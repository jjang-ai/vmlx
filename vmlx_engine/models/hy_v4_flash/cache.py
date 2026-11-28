# SPDX-License-Identifier: Apache-2.0
"""KV, IndexPool, and Engram-history caches for Hy4-flash with exact DFly2 rollback & SSD serialization."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np


class HyV4RotatingMLACache:
    """Sliding-window latent MLA cache (window=512, c_kv=512 + k_pe=64) for the 30 SWA layers.

    Total resident memory across all 30 SWA layers is fixed at 16.875 MiB regardless of context length!
    """

    def __init__(self, max_size: int = 512) -> None:
        self.max_size = int(max_size)
        self.c_kv: Optional[mx.array] = None  # [B, 1, <=max_size, 512]
        self.k_pe: Optional[mx.array] = None  # [B, 1, <=max_size, 64]
        self.offset: int = 0
        self._last_in_ckv: Optional[mx.array] = None
        self._last_in_kpe: Optional[mx.array] = None

    def update_and_fetch(self, c_kv: mx.array, k_pe: mx.array) -> Tuple[mx.array, mx.array]:
        L = int(c_kv.shape[2])
        self._last_in_ckv = c_kv if L <= 32 else None
        self._last_in_kpe = k_pe if L <= 32 else None
        if self.c_kv is None or self.offset == 0:
            full_kv, full_pe = c_kv, k_pe
        else:
            keep_prev = min(int(self.c_kv.shape[2]), self.max_size - 1) if self.max_size > 1 else 0
            if keep_prev > 0:
                full_kv = mx.concatenate([self.c_kv[:, :, -keep_prev:, :], c_kv], axis=2)
                full_pe = mx.concatenate([self.k_pe[:, :, -keep_prev:, :], k_pe], axis=2)
            else:
                full_kv, full_pe = c_kv, k_pe
        self.offset += L
        if full_kv.shape[2] > self.max_size:
            self.c_kv = mx.contiguous(full_kv[:, :, -self.max_size :, :])
            self.k_pe = mx.contiguous(full_pe[:, :, -self.max_size :, :])
        else:
            self.c_kv = mx.contiguous(full_kv)
            self.k_pe = mx.contiguous(full_pe)
        return full_kv, full_pe

    def snapshot(self) -> Tuple[Optional[mx.array], Optional[mx.array], int]:
        return (self.c_kv, self.k_pe, self.offset)

    def rollback(self, snap: Tuple[Optional[mx.array], Optional[mx.array], int], kept: int) -> None:
        prev_kv, prev_pe, prev_off = snap
        if kept <= 0 or self._last_in_ckv is None:
            self.c_kv, self.k_pe, self.offset = prev_kv, prev_pe, prev_off
            return
        k_kv = self._last_in_ckv[:, :, :kept, :]
        k_pe = self._last_in_kpe[:, :, :kept, :]
        if prev_kv is None or prev_off == 0:
            m_kv, m_pe = k_kv, k_pe
        else:
            m_kv = mx.concatenate([prev_kv, k_kv], axis=2)
            m_pe = mx.concatenate([prev_pe, k_pe], axis=2)
        if m_kv.shape[2] > self.max_size:
            m_kv = m_kv[:, :, -self.max_size :, :]
            m_pe = m_pe[:, :, -self.max_size :, :]
        self.c_kv = mx.contiguous(m_kv)
        self.k_pe = mx.contiguous(m_pe)
        self.offset = prev_off + kept

    @property
    def state(self):
        return (self.c_kv, self.k_pe, mx.array([self.offset], dtype=mx.int64))

    @state.setter
    def state(self, v) -> None:
        if v is None:
            self.c_kv = self.k_pe = None
            self.offset = 0
        else:
            self.c_kv, self.k_pe = v[0], v[1]
            self.offset = int(v[2].reshape(-1)[0].item())

    def copy(self) -> "HyV4RotatingMLACache":
        c = HyV4RotatingMLACache(self.max_size)
        c.c_kv = mx.contiguous(self.c_kv) if self.c_kv is not None else None
        c.k_pe = mx.contiguous(self.k_pe) if self.k_pe is not None else None
        c.offset = self.offset
        return c


class HyV4GrowingMLACache:
    """Growing latent MLA cache (`c_kv [B,1,S,512]` + `k_pe [B,1,S,64]`) for the 12 DSA layers.

    Also holds the compressed 4-token IndexPool group keys `[B, G, 128]`, the trailing `<4`-token
    incomplete group buffer (`tail_tokens [B, <4, 128]`), and the most recent top-2048 selected
    indices snapshot (`last_topk_indices [B, L, K]`) so inline DFly2 layers `[0..4]` reuse target
    layers `[25, 29, 33, 37, 41]` caches & sparse indices with ZERO extra KV memory!
    """

    step: int = 256
    group_step: int = 64

    def __init__(self, compress_ratio: int = 4) -> None:
        self.compress_ratio = int(compress_ratio)
        self.c_kv: Optional[mx.array] = None
        self.k_pe: Optional[mx.array] = None
        self.offset: int = 0

        # IndexPool compressed group keys (`[B, G, 128]` BF16) + pending <4-token tail (`[B, <4, 128]`)
        self.group_keys: Optional[mx.array] = None
        self.group_count: int = 0
        self.tail_keys: Optional[mx.array] = None
        self.tail_len: int = 0
        self.tail_base_pos: int = 0

        # Stashed per-step raw token keys for exact IndexPool rollback across 4-token boundaries
        self._last_in_tok_keys: Optional[mx.array] = None
        self._last_in_tok_pos: Optional[mx.array] = None
        # Most recent selected sparse token indices (reused directly by DFly2!)
        self.last_topk_indices: Optional[mx.array] = None

    def update_and_fetch_kv(self, c_kv: mx.array, k_pe: mx.array) -> Tuple[mx.array, mx.array]:
        B, H, L, D_kv = c_kv.shape
        D_pe = k_pe.shape[-1]
        prev = self.offset
        if self.c_kv is None or (prev + L) > self.c_kv.shape[2]:
            n_steps = (self.step + L - 1) // self.step
            k_add = mx.zeros((B, H, n_steps * self.step, D_kv), dtype=c_kv.dtype)
            p_add = mx.zeros((B, H, n_steps * self.step, D_pe), dtype=k_pe.dtype)
            if self.c_kv is not None:
                if prev % self.step != 0:
                    self.c_kv = self.c_kv[..., :prev, :]
                    self.k_pe = self.k_pe[..., :prev, :]
                self.c_kv = mx.concatenate([self.c_kv, k_add], axis=2)
                self.k_pe = mx.concatenate([self.k_pe, p_add], axis=2)
            else:
                self.c_kv, self.k_pe = k_add, p_add
        self.c_kv[..., prev : prev + L, :] = c_kv
        self.k_pe[..., prev : prev + L, :] = k_pe
        self.offset = prev + L
        return self.c_kv[..., : self.offset, :], self.k_pe[..., : self.offset, :]

    def append_group_keys(self, new_groups: mx.array) -> Optional[mx.array]:
        """Append `[B, n_new, 128]` pooled+RoPEd IndexPool group keys and return `[B, G_total, 128]`."""
        B, n_new, D = new_groups.shape
        if n_new > 0:
            prev = self.group_count
            if self.group_keys is None or (prev + n_new) > self.group_keys.shape[1]:
                n_steps = (self.group_step + n_new - 1) // self.group_step
                alloc = mx.zeros((B, n_steps * self.group_step, D), dtype=new_groups.dtype)
                if self.group_keys is not None:
                    if prev % self.group_step != 0:
                        self.group_keys = self.group_keys[:, :prev, :]
                    self.group_keys = mx.concatenate([self.group_keys, alloc], axis=1)
                else:
                    self.group_keys = alloc
            self.group_keys[:, prev : prev + n_new, :] = new_groups
            self.group_count = prev + n_new
        if self.group_keys is None or self.group_count == 0:
            return None
        return self.group_keys[:, : self.group_count, :]

    def fetch_kv(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if self.c_kv is None or self.offset == 0:
            return None, None
        return self.c_kv[..., : self.offset, :], self.k_pe[..., : self.offset, :]

    def fetch_group_keys(self) -> Optional[mx.array]:
        if self.group_keys is None or self.group_count == 0:
            return None
        return self.group_keys[:, : self.group_count, :]

    def snapshot(self) -> Tuple[Any, ...]:
        return (
            self.offset,
            self.group_count,
            self.tail_keys,
            self.tail_len,
            self.tail_base_pos,
            self.last_topk_indices,
        )

    def rollback(
        self,
        snap: Tuple[Any, ...],
        kept: int,
        indexer: Optional[Any] = None,
    ) -> None:
        """Exact O(1) rollback of growing KV + IndexPool 4-token groups & pending tail."""
        prev_off, prev_gc, prev_tail, prev_tlen, prev_tbase, prev_topk = snap
        self.offset = prev_off + kept
        if kept <= 0 or self._last_in_tok_keys is None:
            self.group_count = prev_gc
            self.tail_keys = prev_tail
            self.tail_len = prev_tlen
            self.tail_base_pos = prev_tbase
            self.last_topk_indices = prev_topk
            return

        # Recompute exact group_count + tail from prev_tail + kept slice of incoming token keys
        kept_toks = self._last_in_tok_keys[:, :kept, :]
        if prev_tail is not None and prev_tlen > 0:
            full = mx.concatenate([prev_tail, kept_toks], axis=1)
            base_pos = prev_tbase
        else:
            full = kept_toks
            base_pos = prev_off
        total = int(full.shape[1])
        r = self.compress_ratio
        n_complete = total // r
        usable = n_complete * r
        rem = total - usable
        # Completed groups from [prev_gc .. prev_gc + n_complete) were ALREADY written during verify!
        # Since group pooling depends ONLY on its 4 constituent tokens and verify kept all `kept`
        # prefix tokens, those `n_complete` groups in `self.group_keys` are already bit-exact!
        self.group_count = prev_gc + n_complete
        if rem > 0:
            self.tail_keys = mx.contiguous(full[:, usable:, :])
            self.tail_len = rem
            self.tail_base_pos = base_pos + usable
        else:
            self.tail_keys = None
            self.tail_len = 0
            self.tail_base_pos = base_pos + usable
        if self.last_topk_indices is not None and kept <= self.last_topk_indices.shape[1]:
            self.last_topk_indices = self.last_topk_indices[:, :kept, :]

    @property
    def state(self):
        ckv = self.c_kv[..., : self.offset, :] if (self.c_kv is not None and self.offset > 0) else None
        kpe = self.k_pe[..., : self.offset, :] if (self.k_pe is not None and self.offset > 0) else None
        gk = self.group_keys[:, : self.group_count, :] if (self.group_keys is not None and self.group_count > 0) else None
        meta = mx.array([self.offset, self.group_count, self.tail_len, self.tail_base_pos], dtype=mx.int64)
        return (ckv, kpe, gk, self.tail_keys, meta)

    @state.setter
    def state(self, v) -> None:
        if v is None:
            self.c_kv = self.k_pe = self.group_keys = self.tail_keys = self.last_topk_indices = None
            self.offset = self.group_count = self.tail_len = self.tail_base_pos = 0
        else:
            ckv, kpe, gk, tk, meta = v
            m = [int(x) for x in meta.reshape(-1).tolist()]
            self.c_kv, self.k_pe, self.group_keys, self.tail_keys = ckv, kpe, gk, tk
            self.offset, self.group_count, self.tail_len, self.tail_base_pos = m[0], m[1], m[2], m[3]
            self.last_topk_indices = None

    def copy(self) -> "HyV4GrowingMLACache":
        c = HyV4GrowingMLACache(self.compress_ratio)
        if self.c_kv is not None and self.offset > 0:
            c.c_kv = mx.contiguous(self.c_kv[..., : self.offset, :])
            c.k_pe = mx.contiguous(self.k_pe[..., : self.offset, :])
        c.offset = self.offset
        if self.group_keys is not None and self.group_count > 0:
            c.group_keys = mx.contiguous(self.group_keys[:, : self.group_count, :])
        c.group_count = self.group_count
        c.tail_keys = mx.contiguous(self.tail_keys) if self.tail_keys is not None else None
        c.tail_len = self.tail_len
        c.tail_base_pos = self.tail_base_pos
        c.last_topk_indices = mx.contiguous(self.last_topk_indices) if self.last_topk_indices is not None else None
        return c


def _serializable(arr: Optional[mx.array]) -> mx.array:
    return arr if arr is not None else mx.zeros((0,), dtype=mx.uint8)


def _nullable(arr: Optional[mx.array]) -> Optional[mx.array]:
    if arr is None or arr.size == 0:
        return None
    return arr


class HyV4LayerCache:
    """Unified per-layer cache wrapper compatible with vMLX's cache list & SSD serializer.

    Layer 0 also carries `engram_tail` (last `max_ngram - 1` compressed token IDs on CPU) so
    multi-turn Engram hashing stays 100% self-contained inside the prompt cache list!
    """

    def __init__(self, inner: HyV4RotatingMLACache | HyV4GrowingMLACache, *, is_dsa: bool) -> None:
        self.inner = inner
        self.is_dsa = bool(is_dsa)
        self.engram_tail: Optional[np.ndarray] = None  # stored on layer 0 only

    @property
    def offset(self) -> int:
        return int(self.inner.offset)

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self) -> List[mx.array]:
        eng = mx.array(self.engram_tail, dtype=mx.int64) if self.engram_tail is not None else mx.zeros((0,), dtype=mx.int64)
        if self.is_dsa:
            assert isinstance(self.inner, HyV4GrowingMLACache)
            ckv, kpe = self.inner.fetch_kv()
            gk = self.inner.fetch_group_keys()
            tk = self.inner.tail_keys
            return [
                _serializable(mx.contiguous(ckv) if ckv is not None else None),
                _serializable(mx.contiguous(kpe) if kpe is not None else None),
                _serializable(mx.contiguous(gk) if gk is not None else None),
                _serializable(mx.contiguous(tk) if tk is not None else None),
                eng,
            ]
        assert isinstance(self.inner, HyV4RotatingMLACache)
        return [
            _serializable(mx.contiguous(self.inner.c_kv) if self.inner.c_kv is not None else None),
            _serializable(mx.contiguous(self.inner.k_pe) if self.inner.k_pe is not None else None),
            eng,
        ]

    @state.setter
    def state(self, v) -> None:
        arrs = list(v or ())
        if self.is_dsa:
            if len(arrs) != 5:
                raise ValueError(f"HyV4LayerCache DSA state expects 5 arrays, got {len(arrs)}")
            assert isinstance(self.inner, HyV4GrowingMLACache)
            ckv, kpe, gk, tk, eng = (_nullable(a) for a in arrs)
            self.inner.c_kv = ckv
            self.inner.k_pe = kpe
            self.inner.offset = int(ckv.shape[2]) if ckv is not None else 0
            self.inner.group_keys = gk
            self.inner.group_count = int(gk.shape[1]) if gk is not None else 0
            self.inner.tail_keys = tk
            self.inner.tail_len = int(tk.shape[1]) if tk is not None else 0
            self.inner.last_topk_indices = None
            self.engram_tail = np.array(eng, dtype=np.int64) if eng is not None else None
        else:
            if len(arrs) != 3:
                raise ValueError(f"HyV4LayerCache SWA state expects 3 arrays, got {len(arrs)}")
            assert isinstance(self.inner, HyV4RotatingMLACache)
            ckv, kpe, eng = (_nullable(a) for a in arrs)
            self.inner.c_kv = ckv
            self.inner.k_pe = kpe
            self.engram_tail = np.array(eng, dtype=np.int64) if eng is not None else None

    @property
    def meta_state(self) -> Tuple[str, ...]:
        if self.is_dsa:
            assert isinstance(self.inner, HyV4GrowingMLACache)
            return (
                "hy_v4_flash_cache_v1",
                "dsa",
                str(self.inner.offset),
                str(self.inner.compress_ratio),
                str(self.inner.group_count),
                str(self.inner.tail_len),
                str(self.inner.tail_base_pos),
            )
        assert isinstance(self.inner, HyV4RotatingMLACache)
        return (
            "hy_v4_flash_cache_v1",
            "swa",
            str(self.inner.offset),
            str(self.inner.max_size),
            "0",
            "0",
            "0",
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        meta = tuple(value or ())
        if len(meta) != 7 or meta[0] != "hy_v4_flash_cache_v1":
            raise ValueError("Invalid HyV4LayerCache meta_state")
        off = int(meta[2])
        if self.is_dsa:
            assert isinstance(self.inner, HyV4GrowingMLACache)
            self.inner.offset = off
            self.inner.compress_ratio = int(meta[3])
            self.inner.group_count = int(meta[4])
            self.inner.tail_len = int(meta[5])
            self.inner.tail_base_pos = int(meta[6])
        else:
            assert isinstance(self.inner, HyV4RotatingMLACache)
            self.inner.offset = off
            self.inner.max_size = int(meta[3])

    @classmethod
    def from_state(cls, state, meta_state) -> "HyV4LayerCache":
        meta = tuple(meta_state or ())
        if len(meta) != 7 or meta[0] != "hy_v4_flash_cache_v1":
            raise ValueError("HyV4LayerCache metadata missing hy_v4_flash_cache_v1 header")
        mode = str(meta[1])
        if mode == "dsa":
            rebuilt = cls(HyV4GrowingMLACache(compress_ratio=int(meta[3])), is_dsa=True)
        elif mode == "swa":
            rebuilt = cls(HyV4RotatingMLACache(max_size=int(meta[3])), is_dsa=False)
        else:
            raise ValueError(f"Invalid HyV4LayerCache mode: {mode!r}")
        rebuilt.state = state
        rebuilt.meta_state = meta
        return rebuilt

    def copy(self) -> "HyV4LayerCache":
        c = HyV4LayerCache(self.inner.copy(), is_dsa=self.is_dsa)
        c.engram_tail = self.engram_tail.copy() if self.engram_tail is not None else None
        return c


def clone_hy4_layer_cache(source: HyV4LayerCache, *, copy_fn: Any = None) -> HyV4LayerCache:
    """Return a non-aliasing exact-boundary copy of one Hy4-flash layer cache."""
    if type(source).__name__ != "HyV4LayerCache":
        raise TypeError(f"unexpected cache class: {type(source).__name__}")
    if copy_fn is None:
        return source.copy()
    copied_state = [copy_fn(arr) for arr in source.state]
    return HyV4LayerCache.from_state(copied_state, source.meta_state)


def register_hy4_cache_class() -> None:
    """Register HyV4LayerCache on mlx_lm.models.cache for typed SSD prompt-cache round-trip."""
    try:
        import mlx_lm.models.cache as mlx_cache

        setattr(mlx_cache, "HyV4LayerCache", HyV4LayerCache)
    except Exception:
        pass


register_hy4_cache_class()


def snapshot_hy4_caches(caches: Sequence[HyV4LayerCache]) -> Tuple[List[Any], Optional[np.ndarray]]:
    """O(1) snapshot of all 42 Hy4-flash layer caches + Engram 2-token history."""
    snaps = [lc.inner.snapshot() for lc in caches]
    eng_tail = caches[0].engram_tail.copy() if (caches and caches[0].engram_tail is not None) else None
    return snaps, eng_tail


def rollback_hy4_caches(
    caches: Sequence[HyV4LayerCache],
    snap_bundle: Tuple[List[Any], Optional[np.ndarray]],
    kept: int,
    kept_compressed_tail: Optional[np.ndarray] = None,
) -> None:
    """Bit-exact rollback of all 42 SWA/DSA/IndexPool caches + Engram history to `kept` tokens."""
    snaps, prev_eng_tail = snap_bundle
    for lc, sn in zip(caches, snaps):
        lc.inner.rollback(sn, kept)
    if caches:
        if kept <= 0 or kept_compressed_tail is None:
            caches[0].engram_tail = prev_eng_tail.copy() if prev_eng_tail is not None else None
        else:
            caches[0].engram_tail = kept_compressed_tail.copy()

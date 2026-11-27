# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 HCA/MLA hybrid caches with speculative rollback and prefix-cache support."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import mlx.core as mx


class ErnieKVCache:
    """Append-only KV cache (used for MLA latent/k_pe and indexer key lanes)."""

    step: int = 256

    def __init__(self) -> None:
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_dim = keys.shape
            v_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        if values.shape[3] > 0:
            self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def trim(self, n: int) -> int:
        n = min(self.offset, max(0, int(n)))
        self.offset -= n
        return n

    def is_trimmable(self) -> bool:
        return True

    @property
    def state(self):
        if self.keys is None or self.offset == 0:
            return None, None
        return (
            mx.contiguous(self.keys[..., : self.offset, :]),
            mx.contiguous(self.values[..., : self.offset, :]),
        )

    @state.setter
    def state(self, v) -> None:
        if v is None or v[0] is None:
            self.keys = None
            self.values = None
            self.offset = 0
        else:
            self.keys, self.values = v
            self.offset = int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.nbytes) + (int(self.values.nbytes) if self.values is not None else 0)

    def copy(self) -> "ErnieKVCache":
        c = ErnieKVCache()
        if self.keys is not None and self.offset > 0:
            c.keys = mx.contiguous(self.keys[..., : self.offset, :])
            c.values = mx.contiguous(self.values[..., : self.offset, :])
            c.offset = self.offset
        return c


class ErnieRotatingCache:
    """Sliding-window local KV cache for ERNIE HCA layers (max_size=128) with exact rollback."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = int(max_size)
        self.keys: Optional[mx.array] = None
        self.offset: int = 0
        # Snapshot buffer for exact speculative rollback without deep-copying full caches
        self._rollback_keys: Optional[mx.array] = None
        self._rollback_offset: int = 0

    def update_and_fetch(self, keys: mx.array, values: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # keys: [B, 1, L, head_dim]; keep chronological tail of length <= max_size + L - 1
        # During decode (L=1), retaining the last max_size chronological tokens matches
        # mlx_vlm's RotatingKVCache semantics exactly while avoiding ring-buffer bugs.
        B, H, L, D = keys.shape
        if self.keys is None or self.offset == 0:
            full = keys
        else:
            # Keep at most max_size - 1 previous tokens when L == 1 (so total <= max_size),
            # or max_size - 1 previous tokens + all L new tokens during prefill/verify so
            # every query row in [0..L-1] sees its full sliding_window history!
            keep_prev = min(self.keys.shape[2], self.max_size - 1) if self.max_size > 1 else 0
            if keep_prev > 0:
                prev = self.keys[:, :, -keep_prev:, :]
                full = mx.concatenate([prev, keys], axis=2)
            else:
                full = keys
        self.offset += L
        # Store only the last max_size tokens for subsequent steps; return `full` for this forward!
        if full.shape[2] > self.max_size:
            self.keys = mx.contiguous(full[:, :, -self.max_size :, :])
        else:
            self.keys = mx.contiguous(full)
        empty_v = mx.zeros((B, H, full.shape[2], 0), dtype=keys.dtype)
        return full, empty_v

    def begin_speculative(self) -> None:
        self._rollback_keys = self.keys
        self._rollback_offset = self.offset

    def commit_speculative(self, keep_new: int, new_keys: Optional[mx.array]) -> None:
        """Keep `keep_new` tokens from the speculative verify batch starting at `_rollback_offset`."""
        prev_k = self._rollback_keys
        prev_off = self._rollback_offset
        self._rollback_keys = None
        if keep_new <= 0 or new_keys is None:
            self.keys = prev_k
            self.offset = prev_off
            return
        kept = new_keys[:, :, :keep_new, :]
        if prev_k is None or prev_off == 0:
            merged = kept
        else:
            merged = mx.concatenate([prev_k, kept], axis=2)
        if merged.shape[2] > self.max_size:
            merged = merged[:, :, -self.max_size :, :]
        self.keys = mx.contiguous(merged)
        self.offset = prev_off + keep_new

    def abort_speculative(self) -> None:
        self.keys = self._rollback_keys
        self.offset = self._rollback_offset
        self._rollback_keys = None

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self):
        return (self.keys, mx.array([self.offset], dtype=mx.int64))

    @state.setter
    def state(self, v) -> None:
        if v is None:
            self.keys = None
            self.offset = 0
        else:
            self.keys = v[0]
            self.offset = int(v[1].reshape(-1)[0].item())

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes) if self.keys is not None else 0

    def copy(self) -> "ErnieRotatingCache":
        c = ErnieRotatingCache(self.max_size)
        c.keys = mx.contiguous(self.keys) if self.keys is not None else None
        c.offset = self.offset
        return c


class ErniePoolingCache:
    """Exact non-overlapping HCA window accumulator + pooled KV store (ratio=128) with rollback."""

    step: int = 64

    def __init__(self, compress_ratio: int = 128) -> None:
        self.compress_ratio = int(compress_ratio)
        self.pooled: Optional[mx.array] = None
        self.pooled_count: int = 0
        # Pending partial window (< compress_ratio tokens) stored in FP32
        self.pending_kv: Optional[mx.array] = None
        self.pending_gate: Optional[mx.array] = None
        self.pending_len: int = 0
        self.offset: int = 0
        # Speculative snapshot + tiny verify-input stash for exact 128-boundary rollback
        self._snap: Optional[Tuple[Any, ...]] = None
        self._last_in_kv: Optional[mx.array] = None
        self._last_in_gate: Optional[mx.array] = None

    def accumulate_windows(
        self, kv: mx.array, gate: mx.array, offset: int
    ) -> Tuple[mx.array, mx.array, int]:
        """Accumulate incoming FP32 (kv, gate) into pending buffer and emit complete windows."""
        B, L, D = kv.shape
        if L <= 32:
            self._last_in_kv = kv
            self._last_in_gate = gate
        else:
            self._last_in_kv = None
            self._last_in_gate = None
        if self.pending_kv is not None and self.pending_len > 0:
            full_kv = mx.concatenate([self.pending_kv, kv], axis=1)
            full_gate = mx.concatenate([self.pending_gate, gate], axis=1)
            pool_base = int(offset) - self.pending_len
        else:
            full_kv = kv
            full_gate = gate
            pool_base = int(offset)

        total = full_kv.shape[1]
        n_windows = total // self.compress_ratio
        usable = n_windows * self.compress_ratio
        rem = total - usable

        if rem > 0:
            self.pending_kv = mx.contiguous(full_kv[:, usable:, :])
            self.pending_gate = mx.contiguous(full_gate[:, usable:, :])
            self.pending_len = rem
        else:
            self.pending_kv = None
            self.pending_gate = None
            self.pending_len = 0

        self.offset = int(offset) + L
        if usable == 0:
            return (
                mx.zeros((B, 0, D), dtype=kv.dtype),
                mx.zeros((B, 0, D), dtype=gate.dtype),
                pool_base,
            )
        return full_kv[:, :usable, :], full_gate[:, :usable, :], pool_base

    def update_and_fetch(self, new_pooled: mx.array) -> mx.array:
        B, n_new, D = new_pooled.shape
        if n_new > 0:
            prev = self.pooled_count
            if self.pooled is None or (prev + n_new) > self.pooled.shape[1]:
                n_steps = (self.step + n_new - 1) // self.step
                alloc = mx.zeros((B, n_steps * self.step, D), dtype=new_pooled.dtype)
                if self.pooled is not None:
                    if prev % self.step != 0:
                        self.pooled = self.pooled[:, :prev, :]
                    self.pooled = mx.concatenate([self.pooled, alloc], axis=1)
                else:
                    self.pooled = alloc
            self.pooled[:, prev : prev + n_new, :] = new_pooled
            self.pooled_count += n_new
        if self.pooled is None or self.pooled_count == 0:
            return mx.zeros((B, 0, D), dtype=new_pooled.dtype)
        return self.pooled[:, : self.pooled_count, :]

    def make_mask(self, L: int, offset: int) -> Optional[mx.array]:
        """Causal visibility mask [1, 1, L, pooled_count]: query at abs pos q sees pool window w iff (w+1)*R <= q+1."""
        if self.pooled_count == 0:
            return None
        if L == 1:
            # Single token at `offset` already accumulated its own window if (offset+1)%R==0,
            # so every existing pooled window is visible (no mask needed!).
            return None
        q_pos = int(offset) + mx.arange(1, L + 1, dtype=mx.int32)[:, None]
        w_idx = mx.arange(self.pooled_count, dtype=mx.int32)[None, :]
        return w_idx < (q_pos // self.compress_ratio)

    def begin_speculative(self) -> None:
        self._snap = (
            self.pooled_count,
            self.pending_kv,
            self.pending_gate,
            self.pending_len,
            self.offset,
        )

    def abort_speculative(self) -> None:
        if self._snap is not None:
            (
                self.pooled_count,
                self.pending_kv,
                self.pending_gate,
                self.pending_len,
                self.offset,
            ) = self._snap
            self._snap = None

    @property
    def state(self):
        p = mx.contiguous(self.pooled[:, : self.pooled_count, :]) if (self.pooled is not None and self.pooled_count > 0) else None
        return (
            p,
            self.pending_kv,
            self.pending_gate,
            mx.array([self.offset, self.pending_len, self.pooled_count], dtype=mx.int64),
        )

    @state.setter
    def state(self, v) -> None:
        if v is None:
            self.pooled = None
            self.pooled_count = 0
            self.pending_kv = None
            self.pending_gate = None
            self.pending_len = 0
            self.offset = 0
        else:
            p, pkv, pgate, meta = v
            vals = meta.tolist()
            self.offset = int(vals[0])
            self.pending_len = int(vals[1])
            self.pooled_count = int(vals[2])
            self.pooled = p
            self.pending_kv = pkv
            self.pending_gate = pgate

    @property
    def nbytes(self) -> int:
        tot = 0
        if self.pooled is not None:
            tot += int(self.pooled.nbytes)
        if self.pending_kv is not None:
            tot += int(self.pending_kv.nbytes)
        if self.pending_gate is not None:
            tot += int(self.pending_gate.nbytes)
        return tot

    def copy(self) -> "ErniePoolingCache":
        c = ErniePoolingCache(self.compress_ratio)
        c.pooled = mx.contiguous(self.pooled[:, : self.pooled_count, :]) if (self.pooled is not None and self.pooled_count > 0) else None
        c.pooled_count = self.pooled_count
        c.pending_kv = mx.contiguous(self.pending_kv) if self.pending_kv is not None else None
        c.pending_gate = mx.contiguous(self.pending_gate) if self.pending_gate is not None else None
        c.pending_len = self.pending_len
        c.offset = self.offset
        return c


class ErnieLayerCache:
    """Two-lane per-layer cache (local+pool for HCA, latent+indexer for MLA)."""

    def __init__(self, first: Any, second: Any, is_mla: bool = False) -> None:
        self.caches = [first, second]
        self.is_mla = is_mla

    def __getitem__(self, idx: int) -> Any:
        return self.caches[idx]

    @property
    def offset(self) -> int:
        return int(self.caches[0].offset)

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(c, "nbytes", 0)) for c in self.caches)

    def is_trimmable(self) -> bool:
        return self.is_mla

    def trim(self, n: int) -> int:
        if self.is_mla:
            r0 = self.caches[0].trim(n)
            self.caches[1].trim(n)
            return r0
        return 0

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v) -> None:
        for c, sv in zip(self.caches, v):
            c.state = sv

    def copy(self) -> "ErnieLayerCache":
        return ErnieLayerCache(self.caches[0].copy(), self.caches[1].copy(), is_mla=self.is_mla)

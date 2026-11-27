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
    """Sliding-window local KV cache for ERNIE HCA layers (max_size=128) with preallocated O(1) decode."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = int(max_size)
        self._cap = max(self.max_size * 2 + 32, 288)
        self._buf: Optional[mx.array] = None
        self._start: int = 0
        self._end: int = 0
        self.offset: int = 0
        # Speculative rollback stash
        self._rollback_keys: Optional[mx.array] = None
        self._rollback_offset: int = 0
        self._last_in_keys: Optional[mx.array] = None

    @property
    def keys(self) -> Optional[mx.array]:
        if self._buf is None or self._end <= self._start:
            return None
        return self._buf[:, :, self._start : self._end, :]

    @keys.setter
    def keys(self, val: Optional[mx.array]) -> None:
        if val is None or val.shape[2] == 0:
            self._buf = None
            self._start = 0
            self._end = 0
            return
        B, H, S, D = val.shape
        tail = val[:, :, -self.max_size :, :] if S > self.max_size else val
        tlen = int(tail.shape[2])
        if self._buf is None or self._buf.shape[0] != B or self._buf.shape[1] != H or self._buf.shape[3] != D or self._buf.dtype != val.dtype:
            self._buf = mx.zeros((B, H, self._cap, D), dtype=val.dtype)
        self._buf[:, :, :tlen, :] = tail
        self._start = 0
        self._end = tlen

    def update_and_fetch(self, keys: mx.array, values: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        B, H, L, D = keys.shape
        self._last_in_keys = keys if L <= 32 else None
        # Fast O(1) in-place append path for decode & speculative verify (ZERO concat/contiguous!)
        if L <= 32 and self._buf is not None and self._buf.shape[0] == B and self._end + L <= self._cap:
            prev_len = self._end - self._start
            keep_prev = min(prev_len, self.max_size - 1) if self.max_size > 1 else 0
            ret_start = self._end - keep_prev
            self._buf[:, :, self._end : self._end + L, :] = keys
            self._end += L
            self._start = max(0, self._end - self.max_size)
            self.offset += L
            full = self._buf[:, :, ret_start : self._end, :]
            return full, mx.zeros((B, H, full.shape[2], 0), dtype=keys.dtype)

        # Prefill or buffer compaction (runs once every >=160 decode tokens!)
        prev_k = self.keys
        if prev_k is None or self.offset == 0:
            full = keys
        else:
            keep_prev = min(prev_k.shape[2], self.max_size - 1) if self.max_size > 1 else 0
            if keep_prev > 0:
                full = mx.concatenate([prev_k[:, :, -keep_prev:, :], keys], axis=2)
            else:
                full = keys
        self.offset += L
        tail = full[:, :, -self.max_size :, :] if full.shape[2] > self.max_size else full
        tlen = int(tail.shape[2])
        if self._buf is None or self._buf.shape[0] != B or self._buf.shape[1] != H or self._buf.shape[3] != D or self._buf.dtype != keys.dtype:
            self._buf = mx.zeros((B, H, self._cap, D), dtype=keys.dtype)
        self._buf[:, :, :tlen, :] = tail
        self._start = 0
        self._end = tlen
        return full, mx.zeros((B, H, full.shape[2], 0), dtype=keys.dtype)

    def begin_speculative(self) -> None:
        self._rollback_keys = mx.contiguous(self.keys) if self.keys is not None else None
        self._rollback_offset = self.offset

    def commit_speculative(self, keep_new: int, new_keys: Optional[mx.array]) -> None:
        prev_k = self._rollback_keys
        prev_off = self._rollback_offset
        self._rollback_keys = None
        if keep_new <= 0 or new_keys is None:
            self.keys = prev_k
            self.offset = prev_off
            return
        kept = new_keys[:, :, :keep_new, :]
        merged = kept if (prev_k is None or prev_off == 0) else mx.concatenate([prev_k, kept], axis=2)
        self.keys = merged
        self.offset = prev_off + keep_new

    def abort_speculative(self) -> None:
        self.keys = self._rollback_keys
        self.offset = self._rollback_offset
        self._rollback_keys = None

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self):
        k = mx.contiguous(self.keys) if self.keys is not None else None
        return (k, mx.array([self.offset], dtype=mx.int64))

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
        return int(self._buf.nbytes) if self._buf is not None else 0

    def copy(self) -> "ErnieRotatingCache":
        c = ErnieRotatingCache(self.max_size)
        if self.keys is not None:
            c.keys = mx.contiguous(self.keys)
        c.offset = self.offset
        return c


class ErniePoolingCache:
    """Exact non-overlapping HCA window accumulator + pooled KV store (ratio=128) with O(1) decode buffer."""

    step: int = 64

    def __init__(self, compress_ratio: int = 128) -> None:
        self.compress_ratio = int(compress_ratio)
        self.pooled: Optional[mx.array] = None
        self.pooled_count: int = 0
        # Preallocated FP32 pending window buffer (eliminates 74 concat+copy ops/tok!)
        self._pend_cap: int = max(self.compress_ratio + 32, 160)
        self._pend_kv_buf: Optional[mx.array] = None
        self._pend_gate_buf: Optional[mx.array] = None
        self.pending_len: int = 0
        self.offset: int = 0
        # Speculative snapshot + tiny verify-input stash for exact 128-boundary rollback
        self._snap: Optional[Tuple[Any, ...]] = None
        self._last_in_kv: Optional[mx.array] = None
        self._last_in_gate: Optional[mx.array] = None

    @property
    def pending_kv(self) -> Optional[mx.array]:
        if self._pend_kv_buf is None or self.pending_len <= 0:
            return None
        return self._pend_kv_buf[:, : self.pending_len, :]

    @pending_kv.setter
    def pending_kv(self, val: Optional[mx.array]) -> None:
        if val is None or val.shape[1] == 0:
            self.pending_len = 0
            return
        B, n, D = val.shape
        if self._pend_kv_buf is None or self._pend_kv_buf.shape[0] != B or self._pend_kv_buf.shape[2] != D or self._pend_kv_buf.dtype != val.dtype:
            self._pend_kv_buf = mx.zeros((B, self._pend_cap, D), dtype=val.dtype)
        self._pend_kv_buf[:, :n, :] = val

    @property
    def pending_gate(self) -> Optional[mx.array]:
        if self._pend_gate_buf is None or self.pending_len <= 0:
            return None
        return self._pend_gate_buf[:, : self.pending_len, :]

    @pending_gate.setter
    def pending_gate(self, val: Optional[mx.array]) -> None:
        if val is None or val.shape[1] == 0:
            return
        B, n, D = val.shape
        if self._pend_gate_buf is None or self._pend_gate_buf.shape[0] != B or self._pend_gate_buf.shape[2] != D or self._pend_gate_buf.dtype != val.dtype:
            self._pend_gate_buf = mx.zeros((B, self._pend_cap, D), dtype=val.dtype)
        self._pend_gate_buf[:, :n, :] = val

    def accumulate_windows(
        self, kv: mx.array, gate: mx.array, offset: int
    ) -> Tuple[mx.array, mx.array, int]:
        """Accumulate incoming FP32 (kv, gate) into preallocated pending buffer and emit complete windows."""
        B, L, D = kv.shape
        if L <= 32:
            self._last_in_kv = kv
            self._last_in_gate = gate
        else:
            self._last_in_kv = None
            self._last_in_gate = None

        # Fast O(1) in-place decode/verify path when no 128-token boundary completes (~99.2% of decode steps!)
        if L <= 32 and self.pending_len + L < self.compress_ratio:
            if (
                self._pend_kv_buf is None
                or self._pend_kv_buf.shape[0] != B
                or self._pend_kv_buf.shape[2] != D
                or self._pend_kv_buf.dtype != kv.dtype
            ):
                self._pend_kv_buf = mx.zeros((B, self._pend_cap, D), dtype=kv.dtype)
                self._pend_gate_buf = mx.zeros((B, self._pend_cap, D), dtype=gate.dtype)
            p0 = self.pending_len
            self._pend_kv_buf[:, p0 : p0 + L, :] = kv
            self._pend_gate_buf[:, p0 : p0 + L, :] = gate
            self.pending_len = p0 + L
            self.offset = int(offset) + L
            return (
                mx.zeros((B, 0, D), dtype=kv.dtype),
                mx.zeros((B, 0, D), dtype=gate.dtype),
                int(offset) - p0,
            )

        if self._pend_kv_buf is not None and self.pending_len > 0:
            full_kv = mx.concatenate([self._pend_kv_buf[:, : self.pending_len, :], kv], axis=1)
            full_gate = mx.concatenate([self._pend_gate_buf[:, : self.pending_len, :], gate], axis=1)
            pool_base = int(offset) - self.pending_len
        else:
            full_kv = kv
            full_gate = gate
            pool_base = int(offset)

        total = full_kv.shape[1]
        n_windows = total // self.compress_ratio
        usable = n_windows * self.compress_ratio
        rem = total - usable

        if (
            self._pend_kv_buf is None
            or self._pend_kv_buf.shape[0] != B
            or self._pend_kv_buf.shape[2] != D
            or self._pend_kv_buf.dtype != kv.dtype
        ):
            self._pend_kv_buf = mx.zeros((B, self._pend_cap, D), dtype=kv.dtype)
            self._pend_gate_buf = mx.zeros((B, self._pend_cap, D), dtype=gate.dtype)
        if rem > 0:
            self._pend_kv_buf[:, :rem, :] = full_kv[:, usable:, :]
            self._pend_gate_buf[:, :rem, :] = full_gate[:, usable:, :]
        self.pending_len = rem
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

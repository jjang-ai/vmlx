# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 (minimax_m3_vl) MSA dual-cache for the vMLX runtime.

M3 attention is GQA on every layer, with a block-sparse selection (MSA) added on
the sparse layers (3-59; layers 0-2 are full attention). The sparse layers carry
TWO append-only caches in lockstep:

  * the standard GQA KV cache         keys/values  [B, n_kv(=4), S, head_dim(=128)]
  * the Lightning-Indexer key cache   idx_keys     [B, 1, S, index_dim(=128)]

The indexer scores idx_q (current step) against ALL cached idx_keys, max-pools per
128-token block, and selects top-k blocks; the main branch then attends the
selected K/V blocks. SELECTION IS RECOMPUTED EACH STEP from idx_keys — it is never
cached. So the only persistent state is (keys, values, idx_keys), all three
append-only and the same length. Blocks are anchored to ABSOLUTE position
(block = pos // 128), so the cache is append-only / trim-and-replay only: never
shift, rotate, or evict mid-stream (that would move block boundaries and corrupt
selection). Trimming to N tokens slices all three on the sequence axis — which is
exactly what L1 prefix matching and L2 disk restore need.

This mirrors the composite-cache precedent of DeepseekV4Cache / ZayaCCACache: a
custom cache object plus a `cache_data` tuple type the prefix/paged/disk tiers
serialize. M3 is the simplest of the three (one extra tensor, no compressor pool,
no conv/SSM state, no per-layer heterogeneity beyond dense-vs-sparse).

cache_data tuple types contributed by this module (see block_disk_store.py):
  ("minimax_m3", keys_slice, values_slice, idx_keys_slice)   — sparse layer
  dense layers (0-2) reuse the standard ("kv", keys, values) KVCache tuple.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache, dynamic_roll

CACHE_TUPLE_TAG = "minimax_m3"


class MiniMaxM3SparseCache(KVCache):
    """KVCache + an append-only indexer-key cache (idx_keys).

    Subclasses the stock KVCache so the K/V half inherits the exact
    update_and_fetch / trim / step-growth semantics the runtime already relies
    on; we add a parallel idx_keys buffer that grows in lockstep and rides
    through the same serialization path.
    """

    def __init__(self):
        super().__init__()
        self.idx_keys: mx.array | None = None
        self._idx_offset = 0

    # ── indexer-key side (called by the attention layer each step) ──
    def update_index(self, idx_k: mx.array) -> mx.array:
        """Append this step's idx_k [B, 1, T, D] and return the full idx history.

        Grows with the same step/over-allocation policy as the KV side so the two
        offsets stay aligned; the indexer reads `idx_keys[..., : self.offset, :]`.
        """
        prev = self.idx_keys
        if prev is None:
            self.idx_keys = idx_k
        else:
            self.idx_keys = mx.concatenate([prev, idx_k], axis=2)
        self._idx_offset = self.idx_keys.shape[2]
        # Return idx_keys sliced to the CURRENT KV offset. The attention forward now
        # calls cache.update_and_fetch(k, v) BEFORE the indexer (upstream ordering),
        # so self.offset is already the post-append length and this slice equals the
        # full appended idx history -> Sk matches SDPA's K. Keeps update_index, the
        # indexer scoring, and `state` serialization all consistent on self.offset
        # (the 'return full' variant desynced serialization and broke coherence).
        return self.idx_keys[..., : self.offset, :] if self.offset else self.idx_keys

    # ── serialization: expose the 3-tensor slice the disk tiers pack ──
    @property
    def state(self):  # type: ignore[override]
        k, v = super().state
        idx = None if self.idx_keys is None else self.idx_keys[..., : self.offset, :]
        return k, v, idx

    @state.setter
    def state(self, v):
        if len(v) == 3:
            keys, values, idx = v
            KVCache.state.fset(self, (keys, values))
            self.idx_keys = idx
            self._idx_offset = 0 if idx is None else idx.shape[2]
        else:
            KVCache.state.fset(self, v)

    def to_cache_data(self) -> tuple:
        """The tuple the L1/L2 tiers persist (see block_disk_store contract)."""
        k, v, idx = self.state
        return (CACHE_TUPLE_TAG, k, v, idx)

    def trim(self, n: int) -> int:  # type: ignore[override]
        """Trim the last `n` tokens from BOTH caches (prefix-match downgrade).

        Append-only invariant: K/V and idx_keys are the same length, so the same
        trim count applies to all three. Returns the number actually trimmed.
        """
        trimmed = super().trim(n)
        if self.idx_keys is not None and trimmed:
            self.idx_keys = self.idx_keys[..., : self.offset, :]
            self._idx_offset = self.offset
        return trimmed


class BatchMiniMaxM3SparseCache(BatchKVCache):
    """Batch-aware MiniMax-M3 cache that keeps indexer keys aligned with K/V.

    ``MiniMaxM3SparseCache`` subclasses ``KVCache``, so the generic vMLX batch
    promotion used to silently turn it into a stock ``BatchKVCache``.  That
    discarded ``idx_keys`` and made the first concurrent decode step fail at
    ``update_index``.  This class mirrors every BatchKVCache operation that can
    change sequence or batch alignment and applies it to the third tensor too.
    """

    def __init__(self, left_padding: list[int]):
        super().__init__(left_padding)
        self.idx_keys: mx.array | None = None
        self._idx_offset = 0

    def update_index(self, idx_k: mx.array) -> mx.array:
        """Append batched indexer keys after the matching K/V append."""
        if idx_k.ndim != 4:
            raise ValueError(f"idx_k must have shape [B, H, T, D], got {idx_k.shape}")
        if int(idx_k.shape[0]) != int(self.offset.shape[0]):
            raise ValueError(
                "idx_k batch does not match cache batch: "
                f"{idx_k.shape[0]} != {self.offset.shape[0]}"
            )

        prev = self._idx_offset
        target = prev + int(idx_k.shape[2])
        # QSA appends K/V immediately before calling update_index.  Refuse to
        # continue if a future call site breaks that lockstep invariant.
        if target != self._idx:
            raise ValueError(
                "indexer and K/V cache lengths diverged: "
                f"idx target={target}, kv target={self._idx}"
            )

        if self.idx_keys is None or target > int(self.idx_keys.shape[2]):
            batch, heads, _, dim = idx_k.shape
            n_steps = (self.step + int(idx_k.shape[2]) - 1) // self.step
            new_idx = mx.zeros(
                (batch, heads, n_steps * self.step, dim), dtype=idx_k.dtype
            )
            if self.idx_keys is not None:
                if prev % self.step != 0:
                    self.idx_keys = self.idx_keys[..., :prev, :]
                self.idx_keys = mx.concatenate([self.idx_keys, new_idx], axis=2)
            else:
                self.idx_keys = new_idx

        self.idx_keys[..., prev:target, :] = idx_k
        self._idx_offset = target
        return self.idx_keys[..., :target, :]

    @classmethod
    def merge(cls, caches: list[MiniMaxM3SparseCache]):
        """Right-align single-request sparse caches into one decode batch."""
        if not caches:
            raise ValueError("cannot merge an empty cache list")

        lengths = [int(c.size()) for c in caches]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        if max_length == 0:
            return cls([0] * len(caches))

        for cache in caches:
            if cache.keys is not None and cache.idx_keys is None:
                raise ValueError("sparse cache has K/V state but no idx_keys")

        base = BatchKVCache.merge(caches)
        populated = [c for c in caches if c.idx_keys is not None]
        heads = max(int(c.idx_keys.shape[1]) for c in populated)
        dim = max(int(c.idx_keys.shape[3]) for c in populated)
        dtype = populated[0].idx_keys.dtype
        idx_keys = mx.zeros((len(caches), heads, max_length, dim), dtype=dtype)
        for row, (pad, cache) in enumerate(zip(padding, caches)):
            if cache.idx_keys is None:
                continue
            length = lengths[row]
            idx_keys[row : row + 1, :, pad : pad + length, :] = cache.idx_keys[
                ..., :length, :
            ]

        merged = cls(padding)
        merged.keys = base.keys
        merged.values = base.values
        merged.offset = base.offset
        merged.left_padding = base.left_padding
        merged._idx = base._idx
        merged.idx_keys = idx_keys
        merged._idx_offset = max_length
        return merged

    def extract(self, idx: int) -> MiniMaxM3SparseCache:
        """Extract one row without losing the sparse indexer history."""
        cache = MiniMaxM3SparseCache()
        if self.keys is None:
            return cache
        padding = int(self.left_padding[idx].item())
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        if self.idx_keys is None:
            raise ValueError("batched sparse cache has K/V state but no idx_keys")
        cache.idx_keys = mx.contiguous(
            self.idx_keys[idx : idx + 1, :, padding : self._idx]
        )
        cache.offset = int(cache.keys.shape[2])
        cache._idx_offset = cache.offset
        return cache

    def filter(self, batch_indices) -> None:
        """Keep selected rows and shift all three tensors by the same padding."""
        selected_padding = self.left_padding[batch_indices]
        min_left_pad = int(selected_padding.min().item())
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys[batch_indices]
        super().filter(batch_indices)
        # BatchKVCache.filter() removes the minimum remaining left padding from
        # K/V but retains any 256-token capacity reserve. Apply that exact shift
        # instead of inferring it from the allocated tensor length.
        if self.idx_keys is not None and min_left_pad > 0:
            self.idx_keys = self.idx_keys[..., min_left_pad:, :]
        self._idx_offset = self._idx

    @staticmethod
    def _pad_idx_for_extend(
        cache: BatchMiniMaxM3SparseCache,
        *,
        max_idx: int,
        max_size: int,
        heads: int,
        dim: int,
        dtype,
    ) -> mx.array:
        if cache.idx_keys is None:
            if cache.keys is not None:
                raise ValueError("sparse cache has K/V state but no idx_keys")
            idx_keys = mx.zeros(
                (int(cache.offset.shape[0]), heads, 0, dim), dtype=dtype
            )
        else:
            idx_keys = cache.idx_keys

        left = max_idx - int(cache._idx)
        right = max_size - int(idx_keys.shape[2]) - left
        if right < 0:
            idx_keys = idx_keys[..., :right, :]
            right = 0
        if left or right:
            idx_keys = mx.pad(
                idx_keys,
                [(0, 0), (0, 0), (left, right), (0, 0)],
            )
        return idx_keys

    def extend(self, other) -> None:
        """Append batch rows while retaining right-aligned indexer histories."""
        if not isinstance(other, BatchMiniMaxM3SparseCache):
            raise TypeError(
                "BatchMiniMaxM3SparseCache can only extend the same cache type"
            )
        if self.keys is None and other.keys is None:
            super().extend(other)
            self._idx_offset = self._idx
            return

        populated = [c.idx_keys for c in (self, other) if c.idx_keys is not None]
        if not populated:
            raise ValueError("sparse caches have K/V state but no idx_keys")
        heads = max(int(value.shape[1]) for value in populated)
        dim = max(int(value.shape[3]) for value in populated)
        dtype = populated[0].dtype
        max_idx = max(int(self._idx), int(other._idx))
        max_size = max(
            int(self.keys.shape[2]) if self.keys is not None else 0,
            int(other.keys.shape[2]) if other.keys is not None else 0,
        )
        left_idx = self._pad_idx_for_extend(
            self,
            max_idx=max_idx,
            max_size=max_size,
            heads=heads,
            dim=dim,
            dtype=dtype,
        )
        right_idx = self._pad_idx_for_extend(
            other,
            max_idx=max_idx,
            max_size=max_size,
            heads=heads,
            dim=dim,
            dtype=dtype,
        )
        super().extend(other)
        self.idx_keys = mx.concatenate([left_idx, right_idx], axis=0)
        self._idx_offset = self._idx

    def trim(self, n: int) -> int:
        trimmed = super().trim(n)
        self._idx_offset = self._idx
        return trimmed

    def finalize(self) -> None:
        padding = self._right_padding
        if padding is not None and self.idx_keys is not None:
            self.idx_keys = dynamic_roll(self.idx_keys, padding[:, None], axis=2)
        super().finalize()
        self._idx_offset = self._idx

    @property
    def nbytes(self):
        return super().nbytes + (0 if self.idx_keys is None else self.idx_keys.nbytes)


def restore_minimax_m3_sparse(keys, values, idx) -> MiniMaxM3SparseCache:
    """Rebuild a sparse-layer cache from a persisted ("minimax_m3", ...) tuple."""
    c = MiniMaxM3SparseCache()
    c.state = (keys, values, idx)
    return c


def clone_minimax_m3_sparse(
    cache: Any,
    length: int | None = None,
    *,
    copy_fn=None,
    require_idx_keys: bool = True,
) -> MiniMaxM3SparseCache | None:
    """Clone/slice a MiniMax-M3 sparse cache without dropping idx_keys.

    Generic KV cache helpers see MiniMaxM3SparseCache as a KVCache subclass and
    usually copy only ``(keys, values)``. That corrupts M3 reuse because sparse
    block selection is recomputed from ``idx_keys`` every step. This helper is
    the single safe way for prefix/disk/snapshot paths to rebuild the cache.
    """
    new_cache = MiniMaxM3SparseCache()
    keys = getattr(cache, "keys", None)
    values = getattr(cache, "values", None)
    if keys is None or values is None:
        return new_cache

    idx_keys = getattr(cache, "idx_keys", None)
    if idx_keys is None and require_idx_keys:
        return None

    try:
        candidates = [int(getattr(cache, "offset", 0) or keys.shape[-2])]
        candidates.extend([int(keys.shape[-2]), int(values.shape[-2])])
        if idx_keys is not None:
            candidates.append(int(idx_keys.shape[-2]))
        if length is not None:
            candidates.append(int(length))
        target = min(candidates)
    except Exception:
        return None
    if target < 0:
        return None

    def _slice(value):
        if value is None:
            return None
        sliced = value[..., :target, :]
        return copy_fn(sliced) if copy_fn is not None else sliced

    new_cache.state = (_slice(keys), _slice(values), _slice(idx_keys))
    new_cache.offset = target
    new_cache._idx_offset = 0 if new_cache.idx_keys is None else target
    return new_cache


def truncate_minimax_m3_cache(cache: list, length: int) -> None:
    """Roll a MiniMax-M3 cache list back to an absolute token length.

    Speculative verification appends a draft chain to every target layer, then
    must keep only the accepted prefix. Dense layers are stock KVCache; sparse
    MSA layers override ``trim`` so the K/V and idx_keys streams remain aligned.
    """
    if length < 0:
        raise ValueError("cache truncate length must be non-negative")
    for entry in cache:
        offset = getattr(entry, "offset", None)
        trim = getattr(entry, "trim", None)
        if offset is None or trim is None:
            continue
        if length < offset:
            trim(offset - length)


def make_minimax_m3_cache(config) -> list:
    """Per-layer cache list for the whole model.

    Dense/full-attention layers (0-2) → stock KVCache.
    Sparse MSA layers (3-59)          → MiniMaxM3SparseCache.

    Driven by `sparse_attention_config.sparse_attention_freq` (or moe_layer_freq
    as a proxy), matching the converter/probe layer dispatch.
    """
    tc = getattr(config, "text_config", config)
    n_layers = tc["num_hidden_layers"] if isinstance(tc, dict) else tc.num_hidden_layers
    sca = (tc.get("sparse_attention_config", {}) if isinstance(tc, dict)
           else getattr(tc, "sparse_attention_config", {})) or {}
    freq = sca.get("sparse_attention_freq")
    if freq is None:
        moe = tc.get("moe_layer_freq") if isinstance(tc, dict) else getattr(tc, "moe_layer_freq", None)
        freq = moe if moe is not None else [0, 0, 0] + [1] * (n_layers - 3)
    return [MiniMaxM3SparseCache() if freq[i] else KVCache() for i in range(n_layers)]

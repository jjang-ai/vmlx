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
from mlx_lm.models.cache import KVCache

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


def restore_minimax_m3_sparse(keys, values, idx) -> MiniMaxM3SparseCache:
    """Rebuild a sparse-layer cache from a persisted ("minimax_m3", ...) tuple."""
    c = MiniMaxM3SparseCache()
    c.state = (keys, values, idx)
    return c


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

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
selection). Trimming to N tokens resets all three logical extents. Live backing
capacity remains reusable; state/clone views slice all three on the sequence
axis, as required by L1 prefix matching and L2 disk restore.

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

    # ── lane ownership for in-place appends (fused attention, models/minimax_m3/fused_attn.py) ──
    # The fused decode / verify attention writes new K / V / index rows straight into the lane buffers from a Metal
    # kernel, which MLX cannot see. That is only safe while no other array aliases a lane buffer: MLX's own slice
    # update copies a shared buffer (copy on write), the kernel would overwrite rows under a view someone else keeps
    # (e.g. a zero-copy prefix-store view, once a trim lets new rows land below its extent). So every public read of a
    # lane array (``keys`` / ``values`` / ``idx_keys``, hence ``state`` and the stock update paths) and every
    # assignment of one marks that lane shared; the fused path appends to a shared lane with MLX slice updates (copy
    # on write, ``fused_append``), after which the lane is the cache's own again, and writes in place only while all
    # three lanes are its own (``fused_reserve``). Trim / offset / nbytes never expose a lane.
    def _lane_property(i: int, name: str) -> property:  # noqa: N805 - class-body helper, deleted below
        """A lane attribute: reading or assigning it marks the lane shared."""

        def get(self):
            excl = self.__dict__.get("_lane_excl")
            if excl is not None:
                excl[i] = False
            return self.__dict__.get(name)

        def put(self, value):
            self.__dict__[name] = value
            self.__dict__.setdefault("_lane_excl", [False, False, False])[i] = False

        return property(get, put)

    keys = _lane_property(0, "_keys")
    values = _lane_property(1, "_values")
    idx_keys = _lane_property(2, "_idx_keys")
    del _lane_property

    def __init__(self):
        self._lane_excl = [False, False, False]
        super().__init__()
        self.idx_keys: mx.array | None = None
        self._idx_offset = 0
        # Ephemeral DERIVED state owned by the consumer (e.g. Qwen4's QSA
        # indexer keeps its pooled/normed/roped completed-block keys here so
        # they are not recomputed from the whole raw history every call).
        # Never serialized; cleared on any restore/replacement, truncated on
        # trim. The raw ``idx_keys`` lane stays the only authority.
        self.derived: dict = {}

    @property
    def derived_nbytes(self) -> int:
        """Bytes held by consumer-owned derived state (e.g. Qwen4's retained
        pooled QSA keys); reported to the cache accounting alongside K/V and
        the raw index lane."""
        total = 0
        for value in self.derived.values():
            nb = getattr(value, "nbytes", None)
            if isinstance(nb, int):
                total += nb
        return total

    def _truncate_derived(self) -> None:
        for value in list(self.derived.values()):
            fn = getattr(value, "truncate_to_tokens", None)
            if callable(fn):
                try:
                    fn(int(self.offset or 0))
                except Exception:
                    self.derived.clear()
                    return

    # ── indexer-key side (called by the attention layer each step) ──
    # Physical growth step for the raw index lane (same policy as the K/V
    # side): the lane used to be re-concatenated at its exact length on every
    # step, an O(context) copy per token per sparse layer (13.6 MB at 26k
    # tokens for a 131-wide F32 payload).  ``_idx_offset`` is the LOGICAL
    # length; ``idx_keys.shape[2]`` may exceed it.  Every reader goes through
    # ``state``/``update_index`` (both slice to the logical extent) or
    # ``offset``; the record validator accepts a longer physical backing.
    idx_step = 256

    def update_index(self, idx_k: mx.array) -> mx.array:
        """Append this step's idx_k [B, H, T, D] and return the index history.

        Returns idx_keys sliced to the CURRENT KV offset. The attention forward
        calls cache.update_and_fetch(k, v) BEFORE the indexer (upstream
        ordering), so self.offset is already the post-append length and this
        slice equals the full appended idx history -> Sk matches SDPA's K.
        Keeps update_index, the indexer scoring, and `state` serialization all
        consistent on self.offset (the 'return full' variant desynced
        serialization and broke coherence).
        """
        prev = int(self._idx_offset)
        n = int(idx_k.shape[2])
        if self.idx_keys is None or prev + n > int(self.idx_keys.shape[2]):
            B, H, _, D = idx_k.shape
            n_steps = (self.idx_step + n - 1) // self.idx_step
            grown = mx.zeros((B, H, n_steps * self.idx_step, D), dtype=idx_k.dtype)
            if self.idx_keys is None:
                self.idx_keys = grown
            else:
                if prev % self.idx_step != 0:
                    self.idx_keys = self.idx_keys[..., :prev, :]
                self.idx_keys = mx.concatenate([self.idx_keys, grown], axis=2)
        self.idx_keys[..., prev : prev + n, :] = idx_k
        self._idx_offset = prev + n
        if self.offset and self._idx_offset != int(self.offset):
            # The attention layer appends K/V for exactly these rows before
            # the indexer runs; any other relation means the raw lane and
            # the K/V lane have desynchronised.  Fail closed: a slice past
            # the initialized extent would expose unwritten rows.
            raise ValueError(
                "sparse index lane desynchronised from K/V: "
                f"idx_offset={self._idx_offset} kv_offset={int(self.offset)}"
            )
        logical = int(self.offset) if self.offset else self._idx_offset
        return self.idx_keys[..., :logical, :]

    # ── serialization: expose the 3-tensor slice the disk tiers pack ──
    @property
    def state(self):  # type: ignore[override]
        k, v = super().state
        idx = None if self.idx_keys is None else self.idx_keys[..., : self.offset, :]
        return k, v, idx

    @state.setter
    def state(self, v):
        self.derived = {}
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

        K/V and index history keep the same logical length. Preserve the raw
        index backing just like KVCache preserves its K/V capacity: native MTP
        trims unverified chain rows every cycle, then overwrites that tail with
        confirmed rows. Shrinking the physical lane would force the next
        update_index to concatenate the full prefix again. All readers and
        serialization still slice to offset; stale tail rows are never visible.
        Returns the number actually trimmed.
        """
        trimmed = super().trim(n)
        if self.__dict__.get("_idx_keys") is not None and trimmed:
            self._idx_offset = self.offset
        if trimmed:
            self._truncate_derived()
        return trimmed

    # ── metadata without exposing a lane ──
    @property
    def nbytes(self):  # type: ignore[override]
        keys, values = self.__dict__.get("_keys"), self.__dict__.get("_values")
        return 0 if keys is None else keys.nbytes + values.nbytes

    def empty(self):  # type: ignore[override]
        return self.__dict__.get("_keys") is None

    def has_index_lane(self) -> bool:
        return self.__dict__.get("_idx_keys") is not None

    def __copy__(self):
        # A shallow copy shares the lane buffers: neither object owns them alone any more.
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__["_lane_excl"] = [False, False, False]
        self.__dict__["_lane_excl"] = [False, False, False]
        return new

    def __deepcopy__(self, memo):
        import copy

        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        new.__dict__["_lane_excl"] = [False, False, False]
        self.__dict__["_lane_excl"] = [False, False, False]
        return new

    # ── fused attention appends (models/minimax_m3/fused_attn.py) ──
    def fused_lanes(self):
        """(keys, values, idx_keys) lane buffers, full capacity, without marking them shared: for the fused kernels,
        which read rows < offset + new rows only and write new rows only while ``fused_reserve`` says they may."""
        d = self.__dict__
        return d.get("_keys"), d.get("_values"), d.get("_idx_keys")

    def fused_reserve(self, n: int, kv_heads: int, head_dim: int, k_dtype, v_dtype, idx_dim: int, idx_dtype) -> bool:
        """Capacity for ``n`` more rows on every lane (the stock growth: K / V in steps of ``step``, the index lane in
        steps of ``idx_step``; existing lanes keep their dtypes) and whether those rows may be written in place: True
        only while all three lane buffers are this cache's alone. A buffer this call allocates is; a lane read or
        assigned from outside is not until ``fused_append`` re-owns it."""
        prev = int(self.offset)
        if int(self._idx_offset) != prev and self.__dict__.get("_idx_keys") is not None:
            raise ValueError(
                f"sparse index lane desynchronised from K/V: idx_offset={self._idx_offset} kv_offset={prev}")
        d = self.__dict__
        excl = d.setdefault("_lane_excl", [False, False, False])
        keys, values = d.get("_keys"), d.get("_values")
        if keys is None or prev + n > int(keys.shape[2]):
            n_steps = (self.step + n - 1) // self.step
            new_k = mx.zeros((1, kv_heads, n_steps * self.step, head_dim), keys.dtype if keys is not None else k_dtype)
            new_v = mx.zeros((1, kv_heads, n_steps * self.step, head_dim),
                             values.dtype if values is not None else v_dtype)
            if keys is not None:
                if prev % self.step != 0:
                    keys, values = keys[..., :prev, :], values[..., :prev, :]
                keys, values = mx.concatenate([keys, new_k], axis=2), mx.concatenate([values, new_v], axis=2)
            else:
                keys, values = new_k, new_v
            d["_keys"], d["_values"] = keys, values
            excl[0] = excl[1] = True
        idx = d.get("_idx_keys")
        if idx is None or prev + n > int(idx.shape[2]):
            n_steps = (self.idx_step + n - 1) // self.idx_step
            grown = mx.zeros((1, 1, n_steps * self.idx_step, idx_dim), idx.dtype if idx is not None else idx_dtype)
            if idx is None:
                idx = grown
            else:
                if prev % self.idx_step != 0:
                    idx = idx[..., :prev, :]
                idx = mx.concatenate([idx, grown], axis=2)
            d["_idx_keys"] = idx
            excl[2] = True
        return bool(excl[0] and excl[1] and excl[2])

    def fused_append(self, k: mx.array, v: mx.array, idx_k: mx.array) -> None:
        """Write rows [offset, offset + n) of every lane with MLX slice updates (a shared buffer is copied first, so
        no other holder sees the write); afterwards every lane buffer is this cache's own. The offset moves in
        ``fused_commit``."""
        prev = int(self.offset)
        n = int(k.shape[2])
        d = self.__dict__
        d["_keys"][..., prev:prev + n, :] = k
        d["_values"][..., prev:prev + n, :] = v
        d["_idx_keys"][..., prev:prev + n, :] = idx_k
        d["_lane_excl"] = [True, True, True]

    def fused_commit(self, n: int, after: mx.array | None = None) -> None:
        """Advance every lane's logical length by the ``n`` rows the fused attention wrote (in lockstep).

        ``after``: an output of a launch that ran behind a memory barrier following the in-place writes (the fused
        attention passes its MSA output). The lanes are re-issued as ``mx.depends(lanes, after)`` (same buffers), so
        every later reader of a lane -- a growth copy, a slice update, a ``state`` view -- is encoded after that launch
        and sees the rows. Without it MLX, which does not know a kernel wrote the lane, may encode such a reader before
        the write lands (seen: a lazily chained decode step's growth copy lost the previous steps' rows)."""
        self.offset = int(self.offset) + int(n)
        self._idx_offset = self.offset
        if after is not None:
            d = self.__dict__
            lanes = [d.get("_keys"), d.get("_values"), d.get("_idx_keys")]
            if all(lane is not None for lane in lanes):
                d["_keys"], d["_values"], d["_idx_keys"] = mx.depends(lanes, [after])


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
        self.derived: dict = {}  # never populated on the batched cache

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
    if freq is None and not isinstance(tc, dict):
        # ModelArgs (Model.make_cache) carries the per-layer list flattened. Without
        # this, M3.1 (sparse on every layer) fell back to the MoE layout below and
        # gave layers 0-2 a plain KVCache, so their indexer lost its keys on decode.
        freq = getattr(tc, "sparse_attention_freq", None)
    if freq is None:
        moe = tc.get("moe_layer_freq") if isinstance(tc, dict) else getattr(tc, "moe_layer_freq", None)
        freq = moe if moe is not None else [0, 0, 0] + [1] * (n_layers - 3)
    return [MiniMaxM3SparseCache() if freq[i] else KVCache() for i in range(n_layers)]

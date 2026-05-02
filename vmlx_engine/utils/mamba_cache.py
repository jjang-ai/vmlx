# SPDX-License-Identifier: Apache-2.0
"""
BatchMambaCache and cache patching for hybrid model continuous batching.

This module enables continuous batching for models that use non-KVCache layers
(MambaCache, ArraysCache) -- primarily hybrid SSM+attention architectures like
Qwen3.5-VL, Mamba, and Jamba.

PROBLEM
-------
mlx-lm's BatchGenerator assumes all cache layers are KVCache with ``extract()``
and ``merge()`` methods. But MambaCache (extends ArraysCache) lacks ``extract()``,
and ``_make_cache()`` / ``_merge_caches()`` don't handle non-KV types. This causes
crashes when batching hybrid models.

SOLUTION
--------
1. ``BatchMambaCache``: Batch-aware wrapper around MambaCache with:
   - ``extract(idx)`` -- extract single request's state from batch
   - ``merge(caches)`` -- concatenate multiple MambaCache into batch
   - ``filter(batch_indices)`` -- filter batch AND left_padding together

2. ``patch_mlx_lm_for_mamba()``: Monkey-patches mlx-lm's generate module:
   - ``_make_cache`` -- handles MambaCache -> BatchMambaCache conversion
   - ``_merge_caches`` -- handles all cache types including QuantizedKVCache,
     MambaCache, ArraysCache, RotatingKVCache, and CacheList (recursive)

3. ``ensure_mamba_support()`` -- idempotent entry point (called by scheduler)

INTEGRATION
-----------
Called from ``MLLMScheduler.__init__()`` when ``_is_hybrid_model()`` detects
non-KVCache layers. Must be called before any BatchGenerator usage.

The patched ``_merge_caches`` also handles QuantizedKVCache -> dequantize ->
BatchKVCache merging, which is needed when KV cache quantization is active.
"""

import logging
from typing import List, Optional

import mlx.core as mx

# MambaCache removed in mlx-lm >= 0.30.6, replaced by ArraysCache.
# If upstream adds batch methods to ArraysCache, this wrapper may need updates.
try:
    from mlx_lm.models.cache import MambaCache

    HAS_MAMBA_CACHE = True
except ImportError:
    # Fallback for mlx-lm >= 0.30.6 where MambaCache was removed
    from mlx_lm.models.cache import ArraysCache as MambaCache

    HAS_MAMBA_CACHE = False

logger = logging.getLogger(__name__)

# TurboQuantKVCache class name for isinstance-free detection.
_TQ_CLASS_NAME = "TurboQuantKVCache"


def _is_kv_like(c) -> bool:
    """Check if cache is KVCache-compatible (KVCache or TurboQuantKVCache)."""
    from mlx_lm.models.cache import KVCache
    return isinstance(c, KVCache) or type(c).__name__ == _TQ_CLASS_NAME


# A3-BUG-004: warn-once per unknown kwarg in BatchMambaCache.prepare so we
# notice when upstream mlx-lm grows new prepare() params we don't yet handle.
_seen_unknown_prepare_kwargs: set = set()


def _warn_unknown_prepare_kwarg(name: str) -> None:
    if name in _seen_unknown_prepare_kwargs:
        return
    _seen_unknown_prepare_kwargs.add(name)
    logger.warning(
        "BatchMambaCache.prepare received unknown kwarg %r — silently dropped. "
        "Upstream mlx-lm may have added a new ArraysCache.prepare parameter "
        "that vMLX needs to backport. Re-verify after the next mlx-lm bump.",
        name,
    )


class BatchMambaCache(MambaCache):
    """
    Batch-aware MambaCache for continuous batching.

    This extends MambaCache to support batch operations required by
    mlx-lm's BatchGenerator, specifically the `extract` method.
    """

    def __init__(self, size: int = 2, left_padding: Optional[List[int]] = None):
        """
        Initialize BatchMambaCache.

        Args:
            size: Number of arrays in the cache (passed to ArraysCache)
            left_padding: Amount of left padding for each sequence in batch
        """
        # MambaCache.__init__ dropped the `size` param in mlx-lm 0.30.5+
        # but ArraysCache.__init__ still accepts it. Try the full signature
        # first, fall back to left_padding only.
        try:
            super().__init__(size, left_padding=left_padding)
        except TypeError:
            super().__init__(left_padding=left_padding)
            # Manually set cache list to the requested size
            if not self.cache or len(self.cache) != size:
                self.cache = [None] * size
        self._batch_size = len(left_padding) if left_padding else 0
        # Ensure `lengths` exists on both 0.31.1 (parent has no field) and
        # 0.31.2 (parent sets it in __new__). Harmless overwrite-to-None on
        # 0.31.2 because __new__ already set it to None.
        if not hasattr(self, "lengths"):
            self.lengths = None

    # ------------------------------------------------------------------
    # Backport of ArraysCache.lengths / prepare(lengths=) / advance(N) /
    # finalize() from mlx-lm 0.31.2. These methods enable per-sequence
    # length tracking for batched SSM (variable-length sequences in one
    # batch). On mlx-lm 0.31.2+ parents these methods override the parent
    # implementation with identical semantics — harmless. On 0.31.1 parents
    # these methods are the ONLY implementation available.
    #
    # Upstream reference: /tmp/mlx_lm_latest/mlx_lm/models/cache.py:594-696
    # Hybrid models that call these (in mlx-lm 0.31.2): kimi_linear, plamo2,
    # mamba2, nemotron_h, lfm2_moe, qwen3_5, qwen3_next, lfm2, granitemoehybrid.
    # ------------------------------------------------------------------

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None, **kwargs):
        """Set per-sequence lengths and/or left_padding for batched SSM masking.

        Called once at the start of batched prefill. `lengths` is a per-sequence
        array of remaining-token counts; `left_padding` is a per-sequence array
        of padding offsets. `right_padding` is accepted for API compat but
        ignored by ArraysCache-derived caches (BatchKVCache uses it).

        Forward-compat audit trail (A3-BUG-004): if upstream mlx-lm grows new
        ``prepare()`` keyword arguments, log a one-time warning per kwarg name
        so we notice silent drops at the next bump rather than discovering
        them via wrong outputs months later.
        """
        if lengths is not None:
            self.lengths = mx.array(lengths)
        if left_padding is not None:
            # Track delta relative to existing left_padding, mirroring
            # upstream BatchKVCache.prepare semantics.
            lp = mx.array(left_padding)
            if self.left_padding is not None:
                self.left_padding = self.left_padding + lp
            else:
                self.left_padding = lp
        if kwargs:
            for k in kwargs:
                _warn_unknown_prepare_kwarg(k)

    def advance(self, N: int) -> None:
        """Decrement per-sequence `lengths` and `left_padding` by N tokens.

        Called after each forward step so `make_mask(N)` reflects remaining
        tokens. Only touches `lengths` and `left_padding`; the state arrays
        in `self.cache` are untouched.

        A3-BUG-003 (perf): use in-place ``-=`` to match upstream
        ArraysCache.advance and avoid an N-allocation-per-step hot-path
        cost on hybrid models (~40 layers × 1 alloc/step ≈ 40 allocs/step
        averted on Nemotron Cascade 2).
        """
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def finalize(self) -> None:
        """Clear per-sequence length tracking at end of prefill/generation."""
        self.lengths = None
        self.left_padding = None

    def filter(self, batch_indices) -> None:
        """Filter batch to keep only specified indices.

        Inline None-safe re-implementation of ArraysCache.filter(). We can't
        safely delegate to `super().filter()` because mlx-lm 0.31.1's
        `ArraysCache.filter` does `[c[batch_indices] for c in self.cache]`
        with no None guard — it crashes on empty/partial caches. Upstream
        0.31.2 added the guard. Doing it inline works identically on both
        versions, and also gives us one pass to carry `lengths` and
        `left_padding` through the shrink (upstream still doesn't filter
        `left_padding`).
        """
        self.cache = [
            c[batch_indices] if c is not None else None for c in self.cache
        ]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]

    def extract(self, idx: int) -> MambaCache:
        """
        Extract a single cache from the batch.

        Args:
            idx: Index of the sequence to extract

        Returns:
            A new MambaCache with the extracted state
        """
        num_arrays = len(self.cache) if self.cache else 2
        try:
            cache = MambaCache(num_arrays)
        except TypeError:
            # MambaCache no longer accepts size param
            cache = MambaCache()
            cache.cache = [None] * num_arrays
        # Extract the state arrays for this index
        cache.cache = [
            mx.contiguous(c[idx : idx + 1]) if c is not None else None
            for c in self.cache
        ]
        cache.left_padding = None  # Single sequence, no batch padding
        # Carry a single-element `lengths` for the extracted sequence so the
        # returned cache can continue to participate in batched SSM masking
        # if it's later re-merged. Upstream ArraysCache.extract doesn't do
        # this (lengths dropped on extract), but vMLX re-merges extracted
        # caches through BatchMambaCache.merge so preserving it is cheaper
        # than re-deriving.
        if getattr(self, "lengths", None) is not None:
            try:
                cache.lengths = self.lengths[idx : idx + 1]
            except Exception:
                cache.lengths = None
        else:
            cache.lengths = None
        return cache

    def extend(self, other: "BatchMambaCache") -> None:
        """Extend this batch cache with another batch's SSM states.

        Concatenates each state array along the batch dimension.
        Mirrors BatchKVCache.extend() for continuous batching support.
        """
        if not self.cache or not other.cache:
            return
        merged = []
        for sc, oc in zip(self.cache, other.cache):
            if sc is not None and oc is not None:
                merged.append(mx.concatenate([sc, oc], axis=0))
            elif sc is not None:
                merged.append(sc)
            elif oc is not None:
                merged.append(oc)
            else:
                merged.append(None)
        self.cache = merged
        self._batch_size = (self._batch_size or 0) + (other._batch_size or 0)
        # Reset left_padding — merged batch uses no-op masking
        self.left_padding = None
        # Concatenate per-sequence lengths if both sides have them; otherwise
        # drop (no-op mask on the merged result).
        self_len = getattr(self, "lengths", None)
        other_len = getattr(other, "lengths", None)
        if self_len is not None and other_len is not None:
            try:
                self.lengths = mx.concatenate([self_len, other_len], axis=0)
            except Exception:
                self.lengths = None
        else:
            self.lengths = None

    @classmethod
    def merge(cls, caches: List[MambaCache]) -> "BatchMambaCache":
        """
        Merge multiple MambaCache objects into a BatchMambaCache.

        Args:
            caches: List of MambaCache objects to merge

        Returns:
            A new BatchMambaCache containing all caches
        """
        if not caches:
            return cls(size=2, left_padding=[])

        # Get the structure from the first cache
        batch_size = len(caches)
        num_arrays = len(caches[0].cache) if caches[0].cache else 2

        # Merged caches from prefill don't need padding masks. If decode phase
        # needs SSM padding, this would need to be reconstructed.
        # Setting [0]*N creates a no-op mask (all True) that still causes
        # shape mismatches after filter() shrinks the batch — ArraysCache.filter()
        # doesn't filter left_padding, so make_mask() would return a mask with
        # stale batch dimension. With left_padding=None, make_mask() returns None
        # and SSM layers skip masking entirely (correct for decode phase).
        merged_cache = cls(size=num_arrays, left_padding=None)
        merged_cache._batch_size = batch_size

        # Merge each array in the cache
        merged_cache.cache = []

        try:
            for i in range(num_arrays):
                raw = [c.cache[i] for c in caches]
                non_none = [a for a in raw if a is not None]
                if non_none:
                    # Pad None entries with zeros matching shape of non-None entries
                    # to preserve batch dimension alignment for extract()
                    ref_shape = non_none[0].shape
                    padded = []
                    for a in raw:
                        if a is not None:
                            padded.append(a)
                        else:
                            padded.append(mx.zeros(ref_shape))
                    merged_cache.cache.append(mx.concatenate(padded, axis=0))
                else:
                    merged_cache.cache.append(None)
        except (MemoryError, RuntimeError) as e:
            logger.warning(
                "Out of memory during SSM state merge — try reducing "
                "batch size or sequence length. (%s)", e
            )
            raise

        # Carry per-sequence `lengths` through the merge if every source
        # cache has it set. If any source is missing lengths, drop the field
        # (no-op mask on the merged result). Upstream 0.31.2 ArraysCache
        # doesn't implement merge() at this level — BatchMambaCache is the
        # authoritative merger for vMLX continuous batching.
        try:
            src_lengths = [getattr(c, "lengths", None) for c in caches]
            if all(l is not None for l in src_lengths):
                merged_cache.lengths = mx.concatenate(src_lengths, axis=0)
            else:
                merged_cache.lengths = None
        except Exception:
            merged_cache.lengths = None

        return merged_cache


def patch_mlx_lm_for_mamba():
    """
    Patch mlx-lm to support MambaCache in BatchGenerator.

    This modifies the _make_cache function to handle MambaCache by
    converting it to BatchMambaCache.
    """
    import importlib

    gen_module = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import (
        KVCache,
        ArraysCache,
        RotatingKVCache,
        CacheList,
    )

    # QuantizedKVCache import for safety handling
    try:
        from mlx_lm.models.cache import QuantizedKVCache as _QuantizedKVCache
    except ImportError:
        _QuantizedKVCache = None

    # MambaCache was removed in mlx-lm 0.30.6
    try:
        from mlx_lm.models.cache import MambaCache as OrigMambaCache
    except ImportError:
        OrigMambaCache = ArraysCache  # Fallback
    from mlx_lm.generate import BatchKVCache, BatchRotatingKVCache

    # Patch ArraysCache.make_mask to accept **kwargs.
    # mlx-lm's base.py calls make_mask(N, return_array=..., window_size=...)
    # on ALL cache layers, but ArraysCache.make_mask only takes (self, N).
    # Without this patch, hybrid SSM models crash on every request.
    _orig_ac_make_mask = ArraysCache.make_mask
    if 'kwargs' not in _orig_ac_make_mask.__code__.co_varnames:
        def _ac_make_mask_compat(self, N: int, **kwargs):
            return _orig_ac_make_mask(self, N)
        ArraysCache.make_mask = _ac_make_mask_compat
        logger.debug("Patched ArraysCache.make_mask to accept **kwargs")

    # Store original function
    _original_make_cache = gen_module._make_cache

    def _patched_make_cache(model, left_padding, max_kv_size=None):
        """
        Convert a list of regular caches into their corresponding
        batch-aware caches, with support for MambaCache.

        Args:
            model: The model to create cache for
            left_padding: Left padding for batch
            max_kv_size: Maximum KV cache size (mlx-lm 0.30.6+)
        """

        def to_batch_cache(c):
            if _is_kv_like(c):
                return BatchKVCache(left_padding)
            elif _QuantizedKVCache is not None and isinstance(c, _QuantizedKVCache):
                # QuantizedKVCache → BatchKVCache (dequantize at batch boundary)
                return BatchKVCache(left_padding)
            elif isinstance(c, OrigMambaCache):
                # Handle MambaCache/ArraysCache -> BatchMambaCache
                num_arrays = len(c.cache) if c.cache else 2
                return BatchMambaCache(size=num_arrays, left_padding=left_padding)
            elif isinstance(c, RotatingKVCache):
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        elif max_kv_size is not None:
            # mlx-lm 0.30.6+: Use rotating cache with max_kv_size
            return [
                BatchRotatingKVCache(max_kv_size, left_padding) for _ in model.layers
            ]
        else:
            return [BatchKVCache(left_padding) for _ in model.layers]

    # Patch the module
    gen_module._make_cache = _patched_make_cache

    # Also patch _merge_caches to handle BatchMambaCache
    _original_merge_caches = gen_module._merge_caches

    def _dequantize_layer(layer_cache):
        """Dequantize a QuantizedKVCache layer to KVCache for merging."""
        if layer_cache.keys is None:
            return KVCache()
        kv = KVCache()
        kv.keys = mx.dequantize(
            layer_cache.keys[0], layer_cache.keys[1],
            layer_cache.keys[2], layer_cache.group_size, layer_cache.bits,
        )
        kv.values = mx.dequantize(
            layer_cache.values[0], layer_cache.values[1],
            layer_cache.values[2], layer_cache.group_size, layer_cache.bits,
        )
        kv.offset = layer_cache.offset
        return kv

    def _patched_merge_caches(caches):
        """Merge caches with support for all cache types.

        Empty-input fast path: when ``caches`` is empty OR ``caches[0]`` is
        an empty list, return an empty list. This case is hit by mlx-lm
        0.31.2's ``PromptProcessingBatch.empty()`` constructor which calls
        ``_merge_caches([])`` to initialise an empty prompt batch — without
        the early return the next line `range(len(caches[0]))` raises
        IndexError. Discovered during Phase 4 live test on Nemotron Cascade
        2 30B JANG_2L (audit 2026-04-08, ISSUE-A3-003).
        """
        if not caches or not caches[0]:
            return []
        batch_cache = []
        for i in range(len(caches[0])):
            layer_cache = caches[0][i]
            if _QuantizedKVCache is not None and isinstance(layer_cache, _QuantizedKVCache):
                # Dequantize all layers before merging as regular KVCache
                dequantized = [_dequantize_layer(c[i]) for c in caches]
                cache = BatchKVCache.merge(dequantized)
            elif _is_kv_like(layer_cache):
                # TurboQuantKVCache after compress() has keys=None —
                # BatchKVCache.merge needs actual tensors. Convert TQ
                # layers to KVCache via .state (decoded float16 buffers).
                to_merge = []
                for c in caches:
                    layer = c[i]
                    if (hasattr(layer, 'keys') and layer.keys is None
                            and getattr(layer, '_joined_k', None) is not None):
                        # TQ compressed — extract decoded state as KVCache
                        kv = KVCache()
                        state = layer.state
                        if isinstance(state, tuple) and len(state) == 2:
                            kv.keys, kv.values = state
                            kv.offset = layer.offset
                        to_merge.append(kv)
                    else:
                        to_merge.append(layer)
                cache = BatchKVCache.merge(to_merge)
            elif isinstance(layer_cache, RotatingKVCache):
                cache = BatchRotatingKVCache.merge([c[i] for c in caches])
            elif isinstance(layer_cache, (OrigMambaCache, BatchMambaCache)):
                cache = BatchMambaCache.merge([c[i] for c in caches])
            elif isinstance(layer_cache, ArraysCache):
                # Generic ArraysCache: merge by concatenating arrays
                num_arrays = len(layer_cache.cache) if layer_cache.cache else 0
                merged = type(layer_cache)(num_arrays)
                merged.cache = []
                for j in range(num_arrays):
                    arrays = [
                        c[i].cache[j]
                        for c in caches
                        if c[i].cache[j] is not None
                    ]
                    if arrays:
                        merged.cache.append(mx.concatenate(arrays, axis=0))
                    else:
                        merged.cache.append(None)
                cache = merged
            elif isinstance(layer_cache, CacheList):
                # CacheList: merge each sub-cache independently
                num_sub = len(layer_cache.caches)
                merged_subs = []
                for j in range(num_sub):
                    # Flatten and merge recursively
                    flat = [c[i].caches[j] for c in caches]
                    sub_merged = _patched_merge_caches(
                        [[sc] for sc in flat]
                    )[0]
                    merged_subs.append(sub_merged)
                cache = CacheList(*merged_subs)
            elif type(layer_cache).__name__ in (
                "DeepseekV4Cache", "PoolQuantizedV4Cache"
            ):
                # DSV4 caches own compressor + indexer state buffers that
                # cannot be reconstructed from per-batch slices. For single
                # batch (len(caches)==1) the merge is a no-op — pass through.
                # Multi-batch DSV4 isn't supported yet (the cache class
                # itself raises on batched history); we surface a clearer
                # error if anyone tries.
                if len(caches) == 1:
                    cache = layer_cache
                else:
                    raise ValueError(
                        f"DSV4 cache type {type(layer_cache).__name__} cannot "
                        "merge across multiple batches. Restart the engine "
                        "without --continuous-batching, or send requests "
                        "serially (max_num_seqs=1)."
                    )
            else:
                raise ValueError(
                    f"{type(layer_cache)} does not yet support batching with history"
                )
            batch_cache.append(cache)
        return batch_cache

    gen_module._merge_caches = _patched_merge_caches

    logger.info("Patched mlx-lm for MambaCache batching support")


# Auto-patch when module is imported
_patched = False
_patch_lock = __import__("threading").Lock()


def ensure_mamba_support():
    """Ensure MambaCache batching support is enabled."""
    global _patched
    if _patched:
        return
    with _patch_lock:
        if not _patched:
            patch_mlx_lm_for_mamba()
            _patched = True

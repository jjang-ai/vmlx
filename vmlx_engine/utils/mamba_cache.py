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

    def filter(self, batch_indices) -> None:
        """Filter batch to keep only specified indices.

        Overrides ArraysCache.filter() to also filter left_padding.
        Without this, left_padding retains the original batch_size after
        filtering, causing make_mask() to return a mask with stale dimensions
        that triggers shape mismatches in SSM layers during batched generation.

        This matches BatchKVCache.filter() which also filters its left_padding.
        """
        super().filter(batch_indices)
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
        """Merge caches with support for all cache types."""
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

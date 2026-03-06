# SPDX-License-Identifier: Apache-2.0
"""
Cache type detection and classification for vmlx-engine.

Provides a unified system for identifying and working with all mlx-lm
cache types: KVCache, RotatingKVCache, QuantizedKVCache, MambaCache,
ArraysCache, and CacheList.
"""

import logging
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """All cache types supported by mlx-lm."""

    KV_CACHE = "kv_cache"
    ROTATING_KV_CACHE = "rotating_kv_cache"
    QUANTIZED_KV_CACHE = "quantized_kv_cache"
    MAMBA_CACHE = "mamba_cache"
    ARRAYS_CACHE = "arrays_cache"
    CACHE_LIST = "cache_list"
    UNKNOWN = "unknown"


# Class name -> CacheType mapping for fast detection
_CLASS_NAME_MAP = {
    "KVCache": CacheType.KV_CACHE,
    "BatchKVCache": CacheType.KV_CACHE,
    "RotatingKVCache": CacheType.ROTATING_KV_CACHE,
    "BatchRotatingKVCache": CacheType.ROTATING_KV_CACHE,
    "QuantizedKVCache": CacheType.QUANTIZED_KV_CACHE,
    "MambaCache": CacheType.MAMBA_CACHE,
    "BatchMambaCache": CacheType.MAMBA_CACHE,
    "ArraysCache": CacheType.ARRAYS_CACHE,
    "CacheList": CacheType.CACHE_LIST,
}

# Positional caches store per-token KV data that can be sliced by position
_POSITIONAL_TYPES = {
    CacheType.KV_CACHE,
    CacheType.ROTATING_KV_CACHE,
    CacheType.QUANTIZED_KV_CACHE,
}

# Cumulative caches store rolling state that represents all processed tokens
_CUMULATIVE_TYPES = {
    CacheType.MAMBA_CACHE,
    CacheType.ARRAYS_CACHE,
}


def detect_cache_type(cache_obj: Any) -> CacheType:
    """
    Detect the cache type of a cache object by class name and structure.

    Args:
        cache_obj: A cache object from mlx-lm

    Returns:
        CacheType enum value
    """
    if cache_obj is None:
        return CacheType.UNKNOWN

    class_name = type(cache_obj).__name__

    # Fast path: exact class name match
    if class_name in _CLASS_NAME_MAP:
        return _CLASS_NAME_MAP[class_name]

    # Check inheritance chain
    for cls in type(cache_obj).__mro__:
        if cls.__name__ in _CLASS_NAME_MAP:
            return _CLASS_NAME_MAP[cls.__name__]

    # Structure-based detection (fallback)
    if hasattr(cache_obj, "caches") and hasattr(cache_obj, "__iter__"):
        return CacheType.CACHE_LIST

    if hasattr(cache_obj, "max_size") and hasattr(cache_obj, "keys"):
        return CacheType.ROTATING_KV_CACHE

    if hasattr(cache_obj, "keys_quantized") or hasattr(cache_obj, "quantized"):
        return CacheType.QUANTIZED_KV_CACHE

    if hasattr(cache_obj, "keys") and hasattr(cache_obj, "values"):
        return CacheType.KV_CACHE

    if hasattr(cache_obj, "cache") and isinstance(
        getattr(cache_obj, "cache", None), list
    ):
        # MambaCache/ArraysCache have a .cache list attribute
        return CacheType.MAMBA_CACHE

    return CacheType.UNKNOWN


def detect_cache_type_from_state(
    state: Any, class_name: str = ""
) -> CacheType:
    """
    Detect cache type from an extracted state and class name.

    Used when working with extracted cache states (dicts with 'state' key)
    rather than live cache objects.

    Args:
        state: The cache state (tuple of tensors, list of arrays, etc.)
        class_name: Original cache class name (most reliable)

    Returns:
        CacheType enum value
    """
    # Class name is the most reliable signal
    # Match longest names first to avoid "KVCache" matching before "RotatingKVCache"
    if class_name:
        for name in sorted(_CLASS_NAME_MAP, key=len, reverse=True):
            if name in class_name:
                return _CLASS_NAME_MAP[name]

    if not state:
        return CacheType.UNKNOWN

    # Structure-based detection
    if isinstance(state, (tuple, list)):
        if len(state) == 2:
            first = state[0]
            if hasattr(first, "shape"):
                if len(first.shape) == 4:
                    # 4D tensor: (batch, heads, seq, dim) -> KV cache
                    return CacheType.KV_CACHE
                elif len(first.shape) == 3:
                    # 3D tensor: could be cumulative state
                    return CacheType.MAMBA_CACHE

        if len(state) == 4:
            # Could be quantized: (keys_q, values_q, scales, zero_points)
            first = state[0]
            if hasattr(first, "shape") and len(first.shape) == 4:
                return CacheType.QUANTIZED_KV_CACHE

    return CacheType.UNKNOWN


def is_positional_cache(cache_type: CacheType) -> bool:
    """Check if cache type stores position-indexed KV data (sliceable)."""
    return cache_type in _POSITIONAL_TYPES


def is_cumulative_cache(cache_type: CacheType) -> bool:
    """Check if cache type stores cumulative state (not sliceable)."""
    return cache_type in _CUMULATIVE_TYPES


def get_cache_structure_info(cache_obj: Any) -> Dict[str, Any]:
    """
    Get structural information about a cache object for debugging.

    Args:
        cache_obj: A cache object from mlx-lm

    Returns:
        Dict with cache structure details
    """
    info: Dict[str, Any] = {
        "class_name": type(cache_obj).__name__,
        "cache_type": detect_cache_type(cache_obj).value,
    }

    if hasattr(cache_obj, "keys") and hasattr(cache_obj.keys, "shape"):
        info["keys_shape"] = list(cache_obj.keys.shape)
    if hasattr(cache_obj, "values") and hasattr(cache_obj.values, "shape"):
        info["values_shape"] = list(cache_obj.values.shape)
    if hasattr(cache_obj, "offset"):
        info["offset"] = cache_obj.offset
    if hasattr(cache_obj, "max_size"):
        info["max_size"] = cache_obj.max_size
    if hasattr(cache_obj, "keep"):
        info["keep"] = cache_obj.keep
    if hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, list):
        info["num_arrays"] = len(cache_obj.cache)
        info["array_shapes"] = [
            list(a.shape) if hasattr(a, "shape") else None
            for a in cache_obj.cache
        ]
    if hasattr(cache_obj, "caches"):
        info["num_sub_caches"] = len(cache_obj.caches)
        info["sub_cache_types"] = [
            detect_cache_type(c).value for c in cache_obj.caches
        ]

    return info

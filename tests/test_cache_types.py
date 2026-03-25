# SPDX-License-Identifier: Apache-2.0
"""Tests for cache type detection and classification."""

from unittest.mock import MagicMock

import pytest

from vmlx_engine.utils.cache_types import (
    CacheType,
    detect_cache_type,
    detect_cache_type_from_state,
    get_cache_structure_info,
    is_cumulative_cache,
    is_positional_cache,
)


# =============================================================================
# Mock cache objects for testing (no real model loading needed)
# =============================================================================


def _make_named_mock(class_name, **attrs):
    """Create a mock with a proper named class for type detection."""
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _make_mock_kv_cache():
    """Create a mock KVCache object."""
    return _make_named_mock(
        "KVCache",
        keys=MagicMock(shape=(1, 8, 128, 64)),
        values=MagicMock(shape=(1, 8, 128, 64)),
        offset=128,
    )


def _make_mock_rotating_kv_cache():
    """Create a mock RotatingKVCache object."""
    return _make_named_mock(
        "RotatingKVCache",
        keys=MagicMock(shape=(1, 8, 256, 64)),
        values=MagicMock(shape=(1, 8, 256, 64)),
        max_size=256,
        keep=32,
        offset=128,
    )


def _make_mock_mamba_cache():
    """Create a mock MambaCache object."""
    return _make_named_mock(
        "MambaCache",
        cache=[MagicMock(shape=(1, 16, 64)), MagicMock(shape=(1, 16, 64))],
    )


def _make_mock_arrays_cache():
    """Create a mock ArraysCache object."""
    return _make_named_mock(
        "ArraysCache",
        cache=[MagicMock(shape=(1, 32)), MagicMock(shape=(1, 32))],
    )


def _make_mock_cache_list():
    """Create a mock CacheList object with sub-caches."""
    cache = _make_named_mock(
        "CacheList",
        caches=[_make_mock_kv_cache(), _make_mock_mamba_cache()],
    )
    cache.__iter__ = lambda self: iter(self.caches)
    return cache


def _make_mock_quantized_kv_cache():
    """Create a mock QuantizedKVCache object."""
    return _make_named_mock(
        "QuantizedKVCache",
        keys=MagicMock(shape=(1, 8, 128, 64)),
        values=MagicMock(shape=(1, 8, 128, 64)),
        keys_quantized=True,
    )


def _make_mock_batch_kv_cache():
    """Create a mock BatchKVCache object."""
    return _make_named_mock(
        "BatchKVCache",
        keys=MagicMock(shape=(4, 8, 128, 64)),
        values=MagicMock(shape=(4, 8, 128, 64)),
    )


# =============================================================================
# Tests
# =============================================================================


class TestDetectCacheType:
    """Tests for detect_cache_type."""

    def test_none_returns_unknown(self):
        assert detect_cache_type(None) == CacheType.UNKNOWN

    def test_kv_cache(self):
        cache = _make_mock_kv_cache()
        assert detect_cache_type(cache) == CacheType.KV_CACHE

    def test_rotating_kv_cache(self):
        cache = _make_mock_rotating_kv_cache()
        assert detect_cache_type(cache) == CacheType.ROTATING_KV_CACHE

    def test_mamba_cache(self):
        cache = _make_mock_mamba_cache()
        assert detect_cache_type(cache) == CacheType.MAMBA_CACHE

    def test_arrays_cache(self):
        cache = _make_mock_arrays_cache()
        assert detect_cache_type(cache) == CacheType.ARRAYS_CACHE

    def test_cache_list(self):
        cache = _make_mock_cache_list()
        assert detect_cache_type(cache) == CacheType.CACHE_LIST

    def test_quantized_kv_cache(self):
        cache = _make_mock_quantized_kv_cache()
        assert detect_cache_type(cache) == CacheType.QUANTIZED_KV_CACHE

    def test_batch_kv_cache(self):
        cache = _make_mock_batch_kv_cache()
        assert detect_cache_type(cache) == CacheType.KV_CACHE

    def test_unknown_object(self):
        obj = _make_named_mock("SomeUnknownCache")
        assert detect_cache_type(obj) == CacheType.UNKNOWN


class TestDetectCacheTypeFromState:
    """Tests for detect_cache_type_from_state."""

    def test_class_name_kv(self):
        assert detect_cache_type_from_state(None, "KVCache") == CacheType.KV_CACHE

    def test_class_name_rotating(self):
        assert (
            detect_cache_type_from_state(None, "RotatingKVCache")
            == CacheType.ROTATING_KV_CACHE
        )

    def test_class_name_mamba(self):
        assert (
            detect_cache_type_from_state(None, "MambaCache") == CacheType.MAMBA_CACHE
        )

    def test_class_name_batch_kv(self):
        assert (
            detect_cache_type_from_state(None, "BatchKVCache") == CacheType.KV_CACHE
        )

    def test_class_name_batch_mamba(self):
        assert (
            detect_cache_type_from_state(None, "BatchMambaCache")
            == CacheType.MAMBA_CACHE
        )

    def test_empty_state_no_class(self):
        assert detect_cache_type_from_state(None, "") == CacheType.UNKNOWN

    def test_4d_tensor_state(self):
        tensor = MagicMock()
        tensor.shape = (1, 8, 128, 64)
        state = (tensor, tensor)
        assert detect_cache_type_from_state(state) == CacheType.KV_CACHE

    def test_3d_tensor_state(self):
        tensor = MagicMock()
        tensor.shape = (1, 16, 64)
        state = (tensor, tensor)
        assert detect_cache_type_from_state(state) == CacheType.MAMBA_CACHE


class TestPositionalCumulative:
    """Tests for is_positional_cache and is_cumulative_cache."""

    def test_kv_is_positional(self):
        assert is_positional_cache(CacheType.KV_CACHE) is True

    def test_rotating_is_positional(self):
        assert is_positional_cache(CacheType.ROTATING_KV_CACHE) is True

    def test_quantized_is_positional(self):
        assert is_positional_cache(CacheType.QUANTIZED_KV_CACHE) is True

    def test_mamba_is_not_positional(self):
        assert is_positional_cache(CacheType.MAMBA_CACHE) is False

    def test_arrays_is_not_positional(self):
        assert is_positional_cache(CacheType.ARRAYS_CACHE) is False

    def test_mamba_is_cumulative(self):
        assert is_cumulative_cache(CacheType.MAMBA_CACHE) is True

    def test_arrays_is_cumulative(self):
        assert is_cumulative_cache(CacheType.ARRAYS_CACHE) is True

    def test_kv_is_not_cumulative(self):
        assert is_cumulative_cache(CacheType.KV_CACHE) is False

    def test_cache_list_is_neither(self):
        assert is_positional_cache(CacheType.CACHE_LIST) is False
        assert is_cumulative_cache(CacheType.CACHE_LIST) is False

    def test_unknown_is_neither(self):
        assert is_positional_cache(CacheType.UNKNOWN) is False
        assert is_cumulative_cache(CacheType.UNKNOWN) is False


class TestGetCacheStructureInfo:
    """Tests for get_cache_structure_info."""

    def test_kv_cache_info(self):
        cache = _make_mock_kv_cache()
        info = get_cache_structure_info(cache)
        assert info["class_name"] == "KVCache"
        assert info["cache_type"] == "kv_cache"
        assert "offset" in info

    def test_rotating_kv_cache_info(self):
        cache = _make_mock_rotating_kv_cache()
        info = get_cache_structure_info(cache)
        assert info["class_name"] == "RotatingKVCache"
        assert info["cache_type"] == "rotating_kv_cache"
        assert info["max_size"] == 256
        assert info["keep"] == 32

    def test_mamba_cache_info(self):
        cache = _make_mock_mamba_cache()
        info = get_cache_structure_info(cache)
        assert info["class_name"] == "MambaCache"
        assert info["cache_type"] == "mamba_cache"
        assert info["num_arrays"] == 2

    def test_cache_list_info(self):
        cache = _make_mock_cache_list()
        info = get_cache_structure_info(cache)
        assert info["class_name"] == "CacheList"
        assert info["cache_type"] == "cache_list"
        assert info["num_sub_caches"] == 2

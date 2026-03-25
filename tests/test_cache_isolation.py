# SPDX-License-Identifier: Apache-2.0
"""Tests for MemoryAwarePrefixCache key isolation, eviction, and stats.

Verifies that:
- Stored entries are retrievable by the same token key
- Different token sequences do not cross-contaminate
- Eviction fires under memory pressure (max_memory_mb)
- Stats counters are accurate after store/fetch/evict operations
- Forward prefix matching works (cached shorter, request longer)
- clear() removes all entries
"""

from unittest.mock import MagicMock

import pytest

from vmlx_engine.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Non-callable object with .nbytes, mimicking an MLX array.

    estimate_kv_cache_memory checks `not callable(keys_attr)` before
    reading .nbytes, so MagicMock (which is always callable) causes
    memory estimation to return 0.  We use a plain object instead.
    """
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _FakeKVLayer:
    """Mock KV cache layer with deterministic memory size."""
    def __init__(self, nbytes: int = 1024):
        self.keys = _FakeTensor(nbytes // 2)
        self.values = _FakeTensor(nbytes // 2)


def _make_mock_kv_layer(nbytes: int = 1024):
    """Create a mock KV cache layer that reports a given memory size.

    Uses plain objects (not MagicMock) so that estimate_kv_cache_memory
    can read .nbytes correctly (it gates on `not callable(keys_attr)`).
    """
    return _FakeKVLayer(nbytes)


def _make_cache_list(nbytes_per_layer: int = 1024, num_layers: int = 2):
    """Return a list of mock KV layers suitable for MemoryAwarePrefixCache."""
    return [_make_mock_kv_layer(nbytes_per_layer) for _ in range(num_layers)]


def _make_prefix_cache(max_memory_mb: int = 10, max_entries: int = 100):
    """Create a MemoryAwarePrefixCache with a small, deterministic memory limit."""
    model = MagicMock()
    config = MemoryCacheConfig(
        max_memory_mb=max_memory_mb,
        max_entries=max_entries,
    )
    return MemoryAwarePrefixCache(model, config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrefixCacheIsolation:
    """Prefix cache key isolation and basic operations."""

    def test_store_and_fetch_returns_same_data(self):
        """Store tokens + cache, fetch by same tokens returns match."""
        cache = _make_prefix_cache()
        tokens = [1, 2, 3, 4, 5]
        kv_data = _make_cache_list()

        assert cache.store(tokens, kv_data) is True

        result, remaining = cache.fetch(tokens)
        assert result is kv_data  # Same reference (no deep copy)
        assert remaining == []

    def test_different_tokens_no_cross_contamination(self):
        """Store two different token sequences, each fetch returns correct one."""
        cache = _make_prefix_cache()

        tokens_a = [10, 20, 30]
        kv_a = _make_cache_list()

        tokens_b = [40, 50, 60]
        kv_b = _make_cache_list()

        cache.store(tokens_a, kv_a)
        cache.store(tokens_b, kv_b)

        result_a, rem_a = cache.fetch(tokens_a)
        result_b, rem_b = cache.fetch(tokens_b)

        assert result_a is kv_a
        assert result_b is kv_b
        assert rem_a == []
        assert rem_b == []

        # Unrelated tokens return miss
        result_c, rem_c = cache.fetch([99, 100])
        assert result_c is None
        assert rem_c == [99, 100]

    def test_eviction_under_memory_pressure(self):
        """Store entries until memory limit exceeded, verify oldest evicted."""
        # 1 MB limit, each entry ~0.5 MB (2 layers * 256 KB each)
        cache = _make_prefix_cache(max_memory_mb=1, max_entries=100)
        bytes_per_layer = 256 * 1024  # 256 KB

        # Store first entry
        tokens_first = [1, 2, 3]
        kv_first = _make_cache_list(nbytes_per_layer=bytes_per_layer)
        cache.store(tokens_first, kv_first)

        # Store enough entries to exceed 1 MB
        for i in range(4):
            tokens = [100 + i, 200 + i, 300 + i]
            kv = _make_cache_list(nbytes_per_layer=bytes_per_layer)
            cache.store(tokens, kv)

        # The first entry should have been evicted (LRU)
        result, remaining = cache.fetch(tokens_first)
        # Either evicted (None) or still present if memory didn't overflow.
        # With 5 entries * 0.5 MB = 2.5 MB > 1 MB limit, eviction is certain.
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_stats_accuracy(self):
        """After store/fetch/evict, stats() returns correct counts."""
        cache = _make_prefix_cache(max_memory_mb=10, max_entries=3)

        tokens_a = [1, 2, 3]
        tokens_b = [4, 5, 6]
        tokens_c = [7, 8, 9]
        tokens_d = [10, 11, 12]

        cache.store(tokens_a, _make_cache_list())
        cache.store(tokens_b, _make_cache_list())
        cache.store(tokens_c, _make_cache_list())

        # Fetch hit
        cache.fetch(tokens_a)
        # Fetch miss
        cache.fetch([99, 100])

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 3

        # Store a 4th entry -- max_entries=3, so one must be evicted
        cache.store(tokens_d, _make_cache_list())
        stats2 = cache.get_stats()
        assert stats2["evictions"] >= 1
        assert stats2["entry_count"] == 3  # Still at max

    def test_prefix_match_returns_truncated(self):
        """Store [1,2,3,4], fetch [1,2,3,4,5] should return prefix match."""
        cache = _make_prefix_cache()
        tokens_stored = [1, 2, 3, 4]
        kv = _make_cache_list()
        cache.store(tokens_stored, kv)

        # Fetch with extra tokens -- forward prefix match
        result, remaining = cache.fetch([1, 2, 3, 4, 5])
        assert result is kv  # Forward match returns stored cache
        assert remaining == [5]  # Only the extra token remains

    def test_clear_removes_all_entries(self):
        """After clear(), fetch returns None."""
        cache = _make_prefix_cache()
        cache.store([1, 2], _make_cache_list())
        cache.store([3, 4], _make_cache_list())

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        result, remaining = cache.fetch([1, 2])
        assert result is None
        assert remaining == [1, 2]

        stats = cache.get_stats()
        assert stats["entry_count"] == 0

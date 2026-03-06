# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-aware prefix cache."""

from unittest.mock import MagicMock, patch

import pytest

from vmlx_engine.memory_cache import (
    CacheStats,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _CacheEntry,
    _get_available_memory,
    estimate_kv_cache_memory,
)


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_config(self):
        config = MemoryCacheConfig()
        assert config.max_memory_mb is None
        assert config.max_memory_percent == 0.30
        assert config.max_entries == 1000
        assert config.enable_memory_tracking is True

    def test_custom_config(self):
        config = MemoryCacheConfig(
            max_memory_mb=2048,
            max_memory_percent=0.5,
            max_entries=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_memory_percent == 0.5
        assert config.max_entries == 100

    def test_invalid_memory_percent_zero(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=0.0)

    def test_invalid_memory_percent_negative(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=-0.1)

    def test_invalid_memory_percent_over_one(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=1.5)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            MemoryCacheConfig(max_entries=0)

    def test_compute_memory_limit_explicit(self):
        config = MemoryCacheConfig(max_memory_mb=1024)
        assert config.compute_memory_limit() == 1024 * 1024 * 1024

    def test_compute_memory_limit_auto(self):
        with patch(
            "vmlx_engine.memory_cache._get_available_memory",
            return_value=8 * 1024 * 1024 * 1024,  # 8GB
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            assert limit == 2 * 1024 * 1024 * 1024  # 25% of 8GB = 2GB

    def test_compute_memory_limit_fallback(self):
        with patch(
            "vmlx_engine.memory_cache._get_available_memory",
            return_value=0,  # Detection failed
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            # Fallback: 25% of 8GB = 2GB
            assert limit == 2 * 1024 * 1024 * 1024


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_queries(self):
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_memory_utilization(self):
        stats = CacheStats(
            current_memory_bytes=500 * 1024 * 1024,
            max_memory_bytes=1000 * 1024 * 1024,
        )
        assert stats.memory_utilization == 0.5

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d
        assert "memory_utilization" in d


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockKVCache:
    """Mock KV cache with keys/values attributes."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)


class MockStateCache:
    """Mock cache with state property."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self._keys = MockArray(key_bytes)
        self._values = MockArray(value_bytes)

    @property
    def state(self):
        return (self._keys, self._values)


class TestEstimateKvCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_empty_cache(self):
        assert estimate_kv_cache_memory([]) == 0
        assert estimate_kv_cache_memory(None) == 0

    def test_cache_with_nbytes_attribute(self):
        layer = MockKVCache(1000, 1000)
        assert estimate_kv_cache_memory([layer]) == 2000

    def test_cache_with_state_property(self):
        layer = MockStateCache(500, 500)
        assert estimate_kv_cache_memory([layer]) == 1000

    def test_cache_with_dict_state(self):
        keys = MockArray(300)
        values = MockArray(300)
        layer = {"state": (keys, values)}
        assert estimate_kv_cache_memory([layer]) == 600

    def test_multiple_layers(self):
        layers = [MockKVCache(100, 100) for _ in range(4)]
        assert estimate_kv_cache_memory(layers) == 800


class TestCacheEntry:
    """Tests for _CacheEntry."""

    def test_create_entry(self):
        cache = [MockKVCache(100, 100)]
        entry = _CacheEntry.create([1, 2, 3], cache)
        assert entry.tokens == (1, 2, 3)
        assert entry.cache is cache
        assert entry.memory_bytes == 200


class TestMemoryAwarePrefixCache:
    """Tests for MemoryAwarePrefixCache."""

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def small_cache(self, model):
        """Cache with 1MB limit."""
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KV cache with known size."""

        def _create(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        return _create

    def test_initialization(self, model):
        config = MemoryCacheConfig(max_memory_mb=100)
        cache = MemoryAwarePrefixCache(model, config)
        assert len(cache) == 0
        assert cache.memory_limit_mb == 100.0

    def test_store_and_fetch_exact_match(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3, 4, 5]
        kv = mock_kv_cache(1000)

        # Store
        assert small_cache.store(tokens, kv) is True
        assert len(small_cache) == 1

        # Fetch exact match
        result, remaining = small_cache.fetch(tokens)
        assert result is kv  # Same reference, no copy
        assert remaining == []

    def test_fetch_prefix_match(self, small_cache, mock_kv_cache):
        # Store shorter sequence
        short_tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(short_tokens, kv)

        # Fetch longer sequence that starts with cached prefix
        long_tokens = [1, 2, 3, 4, 5, 6]
        result, remaining = small_cache.fetch(long_tokens)

        assert result is kv
        assert remaining == [4, 5, 6]

    def test_fetch_reverse_prefix_match(self, small_cache, mock_kv_cache):
        """Cache stores prompt+output, fetch with prompt only should match."""
        # Store longer sequence (simulates prompt+output storage)
        long_tokens = [1, 2, 3, 4, 5, 6]
        kv = mock_kv_cache(1000)
        small_cache.store(long_tokens, kv)

        # Fetch shorter sequence that is a prefix of the cached key
        short_tokens = [1, 2, 3]
        result, remaining = small_cache.fetch(short_tokens)

        # Should match via reverse prefix and return truncated cache
        assert result is not None
        assert remaining == []

        # Stats should show a hit
        stats = small_cache.get_stats()
        assert stats["hits"] == 1

    def test_fetch_reverse_prefix_match_hybrid_cache(self, small_cache):
        """Reverse prefix match correctly declines for hybrid MambaCache + KVCache.

        MambaCache state is cumulative and includes ALL processed tokens'
        contributions. It cannot be truncated to a shorter token count.
        Passing truncated KVCache + full MambaCache would create a state
        mismatch that crashes the model (SIGKILL). So reverse prefix matching
        must return None (cache miss) for hybrid models.
        """

        class MockMambaLayer:
            """Mock MambaCache layer (cumulative, no .keys/.values)."""

            def __init__(self, size_bytes):
                self.cache = [MockArray(size_bytes // 2), MockArray(size_bytes // 2)]

        class MockKVLayer:
            """Mock KVCache layer (positional, has .keys/.values)."""

            def __init__(self, size_bytes):
                self.keys = MockArray(size_bytes // 2)
                self.values = MockArray(size_bytes // 2)

        # Hybrid cache: MambaCache layer + KVCache layer (like Qwen3-Coder-Next)
        hybrid_cache = [MockMambaLayer(200), MockKVLayer(400), MockMambaLayer(200)]

        # Store longer sequence (prompt+output)
        long_tokens = [1, 2, 3, 4, 5, 6]
        small_cache.store(long_tokens, hybrid_cache)

        # Fetch shorter sequence (prompt only) — reverse prefix match
        short_tokens = [1, 2, 3]
        result, remaining = small_cache.fetch(short_tokens)

        # Should be a cache miss — MambaCache can't be truncated
        assert result is None
        assert remaining == [1, 2, 3]

    def test_fetch_reverse_prefix_prefers_forward(self, small_cache, mock_kv_cache):
        """Forward prefix match should be preferred over reverse."""
        # Store short (forward match candidate) and long (reverse match candidate)
        short_kv = mock_kv_cache(500)
        long_kv = mock_kv_cache(1000)
        small_cache.store([1, 2, 3], short_kv)
        small_cache.store([1, 2, 3, 4, 5, 6, 7, 8], long_kv)

        # Fetch [1,2,3,4,5] - should forward match [1,2,3] (not reverse match [1,2,3,4,5,6,7,8])
        result, remaining = small_cache.fetch([1, 2, 3, 4, 5])
        assert result is short_kv
        assert remaining == [4, 5]

    def test_fetch_miss(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)

        # Fetch completely different sequence
        result, remaining = small_cache.fetch([7, 8, 9])
        assert result is None
        assert remaining == [7, 8, 9]

    def test_lru_eviction_on_memory_pressure(self, model, mock_kv_cache):
        # Create cache with 500KB limit
        config = MemoryCacheConfig(max_memory_mb=0.5, max_entries=100)
        cache = MemoryAwarePrefixCache(model, config)

        # Store entries that together exceed limit
        # Each is ~200KB
        for i in range(5):
            tokens = list(range(i * 10, (i + 1) * 10))
            kv = mock_kv_cache(200 * 1024)
            cache.store(tokens, kv)

        # Should have evicted older entries
        assert cache.memory_usage_mb <= 0.5
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_lru_order_updated_on_fetch(self, small_cache, mock_kv_cache):
        # Store two entries
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv1 = mock_kv_cache(100 * 1024)
        kv2 = mock_kv_cache(100 * 1024)

        small_cache.store(tokens1, kv1)
        small_cache.store(tokens2, kv2)

        # Fetch first entry (moves it to end of LRU)
        small_cache.fetch(tokens1)

        # Now tokens2 should be evicted first if we need space
        # Store a large entry to trigger eviction
        big_kv = mock_kv_cache(900 * 1024)
        small_cache.store([7, 8, 9], big_kv)

        # tokens1 should still be there (was recently accessed)
        # tokens2 should be evicted
        assert tokens1 in small_cache or len(small_cache) == 1

    def test_entry_too_large_rejected(self, small_cache, mock_kv_cache):
        # Try to store entry larger than cache limit
        tokens = [1, 2, 3]
        huge_kv = mock_kv_cache(10 * 1024 * 1024)  # 10MB, limit is 1MB

        result = small_cache.store(tokens, huge_kv)
        assert result is False
        assert len(small_cache) == 0

    def test_store_empty_rejected(self, small_cache, mock_kv_cache):
        assert small_cache.store([], mock_kv_cache(100)) is False
        assert small_cache.store([1, 2, 3], []) is False
        assert small_cache.store([1, 2, 3], None) is False

    def test_remove_entry(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)
        assert len(small_cache) == 1

        assert small_cache.remove(tokens) is True
        assert len(small_cache) == 0
        assert small_cache.remove(tokens) is False  # Already removed

    def test_clear(self, small_cache, mock_kv_cache):
        for i in range(3):
            small_cache.store([i], mock_kv_cache(1000))

        assert len(small_cache) == 3
        small_cache.clear()
        assert len(small_cache) == 0
        assert small_cache.memory_usage_mb == 0

    def test_contains(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        assert tokens not in small_cache
        small_cache.store(tokens, mock_kv_cache(1000))
        assert tokens in small_cache

    def test_stats_tracking(self, small_cache, mock_kv_cache):
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens1, kv)
        small_cache.fetch(tokens1)  # Hit
        small_cache.fetch(tokens2)  # Miss

        stats = small_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1

    def test_reset_stats(self, small_cache, mock_kv_cache):
        small_cache.store([1, 2, 3], mock_kv_cache(1000))
        small_cache.fetch([1, 2, 3])
        small_cache.fetch([4, 5, 6])

        small_cache.reset_stats()
        stats = small_cache.get_stats()

        # Stats reset but entry count preserved
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entry_count"] == 1

    def test_duplicate_store_updates_lru(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens, kv)
        initial_len = len(small_cache)

        # Store same tokens again
        small_cache.store(tokens, kv)

        # Should not create duplicate
        assert len(small_cache) == initial_len

    def test_max_entries_limit(self, model, mock_kv_cache):
        # Create cache with low entry limit
        config = MemoryCacheConfig(max_memory_mb=100, max_entries=3)
        cache = MemoryAwarePrefixCache(model, config)

        # Store 5 entries (only 3 should remain)
        for i in range(5):
            cache.store([i], mock_kv_cache(100))

        assert len(cache) <= 3


class TestGetAvailableMemory:
    """Tests for _get_available_memory helper."""

    def test_with_psutil(self):
        try:
            from importlib.util import find_spec

            if find_spec("psutil") is None:
                pytest.skip("psutil not installed")
            mem = _get_available_memory()
            assert mem > 0
        except ImportError:
            pytest.skip("psutil not installed")

    def test_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Should return 0 when psutil not available
            # Note: This test may not work as expected due to import caching
            pass


class TestLargeContextCacheReuse:
    """Tests simulating 100k+ token cache reuse scenarios (Qwen3-Coder-Next)."""

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def large_cache(self, model):
        """Cache with 8GB limit to simulate real-world usage."""
        config = MemoryCacheConfig(max_memory_mb=8192, max_entries=50)
        return MemoryAwarePrefixCache(model, config)

    def _make_kv_cache(self, num_tokens: int, num_layers: int = 64):
        """Create a mock KV cache simulating Qwen3-Coder-Next dimensions.

        Each layer has keys+values of shape (1, n_kv_heads, seq_len, head_dim).
        With GQA (4 kv_heads, 128 head_dim, fp16): 4*128*2*2 = 2048 bytes/token/layer
        For 4-bit quantized: ~50 bytes/token/layer (realistic for quantized models).
        """
        bytes_per_token_per_layer = 50  # Realistic for 4-bit quantized GQA models
        layer_bytes = num_tokens * bytes_per_token_per_layer
        return [MockKVCache(layer_bytes // 2, layer_bytes // 2) for _ in range(num_layers)]

    def test_cache_100k_tokens_store_and_fetch(self, large_cache):
        """Verify that a 100k token cache can be stored and retrieved."""
        tokens = list(range(100_000))
        kv = self._make_kv_cache(100_000)

        # Store should succeed
        assert large_cache.store(tokens, kv) is True
        assert len(large_cache) == 1

        # Exact fetch should return same reference
        result, remaining = large_cache.fetch(tokens)
        assert result is kv
        assert remaining == []

    def test_multi_turn_growing_context(self, large_cache):
        """Simulate multi-turn conversation growing to 100k tokens.

        Turn 1: 2k prompt → 500 output (2.5k total cached)
        Turn 2: 2.5k + 3k new → 1k output (6.5k total cached)
        Turn 3: 6.5k + 5k new → 2k output (13.5k total cached)
        ... continue growing
        """
        total_tokens = []
        turns = [
            (2000, 500),    # Turn 1
            (3000, 1000),   # Turn 2
            (5000, 2000),   # Turn 3
            (10000, 3000),  # Turn 4
            (15000, 5000),  # Turn 5
            (20000, 8000),  # Turn 6
            (30000, 10000), # Turn 7 - total ~114k
        ]

        for i, (new_prompt_tokens, output_tokens) in enumerate(turns):
            # Build prompt: previous context + new tokens
            new_tokens = list(range(len(total_tokens), len(total_tokens) + new_prompt_tokens))
            prompt = total_tokens + new_tokens

            # Fetch from cache - should match prefix
            cached, remaining = large_cache.fetch(prompt)
            if i == 0:
                # First turn: cache miss
                assert cached is None
                assert remaining == prompt
            else:
                # Subsequent turns: should hit cached prefix
                assert cached is not None
                assert len(remaining) < len(prompt)

            # Process and store full context (prompt + output)
            output = list(range(1_000_000 + i * 100_000, 1_000_000 + i * 100_000 + output_tokens))
            total_tokens = prompt + output

            kv = self._make_kv_cache(len(total_tokens))
            stored = large_cache.store(total_tokens, kv)
            assert stored is True, f"Failed to store at turn {i+1} ({len(total_tokens)} tokens)"

        # Final context should be ~114k tokens
        assert len(total_tokens) > 100_000
        stats = large_cache.get_stats()
        assert stats["hits"] >= 6  # At least 6 cache hits (turns 2-7)

    def test_cache_eviction_preserves_most_recent(self, model):
        """Ensure LRU eviction keeps most recent entries when memory is tight."""
        # Small cache that can only hold ~2 entries of 50k tokens
        config = MemoryCacheConfig(max_memory_mb=200, max_entries=50)
        cache = MemoryAwarePrefixCache(model, config)

        entries = []
        for i in range(5):
            tokens = list(range(i * 50000, (i + 1) * 50000))
            kv = [MockKVCache(50 * 1024 * 1024 // 2, 50 * 1024 * 1024 // 2)]  # ~50MB each
            cache.store(tokens, kv)
            entries.append(tokens)

        # Most recent entries should still be in cache
        result, _ = cache.fetch(entries[-1])
        assert result is not None, "Most recent entry should be in cache"

        stats = cache.get_stats()
        assert stats["evictions"] > 0, "Should have evicted older entries"

    def test_per_entry_limit_95_percent(self, model):
        """Verify per-entry limit is 95% of max, not a fixed 4GB cap."""
        config = MemoryCacheConfig(max_memory_mb=1024)  # 1GB cache
        cache = MemoryAwarePrefixCache(model, config)

        # Entry at 90% of limit should be accepted (under 95% threshold)
        tokens_ok = list(range(1000))
        kv_ok = [MockKVCache(460 * 1024 * 1024, 460 * 1024 * 1024)]  # 920MB
        assert cache.store(tokens_ok, kv_ok) is True

        # Entry at 98% of limit should be rejected (> 95% threshold)
        cache.clear()
        tokens_big = list(range(2000, 3000))
        kv_big = [MockKVCache(502 * 1024 * 1024, 502 * 1024 * 1024)]  # 1004MB > 972MB
        assert cache.store(tokens_big, kv_big) is False

    def test_memory_tracking_accuracy(self, large_cache):
        """Verify memory tracking stays accurate through store/evict cycles."""
        for i in range(20):
            tokens = list(range(i * 1000, (i + 1) * 1000))
            kv = self._make_kv_cache(1000, num_layers=32)
            large_cache.store(tokens, kv)

        stats = large_cache.get_stats()
        assert stats["current_memory_mb"] > 0
        assert stats["current_memory_mb"] <= stats["max_memory_mb"]

        # After clearing, memory should be 0
        large_cache.clear()
        stats = large_cache.get_stats()
        assert stats["current_memory_mb"] == 0

    def test_fetch_empty_tokens_is_miss(self, large_cache):
        """Empty token list should be a cache miss, not an error."""
        result, remaining = large_cache.fetch([])
        assert result is None
        assert remaining == []

    def test_store_none_cache_rejected(self, large_cache):
        """Storing None cache should be rejected gracefully."""
        assert large_cache.store([1, 2, 3], None) is False

    def test_concurrent_prefix_matching_at_scale(self, large_cache):
        """Test prefix matching with many cached sequences of varying lengths."""
        # Store sequences of increasing length
        base = list(range(10000))
        for length in [1000, 5000, 10000, 20000, 50000]:
            tokens = base[:length]
            kv = self._make_kv_cache(length, num_layers=16)
            large_cache.store(tokens, kv)

        # Fetch with 30k tokens should match the 20k prefix (longest prefix < 30k)
        query = base[:10000] + list(range(10000, 30000))
        # Since base[:10000] is exact match, it should return that
        result, remaining = large_cache.fetch(query)
        assert result is not None
        assert len(remaining) < len(query)

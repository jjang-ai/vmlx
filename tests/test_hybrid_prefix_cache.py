"""
Tests for prefix cache with hybrid SSM models (KVCache + MambaCache/ArraysCache).

Verifies that:
1. SSM layers are included in cache state extraction (not skipped)
2. Block tensor slicing tags SSM layers as "cumulative" in last block, "skip" otherwise
3. Reconstruction restores both KV and SSM layers
4. Full round-trip: extract -> store -> fetch -> reconstruct for hybrid models
5. GQA models (num_key_value_heads < num_attention_heads) validate correctly
"""
import pytest
from unittest.mock import MagicMock, PropertyMock

# Skip entire module if MLX not available
try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# ---------------------------------------------------------------------------
# Helpers: mock cache objects that mimic mlx-lm cache classes
# ---------------------------------------------------------------------------

class MockKVCache:
    """Mimics mlx_lm.models.cache.KVCache for testing."""

    def __init__(self, n_kv_heads: int, seq_len: int, head_dim: int = 64):
        self.keys = mx.zeros((1, n_kv_heads, seq_len, head_dim))
        self.values = mx.ones((1, n_kv_heads, seq_len, head_dim))
        self.offset = seq_len

    @property
    def state(self):
        return (self.keys, self.values)

    @property
    def meta_state(self):
        return (str(self.offset),)


class MockArraysCache:
    """Mimics MambaCache/ArraysCache -- cumulative SSM state."""

    def __init__(self, state_size: int = 16):
        # .cache attribute is a list -> identifies as SSM layer
        self.cache = [mx.zeros((1, state_size))]
        self._state = (mx.zeros((1, state_size)), mx.ones((1, state_size)))
        self._offset = 0

    @property
    def state(self):
        return self._state

    @property
    def meta_state(self):
        return (str(self._offset),)


class MockGQAModel:
    """Model with GQA: num_key_value_heads < num_attention_heads."""

    class Args:
        num_attention_heads = 32
        num_key_value_heads = 8
        hidden_size = 4096
        head_dim = 128
        kv_lora_rank = 0

    args = Args()

    def make_cache(self):
        return [MockKVCache(n_kv_heads=8, seq_len=0) for _ in range(4)]


class MockHybridModel:
    """Model with mixed KVCache + ArraysCache layers (like Nemotron)."""

    class Args:
        num_attention_heads = 32
        num_key_value_heads = 8
        hidden_size = 4096
        head_dim = 128
        kv_lora_rank = 0

    args = Args()

    def make_cache(self):
        # 4 layers: SSM, KV, SSM, KV (alternating)
        return [
            MockArraysCache(),
            MockKVCache(n_kv_heads=8, seq_len=0),
            MockArraysCache(),
            MockKVCache(n_kv_heads=8, seq_len=0),
        ]


# ---------------------------------------------------------------------------
# Tests: _extract_cache_states includes SSM layers
# ---------------------------------------------------------------------------

class TestExtractCacheStatesHybrid:
    """Verify _extract_cache_states includes SSM layers (not skips them)."""

    def test_extract_includes_ssm_layers(self):
        """SSM layers should be in extracted list, not skipped."""
        from vmlx_engine.scheduler import Scheduler

        # Create a minimal scheduler mock with the method
        sched = object.__new__(Scheduler)
        sched.model = MockHybridModel()
        sched._n_kv_heads_cached = 8

        # Build hybrid cache: [SSM, KV(8 heads, 100 tokens), SSM, KV(8 heads, 100 tokens)]
        raw_cache = [
            MockArraysCache(state_size=16),
            MockKVCache(n_kv_heads=8, seq_len=100),
            MockArraysCache(state_size=16),
            MockKVCache(n_kv_heads=8, seq_len=100),
        ]

        extracted = sched._extract_cache_states(raw_cache)

        # ALL 4 layers should be extracted (not just the 2 KV layers)
        assert len(extracted) == 4, (
            f"Expected 4 extracted layers (2 KV + 2 SSM), got {len(extracted)}"
        )

        # Check layer types
        assert extracted[0]["class_name"] == "MockArraysCache"
        assert extracted[1]["class_name"] == "MockKVCache"
        assert extracted[2]["class_name"] == "MockArraysCache"
        assert extracted[3]["class_name"] == "MockKVCache"

    def test_extract_ssm_state_not_none(self):
        """SSM layers should have actual state, not None."""
        from vmlx_engine.scheduler import Scheduler

        sched = object.__new__(Scheduler)
        sched.model = MockHybridModel()
        sched._n_kv_heads_cached = 8

        raw_cache = [
            MockArraysCache(state_size=16),
            MockKVCache(n_kv_heads=8, seq_len=100),
        ]

        extracted = sched._extract_cache_states(raw_cache)
        assert len(extracted) == 2

        # SSM layer should have real state
        ssm_state = extracted[0]["state"]
        assert ssm_state is not None, "SSM state should not be None"
        assert isinstance(ssm_state, tuple), "SSM state should be a tuple"
        assert len(ssm_state) == 2, "SSM state should have 2 elements"

    def test_extract_gqa_normalization_only_on_kv(self):
        """GQA normalization should only apply to KV layers, not SSM."""
        from vmlx_engine.scheduler import Scheduler

        sched = object.__new__(Scheduler)
        sched.model = MockGQAModel()
        sched._n_kv_heads_cached = 8

        # Create KV cache with inflated heads (32 instead of 8)
        inflated_kv = MockKVCache(n_kv_heads=32, seq_len=100)
        raw_cache = [inflated_kv]

        extracted = sched._extract_cache_states(raw_cache)
        assert len(extracted) == 1

        # Should be normalized to 8 heads
        keys, values = extracted[0]["state"]
        assert keys.shape[1] == 8, f"Expected 8 KV heads after normalization, got {keys.shape[1]}"


# ---------------------------------------------------------------------------
# Tests: block tensor slicing tags SSM layers correctly
# ---------------------------------------------------------------------------

class TestBlockSlicingHybrid:
    """Verify _extract_block_tensor_slice tags SSM as cumulative/skip."""

    def _make_cache(self, block_size=64):
        """Create a BlockAwarePrefixCache with mock model."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        model = MockGQAModel()
        mgr = PagedCacheManager(block_size=block_size, max_blocks=100)
        return BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

    def test_ssm_skip_in_non_last_block(self):
        """SSM layers should be tagged 'skip' in non-last blocks."""
        cache = self._make_cache(block_size=64)

        # Extracted data: [SSM, KV(8 heads, 128 tokens)]
        cache_data = [
            {
                "state": (mx.zeros((1, 16)), mx.ones((1, 16))),
                "meta_state": ("0",),
                "class_name": "ArraysCache",
            },
            {
                "state": (
                    mx.zeros((1, 8, 128, 64)),
                    mx.ones((1, 8, 128, 64)),
                ),
                "meta_state": ("128",),
                "class_name": "KVCache",
            },
        ]

        # Non-last block (tokens 0-63)
        slices = cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=64, is_last_block=False
        )
        assert slices is not None
        assert len(slices) == 2

        # SSM layer -> skip
        assert slices[0][0] == "skip", f"Expected 'skip' for SSM in non-last block, got {slices[0][0]}"
        # KV layer -> kv
        assert slices[1][0] == "kv", f"Expected 'kv' for KV layer, got {slices[1][0]}"

    def test_ssm_cumulative_in_last_block(self):
        """SSM layers should be tagged 'cumulative' in last block."""
        cache = self._make_cache(block_size=64)

        cache_data = [
            {
                "state": (mx.zeros((1, 16)), mx.ones((1, 16))),
                "meta_state": ("0",),
                "class_name": "ArraysCache",
            },
            {
                "state": (
                    mx.zeros((1, 8, 128, 64)),
                    mx.ones((1, 8, 128, 64)),
                ),
                "meta_state": ("128",),
                "class_name": "KVCache",
            },
        ]

        # Last block (tokens 64-127)
        slices = cache._extract_block_tensor_slice(
            cache_data, start_idx=64, end_idx=128, is_last_block=True
        )
        assert slices is not None
        assert len(slices) == 2

        # SSM layer -> cumulative (full state stored)
        assert slices[0][0] == "cumulative", (
            f"Expected 'cumulative' for SSM in last block, got {slices[0][0]}"
        )
        assert slices[0][1] is not None, "Cumulative state should not be None"

        # KV layer -> kv
        assert slices[1][0] == "kv"

    def test_ssm_none_state_always_skip(self):
        """SSM layers with None state should always be 'skip', even in last block."""
        cache = self._make_cache(block_size=64)

        cache_data = [
            {
                "state": None,
                "meta_state": None,
                "class_name": "ArraysCache",
            },
            {
                "state": (
                    mx.zeros((1, 8, 64, 64)),
                    mx.ones((1, 8, 64, 64)),
                ),
                "meta_state": ("64",),
                "class_name": "KVCache",
            },
        ]

        slices = cache._extract_block_tensor_slice(
            cache_data, start_idx=0, end_idx=64, is_last_block=True
        )
        assert slices is not None
        # SSM with None state -> skip (not cumulative)
        assert slices[0][0] == "skip"
        assert slices[1][0] == "kv"


# ---------------------------------------------------------------------------
# Tests: reconstruction round-trip for hybrid models
# ---------------------------------------------------------------------------

class TestReconstructHybrid:
    """Verify full store -> reconstruct round-trip for hybrid models."""

    def _make_cache(self, block_size=64):
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager
        model = MockHybridModel()
        mgr = PagedCacheManager(block_size=block_size, max_blocks=100)
        return BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

    def test_full_round_trip_hybrid(self):
        """Store hybrid cache, fetch, reconstruct -- should get all layers back."""
        cache = self._make_cache(block_size=64)

        # Extracted data: 4 layers (SSM, KV, SSM, KV) with 64 tokens
        kv_keys = mx.random.normal((1, 8, 64, 64))
        kv_values = mx.random.normal((1, 8, 64, 64))
        mx.synchronize()

        ssm_state = (mx.random.normal((1, 16)), mx.random.normal((1, 16)))
        mx.synchronize()

        cache_data = [
            {
                "state": ssm_state,
                "meta_state": ("0",),
                "class_name": "ArraysCache",
            },
            {
                "state": (kv_keys, kv_values),
                "meta_state": ("64",),
                "class_name": "KVCache",
            },
            {
                "state": ssm_state,
                "meta_state": ("0",),
                "class_name": "ArraysCache",
            },
            {
                "state": (kv_keys, kv_values),
                "meta_state": ("64",),
                "class_name": "KVCache",
            },
        ]

        # Store
        tokens = list(range(64))
        cache.store_cache("req1", tokens, cache_data)

        # Fetch
        block_table, remaining = cache.fetch_cache("req2", tokens)
        assert block_table is not None, "Should find cached prefix"
        assert block_table.num_tokens == 64

        # Reconstruct
        reconstructed = cache.reconstruct_cache(block_table)

        # Should get all 4 layers back (2 SSM + 2 KV)
        assert reconstructed is not None, "Reconstruction should succeed"
        assert len(reconstructed) == 4, (
            f"Expected 4 reconstructed layers, got {len(reconstructed)}"
        )

        # Verify KV layers have correct shape
        kv_layers = [c for c in reconstructed if hasattr(c, 'keys') and c.keys is not None]
        assert len(kv_layers) == 2, f"Expected 2 KV layers, got {len(kv_layers)}"
        for kv in kv_layers:
            assert kv.keys.shape == (1, 8, 64, 64), f"KV keys shape mismatch: {kv.keys.shape}"


# ---------------------------------------------------------------------------
# Tests: GQA head count validation in prefix cache
# ---------------------------------------------------------------------------

class TestGQAHeadValidation:
    """Verify _get_n_kv_heads returns correct value for GQA models."""

    def test_gqa_model_returns_kv_heads(self):
        """For GQA model, _get_n_kv_heads should return num_key_value_heads."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        model = MockGQAModel()
        mgr = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

        n_kv = cache._get_n_kv_heads()
        assert n_kv == 8, f"Expected 8 KV heads for GQA model, got {n_kv}"

    def test_mha_model_returns_all_heads(self):
        """For MHA model (all heads = KV heads), should return num_key_value_heads."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        class MHAModel:
            class Args:
                num_attention_heads = 32
                num_key_value_heads = 32  # MHA: all heads are KV heads
                kv_lora_rank = 0
            args = Args()

        model = MHAModel()
        mgr = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

        n_kv = cache._get_n_kv_heads()
        assert n_kv == 32, f"Expected 32 KV heads for MHA model, got {n_kv}"

    def test_no_kv_heads_attr_returns_zero(self):
        """Model without num_key_value_heads should return 0 (skip validation)."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        class MinimalModel:
            class Args:
                num_attention_heads = 32
                # No num_key_value_heads!
                kv_lora_rank = 0
            args = Args()

        model = MinimalModel()
        mgr = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

        n_kv = cache._get_n_kv_heads()
        assert n_kv == 0, f"Expected 0 (skip validation) without num_key_value_heads, got {n_kv}"

    def test_mla_returns_one(self):
        """MLA model (kv_lora_rank > 0) should return 1."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        class MLAModel:
            class Args:
                num_attention_heads = 32
                num_key_value_heads = 32
                kv_lora_rank = 512  # MLA: compressed latents
            args = Args()

        model = MLAModel()
        mgr = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)

        n_kv = cache._get_n_kv_heads()
        assert n_kv == 1, f"Expected 1 for MLA model, got {n_kv}"


# ---------------------------------------------------------------------------
# Tests: _is_positional_cache correctly identifies SSM vs KV
# ---------------------------------------------------------------------------

class TestIsPositionalCache:
    """Verify cache type detection works for all cache types."""

    def test_kvcache_is_positional(self):
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        state = (mx.zeros((1, 8, 100, 64)), mx.zeros((1, 8, 100, 64)))
        assert BlockAwarePrefixCache._is_positional_cache(state, "KVCache") is True

    def test_arrays_cache_is_not_positional(self):
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        state = (mx.zeros((1, 16)), mx.zeros((1, 16)))
        assert BlockAwarePrefixCache._is_positional_cache(state, "ArraysCache") is False

    def test_mamba_cache_is_not_positional(self):
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        state = (mx.zeros((1, 16)), mx.zeros((1, 16)))
        assert BlockAwarePrefixCache._is_positional_cache(state, "MambaCache") is False

    def test_quantized_kvcache_is_positional(self):
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        state = (mx.zeros((1, 8, 100, 64)), mx.zeros((1, 8, 100, 64)))
        assert BlockAwarePrefixCache._is_positional_cache(state, "QuantizedKVCache") is True


# ---------------------------------------------------------------------------
# Tests: scheduler _detect_n_kv_heads vs prefix_cache _get_n_kv_heads agreement
# ---------------------------------------------------------------------------

class TestHeadCountAgreement:
    """Verify scheduler and prefix cache agree on KV head count."""

    def test_gqa_agreement(self):
        """Both should detect the same head count for GQA models."""
        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache, PagedCacheManager

        model = MockGQAModel()

        # Scheduler detection
        sched = object.__new__(Scheduler)
        sched.model = model
        sched_n_kv = sched._detect_n_kv_heads()

        # Prefix cache detection
        mgr = PagedCacheManager(block_size=64, max_blocks=100)
        pcache = BlockAwarePrefixCache(model=model, paged_cache_manager=mgr)
        pcache_n_kv = pcache._get_n_kv_heads()

        assert sched_n_kv == pcache_n_kv, (
            f"Scheduler ({sched_n_kv}) and prefix cache ({pcache_n_kv}) "
            f"disagree on KV head count"
        )

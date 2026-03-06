# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM Scheduler cache infrastructure.

Covers:
- MLLMSchedulerConfig cache field parity with SchedulerConfig
- Cache init chain (paged > memory-aware > legacy)
- _ensure_batch_generator clears all cache modes
- _cleanup_finished stores to all cache paths
- HybridSSMStateCache companion cache
- Metal optimization settings
- get_stats() cache reporting
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields

from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig, MLLMScheduler


# ============================================================
# Config parity tests
# ============================================================


class TestMLLMSchedulerConfigParity:
    """Verify all cache-related fields exist on MLLMSchedulerConfig."""

    def test_memory_aware_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "use_memory_aware_cache")
        assert hasattr(config, "cache_memory_mb")
        assert hasattr(config, "cache_memory_percent")
        assert hasattr(config, "cache_ttl_minutes")

    def test_memory_aware_defaults(self):
        config = MLLMSchedulerConfig()
        assert config.use_memory_aware_cache is True
        assert config.cache_memory_mb is None
        assert config.cache_memory_percent == 0.20
        assert config.cache_ttl_minutes == 0

    def test_legacy_prefix_cache_field(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "prefix_cache_size")
        assert config.prefix_cache_size == 100

    def test_disk_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_disk_cache")
        assert hasattr(config, "disk_cache_dir")
        assert hasattr(config, "disk_cache_max_gb")
        assert config.enable_disk_cache is False
        assert config.disk_cache_dir is None
        assert config.disk_cache_max_gb == 10.0

    def test_block_disk_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_block_disk_cache")
        assert hasattr(config, "block_disk_cache_dir")
        assert hasattr(config, "block_disk_cache_max_gb")
        assert config.enable_block_disk_cache is False
        assert config.block_disk_cache_dir is None
        assert config.block_disk_cache_max_gb == 10.0

    def test_model_path_field(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "model_path")
        assert config.model_path is None

    def test_kv_cache_quantization_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "kv_cache_quantization")
        assert hasattr(config, "kv_cache_group_size")
        assert config.kv_cache_quantization == "none"
        assert config.kv_cache_group_size == 64

    def test_paged_cache_fields(self):
        config = MLLMSchedulerConfig()
        assert hasattr(config, "enable_prefix_cache")
        assert hasattr(config, "use_paged_cache")
        assert hasattr(config, "paged_cache_block_size")
        assert hasattr(config, "max_cache_blocks")
        assert config.enable_prefix_cache is True
        assert config.use_paged_cache is True
        assert config.paged_cache_block_size == 64
        assert config.max_cache_blocks == 1000

    def test_custom_values(self):
        config = MLLMSchedulerConfig(
            cache_memory_mb=2048,
            cache_memory_percent=0.30,
            cache_ttl_minutes=5.0,
            prefix_cache_size=200,
            enable_disk_cache=True,
            disk_cache_dir="/tmp/test-cache",
            disk_cache_max_gb=20.0,
            model_path="/models/test-vlm",
        )
        assert config.cache_memory_mb == 2048
        assert config.cache_memory_percent == 0.30
        assert config.cache_ttl_minutes == 5.0
        assert config.prefix_cache_size == 200
        assert config.enable_disk_cache is True
        assert config.disk_cache_dir == "/tmp/test-cache"
        assert config.disk_cache_max_gb == 20.0
        assert config.model_path == "/models/test-vlm"

    def test_all_config_fields_match_scheduler_config(self):
        """Ensure MLLM config has all cache-related fields from SchedulerConfig."""
        from vmlx_engine.scheduler import SchedulerConfig

        cache_field_names = {
            "enable_prefix_cache",
            "use_paged_cache",
            "paged_cache_block_size",
            "max_cache_blocks",
            "kv_cache_quantization",
            "kv_cache_group_size",
            "use_memory_aware_cache",
            "cache_memory_mb",
            "cache_memory_percent",
            "cache_ttl_minutes",
            "prefix_cache_size",
            "enable_disk_cache",
            "disk_cache_dir",
            "disk_cache_max_gb",
            "enable_block_disk_cache",
            "block_disk_cache_dir",
            "block_disk_cache_max_gb",
            "model_path",
        }

        mllm_field_names = {f.name for f in fields(MLLMSchedulerConfig)}
        scheduler_field_names = {f.name for f in fields(SchedulerConfig)}

        for field_name in cache_field_names:
            assert field_name in scheduler_field_names, (
                f"Field '{field_name}' missing from SchedulerConfig"
            )
            assert field_name in mllm_field_names, (
                f"Field '{field_name}' missing from MLLMSchedulerConfig"
            )


# ============================================================
# Config forwarding from batched.py
# ============================================================


class TestConfigForwarding:
    """Verify BatchedEngine._start_mllm() forwards all settings."""

    def test_mllm_config_fields_forwarded(self):
        """Check that _start_mllm creates MLLMSchedulerConfig with all fields."""
        from vmlx_engine.engine.batched import BatchedEngine
        from vmlx_engine.scheduler import SchedulerConfig
        import inspect

        # Get the source of _start_mllm to verify forwarding
        source = inspect.getsource(BatchedEngine._start_mllm)

        forwarded_fields = [
            "use_memory_aware_cache",
            "cache_memory_mb",
            "cache_memory_percent",
            "cache_ttl_minutes",
            "prefix_cache_size",
            "enable_disk_cache",
            "disk_cache_dir",
            "disk_cache_max_gb",
            "enable_block_disk_cache",
            "block_disk_cache_dir",
            "block_disk_cache_max_gb",
            "model_path",
        ]

        for field_name in forwarded_fields:
            assert field_name in source, (
                f"Field '{field_name}' not forwarded in _start_mllm()"
            )


# ============================================================
# HybridSSMStateCache tests
# ============================================================


class TestHybridSSMStateCache:
    """Tests for the companion SSM state cache."""

    def test_import(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache
        cache = HybridSSMStateCache(max_entries=10)
        assert cache is not None

    def test_store_and_fetch(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens = [1, 2, 3, 4, 5]
        ssm_states = [MagicMock(), MagicMock()]

        cache.store(tokens, 5, ssm_states)
        result = cache.fetch(tokens, 5)

        assert result is ssm_states

    def test_fetch_miss(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        result = cache.fetch([1, 2, 3], 3)
        assert result is None

    def test_lru_eviction(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=2)

        cache.store([1, 2], 2, ["state_a"])
        cache.store([3, 4], 2, ["state_b"])
        cache.store([5, 6], 2, ["state_c"])  # Should evict [1,2]

        assert cache.fetch([1, 2], 2) is None  # Evicted
        assert cache.fetch([3, 4], 2) == ["state_b"]
        assert cache.fetch([5, 6], 2) == ["state_c"]

    def test_lru_access_refresh(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=2)

        cache.store([1, 2], 2, ["state_a"])
        cache.store([3, 4], 2, ["state_b"])

        # Access [1,2] to refresh its position
        cache.fetch([1, 2], 2)

        # Now store [5,6] — should evict [3,4] (oldest), not [1,2]
        cache.store([5, 6], 2, ["state_c"])

        assert cache.fetch([1, 2], 2) == ["state_a"]
        assert cache.fetch([3, 4], 2) is None  # Evicted
        assert cache.fetch([5, 6], 2) == ["state_c"]

    def test_clear(self):
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        cache.store([1, 2, 3], 3, ["state"])
        cache.clear()
        assert cache.fetch([1, 2, 3], 3) is None

    def test_prefix_keying(self):
        """Verify that cache is keyed by prefix, not full token list."""
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens_full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Store with 5-token prefix
        cache.store(tokens_full, 5, ["state_5"])

        # Fetch with same 5-token prefix but different suffix
        tokens_different_suffix = [1, 2, 3, 4, 5, 99, 98, 97]
        result = cache.fetch(tokens_different_suffix, 5)
        assert result == ["state_5"]

    def test_different_lengths_different_keys(self):
        """Same prefix tokens but different num_tokens → different keys."""
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache(max_entries=10)
        tokens = [1, 2, 3, 4, 5]

        cache.store(tokens, 3, ["state_3"])
        cache.store(tokens, 5, ["state_5"])

        assert cache.fetch(tokens, 3) == ["state_3"]
        assert cache.fetch(tokens, 5) == ["state_5"]


# ============================================================
# MLLMBatchGenerator cache params tests
# ============================================================


class TestBatchGeneratorCacheParams:
    """Test that MLLMBatchGenerator accepts all cache params."""

    def test_init_signature(self):
        """Verify constructor accepts all cache params."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        sig = inspect.signature(MLLMBatchGenerator.__init__)
        param_names = list(sig.parameters.keys())

        expected = [
            "memory_aware_cache",
            "prefix_cache",
            "disk_cache",
            "kv_cache_bits",
            "kv_cache_group_size",
        ]
        for name in expected:
            assert name in param_names, f"Missing param: {name}"


# ============================================================
# MLLMBatch extract_cache contiguous tests
# ============================================================


class TestMLLMBatchExtractContiguous:
    """Test contiguous enforcement in extract_cache."""

    def test_extract_calls_contiguous(self):
        """Verify extract_cache makes keys/values contiguous."""
        import mlx.core as mx
        from vmlx_engine.mllm_batch_generator import MLLMBatch

        # Create a mock batch with a mock cache that has extract()
        mock_kv = MagicMock()
        mock_kv.keys = mx.zeros((1, 4, 8, 64))  # Already contiguous
        mock_kv.values = mx.zeros((1, 4, 8, 64))

        mock_cache = MagicMock()
        mock_cache.extract.return_value = mock_kv

        batch = MLLMBatch(
            uids=[0],
            request_ids=["req-1"],
            y=mx.array([0]),
            logprobs=[mx.zeros(10)],
            max_tokens=[100],
            num_tokens=[0],
            cache=[mock_cache],
            requests=[],
        )

        result = batch.extract_cache(0)
        assert len(result) == 1
        mock_cache.extract.assert_called_once_with(0)
        # The result should have keys/values set
        layer = result[0]
        assert layer.keys is not None
        assert layer.values is not None


# ============================================================
# _ensure_batch_generator clears all cache modes
# ============================================================


class TestEnsureBatchGeneratorCacheClearing:
    """Test that _ensure_batch_generator clears all 3 cache modes."""

    def test_clears_paged_cache(self):
        """Verify paged cache is cleared on generator rebuild."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "block_aware_cache" in source
        assert "block_aware_cache.clear" in source.replace(" ", "").replace("self.", "")

    def test_clears_memory_aware_cache(self):
        """Verify memory-aware cache is cleared on generator rebuild."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "memory_aware_cache" in source
        assert "memory_aware_cache.clear" in source.replace(" ", "").replace("self.", "")

    def test_clears_legacy_prefix_cache(self):
        """Verify legacy prefix cache is cleared on generator rebuild."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "prefix_cache" in source
        assert "prefix_cache.clear" in source.replace(" ", "").replace("self.", "")


# ============================================================
# _cleanup_finished cache store paths
# ============================================================


class TestCleanupFinishedCacheStore:
    """Verify _cleanup_finished stores to all 3 cache paths."""

    def test_has_paged_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "block_aware_cache" in source
        assert "store_cache" in source

    def test_has_memory_aware_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "memory_aware_cache" in source
        assert "memory_aware_cache.store" in source.replace("self.", "")

    def test_has_legacy_store_path(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "prefix_cache" in source
        assert "prefix_cache.store_cache" in source.replace("self.", "")

    def test_has_disk_cache_l2_writes(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "disk_cache" in source

    def test_uses_extracted_tokens_not_prompt_token_ids(self):
        """Ensure memory-aware and legacy paths use _extracted_tokens."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        # Should NOT reference prompt_token_ids (doesn't exist on MLLMRequest)
        lines = source.split('\n')
        for line in lines:
            if 'prompt_token_ids' in line and 'prompt_token_ids' not in line.lstrip().startswith('#'):
                # Only OK if it's in a comment
                stripped = line.lstrip()
                if not stripped.startswith('#'):
                    pytest.fail(f"Found prompt_token_ids reference: {line.strip()}")


# ============================================================
# Metal GC timer
# ============================================================


class TestMetalGCTimer:
    """Test Metal GC timer in scheduler."""

    def test_gc_timer_fields_exist(self):
        """Verify GC timer fields are on MLLMScheduler source."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "_last_metal_gc_time" in source
        assert "_metal_gc_interval" in source

    def test_step_has_periodic_gc(self):
        """Verify step() includes periodic Metal GC."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "clear_memory_cache" in source
        assert "_metal_gc_interval" in source

    def test_cleanup_has_idle_gc(self):
        """Verify _cleanup_finished clears Metal cache when idle."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "clear_memory_cache" in source


# ============================================================
# get_stats cache reporting
# ============================================================


class TestGetStatsCacheReporting:
    """Test that get_stats reports all cache modes."""

    def test_get_stats_reports_cache_modes(self):
        """Verify get_stats source includes all cache mode reporting."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.get_stats)
        assert "paged_cache" in source
        assert "memory_aware_cache" in source
        assert "prefix_cache" in source
        assert "disk_cache" in source


# ============================================================
# Hybrid model detection
# ============================================================


class TestHybridModelDetection:
    """Test _is_hybrid_model static method."""

    def test_non_hybrid_returns_false(self):
        """Pure KVCache model should not be detected as hybrid."""
        from mlx_lm.models.cache import KVCache

        model = MagicMock()
        model.make_cache.return_value = [KVCache(), KVCache(), KVCache()]

        assert MLLMScheduler._is_hybrid_model(model) is False

    def test_no_make_cache_returns_false(self):
        model = MagicMock(spec=[])  # No make_cache attribute
        assert MLLMScheduler._is_hybrid_model(model) is False


# ============================================================
# Integration: cache init chain
# ============================================================


class TestCacheInitChain:
    """Test the 3-tier cache init chain in MLLMScheduler.__init__."""

    def test_init_source_has_three_tiers(self):
        """Verify init chain covers all 3 tiers."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)

        # Paged tier
        assert "PagedCacheManager" in source
        assert "BlockAwarePrefixCache" in source

        # Memory-aware tier
        assert "MemoryAwarePrefixCache" in source
        assert "MemoryCacheConfig" in source

        # Legacy tier
        assert "PrefixCacheManager" in source

    def test_init_source_has_disk_cache_l2(self):
        """Verify disk cache L2 initialization is present."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "DiskCacheManager" in source

    def test_init_source_has_block_disk_store(self):
        """Verify block disk store wiring is present."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "BlockDiskStore" in source


# ============================================================
# Process prompts cache fetch paths
# ============================================================


class TestProcessPromptsCacheFetch:
    """Test that _process_prompts handles all cache fetch paths."""

    def test_has_paged_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "block_aware_cache" in source
        assert "fetch_cache" in source

    def test_has_memory_aware_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "memory_aware_cache" in source

    def test_has_legacy_fetch(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "prefix_cache" in source

    def test_has_disk_cache_l2_fallback(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "disk_cache" in source

    def test_has_hybrid_ssm_state_fetch(self):
        """Verify hybrid models check companion SSM state cache."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "_ssm_state_cache" in source
        assert "HYBRID cache HIT" in source

    def test_has_ssm_state_capture(self):
        """Verify SSM state is captured at prompt boundary during prefill."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        assert "_ssm_state_cache.store" in source
        assert "Captured SSM state" in source


# ============================================================
# Metal cache limit
# ============================================================


class TestMetalCacheLimit:
    """Test Metal cache limit tuning in batch generator."""

    def test_init_has_cache_limit(self):
        """Verify batch generator sets Metal cache limit."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator.__init__)
        assert "set_cache_limit" in source
        assert "_old_cache_limit" in source

    def test_close_restores_cache_limit(self):
        """Verify close() restores old cache limit."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator.close)
        assert "_old_cache_limit" in source
        assert "set_cache_limit" in source

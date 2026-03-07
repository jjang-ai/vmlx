# SPDX-License-Identifier: Apache-2.0
"""
Tests for Continuous Batching with Hybrid/Mamba Models.

These tests verify that the system correctly identifies Hybrid/Mamba architectures
(those returning MambaCache/ArraysCache from make_cache()) and appropriately
configures their caching strategies, avoiding Memory-Aware cache where incompatible.
"""

from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class KVCache:
    pass


class MambaCache:
    pass


class ArraysCache:
    pass


@pytest.fixture
def mock_standard_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[KVCache(), KVCache()])
    return model


@pytest.fixture
def mock_hybrid_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[KVCache(), MambaCache()])
    return model


@pytest.fixture
def mock_pure_mamba_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[MambaCache(), ArraysCache()])
    return model


class TestHybridBatching:

    def test_hybrid_detection(
        self, mock_standard_model, mock_hybrid_model, mock_pure_mamba_model
    ):
        """Test that _is_hybrid_model correctly identifies non-standard caches."""
        from vmlx_engine.scheduler import Scheduler

        # Standard KV-only
        assert Scheduler._is_hybrid_model(mock_standard_model) is False

        # Mixed KV and Mamba
        assert Scheduler._is_hybrid_model(mock_hybrid_model) is True

        # Pure Mamba/SSM
        assert Scheduler._is_hybrid_model(mock_pure_mamba_model) is True

        # Model with no make_cache
        mock_no_cache = MagicMock(spec=[])
        assert Scheduler._is_hybrid_model(mock_no_cache) is False

    @patch("vmlx_engine.scheduler.Scheduler._is_hybrid_model")
    def test_hybrid_forces_legacy_cache(
        self, mock_is_hybrid, mock_hybrid_model
    ):
        """
        Test that a hybrid model bypasses Memory-Aware cache sizing and
        routes to either Legacy Cache or Paged Cache, depending on settings.
        """
        mock_is_hybrid.return_value = True

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        mock_tokenizer = MagicMock(spec=TokenizerWrapper)
        
        # Scenario 1: Memory-Aware caching requested (the default for simple continuous batching)
        config = SchedulerConfig(
            max_num_seqs=4,
            use_memory_aware_cache=True,  # Default
            use_paged_cache=False
        )

        with patch("vmlx_engine.scheduler.logger") as mock_logger:
            # We must trap model properties required inside __init__
            mock_hybrid_model.config = MagicMock()
            
            scheduler = Scheduler(mock_hybrid_model, mock_tokenizer, config)
            
            # Since memory_aware_cache requires KV caching, hybrid model should force it to False
            # and fall back to Legacy KV caching approach (which doesn't dynamically size physical chunks)
            assert scheduler.config.use_memory_aware_cache is False
            
            # The system warns the user
            mock_logger.info.assert_any_call(
                "Non-standard cache model detected (MambaCache/hybrid layers). "
                "Auto-switching to paged cache for correct cache reuse."
            )

    @patch("vmlx_engine.scheduler.Scheduler._is_hybrid_model")
    def test_hybrid_allows_paged_cache(
        self, mock_is_hybrid, mock_hybrid_model
    ):
        """Test that paged cache is permitted with hybrid models."""
        mock_is_hybrid.return_value = True

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        mock_tokenizer = MagicMock(spec=TokenizerWrapper)

        # User explicitly requested Paged Cache
        config = SchedulerConfig(
            max_num_seqs=4,
            use_paged_cache=True
        )

        mock_hybrid_model.config = MagicMock()
        
        with patch("vmlx_engine.scheduler.logger") as mock_logger:
            scheduler = Scheduler(mock_hybrid_model, mock_tokenizer, config)
            
            # It retains paged cache because paged cache implements custom Mamba block mappings
            assert scheduler.config.use_paged_cache is True
            assert scheduler.block_aware_cache is not None

class TestHybridCacheRefLeak:
    """Tests for the hybrid model paged cache ref_count leak fix.

    When a hybrid VLM gets a paged cache HIT but has no companion SSM state,
    the cache blocks are unusable (full prefill required). The fix ensures:
    1. Block refs are released (not leaked) when cache can't be used
    2. Reconstruction is skipped entirely (no wasteful tensor allocation)
    3. The request still processes correctly with full prefill
    """

    def test_release_cache_called_on_hybrid_no_ssm(self):
        """Block refs must be released when hybrid cache hit lacks SSM state."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        # The fix: release_cache is called before continue
        assert "release_cache(req.request_id)" in source
        assert "hybrid — no SSM state, full prefill required" in source

    def test_ssm_check_before_reconstruction(self):
        """SSM state should be checked BEFORE reconstruct_cache to avoid waste."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        lines = source.split('\n')

        # Find the SSM fetch and reconstruct lines
        ssm_fetch_line = None
        reconstruct_line = None
        for i, line in enumerate(lines):
            if '_ssm_state_cache.fetch' in line and ssm_fetch_line is None:
                ssm_fetch_line = i
            if 'reconstruct_cache(block_table)' in line and reconstruct_line is None:
                reconstruct_line = i

        assert ssm_fetch_line is not None, "SSM state cache fetch not found"
        assert reconstruct_line is not None, "reconstruct_cache not found"
        assert ssm_fetch_line < reconstruct_line, (
            f"SSM check (line {ssm_fetch_line}) must come BEFORE "
            f"reconstruct_cache (line {reconstruct_line}) to avoid "
            f"wasteful tensor allocation for hybrid models without SSM state"
        )

    def test_continue_skips_reconstruction(self):
        """When hybrid has no SSM state, continue should skip reconstruction."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        # After release_cache, continue skips the rest (including reconstruct)
        assert "release_cache(req.request_id)" in source
        # The continue must appear near the release_cache call
        release_idx = source.index("release_cache(req.request_id)")
        # Find the next 'continue' after release_cache
        continue_idx = source.index("continue", release_idx)
        # Should be within ~200 chars (same block)
        assert continue_idx - release_idx < 200, (
            "continue should immediately follow release_cache"
        )

    def test_paged_cache_detach_decrements_refs(self):
        """delete_block_table should decrement ref_counts via free_block."""
        from vmlx_engine.paged_cache import PagedCacheManager

        mgr = PagedCacheManager(block_size=4, max_blocks=10)

        # Create a block table and allocate blocks
        table = mgr.create_block_table("test-req")
        block = mgr.get_new_blocks(1)[0]
        block.token_count = 4
        table.block_ids.append(block.block_id)

        # Increment ref (simulating fetch_cache sharing)
        mgr.increment_ref(block.block_id)
        assert block.ref_count == 2

        # delete_block_table should decrement
        mgr.delete_block_table("test-req")
        assert block.ref_count == 1  # Back to original ref from cache storage

    def test_detach_does_not_free_blocks(self):
        """detach_request should NOT decrement ref_counts (by design)."""
        from vmlx_engine.paged_cache import PagedCacheManager

        mgr = PagedCacheManager(block_size=4, max_blocks=10)

        table = mgr.create_block_table("test-req")
        block = mgr.get_new_blocks(1)[0]
        block.token_count = 4
        table.block_ids.append(block.block_id)
        original_ref = block.ref_count

        mgr.detach_request("test-req")
        # ref_count unchanged — detach only removes tracking, not block refs
        assert block.ref_count == original_ref


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

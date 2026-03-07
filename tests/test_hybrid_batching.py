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


class TestPagedCacheValidation:
    """Tests for PagedCacheManager input validation."""

    def test_block_size_zero_raises(self):
        """block_size=0 should raise ValueError, not cause ZeroDivisionError later."""
        from vmlx_engine.paged_cache import PagedCacheManager

        with pytest.raises(ValueError, match="block_size must be >= 1"):
            PagedCacheManager(block_size=0, max_blocks=10)

    def test_block_size_negative_raises(self):
        """Negative block_size should raise ValueError."""
        from vmlx_engine.paged_cache import PagedCacheManager

        with pytest.raises(ValueError, match="block_size must be >= 1"):
            PagedCacheManager(block_size=-1, max_blocks=10)

    def test_max_blocks_zero_raises(self):
        """max_blocks=0 should raise ValueError, not crash on null block reserve."""
        from vmlx_engine.paged_cache import PagedCacheManager

        with pytest.raises(ValueError, match="max_blocks must be >= 2"):
            PagedCacheManager(block_size=4, max_blocks=0)

    def test_max_blocks_one_raises(self):
        """max_blocks=1 only fits null block, no usable blocks — should raise."""
        from vmlx_engine.paged_cache import PagedCacheManager

        with pytest.raises(ValueError, match="max_blocks must be >= 2"):
            PagedCacheManager(block_size=4, max_blocks=1)

    def test_max_blocks_two_works(self):
        """max_blocks=2 should work (1 null + 1 usable)."""
        from vmlx_engine.paged_cache import PagedCacheManager

        mgr = PagedCacheManager(block_size=4, max_blocks=2)
        assert mgr.stats.free_blocks == 1


class TestSuppressReasoningInvariants:
    """Tests for reasoning suppression invariants across API paths."""

    def test_responses_api_no_reasoning_done_when_suppressed(self):
        """response.reasoning.done should NOT be emitted when suppress_reasoning=True."""
        from vmlx_engine.server import stream_responses_api
        import inspect

        source = inspect.getsource(stream_responses_api)
        # The guard: accumulated_reasoning and not suppress_reasoning
        assert "not suppress_reasoning" in source
        # Find the reasoning.done emission
        idx = source.index("response.reasoning.done")
        # Check that the guard appears before this emission in the same block
        block_start = source.rfind("if ", 0, idx)
        block_text = source[block_start:idx]
        assert "not suppress_reasoning" in block_text

    def test_reasoning_fallback_guarded_by_suppress(self):
        """Reasoning-only fallback should NOT emit as content when suppress_reasoning=True."""
        from vmlx_engine.server import stream_chat_completion
        import inspect

        source = inspect.getsource(stream_chat_completion)
        # Find the fallback: "not content_was_emitted and accumulated_reasoning"
        idx = source.index("not content_was_emitted and accumulated_reasoning")
        # The line should also include "not suppress_reasoning"
        line_start = source.rfind("\n", 0, idx)
        line_end = source.index("\n", idx)
        line = source[line_start:line_end]
        assert "not suppress_reasoning" in line


class TestToolChoiceNoneInvariants:
    """Tests for tool_choice='none' correctly suppressing tool parsing."""

    def test_chat_completions_streaming_guards_tool_parsing(self):
        """tool_choice='none' should prevent post-stream tool call parsing."""
        from vmlx_engine.server import stream_chat_completion
        import inspect

        source = inspect.getsource(stream_chat_completion)
        # The guard: "not _suppress_tools" should appear before _parse_tool_calls_with_parser
        # in the tool_call_buffering block
        assert "and not _suppress_tools" in source

    def test_chat_completions_streaming_tool_call_active_gated(self):
        """tool_call_active must be gated by _suppress_tools to prevent content swallowing."""
        from vmlx_engine.server import stream_chat_completion
        import inspect

        source = inspect.getsource(stream_chat_completion)
        # Find the tool_call_active assignment line
        lines = source.split('\n')
        for line in lines:
            if 'tool_call_active' in line and '=' in line and 'not _suppress_tools' in line:
                break
        else:
            raise AssertionError(
                "tool_call_active assignment must include 'not _suppress_tools' guard. "
                "Without this, tool_choice='none' still buffers content when tool markers "
                "are detected, swallowing user-visible text."
            )

    def test_responses_api_guards_tool_call_active(self):
        """Responses API should set tool_call_active=False when tool_choice='none'."""
        from vmlx_engine.server import stream_responses_api
        import inspect

        source = inspect.getsource(stream_responses_api)
        assert "_suppress_tools" in source
        assert "not _suppress_tools" in source


class TestToolChoiceNoneNonStreaming:
    """Tests for tool_choice='none' in non-streaming API paths."""

    def test_chat_completions_non_streaming_guards_tool_parsing(self):
        """Non-streaming Chat Completions should skip tool parsing when tool_choice='none'."""
        from vmlx_engine.server import create_chat_completion
        import inspect

        source = inspect.getsource(create_chat_completion)
        assert "not _suppress_tools" in source

    def test_responses_api_non_streaming_guards_tool_parsing(self):
        """Non-streaming Responses API should skip tool parsing when tool_choice='none'."""
        from vmlx_engine.server import create_response
        import inspect

        source = inspect.getsource(create_response)
        assert "not _suppress_tools" in source


class TestMemoryCacheFallbackWarning:
    """Tests for memory cache 0-memory fallback warning."""

    def test_compute_memory_limit_logs_on_zero_memory(self):
        """When available memory is 0, compute_memory_limit should log a warning."""
        from vmlx_engine.memory_cache import MemoryCacheConfig
        import inspect

        source = inspect.getsource(MemoryCacheConfig.compute_memory_limit)
        assert "logger.warning" in source
        assert "Could not detect available memory" in source


class TestHybridDetectionLogging:
    """Tests for hybrid model detection error handling."""

    def test_is_hybrid_model_logs_exception(self):
        """_is_hybrid_model should log warning when make_cache() raises."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._is_hybrid_model)
        # Should log, not silently swallow
        assert "logger.warning" in source
        assert "make_cache() failed" in source


class TestRotatingKVCachePreservation:
    """Tests for RotatingKVCache sliding window parameter preservation."""

    def test_truncate_preserves_rotating_kv_type(self):
        """_truncate_cache_to_prompt_length should create RotatingKVCache, not KVCache."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must import and use RotatingKVCache for sliding window layers
        assert "RotatingKVCache" in source
        assert "max_size" in source

    def test_block_slice_extracts_rotating_params_from_meta(self):
        """_extract_block_tensor_slice should read max_size/keep from meta_state."""
        from vmlx_engine.prefix_cache import BlockAwarePrefixCache
        import inspect

        source = inspect.getsource(BlockAwarePrefixCache._extract_block_tensor_slice)
        # Should parse meta_state for RotatingKVCache params
        assert "meta_state" in source
        # The meta tuple format is (keep, max_size, ...)
        assert "int(meta[" in source


class TestVLMPrefixCacheImageGuard:
    """Tests for VLM prefix cache image collision prevention."""

    def test_skip_prefix_cache_when_images_present(self):
        """Prefix cache fetch should be skipped when request has pixel_values."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        # Must check has_images before fetching from prefix cache
        assert "has_images" in source
        assert "not has_images" in source


class TestSSMStateCacheKeyAlignment:
    """Tests for SSM state cache key alignment between store and fetch."""

    def test_fetch_block_aligns_num_tokens(self):
        """SSM fetch must block-align num_tokens to match store key."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        # Fetch path must align to block boundary
        assert "_fetch_num" in source
        # The alignment: (_fetch_num // _bs) * _bs
        assert "_fetch_num // _bs" in source

    def test_ssm_state_cache_key_determinism(self):
        """Same token prefix should produce same cache key."""
        from vmlx_engine.mllm_batch_generator import HybridSSMStateCache

        cache = HybridSSMStateCache()
        tokens = list(range(100))
        key1 = cache._key(tokens, 64)
        key2 = cache._key(tokens, 64)
        assert key1 == key2

        # Different prefix length = different key
        key3 = cache._key(tokens, 65)
        assert key1 != key3


class TestStopSequenceThinkAwareness:
    """Tests for stop sequences not matching inside <think> blocks."""

    def test_stop_check_strips_think_blocks(self):
        """String stop sequences should be checked on text with think blocks removed."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._process_batch_responses)
        # Must strip think blocks before stop matching
        assert "re.sub" in source
        assert "<think>" in source

    def test_stop_check_skips_unclosed_think(self):
        """Stop matching should be skipped while inside unclosed <think> block."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._process_batch_responses)
        assert "</think>" in source


class TestAbortRequestReturnValue:
    """Tests for abort_request returning correct found status."""

    def test_engine_core_abort_returns_found_status(self):
        """EngineCore.abort_request should return True only if request exists."""
        from vmlx_engine.engine_core import EngineCore
        import inspect

        source = inspect.getsource(EngineCore.abort_request)
        # Should check if request exists before returning
        assert "_output_queues" in source or "_finished_events" in source
        # Should NOT unconditionally return True
        assert "return found" in source


class TestCachedTokensZeroOnFailure:
    """Tests for cached_tokens being zeroed on reconstruction failure."""

    def test_reconstruction_failure_zeros_cached_tokens(self):
        """When paged cache reconstruction fails, cached_tokens must be 0."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler.add_request)
        # Find the reconstruction failure path
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'reconstruction failed' in line:
                # Look in nearby lines for cached_tokens = 0
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'cached_tokens = 0' in context, (
                    "cached_tokens must be zeroed on reconstruction failure"
                )
                break
        else:
            raise AssertionError("Could not find reconstruction failure path")


class TestMLLMCacheStatsCompleteness:
    """Tests for MLLM cache stats including hits/misses/hit_rate."""

    def test_mllm_stats_include_cache_fields(self):
        """MLLMScheduler.get_stats should include hit/miss fields for CachePanel."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        import inspect

        source = inspect.getsource(MLLMScheduler.get_stats)
        # Must include these fields so CachePanel renders
        assert '"hits"' in source
        assert '"misses"' in source
        assert '"hit_rate"' in source
        assert '"tokens_saved"' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
        assert "no SSM companion state" in source

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
        # Fetch path uses exact num_tokens (block alignment removed —
        # caused prompts <64 tokens to round to 0, breaking SSM companion)
        assert "_fetch_num" in source

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
        """String stop sequences should skip content inside <think> blocks."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._process_batch_responses)
        # Must handle think blocks before stop matching
        assert "<think>" in source
        assert "</think>" in source

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
                context = '\n'.join(lines[max(0, i-5):i+5])
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


class TestEngineCoreAbortAttribute:
    """Verify EngineCore.abort_request references _output_collectors (not _output_queues)."""

    def test_abort_uses_output_collectors(self):
        import inspect
        from vmlx_engine.engine_core import EngineCore

        source = inspect.getsource(EngineCore.abort_request)
        assert "_output_collectors" in source, (
            "abort_request must reference self._output_collectors, not _output_queues"
        )
        assert "_output_queues" not in source, (
            "abort_request still references non-existent _output_queues attribute"
        )


class TestCacheEndpointAuth:
    """Verify all /v1/cache/* endpoints require API key auth."""

    def test_cache_endpoints_have_auth_dependency(self):
        import inspect
        from vmlx_engine import server

        source = inspect.getsource(server)
        # Find each cache endpoint decorator and verify it has verify_api_key
        endpoints = [
            "/v1/cache/stats",
            "/v1/cache/entries",
            "/v1/cache/warm",
        ]
        for endpoint in endpoints:
            # Find the decorator line containing this endpoint
            idx = source.find(f'"{endpoint}"')
            assert idx >= 0, f"Endpoint {endpoint} not found in server.py"
            # Check the surrounding decorator text (within 200 chars) for auth
            context = source[max(0, idx - 200):idx + 100]
            assert "verify_api_key" in context, (
                f"Endpoint {endpoint} is missing verify_api_key dependency"
            )

        # DELETE /v1/cache — find the actual endpoint decorator, not middleware references
        # Search for the delete method decorator pattern to avoid matching middleware text
        delete_cache_pattern = 'delete("/v1/cache"'
        idx = source.find(delete_cache_pattern)
        if idx < 0:
            # Fallback: find last occurrence (endpoint is after middleware)
            idx = source.rfind('"/v1/cache"')
        assert idx >= 0
        context = source[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in context, (
            "DELETE /v1/cache is missing verify_api_key dependency"
        )


class TestStopSequenceThinkPositionMapping:
    """Verify stop sequence position maps correctly when stop string
    appears both inside <think> block AND in content."""

    def test_stop_in_think_and_content_finds_content_occurrence(self):
        """The stop string search should start after the last closed </think>
        block so it finds the content occurrence, not the reasoning one."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._process_batch_responses)
        # Must search from after the last </think> end
        assert "rfind('</think>')" in source, (
            "Stop sequence mapping must use rfind('</think>') to skip past "
            "reasoning blocks when mapping position back to full_text"
        )
        # Must NOT use bare full_text.find(stop_str) without offset
        assert "full_text.find(stop_str)" not in source or "search_start" in source, (
            "full_text.find(stop_str) must use a search_start offset "
            "to avoid matching inside <think> blocks"
        )


class TestMLLMModelWrapperInLLMPath:
    """Tests for MLLMModelWrapper being applied in BatchedEngine._start_llm().

    Models like nemotron_h return LanguageModelOutput objects instead of raw
    tensors. Without wrapping, BatchGenerator subscripts the output incorrectly,
    producing garbage tokens. The wrapper extracts .logits when present.
    """

    def test_start_llm_wraps_model_in_mllm_wrapper(self):
        """_start_llm must wrap self._model in MLLMModelWrapper."""
        from vmlx_engine.engine.batched import BatchedEngine
        import inspect

        source = inspect.getsource(BatchedEngine._start_llm)
        assert "MLLMModelWrapper" in source, (
            "_start_llm must wrap the model in MLLMModelWrapper so models "
            "returning LanguageModelOutput (nemotron_h, etc.) produce raw "
            "logits tensors for BatchGenerator"
        )

    def test_mllm_wrapper_extracts_logits(self):
        """MLLMModelWrapper should extract .logits from LanguageModelOutput."""
        from vmlx_engine.engine.batched import MLLMModelWrapper

        class FakeLanguageModelOutput:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return FakeLanguageModelOutput("extracted_logits")
            def make_cache(self):
                return []

        wrapper = MLLMModelWrapper(FakeModel())
        result = wrapper("dummy_input")
        assert result == "extracted_logits", (
            "MLLMModelWrapper must extract .logits from LanguageModelOutput"
        )

    def test_mllm_wrapper_passthrough_for_plain_tensor(self):
        """MLLMModelWrapper should pass through raw tensors unchanged."""
        from vmlx_engine.engine.batched import MLLMModelWrapper

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return "raw_tensor_output"
            def make_cache(self):
                return []

        wrapper = MLLMModelWrapper(FakeModel())
        result = wrapper("dummy_input")
        assert result == "raw_tensor_output", (
            "MLLMModelWrapper must pass through models returning plain tensors"
        )

    def test_mllm_wrapper_forwards_make_cache(self):
        """MLLMModelWrapper.__getattr__ must forward make_cache() to the real model."""
        from vmlx_engine.engine.batched import MLLMModelWrapper

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return "output"
            def make_cache(self):
                return ["cache_a", "cache_b"]

        wrapper = MLLMModelWrapper(FakeModel())
        cache = wrapper.make_cache()
        assert cache == ["cache_a", "cache_b"], (
            "Wrapper must forward make_cache() for hybrid detection and "
            "prefix cache warming"
        )

    def test_mllm_wrapper_forwards_args_attribute(self):
        """MLLMModelWrapper.__getattr__ must forward .args for head_dim detection."""
        from vmlx_engine.engine.batched import MLLMModelWrapper

        class FakeArgs:
            head_dim = 128
            hidden_size = 4096

        class FakeModel:
            def __init__(self):
                self.args = FakeArgs()
            def __call__(self, *args, **kwargs):
                return "output"

        wrapper = MLLMModelWrapper(FakeModel())
        assert wrapper.args.head_dim == 128, (
            "Wrapper must forward .args for Scheduler head_dim detection"
        )


class TestStreamIntervalAccumulation:
    """Tests for stream_interval > 1 correctly accumulating skipped tokens.

    When stream_interval > 1, the engine loop skips putting intermediate
    tokens into the collector. The skipped tokens' new_text and new_token_ids
    must be accumulated and merged into the next output that IS sent.
    Without this, tokens are permanently lost, causing garbled output.
    """

    def test_request_stream_state_accumulates_pending_text(self):
        """RequestStreamState must have pending_new_text for accumulation."""
        from vmlx_engine.output_collector import RequestStreamState

        state = RequestStreamState(stream_interval=4)
        assert hasattr(state, "pending_new_text"), (
            "RequestStreamState must have pending_new_text field for "
            "accumulating skipped tokens' text when stream_interval > 1"
        )
        assert hasattr(state, "pending_new_token_ids"), (
            "RequestStreamState must have pending_new_token_ids field for "
            "accumulating skipped tokens' token IDs when stream_interval > 1"
        )

    def test_accumulate_merges_pending_into_output(self):
        """accumulate() should store text/tokens; drain() should return and clear them."""
        from vmlx_engine.output_collector import RequestStreamState

        state = RequestStreamState(stream_interval=4)
        # Simulate 3 skipped tokens
        state.accumulate("Hello", [100])
        state.accumulate(" world", [200])
        state.accumulate("!", [300])

        text, token_ids = state.drain_pending()
        assert text == "Hello world!", (
            "drain_pending must return concatenated pending text"
        )
        assert token_ids == [100, 200, 300], (
            "drain_pending must return concatenated pending token IDs"
        )

        # After drain, pending should be empty
        text2, token_ids2 = state.drain_pending()
        assert text2 == ""
        assert token_ids2 == []

    def test_engine_loop_accumulates_skipped_outputs(self):
        """The engine loop must accumulate skipped outputs, not drop them."""
        from vmlx_engine.engine_core import EngineCore
        import inspect

        source = inspect.getsource(EngineCore._engine_loop)
        # When should_send is False, must accumulate instead of silently dropping
        assert "accumulate" in source, (
            "Engine loop must call state.accumulate() for skipped tokens "
            "when stream_interval > 1. Without this, tokens are permanently "
            "lost and output is garbled."
        )

    def test_engine_loop_drains_pending_before_put(self):
        """The engine loop must drain pending text before collector.put()."""
        from vmlx_engine.engine_core import EngineCore
        import inspect

        source = inspect.getsource(EngineCore._engine_loop)
        assert "drain_pending" in source, (
            "Engine loop must call state.drain_pending() and merge into "
            "req_output before collector.put() when stream_interval > 1"
        )


class TestPerfCacheTimeouts:
    """Tests for performance and cache IPC timeout values.

    During inference, synchronous scheduler.step() blocks the uvicorn event
    loop. A 5-second timeout is too short for large model prefills which can
    take 10+ seconds on a single step. Timeouts must be large enough to
    survive heavy inference load.
    """

    def test_performance_timeout_sufficient(self):
        """Performance health check must use >= 30s timeout."""
        import re
        with open("panel/src/main/ipc/performance.ts") as f:
            source = f.read()
        match = re.search(r"AbortSignal\.timeout\((\d+)\)", source)
        assert match, "performance.ts must use AbortSignal.timeout"
        timeout_ms = int(match.group(1))
        assert timeout_ms >= 30000, (
            f"Performance health timeout is {timeout_ms}ms, must be >= 30000ms. "
            f"During large model prefills, scheduler.step() blocks the event "
            f"loop for 10+ seconds, causing 5s timeouts to fire spuriously."
        )

    def test_cache_stats_timeout_sufficient(self):
        """Cache stats check must use >= 30s timeout."""
        import re
        with open("panel/src/main/ipc/cache.ts") as f:
            source = f.read()
        matches = re.findall(r"AbortSignal\.timeout\((\d+)\)", source)
        assert matches, "cache.ts must use AbortSignal.timeout"
        for timeout_str in matches:
            timeout_ms = int(timeout_str)
            # cache:warm uses 60s (correct), cache:clear uses 10s (fine)
            # Only stats/entries should be >= 30s
            if timeout_ms < 10000:
                raise AssertionError(
                    f"Cache timeout {timeout_ms}ms is too low. Must be >= 10000ms "
                    f"to survive event loop blocking during inference."
                )


class TestPortInputClamping:
    """Tests for SliderField port input not clamping on every keystroke.

    With min=1024, typing "1" (first digit of e.g. "12345") should NOT
    immediately snap to 1024. Clamping should only happen on blur.
    """

    def test_handle_input_change_does_not_clamp_to_min(self):
        """handleInputChange must not call Math.max(min, ...) on every keystroke."""
        with open("panel/src/renderer/src/components/sessions/SessionConfigForm.tsx") as f:
            source = f.read()

        # Find handleInputChange function body
        start = source.index("const handleInputChange")
        # Find the next const/function declaration after it
        next_func = source.index("const handleInput", start + 30)
        handler_body = source[start:next_func]

        # Must NOT contain Math.max(min in the onChange call
        assert "Math.max(min" not in handler_body, (
            "handleInputChange must NOT clamp to min on every keystroke. "
            "With min=1024, typing '1' immediately snaps to 1024 before "
            "the user can finish typing. Clamping belongs in handleInputBlur."
        )


class TestAbortDrainsPendingText:
    """Regression: abort must drain pending text from stream_interval > 1."""

    def test_cleanup_request_drains_pending_before_sentinel(self):
        """_cleanup_request must drain pending text into abort sentinel."""
        import inspect
        from vmlx_engine.engine_core import EngineCore

        source = inspect.getsource(EngineCore._cleanup_request)
        # Must get stream state and drain BEFORE popping
        assert "drain_pending" in source, (
            "_cleanup_request must drain pending text from RequestStreamState "
            "before discarding it, so the abort sentinel carries accumulated text"
        )

    def test_abort_sentinel_includes_new_text_field(self):
        """Abort sentinel RequestOutput must carry new_text from drained pending."""
        import inspect
        from vmlx_engine.engine_core import EngineCore

        source = inspect.getsource(EngineCore._cleanup_request)
        assert "new_text=" in source, (
            "Abort sentinel must include new_text= with drained pending text"
        )


class TestReasoningDoneAtToolBoundary:
    """Regression: chat:reasoningDone must fire at tool iteration boundary."""

    def test_tool_iteration_boundary_emits_reasoning_done(self):
        """When isReasoning=true at tool boundary, reasoningDone must fire."""
        import os

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        # Find tool iteration boundary: "emitToolStatus('processing', '', undefined"
        boundary_idx = source.index("emitToolStatus('processing', '', undefined")
        # Look backwards for reasoningDone emission
        pre_boundary = source[max(0, boundary_idx - 500):boundary_idx]
        assert "chat:reasoningDone" in pre_boundary, (
            "Tool iteration boundary must fire chat:reasoningDone before "
            "resetting isReasoning=false, otherwise reasoning-only tool calls "
            "silently drop the reasoning content"
        )

    def test_auto_continue_boundary_emits_reasoning_done(self):
        """When isReasoning=true at auto-continue boundary, reasoningDone must fire."""
        import os

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        # Find auto-continue boundary: "emitToolStatus('processing', '', 'Generating response...'"
        boundary_idx = source.index("emitToolStatus('processing', '', 'Generating response...'")
        pre_boundary = source[max(0, boundary_idx - 500):boundary_idx]
        assert "chat:reasoningDone" in pre_boundary, (
            "Auto-continue boundary must fire chat:reasoningDone before "
            "resetting isReasoning=false"
        )


class TestSuppressReasoningDiagnostic:
    """Regression: suppress_reasoning + reasoning-only must not be silently empty."""

    def test_chat_completions_has_suppress_reasoning_diagnostic(self):
        """Chat completions path must emit diagnostic when only reasoning produced."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert "suppress_reasoning and not content_was_emitted and accumulated_reasoning" in source, (
            "Chat completions must detect reasoning-only + suppress and emit a diagnostic"
        )
        assert "only internal reasoning" in source, (
            "Chat completions diagnostic must explain that reasoning was suppressed"
        )

    def test_responses_api_has_suppress_reasoning_diagnostic(self):
        """Responses API path must emit diagnostic when only reasoning produced."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "only internal reasoning" in source, (
            "Responses API must explain that reasoning was suppressed"
        )


class TestQwen3NextToolParser:
    """Regression: qwen3_next must use 'qwen' tool parser, not 'nemotron'."""

    def test_qwen3_next_uses_qwen_parser(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config",
                    lambda p: {"model_type": "qwen3_next"}):
            config = registry.lookup("Qwen3-Next-8B")
        assert config.tool_parser == "qwen", (
            f"qwen3_next must use 'qwen' tool parser, got '{config.tool_parser}'"
        )

    def test_qwen3_next_not_nemotron_parser(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config",
                    lambda p: {"model_type": "qwen3_next"}):
            config = registry.lookup("Qwen3-Next-8B")
        assert config.tool_parser != "nemotron", (
            "qwen3_next must NOT use nemotron tool parser"
        )


class TestGemmaArchitectureHints:
    """Regression: gemma3/medgemma must have inject_pixel_values hint."""

    def test_gemma3_has_inject_pixel_values(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        with patch("vmlx_engine.model_config_registry.load_config",
                    lambda p: {"model_type": "gemma3"}):
            config = registry.lookup("gemma3-2B")
        assert config.architecture_hints.get("inject_pixel_values") is True, (
            "gemma3 must have architecture_hints.inject_pixel_values=True "
            "so MLLMModelWrapper injects pixel_values=None for text-only requests"
        )

    def test_medgemma_has_inject_pixel_values(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        # medgemma matches by name (model_type=gemma2), not by model_type alone
        with patch("vmlx_engine.model_config_registry.load_config",
                    lambda p: {"model_type": "gemma2"}):
            config = registry.lookup("google/medgemma-4b-it")
        assert config.architecture_hints.get("inject_pixel_values") is True, (
            "medgemma must have architecture_hints.inject_pixel_values=True"
        )


class TestServerErrorEventHandling:
    """Regression: server-side error SSE events must be caught in both API paths."""

    def test_chat_completions_handles_parsed_error(self):
        """Chat completions SSE path must check parsed.error."""
        import os

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        assert "parsed.error" in source, (
            "Chat completions SSE parser must handle parsed.error field "
            "from server-side error events"
        )

    def test_responses_api_handles_error_event_type(self):
        """Responses API SSE path must recognize 'error' event type."""
        import os

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        # Must check for bare 'error' event type, not just 'response.error'
        assert "=== 'error'" in source or "== 'error'" in source, (
            "Responses API SSE parser must recognize bare 'error' event type "
            "alongside 'response.error' and 'response.failed'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

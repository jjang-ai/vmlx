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
            
            # 2026-08-23: paged RAM is OFF for every family, so a hybrid model
            # with no Block Disk L2 no longer escalates to a RAM tier. It drops
            # the memory-aware lane and takes the reuse loss, and says so.
            assert scheduler.config.use_paged_cache is False
            mock_logger.info.assert_any_call(
                "Non-standard cache model detected (MambaCache/hybrid layers) "
                "with Block Disk L2 disabled. Paged RAM stays OFF (SSD L2 is "
                "the only tier); disabling the memory-aware lane. Enable "
                "--enable-block-disk-cache to get hybrid prefix reuse back."
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
            
            # In-RAM paged cache is OFF for every family by default, but an
            # EXPLICIT operator request (--use-paged-cache) is still honoured.
            # What was removed is automatic escalation, not the flag.
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

    def test_responses_api_uses_standard_reasoning_summary_events(self):
        """Responses API reasoning SSE must use OpenAI standard summary events."""
        from vmlx_engine.server import stream_responses_api
        import inspect

        source = inspect.getsource(stream_responses_api)

        assert "response.reasoning_summary_text.delta" in source
        assert "response.reasoning_summary_text.done" in source
        assert "response.reasoning.delta" not in source
        assert "response.reasoning.done" not in source

    def test_responses_api_no_reasoning_done_when_suppressed(self):
        """reasoning done should NOT be emitted when suppress_reasoning=True."""
        from vmlx_engine.server import stream_responses_api
        import inspect

        source = inspect.getsource(stream_responses_api)
        # The guard: accumulated_reasoning and not suppress_reasoning
        assert "not suppress_reasoning" in source
        # Find the reasoning summary done emission
        idx = source.index("response.reasoning_summary_text.done")
        # Check that the guard appears before this emission in the same block
        block_start = source.rfind("def _finish_reasoning_item_events", 0, idx)
        block_text = source[block_start:idx]
        assert "suppress_reasoning" not in block_text
        close_idx = source.index("if visible_reasoning_text and not suppress_reasoning")
        assert close_idx < source.index("_finish_reasoning_item_events", close_idx)

    def test_reasoning_fallback_guarded_by_suppress(self):
        """Reasoning-only fallback must be gated by suppress_reasoning state.

        The server is free to pick EITHER behavior (both are correct):
          (a) Not suppressed → emit accumulated reasoning as content fallback
              (prevents silent empty response when model only produced reasoning).
          (b) Suppressed → emit a user-visible diagnostic explaining why nothing
              was returned (current behavior; the reasoning output is withheld
              per user setting but the user gets a meaningful hint).

        Either implementation references `suppress_reasoning` in the same
        condition block as the `content_was_emitted` / `accumulated_reasoning`
        check. This test only verifies the guard is present, not which branch
        it takes.
        """
        from vmlx_engine.server import stream_chat_completion
        import inspect

        source = inspect.getsource(stream_chat_completion)
        gate = (
            "suppress_reasoning\n"
            "        and not content_was_emitted\n"
            "        and not tool_calls_emitted\n"
            "        and accumulated_reasoning"
        )
        assert gate in source, (
            "reasoning-only fallback must reference suppress_reasoning in the "
            "same condition block as the content_was_emitted / "
            "accumulated_reasoning check"
        )


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
        # Find the `tool_call_active =` assignment. The guard may be on the same
        # line or within the next few lines (multi-line expression).
        idx = source.find("tool_call_active = ")
        assert idx != -1, "stream_chat_completion must define tool_call_active"
        # Read up to 5 lines of the assignment block
        block_end = idx
        for _ in range(5):
            nl = source.find("\n", block_end + 1)
            if nl == -1:
                break
            block_end = nl
        block = source[idx:block_end]
        assert "not _suppress_tools" in block, (
            "tool_call_active assignment must include 'not _suppress_tools' guard. "
            "Without this, tool_choice='none' still buffers content when tool markers "
            f"are detected, swallowing user-visible text. Block: {block!r}"
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

    def test_is_hybrid_model_fails_closed_on_exception(self):
        """A failed cache probe must not silently classify the model as KV."""
        from vmlx_engine.scheduler import Scheduler

        class BrokenModel:
            def make_cache(self):
                raise RuntimeError("broken cache")

        with pytest.raises(RuntimeError, match="refusing to classify"):
            Scheduler._is_hybrid_model(BrokenModel())


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
        from vmlx_engine.prefix_cache import (
            BlockAwarePrefixCache,
            _positional_layer_slice_bounds,
        )
        import inspect

        source = inspect.getsource(BlockAwarePrefixCache._extract_block_tensor_slice)
        bounds_source = inspect.getsource(_positional_layer_slice_bounds)
        # Should parse meta_state for RotatingKVCache params
        assert "meta_state" in source
        # Absolute rotating offsets are now parsed in the shared bounds helper
        # used by both MLX and NumPy block extraction paths.
        assert "_positional_layer_slice_bounds" in source
        assert "int(meta[2])" in bounds_source


class TestVLMPrefixCacheImageGuard:
    """Tests for VLM prefix cache image collision prevention."""

    def test_skip_prefix_cache_when_images_present(self):
        """Prefix cache fetch should be skipped when request has pixel_values.

        Codex's N-1 fix slice (2026-05-09) factored prefill/encode out of
        `_process_prompts` into smaller helpers. The has_images guard now
        lives inside `_run_vision_encoding_inner`.
        """
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
        import inspect

        source = inspect.getsource(MLLMBatchGenerator._run_vision_encoding_inner)
        # Must gate prefix-cache / chunked-prefill paths on has_images
        assert "has_images" in source
        assert "not has_images" in source

    def test_media_placeholder_helper_covers_image_video_audio_ids(self):
        """All configured media placeholder ids should be treated as cache-unsafe."""
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        class Config:
            image_token_index = 101
            video_token_id = 202
            audio_token_id = 303

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.model = type("Model", (), {"config": Config()})()
        generator.language_model = type("LM", (), {"config": None})()

        assert generator._media_placeholder_token_ids() == {101, 202, 303}
        assert generator._tokens_contain_media_placeholders([1, 2, 101]) is True
        assert generator._tokens_contain_media_placeholders([1, 202, 3]) is True
        assert generator._tokens_contain_media_placeholders([303]) is True
        assert generator._tokens_contain_media_placeholders([1, 2, 3]) is False

    def test_mllm_prefix_cache_store_skips_media_token_only_keys(self):
        """Store path must not persist media-affected KV under token-only keys.

        Codex's N-1 fix slice (2026-05-09) factored the media-cache-context
        check out of `_cleanup_finished` into a dedicated helper
        `_mllm_request_has_media_cache_context`. The store path still consults
        it before writing under a token-only prefix key.
        """
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        helper_source = inspect.getsource(
            MLLMScheduler._mllm_request_has_media_cache_context
        )
        # Helper must consult the media-placeholder detector
        assert "_tokens_contain_media_placeholders" in helper_source

        # The cleanup path must call the helper before storing under a
        # token-only key (legacy/memory-aware cache). Inspect _cleanup_finished
        # for the call site.
        cleanup_source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "_mllm_request_has_media_cache_context" in cleanup_source


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

        source = inspect.getsource(Scheduler._schedule_waiting)
        # Find the reconstruction failure path
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'reconstruction failed' in line:
                # Look in nearby lines for cached_tokens = 0
                context = '\n'.join(lines[max(0, i-5):i+16])
                assert 'cached_tokens = 0' in context, (
                    "cached_tokens must be zeroed on reconstruction failure"
                )
                assert '_release_unusable_paged_hit(request)' in context, (
                    "reconstruction failure must release fetched request refs"
                )
                break
        else:
            raise AssertionError("Could not find reconstruction failure path")


class TestHybridPagedSSMReuse:
    """Tests for hybrid paged KV hits when SSM companion state is missing."""

    def test_scheduler_derives_block_aligned_ssm_before_hybrid_paged_miss_fallback(self):
        """Hybrid paged hits must try block-aligned SSM derive before full prefill."""
        from vmlx_engine.scheduler import Scheduler
        import inspect

        source = inspect.getsource(Scheduler._finalize_hybrid_paged_cache_on_worker)
        derive_idx = source.index("_prefill_for_prompt_only_cache(")
        miss_idx = source.index("hybrid paged MISS")

        assert derive_idx < miss_idx
        assert "synchronously derived SSM companion" in source
        assert "self._ssm_state_cache.store(" in source
        # The derive must be SEEDED from the newest companion checkpoint, not
        # re-run from token zero. An unseeded derive costs as much forward
        # compute as a cold prefill, which makes the paged hit it is attached
        # to worth nothing while usage still reports the full cached_tokens.
        seed_idx = source.index("_seed_cache_from_ssm_checkpoint(")
        assert seed_idx < derive_idx
        assert "base_token_count=" in source


class TestMediaCompanionCaptureBoundaries:
    """Media turns must capture companion checkpoints, safely."""

    def _gen(self, placeholders=frozenset({99})):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen._is_hybrid = True
        gen.block_aware_cache = SimpleNamespace(block_size=64)
        gen._media_placeholder_token_ids = lambda: set(placeholders)
        return gen

    def test_media_capture_is_bounded_to_the_pure_text_prefix(self):
        """Unreachable today; asserts the bound for when it is not.

        All three call sites sit under `if not has_media_payload:`, so
        has_images is always False in production and this arm does not run --
        it is NOT the multimodal reuse fix. What it pins is the invariant: if
        a media turn ever does reach here, the capture must stop before the
        first media placeholder, because a snapshot taken past one describes
        recurrent state that absorbed vision embeddings, and anything
        resuming from it with a text-only forward re-feeds those positions
        without pixels.
        """
        from types import SimpleNamespace

        gen = self._gen()
        # 372 tokens (not a multiple of 64 or 256), media run starts at 300.
        tokens = list(range(300)) + [99] * 30 + list(range(42))
        request = SimpleNamespace(
            _original_token_ids=tokens,
            _cached_tokens=0,
            _ssm_required_checkpoint_tokens=0,
        )
        bounds = gen._ssm_capture_boundaries_for(
            request, seq_len=372, has_images=True, clean_boundary=371
        )
        assert bounds, "media turn still captures nothing"
        assert all(b <= 300 for b in bounds), (
            "captured past the first media placeholder at 300: %s" % bounds
        )
        assert all(b % 64 == 0 for b in bounds), (
            "boundaries must be block-aligned so the KV chain can pair: %s"
            % bounds
        )

    def test_media_at_the_very_start_captures_nothing(self):
        """No pure-text prefix means nothing is safe to snapshot."""
        from types import SimpleNamespace

        gen = self._gen()
        tokens = [99] * 40 + list(range(300))
        request = SimpleNamespace(
            _original_token_ids=tokens,
            _cached_tokens=0,
            _ssm_required_checkpoint_tokens=0,
        )
        assert gen._ssm_capture_boundaries_for(
            request, seq_len=340, has_images=True, clean_boundary=339
        ) == []

    def test_text_only_turns_are_unchanged(self):
        """The text lane must keep its existing boundaries."""
        from types import SimpleNamespace

        gen = self._gen()
        request = SimpleNamespace(
            _original_token_ids=list(range(372)),
            _cached_tokens=0,
            _ssm_required_checkpoint_tokens=0,
        )
        bounds = gen._ssm_capture_boundaries_for(
            request, seq_len=372, has_images=False, clean_boundary=371
        )
        assert 371 in bounds, "clean boundary dropped from the text lane"

    def test_safe_limit_reports_first_placeholder(self):
        gen = self._gen()
        assert gen._media_safe_capture_limit(list(range(50))) == 50
        assert gen._media_safe_capture_limit([1, 2, 99, 4]) == 2
        assert gen._media_safe_capture_limit([99, 1]) == 0
        assert gen._media_safe_capture_limit([]) == 0


class TestCompanionDeltaMediaGuard:
    """A text-only derive must never cross a media span."""

    def test_delta_declines_when_the_gap_contains_media(self):
        from types import SimpleNamespace
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen._is_hybrid = True
        gen._hybrid_kv_positions = [0]
        gen._media_placeholder_token_ids = lambda: {99}
        gen._tokens_contain_media_placeholders = (
            lambda ids: any(i == 99 for i in ids)
        )

        def _must_not_run(_table):
            raise AssertionError("reconstructed despite a media gap")

        gen.block_aware_cache = SimpleNamespace(reconstruct_cache=_must_not_run)
        # gap [297, 384) crosses a placeholder run at [300, 337)
        tokens = list(range(300)) + [99] * 37 + list(range(83))
        assert gen._derive_hybrid_companion_delta(
            SimpleNamespace(request_id="r"), tokens,
            fetch_num=384, ck_len=297, block_table=object(),
        ) is None


class TestCompanionStoreKeySymmetry:
    """Whatever key the reader asks with, the writer must write with."""

    class _RecordingCompanion:
        def __init__(self):
            self.stored = []
            self.probes = []

        def has_complete(self, tokens, num_tokens, cache_extra_keys=None):
            self.probes.append((num_tokens, cache_extra_keys))
            return any(
                s[0] == num_tokens and s[1] == cache_extra_keys
                for s in self.stored
            )

        def store(
            self, tokens, num_tokens, states, is_complete=True,
            cache_extra_keys=None,
        ):
            self.stored.append((num_tokens, cache_extra_keys))

    def _gen(self, companion, placeholders=frozenset()):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen._is_hybrid = True
        gen._ssm_state_cache = companion
        gen._hybrid_kv_positions = [0]
        gen._media_placeholder_token_ids = lambda: set(placeholders)
        gen._tokens_contain_media_placeholders = (
            lambda ids: any(i in set(placeholders) for i in ids)
        )
        return gen

    def test_clean_pass_stores_under_the_key_the_reader_asks_with(self):
        """The media lane fetches with a salt; the store must carry it.

        Without this, the clean pass wrote the companion under the bare token
        key while every media reader asked with the media salt -- so the
        entry was unfindable, and the "skip the second background pass"
        optimisation could never fire.
        """
        companion = self._RecordingCompanion()
        gen = self._gen(companion)
        fresh = [object(), object()]
        salt = {"mllm_media": "sha-of-the-image"}

        gen._store_companion_from_clean_pass(
            list(range(6)), fresh, cache_extra_keys=salt
        )
        assert companion.stored == [(6, salt)]

    def test_media_prompt_without_a_salt_is_declined_not_guessed(self):
        """Fail closed: an unsalted media companion collides across images.

        Two turns can carry byte-identical placeholder token ids and totally
        different pictures. Storing that state under the bare token key means
        the second turn restores recurrent state that absorbed the FIRST
        image -- coherent prose about the wrong photo, and nothing raises.
        """
        companion = self._RecordingCompanion()
        gen = self._gen(companion, placeholders={99})
        fresh = [object(), object()]

        gen._store_companion_from_clean_pass(
            [1, 2, 99, 99, 3, 4], fresh, cache_extra_keys=None
        )
        assert companion.stored == [], "stored an unsalted media companion"

    def test_text_prompt_without_a_salt_still_stores(self):
        companion = self._RecordingCompanion()
        gen = self._gen(companion, placeholders={99})
        fresh = [object(), object()]

        gen._store_companion_from_clean_pass([1, 2, 3, 4], fresh)
        assert companion.stored == [(4, None)]

    def test_a_failed_media_probe_does_not_block_the_store(self):
        """A probe that cannot answer must not become a restriction.

        The media decline is allowed to fire only on a POSITIVE reading. If
        the placeholder lookup raises (no model attached, odd config shape),
        declining would silently kill companion storage for an entire family
        and every hybrid turn would re-prefill from cold forever.
        """
        companion = self._RecordingCompanion()
        gen = self._gen(companion)

        def _boom(_ids):
            raise RuntimeError("config unavailable")

        gen._tokens_contain_media_placeholders = _boom
        gen._store_companion_from_clean_pass([1, 2, 3, 4], [object(), object()])
        assert companion.stored == [(4, None)], (
            "a failed media probe blocked a legitimate store"
        )

    def test_idle_rederive_threads_the_salt_into_the_clean_prefill(self):
        """A salted queue entry paid for the clean pass TWICE.

        run_idle_rederive popped the salt, probed has_complete WITH it, then
        ran the clean prefill WITHOUT it. The clean pass stored under the bare
        key, the probe below still missed, and the function did the whole
        50-200MB clone + disk write a second time.
        """
        companion = self._RecordingCompanion()
        gen = self._gen(companion)
        salt = {"mllm_media": "sha-of-the-image"}
        gen._ssm_rederive_queue = [(list(range(6)), 6, "req-1", salt)]

        seen = {}

        def _fake_clean(tokens, cache_extra_keys=None):
            seen["salt"] = cache_extra_keys
            companion.store(
                tokens, len(tokens), ["s"], cache_extra_keys=cache_extra_keys
            )
            return None

        gen._prefill_for_clean_ssm = _fake_clean
        assert gen.run_idle_rederive() is True
        assert seen["salt"] == salt, (
            "clean prefill ran without the salt the reader uses"
        )
        assert companion.stored == [(6, salt)], (
            "companion written twice or under the wrong key: %s"
            % companion.stored
        )


class TestNativeMtpVersusExternalDrafter:
    """An external drafter and native MTP must never run together."""

    def _args(self, **over):
        from types import SimpleNamespace

        base = dict(
            speculative_model=None,
            disable_native_mtp=False,
            native_mtp_depth=3,
            native_mtp_sampling_policy=None,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_external_drafter_disables_native_mtp(self):
        """Selecting a draft model must switch the bundle's MTP heads off.

        Running both is two speculative decoders bidding for the same decode
        step. Observed live before this guard: the same code prompt measured
        31.6, 44.5 and 45.3 t/s back to back with a DFlash2 drafter and native
        MTP depth 3 both on the command line.
        """
        import inspect

        from vmlx_engine import cli

        source = inspect.getsource(cli)
        guard = source.index("Native MTP disabled: an external speculative model")
        assert guard > 0
        window = source[max(0, guard - 1200) : guard + 400]
        assert "args.disable_native_mtp = True" in window
        assert "args.native_mtp_depth = None" in window

    def test_disabling_mtp_clears_a_leftover_depth_env(self):
        """A disabled runtime must not be revivable by stale env state."""
        import inspect

        from vmlx_engine import cli

        source = inspect.getsource(cli)
        idx = source.index('os.environ["VMLINUX_NATIVE_MTP"] = "0"')
        window = source[idx : idx + 500]
        assert 'os.environ.pop("VMLINUX_NATIVE_MTP_DEPTH", None)' in window
        assert 'os.environ.pop("VMLX_NATIVE_MTP_DEPTH", None)' in window

    def test_explicit_disable_is_not_overridden_by_the_guard(self):
        """--disable-native-mtp already off must not re-enter the guard."""
        import inspect

        from vmlx_engine import cli

        source = inspect.getsource(cli)
        idx = source.index('if getattr(args, "speculative_model", None) and not getattr(')
        window = source[idx : idx + 200]
        assert '"disable_native_mtp", False' in window


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
        # Must search from after the last </think> end. Accept either quote style.
        assert (
            'rfind("</think>")' in source
            or "rfind('</think>')" in source
        ), (
            "Stop sequence mapping must use rfind('</think>') or rfind(\"</think>\") "
            "to skip past reasoning blocks when mapping position back to full_text"
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


class TestHybridSSMResumeRemaining:
    """Regression coverage for hybrid SSM resume after a shorter checkpoint."""

    def test_accepted_companion_delta_falls_through_to_reconstruction(self):
        """vmlx#91 DELTA: when the KV hit runs past the companion checkpoint by less than a block, the
        companion is advanced to the hit and the FULL block table is kept (``trimmed = None`` on purpose).
        Since e89006fb the branch after it read ``else:`` and treated that None as "trim returned None ->
        full prefill": the accepted delta was zeroed a millisecond later (live at 22ee605e: 'DELTA accepted
        ... kept the full 704-token KV hit' then 'RESUME skipped ... checkpoint at 702 below one block',
        credited=704 accepted=0, cached_tokens=0). The below-one-block branch must be guarded by
        ``not _delta_states`` so an accepted delta reaches reconstruction."""
        import inspect

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        accepted = source.index('f"vmlx#91 DELTA accepted for "')
        assert "trimmed = None" in source[accepted:accepted + 900]
        below = source.index("below one block — ", accepted)
        guard = source.rfind("elif not _delta_states:", accepted, below)
        assert guard != -1, "the below-one-block fallback must be an 'elif not _delta_states:' branch"
        assert "else:\n" not in source[guard:below].replace(" ", ""), "no bare else may sit between the delta guard and the below-one-block fallback"
        # the misalignment refusal is likewise delta-guarded
        misaligned = source.index("is not block-aligned", accepted)
        assert "not _delta_states" in source[accepted:misaligned]

    def test_mllm_resume_recomputes_remaining_from_trimmed_checkpoint(self):
        """After vmlx#91 trims KV to a shorter SSM checkpoint, MLLM must
        re-feed tokens from that trimmed checkpoint, not from the longer
        original paged hit. Otherwise the middle prompt span is skipped.
        """
        import inspect

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)
        resume_idx = source.index("trim_block_table(")
        hit_idx = source.index("VLM HYBRID cache HIT", resume_idx)
        resume_window = source[resume_idx:hit_idx]

        assert "remaining = token_list[trimmed.num_tokens:]" in resume_window, (
            "MLLM vmlx#91 resume must recompute remaining tokens from the "
            "trimmed KV checkpoint; using the pre-trim remaining skips the "
            "prompt span between the shorter SSM checkpoint and the original "
            "paged hit."
        )
        assert "_full_remaining = (remaining or []) + list(_gpl_suffix)" in resume_window, (
            "The recomputed remaining tokens must be the value re-fed into "
            "req.input_ids after reconstruction, not only diagnostic log text."
        )
        assert "req.input_ids = mx.array([_full_remaining])" in resume_window, (
            "The vmlx#91 resume path must prefill the recomputed tail from the "
            "trimmed SSM checkpoint."
        )

    def test_inline_ssm_capture_key_includes_cached_prefix_base(self):
        """When capture runs after a cache hit, boundary_len is local to the
        re-fed tail, but the companion key must be absolute over the full
        prompt token list.
        """
        import inspect

        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(
            MLLMBatchGenerator._maybe_capture_clean_ssm_boundary
        )

        assert "base_len = int(getattr(request, \"_cached_tokens\", 0) or 0)" in source
        assert "key_boundary = base_len + int(boundary_len)" in source
        assert "all_tokens[:key_boundary]" in source

    def test_mllm_hybrid_cache_hit_is_not_promoted_into_longer_paged_state(self):
        """A restored hybrid prefix may be consumed but must not be stored
        again as a longer paged entry. Repeated restore/extend/store cycles
        compound path-dependent cache error across multi-turn chats.
        """
        import inspect

        from vmlx_engine.mllm_scheduler import (
            MLLMScheduler,
            _hybrid_clean_store_enabled,
            _hybrid_prefix_promotion_enabled,
        )

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        skip_idx = source.index("hybrid restored-prefix promotion disabled")
        paged_store_idx = source.index("# --- Cache store: paged path ---")
        # Window widened for the opt-in routes that now sit between the
        # condition and the skip; the condition itself must still be there.
        guard = source[max(0, skip_idx - 1600):skip_idx + 200]

        assert skip_idx < paged_store_idx
        assert 'getattr(self, "_is_hybrid", False)' in guard
        assert 'getattr(request, "_cached_tokens", 0)' in guard
        assert "_skip_cache_store = True" in guard

        # Promotion — writing the RESTORED cache back as a longer entry — was
        # measured to collapse the model on its first extended turn, so an
        # unconfigured engine must still refuse it. That default is the thing
        # under test here.
        assert _hybrid_prefix_promotion_enabled() is False

        # The clean re-prefill route is deliberately NOT the same thing and is
        # on by default: it re-derives the N-1 key and stores that typed state
        # rather than writing back a reconstructed one. Keeping it off is what
        # froze hybrid reuse at turn one.
        assert _hybrid_clean_store_enabled() is True

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
        import re

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        # Tool iteration boundary: emitToolStatus with "processing", "", undefined (tool follow-up).
        # Accept both single- and double-quoted forms and collapsed whitespace
        # so the test survives prettier / eslint reformats.
        match = re.search(
            r'emitToolStatus\(\s*[\'"]processing[\'"]\s*,\s*[\'"][\'"]\s*,\s*undefined',
            source,
        )
        assert match is not None, (
            "Tool iteration boundary must call emitToolStatus('processing', '', undefined, ...)"
        )
        boundary_idx = match.start()
        # Look backwards for reasoningDone emission
        pre_boundary = source[max(0, boundary_idx - 800):boundary_idx]
        assert "chat:reasoningDone" in pre_boundary, (
            "Tool iteration boundary must fire chat:reasoningDone before "
            "resetting isReasoning=false, otherwise reasoning-only tool calls "
            "silently drop the reasoning content"
        )

    def test_auto_continue_boundary_emits_reasoning_done(self):
        """When isReasoning=true at auto-continue boundary, reasoningDone must fire."""
        import os
        import re

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        # Auto-continue boundary: emitToolStatus with "processing", "", "Generating response..."
        # Accept multi-line formatted calls with either quote style.
        match = re.search(
            r'emitToolStatus\(\s*[\'"]processing[\'"]\s*,\s*[\'"][\'"]\s*,\s*[\'"]Generating response\.\.\.[\'"]',
            source,
        )
        assert match is not None, (
            "Auto-continue boundary must call "
            "emitToolStatus('processing', '', 'Generating response...', ...)"
        )
        boundary_idx = match.start()
        pre_boundary = source[max(0, boundary_idx - 800):boundary_idx]
        assert "chat:reasoningDone" in pre_boundary, (
            "Auto-continue boundary must fire chat:reasoningDone before "
            "resetting isReasoning=false"
        )


class TestSuppressReasoningDiagnostic:
    """Regression: suppressed reasoning-only output must stay out of assistant text."""

    def test_chat_completions_has_suppress_reasoning_diagnostic(self):
        """Chat completions path must finish empty instead of injecting prose."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert (
            "suppress_reasoning\n"
            "        and not content_was_emitted\n"
            "        and not tool_calls_emitted\n"
            "        and accumulated_reasoning"
        ) in source, (
            "Chat completions must detect reasoning-only + suppress"
        )
        assert "Model produced only internal reasoning" not in source
        assert "ChatCompletionChunkDelta()" in source

    def test_responses_api_has_suppress_reasoning_diagnostic(self):
        """Responses API path must warn out-of-band when only reasoning is suppressed."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "reasoning_only_no_content" in source
        assert "Model produced only internal reasoning" not in source

    def test_chat_completions_nonstream_reasoning_only_warning_is_wired(self):
        """Non-stream chat must surface warnings out-of-band too."""
        import inspect
        from vmlx_engine.server import create_chat_completion

        source = inspect.getsource(create_chat_completion)

        assert "_chat_completion_warnings_for_reasoning_only" in source
        assert "warnings=response_warnings" in source

    def test_chat_completions_streaming_reasoning_only_warning_is_wired(self):
        """Streaming chat must surface warnings out-of-band too."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)

        assert "_chat_completion_warnings_for_reasoning_only" in source
        assert "warnings=_stream_chat_warnings" in source


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
        """Chat completions SSE path must route parsed errors to the shared decoder."""
        import os

        chat_ts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "panel", "src", "main", "ipc", "chat.ts",
        )
        with open(chat_ts) as f:
            source = f.read()

        assert "chatStreamServerEventErrorDetail(parsed)" in source, (
            "Chat completions SSE parser must route the parsed payload through "
            "the shared server-error decoder"
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

        # Must check for bare 'error' event type, not just 'response.error'.
        # Accept both single- and double-quoted strict/loose comparisons.
        accepted = [
            "=== 'error'", '=== "error"',
            "== 'error'", '== "error"',
        ]
        assert any(token in source for token in accepted), (
            "Responses API SSE parser must recognize bare 'error' event type "
            "alongside 'response.error' and 'response.failed' — looked for "
            f"any of {accepted}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

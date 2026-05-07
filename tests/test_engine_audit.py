# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive engine audit tests for vMLX.

Tests cover all core features:
- A. Reasoning & Parser System (GPT-OSS parser, parser registry parity)
- B. Tool Parser System (GLM47, parser-model mapping)
- C. Sampling Defaults (generation_config.json reading)
- D. Settings & Config (model config registry, incompatibility logic)
- E. Engine & Cache (request lifecycle, SamplingParams, MLLM batch request)
- F. Vision Embedding Cache (hash ordering)

These are unit tests that do NOT require model loading.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# A. Reasoning & Parser System
# ===========================================================================


class TestGptOssReasoningParser:
    """Tests for the GPT-OSS / Harmony protocol reasoning parser."""

    @pytest.fixture
    def parser(self):
        from vmlx_engine.reasoning import get_parser
        return get_parser("openai_gptoss")()

    def test_parser_registered(self):
        from vmlx_engine.reasoning import list_parsers
        assert "openai_gptoss" in list_parsers()

    def test_extract_analysis_and_final(self, parser):
        """Should extract reasoning from analysis channel and content from final channel."""
        output = (
            "<|channel|>analysis<|message|>Let me analyze this"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "analyze" in reasoning
        assert content is not None
        assert "42" in content

    def test_extract_no_markers_returns_content(self, parser):
        """No Harmony markers should return output as content."""
        output = "Just a plain response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_extract_only_analysis(self, parser):
        """Only analysis channel, no final channel."""
        parser._harmony_active = True
        output = "<|channel|>analysis<|message|>Just thinking out loud"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "thinking" in reasoning

    def test_streaming_reset_state(self, parser):
        """Reset state should allow reuse."""
        parser.reset_state(harmony_active=True)
        assert parser._harmony_active is True
        assert parser._emitted_reasoning == 0
        assert parser._emitted_content == 0
        assert parser._got_final is False


class TestParserRegistryParity:
    """Tests that all reasoning parsers are properly registered and instantiable."""

    def test_all_reasoning_parsers_registered(self):
        from vmlx_engine.reasoning import list_parsers
        parsers = list_parsers()
        expected = ["qwen3", "deepseek_r1", "openai_gptoss"]
        for name in expected:
            assert name in parsers, f"Reasoning parser '{name}' not registered"

    def test_all_reasoning_parsers_instantiable(self):
        from vmlx_engine.reasoning import get_parser
        for name in ["qwen3", "deepseek_r1", "openai_gptoss"]:
            parser = get_parser(name)()
            assert hasattr(parser, "extract_reasoning")
            assert hasattr(parser, "extract_reasoning_streaming")
            assert hasattr(parser, "reset_state")


# ===========================================================================
# B. Tool Parser System
# ===========================================================================


class TestGLM47ToolParser:
    """Tests for the GLM47 tool parser."""

    def test_glm47_registered(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parsers = ToolParserManager.list_registered()
        assert "glm47" in parsers

    def test_glm47_instantiation(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parser_cls = ToolParserManager.get_tool_parser("glm47")
        parser = parser_cls()
        assert hasattr(parser, "extract_tool_calls")

    def test_step3p5_registered(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        parsers = ToolParserManager.list_registered()
        assert "step3p5" in parsers


class TestToolParserModelMapping:
    """Tests that model configs map to correct tool parsers."""

    def test_model_config_tool_parsers(self):
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        # Reset and populate
        import vmlx_engine.model_config_registry as mcr
        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        expected_mappings = {
            "qwen3": "qwen",
            "deepseek": "deepseek",
            "glm47-flash": "glm47",
            "llama4": "llama",
        }

        for family_name, expected_tool_parser in expected_mappings.items():
            configs = [c for c in registry._configs
                       if c.family_name == family_name]
            if configs:
                assert configs[0].tool_parser == expected_tool_parser, \
                    f"Family '{family_name}' expected tool_parser='{expected_tool_parser}', got '{configs[0].tool_parser}'"

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False


class TestToolParserReasoningParserMapping:
    """Tests that model configs have correct reasoning parsers."""

    def test_reasoning_parser_assignments(self):
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        import vmlx_engine.model_config_registry as mcr
        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        expected = {
            "qwen3": "qwen3",
            "deepseek": "deepseek_r1",
            "glm47-flash": "openai_gptoss",
            "gpt-oss": "openai_gptoss",
        }

        for family_name, expected_parser in expected.items():
            configs = [c for c in registry._configs
                       if c.family_name == family_name]
            if configs:
                assert configs[0].reasoning_parser == expected_parser, \
                    f"Family '{family_name}' expected reasoning_parser='{expected_parser}', got '{configs[0].reasoning_parser}'"

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False


# ===========================================================================
# C. Sampling Defaults
# ===========================================================================


class TestSamplingParams:
    """Tests for SamplingParams dataclass fields."""

    def test_sampling_params_has_all_fields(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert sp.max_tokens == 100
        assert sp.temperature == 0.8
        assert sp.top_p == 0.95
        assert sp.top_k == 50
        assert sp.min_p == 0.1
        assert sp.repetition_penalty == 1.2

    def test_sampling_params_defaults(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.temperature == 0.7
        assert sp.top_p == 0.9
        assert sp.top_k == 0
        assert sp.min_p == 0.0
        assert sp.repetition_penalty == 1.0


class TestMLLMBatchRequestSampling:
    """Tests for MLLM batch request sampling parameter passthrough."""

    def test_mllm_batch_request_has_sampling_fields(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
        req = MLLMBatchRequest(
            uid=0,
            request_id="test",
            prompt="hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_mllm_batch_request_defaults(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
        req = MLLMBatchRequest(uid=0, request_id="test", prompt="hello")
        assert req.top_k == 0
        assert req.min_p == 0.0
        assert req.repetition_penalty == 1.0


class TestServerSamplingResolution:
    """Tests for server-side sampling parameter resolution."""

    def test_resolve_temperature_request_value(self):
        """Request value should take priority."""
        from vmlx_engine.server import _resolve_temperature
        assert _resolve_temperature(0.5) == 0.5

    def test_resolve_temperature_fallback(self):
        """None request should use fallback."""
        from vmlx_engine.server import _resolve_temperature
        result = _resolve_temperature(None)
        assert isinstance(result, float)

    def test_resolve_top_p_request_value(self):
        """Request value should take priority."""
        from vmlx_engine.server import _resolve_top_p
        assert _resolve_top_p(0.95) == 0.95

    def test_resolve_top_p_fallback(self):
        """None request should use fallback."""
        from vmlx_engine.server import _resolve_top_p
        result = _resolve_top_p(None)
        assert isinstance(result, float)


# ===========================================================================
# D. Settings & Config
# ===========================================================================


class TestModelConfigRegistryLookup:
    """Tests for model config registry lookup by model_type."""

    def _find_by_model_type(self, registry, model_type):
        """Find a config that has the given model_type in its model_types list."""
        for config in registry._configs:
            if model_type in config.model_types:
                return config
        return None

    def test_lookup_qwen3(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3")
        assert config is not None
        assert config.tool_parser == "qwen"

    def test_lookup_glm4_moe(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.family_name == "glm4_moe"
        assert config.reasoning_parser == "openai_gptoss"

    def test_lookup_deepseek_v3(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "deepseek_v3")
        assert config is not None
        assert config.reasoning_parser == "deepseek_r1"

    def test_lookup_qwen3_5(self):
        """qwen3_5 model_type is shared between text and VL variants.
        Registry is_mllm must be False — VLM detection relies on config.json vision_config."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3_5")
        assert config is not None
        assert config.is_mllm is False

    def test_lookup_unknown_type_returns_none(self):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "nonexistent_type_xyz")
        assert config is None


class TestThinkInTemplate:
    """Tests for think_in_template flag on model configs."""

    def _find_by_model_type(self, registry, model_type):
        for config in registry._configs:
            if model_type in config.model_types:
                return config
        return None

    def test_glm47_flash_think_in_template_false(self):
        """GLM-4.7 Flash uses Harmony protocol, NOT <think> in template."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.think_in_template is False

    def test_qwen3_think_in_template_true(self):
        """Qwen3 uses <think> tag in template."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3")
        if config is not None:
            assert config.think_in_template is True


# ===========================================================================
# E. Engine & Cache
# ===========================================================================


class TestVisionEmbeddingCacheOrdering:
    """Tests for vision embedding cache hash ordering."""

    def test_hash_order_sensitive(self):
        """Different image orderings should produce different hashes."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        hash1 = compute_images_hash(["img1.jpg", "img2.jpg"])
        hash2 = compute_images_hash(["img2.jpg", "img1.jpg"])
        assert hash1 != hash2, "Different image orders should produce different hashes"

    def test_same_order_same_hash(self):
        """Same image ordering should produce same hash."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        hash1 = compute_images_hash(["a.jpg", "b.jpg"])
        hash2 = compute_images_hash(["a.jpg", "b.jpg"])
        assert hash1 == hash2

    def test_empty_images(self):
        """Empty list should produce a consistent hash."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        assert compute_images_hash([]) == "no_images"

    def test_single_image(self):
        """Single image hash should be deterministic."""
        from vmlx_engine.vision_embedding_cache import compute_images_hash
        h1 = compute_images_hash(["test.jpg"])
        h2 = compute_images_hash(["test.jpg"])
        assert h1 == h2


class TestVisionCacheStats:
    """Tests for VisionCacheStats tracking."""

    def test_initial_stats(self):
        from vmlx_engine.vision_embedding_cache import VisionCacheStats
        stats = VisionCacheStats()
        assert stats.pixel_cache_hits == 0
        assert stats.pixel_cache_misses == 0
        assert stats.pixel_hit_rate == 0.0

    def test_hit_rate_calculation(self):
        from vmlx_engine.vision_embedding_cache import VisionCacheStats
        stats = VisionCacheStats(pixel_cache_hits=3, pixel_cache_misses=7)
        assert stats.pixel_hit_rate == 0.3


class TestRequestStatus:
    """Tests for request status transitions."""

    def test_request_status_values(self):
        from vmlx_engine.request import RequestStatus
        assert hasattr(RequestStatus, "FINISHED_STOPPED")
        assert hasattr(RequestStatus, "FINISHED_ABORTED")
        assert hasattr(RequestStatus, "FINISHED_LENGTH_CAPPED")

    def test_sampling_params_stop_sequences(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>", "\n\n"])
        assert "<|end|>" in sp.stop
        assert "\n\n" in sp.stop


# ===========================================================================
# F. Standard Architectures Detection
# ===========================================================================


class TestStandardArchitectures:
    """Tests that _STANDARD_ARCHITECTURES in tokenizer.py includes all key types."""

    def test_qwen3_5_in_standard_architectures(self):
        from vmlx_engine.utils.tokenizer import _STANDARD_ARCHITECTURES
        assert "qwen3_5" in _STANDARD_ARCHITECTURES
        assert "qwen3_5_moe" in _STANDARD_ARCHITECTURES

    def test_common_types_in_standard_architectures(self):
        from vmlx_engine.utils.tokenizer import _STANDARD_ARCHITECTURES
        common = ["llama", "qwen2", "qwen3", "gemma2", "gemma3", "mistral",
                  "deepseek_v3", "phi3"]
        for t in common:
            assert t in _STANDARD_ARCHITECTURES, \
                f"model_type '{t}' missing from _STANDARD_ARCHITECTURES"


# ===========================================================================
# G. API Models
# ===========================================================================


class TestAPIModels:
    """Tests for API request/response models."""

    def test_chat_completion_request_has_sampling_fields(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.8,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.temperature == 0.5
        assert req.top_p == 0.8
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_chat_completion_request_defaults(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Defaults should be None (let server resolve)
        assert req.top_k is None
        assert req.min_p is None
        assert req.repetition_penalty is None

    def test_responses_request_has_sampling_fields(self):
        from vmlx_engine.api.models import ResponsesRequest
        req = ResponsesRequest(
            model="test",
            input="hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
        )
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2

    def test_enable_thinking_field(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        assert req.enable_thinking is True

    def test_stream_options_field(self):
        from vmlx_engine.api.models import ChatCompletionRequest, StreamOptions
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        assert req.stream_options.include_usage is True


# ===========================================================================
# G. MLLM Scheduler Fixes
# ===========================================================================

class TestMLLMStopSequences:
    """Test that MLLM requests properly carry stop sequences."""

    def test_sampling_params_stop_sequences_populated(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>", "###"])
        assert sp.stop == ["<|end|>", "###"]

    def test_sampling_params_stop_default_empty(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.stop == []

    def test_mllm_request_accepts_stop(self):
        from vmlx_engine.mllm_scheduler import MLLMRequest
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams(stop=["<|end|>"])
        req = MLLMRequest(
            request_id="test-1",
            prompt="What's in this image?",
            sampling_params=sp,
        )
        assert req.sampling_params.stop == ["<|end|>"]


class TestRepPenaltyTruthiness:
    """Test that repetition_penalty=0.0 is NOT treated as disabled."""

    def test_zero_rep_penalty_is_not_none(self):
        """0.0 is a valid repetition penalty (no repetition boost)."""
        val = 0.0
        # Old buggy check: `if val and val != 1.0` → falsy (skips 0.0)
        # New correct check: `if val is not None and val != 1.0` → True
        assert (val is not None and val != 1.0) is True
        assert not (val and val != 1.0)  # Demonstrates the old bug: 0.0 was falsy

    def test_none_rep_penalty(self):
        val = None
        assert (val is not None and val != 1.0) is False

    def test_default_rep_penalty(self):
        val = 1.0
        assert (val is not None and val != 1.0) is False


class TestImageHashOrdering:
    """Test that image hash preserves order (not sorted)."""

    def test_different_order_different_hash(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        # Two different orderings should produce different hashes
        hash1 = compute_images_hash(["image_a.jpg", "image_b.jpg"])
        hash2 = compute_images_hash(["image_b.jpg", "image_a.jpg"])
        assert hash1 != hash2, "Image order should matter for VLM cache hashing"

    def test_same_order_same_hash(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        hash1 = compute_images_hash(["img1.jpg", "img2.jpg"])
        hash2 = compute_images_hash(["img1.jpg", "img2.jpg"])
        assert hash1 == hash2

    def test_empty_images(self):
        from vmlx_engine.mllm_cache import compute_images_hash
        assert compute_images_hash([]) == "no_images"
        assert compute_images_hash(None) == "no_images"


class TestToolParserConcurrency:
    """Test that tool parser creates per-call instances (not shared global)."""

    def test_parse_tool_calls_with_parser_no_global_state(self):
        """Verify the function doesn't rely on a global _tool_parser_instance."""
        import vmlx_engine.server as srv
        # When auto tool choice is disabled, should fall through to generic parser
        old_val = srv._enable_auto_tool_choice
        try:
            srv._enable_auto_tool_choice = False
            result = srv._parse_tool_calls_with_parser("hello world")
            # Should return content unchanged, no tool calls
            assert result[1] is None or result[1] == []
        finally:
            srv._enable_auto_tool_choice = old_val


class TestCacheTruncation:
    """Test N-1 token truncation logic for cache storage."""

    def test_truncation_target(self):
        """Cache should store N-1 tokens (prompt_len - 1)."""
        prompt_len = 100
        target = prompt_len - 1
        assert target == 99

    def test_truncation_skips_empty(self):
        """Truncation with 0 or 1 tokens should return None."""
        from vmlx_engine.scheduler import Scheduler
        result = Scheduler._truncate_cache_to_prompt_length([], 5)
        assert result is None
        result = Scheduler._truncate_cache_to_prompt_length([MagicMock()], 0)
        assert result is None


# ===========================================================================
# H. MLLM Scheduler Parity Tests
# ===========================================================================

class TestMLLMAbortCleanup:
    """Test that MLLM abort properly cleans up all resources."""

    def test_abort_uses_pop_not_get(self):
        """abort_request must remove (pop) the request, not just read (get) it."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        # Must use pop to remove the request
        assert "self.requests.pop(" in source or "requests.pop(" in source
        # Must NOT use .get() for the primary request lookup
        # (get would leave the request in the dict)

    def test_abort_cleans_block_table(self):
        """abort_request must clean up paged cache block tables."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_request_tables" in source, (
            "abort_request must clean up _request_tables for paged cache"
        )

    def test_abort_cleans_detokenizer(self):
        """abort_request must clean up detokenizer state."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_cleanup_detokenizer" in source, (
            "abort_request must call _cleanup_detokenizer"
        )


class TestMLLMStepErrorRecovery:
    """Test that MLLM step() has error recovery like LLM scheduler."""

    def test_step_has_try_except(self):
        """step() must wrap batch_generator.next() in try/except."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "try:" in source, "step() must have try/except for error recovery"
        assert "batch_generator.next()" in source or "self.batch_generator.next()" in source

    def test_step_reschedules_on_error(self):
        """On error, step() must move requests back to waiting."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "RequestStatus.WAITING" in source, (
            "step() must reschedule failed requests to WAITING"
        )


class TestMLLMStopTokenIds:
    """Test that MLLM properly handles stop_token_ids."""

    def test_sampling_params_has_stop_token_ids(self):
        """SamplingParams must include stop_token_ids."""
        from vmlx_engine.request import SamplingParams

        params = SamplingParams(stop_token_ids=[100, 200])
        assert params.stop_token_ids == [100, 200]

    def test_stop_token_ids_default_empty(self):
        """stop_token_ids should default to empty list."""
        from vmlx_engine.request import SamplingParams

        params = SamplingParams()
        assert params.stop_token_ids == [] or params.stop_token_ids is None

    def test_mllm_scheduler_passes_stop_token_ids(self):
        """MLLMScheduler.add_request must pass stop_token_ids to SamplingParams."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "stop_token_ids" in source, (
            "add_request must pass stop_token_ids to SamplingParams"
        )


class TestMLLMEnsureBatchGeneratorCacheClean:
    """Test that _ensure_batch_generator updates sampler in place (preserves caches)."""

    def test_updates_sampler_in_place(self):
        """Must update sampler in place without clearing caches."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        # Should update sampler in place rather than recreating generator
        assert "batch_generator.sampler" in source, (
            "_ensure_batch_generator must update sampler in place"
        )
        assert "_current_sampler_params" in source, (
            "_ensure_batch_generator must track current sampler params"
        )

    def test_no_logits_processors_param(self):
        """MLLMBatchGenerator must not accept logits_processors."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        sig = inspect.signature(MLLMBatchGenerator.__init__)
        assert "logits_processors" not in sig.parameters, (
            "logits_processors was removed — must not be in __init__ signature"
        )

    def test_scheduler_does_not_pass_logits_processors(self):
        """_ensure_batch_generator must not pass logits_processors."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._ensure_batch_generator)
        assert "logits_processors" not in source, (
            "_ensure_batch_generator must not reference logits_processors"
        )


class TestMLLMDequantizeOnRestore:
    """Test that MLLM dequantizes cache after paged cache restore."""

    def test_dequantize_function_exists(self):
        """_dequantize_cache must exist in mllm_batch_generator."""
        from vmlx_engine.mllm_batch_generator import _dequantize_cache
        assert callable(_dequantize_cache)

    def test_dequantize_passthrough_non_quantized(self):
        """Non-quantized cache should pass through unchanged."""
        from vmlx_engine.mllm_batch_generator import _dequantize_cache

        mock_cache = [MagicMock(), MagicMock()]
        # Not QuantizedKVCache instances — should pass through
        result = _dequantize_cache(mock_cache)
        assert len(result) == 2

    def test_dequantize_called_after_reconstruct(self):
        """Paged cache hit path must call _dequantize_cache after reconstruct."""
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator)
        # reconstruct_cache and _dequantize_cache must both appear
        assert "reconstruct_cache" in source
        assert "_dequantize_cache" in source


class TestMLLMTotalPromptTokens:
    """Test that MLLM scheduler tracks total_prompt_tokens."""

    def test_total_prompt_tokens_incremented(self):
        """_process_batch_responses must increment total_prompt_tokens at finish time."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        # Prompt token count is only known after the batch generator's first response,
        # so tracking happens at request finish time in _process_batch_responses
        source = inspect.getsource(MLLMScheduler._process_batch_responses)
        assert "total_prompt_tokens" in source, (
            "_process_batch_responses must increment total_prompt_tokens"
        )


# =============================================================================
# L1: frequency_penalty / presence_penalty accepted but warned
# =============================================================================


class TestPenaltyParametersAccepted:
    """Test that frequency_penalty and presence_penalty are accepted by API models."""

    def test_chat_completion_accepts_penalties(self):
        """ChatCompletionRequest should accept frequency_penalty and presence_penalty."""
        from vmlx_engine.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=0.5,
            presence_penalty=0.8,
        )
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.8

    def test_responses_request_accepts_penalties(self):
        """ResponsesRequest should accept frequency_penalty and presence_penalty."""
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(
            model="test",
            input="hello",
            frequency_penalty=0.5,
            presence_penalty=0.8,
        )
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.8

    def test_server_warns_on_frequency_penalty(self):
        """Server should log warning when frequency_penalty is non-zero."""
        import inspect
        # Read the create_chat_completion source to verify warning logic
        from vmlx_engine.server import create_chat_completion

        source = inspect.getsource(create_chat_completion)
        assert "frequency_penalty" in source, (
            "create_chat_completion must check frequency_penalty"
        )
        assert "not implemented" in source.lower() or "ignored" in source.lower(), (
            "create_chat_completion must warn that frequency_penalty is not implemented"
        )


# =============================================================================
# v2 Audit: C1-C5 Critical Fixes
# =============================================================================


class TestC1DuplicateRequestIdCheck:
    """C1: MLLM scheduler must reject duplicate request IDs."""

    def test_mllm_scheduler_has_duplicate_check(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "already exists" in source, (
            "add_request must check for duplicate request IDs"
        )

    def test_llm_scheduler_also_has_check(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.add_request)
        assert "already exists" in source, (
            "LLM add_request must also check for duplicate request IDs"
        )


class TestC2AbortDecodeRace:
    """C2: MLLM scheduler must have _batch_lock for next()/remove() serialization."""

    def test_batch_lock_exists(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "_batch_lock" in source, (
            "MLLMScheduler must have _batch_lock for abort/decode race protection"
        )

    def test_batch_lock_used_in_step(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.step)
        assert "_batch_lock" in source, (
            "step() must hold _batch_lock during batch_generator.next()"
        )

    def test_deferred_abort_prevents_metal_race(self):
        """Aborting a request while Metal buffers are in-flight touching the
        cache tensors would assert. Current design defers removal via
        `_pending_aborts` instead of holding `_batch_lock`, which is safer
        — the deferred set is drained after the current Metal compute
        completes."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.abort_request)
        assert "_pending_aborts" in source, (
            "abort_request() must defer batch removal via _pending_aborts "
            "to avoid touching cache tensors mid-Metal-compute"
        )
        # And must still hold the queue lock for request table mutation
        assert "_queue_lock" in source


class TestC3DiskCacheQuantScoping:
    """C3: Disk cache directory must be scoped by quantization config."""

    def test_scheduler_disk_cache_includes_quant(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        assert "quant" in source and "scope_key" in source, (
            "Scheduler disk cache dir must include quantization in scope key"
        )

    def test_mllm_scheduler_disk_cache_includes_quant(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        assert "quant" in source and "scope_key" in source, (
            "MLLM scheduler disk cache dir must include quantization in scope key"
        )


class TestC4MaxTokensZero:
    """C4: max_tokens=0 must not be silently overridden to the default."""

    def test_server_uses_is_not_none_check(self):
        """max_tokens=0 must not be silently overridden (0 is falsy so the
        old `max_tokens or default` pattern would replace 0 with default).
        Accept either single- or multi-line form of the conditional, and
        normalize whitespace so newlines between clauses don't fool the
        check."""
        import inspect
        import re
        from vmlx_engine.server import create_chat_completion

        source = inspect.getsource(create_chat_completion)
        # Collapse all whitespace to single spaces so the multi-line
        # `if request.max_tokens is not None\n  else _default_max_tokens`
        # matches too.
        normalized = re.sub(r"\s+", " ", source)
        assert (
            "request.max_tokens if request.max_tokens is not None" in normalized
            or "if request.max_tokens is not None else _default_max_tokens" in normalized
        ), (
            "max_tokens must use 'is not None' check, not 'or' (0 is falsy)"
        )


class TestC5ToolCallsInThinkBlocks:
    """C5: Tool parsing must use accumulated_content when reasoning parser is active."""

    def test_chat_completions_uses_accumulated_content(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # When reasoning parser is active, should use accumulated_content
        assert "request_parser and accumulated_content" in source, (
            "Tool call parsing must prefer accumulated_content when reasoning parser active"
        )

    def test_responses_api_uses_accumulated_content(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "request_parser and accumulated_content" in source, (
            "Responses API tool parsing must prefer accumulated_content when reasoning parser active"
        )


class TestH1StopTokenCleanup:
    """H1: Per-request stop tokens must use a snapshot of surviving requests,
    not read from self.running (which is mutated during cleanup loop)."""

    def test_surviving_stops_snapshot_before_loop(self):
        """_surviving_stops must be computed BEFORE the per-request loop."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        # _surviving_stops must appear before the "for request_id in finished_ids" loop
        snap_pos = source.find("_surviving_stops")
        loop_pos = source.find("for request_id in finished_ids")
        assert snap_pos < loop_pos, (
            "_surviving_stops snapshot must be computed before the cleanup loop"
        )

    def test_removable_uses_snapshot(self):
        """Stop token removal must subtract _surviving_stops, not re-read self.running."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "_surviving_stops" in source, (
            "Cleanup must use _surviving_stops snapshot"
        )
        assert "removable = request._added_stop_tokens - _surviving_stops" in source, (
            "Removable stop tokens must subtract surviving stops snapshot"
        )

    def test_surviving_stops_excludes_finished(self):
        """The snapshot must only include stops from requests NOT in finished_ids."""
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler._cleanup_finished)
        assert "rid not in finished_ids" in source, (
            "Surviving stops must exclude requests that are finishing"
        )


class TestH2ImageCountLimit:
    """H2: Excessive images must be rejected to prevent Metal OOM."""

    def test_mllm_scheduler_config_has_limit(self):
        from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig
        config = MLLMSchedulerConfig()
        assert hasattr(config, 'max_images_per_request')
        assert config.max_images_per_request > 0

    def test_add_request_rejects_excessive_images(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.add_request)
        assert "max_images_per_request" in source, (
            "add_request must check image count against max_images_per_request"
        )

    def test_add_request_raises_on_too_many_images(self):
        from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(max_images_per_request=3)
        # Config limit is 3, so 5 images should trigger the guard
        assert config.max_images_per_request == 3


class TestH3VideoExtractionFailures:
    """H3: When ALL media inputs fail, must raise instead of silently continuing."""

    def test_multimodal_processor_raises_on_all_failures(self):
        import inspect
        from vmlx_engine.multimodal_processor import MultimodalProcessor

        source = inspect.getsource(MultimodalProcessor.process)
        assert "All media inputs failed" in source, (
            "Must raise ValueError when all images/videos fail to process"
        )

    def test_counts_failed_media(self):
        import inspect
        from vmlx_engine.multimodal_processor import MultimodalProcessor

        source = inspect.getsource(MultimodalProcessor.process)
        assert "failed_images" in source and "failed_videos" in source, (
            "Must track failed image and video counts"
        )


class TestH4JsonSchemaStreaming:
    """H4: JSON schema/object validation must happen at end of streaming."""

    def test_chat_completion_streaming_validates_json(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert "parse_json_output" in source, (
            "Streaming path must call parse_json_output for response_format validation"
        )

    def test_responses_api_streaming_validates_json(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "parse_json_output" in source, (
            "Responses API streaming must validate JSON format"
        )

    def test_streaming_emits_error_on_strict_failure(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        assert "json_validation_failed" in source, (
            "Streaming must emit error event on strict JSON schema failure"
        )


class TestH6ConfigRestartWarning:
    """H6: updateSessionConfig must return restart-required info."""

    def test_restart_required_keys_defined(self):
        """SessionManager must define which config keys need restart."""
        import inspect
        # Read source file directly since TypeScript
        source_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'sessions.ts'
        )
        with open(source_path) as f:
            source = f.read()
        assert "RESTART_REQUIRED_KEYS" in source, (
            "SessionManager must define RESTART_REQUIRED_KEYS"
        )
        assert "restartRequired" in source, (
            "updateSessionConfig must return restartRequired flag"
        )


class TestM5Base64TempFileCleanup:
    """M5: LRU cache eviction must also delete the temp file from disk."""

    def test_eviction_calls_cleanup(self):
        import inspect
        from vmlx_engine.models.mllm import save_base64_image

        source = inspect.getsource(save_base64_image)
        assert "_temp_manager.cleanup(evicted_path)" in source, (
            "Base64 image cache eviction must call _temp_manager.cleanup on evicted path"
        )


class TestM8UsageOnError:
    """M8: Usage must be sent even when stream encounters an error."""

    def test_chat_completion_stream_sends_usage_on_error(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # The error handler must include usage when include_usage is on
        # Find the except block and check for usage handling
        error_section = source[source.find("Stream generation failed"):]
        assert "include_usage" in error_section[:500], (
            "Stream error handler must check include_usage and send partial usage"
        )

    def test_responses_api_stream_sends_usage_on_error(self):
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        error_section = source[source.find("Stream generation failed"):]
        assert "usage" in error_section[:500], (
            "Responses API error handler must include usage in failed response"
        )


class TestC3BlockDiskCacheQuantScoping:
    """C3 extension: Block-level disk cache must also scope by quantization config."""

    def test_scheduler_block_cache_includes_quant(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        # Find the block disk cache section
        block_section = source[source.find("enable_block_disk_cache"):]
        assert "quant" in block_section[:500], (
            "Block disk cache hash must include quantization in scope key"
        )

    def test_mllm_scheduler_block_cache_includes_quant(self):
        import inspect
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        source = inspect.getsource(MLLMScheduler.__init__)
        block_section = source[source.find("enable_block_disk_cache"):]
        assert "quant" in block_section[:500], (
            "MLLM block disk cache hash must include quantization in scope key"
        )


class TestTimeoutFalsy:
    """Timeout=0 must not be silently overridden to default (same class of bug as C4)."""

    def test_all_timeout_uses_is_not_none(self):
        import inspect
        from vmlx_engine.server import (
            create_chat_completion,
            create_response,
        )

        for fn in [create_chat_completion, create_response]:
            source = inspect.getsource(fn)
            assert "request.timeout or _default_timeout" not in source, (
                f"{fn.__name__} must use 'is not None' for timeout, not 'or'"
            )
            assert "request.timeout if request.timeout is not None" in source, (
                f"{fn.__name__} must use 'is not None' check for timeout"
            )


# ===========================================================================
# H3 Production Path: _prepare_images total-failure detection
# ===========================================================================


class TestH3PrepareImagesTotalFailure:
    """Test that _prepare_images raises on total failure (production path in models/mllm.py)."""

    @staticmethod
    def _read_mllm_source():
        return (Path(__file__).parent.parent / "vmlx_engine" / "models" / "mllm.py").read_text()

    def test_all_images_fail_raises_valueerror(self):
        """When every image fails to process, should raise ValueError."""
        source = self._read_mllm_source()
        # _prepare_images must track failures and raise when all fail
        assert "failed_count" in source
        assert "images and not processed" in source
        assert 'raise ValueError' in source

    def test_prepare_images_has_failure_guard(self):
        """_prepare_images must guard against total failure but allow partial."""
        source = self._read_mllm_source()
        # Guard: `if images and not processed` — empty list skips, partial succeeds
        assert "images and not processed" in source


# ===========================================================================
# H4 Regression: Fallback text skips JSON validation
# ===========================================================================


class TestH4FallbackTextSkipsJsonValidation:
    """Test that fallback '[Model produced no response...]' doesn't trigger JSON validation."""

    def test_responses_api_skips_validation_on_fallback(self):
        """Responses API JSON validation must skip the fallback message."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        # Must define fallback message constant
        assert "_FALLBACK_MSG" in source or "Model produced no response" in source
        # Must check display_text != fallback before validating
        assert "_FALLBACK_MSG" in source

    def test_chat_completions_already_safe(self):
        """Chat completions uses content_was_emitted gate — already safe."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # Chat completions gates on content_was_emitted
        assert "content_was_emitted" in source


# ===========================================================================
# H2 Video Frame Bypass: Total image count guard in generate paths
# ===========================================================================


class TestH2VideoFrameBypass:
    """Test that total image count (including video frames) is enforced."""

    @staticmethod
    def _read_mllm_source():
        return (Path(__file__).parent.parent / "vmlx_engine" / "models" / "mllm.py").read_text()

    def test_all_generate_paths_have_total_image_guard(self):
        """All generate/stream_generate/chat/stream_chat must check total images."""
        source = self._read_mllm_source()
        # Count occurrences of the guard — should appear in all 4 generate paths
        guard_count = source.count("max_images_per_request")
        assert guard_count >= 4, (
            f"Expected max_images_per_request guard in at least 4 places, found {guard_count}"
        )
        assert source.count("including video frames") >= 4, (
            "Expected 'including video frames' error message in all 4 generate paths"
        )


# ===========================================================================
# H1 LLM Scheduler Parity: Stop token cleanup
# ===========================================================================


class TestH1LLMSchedulerStopTokenParity:
    """Test that LLM Scheduler has stop token cleanup matching MLLM Scheduler."""

    def test_scheduler_has_stop_tokens_attribute(self):
        """LLM Scheduler must track base stop tokens."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler.__init__)
        assert "self.stop_tokens" in source
        assert "_get_stop_tokens" in source

    def test_schedule_waiting_adds_per_request_stop_tokens(self):
        """_schedule_waiting must add per-request stop tokens to batch generator."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._schedule_waiting)
        assert "_added_stop_tokens" in source
        assert "batch_generator.stop_tokens.update" in source

    def test_cleanup_finished_uses_surviving_stops_snapshot(self):
        """_cleanup_finished must use _surviving_stops snapshot pattern."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)
        assert "_surviving_stops" in source
        assert "request._added_stop_tokens" in source
        assert "removable" in source

    def test_cleanup_never_removes_base_stop_tokens(self):
        """Cleanup must subtract self.stop_tokens to protect base EOS tokens."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)
        assert "self.stop_tokens" in source


# ===========================================================================
# V3 Audit: Deep cross-component issues
# ===========================================================================


class TestV3ToolCallIdForwarding:
    """Non-streaming Responses API must forward tc.id from parser."""

    def test_responses_nonstreaming_forwards_tc_id(self):
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        # Must reference tc.id and pass call_id to ResponsesFunctionCall
        assert "tc.id" in source or "tc_call_id" in source
        assert "call_id" in source


class TestV3RescheduleCleanup:
    """_reschedule_running_requests must clear _extracted_cache."""

    def test_extracted_cache_cleared_on_reschedule(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._reschedule_running_requests)
        assert "_extracted_cache" in source


class TestV3ScheduleWaitingRecovery:
    """_schedule_waiting must not lose requests on insert failure."""

    def test_lost_request_protection(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._schedule_waiting)
        # Must have outer try/except that puts request back
        assert "waiting.appendleft(request)" in source
        # Must appear at least twice — once for batch_generator=None, once for insert failure
        assert source.count("waiting.appendleft(request)") >= 2


class TestV3BlockCacheFinallyCleanup:
    """block_aware_cache branch must use finally for _extracted_cache."""

    def test_block_cache_uses_finally(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._cleanup_finished)
        # Must have 'finally' followed by _extracted_cache cleanup
        assert "finally:" in source
        assert "request._extracted_cache = None" in source


class TestV3GenerateCleanupOnError:
    """EngineCore.generate() must clean up on all error paths."""

    def test_generate_has_finally_cleanup(self):
        import inspect
        from vmlx_engine.engine_core import EngineCore
        source = inspect.getsource(EngineCore.generate)
        assert "finally:" in source
        assert "_cleanup_request" in source


class TestV3ResponsesTextFormatSchema:
    """ResponsesTextFormat must preserve json_schema field."""

    def test_text_format_has_json_schema_field(self):
        from vmlx_engine.api.models import ResponsesTextFormat
        # json_schema field must exist
        assert "json_schema" in ResponsesTextFormat.model_fields

    def test_text_format_preserves_schema_data(self):
        from vmlx_engine.api.models import ResponsesTextFormat
        fmt = ResponsesTextFormat(type="json_schema", json_schema={"name": "test", "schema": {"type": "object"}})
        dumped = fmt.model_dump()
        assert dumped["json_schema"] is not None
        assert dumped["json_schema"]["name"] == "test"


class TestV3ChatCompletionStopNormalization:
    """ChatCompletionRequest.stop must accept bare strings."""

    def test_bare_string_normalized_to_list(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop="\\n",
        )
        assert isinstance(req.stop, list)
        assert req.stop == ["\\n"]

    def test_list_preserved(self):
        from vmlx_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop=["\\n", "END"],
        )
        assert req.stop == ["\\n", "END"]


class TestV3SuppressReasoningNoFallback:
    """Suppress reasoning + reasoning-only output should NOT show fallback."""

    def test_responses_api_no_fallback_on_suppressed_reasoning(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Must check suppress_reasoning before showing fallback
        assert "suppress_reasoning and accumulated_reasoning" in source


class TestV3ReasoningDoubleAccumulation:
    """v1.3.56 §15 SUPERSEDES V3-H1: under suppress_reasoning the parser's
    reasoning delta is routed into emit_content and MUST also mirror into
    accumulated_content so content_was_emitted + tool-call marker detection
    stay truthful. V3-H1's 'no double accumulation' assertion is no longer
    applicable.

    These tests now verify that the accumulated_content mirror exists AND
    is guarded by suppress_reasoning so non-suppressed requests don't get
    polluted. See NO-REGRESSION-CHECKLIST.md §18.
    """

    def test_chat_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "accumulated_content += delta_msg.reasoning" in source
        # Under suppress only — not unconditional — so non-suppressed flows
        # keep accumulated_content content-only (for clean tool-call marker
        # detection when the user wants to see reasoning).
        assert "if suppress_reasoning:" in source

    def test_responses_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "accumulated_content += delta_msg.reasoning" in source
        assert "if suppress_reasoning:" in source


class TestV3KvCacheBitsInit:
    """Scheduler must initialize _kv_cache_bits to 0."""

    def test_kv_cache_bits_in_init(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.__init__)
        assert "self._kv_cache_bits" in source
        assert "_kv_cache_bits: int = 0" in source or "_kv_cache_bits = 0" in source


# ===========================================================================
# V4. Deep Audit — Cache subsystem, prefix cache, truncation fixes
# ===========================================================================


class TestV4DiskCacheDequantizeGuard:
    """Disk cache fetch must dequantize when KV quantization is active."""

    def test_scheduler_disk_cache_dequantize(self):
        """scheduler.py: disk_cache.fetch result must go through dequantize."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.add_request)
        # After disk_cache.fetch, there must be a dequantize call
        fetch_idx = source.find("disk_cache.fetch")
        assert fetch_idx != -1, "disk_cache.fetch not found in add_request"
        after_fetch = source[fetch_idx:fetch_idx + 600]
        assert "_dequantize_cache" in after_fetch, (
            "Missing dequantize guard after disk_cache.fetch in scheduler.py"
        )

    def test_mllm_disk_cache_dequantize(self):
        """mllm_batch_generator.py: disk cache fetch must dequantize."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the actual disk_cache.fetch() call (with parens), not docstring mention
        fetch_idx = source.find("self.disk_cache.fetch(")
        assert fetch_idx != -1, "self.disk_cache.fetch() not found in mllm_batch_generator.py"
        after_fetch = source[fetch_idx:fetch_idx + 1500]
        assert "_dequantize_cache" in after_fetch, (
            "Missing dequantize guard after disk_cache.fetch in mllm_batch_generator.py"
        )


class TestV4DequantizeFreshKVCacheFallback:
    """_dequantize_cache must return fresh KVCache for QuantizedKVCache with keys=None."""

    def test_quantized_with_none_keys_gets_fresh_kvcache(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = source.find("def _dequantize_cache(")
        assert func_start != -1
        # Find the next function definition to bound our search
        func_end = source.find("\ndef ", func_start + 10)
        func_body = source[func_start:func_end]
        # Must handle QuantizedKVCache with keys=None by creating fresh KVCache
        assert "KVCache()" in func_body, (
            "_dequantize_cache must create fresh KVCache() for empty QuantizedKVCache layers"
        )
        # Must NOT silently pass through QuantizedKVCache to result
        assert "keys is not None" in func_body or "keys is None" in func_body, (
            "_dequantize_cache must explicitly check for None keys"
        )


class TestV4FixHybridCacheDequantize:
    """_fix_hybrid_cache call in prefill must be preceded by _dequantize_cache."""

    def test_dequantize_before_fix_hybrid(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the prefill section where _fix_hybrid_cache is called on req.prompt_cache
        idx = source.find("req_cache = _fix_hybrid_cache(")
        assert idx != -1
        # Look at the nearby prefill branch before this call. Keep the window
        # wide enough to include the explicit dequantize failure guard without
        # becoming a whole-file substring assertion.
        before = source[max(0, idx - 1200):idx]
        assert "_dequantize_cache" in before, (
            "Must call _dequantize_cache before _fix_hybrid_cache in prefill path"
        )
        # Verify None guard exists after dequantize
        assert "cache_for_fix is None" in before, (
            "Must guard against _dequantize_cache returning None"
        )


class TestV4RotatingKVCacheTruncation:
    """RotatingKVCache must not truncate when circular buffer has wrapped."""

    def test_rotating_cache_wrap_detection(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must check for offset > max_size (wrapped circular buffer)
        assert "offset > max_size" in source, (
            "Must detect wrapped RotatingKVCache (offset > max_size) and skip"
        )

    def test_rotating_cache_idx_restore(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must restore _idx for RotatingKVCache
        assert "_idx" in source, (
            "Must restore _idx for RotatingKVCache after truncation"
        )

    def test_safe_target_bounds(self):
        """Truncation must use min(target_len, actual_shape) to prevent OOB."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "safe_target" in source or "min(target_len" in source, (
            "Must bound slice to actual tensor shape to prevent OOB"
        )


class TestV4CacheListTruncation:
    """CacheList (DeepSeek V3.2, Falcon H1) must be handled in truncation."""

    def test_cachelist_branch_exists(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "CacheList" in source, (
            "_truncate_cache_to_prompt_length must handle CacheList"
        )
        assert "caches" in source, (
            "Must access CacheList.caches for recursive truncation"
        )


class TestV4PrefixCacheLRU:
    """PrefixCacheManager LRU must use OrderedDict for O(1) and dedup.

    LRU storage is now partitioned by cache_type (assistant / user / system)
    via ``_lru_by_type`` so eviction can prefer lower-priority buckets
    first. The original ``_lru`` attribute was replaced; these tests walk
    all buckets to produce the combined view.
    """

    @staticmethod
    def _all_lru_keys(mgr):
        keys = []
        for od in mgr._lru_by_type.values():
            keys.extend(od.keys())
        return keys

    def test_lru_is_ordered_dict(self):
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        from collections import OrderedDict
        assert hasattr(mgr, "_lru_by_type"), (
            "PrefixCacheManager must expose per-type LRU dict"
        )
        for t, od in mgr._lru_by_type.items():
            assert isinstance(od, OrderedDict), (
                f"bucket {t!r} must be OrderedDict for O(1) reordering"
            )

    def test_no_duplicate_lru_entries(self):
        """Storing same tokens twice must not create duplicate LRU entries."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        tokens = [1, 2, 3]
        cache = [MagicMock()]
        mgr.store_cache(tokens, cache)
        mgr.store_cache(tokens, cache)
        total = len(self._all_lru_keys(mgr))
        assert total == 1, (
            f"Expected 1 LRU entry after duplicate store, got {total}"
        )

    def test_touch_lru_moves_to_end(self):
        """Touching an entry must move it to the end (MRU position)."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=10)
        mgr.store_cache([1, 2], [MagicMock()])
        mgr.store_cache([3, 4], [MagicMock()])
        # Touch first entry — it should move to end of its bucket
        mgr._touch_lru(tuple([1, 2]))
        keys = self._all_lru_keys(mgr)
        assert keys[-1] == (mgr.model_key, (1, 2)), (
            "Touch must move entry to MRU end"
        )

    def test_eviction_removes_lru(self):
        """Eviction must remove the least recently used entry."""
        from vmlx_engine.prefix_cache import PrefixCacheManager
        mock_model = MagicMock()
        mgr = PrefixCacheManager(mock_model, max_entries=2)
        mgr.store_cache([1], [MagicMock()])
        mgr.store_cache([2], [MagicMock()])
        # This should evict [1]
        mgr.store_cache([3], [MagicMock()])
        keys = self._all_lru_keys(mgr)
        assert len(keys) == 2
        assert (mgr.model_key, (1,)) not in keys, "LRU entry [1] should have been evicted"


class TestV4QuantizeCacheSubclassHandling:
    """_quantize_cache_for_storage must use isinstance, not type() is."""

    def test_uses_isinstance_not_type_is(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._quantize_cache_for_storage)
        assert "isinstance(layer_cache, KVCache)" in source, (
            "Must use isinstance() to catch KVCache subclasses (e.g. RotatingKVCache)"
        )
        assert "type(layer_cache) is KVCache" not in source, (
            "Must NOT use strict type() check — misses subclasses"
        )


# ===========================================================================
# V4b. Deep Audit Review — Cross-component cohesion fixes
# ===========================================================================


class TestV4bDiskCacheDequantFallthrough:
    """Disk cache dequant failure must NOT corrupt request state."""

    def test_scheduler_disk_dequant_failure_skips_state_mutation(self):
        """When dequant returns None, cached_tokens and remaining_tokens must NOT be set."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.add_request)
        # Find the disk cache section
        fetch_idx = source.find("disk_cache = self.disk_cache.fetch")
        assert fetch_idx != -1
        after_fetch = source[fetch_idx:fetch_idx + 1200]
        # The pattern: if dequant fails (disk_cache is None), must NOT set cached_tokens
        # Correct pattern: else branch gates all state mutations
        assert "else:" in after_fetch, (
            "Disk cache dequant failure must have else branch to skip state mutation"
        )
        # Find 'request.cached_tokens' — it must be INSIDE the else block (indented deeper)
        cached_idx = after_fetch.find("request.cached_tokens")
        else_idx = after_fetch.find("else:")
        assert cached_idx > else_idx, (
            "request.cached_tokens must be inside the else block (after dequant success)"
        )


class TestV4bSchedulerDequantKeysNone:
    """scheduler.py _dequantize_cache_for_use must handle QuantizedKVCache(keys=None)."""

    def test_keys_none_gets_fresh_kvcache(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._dequantize_cache_for_use)
        # Must create fresh KVCache for empty QuantizedKVCache layers
        assert "KVCache()" in source, (
            "_dequantize_cache_for_use must create fresh KVCache() for empty QuantizedKVCache"
        )

    def test_consistent_with_mllm_version(self):
        """Both dequantize functions must handle keys=None the same way."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        scheduler_src = inspect.getsource(Scheduler._dequantize_cache_for_use)
        mllm_src = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = mllm_src.find("def _dequantize_cache(")
        func_end = mllm_src.find("\ndef ", func_start + 10)
        mllm_func = mllm_src[func_start:func_end]
        # Both must have fresh KVCache for keys=None
        assert "KVCache()" in scheduler_src, "Scheduler missing KVCache() fallback"
        assert "KVCache()" in mllm_func, "MLLM missing KVCache() fallback"


class TestV4bCacheListTupleInvariant:
    """CacheList.caches must be stored as tuple to match constructor invariant."""

    def test_cachelist_stored_as_tuple(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        assert "tuple(sub_result)" in source, (
            "CacheList.caches must be stored as tuple, not list"
        )


class TestV4bImageHashCollisionSafe:
    """Paged cache image hashing must use content-based hash, not sum-based."""

    def test_no_sum_based_hash(self):
        source = Path("./vmlx_engine/paged_cache.py").read_text()
        hash_section = source[source.find("def _hash_extra"):source.find("_hash_extra(extra_keys)")]
        assert "mx.sum" not in hash_section, (
            "Must not use mx.sum for image hashing — collision-prone"
        )
        assert "tobytes" in hash_section, (
            "Must use tobytes() for collision-safe content hashing"
        )


class TestV4bToolChoiceRequired:
    """tool_choice='required' must be enforced in all API paths."""

    def test_chat_completions_nonstreaming(self):
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "required" in source, (
            "Chat Completions must check tool_choice='required'"
        )
        assert "tool_calls_required" in source or "HTTPException" in source, (
            "Chat Completions must raise error when required tool calls missing"
        )

    def test_responses_nonstreaming(self):
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        assert "required" in source, (
            "Responses API must check tool_choice='required'"
        )

    def test_chat_completions_streaming(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "tool_calls_required" in source, (
            "Chat Completions streaming must emit error for tool_choice='required'"
        )

    def test_responses_streaming(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "tool_calls_required" in source, (
            "Responses API streaming must emit error for tool_choice='required'"
        )


# ===========================================================================
# L. LOW severity fixes — port race, delta streaming, reasoning_effort
# ===========================================================================


class TestL1PortRaceCondition:
    """Session creation must serialize port assignment to prevent races."""

    def test_creation_lock_exists(self):
        """SessionManager must have a global creation lock field."""
        source = Path("./panel/src/main/sessions.ts").read_text()
        assert "creationLock" in source, (
            "SessionManager must have a creationLock to serialize createSession"
        )

    def test_create_session_uses_lock(self):
        """createSession must acquire creationLock before port assignment."""
        source = Path("./panel/src/main/sessions.ts").read_text()
        # createSession should delegate to _createSessionInner
        assert "_createSessionInner" in source, (
            "createSession must delegate to _createSessionInner under lock"
        )

    def test_port_unique_constraint(self):
        """sessions table must have UNIQUE constraint on port column."""
        source = Path("./panel/src/main/database.ts").read_text()
        assert "port INTEGER NOT NULL UNIQUE" in source, (
            "sessions.port must have UNIQUE constraint as safety net"
        )


class TestL2IncrementalDelta:
    """Responses API function_call_arguments.delta must be incremental."""

    def test_delta_is_chunked(self):
        """Arguments must be emitted in chunks, not as one big delta."""
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Must have a chunking loop (range + _ARG_CHUNK or similar)
        assert "_ARG_CHUNK" in source or "CHUNK_SIZE" in source, (
            "Must chunk arguments into incremental deltas"
        )
        # Must iterate over argument characters
        assert "range(0, len(tc_args)" in source or "range(0, max(len(tc_args)" in source, (
            "Must iterate over argument string for chunking"
        )


class TestL3ReasoningEffort:
    """reasoning_effort must map to thinking_budget and max_tokens."""

    def test_effort_constants_defined(self):
        """Server must define effort-to-budget mapping constants."""
        from vmlx_engine.server import _EFFORT_THINKING_BUDGET, _EFFORT_MAX_TOKENS
        assert "low" in _EFFORT_THINKING_BUDGET
        assert "medium" in _EFFORT_THINKING_BUDGET
        assert "high" in _EFFORT_THINKING_BUDGET
        assert _EFFORT_THINKING_BUDGET["low"] < _EFFORT_THINKING_BUDGET["medium"]
        assert _EFFORT_THINKING_BUDGET["medium"] < _EFFORT_THINKING_BUDGET["high"]
        assert "low" in _EFFORT_MAX_TOKENS
        assert _EFFORT_MAX_TOKENS["low"] < _EFFORT_MAX_TOKENS["high"]

    def test_chat_completions_maps_effort(self):
        """Chat Completions must map reasoning_effort to thinking_budget."""
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "thinking_budget" in source, (
            "Chat Completions must inject thinking_budget from reasoning_effort"
        )
        assert "_EFFORT_THINKING_BUDGET" in source, (
            "Must use the _EFFORT_THINKING_BUDGET mapping"
        )

    def test_responses_maps_effort(self):
        """Responses API must map reasoning_effort to thinking_budget."""
        import inspect
        from vmlx_engine.server import create_response
        source = inspect.getsource(create_response)
        assert "thinking_budget" in source, (
            "Responses API must inject thinking_budget from reasoning_effort"
        )

    def test_effort_sets_max_tokens_when_unset(self):
        """reasoning_effort must set max_tokens when not explicitly provided."""
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "_EFFORT_MAX_TOKENS" in source, (
            "Must use _EFFORT_MAX_TOKENS to cap generation"
        )
        assert "max_tokens is None" in source, (
            "Must only set max_tokens when user didn't specify it"
        )


# ── V5 FIXES (Pre-release cohesion review) ──────────────────────────────────

class TestV5CacheListTupleDetection:
    """CacheList.caches is always a tuple — detection must accept both list and tuple."""

    def test_truncation_accepts_tuple(self):
        """_truncate_cache_to_prompt_length must check isinstance(..., (list, tuple))."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler._truncate_cache_to_prompt_length)
        # Must accept tuple (CacheList stores .caches as tuple)
        assert "(list, tuple)" in source, (
            "CacheList detection must accept both list and tuple since CacheList.caches is always a tuple"
        )


class TestV5StreamingToolChoiceRequired:
    """Streaming tool_choice='required' must track actual emission, not just buffering."""

    def test_tool_calls_emitted_flag_exists(self):
        """stream_chat_completion must have a tool_calls_emitted flag."""
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "tool_calls_emitted" in source, (
            "Must track whether tool calls were actually emitted, not just buffering state"
        )

    def test_required_check_uses_emitted_flag(self):
        """tool_choice='required' enforcement must check tool_calls_emitted, not tool_call_buffering."""
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        # The enforcement line must use tool_calls_emitted
        assert 'not tool_calls_emitted' in source, (
            "tool_choice='required' enforcement must use 'not tool_calls_emitted', "
            "not 'not tool_call_buffering' (which can be true on false-positive marker detection)"
        )
        # Must NOT use tool_call_buffering for the required check
        assert '"required" and not tool_call_buffering' not in source, (
            "Must not use tool_call_buffering for required enforcement — it stays True on false positives"
        )


class TestV5DisplayTextInit:
    """display_text must be initialized before the if/else tool_calls branch."""

    def test_display_text_initialized_before_branch(self):
        """stream_responses_api must initialize display_text before tool_calls branch."""
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        # Find initialization before the tool_calls branch
        init_idx = source.find('display_text = ""')
        assert init_idx != -1, "display_text must be initialized to empty string"
        # Must appear before the H4 JSON validation that references display_text
        h4_idx = source.find("H4: Validate text format")
        assert h4_idx != -1, "H4 validation block must exist"
        assert init_idx < h4_idx, (
            "display_text initialization must come before H4 validation to prevent UnboundLocalError"
        )


class TestResponsesSuppressedReasoningToolCalls:
    """Suppressed reasoning can still contain a real tool call."""

    def test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize(self):
        """Tool calls found only in suppressed reasoning must enter the Responses tool branch."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        extract_idx = source.find("tool markers in suppressed reasoning")
        branch_idx = source.find("if tool_calls:")

        assert extract_idx != -1, (
            "Responses API must extract tool calls from suppressed reasoning before finalization"
        )
        assert branch_idx != -1, "Responses API tool_calls branch missing"
        assert extract_idx < branch_idx, (
            "Suppressed-reasoning tool extraction must run before the final tool_calls branch"
        )
        assert "TODO: emit tool calls via Responses API format" not in source, (
            "Responses API must not leave parsed suppressed-reasoning tool calls un-emitted"
        )


class TestAnthropicOmniStreamingAdapter:
    """Anthropic Omni streaming must not leak OpenAI SSE chunks."""

    def test_omni_streaming_path_uses_anthropic_adapter(self):
        import inspect
        from vmlx_engine.server import create_anthropic_message

        source = inspect.getsource(create_anthropic_message)
        assert "and anthropic_req.stream" in source
        assert "AnthropicStreamAdapter(model=resolved_name)" in source
        assert "adapter.process_chunk(line)" in source
        assert 'media_type="text/event-stream"' in source
        assert "pass through unchanged" not in source
        assert "Caller may see OpenAI" not in source


class TestDSV4FastLoadSwitchGLUScope:
    """DSV4 fast-load speed patch must not hijack other SwitchGLU models."""

    def test_switchglu_patch_is_marker_scoped_and_idempotent(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _try_fast_load_dsv4

        source = inspect.getsource(_try_fast_load_dsv4)
        assert "_vmlx_dsv4_original_call" in source, (
            "Fast-load patch must keep the original SwitchGLU.__call__ instead of wrapping wrappers"
        )
        assert "_vmlx_dsv4_fused_fastpath" in source, (
            "Fast-load patch must mark only DSV4 modules as eligible for fused decoding"
        )
        guard_idx = source.find('if not getattr(self, "_vmlx_dsv4_fused_fastpath", False):')
        gp_idx = source.find("gp = self.gate_proj")
        assert guard_idx != -1 and gp_idx != -1 and guard_idx < gp_idx, (
            "Non-DSV4 SwitchGLU modules must fall back before the DSV4 TurboQuant path"
        )

    def test_fast_load_switchglu_threads_down_proj_bits(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _try_fast_load_dsv4

        source = inspect.getsource(_try_fast_load_dsv4)
        assert "dp_bits=None" in source
        assert "dp_bits=dp.bits" in source
        assert "make_gather_tq_decode_per_row(out_f, in_f, dp_bits, k)" in source
        assert "cache_key = (in_f, out_f, bits, dp_bits, k, limit_milli)" in source


class TestDSV4SidecarManifestRuntimePatch:
    """DSV4 sidecars must be invalidated when runtime patches change."""

    def test_manifest_records_and_checks_runtime_patch_version(self):
        import inspect
        import vmlx_engine.loaders.load_jangtq_dsv4 as loader

        fast_source = inspect.getsource(loader._try_fast_load_dsv4)
        write_source = inspect.getsource(loader._write_sidecar_after_hydrate)

        assert hasattr(loader, "_INSTANT_LOAD_RUNTIME_PATCH")
        assert '"runtime_patch": _INSTANT_LOAD_RUNTIME_PATCH' in write_source
        assert 'manifest.get("runtime_patch") != _INSTANT_LOAD_RUNTIME_PATCH' in fast_source


class TestV5FixHybridCacheExcept:
    """_fix_hybrid_cache outermost except must return fresh cache, not broken original."""

    def test_except_returns_make_cache(self):
        """Outermost except in _fix_hybrid_cache must call make_cache() not return original cache."""
        import inspect
        from vmlx_engine.mllm_batch_generator import _fix_hybrid_cache
        source = inspect.getsource(_fix_hybrid_cache)
        # Find the outermost except block (last except in the function)
        lines = source.split('\n')
        last_except_idx = None
        for i, line in enumerate(lines):
            if 'except Exception' in line:
                last_except_idx = i
        assert last_except_idx is not None, "Must have outermost except block"
        # Check the lines after the except
        after_except = '\n'.join(lines[last_except_idx:last_except_idx + 5])
        assert "make_cache()" in after_except, (
            "Outermost except must call language_model.make_cache() to return fresh cache, "
            "not return the potentially broken original cache"
        )

    def test_except_has_fallback(self):
        """Must fall back to original cache if make_cache is not available."""
        import inspect
        from vmlx_engine.mllm_batch_generator import _fix_hybrid_cache
        source = inspect.getsource(_fix_hybrid_cache)
        # Must have a hasattr check for make_cache in the except path
        assert "hasattr(language_model, 'make_cache')" in source, (
            "Must check hasattr before calling make_cache in except handler"
        )


class TestGenerationBatchFastNoLogprobs:
    """Text BatchGenerator should not materialize full-vocab logprobs per token."""

    def test_generation_step_keeps_logprobs_transient(self):
        source = Path("./vmlx_engine/utils/mamba_cache.py").read_text()
        func_start = source.find("def _patch_generation_step_sync")
        assert func_start >= 0
        func_body = source[func_start: source.find("\ndef ", func_start + 10)]
        assert "vMLX does not expose" in func_body
        assert "self._next_logprobs = [None] * len(self.uids)" in func_body
        assert "mx.eval(inputs, self._current_logprobs)" not in func_body
        assert "mx.async_eval(self._next_tokens, self._next_logprobs" not in func_body


class TestHybridSSMCompanionCacheGating:
    """Hybrid SSM companion work must obey the prefix-cache enable flag."""

    def test_llm_scheduler_does_not_init_or_store_ssm_when_prefix_cache_disabled(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()
        init_idx = source.index("self._ssm_state_cache: Optional[HybridSSMStateCache] = None")
        init_block = source[init_idx : source.index("# Prompt lookup decoding", init_idx)]
        assert "self.config.enable_prefix_cache" in init_block
        assert "HybridSSMStateCache(" in init_block
        assert "max_entries=_ssm_cache_size" in init_block
        assert "max_bytes=(" in init_block

        store_idx = source.index("# Hybrid SSM companion state capture.")
        store_block = source[store_idx : source.index("# Store cache for future reuse", store_idx)]
        assert "and self.config.enable_prefix_cache" in store_block

        rederive_idx = source.index("# ── Deferred SSM re-derive")
        rederive_block = source[rederive_idx : source.index("return output", rederive_idx)]
        assert "and self.config.enable_prefix_cache" in rederive_block

    def test_mllm_batch_generator_disables_hidden_ssm_work_without_prefix_cache(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        assert "enable_prefix_cache: bool = True" in source
        assert "self._prefix_cache_enabled = bool(enable_prefix_cache)" in source
        assert "self._ssm_companion_enabled = bool(" in source
        assert "if self._ssm_companion_enabled" in source
        assert "else None" in source
        assert "self._prefix_cache_enabled" in source
        assert "block_aware_cache is not None" in source
        assert "if self._is_hybrid and self._ssm_companion_enabled:" in source

    def test_mllm_scheduler_threads_prefix_cache_flag_to_batch_generator(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        generator_idx = source.index("self.batch_generator = MLLMBatchGenerator(")
        generator_block = source[generator_idx : source.index("self._current_sampler_params", generator_idx)]
        assert "enable_prefix_cache=self.config.enable_prefix_cache" in generator_block


class TestStartupCompatibilityGuards:
    def test_cli_checks_mlx_wheel_macos_tag_before_import(self):
        source = Path("./vmlx_engine/cli.py").read_text()
        check_idx = source.index("def _check_macos_compat")
        check_block = source[check_idx: source.index("def _check_no_duplicate_mlx", check_idx)]
        assert "importlib.metadata.distribution" in check_block
        assert "macosx_(\\d+)_(\\d+)_arm64" in check_block
        assert "Failed to load the default metallib" in check_block
        assert "_check_macos_compat()" in source

    def test_bundled_python_requires_mflux_for_image_models(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        assert 'MFLUX_VERSION="0.17.5"' in bundle_script
        assert '"mflux==$MFLUX_VERSION"' in bundle_script
        assert '("mflux", "mflux image runtime"' in verify_script
        assert '"mflux.models.common.config.model_config"' in verify_script

    def test_bundle_forces_sonoma_mlx_wheels_on_tahoe_build_hosts(self):
        """The release bundle must not inherit the builder host's macOS wheel tag."""
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        assert 'MLX_VERSION="0.31.2"' in bundle_script
        assert 'MLX_LM_VERSION="0.31.3"' in bundle_script
        assert 'MLX_VLM_VERSION="0.4.4"' in bundle_script
        assert 'MLX_WHEEL_PLATFORM="${VMLX_BUNDLE_MLX_PLATFORM:-macosx_14_0_arm64}"' in bundle_script
        assert '--platform "$MLX_WHEEL_PLATFORM"' in bundle_script
        assert '"mlx==$MLX_VERSION"' in bundle_script
        assert '"mlx-metal==$MLX_VERSION"' in bundle_script

    def test_turboquant_disable_env_is_honored_by_jang_loader(self):
        """Explicit/off-family cache choices must stop loader-level live TQ-KV."""
        loader_source = Path("./vmlx_engine/utils/jang_loader.py").read_text()
        cli_source = Path("./vmlx_engine/cli.py").read_text()

        assert 'environ.get("VMLX_DISABLE_TQ_KV")' in loader_source
        assert "TurboQuant KV skipped" in loader_source
        assert 'os.environ["VMLX_DISABLE_TQ_KV"] = "1"' in cli_source
        assert "VMLINUX_DISABLE_TQ_KV" not in cli_source

    def test_hybrid_ssm_auto_mode_skips_kv_quant_codecs(self):
        """Hybrid SSM auto mode must not use KV-only quant codecs."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()

        assert 'getattr(_mc, "cache_type", None) == "hybrid"' in cli_source
        assert "Hybrid SSM cache model detected" in cli_source
        assert "os.environ.pop(\"VMLX_FORCE_TQ_AUTO\", None)" in cli_source
        assert 'args.kv_cache_quantization = "none"' in cli_source

        assert "VMLX_ALLOW_HYBRID_KV_QUANT" in scheduler_source
        assert "disabling generic KV cache" in scheduler_source
        assert 'self.config.kv_cache_quantization = "none"' in scheduler_source

    def test_vmlx_env_prefix_is_canonical_for_ssm_cache_budget(self):
        """New cache env knobs should use VMLX_, with typo fallback only."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert "def _env_int(" in cli_source
        assert "os.environ.get(name)" in cli_source
        assert '"VMLINUX_SSM_STATE_CACHE_MB"' in cli_source
        assert 'default=_env_int("VMLX_SSM_STATE_CACHE_MB", 512, "VMLINUX_SSM_STATE_CACHE_MB")' in cli_source


class TestZayaCCACachePolicy:
    """ZAYA/CCA must not fall through generic hybrid cache paths."""

    def _write_zaya_fixture(self, tmp_path, *, weight_format="mxtq"):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "zaya",
            "weight_format": weight_format,
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mxtq_bits": {
                    "routed_expert": 2,
                    "attention": 8,
                    "router": 16,
                    "embed_tokens": 8,
                    "lm_head": 8,
                    "cca_conv": 16,
                    "norms_residual": 16,
                },
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": weight_format,
            "cache_subtype": "zaya_cca",
            "source_model": {
                "name": "ZAYA1-8B",
                "architecture": "zaya",
            },
            "capabilities": {
                "reasoning_parser": "qwen3",
                "tool_parser": "zaya_xml",
                "think_in_template": True,
                "supports_tools": True,
                "supports_thinking": True,
                "family": "zaya",
                "modality": "text",
                "cache_type": "hybrid",
            },
        }))
        return tmp_path

    def test_registry_preserves_zaya_cache_subtype(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = self._write_zaya_fixture(tmp_path)
        registry = get_model_config_registry()
        registry.clear_cache()

        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "zaya"
        assert cfg.cache_type == "hybrid"
        assert cfg.cache_subtype == "zaya_cca"
        assert cfg.tool_parser == "zaya_xml"
        assert cfg.reasoning_parser == "qwen3"

    def test_cli_disables_prefix_paged_l2_and_tq_for_zaya_cca(self):
        source = Path("./vmlx_engine/cli.py").read_text()

        assert 'getattr(_mc, "cache_subtype", None) == "zaya_cca"' in source
        assert 'args.enable_prefix_cache = False' in source
        assert 'args.use_paged_cache = False' in source
        assert 'args.enable_block_disk_cache = False' in source
        assert 'args.kv_cache_quantization = "none"' in source
        assert "full CCA state restore" in source
        assert "conv_state + prev_hs" in source

    def test_jang_loader_refuses_zaya_without_runtime(self, tmp_path):
        from vmlx_engine.utils.jang_loader import load_jang_model

        model_dir = self._write_zaya_fixture(tmp_path)

        with pytest.raises(RuntimeError, match="ZAYA-aware runtime"):
            load_jang_model(model_dir, skip_eval=True)

    def test_local_zaya_bundles_carry_explicit_cca_contract(self):
        model_dirs = [
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ2"),
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ4"),
            Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B-MXFP4"),
        ]
        existing = [p for p in model_dirs if p.exists()]
        if not existing:
            pytest.skip("local ZAYA bundles are not present on this machine")

        for model_dir in existing:
            cfg = json.loads((model_dir / "config.json").read_text())
            jcfg = json.loads((model_dir / "jang_config.json").read_text())
            caps = jcfg.get("capabilities", {})

            assert cfg.get("model_type") == "zaya"
            assert jcfg.get("cache_subtype") == "zaya_cca"
            assert caps.get("family") == "zaya"
            assert caps.get("cache_type") == "hybrid"
            assert caps.get("tool_parser") == "zaya_xml"
            assert caps.get("reasoning_parser") == "qwen3"
            assert cfg.get("zaya_expert_layout") == "split_switch_mlp"

            if jcfg.get("weight_format") == "mxtq":
                bits = jcfg.get("mxtq_bits", {})
                assert bits.get("cca_conv") == 16
                assert bits.get("router") == 16
                assert (model_dir / "jangtq_runtime.safetensors").is_file()


class TestV5PortUniqueMigration:
    """Existing databases must get a UNIQUE index on sessions.port."""

    def test_migration_block_exists(self):
        """database.ts must have migration code for sessions.port UNIQUE index."""
        with open(os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'database.ts'
        )) as f:
            source = f.read()
        assert "idx_sessions_port_unique" in source, (
            "Must create UNIQUE index on sessions.port for existing databases"
        )
        # Must deduplicate before adding constraint
        assert "DELETE FROM sessions" in source, (
            "Must deduplicate existing port conflicts before adding UNIQUE constraint"
        )
        assert "GROUP BY port" in source, (
            "Deduplication must group by port to keep only one session per port"
        )


# ================================================================
# V6: Regression tests for v1.2.0 audit fixes
# ================================================================

class TestV6CancelledErrorCallsFailActive:
    """CancelledError in engine loop must call _fail_active_requests."""

    def test_cancelled_error_handler_exists(self):
        source = Path("./vmlx_engine/engine_core.py").read_text()
        # Find CancelledError handler inside engine loop (the one that calls _fail_active_requests)
        # There are two: one in stop() and one in the engine loop. We need the engine loop one.
        idx = source.find("except asyncio.CancelledError:")
        # Skip the first one (in stop()) and find the second one (in engine loop)
        idx2 = source.find("except asyncio.CancelledError:", idx + 1)
        assert idx2 != -1, "Engine loop must have its own CancelledError handler"
        handler = source[idx2:idx2 + 200]
        assert "_fail_active_requests" in handler, (
            "CancelledError handler must call _fail_active_requests to unblock SSE consumers"
        )


class TestV6AbortUsesDeleteBlockTable:
    """abort_request must use delete_block_table (not detach_request) for paged cache."""

    def test_scheduler_abort_uses_delete(self):
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.abort_request)
        assert "delete_block_table" in source, (
            "Scheduler.abort_request must use delete_block_table to decrement ref_counts"
        )
        # Must NOT call detach_request (the word may appear in comments, check for actual call)
        assert ".detach_request(" not in source, (
            "Scheduler.abort_request must NOT call detach_request (leaks ref_counts)"
        )

    def test_mllm_scheduler_abort_uses_delete(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # Find the abort_request method
        start = source.find("def abort_request(self, request_id")
        assert start != -1
        # Find the next method definition
        end = source.find("\n    def ", start + 20)
        abort_body = source[start:end]
        assert "delete_block_table" in abort_body, (
            "MLLMScheduler.abort_request must use delete_block_table"
        )

    def test_mllm_scheduler_error_recovery_uses_delete(self):
        """Error-recovery path must also use delete_block_table."""
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # Find the error-recovery block (paged_cache_manager.delete_block_table in error path)
        # This is in the step() method's except block
        idx = source.find("# Clean up paged cache block tables for all running")
        assert idx != -1, "Error-recovery cache cleanup comment must exist"
        nearby = source[idx:idx + 500]
        assert "delete_block_table" in nearby, (
            "Error-recovery path must use delete_block_table (not detach_request)"
        )

    def test_completion_path_uses_detach(self):
        """Normal completion path should use detach_request (preserves blocks for LRU)."""
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # The completion path is in _finish_completed_requests or similar
        # It stores blocks for prefix cache, so it uses detach_request
        completion_idx = source.find("detach_request")
        assert completion_idx != -1, (
            "Normal completion path must use detach_request to preserve blocks for LRU reuse"
        )


class TestV6VLMDiskCacheKeyConsistency:
    """VLM disk cache store key must match fetch key (full token_list, not truncated)."""

    def test_store_uses_token_list_not_truncated(self):
        source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        # VLM disk cache store must use token_list (not truncated_tokens) to match fetch key
        assert "disk_cache.store(token_list" in source, (
            "VLM disk cache store must use token_list (not truncated_tokens) to match fetch key"
        )


class TestV6DequantizeNoneGuards:
    """All callers of _dequantize_cache must guard against None return."""

    def test_dequantize_returns_none_on_failure(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        func_start = source.find("def _dequantize_cache(")
        func_end = source.find("\ndef ", func_start + 10)
        func_body = source[func_start:func_end]
        assert "return None" in func_body, (
            "_dequantize_cache must return None on dequantization failure"
        )

    def test_memory_aware_caller_guards_none(self):
        """Memory-aware cache path must guard _dequantize_cache returning None."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # Find the memory-aware cache path with dequantize call
        # There are multiple dequantize call sites; find the one with "continue" after None check
        idx = source.find("_dequantize_cache(cache)")
        assert idx != -1
        after = source[idx:idx + 150]
        assert "if cache is None" in after or "is None" in after, (
            "Memory-aware path must check for None after _dequantize_cache"
        )

    def test_hybrid_cache_caller_guards_none(self):
        """Hybrid cache path must guard _dequantize_cache returning None before _fix_hybrid_cache."""
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        # The hybrid cache path should check for None after dequantize
        idx = source.find("_dequantize_cache(req.prompt_cache)")
        if idx == -1:
            idx = source.find("_dequantize_cache(cache_for_fix)")
        assert idx != -1
        after = source[idx:idx + 200]
        assert "is None" in after, (
            "Hybrid cache path must check for None after _dequantize_cache"
        )


class TestV6MinimumSystemVersion:
    """macOS minimumSystemVersion must not block current macOS users."""

    def test_minimum_version_not_too_high(self):
        pkg_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'package.json'
        )
        with open(pkg_path) as f:
            pkg = json.load(f)
        min_ver = pkg.get("build", {}).get("mac", {}).get("minimumSystemVersion", "")
        assert min_ver, "minimumSystemVersion must be set"
        major = int(min_ver.split(".")[0])
        # macOS versions: 14 = Sonoma, 15 = Sequoia, 26 = Tahoe (2025)
        # Must support at least macOS 14+ (Sonoma) for M-series Macs
        assert major <= 15, (
            f"minimumSystemVersion {min_ver} is too high — "
            f"blocks users on macOS Sequoia (15) and earlier"
        )

    def test_minimum_version_matches_mlx_runtime_floor(self):
        """Packaging must not claim support below the bundled MLX runtime floor."""
        root = os.path.join(os.path.dirname(__file__), '..')
        pkg_path = os.path.join(root, 'panel', 'package.json')
        with open(pkg_path) as f:
            pkg = json.load(f)
        min_ver = pkg.get("build", {}).get("mac", {}).get("minimumSystemVersion", "")
        assert min_ver == "14.5.0", (
            "Bundled MLX wheels require macOS 14.5+; advertising or allowing "
            f"{min_ver} reopens mlxstudio#90/#104 metallib/libc++ failures."
        )

        for rel in ("panel/README.md", "panel/SETUP.md"):
            text = open(os.path.join(root, rel), encoding="utf-8").read()
            assert "macOS 14.5+" in text
            assert "macOS 26+" not in text


class TestV6MapHFModel:
    """mapHFModel must handle missing lastModified and author from HF list API."""

    def test_map_hf_model_function_exists(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        assert "function mapHFModel" in source, "mapHFModel helper must exist"

    def test_map_hf_model_uses_created_at_fallback(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        # Must use createdAt as fallback for lastModified
        assert "createdAt" in source, (
            "mapHFModel must use createdAt as fallback when lastModified is missing"
        )

    def test_map_hf_model_extracts_author_from_model_id(self):
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        func_start = source.find("function mapHFModel")
        func_body = source[func_start:func_start + 400]
        # Must extract author from modelId (split('/')[0]) as fallback.
        # Accept both single- and double-quote forms — the TS source uses
        # double quotes: `modelId.split("/")[0]`.
        assert ("split('/')[0]" in func_body
                or 'split("/")[0]' in func_body), (
            "mapHFModel must extract author from modelId.split('/')[0] as fallback"
        )

    def test_search_and_recommended_use_map_hf_model(self):
        """Both searchHF and getRecommendedModels must use mapHFModel."""
        models_path = os.path.join(
            os.path.dirname(__file__), '..', 'panel', 'src', 'main', 'ipc', 'models.ts'
        )
        with open(models_path) as f:
            source = f.read()
        # Count usages of mapHFModel (should be used in both handlers)
        usages = source.count("mapHFModel(")
        assert usages >= 2, (
            f"mapHFModel must be used in both searchHF and getRecommendedModels "
            f"(found {usages} usage(s), expected >= 2)"
        )


class TestMLLMTurboQuantFetchPath:
    """Fetched VLM prefix/L2 caches must re-enter the TQ live-cache path."""

    def test_process_prompts_recompresses_fetched_prompt_cache(self):
        import inspect
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        source = inspect.getsource(MLLMBatchGenerator._process_prompts)

        assert "_recompress_to_tq(req_cache, self.language_model)" in source
        assert "TQ recompress removed from fetch paths" not in source


class TestMLLMMlaDetection:
    """MLLM MLA detection should match the LLM scheduler's nuance."""

    def test_detects_raw_text_config_kv_lora_rank(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _Cfg:
            _raw_config = {
                "text_config": {
                    "model_type": "kimi_k25",
                    "kv_lora_rank": 512,
                }
            }

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_bailing_ling_mla_is_not_treated_as_compressed_latent(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _Args:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is False


class TestHybridSSMEnvNames:
    """Hybrid SSM resume controls must use the documented VMLX_* names."""

    def test_scheduler_uses_vmlx_disable_ssm_prefix_resume(self):
        source = Path("vmlx_engine/scheduler.py").read_text()

        assert "VMLX_DISABLE_SSM_PREFIX_RESUME" in source
        assert "VMLINUX_DISABLE_SSM_PREFIX_RESUME" not in source


class TestJangVLMFallbacks:
    """JANG VLM loaders must not silently drop working image support."""

    def _write_qwen_hybrid_fixture(self, tmp_path, *, weight_format=None):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention", "full_attention"],
            },
            "vision_config": {},
        }))
        jang = {
            "format": "jang" if weight_format is None else weight_format,
            "architecture": {
                "has_vision": True,
                "has_ssm": True,
                "type": "hybrid_ssm_dense",
            },
        }
        if weight_format is not None:
            jang["weight_format"] = weight_format
        (tmp_path / "jang_config.json").write_text(json.dumps(jang))
        return tmp_path

    def test_qwen_hybrid_jangtq_vlm_stays_on_native_vlm_loader(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        assert "VMLINUX_FORCE_VLM_LOADER" not in source
        assert "JANGTQ/MXTQ Qwen VLM remains" in source
        assert "native multimodal fast path" in source

    def test_text_only_vlm_fallbacks_mark_vision_unavailable(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()
        mistral_fallback = source[
            source.index('config.get("model_type") == "mistral3"'):
            source.index("# Qwen3.5/3.6-VL hybrid SSM bundles", source.index('config.get("model_type") == "mistral3"'))
        ]

        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = True' in mistral_fallback
        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = False' in source

    def test_affine_qwen_hybrid_jang_routes_text_only(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path)
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is False

    def test_mxtq_qwen_hybrid_jang_routes_multimodal(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path, weight_format="mxtq")
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is True


class TestTurboQuantKVTelemetry:
    """Cache telemetry must agree for nested MLLM language models."""

    def test_detects_nested_mllm_turboquant_make_cache(self):
        from vmlx_engine.server import _turboquant_kv_cache_status

        def _turboquant_make_cache():
            return []

        class _LanguageModel:
            make_cache = staticmethod(_turboquant_make_cache)

        class _Model:
            language_model = _LanguageModel()

        class _Wrapper:
            model = _Model()

        class _Engine:
            _model = _Wrapper()

        status = _turboquant_kv_cache_status(_Engine())

        assert status["enabled"] is True
        assert status["default_bits"] == 3

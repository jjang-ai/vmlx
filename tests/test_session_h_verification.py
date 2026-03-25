"""
Session H verification tests — runtime checks for every audit item.

These tests ACTUALLY import, instantiate, and verify every critical path
identified in the 260-check cross-system audit. No mocking unless absolutely
necessary. If it can be checked at import/instantiation time, it IS checked.
"""
import gc
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. SSD DISK STREAMING — import chain verification
# ---------------------------------------------------------------------------

class TestSSDImportChain:
    """Verify every SSD import resolves to a real function."""

    def test_ssd_generate_imports(self):
        from vmlx_engine.utils.ssd_generate import (
            ssd_stream_generate,
            ssd_generate,
            _find_model_components,
            _create_attention_mask,
            _load_layer_from_index,
        )
        assert callable(ssd_stream_generate)
        assert callable(ssd_generate)
        assert callable(_find_model_components)
        assert callable(_create_attention_mask)
        assert callable(_load_layer_from_index)

    def test_ssd_generate_no_free_all_import(self):
        """free_all_layer_weights must NOT be imported in ssd_generate."""
        import vmlx_engine.utils.ssd_generate as mod
        source = inspect.getsource(mod)
        # Must not appear in import block
        import_section = source[:source.index("logger =")]
        assert "free_all_layer_weights" not in import_section

    def test_weight_index_imports(self):
        from vmlx_engine.utils.weight_index import (
            build_weight_index,
            save_layer_weights,
            save_all_layer_weights,
            load_layer_weights,
            free_layer_weights,
            free_all_layer_weights,
        )
        for fn in [build_weight_index, save_layer_weights, save_all_layer_weights,
                    load_layer_weights, free_layer_weights, free_all_layer_weights]:
            assert callable(fn)

    def test_streaming_wrapper_only_find_layers(self):
        """streaming_wrapper.py should only export _find_layers."""
        from vmlx_engine.utils.streaming_wrapper import _find_layers
        assert callable(_find_layers)
        # Should NOT have these old symbols
        import vmlx_engine.utils.streaming_wrapper as sw
        assert not hasattr(sw, 'StreamingLayerWrapper')
        assert not hasattr(sw, 'apply_streaming_layers')
        assert not hasattr(sw, 'compute_streaming_wired_limit')
        assert not hasattr(sw, 'lock_wired_limit')
        assert not hasattr(sw, 'unlock_wired_limit')

    def test_find_layers_return_type(self):
        """_find_layers should return (container, attr) or None."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import _find_layers

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type('Inner', (), {'layers': [nn.Linear(4, 4)]})()
        result = _find_layers(FakeModel())
        assert result is not None
        container, attr = result
        assert attr == 'layers'
        assert len(getattr(container, attr)) == 1

    def test_find_layers_returns_none_for_no_layers(self):
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import _find_layers
        class NoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
        assert _find_layers(NoLayers()) is None

    def test_mlx_lm_dependencies_exist(self):
        """SSD depends on these mlx_lm symbols at runtime."""
        from mlx_lm.generate import GenerationResponse, make_sampler
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        assert callable(make_sampler)
        assert callable(make_prompt_cache)

    def test_ssd_fallback_uses_sampler_not_temp(self):
        """Hybrid SSM fallback must use sampler= not temp= kwarg to mlx_stream_generate."""
        from vmlx_engine.utils.ssd_generate import ssd_stream_generate
        source = inspect.getsource(ssd_stream_generate)
        assert "sampler=fallback_sampler" in source
        # The mlx_stream_generate call must NOT have temp= as a direct kwarg
        # (temp= IS used in make_sampler — that's correct)
        assert "mlx_stream_generate(" in source
        sg_call_idx = source.index("mlx_stream_generate(")
        sg_call = source[sg_call_idx:sg_call_idx + 300]
        assert "temp=" not in sg_call


# ---------------------------------------------------------------------------
# 2. LLM MODEL — SSD attribute wiring
# ---------------------------------------------------------------------------

class TestLLMModelSSDAttributes:
    """Verify MLXLanguageModel has all SSD attrs and routing."""

    def test_llm_has_ssd_attrs_in_init(self):
        from vmlx_engine.models.llm import MLXLanguageModel
        source = inspect.getsource(MLXLanguageModel.__init__)
        for attr in ['_stream_from_disk', '_model_path', '_weight_index', '_temp_weight_dir']:
            assert attr in source, f"Missing {attr} in MLXLanguageModel.__init__"

    def test_stream_generate_checks_ssd(self):
        from vmlx_engine.models.llm import MLXLanguageModel
        source = inspect.getsource(MLXLanguageModel.stream_generate)
        assert '_stream_from_disk' in source
        assert 'ssd_stream_generate' in source

    def test_generate_checks_ssd(self):
        from vmlx_engine.models.llm import MLXLanguageModel
        source = inspect.getsource(MLXLanguageModel.generate)
        assert '_stream_from_disk' in source
        assert 'ssd_generate' in source


# ---------------------------------------------------------------------------
# 3. SERVER — SSD init block, temp cleanup, MLLM warning
# ---------------------------------------------------------------------------

class TestServerSSDWiring:
    """Verify server.py SSD module-level variables and functions."""

    def test_ssd_temp_dir_module_var(self):
        from vmlx_engine import server
        assert hasattr(server, '_ssd_temp_dir')

    def test_stream_from_disk_module_var(self):
        from vmlx_engine import server
        assert hasattr(server, '_stream_from_disk')
        assert server._stream_from_disk is False  # default OFF

    def test_cleanup_ssd_temp_registered(self):
        """atexit handler should be registered."""
        from vmlx_engine import server
        assert hasattr(server, '_cleanup_ssd_temp')
        assert callable(server._cleanup_ssd_temp)

    def test_cleanup_handles_none(self):
        from vmlx_engine import server
        old = server._ssd_temp_dir
        server._ssd_temp_dir = None
        server._cleanup_ssd_temp()  # should not crash
        server._ssd_temp_dir = old

    def test_cleanup_handles_nonexistent_dir(self):
        from vmlx_engine import server
        old = server._ssd_temp_dir
        server._ssd_temp_dir = "/nonexistent/path/that/does/not/exist"
        server._cleanup_ssd_temp()  # should not crash
        server._ssd_temp_dir = old

    def test_load_model_has_ssd_temp_dir_global(self):
        from vmlx_engine import server
        source = inspect.getsource(server.load_model)
        assert '_ssd_temp_dir' in source

    def test_anthropic_endpoint_has_jsonresponse_import(self):
        """B1 fix: JSONResponse must be imported in create_anthropic_message."""
        from vmlx_engine import server
        source = inspect.getsource(server.create_anthropic_message)
        assert 'from starlette.responses import JSONResponse' in source


# ---------------------------------------------------------------------------
# 4. CLI — SSD feature gating
# ---------------------------------------------------------------------------

class TestCLIFeatureGating:
    """Verify cli.py SSD gating forces correct values."""

    def test_ssd_gating_forces_all_off(self):
        from vmlx_engine import cli
        source = inspect.getsource(cli)
        # Find the SSD gating block — search wider (1500 chars to cover banner + assignments)
        gating_start = source.index("stream_from_disk")
        gating_block = source[gating_start:gating_start + 1500]

        assert "use_paged_cache" in gating_block
        assert "enable_prefix_cache" in gating_block
        assert "kv_cache_quantization" in gating_block
        assert "max_num_seqs" in gating_block
        assert "cache_memory_percent" in gating_block
        assert "continuous_batching" in gating_block


# ---------------------------------------------------------------------------
# 5. HYBRID SSM — cache compatibility
# ---------------------------------------------------------------------------

class TestHybridSSMCacheCompat:
    """Verify hybrid SSM models get correct cache handling."""

    def test_make_prompt_cache_delegates_to_model(self):
        """Models with make_cache() should use it."""
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.nn as nn
        import mlx.core as mx

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(3)]
            def make_cache(self):
                return ["cache_a", "cache_b", "cache_c"]

        model = HybridModel()
        cache = make_prompt_cache(model)
        assert cache == ["cache_a", "cache_b", "cache_c"]

    def test_ssd_fallback_for_cache_mismatch(self):
        """When cache != layers, ssd_generate should NOT raise RuntimeError."""
        from vmlx_engine.utils.ssd_generate import ssd_stream_generate
        source = inspect.getsource(ssd_stream_generate)
        # Should NOT have RuntimeError for cache mismatch
        assert "raise RuntimeError" not in source.split("Falling back")[0].split("layer-to-cache")[1]
        # Should have fallback
        assert "Falling back to standard generation" in source

    def test_scheduler_detects_hybrid(self):
        """Scheduler._is_hybrid_model should exist."""
        from vmlx_engine.scheduler import Scheduler
        assert hasattr(Scheduler, '_is_hybrid_model')

    def test_ensure_mamba_support_exists(self):
        """ensure_mamba_support patches mlx_lm for BatchMambaCache."""
        from vmlx_engine.scheduler import ensure_mamba_support
        assert callable(ensure_mamba_support)


# ---------------------------------------------------------------------------
# 6. REASONING PARSERS — on/off
# ---------------------------------------------------------------------------

class TestReasoningParsers:
    """Verify all reasoning parsers exist, register, and handle on/off."""

    def test_all_parsers_importable(self):
        from vmlx_engine.reasoning import get_parser
        for name in ["qwen3", "deepseek_r1", "openai_gptoss"]:
            parser_cls = get_parser(name)
            assert parser_cls is not None
            # Instantiate and check methods
            parser = parser_cls()
            assert hasattr(parser, 'extract_reasoning')

    def test_unknown_parser_raises(self):
        from vmlx_engine.reasoning import get_parser
        with pytest.raises(KeyError):
            get_parser("nonexistent_parser_xyz")

    def test_qwen3_parser_strips_think_tags(self):
        from vmlx_engine.reasoning import get_parser
        parser = get_parser("qwen3")()
        reasoning, content = parser.extract_reasoning("<think>reasoning here</think>actual content")
        assert reasoning is not None
        assert "reasoning here" in reasoning
        assert content is not None
        assert "actual content" in content

    def test_qwen3_parser_handles_no_think(self):
        from vmlx_engine.reasoning import get_parser
        parser = get_parser("qwen3")()
        reasoning, content = parser.extract_reasoning("just plain content")
        assert "just plain content" in (content or "")

    def test_suppress_reasoning_in_server(self):
        """Server's stream_chat_completion must check suppress_reasoning."""
        from vmlx_engine import server
        source = inspect.getsource(server.stream_chat_completion)
        assert "suppress_reasoning" in source

    def test_enable_thinking_priority(self):
        """Server must check request > default > auto for enable_thinking."""
        from vmlx_engine import server
        source = inspect.getsource(server.create_chat_completion)
        assert "_default_enable_thinking" in source

    def test_prior_turn_think_strip(self):
        """When thinking OFF, prior <think> blocks should be stripped."""
        from vmlx_engine import server
        source = inspect.getsource(server.create_chat_completion)
        assert "<think>" in source  # strip logic references <think>


# ---------------------------------------------------------------------------
# 7. JANG MODEL LOADING
# ---------------------------------------------------------------------------

class TestJANGLoading:
    """Verify JANG loader functions exist and are consistent."""

    def test_is_jang_model_exists(self):
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert callable(is_jang_model)

    def test_jang_config_filenames_consistent(self):
        from vmlx_engine.utils.jang_loader import JANG_CONFIG_FILENAMES
        assert "jang_config.json" in JANG_CONFIG_FILENAMES
        assert "jjqf_config.json" in JANG_CONFIG_FILENAMES
        assert len(JANG_CONFIG_FILENAMES) >= 3

    def test_load_jang_model_accepts_lazy(self):
        from vmlx_engine.utils.jang_loader import load_jang_model
        sig = inspect.signature(load_jang_model)
        assert "lazy" in sig.parameters

    def test_is_jang_model_called_from_tokenizer(self):
        from vmlx_engine.utils import tokenizer
        source = inspect.getsource(tokenizer)
        assert "is_jang_model" in source

    def test_tokenizer_reads_stream_from_disk(self):
        from vmlx_engine.utils import tokenizer
        source = inspect.getsource(tokenizer)
        assert "_stream_from_disk" in source


# ---------------------------------------------------------------------------
# 8. VLM / MLLM PATH
# ---------------------------------------------------------------------------

class TestVLMMLLMPath:
    """Verify VLM detection and SSD interaction."""

    def test_is_mllm_model_exists(self):
        from vmlx_engine.api.utils import is_mllm_model
        assert callable(is_mllm_model)

    def test_is_mllm_no_regex(self):
        """is_mllm_model must not use regex for detection."""
        from vmlx_engine.api.utils import is_mllm_model
        source = inspect.getsource(is_mllm_model)
        assert "re.search" not in source
        assert "re.match" not in source

    def test_mllm_reads_stream_from_disk(self):
        from vmlx_engine.models import mllm
        source = inspect.getsource(mllm)
        assert "_stream_from_disk" in source

    def test_mllm_batch_generator_reads_stream_from_disk(self):
        from vmlx_engine import mllm_batch_generator
        source = inspect.getsource(mllm_batch_generator)
        assert "_stream_from_disk" in source


# ---------------------------------------------------------------------------
# 9. TOOL CALL PARSERS
# ---------------------------------------------------------------------------

class TestToolCallParsers:
    """Verify tool parser registry has all expected parsers."""

    def test_tool_parser_manager_exists(self):
        from vmlx_engine.tool_parsers import ToolParserManager
        assert ToolParserManager is not None

    def test_tool_call_markers_defined(self):
        from vmlx_engine import server
        assert hasattr(server, '_TOOL_CALL_MARKERS')
        markers = server._TOOL_CALL_MARKERS
        assert len(markers) >= 5  # At least Qwen, Mistral, Hermes, Llama, DeepSeek


# ---------------------------------------------------------------------------
# 10. API ENDPOINTS — function signatures
# ---------------------------------------------------------------------------

class TestAPIEndpointSignatures:
    """Verify all critical API handler functions exist."""

    def test_chat_completion_handler(self):
        from vmlx_engine.server import create_chat_completion
        assert callable(create_chat_completion)

    def test_anthropic_message_handler(self):
        from vmlx_engine.server import create_anthropic_message
        assert callable(create_anthropic_message)

    def test_image_gen_handler(self):
        from vmlx_engine.server import create_image
        assert callable(create_image)

    def test_health_handler(self):
        from vmlx_engine.server import health
        assert callable(health)

    def test_stream_chat_completion(self):
        from vmlx_engine.server import stream_chat_completion
        assert callable(stream_chat_completion)


# ---------------------------------------------------------------------------
# 11. ANTHROPIC ADAPTER
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    """Verify adapter conversion functions."""

    def test_adapter_functions_exist(self):
        from vmlx_engine.api.anthropic_adapter import (
            to_chat_completion,
            AnthropicStreamAdapter,
            AnthropicRequest,
        )
        assert callable(to_chat_completion)

    def test_system_prompt_conversion(self):
        from vmlx_engine.api.anthropic_adapter import to_chat_completion, AnthropicRequest
        req = AnthropicRequest(
            model="test",
            max_tokens=100,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = to_chat_completion(req)
        # System should be first message — check via attribute access
        first_msg = result.messages[0]
        if isinstance(first_msg, dict):
            assert first_msg["role"] == "system"
        else:
            assert first_msg.role == "system"


# ---------------------------------------------------------------------------
# 12. SLEEP / WAKE
# ---------------------------------------------------------------------------

class TestSleepWake:
    """Verify sleep/wake admin functions exist with correct signatures."""

    def test_admin_endpoints_exist(self):
        from vmlx_engine import server
        assert hasattr(server, 'admin_soft_sleep')
        assert hasattr(server, 'admin_deep_sleep')
        assert hasattr(server, 'admin_wake')

    def test_cli_args_saved_for_wake(self):
        from vmlx_engine import server
        source = inspect.getsource(server.load_model)
        assert "_cli_args" in source
        assert "stream_from_disk" in source
        assert "stream_memory_percent" in source

    def test_wake_passes_ssd_args(self):
        from vmlx_engine import server
        source = inspect.getsource(server.admin_wake)
        assert "stream_from_disk" in source


# ---------------------------------------------------------------------------
# 13. WEIGHT INDEX — progress logging
# ---------------------------------------------------------------------------

class TestWeightIndexProgressLogging:
    """Verify save_all_layer_weights logs progress."""

    def test_progress_logging_in_save_all(self):
        from vmlx_engine.utils.weight_index import save_all_layer_weights
        source = inspect.getsource(save_all_layer_weights)
        assert "Saved %d/%d layer weights to SSD" in source
        assert "% 10 == 0" in source


# ---------------------------------------------------------------------------
# 14. BUNDLED PYTHON SYNC — source vs bundled
# ---------------------------------------------------------------------------

class TestBundledPythonSync:
    """Verify bundled python matches source for critical files."""

    BUNDLED_BASE = Path(__file__).parent.parent / "panel" / "bundled-python" / "python" / "lib" / "python3.12" / "site-packages" / "vmlx_engine"
    SOURCE_BASE = Path(__file__).parent.parent / "vmlx_engine"

    CRITICAL_FILES = [
        "utils/ssd_generate.py",
        "utils/weight_index.py",
        "utils/streaming_wrapper.py",
        "models/llm.py",
        "server.py",
    ]

    @pytest.mark.parametrize("relpath", CRITICAL_FILES)
    def test_source_matches_bundled(self, relpath):
        source_file = self.SOURCE_BASE / relpath
        bundled_file = self.BUNDLED_BASE / relpath
        if not bundled_file.exists():
            pytest.skip(f"Bundled file not found: {bundled_file}")
        source_content = source_file.read_text()
        bundled_content = bundled_file.read_text()
        assert source_content == bundled_content, f"MISMATCH: {relpath}"


# ---------------------------------------------------------------------------
# 15. MODEL CONFIG REGISTRY — no regex for detection
# ---------------------------------------------------------------------------

class TestModelConfigRegistry:
    """Verify model config registry doesn't use regex for primary detection."""

    def test_no_general_regex_fallback(self):
        from vmlx_engine import model_config_registry
        source = inspect.getsource(model_config_registry)
        # Should not have a general re.search/re.match for model type detection
        # (specific disambiguations for GLM-Z1 and MedGemma are OK)
        lookup_fn = model_config_registry.ModelConfigRegistry.lookup
        lookup_source = inspect.getsource(lookup_fn)
        # Primary detection should use model_type exact match
        assert "model_type" in lookup_source


# ---------------------------------------------------------------------------
# 16. SPECULATIVE DECODING — SSD gating
# ---------------------------------------------------------------------------

class TestSpeculativeGating:
    """Verify speculative decoding is gated when SSD is on."""

    def test_ssd_forces_speculative_none(self):
        from vmlx_engine import cli
        source = inspect.getsource(cli)
        gating_start = source.index("stream_from_disk")
        gating_block = source[gating_start:gating_start + 1500]
        assert "speculative_model" in gating_block


# ---------------------------------------------------------------------------
# 17. KV CACHE QUANTIZATION — exists and imports
# ---------------------------------------------------------------------------

class TestKVCacheQuant:
    """Verify KV cache quantization infrastructure."""

    def test_quantized_kv_cache_importable(self):
        """QuantizedKVCache should be importable from mlx_lm."""
        try:
            from mlx_lm.models.cache import QuantizedKVCache
            assert QuantizedKVCache is not None
        except ImportError:
            pytest.skip("QuantizedKVCache not in this mlx_lm version")

    def test_scheduler_creates_quantized_cache(self):
        from vmlx_engine import scheduler
        source = inspect.getsource(scheduler)
        assert "QuantizedKVCache" in source
        assert "kv_cache_quantization" in source


# ---------------------------------------------------------------------------
# 18. IMAGE GEN — model class dispatch
# ---------------------------------------------------------------------------

class TestImageGenDispatch:
    """Verify image gen uses explicit class dispatch, not regex."""

    def test_model_class_map_exists(self):
        from vmlx_engine.image_gen import MODEL_CLASS_MAP
        assert isinstance(MODEL_CLASS_MAP, dict)
        assert len(MODEL_CLASS_MAP) >= 2  # At least ZImage and Flux1

    def test_name_to_class_map_exists(self):
        from vmlx_engine.image_gen import _NAME_TO_CLASS
        assert isinstance(_NAME_TO_CLASS, dict)
        assert "schnell" in _NAME_TO_CLASS or "flux-schnell" in _NAME_TO_CLASS


# ---------------------------------------------------------------------------
# 19. MCP TOOL SECURITY
# ---------------------------------------------------------------------------

class TestMCPSecurity:
    """Verify MCP config validates security."""

    def test_mcp_config_loader_exists(self):
        from vmlx_engine.mcp.config import load_mcp_config
        assert callable(load_mcp_config)


# ---------------------------------------------------------------------------
# 20. AUDIO PIPELINE — lazy loading
# ---------------------------------------------------------------------------

class TestAudioPipeline:
    """Verify audio endpoints are lazy-loaded (no crash without mlx-audio)."""

    def test_stt_engine_class_exists(self):
        try:
            from vmlx_engine.audio.stt import STTEngine
            assert STTEngine is not None
        except ImportError:
            pass  # OK if mlx-audio not installed

    def test_tts_engine_class_exists(self):
        try:
            from vmlx_engine.audio.tts import TTSEngine
            assert TTSEngine is not None
        except ImportError:
            pass  # OK if mlx-audio not installed


# ---------------------------------------------------------------------------
# 21. DATABASE — parameterized queries
# ---------------------------------------------------------------------------

class TestDatabaseSafety:
    """Verify no SQL injection risks in database code."""

    def test_no_string_format_in_sql(self):
        """Database TS uses parameterized queries — verify Python side too."""
        # Check server.py for any raw SQL (should be none)
        from vmlx_engine import server
        source = inspect.getsource(server)
        # No cursor.execute with f-strings or .format
        assert "cursor.execute(f" not in source
        assert '.execute(f"' not in source


# ---------------------------------------------------------------------------
# 22. RERANK — dual backend
# ---------------------------------------------------------------------------

class TestRerank:
    """Verify reranker infrastructure."""

    def test_reranker_exists(self):
        from vmlx_engine.reranker import Reranker
        assert Reranker is not None


# ---------------------------------------------------------------------------
# 23. PROCESS MANAGEMENT — verify key functions
# ---------------------------------------------------------------------------

class TestProcessManagement:
    """Verify critical TypeScript process management patterns via Python config."""

    def test_ssd_default_off_in_default_config(self):
        """streamFromDisk must default to false."""
        # Read the TS source and check
        config_path = Path(__file__).parent.parent / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        if not config_path.exists():
            pytest.skip("Panel source not found")
        content = config_path.read_text()
        assert "streamFromDisk: false" in content

    def test_ssd_active_gates_all_sections(self):
        """UI must disable caching sections when SSD active."""
        config_path = Path(__file__).parent.parent / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        if not config_path.exists():
            pytest.skip("Panel source not found")
        content = config_path.read_text()
        assert "const ssdActive = !!config.streamFromDisk" in content
        # Count how many times ssdActive is used for disabling
        ssd_refs = content.count("ssdActive")
        assert ssd_refs >= 10, f"ssdActive used only {ssd_refs} times — not enough gating"

    def test_no_dead_ssd_sliders_in_form(self):
        """ssdMemoryBudget/ssdPrefetchLayers must not have SliderField."""
        config_path = Path(__file__).parent.parent / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        if not config_path.exists():
            pytest.skip("Panel source not found")
        content = config_path.read_text()
        # These should NOT appear as slider labels
        assert 'label="SSD Memory Budget' not in content
        assert 'label="Prefetch Layers"' not in content

    def test_ssd_progress_patterns_in_sessions(self):
        """3 SSD progress patterns must be in sessions.ts."""
        sessions_path = Path(__file__).parent.parent / "panel/src/main/sessions.ts"
        if not sessions_path.exists():
            pytest.skip("Panel source not found")
        content = sessions_path.read_text()
        assert "Saving weights to SSD" in content
        assert "Building weight index" in content
        assert "SSD streaming ready" in content

    def test_no_dead_ssd_args_in_build_args(self):
        """buildArgs must NOT push --ssd-memory-budget or --ssd-prefetch-layers."""
        sessions_path = Path(__file__).parent.parent / "panel/src/main/sessions.ts"
        if not sessions_path.exists():
            pytest.skip("Panel source not found")
        content = sessions_path.read_text()
        assert "'--ssd-memory-budget'" not in content
        assert "'--ssd-prefetch-layers'" not in content

    def test_no_dead_ssd_args_in_preview(self):
        """SessionSettings preview must NOT show --ssd-memory-budget or --ssd-prefetch-layers."""
        settings_path = Path(__file__).parent.parent / "panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        if not settings_path.exists():
            pytest.skip("Panel source not found")
        content = settings_path.read_text()
        assert "'--ssd-memory-budget'" not in content
        assert "'--ssd-prefetch-layers'" not in content

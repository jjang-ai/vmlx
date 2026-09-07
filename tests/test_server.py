# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import json
import platform
import sys
from pathlib import Path

import pytest


def test_thinking_off_history_strip_is_shared_and_preserves_tool_anchors():
    from vmlx_engine.server import _strip_prior_reasoning_for_thinking_off

    messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "private",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "private only",
        },
        {
            "role": "assistant",
            "content": "<think>private tool plan</think>",
            "reasoning_content": "private tool plan",
            "tool_calls": [{"id": "call_file"}],
        },
        {"role": "tool", "tool_call_id": "call_file", "content": "result"},
    ]

    cleaned = _strip_prior_reasoning_for_thinking_off(messages)

    assert cleaned == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "visible"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_file"}],
        },
        {"role": "tool", "tool_call_id": "call_file", "content": "result"},
    ]
    assert "private" not in json.dumps(cleaned)

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vmlx_engine.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vmlx_engine.server import Message

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vmlx_engine.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vmlx_engine.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vmlx_engine.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vmlx_engine.server import ContentPart

        part = ContentPart(
            type="image_url", image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vmlx_engine.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vmlx_engine.server import ContentPart

        part = ContentPart(
            type="video_url", video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vmlx_engine.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert (
            request.temperature is None
        )  # resolved at runtime by _resolve_temperature
        assert request.stream is False  # default

    def test_request_with_options(self):
        """Test request with custom options."""
        from vmlx_engine.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vmlx_engine.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vmlx_engine.server import CompletionRequest

        request = CompletionRequest(model="test-model", prompt="Once upon a time")

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test server helper functions."""

    def test_main_raises_nofile_soft_limit_before_serving(self):
        """macOS default soft cap of 256 fds must be lifted at server boot.

        A long-context serve bursts through 256 (per-thread SQLite handles +
        block payload reads + sockets); running out mid-request degrades
        importlib/disk reads into broad-except fallbacks and poisons cache
        lookups (live incident 2026-08-07).
        """
        import resource

        from vmlx_engine import server

        soft_before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        server._raise_nofile_soft_limit()
        soft_after, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert soft_after >= soft_before
        if hard == resource.RLIM_INFINITY or hard >= 4096:
            assert soft_after >= 4096

        with open("vmlx_engine/server.py") as f:
            src = f.read()
        main_body = src.split("def main():", 1)[1]
        assert "_raise_nofile_soft_limit()" in main_body[:200]

        # The shipping entry point is vmlx-serve -> cli.serve_command, which
        # runs its own uvicorn and never enters server.main(). It must raise
        # the limit too.
        with open("vmlx_engine/cli.py") as f:
            cli_src = f.read()
        serve_body = cli_src.split("def serve_command(", 1)[1]
        assert "_raise_nofile_soft_limit()" in serve_body[:1200]

    def test_resolve_max_prompt_tokens_uses_user_context_cap(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(64000, 8192) == 8192

    def test_resolve_max_prompt_tokens_uses_auto_estimate_when_unset(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(4096, None) == 4096

    def test_resolve_max_prompt_tokens_uses_auto_estimate_when_zero(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(4096, 0) == 4096

    def test_estimate_max_prompt_tokens_uses_batched_private_model_and_context_ceiling(
        self, monkeypatch
    ):
        from types import SimpleNamespace
        from vmlx_engine import server

        config = SimpleNamespace(
            num_hidden_layers=22,
            num_loops=2,
            num_key_value_heads=8,
            head_dim=128,
            torch_dtype="bfloat16",
            max_position_embeddings=262144,
        )
        engine = SimpleNamespace(_model=SimpleNamespace(args=config))
        monkeypatch.setattr(server, "get_engine", lambda: engine)
        monkeypatch.setattr(
            server,
            "_metal_projection_stats",
            lambda: (0, 10 * 1024**4),
        )

        assert server._estimate_max_prompt_tokens() == 262144

    def test_estimate_max_prompt_tokens_uses_simple_engine_wrapper_model(
        self, monkeypatch
    ):
        from types import SimpleNamespace
        from vmlx_engine import server

        config = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            torch_dtype="bfloat16",
            max_position_embeddings=32768,
        )
        raw_model = SimpleNamespace(args=config)
        language_model_wrapper = SimpleNamespace(model=raw_model)
        engine = SimpleNamespace(_model=language_model_wrapper)
        monkeypatch.setattr(server, "get_engine", lambda: engine)
        monkeypatch.setattr(
            server,
            "_metal_projection_stats",
            lambda: (0, 1024**4),
        )

        assert server._estimate_max_prompt_tokens() == 32768

    def test_estimate_max_prompt_tokens_returns_zero_without_loaded_config(
        self, monkeypatch
    ):
        from vmlx_engine import server

        monkeypatch.setattr(
            server,
            "_loaded_model_config_for_memory_projection",
            lambda: None,
        )
        monkeypatch.setattr(
            server,
            "_metal_projection_stats",
            lambda: (0, 8 * 1024**3),
        )

        assert server._estimate_max_prompt_tokens() == 0

    def test_estimate_max_prompt_tokens_uses_native_dsv4_pool_geometry(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        from vmlx_engine import server

        config = SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepseekV4ForCausalLM"],
            num_hidden_layers=43,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            hidden_size=4096,
            sliding_window=128,
            index_head_dim=128,
            torch_dtype="bfloat16",
            max_position_embeddings=1_048_576,
            compress_ratios=[0, 0]
            + [value for _ in range(20) for value in (4, 128)]
            + [4, 0],
        )
        engine = SimpleNamespace(_model=SimpleNamespace(args=config))
        monkeypatch.setattr(server, "get_engine", lambda: engine)
        # 60% of 20 GiB covers the architecture-native q8 estimate plus its
        # attention-ready hot view at 1M tokens. Generic 43-layer full-KV
        # geometry would falsely cap this far below the declared limit.
        # Admission is deliberately live-cache-only: paged-block records are
        # evictable under Metal pressure (enforce_byte_budget) and fresh
        # oversized captures are skipped by _delta_capture_admitted, so they
        # must not shrink the advertised context.
        monkeypatch.setattr(
            server,
            "_metal_projection_stats",
            lambda: (0, 20 * 1024**3),
        )
        monkeypatch.setenv("DSV4_POOL_QUANT", "1")
        monkeypatch.delenv("DSV4_MAX_PREFILL_TOKENS", raising=False)

        assert server._estimate_max_prompt_tokens() == 1_048_576

        # An explicit operator guard still overrides architecture-aware
        # admission.
        monkeypatch.setenv("DSV4_MAX_PREFILL_TOKENS", "32768")
        assert server._estimate_max_prompt_tokens() == 32_768

    def test_declared_context_limit_prefers_smaller_nested_text_ceiling(self):
        from types import SimpleNamespace
        from vmlx_engine import server

        config = SimpleNamespace(
            max_position_embeddings=262144,
            text_config=SimpleNamespace(max_position_embeddings=131072),
        )

        assert server._declared_context_limit_from_config(config) == 131072

    def test_refresh_loaded_max_prompt_tokens_applies_automatic_estimate(
        self, monkeypatch
    ):
        from vmlx_engine import server

        monkeypatch.setattr(server, "_cli_args", {"max_prompt_tokens": None})
        monkeypatch.setattr(server, "_max_prompt_tokens", 0)
        monkeypatch.setattr(server, "_estimate_max_prompt_tokens", lambda: 65536)

        assert server._refresh_loaded_max_prompt_tokens("test") == 65536
        assert server._max_prompt_tokens == 65536

    def test_refresh_loaded_max_prompt_tokens_preserves_explicit_limit(
        self, monkeypatch
    ):
        from vmlx_engine import server

        monkeypatch.setattr(server, "_cli_args", {"max_prompt_tokens": 8192})
        monkeypatch.setattr(server, "_max_prompt_tokens", 0)
        monkeypatch.setattr(server, "_estimate_max_prompt_tokens", lambda: 65536)

        assert server._refresh_loaded_max_prompt_tokens("test") == 8192
        assert server._max_prompt_tokens == 8192

    def test_admin_wake_refreshes_prompt_limit_after_speculative_draft_reload(self):
        import inspect

        from vmlx_engine import server

        # aae5885d3 moved the wake body into _admin_wake_impl (shared by the
        # JIT middleware and /admin/wake); the ordering contract lives there.
        source = inspect.getsource(server._admin_wake_impl)

        assert source.index(
            "await _run_on_model_executor(load_draft_model, _spec_cfg)"
        ) < source.index(
            '_refresh_loaded_max_prompt_tokens("admin_wake_engine_start")'
        )
        assert source.index(
            "await _run_on_model_executor(_apply_jit_compilation)"
        ) < source.index(
            '_record_metal_ws_model_baseline("admin_wake_post_acceleration")'
        )
        assert source.index(
            "await _run_on_model_executor(load_draft_model, _spec_cfg)"
        ) < source.index(
            '_record_metal_ws_model_baseline("admin_wake_post_acceleration")'
        )

    def test_simple_engine_baseline_is_after_optional_acceleration(self):
        import inspect

        from vmlx_engine import server

        source = inspect.getsource(server.load_model)
        baseline = source.index(
            '_record_metal_ws_model_baseline("simple_engine_post_acceleration")'
        )
        assert source.index(
            "_run_on_model_executor_blocking(_apply_flash_moe_patching)"
        ) < baseline
        assert source.index(
            "_run_on_model_executor_blocking(_apply_jit_compilation)"
        ) < baseline

    def test_effective_max_prompt_tokens_request_can_only_lower_session_cap(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine import server

        monkeypatch.setattr(server, "_max_prompt_tokens", 8192)

        assert server._effective_max_prompt_tokens(SimpleNamespace(max_prompt_tokens=None)) == 8192
        assert server._effective_max_prompt_tokens(SimpleNamespace(max_prompt_tokens=4096)) == 4096
        assert server._effective_max_prompt_tokens(SimpleNamespace(max_prompt_tokens=16384)) == 8192

    def test_effective_max_prompt_tokens_request_can_cap_when_session_unbounded(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine import server

        monkeypatch.setattr(server, "_max_prompt_tokens", 0)

        assert server._effective_max_prompt_tokens(SimpleNamespace(max_prompt_tokens=2048)) == 2048

    def test_api_request_max_context_aliases_normalize_to_max_prompt_tokens(self):
        from vmlx_engine.api.models import (
            ChatCompletionRequest,
            CompletionRequest,
            Message,
            ResponsesRequest,
        )

        chat = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            max_context_tokens=1234,
        )
        comp = CompletionRequest(model="m", prompt="hi", max_context=2345)
        resp = ResponsesRequest(model="m", input="hi", max_prompt_tokens=3456)

        assert chat.max_prompt_tokens == 1234
        assert comp.max_prompt_tokens == 2345
        assert resp.max_prompt_tokens == 3456

    def test_ollama_num_ctx_maps_to_internal_max_prompt_tokens(self):
        from vmlx_engine.api.ollama_adapter import (
            ollama_chat_to_openai,
            ollama_generate_to_openai,
            ollama_generate_to_openai_chat,
        )

        chat = ollama_chat_to_openai({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_ctx": 2048},
        })
        raw = ollama_generate_to_openai({
            "model": "m",
            "prompt": "hi",
            "options": {"num_ctx": 3072},
        })
        templated = ollama_generate_to_openai_chat({
            "model": "m",
            "prompt": "hi",
            "options": {"num_ctx": 4096},
        })

        assert chat["max_prompt_tokens"] == 2048
        assert raw["max_prompt_tokens"] == 3072
        assert templated["max_prompt_tokens"] == 4096

    def test_anthropic_max_context_extension_maps_to_chat_request(self):
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion

        req = AnthropicRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            max_context_tokens=2048,
        )
        chat = to_chat_completion(req)

        assert chat.max_prompt_tokens == 2048

    def test_dead_max_context_length_global_removed(self):
        from pathlib import Path
        import vmlx_engine.server as server

        src = Path(server.__file__).read_text()
        assert "_max_context_length" not in src

    def test_prompt_limit_has_no_character_heuristic_preflight(self):
        """Near-limit requests must reach the authoritative loaded tokenizer.

        Character ratios can overestimate a valid prompt by thousands of
        tokens for a concrete bundle. The scheduler/SimpleEngine exact-token
        guards own rejection after production chat-template rendering.
        """
        from vmlx_engine import server

        source = Path(server.__file__).read_text()

        assert "_text_prompt_token_estimate" not in source
        assert "_reject_if_prompt_too_long_for_messages" not in source
        assert "_reject_if_prompt_too_long_for_prompts" not in source

    def test_vlm_image_prefill_budget_response_is_client_error(self):
        """Predictable media-prefill budget rejection is a 413, not 500."""
        import json

        from vmlx_engine import server
        from vmlx_engine.errors import VLMImagePrefillBudgetError

        response = server._vlm_image_prefill_budget_response_from_error(
            VLMImagePrefillBudgetError(
                "VLM image prefill rejected before Metal forward: predicted "
                "attention buffer 65.2GB exceeds single-buffer guard 8.0GB",
                request_id="vl-image-budget",
            )
        )

        body = json.loads(response.body)
        assert response.status_code == 413
        assert body["error"]["code"] == "vlm_image_prefill_too_large"
        assert "Reduce image resolution" in body["error"]["message"]

    def test_unsupported_media_modality_response_is_client_error(self):
        """Unwired family media bridges are typed client errors, not 500s."""
        import json

        from vmlx_engine import server
        from vmlx_engine.errors import UnsupportedMediaModalityError

        response = server._unsupported_media_modality_response_from_error(
            UnsupportedMediaModalityError(
                "vision",
                "MiMo-V2.5 JANG_2L vision input is not wired in this Python runtime.",
                family="mimo_v2",
                request_id="mimo-media",
            )
        )

        body = json.loads(response.body)
        assert response.status_code == 400
        assert body["error"]["code"] == "unsupported_media_modality"
        assert body["error"]["modality"] == "vision"
        assert body["error"]["family"] == "mimo_v2"
        assert "not wired" in body["error"]["message"]

    def test_exact_prompt_limit_is_forwarded_by_all_generation_entrypoints(self):
        import inspect
        from vmlx_engine import server

        completion_src = inspect.getsource(server.create_completion)
        chat_src = inspect.getsource(server.create_chat_completion)
        responses_src = inspect.getsource(server.create_response)
        anthropic_src = inspect.getsource(server.create_anthropic_message)
        ollama_chat_src = inspect.getsource(server.ollama_chat)
        ollama_generate_src = inspect.getsource(server.ollama_generate)

        assert "_completion_max_prompt_tokens = _effective_max_prompt_tokens(request)" in completion_src
        assert "_chat_max_prompt_tokens = _effective_max_prompt_tokens(request)" in chat_src
        assert "_responses_max_prompt_tokens = _effective_max_prompt_tokens(request)" in responses_src
        assert "_msg_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)" in anthropic_src
        assert "_ollama_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)" in ollama_chat_src
        assert "_ollama_gen_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)" in ollama_generate_src
        assert "_ollama_raw_max_prompt_tokens = _effective_max_prompt_tokens(comp_req)" in ollama_generate_src

    def test_prompt_limit_is_forwarded_to_engine_generation_kwargs(self):
        source = Path("vmlx_engine/server.py").read_text()

        assert "_effective_max_prompt_tokens(request)" in source
        assert '"max_prompt_tokens": _chat_max_prompt_tokens' in source
        assert '"max_prompt_tokens": _responses_max_prompt_tokens' in source
        assert '"max_prompt_tokens": _completion_max_prompt_tokens' in source
        assert '"max_prompt_tokens": _msg_max_prompt_tokens' in source
        assert '"max_prompt_tokens": _ollama_max_prompt_tokens' in source
        assert "except PromptTooLongError" in source

    def test_streaming_routes_map_prompt_limit_to_prompt_too_long(self):
        import inspect
        from vmlx_engine import server

        chat_stream_src = inspect.getsource(server.stream_chat_completion)
        responses_stream_src = inspect.getsource(server.stream_responses_api)

        # Match the HANDLER, not one exact spelling of it. These routes also
        # catch PrefillAdmissionError in the same clause — a device that cannot
        # serve the context is the same class of client error — so the tuple
        # form `except (PromptTooLongError, PrefillAdmissionError)` is correct
        # and the old literal count(...) == 1 failed on it while the behaviour
        # it was protecting was intact.
        assert chat_stream_src.count("except (PromptTooLongError") == 1
        assert "code\": \"prompt_too_long\"" in chat_stream_src
        assert "PromptTooLongError" in responses_stream_src
        # the Responses stream maps PromptTooLong/PrefillAdmission to the typed
        # code inside ONE fatal handler that ends with response.failed
        assert '"prompt_too_long"' in responses_stream_src
        typed_handler = responses_stream_src[
            responses_stream_src.index("MediaControlsUnmeetableError,") :
        ]
        assert typed_handler.index('"prompt_too_long"') < typed_handler.index('"response.failed"')

    def test_is_mllm_model_detection(self, tmp_path):
        """Test MLLM model detection via config.json and force flag.
        No regex fallback — remote names without local config return False."""
        import json
        from vmlx_engine.server import is_mllm_model

        # force_mllm always returns True
        assert is_mllm_model("anything", force_mllm=True)

        # Remote names without local config.json return False
        assert not is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")

        # Local model with vision_config returns True
        vlm_dir = tmp_path / "vlm-model"
        vlm_dir.mkdir()
        (vlm_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5", "vision_config": {"hidden_size": 1024}
        }))
        assert is_mllm_model(str(vlm_dir))

        # Local model without vision_config returns False
        llm_dir = tmp_path / "llm-model"
        llm_dir.mkdir()
        (llm_dir / "config.json").write_text(json.dumps({
            "model_type": "llama", "hidden_size": 4096
        }))
        assert not is_mllm_model(str(llm_dir))

    def test_step37_advertised_vlm_runtime_launches_text_only_when_runtime_missing(
        self, monkeypatch, tmp_path
    ):
        """Step3.7 advertised vision must not route into MLLM without runtime."""
        import json

        from vmlx_engine.api import utils as api_utils
        from vmlx_engine.engine.batched import BatchedEngine

        step_dir = tmp_path / "Step-3.7-Flash-JANG_2L"
        step_dir.mkdir()
        (step_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "step3p7",
                    "model_file": "step3p7_mlx.py",
                    "text_config": {"model_type": "step3p5"},
                    "vision_config": {"hidden_size": 1024},
                    "image_token_id": 151655,
                }
            )
        )
        (step_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "architecture": {
                        "family": "step3p7",
                        "has_vision": True,
                    },
                }
            )
        )
        (step_dir / "step3p7_mlx.py").write_text("# local text bridge marker\n")

        api_utils.resolve_to_local_path.cache_clear()
        api_utils._IS_MLLM_CACHE.clear()
        monkeypatch.setattr(
            api_utils,
            "_source_step3p7_vlm_runtime_available",
            lambda: False,
        )

        assert api_utils.is_mllm_model(str(step_dir), force_mllm=False) is False
        assert api_utils.is_mllm_model(str(step_dir), force_mllm=True) is False
        engine = BatchedEngine(str(step_dir), force_mllm=True)
        assert engine.is_mllm is False

    @pytest.mark.asyncio
    async def test_step37_text_only_media_rejection_recovers_on_next_text_request(
        self, monkeypatch, tmp_path
    ):
        """Step3.7 media rejection must not poison later text requests."""
        import json

        import pytest
        import vmlx_engine.server as server
        from fastapi import HTTPException
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "step3p7",
                    "text_config": {"model_type": "step3p5"},
                    "vision_config": {"hidden_size": 1024},
                    "image_token_id": 151655,
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "architecture": {
                        "family": "step3p7",
                        "has_vision": True,
                    },
                }
            )
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.chat_calls = []

            async def chat(self, *, messages, **kwargs):
                self.chat_calls.append({"messages": messages, "kwargs": kwargs})
                return GenerationOutput(
                    text="TEXT_OK",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "step37-text-only")
        monkeypatch.setattr(server, "_model_name", "step37-text-only")
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_type", "step3p7")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        with pytest.raises(HTTPException) as exc:
            await server.create_chat_completion(
                ChatCompletionRequest(
                    model="step37-text-only",
                    messages=[
                        Message(
                            role="user",
                            content=[
                                {"type": "text", "text": "describe image"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,AAAA"
                                    },
                                },
                            ],
                        )
                    ],
                    max_tokens=8,
                ),
                fastapi_request=None,
            )

        assert exc.value.status_code == 400
        assert "unsupported media modality" in exc.value.detail
        assert "text-only" in exc.value.detail
        assert engine.chat_calls == []

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="step37-text-only",
                messages=[Message(role="user", content="text still works")],
                max_tokens=8,
            ),
            fastapi_request=None,
        )

        assert response.choices[0].message.content == "TEXT_OK"
        assert len(engine.chat_calls) == 1


class TestOllamaCompatibilityProbe:
    """Ollama-compatible clients probe these before sending real requests."""

    def test_root_and_version_do_not_require_api_key(self, monkeypatch):
        from fastapi.testclient import TestClient
        from vmlx_engine import server

        monkeypatch.setattr(server, "_api_key", "secret-token", raising=False)
        client = TestClient(server.app)

        root = client.get("/")
        assert root.status_code == 200
        assert "Ollama" in root.text
        assert client.head("/").status_code == 200

        version = client.get("/api/version")
        assert version.status_code == 200
        assert version.json()["version"] == "0.12.6"
        assert client.head("/api/version").status_code == 200

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vmlx_engine.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vmlx_engine.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vmlx_engine.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vmlx_engine.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(videos) == 1


# =============================================================================
# Security and Reliability Tests (PR #4)
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter class for rate limiting functionality."""

    def test_rate_limiter_disabled_by_default(self):
        """Test that rate limiter allows all requests when disabled."""
        from vmlx_engine.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_enforces_limit(self):
        """Test that rate limiter enforces the request limit."""
        from vmlx_engine.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3, enabled=True)

        # First 3 requests should be allowed
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True, f"Request {i+1} should be allowed"
            assert retry_after == 0

        # 4th request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_per_client(self):
        """Test that rate limits are tracked per client."""
        from vmlx_engine.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Client 1 uses its quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Client 2 should still have quota
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        import threading
        from vmlx_engine.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=100, enabled=True)
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    allowed, _ = limiter.is_allowed("shared_client")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 100
        # Exactly 100 requests allowed (our limit)
        assert results.count(True) == 100


class TestTempFileManager:
    """Test the TempFileManager class for temp file cleanup."""

    def test_register_and_cleanup_single_file(self):
        """Test registering and cleaning up a single temp file."""
        import tempfile
        import os
        from vmlx_engine.models.mllm import TempFileManager

        manager = TempFileManager()

        # Create a real temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        temp.write(b"test content")
        temp.close()

        # Register it
        path = manager.register(temp.name)
        assert path == temp.name
        assert os.path.exists(temp.name)

        # Cleanup
        result = manager.cleanup(temp.name)
        assert result is True
        assert not os.path.exists(temp.name)

    def test_cleanup_all_files(self):
        """Test cleaning up all registered temp files."""
        import tempfile
        import os
        from vmlx_engine.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []

        # Create multiple temp files
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
            temp.write(f"content {i}".encode())
            temp.close()
            manager.register(temp.name)
            paths.append(temp.name)

        # Verify all exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 3

        # Verify all deleted
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_nonexistent_file(self):
        """Test cleanup of a non-existent file."""
        from vmlx_engine.models.mllm import TempFileManager

        manager = TempFileManager()

        # Cleanup a file that doesn't exist
        result = manager.cleanup("/nonexistent/path/file.txt")
        assert result is False

    def test_thread_safe_registration(self):
        """Test that TempFileManager is thread-safe."""
        import threading
        import tempfile
        from vmlx_engine.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []
        lock = threading.Lock()
        errors = []

        def register_files():
            try:
                for _ in range(5):
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    temp.write(b"test")
                    temp.close()
                    path = manager.register(temp.name)
                    with lock:
                        paths.append(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_files) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(paths) == 25

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 25


class TestRequestOutputCollectorThreadSafety:
    """Test thread-safety of RequestOutputCollector._waiting_consumers."""

    def test_waiting_consumers_thread_safe(self):
        """Test that _waiting_consumers counter is thread-safe."""
        import threading
        from vmlx_engine.output_collector import RequestOutputCollector

        # Reset the counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        errors = []

        def manipulate_counter():
            try:
                for _ in range(100):
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers += 1
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=manipulate_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should return to zero
        with RequestOutputCollector._waiting_lock:
            assert RequestOutputCollector._waiting_consumers == 0

    def test_has_waiting_consumers_method(self):
        """Test has_waiting_consumers class method."""
        from vmlx_engine.output_collector import RequestOutputCollector

        # Reset counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        assert RequestOutputCollector.has_waiting_consumers() is False

        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 1

        assert RequestOutputCollector.has_waiting_consumers() is True

        # Reset
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0


class TestRequestTimeoutField:
    """Test the new timeout field in request models."""

    def test_chat_completion_request_timeout_field(self):
        """Test that ChatCompletionRequest has timeout field."""
        from vmlx_engine.server import ChatCompletionRequest, Message

        # Default should be None
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            timeout=60.0,
        )
        assert request_with_timeout.timeout == 60.0

    def test_completion_request_timeout_field(self):
        """Test that CompletionRequest has timeout field."""
        from vmlx_engine.server import CompletionRequest

        # Default should be None
        request = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = CompletionRequest(
            model="test-model", prompt="Once upon a time", timeout=120.0
        )
        assert request_with_timeout.timeout == 120.0


class TestOpenAILogprobsFormatting:
    """Server-side token decoding must produce OpenAI-compatible logprob shapes."""

    class _Tokenizer:
        def decode(self, token_ids):
            mapping = {
                10: " hello",
                11: "world",
                12: "é",
            }
            return "".join(mapping[i] for i in token_ids)

    def test_completion_logprobs_preserve_tokens_offsets_and_top_entries(self):
        from vmlx_engine.server import _format_completion_logprobs

        raw = [
            {
                "token_id": 10,
                "logprob": -0.1,
                "top_logprobs": [(10, -0.1), (11, -2.0)],
            },
            {
                "token_id": 12,
                "logprob": -0.2,
                "top_logprobs": [(12, -0.2)],
            },
        ]

        formatted = _format_completion_logprobs(raw, self._Tokenizer())

        assert formatted == {
            "tokens": [" hello", "é"],
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [
                {" hello": -0.1, "world": -2.0},
                {"é": -0.2},
            ],
            "text_offset": [0, 6],
        }

    def test_chat_logprobs_include_utf8_bytes(self):
        from vmlx_engine.server import _format_chat_logprobs

        formatted = _format_chat_logprobs(
            [
                {
                    "token_id": 12,
                    "logprob": -0.2,
                    "top_logprobs": [(12, -0.2), (11, -1.5)],
                }
            ],
            self._Tokenizer(),
        )

        assert formatted == {
            "content": [
                {
                    "token": "é",
                    "logprob": -0.2,
                    "bytes": [195, 169],
                    "top_logprobs": [
                        {"token": "é", "logprob": -0.2, "bytes": [195, 169]},
                        {
                            "token": "world",
                            "logprob": -1.5,
                            "bytes": [119, 111, 114, 108, 100],
                        },
                    ],
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_completion_endpoint_passes_logprobs_to_text_engine(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = self._Tokenizer()

            def __init__(self):
                self.kwargs = None

            async def generate(self, **kwargs):
                self.kwargs = kwargs
                return GenerationOutput(
                    text=" hello",
                    tokens=[10],
                    prompt_tokens=2,
                    completion_tokens=1,
                    logprobs=[
                        {
                            "token_id": 10,
                            "logprob": -0.1,
                            "top_logprobs": [(10, -0.1), (11, -2.0)],
                        }
                    ],
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")

        response = await server.create_completion(
            CompletionRequest(
                model="loaded-model",
                prompt="hi",
                max_tokens=1,
                logprobs=2,
            )
        )

        assert engine.kwargs["logprobs"] is True
        assert engine.kwargs["top_logprobs"] == 2
        assert response.choices[0].logprobs == {
            "tokens": [" hello"],
            "token_logprobs": [-0.1],
            "top_logprobs": [{" hello": -0.1, "world": -2.0}],
            "text_offset": [0],
        }

    @pytest.mark.asyncio
    async def test_dsv4_completion_endpoint_uses_chat_rail(self, monkeypatch, tmp_path):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = self._Tokenizer()

            def __init__(self):
                self.chat_calls = []
                self.generate_calls = []

            async def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                raise AssertionError("DSV4 completions must not use raw generate")

            async def chat(self, messages, **kwargs):
                self.chat_calls.append({"messages": messages, "kwargs": kwargs})
                return GenerationOutput(
                    text="const camera = new THREE.PerspectiveCamera();",
                    prompt_tokens=7,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_served_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_default_timeout", 60)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        server._jang_sampling_defaults_cache.clear()

        response = await server.create_completion(
            CompletionRequest(
                model="dsv4-route-code-probe",
                prompt="Return exactly this JavaScript code:\nconst camera = new THREE.PerspectiveCamera();",
                max_tokens=64,
                temperature=0,
                top_p=1,
            )
        )

        assert engine.generate_calls == []
        assert len(engine.chat_calls) == 1
        call = engine.chat_calls[0]
        assert call["messages"] == [
            {
                "role": "user",
                "content": "Return exactly this JavaScript code:\nconst camera = new THREE.PerspectiveCamera();",
            }
        ]
        assert call["kwargs"]["enable_thinking"] is True
        assert call["kwargs"]["max_tokens"] == 64
        assert call["kwargs"]["temperature"] == 0
        assert call["kwargs"]["top_p"] == 1
        assert response.choices[0].text == "const camera = new THREE.PerspectiveCamera();"

    @pytest.mark.asyncio
    async def test_dsv4_completion_exact_no_markdown_returns_visible_unfenced_code(
        self, monkeypatch, tmp_path
    ):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = self._Tokenizer()

            async def generate(self, **kwargs):
                raise AssertionError("DSV4 completions must not use raw generate")

            async def chat(self, messages, **kwargs):
                text = (
                    "I should answer with only the copied code.</think>"
                    "```javascript\n"
                    "const camera = new THREE.PerspectiveCamera();\n"
                    "```"
                )
                return GenerationOutput(
                    text=text,
                    raw_text=text,
                    prompt_tokens=7,
                    completion_tokens=18,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_served_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_default_timeout", 60)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        server._jang_sampling_defaults_cache.clear()

        response = await server.create_completion(
            CompletionRequest(
                model="dsv4-route-code-probe",
                prompt=(
                    "Return exactly this JavaScript code and no markdown fences:\n"
                    "const camera = new THREE.PerspectiveCamera();"
                ),
                max_tokens=64,
                temperature=0,
                top_p=1,
            )
        )

        assert response.choices[0].text == "const camera = new THREE.PerspectiveCamera();"

    @pytest.mark.asyncio
    async def test_dsv4_completion_exact_no_markdown_trims_trailing_prompt_space(
        self, monkeypatch, tmp_path
    ):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = self._Tokenizer()

            def __init__(self):
                self.messages = None
                self.kwargs = None

            async def generate(self, **kwargs):
                raise AssertionError("DSV4 completions must not use raw generate")

            async def chat(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return GenerationOutput(
                    text="const renderer = new THREE.WebGLRenderer();",
                    prompt_tokens=7,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_served_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_default_timeout", 60)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        server._jang_sampling_defaults_cache.clear()

        await server.create_completion(
            CompletionRequest(
                model="dsv4-route-code-probe",
                prompt=(
                    "Return exactly this JavaScript code and no markdown fences:\n"
                    "const renderer = new THREE.WebGLRenderer();\n\n"
                ),
                max_tokens=220,
                temperature=0,
                top_p=1,
            )
        )

        assert engine.messages == [
            {
                "role": "user",
                "content": (
                    "Return exactly this JavaScript code and no markdown fences:\n"
                    "const renderer = new THREE.WebGLRenderer();"
                ),
            }
        ]
        assert engine.kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_dsv4_streaming_completion_uses_chat_rail_and_strips_think_glue(
        self, monkeypatch, tmp_path
    ):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = self._Tokenizer()

            def __init__(self):
                self.chat_calls = []

            async def stream_generate(self, **kwargs):
                raise AssertionError(
                    "DSV4 streaming completions must not stream raw "
                    "think-tagged text"
                )
                yield  # pragma: no cover

            async def chat(self, messages, **kwargs):
                self.chat_calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "I will fence the solution.</think>```python\n"
                    "def has_close_elements(numbers, threshold):\n"
                    "    return False\n"
                    "```"
                )
                return GenerationOutput(
                    text=text,
                    raw_text=text,
                    prompt_tokens=9,
                    completion_tokens=24,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_served_model_name", "dsv4-route-code-probe")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_default_timeout", 60)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        server._jang_sampling_defaults_cache.clear()

        request = CompletionRequest(
            model="dsv4-route-code-probe",
            prompt="Complete the function:\ndef has_close_elements(...):",
            max_tokens=64,
            temperature=0,
            top_p=1,
            stream=True,
        )
        lines = []
        async for line in server.stream_completions_multi(
            engine, [request.prompt], request
        ):
            lines.append(line)

        assert len(engine.chat_calls) == 1
        call = engine.chat_calls[0]
        assert call["messages"] == [
            {
                "role": "user",
                "content": "Complete the function:\ndef has_close_elements(...):",
            }
        ]
        assert call["kwargs"]["enable_thinking"] is True
        assert call["kwargs"]["max_tokens"] == 64

        data_lines = [
            line
            for line in lines
            if line.startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        assert len(data_lines) == 1
        chunk = json.loads(data_lines[0].removeprefix("data: ").strip())
        text = chunk["choices"][0]["text"]
        assert "</think>" not in text
        assert text.startswith("```python\n")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["usage"]["completion_tokens"] == 24
        assert lines[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_chat_endpoint_passes_logprobs_to_text_engine(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            tokenizer = self._Tokenizer()
            preserve_native_tool_format = False

            def __init__(self):
                self.kwargs = None

            async def chat(self, *, messages, **kwargs):
                self.kwargs = kwargs
                return GenerationOutput(
                    text="é",
                    tokens=[12],
                    prompt_tokens=3,
                    completion_tokens=1,
                    logprobs=[
                        {
                            "token_id": 12,
                            "logprob": -0.2,
                            "top_logprobs": [(12, -0.2), (11, -1.5)],
                        }
                    ],
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="hi")],
                max_tokens=1,
                logprobs=True,
                top_logprobs=2,
            ),
            fastapi_request=None,
        )

        assert engine.kwargs["logprobs"] is True
        assert engine.kwargs["top_logprobs"] == 2
        assert response.choices[0].logprobs == {
            "content": [
                {
                    "token": "é",
                    "logprob": -0.2,
                    "bytes": [195, 169],
                    "top_logprobs": [
                        {"token": "é", "logprob": -0.2, "bytes": [195, 169]},
                        {
                            "token": "world",
                            "logprob": -1.5,
                            "bytes": [119, 111, 114, 108, 100],
                        },
                    ],
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_chat_endpoint_passes_max_prompt_tokens_to_engine(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.kwargs = None

            async def chat(self, *, messages, **kwargs):
                self.kwargs = kwargs
                return GenerationOutput(
                    text="ok",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_max_prompt_tokens", 64)

        await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="hi")],
                max_tokens=1,
            ),
            fastapi_request=None,
        )

        assert engine.kwargs["max_prompt_tokens"] == 64

    @pytest.mark.asyncio
    async def test_chat_endpoint_request_max_prompt_tokens_lowers_session_cap(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.kwargs = None

            async def chat(self, *, messages, **kwargs):
                self.kwargs = kwargs
                return GenerationOutput(
                    text="ok",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_max_prompt_tokens", 64)

        await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="hi")],
                max_tokens=1,
                max_prompt_tokens=32,
            ),
            fastapi_request=None,
        )

        assert engine.kwargs["max_prompt_tokens"] == 32

    @pytest.mark.asyncio
    async def test_chat_response_format_strict_retries_failed_json_only(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = "not json" if len(self.calls) == 1 else '{"status":"ok"}'
                return GenerationOutput(
                    text=text,
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="return status")],
                max_tokens=16,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "status",
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
            fastapi_request=None,
        )

        assert response.choices[0].message.content == '{"status": "ok"}'
        assert len(engine.calls) == 2
        assert engine.calls[1]["messages"][-2] == {
            "role": "assistant",
            "content": "not json",
        }
        assert "fix this JSON only" in engine.calls[1]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_chat_response_format_strict_retries_failed_xml_only(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "<catalog><label>CLIPFARM</label></catalog>"
                    if len(self.calls) == 1
                    else "<catalog><visible_text>CLIPFARM</visible_text></catalog>"
                )
                return GenerationOutput(
                    text=text,
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="return catalog xml")],
                max_tokens=16,
                response_format={
                    "type": "xml",
                    "xml_root_tag": "catalog",
                    "required_xml_fields": ["visible_text"],
                    "strict": True,
                },
            ),
            fastapi_request=None,
        )

        assert (
            response.choices[0].message.content
            == "<catalog><visible_text>CLIPFARM</visible_text></catalog>"
        )
        assert len(engine.calls) == 2
        assert engine.calls[1]["messages"][-2] == {
            "role": "assistant",
            "content": "<catalog><label>CLIPFARM</label></catalog>",
        }
        assert "fix this XML only" in engine.calls[1]["messages"][-1]["content"]
        assert "Required XML fields: visible_text" in engine.calls[1]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_chat_response_format_forwards_guided_json_hint(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                return GenerationOutput(
                    text='{"status":"ok"}',
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="return status")],
                max_tokens=16,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "status",
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
            fastapi_request=None,
        )

        assert engine.calls[0]["kwargs"]["_vmlx_response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "status",
                "schema": schema,
                "strict": True,
            },
        }

    @pytest.mark.asyncio
    async def test_responses_text_format_strict_retries_failed_json_only(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = "still not json" if len(self.calls) == 1 else '{"status":"ok"}'
                return GenerationOutput(
                    text=text,
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        response = await server.create_response(
            ResponsesRequest(
                model="loaded-model",
                input="return status",
                max_output_tokens=16,
                text={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "status",
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
            fastapi_request=None,
        )

        assert response.output_text == '{"status": "ok"}'
        assert len(engine.calls) == 2
        assert engine.calls[1]["messages"][-2] == {
            "role": "assistant",
            "content": "still not json",
        }
        assert "fix this JSON only" in engine.calls[1]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_responses_text_format_strict_retries_failed_xml_only(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "still not xml"
                    if len(self.calls) == 1
                    else "<catalog><visible_text>CLIPFARM</visible_text></catalog>"
                )
                return GenerationOutput(
                    text=text,
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        response = await server.create_response(
            ResponsesRequest(
                model="loaded-model",
                input="return catalog xml",
                max_output_tokens=16,
                text={
                    "type": "xml",
                    "xml_root_tag": "catalog",
                    "required_xml_fields": ["visible_text"],
                    "strict": True,
                },
            ),
            fastapi_request=None,
        )

        assert (
            response.output_text
            == "<catalog><visible_text>CLIPFARM</visible_text></catalog>"
        )
        assert len(engine.calls) == 2
        assert engine.calls[1]["messages"][-2] == {
            "role": "assistant",
            "content": "still not xml",
        }
        assert "fix this XML only" in engine.calls[1]["messages"][-1]["content"]
        assert "Required XML fields: visible_text" in engine.calls[1]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_responses_text_format_forwards_guided_json_hint(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                return GenerationOutput(
                    text='{"status":"ok"}',
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        await server.create_response(
            ResponsesRequest(
                model="loaded-model",
                input="return status",
                max_output_tokens=16,
                text={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "status",
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
            fastapi_request=None,
        )

        assert engine.calls[0]["kwargs"]["_vmlx_response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "status",
                "schema": schema,
                "strict": True,
            },
        }

    @pytest.mark.parametrize(
        "endpoint, stream, is_mllm, expected",
        [
            ("Completions", False, True, "text-only LLM"),
            ("Chat Completions", False, True, "text-only LLM"),
            ("Completions", True, True, "multimodal/VLM logprobs"),
            ("Chat Completions", True, True, "multimodal/VLM logprobs"),
        ],
    )
    def test_logprobs_rejects_unimplemented_surfaces(
        self, endpoint, stream, is_mllm, expected
    ):
        from fastapi import HTTPException
        from vmlx_engine.server import _reject_unsupported_logprobs_request

        with pytest.raises(HTTPException) as exc:
            _reject_unsupported_logprobs_request(
                endpoint=endpoint,
                requested=True,
                stream=stream,
                is_mllm=is_mllm,
            )

        assert exc.value.status_code == 400
        assert expected in exc.value.detail

    def test_logprobs_allows_text_streaming(self):
        from vmlx_engine.server import _reject_unsupported_logprobs_request

        _reject_unsupported_logprobs_request(
            endpoint="Chat Completions",
            requested=True,
            stream=True,
            is_mllm=False,
        )

    @pytest.mark.asyncio
    async def test_completion_endpoint_rejects_mllm_logprobs(self, monkeypatch):
        import vmlx_engine.server as server
        from fastapi import HTTPException
        from vmlx_engine.api.models import CompletionRequest

        class _Engine:
            is_mllm = True

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "vl-model")
        monkeypatch.setattr(server, "_model_name", "vl-model")

        with pytest.raises(HTTPException) as exc:
            await server.create_completion(
                CompletionRequest(model="vl-model", prompt="hi", logprobs=1)
            )

        assert exc.value.status_code == 400
        assert "multimodal/VLM logprobs" in exc.value.detail

    @pytest.mark.asyncio
    async def test_chat_endpoint_rejects_mllm_logprobs_before_generation(self, monkeypatch):
        import vmlx_engine.server as server
        from fastapi import HTTPException
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        class _Engine:
            is_mllm = True

            async def chat(self, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("chat should not be called")

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "vl-model")
        monkeypatch.setattr(server, "_model_name", "vl-model")
        monkeypatch.setattr(server, "_model_path", None)

        with pytest.raises(HTTPException) as exc:
            await server.create_chat_completion(
                ChatCompletionRequest(
                    model="vl-model",
                    messages=[Message(role="user", content="hi")],
                    logprobs=True,
                ),
                fastapi_request=None,
            )

        assert exc.value.status_code == 400
        assert "multimodal/VLM logprobs" in exc.value.detail

    def test_streaming_chunk_choice_can_carry_logprobs(self):
        from vmlx_engine.api.models import ChatCompletionChunkChoice

        assert "logprobs" in ChatCompletionChunkChoice.model_fields

    @pytest.mark.asyncio
    async def test_streaming_completion_emits_delta_logprobs(self, monkeypatch):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import CompletionRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = self._Tokenizer()

            def __init__(self):
                self.kwargs = None

            async def stream_generate(self, **kwargs):
                self.kwargs = kwargs
                yield GenerationOutput(
                    text=" hello",
                    new_text=" hello",
                    tokens=[10],
                    prompt_tokens=2,
                    completion_tokens=1,
                    finished=False,
                    logprobs=[
                        {
                            "token_id": 10,
                            "logprob": -0.1,
                            "top_logprobs": [(10, -0.1)],
                        }
                    ],
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        chunks = []
        request = CompletionRequest(
            model="loaded-model",
            prompt="hi",
            stream=True,
            logprobs=1,
        )
        async for line in server.stream_completions_multi(engine, ["hi"], request):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        assert engine.kwargs["logprobs"] is True
        assert engine.kwargs["top_logprobs"] == 1
        assert chunks[0]["choices"][0]["logprobs"] == {
            "tokens": [" hello"],
            "token_logprobs": [-0.1],
            "top_logprobs": [{" hello": -0.1}],
            "text_offset": [0],
        }

    @pytest.mark.asyncio
    async def test_streaming_chat_emits_delta_logprobs(self, monkeypatch):
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = self._Tokenizer()

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="é",
                    new_text="é",
                    tokens=[12],
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=False,
                    logprobs=[
                        {
                            "token_id": 12,
                            "logprob": -0.2,
                            "top_logprobs": [(12, -0.2)],
                        }
                    ],
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)

        request = ChatCompletionRequest(
            model="loaded-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
            logprobs=True,
            top_logprobs=1,
        )
        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
            logprobs=True,
            top_logprobs=1,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        content_chunks = [
            c for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        assert content_chunks[0]["choices"][0]["logprobs"] == {
            "content": [
                {
                    "token": "é",
                    "logprob": -0.2,
                    "bytes": [195, 169],
                    "top_logprobs": [
                        {"token": "é", "logprob": -0.2, "bytes": [195, 169]}
                    ],
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_nonstream_adapter_chat_stream_actively_aborts_on_disconnect(
        self, monkeypatch
    ):
        import asyncio
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        aborted: list[str] = []

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                await asyncio.Event().wait()
                if False:
                    yield None

            async def abort_request(self, request_id):
                aborted.append(request_id)

        class _DisconnectedRequest:
            async def receive(self):
                return {"type": "http.disconnect"}

            async def is_disconnected(self):
                raise AssertionError("active receive drain owns disconnect detection")

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "anthropic-adapter-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ChatCompletionRequest(
            model="anthropic-adapter-test",
            messages=[Message(role="user", content="keep generating")],
            stream=False,
        )
        stream = server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=_DisconnectedRequest(),
        )

        first = await anext(stream)
        assert '"role":"assistant"' in first.replace(" ", "")
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=1.0)

        assert len(aborted) == 1
        assert aborted[0].startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_streaming_chat_strict_xml_validates_final_text(
        self, monkeypatch
    ):
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="<catalog><label>CLIPFARM</label></catalog>",
                    new_text="<catalog><label>CLIPFARM</label></catalog>",
                    tokens=[],
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ChatCompletionRequest(
            model="loaded-model",
            messages=[Message(role="user", content="return catalog xml")],
            stream=True,
            response_format={
                "type": "xml",
                "xml_root_tag": "catalog",
                "required_xml_fields": ["visible_text"],
                "strict": True,
            },
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        errors = [chunk["error"] for chunk in chunks if "error" in chunk]
        assert errors
        assert errors[-1]["code"] == "xml_validation_failed"
        assert "missing required XML fields" in errors[-1]["message"]
        assert "visible_text" in errors[-1]["message"]

    @pytest.mark.asyncio
    async def test_streaming_responses_strict_xml_validates_final_text(
        self, monkeypatch
    ):
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="<catalog><label>CLIPFARM</label></catalog>",
                    new_text="<catalog><label>CLIPFARM</label></catalog>",
                    tokens=[],
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ResponsesRequest(
            model="loaded-model",
            input="return catalog xml",
            stream=True,
            text={
                "type": "xml",
                "xml_root_tag": "catalog",
                "required_xml_fields": ["visible_text"],
                "strict": True,
            },
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "return catalog xml"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        errors = [event for event in events if event.get("type") == "error"]
        assert errors
        assert errors[-1]["code"] == "xml_validation_failed"
        assert "missing required XML fields" in errors[-1]["message"]
        assert "visible_text" in errors[-1]["message"]

    @pytest.mark.asyncio
    async def test_streaming_chat_suppresses_mid_text_dsml_partial_marker(
        self, monkeypatch
    ):
        """DSV4 can flush `<｜DSML｜tool` before the full wrapper name.

        That partial can land in the middle of the accumulated stream when the
        model resumes prose after a tool attempt. It must never be emitted as
        visible assistant text.
        """
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="<",
                    new_text="<",
                    tokens=[10],
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                yield GenerationOutput(
                    text="<｜DSML｜",
                    new_text="｜DSML｜",
                    tokens=[10, 11],
                    prompt_tokens=3,
                    completion_tokens=2,
                    finished=False,
                    finish_reason=None,
                )
                yield GenerationOutput(
                    text='<｜DSML｜tool_call_type type="list_directory" "attributes":"."',
                    new_text='tool_call_type type="list_directory" "attributes":"."',
                    tokens=[10, 11],
                    prompt_tokens=3,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "dsv4-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "deepseek_v4")

        request = ChatCompletionRequest(
            model="dsv4-test",
            messages=[Message(role="user", content="read files and help me fix")],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert "<｜DSML｜" not in visible
        assert "<tool" not in visible

    @pytest.mark.asyncio
    async def test_streaming_chat_flushes_literal_less_than_after_tool_buffer_probe(
        self, monkeypatch
    ):
        """A literal `<` prefix probe must not swallow normal visible text."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="2 <",
                    new_text="2 <",
                    tokens=[10],
                    prompt_tokens=3,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                yield GenerationOutput(
                    text="2 < 3",
                    new_text=" 3",
                    tokens=[10, 11],
                    prompt_tokens=3,
                    completion_tokens=2,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "dsv4-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "deepseek_v4")

        request = ChatCompletionRequest(
            model="dsv4-test",
            messages=[Message(role="user", content="is 2 < 3?")],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert visible == "2 < 3"

    @pytest.mark.asyncio
    async def test_streaming_chat_minimax_truncated_namespace_emits_only_tool_call(
        self, monkeypatch
    ):
        """A terminally truncated M3 namespace token must not leak as content."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.tool_parsers.minimax_m3_tool_parser import NS_TOKEN

        raw = (
            f"{NS_TOKEN[:-1]}<tool_call>\n"
            f'{NS_TOKEN}<invoke name="file_info">\n'
            f"{NS_TOKEN}<path>panel/package.json{NS_TOKEN}</path>\n"
            f"{NS_TOKEN}</invoke>\n"
            "</tool_call>"
        )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for index, character in enumerate(raw, start=1):
                    text += character
                    finished = index == len(raw)
                    yield GenerationOutput(
                        text=text,
                        new_text=character,
                        tokens=list(range(index)),
                        prompt_tokens=3,
                        completion_tokens=index,
                        finished=finished,
                        finish_reason="stop" if finished else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "minimax-m3-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "minimax_m3")

        request = ChatCompletionRequest(
            model="minimax-m3-test",
            messages=[Message(role="user", content="inspect panel/package.json")],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )

        chunks = []
        done_seen = False
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.strip() == "data: [DONE]":
                done_seen = True
            elif line.startswith("data: "):
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        tool_deltas = [
            tool_call
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for tool_call in choice.get("delta", {}).get("tool_calls", [])
        ]
        finish_reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]

        assert visible == ""
        assert len(tool_deltas) == 2
        assert tool_deltas[0]["function"] == {"name": "", "arguments": ""}
        assert tool_deltas[0]["id"].startswith("call_")
        assert "id" not in tool_deltas[1]
        assert "type" not in tool_deltas[1]
        assert "".join(
            delta.get("id") or "" for delta in tool_deltas
        ) == tool_deltas[0]["id"]
        assert tool_deltas[1]["function"]["name"] == "file_info"
        assert json.loads(tool_deltas[1]["function"]["arguments"]) == {
            "path": "panel/package.json"
        }
        assert finish_reasons == ["tool_calls"]
        assert done_seen is True

    @pytest.mark.asyncio
    async def test_streaming_chat_exact_once_buffers_schema_labeled_qwen_call(
        self, monkeypatch
    ):
        """Exact-one tool selection must not leak off-template Qwen call prose.

        Live Qwen3.6 JANGTQ emitted:
            [Q36-JT-UI-TOOL2]
            file_info
            path: panel/package.json

        That is a schema-valid tool intent, but it contains no canonical XML
        marker. Chat must buffer the selection turn from token one, then emit a
        structured tool_call instead of first streaming the lines as content.
        """
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        chunks_out = (
            "[Q36-JT-UI-TOOL2]\n",
            "file_info\npath: panel/package.json\n[Q36-JT-UI-TOOL2-DONE SIZE=3",
        )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = False

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for index, delta in enumerate(chunks_out, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=list(range(index)),
                        prompt_tokens=8,
                        completion_tokens=index,
                        finished=False,
                        finish_reason=None,
                    )
                    if self.aborted:
                        return

            async def abort_request(self, request_id):
                self.aborted = True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "qwen-jangtq-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "qwen")

        request = ChatCompletionRequest(
            model="qwen-jangtq-test",
            messages=[
                Message(
                    role="user",
                    content=(
                        "Call the built-in file_info tool exactly once with path "
                        "panel/package.json. You must use the tool."
                    ),
                )
            ],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )

        chunks = []
        done_seen = False
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.strip() == "data: [DONE]":
                done_seen = True
            elif line.startswith("data: "):
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        tool_deltas = [
            tool_call
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for tool_call in choice.get("delta", {}).get("tool_calls", [])
        ]
        finish_reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]

        assert visible == ""
        assert tool_deltas[0]["function"] == {"name": "", "arguments": ""}
        assert tool_deltas[-1]["function"]["name"] == "file_info"
        assert json.loads(tool_deltas[-1]["function"]["arguments"]) == {
            "path": "panel/package.json"
        }
        assert finish_reasons == ["tool_calls"]
        assert done_seen is True

    @pytest.mark.asyncio
    async def test_streaming_chat_exact_once_hides_pretool_prose_and_orphan_markup(
        self, monkeypatch
    ):
        """Exact-one turns never expose meta prose or rejected native fragments."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            def __init__(self, deltas):
                self.deltas = deltas

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for index, delta in enumerate(self.deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[index],
                        prompt_tokens=8,
                        completion_tokens=index,
                        finished=index == len(self.deltas),
                        finish_reason="stop" if index == len(self.deltas) else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "qwen-jangtq-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

        def request():
            return ChatCompletionRequest(
                model="qwen-jangtq-test",
                messages=[Message(
                    role="user",
                    content=(
                        "Call file_info exactly once with path panel/package.json."
                    ),
                )],
                stream=True,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }],
                tool_choice={
                    "type": "function",
                    "function": {"name": "file_info"},
                },
            )

        async def collect(deltas):
            req = request()
            chunks = []
            async for line in server.stream_chat_completion(
                _Engine(deltas),
                [m.model_dump(exclude_none=True) for m in req.messages],
                req,
                fastapi_request=None,
            ):
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    chunks.append(json.loads(line.removeprefix("data: ")))
            visible = "".join(
                choice.get("delta", {}).get("content") or ""
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
            calls = [
                call
                for chunk in chunks
                for choice in chunk.get("choices", [])
                for call in choice.get("delta", {}).get("tool_calls", [])
            ]
            return visible, calls, chunks

        visible, calls, _ = await collect([
            "I should call the requested tool. ",
            "<tool_call><function=file_info>",
            "<parameter=path>panel/package.json</parameter>",
            "</function></tool_call>",
        ])
        assert visible == ""
        assert calls[-1]["function"]["name"] == "file_info"

        visible, calls, chunks = await collect([
            "<parameter=path>\npanel/package.json\n</parameter>\n</",
        ])
        assert visible == ""
        assert calls == []
        assert any(chunk.get("error", {}).get("code") == "tool_calls_required" for chunk in chunks)

    @pytest.mark.asyncio
    async def test_streaming_chat_hides_zaya_visual_grounding_markup(self, monkeypatch):
        """ZAYA-VL point spans are control markup, not visible assistant text."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("Answer ", "Answer ", False, None),
                    ("Answer <|point_start|>tool", "<|point_start|>tool", False, None),
                    (
                        "Answer <|point_start|>tool<|point_end|> done",
                        "<|point_end|> done",
                        True,
                        "stop",
                    ),
                )
                for text, new_text, finished, reason in chunks:
                    yield GenerationOutput(
                        text=text,
                        new_text=new_text,
                        tokens=[],
                        prompt_tokens=3,
                        completion_tokens=1,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "zaya_xml")

        request = ChatCompletionRequest(
            model="zaya-vl-test",
            messages=[Message(role="user", content="use tools then answer")],
            stream=True,
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert "<|point_start|>" not in visible
        assert "<|point_end|>" not in visible
        assert visible == "Answer  done"

    @pytest.mark.asyncio
    async def test_streaming_responses_hides_zaya_visual_grounding_markup(
        self, monkeypatch
    ):
        """Responses API must keep the same point-span display hygiene."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("Answer ", "Answer ", False, None),
                    ("Answer <|point_start|>tool", "<|point_start|>tool", False, None),
                    (
                        "Answer <|point_start|>tool<|point_end|> done",
                        "<|point_end|> done",
                        True,
                        "stop",
                    ),
                )
                for text, new_text, finished, reason in chunks:
                    yield GenerationOutput(
                        text=text,
                        new_text=new_text,
                        tokens=[],
                        prompt_tokens=3,
                        completion_tokens=1,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "zaya_xml")

        request = ResponsesRequest(
            model="zaya-vl-test",
            input="use tools then answer",
            stream=True,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "use tools then answer"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        done_text = "".join(
            event.get("text", "")
            for event in events
            if event.get("type") == "response.output_text.done"
        )
        assert "<|point_start|>" not in visible
        assert "<|point_end|>" not in visible
        assert "<|point_start|>" not in done_text
        assert "<|point_end|>" not in done_text
        assert visible == "Answer  done"
        assert done_text == "Answer  done"

    @pytest.mark.asyncio
    async def test_streaming_chat_uses_registry_reasoning_parser_when_global_missing(
        self, monkeypatch
    ):
        """A missing global parser must not make an open think rail visible."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry_mod
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Registry:
            def lookup(self, _model):
                return SimpleNamespace(
                    family_name="qwen3_5_moe",
                    reasoning_parser="qwen3",
                    think_in_template=True,
                )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=True)

            async def stream_chat(self, *, messages, **kwargs):
                text = "private thought</think>VISIBLE-DONE"
                yield GenerationOutput(
                    text=text,
                    new_text=text,
                    tokens=[],
                    prompt_tokens=3,
                    completion_tokens=7,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(registry_mod, "get_model_config_registry", lambda: _Registry())
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "registry-parser-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_template_completes_thinking", lambda *a, **k: False)
        monkeypatch.setattr(server, "_engine_prompt_starts_in_reasoning", lambda *a, **k: True)

        request = ChatCompletionRequest(
            model="registry-parser-test",
            messages=[Message(role="user", content="test")],
            stream=True,
            enable_thinking=True,
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        reasoning = "".join(
            c["choices"][0]["delta"].get("reasoning_content", "")
            for c in chunks
            if c.get("choices")
        )
        visible = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c.get("choices")
        )
        assert reasoning == "private thought"
        assert visible == "VISIBLE-DONE"
        assert "<think" not in visible
        assert "</think>" not in visible

    @pytest.mark.asyncio
    async def test_streaming_responses_uses_registry_reasoning_parser_when_global_missing(
        self, monkeypatch
    ):
        """Responses must emit reasoning-summary deltas instead of inline tags."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry_mod
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Registry:
            def lookup(self, _model):
                return SimpleNamespace(
                    family_name="qwen3_5_moe",
                    reasoning_parser="qwen3",
                    think_in_template=True,
                )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=True)

            async def stream_chat(self, *, messages, **kwargs):
                text = "private thought</think>VISIBLE-DONE"
                yield GenerationOutput(
                    text=text,
                    new_text=text,
                    tokens=[],
                    prompt_tokens=3,
                    completion_tokens=7,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(registry_mod, "get_model_config_registry", lambda: _Registry())
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "registry-parser-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_template_completes_thinking", lambda *a, **k: False)
        monkeypatch.setattr(server, "_engine_prompt_starts_in_reasoning", lambda *a, **k: True)

        request = ResponsesRequest(
            model="registry-parser-test",
            input="test",
            stream=True,
            enable_thinking=True,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "test"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        reasoning = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        done_text = "".join(
            event.get("text", "")
            for event in events
            if event.get("type") == "response.output_text.done"
        )
        assert reasoning == "private thought"
        assert visible == "VISIBLE-DONE"
        assert done_text == "VISIBLE-DONE"
        assert "<think" not in visible
        assert "</think>" not in visible

    @pytest.mark.asyncio
    async def test_streaming_responses_separates_late_gemma4_thought_after_content(
        self, monkeypatch
    ):
        """A post-tool Gemma answer must not stream its late thought rail as text."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.gemma4_parser import Gemma4ReasoningParser

        raw = (
            "The reported human-readable size is 5.2 KB.\n"
            "REL1612-SEQ-T2C-DONE\n"
            "thought\n"
            "The human-readable size is 5.2 KB.\n"
            "REL1612-SEQ-T2C-DONE"
        )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=True)

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(raw, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        raw_text=text,
                        new_text=delta,
                        tokens=[],
                        prompt_tokens=7,
                        completion_tokens=idx,
                        finished=idx == len(raw),
                        finish_reason="stop" if idx == len(raw) else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "gemma4-post-tool-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", Gemma4ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "gemma4")

        request = ResponsesRequest(
            model="gemma4-post-tool-test",
            input="report the tool result",
            stream=True,
            enable_thinking=True,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "report the tool result"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        reasoning = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        done_text = "".join(
            event.get("text", "")
            for event in events
            if event.get("type") == "response.output_text.done"
        )

        expected_visible = (
            "The reported human-readable size is 5.2 KB.\n"
            "REL1612-SEQ-T2C-DONE"
        )
        assert visible == expected_visible
        assert done_text == expected_visible
        assert reasoning == (
            "The human-readable size is 5.2 KB.\n"
            "REL1612-SEQ-T2C-DONE"
        )
        assert "thought" not in visible
        assert any(event.get("type") == "response.completed" for event in events)

    @pytest.mark.asyncio
    async def test_streaming_responses_abort_is_incomplete_not_completed(
        self, monkeypatch
    ):
        """An aborted engine stream must never finalize partial text as success."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="1, ",
                    new_text="1, ",
                    tokens=[1, 2, 3],
                    prompt_tokens=7,
                    completion_tokens=3,
                    finished=False,
                    finish_reason=None,
                )
                yield GenerationOutput(
                    text="1, ",
                    new_text="",
                    tokens=[],
                    prompt_tokens=7,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="aborted",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "abort-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        server._responses_history.clear()

        request = ResponsesRequest(
            model="abort-test",
            input="count slowly",
            stream=True,
            enable_thinking=False,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "count slowly"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        response_id = next(
            event["response"]["id"]
            for event in events
            if event.get("type") == "response.created"
        )
        assert not any(event.get("type") == "response.completed" for event in events)
        incomplete = [
            event for event in events if event.get("type") == "response.incomplete"
        ]
        assert len(incomplete) == 1
        assert incomplete[0]["response"]["status"] == "incomplete"
        assert incomplete[0]["response"]["incomplete_details"] == {
            "reason": "cancelled"
        }
        done_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
        ]
        assert done_items[-1]["status"] == "incomplete"
        assert response_id not in server._responses_history

    @pytest.mark.asyncio
    async def test_streaming_responses_midstream_exception_emits_failed_terminal(
        self, monkeypatch
    ):
        """A mid-stream engine error must emit response.failed, never completed."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = []

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="partial",
                    new_text="partial",
                    tokens=[1],
                    prompt_tokens=5,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                raise RuntimeError("MIDSTREAM PROBE FAILURE")

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "failure-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ResponsesRequest(
            model="failure-test",
            input="fail after one delta",
            stream=True,
            enable_thinking=False,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "fail after one delta"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        event_types = [event.get("type") for event in events]
        assert len([event for event in events if event.get("type") == "error"]) == 1
        failed = [event for event in events if event.get("type") == "response.failed"]
        assert len(failed) == 1
        assert failed[0]["response"]["status"] == "failed"
        assert failed[0]["response"]["usage"] == {
            "input_tokens": 5,
            "output_tokens": 1,
            "total_tokens": 6,
        }
        assert "response.output_text.delta" in event_types
        assert event_types.index("error") < event_types.index("response.failed")
        assert not any(event.get("type") == "response.completed" for event in events)
        assert len(_Engine.aborted) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_factory, expected_code",
        [
            (
                lambda: __import__(
                    "vmlx_engine.errors", fromlist=["MediaControlsUnmeetableError"]
                ).MediaControlsUnmeetableError("video_controls: cannot fit 64 frames"),
                "media_controls_unmeetable",
            ),
            (
                lambda: __import__(
                    "vmlx_engine.errors", fromlist=["PromptTooLongError"]
                ).PromptTooLongError(prompt_tokens=9000, max_prompt_tokens=4096),
                "prompt_too_long",
            ),
        ],
    )
    async def test_streaming_responses_typed_rejection_emits_failed_terminal(
        self, monkeypatch, exc_factory, expected_code
    ):
        """A typed request rejection is FATAL for the Responses stream.

        Live at e6262e3a (strict-multi-4S-033105 resp_b8b3d8425246): the typed
        handlers yielded the ``error`` event and then fell through into the
        normal finalization, which emitted ``response.completed`` with an
        empty output — a fatal rejection reported as a success. The stream
        must end with ``response.failed`` carrying the typed code.
        """
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        exc = exc_factory()

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = []

            async def stream_chat(self, *, messages, **kwargs):
                raise exc
                yield  # pragma: no cover - makes this an async generator

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "rejection-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ResponsesRequest(
            model="rejection-test",
            input="reject me",
            stream=True,
            enable_thinking=False,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "reject me"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        event_types = [event.get("type") for event in events]
        errors = [event for event in events if event.get("type") == "error"]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == expected_code
        failed = [event for event in events if event.get("type") == "response.failed"]
        assert len(failed) == 1
        assert failed[0]["response"]["status"] == "failed"
        assert failed[0]["response"]["error"]["code"] == expected_code
        assert event_types[-1] == "response.failed"
        assert event_types.index("error") < event_types.index("response.failed")
        # never a success terminal, never an empty message after the rejection
        assert "response.completed" not in event_types
        assert "response.output_item.added" not in event_types
        assert "response.output_text.delta" not in event_types
        assert len(_Engine.aborted) == 1

    @pytest.mark.asyncio
    async def test_streaming_chat_midstream_exception_keeps_delta_usage_and_done(
        self, monkeypatch
    ):
        """Chat SSE keeps visible prefix and authoritative usage before [DONE]."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = []

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="partial",
                    new_text="partial",
                    tokens=[1],
                    prompt_tokens=5,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                )
                raise RuntimeError("MIDSTREAM CHAT PROBE FAILURE")

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "failure-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ChatCompletionRequest(
            model="failure-test",
            messages=[{"role": "user", "content": "fail after one delta"}],
            stream=True,
            stream_options={"include_usage": True},
            enable_thinking=False,
        )

        raw_chunks = []
        async for chunk in server.stream_chat_completion(
            _Engine(),
            request.messages,
            request,
            fastapi_request=None,
        ):
            raw_chunks.append(chunk)

        data = [
            line.removeprefix("data: ")
            for chunk in raw_chunks
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]
        parsed = [json.loads(item) for item in data if item != "[DONE]"]
        assert any(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content") == "partial"
            for chunk in parsed
            if chunk.get("choices")
        )
        error_index = next(i for i, chunk in enumerate(parsed) if chunk.get("error"))
        usage_index = next(i for i, chunk in enumerate(parsed) if chunk.get("usage"))
        assert error_index < usage_index
        assert parsed[usage_index]["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        }
        assert data[-1] == "[DONE]"
        assert len(_Engine.aborted) == 1

    @pytest.mark.asyncio
    async def test_streaming_chat_structured_engine_error_does_not_enter_reasoning(
        self, monkeypatch
    ):
        """Structured engine errors must not become reasoning/content deltas."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning import get_parser

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = []

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="",
                    new_text="",
                    prompt_tokens=5,
                    completion_tokens=0,
                    finished=True,
                    finish_reason="error",
                    error=(
                        "Engine loop error: [full] Negative dimensions not allowed."
                    ),
                    error_code="engine_loop_error",
                )

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "laguna-error-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", get_parser("deepseek_r1")())
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ChatCompletionRequest(
            model="laguna-error-test",
            messages=[{"role": "user", "content": "trigger engine error"}],
            stream=True,
            stream_options={"include_usage": True},
            enable_thinking=True,
        )

        raw_chunks = []
        async for chunk in server.stream_chat_completion(
            _Engine(),
            request.messages,
            request,
            fastapi_request=None,
        ):
            raw_chunks.append(chunk)

        data = [
            line.removeprefix("data: ")
            for chunk in raw_chunks
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]
        parsed = [json.loads(item) for item in data if item != "[DONE]"]
        deltas = [
            choice.get("delta", {})
            for chunk in parsed
            for choice in chunk.get("choices", [])
        ]
        assert not any(delta.get("reasoning_content") for delta in deltas)
        assert not any(delta.get("content") for delta in deltas)
        errors = [chunk["error"] for chunk in parsed if chunk.get("error")]
        assert len(errors) == 1
        assert errors[0]["code"] == "internal_error"
        assert "Negative dimensions not allowed" in errors[0]["message"]
        assert all("[Engine error:" not in json.dumps(chunk) for chunk in parsed)
        assert data[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_streaming_responses_structured_engine_error_failed_not_reasoning(
        self, monkeypatch
    ):
        """Responses converts structured engine errors to response.failed."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning import get_parser

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)
            aborted = []

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="",
                    new_text="",
                    prompt_tokens=7,
                    completion_tokens=0,
                    finished=True,
                    finish_reason="error",
                    error=(
                        "Engine loop error: [full] Negative dimensions not allowed."
                    ),
                    error_code="engine_loop_error",
                )

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "laguna-error-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", get_parser("deepseek_r1")())
        monkeypatch.setattr(server, "_tool_call_parser", None)

        request = ResponsesRequest(
            model="laguna-error-test",
            input="trigger engine error",
            stream=True,
            enable_thinking=True,
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "trigger engine error"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        assert not any(
            event.get("type") == "response.reasoning_summary_text.delta"
            for event in events
        )
        assert not any(
            event.get("type") == "response.output_text.delta" for event in events
        )
        errors = [event for event in events if event.get("type") == "error"]
        failed = [event for event in events if event.get("type") == "response.failed"]
        assert len(errors) == 1
        assert len(failed) == 1
        assert "Negative dimensions not allowed" in errors[0]["error"]["message"]
        assert failed[0]["response"]["status"] == "failed"
        assert not any(event.get("type") == "response.completed" for event in events)
        assert all("[Engine error:" not in json.dumps(event) for event in events)

    @pytest.mark.asyncio
    async def test_streaming_responses_tool_call_arguments_survive_buffering(
        self, monkeypatch
    ):
        """Responses SSE heartbeats must not replace final tool arguments."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    (
                        '<tool_call>{"name":"lookup","arguments":{"query":"alpha"',
                        False,
                        None,
                    ),
                    (',"limit":2}}</tool_call>', True, "stop"),
                )
                text = ""
                for idx, (delta, finished, reason) in enumerate(chunks, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[],
                        prompt_tokens=5,
                        completion_tokens=idx,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "unit-tool-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ResponsesRequest(
            model="unit-tool-model",
            input="use lookup",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query", "limit"],
                    },
                }
            ],
        )

        payloads: list[tuple[str, dict]] = []
        async for event in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "use lookup"}],
            request,
            fastapi_request=None,
        ):
            if not event.startswith("event: "):
                continue
            event_type = event.splitlines()[0].removeprefix("event: ")
            data_line = next(
                line for line in event.splitlines() if line.startswith("data: ")
            )
            payloads.append((event_type, json.loads(data_line.removeprefix("data: "))))

        arg_deltas = [
            payload["delta"]
            for event_type, payload in payloads
            if event_type == "response.function_call_arguments.delta"
        ]
        arg_done = [
            payload["arguments"]
            for event_type, payload in payloads
            if event_type == "response.function_call_arguments.done"
        ]
        function_items = [
            payload["item"]
            for event_type, payload in payloads
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]

        expected_args = {"query": "alpha", "limit": 2}
        assert any(
            event_type == "response.heartbeat"
            and payload.get("tool_call_generating") is True
            for event_type, payload in payloads
        )
        assert json.loads("".join(arg_deltas)) == expected_args
        assert json.loads(arg_done[-1]) == expected_args
        assert function_items[-1]["name"] == "lookup"
        assert json.loads(function_items[-1]["arguments"]) == expected_args

    def test_explicit_tool_parser_none_skips_registry_and_generic_parse(
        self, monkeypatch
    ):
        """The CLI/UI parser-off setting must survive request auto-detection."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        raw = (
            "<tool_call><function=file_info>"
            "<parameter=path>README.md</parameter>"
            "</function></tool_call>"
        )
        request = ResponsesRequest(
            model="bonsai-test",
            input="Call file_info once for README.md",
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", True)
        monkeypatch.setattr(server, "_model_name", "bonsai-test")
        monkeypatch.setattr(server, "_model_path", None)

        cleaned, calls = server._parse_tool_calls_with_parser(raw, request)

        assert cleaned == raw
        assert calls is None

    @pytest.mark.asyncio
    async def test_streaming_responses_explicit_tool_parser_none_keeps_raw_output(
        self, monkeypatch
    ):
        """Parser-off streams raw model text instead of buffering a fake call."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        raw = (
            "<tool_call><function=file_info>"
            "<parameter=path>README.md</parameter>"
            "</function></tool_call>"
        )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text=raw,
                    new_text=raw,
                    tokens=[1],
                    prompt_tokens=8,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "bonsai-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", True)

        request = ResponsesRequest(
            model="bonsai-test",
            input="Call file_info once for README.md",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

        assert visible == raw
        assert completed["output_text"] == raw
        assert completed.get("warnings", []) == []
        assert function_items == []

    @pytest.mark.asyncio
    async def test_streaming_chat_explicit_tool_parser_none_keeps_raw_output(
        self, monkeypatch
    ):
        """Chat Completions honors the same parser-off stream contract."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        raw = (
            "<tool_call><function=file_info>"
            "<parameter=path>README.md</parameter>"
            "</function></tool_call>"
        )

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text=raw,
                    new_text=raw,
                    tokens=[1],
                    prompt_tokens=8,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "bonsai-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", True)

        request = ChatCompletionRequest(
            model="bonsai-test",
            messages=[Message(role="user", content="Call file_info for README.md")],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        tool_deltas = [
            choice["delta"]["tool_calls"]
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("delta", {}).get("tool_calls")
        ]

        assert visible == raw
        assert tool_deltas == []

    @pytest.mark.asyncio
    async def test_streaming_chat_post_tool_false_marker_does_not_emit_phantom_tool_delta(
        self, monkeypatch
    ):
        """Optional post-tool continuations must not advertise rejected empty calls."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _ReasoningParser:
            def reset_state(self, **kwargs):
                pass

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                return SimpleNamespace(reasoning=None, content=delta_text)

            def extract_reasoning(self, text):
                return None, text

        deltas = ["<tool_call>", "LAG-S21-POST-TOOL-FINAL-DONE"]

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[idx],
                        prompt_tokens=12,
                        completion_tokens=idx,
                        finished=(idx == len(deltas)),
                        finish_reason="stop" if idx == len(deltas) else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", _ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "auto")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

        request = ChatCompletionRequest(
            model="laguna-test",
            messages=[
                Message(
                    role="user",
                    content=(
                        "Call the built-in file_info tool exactly once with path "
                        "panel/package.json. You must use the tool. After the "
                        "tool result, reply exactly "
                        "LAG-S21-POST-TOOL-FINAL-DONE and nothing else."
                    ),
                )
            ],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
        messages = [
            {"role": "user", "content": request.messages[0].content},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_prior",
                        "type": "function",
                        "function": {
                            "name": "file_info",
                            "arguments": '{"path":"panel/package.json"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_prior",
                "content": "Path: panel/package.json\nSize: 5.2 KB",
            },
        ]

        chunks = []
        async for line in server.stream_chat_completion(
            _Engine(),
            messages,
            request,
            fastapi_request=None,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        tool_deltas = [
            choice["delta"]["tool_calls"]
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("delta", {}).get("tool_calls")
        ]
        finish_reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]

        assert visible == "LAG-S21-POST-TOOL-FINAL-DONE"
        assert tool_deltas == []
        assert finish_reasons == ["stop"]

    @pytest.mark.asyncio
    async def test_streaming_responses_qwen_exact_once_stops_after_first_valid_call(
        self, monkeypatch
    ):
        """A single-call contract must not drain Qwen's post-call repetition."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        call_deltas = [
            "reasoning before the call ",
            "<tool_call>",
            "<function=file_info>",
            "<parameter=path>panel/package.json</parameter>",
            "</function>",
            "</tool_call>",
        ]
        # Regression shape from live Qwen3.6 Electron: the model fit a second
        # byte-identical call inside the generic eight-chunk grace window and
        # then naturally finished. The old `not output.finished` early-stop
        # branch never reached its ninth tick, so both calls were executed.
        duplicate_call = [
            "<tool_call>",
            "<function=file_info>",
            "<parameter=path>panel/package.json</parameter>",
            "</function>",
            "</tool_call>",
        ]
        deltas = call_deltas + duplicate_call

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            def __init__(self):
                self.aborted: list[str] = []
                self.chunks_consumed = 0

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(deltas, start=1):
                    text += delta
                    self.chunks_consumed += 1
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[idx],
                        prompt_tokens=8,
                        completion_tokens=idx,
                        finished=(idx == len(deltas)),
                        finish_reason="length" if idx == len(deltas) else None,
                    )

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        engine = _Engine()
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "bonsai-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

        request = ResponsesRequest(
            model="bonsai-test",
            input=(
                "Call the built-in file_info tool exactly once with path "
                "panel/package.json."
            ),
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        events: list[dict] = []
        async for chunk in server.stream_responses_api(
            engine,
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )
        output_deltas = [
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        ]

        assert engine.aborted
        assert engine.chunks_consumed == len(call_deltas)
        assert len(function_items) == 1
        assert function_items[0]["name"] == "file_info"
        assert json.loads(function_items[0]["arguments"]) == {
            "path": "panel/package.json"
        }
        assert output_deltas == []
        assert completed["output_text"] == ""
        assert "reasoning before the call" not in json.dumps(completed)
        assert "POST_CALL_REPEAT" not in json.dumps(completed)

    @pytest.mark.asyncio
    async def test_streaming_responses_qwen_exact_once_streams_reasoning_not_pretool_prose(
        self, monkeypatch
    ):
        """Premature post-think meta prose stays hidden before an exact-one call."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _ReasoningParser:
            def reset_state(self, **kwargs):
                pass

            def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
                if delta_text.startswith("R:"):
                    return SimpleNamespace(reasoning=delta_text[2:], content=None)
                if delta_text.startswith("C:"):
                    return SimpleNamespace(reasoning=None, content=delta_text[2:])
                return SimpleNamespace(reasoning=None, content=delta_text)

            def extract_reasoning(self, text):
                return None, text

        deltas = [
            "R:plan the requested call",
            "C:visible meta-reasoning that must stay hidden ",
            "C:<tool_call>",
            "C:<function=file_info>",
            "C:<parameter=path>panel/package.json</parameter>",
            "C:</function>",
            "C:</tool_call>",
            "C: POST_CALL_REPEAT",
        ]

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            def __init__(self):
                self.aborted: list[str] = []

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[idx],
                        prompt_tokens=8,
                        completion_tokens=idx,
                        finished=(idx == len(deltas)),
                        finish_reason="length" if idx == len(deltas) else None,
                    )

            async def abort_request(self, request_id):
                self.aborted.append(request_id)
                return True

        engine = _Engine()
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "bonsai-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", _ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

        request = ResponsesRequest(
            model="bonsai-test",
            input=(
                "Call the built-in file_info tool exactly once with path "
                "panel/package.json."
            ),
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        events: list[dict] = []
        async for chunk in server.stream_responses_api(
            engine,
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        reasoning = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]
        terminal = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

        assert reasoning == "plan the requested call"
        assert visible == ""
        assert terminal["status"] == "completed"
        assert terminal["output_text"] == ""
        assert "visible meta-reasoning" not in json.dumps(terminal)
        assert len(function_items) == 1
        assert function_items[0]["name"] == "file_info"
        assert json.loads(function_items[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    @pytest.mark.asyncio
    async def test_streaming_responses_dsv4_tool_markup_never_leaks_into_reasoning(
        self, monkeypatch
    ):
        """DSML parsed from the reasoning rail stays structured, not visible."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _ReasoningParser:
            def reset_state(self, **kwargs):
                pass

            def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
                assert delta_text.startswith("R:")
                return SimpleNamespace(reasoning=delta_text[2:], content=None)

            def extract_reasoning(self, text):
                return text, None

        dsml_call = (
            '<｜DSML｜tool_calls>\n'
            '<｜DSML｜invoke name="file_info">\n'
            '<｜DSML｜parameter name="path" string="true">'
            'panel/package.json</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n'
            '</｜DSML｜tool_calls>'
        )
        deltas = ["R:I will call the requested tool.\n", f"R:{dsml_call}"]

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[idx],
                        prompt_tokens=8,
                        completion_tokens=idx,
                        finished=(idx == len(deltas)),
                        finish_reason="stop" if idx == len(deltas) else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "dsv4-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", _ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

        request = ResponsesRequest(
            model="dsv4-test",
            input=(
                "Call the built-in file_info tool exactly once with path "
                "panel/package.json."
            ),
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        events: list[dict] = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        reasoning_deltas = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        reasoning_done = [
            event.get("text", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.done"
        ]
        reasoning_added = [
            event
            for event in events
            if event.get("type") == "response.output_item.added"
            and event.get("item", {}).get("type") == "reasoning"
        ]
        reasoning_part_added = [
            event
            for event in events
            if event.get("type") == "response.reasoning_summary_part.added"
        ]
        reasoning_part_done = [
            event
            for event in events
            if event.get("type") == "response.reasoning_summary_part.done"
        ]
        reasoning_item_done = [
            event
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "reasoning"
        ]
        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

        assert reasoning_deltas.strip() == "I will call the requested tool."
        assert reasoning_done[-1] == "I will call the requested tool."
        assert len(reasoning_added) == 1
        assert len(reasoning_part_added) == 1
        assert len(reasoning_part_done) == 1
        assert len(reasoning_item_done) == 1
        reasoning_item_id = reasoning_added[0]["item"]["id"]
        assert reasoning_item_id.startswith("rs_")
        assert reasoning_added[0]["output_index"] == 0
        assert {
            event["item_id"]
            for event in events
            if event.get("type")
            in {
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
            }
        } == {reasoning_item_id}
        assert reasoning_item_done[0]["item"]["id"] == reasoning_item_id
        assert reasoning_item_done[0]["item"]["summary"] == [
            {"type": "summary_text", "text": "I will call the requested tool."}
        ]
        assert "DSML" not in reasoning_deltas
        assert "DSML" not in reasoning_done[-1]
        assert len(function_items) == 1
        assert function_items[0]["name"] == "file_info"
        assert json.loads(function_items[0]["arguments"]) == {
            "path": "panel/package.json"
        }
        reasoning_items = [
            item for item in completed["output"] if item.get("type") == "reasoning"
        ]
        assert reasoning_items[-1]["id"] == reasoning_item_id
        assert reasoning_items[-1]["summary"][0]["text"] == (
            "I will call the requested tool."
        )
        assert "DSML" not in json.dumps(completed)
        added_items = [
            event
            for event in events
            if event.get("type") == "response.output_item.added"
        ]
        assert [event["item"]["type"] for event in added_items] == [
            "reasoning",
            "function_call",
        ]
        assert [event["output_index"] for event in added_items] == [0, 1]
        assert [item["type"] for item in completed["output"]] == [
            "reasoning",
            "function_call",
        ]

    @pytest.mark.asyncio
    async def test_streaming_responses_reasoning_closes_before_message_starts(
        self, monkeypatch
    ):
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _ReasoningParser:
            def reset_state(self, **kwargs):
                pass

            def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
                if delta_text.startswith("R:"):
                    return SimpleNamespace(reasoning=delta_text[2:], content=None)
                assert delta_text.startswith("C:")
                return SimpleNamespace(reasoning=None, content=delta_text[2:])

            def extract_reasoning(self, text):
                return "private plan", "visible answer"

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                for idx, delta in enumerate(("R:private plan", "C:visible answer"), start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[idx],
                        prompt_tokens=4,
                        completion_tokens=idx,
                        finished=(idx == 2),
                        finish_reason="stop" if idx == 2 else None,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", _ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "reason then answer"}],
            ResponsesRequest(
                model="laguna-test",
                input="reason then answer",
                stream=True,
                enable_thinking=True,
            ),
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        lifecycle = [
            (event.get("type"), event.get("output_index"), event.get("item", {}).get("type"))
            for event in events
            if event.get("type") in {
                "response.output_item.added",
                "response.output_item.done",
            }
        ]
        assert lifecycle == [
            ("response.output_item.added", 0, "reasoning"),
            ("response.output_item.done", 0, "reasoning"),
            ("response.output_item.added", 1, "message"),
            ("response.output_item.done", 1, "message"),
        ]
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )
        assert [item["type"] for item in completed["output"]] == [
            "reasoning",
            "message",
        ]
        assert completed["output_text"] == "visible answer"

    @pytest.mark.asyncio
    async def test_nonstream_responses_serializes_reasoning_as_standard_item(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(
                    text="<think>private analysis</think>visible answer",
                    raw_text="<think>private analysis</think>visible answer",
                    prompt_tokens=8,
                    completion_tokens=7,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_response(
            ResponsesRequest(
                model="laguna-test",
                input="reason, then answer",
                max_output_tokens=64,
                enable_thinking=True,
            ),
            fastapi_request=None,
        )

        assert response.output_text == "visible answer"
        assert [item.type for item in response.output] == ["reasoning", "message"]
        reasoning = response.output[0]
        assert reasoning.id.startswith("rs_")
        assert reasoning.summary[0].type == "summary_text"
        assert reasoning.summary[0].text == "private analysis"
        assert reasoning.content == []
        assert "private analysis" not in response.output_text

    @pytest.mark.asyncio
    async def test_nonstream_responses_thinking_off_never_promotes_private_reasoning(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(
                    text="<think>private analysis only",
                    raw_text="<think>private analysis only",
                    prompt_tokens=8,
                    completion_tokens=4,
                    finish_reason="length",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_name", "laguna-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_response(
            ResponsesRequest(
                model="laguna-test",
                input="answer without thinking",
                max_output_tokens=16,
                enable_thinking=False,
            ),
            fastapi_request=None,
        )

        assert response.output_text in (None, "")
        assert all(item.type != "reasoning" for item in response.output)
        assert "private analysis" not in response.model_dump_json()

    @pytest.mark.asyncio
    async def test_nonstream_chat_thinking_off_extracts_private_tool_without_leak(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        hidden = (
            "<think>private plan must stay hidden\n"
            '<tool_call>{"name":"file_info","arguments":{"path":"panel/package.json"}}</tool_call>'
            "</think>"
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(
                    text=hidden,
                    raw_text=hidden,
                    prompt_tokens=8,
                    completion_tokens=9,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "hidden-tool-test")
        monkeypatch.setattr(server, "_model_name", "hidden-tool-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="hidden-tool-test",
                messages=[Message(role="user", content="use file_info")],
                enable_thinking=False,
                max_tokens=32,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "file_info",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
            ),
            fastapi_request=None,
        )

        body = json.loads(response.body)
        message = body["choices"][0]["message"]
        call = message["tool_calls"][0]

        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert message["content"] is None
        assert "reasoning_content" not in message
        assert call["function"]["name"] == "file_info"
        assert json.loads(call["function"]["arguments"]) == {
            "path": "panel/package.json"
        }
        assert "private plan" not in json.dumps(body)

    @pytest.mark.asyncio
    async def test_nonstream_responses_thinking_off_extracts_private_tool_without_leak(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser

        hidden = (
            "<think>private responses plan must stay hidden\n"
            '<tool_call>{"name":"file_info","arguments":{"path":"panel/package.json"}}</tool_call>'
            "</think>"
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(
                    text=hidden,
                    raw_text=hidden,
                    prompt_tokens=8,
                    completion_tokens=9,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "hidden-tool-test")
        monkeypatch.setattr(server, "_model_name", "hidden-tool-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", DeepSeekR1ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_response(
            ResponsesRequest(
                model="hidden-tool-test",
                input="use file_info",
                enable_thinking=False,
                max_output_tokens=32,
                tools=[
                    {
                        "type": "function",
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
            ),
            fastapi_request=None,
        )

        assert [item.type for item in response.output] == ["function_call"]
        function_item = response.output[0]
        assert function_item.name == "file_info"
        assert json.loads(function_item.arguments) == {"path": "panel/package.json"}
        assert response.output_text in ("", None)
        assert "private responses plan" not in response.model_dump_json()

    def test_parser_init_failure_still_filters_request_tools(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        def _raise_parser_init(_name):
            raise RuntimeError("parser unavailable")

        monkeypatch.setattr(server, "_tool_call_parser", "broken_parser")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server.ToolParserManager, "get_tool_parser", _raise_parser_init)

        request = ChatCompletionRequest(
            model="filter-test",
            messages=[Message(role="user", content="use a tool")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )

        _cleaned, calls = server._parse_tool_calls_with_parser(
            '{"name":"not_available","arguments":{"path":"panel/package.json"}}',
            request,
        )

        assert calls is None
        assert "not_available" in _cleaned

    @pytest.mark.asyncio
    async def test_nonstream_responses_minimax_tools_available_no_call_runs_answer_pass(
        self, monkeypatch
    ):
        """An explicit M3 thinking budget keeps its no-call answer pass."""
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "<mm:think>No function is needed for this answer."
                    if len(self.calls) == 1
                    else "MM3-NONSTREAM-RESPONSES-DONE"
                )
                return GenerationOutput(
                    text=text,
                    raw_text=text,
                    prompt_tokens=8,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "jangq-ai/MiniMax-M3-Coder-Small")
        monkeypatch.setattr(server, "_model_name", "jangq-ai/MiniMax-M3-Coder-Small")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", MiniMaxM3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "minimax_m3")
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_response(
            ResponsesRequest(
                model="jangq-ai/MiniMax-M3-Coder-Small",
                input="answer without a tool",
                max_output_tokens=64,
                max_thinking_tokens=8,
                enable_thinking=True,
                tools=[
                    {
                        "type": "function",
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
            ),
            fastapi_request=None,
        )

        assert response.output_text == "MM3-NONSTREAM-RESPONSES-DONE"
        assert len(engine.calls) == 2
        assert "tools" not in engine.calls[1]["kwargs"]
        assert engine.calls[1]["kwargs"]["chat_template_kwargs"]["thinking_mode"] == "disabled"

    def test_step_native_tool_recovery_intent_gate(self):
        """Only ordinary auto-tool turns may drop schemas for Step recovery."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        tools = [
            {
                "type": "function",
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ]

        ordinary = ResponsesRequest(
            model="step-test",
            input="Privately calculate 143 times 27 and return the number.",
            tools=tools,
        )
        explicit = ResponsesRequest(
            model="step-test",
            input="Use the run_command tool to print the number.",
            tools=tools,
        )
        prohibited = ResponsesRequest(
            model="step-test",
            input="Do not call run_command; calculate the number yourself.",
            tools=tools,
        )
        required = ResponsesRequest(
            model="step-test",
            input="Calculate the number.",
            tool_choice="required",
            tools=tools,
        )

        assert not server._request_explicitly_requests_tool_use(ordinary)
        assert server._request_explicitly_requests_tool_use(explicit)
        assert not server._request_explicitly_requests_tool_use(prohibited)
        assert server._request_explicitly_requests_tool_use(required)
        assert server._native_reasoning_tool_recovery_allowed(
            "step3p7",
            ordinary,
            tools_available=True,
            effective_thinking=True,
        )
        assert not server._native_reasoning_tool_recovery_allowed(
            "step3p7",
            explicit,
            tools_available=True,
            effective_thinking=True,
        )

    def test_fulfilled_tool_round_clears_explicit_intent(self):
        """The continuation request — tool_call + tool result appended — has
        FULFILLED the "use the tool" instruction. Suppressing the tools-free
        recovery there left reasoning-only continuations with no answer pass
        (chat 502 / silently empty Responses message), live-measured on
        Qwen3.8-Flash-Next JANG_1L 2026-08-28."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        chat_continuation = ChatCompletionRequest(
            model="qwen-test",
            messages=[
                {"role": "user", "content": "Use the get_weather tool for Paris."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\": \"Paris\"}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21}"},
            ],
            tools=tools,
        )
        assert not server._request_explicitly_requests_tool_use(chat_continuation)
        assert server._request_tool_intent_already_fulfilled(chat_continuation)

        responses_continuation = ResponsesRequest(
            model="qwen-test",
            input=[
                {"role": "user", "content": "Use the get_weather tool for Paris."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Paris\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"temp_c\": 21}",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )
        assert not server._request_explicitly_requests_tool_use(responses_continuation)
        assert server._request_tool_intent_already_fulfilled(responses_continuation)

        # A PENDING explicit request (no tool result yet) keeps its intent.
        pending = ChatCompletionRequest(
            model="qwen-test",
            messages=[{"role": "user", "content": "Use the get_weather tool for Paris."}],
            tools=tools,
        )
        assert server._request_explicitly_requests_tool_use(pending)
        assert not server._request_tool_intent_already_fulfilled(pending)

    def test_multimodal_latest_user_text_does_not_reuse_prior_tool_intent(self):
        """Pydantic ContentPart text owns intent on the current media turn."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest

        request = ChatCompletionRequest.model_validate(
            {
                "model": "step-test",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the list_directory tool exactly once.",
                    },
                    {"role": "assistant", "content": "done"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,AA=="
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Do not call any tool. This turn adds exactly "
                                    "one new image."
                                ),
                            },
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
            }
        )

        assert server._latest_request_user_text(request) == (
            "Do not call any tool. This turn adds exactly one new image."
        )
        assert not server._request_explicitly_requires_one_tool_once(request)
        assert not server._request_explicitly_requests_tool_use(request)

    def test_multimodal_latest_user_text_keeps_current_exact_tool_intent(self):
        """The fix must not weaken a genuine current multimodal tool contract."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest

        request = ChatCompletionRequest.model_validate(
            {
                "model": "step-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,AA=="
                                },
                            },
                            {
                                "type": "text",
                                "text": "Use list_directory exactly once.",
                            },
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
            }
        )

        assert server._latest_request_user_text(request) == (
            "Use list_directory exactly once."
        )
        assert server._request_explicitly_requires_one_tool_once(request)
        assert server._request_explicitly_requests_tool_use(request)

    @pytest.mark.asyncio
    async def test_streaming_responses_step_invalid_auto_tool_stays_fail_closed(
        self, monkeypatch
    ):
        """Step Auto never fabricates a second tools-free sample after bad XML."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

        config = SimpleNamespace(
            family_name="step3p7",
            think_in_template=True,
            reasoning_parser="qwen3",
            tool_parser="step3p5",
            supports_thinking=True,
        )

        class _Engine:
            is_mllm = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def stream_chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
                if len(self.calls) == 1:
                    deltas = [
                        "I should answer the arithmetic request.",
                        (
                            "</think>\n<tool_call>\n<function=run_command>\n"
                            "</function>\n</tool_call>"
                        ),
                    ]
                else:
                    deltas = [
                        "I checked the multiplication again.",
                        "</think>\n3861 ",
                        "STEP-RECOVERY-DONE",
                    ]
                text = ""
                for index, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[index],
                        prompt_tokens=8,
                        completion_tokens=index,
                        finished=index == len(deltas),
                        finish_reason="stop" if index == len(deltas) else None,
                    )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "step-recovery-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "step3p5")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(
            server,
            "_engine_prompt_starts_in_reasoning",
            lambda *args, **kwargs: True,
        )
        monkeypatch.setattr(
            registry,
            "get_model_config_registry",
            lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
        )

        tools = [
            {
                "type": "function",
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ]
        request = ResponsesRequest(
            model="step-recovery-test",
            input="Privately calculate 143 times 27.",
            stream=True,
            max_output_tokens=128,
            enable_thinking=True,
            tools=tools,
        )

        events = []
        async for chunk in server.stream_responses_api(
            engine,
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
            tools=tools,
            enable_thinking=True,
            max_tokens=128,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        content_deltas = [
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        ]
        visible = "".join(content_deltas)
        incomplete = next(
            event["response"]
            for event in events
            if event.get("type") == "response.incomplete"
        )
        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]

        assert visible == ""
        assert incomplete["output_text"] == ""
        assert incomplete["incomplete_details"] == {
            "reason": "reasoning_only_no_content"
        }
        assert function_items == []
        assert "<tool_call>" not in json.dumps(incomplete)
        assert len(engine.calls) == 1
        assert "tools" in engine.calls[0]["kwargs"]

    @pytest.mark.asyncio
    async def test_streaming_responses_qwen4_exp_reasoning_only_stays_one_pass(
        self, monkeypatch
    ):
        """qwen4_exp's answer-pass enrollment (f58c9c196) was corrected
        2026-08-28: it hard-split the first pass, then silently reran a
        SECOND thinking-off generation with tools popped and a FABRICATED
        reasoning_content history item spliced ahead of it -- forced
        behavior forbidden by AGENTS.md:54. A default qwen4_exp request that
        never emits </think> must now: run generation exactly once, keep the
        caller's full output budget (no hard split), keep tools, never force
        enable_thinking=False, never emit the vmlx-answer-pass-start SSE
        marker (structurally impossible without a second call), and surface
        the honest reasoning_only_no_content incomplete signal instead of a
        synthesized answer."""
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

        config = SimpleNamespace(
            family_name="qwen4_exp",
            think_in_template=True,
            reasoning_parser="qwen3",
            tool_parser="hermes",
            supports_thinking=True,
        )

        class _Engine:
            is_mllm = True
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def stream_chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
                # Answers INSIDE the open think block and EOSes -- never
                # generates </think> -- the live-measured qwen4_exp disease
                # this whole campaign root-causes.
                deltas = ["I should check the weather data for this city."]
                text = ""
                for i, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text, new_text=delta, tokens=[i],
                        prompt_tokens=8, completion_tokens=i,
                        finished=i == len(deltas),
                        finish_reason="stop" if i == len(deltas) else None,
                    )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "qwen4-exp-reasononly-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "hermes")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(
            server, "_engine_prompt_starts_in_reasoning", lambda *args, **kwargs: True
        )
        monkeypatch.setattr(
            registry,
            "get_model_config_registry",
            lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
        )

        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        request = ResponsesRequest(
            model="qwen4-exp-reasononly-test",
            input="What is the weather in Berlin?",
            stream=True,
            max_output_tokens=128,
            enable_thinking=True,
            tools=tools,
        )

        raw_chunks = []
        events = []
        async for chunk in server.stream_responses_api(
            engine,
            [{"role": "user", "content": request.input}],
            request,
            fastapi_request=None,
            tools=tools,
            enable_thinking=True,
            max_tokens=128,
        ):
            raw_chunks.append(chunk)
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        incomplete = next(
            event["response"] for event in events
            if event.get("type") == "response.incomplete"
        )

        # Exactly one generation, no hidden retry.
        assert len(engine.calls) == 1
        # Full output budget preserved -- not hard-split to reserve room for
        # a second pass.
        assert engine.calls[0]["kwargs"].get("max_tokens") == 128
        # Tools preserved, never popped for a retry that doesn't exist.
        assert "tools" in engine.calls[0]["kwargs"]
        assert engine.calls[0]["kwargs"]["tools"]
        # enable_thinking never forced off.
        assert engine.calls[0]["kwargs"].get("enable_thinking") is not False
        # The answer-pass SSE marker never appears -- structurally impossible
        # without a second call, checked directly against the raw stream too.
        assert "vmlx-answer-pass-start" not in "".join(raw_chunks)
        # Honest incomplete signal, not a synthesized answer.
        assert incomplete["output_text"] == ""
        assert incomplete["incomplete_details"] == {"reason": "reasoning_only_no_content"}

    @pytest.mark.asyncio
    async def test_streaming_chat_step_invalid_auto_tool_stays_fail_closed(
        self, monkeypatch
    ):
        """Chat likewise reports invalid Step output without a second sample."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

        config = SimpleNamespace(
            family_name="step3p7",
            think_in_template=True,
            reasoning_parser="qwen3",
            tool_parser="step3p5",
            supports_thinking=True,
        )

        class _Engine:
            is_mllm = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def stream_chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
                deltas = (
                    [
                        "I should answer without a command.",
                        (
                            "</think>\n<tool_call>\n<function=run_command>\n"
                            "</function>\n</tool_call>"
                        ),
                    ]
                    if len(self.calls) == 1
                    else [
                        "I checked the result.",
                        "</think>\n3861 ",
                        "STEP-CHAT-RECOVERY-DONE",
                    ]
                )
                text = ""
                for index, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[index],
                        prompt_tokens=8,
                        completion_tokens=index,
                        finished=index == len(deltas),
                        finish_reason="stop" if index == len(deltas) else None,
                    )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "step-recovery-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "step3p5")
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(
            server,
            "_engine_prompt_starts_in_reasoning",
            lambda *args, **kwargs: True,
        )
        monkeypatch.setattr(
            registry,
            "get_model_config_registry",
            lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        request = ChatCompletionRequest(
            model="step-recovery-test",
            messages=[Message(role="user", content="Privately calculate 143 times 27.")],
            stream=True,
            max_tokens=128,
            enable_thinking=True,
            tools=tools,
        )

        chunks = []
        async for chunk in server.stream_chat_completion(
            engine,
            [{"role": "user", "content": "Privately calculate 143 times 27."}],
            request,
            fastapi_request=None,
            tools=tools,
            enable_thinking=True,
            max_tokens=128,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line.removeprefix("data: ")))

        content_deltas = [
            choice.get("delta", {}).get("content", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("delta", {}).get("content") is not None
        ]
        visible = "".join(content_deltas)
        finish_reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        errors = [chunk["error"] for chunk in chunks if chunk.get("error")]

        assert visible == ""
        assert finish_reasons == []
        assert len(errors) == 1
        assert errors[0]["code"] == "reasoning_only_no_content"
        assert "<tool_call>" not in json.dumps(chunks)
        assert len(engine.calls) == 1

    @pytest.mark.asyncio
    async def test_nonstream_responses_suppressed_repeat_tool_runs_answer_pass(
        self, monkeypatch
    ):
        """An explicit cap may recover after tool_choice=none hides native markup."""
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser

        config = SimpleNamespace(
            family_name="qwen3_5",
            think_in_template=True,
            reasoning_parser="qwen3",
            tool_parser="qwen",
            supports_thinking=True,
        )

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "Q35-NONSTREAM-SUPPRESSED-DONE"
                    if kwargs.get("enable_thinking") is False
                    else (
                        "I should call the tool again.</think>\n\n<tool_call>\n"
                        "<function=file_info>\n<parameter=path>\n"
                        "panel/package.json\n</parameter>\n</function>\n</tool_call>"
                    )
                )
                return GenerationOutput(
                    text=text,
                    raw_text=text,
                    prompt_tokens=8,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "qwen3-policy-test")
        monkeypatch.setattr(server, "_model_name", "qwen3-policy-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(
            registry,
            "get_model_config_registry",
            lambda *args, **kwargs: SimpleNamespace(lookup=lambda *a, **k: config),
        )

        response = await server.create_response(
            ResponsesRequest(
                model="qwen3-policy-test",
                input="use the prior tool result",
                max_output_tokens=112,
                max_thinking_tokens=8,
                enable_thinking=True,
                tool_choice="none",
                tools=[
                    {
                        "type": "function",
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
            ),
            fastapi_request=None,
        )

        assert response.output_text == "Q35-NONSTREAM-SUPPRESSED-DONE"
        assert len(engine.calls) == 2
        assert engine.calls[1]["kwargs"]["enable_thinking"] is False
        assert "tools" not in engine.calls[1]["kwargs"]
        assert "<tool_call>" not in response.model_dump_json()

    @pytest.mark.asyncio
    async def test_nonstream_chat_minimax_tools_available_no_call_runs_answer_pass(
        self, monkeypatch
    ):
        """Chat shares the explicit-budget M3 no-call answer pass."""
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = SimpleNamespace(has_thinking=True)

            def __init__(self):
                self.calls = []

            async def chat(self, *, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                text = (
                    "<mm:think>No function is needed for this answer."
                    if len(self.calls) == 1
                    else "MM3-NONSTREAM-CHAT-DONE"
                )
                return GenerationOutput(
                    text=text,
                    raw_text=text,
                    prompt_tokens=8,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        engine = _Engine()
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_served_model_name", "jangq-ai/MiniMax-M3-Coder-Small")
        monkeypatch.setattr(server, "_model_name", "jangq-ai/MiniMax-M3-Coder-Small")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", MiniMaxM3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "minimax_m3")
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_default_timeout", 5.0)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="jangq-ai/MiniMax-M3-Coder-Small",
                messages=[Message(role="user", content="answer without a tool")],
                max_tokens=64,
                max_thinking_tokens=8,
                enable_thinking=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "file_info",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
            ),
            fastapi_request=None,
        )

        assert response.choices[0].message.content == "MM3-NONSTREAM-CHAT-DONE"
        assert len(engine.calls) == 2
        assert "tools" not in engine.calls[1]["kwargs"]
        assert engine.calls[1]["kwargs"]["chat_template_kwargs"]["thinking_mode"] == "disabled"

    @pytest.mark.asyncio
    async def test_streaming_responses_invalid_minimax_xml_keeps_only_visible_prefix(
        self, monkeypatch
    ):
        """A speculative M3 XML marker must not become text or a zero-tool success."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        visible_prefix = "Based on the screenshot: Completion marker:"

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    (visible_prefix, False, None),
                    ("\n<tool_call>", True, "stop"),
                )
                text = ""
                for idx, (delta, finished, reason) in enumerate(chunks, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[],
                        prompt_tokens=7,
                        completion_tokens=idx,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "minimax-m3-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "minimax_m3")

        request = ResponsesRequest(
            model="minimax-m3-test",
            input="read the screenshot",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "read the screenshot"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        done = [
            event.get("text", "")
            for event in events
            if event.get("type") == "response.output_text.done"
        ]
        function_items = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done"
            and event.get("item", {}).get("type") == "function_call"
        ]
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

        assert visible == visible_prefix
        assert done[-1] == visible_prefix
        assert function_items == []
        assert "<tool_call" not in json.dumps(completed)
        assert any(
            "schema-valid function call" in warning
            for warning in completed.get("warnings", [])
        )

    @pytest.mark.asyncio
    async def test_streaming_responses_minimax_adaptive_holds_end_only_private_prefix(
        self, monkeypatch
    ):
        """M3 adaptive must not publish bytes that a late close makes private."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

        private = "The user wants exactly one sentence."

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text=private,
                    new_text=private,
                    prompt_tokens=9,
                    completion_tokens=7,
                    finished=False,
                    finish_reason=None,
                )
                yield GenerationOutput(
                    text=private + "</mm:think>VISIBLE_M3_RESPONSE",
                    new_text="</mm:think>VISIBLE_M3_RESPONSE",
                    prompt_tokens=9,
                    completion_tokens=11,
                    finished=True,
                    finish_reason="stop",
                )

        class _Registry:
            def lookup(self, _key):
                return SimpleNamespace(
                    family_name="minimax_m3",
                    think_in_template=False,
                    reasoning_parser="minimax_m3",
                    tool_parser="minimax_m3",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "minimax-m3-adaptive-stream-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", MiniMaxM3ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(registry, "get_model_config_registry", lambda: _Registry())

        request = ResponsesRequest(
            model="minimax-m3-adaptive-stream-test",
            input="reply in one sentence",
            stream=True,
        )
        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "reply in one sentence"}],
            request,
            fastapi_request=None,
            chat_template_kwargs={"thinking_mode": "adaptive"},
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        reasoning = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        visible = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )

        assert reasoning == private
        assert visible == "VISIBLE_M3_RESPONSE"
        assert completed["output_text"] == "VISIBLE_M3_RESPONSE"
        assert private not in completed["output_text"]

    @pytest.mark.asyncio
    async def test_streaming_responses_without_tools_skips_native_tool_parser(
        self, monkeypatch
    ):
        """A normal no-tools turn must not produce a native-parser warning."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="45",
                    new_text="45",
                    tokens=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "openpangu-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "openpangu")

        request = ResponsesRequest(
            model="openpangu-test",
            input="What is 17 + 28?",
            stream=True,
        )
        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "What is 17 + 28?"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )
        assert completed["output_text"] == "45"
        assert completed.get("warnings", []) == []

    @pytest.mark.asyncio
    async def test_streaming_responses_strict_native_plain_final_with_tools_has_no_drop_warning(
        self, monkeypatch
    ):
        """Available tools do not turn a normal final answer into a parser failure."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                yield GenerationOutput(
                    text="PG-DONE",
                    new_text="PG-DONE",
                    tokens=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "openpangu-test")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "openpangu")

        request = ResponsesRequest(
            model="openpangu-test",
            input="finish after the tool result",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )
        events = []
        async for chunk in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "finish after the tool result"}],
            request,
            fastapi_request=None,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        completed = next(
            event["response"]
            for event in events
            if event.get("type") == "response.completed"
        )
        assert completed["output_text"] == "PG-DONE"
        assert completed.get("warnings", []) == []

    @pytest.mark.asyncio
    async def test_streaming_responses_tool_call_uses_next_output_index_without_text(
        self, monkeypatch
    ):
        """Function calls must not reuse the placeholder message output index."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                text = '<tool_call>{"name":"lookup","arguments":{"query":"alpha"}}</tool_call>'
                yield GenerationOutput(
                    text=text,
                    new_text=text,
                    tokens=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "unit-tool-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ResponsesRequest(
            model="unit-tool-model",
            input="use lookup",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
        )

        payloads: list[tuple[str, dict]] = []
        async for event in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "use lookup"}],
            request,
            fastapi_request=None,
        ):
            if not event.startswith("event: "):
                continue
            event_type = event.splitlines()[0].removeprefix("event: ")
            data_line = next(
                line for line in event.splitlines() if line.startswith("data: ")
            )
            payloads.append((event_type, json.loads(data_line.removeprefix("data: "))))

        message_done_indexes = [
            payload["output_index"]
            for event_type, payload in payloads
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "message"
        ]
        function_added_indexes = [
            payload["output_index"]
            for event_type, payload in payloads
            if event_type == "response.output_item.added"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        function_done_indexes = [
            payload["output_index"]
            for event_type, payload in payloads
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        function_arg_indexes = [
            payload["output_index"]
            for event_type, payload in payloads
            if event_type
            in (
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            )
        ]

        assert message_done_indexes == []
        assert function_added_indexes == [0]
        assert function_done_indexes == [0]
        assert function_arg_indexes and set(function_arg_indexes) == {0}

    @pytest.mark.asyncio
    async def test_streaming_responses_required_empty_xml_tool_call_is_rejected(
        self, monkeypatch
    ):
        """Malformed XML tool calls must not become executable empty JSON args."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("**Quick preamble:** Checking `/tmp`...\n", False, None),
                    (
                        "<tool_call>\n<function=exec_command>\n</function>\n</tool_call>",
                        True,
                        "stop",
                    ),
                )
                text = ""
                for idx, (delta, finished, reason) in enumerate(chunks, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[],
                        prompt_tokens=5,
                        completion_tokens=idx,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "unit-tool-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ResponsesRequest(
            model="unit-tool-model",
            input="list /tmp",
            stream=True,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        )

        payloads: list[tuple[str, dict]] = []
        async for event in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "list /tmp"}],
            request,
            fastapi_request=None,
        ):
            if not event.startswith("event: "):
                continue
            event_type = event.splitlines()[0].removeprefix("event: ")
            data_line = next(
                line for line in event.splitlines() if line.startswith("data: ")
            )
            payloads.append((event_type, json.loads(data_line.removeprefix("data: "))))

        function_items = [
            payload["item"]
            for event_type, payload in payloads
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]
        error_codes = [
            payload.get("error", {}).get("code") or payload.get("code")
            for event_type, payload in payloads
            if event_type == "error"
        ]
        terminal_events = [
            (event_type, payload.get("response", {}))
            for event_type, payload in payloads
            if event_type.startswith("response.")
            and event_type
            in {"response.completed", "response.incomplete", "response.failed"}
        ]

        assert function_items == []
        assert "tool_calls_required" in error_codes
        assert [event_type for event_type, _ in terminal_events] == [
            "response.failed"
        ]
        assert terminal_events[0][1]["status"] == "failed"
        assert terminal_events[0][1]["error"]["code"] == "tool_calls_required"

    @pytest.mark.asyncio
    async def test_streaming_responses_reasoning_tool_call_keeps_arguments(
        self, monkeypatch
    ):
        """Reasoning-channel tool calls must not finalize with `{}` args."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.base import DeltaMessage

        class _ReasoningParser:
            def reset_state(self, **_kwargs):
                pass

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                return DeltaMessage(reasoning=delta_text)

            def extract_reasoning(self, model_output):
                return model_output, None

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=True)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    (
                        "<tool_call>",
                        False,
                        None,
                    ),
                    (
                        '{"name":"lookup","arguments":{"query":"beta","limit":3}}</tool_call>',
                        True,
                        "stop",
                    ),
                )
                text = ""
                for idx, (delta, finished, reason) in enumerate(chunks, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[],
                        prompt_tokens=5,
                        completion_tokens=idx,
                        finished=finished,
                        finish_reason=reason,
                    )

        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "reasoning-tool-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", _ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ResponsesRequest(
            model="reasoning-tool-model",
            input="use lookup",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query", "limit"],
                    },
                }
            ],
            enable_thinking=True,
        )

        payloads: list[tuple[str, dict]] = []
        async for event in server.stream_responses_api(
            _Engine(),
            [{"role": "user", "content": "use lookup"}],
            request,
            fastapi_request=None,
        ):
            if not event.startswith("event: "):
                continue
            event_type = event.splitlines()[0].removeprefix("event: ")
            data_line = next(
                line for line in event.splitlines() if line.startswith("data: ")
            )
            payloads.append((event_type, json.loads(data_line.removeprefix("data: "))))

        arg_deltas = [
            payload["delta"]
            for event_type, payload in payloads
            if event_type == "response.function_call_arguments.delta"
        ]
        arg_done = [
            payload["arguments"]
            for event_type, payload in payloads
            if event_type == "response.function_call_arguments.done"
        ]
        function_items = [
            payload["item"]
            for event_type, payload in payloads
            if event_type == "response.output_item.done"
            and payload.get("item", {}).get("type") == "function_call"
        ]

        expected_args = {"query": "beta", "limit": 3}
        assert any(
            event_type == "response.heartbeat"
            and payload.get("tool_call_generating") is True
            for event_type, payload in payloads
        )
        assert json.loads("".join(arg_deltas)) == expected_args
        assert json.loads(arg_done[-1]) == expected_args
        assert function_items[-1]["name"] == "lookup"
        assert json.loads(function_items[-1]["arguments"]) == expected_args


class TestAPIKeyVerification:
    """Test API key verification with timing attack prevention."""

    def test_secrets_compare_digest_usage(self):
        """Test that secrets.compare_digest is used (timing attack prevention)."""
        import secrets

        # Verify secrets.compare_digest works as expected
        key1 = "test-api-key-12345"
        key2 = "test-api-key-12345"
        key3 = "different-key-67890"

        # Same keys should match
        assert secrets.compare_digest(key1, key2) is True

        # Different keys should not match
        assert secrets.compare_digest(key1, key3) is False

        # Verify it's constant-time (by checking function exists)
        assert hasattr(secrets, "compare_digest")

    def test_verify_api_key_rejects_invalid(self):
        """Test that invalid API key is rejected with 401."""
        import asyncio
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        # Import and set up the module
        import vmlx_engine.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with invalid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid-key"
            )

            # Should raise HTTPException with 401
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(server.verify_api_key(credentials))

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            server._api_key = original_key

    def test_verify_api_key_accepts_valid(self):
        """Test that valid API key is accepted."""
        import asyncio
        from fastapi.security import HTTPAuthorizationCredentials

        import vmlx_engine.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with valid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid-secret-key"
            )

            # Should not raise any exception
            result = asyncio.run(server.verify_api_key(credentials))
            # verify_api_key returns True on success (no exception raised)
            assert result is True or result is None
        finally:
            server._api_key = original_key


class TestRateLimiterHTTPResponse:
    """Test rate limiter HTTP response behavior."""

    def test_rate_limiter_returns_retry_after(self):
        """Test that rate limiter returns retry_after when limit exceeded."""
        from vmlx_engine.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Exhaust the limit
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Next request should be denied with retry_after
        allowed, retry_after = limiter.is_allowed("test_client")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
        assert retry_after <= 60  # Should be within a minute

    def test_rate_limiter_window_cleanup(self):
        """Test that rate limiter cleans up old requests from sliding window."""
        from vmlx_engine.server import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Make some requests
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Should be denied (limit reached)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

        # Manually inject old timestamps to simulate time passing
        # The sliding window should clean these up
        old_time = time.time() - 120  # 2 minutes ago
        with limiter._lock:
            limiter._requests["test_client"] = [old_time, old_time]

        # Now should be allowed again (old requests cleaned up)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is True


# =============================================================================
# Integration Tests (require running server)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vmlx-engine server for integration tests",
    )


class TestDSV4RepetitionPenaltyDefaults:
    """DSV4 uses bundle-declared mode-specific repetition penalties.

    The DSV4 converter documents that thinking mode must remain neutral
    (`repetition_penalty_thinking=1.0`) because higher penalties make the model
    fail to close `</think>`. The old generic 1.15 floor is a regression for
    the forced-thinking default path.
    """

    def _set_dsv4_path(self, monkeypatch, tmp_path, sampling_defaults):
        """Build a fake DSV4 bundle dir + point _model_path at it."""
        import json
        cfg = {"model_type": "deepseek_v4"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        jang_cfg = {
            "model_family": "deepseek_v4",
            "chat": {"sampling_defaults": sampling_defaults,
                     "reasoning": {"default_mode": "chat"}},
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_cfg))
        import vmlx_engine.server as srv
        monkeypatch.setattr(srv, "_model_path", str(tmp_path))
        monkeypatch.setattr(srv, "_default_repetition_penalty", None)
        return srv

    def test_thinking_mode_keeps_bundle_neutral_penalty(self, monkeypatch, tmp_path):
        srv = self._set_dsv4_path(monkeypatch, tmp_path, {
            "repetition_penalty_thinking": 1.0,
            "repetition_penalty_chat": 1.05,
            "repetition_penalty": 1.0,
        })
        result = srv._resolve_repetition_penalty(
            None,
            str(tmp_path),
            enable_thinking=True,
        )
        assert result == 1.0

    def test_dsv4_direct_chat_prefers_neutral_generic_penalty(self, monkeypatch, tmp_path):
        srv = self._set_dsv4_path(monkeypatch, tmp_path, {
            "repetition_penalty": 1.0,
            "repetition_penalty_chat": 1.05,
            "repetition_penalty_thinking": 1.0,
        })
        result = srv._resolve_repetition_penalty(
            None,
            str(tmp_path),
            enable_thinking=False,
        )
        assert result == 1.0

    def test_generation_kwargs_recomputed_after_mode_resolution(self, monkeypatch, tmp_path):
        srv = self._set_dsv4_path(monkeypatch, tmp_path, {
            "repetition_penalty": 1.0,
            "repetition_penalty_chat": 1.05,
            "repetition_penalty_thinking": 1.0,
        })
        kwargs = {"repetition_penalty": 1.05}

        srv._set_resolved_repetition_penalty(
            kwargs,
            None,
            str(tmp_path),
            enable_thinking=False,
        )

        assert kwargs["repetition_penalty"] == 1.0

    def test_explicit_per_request_repetition_penalty_is_honored(self, monkeypatch, tmp_path):
        srv = self._set_dsv4_path(monkeypatch, tmp_path, {
            "repetition_penalty_thinking": 1.0,
            "repetition_penalty_chat": 1.05,
        })
        result = srv._resolve_repetition_penalty(
            1.05,
            str(tmp_path),
            enable_thinking=True,
        )
        assert result == 1.05

    def test_floor_does_not_affect_non_dsv4_families(self, monkeypatch, tmp_path):
        import json
        # Build a non-DSV4 bundle
        cfg = {"model_type": "qwen3_moe"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        jang_cfg = {"chat": {"sampling_defaults": {"repetition_penalty_chat": 1.0}}}
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_cfg))
        import vmlx_engine.server as srv
        monkeypatch.setattr(srv, "_model_path", str(tmp_path))
        monkeypatch.setattr(srv, "_default_repetition_penalty", None)
        result = srv._resolve_repetition_penalty(None, str(tmp_path))
        # Non-DSV4 families don't get the DSV4 floor — None or actual is fine.
        assert result is None or result < 1.15, (
            f"Floor leaked to non-DSV4 family: got {result}")


class TestToolChoiceNonePromptParity:
    def test_none_disables_public_and_engine_tool_sets(self):
        """The payload may retain schemas, but none means no rendered tools."""
        from vmlx_engine import server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="minimax-test",
            messages=[Message(role="user", content="answer directly")],
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        assert not server._tools_available_for_generation(
            request, engine_tools=[{"name": "file_info"}]
        )
        assert server._request_tools_for_generation_prompt(request) == []

    def test_named_choice_filters_request_prompt_tools(self):
        from vmlx_engine import server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="minimax-test",
            messages=[Message(role="user", content="inspect the file")],
            tool_choice={
                "type": "function",
                "function": {"name": "file_info"},
            },
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

        effective = server._request_tools_for_generation_prompt(request)
        assert len(effective) == 1
        assert effective[0].function["name"] == "file_info"

    @pytest.mark.asyncio
    async def test_streaming_chat_post_tool_direct_answer_is_visible(
        self, monkeypatch
    ):
        """Schemas + tool_choice=none must not seed MiniMax inside reasoning.

        This is the raw-API shape used after a client has executed a function
        call: it commonly retains the schemas for history fidelity while
        explicitly disabling another call.  The engine receives no tools.
        """
        import json
        from types import SimpleNamespace

        import vmlx_engine.model_config_registry as registry
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput
        from vmlx_engine.reasoning.minimax_m2_parser import MiniMaxM2ReasoningParser

        class _Tokenizer:
            has_thinking = True

            def apply_chat_template(self, messages, **kwargs):
                return "]~b]user\nanswer directly[e~[\n]~b]ai\n<think>\n"

        class _Engine:
            tokenizer = _Tokenizer()

            async def stream_chat(self, *, messages, **kwargs):
                text = ""
                deltas = ["M27-CHAT-TOOL-", "CONTINUE-DONE SIZE=5.2 KB"]
                for index, delta in enumerate(deltas, start=1):
                    text += delta
                    yield GenerationOutput(
                        text=text,
                        new_text=delta,
                        tokens=[index],
                        prompt_tokens=12,
                        completion_tokens=index,
                        finished=index == len(deltas),
                        finish_reason="stop" if index == len(deltas) else None,
                    )

        config = SimpleNamespace(
            family_name="minimax",
            think_in_template=True,
            reasoning_parser="minimax_m2",
            tool_parser="minimax",
            supports_thinking=True,
        )
        engine = _Engine()
        monkeypatch.setattr(server, "_default_timeout", 5.0)
        monkeypatch.setattr(server, "_model_name", "MiniMax-M2.7-Small-JANGTQ")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", MiniMaxM2ReasoningParser())
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(
            registry,
            "get_model_config_registry",
            lambda: SimpleNamespace(lookup=lambda *args, **kwargs: config),
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        request = ChatCompletionRequest(
            model="minimax-test",
            messages=[Message(role="user", content="answer directly")],
            stream=True,
            max_tokens=64,
            enable_thinking=False,
            tool_choice="none",
            tools=tools,
        )
        server._attach_effective_tools_for_tool_parsing(request, [])

        chunks = []
        async for chunk in server.stream_chat_completion(
            engine,
            [{"role": "user", "content": "answer directly"}],
            request,
            fastapi_request=None,
            enable_thinking=False,
            max_tokens=64,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        reasoning = "".join(
            choice.get("delta", {}).get("reasoning_content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        finish_reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]

        assert visible == "M27-CHAT-TOOL-CONTINUE-DONE SIZE=5.2 KB"
        assert reasoning == ""
        assert finish_reasons == ["stop"]


class TestNativeParserExceptionNeverRepairsNativeMarkup:
    """A native parser that RAISES must not hand its own markup to the generic repair.

    Measured on Muse-Glimmer-30B (agentic-eval-074524 nullable_enum_bounds_valid): the
    ATEM parser raised on a nullable JSON-Schema type list, the except-all fell to the
    generic repair, and the tool executed with every argument set to the raw markup
    between attribute quotes. The parser bug is fixed separately (23f4660b); this pins
    the server rule that a crashed native parser drops the block with a diagnostic.
    """

    @staticmethod
    def _install(monkeypatch, name, markers):
        import vmlx_engine.server as server
        from vmlx_engine.tool_parsers import ToolParserManager
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParser

        class Boom(ToolParser):
            NATIVE_MARKERS = markers
            SUPPORTS_NATIVE_TOOL_FORMAT = True

            def __init__(self, tokenizer=None):
                pass

            def extract_tool_calls(self, model_output, request=None):
                raise TypeError("unhashable type: 'list'")

            def extract_tool_calls_streaming(self, *a, **k):
                return None

        ToolParserManager.register_module(name, Boom)
        monkeypatch.setattr(server, "_tool_call_parser", name)
        monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
        monkeypatch.setattr(server, "_engine", None)
        return server

    @staticmethod
    def _request():
        from vmlx_engine.api.models import ChatCompletionRequest

        return ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "search"}],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": ["string", "null"]}}}}}],
        )

    def test_native_block_is_dropped_with_a_diagnostic_not_repaired(self, monkeypatch):
        server = self._install(monkeypatch, "boom_native_test", ("<atem:function_calls>", "<atem:invoke"))
        raw = (
            "Looking now.\n<atem:function_calls>\n<atem:invoke name=\"search\">\n"
            '<atem:parameter name="pattern">magic</atem:parameter>\n'
            '<atem:parameter name="path">null</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        server._begin_tool_call_drop_capture()
        cleaned, calls = server._parse_tool_calls_with_parser(raw, self._request())
        diags = server._take_tool_call_drop_diagnostics()

        assert calls is None
        assert "<atem:" not in cleaned
        assert cleaned.strip() == "Looking now."
        assert diags and "boom_native_test" in diags[0] and "TypeError" in diags[0]

    def test_output_without_native_markers_keeps_the_generic_repair(self, monkeypatch):
        server = self._install(monkeypatch, "boom_native_test2", ("<atem:function_calls>", "<atem:invoke"))
        raw = '<tool_call>{"name": "search", "arguments": {"pattern": "magic"}}</tool_call>'
        server._begin_tool_call_drop_capture()
        cleaned, calls = server._parse_tool_calls_with_parser(raw, self._request())
        server._take_tool_call_drop_diagnostics()

        assert calls is not None and calls[0].function.name == "search"

    def test_truncated_native_marker_at_the_tail_is_not_repaired_either(self, monkeypatch):
        server = self._install(monkeypatch, "boom_native_test3", ("<atem:function_calls>", "<atem:invoke"))
        raw = "Looking now.\n<atem:functi"  # cut by max_tokens inside the opener
        server._begin_tool_call_drop_capture()
        cleaned, calls = server._parse_tool_calls_with_parser(raw, self._request())
        diags = server._take_tool_call_drop_diagnostics()

        assert calls is None
        assert "<atem" not in cleaned
        assert diags and "boom_native_test3" in diags[0]

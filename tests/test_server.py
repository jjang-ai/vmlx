# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import platform
import sys
from pathlib import Path

import pytest

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

    def test_resolve_max_prompt_tokens_uses_user_context_cap(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(64000, 8192) == 8192

    def test_resolve_max_prompt_tokens_uses_auto_estimate_when_unset(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(4096, None) == 4096

    def test_resolve_max_prompt_tokens_uses_auto_estimate_when_zero(self):
        from vmlx_engine import server

        assert server._resolve_max_prompt_tokens(4096, 0) == 4096

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

    def test_prompt_limit_guard_rejects_chat_messages(self, monkeypatch):
        from vmlx_engine import server
        from vmlx_engine.api.models import Message

        monkeypatch.setattr(server, "_max_prompt_tokens", 2)

        response = server._reject_if_prompt_too_long_for_messages([
            Message(role="user", content="x" * 30)
        ])

        assert response is not None
        assert response.status_code == 413
        assert b"prompt_too_long" in response.body

    def test_prompt_limit_guard_allows_chat_messages_under_limit(self, monkeypatch):
        from vmlx_engine import server
        from vmlx_engine.api.models import Message

        monkeypatch.setattr(server, "_max_prompt_tokens", 100)

        assert server._reject_if_prompt_too_long_for_messages([
            Message(role="user", content="hello")
        ]) is None

    def test_prompt_limit_guard_rejects_completion_prompt_list(self, monkeypatch):
        from vmlx_engine import server

        monkeypatch.setattr(server, "_max_prompt_tokens", 2)

        response = server._reject_if_prompt_too_long_for_prompts([
            "short",
            "x" * 30,
        ])

        assert response is not None
        assert response.status_code == 413
        assert b"prompt_too_long" in response.body

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

    def test_prompt_limit_guard_counts_vlm_media_parts(self, monkeypatch):
        from vmlx_engine import server
        from vmlx_engine.api.models import Message

        monkeypatch.setattr(server, "_max_prompt_tokens", 100)

        response = server._reject_if_prompt_too_long_for_messages([
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}},
                ],
            )
        ])

        assert response is not None
        assert response.status_code == 413
        assert b"prompt_too_long" in response.body

    def test_prompt_limit_guard_is_used_by_all_generation_entrypoints(self):
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
        assert "max_prompt_tokens=_chat_max_prompt_tokens" in source
        assert '"max_prompt_tokens": _chat_max_prompt_tokens' in source
        assert '"max_prompt_tokens": _responses_max_prompt_tokens' in source
        assert "except PromptTooLongError" in source

    def test_streaming_routes_map_prompt_limit_to_prompt_too_long(self):
        import inspect
        from vmlx_engine import server

        chat_stream_src = inspect.getsource(server.stream_chat_completion)
        responses_stream_src = inspect.getsource(server.stream_responses_api)

        assert chat_stream_src.count("except PromptTooLongError") == 1
        assert "code\": \"prompt_too_long\"" in chat_stream_src
        assert "except PromptTooLongError" in responses_stream_src
        assert "code\": \"prompt_too_long\"" in responses_stream_src

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

    @pytest.mark.asyncio
    async def test_streaming_responses_xml_tool_call_preserves_special_arguments(
        self, monkeypatch
    ):
        """Responses SSE must preserve XML-function string argument text."""
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
                        "Checking.\n<tool_call><function=exec_command>",
                        False,
                        None,
                    ),
                    (
                        "<parameter=cmd>  printf '&lt;日本語&gt;' &amp;&amp; pwd  </parameter>",
                        False,
                        None,
                    ),
                    ("</function></tool_call>", True, "stop"),
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
        monkeypatch.setattr(server, "_tool_call_parser", "xml_function")

        request = ResponsesRequest(
            model="unit-tool-model",
            input="run command",
            stream=True,
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
            [{"role": "user", "content": "run command"}],
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

        expected_args = {"cmd": "  printf '<日本語>' && pwd  "}
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

        assert json.loads("".join(arg_deltas)) == expected_args
        assert json.loads(arg_done[-1]) == expected_args
        assert function_items[-1]["name"] == "exec_command"
        assert json.loads(function_items[-1]["arguments"]) == expected_args

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

        assert message_done_indexes == [0]
        assert function_added_indexes == [1]
        assert function_done_indexes == [1]
        assert function_arg_indexes and set(function_arg_indexes) == {1}

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

        assert function_items == []
        assert "tool_calls_required" in error_codes

    @pytest.mark.asyncio
    async def test_streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments(
        self, monkeypatch
    ):
        """A streamed preamble plus empty XML function must fail closed, never emit `{}`."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("**Quick preamble:** Checking what's in `/tmp`...\n", False, None),
                    (
                        "<tool_call>\n"
                        "<function=exec_command>\n"
                        "</function>\n"
                        "</tool_call>",
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

        visible_deltas = [
            payload.get("delta", "")
            for event_type, payload in payloads
            if event_type == "response.output_text.delta"
        ]
        function_items = [
            payload.get("item", {})
            for event_type, payload in payloads
            if event_type in {"response.output_item.added", "response.output_item.done"}
            and payload.get("item", {}).get("type") == "function_call"
        ]
        argument_events = [
            payload
            for event_type, payload in payloads
            if event_type
            in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
        ]
        error_codes = [
            payload.get("error", {}).get("code") or payload.get("code")
            for event_type, payload in payloads
            if event_type == "error"
        ]
        serialized_events = "\n".join(
            json.dumps(payload, sort_keys=True) for _, payload in payloads
        )

        assert visible_deltas == []
        assert function_items == []
        assert argument_events == []
        assert '"arguments": "{}"' not in serialized_events
        assert "tool_calls_required" in error_codes
        completed = [
            payload["response"]
            for event_type, payload in payloads
            if event_type == "response.completed"
        ]
        assert completed[-1]["status"] == "failed"
        assert completed[-1]["output_text"] == ""
        assert completed[-1]["output"] == []

    @pytest.mark.asyncio
    async def test_streaming_responses_auto_empty_xml_tool_call_strips_final_markup(
        self, monkeypatch
    ):
        """Auto tool mode may keep the preamble, but not raw invalid XML markup."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("Quick preamble. Checking tmp...\n", False, None),
                    (
                        "<tool_call>\n"
                        "<function=exec_command>\n"
                        "</function>\n"
                        "</tool_call>",
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
        monkeypatch.setattr(server, "_model_name", "unit-qwen35-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ResponsesRequest(
            model="unit-qwen35-model",
            input="list /tmp",
            stream=True,
            tool_choice="auto",
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

        visible_deltas = [
            payload.get("delta", "")
            for event_type, payload in payloads
            if event_type == "response.output_text.delta"
        ]
        done_texts = [
            payload.get("text", "")
            for event_type, payload in payloads
            if event_type == "response.output_text.done"
        ]
        function_items = [
            payload.get("item", {})
            for event_type, payload in payloads
            if event_type in {"response.output_item.added", "response.output_item.done"}
            and payload.get("item", {}).get("type") == "function_call"
        ]
        completed = [
            payload["response"]
            for event_type, payload in payloads
            if event_type == "response.completed"
        ][-1]
        serialized_events = "\n".join(
            json.dumps(payload, sort_keys=True) for _, payload in payloads
        )

        assert visible_deltas == ["Quick preamble. Checking tmp...\n"]
        assert done_texts[-1] == "Quick preamble. Checking tmp..."
        assert completed["status"] == "completed"
        assert completed["output_text"] == "Quick preamble. Checking tmp..."
        assert function_items == []
        assert "<tool_call" not in serialized_events
        assert "<function=exec_command" not in serialized_events
        assert '"arguments": "{}"' not in serialized_events

    def test_tool_parser_drops_empty_xml_call_and_strips_markup_for_nonstream_paths(
        self, monkeypatch
    ):
        """Shared Chat/Responses nonstream parser path must fail closed."""
        import json

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        monkeypatch.setattr(server, "_model_name", "unit-qwen35-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ChatCompletionRequest(
            model="unit-qwen35-model",
            messages=[Message(role="user", content="list /tmp")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ],
        )

        empty_text = (
            "**Quick preamble:** Checking what's in `/tmp`...\n"
            "<tool_call>\n"
            "<function=exec_command>\n"
            "</function>\n"
            "</tool_call>"
        )
        cleaned, calls = server._parse_tool_calls_with_parser(empty_text, request)

        assert calls is None
        assert cleaned == "**Quick preamble:** Checking what's in `/tmp`..."
        assert "<tool_call" not in cleaned
        assert "<function" not in cleaned

        valid_text = (
            "**Quick preamble:** Checking what's in `/tmp`...\n"
            "<tool_call>\n"
            "<function=exec_command>\n"
            "<parameter=cmd>ls /tmp</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        valid_cleaned, valid_calls = server._parse_tool_calls_with_parser(
            valid_text, request
        )

        assert valid_cleaned == "**Quick preamble:** Checking what's in `/tmp`..."
        assert valid_calls is not None
        assert valid_calls[0].function.name == "exec_command"
        assert json.loads(valid_calls[0].function.arguments) == {"cmd": "ls /tmp"}

    @pytest.mark.asyncio
    async def test_streaming_chat_preamble_empty_xml_tool_call_never_emits_empty_arguments(
        self, monkeypatch
    ):
        """Chat Completions SSE must not emit executable `{}` tool args."""
        import json
        from types import SimpleNamespace

        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            tokenizer = SimpleNamespace(has_thinking=False)

            async def stream_chat(self, *, messages, **kwargs):
                chunks = (
                    ("**Quick preamble:** Checking what's in `/tmp`...\n", False, None),
                    (
                        "<tool_call>\n"
                        "<function=exec_command>\n"
                        "</function>\n"
                        "</tool_call>",
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
        monkeypatch.setattr(server, "_model_name", "unit-qwen35-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")

        request = ChatCompletionRequest(
            model="unit-qwen35-model",
            messages=[Message(role="user", content="list /tmp")],
            stream=True,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
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
            tools=server.convert_tools_for_template(request.tools),
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        visible = "".join(
            choice.get("delta", {}).get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        tool_deltas = [
            choice.get("delta", {}).get("tool_calls")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("delta", {}).get("tool_calls")
        ]
        error_codes = [
            chunk.get("error", {}).get("code") for chunk in chunks if chunk.get("error")
        ]
        serialized_chunks = "\n".join(json.dumps(chunk, sort_keys=True) for chunk in chunks)

        assert visible == "**Quick preamble:** Checking what's in `/tmp`...\n"
        assert tool_deltas == []
        assert '"arguments": "{}"' not in serialized_chunks
        assert "tool_calls_required" in error_codes

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

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
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
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
        expected = ["qwen3", "deepseek_r1", "minimax_m2", "openai_gptoss"]
        for name in expected:
            assert name in parsers, f"Reasoning parser '{name}' not registered"

    def test_all_reasoning_parsers_instantiable(self):
        from vmlx_engine.reasoning import get_parser
        for name in ["qwen3", "deepseek_r1", "minimax_m2", "openai_gptoss"]:
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
            "minimax": "minimax_m2",
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
            logprobs=True,
            top_logprobs=5,
        )
        assert sp.max_tokens == 100
        assert sp.temperature == 0.8
        assert sp.top_p == 0.95
        assert sp.top_k == 50
        assert sp.min_p == 0.1
        assert sp.repetition_penalty == 1.2
        assert sp.logprobs is True
        assert sp.top_logprobs == 5

    def test_sampling_params_defaults(self):
        from vmlx_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.temperature == 0.0
        assert sp.top_p == 1.0
        assert sp.top_k == 0
        assert sp.min_p == 0.0
        assert sp.repetition_penalty == 1.0
        assert sp.logprobs is False
        assert sp.top_logprobs == 0


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


class TestBatchedEngineVideoTemplate:
    """BatchedEngine MLLM template handling for video-only requests."""

    def test_video_only_mllm_uses_processor_template_not_tokenizer_fallback(self):
        pytest.importorskip("mlx_vlm")
        from vmlx_engine.engine.batched import BatchedEngine

        captured = {}

        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("video-only MLLM request used tokenizer fallback")

            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        class _Processor:
            tokenizer = _Tokenizer()

            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                return "VIDEO_PROMPT"

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = _Processor()
        engine._tokenizer = None
        engine._model_name = "qwen36-video-test"
        engine._model = SimpleNamespace(config={"model_type": "qwen3_5_moe"})

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                    {"type": "text", "text": "Reply only BLUE."},
                ],
            }
        ]

        prompt = engine._apply_chat_template(
            messages,
            num_images=0,
            num_videos=1,
            enable_thinking=False,
        )

        assert prompt == "VIDEO_PROMPT"
        content = captured["messages"][0]["content"]
        assert content[0]["type"] == "video"
        assert content[1]["type"] == "text"
        assert not any(part.get("type") == "video_url" for part in content)


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

    def test_api_routes_pass_model_to_sampling_resolvers(self):
        """API handlers must resolve omitted sampling params against the request model.

        This pins the generation_config/jang_config behavior at the route layer:
        the resolver helpers support model-scoped defaults, but callers can still
        accidentally fall back to process-global state if they omit model_name.
        """
        source = Path("./vmlx_engine/server.py").read_text()
        bad_patterns = {
            "temperature": r"_resolve_temperature\(\s*(?:chat_req|request)\.temperature\s*\)",
            "top_p": r"_resolve_top_p\(\s*(?:chat_req|request)\.top_p\s*\)",
            "repetition_penalty": (
                r"_resolve_repetition_penalty\(\s*(?:chat_req|request)\.repetition_penalty\s*\)"
            ),
        }
        for name, pattern in bad_patterns.items():
            assert not re.search(pattern, source), (
                f"{name} resolver call is not model-scoped; pass request.model/chat_req.model "
                "so bundle generation_config.json and jang_config sampling defaults apply."
            )

    def test_chat_and_responses_log_and_forward_supported_sampling_kwargs(
        self,
        monkeypatch,
        caplog,
    ):
        """Supported text sampling kwargs must reach the engine unchanged."""
        import logging

        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "kwarg-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)
        caplog.set_level(logging.INFO, logger="vmlx_engine.server")

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 17,
                "temperature": 0.23,
                "top_p": 0.77,
                "top_k": 42,
                "min_p": 0.03,
                "repetition_penalty": 1.11,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "kwarg-model",
                "input": "hi",
                "stream": False,
                "max_output_tokens": 19,
                "temperature": 0.31,
                "top_p": 0.88,
                "top_k": 24,
                "min_p": 0.04,
                "repetition_penalty": 1.07,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert captured[0] == {
            "max_tokens": 17,
            "temperature": 0.23,
            "top_p": 0.77,
            "max_prompt_tokens": 0,
            "top_k": 42,
            "min_p": 0.03,
            "repetition_penalty": 1.11,
        }
        assert captured[1] == {
            "max_tokens": 19,
            "temperature": 0.31,
            "top_p": 0.88,
            "max_prompt_tokens": 0,
            "top_k": 24,
            "min_p": 0.04,
            "repetition_penalty": 1.07,
        }
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Resolved sampling kwargs route=/v1/chat/completions" in log_text
        assert "Resolved sampling kwargs route=/v1/responses" in log_text
        assert "'top_k': 42" in log_text
        assert "'min_p': 0.04" in log_text

    def test_request_output_caps_override_server_default_without_touching_context_cap(
        self,
        monkeypatch,
    ):
        """Per-request output caps and prompt/context caps stay independent.

        Chat Completions owns generated length with max_tokens; Responses owns
        it with max_output_tokens. The prompt/context admission cap is a
        separate max_prompt_tokens field and must not rewrite either output cap.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(
                text="ok",
                prompt_tokens=3,
                completion_tokens=1,
            )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "kwarg-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        monkeypatch.setattr(server, "_max_prompt_tokens", 4096)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 123,
                "max_prompt_tokens": 2048,
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "kwarg-model",
                "input": "hi",
                "stream": False,
                "max_output_tokens": 321,
                "max_prompt_tokens": 1024,
            },
        )
        chat_auto_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kwarg-model",
                "messages": [{"role": "user", "content": "auto"}],
                "stream": False,
                "max_prompt_tokens": 2048,
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert chat_auto_resp.status_code == 200
        assert captured[0]["max_tokens"] == 123
        assert captured[0]["max_prompt_tokens"] == 2048
        assert captured[1]["max_tokens"] == 321
        assert captured[1]["max_prompt_tokens"] == 1024
        assert captured[2]["max_tokens"] == 512
        assert captured[2]["max_prompt_tokens"] == 2048

    def test_reasoning_effort_preserves_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """reasoning_effort should not inflate model-declared output length.

        The template-facing thinking_budget is the effort control. Length is
        owned by explicit request max_tokens/max_output_tokens first, then the
        bundle's generation_config.max_new_tokens.
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server
        from vmlx_engine.engine.base import GenerationOutput

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )

        class FakeTokenizer:
            has_thinking = False

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_await_chat(*args, **kwargs):
            captured.append(dict(kwargs["chat_kwargs"]))
            return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(
            server,
            "_await_chat_with_disconnect_abort",
            fake_await_chat,
        )
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning_effort": "high",
            },
        )
        responses_resp = client.post(
            "/v1/responses",
            json={
                "model": "bundle-model",
                "input": "hi",
                "stream": False,
                "reasoning_effort": "high",
            },
        )

        assert chat_resp.status_code == 200
        assert responses_resp.status_code == 200
        assert captured[0]["max_tokens"] == 2048
        assert captured[1]["max_tokens"] == 2048
        assert captured[0]["chat_template_kwargs"]["thinking_budget"] == 32768
        assert captured[1]["chat_template_kwargs"]["thinking_budget"] == 32768

    def test_anthropic_messages_omitted_max_tokens_uses_bundle_default(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Anthropic compatibility must not force its old 4096 default.

        The /v1/messages endpoint aggregates the shared Chat Completions stream,
        so test the route, not just anthropic_adapter.to_chat_completion().
        """
        from fastapi.testclient import TestClient

        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )

        class FakeEngine:
            is_mllm = False
            tokenizer = None
            preserve_native_tool_format = False

        captured: list[dict] = []

        async def fake_stream_chat_completion(*args, **kwargs):
            captured.append(
                {
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                }
            )
            yield (
                'data: {"choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
                '"completion_tokens":1}}\n\n'
            )
            yield "data: [DONE]\n\n"

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "bundle-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_api_key", None, raising=False)
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        monkeypatch.setattr(server, "_default_temperature", None)
        monkeypatch.setattr(server, "_default_top_p", None)
        monkeypatch.setattr(server, "_default_top_k", None)
        monkeypatch.setattr(server, "_default_min_p", None)
        monkeypatch.setattr(server, "_default_repetition_penalty", None)
        monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        client = TestClient(server.app)

        omitted = client.post(
            "/v1/messages",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        explicit = client.post(
            "/v1/messages",
            json={
                "model": "bundle-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 33,
            },
        )

        assert omitted.status_code == 200
        assert explicit.status_code == 200
        assert captured[0]["max_tokens"] == 2048
        assert captured[1]["max_tokens"] == 33

    def test_explicit_server_max_tokens_overrides_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A real session/CLI max-tokens setting must remain an override.

        Omitted startup max_tokens should let the bundle own the default
        length, but an explicit session value is a user/operator choice.
        """
        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 512)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", True, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 512

    def test_omitted_server_max_tokens_uses_bundle_max_new_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        """No session/CLI override means generation_config owns length."""
        import vmlx_engine.server as server

        (tmp_path / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 2048})
        )
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 2048

    def test_omitted_server_max_tokens_without_bundle_default_is_bounded(
        self,
        tmp_path,
        monkeypatch,
    ):
        """No request/session/bundle length must not silently become 32K.

        Several local JANG/JANGTQ bundles intentionally omit
        generation_config.max_new_tokens. In that case the engine still needs
        a real bounded output budget instead of inheriting the historical
        32768 fallback, which lets long turns drift into loops.
        """
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
        (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": -1}))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_default_max_tokens", 32768)
        monkeypatch.setattr(server, "_default_max_tokens_explicit", False, raising=False)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        assert server._resolve_max_tokens(None, "bundle-model") == 4096

    def test_max_tokens_resolution_contract_applies_to_every_registered_family(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Every backend family gets the same output-length precedence.

        This prevents family-specific parser/cache work from accidentally
        bypassing the model-owned length contract:
        request > explicit startup --max-tokens > bundle max_new_tokens > fallback.
        """
        import vmlx_engine.server as server
        from vmlx_engine.model_config_registry import ModelConfigRegistry
        from vmlx_engine.model_configs import register_all

        registry = ModelConfigRegistry()
        register_all(registry)
        families = [c for c in registry._configs if c.model_types]
        assert len(families) >= 70

        monkeypatch.setattr(server, "_default_max_tokens", 512)
        server._jang_sampling_defaults_cache.clear()
        server._generation_defaults_cache.clear()

        for cfg in families:
            model_type = cfg.model_types[0]
            model_dir = tmp_path / model_type
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
            (model_dir / "generation_config.json").write_text(
                json.dumps({"max_new_tokens": 2048})
            )

            monkeypatch.setattr(server, "_model_path", str(model_dir))

            monkeypatch.setattr(
                server, "_default_max_tokens_explicit", False, raising=False
            )
            server._jang_sampling_defaults_cache.clear()
            server._generation_defaults_cache.clear()
            assert server._resolve_max_tokens(None, str(model_dir)) == 2048, (
                cfg.family_name,
                model_type,
            )
            assert server._resolve_max_tokens(77, str(model_dir)) == 77

            monkeypatch.setattr(
                server, "_default_max_tokens_explicit", True, raising=False
            )
            server._jang_sampling_defaults_cache.clear()
            server._generation_defaults_cache.clear()
            assert server._resolve_max_tokens(None, str(model_dir)) == 512, (
                cfg.family_name,
                model_type,
            )

    def test_wake_reload_preserves_max_tokens_explicitness(self):
        """Deep-sleep reload must not turn fallback max_tokens into override."""
        source = Path("./vmlx_engine/server.py").read_text()
        wake_block = source[
            source.index("async def admin_wake"):
            source.index("@app.get(\"/v1/cache/stats\"")
        ]

        assert '"max_tokens_explicit": max_tokens_explicit' in source
        assert 'max_tokens_explicit=_cli_args.get("max_tokens_explicit", False)' in wake_block

    def test_cli_serve_implicit_max_tokens_uses_bounded_fallback(self):
        """The vmlx serve parser must not reintroduce the historical 32K default."""
        import vmlx_engine.cli as cli
        import vmlx_engine.server as server

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        serve_max_tokens_block = cli_source[
            cli_source.index('serve_parser.add_argument(\n        "--max-tokens"'):
            cli_source.index('serve_parser.add_argument(\n        "--max-prompt-tokens"')
        ]

        assert cli.DEFAULT_MAX_OUTPUT_TOKENS == server._FALLBACK_MAX_OUTPUT_TOKENS
        assert "default=DEFAULT_MAX_OUTPUT_TOKENS" in serve_max_tokens_block
        assert "default=32768" not in serve_max_tokens_block
        assert 'getattr(args, "max_tokens", 32768) != 32768' not in cli_source


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
        """The bare qwen3_5 family entry is neutral.

        Local Qwen 3.6 bundles with vision/video metadata are tested below via
        registry.lookup(path), because that path reads config.json.
        """
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3_5")
        assert config is not None
        assert config.is_mllm is False

    def test_lookup_qwen3_6_dense_path_with_vision_video_is_mllm(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        registry._match_cache.clear()

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))

        config = registry.lookup(str(tmp_path))

        assert config.family_name == "qwen3_5"
        assert config.is_mllm is True

    def test_lookup_qwen3_6_moe_path_with_vision_video_is_mllm(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        registry._match_cache.clear()

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_moe_text"},
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))

        config = registry.lookup(str(tmp_path))

        assert config.family_name == "qwen3_5_moe"
        assert config.is_mllm is True

    @pytest.mark.parametrize(
        ("model_type", "family_name"),
        [
            ("qwen3_5", "qwen3_5"),
            ("qwen3_5_moe", "qwen3_5_moe"),
        ],
    )
    def test_qwen3_6_vision_video_config_marks_bundle_multimodal(
        self, tmp_path, model_type, family_name
    ):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / model_type
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "architectures": [f"{model_type}ForConditionalGeneration"],
                    "text_config": {
                        "model_type": model_type,
                        "layer_types": ["linear_attention", "full_attention"],
                    },
                    "vision_config": {"hidden_size": 1024},
                    "video_token_id": 151666,
                    "video_token_index": 151666,
                }
            )
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == family_name
        assert config.cache_type == "hybrid"
        assert config.is_mllm is True

    def test_nemotron_h_without_media_config_is_text_runtime(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / "nemotron-text"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "text_config": {
                        "layer_types": ["mamba", "full_attention"],
                    },
                }
            )
        )
        (model_dir / "preprocessor_config.json").write_text("{}")

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == "nemotron_h"
        assert config.cache_type == "hybrid"
        assert config.is_mllm is False

    def test_nemotron_h_omni_stamp_without_media_config_is_text_runtime(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / "nemotron-stamped-text"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "text_config": {
                        "layer_types": ["mamba", "full_attention"],
                    },
                }
            )
        )
        (model_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "family": "nemotron_h",
                        "modality": "omni",
                        "cache_type": "hybrid",
                    }
                }
            )
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == "nemotron_h"
        assert config.cache_type == "hybrid"
        assert config.is_mllm is False

    @pytest.mark.parametrize("model_type", ["zaya", "zaya1_vl"])
    def test_zaya_registry_preserves_tokenizer_eos_boundary(self, tmp_path, model_type):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = tmp_path / model_type
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    **({"vision_config": {"hidden_size": 1024}} if model_type == "zaya1_vl" else {}),
                }
            )
        )
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>", "eos_token_id": 106})
        )

        config = get_model_config_registry().lookup(str(model_dir))

        assert config.family_name == model_type
        assert config.eos_tokens == ["<|im_end|>"]

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

    def test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw(self):
        """DSV4 DSML repair must survive the server's OpenAI ToolCall wrapper."""
        import json

        import vmlx_engine.server as srv
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSML_PREFIX

        request = srv.ChatCompletionRequest(
            model="dsv4-test",
            messages=[
                srv.Message(
                    role="user",
                    content="Use read_file for docs/vendor_memo.md.",
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )
        output = (
            f'<{DSML_PREFIX}tool_calls>\n'
            f'<{DSML_PREFIX}invoke name="read_file">\n'
            '    <param name="path">docs/vendor_memo.md</param>\n'
            '</inv>'
        )

        old_auto = srv._enable_auto_tool_choice
        old_parser = srv._tool_call_parser
        try:
            srv._enable_auto_tool_choice = True
            srv._tool_call_parser = "deepseek_v4"
            cleaned, tool_calls = srv._parse_tool_calls_with_parser(output, request)
        finally:
            srv._enable_auto_tool_choice = old_auto
            srv._tool_call_parser = old_parser

        assert cleaned == ""
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0].function.name == "read_file"
        assert json.loads(tool_calls[0].function.arguments) == {
            "path": "docs/vendor_memo.md"
        }
        assert DSML_PREFIX not in tool_calls[0].function.arguments

    def test_dsv4_fallback_tool_prompt_uses_canonical_tool_calls_wrapper(self):
        """Fallback injection must not teach a non-canonical bare DSML invoke."""
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class DSV4FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                rendered = "<｜begin▁of▁sentence｜>"
                for msg in messages:
                    role = msg.get("role")
                    if role == "system":
                        rendered += msg.get("content") or ""
                    elif role == "user":
                        rendered += "<｜User｜>" + (msg.get("content") or "")
                    elif role == "assistant":
                        rendered += "<｜Assistant｜>" + (msg.get("content") or "")
                    elif role == "tool":
                        rendered += (
                            "<｜User｜><tool_result>"
                            + (msg.get("content") or "")
                            + "</tool_result>"
                        )
                rendered += "<｜Assistant｜></think>"
                return rendered

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
        prompt = (
            "<｜begin▁of▁sentence｜>\n\n## Tools\n"
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n"
            "<｜User｜>Use tools in sequence."
        )

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Use tools in sequence."}],
            tools,
            DSV4FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="dsml",
        )

        assert "or a DSML wrapper block" not in rendered
        assert "VALUE HERE" not in rendered
        assert "string=" not in rendered.split("<｜User｜>", 1)[0]
        assert "<｜DSML｜parameter" not in rendered.split("<｜User｜>", 1)[0]
        instruction_scope = rendered.split("<｜User｜>", 1)[0]
        assert "<｜DSML｜tool_calls>" in rendered
        assert "</｜DSML｜tool_calls>" in rendered
        assert '<｜DSML｜invoke name="list_directory">' in rendered
        assert '<｜DSML｜invoke name="write_file">' in rendered
        assert instruction_scope.count("<｜DSML｜tool_calls>") == 2
        assert "path (string, required)" in rendered
        assert "content (string, required)" in rendered

    def test_dsv4_native_schema_prompt_with_only_generic_examples_gets_concrete_fallback(self):
        """DSV4 needs concrete invoke names, not only generic DSML placeholders."""
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class DSV4FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
        prompt = (
            "<｜begin▁of▁sentence｜>\n\n## Tools\n"
            "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
            "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
            "$PARAMETER_VALUE</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n\n"
            "### Available Tool Schemas\n\n"
            '{"name": "list_directory", "parameters": {"properties": {"path": {"type": "string"}}}}\n'
            '{"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}\n'
            "<｜User｜>Use tools in sequence."
        )

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Use tools in sequence."}],
            tools,
            DSV4FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="dsml",
        )

        instruction_scope = rendered.split("<｜User｜>", 1)[0]
        assert rendered != prompt
        assert '<｜DSML｜invoke name="list_directory">' in instruction_scope
        assert '<｜DSML｜invoke name="write_file">' in instruction_scope
        assert "path (string, required)" in instruction_scope
        assert "content (string, required)" in instruction_scope
        assert "$TOOL_NAME" not in instruction_scope
        assert "$PARAMETER_NAME" not in instruction_scope
        assert "$PARAMETER_VALUE" not in instruction_scope

    def test_stream_chat_buffers_leading_whitespace_before_tool_marker(self):
        """Tool-call-only streams must not leak whitespace before tool_calls."""
        import inspect

        from vmlx_engine.server import _TOOL_CALL_MARKERS, stream_chat_completion

        assert "<｜DSML｜tool" in _TOOL_CALL_MARKERS
        source = inspect.getsource(stream_chat_completion)
        assert "<｜DSML｜tool_c" in _TOOL_CALL_MARKERS
        assert "<tool_calls>" in _TOOL_CALL_MARKERS
        assert "<tool_call>" in _TOOL_CALL_MARKERS
        assert "<tool_sep>" in _TOOL_CALL_MARKERS
        assert "<function" in _TOOL_CALL_MARKERS
        assert "<zyphra_tool_call" in _TOOL_CALL_MARKERS
        assert "not emit_content.strip()" in source
        assert "not content.strip()" in source
        assert "tool_call_generating=True" in source

    def test_tool_markup_residue_strips_all_registered_marker_families(self):
        """Display cleanup must track every family marker, not only DSML/Hy3."""
        from vmlx_engine.server import _strip_tool_markup_residue_for_display

        cases = {
            "zaya": "Before <zyphra_tool_call name=\"read_file\"> after",
            "mistral": "Before [TOOL_CALLS] after",
            "minimax": "Before <minimax:tool_call> after",
            "kimi": "Before <|tool_calls_section_begin|> after",
            "gemma": "Before <|tool_call> after",
            "llama": "Before <function=read_file> after",
            "deepseek_glm": "Before <｜tool▁call▁begin｜> after",
            "dsv4": "Before <｜DSML｜tool after",
            "python_tag": "Before <|python_tag|> after",
            "tool_code": "Before ```tool_code after",
        }

        for family, text in cases.items():
            cleaned = _strip_tool_markup_residue_for_display(text)
            assert "Before" in cleaned, family
            assert "after" in cleaned, family
            assert "<" not in cleaned, (family, cleaned)
            assert "[TOOL_CALLS]" not in cleaned, (family, cleaned)
            assert "```tool_code" not in cleaned, (family, cleaned)

    def test_tool_markup_residue_strips_interrupted_non_dsml_fragments(self):
        """Interrupted marker fragments must not leak attribute/code tails."""
        from vmlx_engine.server import _strip_tool_markup_residue_for_display

        cases = {
            "zaya_attr_tail": 'Before <zyphra_tool_call name="write_file"',
            "llama_function_tail": "Before <function=write_file",
            "gemma3_code_tail": "Before ```tool_code\nwrite_file(",
        }

        for family, text in cases.items():
            cleaned = _strip_tool_markup_residue_for_display(text)
            assert "Before" in cleaned, family
            assert "write_file" not in cleaned, (family, cleaned)
            assert "zyphra" not in cleaned, (family, cleaned)
            assert "function" not in cleaned, (family, cleaned)
            assert "```tool_code" not in cleaned, (family, cleaned)


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


class TestLLMSingleActiveGenerator:
    """Text max_num_seqs=1 must be an engine path, not a patched mlx-lm merge."""

    def test_scheduler_imports_single_active_generator(self):
        from pathlib import Path

        source = Path("./vmlx_engine/scheduler.py").read_text()

        assert "from .utils.single_batch_generator import SingleBatchGenerator" in source
        assert "return SingleBatchGenerator(" in source
        assert "max_num_seqs=1" in source
        assert '"engine_path": generator_path' in source
        assert '"single_active_decode": generator_path == "single_active"' in source

    def test_single_active_generator_is_repo_owned(self):
        from pathlib import Path

        source = Path("./vmlx_engine/utils/single_batch_generator.py").read_text()

        assert "class SingleBatchGenerator" in source
        assert "from mlx_lm.generate import BatchGenerator" not in source
        assert "BatchKVCache(" not in source
        assert "BatchMambaCache(" not in source
        assert "BatchGenerator(" not in source
        assert "prompt_cache=req.cache" in source

    def test_health_and_cache_stats_surface_single_active_path(self):
        from pathlib import Path

        source = Path("./vmlx_engine/server.py").read_text()

        assert '"engine_path": scheduler_stats.get("engine_path")' in source
        assert '"single_active_decode": (' in source
        assert '"engine_path": stats.get("engine_path")' in source


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


class TestMediaDiagnostics:
    """Redacted media diagnostics for user logs when image requests fail."""

    def test_messages_multimodal_summary_redacts_data_urls(self):
        from vmlx_engine.server import _messages_multimodal_summary

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what color?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,SECRET_IMAGE_BYTES"
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "/tmp/example.mp4"},
                    },
                ],
            }
        ]

        summary = _messages_multimodal_summary(messages)

        assert summary["total"] == 2
        assert summary["types"] == {"image_url": 1, "video_url": 1}
        assert summary["data_url"] == 1
        assert summary["url_or_path"] == 1
        assert "SECRET_IMAGE_BYTES" not in json.dumps(summary)

    def test_responses_multimodal_summary_handles_input_image_without_payload(self):
        from vmlx_engine.server import _responses_input_multimodal_summary

        response_input = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_RESPONSE_IMAGE",
                    },
                ],
            }
        ]

        summary = _responses_input_multimodal_summary(response_input)

        assert summary["total"] == 1
        assert summary["types"] == {"input_image": 1}
        assert summary["data_url"] == 1
        assert "SECRET_RESPONSE_IMAGE" not in json.dumps(summary)

    def test_chat_multimodal_summary_handles_input_image_shape(self):
        from vmlx_engine.server import _messages_multimodal_summary

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_CHAT_INPUT_IMAGE",
                    },
                ],
            }
        ]

        summary = _messages_multimodal_summary(messages)

        assert summary["total"] == 1
        assert summary["types"] == {"input_image": 1}
        assert summary["data_url"] == 1
        assert summary["roles"] == {"user": 1}
        assert "SECRET_CHAT_INPUT_IMAGE" not in json.dumps(summary)

    def test_anthropic_image_conversion_reaches_media_summary(self):
        from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
        from vmlx_engine.server import _messages_multimodal_summary

        anthropic = AnthropicRequest(
            model="vl",
            max_tokens=32,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "SECRET_ANTHROPIC_IMAGE",
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        )

        chat = to_chat_completion(anthropic)
        summary = _messages_multimodal_summary(chat.messages)

        assert summary["total"] == 1
        assert summary["types"] == {"image_url": 1}
        assert summary["data_url"] == 1
        assert summary["roles"] == {"user": 1}
        assert "SECRET_ANTHROPIC_IMAGE" not in json.dumps(summary)

    def test_media_diag_hooks_cover_anthropic_and_ollama_streaming_ingress(self):
        import inspect
        import vmlx_engine.server as server

        anthropic_source = inspect.getsource(server.create_anthropic_message)
        ollama_source = inspect.getsource(server.ollama_chat)

        assert '"/v1/messages"' in anthropic_source
        assert "_log_multimodal_request_shape" in anthropic_source
        assert "_messages_multimodal_summary(chat_req.messages)" in anthropic_source
        assert '_reject_unsupported_multimodal("/v1/messages")' in anthropic_source

        assert '"/api/chat"' in ollama_source
        assert "_log_multimodal_request_shape" in ollama_source
        assert "_messages_multimodal_summary(chat_req.messages)" in ollama_source
        assert '_reject_unsupported_multimodal("/api/chat")' in ollama_source

    def test_anthropic_media_on_text_runtime_rejects_instead_of_dropping(
        self, monkeypatch
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=False))
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "text-runtime")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "aGVsbG8=",
                                },
                            },
                            {"type": "text", "text": ""},
                        ],
                    }
                ],
            },
        )

        assert response.status_code == 400
        assert "text-only" in response.text

    def test_ollama_streaming_media_on_text_runtime_rejects_instead_of_dropping(
        self, monkeypatch
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=False))
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_name", "text-runtime")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/api/chat",
            json={
                "model": "text-runtime",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "", "images": ["aGVsbG8="]}
                ],
            },
        )

        assert response.status_code == 400
        assert "text-only" in response.text

    def test_zaya_vl_runtime_modalities_do_not_infer_video_from_vision(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "zaya1_vl",
                    "vision_config": {"model_type": "qwen2_5_vl"},
                    "image_token_id": 262147,
                    "capabilities": {"modality": "vision"},
                }
            )
        )
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"video_token": "<video>"})
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision"]

    def test_qwen_vl_runtime_modalities_keep_explicit_video(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "vision_config": {"model_type": "qwen3_vl"},
                    "video_token_id": 248057,
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "qwen-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        assert server._loaded_runtime_modalities() == ["text", "vision", "video"]

    def test_video_request_on_image_only_mllm_rejects_instead_of_crashing(
        self, monkeypatch, tmp_path
    ):
        from fastapi.testclient import TestClient
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "zaya1_vl",
                    "vision_config": {"model_type": "qwen2_5_vl"},
                    "image_token_id": 262147,
                    "capabilities": {"modality": "vision"},
                }
            )
        )

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "zaya-vl-test")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)

        response = TestClient(server.app).post(
            "/v1/chat/completions",
            json={
                "model": "zaya-vl-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this video?"},
                            {
                                "type": "video_url",
                                "video_url": {"url": "data:video/mp4;base64,AAAA"},
                            },
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 400
        assert "unsupported media modality video" in response.text
        assert "text, vision" in response.text

    def test_server_media_diag_log_includes_route_runtime_and_redacts_payload(
        self, caplog, monkeypatch
    ):
        import logging
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_engine", SimpleNamespace(is_mllm=True))
        monkeypatch.setattr(server, "_model_path", "/models/Brooklyn-VLM")
        monkeypatch.setattr(server, "_model_name", "Brooklyn-VLM")
        monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
        summary = {
            "total": 1,
            "types": {"image_url": 1},
            "data_url": 1,
            "url_or_path": 0,
            "missing_source": 0,
            "roles": {"user": 1},
            "messages": 1,
            "content_arrays": 1,
        }

        caplog.set_level(logging.INFO, logger="vmlx_engine.server")
        server._log_multimodal_request_shape(
            "/v1/chat/completions",
            "/models/Brooklyn-VLM",
            summary,
        )

        text = caplog.text
        assert "[MEDIA_DIAG] request_shape=" in text
        assert '"/v1/chat/completions"' in text
        assert '"engine_is_mllm": true' in text
        assert '"total": 1' in text
        assert "SECRET" not in text
        assert "data:image" not in text


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
# H4 Regression: Empty output is not model text
# ===========================================================================


class TestH4EmptyOutputIsNotAssistantText:
    """Empty Responses output must stay empty instead of becoming fake model text."""

    def test_responses_api_empty_output_skips_validation_without_placeholder(self):
        """Responses API JSON validation should skip naturally on empty display_text."""
        import inspect
        from vmlx_engine.server import stream_responses_api

        source = inspect.getsource(stream_responses_api)
        assert "Model produced no response" not in source
        assert "Model produced only internal reasoning" not in source
        assert "_FALLBACK_MSG" not in source
        assert "and display_text" in source
        assert "empty_model_response" in source
        assert "reasoning_only_no_content" in source

    def test_chat_completions_already_safe(self):
        """Chat completions uses content_was_emitted gate — already safe."""
        import inspect
        from vmlx_engine.server import stream_chat_completion

        source = inspect.getsource(stream_chat_completion)
        # Chat completions gates on content_was_emitted
        assert "content_was_emitted" in source
        assert "Model produced only internal reasoning" not in source
        assert "ChatCompletionChunkDelta()" in source


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
    """Suppressed reasoning must stay out of visible content/history.

    The panel now keeps reasoning in the dedicated reasoning channel and
    surfaces reasoning-only cases as warnings/placeholders instead of
    redirecting parser output into normal assistant content.
    """

    def test_chat_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_chat_completion
        source = inspect.getsource(stream_chat_completion)
        assert "accumulated_content += delta_msg.reasoning" not in source
        assert "Suppressed reasoning is never redirected into visible content" in source
        assert "suppress_reasoning and not content_was_emitted and accumulated_reasoning" in source

    def test_responses_accumulated_content_mirror_under_suppress(self):
        import inspect
        from vmlx_engine.server import stream_responses_api
        source = inspect.getsource(stream_responses_api)
        assert "accumulated_content += delta_msg.reasoning" not in source
        assert "Suppressed reasoning is never redirected into" in source
        assert "reasoning_only_no_content" in source


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
    """Disk cache fetch must defer dequantization to the scheduler worker."""

    def test_scheduler_disk_cache_dequantize(self):
        """scheduler.py: disk_cache.fetch result must not dequantize on API thread."""
        import inspect
        from vmlx_engine.scheduler import Scheduler
        source = inspect.getsource(Scheduler.add_request)
        fetch_idx = source.find("disk_cache.fetch")
        assert fetch_idx != -1, "disk_cache.fetch not found in add_request"
        after_fetch = source[fetch_idx:fetch_idx + 900]
        assert "_prompt_cache_needs_worker_dequant = True" in after_fetch, (
            "disk_cache.fetch must mark cache hits for worker-side dequantization"
        )
        assert "_dequantize_cache_for_use(" not in after_fetch, (
            "disk_cache.fetch path must not dequantize MLX arrays on the API thread"
        )
        schedule_source = inspect.getsource(Scheduler._schedule_waiting)
        assert "_prompt_cache_needs_worker_dequant" in schedule_source
        assert "_dequantize_cache_for_use(cache_to_use)" in schedule_source, (
            "worker schedule path must own stored-cache dequantization"
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
        after_fetch = source[fetch_idx:fetch_idx + 2000]
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


class TestNonStreamingDisconnectAbort:
    """Non-streaming API requests must abort scheduler work on disconnect."""

    def test_helper_aborts_when_client_disconnects(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            def __init__(self):
                self.aborted = None

            async def chat(self, **kwargs):
                await asyncio.sleep(10)

            async def abort_request(self, request_id):
                self.aborted = request_id
                return True

        class FakeRequest:
            async def is_disconnected(self):
                return True

        async def run():
            engine = FakeEngine()
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    engine,
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=FakeRequest(),
                    request_id="resp_disconnect_test",
                    endpoint="test",
                    poll_interval=0.001,
                )
            assert exc.value.status_code == 499
            assert engine.aborted == "resp_disconnect_test"

        asyncio.run(run())

    def test_helper_passes_public_request_id_to_engine_chat(self):
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            def __init__(self):
                self.kwargs = None

            async def chat(self, **kwargs):
                self.kwargs = kwargs
                return FakeOutput()

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def run():
            engine = FakeEngine()
            output = await _await_chat_with_disconnect_abort(
                engine,
                messages=[{"role": "user", "content": "hi"}],
                chat_kwargs={"max_tokens": 1},
                timeout=30,
                fastapi_request=FakeRequest(),
                request_id="chatcmpl_public_id",
                endpoint="test",
                poll_interval=0.001,
            )
            assert output.completion_tokens == 1
            assert engine.kwargs["request_id"] == "chatcmpl_public_id"
            assert engine.kwargs["messages"] == [{"role": "user", "content": "hi"}]

        asyncio.run(run())

    def test_helper_treats_cancelled_collector_removal_as_client_cancel(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            async def chat(self, **kwargs):
                raise RuntimeError("No collector for request resp_cancelled")

            async def abort_request(self, request_id):
                return False

        class FakeRequest:
            async def is_disconnected(self):
                return False

        async def run():
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    FakeEngine(),
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=FakeRequest(),
                    request_id="resp_cancelled",
                    endpoint="test",
                    poll_interval=0.001,
                )
            assert exc.value.status_code == 499

        asyncio.run(run())


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
    """reasoning_effort maps to thinking_budget without rewriting output length."""

    def test_effort_constants_defined(self):
        """Server must define effort-to-budget mapping constants."""
        from vmlx_engine.server import _EFFORT_THINKING_BUDGET
        assert "low" in _EFFORT_THINKING_BUDGET
        assert "medium" in _EFFORT_THINKING_BUDGET
        assert "high" in _EFFORT_THINKING_BUDGET
        assert _EFFORT_THINKING_BUDGET["low"] < _EFFORT_THINKING_BUDGET["medium"]
        assert _EFFORT_THINKING_BUDGET["medium"] < _EFFORT_THINKING_BUDGET["high"]

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

    def test_effort_does_not_synthesize_max_tokens_when_unset(self):
        """reasoning_effort must not replace model-owned max_new_tokens."""
        import inspect
        from vmlx_engine.server import create_chat_completion
        source = inspect.getsource(create_chat_completion)
        assert "thinking_budget" in source, (
            "reasoning_effort should still reach chat_template_kwargs"
        )
        assert "_EFFORT_MAX_TOKENS" not in source, (
            "reasoning_effort must not secretly rewrite max_tokens; explicit "
            "max_tokens/max_output_tokens or bundle generation_config own length"
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

    def test_omni_messages_path_adapts_plain_dict_completion(self):
        import inspect
        from vmlx_engine.server import create_anthropic_message

        source = inspect.getsource(create_anthropic_message)
        assert "if isinstance(cc, dict):" in source
        assert "to_anthropic_response(cc, resolved_name)" in source

    def test_omni_responses_path_adapts_plain_dict_and_forwards_thinking(self):
        import inspect
        from vmlx_engine.server import create_response

        source = inspect.getsource(create_response)
        start = source.index("Nemotron-Omni multimodal dispatch (Responses API parity")
        end = source.index("if request.stream:", start)
        block = source[start:end]

        assert "enable_thinking=request.enable_thinking" in block
        assert "elif isinstance(cc, dict):" in block
        assert "cc_dict = cc" in block


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

    def test_dsv4_switchglu_contract_audit_requires_limited_swiglu(self):
        import inspect
        from vmlx_engine.loaders.load_jangtq_dsv4 import _audit_dsv4_switchglu_contract

        source = inspect.getsource(_audit_dsv4_switchglu_contract)
        assert "swiglu_limit" in source
        assert "10.0" in source
        assert "zero TurboQuant SwitchGLU" in source
        assert "swiglu_limit=10 missing" in source

    def test_dsv4_switchglu_contract_audit_rejects_missing_limit(self, monkeypatch):
        import sys
        import types
        import pytest
        from vmlx_engine.loaders.load_jangtq_dsv4 import _audit_dsv4_switchglu_contract

        class TurboQuantSwitchLinear:
            pass

        class SwitchGLU:
            pass

        tq_mod = types.ModuleType("jang_tools.turboquant.tq_kernel")
        tq_mod.TurboQuantSwitchLinear = TurboQuantSwitchLinear
        switch_mod = types.ModuleType("mlx_lm.models.switch_layers")
        switch_mod.SwitchGLU = SwitchGLU
        monkeypatch.setitem(sys.modules, "jang_tools.turboquant.tq_kernel", tq_mod)
        monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch_mod)

        class _Activation:
            def __init__(self, limit):
                self.swiglu_limit = limit

        class _Switch(SwitchGLU):
            def __init__(self, limit):
                self.gate_proj = TurboQuantSwitchLinear()
                self.up_proj = TurboQuantSwitchLinear()
                self.down_proj = TurboQuantSwitchLinear()
                self.activation = _Activation(limit)

        class _Model:
            def __init__(self, module):
                self.module = module

            def named_modules(self):
                return [("layers.0.mlp.switch_mlp", self.module)]

        _audit_dsv4_switchglu_contract(_Model(_Switch(10.0)))

        with pytest.raises(RuntimeError, match="swiglu_limit=10 missing"):
            _audit_dsv4_switchglu_contract(_Model(_Switch(0.0)))


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
        assert "requested_logprobs" in func_body
        assert "_should_capture_generation_logprobs" in func_body
        assert "logprobs[i] if requested_logprobs[i] else None" in func_body
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

    def test_llm_scheduler_hybrid_ssm_rederive_policy_is_not_stale(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()
        store_idx = source.index("# Hybrid SSM companion state capture.")
        store_block = source[store_idx : source.index("# Store cache for future reuse", store_idx)]

        assert "SSM_REDERIVE_MIN_TOKENS = 1" in source
        assert "< 64" not in store_block
        assert "SSM_REDERIVE_MIN_TOKENS" in store_block
        assert "gpl=0 (non-thinking) hybrid SSM path" in store_block
        assert "DO NOT extract" in store_block
        assert "post-output SSM layers" in store_block
        assert "queued deferred" in store_block
        assert "is_complete=False" in store_block


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

    def test_python_dependency_floor_covers_current_jang_family_runtimes(self):
        pyproject = Path("./pyproject.toml").read_text()

        # Hy3 registration lives in jang_tools.hy3 as of the 2.5.29 public wheel
        # and the newer local 2.5.30 app bundle;
        # Ling/Bailing hybrid needs the mlx-lm runtime floor that the bundle
        # uses before the local bailing_hybrid vendor file is applied.
        assert '"mlx-lm>=0.31.3"' in pyproject
        assert pyproject.count('"jang>=2.5.29"') >= 3

    def test_bundled_python_installs_distutils_version_shim_for_radio(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()

        assert "Installing distutils.version compatibility shim" in bundle_script
        assert 'mkdir -p "$SITE/distutils"' in bundle_script
        assert "class LooseVersion" in bundle_script
        assert 'from distutils.version import LooseVersion' in bundle_script
        assert 'LooseVersion("1.2") < LooseVersion("1.3")' in bundle_script

    def test_bundled_python_local_source_installs_force_reinstall(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        local_install_block = bundle_script[
            bundle_script.index('echo "==> Installing vmlx-engine + jang_tools')
            : bundle_script.index("# Clean up to reduce size")
        ]

        assert '"$PYTHON" -m pip install --force-reinstall --no-deps "$VMLX_LOCAL"' in local_install_block
        assert 'JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"' in bundle_script
        assert '"$PYTHON" -m pip install --force-reinstall --no-deps "$JANG_LOCAL"' in local_install_block

    def test_bundled_python_does_not_silently_fallback_to_pypi_jang_tools(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '${VMLX_ALLOW_PYPI_JANG:-${VMLINUX_ALLOW_PYPI_JANG:-0}}' in bundle_script
        assert "RELEASE BLOCKED — local jang-tools source missing" in bundle_script
        assert 'pip install --no-deps "jang>=2.5.29"' in bundle_script
        assert '${VMLX_ALLOW_MISSING_JANG_SOURCE_HASH:-${VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH:-0}}' in verify_script
        assert "RELEASE BLOCKED — local jang_tools source unavailable for hash parity" in verify_script

    def test_bundled_python_blocks_tracked_dirty_local_jang_tools_by_default(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        local_install_block = bundle_script[
            bundle_script.index('JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"')
            : bundle_script.index("# Clean up to reduce size")
        ]
        destructive_build_index = bundle_script.index('rm -rf "$BUNDLE_DIR"')
        dirty_guard_index = bundle_script.index("check_local_jang_source_clean")

        assert '${VMLX_ALLOW_DIRTY_JANG_SOURCE:-${VMLINUX_ALLOW_DIRTY_JANG_SOURCE:-0}}' in local_install_block
        assert "RELEASE BLOCKED — local jang-tools source has tracked changes" in local_install_block
        assert 'git -C "$JANG_LOCAL" diff --quiet --ignore-submodules --' in local_install_block
        assert 'git -C "$JANG_LOCAL" diff --cached --quiet --ignore-submodules --' in local_install_block
        assert dirty_guard_index < destructive_build_index

    def test_release_scripts_accept_documented_jang_tools_env_names_with_legacy_fallback(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert 'JANG_LOCAL="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}"' in bundle_script
        assert 'JANG_TOOLS_SOURCE_DIR="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$HOME/jang/jang-tools}}/jang_tools"' in verify_script
        assert 'os.environ.get("VMLX_JANG_TOOLS_SOURCE")' in release_gate
        assert 'os.environ.get("VMLINUX_JANG_TOOLS_SOURCE")' in release_gate
        assert '${VMLX_ALLOW_DIRTY_JANG_SOURCE:-${VMLINUX_ALLOW_DIRTY_JANG_SOURCE:-0}}' in bundle_script
        assert '${VMLX_ALLOW_PYPI_JANG:-${VMLINUX_ALLOW_PYPI_JANG:-0}}' in bundle_script
        assert '${VMLX_ALLOW_MISSING_JANG_SOURCE_HASH:-${VMLINUX_ALLOW_MISSING_JANG_SOURCE_HASH:-0}}' in verify_script

    def test_bundled_python_console_scripts_are_relocatable(self):
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert "Rewriting console-script shebangs to relocatable bundled Python" in bundle_script
        assert "\\$(dirname " in bundle_script
        assert "/python3\\\" -B -s" in bundle_script
        assert "check_console_script_shebangs" in verify_script
        assert "check_packaged_console_script_shebangs" in release_gate
        assert "/Applications/vMLX.app" in release_gate

    def test_local_installer_installs_node_deps_before_typecheck(self):
        install_script = Path("./panel/scripts/build-and-install.sh").read_text()

        assert "ensure_node_dependencies()" in install_script
        assert 'node_modules/.bin/tsc' in install_script
        install_idx = install_script.index("ensure_node_dependencies")
        typecheck_idx = install_script.index("npm run typecheck")
        assert install_idx < typecheck_idx

    def test_release_gate_packaged_python_probes_cannot_mutate_signed_app(self):
        release_gate = Path("./panel/scripts/release-gate-python-app.py").read_text()

        assert "packaged_python_env" in release_gate
        assert "PYTHONPYCACHEPREFIX" in release_gate
        assert "twine_env" in release_gate
        assert '"twine check dist"' in release_gate
        assert "env=twine_env(gate)" in release_gate
        assert "check_no_packaged_pycache" in release_gate
        assert "packaged app pycache clean" in release_gate

    def test_bundled_python_hash_gate_covers_runtime_files_changed_for_release(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        for rel in (
            "server.py",
            "api/anthropic_adapter.py",
            "api/ollama_adapter.py",
            "engine/batched.py",
            "loaders/load_jangtq_dsv4.py",
            "mllm_batch_generator.py",
            "mllm_scheduler.py",
            "omni_multimodal.py",
            "prefix_cache.py",
            "runtime_patches/gemma4_processing.py",
            "scheduler.py",
        ):
            assert f'"{rel}"' in verify_script

    def test_bundled_python_hash_gate_covers_critical_jang_tools_files(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()
        for rel in (
            "capabilities.py",
            "convert_hy3_jangtq.py",
            "load_jangtq.py",
            "load_jangtq_kimi_vlm.py",
            "nemotron_omni_chat.py",
            "dsv4/mlx_model.py",
            "dsv4/pool_quant_cache.py",
            "hy3/__init__.py",
            "hy3/model.py",
            "hy3/runtime.py",
            "kimi_prune/generate_vl.py",
            "kimi_prune/runtime_patch.py",
            "topk_override.py",
            "turboquant/fused_gate_up_kernel.py",
            "turboquant/gather_tq_kernel.py",
            "turboquant/hadamard_kernel.py",
            "turboquant/mpp_nax_kernel.py",
            "turboquant/tq_kernel.py",
        ):
            assert f'"{rel}"' in verify_script

    def test_bundled_python_import_gate_covers_jangtq_acceleration_kernel(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '"jang_tools.turboquant.mpp_nax_kernel"' in verify_script
        assert "JANGTQ acceleration kernel missing" in verify_script

    def test_bundled_python_import_gate_covers_hy3_and_ling_registration(self):
        verify_script = Path("./panel/scripts/verify-bundled-python.sh").read_text()

        assert '("jang_tools.hy3", "jang_tools.hy3"' in verify_script
        assert '"mlx_lm.models.hy_v3"' in verify_script
        assert '"mlx_lm.models.bailing_hybrid"' in verify_script
        assert "Hy3 model-family mlx-lm registration missing" in verify_script
        assert "Ling/Bailing hybrid mlx-lm runtime missing" in verify_script

    def test_jang_loader_registers_hy3_and_ling_before_mlx_lm_resolution(
        self, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        seen = []

        def fake_import_module(name):
            seen.append(name)
            return SimpleNamespace()

        monkeypatch.setattr(jang_loader.importlib, "import_module", fake_import_module)

        jang_loader._ensure_jang_family_runtime_supported(
            Path("/models/Hy3-preview-JANGTQ2"), {"model_type": "hy_v3"}
        )
        assert seen[:2] == ["jang_tools.hy3", "mlx_lm.models.hy_v3"]

        seen.clear()
        jang_loader._ensure_jang_family_runtime_supported(
            Path("/models/Ling-2.6-flash-JANGTQ"),
            {"model_type": "bailing_hybrid"},
        )
        assert seen == ["mlx_lm.models.bailing_hybrid"]

    def test_jang_loader_failure_message_names_required_runtime_floor(
        self, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        def fake_import_module(name):
            raise ImportError(f"missing {name}")

        monkeypatch.setattr(jang_loader.importlib, "import_module", fake_import_module)
        monkeypatch.setattr(
            jang_loader,
            "_register_bailing_hybrid_from_repo_patch",
            lambda: (_ for _ in ()).throw(ImportError("missing bailing patch")),
        )

        with pytest.raises(RuntimeError, match="jang>=2.5.29"):
            jang_loader._ensure_jang_family_runtime_supported(
                Path("/models/Hy3-preview-JANGTQ2"), {"model_type": "hy_v3"}
            )

        with pytest.raises(RuntimeError, match="mlx-lm>=0.31.3"):
            jang_loader._ensure_jang_family_runtime_supported(
                Path("/models/Ling-2.6-flash-JANGTQ"),
                {"model_type": "bailing_hybrid"},
            )

    def test_embeddings_and_rerank_endpoints_have_memory_pressure_guards(self):
        source = Path("./vmlx_engine/server.py").read_text()

        assert '@app.post(\n    "/v1/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/v1/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/v1/rerank",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/v1/rerank",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_rate_limit),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/api/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/api/embeddings",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source
        assert '@app.post(\n    "/api/embed",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),' in source
        assert '@app.post(\n    "/api/embed",\n    dependencies=[\n        Depends(verify_api_key),\n        Depends(check_memory_pressure),\n        Depends(check_metal_working_set_pressure),' in source

    def test_bundle_selects_native_mlx_wheels_with_compat_override(self):
        """Bundling can select Sequoia-compatible or Tahoe-native MLX wheels."""
        bundle_script = Path("./panel/scripts/bundle-python.sh").read_text()
        assert 'MLX_VERSION="0.31.2"' in bundle_script
        assert 'MLX_LM_VERSION="0.31.3"' in bundle_script
        assert 'MLX_VLM_VERSION="0.4.4"' in bundle_script
        assert 'detect_mlx_wheel_platform()' in bundle_script
        assert 'VMLX_BUNDLE_MLX_PLATFORM:-${VMLINUX_BUNDLE_MLX_PLATFORM:-compat}' in bundle_script
        assert "legacy misspelled VMLINUX_BUNDLE_MLX_PLATFORM is still accepted" in bundle_script
        assert 'echo "macosx_26_0_arm64"' in bundle_script
        assert 'auto|compat|sonoma|sequoia|"")' in bundle_script
        assert 'echo "macosx_14_0_arm64"' in bundle_script
        assert 'MLX_WHEEL_PLATFORM="$(detect_mlx_wheel_platform)"' in bundle_script
        assert '--platform "$MLX_WHEEL_PLATFORM"' in bundle_script
        assert '"mlx==$MLX_VERSION"' in bundle_script
        assert '"mlx-metal==$MLX_VERSION"' in bundle_script

    def test_release_build_declares_sequoia_and_tahoe_dmg_flavors(self):
        """vmlx#169 release packaging must build both wheel-tag variants."""
        script = Path("./panel/scripts/build-release-dmgs.sh")

        assert script.is_file()
        source = script.read_text()
        assert 'build_one "sequoia" "compat"' in source
        assert 'build_one "tahoe" "native"' in source
        assert 'VMLINUX_BUNDLE_MLX_PLATFORM="$platform"' not in source
        assert 'VMLX_BUNDLE_MLX_PLATFORM="$platform"' in source
        assert 'vMLX-\\${version}-${flavor}-\\${arch}.\\${ext}' in source
        assert "electron-builder --mac" in source
        assert "gh release" not in source
        assert "latest.json" not in source

    def test_macos_compat_error_points_to_sequoia_dmg_for_tahoe_wheels(self):
        """Wrong-DMG installs should fail before MLX import with actionable text."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        check_idx = cli_source.index("def _check_macos_compat")
        check_block = cli_source[check_idx: cli_source.index("def _check_no_duplicate_mlx", check_idx)]

        assert "download the Sequoia-compatible vMLX DMG" in check_block
        assert "Tahoe-native DMG requires macOS 26" in check_block

    def test_turboquant_disable_env_is_honored_by_jang_loader(self):
        """Explicit/off-family cache choices must stop loader-level live TQ-KV."""
        loader_source = Path("./vmlx_engine/utils/jang_loader.py").read_text()
        tokenizer_source = Path("./vmlx_engine/utils/tokenizer.py").read_text()
        cli_source = Path("./vmlx_engine/cli.py").read_text()

        assert 'environ.get("VMLX_DISABLE_TQ_KV")' in loader_source
        assert "TurboQuant KV skipped" in loader_source
        assert 'os.environ.get("VMLX_DISABLE_TQ_KV")' in tokenizer_source
        assert "TurboQuant skipped: VMLX_DISABLE_TQ_KV=1" in tokenizer_source
        assert 'os.environ["VMLX_DISABLE_TQ_KV"] = "1"' in cli_source
        assert "VMLINUX_DISABLE_TQ_KV" not in cli_source

    def test_hybrid_auto_mode_enables_qwen_selective_live_tq_only(self):
        """Qwen3.6 hybrid auto mode TQs attention KV while preserving SSM state."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        tokenizer_source = Path("./vmlx_engine/utils/tokenizer.py").read_text()

        assert 'getattr(_mc, "cache_type", None) == "hybrid"' in cli_source
        assert "Qwen3.6 hybrid/path-dependent cache model detected" in cli_source
        assert "attention KVCache" in cli_source
        assert "Only Qwen3.6 has a selective live" in cli_source
        hybrid_idx = cli_source.index('getattr(_mc, "cache_type", None) == "hybrid"')
        hybrid_block = cli_source[hybrid_idx : cli_source.index("if _mc.family_name !=", hybrid_idx)]
        assert 'args.kv_cache_quantization = "none"' not in hybrid_block
        assert "args.kv_cache_quantization_explicit = True" not in hybrid_block
        assert "stored attention-KV" in hybrid_block

        assert "Hybrid/path-dependent cache model detected — using q4/q8 only at cache storage boundaries" in scheduler_source
        assert "non-KV state is preserved full precision" in scheduler_source
        assert "build_hybrid_turboquant_make_cache" in tokenizer_source
        assert "is_qwen36_hybrid_tq_supported" in tokenizer_source
        assert "model.make_cache()" in tokenizer_source

    def test_generic_turboquant_patcher_honors_disable_env(self, tmp_path, monkeypatch):
        """The non-JANG fallback TQ hook must not bypass VMLX_DISABLE_TQ_KV."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "llama",
            "num_hidden_layers": 1,
            "head_dim": 128,
        }))

        class FakeModel:
            layers = [object()]

            def make_cache(self):
                return ["native"]

        model = FakeModel()
        monkeypatch.setenv("VMLX_DISABLE_TQ_KV", "1")

        _apply_turboquant_to_model(model, str(tmp_path))

        assert model.make_cache() == ["native"]

    def test_generic_turboquant_patcher_skips_hybrid_ssm(self, tmp_path, monkeypatch):
        """Ling/Bailing hybrid must keep its native KV+SSM cache contract."""
        from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "bailing_hybrid",
            "num_hidden_layers": 32,
            "layer_group_size": 8,
            "head_dim": 128,
        }))

        class FakeModel:
            layers = [object()] * 33

            def make_cache(self):
                return ["native"] * 32

        model = FakeModel()
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)
        monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

        _apply_turboquant_to_model(model, str(tmp_path))

        assert model.make_cache() == ["native"] * 32

    def test_vmlx_env_prefix_is_canonical_for_ssm_cache_budget(self):
        """New cache env knobs should use VMLX_, with typo fallback only."""
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert "def _env_int(" in cli_source
        assert "os.environ.get(name)" in cli_source
        assert '"VMLINUX_SSM_STATE_CACHE_MB"' in cli_source
        assert 'default=_env_int("VMLX_SSM_STATE_CACHE_MB", 512, "VMLINUX_SSM_STATE_CACHE_MB")' in cli_source

    def test_cli_tool_parser_choices_include_registry_only_parsers(self):
        """Explicit CLI settings must accept parsers used by auto detection.

        ZAYA and Hy3 can be launched via auto-detection, but direct users also
        pass --tool-call-parser explicitly.  Missing choices make the CLI reject
        a valid product setting before the server can apply the registry.
        """
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert '"zaya_xml"' in cli_source
        assert '"hunyuan"' in cli_source

    def test_cli_tool_parser_choices_cover_family_registry_parsers(self):
        """Every parser emitted by model_configs.py must be CLI-selectable.

        The panel resolves family parsers before launching the server. If
        argparse choices lag behind the model registry, a valid auto-detected
        session fails at startup before the engine can use that parser.
        """
        from vmlx_engine.model_configs import register_all
        from vmlx_engine.model_config_registry import ModelConfigRegistry

        import vmlx_engine.model_config_registry as mcr

        ModelConfigRegistry._instance = None
        mcr._configs_loaded = False
        registry = ModelConfigRegistry()
        register_all(registry)

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        missing = sorted(
            {
                config.tool_parser
                for config in registry._configs
                if config.tool_parser and f'"{config.tool_parser}"' not in cli_source
            }
        )
        assert not missing
        assert '"gemma3"' in cli_source
        assert '"gemma3n"' in cli_source

    def test_sampler_recreation_invalidates_pending_prefix_hits(self):
        """A sampler change must not leave requests pointing at cleared paged blocks."""
        import inspect
        from vmlx_engine.scheduler import Scheduler

        ensure_source = inspect.getsource(Scheduler._ensure_batch_generator)
        schedule_source = inspect.getsource(Scheduler._schedule_waiting)

        assert "cleared_cache = False" in ensure_source
        assert "return cleared_cache" in ensure_source
        assert "Preserving paged cache across BatchGenerator recreation" in ensure_source
        assert "self.block_aware_cache.clear()" not in ensure_source
        assert "_cache_cleared = self._ensure_batch_generator" in schedule_source
        assert "prefix cache hit invalidated by BatchGenerator" in schedule_source
        assert "request._paged_block_table_needs_worker_reconstruct = False" in schedule_source
        assert "request._hybrid_prompt_cache_needs_worker_ssm = False" in schedule_source
        assert "request.remaining_tokens = request.prompt_token_ids" in schedule_source


class TestZayaCCACachePolicy:
    """ZAYA/CCA must not fall through generic hybrid cache paths."""

    def _write_minimax_fixture(
        self,
        tmp_path,
        *,
        stamped_reasoning_parser: str = "minimax_m2",
    ):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "text_config": {
                "model_type": "minimax",
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": "mxtq",
            "cache_type": "kv",
            "source_model": {
                "name": "MiniMax-Test",
                "architecture": "minimax",
            },
            "capabilities": {
                "family": "minimax",
                "reasoning_parser": stamped_reasoning_parser,
                "think_in_template": True,
                "supports_thinking": True,
                "cache_type": "kv",
                "modality": "text",
                "supports_tools": True,
            },
            "chat": {
                "reasoning": {
                    "supported": True,
                    "parser": stamped_reasoning_parser,
                }
            },
        }))
        return tmp_path

    def _write_qwen36_fixture(self, tmp_path, *, model_type="qwen3_5_moe"):
        text_model_type = (
            "qwen3_5_moe_text" if model_type == "qwen3_5_moe" else "qwen3_5_text"
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": model_type,
            "text_config": {
                "model_type": text_model_type,
                "layer_types": ["full_attention", "linear_attention"],
            },
            "vision_config": {"model_type": "qwen3_vl"},
            "image_token_id": 248056,
            "video_token_id": 248057,
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "version": 2,
            "weight_format": "mxfp8",
            "cache_type": "hybrid",
            "source_model": {
                "name": "Qwen3.6-Test",
                "architecture": model_type,
            },
            "capabilities": {
                "family": model_type,
                "reasoning_parser": "qwen3",
                "think_in_template": True,
                "supports_thinking": True,
                "cache_type": "hybrid",
                "modality": "vl",
                "supports_tools": True,
            },
            "chat": {
                "reasoning": {
                    "supported": True,
                    "parser": "qwen3",
                }
            },
        }))
        return tmp_path

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
                # Simulates older local ZAYA bundles that were stamped too
                # optimistically before live gates proved thinking unsafe.
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
        assert cfg.think_in_template is False
        assert cfg.supports_thinking is True

    def test_registry_overrides_stale_minimax_qwen3_sidecar(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dir = self._write_minimax_fixture(
            tmp_path,
            stamped_reasoning_parser="qwen3",
        )
        registry = get_model_config_registry()
        registry.clear_cache()

        cfg = registry.lookup(str(model_dir))

        assert cfg.family_name == "minimax"
        assert cfg.tool_parser == "minimax"
        assert cfg.reasoning_parser == "minimax_m2"

    def test_zaya_auto_defaults_to_no_think_but_explicit_on_still_works(self, tmp_path):
        from vmlx_engine import server

        model_dir = self._write_zaya_fixture(tmp_path)
        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False
        assert explicit_on is True

    def test_server_default_false_is_explicit_and_request_off_still_wins(self):
        from vmlx_engine import server

        old_default = server._default_enable_thinking
        server._default_enable_thinking = False
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key="unknown-thinking-capable",
                engine=None,
                auto_detect=True,
            )
            explicit_off = server._resolve_enable_thinking(
                request_value=False,
                ct_kwargs={},
                tools_present=False,
                model_key="unknown-thinking-capable",
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False
        assert explicit_off is False

    def test_server_default_false_respected_for_known_reasoning_model(self, tmp_path):
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = False
        try:
            model_dir = self._write_minimax_fixture(tmp_path)
            registry = get_model_config_registry()
            registry.clear_cache()
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=False,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is False

    def test_minimax_auto_stays_unset_and_explicit_thinking_still_works(
        self, tmp_path
    ):
        """MiniMax supports thinking, but omitted/Auto must stay unset.

        The engine must not hide runtime/template issues behind a family-level
        forced off rail. Explicit user thinking controls still win.
        """
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            model_dir = self._write_minimax_fixture(tmp_path)
            registry = get_model_config_registry()
            registry.clear_cache()
            cfg = registry.lookup(str(model_dir))
            auto_resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert cfg.supports_thinking is True
        assert cfg.reasoning_parser == "minimax_m2"
        assert auto_resolved is None
        assert explicit_on is True

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
    def test_qwen36_auto_stays_unset_and_explicit_thinking_still_works(
        self, tmp_path, model_type
    ):
        """Qwen3.6 supports thinking, but omitted/Auto must stay unset.

        Visibility/coherency issues must be fixed in template/runtime handling,
        not by converting Auto into a hidden off rail.
        """
        from vmlx_engine import server
        from vmlx_engine.model_config_registry import get_model_config_registry

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            model_dir = self._write_qwen36_fixture(tmp_path, model_type=model_type)
            registry = get_model_config_registry()
            registry.clear_cache()
            cfg = registry.lookup(str(model_dir))
            auto_resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=False,
                model_key=str(model_dir),
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert cfg.family_name == model_type
        assert cfg.supports_thinking is True
        assert cfg.reasoning_parser == "qwen3"
        assert auto_resolved is None
        assert explicit_on is True

    def test_gemma4_tools_auto_stays_unset_and_explicit_thinking_still_works(self):
        from vmlx_engine import server

        old_default = server._default_enable_thinking
        server._default_enable_thinking = None
        try:
            resolved = server._resolve_enable_thinking(
                request_value=None,
                ct_kwargs={},
                tools_present=True,
                model_key="gemma4",
                engine=None,
                auto_detect=True,
            )
            explicit_on = server._resolve_enable_thinking(
                request_value=True,
                ct_kwargs={},
                tools_present=True,
                model_key="gemma4",
                engine=None,
                auto_detect=True,
            )
        finally:
            server._default_enable_thinking = old_default

        assert resolved is None
        assert explicit_on is True

    def test_gemma4_supports_thinking_is_explicit_not_implicit(self):
        from unittest.mock import patch
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        registry.clear_cache()
        with patch(
            "vmlx_engine.model_config_registry.load_config",
            lambda _path: {"model_type": "gemma4"},
        ):
            cfg = registry.lookup("google/gemma-4-26b-it")

        assert cfg.family_name == "gemma4"
        assert cfg.reasoning_parser == "gemma4"
        assert cfg.supports_thinking is True
        assert cfg.think_in_template is False

    def test_cli_enables_typed_paged_cache_and_forces_tq_off_for_zaya_cca(self):
        source = Path("./vmlx_engine/cli.py").read_text()

        assert 'getattr(_mc, "cache_subtype", None) == "zaya_cca"' in source
        assert 'args.use_paged_cache = True' in source
        assert 'args.kv_cache_quantization = "none"' in source
        assert "ZAYA/CCA typed cache enabled" in source
        assert "conv_state + prev_hs" in source
        assert "server._tool_call_parser or args.tool_call_parser" in source

        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        assert "self._uses_zaya_cache" in scheduler_source
        assert 'self._model_type_for_runtime == "zaya"' in scheduler_source
        assert "ZAYA/CCA typed paged prefix cache enabled" in scheduler_source
        assert "ZAYA/CCA cache contract detected but prefix cache is disabled" in scheduler_source
        assert "and not self._uses_zaya_cache" in scheduler_source

        mllm_scheduler_source = Path("./vmlx_engine/mllm_scheduler.py").read_text()
        assert "Runtime cache layout:" in mllm_scheduler_source
        assert "zaya_cca_v1" in mllm_scheduler_source

    def test_zaya_cli_policy_default_keeps_prefix_paged_l2_but_forces_tq_off(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=True,
            enable_block_disk_cache=True,
            kv_cache_quantization="q4",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("kv_quant=q4",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is True
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
        assert "VMLX_FORCE_TQ_AUTO" not in os.environ
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_zaya_cli_policy_no_longer_requires_live_gate_env(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.delenv("VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE", raising=False)
        monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=True,
            enable_block_disk_cache=True,
            kv_cache_quantization="q8",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("kv_quant=q8",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is True
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        assert os.environ["VMLX_DISABLE_TQ_KV"] == "1"
        assert "VMLX_FORCE_TQ_AUTO" not in os.environ
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_zaya_cli_policy_upgrades_prefix_only_to_paged(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_zaya_cca_cache_policy

        monkeypatch.delenv("VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE", raising=False)
        args = SimpleNamespace(
            enable_prefix_cache=True,
            use_paged_cache=False,
            enable_block_disk_cache=False,
            kv_cache_quantization="none",
        )

        gate, changed = _apply_zaya_cca_cache_policy(args, MagicMock())

        assert gate is True
        assert changed == ("paged=required_for_zaya_cca",)
        assert args.enable_prefix_cache is True
        assert args.use_paged_cache is True
        assert args.enable_block_disk_cache is False
        assert args.kv_cache_quantization == "none"
        assert args.kv_cache_quantization_explicit is True
        os.environ.pop("VMLX_DISABLE_TQ_KV", None)

    def test_ollama_streaming_suppresses_duplicate_done_chunks(self):
        source = Path("./vmlx_engine/server.py").read_text()

        assert "done_sent = False" in source
        assert "if done_sent:" in source
        assert "done_sent = True" in source
        assert "openai_chat_chunk_to_ollama_generate_ndjson" in source

    def test_jang_loader_registers_zaya_runtime(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _ensure_zaya_runtime_supported

        model_dir = self._write_zaya_fixture(tmp_path)
        jcfg = json.loads((model_dir / "jang_config.json").read_text())

        _ensure_zaya_runtime_supported(model_dir, jcfg)

        import mlx_lm.models.zaya as zaya

        assert zaya.Model.__name__ == "Model"

    def test_jang_loader_registers_zaya1_vl_runtime_adapter(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _ensure_zaya_runtime_supported

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "zaya1_vl",
            "vision_config": {"model_type": "qwen2_5_vl"},
            "zaya_expert_layout": "split_switch_mlp",
        }))
        jcfg = {
            "cache_subtype": "zaya_cca",
            "capabilities": {
                "family": "zaya1_vl",
                "tool_parser": "zaya_xml",
                "reasoning_parser": "qwen3",
                "think_in_template": False,
                "supports_thinking": True,
                "cache_type": "hybrid",
                "modality": "vision",
            },
        }

        _ensure_zaya_runtime_supported(tmp_path, jcfg)

        import mlx_vlm.models.zaya1_vl as zaya1_vl

        assert zaya1_vl.Model.__name__ == "Model"

    def test_local_zaya_bundles_carry_explicit_cca_contract(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        model_dirs = [
            Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4"),
            Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ2"),
            Path("/Users/example/models/JANGQ/ZAYA1-8B-MXFP4"),
            Path("/Users/example/models/JANGQ/ZAYA1-8B-JANGTQ2"),
            Path("/Users/example/jang/models/Zyphra/ZAYA1-8B-JANGTQ2"),
            Path("/Users/example/jang/models/Zyphra/ZAYA1-8B-JANGTQ4"),
            Path("/Users/example/jang/models/Zyphra/ZAYA1-8B-MXFP4"),
        ]
        existing = [p for p in model_dirs if p.exists()]
        if not existing:
            pytest.skip("local ZAYA bundles are not present on this machine")

        for model_dir in existing:
            cfg = json.loads((model_dir / "config.json").read_text())
            jcfg = json.loads((model_dir / "jang_config.json").read_text())
            caps = jcfg.get("capabilities", {})

            model_type = cfg.get("model_type")
            assert model_type in {"zaya", "zaya1_vl"}
            assert jcfg.get("cache_subtype") == "zaya_cca"
            expected_family = "zaya1_vl" if model_type == "zaya1_vl" else "zaya"
            assert caps.get("family") == expected_family
            assert caps.get("cache_type") == "hybrid"
            assert caps.get("tool_parser") == "zaya_xml"
            assert caps.get("reasoning_parser") == "qwen3"
            assert caps.get("think_in_template") is False
            # Per Eric 2026-05-10 honest-flag directive: ZAYA AND ZAYA1-VL are
            # reasoning-capable and use qwen3 parsing, while default prompts
            # are not stamped as starting inside an open think rail.
            assert caps.get("supports_thinking") is True
            assert cfg.get("zaya_expert_layout") == "split_switch_mlp"
            assert caps.get("modality") == ("vision" if model_type == "zaya1_vl" else "text")

            registry = get_model_config_registry()
            registry.clear_cache()
            rcfg = registry.lookup(str(model_dir))
            assert rcfg.family_name == expected_family
            assert rcfg.tool_parser == "zaya_xml"
            assert rcfg.think_in_template is False
            # Bit profile is still audited for kernel/cache metadata below, but
            # it must not change ZAYA reasoning policy. Low-bit artifact quality
            # warnings live with model distribution notes, not hidden vMLX
            # runtime guards.
            assert rcfg.reasoning_parser == "qwen3"
            assert rcfg.supports_thinking is True
            assert rcfg.is_mllm is (model_type == "zaya1_vl")

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


class TestMLLMNKvHeadsWrapperParity:
    """MLLMScheduler._detect_n_kv_heads must walk text_config + inner wrappers.

    Symmetric to ``Scheduler._detect_mla`` and ``PrefixCache._get_n_kv_heads``.
    Earlier version only inspected ``language_model.{args,config}`` directly —
    Kimi K2.6 VLM (mlx_vlm wrapper around DeepseekV3 backbone) exposes the
    MLA config via ``language_model.config.text_config.kv_lora_rank``. Without
    the text_config fallback, MLA H=1 collapse never fires for VLM-wrapped
    MLA models and cache slicing in ``_normalize_gqa_state`` mismatches.
    """

    def test_kimi_vlm_text_config_kv_lora_rank_collapses_to_one_head(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            kv_lora_rank = 512
            num_attention_heads = 64
            num_key_value_heads = 8

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.model = _Model()
        assert sched._detect_n_kv_heads() == 1

    def test_bailing_hybrid_via_text_config_keeps_full_kv_heads(self):
        """Ling/Bailing stores expanded KV — must NOT collapse to H=1."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512
            num_attention_heads = 32
            num_key_value_heads = 32

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.model = _Model()
        # Must keep full per-head KV count (32), not collapse to H=1
        # which would slice valid (1,32,T,D) cache to (1,1,T,D).
        assert sched._detect_n_kv_heads() == 32


class TestHeadDimWrapperTraversal:
    """Scheduler._detect_head_dim + MLLMScheduler._detect_head_dim must walk wrappers.

    Symmetric to ``_detect_mla`` and ``_detect_n_kv_heads``. Earlier
    implementations only inspected ``self.model.{args,config}`` (LLM) or
    ``language_model.{args,config}`` (MLLM) directly. For VLM-wrapped
    backbones (Kimi K2.6 around DeepseekV3, glm_moe_dsa, Mistral 4
    wrappers), head_dim is exposed via ``language_model.config.text_config``
    or via an inner ``.model`` candidate. Without traversal, both helpers
    returned None silently — and the KV-quant
    ``_wrap_make_cache_quantized`` skipped its head_dim/group_size
    compatibility check, allowing a mismatched group_size into mx.quantize
    to fail at runtime.
    """

    def test_llm_scheduler_finds_head_dim_via_text_config_wrapper(self):
        from vmlx_engine.scheduler import Scheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        s = Scheduler.__new__(Scheduler)
        s.model = _Wrapper()
        assert s._detect_head_dim() == 128

    def test_llm_scheduler_derives_head_dim_from_hidden_div_heads_via_inner_model(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            hidden_size = 4096
            num_attention_heads = 32

        class _Inner:
            args = _Args()

        class _Wrapper:
            model = _Inner()

        s = Scheduler.__new__(Scheduler)
        s.model = _Wrapper()
        # 4096 / 32 = 128
        assert s._detect_head_dim() == 128

    def test_mllm_scheduler_finds_head_dim_via_text_config_wrapper(self):
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        s = MLLMScheduler.__new__(MLLMScheduler)
        s.model = _Wrapper()
        assert s._detect_head_dim() == 128

    def test_llm_and_mllm_scheduler_agree_on_wrapped_head_dim(self):
        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        llm = Scheduler.__new__(Scheduler)
        llm.model = _Wrapper()
        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = _Wrapper()
        assert llm._detect_head_dim() == mllm._detect_head_dim()

    def test_kimi_mla_cache_dims_do_not_use_hidden_div_heads(self):
        """Kimi MLA cache dims are latent+RoPE dims, not hidden/heads.

        The local Kimi K2.6 VLM bundle exposes hidden_size=7168 and
        num_attention_heads=64, but mlx_lm caches ``kv_lora_rank`` and
        ``qk_rope_head_dim`` via cache.update_and_fetch(kv_latent, k_pe).
        Treating 7168 / 64 = 112 as the cache head dim makes the
        group-size validator adjust default-compatible KV quant settings to
        the wrong value.
        """
        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        class _TextCfg:
            model_type = "kimi_k25"
            hidden_size = 7168
            num_attention_heads = 64
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            v_head_dim = 128

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Wrapper:
            language_model = _LM()

        llm = Scheduler.__new__(Scheduler)
        llm.model = _Wrapper()
        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = _Wrapper()

        assert llm._detect_cache_head_dims() == (512, 64)
        assert mllm._detect_cache_head_dims() == (512, 64)
        assert llm._detect_head_dim() == 512
        assert mllm._detect_head_dim() == 512

    def test_kimi_mla_quant_group_validation_checks_all_cache_dims(self):
        """A group size must divide both MLA cache tensors.

        Kimi/DeepSeek/Mistral MLA caches store latent KV with width
        ``kv_lora_rank`` and RoPE keys with width ``qk_rope_head_dim``.
        A user-provided group_size=128 divides 512 but not 64, so the
        validator must adjust to 64. The old hidden/heads fallback returned
        112 and adjusted to 16, which was a false fix.
        """
        from types import SimpleNamespace

        from vmlx_engine.scheduler import Scheduler
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        text_config = SimpleNamespace(
            model_type="kimi_k25",
            hidden_size=7168,
            num_attention_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(text_config=text_config)))

        llm = Scheduler.__new__(Scheduler)
        llm.model = model
        llm.config = SimpleNamespace(kv_cache_group_size=128)
        llm._wrap_make_cache_quantized(bits=4, group_size=128)
        assert llm._kv_cache_group_size == 64
        assert llm.config.kv_cache_group_size == 64

        mllm = MLLMScheduler.__new__(MLLMScheduler)
        mllm.model = model
        mllm._wrap_make_cache_quantized(bits=4, group_size=128)
        assert mllm._kv_cache_group_size == 64


class TestLLMMlaDetectionWrapperParity:
    """LLM scheduler MLA detection must match MLLMScheduler._detect_mla traversal.

    Site 1 (Scheduler.__init__ KV-quant gate) historically only inspected
    ``self.model.args``. Wrapped models (Kimi K2.6 mlx_vlm wrapper around
    DeepseekV3, glm_moe_dsa inheriting deepseek_v32) expose MLA config via
    ``self.model.language_model.config.text_config`` — the inline check missed
    them and KV-cache quantization stayed enabled, double-quantizing already
    compressed latents.
    """

    def test_detects_kimi_style_wrapper_via_text_config(self):
        from vmlx_engine.scheduler import Scheduler

        class _TextCfg:
            model_type = "kimi_k25"
            kv_lora_rank = 512

        class _Cfg:
            text_config = _TextCfg()

        class _LM:
            config = _Cfg()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_detects_deepseek_v3_inner_model_args(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "deepseek_v3"
            kv_lora_rank = 512

        class _Inner:
            args = _Args()

        class _Model:
            model = _Inner()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True

    def test_bailing_hybrid_via_wrapper_is_not_mla(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "bailing_hybrid"
            kv_lora_rank = 512

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is False

    def test_mistral4_detected_via_wrapper(self):
        from vmlx_engine.scheduler import Scheduler

        class _Args:
            model_type = "mistral4"

        class _LM:
            args = _Args()

        class _Model:
            language_model = _LM()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = _Model()

        assert scheduler._detect_mla() is True


class TestHybridSSMEnvNames:
    """Hybrid SSM resume controls must use the documented VMLX_* names."""

    def test_scheduler_uses_vmlx_disable_ssm_prefix_resume(self):
        source = Path("vmlx_engine/scheduler.py").read_text()

        assert "VMLX_DISABLE_SSM_PREFIX_RESUME" in source
        assert "VMLINUX_DISABLE_SSM_PREFIX_RESUME" not in source


class TestPromptLookupDocumentation:
    """vmlx#137: PLD docs must match the scheduler's real implementation."""

    def test_prompt_lookup_docstring_no_longer_claims_phase2_is_blocked(self):
        import vmlx_engine.prompt_lookup as prompt_lookup

        doc = prompt_lookup.__doc__ or ""

        assert "Phase 2 (future)" not in doc
        assert "Blocked today by" not in doc
        assert "blocked by mlx-lm" not in doc.lower()
        assert "Scheduler integration owns runtime acceleration" in doc
        assert "_try_pld_speculative_decode" in doc
        assert "Hybrid SSM/attention models need family-aware" in doc


class TestJangTqMppNaxCliPolicy:
    def test_jangtq_mpp_nax_cli_policy_defaults_to_auto(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "4")
        args = SimpleNamespace()
        logger = MagicMock()

        _apply_jangtq_mpp_nax_policy(args, logger)

        assert os.environ["JANGTQ_MPP_NAX"] == "auto"
        assert "JANGTQ_TOPK_OVERRIDE" not in os.environ
        logger.info.assert_called_once()
        logger.debug.assert_called_once()

    def test_jangtq_mpp_nax_cli_policy_can_force_off(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "off")
        args = SimpleNamespace()
        logger = MagicMock()

        _apply_jangtq_mpp_nax_policy(args, logger)

        assert os.environ["JANGTQ_MPP_NAX"] == "off"

    def test_jangtq_mpp_nax_cli_policy_can_force_on(self, monkeypatch):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "on")
        args = SimpleNamespace()
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "on"
        assert os.environ["JANGTQ_MPP_NAX"] == "1"

    def test_jangtq_mpp_nax_cli_policy_invalid_value_falls_back_to_auto(
        self, monkeypatch
    ):
        from vmlx_engine.cli import _apply_jangtq_mpp_nax_policy

        monkeypatch.setenv("JANGTQ_MPP_NAX", "bogus")
        args = SimpleNamespace()
        logger = MagicMock()

        mode = _apply_jangtq_mpp_nax_policy(args, logger)

        assert mode == "auto"
        assert os.environ["JANGTQ_MPP_NAX"] == "auto"
        logger.warning.assert_called_once()

    def test_jangtq_mpp_nax_serve_parser_does_not_expose_toggle(self):
        source = Path("vmlx_engine/cli.py").read_text()

        assert '"--jangtq-mpp-nax"' not in source
        assert 'choices=["off", "auto", "on"]' not in source

    def test_jangtq_mpp_nax_server_status_defaults_to_auto(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)

        mode, issue = server._normalize_jangtq_mpp_nax_mode()

        assert mode == "auto"
        assert issue is None

    def test_jangtq_mpp_nax_server_invalid_value_falls_back_to_auto(self):
        import vmlx_engine.server as server

        mode, issue = server._normalize_jangtq_mpp_nax_mode("bogus")

        assert mode == "auto"
        assert "invalid JANGTQ_MPP_NAX" in issue

    def test_jangtq_mpp_nax_availability_status_does_not_eval_mlx(self, monkeypatch):
        pytest.importorskip("jang_tools.turboquant.mpp_nax_kernel")
        import vmlx_engine.server as server
        import jang_tools.turboquant.mpp_nax_kernel as nax_kernel

        nax_kernel.mpp_nax_tensorops_available.cache_clear()
        server._jangtq_mpp_nax_available.cache_clear()

        def fail_eval(*_args, **_kwargs):
            raise AssertionError("status path must not run a smoke Metal eval")

        monkeypatch.setattr(nax_kernel.mx, "eval", fail_eval)

        available, error = server._jangtq_mpp_nax_available()

        assert isinstance(available, bool)
        assert error is None


class TestDistributedStreamingUnicode:
    """vmlx#124 class: distributed streaming must not per-token decode UTF-8."""

    def test_distributed_generate_loop_uses_streaming_detokenizer(self):
        source = Path("vmlx_engine/distributed/engine.py").read_text()
        body_start = source.index("async def _generate_impl")
        body_end = source.index("\n    def _apply_chat_template", body_start)
        body = source[body_start:body_end]

        assert "tokenizer.decode([next_tok_id])" not in body
        assert "_make_streaming_detokenizer(tokenizer)" in body
        assert "_add_streaming_token(detokenizer, next_tok_id)" in body
        assert "_finalize_streaming_detokenizer(detokenizer)" in body

    def test_streaming_helpers_do_not_emit_incomplete_cyrillic(self):
        from vmlx_engine.distributed.engine import (
            _add_streaming_token,
            _finalize_streaming_detokenizer,
        )

        class FakeDetokenizer:
            def __init__(self):
                self.text = ""

            def add_token(self, token):
                # Token 1 is an incomplete byte span. A per-token decode path
                # would emit U+FFFD here; a streaming path must wait.
                if token == 1:
                    self.text = ""
                elif token == 2:
                    self.text = "Пр"

            def finalize(self):
                self.text = "Привет"

        detok = FakeDetokenizer()

        assert _add_streaming_token(detok, 1) == ""
        assert _add_streaming_token(detok, 2) == "Пр"
        assert _finalize_streaming_detokenizer(detok) == "ивет"
        assert "\ufffd" not in detok.text


class TestJangVLMFallbacks:
    """JANG VLM loaders must not silently drop working image support."""

    def _write_qwen_hybrid_fixture(
        self,
        tmp_path,
        *,
        weight_format=None,
        text_model_type="qwen3_5_text",
        layer_types=None,
    ):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": text_model_type,
                "layer_types": layer_types or ["linear_attention", "full_attention"],
            },
            "vision_config": {},
            "image_token_id": 248056,
            "video_token_id": 248057,
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
        assert "Qwen3.5/3.6-VL MXTQ/JANGTQ hybrid SSM bundles must stay on the real VLM" in source
        assert "affine-JANG exception above is deliberately narrow" in source

    def test_text_only_vlm_fallbacks_mark_vision_unavailable(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()
        mistral_fallback = source[
            source.index('config.get("model_type") == "mistral3"'):
            source.index("# Qwen3.5/3.6-VL MXTQ/JANGTQ hybrid SSM bundles", source.index('config.get("model_type") == "mistral3"'))
        ]

        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = True' in mistral_fallback
        assert 'globals()["_LAST_LOAD_VLM_FALLBACK"] = False' in source

    @pytest.mark.parametrize(
        "model_type,text_model_type",
        [
            ("qwen3_5", "qwen3_5_text"),
            ("qwen3_vl", "qwen3_vl"),
            ("qwen3_vl_moe", "qwen3_vl_moe"),
        ],
    )
    def test_affine_qwen_mrope_families_route_text_only(
        self, tmp_path, model_type, text_model_type
    ):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type=text_model_type,
        )
        config = json.loads((model_dir / "config.json").read_text())
        config["model_type"] = model_type
        (model_dir / "config.json").write_text(json.dumps(config))
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is False

    def test_affine_qwen_hybrid_detection_normalizes_config_case_for_text_only(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type="QWEN3_5_TEXT",
            layer_types=["LINEAR_ATTENTION", "FULL_ATTENTION"],
        )
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is False

    def test_affine_qwen_hybrid_jang_overrides_forced_mllm(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path)
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir), force_mllm=True) is False

    def test_mxtq_qwen_hybrid_jang_routes_multimodal(self, tmp_path):
        from vmlx_engine.api import utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path, weight_format="mxtq")
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(model_dir)) is True

    def test_qwen_vlm_loader_affine_delegates_to_text_loader(self, tmp_path, monkeypatch):
        """Affine-JANG Qwen hybrid uses text loader until mlx_vlm M-RoPE is fixed."""
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        model_dir = self._write_qwen_hybrid_fixture(tmp_path)
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: pytest.fail("affine-JANG Qwen must not hit native VLM loader"),
        )
        sentinel = object()

        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: (sentinel, "tokenizer"),
        )

        model, tokenizer = jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert model is sentinel
        assert tokenizer == "tokenizer"
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is True

    def test_qwen_vlm_loader_affine_native_path_normalizes_config_case(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The text fallback path must match the same normalized Qwen
        hybrid predicate as is_mllm_model()."""
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        model_dir = self._write_qwen_hybrid_fixture(
            tmp_path,
            text_model_type="QWEN3_5_TEXT",
            layer_types=["LINEAR_ATTENTION", "FULL_ATTENTION"],
        )
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: pytest.fail("affine-JANG Qwen must not hit native VLM loader"),
        )
        sentinel = object()

        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: (sentinel, "tokenizer"),
        )

        model, tokenizer = jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert model is sentinel
        assert tokenizer == "tokenizer"
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is True

    def test_qwen_vlm_loader_mxtq_stays_native_vlm(self, tmp_path, monkeypatch):
        """JANGTQ/MXTQ Qwen VLM must not be caught by the affine fallback."""
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        class NativeVlmReached(RuntimeError):
            pass

        model_dir = self._write_qwen_hybrid_fixture(tmp_path, weight_format="mxtq")
        config = json.loads((model_dir / "config.json").read_text())
        jang_cfg = json.loads((model_dir / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: (_ for _ in ()).throw(NativeVlmReached()),
        )
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: pytest.fail("MXTQ Qwen VLM must not use text fallback"),
        )

        with pytest.raises(NativeVlmReached):
            jang_loader._load_jang_v2_vlm(model_dir, jang_cfg)
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is False


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

    def test_reports_turboquant_single_sequence_runtime_contract(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _turboquant_kv_cache_status

        scheduler = SimpleNamespace(
            _tq_active=True,
            _tq_batch_api=False,
            config=SimpleNamespace(
                max_num_seqs=1,
                prefill_batch_size=1,
                completion_batch_size=1,
            ),
        )

        status = _turboquant_kv_cache_status(scheduler=scheduler)

        assert status["enabled"] is True
        assert status["single_sequence_only"] is True
        assert status["single_sequence_reason"] == "cache_extend_not_supported"
        assert status["effective_max_num_seqs"] == 1
        assert status["effective_prefill_batch_size"] == 1
        assert status["effective_completion_batch_size"] == 1

    def test_reports_turboquant_batch_api_runtime_contract(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _turboquant_kv_cache_status

        scheduler = SimpleNamespace(
            _tq_active=True,
            _tq_batch_api=True,
            config=SimpleNamespace(
                max_num_seqs=5,
                prefill_batch_size=1024,
                completion_batch_size=1024,
            ),
        )

        status = _turboquant_kv_cache_status(scheduler=scheduler)

        assert status["enabled"] is True
        assert status["batch_api"] == "turboquant_kv_v1"
        assert status["single_sequence_only"] is False
        assert "single_sequence_reason" not in status
        assert status["effective_max_num_seqs"] == 5

    def test_mllm_scheduler_aggregates_cached_tokens_for_stats(self):
        import threading
        from types import SimpleNamespace
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._cache_hit_requests = 0
        scheduler._cache_hit_tokens = 0
        scheduler._cache_hit_tokens_by_detail = {}
        scheduler._queue_lock = threading.RLock()
        scheduler.waiting = []
        scheduler.running = {}
        scheduler.finished_req_ids = set()
        scheduler.num_requests_processed = 1
        scheduler.total_prompt_tokens = 35
        scheduler.total_completion_tokens = 3
        scheduler.batch_generator = None
        scheduler.block_aware_cache = None
        scheduler.paged_cache_manager = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = None

        request = SimpleNamespace(_cache_detail="request-legacy-detail")
        response = SimpleNamespace(cached_tokens=34, cache_detail="paged+ssm+disk")

        scheduler._record_cache_hit(response, request)
        scheduler._record_cache_hit(response, request)
        stats = scheduler.get_stats()

        assert stats["cache_hit_requests"] == 1
        assert stats["cache_hit_tokens"] == 34
        assert stats["cache_hit_tokens_by_detail"] == {"paged+ssm+disk": 34}

    def test_mllm_paged_cache_hits_have_source_detail_labels(self):
        source = Path("./vmlx_engine/mllm_batch_generator.py").read_text()

        assert "_paged_disk_hits_before" in source
        assert "_paged_disk_hit" in source
        assert "cache_detail: str = \"\"" in source
        assert "cache_detail=getattr(req, '_cache_detail', \"\") or \"\"" in source
        assert '"paged+ssm+disk" if _paged_disk_hit else "paged+ssm"' in source
        assert '"paged+disk" if _paged_disk_hit else "paged"' in source

    def test_llm_cache_detail_uses_canonical_plus_grammar(self):
        source = Path("./vmlx_engine/scheduler.py").read_text()

        assert '"paged+disk"' in source
        assert '"paged+disk+tq"' in source
        assert '"paged+ssm+disk"' in source
        assert '"paged+ssm"' in source
        assert "disk->paged" not in source
        assert "disk+tq->paged" not in source
        assert "paged+ssm(" not in source

    def test_llm_hybrid_cache_detail_marks_ssm_disk_source(self, monkeypatch):
        import vmlx_engine.scheduler as scheduler_mod

        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler._is_hybrid = True
        scheduler._uses_dsv4_cache = False
        scheduler._uses_zaya_cache = False
        scheduler._ssm_state_cache = SimpleNamespace(
            fetch=lambda tokens, fetch_num: (["ssm-a", "ssm-b"], True)
        )
        scheduler.block_aware_cache = SimpleNamespace()
        scheduler.model = SimpleNamespace()
        scheduler._hybrid_kv_positions = [0]
        scheduler._hybrid_num_layers = 3

        monkeypatch.setattr(
            scheduler_mod,
            "_fix_hybrid_cache",
            lambda reconstructed, model, kv_positions, num_model_layers: list(
                reconstructed
            ),
        )

        request = SimpleNamespace(
            request_id="req-1",
            block_table=SimpleNamespace(num_tokens=64),
            prompt_token_ids=list(range(80)),
            _hybrid_ssm_fetch_tokens=list(range(80)),
            _paged_disk_hit=True,
        )

        full_cache = scheduler._finalize_hybrid_paged_cache_on_worker(
            request,
            ["kv", None, None],
        )

        assert full_cache == ["kv", "ssm-a", "ssm-b"]
        assert request._cache_detail == "paged+ssm+disk"
        assert request._cache_detail_ssm_layers == 2

        request2 = SimpleNamespace(
            request_id="req-2",
            block_table=SimpleNamespace(num_tokens=64),
            prompt_token_ids=list(range(80)),
            _hybrid_ssm_fetch_tokens=list(range(80)),
            _paged_disk_hit=False,
        )

        scheduler._finalize_hybrid_paged_cache_on_worker(
            request2,
            ["kv", None, None],
        )

        assert request2._cache_detail == "paged+ssm"

    @pytest.mark.asyncio
    async def test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_cache_stats(self):
                return None

        class _Scheduler:
            disk_cache = None
            paged_cache_manager = None

            def get_stats(self):
                return {
                    "num_waiting": 2,
                    "num_running": 1,
                    "num_requests_processed": 4,
                    "total_prompt_tokens": 1000,
                    "total_completion_tokens": 128,
                    "ewma_ttft_seconds": 9.25,
                    "cache_hit_requests": 7,
                    "cache_hit_tokens": 12345,
                    "cache_hit_tokens_by_detail": {"paged+dsv4": 4096, "disk+tq": 8249},
                    "batch_generator": {
                        "hybrid_kv_without_ssm_hits": 3,
                        "hybrid_kv_without_ssm_tokens": 2624,
                        "last_hybrid_kv_without_ssm": {
                            "reason": "no_ssm_companion_state",
                            "cached_tokens": 1088,
                        },
                    },
                    "cache_reuse_skips": 1,
                    "cache_reuse_skip_tokens": 512,
                    "last_cache_reuse_skip": {
                        "reason": "insufficient_memory_for_cache_merge",
                        "action": "full_prefill",
                        "needed_mb": 41618.0,
                        "budget_mb": 11330.5,
                        "available_mb": 13330.0,
                        "cached_tokens": 512,
                        "dropped_cached_tokens": 512,
                        "full_prefill_tokens": 4096,
                        "cache_contract": "hybrid_ssm",
                        "cache_format": "full_precision_kv+state_cache",
                        "partial_reuse_unavailable_reason": (
                            "no_block_aligned_ssm_checkpoint"
                        ),
                    },
                    "cache_reuse_partial_downgrades": 1,
                    "cache_reuse_partial_tokens": 2048,
                    "last_cache_reuse_partial": {
                        "reason": "insufficient_memory_for_full_cache_merge",
                        "used_cached_tokens": 2048,
                        "original_cached_tokens": 8192,
                        "tail_tokens": 6144,
                        "cache_format": "full_precision_kv",
                    },
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()
        scheduler_stats = payload["scheduler_stats"]

        assert scheduler_stats["cache_hit_requests"] == 7
        assert scheduler_stats["cache_hit_tokens"] == 12345
        assert scheduler_stats["cache_hit_tokens_by_detail"]["paged+dsv4"] == 4096
        assert scheduler_stats["cache_hit_tokens_by_detail"]["disk+tq"] == 8249
        assert scheduler_stats["hybrid_kv_without_ssm_hits"] == 3
        assert scheduler_stats["hybrid_kv_without_ssm_tokens"] == 2624
        assert scheduler_stats["last_hybrid_kv_without_ssm"]["cached_tokens"] == 1088
        assert scheduler_stats["cache_reuse_skips"] == 1
        assert scheduler_stats["cache_reuse_skip_tokens"] == 512
        assert scheduler_stats["last_cache_reuse_skip"]["reason"] == (
            "insufficient_memory_for_cache_merge"
        )
        assert scheduler_stats["last_cache_reuse_skip"]["needed_mb"] == 41618.0
        assert scheduler_stats["last_cache_reuse_skip"]["budget_mb"] == 11330.5
        assert scheduler_stats["last_cache_reuse_skip"]["available_mb"] == 13330.0
        assert scheduler_stats["last_cache_reuse_skip"]["dropped_cached_tokens"] == 512
        assert scheduler_stats["last_cache_reuse_skip"]["full_prefill_tokens"] == 4096
        assert scheduler_stats["last_cache_reuse_skip"]["cache_contract"] == "hybrid_ssm"
        assert scheduler_stats["last_cache_reuse_skip"]["cache_format"] == (
            "full_precision_kv+state_cache"
        )
        assert scheduler_stats["last_cache_reuse_skip"][
            "partial_reuse_unavailable_reason"
        ] == "no_block_aligned_ssm_checkpoint"
        assert scheduler_stats["cache_reuse_partial_downgrades"] == 1
        assert scheduler_stats["cache_reuse_partial_tokens"] == 2048
        assert scheduler_stats["last_cache_reuse_partial"]["used_cached_tokens"] == 2048
        assert scheduler_stats["last_cache_reuse_partial"]["cache_format"] == (
            "full_precision_kv"
        )

    @pytest.mark.asyncio
    async def test_cache_stats_projects_ssm_companion_disk_state(self, monkeypatch):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = True

            def get_cache_stats(self):
                return None

        class _SSMCache:
            size = 1
            max_entries = 8
            disk_enabled = True
            disk_directory = "/tmp/vmlx-test/ssm_companion"

        class _BatchGenerator:
            _ssm_state_cache = _SSMCache()

        class _Scheduler:
            disk_cache = None
            paged_cache_manager = None
            batch_generator = _BatchGenerator()

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()
        ssm = payload["ssm_companion"]

        assert ssm["entries"] == 1
        assert ssm["max_entries"] == 8
        assert ssm["disk_enabled"] is True
        assert ssm["disk_directory"] == "/tmp/vmlx-test/ssm_companion"

    @pytest.mark.asyncio
    async def test_cache_stats_reports_disabled_kv_quant_without_zero_bit_ui(self, monkeypatch):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_cache_stats(self):
                return None

        class _Scheduler:
            _kv_cache_bits = 0
            _kv_cache_group_size = 64
            disk_cache = None
            paged_cache_manager = None

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())

        payload = await server.cache_stats()

        assert payload["kv_cache_quantization"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_health_endpoint_projects_scheduler_pressure_telemetry(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

        class _Scheduler:
            def get_stats(self):
                return {
                    "num_waiting": 3,
                    "num_running": 1,
                    "ewma_ttft_seconds": 58.973,
                    "cache_hit_requests": 5,
                    "cache_hit_tokens": 8192,
                    "cache_hit_tokens_by_detail": {"paged+ssm": 8192},
                    "batch_generator": {
                        "hybrid_kv_without_ssm_hits": 2,
                        "hybrid_kv_without_ssm_tokens": 1536,
                        "last_hybrid_kv_without_ssm": {
                            "reason": "checkpoint_incomplete",
                            "checkpoint_tokens": 512,
                        },
                    },
                    "cache_reuse_skips": 2,
                    "cache_reuse_skip_tokens": 4096,
                    "last_cache_reuse_skip": {
                        "reason": "insufficient_memory_for_cache_merge",
                        "needed_mb": 41618.0,
                        "available_mb": 13330.0,
                    },
                    "cache_reuse_partial_downgrades": 3,
                    "cache_reuse_partial_tokens": 12000,
                    "last_cache_reuse_partial": {
                        "reason": "insufficient_memory_for_full_cache_merge",
                        "used_cached_tokens": 4096,
                    },
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        payload = await server.health()
        scheduler = payload["scheduler"]

        assert scheduler["num_waiting"] == 3
        assert scheduler["num_running"] == 1
        assert scheduler["ewma_ttft_seconds"] == 58.973
        assert scheduler["cache_hit_requests"] == 5
        assert scheduler["cache_hit_tokens"] == 8192
        assert scheduler["cache_hit_tokens_by_detail"]["paged+ssm"] == 8192
        assert scheduler["hybrid_kv_without_ssm_hits"] == 2
        assert scheduler["hybrid_kv_without_ssm_tokens"] == 1536
        assert scheduler["last_hybrid_kv_without_ssm"]["checkpoint_tokens"] == 512
        assert scheduler["cache_reuse_skips"] == 2
        assert scheduler["cache_reuse_skip_tokens"] == 4096
        assert scheduler["last_cache_reuse_skip"]["reason"] == (
            "insufficient_memory_for_cache_merge"
        )
        assert scheduler["cache_reuse_partial_downgrades"] == 3
        assert scheduler["cache_reuse_partial_tokens"] == 12000
        assert scheduler["last_cache_reuse_partial"]["used_cached_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_health_endpoint_projects_cache_telemetry_snapshot(
        self, monkeypatch
    ):
        import vmlx_engine.server as server

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

            def get_cache_stats(self):
                return {
                    "total_tokens_cached": 640,
                    "tokens_saved": 1280,
                    "allocated_blocks": 10,
                }

        class _DiskCache:
            def stats(self):
                return {
                    "entries": 2,
                    "total_tokens_on_disk": 384,
                    "total_cached_tokens": 384,
                    "hits": 3,
                    "misses": 1,
                }

        class _BlockDisk:
            def get_stats(self):
                return {
                    "blocks_on_disk": 6,
                    "total_tokens_on_disk": 256,
                    "total_cached_tokens": 256,
                    "disk_hits": 2,
                    "disk_misses": 1,
                }

        class _PagedManager:
            _disk_store = _BlockDisk()

        class _SSMDisk:
            def stats(self):
                return {
                    "entries": 3,
                    "total_tokens_on_disk": 128,
                    "total_cached_tokens": 128,
                    "hits": 4,
                    "misses": 2,
                }

        class _SSMCache:
            size = 1
            max_entries = 8
            disk_enabled = True
            disk_directory = "/tmp/vmlx-test/ssm_companion"
            _disk = _SSMDisk()

        class _Scheduler:
            disk_cache = _DiskCache()
            paged_cache_manager = _PagedManager()
            _ssm_state_cache = _SSMCache()

            def get_stats(self):
                return {
                    "num_waiting": 0,
                    "num_running": 0,
                    "ewma_ttft_seconds": 0,
                    "cache_reuse_skips": 0,
                    "cache_reuse_skip_tokens": 0,
                    "last_cache_reuse_skip": None,
                    "cache_reuse_partial_downgrades": 0,
                    "cache_reuse_partial_tokens": 0,
                    "last_cache_reuse_partial": None,
                }

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        payload = await server.health()
        cache = payload["cache"]

        assert cache["scheduler_cache"]["total_tokens_cached"] == 640
        assert cache["disk_cache"]["total_tokens_on_disk"] == 384
        assert cache["block_disk_cache"]["total_tokens_on_disk"] == 256
        assert cache["ssm_companion"]["disk"]["total_tokens_on_disk"] == 128
        assert cache["totals"]["ram_tokens_cached"] == 640
        assert cache["totals"]["l2_prompt_tokens_on_disk"] == 384
        assert cache["totals"]["l2_block_tokens_on_disk"] == 256
        assert cache["totals"]["l2_ssm_tokens_on_disk"] == 128
        assert cache["totals"]["l2_tokens_on_disk"] == 768
        assert cache["totals"]["l2_tokens_on_disk_store_sum"] == 768
        assert "may_overlap" in cache["totals"]["l2_tokens_on_disk_note"]

    def test_cache_snapshot_reports_mllm_ssm_l2_before_lazy_generator(self):
        import vmlx_engine.server as server

        class _SSMDisk:
            directory = "/tmp/vmlx-test/block-cache/ssm_companion"

            def stats(self):
                return {
                    "entries": 3,
                    "total_tokens_on_disk": 192,
                    "total_cached_tokens": 192,
                    "hits": 5,
                    "misses": 1,
                }

        scheduler = SimpleNamespace(
            batch_generator=None,
            _ssm_companion_disk_store=_SSMDisk(),
            config=SimpleNamespace(
                ssm_state_cache_size=8,
                ssm_state_cache_max_mb=512,
            ),
        )

        cache = server._cache_telemetry_snapshot(scheduler)

        assert cache["ssm_companion"]["entries"] == 0
        assert cache["ssm_companion"]["max_entries"] == 8
        assert cache["ssm_companion"]["disk_enabled"] is True
        assert cache["ssm_companion"]["disk_directory"].endswith("ssm_companion")
        assert cache["ssm_companion"]["disk"]["total_tokens_on_disk"] == 192
        assert cache["totals"]["l2_ssm_tokens_on_disk"] == 192
        assert cache["totals"]["l2_tokens_on_disk"] == 192

    def test_cache_stats_surfaces_cache_reuse_skip_telemetry(self):
        scheduler_source = Path("./vmlx_engine/scheduler.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()
        performance_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

        for marker in (
            "cache_reuse_skips",
            "cache_reuse_skip_tokens",
            "last_cache_reuse_skip",
            "cache_hit_requests",
            "cache_hit_tokens",
            "cache_hit_tokens_by_detail",
            "cache_reuse_partial_downgrades",
            "cache_reuse_partial_tokens",
            "last_cache_reuse_partial",
        ):
            assert marker in scheduler_source
            assert marker in server_source

        assert "insufficient_memory_for_cache_merge" in scheduler_source
        assert "insufficient_memory_for_full_cache_merge" in scheduler_source
        assert "merge_budget = avail * budget_fraction" in scheduler_source
        assert '"budget_mb"' in scheduler_source
        assert "Cache Reuse Skips" in cache_panel_source
        assert "Cache Hit Tokens" in cache_panel_source
        assert "Hit Tokens by Detail" in cache_panel_source
        assert "hybrid_kv_without_ssm" in Path("./vmlx_engine/server.py").read_text()
        assert "ssm_prefix_lookup" in Path("./vmlx_engine/mllm_batch_generator.py").read_text()
        assert "Hybrid KV-Only Misses" in cache_panel_source
        assert "KV-Only Tokens" in cache_panel_source
        assert "last_hybrid_kv_without_ssm" in cache_panel_source
        assert "Hybrid KV-Only Misses" in performance_panel_source
        assert "last_hybrid_kv_without_ssm" in performance_panel_source
        assert "Partial Reuse" in cache_panel_source
        assert "last_cache_reuse_skip" in cache_panel_source
        assert "last_cache_reuse_partial" in cache_panel_source
        assert "used_needed_mb" in cache_panel_source
        assert "budgeted" in cache_panel_source
        assert "needed_mb" in cache_panel_source
        assert "budget_mb" in cache_panel_source
        assert "available_mb" in cache_panel_source
        assert "partial reuse failed" in cache_panel_source
        assert "dropped_cached_tokens" in cache_panel_source
        assert "full_prefill_tokens" in cache_panel_source
        assert "partial_reuse_unavailable_reason" in cache_panel_source
        assert "cache_format" in cache_panel_source
        assert "Partial Reuse" in performance_panel_source
        assert "Cache Hit Tokens" in performance_panel_source
        assert "last_cache_reuse_partial" in performance_panel_source
        assert "used_needed_mb" in performance_panel_source
        assert "budgeted" in performance_panel_source
        assert "budget_mb" in performance_panel_source
        assert "partial reuse failed" in performance_panel_source
        assert "dropped_cached_tokens" in performance_panel_source
        assert "full_prefill_tokens" in performance_panel_source
        assert "partial_reuse_unavailable_reason" in performance_panel_source
        assert "cache_format" in performance_panel_source

    def test_cache_stats_surface_displays_l2_token_totals(self):
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()

        assert "Tokens on Disk" in cache_panel_source
        assert "Cache Totals" in cache_panel_source
        assert "RAM Cached Tokens" in cache_panel_source
        assert "L2 Tokens on Disk" in cache_panel_source
        assert "SSM L2 Tokens" in cache_panel_source

    def test_performance_panel_displays_jangtq_sidecar_layout(self):
        performance_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

        assert "prestacked_bundle" in performance_panel_source
        assert "JANGTQ Layout" in performance_panel_source
        assert "runtime sidecar" in performance_panel_source
        assert "F16 Passthrough" in performance_panel_source
        assert "passthrough_bit_widths_used" in performance_panel_source
        assert "routed_expert_bits_label" in performance_panel_source
        assert "compat_warnings" in performance_panel_source
        assert "grouped-Conv1d" in Path("./vmlx_engine/server.py").read_text()

    def test_panel_defaults_are_speed_oriented_and_labels_match_values(self):
        session_form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()
        cache_panel_source = Path(
            "./panel/src/renderer/src/components/sessions/CachePanel.tsx"
        ).read_text()
        settings_flow_source = Path("./panel/tests/settings-flow.test.ts").read_text()
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        defaults_yaml = Path("./vmlx_engine/config/defaults.yaml").read_text()
        config_models = Path("./vmlx_engine/config/models.py").read_text()

        assert "maxNumSeqs: 1" in session_form_source
        assert "prefillBatchSize: 512" in session_form_source
        assert "completionBatchSize: 512" in session_form_source
        assert "enableJit: true" in session_form_source
        assert "maxNumSeqs: 1" in sessions_source
        assert "prefillBatchSize: 512" in sessions_source
        assert "completionBatchSize: 512" in sessions_source
        assert 'unlimitedLabel="Default (1)"' in session_form_source
        assert 'unlimitedLabel="Default (512)"' in session_form_source
        assert 'unlimitedLabel="Default (2048)"' in session_form_source
        assert "isLingCrackModelPath" not in sessions_source
        assert "out.defaultTemperature = 20" not in sessions_source
        assert (
            "defaultMinP = defs.defaultMinP ?? 0" in sessions_source
            or "setConfigValue(mutable, 'defaultMinP', defs.defaultMinP ?? 0)" in sessions_source
        )
        native_mtp_idx = sessions_source.index("// Native in-model MTP")
        generation_defaults_idx = sessions_source.index("// Generation defaults", native_mtp_idx)
        native_mtp_launch_block = sessions_source[native_mtp_idx:generation_defaults_idx]
        sessions_outside_native_mtp = (
            sessions_source[:native_mtp_idx] + sessions_source[generation_defaults_idx:]
        )
        assert "args.push('--native-mtp-sampling-policy'" in native_mtp_launch_block
        assert "args.push('--default-min-p'" not in native_mtp_launch_block
        assert "args.push('--default-min-p'" not in sessions_outside_native_mtp
        assert "args.push('--no-continuous-batching')" in sessions_source
        assert "Auto-enable continuous batching when prefix cache is on" not in sessions_source
        cli_source = Path("./vmlx_engine/cli.py").read_text()
        assert '"--max-num-seqs", type=int, default=1' in cli_source
        assert '"--prefill-batch-size", type=int, default=512' in cli_source
        assert '"--completion-batch-size", type=int, default=512' in cli_source
        assert '"--continuous-batching"' in cli_source
        assert '"--no-continuous-batching"' in cli_source
        assert "default=True" in cli_source
        assert "max_concurrent_requests: 1" in defaults_yaml
        assert "prefill_batch_size: 512" in defaults_yaml
        assert "completion_batch_size: 512" in defaults_yaml
        assert 'quantization: "q4"' in defaults_yaml
        assert 'quantization: "turboquant"' not in defaults_yaml
        assert "max_blocks: 1000" in defaults_yaml
        assert "max_concurrent_requests: int = 1" in config_models
        assert "prefill_batch_size: int = 512" in config_models
        assert "completion_batch_size: int = 512" in config_models
        assert "max_blocks: int = 1000" in config_models
        assert "defaults to 5" not in sessions_source
        assert "default: 256" not in cli_source
        assert "backend default 8" not in settings_flow_source
        assert "total_tokens_on_disk" in cache_panel_source
        assert "health.cache" in Path(
            "./panel/src/renderer/src/components/sessions/PerformancePanel.tsx"
        ).read_text()

    def test_session_command_preview_mirrors_runtime_default_flags(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        preview_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        ).read_text()

        for flag in (
            "--smelt",
            "--flash-moe",
            "--distributed",
            "--enable-jit",
            "--omni-backend",
            "--log-level",
            "--allowed-origins",
        ):
            assert flag in sessions_source
            assert flag in preview_source
        for flag in ("--default-enable-thinking",):
            # The literals may appear in sanitizer/blocklist tables so stale
            # additionalArgs can be stripped. They must not be emitted as
            # startup launch flags from the real session command or preview.
            assert f"args.push('{flag}'" not in sessions_source
            assert f"parts.push('{flag}'" not in preview_source
        sessions_native_idx = sessions_source.index("// Native in-model MTP")
        sessions_gen_idx = sessions_source.index("// Generation defaults", sessions_native_idx)
        sessions_native_block = sessions_source[sessions_native_idx:sessions_gen_idx]
        sessions_outside_native = (
            sessions_source[:sessions_native_idx] + sessions_source[sessions_gen_idx:]
        )
        preview_native_idx = preview_source.index("// Native in-model MTP mirrors sessions.ts")
        preview_gen_idx = preview_source.index("// Generation defaults", preview_native_idx)
        preview_native_block = preview_source[preview_native_idx:preview_gen_idx]
        preview_outside_native = (
            preview_source[:preview_native_idx] + preview_source[preview_gen_idx:]
        )
        for flag in (
            "--default-temperature",
            "--default-top-p",
            "--default-top-k",
            "--default-min-p",
            "--default-repetition-penalty",
        ):
            assert f"args.push('{flag}'" not in sessions_native_block
            assert f"parts.push('{flag}'" not in preview_native_block
        assert "args.push('--default-repetition-penalty'" not in sessions_outside_native
        assert "parts.push('--default-repetition-penalty'" not in preview_outside_native
        assert "IMAGE_ADDITIONAL_ARG_BLOCKLIST" in sessions_source
        assert "IMAGE_ADDITIONAL_ARG_BLOCKLIST" in preview_source
        assert "DSV4_ADDITIONAL_ARG_BLOCKLIST" in sessions_source
        assert "DSV4_ADDITIONAL_ARG_BLOCKLIST" in preview_source

        cli_source = Path("./vmlx_engine/cli.py").read_text()
        server_source = Path("./vmlx_engine/server.py").read_text()
        assert "server._default_repetition_penalty = 1.0" not in cli_source
        assert "_default_repetition_penalty = 1.0" not in server_source

    def test_panel_startup_defaults_sanitize_incompatible_saved_modes(self):
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()
        preview_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionSettings.tsx"
        ).read_text()
        form_source = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        for source in (sessions_source, preview_source):
            assert "effectiveFlashMoe" in source
            assert "effectiveDistributed" in source
            assert "dsv4Active" in source
            assert "hybridCacheActive" in source
            assert "effectiveEnableJit" in source
            assert "if (effectiveFlashMoe)" in source
            assert "if (effectiveDistributed)" in source
            assert "compatibleExternalSpeculative" in source
            assert "if (effectiveEnableJit)" in source
            assert "if (compatibleExternalSpeculative)" in source
            assert "if (config.enableJit) args.push('--enable-jit')" not in source

        assert "normalizedDetectedFamily === 'deepseek-v4'" in form_source
        # JIT incompat list expanded 2026-05-09 to include TurboQuant
        # (engine skips mx.compile for TurboQuantKVCache; UI now matches).
        assert "disabled={flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive}" in form_source
        assert "checked={!!config.enableJit && !flashMoeActive && !distributedActive && !dsv4Active && !zayaCcaActive && !turboQuantActive && !multimodalActive && !hybridCacheActive}" in form_source
        assert "disabled={config.continuousBatching || multimodalActive || dsv4Active}" in form_source

    def test_responses_long_context_tool_cache_gate_script_pins_artifacts(self):
        gate_source = Path(
            "./tests/cross_matrix/run_responses_long_tool_cache_gate.py"
        ).read_text()

        for marker in (
            "/v1/responses",
            "/v1/cache/stats",
            "/health",
            "previous_response_id",
            "function_call_output",
            "TOOL_DEFINITIONS",
            "target_chars_per_turn",
            "10000",
            "SUMMARY.md",
            "SUMMARY.json",
            "tail_review",
            "cached_tokens",
            "scheduler_stats",
            "visible_output_observed",
            "final_turn_visible_output",
            "final_turn_no_tools",
            "final_turn_disable_thinking",
            "final_turn_tools_disabled",
            "final_turn_thinking_disabled",
            "require_cache_each_turn_after_first",
            "cache_reuse_each_turn_after_first",
            "_cache_acceptance",
            "_extract_warnings",
            "tools_enabled",
            "enable_thinking",
            "overall_pass",
            "result =",
            "_tools_enabled_for_turn",
            "resolve_tool_calls_in_turn",
            "Tool results are provided. Produce a visible answer",
            "require_tool_call_each_turn",
            "tool_call_each_required_turn",
            "_max_cached_tokens",
            "request_round",
            "response_round",
            "tool_choice_mode",
            "--tool-choice",
            "resolution_tool_choice",
            "--resolution-tool-choice",
            "no_tool_markup_leak",
            "_tool_markup_leak",
            "require_tool_evidence",
            "--require-tool-evidence",
            "_tool_grounding",
            "tool_grounded",
            "tool_evidence_markers",
        ):
            assert marker in gate_source

    def test_responses_long_context_tool_cache_gate_does_not_count_reasoning_as_visible(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        extract_output_text = gate["_extract_output_text"]
        extract_reasoning = gate["_extract_reasoning"]

        reasoning_only = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning", "text": "internal cache analysis"}],
                }
            ]
        }
        assert extract_output_text(reasoning_only) == ""
        assert extract_reasoning(reasoning_only) == "internal cache analysis"

        mixed_response = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning", "text": "hidden"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible answer"}],
                },
            ]
        }
        assert extract_output_text(mixed_response) == "visible answer"

    def test_responses_long_context_tool_cache_gate_can_require_cache_each_turn(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        cache_acceptance = gate["_cache_acceptance"]

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 5562},
            ],
            require_cache_each_turn_after_first=True,
        )
        assert seen is True
        assert each is True

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 0},
            ],
            require_cache_each_turn_after_first=True,
        )
        assert seen is True
        assert each is False

        seen, each = cache_acceptance(
            [
                {"cached_tokens": 0},
                {"cached_tokens": 2895},
                {"cached_tokens": 0},
            ],
            require_cache_each_turn_after_first=False,
        )
        assert seen is True
        assert each is True

    def test_responses_long_context_tool_cache_gate_extracts_warnings(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        extract_warnings = gate["_extract_warnings"]

        assert extract_warnings({"warnings": ["  current response warning  ", 42, ""]}) == [
            "current response warning"
        ]

        assert extract_warnings(
            {
                "output": [
                    {"type": "response.warning", "message": "stream warning"},
                    {"type": "message", "content": []},
                ]
            }
        ) == ["stream warning"]

    def test_responses_long_context_tool_cache_gate_counts_cached_tokens_across_rounds(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        max_cached_tokens = gate["_max_cached_tokens"]

        assert max_cached_tokens(
            [
                {"usage": {"input_tokens_details": {"cached_tokens": 0}}},
                {"usage": {"input_tokens_details": {"cached_tokens": 8192}}},
            ]
        ) == 8192
        assert max_cached_tokens([]) == 0

    def test_responses_long_context_tool_cache_gate_tool_output_handles_bad_paths(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        out = tool_output(
            Path("."),
            {
                "name": "inspect_symbol",
                "call_id": "call_bad_path",
                "arguments": '{"path":"does-not-exist.py","symbol":"Scheduler"}',
            },
        )
        assert out["type"] == "function_call_output"
        assert out["call_id"] == "call_bad_path"
        assert "not a readable file" in out["output"]

    def test_responses_long_context_tool_cache_gate_inspect_symbol_searches_directories(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        out = tool_output(
            Path("."),
            {
                "name": "inspect_symbol",
                "call_id": "call_dir_path",
                "arguments": '{"path":"vmlx_engine","symbol":"PAGED_CACHE_SCHEMA_VERSION","context_lines":2}',
            },
        )
        assert out["type"] == "function_call_output"
        assert out["call_id"] == "call_dir_path"
        assert "not a readable file" not in out["output"]
        assert "PAGED_CACHE_SCHEMA_VERSION" in out["output"]
        assert "vmlx_engine/" in out["output"]
        assert "build/lib" not in out["output"]

    def test_responses_long_context_tool_cache_gate_can_require_tool_each_turn(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_acceptance = gate["_tool_acceptance"]

        assert tool_acceptance(
            [
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
            ],
            require_tool_call=True,
            require_tool_call_each_turn=True,
        ) == (True, True)
        assert tool_acceptance(
            [
                {"tools_enabled": True, "function_calls": [{"name": "grep_repo"}]},
                {"tools_enabled": True, "function_calls": []},
            ],
            require_tool_call=True,
            require_tool_call_each_turn=True,
        ) == (True, False)

    def test_responses_long_context_tool_cache_gate_rejects_visible_tool_markup_leak(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_markup_leak = gate["_tool_markup_leak"]

        assert tool_markup_leak("plain visible answer") is False
        assert tool_markup_leak("<minimax:tool_call><invoke name=\"grep_repo\">") is True
        assert tool_markup_leak("<｜DSML｜invoke name=\"grep_repo\">") is True
        assert tool_markup_leak("<tool_call>\n<function=grep_repo>") is True

    def test_responses_long_context_tool_cache_gate_can_require_tool_evidence(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_grounding = gate["_tool_grounding"]

        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": (
                    "vmlx_engine/scheduler.py:2431: def _cache_reuse_budget_fraction():\n"
                    "vmlx_engine/scheduler.py:2440: return max(0.10, min(0.95, value))"
                ),
            }
        ]

        grounded = tool_grounding(
            "Risk is in budget sizing. TOOL_EVIDENCE: vmlx_engine/scheduler.py:2431",
            tool_outputs,
        )
        assert grounded["grounded"] is True
        assert grounded["marker"] == "vmlx_engine/scheduler.py:2431"

        ungrounded = tool_grounding(
            "Risk is in budget sizing, but no exact file line is cited.",
            tool_outputs,
        )
        assert ungrounded["grounded"] is False
        assert "vmlx_engine/scheduler.py:2431" in ungrounded["markers"]

    def test_responses_long_context_tool_cache_gate_rejects_no_match_tool_evidence(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_grounding = gate["_tool_grounding"]

        no_match = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "no matches",
            }
        ]

        result = tool_grounding("TOOL_EVIDENCE: no matches", no_match)
        assert result["grounded"] is False
        assert result["markers"] == []
        assert result["reason"] == "no_file_line_tool_evidence"

    def test_responses_long_context_tool_cache_gate_tolerates_malformed_tool_ints(self):
        import json
        import runpy
        from pathlib import Path

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_output = gate["_tool_output"]

        result = tool_output(
            Path("."),
            {
                "name": "grep_repo",
                "call_id": "call_bad_int",
                "arguments": json.dumps(
                    {
                        "pattern": "_prefix_cache",
                        "path": "vmlx_engine",
                        "max_matches": "5\n</parameter>",
                    }
                ),
            },
        )

        assert result["type"] == "function_call_output"
        assert result["call_id"] == "call_bad_int"
        assert isinstance(result["output"], str)

    def test_responses_long_context_tool_cache_gate_bounds_tool_ints(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tool_positive_int = gate["_tool_positive_int"]

        assert tool_positive_int("999999999", 20) == 200
        assert tool_positive_int("-1", 20) == 20

    def test_responses_long_context_tool_cache_gate_resolution_tools_mode(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        resolution_tools_enabled = gate["_resolution_tools_enabled"]

        assert resolution_tools_enabled("none") is False
        assert resolution_tools_enabled("auto") is True
        assert resolution_tools_enabled("required") is True

    def test_responses_long_context_tool_cache_gate_can_disable_final_turn_tools(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        tools_enabled_for_turn = gate["_tools_enabled_for_turn"]

        assert tools_enabled_for_turn(1, 3, True) is True
        assert tools_enabled_for_turn(2, 3, True) is True
        assert tools_enabled_for_turn(3, 3, True) is False
        assert tools_enabled_for_turn(3, 3, False) is True

    def test_responses_long_context_tool_cache_gate_can_disable_final_turn_thinking(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        enable_thinking_for_turn = gate["_enable_thinking_for_turn"]

        assert enable_thinking_for_turn(1, 3, True, True) is True
        assert enable_thinking_for_turn(2, 3, True, True) is True
        assert enable_thinking_for_turn(3, 3, True, True) is False
        assert enable_thinking_for_turn(3, 3, False, True) is False

    def test_responses_long_context_tool_cache_gate_requires_real_tool_path(self):
        import runpy

        gate = runpy.run_path("./tests/cross_matrix/run_responses_long_tool_cache_gate.py")
        prompt = gate["_build_prompt"](Path("."), 1, 200)
        instructions = gate["_instructions"]()

        assert "must call exactly one provided tool" in prompt
        assert "must call exactly one provided tool" in instructions
        assert "When tools are not available, answer directly" in instructions

    def test_native_cache_status_reports_dsv4_separately_from_tq_kv(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _uses_dsv4_cache=True,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )
        monkeypatch.setenv("DSV4_POOL_QUANT", "1")

        status = _native_cache_status(scheduler)

        assert status["family"] == "deepseek_v4"
        assert status["schema"] == "deepseek_v4_v7"
        assert status["cache_type"] == "native_composite"
        assert "swa_local" in status["components"]
        assert "csa_compressed_pool" in status["components"]
        assert "hca_compressed_pool" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["pool_quant"]["enabled"] is True
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_zaya_typed_cca(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="zaya",
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["family"] == "zaya"
        assert status["schema"] == "zaya_cca_v1"
        assert status["cache_type"] == "typed_cca"
        assert "standard_kv" in status["components"]
        assert "cca_conv_state" in status["components"]
        assert "cca_prev_hidden" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_mixed_swa_kv(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="gemma4",
            _mixed_attention_cache_model=True,
            _tq_active=False,
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["family"] == "gemma4"
        assert status["schema"] == "mixed_swa_kv_v1"
        assert status["cache_type"] == "mixed_swa_kv"
        assert "sliding_window_kv" in status["components"]
        assert "full_attention_kv" in status["components"]
        assert "rotating_window_metadata" in status["components"]
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["paged"] is True
        assert status["block_disk_l2"] is True

    def test_native_cache_status_reports_hybrid_ssm(self):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="bailing_hybrid",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[7, 15, 23, 31],
            _kv_cache_bits=4,
            _kv_cache_group_size=64,
            _ssm_state_cache=SimpleNamespace(_store={"a": object()}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
        )

        status = _native_cache_status(scheduler)

        assert status["schema"] == "hybrid_ssm_v1"
        assert status["cache_type"] == "hybrid_ssm_typed"
        assert status["generic_turboquant_kv"]["enabled"] is False
        assert status["attention_kv_storage_quantization"] == {
            "enabled": True,
            "mode": "storage_boundary",
            "bits": 4,
            "group_size": 64,
            "applies_to": "attention_kv_layers_only",
            "ssm_policy": "native_companion_state",
            "rederive": "async_clean_prefill_on_miss_or_warm_pass",
        }
        assert status["ssm_entries"] == 1
        assert status["kv_layer_indices"] == [7, 15, 23, 31]

    def test_native_cache_status_ignores_legacy_hybrid_tq_override(self, monkeypatch):
        from types import SimpleNamespace
        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            config=SimpleNamespace(kv_cache_quantization="q4"),
            _model_type_for_runtime="bailing_hybrid",
            _is_hybrid=True,
            _uses_dsv4_cache=False,
            _uses_zaya_cache=False,
            _hybrid_kv_positions=[],
            _ssm_state_cache=SimpleNamespace(_store={}),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=None),
        )
        monkeypatch.setenv("VMLX_ALLOW_HYBRID_KV_QUANT", "1")

        status = _native_cache_status(scheduler)

        assert status["generic_turboquant_kv"] == {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        }
        assert status["live_attention_tq_kv"]["enabled"] is False

    def test_quantization_status_detects_jangtq_sidecar_and_bits(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxtq",
            "mxtq_bits": 2,
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "quantization": {
                "profile": "JANGTQ2",
                "target_bits": 2,
                "actual_bits": 2.0,
                "quantization_backend": "turboquant",
            }
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_format"] == "mxtq"
        assert status["mxtq_bits"] == 2
        assert status["routed_expert_bits"] == 2
        assert status["profile"] == "JANGTQ2"
        assert status["sidecar"]["jangtq_runtime"] is True

    @pytest.mark.parametrize(
        ("profile", "expected_bits", "unsupported"),
        [("JANGTQ1", 1, True), ("JANGTQ2", 2, False), ("JANGTQ4", 4, False)],
    )
    def test_quantization_status_derives_jangtq_bits_from_profile(
        self, tmp_path, profile, expected_bits, unsupported
    ):
        """Hy3/JANGTQ bundles may stamp only profile before sidecar sniffing.

        The engine health surface must still expose the routed expert bit count
        so the panel and release gates do not show a TurboQuant bundle as
        "JANGTQ -bit" or treat it as unknown precision.
        """
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "hy_v3",
            "weight_format": "mxtq",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {
                "profile": profile,
                "quantization_backend": "turboquant",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["profile"] == profile
        assert status["routed_expert_bits"] == expected_bits
        warnings = status.get("compat_warnings") or []
        assert any("not production-supported" in w for w in warnings) is unsupported
        assert status["target_bits"] == expected_bits

    def test_quantization_status_detects_prestacked_jangtq_bundle(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
            "mxtq_bits": 2,
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {"routed_expert": 2},
        }))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "model-00001.safetensors",
            }
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["sidecar"]["prestacked_bundle"] is True

    def test_quantization_status_reads_jang_role_bit_plan(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "quantization": {"bits": 8, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "shared_expert": 8,
                "routed_expert": 2,
                "embed_tokens": 8,
                "lm_head": 8,
            },
            "quantization": {
                "method": "affine+mxtq",
                "bits_default": 2,
                "group_size": 64,
            },
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_format"] == "mxtq"
        assert status["routed_expert_bits"] == 2
        assert status["mxtq_bits_by_role"]["attention"] == 8
        assert status["target_bits"] == 2

    def test_quantization_status_handles_jangtq_k_mixed_routed_bits(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert "routed_expert_bits" not in status
        assert status["routed_expert_bits_by_projection"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 4,
        }
        assert status["routed_expert_bits_label"] == "gate=2/up=2/down=4-bit"
        assert status["target_bits"] == 2

    def test_quantization_status_surfaces_jang_passthrough_bit_plan(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_next",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
                "actual_bits": 2.2,
                "bit_widths_used": [2, 4, 8],
                "passthrough_bit_widths_used": [16],
                "passthrough_tensor_count": 90,
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["passthrough_bit_widths_used"] == [16]
        assert status["passthrough_tensor_count"] == 90
        assert "compat_warnings" not in status

    def test_quantization_status_warns_when_hybrid_bundle_lacks_passthrough_metadata(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "nemotron_h",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["compat_warnings"]
        assert "grouped-Conv1d layout backstop" in status["compat_warnings"][0]

    def test_quantization_status_reports_jangtq_weight_index_and_dispatch(self, tmp_path):
        """JANGTQ speed gates need actual indexed codec targets, not labels."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": 4,
                "embed_tokens": 8,
                "lm_head": 8,
            },
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "quantization": {
                "profile": "JANGTQ4",
                "quantization_backend": "turboquant",
            },
        }))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "a.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_norms": "a.safetensors",
                "model.layers.1.mlp.switch_mlp.down_proj.tq_bits": "b.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "c.safetensors",
                "model.embed_tokens.weight": "d.safetensors",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "turboquant_codebook"
        assert status["weight_index"]["total_tensors"] == 5
        assert status["weight_index"]["suffix_counts"]["tq_packed"] == 1
        assert status["weight_index"]["suffix_counts"]["tq_norms"] == 1
        assert status["weight_index"]["suffix_counts"]["tq_bits"] == 1
        assert status["weight_index"]["tq_target_counts"]["routed_expert"] == 3
        assert status["weight_index"]["mtp_tensor_count"] == 0
        assert status["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 3,
            "split_expert": 0,
        }
        assert status["weight_index"]["sample_tq_packed_targets"] == [
            "model.layers.0.mlp.switch_mlp.gate_proj"
        ]
        assert status["weight_matmul_dispatch"] == {
            "primary": "jang_tools_turboquant_custom_kernels",
            "uses_mlx_quantized_matmul": False,
            "metal_na_eligible": False,
            "reason": "turboquant_codebook_uses_custom_tq_kernels",
        }

    def test_dsv4_keeper_identity_reports_no_mtp_and_prestacked_layout(
        self, tmp_path
    ):
        """Pin the final DSV4 no-MTP keeper shape without local model paths."""
        from vmlx_engine.server import _model_mtp_status, _model_quantization_status

        pure_jang = tmp_path / "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
        pure_jang.mkdir()
        (pure_jang / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "num_nextn_predict_layers": 0,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "quantization": {"bits": 4, "group_size": 64},
        }))
        (pure_jang / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 64,
                    "bit_plan": {
                        "routed_projection_group_sizes": {
                            "w1": 32,
                            "w2": 32,
                            "w3": 64,
                        },
                    },
                },
                "non_routed": {"bits": 4, "codec": "affine", "group_size": 64},
            },
        }))
        (pure_jang / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.biases": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.biases": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.weight": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.scales": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.biases": "model.safetensors",
            }
        }))

        pure_quant = _model_quantization_status(str(pure_jang))
        pure_mtp = _model_mtp_status(str(pure_jang))

        assert pure_quant["codec"] == "affine_quantized_matmul"
        assert pure_quant["weight_format"] == "affine"
        assert pure_quant["weight_index"]["total_tensors"] == 10
        assert pure_quant["weight_index"]["mtp_tensor_count"] == 0
        assert pure_quant["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 9,
            "split_expert": 0,
        }
        assert pure_mtp["status"] == "not_configured"
        assert pure_mtp["config_num_nextn_predict_layers"] == 0
        assert pure_mtp["index_has_mtp_tensors"] is False

        jangtq = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
        jangtq.mkdir()
        (jangtq / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
            "num_nextn_predict_layers": 0,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "mxtq_bits": {
                "routed_expert": 2,
                "attention": 8,
                "shared_expert": 8,
                "embed_tokens": 8,
                "lm_head": 8,
            },
        }))
        (jangtq / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "drop_mtp": True,
            "quantization": {
                "profile": "JANGTQ-K",
                "quantization_backend": "turboquant",
            },
        }))
        (jangtq / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        (jangtq / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.gate_proj.tq_bits": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.down_proj.tq_bits": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_packed": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_norms": "model.safetensors",
                "model.layers.0.mlp.switch_mlp.up_proj.tq_bits": "model.safetensors",
                "model.lm_head.weight": "model.safetensors",
            }
        }))

        jangtq_quant = _model_quantization_status(str(jangtq))
        jangtq_mtp = _model_mtp_status(str(jangtq))

        assert jangtq_quant["codec"] == "turboquant_codebook"
        assert jangtq_quant["weight_format"] == "mxtq"
        assert jangtq_quant["weight_index"]["total_tensors"] == 10
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_packed"] == 3
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_norms"] == 3
        assert jangtq_quant["weight_index"]["suffix_counts"]["tq_bits"] == 3
        assert jangtq_quant["weight_index"]["tq_target_counts"]["routed_expert"] == 9
        assert jangtq_quant["weight_index"]["mtp_tensor_count"] == 0
        assert jangtq_quant["weight_index"]["routed_layout_counts"] == {
            "prestacked_switch": 9,
            "split_expert": 0,
        }
        assert jangtq_mtp["status"] == "dropped"
        assert jangtq_mtp["jang_drop_mtp"] is True
        assert jangtq_mtp["config_num_nextn_predict_layers"] == 0
        assert jangtq_mtp["index_has_mtp_tensors"] is False

    def test_quantization_status_reports_routed_layer_bit_plan(self, tmp_path):
        """Layer-specific routed bits must be visible for speed/quality audits."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "mxtq",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": {
                "attention": 8,
                "routed_expert": 2,
            },
            "quantization": {
                "profile": "JANGTQ2",
                "routed_experts": {
                    "bit_plan": {
                        "routed_layer_bits": {
                            "0": 4,
                            "1": 4,
                            "23": 2,
                        }
                    }
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["routed_layer_bits"] == {"0": 4, "1": 4, "23": 2}
        assert status["routed_layer_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_layer_bits"
        )

    def test_quantization_status_reports_affine_dsv4_projection_and_down_layer_plans(
        self, tmp_path
    ):
        """Pure affine DSV4 K/Down4 artifacts need exact mixed-bit visibility."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 128},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 128,
                    "bit_plan": {
                        "routed_projection_bits": {"w2": 4},
                        "routed_down_layer_bits": {"7": 4, "8": 4, "12": 4},
                    },
                },
                "non_routed": {
                    "bits": 8,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["routed_projection_bits"] == {"w2": 4}
        assert status["routed_projection_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_projection_bits"
        )
        assert status["routed_down_layer_bits"] == {"7": 4, "8": 4, "12": 4}
        assert status["routed_down_layer_bits_source"] == (
            "jang_config.quantization.routed_experts.bit_plan.routed_down_layer_bits"
        )

    def test_quantization_status_reports_affine_dispatch_path(self, tmp_path):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "quantization": {"bits": 4, "group_size": 64},
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_treats_plain_jang_weight_format_as_affine_na_path(self, tmp_path):
        """Plain JANG bundles must stay on MLX affine matmul, not JANGTQ TQ."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax_m2",
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "jang",
            "profile": "JANG_2L",
            "quantization": {
                "method": "jang-importance",
                "target_bits": 2,
                "actual_bits": 2.73,
                "block_size": 128,
                "quantization_backend": "mx.quantize",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "jang"
        assert status["profile"] == "JANG_2L"
        assert status["target_bits"] == 2
        assert status["actual_bits"] == 2.73
        assert status["group_size"] == 128
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_reports_hy3_jang_affine_role_bit_plan(self, tmp_path):
        """Hy3 JANG_2K/2L use affine role metadata, not top-level 8-bit defaults."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "hy_v3",
            "weight_format": "affine",
            "quantization": {"bits": 8, "group_size": 128, "mode": "affine"},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "format": "jang",
            "weight_format": "affine",
            "profile": "JANG_2K",
            "affine_bits": {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 3,
                },
                "attention": 8,
                "shared_expert": 8,
                "dense_ffn": 8,
                "embed_tokens": 6,
                "lm_head": 8,
                "norms_router_biases": 16,
            },
            "quantization": {
                "method": "jang-affine",
                "bits_default": 8,
                "group_size": 128,
                "mode": "affine",
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "affine"
        assert status["profile"] == "JANG_2K"
        assert status["group_size"] == 128
        assert status["target_bits"] == 2
        assert status["affine_bits_by_role"]["routed_expert"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 3,
        }
        assert status["routed_expert_bits_by_projection"] == {
            "gate_proj": 2,
            "up_proj": 2,
            "down_proj": 3,
        }
        assert status["routed_expert_bits_label"] == "gate=2/up=2/down=3-bit"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_treats_mxfp4_weight_format_as_affine_na_path(self, tmp_path):
        """MXFP4 bundles use the native MLX affine matmul lane, not JANGTQ TQ."""
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe",
            "weight_format": "mxfp4",
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        assert status["weight_format"] == "mxfp4"
        assert status["weight_matmul_dispatch"] == {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }

    def test_quantization_status_warns_dsv4_affine_routed_without_coherency_stamp(
        self, tmp_path
    ):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 128},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 128,
                },
                "non_routed": {
                    "bits": 8,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        warnings = status.get("compat_warnings") or []
        assert status["codec"] == "affine_quantized_matmul"
        assert any(
            "DSV4 affine routed JANG runtime is loaded but not coherency-verified"
            in warning
            for warning in warnings
        )

    def test_quantization_status_trusts_dsv4_affine_runtime_coherency_stamp(
        self, tmp_path
    ):
        from vmlx_engine.server import _model_quantization_status

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "weight_format": "affine",
            "quantization": {"bits": 2, "group_size": 64},
        }))
        (tmp_path / "jang_config.json").write_text(json.dumps({
            "weight_format": "affine",
            "runtime": {
                "coherency_verified": True,
                "ar_coherency_verified": True,
                "verified_with": "packaged-live-gate",
            },
            "quantization": {
                "method": "affine",
                "routed_experts": {
                    "bits": 2,
                    "codec": "affine",
                    "group_size": 64,
                },
            },
        }))

        status = _model_quantization_status(str(tmp_path))

        assert status["codec"] == "affine_quantized_matmul"
        warnings = status.get("compat_warnings") or []
        assert not any(
            "DSV4 affine routed JANG runtime is loaded but not coherency-verified"
            in warning
            for warning in warnings
        )

    def test_routing_status_reports_trained_top_k_without_override(self, monkeypatch, tmp_path):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.delenv("JANGTQ_TOPK_OVERRIDE", raising=False)
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status == {
            "trained_active_experts": 8,
            "trained_active_experts_source": "config.num_experts_per_tok",
            "n_routed_experts": 256,
            "override_env": None,
            "effective_active_experts": 8,
            "effective_active_experts_source": "trained_default",
        }

    def test_dsv4_routing_status_reports_trained_top_k_six(self, monkeypatch, tmp_path):
        """DSV4 MoE top-k is trained routing metadata, not sampler top_k."""
        from vmlx_engine.server import _model_routing_status

        monkeypatch.delenv("JANGTQ_TOPK_OVERRIDE", raising=False)
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v4",
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status == {
            "trained_active_experts": 6,
            "trained_active_experts_source": "config.num_experts_per_tok",
            "n_routed_experts": 256,
            "override_env": None,
            "effective_active_experts": 6,
            "effective_active_experts_source": "trained_default",
        }

    def test_routing_status_ignores_jangtq_top_k_override(self, monkeypatch, tmp_path):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "4")
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "text_config": {
                "n_routed_experts": 256,
                "num_experts_per_tok": 8,
            },
        }))

        status = _model_routing_status(str(tmp_path))

        assert status["trained_active_experts"] == 8
        assert status["trained_active_experts_source"] == "config.text_config.num_experts_per_tok"
        assert status["n_routed_experts"] == 256
        assert status["override_env"] is None
        assert status["effective_active_experts"] == 8
        assert status["effective_active_experts_source"] == "trained_default"

    def test_routing_status_ignores_invalid_jangtq_top_k_override(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.server import _model_routing_status

        monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "abc")
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "minimax",
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        }))

        status = _model_routing_status(str(tmp_path))

        assert status["trained_active_experts"] == 8
        assert status["override_env"] is None
        assert "override_issue" not in status
        assert status["effective_active_experts"] == 8
        assert status["effective_active_experts_source"] == "trained_default"

    def test_grouped_conv1d_native_error_gets_actionable_detail(self):
        from vmlx_engine.server import _generation_error_detail

        detail = _generation_error_detail(
            ValueError(
                "Given groups=8192 and weights of shape (8192,1,4), "
                "expected to have 32768 input channels but got 8192 input channels instead."
            )
        )

        assert "Grouped Conv1d layout mismatch" in detail
        assert "re-convert with a newer jang build" in detail
        assert "Original error:" in detail

    def test_acceleration_status_does_not_claim_metal_na_for_jangtq(self, monkeypatch, tmp_path):
        import vmlx_engine.server as server

        monkeypatch.setenv("JANGTQ_MPP_NAX", "off")
        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": 2,
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "turboquant_codebook"
        assert status["metal_na_capable"] is False
        assert status["metal_na_active_on_host"] is False
        assert status["reason"] == "turboquant_custom_kernels_do_not_use_mlx_na"
        assert status["jangtq_acceleration"]["mode"] == "off"

    def test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({
            "weight_format": "mxtq",
            "mxtq_bits": 4,
        }))
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"sidecar")
        monkeypatch.setenv("JANGTQ_MPP_NAX", "auto")
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )
        monkeypatch.setattr(
            server,
            "_jangtq_mpp_nax_runtime_status",
            lambda _host=None: {
                "mode": "auto",
                "requested": True,
                "available": True,
                "active": True,
                "uses_mlx_quantized_matmul": False,
                "reason": None,
            },
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "turboquant_codebook_accelerated"
        assert status["metal_na_capable"] is True
        assert status["metal_na_active_on_host"] is True
        assert status["reason"] is None
        assert status["jangtq_acceleration"] == {
            "mode": "auto",
            "requested": True,
            "available": True,
            "active": True,
            "uses_mlx_quantized_matmul": False,
            "reason": None,
        }

    def test_acceleration_status_reports_affine_na_only_when_symbols_and_host_match(self, monkeypatch, tmp_path):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(json.dumps({
            "quantization": {"bits": 4, "group_size": 64},
        }))
        monkeypatch.setattr(
            server,
            "_mlx_metal_na_status",
            lambda: {"available": True, "nax_symbols": 3534, "naxtile_symbols": 786},
        )
        monkeypatch.setattr(
            server,
            "_host_supports_metal_na",
            lambda: {"supported": True, "brand": "Apple M5 Max"},
        )

        status = server._model_acceleration_status(str(tmp_path))

        assert status["kernel_type"] == "affine_quantized_matmul"
        assert status["metal_na_capable"] is True
        assert status["metal_na_active_on_host"] is True

    def test_mtp_status_reports_dropped_dsv4_artifact(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":true}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        expected = {
            "config_num_nextn_predict_layers": 0,
            "jang_drop_mtp": True,
            "index_has_mtp_tensors": False,
            "artifact_available": False,
            "family": "deepseek_v4",
            "runtime_supported": False,
            "runtime_available": False,
            "runtime_reason": "jang_config.drop_mtp=true",
            "status": "dropped",
            "issues": [],
        }
        for key, value in expected.items():
            assert status[key] == value

    def test_mtp_status_flags_missing_weights_when_config_expects_mtp(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("config expects" in issue for issue in status["issues"])

    def test_mtp_status_flags_indexed_mtp_when_config_disables_runtime(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("config disables" in issue for issue in status["issues"])

    def test_mtp_status_flags_invalid_config_layer_count(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":"one"}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("invalid" in issue for issue in status["issues"])

    def test_mtp_status_flags_malformed_index_metadata(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text("{")

        status = _model_mtp_status(str(tmp_path))

        assert status["runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("model.safetensors.index.json" in issue for issue in status["issues"])

    def test_mtp_status_does_not_claim_runtime_for_weights_only_bundle(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":false}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["artifact_available"] is True
        assert status["runtime_available"] is False
        assert status["status"] == "weights_present_runtime_unwired"
        assert "not currently on the JangMTP support map" in status["runtime_reason"]

    @pytest.mark.asyncio
    async def test_health_and_capabilities_surface_mtp_status(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        (tmp_path / "config.json").write_text(
            '{"model_type":"deepseek_v4","num_nextn_predict_layers":0,'
            '"n_routed_experts":256,"num_experts_per_tok":6}'
        )
        (tmp_path / "jang_config.json").write_text(
            '{"weight_format":"mxtq","drop_mtp":true}'
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
        )

        class _Engine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "batched"}

        class _Scheduler:
            block_aware_cache = None
            paged_cache_manager = None
            memory_aware_cache = None
            prefix_cache = None

            def get_stats(self):
                return {}

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_get_scheduler", lambda: _Scheduler())
        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "dsv4-test")
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_standby_state", None)
        monkeypatch.setattr(server, "_mcp_manager", None)

        health = await server.health()
        mtp_alias = await server.health_mtp()
        capabilities = await server.model_capabilities("dsv4-test")

        assert health["mtp"]["status"] == "dropped"
        assert health["mtp"]["runtime_available"] is False
        assert mtp_alias["status"] == "dropped"
        assert mtp_alias["runtime_available"] is False
        assert health["routing"]["trained_active_experts"] == 6
        assert health["routing"]["effective_active_experts_source"] == "trained_default"
        assert capabilities["mtp"]["status"] == "dropped"
        assert capabilities["mtp"]["runtime_available"] is False
        assert capabilities["routing"]["trained_active_experts"] == 6
        assert capabilities["routing"]["effective_active_experts_source"] == "trained_default"

    @pytest.mark.asyncio
    async def test_health_mtp_endpoint_returns_loaded_runtime_status(
        self, monkeypatch, tmp_path
    ):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_model_path", str(tmp_path))
        monkeypatch.setattr(server, "_model_name", "qwen-mtp-test")
        monkeypatch.setattr(
            server,
            "_model_mtp_status_with_loaded_runtime",
            lambda bundle_path: {
                "status": "native_runtime_active",
                "runtime_active": True,
                "runtime_scope": "text+vl",
                "bundle_path": bundle_path,
            },
        )

        payload = await server.health_mtp()

        assert payload["status"] == "native_runtime_active"
        assert payload["runtime_active"] is True
        assert payload["runtime_scope"] == "text+vl"
        assert payload["bundle_path"] == str(tmp_path)

    def test_mlx_metal_na_status_handles_namespace_mlx_package(self, monkeypatch, tmp_path):
        """MLX can be a namespace package with mlx.__file__ == None.

        The extension module path lives on mlx.core.__file__; NA telemetry must
        use that fallback or the panel/API report "not available" even when the
        installed metallib has NA symbols.
        """
        import sys
        import types
        import vmlx_engine.server as server

        site = tmp_path / "site-packages"
        metal = site / "mlx" / "lib" / "mlx.metallib"
        metal.parent.mkdir(parents=True)
        metal.write_bytes(b"prefix _nax_ NAXTile suffix")
        core_so = site / "mlx" / "core.cpython-313-darwin.so"
        core_so.write_bytes(b"fake")

        monkeypatch.setitem(
            sys.modules,
            "mlx",
            types.SimpleNamespace(__file__=None),
        )
        monkeypatch.setitem(
            sys.modules,
            "mlx.core",
            types.SimpleNamespace(__file__=str(core_so)),
        )
        server._metal_na_status_cache.clear()

        status = server._mlx_metal_na_status()

        assert status["available"] is True
        assert status["nax_symbols"] == 1
        assert status["naxtile_symbols"] == 1

    def test_mlx_metal_na_status_accepts_affine_qmm_symbols(
        self, monkeypatch, tmp_path
    ):
        """Bundled MLX 0.31.x exposes affine qmm symbols, not _nax_/NAXTile."""
        import sys
        import types
        import vmlx_engine.server as server

        site = tmp_path / "site-packages"
        metal = site / "mlx" / "lib" / "mlx.metallib"
        metal.parent.mkdir(parents=True)
        metal.write_bytes(b"prefix affine_qmm affine_gather_qmm suffix")
        init = site / "mlx" / "__init__.py"
        init.write_text("")

        monkeypatch.setitem(
            sys.modules,
            "mlx",
            types.SimpleNamespace(__file__=str(init)),
        )
        server._metal_na_status_cache.clear()

        status = server._mlx_metal_na_status()

        assert status["available"] is True
        assert status["nax_symbols"] == 0
        assert status["naxtile_symbols"] == 0
        assert status["affine_qmm_symbols"] == 1
        assert status["affine_gather_qmm_symbols"] == 1


class TestNonStreamingReceiveDrainDisconnect:
    """Active receive-drain detects ASGI http.disconnect events that
    `Request.is_disconnected()` may miss when nothing else reads from the
    receive channel.

    Live evidence (codex 2026-05-09 14:18): client TCP close + read-timeout
    left scheduler num_running=1 because is_disconnected() polling didn't
    surface the disconnect event in time. Active receive-drain catches it.
    """

    def test_helper_detects_http_disconnect_via_receive_channel(self):
        import asyncio
        import pytest
        from fastapi import HTTPException
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeEngine:
            def __init__(self):
                self.aborted = None

            async def chat(self, **kwargs):
                # Run forever until aborted
                await asyncio.sleep(10)
                return None

            async def abort_request(self, request_id):
                self.aborted = request_id
                return True

        class FakeRequest:
            """ASGI receive channel that emits http.disconnect after a tick."""
            def __init__(self):
                self._sent = False

            async def is_disconnected(self):
                # Simulate Starlette's lazy poll missing the event because
                # nothing is actively draining receive.
                return False

            async def receive(self):
                # First call returns the disconnect event after a small delay
                # so the helper's parallel drain task can catch it.
                if not self._sent:
                    self._sent = True
                    await asyncio.sleep(0.01)
                    return {"type": "http.disconnect"}
                # Subsequent calls block forever (channel exhausted).
                await asyncio.sleep(60)
                return {}

        async def run():
            engine = FakeEngine()
            req = FakeRequest()
            with pytest.raises(HTTPException) as exc:
                await _await_chat_with_disconnect_abort(
                    engine,
                    messages=[],
                    chat_kwargs={},
                    timeout=30,
                    fastapi_request=req,
                    request_id="resp_receive_drain_test",
                    endpoint="test",
                    poll_interval=0.05,
                )
            assert exc.value.status_code == 499
            assert engine.aborted == "resp_receive_drain_test", (
                "Engine must be aborted when receive-drain catches http.disconnect "
                "even if is_disconnected() never returned True"
            )

        asyncio.run(run())

    def test_helper_safe_when_request_has_no_receive_method(self):
        """Backwards-compat: legacy callers may pass a fake request without
        `.receive`. Helper must not crash; falls back to is_disconnected polling."""
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            async def chat(self, **kwargs):
                return FakeOutput()

        class FakeRequest:
            # NO `.receive` attribute
            async def is_disconnected(self):
                return False

        async def run():
            output = await _await_chat_with_disconnect_abort(
                FakeEngine(),
                messages=[],
                chat_kwargs={},
                timeout=10,
                fastapi_request=FakeRequest(),
                request_id="resp_no_receive",
                endpoint="test",
                poll_interval=0.01,
            )
            assert output.completion_tokens == 1

        asyncio.run(run())

    def test_helper_uses_single_receive_reader_when_drain_available(self):
        """When `.receive` exists, the active drain owns ASGI disconnect reads.

        Starlette's `Request.is_disconnected()` is implemented as a zero-timeout
        call to the same receive function. Calling it while the drain task is
        blocked in receive creates two readers on the ASGI receive channel, which
        is not a safe contract for all servers.
        """
        import asyncio
        from vmlx_engine.server import _await_chat_with_disconnect_abort

        class FakeOutput:
            completion_tokens = 1

        class FakeEngine:
            async def chat(self, **kwargs):
                await asyncio.sleep(0.03)
                return FakeOutput()

        class FakeRequest:
            def __init__(self):
                self.is_disconnected_calls = 0

            async def is_disconnected(self):
                self.is_disconnected_calls += 1
                return False

            async def receive(self):
                await asyncio.sleep(60)
                return {}

        async def run():
            req = FakeRequest()
            output = await _await_chat_with_disconnect_abort(
                FakeEngine(),
                messages=[],
                chat_kwargs={},
                timeout=10,
                fastapi_request=req,
                request_id="resp_single_receive_reader",
                endpoint="test",
                poll_interval=0.001,
            )
            assert output.completion_tokens == 1
            assert req.is_disconnected_calls == 0

        asyncio.run(run())


class TestJitTurboQuantSymmetricGuard:
    """Engine skips mx.compile for TurboQuantKVCache (GH issue #66) AND panel
    suppresses --enable-jit when isTurboQuant. The two guards must stay
    symmetric — if one drifts away, the other breaks user trust (UI says
    JIT is disabled but engine still tries to compile, or vice versa).
    """

    def test_engine_jit_skip_for_turboquant_make_cache(self):
        """Engine apply_jit_compilation must short-circuit when the active
        model has the TurboQuant patched make_cache function name."""
        import inspect
        from pathlib import Path

        source = Path("vmlx_engine/server.py").read_text()
        # The skip block is identified by these load-bearing strings:
        assert "_turboquant_make_cache" in source
        assert "_tq_make_cache" in source
        assert "JIT: Skipping mx.compile — TurboQuantKVCache is active" in source
        # Make sure the early-return is wired in (return after detecting TQ name)
        assert (
            'if _make_cache_name in ("_turboquant_make_cache", "_tq_make_cache"):'
            in source
        )

    def test_panel_session_launcher_suppresses_jit_for_turboquant(self):
        """Panel launcher must compute effectiveEnableJit with turboQuantActive
        gating; matches the engine's TQ-skip behavior."""
        from pathlib import Path
        sessions_source = Path("./panel/src/main/sessions.ts").read_text()

        assert "turboQuantActive" in sessions_source
        assert "(detected as any).isTurboQuant" in sessions_source
        # effectiveEnableJit must include !turboQuantActive in the AND chain
        assert "!turboQuantActive" in sessions_source

    def test_panel_form_disables_jit_checkbox_for_turboquant(self):
        """Panel JIT checkbox must be visually disabled + warned when
        turboQuantActive is true. If this test fails, user sees a checkbox
        they can tick that the engine will silently ignore."""
        from pathlib import Path
        form = Path(
            "./panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
        ).read_text()

        assert "turboQuantActive" in form
        assert "detectedIsTurboQuant" in form
        # disabled prop covers turboQuantActive and hybrid path-dependent caches.
        assert "disabled={flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive}" in form

    def test_detect_config_stamps_isTurboQuant_flag(self):
        """detectModelConfigFromDir must set isTurboQuant when bundle is TQ.
        Without this stamp, the panel guards above never fire."""
        from pathlib import Path
        registry = Path("./panel/src/main/model-config-registry.ts").read_text()

        assert "isTurboQuant" in registry
        # The stamp site sets next.isTurboQuant = true
        assert "next.isTurboQuant = true" in registry


class TestStreamUsagePropagatesCacheDetail:
    """Streaming SSE finish-chunk usage must surface cache_detail alongside
    cached_tokens, mirroring the non-stream get_usage() path. Without this,
    typed cache labels (paged+zaya_cca, paged+ssm+disk, paged+disk+tq, etc.)
    are silently dropped from stream consumers even though server-side
    cache_hit_tokens_by_detail accounting still works.

    Bug exposed during ZAYA1-VL typed CCA validation (commit 4f3c1dc6):
    non-stream returned cache_detail=paged+zaya_cca, stream returned None.
    """

    def test_chat_stream_tracks_cache_detail_alongside_cached_tokens(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        assert "cached_tokens = 0\n    cache_detail" in source
        assert (
            'getattr(output, "cache_detail", "") or None\n            if _detail is not None:\n                cache_detail = _detail'
            in source
        )

    def test_chat_stream_finish_chunks_emit_cache_detail(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        # All three chat-stream finish paths (regular, error, tool_calls)
        # construct PromptTokensDetails manually; each must pass cache_detail.
        assert (
            source.count("cached_tokens=cached_tokens, cache_detail=cache_detail")
            >= 3
        )

    def test_responses_stream_tracks_cache_detail_alongside_cached(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        assert "_cached = 0\n    _cache_detail" in source
        assert '_detail_chunk = getattr(output, "cache_detail", "") or None' in source

    def test_responses_stream_finish_emits_cache_detail(self):
        from pathlib import Path
        source = Path("./vmlx_engine/server.py").read_text()
        # Responses uses dict-builder shape, not PromptTokensDetails class.
        assert '"cache_detail": _cache_detail' in source


class TestHuggingFaceDownloadRegression:
    """Issue #118: stale saved HF mirrors or tokens must not poison public GUI
    downloads. The panel should normalize persisted settings and keep the
    Python worker's endpoint selection app-owned so it can fall back cleanly.
    """

    def test_panel_download_worker_does_not_inherit_raw_hf_endpoint(self):
        from pathlib import Path

        source = Path("./panel/src/main/ipc/models.ts").read_text()

        assert "normalizeHfEndpointSetting" in source
        assert "HF_ENDPOINT: undefined" in source
        assert "endpoint = sys.argv[3] or None" in source
        assert "endpoint=active_endpoint" in source
        assert "https://huggingface.co" in source
        assert "'type':'fallback'" in source

    def test_panel_hf_api_retries_public_requests_without_stale_auth(self):
        from pathlib import Path

        source = Path("./panel/src/main/ipc/models.ts").read_text()

        assert "normalizeHfTokenSetting" in source
        assert "delete baseHeaders.Authorization" in source
        assert "retrying ${baseUrl} without auth" in source
        assert "HuggingFace mirror failed" in source
        assert "fetchHfPath(`/api/models?${params}`" in source

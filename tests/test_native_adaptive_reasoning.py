"""Native adaptive intent must survive aliases, defaults and family policy."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest

MESSAGES = [{"role": "user", "content": "Solve it."}]


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("controls", [
    {"thinking_mode": "adaptive"},
    {"chat_template_kwargs": {"thinking_mode": "adaptive"}},
    {"thinking_mode": "adaptive", "reasoning": {"budget_tokens": 1024}},
])
def test_adaptive_alias_keeps_native_mode_without_forcing_on(request_type, controls):
    inputs = {"messages": MESSAGES} if request_type is ChatCompletionRequest else {"input": "Solve it."}
    req = request_type(model="native-model", **inputs, **controls)
    assert req.thinking_mode == "adaptive"
    assert req.enable_thinking is None
    assert req.reasoning_effort is None
    assert req.chat_template_kwargs["thinking_mode"] == "adaptive"


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("controls", [
    {"enable_thinking": True}, {"enable_thinking": False},
    {"chat_template_kwargs": {"enable_thinking": True}},
    {"reasoning_effort": "none"},
    {"chat_template_kwargs": {"thinking_mode": "enabled"}},
])
def test_adaptive_conflicts_fail_request_validation(request_type, controls):
    inputs = {"messages": MESSAGES} if request_type is ChatCompletionRequest else {"input": "Solve it."}
    with pytest.raises(ValueError, match="conflict"):
        request_type(model="native-model", **inputs, thinking_mode="adaptive", **controls)


def test_anthropic_preserves_native_adaptive_mode():
    req = to_chat_completion(AnthropicRequest(model="native-model", messages=MESSAGES, thinking={"type": "adaptive"}))
    assert req.enable_thinking is None
    assert req.thinking_mode == "adaptive"
    assert req.chat_template_kwargs["thinking_mode"] == "adaptive"


@pytest.mark.parametrize("thinking,extra", [
    ({"type": "not-a-mode"}, {}),
    ({"type": "adaptive", "budget_tokens": 1024}, {}),
    ({"type": "adaptive"}, {"enable_thinking": False}),
    ({"type": "adaptive"}, {"enable_thinking": True}),
    ({"type": "adaptive"}, {"chat_template_kwargs": {"thinking_mode": "disabled"}}),
])
def test_invalid_anthropic_adaptive_controls_rejected(thinking, extra):
    with pytest.raises(ValueError):
        AnthropicRequest(model="native-model", messages=MESSAGES, thinking=thinking, **extra)


def setup_registry(monkeypatch, supported=True):
    from vmlx_engine import server
    cfg = SimpleNamespace(family_name="minimax_m3" if supported else "qwen3_5", model_type="minimax_m3_vl" if supported else "qwen3_5", supports_thinking=True, supports_instruct_mode=True, architecture_hints={"native_thinking_modes": ["enabled", "disabled", "adaptive"]} if supported else {})
    monkeypatch.setattr("vmlx_engine.model_config_registry.get_model_config_registry", lambda: SimpleNamespace(lookup=lambda key: cfg))
    return server


@pytest.mark.parametrize("default", [None, True, False])
@pytest.mark.parametrize("mode,expected", [("adaptive", None), ("enabled", True), ("disabled", False)])
def test_native_mode_precedes_server_boolean_default(monkeypatch, default, mode, expected):
    server = setup_registry(monkeypatch)
    monkeypatch.setattr(server, "_default_enable_thinking", default)
    ct = {"thinking_mode": mode}
    value = server._resolve_enable_thinking(None, ct, False, "native-model", reasoning_effort=None)
    assert value is expected
    server._normalize_minimax_m3_thinking_mode(ct, SimpleNamespace(enable_thinking=value), "native-model")
    assert ct["thinking_mode"] == mode


@pytest.mark.parametrize("mode", ["adaptive", "enabled", "disabled"])
def test_unsupported_native_mode_fails_before_generation(monkeypatch, mode):
    server = setup_registry(monkeypatch, supported=False)
    with pytest.raises(HTTPException) as exc:
        server._resolve_enable_thinking(None, {"thinking_mode": mode}, False, "native-model")
    assert exc.value.status_code == 400
    assert "native" in exc.value.detail


def test_registered_minimax_declares_its_native_template_modes(tmp_path):
    import json
    from vmlx_engine.model_config_registry import get_model_config_registry
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl"}))
    cfg = get_model_config_registry().lookup(str(tmp_path))
    assert cfg.architecture_hints["native_thinking_modes"] == ["enabled", "disabled", "adaptive"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["enabled", "disabled", "adaptive"])
async def test_omni_dispatch_rejects_unsupported_native_mode_before_load(monkeypatch, mode):
    from vmlx_engine.omni_multimodal import dispatch_omni_chat_completion, OmniMultimodalDispatcher
    monkeypatch.setattr(OmniMultimodalDispatcher, "get", lambda *a, **kw: pytest.fail("unsupported mode loaded Omni"))
    req = SimpleNamespace(messages=MESSAGES, chat_template_kwargs={"thinking_mode": mode})
    with pytest.raises(HTTPException) as exc:
        await dispatch_omni_chat_completion(req, "/unused/no-model-loaded")
    assert exc.value.status_code == 400
    assert "native thinking_mode" in exc.value.detail


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/api/chat"])
@pytest.mark.parametrize("default", [False, True])
def test_adaptive_http_handoff_overrides_boolean_defaults(monkeypatch, path, default):
    """Real HTTP validation/adapters with synthetic generation handoff."""
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    _run_streaming_ollama_chat(monkeypatch, family_name="minimax_m3", model_type="minimax_m3_vl",
                              body={"model": "native-model", "messages": MESSAGES, "stream": True})
    server = setup_registry(monkeypatch)
    monkeypatch.setattr(server, "_default_enable_thinking", default)
    captured = {}
    async def capture(engine, messages, request, fastapi_request=None, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        yield 'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        yield 'data: [DONE]\n\n'
    monkeypatch.setattr(server, "stream_chat_completion", capture)
    monkeypatch.setattr(server, "stream_responses_api", capture)
    body = {"model": "native-model", "stream": True}
    if path == "/v1/responses":
        body.update(input="Solve it.", thinking_mode="adaptive")
    else:
        body["messages"] = MESSAGES
        if path == "/v1/messages":
            body.update(thinking={"type": "adaptive"}, max_tokens=128)
        else:
            body["chat_template_kwargs"] = {"thinking_mode": "adaptive"}
    with TestClient(server.app) as client:
        response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert captured["request"].enable_thinking is None
    assert captured["kwargs"].get("enable_thinking") is None
    assert captured["kwargs"]["chat_template_kwargs"]["thinking_mode"] == "adaptive"


@pytest.mark.parametrize("mode", ["enabled", "disabled", "adaptive"])
@pytest.mark.parametrize("default", [True, False])
def test_native_request_mode_overrides_inherited_template_boolean(monkeypatch, mode, default):
    server = setup_registry(monkeypatch)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"enable_thinking": default, "other_kwarg": 7})
    result = server._merge_ct_kwargs({"thinking_mode": mode})
    assert result == {"thinking_mode": mode, "other_kwarg": 7}
    assert server._default_chat_template_kwargs["enable_thinking"] is default


@pytest.mark.parametrize("value", [True, False])
def test_request_boolean_overrides_inherited_native_mode(monkeypatch, value):
    server = setup_registry(monkeypatch)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"thinking_mode": "adaptive"})
    assert server._merge_ct_kwargs(None, enable_thinking=value) == {}
    # Two explicit per-request controls still conflict instead of overriding.
    assert server._merge_ct_kwargs({"thinking_mode": "adaptive"}, enable_thinking=value)["thinking_mode"] == "adaptive"


@pytest.mark.parametrize("thinking", [{}, {"budget_tokens": 1024}])
def test_incomplete_anthropic_thinking_does_not_implicitly_force_on(thinking):
    with pytest.raises(ValueError, match="type"):
        AnthropicRequest(model="native-model", messages=MESSAGES, thinking=thinking)

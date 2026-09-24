"""Native Anthropic effort is distinct from token budgets and extensions."""

import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion

MESSAGES = [{"role": "user", "content": "Solve the problem."}]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}, {"type": "enabled", "budget_tokens": 40000}])
def test_native_effort_survives_conversion(effort, thinking):
    request = to_chat_completion(AnthropicRequest(
        model="native-family", messages=MESSAGES,
        thinking=thinking, output_config={"effort": effort},
    ))
    assert request.reasoning_effort == effort
    if thinking is not None:
        assert request.enable_thinking is (thinking["type"] == "enabled")
    assert request.max_thinking_tokens == (40000 if thinking and thinking["type"] == "enabled" else None)


@pytest.mark.parametrize("effort", ["", "ultra", "none", True, 2, ["low"]])
def test_invalid_native_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="effort"):
        AnthropicRequest(model="native-family", messages=MESSAGES, output_config={"effort": effort})


@pytest.mark.parametrize("extension", [
    {"reasoning_effort": "high"},
    {"chat_template_kwargs": {"reasoning_effort": "high"}},
])
def test_conflicting_native_effort_is_rejected(extension):
    with pytest.raises(ValueError, match="conflict"):
        AnthropicRequest(model="native-family", messages=MESSAGES,
                         output_config={"effort": "low"}, **extension)


def test_matching_effort_aliases_and_omitted_native_effort():
    for output_config in (None, {}, {"effort": None}, {"effort": "low"}):
        converted = to_chat_completion(AnthropicRequest(
            model="native-family", messages=MESSAGES, output_config=output_config,
            reasoning_effort="low", chat_template_kwargs={"reasoning_effort": "low"},
        ))
        assert converted.reasoning_effort == "low"


def test_unsupported_output_format_is_not_silently_ignored():
    with pytest.raises(ValueError, match="format"):
        AnthropicRequest(model="native-family", messages=MESSAGES,
                         output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_native_effort_reaches_http_generation(monkeypatch, stream, effort):
    """Real HTTP request handling; synthetic generation is not model proof."""
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server

    _run_streaming_ollama_chat(
        monkeypatch, family_name="qwen4_exp", model_type="qwen4_exp",
        body={"model": "native-family", "messages": MESSAGES, "stream": True},
    )
    captured = {}
    async def capture(engine, messages, request, fastapi_request=None, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        yield 'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(server, "stream_chat_completion", capture)
    with TestClient(server.app) as client:
        response = client.post("/v1/messages", json={
            "model": "native-family", "messages": MESSAGES, "max_tokens": 50000,
            "stream": stream, "output_config": {"effort": effort},
            "thinking": {"type": "enabled", "budget_tokens": 40000},
        })
    assert response.status_code == 200, response.text
    assert captured["request"].reasoning_effort == effort
    assert captured["request"].max_thinking_tokens == 40000
    assert captured["kwargs"]["chat_template_kwargs"]["reasoning_effort"] == effort
    assert captured["kwargs"]["chat_template_kwargs"]["thinking_budget"] == 40000


@pytest.mark.parametrize("body", [
    {"output_config": {"effort": "ultra"}},
    {"output_config": {"effort": True}},
    {"output_config": {"effort": "low"}, "reasoning_effort": "high"},
    {"output_config": {"effort": "low"}, "chat_template_kwargs": {"reasoning_effort": "high"}},
    {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}},
])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_output_config_returns_anthropic_error_before_generation(monkeypatch, body, stream):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server

    _run_streaming_ollama_chat(
        monkeypatch, family_name="qwen4_exp", model_type="qwen4_exp",
        body={"model": "native-family", "messages": MESSAGES, "stream": True},
    )
    async def unexpected_generation(*args, **kwargs):
        pytest.fail("invalid controls reached generation")
        yield
    monkeypatch.setattr(server, "stream_chat_completion", unexpected_generation)
    with TestClient(server.app) as client:
        response = client.post("/v1/messages", json={
            "model": "native-family", "messages": MESSAGES, "max_tokens": 128,
            "stream": stream, **body,
        })
    assert response.status_code == 400, response.text
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"

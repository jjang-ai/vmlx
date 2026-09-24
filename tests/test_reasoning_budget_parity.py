"""Token budgets and native effort tiers are independent request controls."""

import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest


MESSAGES = [{"role": "user", "content": "Solve the problem."}]


@pytest.mark.parametrize("budget", [32767, 32768, 40000])
@pytest.mark.parametrize("effort", [None, "low", "xhigh", "max"])
def test_anthropic_budget_does_not_invent_effort(budget, effort):
    request = to_chat_completion(AnthropicRequest(
        model="native-family", messages=MESSAGES, max_tokens=50000,
        thinking={"type": "enabled", "budget_tokens": budget},
        reasoning_effort=effort,
    ))
    assert request.enable_thinking is True
    assert request.max_thinking_tokens == budget
    assert request.reasoning_effort == effort
    assert request.chat_template_kwargs["thinking_budget"] == budget
    assert "reasoning_effort" not in request.chat_template_kwargs


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("effort", [None, "low", "xhigh", "max"])
def test_nested_budget_survives_top_level_effort(request_type, effort):
    inputs = {"messages": MESSAGES} if request_type is ChatCompletionRequest else {"input": "Solve it."}
    request = request_type(model="native-family", **inputs,
                           reasoning_effort=effort, reasoning={"budget_tokens": 4096})
    assert request.max_thinking_tokens == 4096
    assert request.reasoning_effort == effort


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, "4096"])
def test_nested_budget_cannot_bypass_validation(request_type, budget):
    inputs = {"messages": MESSAGES} if request_type is ChatCompletionRequest else {"input": "Solve it."}
    with pytest.raises(ValueError, match="budget_tokens"):
        request_type(model="native-family", **inputs, reasoning={"budget_tokens": budget})


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, ResponsesRequest])
def test_explicit_thinking_cap_precedes_nested_alias(request_type):
    inputs = {"messages": MESSAGES} if request_type is ChatCompletionRequest else {"input": "Solve it."}
    request = request_type(model="native-family", **inputs, max_thinking_tokens=512,
                           reasoning_effort="low", reasoning={"budget_tokens": 4096})
    assert request.max_thinking_tokens == 512
    assert request.reasoning_effort == "low"


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, "4096"])
def test_anthropic_dict_budget_cannot_bypass_validation(budget):
    with pytest.raises(ValueError, match="thinking.budget_tokens"):
        AnthropicRequest(model="native-family", messages=MESSAGES,
                         thinking={"type": "enabled", "budget_tokens": budget})


def test_anthropic_preserves_explicit_template_effort_with_large_budget():
    request = to_chat_completion(AnthropicRequest(
        model="native-family", messages=MESSAGES,
        thinking={"type": "enabled", "budget_tokens": 40000},
        chat_template_kwargs={"reasoning_effort": "low"},
    ))
    assert request.reasoning_effort == "low"
    assert request.chat_template_kwargs["reasoning_effort"] == "low"
    assert request.max_thinking_tokens == 40000


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
@pytest.mark.parametrize("effort", [None, "low", "xhigh"])
def test_http_generation_handoff_keeps_budget_and_effort_independent(monkeypatch, path, effort):
    """Exercise real routes with synthetic generation, not model inference."""
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
    monkeypatch.setattr(server, "stream_responses_api", capture)
    body = {"model": "native-family", "stream": True, "reasoning_effort": effort}
    if path == "/v1/messages":
        body.update(messages=MESSAGES, max_tokens=50000,
                    thinking={"type": "enabled", "budget_tokens": 40000})
    elif path == "/v1/responses":
        body.update(input="Solve it.", max_output_tokens=50000,
                    reasoning={"budget_tokens": 40000})
    else:
        body.update(messages=MESSAGES, max_tokens=50000,
                    reasoning={"budget_tokens": 40000})
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert captured["request"].max_thinking_tokens == 40000
    assert captured["request"].reasoning_effort == effort
    assert captured["kwargs"]["chat_template_kwargs"]["thinking_budget"] == 40000
    assert captured["kwargs"]["chat_template_kwargs"].get("reasoning_effort") == effort

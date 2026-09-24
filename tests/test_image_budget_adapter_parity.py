"""Image token budgets must survive every adapter and the engine kwargs hop."""
import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
from vmlx_engine.api.models import ChatCompletionRequest
from vmlx_engine.api.ollama_adapter import ollama_chat_to_openai, ollama_generate_to_openai_chat
from vmlx_engine.video_controls import video_control_kwargs


MESSAGES = [{"role": "user", "content": "Describe the image."}]


@pytest.mark.parametrize("budget", [70, 140, 280, 560, 1120])
def test_anthropic_budget_reaches_engine_kwargs(budget):
    req = AnthropicRequest(model="gemma", messages=MESSAGES, max_tokens=16, image_token_budget=budget)
    chat = to_chat_completion(req)
    assert chat.image_token_budget == budget
    assert video_control_kwargs(chat)["image_token_budget"] == budget


@pytest.mark.parametrize("adapter", [ollama_chat_to_openai, ollama_generate_to_openai_chat])
@pytest.mark.parametrize("controls", [
    {"image_token_budget": 1120},
    {"options": {"image_token_budget": 1120}},
    {"image_token_budget": 1120, "options": {"image_token_budget": 280}},
    {"image_token_budget": None, "options": {"image_token_budget": 1120}},
])
def test_ollama_budget_precedence_reaches_engine_kwargs(adapter, controls):
    converted = adapter({"model": "gemma", "messages": MESSAGES, "prompt": "Describe the image.", **controls})
    assert converted["image_token_budget"] == 1120
    req = ChatCompletionRequest(**converted)
    assert video_control_kwargs(req)["image_token_budget"] == 1120


@pytest.mark.parametrize("budget", [0, -1, 1000])
def test_anthropic_invalid_budget_rejected_at_request_boundary(budget):
    with pytest.raises(ValueError, match="image_token_budget must be one of"):
        AnthropicRequest(model="gemma", messages=MESSAGES, max_tokens=16, image_token_budget=budget)


@pytest.mark.parametrize("adapter", [ollama_chat_to_openai, ollama_generate_to_openai_chat])
def test_ollama_invalid_top_level_budget_not_replaced_by_valid_default(adapter):
    converted = adapter({"model": "gemma", "messages": MESSAGES, "prompt": "Hi", "image_token_budget": 0, "options": {"image_token_budget": 280}})
    assert converted["image_token_budget"] == 0
    with pytest.raises(ValueError, match="image_token_budget must be one of"):
        ChatCompletionRequest(**converted)


def test_unset_budget_stays_absent_from_engine_kwargs():
    req = to_chat_completion(AnthropicRequest(model="gemma", messages=MESSAGES, max_tokens=16))
    assert req.image_token_budget is None
    assert "image_token_budget" not in video_control_kwargs(req)
    assert "image_token_budget" not in video_control_kwargs({"image_token_budget": None})


def test_mapping_kwargs_preserve_explicit_image_budget():
    assert video_control_kwargs({"image_token_budget": 560})["image_token_budget"] == 560


@pytest.mark.parametrize("path,stream", [
    ("/api/chat", True), ("/api/generate", True),
    ("/v1/messages", False), ("/v1/messages", True),
])
def test_http_routes_forward_budget_to_generation(monkeypatch, path, stream):
    """Real HTTP route/adapter/kwargs execution with a synthetic generator.

    This proves transport, not model preprocessing or media accuracy.
    """
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server

    captured = _run_streaming_ollama_chat(
        monkeypatch, family_name="gemma4", model_type="gemma4",
        body={"model": "gemma-budget-test", "messages": MESSAGES,
              "stream": True, "image_token_budget": 280},
    )
    assert captured["kwargs"]["image_token_budget"] == 280
    captured.clear()
    stream_impl = server.stream_chat_completion

    async def positional_or_named_stream(*args, **kwargs):
        # Anthropic names these arguments; Ollama passes them positionally.
        if "engine" in kwargs:
            args = (kwargs.pop("engine"), kwargs.pop("messages"), kwargs.pop("request"))
        async for chunk in stream_impl(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(server, "stream_chat_completion", positional_or_named_stream)
    body = {"model": "gemma-budget-test", "messages": MESSAGES,
            "prompt": "Describe the image.", "max_tokens": 16,
            "stream": stream, "image_token_budget": 1120}
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert captured["kwargs"]["image_token_budget"] == 1120

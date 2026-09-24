"""Omni must not silently accept a thinking cap it cannot enforce."""

import pytest
from fastapi import HTTPException

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion
from vmlx_engine.api.models import ChatCompletionRequest
from vmlx_engine.omni_multimodal import dispatch_omni_chat_completion


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [
    {"max_thinking_tokens": 128},
    {"reasoning": {"budget_tokens": 128}},
    {"chat_template_kwargs": {"thinking_budget": 128}},
    {"max_thinking_tokens": 128, "enable_thinking": False},
])
async def test_omni_rejects_unenforced_budget_before_component_or_model_load(monkeypatch, controls):
    monkeypatch.setattr(
        "vmlx_engine.omni_multimodal.omni_multimodal_component_status",
        lambda path: pytest.fail("unsupported budget reached component inventory"),
    )
    request = ChatCompletionRequest(
        model="omni", messages=[{"role": "user", "content": "Hello"}], **controls,
    )
    with pytest.raises(HTTPException) as exc:
        await dispatch_omni_chat_completion(request, "/not-loaded")
    assert exc.value.status_code == 400
    assert "thinking-token budget" in exc.value.detail


@pytest.mark.asyncio
async def test_anthropic_native_budget_cannot_bypass_omni_guard(monkeypatch):
    monkeypatch.setattr(
        "vmlx_engine.omni_multimodal.omni_multimodal_component_status",
        lambda path: pytest.fail("unsupported budget reached component inventory"),
    )
    request = to_chat_completion(AnthropicRequest(
        model="omni", messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "enabled", "budget_tokens": 128},
    ))
    with pytest.raises(HTTPException) as exc:
        await dispatch_omni_chat_completion(request, "/not-loaded")
    assert exc.value.status_code == 400
    assert "thinking-token budget" in exc.value.detail


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_media_http_budget_rejection_survives_protocol_adapters(monkeypatch, path, stream):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server

    _run_streaming_ollama_chat(
        monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr("vmlx_engine.omni_multimodal.is_omni_multimodal_bundle", lambda path: True)
    monkeypatch.setattr(
        "vmlx_engine.omni_multimodal.omni_multimodal_component_status",
        lambda path: {"bundle_compatible": True, "modalities": ["text", "image"]},
    )
    monkeypatch.setattr(
        "vmlx_engine.omni_multimodal.OmniMultimodalDispatcher.get",
        lambda *args, **kwargs: pytest.fail("budget was lost before Omni model loading"),
    )
    image = "data:image/png;base64,AA=="
    body = {"model": "test-model", "stream": stream}
    if path == "/v1/responses":
        body.update(input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Describe this."},
            {"type": "input_image", "image_url": image},
        ]}], reasoning={"budget_tokens": 128})
    elif path == "/v1/messages":
        body.update(max_tokens=256, thinking={"type": "enabled", "budget_tokens": 128},
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
                    ]}])
    else:
        body.update(max_thinking_tokens=128, messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": image}},
        ]}])
    # The fixture already supplies an engine; these are adapter tests, not
    # model-loading/lifespan tests. Do not start the real idle-model monitor.
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "thinking-token budget" in response.text

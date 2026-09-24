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
    if path == "/v1/messages":
        payload = response.json()
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
        assert "thinking-token budget" in payload["error"]["message"]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_omni_cache_controls_survive_protocol_adapters(monkeypatch, path):
    from fastapi.testclient import TestClient
    from starlette.responses import JSONResponse
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server

    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr("vmlx_engine.omni_multimodal.is_omni_multimodal_bundle", lambda path: True)
    monkeypatch.setattr("vmlx_engine.omni_multimodal.omni_multimodal_component_status", lambda path: {"bundle_compatible": True, "modalities": ["text", "image"]})
    from types import SimpleNamespace
    monkeypatch.setattr(server._engine, "_scheduler_config", SimpleNamespace(
        enable_block_disk_cache=True, block_disk_cache_dir="/tmp/selected-native-pool",
        block_disk_cache_max_gb=1.25, cache_ttl_minutes=12,
    ), raising=False)
    captured = []
    async def dispatch(request, *args, **kwargs):
        captured.append((request.skip_prefix_cache, request.cache_salt))
        assert request.video_fps == 3
        assert request.video_max_frames == 2
        assert request.video_token_budget == 4096
        assert kwargs["disk_cache_enabled"] is True
        assert kwargs["disk_cache_policy"] == {
            "root": "/tmp/selected-native-pool", "max_size_bytes": int(1.25 * 1024**3),
            "ttl_minutes": 0.0,
        }
        return JSONResponse({"id": "test", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    monkeypatch.setattr("vmlx_engine.omni_multimodal.dispatch_omni_chat_completion", dispatch)
    body = {"model": "test-model", "skip_prefix_cache": True, "cache_salt": "request-salt",
            "video_fps": 3, "video_max_frames": 2, "video_token_budget": 4096}
    if path.endswith("responses"):
        body["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Describe."}, {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]
    elif path.endswith("messages"):
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "Describe."}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}]
    else:
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "Describe."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert captured == [(True, "request-salt")]

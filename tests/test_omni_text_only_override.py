"""Explicit text-only mode must win over native media bundle detection."""
import pytest


@pytest.fixture
def native_bundle(monkeypatch):
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server, omni_multimodal as omni
    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/native-omni")
    monkeypatch.setattr(server, "_force_text_only", True)
    monkeypatch.setattr(omni, "is_omni_multimodal_bundle", lambda _: True)
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {
        "bundle_compatible": True, "modalities": ["text", "audio", "image", "video"]})
    calls = []
    async def dispatch(*args, **kwargs):
        calls.append("native media")
        return {"choices": [{"message": {"role": "assistant", "content": "wrong route"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(omni, "dispatch_omni_chat_completion", dispatch)
    return server, calls


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/api/chat"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("in_history", [False, True])
def test_forced_text_only_rejects_media_before_dispatch(native_bundle, path, stream, in_history):
    from fastapi.testclient import TestClient
    server, calls = native_bundle
    image = "data:image/png;base64,AA=="
    body = {"model": "test-model", "stream": stream, "max_tokens": 32}
    if path.endswith("responses"):
        part = {"type": "input_image", "image_url": image}
    elif path.endswith("messages"):
        part = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
    else:
        part = {"type": "image_url", "image_url": {"url": image}}
    messages = [{"role": "user", "content": [part]}]
    if path == "/api/chat":
        messages = [{"role": "user", "content": "describe", "images": ["AA=="]}]
    if in_history:
        messages += [{"role": "assistant", "content": "old answer"}, {"role": "user", "content": "continue"}]
    body["input" if path.endswith("responses") else "messages"] = messages
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "text-only" in response.text
    assert not calls
    if path.endswith("messages"):
        assert response.json()["error"]["type"] == "invalid_request_error"


def test_forced_text_only_capabilities_override_native_and_generic_detection(native_bundle, monkeypatch):
    from fastapi.testclient import TestClient
    server, _ = native_bundle
    monkeypatch.setattr(server._engine, "is_mllm", True)
    assert server._loaded_omni_modalities() is None
    assert server._loaded_runtime_modalities() == ["text"]
    client = TestClient(server.app)
    try:
        response = client.get("/v1/capabilities")
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert response.json()["modalities"] == ["text"]
    assert response.json()["media"]["runtime_modalities"] == ["text"]


def test_forced_text_only_rejects_restored_response_media(native_bundle, monkeypatch):
    from fastapi.testclient import TestClient
    server, calls = native_bundle
    monkeypatch.setattr(server, "_responses_get_history", lambda _: [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
        {"role": "assistant", "content": "previous answer"}])
    client = TestClient(server.app)
    try:
        response = client.post("/v1/responses", json={"model": "test-model", "stream": False,
            "previous_response_id": "old-media", "input": "continue"})
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "text-only" in response.text
    assert not calls


def test_forced_text_only_keeps_text_requests_available(native_bundle):
    from fastapi.testclient import TestClient
    server, calls = native_bundle
    client = TestClient(server.app)
    try:
        response = client.post("/v1/chat/completions", json={"model": "test-model",
            "stream": True, "messages": [{"role": "user", "content": "hello"}]})
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert "data: [DONE]" in response.text
    assert not calls


@pytest.mark.parametrize("part", [
    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    {"type": "video_url", "video_url": {"url": "file:///unused.mp4"}},
])
def test_forced_text_only_rejects_other_native_media(native_bundle, part):
    from fastapi.testclient import TestClient
    server, calls = native_bundle
    client = TestClient(server.app)
    try:
        response = client.post("/v1/chat/completions", json={"model": "test-model",
            "messages": [{"role": "user", "content": [part]}]})
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "text-only" in response.text
    assert not calls

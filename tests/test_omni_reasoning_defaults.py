"""Native media uses the same request/server thinking precedence as text."""
import pytest


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
@pytest.mark.parametrize("server_default,template_default,controls,expected", [
    (False, {}, {}, False),
    (True, {}, {"chat_template_kwargs": {"enable_thinking": "false"}}, False),
    (None, {"enable_thinking": "off"}, {}, False),
    (True, {}, {"chat_template_kwargs": {"enable_thinking": None}}, True),
    (False, {}, {"enable_thinking": True}, True),
    (True, {}, {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}}, False),
    (None, {}, {}, True),
])
def test_native_thinking_default_precedence(monkeypatch, path, server_default, template_default, controls, expected):
    from fastapi.testclient import TestClient
    from starlette.responses import JSONResponse
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server, omni_multimodal as omni

    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr(server, "_default_enable_thinking", server_default)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", template_default)
    monkeypatch.setattr(omni, "is_omni_multimodal_bundle", lambda _: True)
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"bundle_compatible": True, "modalities": ["text", "image"]})
    captured = []
    async def dispatch(request, *args, **kwargs):
        captured.append(request.enable_thinking)
        return JSONResponse({"id": "test", "choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    monkeypatch.setattr(omni, "dispatch_omni_chat_completion", dispatch)
    image = "data:image/png;base64,AA=="
    body = {"model": "test-model", "stream": False, **controls}
    if path.endswith("responses"):
        body['input'] = [{"role": "user", "content": [{"type": "input_image", "image_url": image}]}]
    elif path.endswith("messages"):
        body.update(max_tokens=32, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}])
    else:
        body['messages'] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}]
    with_context = dict(template_default)
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert captured == [expected]
    assert template_default == with_context, "request must not mutate server defaults"

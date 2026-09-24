"""Native media routes cannot silently discard client constraints."""
import pytest
from fastapi import HTTPException

from vmlx_engine import omni_multimodal as omni
from vmlx_engine.api.models import ChatCompletionRequest

TOOL = {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {}}}}
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [
    {"tools": [TOOL], "tool_choice": "required"},
    {"response_format": {"type": "json_object"}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}, "strict": True}}},
    {"stop": ["END"]}, {"seed": 7}, {"top_k": 8}, {"min_p": .1},
    {"repetition_penalty": 1.1}, {"frequency_penalty": .1},
    {"presence_penalty": .1}, {"logit_bias": {"42": 2}}, {"logprobs": True},
    {"image_token_budget": 280}, {"image_max_pixels": 100352},
    {"image_resized_height": 224, "image_resized_width": 224},
])
async def test_unimplemented_native_constraints_fail_before_loading(monkeypatch, controls):
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"modalities": ["text", "image"]})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: pytest.fail("ignored constraint reached native load"))
    request = ChatCompletionRequest(model="omni", messages=[{"role": "user", "content": [IMAGE]}], **controls)
    with pytest.raises(HTTPException) as error:
        await omni.dispatch_omni_chat_completion(request, "/unused")
    assert error.value.status_code == 400
    assert "native omni" in error.value.detail.lower()


def test_neutral_controls_and_explicit_no_tools_remain_valid():
    request = ChatCompletionRequest(model="omni", messages=[{"role": "user", "content": [IMAGE]}],
        tools=[TOOL], tool_choice="none", response_format={"type": "text"},
        top_k=0, min_p=0, repetition_penalty=1, frequency_penalty=0,
        presence_penalty=0, logit_bias={}, logprobs=False)
    omni._validate_native_media_controls(request, [m.model_dump(exclude_none=True) for m in request.messages])


def test_malformed_native_tool_history_is_rejected_before_rendering():
    from vmlx_engine.omni_native_tools import prepare_native_tools
    with pytest.raises(HTTPException, match="function tool history"):
        prepare_native_tools([], None, [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call1"}]},
            {"role": "tool", "tool_call_id": "call1", "content": "result"},
        ])


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
@pytest.mark.parametrize("control", ["top_k", "image_max_pixels"])
def test_native_rejection_survives_each_protocol_adapter(monkeypatch, path, control):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server
    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr(omni, "is_omni_multimodal_bundle", lambda _: True)
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"bundle_compatible": True, "modalities": ["text", "image"]})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: pytest.fail("constraint lost in protocol bridge"))
    body = {"model": "test-model", "stream": True}
    if path.endswith("responses"):
        body['input'] = [{"role": "user", "content": [{"type": "input_image", "image_url": IMAGE['image_url']['url']}]}]
    elif path.endswith("messages"):
        body.update(max_tokens=32, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}])
    else:
        body['messages'] = [{"role": "user", "content": [IMAGE]}]
    body[control] = {"top_k": 8, "image_max_pixels": 100352, "tools": [TOOL]}[control]
    if control == "tools" and path.endswith("messages"):
        body['tools'] = [{"name": "inspect", "input_schema": {"type": "object", "properties": {}}}]
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "Native Omni media" in response.text


@pytest.mark.parametrize("text_format", [{"type": "json_object"}, {"format": {"type": "json_schema", "name": "result", "schema": {"type": "object", "properties": {}}, "strict": True}}])
def test_responses_structured_output_is_not_lost_in_native_bridge(monkeypatch, text_format):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server
    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr(omni, "is_omni_multimodal_bundle", lambda _: True)
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"bundle_compatible": True, "modalities": ["text", "image"]})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: pytest.fail("structured output lost"))
    client = TestClient(server.app)
    try:
        response = client.post('/v1/responses', json={"model": "test-model", "stream": True, "text": text_format,
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": IMAGE['image_url']['url']}]}]})
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert "structured output" in response.text


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_native_route_failure_never_falls_back_to_text(monkeypatch, path):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server
    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
        body={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True})
    monkeypatch.setattr(server, "_model_path", "/not-loaded-omni")
    monkeypatch.setattr(omni, "is_omni_multimodal_bundle", lambda _: True)
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"bundle_compatible": True, "modalities": ["text", "image"]})
    async def broken(*a, **kw):
        raise RuntimeError("native encoder unavailable")
    monkeypatch.setattr(omni, "dispatch_omni_chat_completion", broken)
    body = {"model": "test-model", "stream": True}
    if path.endswith("responses"):
        body['input'] = [{"role": "user", "content": [{"type": "input_image", "image_url": IMAGE['image_url']['url']}]}]
    elif path.endswith("messages"):
        body.update(max_tokens=32, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}])
    else:
        body['messages'] = [{"role": "user", "content": [IMAGE]}]
    client = TestClient(server.app)
    try:
        response = client.post(path, json=body)
    finally:
        client.close()
    assert response.status_code == 500, response.text
    assert "native encoder unavailable" in response.text
    if path.endswith("messages"):
        assert response.json()['error']['type'] == 'api_error'

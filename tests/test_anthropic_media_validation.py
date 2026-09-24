"""Recognized media blocks must never disappear during Messages conversion."""
import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion


@pytest.mark.parametrize("block", [
    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
    {"type": "image", "source": {}},
    {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": ""}},
    {"type": "video", "source": {"type": "url", "url": ""}},
    {"type": "image", "source": {"type": "file", "path": "/missing"}},
    {"type": "audio", "source": "not-an-object"},
])
def test_malformed_media_source_rejects_instead_of_becoming_text_only(block):
    request = AnthropicRequest(model="test", messages=[{
        "role": "user", "content": [{"type": "text", "text": "Describe."}, block],
    }])
    with pytest.raises(ValueError, match="source"):
        to_chat_completion(request)


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_media_has_native_http_error_without_engine(monkeypatch, stream):
    from fastapi.testclient import TestClient
    from vmlx_engine import server

    monkeypatch.setattr(server, "get_engine", lambda: pytest.fail("malformed media reached model"))
    client = TestClient(server.app)
    try:
        response = client.post('/v1/messages', json={
            "model": "test", "stream": stream, "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Transcribe."},
                {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
            ]}],
        })
    finally:
        client.close()
    assert response.status_code == 400, response.text
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "source" in response.json()["error"]["message"]

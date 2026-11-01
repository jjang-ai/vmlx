"""Chat must retain native MiniMax media in the message that supplied it."""
from types import SimpleNamespace

import httpx
import pytest

from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_m3_chat_preserves_historical_image_and_current_video(monkeypatch, stream):
    captured = []

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = False
        preserve_native_tool_format = True

        async def chat(self, **kwargs):
            captured.append(kwargs)
            return GenerationOutput(text="red then blue", tokens=[], prompt_tokens=20,
                                    completion_tokens=3, finished=True, finish_reason="stop")

        async def stream_chat(self, **kwargs):
            captured.append(kwargs)
            yield GenerationOutput(text="red then blue", new_text="red then blue",
                                   tokens=[], prompt_tokens=20, completion_tokens=3,
                                   finished=True, finish_reason="stop")

    monkeypatch.setattr(server, "_engine", Engine())
    monkeypatch.setattr(server, "_m3_vl_media_ok", lambda _: True)
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_model_path", None)
    for name in ("_model_name", "_served_model_name"):
        monkeypatch.setattr(server, name, "media-placement-test")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Earlier still."},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
        {"role": "assistant", "content": "That still is red."},
        {"role": "user", "content": [{"type": "text", "text": "Describe this video."},
         {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}}]},
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "media-placement-test", "messages": messages, "stream": stream,
        })
    assert response.status_code == 200, response.text
    assert len(captured) == 1
    turns = captured[0]["messages"]
    assert turns[0]["content"][1]["type"] == "image_url"
    assert turns[2]["content"][1]["type"] == "video_url"
    assert not captured[0].get("images")
    assert not captured[0].get("videos")

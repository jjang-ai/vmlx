"""/v1/responses serves DeepSeek-V4.1 images (text-routed VL), like /v1/chat/completions.

The vMLX panel's default chat wire is /v1/responses. Its media guard knew the M3-VL carve-out only, so a V4.1 VL
build answered image turns with HTTP 400 "the loaded runtime is text-only" while /v1/chat/completions served them
(live Electron run of the DeepSeek-V4.1 JANGHT bundle, 2026-10-26)."""
import asyncio
from types import SimpleNamespace

import pytest

RED = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


class _Captured(Exception):
    pass


def _engine(seen):
    async def chat(messages=None, **kw):
        seen.append({"messages": messages, **kw})
        raise _Captured

    return SimpleNamespace(is_mllm=False, preserve_native_tool_format=False, _dsv41_vl_preprocess=object(),
                           chat=chat, stream_chat=chat)


def _request(server_mod):
    from vmlx_engine.api.models import ResponsesRequest
    return ResponsesRequest(model="test", stream=False, input=[{"role": "user", "content": [
        {"type": "input_text", "text": "What color is this image? One word."},
        {"type": "input_image", "image_url": RED}]}])


def test_responses_hands_dsv41_images_to_the_engine(tmp_path, monkeypatch):
    # Plain asyncio.run: no async test plugin needed.
    from fastapi import Request
    from vmlx_engine import server
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_resolve_model_name", lambda: "test")
    seen = []
    monkeypatch.setattr(server, "get_engine", lambda: _engine(seen))
    monkeypatch.setattr(server, "_dsv41_vl_media_ok", lambda engine: True)
    with pytest.raises(Exception):
        asyncio.run(server.create_response(_request(server), Request({"type": "http", "headers": []})))
    assert seen, "the request never reached the engine"
    kw = seen[0]
    assert kw.get("images") and len(kw["images"]) == 1
    src = kw.get("_dsv41_vl_source_messages")
    assert src and any(isinstance(m.get("content"), list) for m in src)        # image part kept in its turn
    assert all(not isinstance(m.get("content"), list) for m in kw["messages"])  # text route


def test_responses_still_rejects_images_on_text_only_runtime(tmp_path, monkeypatch):
    from fastapi import HTTPException, Request
    from vmlx_engine import server
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_resolve_model_name", lambda: "test")
    seen = []
    monkeypatch.setattr(server, "get_engine", lambda: _engine(seen))
    monkeypatch.setattr(server, "_dsv41_vl_media_ok", lambda engine: False)
    with pytest.raises(HTTPException) as err:
        asyncio.run(server.create_response(_request(server), Request({"type": "http", "headers": []})))
    assert err.value.status_code == 400 and not seen

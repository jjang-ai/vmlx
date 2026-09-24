"""Native restored tokens remain input tokens across protocol adapters."""
import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vmlx_engine import omni_multimodal as omni


@pytest.mark.parametrize("mode", ["matching", "restored", "bypass"])
def test_native_usage_counts_restored_offset_before_decode(tmp_path, monkeypatch, mode):
    class Session:
        _cache = [SimpleNamespace(offset=9000), SimpleNamespace(offset=320)]
        mlx_model = SimpleNamespace(backbone=SimpleNamespace(fa_idx=1))
        _last_prompt_tokens = 17
        _last_completion_tokens = 4

        def reset(self):
            self._cache = None

        def turn(self, **kwargs):
            # Do not count generated tokens or allocated cache capacity.
            self._cache = [SimpleNamespace(offset=9999), SimpleNamespace(offset=340)]
            return "answer"

    d = omni.OmniMultimodalDispatcher.__new__(omni.OmniMultimodalDispatcher)
    d._session = Session()
    d._backend = "stage1"
    d._lock = threading.Lock()
    d._scratch_dir = tmp_path
    messages = [{"role": "user", "content": "old"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "next"}]
    d._last_signature = omni._conversation_signature(messages[:-1], False, None)
    if mode == "restored":
        native_cache = d._session._cache
        d._session._cache = None
        d._last_signature = None

        def restore(signature):
            d._session._cache = native_cache
            d._last_signature = signature
            return True

        monkeypatch.setattr(d, "_try_restore_session_snapshot", restore)
    monkeypatch.setattr(omni, "_run_omni_full_history", lambda session, *a, **kw: session.turn())
    bypass = mode == "bypass"
    result = d.chat(messages, enable_thinking=False, force_reset=bypass)
    assert result["prompt_tokens"] == (17 if bypass else 337)
    assert result["cached_tokens"] == (0 if bypass else 320)
    assert result["completion_tokens"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_chat_usage_preserves_cache_details(monkeypatch, stream):
    class Dispatcher:
        def chat(self, **kwargs):
            if kwargs.get("token_callback"):
                kwargs["token_callback"](1, "answer")
            return {"content": "answer", "prompt_tokens": 337,
                    "cached_tokens": 320, "completion_tokens": 4}

        def finish_request_cache(self):
            pass

        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"modalities": ["text"]})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: Dispatcher())
    request = SimpleNamespace(model="omni", messages=[{"role": "user", "content": "next"}], stream=stream)
    response = await omni.dispatch_omni_chat_completion(request, "/unused")
    if stream:
        payloads = [json.loads(line[6:]) async for raw in response.body_iterator
                    for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        response = next(p for p in payloads if p.get("usage"))
    assert response["usage"] == {"prompt_tokens": 337, "completion_tokens": 4,
                                  "total_tokens": 341,
                                  "prompt_tokens_details": {"cached_tokens": 320}}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_responses_usage_preserves_cache_details(monkeypatch, stream):
    from starlette.responses import StreamingResponse
    from vmlx_engine import server

    usage = {"prompt_tokens": 337, "completion_tokens": 4, "total_tokens": 341,
             "prompt_tokens_details": {"cached_tokens": 320}}
    if stream:
        async def chunks():
            yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}], "usage": usage}) + '\n\n'
            yield 'data: [DONE]\n\n'
        monkeypatch.setattr(server, "_responses_store_history", lambda *a, **kw: None)
        events = [json.loads(line[6:]) async for raw in server._adapt_omni_chat_stream_to_responses(
            StreamingResponse(chunks()), SimpleNamespace(model="omni"))
            for line in raw.splitlines() if line.startswith("data: ")]
        response = next(e["response"] for e in events if e["type"] == "response.completed")
    else:
        response = server._adapt_omni_chat_completion_to_responses_payload({
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": usage}, "omni")
    assert response["usage"] == {"input_tokens": 337, "output_tokens": 4, "total_tokens": 341,
                                  "input_tokens_details": {"cached_tokens": 320}}

"""Native media honors the public no-read/no-write cache bypass contract."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vmlx_engine import omni_multimodal as omni


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("controls", [{"skip_prefix_cache": True}, {"cache_salt": "private-run"}])
async def test_native_bypass_skips_restore_and_publication_but_releases_state(monkeypatch, stream, controls):
    events = []
    class Dispatcher:
        def submit(self, fn, *args):
            future = Future()
            try:
                future.set_result(fn(*args))
            except Exception as error:
                future.set_exception(error)
            return future

        def chat(self, **kwargs):
            events.append(("decode", kwargs["force_reset"]))
            if kwargs.get("token_callback"):
                kwargs["token_callback"](1, "answer")
            return {"content": "answer", "prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 1}

        def finish_request_cache(self):
            events.append(("write", True))

        def reset(self):
            events.append(("release", True))

    monkeypatch.setattr(omni.OmniMultimodalDispatcher, "get", lambda *a, **kw: Dispatcher())
    monkeypatch.setattr(omni, "omni_multimodal_component_status", lambda _: {"modalities": ["text"]})
    request = SimpleNamespace(model="omni", messages=[{"role": "user", "content": "hello"}], enable_thinking=False, stream=stream, **controls)
    response = await omni.dispatch_omni_chat_completion(request, "/unused", disk_cache_enabled=True)
    if stream:
        chunks = [chunk async for chunk in response.body_iterator]
        assert chunks[-1] == 'data: [DONE]\n\n'
    else:
        assert response['choices'][0]['message']['content'] == 'answer'
    assert events == [("decode", True), ("release", True)]

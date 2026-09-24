"""A media response cannot finish or begin the next decode ahead of its SSD write."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vmlx_engine.omni_multimodal import OmniMultimodalDispatcher, dispatch_omni_chat_completion


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_omni_terminal_waits_for_request_snapshot(monkeypatch, stream):
    writer_started = threading.Event()
    release_writer = threading.Event()
    events = []
    owner = ThreadPoolExecutor(max_workers=1)

    class Dispatcher:
        def submit(self, fn, *args, **kwargs):
            return owner.submit(fn, *args, **kwargs)

        def chat(self, token_callback=None, **kwargs):
            events.append("decode")
            if token_callback:
                token_callback(1, "answer")
            return {"content": "answer", "completion_tokens": 1, "finish_reason": "stop"}

        def finish_request_cache(self):
            return self._persist()

        def schedule_session_l2_persist(self):
            return self.submit(self._persist)

        def _persist(self):
            writer_started.set()
            assert release_writer.wait(5), "test did not release snapshot writer"
            events.append("persisted")
            return True

    dispatcher = Dispatcher()
    monkeypatch.setattr(OmniMultimodalDispatcher, "get", lambda *a, **k: dispatcher)
    monkeypatch.setattr("vmlx_engine.omni_multimodal.omni_multimodal_component_status", lambda _: {"modalities": ["text"]})
    request = SimpleNamespace(model="omni", messages=[{"role": "user", "content": "hello"}], stream=stream, enable_thinking=False)
    chunks = []

    async def consume():
        response = await dispatch_omni_chat_completion(request, "/test", disk_cache_enabled=True)
        if stream:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
        events.append("terminal")

    task = asyncio.create_task(consume())
    try:
        assert await asyncio.to_thread(writer_started.wait, 3)
        # Let the HTTP consumer process every ready event; the blocked native
        # writer remains the only reason terminal completion may be pending.
        await asyncio.sleep(0.02)
        assert not task.done(), events
        assert not any("[DONE]" in chunk for chunk in chunks)
        if stream:
            assert any('"content": "answer"' in chunk for chunk in chunks)
        release_writer.set()
        await asyncio.wait_for(task, 3)
        assert events == ["decode", "persisted", "terminal"]
    finally:
        release_writer.set()
        await asyncio.gather(task, return_exceptions=True)
        owner.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_omni_write_failure_is_not_a_successful_completion(monkeypatch, stream):
    from concurrent.futures import Future
    from fastapi import HTTPException

    class Dispatcher:
        finish_request_cache = OmniMultimodalDispatcher.finish_request_cache
        _disk_cache_enabled = True
        _backend = "stage1"
        _session_l2_stats = {"last_error": "disk full"}
        resets = 0

        def submit(self, fn, *args, **kwargs):
            f = Future()
            try:
                f.set_result(fn(*args, **kwargs))
            except Exception as e:
                f.set_exception(e)
            return f

        def chat(self, **kwargs):
            return {"content": "answer", "completion_tokens": 1, "finish_reason": "stop"}

        def _persist_session_snapshot(self):
            return False

        def reset(self):
            self.resets += 1

    dispatcher = Dispatcher()
    monkeypatch.setattr(OmniMultimodalDispatcher, "get", lambda *a, **k: dispatcher)
    monkeypatch.setattr("vmlx_engine.omni_multimodal.omni_multimodal_component_status", lambda _: {"modalities": ["text"]})
    request = SimpleNamespace(model="omni", messages=[{"role": "user", "content": "hello"}], stream=stream, enable_thinking=False)
    if stream:
        response = await dispatch_omni_chat_completion(request, "/test", disk_cache_enabled=True)
        chunks = [chunk async for chunk in response.body_iterator]
        assert any('"error"' in chunk and "disk full" in chunk for chunk in chunks)
        assert not any('"finish_reason": "stop"' in chunk for chunk in chunks)
    else:
        with pytest.raises(HTTPException, match="disk full") as error:
            await dispatch_omni_chat_completion(request, "/test", disk_cache_enabled=True)
        assert error.value.status_code == 500
    assert dispatcher.resets == 1


@pytest.mark.parametrize("disk_enabled", [False, True])
def test_native_payload_is_released_only_after_its_write(disk_enabled):
    session = SimpleNamespace(_cache=[object()], _history_text=[{"role": "user", "content": "history"}])
    order = []
    d = OmniMultimodalDispatcher.__new__(OmniMultimodalDispatcher)
    d._backend = 'stage1'
    d._disk_cache_enabled = disk_enabled
    d._lock = threading.Lock()
    d._session = session
    d._last_signature = 'boundary'
    def persist():
        assert session._cache and session._history_text
        order.append('persisted')
        return True
    def reset():
        order.append('released')
        session._cache = None
        session._history_text = []
    session.reset = reset
    d._persist_session_snapshot = persist
    d.finish_request_cache()
    assert order == (['persisted', 'released'] if disk_enabled else ['released'])
    assert session._cache is None and session._history_text == []
    assert d._last_signature is None

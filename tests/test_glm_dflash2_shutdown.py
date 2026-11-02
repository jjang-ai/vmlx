"""Native SSD lifecycle checks without loading a model or creating MLX arrays."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx.core")
from vmlx_engine import glm_dflash2 as runtime
from vmlx_engine.engine.simple import SimpleEngine


def test_close_drains_queued_writes_and_terminates_worker(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    written = []

    def write(self, name, key, snapshot):
        entered.set()
        assert release.wait(3)
        written.append(name)
        return {"outcome": "stored", "durable": True, "retained_tokens": len(key)}

    monkeypatch.setattr(runtime.PrefixDiskTier, "_write", write)
    monkeypatch.setattr(runtime.PrefixDiskTier, "_freeze", lambda self, key, snap: snap)
    disk = runtime.PrefixDiskTier(tmp_path, 1024)
    first = disk.put((1,), None)
    second = disk.put((2,), None)
    assert entered.wait(1)
    closing = threading.Thread(target=disk.close)
    closing.start()
    try:
        closing.join(0.05)
        assert closing.is_alive(), "close returned while a snapshot was unwritten"
    finally:
        release.set()
        closing.join(3)
    assert not closing.is_alive()
    assert written == [disk.digest((1,)), disk.digest((2,))]
    assert first.result()["durable"] and second.result()["durable"]
    assert not disk._thread.is_alive()
    assert disk._queue.unfinished_tasks == 0
    disk.close()  # idempotent teardown


def test_simple_stop_drains_native_store_before_releasing_model(monkeypatch):
    events = []
    store = SimpleNamespace(disk=SimpleNamespace(close=lambda: events.append("disk")))
    target = SimpleNamespace(close=lambda: events.append("adapter"))
    monkeypatch.setattr(runtime, "_PREFIX_STORES", {1: store})
    monkeypatch.setattr(runtime, "_TARGETS", {2: target})

    async def run():
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        engine._loaded = True
        engine._model = object()

        async def on_model_worker(fn):
            assert engine._model is not None
            fn()
        engine._run_model_call = on_model_worker
        await engine.stop()
        assert engine._model is None
        assert not engine._loaded

    asyncio.run(run())
    assert events == ["disk", "adapter"]
    assert runtime._PREFIX_STORES == {}
    assert runtime._TARGETS == {}


def test_native_writer_receipt_fences_next_tool_but_not_terminal(tmp_path, monkeypatch):
    release = threading.Event()
    def write(self, name, key, snapshot):
        assert release.wait(3)
        return {"outcome": "stored", "durable": True, "retained_tokens": len(key)}
    monkeypatch.setattr(runtime.PrefixDiskTier, "_write", write)
    monkeypatch.setattr(runtime.PrefixDiskTier, "_freeze", lambda self, key, snap: snap)
    disk = runtime.PrefixDiskTier(tmp_path, 1024)
    receipt = disk.put((1, 2, 3), None)

    async def run():
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        chunk = SimpleNamespace(text="tool", finish_reason="stop", prompt_tokens=3,
                                completion_tokens=1, cached_tokens=0,
                                persistence_future=receipt)
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=lambda **kwargs: iter([chunk]))
        async def worker(fn):
            return fn()
        engine._run_model_call = worker
        stream = engine.stream_chat([], request_id="native-tool")
        terminal = await asyncio.wait_for(anext(stream), 1)
        assert terminal.finished and not receipt.done()
        await stream.aclose()
        continuation = asyncio.create_task(engine._await_prior_terminal_persistence("next-tool"))
        await asyncio.sleep(0.05)
        assert not continuation.done()
        release.set()
        await asyncio.wait_for(continuation, 1)
        assert engine.get_stats()["last_durability"]["durable"] is True
    try:
        asyncio.run(run())
    finally:
        release.set()
        disk.close()


def test_writer_failure_is_settled_as_not_durable(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk write failed")
    monkeypatch.setattr(runtime.PrefixDiskTier, "_write", fail)
    monkeypatch.setattr(runtime.PrefixDiskTier, "_freeze", lambda self, key, snap: snap)
    disk = runtime.PrefixDiskTier(tmp_path, 1024)
    try:
        result = disk.put((1, 2), None).result(timeout=1)
        assert result["outcome"] == "failed" and result["durable"] is False
        assert result["retained_tokens"] == 0
    finally:
        disk.close()

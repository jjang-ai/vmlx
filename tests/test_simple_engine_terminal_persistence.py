"""Actual SimpleEngine stream bridge, with CPU-only model outputs."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.engine.simple import SimpleEngine


def test_tool_terminal_drains_its_own_write_and_preserves_usage_in_strict_mode(monkeypatch):
    monkeypatch.setenv("VMLX_TERMINAL_PUBLISH", "strict")
    async def run():
        receipt = Future()
        chunk = SimpleNamespace(text="<tool_call>done</tool_call>", finish_reason="stop",
                                prompt_tokens=100, completion_tokens=7, cached_tokens=97,
                                cache_detail="dflash2-ssd", persistence_future=receipt)
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=lambda **_: iter([chunk]))
        async def call(fn):
            return fn()
        engine._run_model_call = call
        assert not await engine.request_graceful_stop(None)
        stream = engine.stream_chat([], request_id="tool-a")
        content = await anext(stream)
        assert not content.finished and content.new_text == chunk.text
        assert await engine.request_graceful_stop("tool-a")
        assert not await engine.request_graceful_stop("another-request")
        assert engine.get_stats()["terminal_cleanup_pending"]
        terminal_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not terminal_task.done()
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": 106})
        terminal = await terminal_task
        assert terminal.finished and terminal.new_text == ""
        assert terminal.cached_tokens == 97 and terminal.completion_tokens == 7
        assert terminal.generated_at == content.generated_at
        assert engine.get_stats()["last_durability"]["durable"]
        await stream.aclose()
        assert not await engine.request_graceful_stop("tool-a")
    asyncio.run(run())


def _engine_with_chunks(chunks_per_call):
    with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("fake")
    engine._loaded = True
    calls = iter(chunks_per_call)
    engine._model = SimpleNamespace(stream_chat=lambda **_: iter(next(calls)))
    async def call(fn):
        return fn()
    engine._run_model_call = call
    return engine


def test_tool_terminal_publishes_before_its_write_and_next_turn_waits_for_it():
    async def run():
        receipt = Future()
        tool_chunk = SimpleNamespace(text="<tool_call>done</tool_call>", finish_reason="stop",
                                     prompt_tokens=100, completion_tokens=7, cached_tokens=97,
                                     cache_detail="dflash2-ssd", persistence_future=receipt)
        next_chunk = SimpleNamespace(text="answer", finish_reason="stop", prompt_tokens=120,
                                     completion_tokens=3, cached_tokens=106, cache_detail="dflash2-ssd",
                                     prefill_usage={"tokens": 14, "seconds": 0.05})
        engine = _engine_with_chunks([[tool_chunk], [next_chunk]])

        stream = engine.stream_chat([], request_id="tool-a")
        terminal = await asyncio.wait_for(anext(stream), timeout=1.0)
        # Published at once, with its full content and usage, before the write.
        assert terminal.finished and terminal.new_text == tool_chunk.text
        assert terminal.cached_tokens == 97 and terminal.completion_tokens == 7
        assert not receipt.done()
        await stream.aclose()
        assert engine.get_stats()["terminal_cleanup_pending"]
        assert await engine.request_graceful_stop("tool-a")

        # The continuation is admitted only after the write settles.
        follow = engine.stream_chat([], request_id="tool-b")
        pending = asyncio.create_task(anext(follow))
        await asyncio.sleep(0.05)
        assert not pending.done()
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": 106})
        answer = await asyncio.wait_for(pending, timeout=1.0)
        assert answer.finished and answer.cached_tokens == 106
        assert answer.prefill_usage["tokens"] == 14
        assert answer.prefill_usage["admission_wait_seconds"] >= 0.04

        record = engine.get_stats()["last_durability"]
        assert record["request_id"] == "tool-a" and record["durable"]
        assert record["published_early"] is True
        assert not engine.get_stats()["terminal_cleanup_pending"]
        await follow.aclose()
    asyncio.run(run())


def test_published_terminal_write_survives_reader_close_and_stop_waits_for_it():
    async def run():
        receipt = Future()
        chunk = SimpleNamespace(text="final", finish_reason="stop", prompt_tokens=10,
                                completion_tokens=2, cached_tokens=0, cache_detail="",
                                persistence_future=receipt)
        engine = _engine_with_chunks([[chunk]])
        stream = engine.stream_chat([], request_id="final-a")
        await anext(stream)
        await stream.aclose()
        assert not receipt.cancelled()

        stopping = asyncio.create_task(engine.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done()
        receipt.set_result({"outcome": "stored", "durable": True})
        await asyncio.wait_for(stopping, timeout=1.0)
        assert engine.get_stats()["last_durability"]["durable"]
    asyncio.run(run())


def _ram_served_receipt():
    receipt = Future()
    receipt.vmlx_lookup_served_from_ram = True        # as dflash2_runtime._store_session marks it
    return receipt


def test_ram_served_terminal_write_does_not_hold_the_next_turn_but_stop_waits():
    """A DFlash2 session entry is in the RAM session store before its SSD write
    starts: the next turn is admitted at once, the write keeps settling in the
    background, and teardown still waits for it."""
    async def run():
        receipt = _ram_served_receipt()
        tool_chunk = SimpleNamespace(text="<tool_call>done</tool_call>", finish_reason="stop",
                                     prompt_tokens=100, completion_tokens=7, cached_tokens=97,
                                     cache_detail="dflash2-ram", persistence_future=receipt)
        next_chunk = SimpleNamespace(text="answer", finish_reason="stop", prompt_tokens=120,
                                     completion_tokens=3, cached_tokens=106, cache_detail="dflash2-ram")
        engine = _engine_with_chunks([[tool_chunk], [next_chunk]])
        stream = engine.stream_chat([], request_id="tool-a")
        terminal = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert terminal.finished and not receipt.done()
        await stream.aclose()

        follow = engine.stream_chat([], request_id="tool-b")
        answer = await asyncio.wait_for(anext(follow), timeout=1.0)   # not held by the pending write
        assert answer.finished and answer.cached_tokens == 106
        assert not receipt.done()
        await follow.aclose()

        stopping = asyncio.create_task(engine.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done(), "stop returned before the RAM-served write reached disk"
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": 106})
        await asyncio.wait_for(stopping, timeout=1.0)
        record = engine.get_stats()["last_durability"]
        assert record["request_id"] == "tool-a" and record["durable"]
    asyncio.run(run())


def test_cache_clear_waits_for_ram_served_writes():
    async def run():
        receipt = _ram_served_receipt()
        chunk = SimpleNamespace(text="final", finish_reason="stop", prompt_tokens=10,
                                completion_tokens=2, cached_tokens=0, cache_detail="",
                                persistence_future=receipt)
        engine = _engine_with_chunks([[chunk]])
        stream = engine.stream_chat([], request_id="final-a")
        await anext(stream)
        await stream.aclose()
        waiting = asyncio.create_task(
            engine._await_prior_terminal_persistence(None, include_background=True)
        )
        await asyncio.sleep(0.05)
        assert not waiting.done()
        assert await engine._await_prior_terminal_persistence("next") == 0.0
        receipt.set_result({"outcome": "stored", "durable": True})
        await asyncio.wait_for(waiting, timeout=1.0)
    asyncio.run(run())

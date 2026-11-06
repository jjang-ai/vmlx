# SPDX-License-Identifier: Apache-2.0
"""Text-only targets on the DFlash2 lane (Ripple): SimpleEngine's LLM path carries the lane's multi-token chunks,
prefill receipt, speculative stats and session-cache write exactly as the multimodal DFlash2 path does."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.engine import simple as simple_mod
from vmlx_engine.engine.simple import SimpleEngine

USAGE = {"tokens": 40, "seconds": 0.2, "scope": "model_prefill_and_prompt_state", "path": "dflash2"}


class _TextLaneModel:
    """Stands in for MLXLanguageModel on the DFlash2 text lane: cumulative completion_tokens, several ids per chunk."""

    def __init__(self, runs):
        self._runs = iter(runs)
        self.tokenizer = None
        self.calls = []

    def dflash2_text_lane(self, kwargs=None):
        kw = kwargs or {}
        return not (kw.get("logprobs") or kw.get("_vmlx_response_format"))

    def stream_generate(self, **kwargs):
        self.calls.append(kwargs)
        return iter(next(self._runs))

    def generate(self, **_kwargs):
        raise AssertionError("the DFlash2 text lane must drain the stream route")


def _chunks(receipt=None, spec=None):
    return [
        SimpleNamespace(text="Hel", finished=False, finish_reason=None, completion_tokens=1, prompt_tokens=100,
                        cached_tokens=60, cache_detail="dflash2-ssd", prefill_usage=USAGE),
        SimpleNamespace(text="lo, wor", finished=False, finish_reason=None, completion_tokens=5, prompt_tokens=100,
                        cached_tokens=60, cache_detail="dflash2-ssd", prefill_usage=USAGE),
        SimpleNamespace(text="ld", finished=True, finish_reason="stop", completion_tokens=7, prompt_tokens=100,
                        cached_tokens=60, cache_detail="dflash2-ssd", prefill_usage=USAGE, tokens=[1, 2, 3, 4, 5, 6, 7],
                        persistence_future=receipt, spec_stats=spec),
    ]


def _engine(model):
    with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("fake")
    engine._loaded = True
    engine._model = model

    async def call(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    engine._run_model_call = call
    return engine


def test_multi_token_chunks_count_tokens_and_carry_the_prefill_receipt(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)
    spec = {"kind": "dflash2", "lane": "text", "cycles": 2}

    async def run():
        engine = _engine(_TextLaneModel([_chunks(spec=spec)]))
        outs = [o async for o in engine.stream_generate("p", max_tokens=6)]
        # max_tokens=6 is below the lane's cumulative 7 only at the terminal: the lane's own finish wins.
        assert [o.completion_tokens for o in outs] == [1, 5, 7]
        assert outs[-1].finished and outs[-1].finish_reason == "stop"
        assert outs[-1].tokens == [1, 2, 3, 4, 5, 6, 7] and outs[0].tokens == []
        assert outs[-1].prefill_usage["tokens"] == 40 and outs[-1].prefill_usage["scope"] == USAGE["scope"]
        assert outs[-1].cached_tokens == 60 and outs[-1].cache_detail == "dflash2-ssd"
        assert outs[-1].spec_stats == spec and outs[0].spec_stats is None
        assert outs[-1].text == "Hello, world"

    asyncio.run(run())


def test_max_tokens_is_the_lanes_to_enforce(monkeypatch):
    """A chunk that reaches max_tokens mid-stream must not end it: the lane's terminal chunk (finish reason, write
    receipt) follows. Cutting early returned finish_reason None (HTTP 502) and dropped the session write."""
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)

    async def run():
        receipt = Future()
        receipt.set_result({"outcome": "stored", "durable": True})
        chunks = [
            SimpleNamespace(text="a b c d", finished=False, finish_reason=None, completion_tokens=4, prompt_tokens=9),
            SimpleNamespace(text="", finished=True, finish_reason="length", completion_tokens=4, prompt_tokens=9,
                            tokens=[1, 2, 3, 4], persistence_future=receipt),
        ]
        engine = _engine(_TextLaneModel([chunks]))
        outs = [o async for o in engine.stream_generate("p", max_tokens=4)]
        assert len(outs) == 2 and not outs[0].finished
        assert outs[-1].finished and outs[-1].finish_reason == "length" and outs[-1].completion_tokens == 4

    asyncio.run(run())


def test_terminal_publishes_before_the_session_write_and_the_next_request_waits(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)
    monkeypatch.delenv("VMLX_TERMINAL_PUBLISH", raising=False)

    async def run():
        receipt = Future()
        engine = _engine(_TextLaneModel([_chunks(receipt=receipt), _chunks()]))
        outs = [o async for o in engine.stream_generate("p", max_tokens=64, request_id="tool-a")]
        assert outs[-1].finished and not receipt.done()          # published before the write settled
        follow = asyncio.create_task(anext(engine.stream_generate("p2", max_tokens=64, request_id="tool-b")))
        await asyncio.sleep(0.05)
        assert not follow.done()                                  # admission fenced on the prior write
        receipt.set_result({"outcome": "stored", "durable": True, "retained_tokens": 106})
        first = await asyncio.wait_for(follow, timeout=2.0)
        assert first.text == "Hel"
        assert "admission_wait_seconds" in first.prefill_usage

    asyncio.run(run())


def test_nonstream_generate_and_chat_drain_the_stream_route(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)

    async def run():
        model = _TextLaneModel([_chunks(), _chunks()])
        engine = _engine(model)
        out = await engine.generate("p", max_tokens=64)
        assert out.text == "Hello, world" and out.completion_tokens == 7 and out.prefill_usage["tokens"] == 40
        assert out.cached_tokens == 60 and out.tokens == [1, 2, 3, 4, 5, 6, 7]
        # chat renders the template, then the same stream route
        model.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: "rendered")
        out = await engine.chat([{"role": "user", "content": "hi"}], max_tokens=64)
        assert out.completion_tokens == 7 and model.calls[-1]["prompt"] == "rendered"

    asyncio.run(run())


def test_logprobs_and_guided_json_requests_leave_the_lane(monkeypatch):
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: True)
    engine = _engine(_TextLaneModel([]))
    assert engine._llm_dflash2_lane({})
    assert not engine._llm_dflash2_lane({"logprobs": True})
    assert not engine._llm_dflash2_lane({"_vmlx_response_format": {"type": "json_object"}})
    monkeypatch.setattr(simple_mod, "_dflash2_enabled", lambda: False)
    assert not engine._llm_dflash2_lane({})

"""Anthropic /v1/messages reports the user stop string that ended a generation (2026-10-27).

Live (DSV4.1 JANGHT, `vmlx serve`): `stop_sequences=["Mars"]` truncated the text correctly ("Mercury, Venus, Earth, ")
but answered `stop_reason: "end_turn"`, `stop_sequence: null`. Anthropic clients branch on `stop_sequence` to tell a
stop-string hit from a natural end. The Chat Completions chunks only say "stop", so the schedulers now record which
stop string matched (RequestOutput -> GenerationOutput -> the request), and both Anthropic paths report it.
"""
import inspect
import json
from types import SimpleNamespace

import pytest

from vmlx_engine.api.anthropic_adapter import AnthropicStreamAdapter


def _chunk(delta, finish=None, usage=None):
    body = {"id": "chatcmpl-x", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return "data: " + json.dumps(body) + "\n\n"


def _message_delta(events):
    rows = [json.loads(line[6:]) for e in events for line in e.splitlines() if line.startswith("data: ")]
    deltas = [r["delta"] for r in rows if r.get("type") == "message_delta"]
    assert len(deltas) == 1, deltas
    return deltas[0]


def _stream(chunks, **finalize_kwargs):
    a = AnthropicStreamAdapter("test-model", "msg_test")
    out = []
    for c in chunks:
        out += a.process_chunk(c)
    return out + a.finalize(**finalize_kwargs)


def test_stream_adapter_reports_the_matched_stop_sequence():
    events = _stream([_chunk({"role": "assistant"}), _chunk({"content": "Mercury, Venus, Earth, "}),
                      _chunk({}, finish="stop")], stop_sequence="Mars")
    assert _message_delta(events) == {"stop_reason": "stop_sequence", "stop_sequence": "Mars"}


def test_stream_adapter_natural_end_and_budget_keep_their_reasons():
    natural = _stream([_chunk({"role": "assistant"}), _chunk({"content": "Done."}), _chunk({}, finish="stop")])
    assert _message_delta(natural) == {"stop_reason": "end_turn", "stop_sequence": None}
    # a token budget that ran out is max_tokens even if a stop string was noted earlier
    budget = _stream([_chunk({"role": "assistant"}), _chunk({"content": "Mercury"}), _chunk({}, finish="length")],
                     stop_sequence="Mars")
    assert _message_delta(budget) == {"stop_reason": "max_tokens", "stop_sequence": None}


def _patch_server(monkeypatch, server, matched):
    async def fake_stream_chat_completion(*args, **kwargs):
        yield _chunk({"role": "assistant"})
        yield _chunk({"content": "Mercury, Venus, Earth, "})
        # the engine's terminal output carries the matched stop string (None for a natural end)
        server._note_stop_sequence(kwargs["request"], SimpleNamespace(stop_sequence=matched))
        yield _chunk({}, finish="stop", usage={"prompt_tokens": 9, "completion_tokens": 7})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(server, "stream_chat_completion", fake_stream_chat_completion)
    monkeypatch.setattr(server, "_model_name", "stop-seq-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "get_engine",
                        lambda: SimpleNamespace(tokenizer=SimpleNamespace(has_thinking=False), is_mllm=False))


class _Request:
    def __init__(self, stream):
        self.stream = stream

    async def json(self):
        return {"model": "stop-seq-test", "max_tokens": 32, "stream": self.stream, "stop_sequences": ["Mars"],
                "messages": [{"role": "user", "content": "List the planets in order, comma separated."}]}

    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("matched, reason", [("Mars", "stop_sequence"), (None, "end_turn")])
async def test_non_stream_messages_reports_stop_sequence(monkeypatch, matched, reason):
    import vmlx_engine.server as server

    _patch_server(monkeypatch, server, matched)
    response = await server.create_anthropic_message(_Request(stream=False))
    assert response["stop_reason"] == reason
    assert response["stop_sequence"] == matched
    assert response["content"] == [{"type": "text", "text": "Mercury, Venus, Earth, "}]


@pytest.mark.asyncio
@pytest.mark.parametrize("matched, reason", [("Mars", "stop_sequence"), (None, "end_turn")])
async def test_streamed_messages_reports_stop_sequence(monkeypatch, matched, reason):
    import vmlx_engine.server as server

    _patch_server(monkeypatch, server, matched)
    response = await server.create_anthropic_message(_Request(stream=True))
    events = []
    async for part in response.body_iterator:
        events.append(part if isinstance(part, str) else part.decode())
    assert _message_delta(events) == {"stop_reason": reason, "stop_sequence": matched}


def test_schedulers_and_engines_carry_the_matched_stop():
    """Every stop-string match site records the string; every scheduler->engine copy keeps it."""
    from vmlx_engine.engine import batched
    from vmlx_engine.engine.base import GenerationOutput
    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.request import RequestOutput
    from vmlx_engine.scheduler import Scheduler

    assert RequestOutput(request_id="r").stop_sequence is None
    assert GenerationOutput(text="").stop_sequence is None
    llm = inspect.getsource(Scheduler._process_batch_responses)
    assert "matched_stop = stop_str" in llm and "output.stop_sequence = matched_stop" in llm
    mllm = inspect.getsource(MLLMScheduler._process_batch_responses)
    assert mllm.count("request._stop_matched") == 2  # streamed match + match completed by finalize
    assert 'output.stop_sequence = getattr(request, "_stop_matched", None)' in mllm
    src = inspect.getsource(batched)
    assert src.count("finish_reason=output.finish_reason,") == src.count(
        'stop_sequence=getattr(output, "stop_sequence", None),') == 4

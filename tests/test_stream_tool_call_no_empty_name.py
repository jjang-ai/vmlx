# SPDX-License-Identifier: Apache-2.0
"""Streaming tool_calls must never emit an empty function.name.

Regression for the live OpenCode/Cline bug: the streaming emitter announced a
tool call with a placeholder ``{"name": "", "arguments": ""}`` START delta on
the first buffering tick, then sent the resolved name only in a later delta.

Clients built on the Vercel AI SDK ``@ai-sdk/openai-compatible`` provider
initialize a tool call from the FIRST delta seen for a given index and ignore
``name`` on all subsequent deltas. The empty-name START delta therefore
registers a tool named "", the real name is dropped, and every streamed tool
call is silently discarded.

Contract pinned here: for a streaming tool-call turn, EVERY emitted
tool_calls delta carries a non-empty ``function.name``, and the first such
delta already carries the fully-resolved name. ``finish_reason="tool_calls"``
must still terminate the stream.

The buffer-then-parse emitter has two identical START-delta blocks that split
on whether a reasoning parser is configured (server.py). MiniMax-M2 emits
reasoning_content and takes the reasoning-parser branch; a no-parser model
takes the other. Both are exercised below.
"""

from types import SimpleNamespace

import json

import pytest

from vmlx_engine import server
from vmlx_engine.api.models import ChatCompletionRequest, Message
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.reasoning.minimax_m2_parser import MiniMaxM2ReasoningParser


def _make_engine(deltas):
    class _Engine:
        tokenizer = SimpleNamespace(has_thinking=False)

        def __init__(self):
            self.aborted: list[str] = []

        async def stream_chat(self, *, messages, **kwargs):
            text = ""
            for i, d in enumerate(deltas):
                text += d
                yield GenerationOutput(
                    text=text,
                    new_text=d,
                    tokens=[i],
                    prompt_tokens=10,
                    completion_tokens=i + 1,
                    finished=(i == len(deltas) - 1),
                    finish_reason="stop" if i == len(deltas) - 1 else None,
                )

        async def abort_request(self, request_id):
            self.aborted.append(request_id)
            return True

    return _Engine()


def _request():
    return ChatCompletionRequest(
        model="MiniMax-M2",
        messages=[Message(role="user", content="weather in Paris?")],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
    )


async def _collect_tool_call_deltas(engine, request):
    chunks = []
    async for line in server.stream_chat_completion(
        engine,
        [m.model_dump(exclude_none=True) for m in request.messages],
        request,
        fastapi_request=None,
    ):
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))

    tc_deltas = []
    finish_reasons = []
    for c in chunks:
        for choice in c.get("choices") or []:
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                tc_deltas.append(tc)
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
    return tc_deltas, finish_reasons


def _assert_contract(tc_deltas, finish_reasons):
    assert tc_deltas, "a streaming tool-call turn must emit tool_calls deltas"
    # No delta may carry an empty function.name — the bug.
    for tc in tc_deltas:
        name = (tc.get("function") or {}).get("name")
        assert name, f"tool_calls delta emitted with empty function.name: {tc}"
    # The FIRST tool_calls delta already carries the resolved name.
    first = tc_deltas[0]
    assert first["function"]["name"] == "get_weather"
    assert finish_reasons and finish_reasons[-1] == "tool_calls"


@pytest.mark.asyncio
async def test_no_empty_name_reasoning_parser_branch(monkeypatch):
    """MiniMax-M2 path: a reasoning parser is configured (branch that also
    handles reasoning_content)."""
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "MiniMax-M2")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", MiniMaxM2ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "minimax")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

    deltas = [
        "I should check the weather.",
        "</think>",
        '<minimax:tool_call>\n<invoke name="get_weather">\n'
        '<parameter name="location">Paris</parameter>\n'
        "</invoke>\n</minimax:tool_call>",
    ]
    engine = _make_engine(deltas)
    tc_deltas, finish_reasons = await _collect_tool_call_deltas(engine, _request())
    _assert_contract(tc_deltas, finish_reasons)


@pytest.mark.asyncio
async def test_no_empty_name_no_reasoning_parser_branch(monkeypatch):
    """No reasoning parser configured: the other START-delta block."""
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "MiniMax-M2")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "minimax")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)

    deltas = [
        "Let me check. ",
        '<minimax:tool_call>\n<invoke name="get_weather">\n'
        '<parameter name="location">Paris</parameter>\n'
        "</invoke>\n</minimax:tool_call>",
    ]
    engine = _make_engine(deltas)
    tc_deltas, finish_reasons = await _collect_tool_call_deltas(engine, _request())
    _assert_contract(tc_deltas, finish_reasons)

# SPDX-License-Identifier: Apache-2.0
"""Responses API conversation-history regression tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _sse_payloads(events: list[str], event_type: str) -> list[dict]:
    payloads: list[dict] = []
    prefix = f"event: {event_type}\n"
    for event in events:
        if not event.startswith(prefix):
            continue
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))
                break
    return payloads


def test_responses_output_history_converter_accepts_pydantic_and_dict_items():
    from vmlx_engine.api.models import ResponsesOutputMessage, ResponsesOutputText
    from vmlx_engine.server import _responses_output_to_assistant_messages

    pydantic_items = [
        ResponsesOutputMessage(
            role="assistant",
            content=[ResponsesOutputText(type="output_text", text="visible answer")],
        )
    ]
    dict_items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "streamed answer"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"teal"}',
        },
    ]

    assert _responses_output_to_assistant_messages(pydantic_items) == [
        {"role": "assistant", "content": "visible answer"}
    ]
    assert _responses_output_to_assistant_messages(dict_items) == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"teal"}'},
                }
            ],
        },
        {"role": "assistant", "content": "streamed answer"},
    ]


def test_dsv4_responses_tool_choice_none_does_not_synthesize_tools():
    """Historical tool schemas must not override explicit tool_choice=none."""
    import inspect
    from vmlx_engine import server

    src = inspect.getsource(server.create_response)

    assert "elif not _suppress_tools and _is_dsv4_resp_msgs" in src
    assert "_synthesize_tools_from_message_tool_calls(messages)" in src


@pytest.mark.asyncio
async def test_responses_streaming_stores_history_for_previous_response_id():
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine import server

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, messages, **kwargs):
            yield SimpleNamespace(
                new_text="stored answer",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                finished=True,
            )

    server._responses_history.clear()
    server._reasoning_parser = None

    request = ResponsesRequest(model="unit-test-model", input="hello", stream=True)
    messages = [{"role": "user", "content": "hello"}]

    events = [
        event async for event in server.stream_responses_api(
            FakeEngine(), messages, request
        )
    ]
    completed = _sse_payloads(events, "response.completed")
    assert completed, "stream must emit response.completed"
    response_id = completed[-1]["response"]["id"]

    history = server._responses_get_history(response_id)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "stored answer"},
    ]

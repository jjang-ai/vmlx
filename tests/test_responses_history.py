# SPDX-License-Identifier: Apache-2.0
"""Responses API conversation-history regression tests."""

from __future__ import annotations

import json
import inspect
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


def test_responses_object_exposes_output_text_convenience_field():
    from vmlx_engine.api.models import (
        ResponsesFunctionCall,
        ResponsesObject,
        ResponsesOutputMessage,
        ResponsesOutputText,
        ResponsesUsage,
    )

    response = ResponsesObject(
        model="unit-test",
        output=[
            ResponsesOutputMessage(
                role="assistant",
                content=[ResponsesOutputText(type="reasoning", text="hidden")],
            ),
            ResponsesOutputMessage(
                role="assistant",
                content=[ResponsesOutputText(type="output_text", text="Paris")],
            ),
            ResponsesFunctionCall(name="lookup", arguments="{}"),
        ],
        usage=ResponsesUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )

    dumped = response.model_dump()

    assert response.output_text == "Paris"
    assert dumped["output_text"] == "Paris"


def test_dsv4_responses_tool_choice_none_does_not_synthesize_tools():
    """Historical tool schemas must not override explicit tool_choice=none."""
    import inspect
    from vmlx_engine import server

    src = inspect.getsource(server.create_response)

    assert "elif not _suppress_tools and _is_dsv4_resp_msgs" in src
    assert "_synthesize_tools_from_message_tool_calls(messages)" in src


def test_responses_previous_history_keeps_instructions_as_leading_system():
    from vmlx_engine.server import (
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="new response instructions",
        preserve_multimodal=False,
    )

    normalized = _normalize_leading_system_messages(
        previous_messages + current_messages
    )

    assert [m["role"] for m in normalized] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert normalized[0]["content"] == (
        "base instructions\n\nnew response instructions"
    )
    assert normalized[-1] == {"role": "user", "content": "second turn"}


def test_normalize_leading_system_messages_moves_mid_conversation_system():
    from vmlx_engine.server import _normalize_leading_system_messages

    normalized = _normalize_leading_system_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": "developer policy"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "json only"},
        ]
    )

    assert [m["role"] for m in normalized] == ["system", "user", "assistant"]
    assert normalized[0]["content"] == "developer policy\n\njson only"
    assert normalized[1:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_responses_history_json_instruction_stays_in_leading_system():
    from vmlx_engine.server import (
        _inject_json_instruction,
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="new response instructions",
        preserve_multimodal=False,
    )
    messages = _inject_json_instruction(
        previous_messages + current_messages,
        "Return valid JSON only.",
    )
    normalized = _normalize_leading_system_messages(messages)

    assert [m["role"] for m in normalized] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert normalized[0]["content"] == (
        "base instructions\n\nReturn valid JSON only.\n\n"
        "new response instructions"
    )


def test_responses_and_chat_call_sites_normalize_before_template_rendering():
    from vmlx_engine import server

    responses_src = inspect.getsource(server.create_response)
    chat_src = inspect.getsource(server.create_chat_completion)

    assert "messages = _normalize_leading_system_messages(messages)" in responses_src
    assert "messages = _normalize_leading_system_messages(messages)" in chat_src
    assert responses_src.index(
        "messages = _normalize_leading_system_messages(messages)"
    ) > responses_src.index("_inject_json_instruction(messages, json_instruction)")
    assert chat_src.index(
        "messages = _normalize_leading_system_messages(messages)"
    ) > chat_src.index("_inject_json_instruction(messages, json_instruction)")


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
    assert completed[-1]["response"]["output_text"] == "stored answer"

    history = server._responses_get_history(response_id)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "stored answer"},
    ]

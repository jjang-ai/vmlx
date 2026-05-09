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
            "content": "streamed answer",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"teal"}'},
                }
            ],
        },
    ]


def test_responses_output_merges_visible_text_with_tool_calls_for_tool_adjacency():
    """A Responses item stream may contain visible assistant text plus a
    function_call. Store both on the same assistant message so the next
    function_call_output is adjacent to the assistant.tool_calls turn."""
    from vmlx_engine.server import (
        _responses_input_to_messages,
        _responses_output_to_assistant_messages,
    )

    user_messages = [{"role": "user", "content": "review cache path"}]
    output_items = [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "I will inspect it."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "inspect_symbol",
            "arguments": '{"path":"vmlx_engine/server.py","symbol":"cache_stats"}',
        },
    ]
    stored = user_messages + _responses_output_to_assistant_messages(output_items)
    current = _responses_input_to_messages(
        [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "cache_stats source excerpt",
            },
            {"role": "user", "content": "continue"},
        ]
    )
    combined = stored + current

    tool_index = next(i for i, msg in enumerate(combined) if msg.get("role") == "tool")
    previous = combined[tool_index - 1]
    assert previous["role"] == "assistant"
    assert previous.get("tool_calls")
    assert previous["content"] == "I will inspect it."


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


def test_responses_previous_history_dedupes_identical_instructions_for_prefix_stability():
    from vmlx_engine.server import (
        _normalize_leading_system_messages,
        _responses_input_to_messages,
    )

    previous_messages = [
        {"role": "system", "content": "stable coding-review instructions"},
        {"role": "user", "content": "first turn"},
    ]
    current_messages = _responses_input_to_messages(
        "second turn",
        instructions="stable coding-review instructions",
        preserve_multimodal=False,
    )

    normalized = _normalize_leading_system_messages(
        previous_messages + current_messages
    )

    assert [m["role"] for m in normalized] == ["system", "user", "user"]
    assert normalized[0]["content"] == "stable coding-review instructions"
    assert normalized[1:] == [
        {"role": "user", "content": "first turn"},
        {"role": "user", "content": "second turn"},
    ]


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


@pytest.mark.asyncio
async def test_responses_streaming_reasoning_only_stores_placeholder_and_marker():
    from vmlx_engine.api.models import ResponsesRequest
    from vmlx_engine.reasoning.base import DeltaMessage
    from vmlx_engine import server

    class FakeReasoningParser:
        def reset_state(self, **_kwargs):
            pass

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            return DeltaMessage(reasoning=delta_text)

        def extract_reasoning(self, model_output):
            return model_output, None

    class FakeEngine:
        tokenizer = SimpleNamespace(has_thinking=True)

        async def stream_chat(self, messages, **kwargs):
            yield SimpleNamespace(
                new_text="thinking only",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                finished=True,
                cached_tokens=0,
            )

    server._responses_history.clear()
    server._responses_was_reasoning_only.clear()
    original_parser = server._reasoning_parser
    server._reasoning_parser = FakeReasoningParser()
    try:
        request = ResponsesRequest(
            model="unit-test-model",
            input="hello",
            stream=True,
            enable_thinking=True,
        )
        messages = [{"role": "user", "content": "hello"}]

        events = [
            event async for event in server.stream_responses_api(
                FakeEngine(), messages, request
            )
        ]
        completed = _sse_payloads(events, "response.completed")
        assert completed, "stream must emit response.completed"
        response_id = completed[-1]["response"]["id"]

        assert server._responses_get_history(response_id) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        assert server._chain_warnings_for_previous_response_id(response_id)
        warnings = completed[-1]["response"].get("warnings") or []
        assert any("produced reasoning only" in w for w in warnings)
    finally:
        server._reasoning_parser = original_parser
        server._responses_history.clear()
        server._responses_was_reasoning_only.clear()


# ─── Reasoning-only history alignment fix (claude 04:50 audit Option A) ────
# When the model returns ONLY reasoning items (no output_text, no function_call),
# `_responses_output_to_assistant_messages` historically returned [], leaving the
# stored history without an assistant turn. The next `previous_response_id` chain
# then loses cache prefix alignment AND loses the model's record that turn N
# happened. Fix: inject an empty-content placeholder assistant turn so the chain
# template still has a valid user→assistant→user sequence.
#
# See docs/internal/AUDIT_responses_reasoning_history_alignment_2026_05_09.md
# for the full root-cause trace and fix-design discussion.


def test_reasoning_only_output_emits_assistant_placeholder():
    """When output is reasoning-only, an empty-content assistant turn must be
    emitted so the next-turn cache prefix has a proper anchor and the model
    sees that the prior turn completed."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
        }
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert msgs == [
        {"role": "assistant", "content": ""}
    ], "reasoning-only output must inject empty placeholder assistant turn"


def test_message_item_with_only_reasoning_content_emits_assistant_placeholder():
    """Some Responses-compatible payloads represent reasoning as content parts
    on a message item rather than a top-level `type=reasoning` item. That is
    still reasoning-only output and must preserve the assistant history anchor.
    """
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "message",
            "content": [{"type": "reasoning", "text": "thinking..."}],
        }
    ]

    assert _responses_output_to_assistant_messages(items) == [
        {"role": "assistant", "content": ""}
    ]


def test_reasoning_only_detection_ignores_empty_output_text_message():
    """Streaming Responses emits an output_text message shell before it knows
    whether the model will produce visible content. An empty shell must not
    mask a reasoning-only completion.
    """
    from vmlx_engine.server import _responses_output_is_reasoning_only

    assert _responses_output_is_reasoning_only(
        [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": ""}],
            },
            {
                "type": "reasoning",
                "content": [{"type": "reasoning", "text": "thinking..."}],
            },
        ]
    )


def test_current_response_warning_for_reasoning_only_output():
    from vmlx_engine.server import _current_response_warnings_for_reasoning_only

    warnings = _current_response_warnings_for_reasoning_only(True)

    assert warnings is not None
    assert any("produced reasoning only" in w for w in warnings)
    assert _current_response_warnings_for_reasoning_only(False) is None


def test_empty_output_does_NOT_emit_placeholder():
    """An empty output_items list means the model truly produced nothing.
    Do not invent an assistant turn — that's different from reasoning-only."""
    from vmlx_engine.server import _responses_output_to_assistant_messages
    assert _responses_output_to_assistant_messages([]) == []


def test_visible_text_output_does_NOT_get_placeholder():
    """When output_text is present, replay normally (no placeholder injection)."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
        },
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "the answer is 42"}],
        },
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert msgs == [
        {"role": "assistant", "content": "the answer is 42"}
    ], "output_text should replay as assistant content; reasoning still dropped"


def test_function_call_only_does_NOT_get_placeholder():
    """function_call items already produce an assistant.tool_calls turn — no
    extra placeholder needed."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "call_id": "call_abc",
        }
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0].get("tool_calls"), "function_call must produce tool_calls turn"


def test_reasoning_plus_function_call_does_NOT_get_extra_placeholder():
    """Mixed reasoning + function_call: only the function_call assistant turn."""
    from vmlx_engine.server import _responses_output_to_assistant_messages

    items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]},
        {
            "type": "function_call",
            "name": "x",
            "arguments": "{}",
            "call_id": "c1",
        },
    ]
    msgs = _responses_output_to_assistant_messages(items)
    assert len(msgs) == 1
    assert msgs[0].get("tool_calls"), "function_call assistant turn already present"


def test_reasoning_only_history_round_trips_through_store():
    """Integration: store + load preserves the placeholder so chain coherence
    is end-to-end correct, not just at the converter."""
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_get_history,
        _responses_history,
    )

    _responses_history.clear()
    user_messages = [{"role": "user", "content": "what is 2+2?"}]
    output_items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]}
    ]
    full = user_messages + _responses_output_to_assistant_messages(output_items)
    _responses_store_history("resp_test_reasoning_only", full)

    loaded = _responses_get_history("resp_test_reasoning_only")
    assert loaded == [
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": ""},
    ], "stored history must preserve the assistant placeholder for chain anchoring"

    _responses_history.clear()


# ─── Option C: warning surface for reasoning-only chains ──────────────────
# When a `previous_response_id` chain references a response that produced only
# reasoning (no visible output, no tool calls), the chained response should
# expose a `warnings` array so clients can surface the coherence/cache-prefix
# risk without breaking the API contract.


def test_responses_object_supports_warnings_field():
    """ResponsesObject must allow a `warnings: list[str] | None` field."""
    from vmlx_engine.api.models import ResponsesObject
    obj = ResponsesObject(model="m", warnings=["test warning"])
    assert obj.warnings == ["test warning"]


def test_responses_object_warnings_default_none():
    from vmlx_engine.api.models import ResponsesObject
    obj = ResponsesObject(model="m")
    assert obj.warnings is None


def test_reasoning_only_marker_stored_per_response_id():
    """Storing a reasoning-only response must mark it so chained turns can warn."""
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    user_messages = [{"role": "user", "content": "hi"}]
    items = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]}
    ]
    full = user_messages + _responses_output_to_assistant_messages(items)
    _responses_store_history("resp_ro_marker", full, reasoning_only=True)

    assert "resp_ro_marker" in _responses_was_reasoning_only

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_visible_response_does_not_set_reasoning_only_marker():
    from vmlx_engine.server import (
        _responses_output_to_assistant_messages,
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    user_messages = [{"role": "user", "content": "hi"}]
    items = [{"type": "message", "content": [{"type": "output_text", "text": "yes"}]}]
    full = user_messages + _responses_output_to_assistant_messages(items)
    _responses_store_history("resp_normal", full, reasoning_only=False)

    assert "resp_normal" not in _responses_was_reasoning_only

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_chained_response_helper_emits_warning_for_reasoning_only_predecessor():
    """`_chain_warnings_for_previous_response_id` returns a warning when the
    predecessor was reasoning-only, else None."""
    from vmlx_engine.server import (
        _responses_history,
        _responses_store_history,
        _responses_was_reasoning_only,
        _chain_warnings_for_previous_response_id,
    )
    _responses_history.clear()
    _responses_was_reasoning_only.clear()
    _responses_store_history(
        "resp_ro_predecessor",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
        reasoning_only=True,
    )

    warnings = _chain_warnings_for_previous_response_id("resp_ro_predecessor")
    assert warnings is not None
    assert any("reasoning" in w.lower() for w in warnings)

    assert _chain_warnings_for_previous_response_id("resp_normal") is None
    assert _chain_warnings_for_previous_response_id(None) is None
    assert _chain_warnings_for_previous_response_id("") is None

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_chained_response_helper_ignores_stale_marker_without_history_slot():
    """A marker without a matching history slot must not warn.

    The runtime store path evicts both in lockstep, but tests and future admin
    cleanup paths may clear `_responses_history` directly. The warning helper
    should defend the invariant at read time too, otherwise a stale marker can
    create a false-positive warning for a missing predecessor.
    """
    from vmlx_engine.server import (
        _responses_history,
        _responses_was_reasoning_only,
        _chain_warnings_for_previous_response_id,
    )

    _responses_history.clear()
    _responses_was_reasoning_only.clear()
    _responses_was_reasoning_only.add("resp_stale_marker")

    assert _chain_warnings_for_previous_response_id("resp_stale_marker") is None

    _responses_history.clear()
    _responses_was_reasoning_only.clear()


def test_reasoning_only_marker_eviction_with_history_capacity():
    """When history slots are evicted by capacity, marker must also be evicted
    to avoid stale-warning false positives on response_id collisions."""
    from vmlx_engine.server import (
        _responses_store_history,
        _responses_history,
        _responses_was_reasoning_only,
    )
    import vmlx_engine.server as server
    _responses_history.clear()
    _responses_was_reasoning_only.clear()

    original_max = server._RESPONSES_HISTORY_MAX
    try:
        server._RESPONSES_HISTORY_MAX = 2
        _responses_store_history("resp_a", [{"role": "user", "content": "a"}], reasoning_only=True)
        _responses_store_history("resp_b", [{"role": "user", "content": "b"}], reasoning_only=False)
        _responses_store_history("resp_c", [{"role": "user", "content": "c"}], reasoning_only=False)
        # resp_a should be evicted
        assert "resp_a" not in _responses_history
        assert "resp_a" not in _responses_was_reasoning_only, (
            "evicted history slot must also drop its reasoning_only marker"
        )
    finally:
        server._RESPONSES_HISTORY_MAX = original_max
        _responses_history.clear()
        _responses_was_reasoning_only.clear()

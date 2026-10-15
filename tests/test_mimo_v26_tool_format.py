"""MiMo-V2.6 tool-call format round trip (live raw-API row 2026-10-15).

The vendor template renders a string ``arguments`` verbatim inside ``<function=NAME>...</function>`` and only a dict
as ``<parameter=KEY>VALUE</parameter>``. OpenAI histories carry JSON strings, so earlier calls were rendered as JSON
bodies, the model copied that form, and the XML function parser dropped the call as "missing required argument".
"""
from __future__ import annotations

import json

from vmlx_engine.models.mimo_v26_contract import canonicalize_mimo_v26_tool_results
from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser


def _history():
    return [
        {"role": "user", "content": "How many units of amber-207-Z?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "lookup_inventory", "arguments": '{"item": "amber-207-Z"}'}},
            {"id": "call_b", "type": "function", "function": {"name": "raw_tool", "arguments": "not json at all"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": '{"units": 88}'},
        {"role": "tool", "tool_call_id": "call_b", "content": "ok"},
    ]


def test_history_tool_call_json_arguments_become_dicts_for_the_parameter_form():
    out = canonicalize_mimo_v26_tool_results(_history())
    calls = out[1]["tool_calls"]
    assert calls[0]["function"]["arguments"] == {"item": "amber-207-Z"}      # rendered as <parameter=item>
    assert calls[1]["function"]["arguments"] == "not json at all"            # verbatim form kept
    assert _history()[1]["tool_calls"][0]["function"]["arguments"] == '{"item": "amber-207-Z"}'  # input not mutated


def test_parser_reads_the_template_json_body_form():
    parser = XMLFunctionToolParser(None)
    text = '<tool_call><function=lookup_inventory>{"item": "amber-7319-Z"}</function></tool_call>'
    result = parser.extract_tool_calls(text, {"tools": [{"type": "function", "function": {
        "name": "lookup_inventory", "parameters": {"type": "object", "properties": {"item": {"type": "string"}},
                                                   "required": ["item"]}}}]})
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"item": "amber-7319-Z"}


def test_parser_still_prefers_the_parameter_form():
    parser = XMLFunctionToolParser(None)
    text = "<tool_call><function=lookup_inventory><parameter=item>amber-1-Z</parameter></function></tool_call>"
    result = parser.extract_tool_calls(text, None)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"item": "amber-1-Z"}


def test_chat_entry_points_keep_the_gpu_awake():
    """SimpleEngine / BatchedEngine chat + stream_chat touch the GPU keep-alive (it was only touched by generate /
    stream_generate, so every chat turn after >= 1.5 s idle paid the 0.65-1.25 s wake penalty on SimpleEngine lanes:
    MiMo DFlash measured follow-up TTFT 1.03-1.37 s after 6 s idle vs 0.25 s back to back, allocator release on or off)."""
    import inspect

    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.engine.simple import SimpleEngine

    for cls in (SimpleEngine, BatchedEngine):
        for name in ("chat", "stream_chat"):
            assert "_gpu_keepalive_touch()" in inspect.getsource(getattr(cls, name)), (cls.__name__, name)

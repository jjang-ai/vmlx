# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MiniMax-M3 native XML tool parsing."""

import json

from vmlx_engine.tool_parsers.minimax_m3_tool_parser import (
    NS_TOKEN,
    MiniMaxM3ToolParser,
)


def test_terminally_truncated_namespace_before_tool_call_is_not_visible_content():
    """The observed one-character namespace truncation is control markup."""
    raw = (
        f"{NS_TOKEN[:-1]}<tool_call>\n"
        f'{NS_TOKEN}<invoke name="file_info">\n'
        f"{NS_TOKEN}<path>panel/package.json{NS_TOKEN}</path>\n"
        f"{NS_TOKEN}</invoke>\n"
        "</tool_call>"
    )

    result = MiniMaxM3ToolParser().extract_tool_calls(raw)

    assert result.tools_called is True
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "file_info"
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "path": "panel/package.json"
    }


def test_namespace_cleanup_does_not_rewrite_normal_minimax_prose():
    text = "MiniMax remains visible, as does plain minimax prose."

    result = MiniMaxM3ToolParser().extract_tool_calls(text)

    assert result.tools_called is False
    assert result.content == text


def test_malformed_native_call_keeps_only_preceding_prose():
    raw = (
        'Checking now.\n]<]minimax[>[<tool_call>name="run_command">\n'
        '<parameter name="command">python3 -c "print(866525 % 97)"</parameter>\n'
        '</invoke>\n{"status": "success", "output": "24"}'
    )
    result = MiniMaxM3ToolParser().extract_tool_calls(raw)
    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == "Checking now."


def test_complete_native_call_stops_before_fabricated_result():
    parser = MiniMaxM3ToolParser()
    raw = ('Checking now.\n<tool_call><invoke name="run_command">'
           '<command>echo alpha</command></invoke></tool_call>')
    tail = '\nThe command returned alpha.'
    assert parser.STREAM_STOPS_AFTER_COMPLETE_CALL
    assert parser.stream_tool_calls_complete(raw + tail)
    assert parser.stream_tool_call_stop_truncate(raw + tail) == raw


def test_open_sibling_native_call_is_not_a_completed_turn():
    parser = MiniMaxM3ToolParser()
    raw = ('<tool_call><invoke name="run_command"><command>echo alpha</command>'
           '</invoke></tool_call><tool_call><invoke name="run_command">')
    assert not parser.stream_tool_calls_complete(raw)


def test_complete_call_extracts_only_prose_before_control_handoff():
    raw = ('Checking now.\n<tool_call><invoke name="run_command">'
           '<command>echo alpha</command></invoke></tool_call>'
           '\nThe command returned alpha.')
    result = MiniMaxM3ToolParser().extract_tool_calls(raw)
    assert result.tools_called
    assert result.content == "Checking now."


def test_native_stop_requires_advertised_schema_valid_call():
    parser = MiniMaxM3ToolParser()
    parser._stream_stop_request = {"tools": [{"type": "function", "function": {
        "name": "run_command", "parameters": {"type": "object",
        "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }}]}
    raw = '<tool_call><invoke name="run_command"><command>echo alpha</command></invoke></tool_call>'
    assert parser.stream_tool_calls_complete(raw)
    assert not parser.stream_tool_calls_complete(raw.replace('run_command', 'not_offered'))
    assert not parser.stream_tool_calls_complete(raw.replace('echo alpha', '42'))
    assert not parser.stream_tool_calls_complete(raw + '<tool_ca')
    assert not parser.stream_tool_calls_complete(raw + NS_TOKEN[:5])


def test_native_stop_preserves_multiple_complete_calls():
    first = '<tool_call><invoke name="one"></invoke></tool_call>'
    second = '<tool_call><invoke name="two"></invoke></tool_call>'
    parser = MiniMaxM3ToolParser()
    assert parser.stream_tool_calls_complete(first + second)
    assert parser.stream_tool_call_stop_truncate(first + second + 'done') == first + second

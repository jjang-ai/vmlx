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

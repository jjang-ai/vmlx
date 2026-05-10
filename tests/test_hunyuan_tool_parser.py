# SPDX-License-Identifier: Apache-2.0
"""Hunyuan / Tencent Hy3 tool-call parser tests.

Contract source:
    ~/jang/jang-tools/examples/hy3/python_runtime/hy3_parser_contract.py
    ~/jang/docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md
"""

import json

import pytest

from vmlx_engine.tool_parsers.hunyuan_tool_parser import HunyuanToolParser


@pytest.fixture
def parser():
    return HunyuanToolParser(tokenizer=None)


class TestHunyuanToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_single_tool_call_with_string_arg(self, parser):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>Paris</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        assert out.content is None  # all text was tool envelope

    def test_multiple_args_with_type_coercion(self, parser):
        """Per contract: values try json.loads first, fall back to string."""
        text = (
            "<tool_calls>\n"
            "<tool_call>set_temperature<tool_sep>\n"
            "<arg_key>celsius</arg_key>\n"
            "<arg_value>22.5</arg_value>\n"
            "<arg_key>auto_adjust</arg_key>\n"
            "<arg_value>true</arg_value>\n"
            "<arg_key>label</arg_key>\n"
            "<arg_value>kitchen</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        out = parser.extract_tool_calls(text)
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"celsius": 22.5, "auto_adjust": True, "label": "kitchen"}

    def test_multiple_tool_calls_in_one_envelope(self, parser):
        text = (
            "<tool_calls>\n"
            "<tool_call>fn_a<tool_sep>\n"
            "<arg_key>x</arg_key><arg_value>1</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>fn_b<tool_sep>\n"
            "<arg_key>y</arg_key><arg_value>2</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0]["name"] == "fn_a"
        assert out.tool_calls[1]["name"] == "fn_b"
        assert json.loads(out.tool_calls[0]["arguments"]) == {"x": 1}
        assert json.loads(out.tool_calls[1]["arguments"]) == {"y": 2}

    def test_visible_text_before_and_after_tool_calls_preserved(self, parser):
        text = (
            "Sure, calling the function now.\n"
            "<tool_calls>\n"
            "<tool_call>ping<tool_sep>\n"
            "<arg_key>host</arg_key><arg_value>example.com</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>\n"
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        # Surrounding visible text retained
        assert out.content is not None
        assert "Sure, calling the function now." in out.content
        assert "Done." in out.content
        assert "<tool_calls>" not in out.content
        assert "<tool_call>" not in out.content
        assert "<arg_key>" not in out.content

    def test_no_tool_sep_in_call_is_skipped(self, parser):
        """Malformed call without <tool_sep> is silently dropped (defensive)."""
        text = (
            "<tool_calls>\n"
            "<tool_call>just_text_no_separator</tool_call>\n"
            "</tool_calls>"
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False  # nothing extractable
        assert out.tool_calls == []

    def test_streaming_yields_content_until_envelope_close(self, parser):
        """Streaming should pass-through deltas until the closing tag arrives,
        then emit consolidated tool_calls."""
        prev = ""
        cur = "Hello"
        delta = "Hello"
        result = parser.extract_tool_calls_streaming(prev, cur, delta)
        assert result == {"content": "Hello"}

        # Open envelope — still pass-through (no close yet in delta)
        prev = "Hello"
        cur = "Hello\n<tool_calls>\n<tool_call>fn<tool_sep>"
        delta = "\n<tool_calls>\n<tool_call>fn<tool_sep>"
        result = parser.extract_tool_calls_streaming(prev, cur, delta)
        # Once the envelope is open we stop emitting visible text deltas
        assert result is None

        # Close arrives in delta — emit tool_calls payload
        prev = cur
        cur = (
            "Hello\n<tool_calls>\n<tool_call>fn<tool_sep>\n"
            "<arg_key>k</arg_key><arg_value>v</arg_value>\n"
            "</tool_call>\n</tool_calls>"
        )
        delta = (
            "\n<arg_key>k</arg_key><arg_value>v</arg_value>\n"
            "</tool_call>\n</tool_calls>"
        )
        result = parser.extract_tool_calls_streaming(prev, cur, delta)
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "fn"

    def test_registry_aliases_resolve(self):
        """Verify the parser registers under all 3 aliases."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("hunyuan", "hy_v3", "tencent"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is HunyuanToolParser, (
                f"alias {alias!r} should resolve to HunyuanToolParser, got {cls}"
            )

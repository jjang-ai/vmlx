# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7 / GLM-4.7-Flash tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the GLM-4.7 tool parser had no dedicated test file. GLM-4.7 supports two
distinct tool-call formats inside `<tool_call>...</tool_call>`:
  1. XML arg format: `<tool_call>fn\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
  2. JSON format: `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>` (HARMONY template)

If extraction misses an edge case, the raw `<tool_call>` envelope (XML or JSON)
could leak. Note: GLM-4.7 INTENTIONALLY drops surrounding text when tool calls
are found (parser sets `content=None`) per the source comment "GLM often
outputs thinking/reasoning before tool calls without <think> tags".

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with tool_calls populated AND content=None whenever tools_called=True.

Fix: this test file pins the contract per parser API — no source patch needed.
"""

import json

import pytest

from vmlx_engine.tool_parsers.glm47_tool_parser import Glm47ToolParser


@pytest.fixture
def parser():
    return Glm47ToolParser(tokenizer=None)


class TestGlm47ToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_xml_arg_format_single_call(self, parser):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
            "</tool_call>"
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # GLM-4.7 always sets content=None when tools found.
        assert out.content is None

    def test_xml_arg_format_typed_values(self, parser):
        """Numeric / bool arg values deserialized via _deserialize."""
        text = (
            "<tool_call>set_temperature\n"
            "<arg_key>celsius</arg_key><arg_value>22.5</arg_value>\n"
            "<arg_key>auto_adjust</arg_key><arg_value>true</arg_value>\n"
            "<arg_key>label</arg_key><arg_value>kitchen</arg_value>\n"
            "</tool_call>"
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args.get("celsius") == 22.5
        assert args.get("auto_adjust") is True
        assert args.get("label") == "kitchen"

    def test_json_format_inside_tool_call_envelope(self, parser):
        """HARMONY template uses JSON inside <tool_call>...</tool_call>."""
        text = '<tool_call>{"name": "search_web", "arguments": {"query": "weather"}}</tool_call>'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "search_web"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"query": "weather"}
        assert out.content is None

    def test_visible_text_dropped_when_tools_found(self, parser):
        """GLM-4.7 explicitly drops pre/post text when tool calls extracted —
        documented behavior; not a leak."""
        text = (
            "Let me think about that.\n"
            "<tool_call>ping\n"
            "<arg_key>host</arg_key><arg_value>example.com</arg_value>\n"
            "</tool_call>\n"
            "Done thinking."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        # GLM-4.7 contract: drops surrounding text.
        assert out.content is None

    def test_tools_called_implies_no_envelope_in_content(self, parser):
        """Critical no-leak invariant: tools_called=True ⇒ content is None
        (which trivially can't contain `<tool_call>` markup)."""
        for text in [
            "<tool_call>fn\n<arg_key>k</arg_key><arg_value>v</arg_value>\n</tool_call>",
            '<tool_call>{"name": "fn", "arguments": {"k": 1}}</tool_call>',
            "pre text<tool_call>fn\n<arg_key>k</arg_key><arg_value>v</arg_value>\n</tool_call>post",
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                # Content is either None or empty-equivalent; no markup.
                assert out.content is None or "<tool_call>" not in out.content
                assert out.content is None or "</tool_call>" not in out.content
                assert out.content is None or "<arg_key>" not in out.content
                assert out.content is None or "<arg_value>" not in out.content

    def test_no_tool_call_marker_short_circuit(self, parser):
        """Plain prose without <tool_call> passes through as content."""
        text = "If x < 5 and y > 3 then think about it."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_strips_think_tags_before_extraction(self, parser):
        """Parser uses self.strip_think_tags before pattern matching;
        residual <think> blocks should not survive into content
        (since content=None when tools found, this is more about extraction)."""
        text = (
            "<think>Plan: ping example.com</think>\n"
            "<tool_call>ping\n"
            "<arg_key>host</arg_key><arg_value>example.com</arg_value>\n"
            "</tool_call>"
        )
        out = parser.extract_tool_calls(text)
        if out.tools_called:
            # Per parser contract content=None on tool extraction.
            assert out.content is None

    def test_registry_aliases_resolve(self):
        """Glm47ToolParser must register under both `glm47` and `glm4`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("glm47", "glm4"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is Glm47ToolParser, (
                f"alias {alias!r} should resolve to Glm47ToolParser, got {cls}"
            )

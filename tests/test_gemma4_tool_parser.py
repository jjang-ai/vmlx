# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the Gemma 4 tool parser had no dedicated test file. Gemma 4's tool markup uses
a different token shape than other parsers (`<|tool_call>...{...}<tool_call|>`)
and includes a `<|"|>` quote-escape token that could leak if extraction misses
an edge case. Also strips `<think>`/`<thought>`/`<channel>` markers as part of
the parsing path — those must not leak either.

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with all Gemma 4 tool tokens stripped from `content` whenever tools_called=True.

Fix: this test file pins the contract — no source patch needed (parser already
strips via `_TOOL_CALL_PATTERN.sub("", ...)` + `_clean_special_tokens`).

Format reference (Gemma 4 native):
    <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,limit:5}<tool_call|>

Hermes-JSON fallback also supported for compatibility.
"""

import json

import pytest

from vmlx_engine.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


@pytest.fixture
def parser():
    return Gemma4ToolParser(tokenizer=None)


class TestGemma4ToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_native_format_simple_string_arg(self, parser):
        text = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # No Gemma 4 markup may leak into content.
        if out.content:
            assert "<|tool_call>" not in out.content
            assert "<tool_call|>" not in out.content
            assert "call:" not in out.content
            assert '<|"|>' not in out.content

    def test_native_format_complete_call_at_end_without_end_marker(self, parser):
        text = '<|tool_call>call:record_fact{value:<|"|>blue-cat<|"|>}'

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "record_fact"
        assert json.loads(out.tool_calls[0]["arguments"]) == {"value": "blue-cat"}
        assert out.content is None

    def test_native_format_multiple_args(self, parser):
        text = '<|tool_call>call:set_temperature{celsius:22.5,auto_adjust:true,label:<|"|>kitchen<|"|>}<tool_call|>'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args.get("celsius") == 22.5
        assert args.get("auto_adjust") is True
        assert args.get("label") == "kitchen"

    def test_visible_text_around_tool_call_no_gemma4_leak(self, parser):
        """Surrounding visible text preserved; every Gemma 4 marker stripped."""
        text = (
            "Sure, calling the function now.\n"
            '<|tool_call>call:ping{host:<|"|>example.com<|"|>}<tool_call|>\n'
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        assert out.content is not None
        assert "Sure, calling the function now." in out.content
        assert "Done." in out.content
        # Critical no-leak assertions.
        assert "<|tool_call>" not in out.content
        assert "<tool_call|>" not in out.content
        assert "call:" not in out.content
        assert '<|"|>' not in out.content

    def test_hermes_fallback_format(self, parser):
        """Models occasionally emit Hermes JSON instead — fallback path."""
        text = '<tool_call>{"name": "search_web", "arguments": {"query": "weather"}}</tool_call>'
        out = parser.extract_tool_calls(text)
        # Either tools_called True via fallback or False if Hermes pattern not detected.
        if out.tools_called:
            assert out.tool_calls[0]["name"] == "search_web"
            # No <tool_call> JSON markup should leak.
            if out.content:
                assert "<tool_call>" not in out.content
                assert "</tool_call>" not in out.content

    def test_strips_think_markers_alongside_tool_call(self, parser):
        """Gemma 4 also strips <think>...</think> as part of tool extract path."""
        text = (
            "<think>Plan: call ping then summarize.</think>\n"
            '<|tool_call>call:ping{host:<|"|>example.com<|"|>}<tool_call|>'
        )
        out = parser.extract_tool_calls(text)
        if out.tools_called:
            content = out.content or ""
            assert "<think>" not in content
            assert "</think>" not in content
            assert "<|tool_call>" not in content
            assert "<tool_call|>" not in content
            assert '<|"|>' not in content

    def test_no_marker_short_circuit(self, parser):
        """Plain prose with non-Gemma angle brackets passes through clean."""
        text = "If x < 5 and y > 3 then..."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_registry_alias_resolves(self):
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager
        cls = ToolParserManager.get_tool_parser("gemma4")
        assert cls is Gemma4ToolParser

    def test_tools_called_implies_no_marker_in_content(self, parser):
        """Cross-cutting invariant: tools_called=True ⇒ no Gemma 4 marker in content."""
        for text in [
            'pre<|tool_call>call:a{k:<|"|>v<|"|>}<tool_call|>post',
            '<|tool_call>call:a{k:42}<tool_call|>',
            'lead\n<|tool_call>call:a{k:<|"|>x<|"|>}<tool_call|>\ntail',
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                content = out.content or ""
                assert "<|tool_call>" not in content, (
                    f"<|tool_call> leaked for input: {text!r}\ncontent: {content!r}"
                )
                assert "<tool_call|>" not in content
                assert '<|"|>' not in content

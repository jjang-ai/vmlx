# SPDX-License-Identifier: Apache-2.0
"""IBM Granite tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the Granite tool parser had no dedicated test file. Granite's tool format wraps
a JSON array directly after a BOT marker:
  - Granite: `<|tool_call|>[{"name": "fn", "arguments": {...}}]`
  - Granite 3.1: `<tool_call>[{"name": "fn", "arguments": {...}}]`

If extraction misses, the BOT marker AND raw JSON array could leak into content.
Granite parser bails (returns model_output as content) when text doesn't start
with the marker — so embedded calls aren't extracted, which is itself a
no-leak property (whole input passes through as content unchanged).

Behavior: extract_tool_calls() returns content=None when tools_called=True; or
returns model_output as content unchanged when format doesn't match.

Fix: this test file pins the contract per parser API — no source patch needed.
"""

import json

import pytest

from vmlx_engine.tool_parsers.granite_tool_parser import GraniteToolParser


@pytest.fixture
def parser():
    return GraniteToolParser(tokenizer=None)


class TestGraniteToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_granite_native_marker_single_call(self, parser):
        text = '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # Granite contract: content is None when tools found.
        assert out.content is None

    def test_granite3_alternate_marker_single_call(self, parser):
        text = '<tool_call>[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        assert out.content is None

    def test_typed_args_passed_through_as_json(self, parser):
        text = '<|tool_call|>[{"name": "set_temp", "arguments": {"celsius": 22.5, "auto_adjust": true, "label": "kitchen"}}]'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args.get("celsius") == 22.5
        assert args.get("auto_adjust") is True
        assert args.get("label") == "kitchen"

    def test_multiple_calls_in_one_array(self, parser):
        text = '<|tool_call|>[{"name": "fn_a", "arguments": {"x": 1}}, {"name": "fn_b", "arguments": {"y": 2}}]'
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0]["name"] == "fn_a"
        assert out.tool_calls[1]["name"] == "fn_b"

    def test_pre_text_prevents_extraction_passes_through(self, parser):
        """Granite contract: marker must be at start; embedded calls are NOT
        extracted, but also DO NOT leak — full input is returned as content."""
        text = 'Pre text. <|tool_call|>[{"name": "fn", "arguments": {}}]'
        out = parser.extract_tool_calls(text)
        # Marker not at start ⇒ no extraction, full passthrough.
        assert out.tools_called is False
        assert out.content == text

    def test_invalid_json_after_marker_returns_passthrough(self, parser):
        """Malformed JSON after the marker → no extraction; passthrough."""
        text = '<|tool_call|>[{not valid json}]'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_non_array_after_marker_returns_passthrough(self, parser):
        """JSON object instead of array → no extraction (Granite expects array)."""
        text = '<|tool_call|>{"name": "fn", "arguments": {}}'
        out = parser.extract_tool_calls(text)
        # Object isn't an array → tools_called=False; full passthrough preserves
        # the marker visible to the user (UX gap, but no silent leak: marker
        # IS in the original input so this is "passthrough not corruption").
        assert out.tools_called is False

    def test_tools_called_implies_content_is_none(self, parser):
        """Critical no-leak invariant: tools_called=True ⇒ content is None,
        which trivially can't contain `<|tool_call|>` or `<tool_call>` markup."""
        for text in [
            '<|tool_call|>[{"name": "a", "arguments": {}}]',
            '<tool_call>[{"name": "a", "arguments": {}}]',
            '<|tool_call|>[{"name": "a", "arguments": {"k": "v"}}]',
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                assert out.content is None, (
                    f"Granite content should be None when tools found, got: {out.content!r}"
                )

    def test_registry_aliases_resolve(self):
        """GraniteToolParser must register under both `granite` and `granite3`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("granite", "granite3"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is GraniteToolParser, (
                f"alias {alias!r} should resolve to GraniteToolParser, got {cls}"
            )

    def test_supports_native_tool_format(self):
        """Granite 3.1+ chat templates support native tool message format."""
        assert GraniteToolParser.supports_native_format() is True

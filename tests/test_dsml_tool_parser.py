# SPDX-License-Identifier: Apache-2.0
"""DSV4 DSML tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the DSML tool parser (used by deepseek_v4 / DSV4 Flash) had no dedicated test
file. Tool XML markup (`<｜DSML｜invoke>`, `<｜DSML｜parameter>`) could silently
leak into the user-visible `content` if any extraction path regressed.

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with all DSML markup stripped from `content` whenever tools_called=True.

Fix: this test file pins the contract per parser API — no source patch needed
(parser already strips correctly via `_INVOKE_RE.sub("", model_output).strip()`).

Format reference (DSML / DeepSeek Markup Language):
    <｜DSML｜invoke name="search_web">
    <｜DSML｜parameter name="query" string="true">weather in LA</｜DSML｜parameter>
    <｜DSML｜parameter name="limit" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>

DSML delimiter is fullwidth vertical bar `｜` (U+FF5C), same character class as
DeepSeek's other special tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`,
`<｜Assistant｜>`).
"""

import json

import pytest

from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser, DSML_PREFIX


@pytest.fixture
def parser():
    return DSMLToolParser(tokenizer=None)


class TestDSMLToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        """Plain text without DSML markup must pass through verbatim."""
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_single_invoke_with_string_param(self, parser):
        """Single invoke with one string="true" parameter."""
        text = (
            f'<{DSML_PREFIX}invoke name="get_weather">\n'
            f'<{DSML_PREFIX}parameter name="city" string="true">Paris</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # No DSML markup may leak into content.
        assert out.content is None or DSML_PREFIX not in out.content
        assert out.content is None or "<" + DSML_PREFIX not in out.content

    def test_invoke_with_typed_params_string_false_decodes_json(self, parser):
        """string="false" parameters parse as JSON (numbers, bools, arrays)."""
        text = (
            f'<{DSML_PREFIX}invoke name="set_temperature">\n'
            f'<{DSML_PREFIX}parameter name="celsius" string="false">22.5</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="auto_adjust" string="false">true</{DSML_PREFIX}parameter>\n'
            f'<{DSML_PREFIX}parameter name="label" string="true">kitchen</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"celsius": 22.5, "auto_adjust": True, "label": "kitchen"}

    def test_multiple_invokes_in_one_completion(self, parser):
        """Two invoke blocks back-to-back must each become their own tool call."""
        text = (
            f'<{DSML_PREFIX}invoke name="fn_a">\n'
            f'<{DSML_PREFIX}parameter name="x" string="false">1</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            f'<{DSML_PREFIX}invoke name="fn_b">\n'
            f'<{DSML_PREFIX}parameter name="y" string="false">2</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0]["name"] == "fn_a"
        assert out.tool_calls[1]["name"] == "fn_b"
        assert json.loads(out.tool_calls[0]["arguments"]) == {"x": 1}
        assert json.loads(out.tool_calls[1]["arguments"]) == {"y": 2}

    def test_visible_text_around_invoke_preserved_no_dsml_leak(self, parser):
        """Surrounding visible text stays in content; DSML markup must not leak."""
        text = (
            "Sure, calling the function now.\n"
            f'<{DSML_PREFIX}invoke name="ping">\n'
            f'<{DSML_PREFIX}parameter name="host" string="true">example.com</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>\n'
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        assert out.content is not None
        assert "Sure, calling the function now." in out.content
        assert "Done." in out.content
        # Critical no-leak assertions — every DSML token must be stripped.
        assert "<" + DSML_PREFIX + "invoke" not in out.content
        assert "</" + DSML_PREFIX + "invoke>" not in out.content
        assert "<" + DSML_PREFIX + "parameter" not in out.content
        assert "</" + DSML_PREFIX + "parameter>" not in out.content
        assert DSML_PREFIX not in out.content

    def test_dsml_only_completion_returns_none_content(self, parser):
        """When the whole completion is DSML markup, content should be None."""
        text = (
            f'<{DSML_PREFIX}invoke name="fn">\n'
            f'<{DSML_PREFIX}parameter name="k" string="true">v</{DSML_PREFIX}parameter>\n'
            f'</{DSML_PREFIX}invoke>'
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        # All text was tool envelope — content must be None or empty/whitespace.
        assert out.content is None or out.content.strip() == ""

    def test_no_dsml_markers_short_circuit_returns_content(self, parser):
        """Short-circuit: text without DSML prefix or `<invoke_` returns content."""
        # Common prose-only chat with non-DSML angle brackets (e.g. inequalities).
        text = "If x < 5 and y > 3 then..."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_registry_aliases_resolve(self):
        """DSMLToolParser must register under both `dsml` and `deepseek_v4`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("dsml", "deepseek_v4"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is DSMLToolParser, (
                f"alias {alias!r} should resolve to DSMLToolParser, got {cls}"
            )

    def test_supports_native_tool_format(self):
        """DSMLToolParser exposes supports_native_format()=True so server.py
        keeps the raw template's tool affordances rather than injecting the
        generic JSON-tools instructions block."""
        assert DSMLToolParser.supports_native_format() is True

    def test_tools_called_implies_no_dsml_in_content(self, parser):
        """Cross-cutting invariant: whenever tools_called=True the visible
        content field must be free of every DSML token form. Regression guard
        per AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md Finding T1."""
        for text in [
            f'pre<{DSML_PREFIX}invoke name="a"><{DSML_PREFIX}parameter name="k" string="true">v</{DSML_PREFIX}parameter></{DSML_PREFIX}invoke>post',
            f'<{DSML_PREFIX}invoke name="a"><{DSML_PREFIX}parameter name="k" string="false">42</{DSML_PREFIX}parameter></{DSML_PREFIX}invoke>',
            f'lead\n<{DSML_PREFIX}invoke name="a">\n<{DSML_PREFIX}parameter name="k" string="true">x</{DSML_PREFIX}parameter>\n</{DSML_PREFIX}invoke>\ntail',
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                content = out.content or ""
                assert DSML_PREFIX not in content, (
                    f"DSML token leaked into content for input: {text!r}\n"
                    f"content: {content!r}"
                )
                assert "<" + DSML_PREFIX not in content
                assert "</" + DSML_PREFIX not in content

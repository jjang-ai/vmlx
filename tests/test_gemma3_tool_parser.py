# SPDX-License-Identifier: Apache-2.0
"""Gemma 3 / Gemma 3n tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the Gemma 3 tool parser had no dedicated test file. Google's documented
function-calling format for Gemma 3 / 3n uses ` ```tool_code ` markdown blocks
with Python-style calls. If extraction misses an edge case, the raw markdown
block (including triple backticks + `tool_code` language tag) could leak into
the user-visible content.

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with the tool_code markdown block stripped from `content` whenever
tools_called=True.

Fix: this test file pins the contract per parser API — no source patch needed
(parser already strips via `_TOOL_CODE_BLOCK.sub("", ...)`).

Format reference (Gemma 3 / 3n native):
    ```tool_code
    get_weather(city="Paris")
    ```
"""

import json

import pytest

from vmlx_engine.tool_parsers.gemma3_tool_parser import Gemma3ToolParser


@pytest.fixture
def parser():
    return Gemma3ToolParser(tokenizer=None)


class TestGemma3ToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_single_tool_code_block_string_arg(self, parser):
        text = '```tool_code\nget_weather(city="Paris")\n```'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # Strict no-leak: the markdown fence + language tag must be stripped.
        if out.content:
            assert "```tool_code" not in out.content
            assert "```" not in out.content
            assert "get_weather(" not in out.content

    def test_typed_args_python_literal(self, parser):
        """Numeric, boolean, None args parsed via ast.literal_eval (safe — no code execution)."""
        text = '```tool_code\nset_temperature(celsius=22.5, auto_adjust=True, label="kitchen")\n```'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args.get("celsius") == 22.5
        assert args.get("auto_adjust") is True
        assert args.get("label") == "kitchen"

    def test_multiple_tool_code_blocks(self, parser):
        text = (
            '```tool_code\n'
            'fn_a(x=1)\n'
            '```\n'
            '```tool_code\n'
            'fn_b(y=2)\n'
            '```'
        )
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        names = [c["name"] for c in out.tool_calls]
        assert names == ["fn_a", "fn_b"]
        assert json.loads(out.tool_calls[0]["arguments"]) == {"x": 1}
        assert json.loads(out.tool_calls[1]["arguments"]) == {"y": 2}

    def test_visible_text_around_tool_code_no_markdown_leak(self, parser):
        text = (
            "Sure, calling the function now.\n"
            "```tool_code\n"
            'ping(host="example.com")\n'
            "```\n"
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        assert out.content is not None
        assert "Sure, calling the function now." in out.content
        assert "Done." in out.content
        # Critical no-leak.
        assert "```tool_code" not in out.content
        assert "```" not in out.content
        assert "ping(host=" not in out.content

    def test_no_tool_code_block_short_circuit(self, parser):
        """Plain prose without ```tool_code passes through verbatim."""
        text = "Here is some prose with backticks like `code` inline."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_empty_tool_code_block_falls_back_to_no_tool(self, parser):
        """Empty tool_code block must NOT extract a phantom tool call."""
        text = '```tool_code\n\n```'
        out = parser.extract_tool_calls(text)
        # No callable line ⇒ tools_called should be False; content stays as input.
        assert out.tools_called is False

    def test_registry_aliases_resolve(self):
        """Gemma3ToolParser must register under both `gemma3` and `gemma3n`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("gemma3", "gemma3n"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is Gemma3ToolParser, (
                f"alias {alias!r} should resolve to Gemma3ToolParser, got {cls}"
            )

    def test_tools_called_implies_no_markdown_in_content(self, parser):
        """Cross-cutting invariant: tools_called=True ⇒ no tool_code fence in content."""
        for text in [
            'pre\n```tool_code\nfn(k="v")\n```\npost',
            '```tool_code\nfn(k=42)\n```',
            'lead```tool_code\nfn(k="x")\n```tail',
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                content = out.content or ""
                assert "```tool_code" not in content, (
                    f"```tool_code leaked for input: {text!r}\ncontent: {content!r}"
                )
                assert "```" not in content

# SPDX-License-Identifier: Apache-2.0
"""StepFun Step-3.5 tool parser tests — no-leak coverage.

Issue: per `docs/internal/AUDIT_2026_05_10_PARSER_NO_LEAK_COVERAGE.md` Finding T1,
the Step-3.5 tool parser had no dedicated test file. Step-3.5 uses XML-style
tool calls similar to Nemotron, with optional schema-driven parameter type
coercion:
    <tool_call>
    <function=get_weather>
    <parameter=city>Paris</parameter>
    </function>
    </tool_call>

Also accepts JSON args inside `<function=name>{...}</function>`.

If extraction misses an edge case, the XML envelope (`<tool_call>`,
`<function=...>`, `<parameter=...>`) could leak into user-visible content.

Behavior: extract_tool_calls() must produce a clean ExtractedToolCallInformation
with all Step-3.5 XML markup stripped from `content` whenever tools_called=True.

Fix: this test file pins the contract per parser API — no source patch needed
(parser already strips via `TOOL_CALL_PATTERN.sub("", ...)`).
"""

import json

import pytest

from vmlx_engine.tool_parsers.step3p5_tool_parser import Step3p5ToolParser


@pytest.fixture
def parser():
    return Step3p5ToolParser(tokenizer=None)


class TestStep3p5ToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("Hello world, no tools here.")
        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "Hello world, no tools here."

    def test_xml_format_single_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"city": "Paris"}
        # No leak.
        if out.content:
            assert "<tool_call>" not in out.content
            assert "<function=" not in out.content
            assert "<parameter=" not in out.content

    def test_json_format_inside_function_tag(self, parser):
        """Step-3.5 also accepts raw JSON inside <function=name>{...}</function>."""
        text = '<tool_call><function=search_web>{"query": "weather"}</function></tool_call>'
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "search_web"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"query": "weather"}
        if out.content:
            assert "<tool_call>" not in out.content
            assert "<function=" not in out.content

    def test_json_string_argument_trims_only_wrapper_newlines_with_schema(self, parser):
        text = '<tool_call><function=record_fact>{"value": "\\nblue-cat\\n"}</function></tool_call>'
        request = {
            "tools": [{
                "function": {
                    "name": "record_fact",
                    "parameters": {
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            }],
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_typed_args_with_schema_coercion(self, parser):
        """Schema in request triggers numeric coercion for string parameters."""
        text = (
            "<tool_call>\n"
            "<function=set_temperature>\n"
            "<parameter=celsius>22.5</parameter>\n"
            "<parameter=auto_adjust>true</parameter>\n"
            "<parameter=label>kitchen</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = {
            "tools": [{
                "function": {
                    "name": "set_temperature",
                    "parameters": {
                        "properties": {
                            "celsius": {"type": "number"},
                            "auto_adjust": {"type": "boolean"},
                            "label": {"type": "string"},
                        },
                    },
                },
            }],
        }
        out = parser.extract_tool_calls(text, request=request)
        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args.get("celsius") == 22.5
        assert args.get("auto_adjust") is True
        assert args.get("label") == "kitchen"

    def test_string_parameter_trims_only_xml_wrapper_newlines(self, parser):
        text = (
            "<tool_call>\n"
            "<function=record_fact>\n"
            "<parameter=value>\nblue-cat\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = {
            "tools": [{
                "function": {
                    "name": "record_fact",
                    "parameters": {
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            }],
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_string_parameter_trims_wrapper_newlines_with_flat_tool_schema(self, parser):
        text = (
            "<tool_call>\n"
            "<function=record_fact>\n"
            "<parameter=value>\nblue-cat\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = {
            "tools": [{
                "type": "function",
                "name": "record_fact",
                "parameters": {
                    "properties": {
                        "value": {"type": "string"},
                    },
                },
            }],
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_string_parameter_preserves_same_line_spaces_and_multiline_payload(self, parser):
        request = {
            "tools": [{
                "function": {
                    "name": "write_note",
                    "parameters": {
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            }],
        }

        same_line = parser.extract_tool_calls(
            "<tool_call><function=write_note><parameter=value>  blue-cat  </parameter></function></tool_call>",
            request=request,
        )
        multiline = parser.extract_tool_calls(
            "<tool_call><function=write_note><parameter=value>line1\nline2</parameter></function></tool_call>",
            request=request,
        )

        assert json.loads(same_line.tool_calls[0]["arguments"]) == {
            "value": "  blue-cat  "
        }
        assert json.loads(multiline.tool_calls[0]["arguments"]) == {
            "value": "line1\nline2"
        }

    def test_visible_text_around_tool_call_no_xml_leak(self, parser):
        text = (
            "Sure, calling now.\n"
            "<tool_call>\n"
            "<function=ping>\n"
            "<parameter=host>example.com</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "Done."
        )
        out = parser.extract_tool_calls(text)
        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "ping"
        assert out.content is not None
        assert "Sure, calling now." in out.content
        assert "Done." in out.content
        # Critical no-leak.
        assert "<tool_call>" not in out.content
        assert "</tool_call>" not in out.content
        assert "<function=" not in out.content
        assert "<parameter=" not in out.content

    def test_strips_think_tags_before_extraction(self, parser):
        """Parser uses self.strip_think_tags before pattern matching."""
        text = (
            "<think>Plan: ping example.com</think>\n"
            "<tool_call>\n"
            "<function=ping>\n"
            "<parameter=host>example.com</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        out = parser.extract_tool_calls(text)
        if out.tools_called:
            content = out.content or ""
            assert "<think>" not in content
            assert "</think>" not in content
            assert "<tool_call>" not in content

    def test_no_tool_call_marker_short_circuit(self, parser):
        text = "If x < 5 then..."
        out = parser.extract_tool_calls(text)
        assert out.tools_called is False
        assert out.content == text

    def test_multiple_calls_in_one_completion(self, parser):
        text = (
            "<tool_call><function=fn_a><parameter=x>1</parameter></function></tool_call>"
            "<tool_call><function=fn_b><parameter=y>2</parameter></function></tool_call>"
        )
        out = parser.extract_tool_calls(text)
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0]["name"] == "fn_a"
        assert out.tool_calls[1]["name"] == "fn_b"

    def test_empty_function_with_required_schema_fails_closed(self, parser):
        text = "<tool_call><function=exec_command></function></tool_call>"
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ]
        }

        out = parser.extract_tool_calls(text, request=request)

        assert out.tools_called is False
        assert out.tool_calls == []

    def test_registry_aliases_resolve(self):
        """Step3p5ToolParser must register under both `step3p5` and `stepfun`."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager

        for alias in ("step3p5", "stepfun"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is Step3p5ToolParser, (
                f"alias {alias!r} should resolve to Step3p5ToolParser, got {cls}"
            )

    def test_tools_called_implies_no_xml_in_content(self, parser):
        """Cross-cutting invariant: tools_called=True ⇒ no Step-3.5 XML in content."""
        for text in [
            "pre<tool_call><function=a><parameter=k>v</parameter></function></tool_call>post",
            '<tool_call><function=a>{"k": 1}</function></tool_call>',
            "lead\n<tool_call>\n<function=a>\n<parameter=k>x</parameter>\n</function>\n</tool_call>\ntail",
        ]:
            out = parser.extract_tool_calls(text)
            if out.tools_called:
                content = out.content or ""
                assert "<tool_call>" not in content, (
                    f"<tool_call> leaked for input: {text!r}\ncontent: {content!r}"
                )
                assert "<function=" not in content
                assert "<parameter=" not in content

# SPDX-License-Identifier: Apache-2.0
"""XML-function tool parser tests.

This covers MiMo-style ``<tool_call><function=...><parameter=...>`` markup as a
no-heavy parser contract. It does not claim MiMo generation quality.
"""

import json

import pytest

from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParserManager
from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser


@pytest.fixture
def parser():
    return XMLFunctionToolParser(tokenizer=None)


class TestXMLFunctionToolParser:
    def test_no_tool_calls_returns_content_unchanged(self, parser):
        out = parser.extract_tool_calls("plain response")

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "plain response"

    def test_single_function_with_typed_parameters(self, parser):
        text = """
<tool_call>
<function=write_file>
<parameter=path>"notes/result.txt"</parameter>
<parameter=overwrite>true</parameter>
<parameter=count>3</parameter>
</function>
</tool_call>
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "write_file"
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"path": "notes/result.txt", "overwrite": True, "count": 3}
        assert out.content is None

    def test_value_wrapper_is_unwrapped_before_json_coercion(self, parser):
        text = """
<tool_call>
<function=search>
<parameter=query><value>"qwen parser edge"</value></parameter>
</function>
</tool_call>
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"query": "qwen parser edge"}

    def test_hyphenated_literal_parameter_is_preserved(self, parser):
        text = """
<tool_call>
<function=record_fact>
<parameter=value>blue-cat</parameter>
</function>
</tool_call>
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_pretty_wrapper_newlines_trim_scalar_not_inner_spacing(self, parser):
        text = """
<tool_call>
<function=record_fact>
<parameter=value>
blue-cat
</parameter>
<parameter=cmd>  printf '&lt;日本語&gt;' &amp;&amp; pwd  </parameter>
<parameter=content>line one
line two</parameter>
</function>
</tool_call>
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args["value"] == "blue-cat"
        assert args["cmd"] == "  printf '<日本語>' && pwd  "
        assert args["content"] == "line one\nline two"

    def test_visible_text_around_tool_call_has_no_xml_function_leak(self, parser):
        text = """
I will write the file now.
<tool_call>
<function=write_file>
<parameter=path>out.txt</parameter>
<parameter=content>hello</parameter>
</function>
</tool_call>
Done.
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        assert out.content is not None
        assert "I will write the file now." in out.content
        assert "Done." in out.content
        assert "<tool_call>" not in out.content
        assert "</tool_call>" not in out.content
        assert "<function=" not in out.content
        assert "<parameter=" not in out.content

    def test_multiple_functions_in_one_tool_call(self, parser):
        text = """
<tool_call>
<function=first><parameter=x>1</parameter></function>
<function=second><parameter=y>2</parameter></function>
</tool_call>
"""

        out = parser.extract_tool_calls(text)

        assert out.tools_called is True
        assert [call["name"] for call in out.tool_calls] == ["first", "second"]
        assert json.loads(out.tool_calls[0]["arguments"]) == {"x": 1}
        assert json.loads(out.tool_calls[1]["arguments"]) == {"y": 2}

    def test_streaming_waits_until_tool_call_close(self, parser):
        previous = "<tool_call><function=write_file>"
        current = previous + "<parameter=path>out.txt</parameter></function></tool_call>"

        early = parser.extract_tool_calls_streaming("", previous, previous)
        final = parser.extract_tool_calls_streaming(previous, current, "</tool_call>")

        assert early is None
        assert final is not None
        assert final["tool_calls"][0]["function"]["name"] == "write_file"

    def test_registry_aliases_resolve(self):
        for alias in ("xml_function", "mimo_xml_function"):
            cls = ToolParserManager.get_tool_parser(alias)
            assert cls is XMLFunctionToolParser

    def test_repairs_missing_opening_tool_call_when_schema_allows_function(self, parser):
        text = """```xml
<function=get_weather>
<parameter=city>San Francisco</parameter>
</function>
</tool_call>
```"""

        out = parser.extract_tool_calls(
            text,
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ]
            },
        )

        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "get_weather"
        assert json.loads(out.tool_calls[0]["arguments"]) == {
            "city": "San Francisco"
        }
        assert out.content is None

    def test_missing_opening_tool_call_repair_rejects_unknown_function(self, parser):
        text = """
<function=delete_everything>
<parameter=path>/</parameter>
</function>
</tool_call>
"""

        out = parser.extract_tool_calls(
            text,
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ]
            },
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == text

    def test_repairs_missing_closing_tool_call_when_schema_allows_function(self, parser):
        text = """<tool_call>
<function=record_fact>
<parameter=value>blue-cat</parameter>
</function>"""

        out = parser.extract_tool_calls(
            text,
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "record_fact",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        },
                    }
                ]
            },
        )

        assert out.tools_called is True
        assert out.tool_calls[0]["name"] == "record_fact"
        assert json.loads(out.tool_calls[0]["arguments"]) == {"value": "blue-cat"}
        assert out.content is None

    def test_missing_closing_tool_call_repair_rejects_unknown_function(self, parser):
        text = """<tool_call>
<function=delete_everything>
<parameter=path>/</parameter>
</function>"""

        out = parser.extract_tool_calls(
            text,
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "record_fact",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            },
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == text

    def test_bare_incomplete_tool_call_still_fails_closed(self, parser):
        out = parser.extract_tool_calls(
            "<tool_call>",
            request={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            },
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == "<tool_call>"

    def test_empty_function_with_required_schema_fails_closed(self, parser):
        text = """Quick preamble.
<tool_call>
<function=exec_command>
</function>
</tool_call>"""

        out = parser.extract_tool_calls(
            text,
            request={
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
            },
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == text

    def test_nested_invoke_missing_required_schema_fails_closed(self, parser):
        text = """<tool_call>
<invoke>
<tool_name>exec_command</tool_name>
<arguments></arguments>
</invoke>
</tool_call>"""

        out = parser.extract_tool_calls(
            text,
            request={
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
            },
        )

        assert out.tools_called is False
        assert out.tool_calls == []
        assert out.content == text

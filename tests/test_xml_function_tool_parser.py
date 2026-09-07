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


def test_nullable_schema_field_turns_none_spelling_into_json_null(parser):
    """Live 2026-09-07 (Qwen3.8-27B, xml_function): the model wrote <parameter name="path">None</parameter> for a
    nullable field and the STRING "None" reached the tool. Only a schema that allows null coerces it; a plain
    string field keeps the text."""
    output = (
        "<tool_call>\n<function=search>\n<parameter=pattern>magic</parameter>\n<parameter=path>None</parameter>\n"
        "<parameter=label>None</parameter>\n</function>\n</tool_call>"
    )
    request = {"tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string"},
        "path": {"type": ["string", "null"]},
        "label": {"type": "string"},
    }}}}]}
    info = parser.extract_tool_calls(output, request)
    assert info.tools_called and len(info.tool_calls) == 1
    args = json.loads(info.tool_calls[0]["arguments"])
    assert args["path"] is None            # nullable: None spelling -> JSON null
    assert args["label"] == "None"         # plain string: the model's text is kept
    assert args["pattern"] == "magic"
    # without a request/schema nothing is coerced
    info2 = parser.extract_tool_calls(output, None)
    assert json.loads(info2.tool_calls[0]["arguments"])["path"] == "None"
    # anyOf / nullable:true spellings also count as allowing null
    request2 = {"tools": [{"type": "function", "name": "search", "parameters": {"type": "object", "properties": {"path": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}]}
    assert json.loads(parser.extract_tool_calls(output, request2).tool_calls[0]["arguments"])["path"] is None


class TestEscapingIsData:
    """Newline U+000A, the two characters backslash+n, slash+n, CRLF, tab, quotes, backslashes and
    XML entities are distinct data. A string parameter's bytes must survive parse -> JSON arguments
    -> JSON decode unchanged (LOSSLESS-API-TOOL-HISTORY-CONTRACT). The live 27B-4D escaping case
    (probe-27b-replay-164849) diverged at GENERATION: the model wrote a real newline for the
    prompt's literal backslash+n, and the parser preserved that byte-for-byte."""

    REQUEST = {"tools": [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]}

    @pytest.mark.parametrize("payload", [
        'He said "hi\\n" — 日本語 🚀 \\ path\\to\\file\nsecond line',  # literal backslash+n AND a real newline
        "line/none\r\nwindows\tTab",  # slash+n, CRLF, tab
        "a &amp; b &lt;c&gt; 'q' \"dq\"",  # XML entities and quotes stay literal
        "\\\\double \\n\\t single-escapes stay two chars",
        "    indented first line\n\n\ntrailing blank lines\n\n",
    ])
    def test_string_parameter_round_trips_byte_for_byte(self, parser, payload):
        block = (
            "<tool_call>\n<function=write_file>\n<parameter=path>\nout/x.txt\n</parameter>\n"
            f"<parameter=content>\n{payload}\n</parameter>\n</function>\n</tool_call>"
        )
        out = parser.extract_tool_calls(block, self.REQUEST)
        assert out.tools_called
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args["content"] == payload
        assert args["path"] == "out/x.txt"

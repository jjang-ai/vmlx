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

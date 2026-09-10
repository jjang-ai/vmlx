# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from vmlx_engine.tool_parsers import ToolParserManager


def parse(value, declared_type="string", flat=False):
    fn = {"name": "write_file", "parameters": {"type": "object", "properties": {"content": {"type": declared_type}}}}
    tool = {"type": "function", **fn} if flat else {"type": "function", "function": fn}
    parser = ToolParserManager.get_tool_parser("spark25")(None)
    return parser.extract_tool_calls(
        "<tool_call>write_file<arg_key>content</arg_key><arg_value>" + value + "</arg_value></tool_call>",
        {"tools": [tool]},
    )


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("value", ["  indented\n\n", "true", "001", "null", "", "\\n", "日本語"])
def test_string_bytes_survive_chat_and_responses_schema(value, flat):
    result = parse(value, flat=flat)
    assert result.tools_called and len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0]["arguments"]) == {"content": value}
    assert result.content is None


@pytest.mark.parametrize("raw,kind,expected", [
    ("null", ["string", "null"], None), ("  text\n", ["string", "null"], "  text\n"),
    ("42", "integer", 42), ("true", "boolean", True),
    ('{"a":[1,false]}', "object", {"a": [1, False]}),
])
def test_native_json_values(raw, kind, expected):
    assert json.loads(parse(raw, kind).tool_calls[0]["arguments"])["content"] == expected


def test_multiple_calls_keep_visible_content_and_unique_ids():
    parser = ToolParserManager.get_tool_parser("spark25")(None)
    text = "Before.<tool_call>one</tool_call><tool_call>two</tool_call>After."
    result = parser.extract_tool_calls(text)
    assert [c["name"] for c in result.tool_calls] == ["one", "two"]
    assert len({c["id"] for c in result.tool_calls}) == 2
    assert result.content == "Before.After."


@pytest.mark.parametrize("text", [
    "<tool_call>one",
    "<tool_call>one<arg_key>x</arg_key></tool_call>",
    "<tool_call>one<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>",
    "<tool_call>one<arg_key>x</arg_key><arg_value>1</arg_value>garbage</tool_call>",
])
def test_malformed_calls_are_not_repaired_into_arguments(text):
    with pytest.raises(ValueError):
        ToolParserManager.get_tool_parser("spark25")(None).extract_tool_calls(text)


def test_implicit_reasoning_opener_is_not_visible_content():
    result = ToolParserManager.get_tool_parser("spark25")(None).extract_tool_calls(
        "I should read the file.</think><tool_call>read_file</tool_call>")
    assert result.content is None
    assert result.tool_calls[0]["name"] == "read_file"

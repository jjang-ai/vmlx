# SPDX-License-Identifier: Apache-2.0
"""dots (dots3_note) tool-call parsing.

Pins the dialect contract from the bundle's chat_template.jinja:
- multiple <invoke> blocks per <dots_function_call>;
- parameter bodies framed by exactly one newline each side — inner
  whitespace (code indentation) is payload;
- string args verbatim, non-string args via |tojson, with the request tool
  schema authoritative for the read-back direction;
- truncated (max_tokens) native blocks still parse leniently;
- <think> reasoning is stripped before parsing.
"""

import json

import pytest

from vmlx_engine.tool_parsers import ToolParserManager


@pytest.fixture
def parser():
    return ToolParserManager.get_tool_parser("dots")(None)


def _args(result, index=0):
    return json.loads(result.tool_calls[index]["arguments"])


WEATHER_REQUEST = {
    "tools": [
        {
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer"},
                        "opts": {"type": "object"},
                    }
                },
            }
        },
        {
            "function": {
                "name": "write_file",
                "parameters": {
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    }
                },
            }
        },
    ]
}


def test_registered_under_every_documented_alias():
    for alias in ("dots", "dots3", "dots3_note"):
        cls = ToolParserManager.get_tool_parser(alias)
        assert cls.__name__ == "DotsToolParser"


def test_single_call_schema_typed(parser):
    out = (
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\n'
        "San Francisco\n"
        "</parameter>\n"
        '<parameter name="days">\n'
        "3\n"
        "</parameter>\n"
        '<parameter name="opts">\n'
        '{"units": "metric"}\n'
        "</parameter>\n"
        "</invoke>\n"
        "</dots_function_call>"
    )
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert result.tools_called
    assert result.tool_calls[0]["name"] == "get_weather"
    args = _args(result)
    assert args["location"] == "San Francisco"
    assert args["days"] == 3
    assert args["opts"] == {"units": "metric"}


def test_string_arg_that_looks_like_json_stays_verbatim(parser):
    out = (
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\n'
        '["not", "a", "list"]\n'
        "</parameter>\n"
        "</invoke>\n"
        "</dots_function_call>"
    )
    args = _args(parser.extract_tool_calls(out, request=WEATHER_REQUEST))
    assert args["location"] == '["not", "a", "list"]'


def test_code_argument_preserves_indentation_and_inner_newlines(parser):
    code = "def f():\n    return 1\n\n\nprint(f())"
    out = (
        "<dots_function_call>\n"
        '<invoke name="write_file">\n'
        '<parameter name="path">\n'
        "/tmp/x.py\n"
        "</parameter>\n"
        '<parameter name="content">\n'
        f"{code}\n"
        "</parameter>\n"
        "</invoke>\n"
        "</dots_function_call>"
    )
    args = _args(parser.extract_tool_calls(out, request=WEATHER_REQUEST))
    assert args["content"] == code
    assert args["path"] == "/tmp/x.py"


def test_leading_indentation_on_first_line_survives(parser):
    # Exactly ONE frame newline is stripped; the following spaces are payload.
    out = (
        "<dots_function_call>\n"
        '<invoke name="write_file">\n'
        '<parameter name="content">\n'
        "    indented first line\n"
        "</parameter>\n"
        "</invoke>\n"
        "</dots_function_call>"
    )
    args = _args(parser.extract_tool_calls(out, request=WEATHER_REQUEST))
    assert args["content"] == "    indented first line"


def test_multiple_invokes_in_one_block(parser):
    out = (
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\nParis\n</parameter>\n'
        "</invoke>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\nTokyo\n</parameter>\n'
        "</invoke>\n"
        "</dots_function_call>"
    )
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert len(result.tool_calls) == 2
    assert _args(result, 0)["location"] == "Paris"
    assert _args(result, 1)["location"] == "Tokyo"


def test_untyped_request_falls_back_to_json_then_string(parser):
    out = (
        "<dots_function_call>\n"
        '<invoke name="unknown_tool">\n'
        '<parameter name="count">\n7\n</parameter>\n'
        '<parameter name="note">\nhello world\n</parameter>\n'
        "</invoke>\n"
        "</dots_function_call>"
    )
    args = _args(parser.extract_tool_calls(out, request=None))
    assert args["count"] == 7
    assert args["note"] == "hello world"


def test_content_before_call_is_preserved_and_markup_removed(parser):
    out = (
        "Let me check the weather.\n"
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\nOslo\n</parameter>\n'
        "</invoke>\n"
        "</dots_function_call>"
    )
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert result.tools_called
    assert result.content == "Let me check the weather."
    assert "<dots_function_call>" not in (result.content or "")


def test_think_block_stripped_before_parsing(parser):
    out = (
        "<think>\nI should call the tool.\n</think>\n"
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\nRome\n</parameter>\n'
        "</invoke>\n"
        "</dots_function_call>"
    )
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert result.tools_called
    assert _args(result)["location"] == "Rome"
    assert "think" not in (result.content or "")


def test_truncated_generation_parses_leniently(parser):
    # max_tokens hit before any closing tag was emitted.
    out = (
        "<dots_function_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="location">\nBer'
    )
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert result.tools_called
    assert _args(result)["location"] == "Ber"


def test_plain_text_is_not_a_tool_call(parser):
    out = "The dots XML dialect uses <invoke name=...> inside a wrapper."
    result = parser.extract_tool_calls(out, request=WEATHER_REQUEST)
    assert not result.tools_called
    assert result.content == out


def test_type_list_keeps_strings_verbatim_and_decodes_null_when_allowed():
    """JSON-Schema type lists are declarations: ["string","null"] keeps "123" as text."""
    from vmlx_engine.tool_parsers.dots_tool_parser import DotsToolParser

    coerce = DotsToolParser._coerce_value
    assert coerce("123", {"type": ["string", "null"]}) == "123"
    assert coerce("null", {"type": ["string", "null"]}) is None
    assert coerce("null", {"type": "string"}) == "null"
    assert coerce("3", {"type": ["integer", "null"]}) == 3
    assert coerce("{\"a\": 1}", {"type": ["object", "null"]}) == {"a": 1}

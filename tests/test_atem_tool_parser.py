# SPDX-License-Identifier: Apache-2.0
"""ATEM (Muse Glimmer) tool-call parsing.

The bundle's own chat template states the output "is not expected to be valid
XML and is parsed with regular expressions", so these tests pin tolerance of
the exact things an XML parser would reject: unescaped markup inside values,
truncated tails, and missing closing tags.
"""

import json

import pytest

from vmlx_engine.tool_parsers import ToolParserManager


@pytest.fixture
def parser():
    return ToolParserManager.get_tool_parser("atem")(None)


def _args(result, index=0):
    # FLAT shape: the engine reads tc["name"]/tc["arguments"] directly and
    # builds the OpenAI envelope itself. Asserting a pre-nested
    # {"function": {...}} here is what let the real defect ship — the
    # dispatcher KeyError'd and leaked raw <atem:...> markup to the user.
    return json.loads(result.tool_calls[index]["arguments"])


WEATHER_REQUEST = {
    "tools": [
        {
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                        "days": {"type": "integer"},
                        "metric": {"type": "boolean"},
                        "opts": {"type": "object"},
                    }
                },
            }
        }
    ]
}


def test_registered_under_every_documented_alias():
    for alias in ("atem", "muse_glimmer", "muse"):
        assert ToolParserManager.get_tool_parser(alias).__name__ == "AtemToolParser"


def test_parses_a_well_formed_call(parser):
    out = (
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">San Francisco</atem:parameter>\n'
        '<atem:parameter name="unit">celsius</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    result = parser.extract_tool_calls(out, WEATHER_REQUEST)

    assert result.tools_called
    assert result.tool_calls[0]["name"] == "get_weather"
    assert _args(result) == {"location": "San Francisco", "unit": "celsius"}


def test_preserves_prose_before_the_call(parser):
    out = (
        "Let me check that for you.\n"
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    result = parser.extract_tool_calls(out, WEATHER_REQUEST)

    assert result.tools_called
    assert result.content == "Let me check that for you."


def test_plain_prose_is_never_promoted_to_a_call(parser):
    out = "I cannot call a tool, but <atem: is a namespace prefix."
    result = parser.extract_tool_calls(out, WEATHER_REQUEST)

    assert not result.tools_called
    assert result.content == out


def test_multiple_invokes_in_one_block(parser):
    out = (
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">Oslo</atem:parameter>\n</atem:invoke>\n'
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">Lima</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls>"
    )
    result = parser.extract_tool_calls(out, WEATHER_REQUEST)

    assert len(result.tool_calls) == 2
    assert _args(result, 0)["location"] == "Oslo"
    assert _args(result, 1)["location"] == "Lima"


class TestMalformedTolerance:
    """An XML parser would reject every input in this class."""

    def test_unescaped_markup_inside_a_value_survives(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="location">Tom &amp Jerry < 5 & rising</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert result.tools_called
        assert _args(result)["location"] == "Tom &amp Jerry < 5 & rising"

    def test_truncated_mid_invoke_still_salvages_the_call(self, parser):
        # What a max_tokens cut actually looks like: no closing tags at all.
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="location">San Francisco</atem:parameter>\n'
            '<atem:parameter name="unit">celsius'
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert result.tools_called, "truncated call was dropped instead of salvaged"
        assert _args(result)["location"] == "San Francisco"
        assert _args(result)["unit"] == "celsius"

    def test_missing_outer_close_tag(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="location">Kyoto</atem:parameter>\n'
            "</atem:invoke>"
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert result.tools_called
        assert _args(result)["location"] == "Kyoto"

    def test_single_quoted_and_bare_names(self, parser):
        out = (
            "<atem:function_calls>\n<atem:invoke name='get_weather'>\n"
            "<atem:parameter name=location>Cairo</atem:parameter>\n"
            "</atem:invoke>\n</atem:function_calls>"
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert result.tool_calls[0]["name"] == "get_weather"
        assert _args(result)["location"] == "Cairo"


class TestValueDecoding:
    """The template renders bool/None/JSON/scalars; inverting that is ambiguous."""

    def test_schema_declared_string_keeps_a_literal_true(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="unit">true</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert _args(result)["unit"] == "true", "string param was coerced to a boolean"

    def test_schema_declared_boolean_decodes(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="metric">false</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        assert _args(parser.extract_tool_calls(out, WEATHER_REQUEST))["metric"] is False

    def test_schema_declared_integer_decodes(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="days">3</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        assert _args(parser.extract_tool_calls(out, WEATHER_REQUEST))["days"] == 3

    def test_object_parameter_round_trips_through_json(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="opts">{"a": 1, "b": [2, 3]}</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        assert _args(parser.extract_tool_calls(out, WEATHER_REQUEST))["opts"] == {
            "a": 1,
            "b": [2, 3],
        }

    def test_malformed_json_in_an_object_param_falls_back_to_the_raw_string(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="opts">{"a": 1,,}</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        result = parser.extract_tool_calls(out, WEATHER_REQUEST)

        assert result.tools_called, "malformed JSON must not drop the whole call"
        assert _args(result)["opts"] == '{"a": 1,,}'

    def test_no_schema_falls_back_to_json_shape(self, parser):
        out = (
            '<atem:function_calls>\n<atem:invoke name="unknown_fn">\n'
            '<atem:parameter name="n">7</atem:parameter>\n'
            '<atem:parameter name="s">hello</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        args = _args(parser.extract_tool_calls(out, WEATHER_REQUEST))

        assert args["n"] == 7
        assert args["s"] == "hello"


class TestStreaming:
    def test_holds_until_the_call_closes(self, parser):
        partial = '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        assert parser.extract_tool_calls_streaming(
            "", partial, partial, [], [], [], WEATHER_REQUEST
        ) is None

    def test_emits_once_when_complete_and_not_again(self, parser):
        complete = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="location">Rome</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        first = parser.extract_tool_calls_streaming(
            "<atem:function_calls>", complete, "", [], [], [], WEATHER_REQUEST
        )
        assert first is not None and first.tools_called

        again = parser.extract_tool_calls_streaming(
            complete, complete + " trailing", "", [], [], [], WEATHER_REQUEST
        )
        assert again is None, "the same call was emitted twice"


def test_tool_call_dicts_match_the_engine_contract(parser):
    """Pin the shape the server actually consumes.

    vmlx_engine/server.py reads ``tc["name"]`` and ``tc["arguments"]`` off each
    returned dict and builds the OpenAI ``{"type": "function", "function": ...}``
    envelope itself. A parser that returns a pre-nested envelope raises KeyError
    inside the dispatcher, which catches it and falls through to passing the raw
    model output to the user — so the failure surfaces as visible ``<atem:...>``
    markup and ``tool_calls: null``, never as an error. This test asserts the
    flat contract directly instead of trusting a helper.
    """
    out = (
        "<atem:function_calls>\n"
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="location">Oslo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    result = parser.extract_tool_calls(out, WEATHER_REQUEST)

    assert result.tools_called
    for call in result.tool_calls:
        assert "function" not in call, (
            "pre-nested envelope: the engine builds that itself"
        )
        assert isinstance(call["id"], str) and call["id"]
        assert isinstance(call["name"], str) and call["name"]
        # arguments must be a JSON *string*, not a dict
        assert isinstance(call["arguments"], str)
        json.loads(call["arguments"])


class TestNullableTypeLists:
    """JSON Schema type LISTS (nullable parameters) must decode, never raise.

    Live shape (Muse-Glimmer-30B, agentic-eval-074524 nullable_enum_bounds_valid,
    reproduced byte-for-byte in probe-muse-atem2-083555): the suite's search tool
    declares ``path: {"type": ["string", "null"]}``. ``_coerce`` hashed the list,
    raised TypeError, the server fell back to generic repair and every argument
    came back as ``>magic</atem:parameter>\\n<atem:parameter name=``.
    """

    SEARCH_REQUEST = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                            "mode": {"type": "string", "enum": ["literal", "regex"]},
                            "path": {"type": ["string", "null"], "description": "Subdirectory or null for the whole workspace"},
                        },
                        "required": ["pattern"],
                    },
                },
            }
        ]
    }

    BLOCK = (
        "<atem:function_calls>\n<atem:invoke name=\"search\">\n"
        '<atem:parameter name="pattern">magic</atem:parameter>\n'
        '<atem:parameter name="max_results">3</atem:parameter>\n'
        '<atem:parameter name="mode">literal</atem:parameter>\n'
        "{path}"
        "</atem:invoke>\n</atem:function_calls>"
    )

    def test_exact_suite_shape_decodes_every_parameter(self, parser):
        out = parser.extract_tool_calls(self.BLOCK.format(path=""), self.SEARCH_REQUEST)
        assert out.tools_called
        args = json.loads(out.tool_calls[0]["arguments"])
        assert args == {"pattern": "magic", "max_results": 3, "mode": "literal"}

    def test_null_spelling_decodes_to_none_when_null_is_allowed(self, parser):
        out = parser.extract_tool_calls(
            self.BLOCK.format(path='<atem:parameter name="path">null</atem:parameter>\n'), self.SEARCH_REQUEST
        )
        assert json.loads(out.tool_calls[0]["arguments"])["path"] is None

    def test_nullable_string_keeps_a_real_value_verbatim(self, parser):
        out = parser.extract_tool_calls(
            self.BLOCK.format(path='<atem:parameter name="path">src/null-safe </atem:parameter>\n'), self.SEARCH_REQUEST
        )
        assert json.loads(out.tool_calls[0]["arguments"])["path"] == "src/null-safe "

    def test_plain_string_type_never_turns_null_into_none(self, parser):
        out = parser.extract_tool_calls(
            self.BLOCK.format(path='<atem:parameter name="pattern">null</atem:parameter>\n'), self.SEARCH_REQUEST
        )
        # last occurrence wins for a repeated key; the declared plain string stays the text "null"
        assert json.loads(out.tool_calls[0]["arguments"])["pattern"] == "null"

    def test_nullable_integer_decodes_the_number(self):
        from vmlx_engine.tool_parsers.atem_tool_parser import _coerce

        assert _coerce("3", ["integer", "null"]) == 3
        assert _coerce("none", ["integer", "null"]) is None
        assert _coerce("x", ["integer", "string"]) == "x"

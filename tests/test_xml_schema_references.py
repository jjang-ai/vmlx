"""Native XML values retain declared types behind composed/local schemas."""
import copy
import json

import pytest

from vmlx_engine.tool_parsers.xml_function_tool_parser import XMLFunctionToolParser
from vmlx_engine.tool_parsers.nemotron_tool_parser import NemotronToolParser
from vmlx_engine.tool_parsers.schema_types import xml_parameter_type_hints


@pytest.mark.parametrize("flat", [False, True], ids=["chat", "responses"])
@pytest.mark.parametrize("parser_type", [XMLFunctionToolParser, NemotronToolParser])
@pytest.mark.parametrize("prop", [
    {"$ref": "#/$defs/text"},
    {"allOf": [{"$ref": "#/$defs/text"}, {"minLength": 1}]},
    {"anyOf": [{"$ref": "#/$defs/text"}, {"type": "null"}]},
    {"$ref": "#/$defs/text~1value~0"},
    {"$ref": "#text_anchor"},
])
@pytest.mark.parametrize("payload", ["123", '{"n":1}\n', '  日本語 \\n "quotes"  '])
def test_referenced_strings_are_literal_in_stream_and_nonstream(flat, parser_type, prop, payload):
    schema = {"type": "object", "$defs": {
        "text": {"type": "string", "$anchor": "text_anchor"},
        "text/value~": {"type": "string"},
    }, "properties": {"content": prop}}
    fn = {"name": "write_file", "parameters": schema}
    request = {"tools": [{"type": "function", **fn} if flat else {"type": "function", "function": fn}]}
    original = copy.deepcopy(request)
    block = f"<tool_call><function=write_file><parameter=content>\n{payload}\n</parameter></function></tool_call>"
    parser = parser_type(None)
    result = parser.extract_tool_calls(block, request)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"content": payload}
    streamed = parser.extract_tool_calls_streaming("", block, "</tool_call>", request=request)
    assert json.loads(streamed["tool_calls"][0]["function"]["arguments"]) == {"content": payload}
    assert request == original


def test_referenced_boolean_and_ambiguous_union():
    schema = {"$defs": {"flag": {"type": "boolean"}}, "properties": {
        "flag": {"$ref": "#/$defs/flag"},
        "ambiguous": {"anyOf": [{"$ref": "#/$defs/flag"}, {"type": "string"}]},
    }}
    hints = xml_parameter_type_hints(schema)
    assert XMLFunctionToolParser._coerce_value("False", hints["flag"]) is False
    assert XMLFunctionToolParser._coerce_value("False", hints["ambiguous"]) == "False"
    assert XMLFunctionToolParser._coerce_value('"False"', hints["flag"]) == "False"


@pytest.mark.parametrize("draft,expected", [
    ("http://json-schema.org/draft-07/schema#", {"string"}),
    ("https://json-schema.org/draft/2020-12/schema", set()),
])
def test_declared_draft_controls_ref_sibling_semantics(draft, expected):
    schema = {"$schema": draft, "definitions": {"text": {"type": "string"}},
              "properties": {"value": {"$ref": "#/definitions/text", "type": "boolean"}}}
    assert set(xml_parameter_type_hints(schema)["value"]["type"]) == expected


def test_embedded_resource_reference_has_correct_base_scope():
    schema = {"$id": "https://example.invalid/root/", "$defs": {
        "embedded": {"$id": "nested/", "$defs": {"text": {"type": "string"}}, "$ref": "#/$defs/text"},
    }, "properties": {"value": {"$ref": "nested/"}}}
    assert xml_parameter_type_hints(schema)["value"]["type"] == ["string"]


def test_unknown_remote_and_recursive_refs_do_not_fetch_or_guess(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network access"))
    schema = {"$defs": {"loop": {"$ref": "#/$defs/loop"}}, "properties": {
        "remote": {"$ref": "https://example.invalid/schema"},
        "cycle": {"$ref": "#/$defs/loop"},
    }}
    for hint in xml_parameter_type_hints(schema).values():
        assert not XMLFunctionToolParser._schema_is_string_or_null(hint)
        assert not XMLFunctionToolParser._schema_is_boolean_or_null(hint)


def test_nemotron_native_boolean_and_json_envelope_are_distinct():
    request = {"tools": [{"type": "function", "function": {"name": "set_flag", "parameters": {
        "properties": {"flag": {"type": "boolean"}, "text": {"type": "string"}},
    }}}]}
    p = NemotronToolParser(None)
    text = "<tool_call><function=set_flag><parameter=flag>False</parameter><parameter=text>123</parameter></function></tool_call>"
    for out in [p.extract_tool_calls(text, request).tool_calls[0],
                p.extract_tool_calls_streaming("", text, "</tool_call>", request=request)["tool_calls"][0]["function"]]:
        assert json.loads(out["arguments"]) == {"flag": False, "text": "123"}
    # JSON-native calls retain generated types, even when they violate schema.
    text = '<tool_call><function=set_flag>{"flag":"False","text":123}</function></tool_call>'
    assert json.loads(p.extract_tool_calls(text, request).tool_calls[0]["arguments"]) == {"flag": "False", "text": 123}


@pytest.mark.parametrize('prop', [
    {'type': ['integer', 'null']}, {'type': ['number', 'null']},
    {'type': ['boolean', 'null']}, {'type': ['array', 'null']},
    {'type': ['object', 'null']}, {'type': 'null'},
    {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
    {'$ref': '#/$defs/nullable'},
])
def test_nemotron_native_none_roundtrips_nullable_nonstring_parameters(prop):
    request = {'tools': [{'type': 'function', 'function': {'name': 'record', 'parameters': {
        '$defs': {'nullable': {'type': ['integer', 'null']}}, 'properties': {'value': prop},
    }}}]}
    parser = NemotronToolParser()
    raw = '<tool_call><function=record><parameter=value>\nNone\n</parameter></function></tool_call>'
    result = parser.extract_tool_calls(raw, request)
    assert json.loads(result.tool_calls[0]['arguments']) == {'value': None}
    streamed = parser.extract_tool_calls_streaming('', raw, '</tool_call>', request=request)
    assert json.loads(streamed['tool_calls'][0]['function']['arguments']) == {'value': None}
    # JSON-native envelopes retain their actual generated types.
    raw = '<tool_call><function=record>{"value":"None"}</function></tool_call>'
    assert json.loads(parser.extract_tool_calls(raw, request).tool_calls[0]['arguments']) == {'value': 'None'}


@pytest.mark.parametrize('prop', [
    {'type': 'string'}, {'type': 'integer'}, {},
    {'type': ['string', 'integer', 'null']},
    {'$ref': 'https://example.invalid/unavailable-schema'},
])
def test_nemotron_none_is_not_guessed_for_strings_or_unresolved_schemas(prop):
    request = {'tools': [{'type': 'function', 'function': {'name': 'record', 'parameters': {
        'properties': {'value': prop},
    }}}]}
    raw = '<tool_call><function=record><parameter=value>None</parameter></function></tool_call>'
    assert json.loads(NemotronToolParser().extract_tool_calls(raw, request).tool_calls[0]['arguments']) == {'value': 'None'}

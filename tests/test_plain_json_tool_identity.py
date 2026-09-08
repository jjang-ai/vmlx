"""Ordinary JSON names are data, not authority to create a tool invocation."""
import json

import pytest

from vmlx_engine.api.tool_calling import parse_tool_calls


@pytest.mark.parametrize("text", [
    '{"name":"demo","ports":[8080,8443]}',
    'The service is {"name":"demo","ports":[8080,8443]}.',
    '[{"name":"demo"},{"name":"other"}]',
    '{"name":"read_file"}',
    '{"name":"demo","parameters":null}',
    '{"name":"demo","arguments":[]}',
])
def test_plain_named_json_remains_content(text):
    content, calls = parse_tool_calls(text)
    assert content == text
    assert not calls


@pytest.mark.parametrize("key", ["arguments", "parameters"])
@pytest.mark.parametrize("array", [False, True])
def test_explicit_raw_call_keeps_empty_arguments(key, array):
    obj = {"name": "ping", key: {}}
    _, calls = parse_tool_calls(json.dumps([obj] if array else obj))
    assert len(calls) == 1
    assert calls[0].function.name == "ping"
    assert json.loads(calls[0].function.arguments) == {}


@pytest.mark.parametrize("parser", [None, "xml_function", "qwen", "auto"])
def test_plain_json_does_not_become_an_available_tool_or_warning(monkeypatch, parser):
    from vmlx_engine import server
    from vmlx_engine.api.models import ChatCompletionRequest

    monkeypatch.setattr(server, "_tool_call_parser", parser)
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", None)
    request = ChatCompletionRequest(model="test", messages=[{"role":"user","content":"Describe the service."}],
        tools=[{"type":"function","function":{"name":"demo","parameters":{"type":"object","properties":{}}}}])
    server._begin_tool_call_drop_capture()
    text = 'The service object is {"name":"demo","ports":[8080,8443]}.'
    content, calls = server._parse_tool_calls_with_parser(text, request)
    assert content == text
    assert not calls
    assert not server._take_tool_call_drop_diagnostics()


def test_auto_parser_preserves_granite_style_explicit_call():
    from vmlx_engine.tool_parsers.auto_tool_parser import AutoToolParser
    result = AutoToolParser().extract_tool_calls('{"type":"ping","arguments":{}}')
    assert result.tools_called
    assert result.tool_calls[0]["name"] == "ping"

"""Validate the emitted argument object, not an empty surrogate."""
import pytest

@pytest.mark.parametrize('mode', ['off', 'warn', 'enforce'])
@pytest.mark.parametrize('arguments', ['[]', '123', 'null', 'true', '"text"'])
def test_non_object_native_arguments_are_not_executable(monkeypatch, mode, arguments):
    from vmlx_engine import server
    from vmlx_engine.api.models import ChatCompletionRequest
    monkeypatch.setenv('VMLX_TOOL_ARGS_SCHEMA_VALIDATION', mode)
    monkeypatch.setattr(server, '_tool_call_parser', 'minimax')
    monkeypatch.setattr(server, '_tool_call_parser_disabled_explicitly', False)
    request = ChatCompletionRequest(model='test', messages=[{'role':'user','content':'inspect'}],
        tools=[{'type':'function','function':{'name':'inspect','parameters':{
            'type':'object','properties':{'path':{'type':'string'}}}}}])
    server._begin_tool_call_drop_capture()
    _, calls = server._parse_tool_calls_with_parser(
        f'<minimax:tool_call><invoke name="inspect">{arguments}</invoke></minimax:tool_call>', request)
    assert not calls
    assert any('JSON object' in item for item in server._take_tool_call_drop_diagnostics())

@pytest.mark.parametrize('arguments', ['{}', '{"path":"  a\\nb  "}'])
def test_valid_native_object_is_preserved(monkeypatch, arguments):
    from vmlx_engine import server
    from vmlx_engine.api.models import ChatCompletionRequest
    monkeypatch.setattr(server, '_tool_call_parser', 'minimax')
    monkeypatch.setattr(server, '_tool_call_parser_disabled_explicitly', False)
    request = ChatCompletionRequest(model='test', messages=[{'role':'user','content':'inspect'}],
        tools=[{'type':'function','function':{'name':'inspect','parameters':{
            'type':'object','properties':{'path':{'type':'string'}}}}}])
    server._begin_tool_call_drop_capture()
    _, calls = server._parse_tool_calls_with_parser(
        f'<minimax:tool_call><invoke name="inspect">{arguments}</invoke></minimax:tool_call>', request)
    assert calls and calls[0].function.arguments == arguments
    assert not server._take_tool_call_drop_diagnostics()

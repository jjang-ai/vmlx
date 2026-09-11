import json
import pytest
from vmlx_engine.tool_parsers.minimax_tool_parser import MiniMaxToolParser

@pytest.mark.parametrize('body', [
    '<invoke name="inspect">{"path":</invoke>',
    '<inspect>{"path":</inspect>',
    '<invoke name="inspect">unstructured prose</invoke>',
])
def test_malformed_native_arguments_never_become_raw_property(body):
    raw=f'Before. <minimax:tool_call>{body}</minimax:tool_call> After.'
    result=MiniMaxToolParser(None).extract_tool_calls(raw)
    assert not result.tools_called
    assert result.tool_calls == []
    assert 'minimax:tool_call' not in (result.content or '')
    assert 'Before.' in result.content and 'After.' in result.content

def test_server_does_not_repair_rejected_minimax_native_block(monkeypatch):
    from vmlx_engine import server
    from vmlx_engine.api.models import ChatCompletionRequest
    monkeypatch.setattr(server, '_tool_call_parser', 'minimax')
    monkeypatch.setattr(server, '_tool_call_parser_disabled_explicitly', False)
    request=ChatCompletionRequest(model='test',messages=[{'role':'user','content':'inspect'}],
        tools=[{'type':'function','function':{'name':'inspect','parameters':{
            'type':'object','properties':{'raw':{'type':'string'}}}}}])
    server._begin_tool_call_drop_capture()
    text,calls=server._parse_tool_calls_with_parser(
        '<minimax:tool_call><invoke name="inspect">{"path":</invoke></minimax:tool_call>',request)
    assert not calls and not text
    assert server._take_tool_call_drop_diagnostics()

def test_explicit_raw_parameter_remains_legitimate():
    result=MiniMaxToolParser(None).extract_tool_calls(
        '<minimax:tool_call><invoke name="inspect"><parameter name="raw">literal text</parameter></invoke></minimax:tool_call>')
    assert result.tools_called
    assert json.loads(result.tool_calls[0]['arguments'])=={'raw':'literal text'}

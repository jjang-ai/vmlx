"""Rejected native calls are not evidence of reasoning-only generation."""
import json
from types import SimpleNamespace

import pytest
from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput


@pytest.fixture(autouse=True)
def reset_capture():
    server._begin_tool_call_drop_capture()
    yield
    server._begin_tool_call_drop_capture()


def test_media_and_delivered_schema_warnings_are_not_rejections():
    from vmlx_engine.request_diagnostics import record
    record('video_controls: effective frame count 4')
    server._record_tool_call_drop('delivered invalid argument', rejected=False)
    assert not server._TOOL_CALL_REJECTED.get()
    assert len(server._take_tool_call_drop_diagnostics()) == 2


def test_rejection_state_survives_warning_drain_but_not_next_request():
    server._record_tool_call_drop('unavailable name')
    assert server._take_tool_call_drop_diagnostics() == ['unavailable name']
    assert server._TOOL_CALL_REJECTED.get()
    assert server._reasoning_only_chat_error_payload('id', tool_calls_rejected=True)['error']['code'] == 'tool_calls_rejected'
    assert server._current_response_warnings_for_reasoning_only(True, 'stop', tool_calls_rejected=True) is None
    server._begin_tool_call_drop_capture()
    assert not server._TOOL_CALL_REJECTED.get()
    assert server._reasoning_only_chat_error_payload('id')['error']['code'] == 'reasoning_only_no_content'


@pytest.mark.parametrize('finish,reason', [('stop', 'tool_calls_rejected'), ('length', 'max_output_tokens'), ('cancelled', 'cancelled')])
def test_responses_rejection_does_not_erase_actual_stop_cause(finish, reason):
    state = server._responses_terminal_state(finish, tool_calls_rejected=True, reasoning_only_no_content=True)
    assert state.event_type == 'response.incomplete'
    assert state.incomplete_details == {'reason': reason}


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['chat', 'responses'])
@pytest.mark.parametrize('stream', [True, False])
async def test_native_rejected_call_stream_has_typed_terminal(monkeypatch, surface, stream):
    text = '<tool_call>unavailable_reader<arg_key>path</arg_key><arg_value>rates.json</arg_value></tool_call>'
    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = False
        preserve_native_tool_format = True
        async def chat(self, **kwargs):
            return GenerationOutput(text=text, raw_text=text, tokens=[], prompt_tokens=10,
                                    completion_tokens=20, finished=True, finish_reason='stop')
        async def stream_chat(self, **kwargs):
            yield GenerationOutput(text=text, new_text=text, tokens=[], prompt_tokens=10,
                                   completion_tokens=20, finished=True, finish_reason='stop')
    engine = Engine()
    monkeypatch.setattr(server, '_engine', engine)
    monkeypatch.setattr(server, '_model_name', 'spark-rejection-test')
    monkeypatch.setattr(server, '_served_model_name', 'spark-rejection-test')
    monkeypatch.setattr(server, '_model_path', None)
    monkeypatch.setattr(server, '_reasoning_parser', None)
    monkeypatch.setattr(server, '_tool_call_parser', 'spark25')
    monkeypatch.setattr(server, '_tool_call_parser_disabled_explicitly', False)
    function = {'name': 'read_file', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}
    messages = [{'role': 'user', 'content': 'Read rates.json'}]
    if surface == 'chat':
        tools = [{'type': 'function', 'function': function}]
        request = server.ChatCompletionRequest(model='spark-rejection-test', messages=messages, tools=tools, stream=True, max_tokens=128)
        iterator = server.stream_chat_completion(engine, messages, request, fastapi_request=None, tools=tools, max_tokens=128)
    else:
        tools = [{'type': 'function', **function}]
        request = server.ResponsesRequest(model='spark-rejection-test', input='Read rates.json', tools=tools, stream=True, max_output_tokens=128)
        iterator = server.stream_responses_api(engine, messages, request, fastapi_request=None, tools=tools, max_tokens=128)
    if not stream:
        request.stream = False
        if surface == 'chat':
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as error:
                await server.create_chat_completion(request, fastapi_request=None)
            assert error.value.status_code == 502
            assert error.value.detail['code'] == 'tool_calls_rejected'
        else:
            response = await server.create_response(request, fastapi_request=None)
            assert response.status == 'incomplete'
            assert response.incomplete_details == {'reason': 'tool_calls_rejected'}
            assert '<tool_call>' not in response.model_dump_json()
        return
    chunks = [chunk async for chunk in iterator]
    events = [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
    if surface == 'chat':
        errors = [e['error'] for e in events if e.get('error')]
        assert len(errors) == 1
        assert errors[0]['code'] == 'tool_calls_rejected'
    else:
        terminals = [e for e in events if e.get('type') in {'response.completed', 'response.failed', 'response.incomplete'}]
        assert len(terminals) == 1
        assert terminals[0]['response'].get('incomplete_details') == {'reason': 'tool_calls_rejected'}, terminals[0]
    assert 'reasoning_only_no_content' not in json.dumps(events)
    assert '<tool_call>' not in json.dumps(events)

"""Native Omni tool calls must survive the Responses transport and history."""
import json
from types import SimpleNamespace

import pytest
from starlette.responses import StreamingResponse


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('reasoning', ['', 'Inspect the files.'])
async def test_native_responses_preserves_tools_and_continuation(monkeypatch, stream, reasoning):
    from vmlx_engine import server
    calls = [{'id': 'call_a', 'type': 'function', 'function': {
        'name': 'read_file', 'arguments': '{"path":"007"}'}},
        {'id': 'call_b', 'type': 'function', 'function': {
        'name': 'read_file', 'arguments': '{"path":"second.txt"}'}}]
    stored = []
    monkeypatch.setattr(server, '_responses_store_history', lambda *a, **kw: stored.append((a, kw)))
    if stream:
        async def chunks():
            for delta in [{'reasoning_content': reasoning}, {'content': '\n'},
                          {'tool_calls': [{'index': i, **call} for i, call in enumerate(calls)]}]:
                yield 'data: ' + json.dumps({'choices': [{'delta': delta}]}) + '\n\n'
            yield 'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]}) + '\n\n'
            yield 'data: [DONE]\n\n'
        events = [json.loads(line[6:]) async for raw in server._adapt_omni_chat_stream_to_responses(
            StreamingResponse(chunks()), SimpleNamespace(model='omni'),
            history_messages=[{'role': 'user', 'content': 'read'}])
            for line in raw.splitlines() if line.startswith('data: ')]
        response = next(e['response'] for e in events if e['type'] == 'response.completed')
        added = [e for e in events if e['type'] == 'response.output_item.added']
        done = [e for e in events if e['type'] == 'response.output_item.done']
        assert [e['output_index'] for e in added] == list(range(len(added)))
        assert {e['item']['id'] for e in added} == {e['item']['id'] for e in done}
        assert len([e for e in events if e['type'] == 'response.function_call_arguments.done']) == 2
        assert not any(e['type'] == 'response.warning' for e in events)
        assert stored and not stored[0][1]['reasoning_only']
        saved_calls = [c for m in stored[0][0][1] for c in m.get('tool_calls', [])]
        assert [c['id'] for c in saved_calls] == ['call_a', 'call_b']
    else:
        response = server._adapt_omni_chat_completion_to_responses_payload({
            'choices': [{'message': {'content': '', 'reasoning_content': reasoning,
                                     'tool_calls': calls}, 'finish_reason': 'tool_calls'}]}, 'omni')
        assert not response.get('warnings')
    output = [item for item in response['output'] if item['type'] == 'function_call']
    assert [(item['call_id'], item['name'], item['arguments']) for item in output] == [
        (call['id'], call['function']['name'], call['function']['arguments']) for call in calls]
    assert response['status'] == 'completed'


@pytest.mark.parametrize('bad_arguments', [None, '{', '[]', '{"x":1,"x":2}'])
def test_nonstream_native_response_stores_media_and_call_for_previous_id(monkeypatch, bad_arguments):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server, omni_multimodal as omni
    _run_streaming_ollama_chat(monkeypatch, family_name='nemotron_h', model_type='nemotron_h',
        body={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True})
    monkeypatch.setattr(server, '_model_path', '/native-omni')
    monkeypatch.setattr(omni, 'is_omni_multimodal_bundle', lambda p: True)
    monkeypatch.setattr(omni, 'omni_multimodal_component_status', lambda p: {
        'bundle_compatible': True, 'modalities': ['text', 'image']})
    saved = []
    monkeypatch.setattr(server, '_responses_store_history', lambda *a, **kw: saved.append((a, kw)))
    async def dispatch(*args, **kwargs):
        assert bad_arguments is None, 'malformed history reached native generation'
        return {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_one', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"first.txt"}'}}
        ]}, 'finish_reason': 'tool_calls'}]}
    monkeypatch.setattr(omni, 'dispatch_omni_chat_completion', dispatch)
    inputs = [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}]}]
    if bad_arguments is not None:
        inputs += [{'type': 'function_call', 'call_id': 'old', 'name': 'read_file', 'arguments': bad_arguments},
                   {'type': 'function_call_output', 'call_id': 'old', 'output': 'old result'}]
    with TestClient(server.app) as client:
        response = client.post('/v1/responses', json={'model': 'test-model', 'stream': False,
            'input': inputs})
    if bad_arguments is not None:
        # Pydantic rejects malformed/non-object JSON at the request boundary;
        # native history admission additionally rejects duplicate JSON keys.
        assert response.status_code in (400, 422), response.text
        assert not saved
        return
    assert response.status_code == 200, response.text
    assert saved and saved[0][0][0] == response.json()['id']
    history = saved[0][0][1]
    assert history[0]['content'][0]['type'] == 'image_url'
    assert history[-1]['tool_calls'][0]['id'] == 'call_one'


def test_previous_id_tool_result_with_repeated_request_instructions(monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat
    from vmlx_engine import server, omni_multimodal as omni
    from vmlx_engine.omni_native_tools import prepare_native_tools
    _run_streaming_ollama_chat(monkeypatch, family_name='nemotron_h', model_type='nemotron_h',
        body={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True})
    monkeypatch.setattr(server, '_model_path', '/native-omni')
    monkeypatch.setattr(omni, 'is_omni_multimodal_bundle', lambda p: True)
    monkeypatch.setattr(omni, 'omni_multimodal_component_status', lambda p: {
        'bundle_compatible': True, 'modalities': ['text', 'image']})
    stored = {}; dispatched = []
    monkeypatch.setattr(server, '_responses_store_history', lambda identifier, messages, **kw: stored.update({identifier: messages}))
    monkeypatch.setattr(server, '_responses_get_history', lambda identifier: stored.get(identifier))
    async def dispatch(request, *args, **kwargs):
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        prepare_native_tools(request.tools, request.tool_choice, messages)
        dispatched.append(messages)
        message = {'role': 'assistant', 'content': 'done'}
        if len(dispatched) == 1:
            message = {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_one', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"first.txt"}'}}]}
        return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if len(dispatched) == 1 else 'stop'}]}
    monkeypatch.setattr(omni, 'dispatch_omni_chat_completion', dispatch)
    common = {'model': 'test-model', 'stream': False, 'instructions': 'Read the requested files.'}
    with TestClient(server.app) as client:
        first = client.post('/v1/responses', json={**common, 'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}]}]})
        assert first.status_code == 200, first.text
        second = client.post('/v1/responses', json={**common, 'previous_response_id': first.json()['id'],
            'input': [{'type': 'function_call_output', 'call_id': 'call_one', 'output': 'file contents'}]})
        assert second.status_code == 200, second.text
    assert len(dispatched) == 2
    assert [m['role'] for m in dispatched[-1]] == ['system', 'user', 'assistant', 'tool']

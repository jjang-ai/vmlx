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

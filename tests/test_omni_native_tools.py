"""Native media tools preserve schemas, causal result order and wire types."""
from copy import deepcopy
import json

import pytest
from fastapi import HTTPException

from vmlx_engine.omni_native_tools import prepare_native_tools, NativeToolOutput


def catalog(schema=None):
    return [{'type': 'function', 'function': {'name': 'record', 'parameters': schema or {
        'type': 'object', 'properties': {'value': {'type': 'string'}},
        'required': ['value'], 'additionalProperties': False,
    }}}]


def call(identifier, value):
    return {'id': identifier, 'type': 'function', 'function': {
        'name': 'record', 'arguments': json.dumps({'value': value})}}


def history():
    return [
        {'role': 'user', 'content': 'inspect image'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'inspect both',
         'tool_calls': [call('a', '007'), call('b', '  code\n')]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'},
    ]


def test_result_order_follows_call_ids_without_mutating_client_history():
    original = history(); before = deepcopy(original)
    contract = prepare_native_tools(catalog(), 'auto', original)
    assert original == before
    assert [m['tool_call_id'] for m in contract.messages[2:]] == ['a', 'b']
    assert contract.messages[1]['tool_calls'][0]['function']['arguments'] == {'value': '007'}
    assert contract.messages[1]['tool_calls'][1]['function']['arguments'] == {'value': '  code\n'}
    assert contract.messages[1]['reasoning_content'] == 'inspect both'
    assert contract.enabled and contract.active and contract.template_tools == catalog()


@pytest.mark.parametrize('arguments', ['{', '[]', 'null', '{"value":"x","value":"y"}', '{"value":NaN}'])
def test_malformed_history_arguments_never_become_empty_objects(arguments):
    messages = history(); messages[1]['tool_calls'][0]['function']['arguments'] = arguments
    with pytest.raises(HTTPException) as error:
        prepare_native_tools(catalog(), 'auto', messages)
    assert error.value.status_code == 400


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown', 'interrupted', 'duplicate_call_id'])
def test_incomplete_or_ambiguous_result_batches_rejected(mutation):
    messages = history()
    if mutation == 'missing': messages.pop()
    elif mutation == 'duplicate': messages[-1]['tool_call_id'] = 'b'
    elif mutation == 'unknown': messages[-1]['tool_call_id'] = 'unknown'
    elif mutation == 'interrupted': messages.insert(3, {'role': 'user', 'content': 'continue'})
    else: messages[1]['tool_calls'][1]['id'] = 'a'
    with pytest.raises(HTTPException):
        prepare_native_tools(catalog(), None, messages)


def test_none_retains_valid_history_but_does_not_advertise_tools():
    contract = prepare_native_tools(catalog(), 'none', history())
    assert not contract.enabled and contract.active and contract.template_tools == []


@pytest.mark.parametrize('choice', ['required', {'type': 'function', 'function': {'name': 'record'}}])
def test_forced_choices_remain_explicitly_unsupported(choice):
    with pytest.raises(HTTPException, match='tool_choice'):
        prepare_native_tools(catalog(), choice, [{'role': 'user', 'content': 'go'}])


def test_original_local_reference_schema_validates_without_flattening():
    schema = {'type': 'object', 'properties': {'value': {'$ref': '#/$defs/literal'}},
              '$defs': {'literal': {'type': 'string', 'pattern': '^0'}}, 'required': ['value']}
    tools = catalog(schema)
    contract = prepare_native_tools(tools, None, [{'role': 'user', 'content': 'go'}])
    assert contract.template_tools == tools
    contract.validate_arguments('record', {'value': '007'})
    with pytest.raises(ValueError, match='schema'):
        contract.validate_arguments('record', {'value': 'wrong'})


def test_remote_reference_fails_before_inference_without_retrieval(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network schema retrieval'))
    with pytest.raises(HTTPException, match='reference'):
        prepare_native_tools(catalog({'type': 'object', 'properties': {
            'value': {'$ref': 'https://invalid.example/schema'}}}), None, [])


@pytest.mark.parametrize('tools', [catalog() * 2, [{'type': 'web_search'}],
    [{'type': 'function', 'function': {'name': 'bad>name', 'parameters': {}}}],
    catalog({'type': 'bogus'})])
def test_invalid_catalogs_rejected(tools):
    with pytest.raises(HTTPException): prepare_native_tools(tools, None, [])


def test_no_tools_preserves_ordinary_request():
    messages = [{'role': 'user', 'content': 'hello'}]
    contract = prepare_native_tools(None, None, messages)
    assert not contract.active and not contract.enabled
    assert contract.messages == messages


@pytest.mark.parametrize('width', [1, 2, 7, 1000])
def test_visible_stream_holds_arbitrarily_split_calls_and_preserves_strings(width):
    contract = prepare_native_tools(catalog(), None, [])
    output = NativeToolOutput(contract)
    raw = 'I will record it. <tool_call>\n<function=record>\n<parameter=value>\n  007\n\n</parameter>\n</function>\n</tool_call>'
    visible = ''.join(output.feed(raw[i:i+width]) for i in range(0, len(raw), width))
    tail, calls = output.finish('stop')
    assert visible + tail == 'I will record it. '
    assert len(calls) == 1
    assert json.loads(calls[0]['function']['arguments']) == {'value': '  007\n'}


@pytest.mark.parametrize('raw', [
    '<tool_call><function=record><parameter=value>x</parameter></function>',
    '<function=record><parameter=value>x</parameter><parameter=value>y</parameter></function>',
    '<function=missing>{"value":"x"}</function>',
    '<function=record>{"value":1}</function>',
    '<function=record>{"value":"x","value":"y"}</function>',
    '<function=record>{"value":"x"}</function> trailing text',
    '<function=record><parameter=value>x</parameter> garbage </function>',
    '<tool_call><function=record>', '<tool_call',
])
def test_malformed_or_schema_invalid_output_cannot_execute(raw):
    output = NativeToolOutput(prepare_native_tools(catalog(), None, []))
    assert output.feed(raw) == ''
    with pytest.raises(ValueError): output.finish('stop')


def test_call_batch_is_validated_atomically_and_length_cannot_execute():
    output = NativeToolOutput(prepare_native_tools(catalog(), None, []))
    output.feed('<function=record>{"value":"x"}</function>')
    with pytest.raises(ValueError, match='complete stop'): output.finish('length')


def test_two_valid_calls_emit_once_in_order():
    output = NativeToolOutput(prepare_native_tools(catalog(), None, []))
    output.feed('<function=record>{"value":"first"}</function>\n<function=record>{"value":"second"}</function>')
    _, calls = output.finish('stop')
    assert [json.loads(c['function']['arguments'])['value'] for c in calls] == ['first', 'second']
    assert len({c['id'] for c in calls}) == 2


def test_literal_less_than_flushes_as_visible_text():
    output = NativeToolOutput(prepare_native_tools(catalog(), None, []))
    assert output.feed('x <') == 'x '
    assert output.finish('stop') == ('<', [])


def test_history_can_reference_a_previous_catalog():
    contract = prepare_native_tools(None, None, history())
    assert contract.active and not contract.enabled
    assert contract.messages[1]['tool_calls'][0]['function']['arguments'] == {'value': '007'}


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('prefix,off,value', [
    ('<think>Example: <function=wrong>{}</function></think>', False, '007'),
    ('', True, '<think>literal code</think>'),
    ('Example: <function=wrong>{}</function></think>', False, '<think>literal code</think>'),
])
async def test_http_calls_follow_durable_owner_and_ignore_reasoning_examples(monkeypatch, stream, prefix, off, value):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from vmlx_engine import omni_multimodal as omni
    from vmlx_engine.api.models import ChatCompletionRequest
    raw = prefix + 'I will record it. <function=record>' + json.dumps({'value': value}) + '</function>'
    entered, release, durable = threading.Event(), threading.Event(), threading.Event()
    class Dispatcher:
        _backend = 'stage1'
        def chat(self, **kwargs):
            assert kwargs['tools'] == catalog() and kwargs['tool_context'] is True
            if kwargs['prompt_rail_callback']: kwargs['prompt_rail_callback'](off)
            if kwargs['token_callback']:
                for index, char in enumerate(raw): kwargs['token_callback'](index, char)
            return {'content': raw, 'prompt_thinking_off': off, 'finish_reason': 'stop',
                    'prompt_tokens': 90, 'completion_tokens': 40, 'cached_tokens': 60}
        def finish_request_cache(self):
            entered.set()
            assert release.wait(5), 'test did not release the persistence fence'
            durable.set()
        def reset(self): pass
        def submit(self, fn, *args): return executor.submit(fn, *args)
    monkeypatch.setattr(omni, 'omni_multimodal_component_status', lambda p: {'modalities': ['text']})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, 'get', lambda *a, **kw: Dispatcher())
    request = ChatCompletionRequest(model='native', messages=[{'role': 'user', 'content': 'go'}],
                                    tools=catalog(), stream=stream)
    payloads = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        async def collect():
            response = await omni.dispatch_omni_chat_completion(request, '/unused')
            if not stream:
                assert durable.is_set()
                return response
            async for event in response.body_iterator:
                for line in event.splitlines():
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        payload = json.loads(line[6:]); payloads.append(payload)
                        for choice in payload.get('choices', []):
                            if choice.get('delta', {}).get('tool_calls'):
                                assert durable.is_set()
            return payloads
        task = asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not any(c.get('delta', {}).get('tool_calls') for p in payloads for c in p.get('choices', []))
        finally:
            release.set()
        response = await task
    if stream:
        deltas = [c.get('delta', {}) for p in response for c in p.get('choices', [])]
        calls = [call for d in deltas for call in d.get('tool_calls', [])]
        assert ''.join(d.get('content', '') for d in deltas) == 'I will record it. '
        assert ('wrong' in ''.join(d.get('reasoning_content', '') for d in deltas)) is not off
        assert response[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    else:
        calls = response['choices'][0]['message']['tool_calls']
        assert response['choices'][0]['finish_reason'] == 'tool_calls'
    assert len(calls) == 1 and calls[0]['function']['name'] == 'record'
    assert json.loads(calls[0]['function']['arguments']) == {'value': value}


def test_tool_result_batch_boundaries_and_catalog_are_bound_to_keys():
    from vmlx_engine.omni_native_prefix import source_prefix_keys
    messages = prepare_native_tools(catalog(), None, history()).messages
    keys = source_prefix_keys(messages, {}, catalog())
    assert [count for count, key in keys] == [1, 4]
    changed = catalog(); changed[0]['function']['description'] = 'new catalog'
    assert source_prefix_keys(messages, {}, changed) != keys
    messages[-1]['content'] = 'changed result'
    modified = source_prefix_keys(messages, {}, catalog())
    assert modified[0] == keys[0] and modified[-1] != keys[-1]

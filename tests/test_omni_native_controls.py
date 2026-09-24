"""Native template semantics must reach the prompt, cache and HTTP rails."""
from concurrent.futures import Future
from types import SimpleNamespace
import json
import threading

import pytest
from fastapi import HTTPException

from vmlx_engine.omni_native_controls import (
    native_template_options, has_native_thinking_directive, record_prompt_rail,
)
from vmlx_engine import omni_multimodal as omni


@pytest.mark.parametrize('kwargs', [
    {'reasoning_budget': True}, {'reasoning_budget': -1},
    {'reasoning_budget': 1.5}, {'reasoning_budget': '128'},
    {'truncate_history_thinking': 0}, {'truncate_history_thinking': 'false'},
])
def test_invalid_native_options_are_rejected(kwargs):
    with pytest.raises(HTTPException) as error:
        native_template_options(kwargs)
    assert error.value.status_code == 400


def test_native_options_preserve_defaults_and_soft_budget():
    assert native_template_options({}) == {}
    assert native_template_options({'reasoning_budget': None}) == {}
    assert native_template_options({'reasoning_budget': 0, 'truncate_history_thinking': False}) == {
        'reasoning_budget': 0, 'truncate_history_thinking': False,
    }


@pytest.mark.parametrize('role,content,expected', [
    ('system', 'Use /no_think', True), ('user', '/think now', True),
    ('user', [{'type': 'text', 'text': '/no_think now'}], True),
    ('assistant', 'I saw /think', False), ('user', 'Think clearly', False),
    ('user', [{'type': 'image_url', 'image_url': {'url': '/think.png'}}], False),
])
def test_directives_require_native_full_history_only_in_template_roles(role, content, expected):
    assert has_native_thinking_directive([{'role': role, 'content': content}]) is expected


@pytest.mark.parametrize('suffix,off', [('<think>\n', False), ('<think></think>', True)])
def test_rail_comes_from_rendered_generation_suffix(suffix, off):
    observed = []
    session = SimpleNamespace(_vmlx_prompt_rail_callback=observed.append)
    record_prompt_rail(session, 'old </think> prompt ' + suffix)
    assert observed == [off]
    assert session._vmlx_prompt_thinking_off is off


def test_unknown_template_suffix_fails_before_decode():
    with pytest.raises(ValueError, match='generation suffix'):
        record_prompt_rail(SimpleNamespace(), 'assistant:')


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('request_off,prompt_off', [(False, True), (True, False)])
async def test_http_rail_follows_prompt_before_first_token(monkeypatch, stream, request_off, prompt_off):
    raw = 'visible answer' if prompt_off else 'still reasoning'
    class Dispatcher:
        _backend = 'stage1'
        def chat(self, *, token_callback, prompt_rail_callback, template_options, **kwargs):
            assert template_options == {'reasoning_budget': 128, 'truncate_history_thinking': False}
            if prompt_rail_callback:
                prompt_rail_callback(prompt_off)
            if token_callback:
                for i, word in enumerate(raw.split(' ')):
                    token_callback(i, (' ' if i else '') + word)
            return {'content': raw, 'prompt_thinking_off': prompt_off,
                    'prompt_tokens': 12, 'completion_tokens': 2, 'finish_reason': 'length'}
        def submit(self, fn, *args):
            f = Future()
            try: f.set_result(fn(*args))
            except Exception as e: f.set_exception(e)
            return f
        def finish_request_cache(self): pass
        def reset(self): pass
    monkeypatch.setattr(omni, 'omni_multimodal_component_status', lambda p: {'modalities': ['text']})
    monkeypatch.setattr(omni.OmniMultimodalDispatcher, 'get', lambda *a, **kw: Dispatcher())
    from vmlx_engine.api.models import ChatCompletionRequest
    request = ChatCompletionRequest(model='omni', stream=stream, enable_thinking=not request_off,
        messages=[{'role': 'user', 'content': '/no_think' if prompt_off else '/think'}],
        chat_template_kwargs={'reasoning_budget': 128, 'truncate_history_thinking': False})
    response = await omni.dispatch_omni_chat_completion(request, '/unused')
    if stream:
        payloads = []
        async for event in response.body_iterator:
            for line in event.splitlines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    payloads.append(json.loads(line[6:]))
        deltas = [p['choices'][0].get('delta', {}) for p in payloads if p.get('choices')]
        content = ''.join(d.get('content', '') for d in deltas)
        reasoning = ''.join(d.get('reasoning_content', '') for d in deltas)
        assert sum(p['choices'][0].get('finish_reason') is not None for p in payloads if p.get('choices')) == 1
    else:
        message = response['choices'][0]['message']
        content, reasoning = message['content'], message.get('reasoning_content', '')
    assert content == (raw if prompt_off else '')
    assert reasoning == ('' if prompt_off else raw)


@pytest.mark.parametrize('options,text', [
    ({'reasoning_budget': 128}, 'Describe'),
    ({'truncate_history_thinking': False}, 'Describe'),
    ({}, '/no_think Describe'),
])
def test_sensitive_controls_never_restore_completed_turn_state(tmp_path, monkeypatch, options, text):
    d = omni.OmniMultimodalDispatcher.__new__(omni.OmniMultimodalDispatcher)
    class Session:
        def reset(self): self._cache = None
        def turn(self, **kwargs): pytest.fail('controls took incremental prompt path')
    d._session = Session(); d._backend = 'stage1'; d._lock = threading.Lock()
    d._scratch_dir = tmp_path; d._last_signature = None; d._disk_cache_enabled = False
    d._try_restore_session_snapshot = lambda signature: pytest.fail('restored completed-turn state')
    calls = []
    def assemble(session, messages, **kwargs):
        calls.append(kwargs['template_options'])
        record_prompt_rail(session, 'assistant<think></think>')
        return 'answer'
    monkeypatch.setattr(omni, '_run_omni_full_history', assemble)
    result = d.chat([{'role': 'user', 'content': text}], enable_thinking=False, template_options=options)
    assert calls == [options] and result['prompt_thinking_off'] is True
    assert d._last_signature is None
    assert d._last_snapshot_skip_reason == 'native_template_controls_require_exact_prefix'
    assert d._session._vmlx_prompt_rail_callback is None


def test_changed_budget_rejects_cached_tokens_and_survives_cold_retry(tmp_path):
    from copy import deepcopy
    from tests.test_omni_prefill_checkpoints import owner, history_session
    from vmlx_engine.omni_native_prefix import NativePrefillCheckpoints
    from vmlx_engine.omni_native_prompt import run_full_history

    d = owner(tmp_path)
    d._session = history_session(d._session.mlx_model)
    tokenizer = d._session.tokenizer
    original = tokenizer.apply_chat_template
    calls = []
    def render(messages, **kwargs):
        calls.append(dict(kwargs))
        messages = deepcopy(messages)
        # A soft hint belongs to the latest user, so changing it invalidates
        # cached recurrent state even when the wire messages have not changed.
        messages[-1]['content'] += str(kwargs['reasoning_budget'])
        return original(messages, **kwargs)
    tokenizer.apply_chat_template = render
    messages = [{'role': 'user', 'content': 'Describe'}]
    for budget in (128, 64):
        d._session._cache = None
        run_full_history(d._session, messages, scratch_dir=tmp_path,
            extract_parts=lambda *a: pytest.fail('text turn encoded media'),
            enable_thinking=False, max_tokens=4, temperature=0, top_p=1,
            template_options={'reasoning_budget': budget, 'truncate_history_thinking': False},
            checkpoints=NativePrefillCheckpoints(d, messages, {}, publish=True))
    assert d._session._vmlx_restored_prefix_tokens == 0
    assert d._session_l2_stats['hits'] == 0
    assert [c['reasoning_budget'] for c in calls] == [128, 128, 64, 64, 64]
    assert all(c['truncate_history_thinking'] is False for c in calls)

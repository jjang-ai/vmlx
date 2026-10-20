"""Declared mode defaults follow the effective rail and preserve caller choices."""
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def mode_server(tmp_path, monkeypatch):
    from vmlx_engine import server

    (tmp_path / 'jang_config.json').write_text(json.dumps({'chat': {
        'sampling_defaults': {'temperature': 1.0, 'top_p': .95, 'top_k': 20},
        'sampling_modes': {
            'thinking': {'temperature': 1.0, 'top_p': .95, 'presence_penalty': 0},
            'instruct_nothinking': {'temperature': .7, 'top_p': .8, 'top_k': 20,
                                   'min_p': 0, 'presence_penalty': 1.5, 'repetition_penalty': 1},
        },
    }}))
    monkeypatch.setattr(server, '_model_path', str(tmp_path))
    monkeypatch.setattr(server, '_jang_sampling_modes_cache', {})
    monkeypatch.setattr(server, '_jang_sampling_defaults_cache', {})
    monkeypatch.setattr(server, '_generation_defaults_cache', {})
    monkeypatch.setattr(server, '_native_mtp_sampling_policy', 'adaptive')
    for field in ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty']:
        monkeypatch.setattr(server, '_default_' + field, None)
    return server


@pytest.mark.parametrize('enabled,temp,penalty', [(True, 1., 0.), (False, .7, 1.5)])
def test_mode_defaults_follow_effective_rail(mode_server, enabled, temp, penalty):
    target = {'enable_thinking': enabled, 'temperature': 1., 'top_p': .95}
    mode_server._apply_bundle_sampling_mode(target, SimpleNamespace())
    assert target['temperature'] == temp
    assert target['presence_penalty'] == penalty


def test_explicit_zero_and_cli_defaults_win(mode_server, monkeypatch):
    monkeypatch.setattr(mode_server, '_default_top_p', .4)
    target = {'enable_thinking': False, 'temperature': 0., 'top_p': .4,
              'top_k': 7, 'presence_penalty': 0.}
    mode_server._apply_bundle_sampling_mode(
        target, SimpleNamespace(temperature=0., top_k=7, presence_penalty=0.),
    )
    assert target['temperature'] == 0 and target['top_p'] == .4
    assert target['top_k'] == 7 and target['presence_penalty'] == 0


def test_greedy_stays_greedy_without_inherited_filters(mode_server):
    target = {'enable_thinking': False, 'temperature': 0.}
    mode_server._apply_bundle_sampling_mode(target, SimpleNamespace(temperature=0.))
    assert target['temperature'] == 0 and target['top_p'] == 1
    assert 'top_k' not in target


def test_absent_or_unknown_mode_does_not_change_values(mode_server):
    target = {'temperature': .25, 'presence_penalty': -.5}
    mode_server._apply_bundle_sampling_mode(target, SimpleNamespace())
    assert target == {'temperature': .25, 'presence_penalty': -.5}
    mode_server._jang_sampling_modes_cache[mode_server._model_path] = {'thinking_general': {'temperature': 2}}
    target['enable_thinking'] = True
    mode_server._apply_bundle_sampling_mode(target, SimpleNamespace())
    assert target['temperature'] == .25 and target['presence_penalty'] == -.5


@pytest.mark.parametrize('route,payload', [
    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}], 'enable_thinking': False}),
    ('/v1/responses', {'input': 'Hello', 'enable_thinking': False}),
    ('/v1/messages', {'messages': [{'role': 'user', 'content': 'Hello'}], 'thinking': {'type': 'disabled'}, 'max_tokens': 16}),
    ('/api/chat', {'messages': [{'role': 'user', 'content': 'Hello'}], 'think': False}),
])
def test_four_routes_forward_mode_defaults(mode_server, monkeypatch, route, payload):
    from fastapi.testclient import TestClient
    from vmlx_engine.engine.base import GenerationOutput

    class FakeEngine:
        is_mllm = False
        preserve_native_tool_format = False
        tokenizer = SimpleNamespace(has_thinking=True)

        async def stream_chat(self, messages, **kwargs):
            captured.append(kwargs)
            yield GenerationOutput(text='ok', new_text='ok', prompt_tokens=3,
                                   completion_tokens=1, finish_reason='stop', finished=True)

    captured = []
    async def chat(*args, **kwargs):
        captured.append(kwargs['chat_kwargs'])
        return GenerationOutput(text='ok', prompt_tokens=3, completion_tokens=1)

    monkeypatch.setattr(mode_server, '_engine', FakeEngine())
    monkeypatch.setattr(mode_server, '_model_name', 'mode-test')
    monkeypatch.setattr(mode_server, '_mcp_manager', None)
    monkeypatch.setattr(mode_server, '_reasoning_parser', None)
    monkeypatch.setattr(mode_server, '_api_key', None)
    monkeypatch.setattr(mode_server, '_await_chat_with_disconnect_abort', chat)
    response = TestClient(mode_server.app).post(route, json={'model': 'mode-test', 'stream': False, **payload})
    assert response.status_code == 200, response.text
    assert len(captured) == 1
    assert captured[0]['temperature'] == .7
    assert captured[0]['top_p'] == .8
    assert captured[0]['presence_penalty'] == 1.5

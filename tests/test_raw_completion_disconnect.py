"""Real ASGI receive-channel cancellation without loading a model."""
import asyncio
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('mode', ['disconnect', 'normal', 'error', 'timeout'])
@pytest.mark.parametrize('route,mllm', [('/v1/completions', False), ('/v1/completions', True), ('/api/generate', False)])
def test_nonstream_raw_disconnect_aborts_producer(monkeypatch, route, mllm, mode):
    import vmlx_engine.server as server
    from fastapi import FastAPI

    assert Path(server.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]

    async def scenario():
        entered = asyncio.Event()
        stopped = asyncio.Event()
        aborted = []
        ids = []

        class Engine:
            is_mllm = mllm

            async def generate(self, **kwargs):
                ids.append(kwargs.get('request_id'))
                entered.set()
                try:
                    if mode == 'normal':
                        from vmlx_engine.engine.base import GenerationOutput
                        return GenerationOutput(text='ok', tokens=[1], prompt_tokens=3, completion_tokens=1, finished=True, finish_reason='stop')
                    if mode == 'error':
                        raise RuntimeError('fixture failure')
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            chat = generate

            async def abort_request(self, request_id):
                aborted.append(request_id)
                return True

        monkeypatch.setattr(server, 'get_engine', lambda: Engine())
        monkeypatch.setattr(server, '_resolve_model_name', lambda: 'disconnect-fixture')
        monkeypatch.setattr(server, '_is_loaded_dsv4_model', lambda _: False)
        monkeypatch.setattr(server, '_is_loaded_mimo_v2_model', lambda _: False)
        if mode == 'timeout':
            monkeypatch.setattr(server, '_default_timeout', 0.02)
        app = FastAPI()
        app.post('/v1/completions')(server.create_completion)
        app.post('/api/generate')(server.ollama_generate)
        body = json.dumps({'model': 'disconnect-fixture', 'prompt': 'hello', 'stream': False, 'raw': True, 'max_tokens': 10}).encode()
        sent_body = False
        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await entered.wait()
            if mode != 'disconnect':
                await asyncio.Event().wait()
            return {'type': 'http.disconnect'}
        messages = []
        async def send(message):
            messages.append(message)
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST', 'scheme': 'http', 'path': route, 'raw_path': route.encode(), 'query_string': b'', 'headers': [(b'content-type', b'application/json')], 'client': ('127.0.0.1', 123), 'server': ('127.0.0.1', 80)}
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(asyncio.shield(task), 1)
            assert stopped.is_set()
            assert ids[0].startswith('cmpl-')
            if mode in ('disconnect', 'timeout'):
                assert len(aborted) == 1 and aborted == ids
            else:
                assert aborted == []
            status = next(x['status'] for x in messages if x['type'] == 'http.response.start')
            assert status == {'disconnect': 499, 'normal': 200, 'error': 500, 'timeout': 504}[mode]
            if mode == 'normal':
                payload = json.loads(b''.join(x.get('body', b'') for x in messages))
                assert (payload['response'] if route == '/api/generate' else payload['choices'][0]['text']) == 'ok'
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())

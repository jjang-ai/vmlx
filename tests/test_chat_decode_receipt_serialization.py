"""Native-free tests of the production Chat serializer's negotiated receipt."""
import ast
import json
import math
from pathlib import Path
import types

import pytest


@pytest.mark.parametrize('telemetry', [False, True])
@pytest.mark.parametrize('first_count', [4, 12])
def test_terminal_receipt_is_negotiated_and_excludes_initial_burst(monkeypatch, telemetry, first_count):
    source = ast.parse(Path('vmlx_engine/server.py').read_text())
    snapshot = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_usage_snapshot')
    stream = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'stream_chat_completion')
    serializer = next(n for n in stream.body if isinstance(n, ast.FunctionDef) and n.name == '_dump_chat_chunk')
    context = types.ModuleType('vmlx_engine.context_limits')
    context.pop_context_clamp = lambda _: None
    context.pop_effort_substitution = lambda _: None
    monkeypatch.setitem(__import__('sys').modules, 'vmlx_engine.context_limits', context)
    ns = dict(json=json, math=math, include_usage=True, response_id='test',
              _prefill_usage_extension=telemetry, _prefill_usage=None,
              _decode_first_ts=10., _decode_last_ts=12., _decode_first_count=first_count,
              # the serializer reads reasoning-token usage from the stream's last output (None: no reasoning span)
              _reasoning_usage_details=lambda _output: None, last_output=None)
    exec(compile(ast.Module(body=[snapshot, serializer], type_ignores=[]), '<production-chat-serializer>', 'exec'), ns)
    dump = ns['_dump_chat_chunk']
    assert json.loads(dump({'choices': [{'delta': {'content': 'x'}}]}))['usage'] is None
    usage = {'prompt_tokens': 20, 'completion_tokens': 12, 'total_tokens': 32}
    payload = json.loads(dump({'choices': [], 'usage': usage.copy()}, terminal_usage=True))
    expected = dict(usage)
    if telemetry and first_count < 12:
        expected['vmlx_decode'] = {'tokens': 8, 'seconds': 2., 'tokens_per_second': 4.}
    assert payload == {'choices': [], 'usage': expected}

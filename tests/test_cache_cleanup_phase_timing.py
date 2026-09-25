"""CPU-only owning-method checks: tracing must not change cleanup semantics."""
import ast
import gc
import os
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'vmlx_engine'


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    # store_cache occurs in two classes; only the lifecycle wrapper owns the fence.
    if 'store_cache' in names:
        selected = [n for n in selected if n.name != 'store_cache' or '_settle_native_write_fence' in ast.unparse(n)]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('fail_gc', [False, True])
@pytest.mark.parametrize('owner', ['store', 'terminal'])
def test_phase_trace_preserves_cleanup_order(monkeypatch, enabled, fail_gc, owner):
    monkeypatch.setenv('VMLX_CACHE_CLEANUP_PHASE_TRACE', '1' if enabled else '0')
    order, logs, clocks = [], [], []
    def collect():
        order.append('gc')
        if fail_gc:
            raise RuntimeError('collection failed')
        return 7
    monkeypatch.setattr(gc, 'collect', collect)
    def clock():
        clocks.append(1)
        return len(clocks) / 1000
    ns = dict(os=os, time=SimpleNamespace(perf_counter=clock), logger=SimpleNamespace(info=lambda *a: logs.append(a), debug=lambda *a: None), HAS_MLX=True, mx=SimpleNamespace(clear_cache=lambda: order.append('clear')))
    load_functions(ROOT/'prefix_cache.py', {'_cleanup_phase_start', '_cleanup_phase_finish', 'store_cache'}, ns)
    if owner == 'store':
        def store(*args, **kwargs):
            order.append('store')
            kwargs['_write_fence']['disk_only_fallbacks'] = True
            return 'result'
        obj = SimpleNamespace(_shape_scoped_cache_extra_keys=lambda *a, **k: None, _store_cache_impl=store, _settle_native_write_fence=lambda *a: order.append('fence'))
        assert ns['store_cache'](obj, 'req', [1], []) == 'result'
        assert order == ['store', 'fence', 'gc'] + ([] if fail_gc else ['clear'])
    else:
        ns['clear_mlx_memory_cache'] = lambda **k: order.append('clear')
        load_functions(ROOT/'mllm_scheduler.py', {'_cleanup_finished_after_terminal_dispatch'}, ns)
        obj = SimpleNamespace(_queue_lock=Lock(), _cleanup_finished=lambda ids: order.append('cleanup'), running=set())
        ns['_cleanup_finished_after_terminal_dispatch'](obj, {'req'})
        assert order == ['cleanup', 'gc', 'clear']
    assert bool(clocks) == enabled
    assert bool(logs) == enabled
    if enabled:
        assert len(clocks) == 2 * len(logs)
        assert logs[0][-2] is (not fail_gc)
        assert logs[0][-1] == (None if fail_gc else 7)


def test_trace_logging_failure_does_not_mask_cleanup():
    def fail(*args):
        raise RuntimeError('logger unavailable')
    ns = dict(os=os, time=SimpleNamespace(perf_counter=lambda: 1), logger=SimpleNamespace(info=fail))
    load_functions(ROOT/'prefix_cache.py', {'_cleanup_phase_finish'}, ns)
    ns['_cleanup_phase_finish'](0, 'site', 'req', False)

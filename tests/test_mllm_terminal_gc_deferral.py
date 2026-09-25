"""Idle terminal GC repayment, including failure and native-owner lifetime."""
import gc
import os
import weakref
from threading import Lock
from types import SimpleNamespace

import pytest

from test_cache_cleanup_phase_timing import ROOT, load_functions


def namespace(order):
    ns = dict(os=os, logger=SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None), HAS_MLX=True, mx=SimpleNamespace(clear_cache=lambda: order.append('store_clear')), _cleanup_phase_start=lambda: None, _cleanup_phase_finish=lambda *a: None, clear_mlx_memory_cache=lambda **k: order.append('outer_clear'))
    load_functions(ROOT/'prefix_cache.py', {'store_cache'}, ns)
    load_functions(ROOT/'mllm_scheduler.py', {'_cleanup_finished_after_terminal_dispatch'}, ns)
    return ns


@pytest.mark.parametrize('running,finished,expected', [({'r'}, {'r'}, True), ({'r','other'}, {'r'}, False), ({'r','other'}, {'r','other'}, False), (set(), {'r'}, False), (set(), set(), False)])
def test_deferral_is_single_terminal_only(running, finished, expected):
    order = []
    ns = namespace(order)
    received = []
    obj = SimpleNamespace(_queue_lock=Lock(), running=dict.fromkeys(running))
    def cleanup(ids, **kwargs):
        received.append(kwargs.get('_defer_post_fence_gc', False))
        for rid in ids:
            obj.running.pop(rid, None)
    obj._cleanup_finished = cleanup
    ns['_cleanup_finished_after_terminal_dispatch'](obj, finished)
    assert received == [expected]


@pytest.mark.parametrize('fail_cleanup', [False, True])
def test_repayment_follows_frame_lifetime_and_preserves_failure(monkeypatch, fail_cleanup):
    order, refs = [], []
    ns = namespace(order)
    real_collect = gc.collect
    def collect():
        order.append('gc')
        return real_collect()
    monkeypatch.setattr(gc, 'collect', collect)
    class Payload:
        pass
    obj = SimpleNamespace(_queue_lock=Lock(), running={'r': object()})
    error = RuntimeError('bookkeeping failure')
    def cleanup(ids, **kwargs):
        payload = Payload()
        payload.cycle = payload
        refs.append(weakref.ref(payload))
        def store(*a, **kw):
            order.append('store')
            kw['_write_fence']['disk_only_fallbacks'] = True
            return 'stored'
        cache = SimpleNamespace(_shape_scoped_cache_extra_keys=lambda *a, **k: None, _store_cache_impl=store, _settle_native_write_fence=lambda *a: order.append('fence'))
        assert ns['store_cache'](cache, 'r', [1], [], defer_post_fence_gc=kwargs['_defer_post_fence_gc']) == 'stored'
        assert refs[0]() is payload
        if fail_cleanup:
            raise error
        obj.running.clear()
    obj._cleanup_finished = cleanup
    if fail_cleanup:
        try:
            ns['_cleanup_finished_after_terminal_dispatch'](obj, {'r'})
        except RuntimeError as caught:
            assert caught is error
        else:
            raise AssertionError('original exception lost')
        # Traceback still owns payload: collection cannot free reachable locals.
        assert refs[0]() is not None
        error.__traceback__ = None
        real_collect()
        assert refs[0]() is None
    else:
        ns['_cleanup_finished_after_terminal_dispatch'](obj, {'r'})
        assert refs[0]() is None  # Outer collection is after the successful frame.
    assert order == ['store', 'fence', 'store_clear', 'gc', 'outer_clear']


def test_cleanup_error_survives_repayment_error(monkeypatch):
    ns = namespace([])
    original = ValueError('original')
    def cleanup(*a, **k):
        raise original
    def fail_clear(**k):
        raise RuntimeError('clear failed')
    ns['clear_mlx_memory_cache'] = fail_clear
    obj = SimpleNamespace(_queue_lock=Lock(), running={'r': object()}, _cleanup_finished=cleanup)
    try:
        ns['_cleanup_finished_after_terminal_dispatch'](obj, {'r'})
    except ValueError as caught:
        assert caught is original
    else:
        raise AssertionError('original exception lost')


@pytest.mark.parametrize('defer', [False, True])
@pytest.mark.parametrize('failure', ['producer', 'settle', 'refused'])
def test_store_fence_failures_keep_original_outcomes(monkeypatch, defer, failure):
    order = []
    ns = namespace(order)
    monkeypatch.setattr(gc, 'collect', lambda: order.append('gc'))
    original = RuntimeError(failure)
    def seal(fence_id, *, producer_aborted):
        order.append(('seal', fence_id, producer_aborted))
        return True
    disk = SimpleNamespace(seal_write_fence=seal)
    def store(*args, **kwargs):
        order.append('store')
        fence = kwargs['_write_fence']
        fence.update(disk_only_fallbacks=True, disk_store=disk, fence_id='f')
        if failure == 'producer':
            raise original
        return 'stored'
    def settle(request_id, fence):
        order.append('settle')
        if failure == 'settle':
            raise original
        fence['seal_attempted'] = True
        fence['disk_only_publication_refused'] = True
    obj = SimpleNamespace(_shape_scoped_cache_extra_keys=lambda *a, **k: None, _store_cache_impl=store, _settle_native_write_fence=settle)
    if failure == 'refused':
        assert ns['store_cache'](obj, 'r', [1], [], defer_post_fence_gc=defer) is None
        assert order == ['store', 'settle'] + ([] if defer else ['gc']) + ['store_clear']
    else:
        try:
            ns['store_cache'](obj, 'r', [1], [], defer_post_fence_gc=defer)
        except RuntimeError as caught:
            assert caught is original
        else:
            raise AssertionError('store failure was suppressed')
        assert order == ['store'] + (['settle'] if failure == 'settle' else []) + [('seal', 'f', True)]


def test_outer_collection_failure_still_clears(monkeypatch):
    order = []
    ns = namespace(order)
    def collect():
        order.append('gc')
        raise RuntimeError('gc failed')
    monkeypatch.setattr(gc, 'collect', collect)
    obj = SimpleNamespace(_queue_lock=Lock(), running={'r': object()})
    def cleanup(*a, **k):
        assert k['_defer_post_fence_gc'] is True
        obj.running.clear()
    obj._cleanup_finished = cleanup
    ns['_cleanup_finished_after_terminal_dispatch'](obj, {'r'})
    assert order == ['gc', 'outer_clear']

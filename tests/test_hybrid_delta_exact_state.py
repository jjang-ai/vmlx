"""Tiny native-array regression for hybrid delta ownership; no model weights."""
import ast
import inspect
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import Mock

import pytest

mx = pytest.importorskip('mlx.core')
from mlx_lm.models.cache import KVCache
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache


@pytest.fixture
def owner():
    source = Path(os.environ.get('VMLX_DELTA_TEST_SOURCE', Path(__file__).parents[1] / 'vmlx_engine/mllm_batch_generator.py'))
    tree = ast.parse(source.read_text())
    names = {'_derive_hybrid_companion_delta', '_prefill_for_clean_path_dependent_cache'}
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Any=Any, List=List, Optional=Optional, mx=mx, logger=logging.getLogger(__name__), __package__='vmlx_engine',
        _HYBRID_BASE_SPLICE=True, _cache_requires_one_shot_rederive=lambda *a, **kw: False,
        _validate_prompt_cache=lambda cache, **kw: all(getattr(c, 'idx_keys', None) is not None for c in cache))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    cls = type('DeltaOwner', (), {name: namespace[name] for name in names})
    obj = cls()
    obj._is_hybrid = True
    obj._hybrid_kv_positions = [0]
    obj._model_type = 'qwen4_exp'
    obj._tokens_contain_media_placeholders = lambda tokens: 99 in tokens
    obj._complete_hybrid_base_from_companion = Mock(return_value=None)
    obj._store_companion_from_clean_pass = Mock()
    obj.prefill_step_size = 1
    return obj


def strict_call(owner, tokens, cache, count, positions=None):
    method = owner._prefill_for_clean_path_dependent_cache
    kwargs = {}
    if 'require_base' in inspect.signature(method).parameters:
        kwargs = dict(require_base=True, delta_position_ids=positions)
    return method(tokens, cache, count, cache_extra_keys=('media',), **kwargs)


class Language:
    def __init__(self, linear=False):
        self.layers = [SimpleNamespace(is_linear=linear)]
        self.calls = []
        self._position_ids = 'saved-position'
        self._rope_deltas = 'saved-delta'

    def __call__(self, ids, **kwargs):
        self.calls.append((ids.tolist(), kwargs))


def test_failed_splice_never_replays_media_or_stores(owner):
    owner.language_model = Language(linear=True)
    owner._cache_model = SimpleNamespace(make_cache=lambda: [KVCache()])
    assert strict_call(owner, [99, 1, 2, 3], [KVCache()], 2) is None
    assert owner.language_model.calls == []
    owner._store_companion_from_clean_pass.assert_not_called()


def test_explicit_compressed_positions_sliced_per_chunk_and_restored(owner):
    owner.language_model = Language()
    positions = mx.array([[[7, 8]], [[7, 8]], [[7, 8]]])
    result = strict_call(owner, [99, 1, 2, 3], [KVCache()], 2, positions)
    assert result is not None
    assert [call[0] for call in owner.language_model.calls] == [[[2]], [[3]]]
    assert [call[1].get('position_ids').tolist() for call in owner.language_model.calls] == [[[[7]], [[7]], [[7]]], [[[8]], [[8]], [[8]]]]
    assert owner.language_model._position_ids == 'saved-position'
    assert owner.language_model._rope_deltas == 'saved-delta'
    owner._store_companion_from_clean_pass.assert_called_once()


def sparse():
    cache = MiniMaxM3SparseCache()
    cache.keys = mx.arange(16).reshape(1, 1, 8, 2)
    cache.values = cache.keys + 100
    cache.idx_keys = mx.arange(24).reshape(1, 1, 8, 3)
    cache.offset = 6
    cache._idx_offset = 6
    return cache


def setup_derive(owner, layer, media=False):
    owner.block_aware_cache = SimpleNamespace(reconstruct_cache=lambda table: [layer])
    capture = Mock(return_value=None)
    owner._prefill_for_clean_path_dependent_cache = capture
    owner._prefill_for_clean_ssm = capture
    tokens = [99 if media else 1, 2, 3, 4, 5, 6]
    request = SimpleNamespace(request_id='tiny', input_ids=mx.array([tokens]), image_grid_thw='current-grid')
    def positions(req, ids, ck):
        assert req is not request
        assert req.image_grid_thw == 'current-grid'
        req._mrope_full_position_ids = mx.array([[[0, 1, 2, 3, 1, 2]]] * 3)
        return True
    owner._mrope_tail_position_ids = positions
    return capture, tokens, request


def test_sparse_slice_keeps_all_native_lanes_and_parent(owner):
    layer = sparse()
    capture, tokens, request = setup_derive(owner, layer)
    before = [x.tolist() for x in (layer.keys, layer.values, layer.idx_keys)]
    owner._derive_hybrid_companion_delta(request, tokens, 6, 4, object())
    capture.assert_called_once()
    clone = capture.call_args.args[1][0]
    assert clone.offset == clone._idx_offset == 4
    for original, copied in zip((layer.keys, layer.values, layer.idx_keys), (clone.keys, clone.values, clone.idx_keys)):
        assert copied is not None
        assert copied.tolist() == original[..., :4, :].tolist()
    assert before == [x.tolist() for x in (layer.keys, layer.values, layer.idx_keys)]
    assert layer.offset == layer._idx_offset == 6


@pytest.mark.parametrize('extent', [0, 3, 5, 9])
def test_sparse_uninitialized_or_inconsistent_extent_declines(owner, extent):
    layer = sparse()
    layer._idx_offset = extent
    capture, tokens, request = setup_derive(owner, layer)
    assert owner._derive_hybrid_companion_delta(request, tokens, 6, 4, object()) is None
    capture.assert_not_called()


def test_media_delta_uses_current_full_request_positions_without_mutation(owner):
    capture, tokens, request = setup_derive(owner, sparse(), media=True)
    owner._derive_hybrid_companion_delta(request, tokens, 6, 4, object())
    capture.assert_called_once()
    assert capture.call_args.kwargs['delta_position_ids'].tolist() == [[[1, 2]]] * 3
    assert capture.call_args.kwargs['require_base'] is True
    assert not hasattr(request, '_mrope_full_position_ids')


def test_media_without_exact_positions_declines(owner):
    capture, tokens, request = setup_derive(owner, sparse(), media=True)
    owner._mrope_tail_position_ids = lambda *args: False
    owner._derive_hybrid_companion_delta(request, tokens, 6, 4, object())
    capture.assert_not_called()


def test_missing_sparse_index_declines(owner):
    layer = sparse()
    layer.idx_keys = None
    capture, tokens, request = setup_derive(owner, layer)
    owner._derive_hybrid_companion_delta(request, tokens, 6, 4, object())
    capture.assert_not_called()


def test_forward_failure_restores_module_state_and_never_stores(owner):
    class FailingLanguage(Language):
        def __call__(self, ids, **kwargs):
            self._position_ids = 'mutated'
            self._rope_deltas = 'mutated'
            raise RuntimeError('injected forward failure')
    owner.language_model = FailingLanguage()
    assert strict_call(owner, [99, 1, 2, 3], [KVCache()], 2, mx.zeros((3, 1, 2))) is None
    assert owner.language_model._position_ids == 'saved-position'
    assert owner.language_model._rope_deltas == 'saved-delta'
    owner._store_companion_from_clean_pass.assert_not_called()


def test_invalid_position_length_never_forwards_or_stores(owner):
    owner.language_model = Language()
    assert strict_call(owner, [99, 1, 2, 3], [KVCache()], 2, mx.zeros((3, 1, 1))) is None
    assert owner.language_model.calls == []
    owner._store_companion_from_clean_pass.assert_not_called()

import importlib

import pytest

from vmlx_engine.utils.mamba_cache import ensure_mamba_support


def test_single_sequence_merge_keeps_native_kv_cache_objects():
    ensure_mamba_support()

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    gen = importlib.import_module("mlx_lm.generate")
    kv = KVCache()
    rot = RotatingKVCache(max_size=128)

    merged = gen._merge_caches([[kv, rot]])

    assert merged == [kv, rot]
    assert merged[0] is kv
    assert merged[1] is rot


def test_native_kv_single_sequence_batch_api_is_explicit():
    ensure_mamba_support()

    from mlx_lm.models.cache import KVCache

    kv = KVCache()

    kv.filter([0])
    assert kv.extract(0) is kv
    kv.prepare(right_padding=[0])
    kv.finalize()

    with pytest.raises(NotImplementedError):
        kv.filter([0, 1])
    with pytest.raises(IndexError):
        kv.extract(1)
    with pytest.raises(NotImplementedError):
        kv.prepare(right_padding=[1])


def test_turboquant_kv_single_sequence_batch_api_is_explicit():
    ensure_mamba_support()

    pytest.importorskip("jang_tools.turboquant.cache")
    from jang_tools.turboquant.cache import TurboQuantKVCache

    tq = TurboQuantKVCache(key_dim=8, value_dim=8)

    tq.filter([0])
    assert tq.extract(0) is tq
    tq.prepare(right_padding=[0])
    tq.finalize()

    with pytest.raises(NotImplementedError):
        tq.filter([0, 1])
    with pytest.raises(IndexError):
        tq.extract(1)

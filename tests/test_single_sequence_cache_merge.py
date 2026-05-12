import importlib

import mlx.core as mx
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

    assert getattr(tq, "_vmlx_batch_api", None) == "turboquant_kv_v1"

    tq.filter([0])
    extracted_empty = tq.extract(0)
    assert isinstance(extracted_empty, TurboQuantKVCache)
    assert extracted_empty is not tq
    assert extracted_empty.offset == 0
    tq.prepare(right_padding=[0])
    tq.finalize()

    empty_batch = TurboQuantKVCache(key_dim=8, value_dim=8)
    empty_batch.filter([0, 1])
    assert empty_batch._idx == 0
    assert empty_batch.offset.shape == (2,)


def test_turboquant_kv_extract_slices_single_batch_row():
    ensure_mamba_support()

    pytest.importorskip("jang_tools.turboquant.cache")
    from jang_tools.turboquant.cache import TurboQuantKVCache

    tq = TurboQuantKVCache(key_dim=8, value_dim=8)
    keys = mx.arange(24, dtype=mx.float16).reshape(1, 1, 3, 8)
    values = mx.arange(24, dtype=mx.float16).reshape(1, 1, 3, 8) + 100
    tq.update_and_fetch(keys, values)

    extracted = tq.extract(0)

    assert isinstance(extracted, TurboQuantKVCache)
    assert extracted is not tq
    assert extracted.offset == 3
    assert extracted.keys.shape == (1, 1, 3, 8)
    assert extracted.values.shape == (1, 1, 3, 8)
    assert mx.array_equal(extracted.keys, keys).item()
    assert mx.array_equal(extracted.values, values).item()

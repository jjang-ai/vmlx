"""DeepSeek-V4.1 in the memory-aware RAM prefix cache (CPU only, synthetic DeepseekV41Cache states).

The store path used to rebuild every DSV4-family layer as a V4.0 ``DeepseekV4Cache`` and assign the V4.1 state to it,
which raised ("too many values to unpack"), was logged at debug, and dropped every V4.1 store. A DeepseekV41Cache list
is now stored as a copy only when it already sits at the N-1 prompt boundary (it cannot be trimmed), and a hit hands
back an independent copy (the stored entry is never aliased by the decoding request).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import mlx.core as mx

mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")

from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig, clone_dsv41_cache_at  # noqa: E402
from vmlx_engine.scheduler import Scheduler  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _caches(n_tokens, ratios=(0, 2, 1), seed=0):
    rng = np.random.default_rng(seed)
    seq = mlx_model._SeqState()
    caches = [mlx_model.DeepseekV41Cache(seq, i, r) for i, r in enumerate(ratios)]
    seq.caches = caches
    for c in caches:
        c.window = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)
        if c._ratio:
            rows = n_tokens // c._ratio
            c.compress_kv.append(mx.array(rng.standard_normal((1, rows, 64)).astype(np.float32)).astype(mx.bfloat16))
            c.index_k.append(mx.array(rng.standard_normal((1, rows, 64)).astype(np.float32)).astype(mx.bfloat16))
    seq.pos = n_tokens
    seq.history = rng.integers(0, 50000, n_tokens).astype(np.int64)
    return caches


def _same_state(a, b):
    assert [type(x).__name__ for x in a] == [type(x).__name__ for x in b]
    for x, y in zip(a, b):
        assert x.offset == y.offset
        assert json.loads(x.meta_state) == json.loads(y.meta_state)
        for u, v in zip(x.state, y.state):
            assert np.array_equal(np.array(u), np.array(v))


def test_store_truncation_copies_a_boundary_snapshot():
    prompt_len = 23  # a short prompt: key = 22 tokens, the snapshot sits at 22
    snap = _caches(prompt_len - 1)
    out = Scheduler._truncate_cache_to_prompt_length(snap, prompt_len)
    assert out is not None
    assert all(type(c).__name__ == "DeepseekV41Cache" for c in out)
    assert all(o is not s for o, s in zip(out, snap))
    _same_state(out, snap)
    # one shared sequence state in the copy, independent of the source
    assert len({id(c._seq) for c in out}) == 1 and out[0]._seq is not snap[0]._seq
    out[0]._seq.pos += 5
    assert snap[0].offset == prompt_len - 1


def test_store_truncation_refuses_a_cache_off_the_boundary():
    live = _caches(40)  # post-decode live cache: cannot be trimmed back to 22
    assert Scheduler._truncate_cache_to_prompt_length(live, 23) is None
    assert clone_dsv41_cache_at(live[:1] + [object()], 40) is None  # mixed list


def test_memory_aware_round_trip_isolated():
    cache = MemoryAwarePrefixCache(model=object(), config=MemoryCacheConfig(max_memory_mb=256))
    tokens = list(range(500, 523))  # 23 prompt tokens
    snap = _caches(len(tokens) - 1)
    stored = Scheduler._truncate_cache_to_prompt_length(snap, len(tokens))
    assert cache.store(tokens[:-1], stored)

    hit, remaining = cache.fetch(tokens)
    assert hit is not None, "a DSV4.1 entry must be reusable, not forced to a miss"
    assert remaining == tokens[-1:]
    assert all(h is not s for h, s in zip(hit, stored))
    _same_state(hit, stored)
    # decoding into the hit never touches the stored entry
    hit[0]._seq.pos += 3
    hit[1].window = None
    again, _ = cache.fetch(tokens)
    _same_state(again, snap)


def test_memory_aware_reverse_match_is_a_miss():
    """A shorter request than the stored key would need a trim, which V4.1 cannot do."""
    cache = MemoryAwarePrefixCache(model=object(), config=MemoryCacheConfig(max_memory_mb=256))
    tokens = list(range(500, 541))
    stored = Scheduler._truncate_cache_to_prompt_length(_caches(len(tokens) - 1), len(tokens))
    assert cache.store(tokens[:-1], stored)
    hit, remaining = cache.fetch(tokens[:20])
    assert hit is None and remaining == tokens[:20]


def test_paged_state_dict_position():
    c = _caches(30)
    sd = {"state": c[0].state, "meta_state": c[0].meta_state, "class_name": "DeepseekV41Cache"}
    assert Scheduler._dsv41_state_dict_pos(sd) == 30
    assert Scheduler._dsv41_state_dict_pos({**sd, "meta_state": (c[0].meta_state,)}) == 30
    assert Scheduler._dsv41_state_dict_pos({"meta_state": "not json"}) is None

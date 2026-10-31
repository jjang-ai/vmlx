"""Memory-aware prefix cache: partial (longest-common-prefix) reuse (CPU only).

A stored key and a request that share a prefix and then differ (same system prompt, same document with another
question, an edited last message) used to miss on the RAM-only tier: only exact, forward (stored key is a prefix
of the request) and reverse (request is a prefix of the stored key) matches were looked up. A cache that can be cut
is now reused up to the common prefix; one that cannot (DeepSeek-V4.1) falls back to the other matches.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig, _common_prefix_len


@pytest.fixture(autouse=True)
def _cpu_only():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _kv(n, layers=2, seed=0):
    out = []
    for i in range(layers):
        c = KVCache()
        k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed + i))
        c.update_and_fetch(k, k + 1)
        mx.eval(c.keys, c.values)
        out.append(c)
    return out


def test_common_prefix_len():
    assert _common_prefix_len((1, 2, 3, 4), (1, 2, 9)) == 2
    assert _common_prefix_len((1, 2), (1, 2)) == 2
    assert _common_prefix_len((5,), (1,)) == 0


def test_partial_hit_cut_at_the_common_prefix():
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    stored = list(range(100, 197))                       # 97 tokens (not a block multiple)
    src = _kv(len(stored))
    assert mem.store(stored, src)
    req = stored[:61] + [5, 6, 7, 8, 9]                   # same first 61 tokens, then differs
    cache, remaining = mem.fetch(req)
    assert mem.last_fetch_kind == "partial"
    assert cache is not None and remaining == req[61:]
    for c, s in zip(cache, src):
        assert c.offset == 61 and c.keys[..., : c.offset, :].shape[2] == 61   # buffer rows == logical offset
        assert mx.array_equal(c.keys[..., :61, :], s.keys[..., :61, :])
    # the stored entry is untouched
    assert all(s.offset == len(stored) for s in src)


def test_forward_match_preferred_when_longer():
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    a = list(range(1, 41))
    b = a + list(range(500, 560))                        # 100 tokens, extends a
    mem.store(a, _kv(len(a)))
    mem.store(b[:90] + [7, 7, 7], _kv(93, seed=3))       # shares 90 with the request
    req = b[:95]
    cache, remaining = mem.fetch(req)
    assert mem.last_fetch_kind == "partial" and len(remaining) == 5        # 90 > forward 40
    mem2 = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    mem2.store(a, _kv(len(a)))
    mem2.store(a[:20] + [9] * 30, _kv(50, seed=5))       # shares only 20 (< forward 40)
    cache, remaining = mem2.fetch(a + [1, 2, 3])
    assert mem2.last_fetch_kind == "forward" and remaining == [1, 2, 3]


def test_untrimmable_cache_falls_back(monkeypatch):
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    short = list(range(1, 21))
    mem.store(short, _kv(20))
    mem.store(short + list(range(50, 90)), _kv(60, seed=2))
    real = mem._truncate_cache
    req = short + list(range(50, 70)) + [1, 1]
    # the long entry cannot be cut at the 40-token common prefix (e.g. a DeepseekV41Cache list)
    monkeypatch.setattr(mem, "_truncate_cache", lambda cache, n, *a, **k: None if n == 40 else real(cache, n, *a, **k))
    cache, remaining = mem.fetch(req)
    assert mem.last_fetch_kind == "forward" and remaining == req[20:]


def test_partial_cut_runs_on_the_stream_owner_and_never_copies_cumulative_state(monkeypatch):
    """The cut array must be born on the generation thread (else decode fails with "no Stream(gpu, 0) in current
    thread", measured on MiniMax-M3.1), and a partial hit must not allow the cumulative-state copy."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-worker",
                            initializer=lambda: mx.set_default_device(mx.cpu))
    mem.set_clone_executor(ex)
    stored = list(range(100, 160))
    mem.store(stored, _kv(len(stored)))
    seen = []
    real = mem._truncate_cache

    def spy(cache, n, allow=False):
        seen.append((threading.current_thread().name, n, allow))
        return real(cache, n, allow)
    monkeypatch.setattr(mem, "_truncate_cache", spy)
    cache, remaining = mem.fetch(stored[:33] + [1, 2])
    ex.shutdown()
    assert cache is not None and remaining == [1, 2]
    assert seen == [(seen[0][0], 33, False)] and seen[0][0].startswith("llm-worker")

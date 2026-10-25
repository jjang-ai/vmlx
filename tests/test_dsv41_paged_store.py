"""DeepSeek-V4.1 paged prefix cache round-trip (CPU only, synthetic DeepseekV41Cache states).

- a stored DSV4.1 entry restores bit-exactly (4-bit pools as stored, format v2);
- only the terminal block of a table carries the composite state (no per-block duplication), and
  that state is ~4-bit size;
- a state whose sequence position disagrees with the block table is rejected (miss), never used.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx


mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")

from vmlx_engine.paged_cache import PagedCacheManager  # noqa: E402
from vmlx_engine.prefix_cache import BlockAwarePrefixCache  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    """CPU only (another process owns the GPU); restore the previous default afterwards."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)

BLOCK = 64


def _caches(n_tokens, ratios=(0, 2, 1), seed=0):
    rng = np.random.default_rng(seed)
    seq = mlx_model._SeqState()
    caches = [mlx_model.DeepseekV41Cache(seq, i, r) for i, r in enumerate(ratios)]
    seq.caches = caches
    for c in caches:
        c.window = mx.array(rng.standard_normal((1, 128, 512)).astype(np.float32)).astype(mx.bfloat16)
        if c._ratio:
            rows = n_tokens // c._ratio
            c.compress_kv.append(mx.array(rng.standard_normal((1, rows, 512)).astype(np.float32)).astype(mx.bfloat16))
            c.index_k.append(mx.array(rng.standard_normal((1, rows, 128)).astype(np.float32)).astype(mx.bfloat16))
    seq.pos = n_tokens
    seq.history = rng.integers(0, 50000, n_tokens).astype(np.int64)
    return caches


def _states(caches):
    return [
        {
            "state": c.state,
            "meta_state": c.meta_state,
            "class_name": "DeepseekV41Cache",
            "compress_ratio": c.compress_ratio,
            "sliding_window": c.sliding_window,
        }
        for c in caches
    ]


def _u16(a):
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)


def _store(n_tokens, pos=None):
    caches = _caches(n_tokens)
    if pos is not None:
        caches[0]._seq.pos = pos
    mgr = PagedCacheManager(block_size=BLOCK, max_blocks=400)
    bac = BlockAwarePrefixCache(object(), mgr)
    tokens = list(range(1000, 1000 + n_tokens))
    table = bac.store_cache("store-req", tokens, _states(caches))
    return bac, mgr, tokens, caches, table


def test_round_trip_bit_exact_and_terminal_block_only():
    n = 10 * BLOCK
    bac, mgr, tokens, src, table = _store(n)
    assert table is not None
    fetched, remaining = bac.fetch_cache("fetch-req", tokens + [7, 8, 9])
    assert fetched is not None and fetched.num_tokens == n
    rec = bac.reconstruct_cache(fetched)
    assert rec is not None and len(rec) == len(src)
    for a, b in zip(src, rec):
        assert type(b).__name__ == "DeepseekV41Cache"
        assert int(b.offset) == n and b.compress_ratio == a.compress_ratio
        assert np.array_equal(_u16(a.window), _u16(b.window))
        for qa, qb in ((a.compress_kv, b.compress_kv), (a.index_k, b.index_k)):
            assert qa.n == qb.n
            if qa.n:
                for x, y in ((qa.wq, qb.wq), (qa.scales, qb.scales), (qa.biases, qb.biases)):
                    assert np.array_equal(_u16(x[:, : qa.n]), _u16(y[:, : qb.n]))
    assert np.array_equal(rec[0]._seq.history, src[0]._seq.history)

    # Only the terminal block carries the composite state; all others are pending markers.
    tagged = []
    for bid in fetched.block_ids:
        blk = mgr.allocated_blocks.get(bid) if hasattr(mgr, "allocated_blocks") else None
        data = getattr(blk, "cache_data", None) if blk is not None else None
        if data is None:
            continue
        tags = {entry[0] for entry in data if isinstance(entry, (tuple, list)) and entry}
        tagged.append(tags)
    assert len(tagged) == len(fetched.block_ids) == n // BLOCK
    assert all(t == {"deepseek_v4_pending"} for t in tagged[:-1])
    assert tagged[-1] == {"deepseek_v4"}


def test_stored_state_is_four_bit_size():
    n = 16 * BLOCK
    src = _caches(n)
    total = sum(int(a.nbytes) for c in src for a in c.state)
    fixed = len(src) * 128 * 512 * 2 + n * 4  # windows + layer-0 history
    pools = total - fixed
    # ratio-2 layer: n/2 rows, ratio-1 layer: n rows; each row (256+32) + (64+8) bytes at 4 bit
    assert pools == (n // 2 + n) * (288 + 72)


def test_offset_mismatch_is_rejected():
    n = 6 * BLOCK
    bac, mgr, tokens, src, table = _store(n, pos=n + 1)  # state one token past its key
    fetched, _ = bac.fetch_cache("fetch-bad", tokens + [1])
    assert fetched is not None and fetched.num_tokens == n
    assert bac.reconstruct_cache(fetched) is None

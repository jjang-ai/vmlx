# SPDX-License-Identifier: Apache-2.0
"""The served MiniMax-M3 module and the prefix-cache restore share ONE MiniMaxM3SparseCache class.

minimax_m3_register loads models/minimax_m3/minimax_m3.py as ``mlx_lm.models.minimax_m3_vl`` (the path every served
bundle takes). Under that name the file's relative ``.cache`` import resolves to mlx-lm's own cache module, and the
old fallback loaded models/minimax_m3/cache.py a second time as a top-level ``cache`` module. Caches rebuilt by the
prefix-cache restore (vmlx_engine.models.minimax_m3.cache) were then not instances of the served model's class: the
Indexer skipped the index lane on every continued prefill, and the DSpark lane's first verify raised "sparse index
lane desynchronised from K/V" (the live M3.1 tool loop, 2026-10-28: every cached turn came back empty).

Tiny synthetic M3 of tests/test_minimax_m3_dspark.py (MSA on layers 1-3, 32-token blocks, top-6); no real weights."""

from __future__ import annotations

import sys

import pytest

mx = pytest.importorskip("mlx.core")

import tests.test_minimax_m3_dspark as T  # noqa: E402


def _served_module():
    from vmlx_engine.models.minimax_m3.minimax_m3_register import register_minimax_m3_runtime

    register_minimax_m3_runtime()
    return sys.modules["mlx_lm.models.minimax_m3_vl"]


def test_served_module_shares_the_package_cache_class():
    from vmlx_engine.models.minimax_m3 import cache as pkg_cache

    served = _served_module()
    assert served.MiniMaxM3SparseCache is pkg_cache.MiniMaxM3SparseCache
    assert not [m for m in ("cache", "msa", "source_format", "m3_affine2_switch") if m in sys.modules]


@pytest.mark.parametrize("cut, n", [(200, 301), (230, 233)])
def test_restored_cache_continues_in_lockstep_on_the_served_model(monkeypatch, cut, n):
    """A model built from the served module: cold prefill [0, cut) + [cut, n) vs a cache restored at ``cut`` (the
    package clone the prefix cache uses) + [cut, n). Every sparse layer keeps idx_keys in lockstep with K/V, and the
    continuation's logits and final cache state equal the cold run's bit for bit. n is past the sparse switch
    (6 x 32 = 192 keys), so block selection reads the index lane; (230, 233) is the short gpl+1 tail of an exact-K hit."""
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.models.minimax_m3.cache import clone_minimax_m3_sparse

    served = _served_module()
    monkeypatch.setattr(T, "M", served)
    model = T._model(seed=5)
    ids = T._ids(n, 9)

    cold = model.make_cache()
    model(mx.array(ids[:cut])[None], cache=cold)
    out_cold = model(mx.array(ids[cut:])[None], cache=cold)

    warm = model.make_cache()
    model(mx.array(ids[:cut])[None], cache=warm)
    restored = []
    for c in warm:
        if hasattr(c, "idx_keys"):
            restored.append(clone_minimax_m3_sparse(c, cut, copy_fn=lambda a: mx.array(a)))
        else:
            k, v = c.state
            fresh = KVCache()
            fresh.state = (mx.array(k), mx.array(v))
            restored.append(fresh)
    out_hit = model(mx.array(ids[cut:])[None], cache=restored)
    mx.eval(out_cold, out_hit)

    sparse = [c for c in restored if hasattr(c, "idx_keys")]
    assert len(sparse) == 3
    assert [(int(c._idx_offset), int(c.offset)) for c in sparse] == [(n, n)] * 3
    assert T._eq(out_hit, out_cold)
    for a, b in zip(restored, cold):
        for x, y in zip(a.state, b.state):
            assert (x is None) == (y is None)
            if x is not None:
                assert T._eq(x, y)

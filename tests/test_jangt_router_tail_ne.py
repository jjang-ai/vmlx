"""router_tail (jangt/router.py) == the reference sigmoid-noaux_tc router for every supported expert count.

GLM-5.3 has 288 experts = 9 simdgroups; the scratch arrays were sized for 8 (N0.5's 256), so the 9th simdgroup
wrote out of bounds and the selection was wrong (2026-10-10: KL 0.11 vs the reference router on GLM decode).
"""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.jangt.router import router_tail


def _ref(logits, bias, k, norm, scaling):
    s = 1.0 / (1.0 + np.exp(-logits))
    c = s + bias
    idx = np.argsort(-c, axis=-1, kind="stable")[:, :k]        # ties -> lower id, like the kernel
    w = np.take_along_axis(s, idx, axis=-1)
    if norm:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx, w * scaling


@pytest.mark.parametrize("ne", [256, 288, 384, 1024])
def test_router_tail_matches_reference(ne):
    rng = np.random.default_rng(ne)
    logits = rng.normal(size=(5, ne)).astype(np.float32) * 3
    bias = rng.normal(size=(ne,)).astype(np.float32) * 0.5
    i, w = router_tail(mx.array(logits), mx.array(bias), 8, True, 2.5)
    ri, rw = _ref(logits, bias, 8, True, 2.5)
    i, w = np.array(i), np.array(w)
    for t in range(5):
        assert set(i[t]) == set(ri[t]), (ne, t, sorted(i[t]), sorted(ri[t]))
        o1, o2 = np.argsort(i[t]), np.argsort(ri[t])
        assert np.allclose(w[t][o1], rw[t][o2], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("T", [1, 4, 9])
def test_router_logits_matches_fp32_matmul(T):
    from vmlx_engine.jangt.router import router_logits
    rng = np.random.default_rng(T)
    x = mx.array(rng.normal(size=(T, 4096)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.normal(size=(288, 4096)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    got = np.array(router_logits(x, w))
    ref = np.array(x.astype(mx.float32) @ w.astype(mx.float32).T)
    assert got.shape == (T, 288) and np.max(np.abs(got - ref)) <= 1e-4 * np.max(np.abs(ref))

"""Fused mHC prefill kernels (metal/glm5_mhc_prefill_fused) == the stock prefill graph.

Slab sizes: 5 (smallest row count routed here; DFlash2 verify blocks), 100, 2048 (a production prefill chunk).
post/comb: against a float64 reference (the stock graph's fp32 matmul runs in TF32 under MLX defaults: 5e-4 off
float64, the kernel 3e-7); normalized: against the float64 path (collapse rounded to bf16, RMSNorm, weight) within two bf16 ULP, more than one in <= 0.1% of elements (the stock graph
is 1.5 ULP off it under TF32, the kernel 0.5); placed streams: vs the stock graph, one bf16 ULP, <= 1e-6 of the
elements may differ by 2 (an fp32 accumulation-order flip ahead of two bf16 roundings).
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import glm5_mhc_prefill_fused as F
from vmlx_engine.models.glm5_next.glm5_next import HyperConnection, RMSNorm, hc_place

ARGS = SimpleNamespace(hc_mult=4, hidden_size=4096, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5)


def _mods(seed):
    mx.random.seed(seed)
    hc = HyperConnection(ARGS)
    hc.hc_fn = (mx.random.normal((24, 16384)) * 0.02).astype(mx.bfloat16)
    hc.hc_base = mx.random.normal((24,)) * 0.5
    hc.hc_scale = mx.array([0.7, 0.9, 1.3])
    norm = RMSNorm(4096, 1e-5)
    norm.weight = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
    mx.eval(hc.parameters(), norm.parameters())
    return hc, norm


def _ulp(a, b, frac=0.0, ulps=1, abs_rel=0.0):
    """<= `ulps` exact bf16 ULPs everywhere, plus abs_rel x row RMS for near-zero values (cancellation: one absolute
    error is hundreds of ULPs there); more than one ULP in at most `frac` of the elements."""
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    m = np.maximum(np.abs(a), np.abs(b))
    tol = np.where(m > 0, 2.0 ** (np.floor(np.log2(np.maximum(m, 1e-30))) - 7), 1e-30)   # exact bf16 ULP
    rms = np.sqrt((a * a).mean(-1, keepdims=True))
    bad = np.abs(a - b) > tol
    ok = bad.mean() <= frac and bool(np.all(np.abs(a - b) <= ulps * tol + abs_rel * rms))
    return ok, bad.mean(), np.abs(a - b).max()


def _f64_post_comb(s, hc):
    f = np.array(s.astype(mx.float32))[0].reshape(s.shape[1], -1).astype(np.float64)
    f = f / np.sqrt((f * f).mean(-1, keepdims=True) + hc.rms_eps)
    mix = f @ np.array(hc.hc_fn.astype(mx.float32)).astype(np.float64).T
    b = np.array(hc.hc_base).astype(np.float64); sc = np.array(hc.hc_scale).astype(np.float64); eps = hc.eps
    post = 2 / (1 + np.exp(-(mix[:, 4:8] * sc[1] + b[4:8])))
    c = (mix[:, 8:] * sc[2] + b[8:]).reshape(-1, 4, 4)
    c = np.exp(c - c.max(-1, keepdims=True)); c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(hc.iters - 1):
        c = c / (c.sum(-1, keepdims=True) + eps); c = c / (c.sum(-2, keepdims=True) + eps)
    pre = 1 / (1 + np.exp(-(mix[:, :4] * sc[0] + b[:4]))) + eps
    return post, c, pre


@pytest.mark.parametrize("T", [5, 100, 2048])
def test_mix_norm_matches_stock(monkeypatch, T):
    hc, norm = _mods(T)
    s = (mx.random.normal((1, T, 4, 4096)) * 3).astype(mx.bfloat16)
    monkeypatch.setenv("VMLX_GLM5_FUSED_MHC", "0")
    hc._fused_decode = hc._fused_prefill = False
    p0, c0, x0 = hc(s); n0 = norm(x0)
    p1, c1, n1 = F.glm5_mhc_mix_norm_prefill(s, hc, norm)
    mx.eval(p0, c0, n0, p1, c1, n1)
    post, comb, pre = _f64_post_comb(s, hc)
    assert np.abs(np.array(p1)[0] - post).max() < 1e-5 and np.abs(np.array(c1)[0] - comb).max() < 1e-5
    S = np.array(s.astype(mx.float32))[0].astype(np.float64)
    col = mx.array(((pre[:, :, None] * S).sum(1)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    col = np.array(col).astype(np.float64)
    nrm = col / np.sqrt((col ** 2).mean(-1, keepdims=True) + norm.eps)
    nrm = np.array(mx.array(nrm.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)).astype(np.float64)
    ref = mx.array((nrm * np.array(norm.weight.astype(mx.float32)))[None].astype(np.float32))
    ok, frac, mx_ = _ulp(ref, n1, frac=1e-3, ulps=2.01, abs_rel=5e-3)   # collapse bf16 rounding flip + output rounding
    assert ok, (frac, mx_)


@pytest.mark.parametrize("T", [5, 100, 2048])
def test_place_matches_stock(monkeypatch, T):
    hc, _ = _mods(T + 1)
    s = (mx.random.normal((1, T, 4, 4096)) * 3).astype(mx.bfloat16)
    out = mx.random.normal((1, T, 4096)).astype(mx.bfloat16)
    post, comb, _ = hc(s)
    monkeypatch.setenv("VMLX_GLM5_MHC_PREFILL_FUSED", "0")
    ref = hc_place(post, comb, out, s)
    got = F.glm5_hc_place_prefill(post, comb, out, s)
    mx.eval(ref, got)
    ok, frac, mx_ = _ulp(ref, got, frac=1e-6, ulps=2.01, abs_rel=5e-3)   # cancellation: a + m with large opposite terms
    assert ok, (frac, mx_)


def test_routing_uses_fused_kernels(monkeypatch):
    """Fail closed: the comparison is moot if hc_place never routes prefill rows to the kernel."""
    hc, _ = _mods(9)
    s = mx.random.normal((1, 8, 4, 4096)).astype(mx.bfloat16)
    post, comb, x = hc(s)
    monkeypatch.setenv("VMLX_GLM5_MHC_PREFILL_FUSED", "1")
    n0 = F._OBSERVED["place"]
    mx.eval(hc_place(post, comb, x, s))
    assert F._OBSERVED["place"] == n0 + 1

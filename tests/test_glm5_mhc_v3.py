"""mHC v3 kernels (1024-thread projection + SIMD-parallel 4x4 Sinkhorn) match v2 up to summation order.

v3 is the default (VMLX_GLM5_MHC_V3); v2 is the rollback. Both kernels are covered: the AR compound
mHC + weighted RMSNorm (glm5_mhc_norm) and the generic decode/verify path (glm5_mhc_decode, 1-4 rows).
Tolerances: post/comb are fp32 (reordered sums: 1e-5), normalized/collapsed are bf16 (one ULP).
"""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import glm5_mhc_decode as D
from vmlx_engine.metal import glm5_mhc_norm as N

KW = dict(rms_eps=1e-5, sink_eps=1e-6, iterations=20)


def _inputs(rows=1, seed=0, scale=1.0):
    mx.random.seed(seed)
    streams = (mx.random.normal((1, rows, 4, 4096)) * scale).astype(mx.bfloat16)
    hc_fn = (mx.random.normal((24, 16384)) * 0.02).astype(mx.bfloat16)
    hc_base = mx.random.normal((24,)) * 0.5
    hc_scale = mx.array([0.7, 0.9, 1.3])
    weight = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
    return streams, hc_fn, hc_base, hc_scale, weight


def _ulp_ok(a, b):
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    ulp = np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -7 + 1e-30
    return bool(np.all(np.abs(a - b) <= ulp))


@pytest.mark.parametrize("seed,scale", [(0, 1.0), (1, 30.0), (2, 1e-3)])
def test_norm_kernel_v3_matches_v2(monkeypatch, seed, scale):
    s, f, b, sc, w = _inputs(1, seed, scale)
    out = {}
    for v in ("0", "1"):
        monkeypatch.setenv("VMLX_GLM5_MHC_V3", v)
        out[v] = N.glm5_mhc_norm_decode(s, f, b, sc, w, norm_eps=1e-5, enabled=True, **KW)
        assert out[v] is not None
        mx.eval(out[v])
    for i in (0, 1):
        assert float(mx.abs(out["0"][i] - out["1"][i]).max()) < 1e-5
    assert _ulp_ok(out["0"][2], out["1"][2])


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_generic_kernel_v3_matches_v2(monkeypatch, rows):
    s, f, b, sc, _ = _inputs(rows, rows)
    out = {}
    for v in ("0", "1"):
        monkeypatch.setenv("VMLX_GLM5_MHC_V3", v)
        out[v] = D.glm5_mhc_decode(s, f, b, sc, enabled=True, verify_enabled=True, **KW)
        assert out[v] is not None
        mx.eval(out[v])
    for i in (0, 1):
        assert float(mx.abs(out["0"][i] - out["1"][i]).max()) < 1e-5
    assert _ulp_ok(out["0"][2], out["1"][2])


def test_v3_comb_is_doubly_stochastic(monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_MHC_V3", "1")
    s, f, b, sc, _ = _inputs(4, 7)
    _, comb, _ = D.glm5_mhc_decode(s, f, b, sc, enabled=True, verify_enabled=True, **KW)
    c = np.array(comb)
    assert np.allclose(c.sum(-2), 1.0, atol=1e-3) and np.allclose(c.sum(-1), 1.0, atol=1e-3)


def test_switch_selects_distinct_kernels(monkeypatch):
    """Fail closed: the comparison above is vacuous if both settings build the same kernel."""
    assert D._EPILOGUE_SOURCE_V3 != D._EPILOGUE_SOURCE and N._SOURCE_V3 != N._SOURCE
    assert "simd_shuffle_xor" in D._EPILOGUE_SOURCE_V3 and "simd_shuffle_xor" not in D._EPILOGUE_SOURCE
    monkeypatch.setenv("VMLX_GLM5_MHC_V3", "0"); assert not D.glm5_mhc_v3_requested()
    monkeypatch.delenv("VMLX_GLM5_MHC_V3"); assert D.glm5_mhc_v3_requested()

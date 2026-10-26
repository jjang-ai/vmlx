"""Fused JANGT routed + mxfp8 shared MoE decode (vmlx_engine/jangt/fused_moe.py) is bit-identical to the stock path
(JTMixedSwitchGLU.routed + the shared expert's mxfp8 w1/w3/w2 with clamp and SwiGLU), each row run alone, at 1-8
rows; trellis K = 1 / 1.5 / 2, with and without outlier columns, clamped and plain SwiGLU; input widths with masked
tail blocks (D = 1280, F = 768: neither a multiple of 512)."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.jangt import fused_moe as FM  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

D, F, E = 1280, 768, 12


def _mx8(o, i, key):
    lin = nn.QuantizedLinear(i, o, bias=False, group_size=32, bits=8, mode="mxfp8")
    w = (mx.random.normal((o, i), key=key) * 0.03).astype(mx.bfloat16)
    lin.weight, lin.scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    return lin


def _layer(K, Kd, n_out, limit, seed):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, F, E, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": Kd}, n_out, n_out, limit)
    for p in (m.gate_proj, m.up_proj, m.down_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            p.jt_ocol = mx.array(np.tile(np.arange(n_out, dtype=np.int32) * 7, (E, 1)))
            p.jt_wout = mx.array(rng.standard_normal(p.jt_wout.shape).astype(np.float16) * 0.05)
    m.down_proj.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=m.down_proj.jt_cs.shape).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su
    k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
    shared = SimpleNamespace(w1=_mx8(F, D, k1), w3=_mx8(F, D, k2), w2=_mx8(D, F, k3))
    return m, shared


@pytest.mark.parametrize("K,Kd", [(1, 1.5), (1.5, 1.5), (2, 2.5)])
@pytest.mark.parametrize("n_out,limit", [(0, 10.0), (4, 10.0), (3, 0.0)])
@pytest.mark.parametrize("R", [1, 2, 5, 8])
def test_fused_moe_bit_identical_to_stock_rows(K, Kd, n_out, limit, R):
    FM._STATE["disabled"] = False
    FM._STATE["checked"].clear()
    m, shared = _layer(K, Kd, n_out, limit, seed=int(K * 10) + R + n_out)
    rng = np.random.default_rng(R)
    x = mx.array((rng.standard_normal((R, D)) * 0.7).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.choice(E, 6, replace=False) for _ in range(R)]).astype(np.int32))
    w = mx.array(rng.uniform(0.05, 0.5, size=(R, 6)).astype(np.float32))
    y = FM._run(x, idx, w, m, shared, limit)
    ref = mx.concatenate([FM._stock_row(x[i:i + 1], idx[i:i + 1], w[i:i + 1], m, shared, limit) for i in range(R)], 0)
    mx.eval(y, ref)
    assert np.abs(np.array(ref.astype(mx.float32))).max() > 0
    assert mx.array_equal(y, ref).item(), float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item())
    # the public entry point admits it (self-check passes) and returns the same array
    y2 = FM.jt_moe_decode(x, idx, w, m, shared, limit)
    assert y2 is not None and mx.array_equal(y2, ref).item()


def test_not_eligible_returns_none():
    m, shared = _layer(1, 1.5, 0, 10.0, seed=1)
    x = mx.zeros((9, D), mx.bfloat16)
    assert FM.jt_moe_decode(x, mx.zeros((9, 6), mx.int32), mx.zeros((9, 6)), m, shared, 10.0) is None   # > 8 rows
    aff = SimpleNamespace(w1=nn.QuantizedLinear(D, F, bias=False, group_size=64, bits=8), w3=shared.w3, w2=shared.w2)
    x = mx.zeros((1, D), mx.bfloat16)
    assert FM.jt_moe_decode(x, mx.zeros((1, 6), mx.int32), mx.zeros((1, 6)), m, aff, 10.0) is None     # affine shared

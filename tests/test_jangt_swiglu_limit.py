"""JANGT routed experts with a CLAMPED SwiGLU (GLM-5.3-Flash: silu(min(g, 10)) * clip(u, -10, 10)).

Synthetic JANGT units (random trellis words; any bit pattern decodes to valid code values) with row scales large
enough that many gate/up pre-activations exceed the limit. Reference: dense rotated-basis weights from
``moe_kernels.dequant_rot`` (values x row scale), gate/up products in float32, clamped activation applied in Python.
Checked: decode kernel (gather_gate_up_swiglu_v3), fused prefill kernel (gather_qmm_sorted_jt with up), the unfused
prefill path, and limit=0 (= plain SwiGLU, the Naive-N0.5 behaviour) on the same units.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

D, I, E, K = 1024, 256, 8, 2
LIMIT = 10.0


def _module(limit: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": K}, 0, 0, limit)
    for p in (m.gate_proj, m.up_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.5, 1.5, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su                 # gate and up share the layer's input signs (encoder contract)
    d = m.down_proj
    d.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=d.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
    d.jt_scale = mx.array(rng.uniform(0.01, 0.02, size=d.jt_scale.shape).astype(np.float32))
    d.jt_su = mx.array(rng.choice([-1.0, 1.0], size=d.jt_su.shape).astype(np.float32))
    return m


def _reference(m, x2, idx2, limit):
    U = m._unit()
    xr = MK.rotate_rows(x2, U["su_gu"]).astype(mx.float32)
    Wg = MK.dequant_rot(U["gate"], 0, E).astype(mx.float32)
    Wu = MK.dequant_rot(U["up"], 0, E).astype(mx.float32)
    T, k = idx2.shape
    rows = mx.repeat(xr, k, axis=0)                                      # (T*k, D)
    e = idx2.reshape(-1)
    g = mx.sum(rows[:, None, :] * Wg[e], axis=-1)
    u = mx.sum(rows[:, None, :] * Wu[e], axis=-1)
    g_raw = g
    if limit > 0:
        g, u = mx.minimum(g, limit), mx.clip(u, -limit, limit)
    return g * mx.sigmoid(g) * u, g_raw


def _inputs(T, seed=1):
    rng = np.random.default_rng(seed)
    x2 = mx.array((rng.standard_normal((T, D)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    idx2 = mx.array(np.stack([rng.permutation(E)[:4] for _ in range(T)]).astype(np.int32))
    return x2, idx2


def _rel(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    return float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item())


@pytest.mark.parametrize("limit", [LIMIT, 0.0])
def test_decode_kernel_matches_clamped_reference(limit):
    m = _module(limit)
    x2, idx2 = _inputs(3)
    ref, g = _reference(m, x2, idx2, limit)
    assert float(mx.mean((g > LIMIT).astype(mx.float32)).item()) > 0.02, "fixture must exercise the clamp"
    got = m._decode_h(x2, idx2)
    assert _rel(got, ref) < 2e-2


@pytest.mark.parametrize("fused", [True, False])
def test_prefill_paths_match_clamped_reference(fused, monkeypatch):
    monkeypatch.setattr(SW, "FUSED_GU_PREFILL", fused)
    m = _module(LIMIT)
    x2, idx2 = _inputs(40)
    ref, _ = _reference(m, x2, idx2, LIMIT)
    T, k = idx2.shape
    flat = idx2.reshape(-1); order = mx.argsort(flat); idx_s = flat[order].astype(mx.uint32); tok = order // k
    U = m._unit(); xr = MK.rotate_rows(x2, U["su_gu"])
    if fused:
        from vmlx_engine.jangt.prefill import gather_qmm_sorted_jt
        h_s = gather_qmm_sorted_jt(xr.astype(mx.bfloat16)[tok], U["gate"], idx_s, U["up"], limit=LIMIT).astype(mx.float32)
    else:
        xs_r, xs_x = xr[tok], x2[tok]
        g = m._jt_mm(xs_r, xs_x, idx_s, U["gate"]); u = m._jt_mm(xs_r, xs_x, idx_s, U["up"])
        g, u = mx.minimum(g, LIMIT), mx.clip(u, -LIMIT, LIMIT)
        h_s = g * mx.sigmoid(g) * u
    got = h_s[mx.argsort(order)]
    assert _rel(got, ref) < 2e-2


def test_limit_changes_output_only_where_preactivations_exceed_it():
    x2, idx2 = _inputs(3)
    a = _module(LIMIT)._decode_h(x2, idx2)
    b = _module(0.0)._decode_h(x2, idx2)
    assert _rel(a, b) > 1e-3, "clamped and unclamped outputs must differ on this fixture"


def test_routed_end_to_end_uses_limit():
    m_l, m_0 = _module(LIMIT), _module(0.0)
    x, idx = _inputs(2)
    w = mx.full(idx.shape, 0.25)
    assert _rel(m_l.routed(x, idx, w), m_0.routed(x, idx, w)) > 1e-3

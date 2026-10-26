"""Grouped few-row JANGT decode (each expert decoded once per pass of the rows routed to it) is bit-identical to the v3
per-(row, slot) kernels: gate/up SwiGLU (plain and clamped, with and without outlier columns, trellis K = 1 / 1.5 / 2 /
2.5 / 3) and the weighted down projection (slot-ordered combine), at 1-16 rows with heavily shared routing, any rows per
pass, with input widths that are NOT multiples of 512 (masked tail block: D = 1152, I = 640)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

D, I, E = 1152, 640, 12


def _module(K, Kd, n_out: int, limit: float, seed: int):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": Kd}, n_out, n_out, limit)
    for p in (m.gate_proj, m.up_proj, m.down_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            p.jt_ocol = mx.array(np.tile(np.arange(n_out, dtype=np.int32) * 7, (E, 1)))
            p.jt_wout = mx.array(rng.standard_normal(p.jt_wout.shape).astype(np.float16) * 0.05)
    if hasattr(m.down_proj, "jt_cs"):
        m.down_proj.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=m.down_proj.jt_cs.shape).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su
    return m


def _routing(T, k, seed):
    """Rows drawn from a small expert pool so most experts are shared by several rows (and some rows repeat sets)."""
    rng = np.random.default_rng(seed)
    pool = min(E, k + 3)
    idx = np.stack([rng.choice(pool, size=k, replace=False) for _ in range(T)]).astype(np.int32)
    if T >= 3:
        idx[2] = idx[0]                                  # an identical expert set
    x = (rng.standard_normal((T, D)) * 0.5).astype(np.float32)
    x[0, :16] *= 3000.0                                  # one lane over the fp16 range guard's threshold
    w = rng.uniform(0.05, 0.4, size=(T, k)).astype(np.float32)
    return mx.array(x).astype(mx.bfloat16), mx.array(idx), mx.array(w)


@pytest.mark.parametrize("K", [1, 1.5, 2, 2.5, 3])
@pytest.mark.parametrize("n_out,limit", [(0, 0.0), (3, 0.0), (0, 7.0), (5, 10.0)])
@pytest.mark.parametrize("T,k", [(1, 4), (2, 6), (3, 4), (5, 6), (8, 8), (16, 4)])
@pytest.mark.parametrize("rp", [1, 2, 4])
def test_grouped_decode_is_bit_identical_to_v3(K, n_out, limit, T, k, rp):
    Kd = 1.5 if K < 2 else 2.5
    m = _module(K, Kd, n_out, limit, seed=int(K * 10) + T + k)
    x, idx, w = _routing(T, k, seed=T * 31 + k)
    U = m._unit()
    xr = MK.rotate_rows(x, U["su_gu"])
    h_ref = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, limit=limit)
    h_grp = MK.gather_gate_up_swiglu_grouped(xr, x, idx, U, out_dtype=mx.float32, limit=limit, rows_per_pass=rp)
    mx.eval(h_ref, h_grp)
    assert np.isfinite(np.array(h_ref)).all() and np.abs(np.array(h_ref)).max() > 0      # non-empty baseline
    assert np.array_equal(np.array(h_ref), np.array(h_grp))
    hr = MK.rotate_rows(h_ref, U["sv_dn"], idx.reshape(-1))
    y_ref = MK.gather_down_weighted_v3(hr, h_ref, idx, w, U)
    y_grp = MK.gather_down_weighted_grouped(hr, h_ref, idx, w, U, rows_per_pass=rp)
    mx.eval(y_ref, y_grp)
    assert np.abs(np.array(y_ref)).max() > 0
    assert np.array_equal(np.array(y_ref), np.array(y_grp))


@pytest.mark.parametrize("T", [2, 6])
def test_routed_grouped_route_matches_v3_route(T, monkeypatch):
    """JTMixedSwitchGLU.routed with the grouped decode route == the v3 route, bit for bit (multi-row decode calls)."""
    m = _module(1.5, 1.5, 3, 10.0, seed=7)
    x, idx, w = _routing(T, 6, seed=11)
    monkeypatch.setenv("JANGT_GROUPED_DECODE", "0")
    y0 = m.routed(x, idx, w)
    monkeypatch.setenv("JANGT_GROUPED_DECODE", "1")
    y1 = m.routed(x, idx, w)
    mx.eval(y0, y1)
    assert np.abs(np.array(y0.astype(mx.float32))).max() > 0
    assert np.array_equal(np.array(y0.astype(mx.float32)), np.array(y1.astype(mx.float32)))

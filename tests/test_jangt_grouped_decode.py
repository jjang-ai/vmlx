# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped few-row JANGT decode (vmlx_engine/jangt/grouped_decode.py) is bit-identical to the per-(row, slot)
v3 kernels it replaces in a verify: gate / up + SwiGLU (plain SiLU, clamped SiLU, SwiGLU-OAI) and the weighted down
(slot-ordered combine), trellis K = 1 / 1.5 / 2 / 2.5 / 3, with and without outlier columns, input widths with and
without a kin % 512 tail, 1..16 rows with distinct, shared and fully repeated experts, every rows-per-pass / output-
rows-per-simdgroup shape. Control: perturbing one row's accumulation in the grouped kernel is caught."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt import grouped_decode as G  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402

E = 12


def _module(D, I, Kgu, Kdn, n_out, limit, act, seed):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": Kgu}, {"mode": "jangt", "kbits": Kdn}, n_out, n_out,
                            limit, act, 1.702 if act == "oai" else 0.0)
    for p, n_in in ((m.gate_proj, D), (m.up_proj, D), (m.down_proj, I)):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            if p is m.down_proj:
                cols = np.stack([rng.choice(n_in, n_out, replace=False) for _ in range(E)]).astype(np.int32)
                cols[1, -1] = -1                                     # padding entry
            else:
                cols = np.tile(rng.choice(n_in, n_out, replace=False), (E, 1)).astype(np.int32)
            p.jt_ocol = mx.array(cols)
            p.jt_wout = mx.array((rng.standard_normal(p.jt_wout.shape) * 0.05).astype(np.float16))
        if p is m.down_proj:
            p.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=(E, n_in)).astype(np.float32))
    m.up_proj.jt_su = m.gate_proj.jt_su
    m.up_proj.jt_ocol = m.gate_proj.jt_ocol
    return m


def _routing(T, k, seed, mode):
    rng = np.random.default_rng(seed)
    if mode == "same":                                       # every row picks the same experts
        idx = np.tile(rng.choice(E, k, replace=False), (T, 1))
    elif mode == "distinct":                                 # as few repeats as E allows
        idx = np.stack([(np.arange(k) + r * k) % E for r in range(T)])
    else:                                                    # a small pool: most experts shared by several rows
        pool = min(E, k + 3)
        idx = np.stack([rng.choice(pool, size=k, replace=False) for _ in range(T)])
        if T >= 3:
            idx[2] = idx[0][::-1]
    return mx.array(idx.astype(np.int32)), mx.array(rng.uniform(0.05, 0.4, size=(T, k)).astype(np.float32)), None


def _x(T, D, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((T, D)) * 0.5).astype(np.float32)
    x[0, :16] *= 3000.0                                      # one lane chunk past the fp16 range guard
    return mx.array(x).astype(mx.bfloat16)


def _check(m, x, idx, w, rp, o):
    U = m._unit()
    xr = MK.rotate_rows(x, U["su_gu"])
    h_ref = MK.gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.float32, limit=m.limit, act=m.act, alpha=m.alpha)
    h_grp = G.gate_up_swiglu_grouped(xr, x, idx, U, out_dtype=mx.float32, limit=m.limit, act=m.act, alpha=m.alpha,
                                     rows_per_pass=rp, out_rows=o)
    hr = MK.rotate_rows(h_ref, U["sv_dn"], idx.reshape(-1))
    y_ref = MK.gather_down_weighted_v3(hr, h_ref, idx, w, U)
    y_grp = G.down_weighted_grouped(hr, h_ref, idx, w, U, rows_per_pass=rp, out_rows=o)
    mx.eval(h_ref, h_grp, y_ref, y_grp)
    assert float(mx.abs(h_ref).max()) > 0 and float(mx.abs(y_ref).max()) > 0
    return bool(mx.array_equal(h_ref, h_grp).item()), bool(mx.array_equal(y_ref, y_grp).item())


@pytest.mark.parametrize("Kgu,Kdn", [(1, 1.5), (1.5, 2.5), (2, 2), (2.5, 1), (3, 2.5)])
@pytest.mark.parametrize("n_out,limit,act", [(0, 0.0, "silu"), (3, 7.0, "oai"), (2, 10.0, "silu")])
def test_grouped_equals_v3_m3_widths(Kgu, Kdn, n_out, limit, act):
    """M3.1-like widths (kin multiple of 512) at 1..16 rows, shared routing."""
    m = _module(1024, 512, Kgu, Kdn, n_out, limit, act, seed=int(Kgu * 10 + Kdn))
    for T in (1, 2, 5, 8, 16):
        idx, w, _ = _routing(T, 4, seed=T, mode="pool")
        assert _check(m, _x(T, 1024, T), idx, w, 0, 0) == (True, True), T


@pytest.mark.parametrize("mode", ["same", "distinct", "pool"])
@pytest.mark.parametrize("rp,o", [(1, 4), (2, 2), (4, 4), (8, 1), (3, 2)])
def test_grouped_equals_v3_tails_and_shapes(mode, rp, o):
    """Input widths with a kin % 512 tail (gate/up 1280, down 640), every pass / output-row shape."""
    m = _module(1280, 640, 1.5, 2.5, 2, 7.0, "oai", seed=rp * 10 + o)
    for T in (1, 3, 7, 16):
        idx, w, _ = _routing(T, 4, seed=100 + T, mode=mode)
        assert _check(m, _x(T, 1280, T + 50), idx, w, rp, o) == (True, True), (T, mode, rp, o)


def test_grouped_rows_are_independent_of_the_window():
    """Row r computed alone == row r inside a 16-row window (the decode-step identity the verify relies on)."""
    m = _module(1024, 512, 1.5, 2.5, 1, 7.0, "oai", seed=7)
    U = m._unit()
    idx, w, _ = _routing(16, 4, seed=9, mode="pool")
    x = _x(16, 1024, 9)
    xr = MK.rotate_rows(x, U["su_gu"])
    h = G.gate_up_swiglu_grouped(xr, x, idx, U, limit=7.0, act="oai", alpha=1.702)
    hr = MK.rotate_rows(h, U["sv_dn"], idx.reshape(-1))
    y = G.down_weighted_grouped(hr, h, idx, w, U)
    for r in (0, 5, 15):
        xr1 = MK.rotate_rows(x[r:r + 1], U["su_gu"])
        h1 = G.gate_up_swiglu_grouped(xr1, x[r:r + 1], idx[r:r + 1], U, limit=7.0, act="oai", alpha=1.702)
        hr1 = MK.rotate_rows(h1, U["sv_dn"], idx[r:r + 1].reshape(-1))
        y1 = G.down_weighted_grouped(hr1, h1, idx[r:r + 1], w[r:r + 1], U)
        assert bool(mx.array_equal(h1, h[4 * r:4 * r + 4]).item()) and bool(mx.array_equal(y1, y[r:r + 1]).item())


def test_control_perturbed_row_arithmetic_is_caught(monkeypatch):
    """Control: the grouped gate/up with the second row's accumulation order changed (steps 0 and 1 swapped) differs
    from v3 on the rows of shared experts, so the equality checks above bite."""
    m = _module(1024, 512, 1.5, 2.5, 0, 7.0, "oai", seed=3)
    real = G._apply

    def swapped(xh, acc):
        if xh != "xh1":
            return real(xh, acc)
        order = [1, 0, 2, 3, 4, 5, 6, 7]
        return " ".join(f"{acc} = fma({xh}[{i}], pv[{i}], {acc});" for i in order)

    monkeypatch.setattr(G, "_apply", swapped)
    G._gu_grouped.cache_clear()
    monkeypatch.setattr(MK, "_kname", lambda K: str(K).replace(".", "p") + "_ctl")
    try:
        idx, w, _ = _routing(8, 4, seed=1, mode="same")
        h_ok, _ = _check(m, _x(8, 1024, 1), idx, w, 4, 4)
        assert not h_ok
    finally:
        G._gu_grouped.cache_clear()

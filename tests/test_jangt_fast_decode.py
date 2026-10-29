# SPDX-License-Identifier: Apache-2.0
"""Fast JANGT decode kernels (vmlx_engine/jangt/fast_decode.py, VMLX_JANGT_FAST=1).

Not bit-identical to the v3 kernels by design; checked instead:
  * accuracy: against a float64 dequantize-and-matmul reference, the v4 error is at most v3's (gate / up + SwiGLU-OAI
    and the weighted down, trellis K 1 / 1.5 / 2 / 2.5 / 3, outlier columns on and off);
  * row invariance: a row decoded alone (decode step: one-row kernels incl. the in-kernel slot combine) equals the
    same row inside a 2..16-row window (verify: per-slot down + combine kernel, grouped variant too), distinct /
    shared / identical routing;
  * the switch: JTMixedSwitchGLU.routed with the lever on gives decode == verify rows; widths that are not whole
    512-value blocks keep the v3 kernels;
  * the DSpark target with the lever on: verify rows == decode steps;
  * control: one row's fold perturbed is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import bench.jangt_decode_error as BE  # noqa: E402
import tests.test_jangt_grouped_decode as TG  # noqa: E402
from vmlx_engine.jangt import fast_decode as FD  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _rms_rel(got, ref):
    got = np.asarray(got, np.float64)
    return float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(BE, "D", 1024)
    monkeypatch.setattr(BE, "I", 512)
    return BE


@pytest.mark.parametrize("K", [1, 1.5, 2, 2.5, 3])
@pytest.mark.parametrize("n_cols", [0, 2])
def test_gate_up_error_vs_float64_at_most_v3(small, K, n_cols):
    rng = np.random.default_rng(int(K * 10) + n_cols)
    D, I = small.D, small.I
    g, gr, su, cols = small.unit(D, I, K, n_cols, False, rng)
    u, ur, _, _ = small.unit(D, I, K, n_cols, False, rng, su=su, cols=cols)
    m = np.ones(D, np.float32)
    m[cols[0][cols[0] >= 0]] = 0.0
    U = {"L": 12, "gate": g, "up": u, "K": K, "su_gu": mx.array(su * m)}
    for T in (1, 7):
        idx = np.stack([rng.permutation(small.E)[:4] for _ in range(T)]).astype(np.int32)
        xb = mx.array(rng.standard_normal((T, D)).astype(np.float32)).astype(mx.bfloat16)
        x64 = np.array(xb.astype(mx.float32)).astype(np.float64)
        ref = np.stack([small.swiglu64(gr[e] @ x64[t], ur[e] @ x64[t]) for t in range(T) for e in idx[t]])
        v3 = MK.gather_gate_up_swiglu_v3(MK.rotate_rows(xb, U["su_gu"]), xb, mx.array(idx), U, out_dtype=mx.float32,
                                         limit=7.0, act="oai", alpha=1.702)
        v4 = FD.routed_gate_up(xb, mx.array(idx), U, 7.0, "oai", 1.702)
        e3, e4 = _rms_rel(v3, ref), _rms_rel(v4, ref)
        assert e3 < 3e-3 and e4 <= e3 * 1.01 + 1e-6, (T, e3, e4)


@pytest.mark.parametrize("K", [1, 1.5, 2, 2.5, 3])
def test_down_error_vs_float64_at_most_v3(small, K):
    rng = np.random.default_rng(int(K * 10) + 5)
    D, I, E = small.D, small.I, small.E
    d, dr, sud, cols = small.unit(I, D, K, 6, True, rng)
    md = np.ones((E, I), np.float32)
    for e in range(E):
        md[e, cols[e][cols[e] >= 0]] = 0.0
    U = {"L": 12, "down": d, "K_dn": K, "sv_dn": d["cs"] * mx.array(sud)[None, :] * mx.array(md)}
    for T in (1, 9):
        idx = np.stack([rng.permutation(E)[:4] for _ in range(T)]).astype(np.int32)
        w = rng.uniform(0.1, 0.6, (T, 4)).astype(np.float32)
        h = (rng.standard_normal((T * 4, I)) * 1.5).astype(np.float32)
        ref = np.stack([sum(float(w[t, j]) * (dr[idx[t, j]] @ h[t * 4 + j].astype(np.float64)) for j in range(4))
                        for t in range(T)])
        hm = mx.array(h)
        v3 = MK.gather_down_weighted_v3(MK.rotate_rows(hm, U["sv_dn"], mx.array(idx).reshape(-1)), hm, mx.array(idx),
                                        mx.array(w), U)
        v4 = FD.routed_down(hm, mx.array(idx), mx.array(w), U)
        e3, e4 = _rms_rel(v3, ref), _rms_rel(v4, ref)
        assert e3 < 3e-3 and e4 <= e3 * 1.01 + 1e-6, (T, e3, e4)


@pytest.mark.parametrize("mode", ["same", "distinct", "pool"])
@pytest.mark.parametrize("Kgu,Kdn,nout", [(1.5, 2.5, 2), (1, 1.5, 0), (2, 2, 3), (2.5, 1, 8)])
def test_rows_alone_equal_rows_in_a_window(mode, Kgu, Kdn, nout):
    """Decode step (one row: one-row gate/up, in-kernel slot combine) == the row inside a 16-row verify (per-slot
    down + combine kernel); the grouped gate / up equals the ungrouped one."""
    m = TG._module(1024, 512, Kgu, Kdn, nout, 7.0, "oai", seed=int(Kgu * 10 + Kdn))
    U = m._unit()
    idx, w, _ = TG._routing(16, 4, seed=3, mode=mode)
    x = TG._x(16, 1024, 4)
    full = FD.routed_fast(x, idx, w, U, 7.0, "oai", 1.702)
    h16 = FD.routed_gate_up(x, idx, U, 7.0, "oai", 1.702)
    hg = FD.gate_up4g(FD.pack_rows1(x, U["su_gu"]), x, idx, U, limit=7.0, act="oai", alpha=1.702)
    assert _eq(h16, hg)
    for r in (0, 5, 15):
        alone = FD.routed_fast(x[r:r + 1], idx[r:r + 1], w[r:r + 1], U, 7.0, "oai", 1.702)
        assert _eq(alone, full[r:r + 1]), r
    for T in (2, 3, 9):
        part = FD.routed_fast(x[:T], idx[:T], w[:T], U, 7.0, "oai", 1.702)
        assert _eq(part, full[:T]), T


def test_switch_lever_decode_equals_verify_and_tracks_v3(monkeypatch):
    m = TG._module(1024, 512, 1.5, 2.5, 2, 7.0, "oai", seed=1)
    idx, w, _ = TG._routing(8, 4, seed=1, mode="pool")
    x = TG._x(8, 1024, 1)[None]
    monkeypatch.setenv("VMLX_JANGT_FAST", "0")
    off = m.routed(x, idx[None], w[None])
    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    on = m.routed(x, idx[None], w[None])
    steps = mx.concatenate([m.routed(x[:, r:r + 1], idx[None, r:r + 1], w[None, r:r + 1]) for r in range(8)], axis=1)
    assert _eq(on, steps)
    rel = float(mx.abs(on.astype(mx.float32) - off.astype(mx.float32)).max() / mx.abs(off.astype(mx.float32)).max())
    assert 0 < rel < 2e-2, rel                        # bf16 outputs: v4 != v3 bits, within rounding


def test_unsupported_widths_keep_v3(monkeypatch):
    m = TG._module(1280, 640, 1.5, 2.5, 2, 7.0, "oai", seed=2)       # not whole 512-value blocks
    assert not FD.supported(m._unit(), 1280, 640)
    idx, w, _ = TG._routing(3, 4, seed=2, mode="pool")
    x = TG._x(3, 1280, 2)[None]
    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    on = m.routed(x, idx[None], w[None])
    monkeypatch.setenv("VMLX_JANGT_FAST", "0")
    assert _eq(on, m.routed(x, idx[None], w[None]))


def test_dspark_verify_with_fast_jangt_equals_decode_steps(monkeypatch):
    import tests.test_minimax_m3_fused_moe as TF
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    tgt = MiniMaxM3DSparkTarget(TF._model())
    prompt = mx.array([np.random.default_rng(1).integers(0, 511, 61).tolist()])
    for width in (4, 16):
        rows = mx.array([np.random.default_rng(width).integers(0, 511, width).tolist()])
        cache = tgt.make_cache()
        tgt.prefill(prompt, cache, (), 32)
        logits, _, _ = tgt.verify(rows, cache, ())
        ref = tgt.make_cache()
        tgt.prefill(prompt, ref, (), 32)
        steps = [tgt.decode(rows[0, r:r + 1], ref) for r in range(width)]
        assert all(_eq(logits[r], steps[r]) for r in range(width)), width


def test_control_perturbed_row_fold_is_caught(monkeypatch):
    """Control: the one-row gate / up kernel with a perturbed fold for the first output row differs from the same row
    in the 16-row launch built from the unperturbed generator."""
    m = TG._module(1024, 512, 1.5, 2.5, 0, 7.0, "oai", seed=4)
    U = m._unit()
    idx, w, _ = TG._routing(16, 4, seed=5, mode="pool")
    x = TG._x(16, 1024, 6)
    full = FD.routed_gate_up(x, idx, U, 7.0, "oai", 1.702)
    real = FD._main_loop

    def perturbed(*a, **k):
        return real(*a, **k).replace("fma(float(a_.x), x0, g[0])", "fma(float(a_.x), x0 * 1.0001f, g[0])")

    monkeypatch.setattr(FD, "_main_loop", perturbed)
    FD.gen_gu4.cache_clear()
    monkeypatch.setattr(FD, "GU_CFG", dict(FD.GU_CFG, wload="packed"))     # new kernel name
    try:
        alone = FD.routed_gate_up(x[:1], idx[:1], U, 7.0, "oai", 1.702)
        assert not _eq(alone, full[:4])
    finally:
        FD.gen_gu4.cache_clear()


@pytest.mark.parametrize("T", [1, 3, 16])
def test_fused_moe_with_fast_jangt_equals_unfused_with_fast_jangt(monkeypatch, T):
    """VMLX_M3_FUSED_MOE=1 + VMLX_JANGT_FAST=1: routing and the shared expert in the fused kernels, the routed experts
    on the fast kernels -- bit-identical to the unfused block with the fast kernels, rows alone == rows in a window."""
    import tests.test_minimax_m3_fused_moe as TF
    from vmlx_engine.models.minimax_m3 import fused_moe as F

    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    b = TF._block(Kgu=1.5, Kdn=2.5, nog=1, nod=3, seed=7)
    x, res = TF._rows(T, 1024, seed=T)
    ref = F.unfused_moe_residual(b, x, res)
    got = F.fused_moe_residual(b, x, res)
    assert _eq(got, ref)
    alone = F.fused_moe_residual(b, x[:, :1], res[:, :1])
    assert _eq(alone, got[:, :1])

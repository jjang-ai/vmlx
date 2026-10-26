"""Sub-2-bit JANGT trellis units (K = 1 and K = 1.5, Qwen3.8-Flash-Next JANGHT, 2026-10-16) in the vMLX kernels.

The encoder (jang_tools.jangt.v2l + the campaign's jt_patch.py) defines the two new widths: kpat(1.5) = (4, 2) new bits
per (even, odd) step = 6 bits per 4 weights, kpat(1) = (2, 2); hash multipliers MULTS[1.5] = 2124823417,
MULTS[1] = 590428815 (MASK/ORB unchanged). These tests pin the vMLX side to that definition and check every decode
kernel that a served bundle runs against a dense dequantized reference built WITHOUT the kernels
(tests/jangt_lowbit_support.py), at the Qwen geometry D = 2560, I = 640 (640 % 512 != 0: the masked tail block):
  * constants / packing: vMLX v2l == encoder v2l with jt_patch (skipped only when the jang-tools source is absent);
  * v3 decode (the runtime path, JTMixedSwitchGLU._decode_h / _decode_down): gate+up+SwiGLU and the router-weighted
    down, with layer-wide gate/up outlier columns, per-expert down outliers (-1 padded) and down column scales;
  * dequant_rot (prefill bring-up) including the 640-wide tail;
  * the word-aligned WPL fast path (64 weights per lane at K=1.5, 32 at K=1) where kin divides its chunk.
Tolerances: the v3 inner products run on half2 by design (8 steps per lane in fp16, fp32 across lanes), for every
width; the bound is checked against the K=2 error on the same data (the width already in production) and an
absolute ceiling, and a wrong-phase oracle must FAIL (the comparison discriminates).
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402
from vmlx_engine.jangt.v2core import MULT  # noqa: E402

from tests import jangt_lowbit_support as S  # noqa: E402

D, I, E, TOPK = 2560, 640, 16, 10
WIDTHS = (1, 1.5, 2, 2.5)
ORACLE = S.load_encoder_oracle()
REF = ORACLE if ORACLE is not None else VL       # dense reference definition (never the Metal kernels)


# ----------------------------------------------------------------------------------------------- constants, packing
def test_lowbit_trellis_definition():
    assert VL.kpat(1.5) == (4, 2) and VL.kpat(1) == (2, 2) and VL.kpat(2) == (4, 4) and VL.kpat(2.5) == (6, 4)
    assert VL.MULTS[1.5] == 2124823417 and VL.MULTS[1] == 590428815 and VL.MULTS[2] == MULT
    assert list(VL.step_offsets(1.5, 6)) == [0, 2, 6, 8, 12, 14]          # t odd: 2 bits, t even: 4 bits
    assert list(VL.step_offsets(1, 4)) == [0, 2, 4, 6]
    # words_per_row at the Qwen geometry (the encoder's jt_off strides)
    assert [VL.words_per_row(k, 12, K) for K in (1.5, 1) for k in (D, I)] == [122, 32, 82, 22]
    # lane windows: (bits per 16-weight lane, words loaded)
    assert {K: (MK._window(K, 12)[0], MK._window(K, 12)[2]) for K in (1, 1.5, 2, 2.5, 3)} == \
        {1: (16, 2), 1.5: (24, 2), 2: (32, 2), 2.5: (40, 3), 3: (48, 3)}
    # word-aligned lane segments of the WPL fast path
    assert {K: (MK.WPL[K], MK._lane_words(K)) for K in (1, 1.5, 2, 2.5, 3)} == \
        {1: (32, 1), 1.5: (64, 3), 2: (16, 1), 2.5: (64, 5), 3: (32, 3)}


def test_constants_match_the_encoder_patch_file():
    mults = S.jt_patch_mults()
    if mults is None:
        pytest.skip(f"encoder campaign file absent: {S.JT_PATCH}")
    assert mults == {1.5: VL.MULTS[1.5], 1.0: VL.MULTS[1]}


@pytest.mark.skipif(ORACLE is None, reason="jang-tools source (JANG_TOOLS_SRC or ~/jang/jang-tools) not present")
@pytest.mark.parametrize("K", (1, 1.5, 2, 2.5, 3))
def test_packing_matches_the_encoder_oracle(K):
    assert ORACLE.kpat(K) == VL.kpat(K) and ORACLE.MULTS[K] == VL.MULTS[K]
    for kin in (D, I):
        assert ORACLE.words_per_row(kin, 12, K) == VL.words_per_row(kin, 12, K)
        np.testing.assert_array_equal(ORACLE.step_offsets(K, kin // 2), VL.step_offsets(K, kin // 2))
    mo, io = ORACLE.code_stats(12, K); mv, iv = VL.code_stats(12, K)
    np.testing.assert_array_equal(mo, mv); np.testing.assert_array_equal(io, iv)
    rng = np.random.default_rng(int(K * 10))
    states = rng.integers(0, 1 << 12, size=(24, I // 2)).astype(np.uint16)
    po = ORACLE.pack_states(states, I, 12, K)
    pv = VL.pack_states(states, I, 12, K)
    np.testing.assert_array_equal(po, pv)
    uo = ORACLE.unpack_states(po, I, 12, K)
    np.testing.assert_array_equal(uo, VL.unpack_states(pv, I, 12, K))
    # the canonical path of random states: state t carries its own top kpat bits, the rest comes from t-1
    np.testing.assert_array_equal(ORACLE.pack_states(uo, I, 12, K), po)
    np.testing.assert_array_equal(np.array(ORACLE.values_of(mx.array(uo), 1.0, 12, K)),
                                  np.array(VL.values_of(mx.array(uo), 1.0, 12, K)))


# ----------------------------------------------------------------------------------------------- v3 decode kernels
_CACHE = {}


def _layer(K):
    if K not in _CACHE:
        rng = np.random.default_rng(100 + int(K * 10))
        P = S.layer_unit(rng, D, I, E, K, K, 3, 2, VL)
        m = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": K}, 3, 2, 0.0)
        S.bind(m, P)
        Wg, Wu, Wd = (S.dense_reference(P[p], REF) for p in ("gate", "up", "down"))
        T = 3
        x = (rng.standard_normal((T, D)) * 0.5).astype(np.float32)
        idx = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
        w = rng.uniform(0.02, 0.2, size=(T, TOPK)).astype(np.float32)
        h, y = S.moe_reference(Wg, Wu, Wd, x.astype(np.float64), idx, w.astype(np.float64))
        _CACHE[K] = (m, P, x, idx, w, h, y)
    return _CACHE[K]


def _decode_errors(K):
    m, P, x, idx, w, h, y = _layer(K)
    hk = m._decode_h(mx.array(x), mx.array(idx))                               # gate+up+SwiGLU, (T*k, I) f32
    # down from the EXACT h, so the down error is the down kernel's own
    yk = m._decode_down(mx.array(h.astype(np.float32)), mx.array(idx), mx.array(w), mx.float32)
    mx.eval(hk, yk)
    return S.rel_l2(np.array(hk), h), S.rel_l2(np.array(yk), y)


@pytest.mark.parametrize("K", WIDTHS)
def test_v3_decode_matches_dense_reference(K):
    eg, ed = _decode_errors(K)
    eg2, ed2 = _decode_errors(2)
    print(f"K={K}: gate/up rel-L2 {eg:.2e} (K=2 {eg2:.2e}), down rel-L2 {ed:.2e} (K=2 {ed2:.2e})")
    assert eg < 5e-3 and ed < 5e-3, (eg, ed)
    assert eg < 3 * eg2 + 1e-4 and ed < 3 * ed2 + 1e-4, (eg, eg2, ed, ed2)


def _rotated_values(p: dict, e: int) -> np.ndarray:
    """(N, kin) float32 weights of expert e in the ROTATED basis (code value x row scale), from the oracle."""
    K, kin = p["K"], p["kin"]
    st = mx.array(REF.unpack_states(p["packed"][e], kin, 12, K))
    return np.array(REF.values_of(st, 1.0, 12, K)) * p["scale"][e][:, None]


@pytest.mark.parametrize("K", WIDTHS)
def test_v3_decode_every_weight_exact(K):
    """Exhaustive bit-extraction check at fp32 tolerance: a one-hot rotated input selects ONE weight per output row, so
    the half2 trellis dot is exact (1 x code) and the kernel output must equal the oracle's decoded weight for every
    (row, column) of an expert, including the 512..639 tail block of the down projection and the outlier columns
    (zero rotated input, one-hot raw input -> the fp16 outlier weight)."""
    m, P, *_ = _layer(K)
    U = m._unit(); e = 5
    # gate/up: token j = one-hot column j (D tokens) + one token per outlier column; slot -> expert e
    Vg, Vu = _rotated_values(P["gate"], e), _rotated_values(P["up"], e)
    cols = P["gate"]["ocol"][e][P["gate"]["ocol"][e] >= 0]
    xr = np.concatenate([np.eye(D, dtype=np.float32), np.zeros((len(cols), D), np.float32)])
    xraw = np.zeros_like(xr); xraw[D + np.arange(len(cols)), cols] = 1.0
    idx = mx.full((xr.shape[0], 1), e, dtype=mx.int32)
    hk = np.array(MK.gather_gate_up_swiglu_v3(mx.array(xr), mx.array(xraw), idx, U, out_dtype=mx.float32))
    g = np.concatenate([Vg.T, P["gate"]["wout"][e][:, :len(cols)].T.astype(np.float32)])
    u = np.concatenate([Vu.T, P["up"]["wout"][e][:, :len(cols)].T.astype(np.float32)])
    want = S.silu(g.astype(np.float64)) * u
    np.testing.assert_allclose(hk, want, rtol=2e-5, atol=1e-7 * float(np.abs(want).max()))
    # down: token j = one-hot column j of the 640 inputs + one token per outlier column, weight 1
    Vd = _rotated_values(P["down"], e)
    oc = P["down"]["ocol"][e]; dcols = oc[oc >= 0]
    hr = np.concatenate([np.eye(I, dtype=np.float32), np.zeros((len(dcols), I), np.float32)])
    hraw = np.zeros_like(hr); hraw[I + np.arange(len(dcols)), dcols] = 1.0
    n = hr.shape[0]
    yk = np.array(MK.gather_down_weighted_v3(mx.array(hr), mx.array(hraw), mx.full((n, 1), e, dtype=mx.int32),
                                             mx.ones((n, 1)), U))
    want_d = np.concatenate([Vd.T, P["down"]["wout"][e][:, :len(dcols)].T.astype(np.float32)])
    np.testing.assert_allclose(yk, want_d, rtol=1e-6, atol=1e-7 * float(np.abs(want_d).max()))
    assert np.abs(yk[512:I]).min(axis=1).max() > 0                                # tail columns decoded, not zero


@pytest.mark.parametrize("K,control", [(1, "mult"), (1.5, "mult"), (1.5, "phase")])
def test_v3_decode_rejects_a_wrong_oracle(K, control):
    """Negative controls: a reference with the K=2 hash multiplier, or (K=1.5) the (even, odd) bit pattern swapped,
    must NOT match the kernel (K=1's (2, 2) pattern has no phase to swap)."""
    m, P, x, idx, w, h, y = _layer(K)
    hk = np.array(m._decode_h(mx.array(x), mx.array(idx)))
    orig_kpat, orig_mult = VL.kpat, VL.MULTS[K]
    try:
        if control == "phase":
            VL.kpat = lambda k: tuple(reversed(orig_kpat(k))) if k == K else orig_kpat(k)
        else:
            VL.MULTS[K] = MULT
        VL.code_stats.cache_clear()
        Wg, Wu = (S.dense_reference(P[p], VL) for p in ("gate", "up"))
    finally:
        VL.kpat, VL.MULTS[K] = orig_kpat, orig_mult
        VL.code_stats.cache_clear()
    hw, _ = S.moe_reference(Wg, Wu, np.zeros((E, 1, I)), x.astype(np.float64), idx, w.astype(np.float64))
    assert S.rel_l2(hk, hw) > 0.3


@pytest.mark.parametrize("K", WIDTHS)
def test_dequant_rot_matches_including_the_tail_block(K):
    m, P, *_ = _layer(K)
    U = m._unit()
    for name, kin in (("gate", D), ("down", I)):
        p = P[name]; e0, e1 = 3, 7
        got = np.array(MK.dequant_rot(U[name], e0, e1).astype(mx.float32))
        ref = np.stack([np.array(REF.values_of(mx.array(REF.unpack_states(p["packed"][e], kin, 12, K)), 1.0, 12, K))
                        * p["scale"][e][:, None] for e in range(e0, e1)])
        assert got.shape == ref.shape == (e1 - e0, p["N"], kin)
        assert S.rel_l2(got, ref) < 4e-3, name                                     # bf16 output
        assert np.abs(got[..., -128:]).max() > 0                                   # the tail columns are written


# ----------------------------------------------------------------------------------------------- WPL fast path
@pytest.mark.parametrize("K", (1, 1.5))
def test_word_aligned_fast_path(K):
    Df = If = 2048                                   # divisible by the 32 x WPL chunk (1024 / 2048)
    rng = np.random.default_rng(7 + int(K * 10))
    P = S.layer_unit(rng, Df, If, 4, K, K, 3, 2, VL)
    m = SW.JTMixedSwitchGLU(Df, If, 4, {"mode": "jangt", "kbits": K}, {"mode": "jangt", "kbits": K}, 3, 2, 0.0)
    S.bind(m, P); U = m._unit()
    Wg, Wu, Wd = (S.dense_reference(P[p], REF) for p in ("gate", "up", "down"))
    x = (rng.standard_normal((2, Df)) * 0.5).astype(np.float32)
    idx = np.array([[0, 2, 3], [1, 3, 0]], np.int32); w = rng.uniform(0.1, 0.4, size=(2, 3)).astype(np.float32)
    h, y = S.moe_reference(Wg, Wu, Wd, x.astype(np.float64), idx, w.astype(np.float64))
    X = mx.array(x)
    hk = MK.gather_gate_up_swiglu_fast(MK.rotate_rows(X, U["su_gu"]), X, mx.array(idx), U, out_dtype=mx.float32)
    H = mx.array(h.astype(np.float32))
    yk = MK.gather_down_weighted_fast(MK.rotate_rows(H, U["sv_dn"], mx.array(idx).reshape(-1)), H, mx.array(idx), mx.array(w), U)
    mx.eval(hk, yk)
    assert S.rel_l2(np.array(hk), h) < 1e-4 and S.rel_l2(np.array(yk), y) < 1e-4       # fp32 FMAs on this path


# ----------------------------------------------------------------------------------------------- NAX prefill tiles
@pytest.mark.skipif(not __import__("vmlx_engine.jangh.kernels", fromlist=["x"]).nax_available(),
                    reason="NAX prefill kernels exist only on NAX GPUs (decode serves every batch elsewhere)")
@pytest.mark.parametrize("K", WIDTHS)
@pytest.mark.parametrize("tiles", ("rows", "expert"))
def test_nax_prefill_tile_loader_every_weight(K, tiles):
    """The sorted prefill's trellis tile loader (jangt/prefill.py JTBlockLoader) at every weight: identity rows (bf16,
    exact) through gather_qmm_sorted_jt return the decoded weights of expert e in the rotated basis, x row scale,
    rounded once to bf16 (measured bit-exact) -- for the 2560-wide gate and the 640-wide down."""
    from vmlx_engine.jangh import kernels as JK
    from vmlx_engine.jangt.prefill import gather_qmm_sorted_jt
    m, P, *_ = _layer(K)
    U = m._unit(); e = 9
    for name, kin in (("gate", D), ("down", I)):
        x = mx.eye(kin, dtype=mx.bfloat16)
        idx = mx.full((kin,), e, dtype=mx.uint32)
        plan = JK.expert_tile_plan(idx, E) if tiles == "expert" else None
        got = np.array(gather_qmm_sorted_jt(x, U[name], idx, plan=plan).astype(mx.float32))
        want = np.array(mx.array(_rotated_values(P[name], e).T).astype(mx.bfloat16).astype(mx.float32))
        assert got.shape == want.shape == (kin, P[name]["N"])
        np.testing.assert_array_equal(got, want, err_msg=name)            # same fp32 ops, one bf16 rounding: exact

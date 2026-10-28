"""JANGT routed experts with the SwiGLU-OAI activation (MiniMax-M3 / M3.1):

    h = g' * sigmoid(alpha * g') * (u' + 1),   g' = min(g, limit),  u' = clip(u, -limit, limit),  alpha 1.702, limit 7

Every path of JTMixedSwitchGLU that applies the activation is compared with a float64 NumPy reference built from the
units' own bits (trellis states unpacked and valued by jangt.v2l, JANGH codes by jangh.format, Hadamard rotations as
explicit Sylvester matrices): the v3 decode gate/up kernel (also with the Hadamard-32 output epilogue), the decode
route end to end, the NAX sorted prefill (fused gate/up with outlier columns in the epilogue, fused without outliers,
unfused trellis matmuls; row-tiled and expert-aligned schedules; the H32 store epilogue feeding a JANGH down), JANGH
gate/up units (unfused fallback), and the fused routed + mxfp8 shared decode (fused_moe, bit-identical to its stock
path). Shapes: D = 1280 and I = 768 are not multiples of the 512-value decode block (masked tail lanes), 37 tokens x 4
slots = 148 rows is not a multiple of the 64-row prefill tile, and 2 of 8 experts receive no rows in the prefill
batches. Outlier columns are present on gate/up (layer-wide) and down (per expert, with -1 padding).

Control: the same comparisons with the SiLU kernels (plain, and clamped at the same limit) must fail the tolerance,
so the tests can tell the activations apart.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.jangh import format as JF  # noqa: E402
from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import fused_moe as FM  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import prefill as PF  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402

D, I, E, TOPK = 1280, 768, 8, 4
ALPHA, LIMIT = 1.702, 7.0
OAI = ("oai", LIMIT, ALPHA)
CONTROLS = (("silu", 0.0, 0.0), ("silu", LIMIT, 0.0))       # plain SiLU, clamped SiLU (GLM rule) at the same limit

# Relative L2 error vs the float64 reference. Measured on M5 Max (2026-10-27): decode h 7.7-8.2e-4, decode route
# 1.8-1.9e-3, NAX prefill route 4.7-5.2e-3, JANGH gate/up h 1.7e-7. Controls: plain SiLU 1.1-1.6, SiLU clamped at the
# same limit 0.19-0.21 (large |g|, |u| dominate the norm, where OAI and clamped SiLU differ by the "+1" and the gate
# slope only). Each tolerance is ~3x the measured error and >= 10x below every control.
TOL_DECODE = 3e-3        # fp16 x / half2 trellis dot, fp32 epilogue (gate/up activations)
TOL_ROUTED = 6e-3        # decode route end to end (+ the down projection)
TOL_PREFILL = 1.5e-2     # bf16 x and weight tiles in the NAX matmul, bf16 h stored between gate/up and down
CTL = 10                 # a control must miss the tolerance by at least this factor

NAX = JK.nax_available()
needs_nax = pytest.mark.skipif(not NAX, reason="NAX prefill kernels need macOS 26.2+ on M5")


# ------------------------------------------------------------------------------------------------ fixtures
def _rand_words(rng, shape):
    return mx.array(rng.integers(0, 2 ** 32, size=shape, dtype=np.uint64).astype(np.uint32))


def _fill_jt(p, rng, lo, hi, ocol=None):
    p.jt_packed = _rand_words(rng, p.jt_packed.shape)
    p.jt_scale = mx.array(rng.uniform(lo, hi, size=p.jt_scale.shape).astype(np.float32))
    p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
    if ocol is not None:
        p.jt_ocol = mx.array(ocol.astype(np.int32))
        wo = rng.standard_normal((E, p.output_dims, ocol.shape[1])) * 0.05
        wo *= (ocol >= 0)[:, None, :]       # bundle contract (the encoders): a -1 padding slot has zero weights
        p.jt_wout = mx.array(wo.astype(np.float16))


def _fill_tq(p, rng, lo, hi):
    p.tq2_packed = _rand_words(rng, p.tq2_packed.shape)
    p.tq2_scales = mx.array(rng.uniform(lo, hi, size=p.tq2_scales.shape).astype(np.float16))


GU_COLS = np.array([5, 77, 1001], dtype=np.int32)          # layer-wide gate/up outlier input columns


def _dn_cols(rng):
    c = np.stack([rng.choice(I, size=2, replace=False) for _ in range(E)]).astype(np.int32)
    c[3, 1] = -1                                             # padding slot (an expert with fewer outlier columns)
    return c


def build(gu="jangt", dn="jangt", kg=2, kd=2.5, n_out=True, act=OAI, seed=0):
    """A JTMixedSwitchGLU with deterministic random units (same seed -> same weights for every act)."""
    rng = np.random.default_rng(seed)
    gu_e = {"mode": "jangt", "kbits": kg} if gu == "jangt" else {"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}
    dn_e = {"mode": "jangt", "kbits": kd} if dn == "jangt" else {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}
    no_gu, no_dn = (3, 2) if n_out else (0, 0)
    kind, limit, alpha = act
    m = SW.JTMixedSwitchGLU(D, I, E, gu_e, dn_e, no_gu if gu == "jangt" else 0, no_dn if dn == "jangt" else 0,
                            limit, kind, alpha)
    # gate/up row scales put |g|, |u| well past the limit for a sizeable fraction of rows (checked in the tests)
    if gu == "jangt":
        cols = np.tile(GU_COLS, (E, 1)) if n_out else None
        _fill_jt(m.gate_proj, rng, 0.2, 0.45, cols)
        _fill_jt(m.up_proj, rng, 0.2, 0.45, cols)
        m.up_proj.jt_su = m.gate_proj.jt_su              # gate and up share the layer's input signs (encoder contract)
    else:
        _fill_tq(m.gate_proj, rng, 0.2, 0.45)
        _fill_tq(m.up_proj, rng, 0.2, 0.45)
    if dn == "jangt":
        _fill_jt(m.down_proj, rng, 0.004, 0.008, _dn_cols(rng) if n_out else None)
        m.down_proj.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=m.down_proj.jt_cs.shape).astype(np.float32))
    else:
        _fill_tq(m.down_proj, rng, 0.004, 0.008)
    return m


def inputs(T, seed=1, experts=E):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal((T, D)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.choice(experts, size=TOPK, replace=False) for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.uniform(0.1, 0.4, size=(T, TOPK)).astype(np.float32))
    return x, idx, w


# ------------------------------------------------------------------------------------------------ float64 reference
@functools.lru_cache(maxsize=None)
def _sylvester(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def hblk(x, n):
    """Blockwise normalized Walsh-Hadamard (natural order) over the last axis in n-wide blocks."""
    with np.errstate(all="ignore"):         # Accelerate's matmul raises spurious FP flags; finiteness is asserted
        return (x.reshape(*x.shape[:-1], -1, n) @ _sylvester(n)).reshape(x.shape)


def _np(a, dt=np.float64):
    return np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).astype(dt)


def dense_jt(p):
    """JTSwitchLinear -> (E, N, K_in) float64 rotated-basis rows: trellis states unpacked and valued, x row scale."""
    out = []
    for e in range(p.num_experts):
        st = VL.unpack_states(np.array(p.jt_packed[e]), p.input_dims, 12, p.kbits)
        v = np.array(VL.values_of(mx.array(st), 1.0, 12, p.kbits)).astype(np.float64)
        out.append(v * _np(p.jt_scale[e])[:, None])
    return np.stack(out)


def dense_tq(p):
    """TQSwitchLinear -> (E, N, K_in) float64 rows in the Hadamard-32 basis (codebook level x row scale)."""
    q = np.array(JF.unpack_bitstream(p.tq2_packed, p.bits, p.input_dims)).astype(np.int64)
    return JF.codebook(p.bits).astype(np.float64)[q] * _np(p.tq2_scales)[..., None]


def act_ref(g, u, kind, limit, alpha):
    if limit > 0:
        g, u = np.minimum(g, limit), np.clip(u, -limit, limit)
    if kind == "oai":
        return g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)
    return g / (1.0 + np.exp(-g)) * u


@functools.lru_cache(maxsize=None)
def _dense(key):
    m = build(*key)
    d = {"gu": (dense_jt(m.gate_proj), dense_jt(m.up_proj)) if m.gu_fmt == "jangt" else
               (dense_tq(m.gate_proj), dense_tq(m.up_proj)),
         "dn": dense_jt(m.down_proj) if m.dn_fmt == "jangt" else dense_tq(m.down_proj)}
    return d


def pre_act(m, key, x, idx):
    """(g, u) float64 (T*k, I) pre-activations of every (token, slot)."""
    Wg, Wu = _dense(key)["gu"]
    xn = _np(x)
    T, k = idx.shape
    e = np.array(idx).reshape(-1)
    rows = np.repeat(np.arange(T), k)
    if m.gu_fmt == "jangt":
        gp, upp = m.gate_proj, m.up_proj
        oc = np.array(gp.jt_ocol)
        mask = np.ones(D)
        mask[oc[0][oc[0] >= 0]] = 0.0
        xr = hblk(xn * _np(gp.jt_su) * mask, 128)
        g = np.einsum("snk,sk->sn", Wg[e], xr[rows])
        u = np.einsum("snk,sk->sn", Wu[e], xr[rows])
        if oc.shape[1] and oc.max() >= 0:
            wg, wu = _np(gp.jt_wout), _np(upp.jt_wout)
            for j in range(oc.shape[1]):
                c = oc[e, j]
                ok = c >= 0
                xo = np.where(ok, xn[rows, np.maximum(c, 0)], 0.0)
                g += wg[e, :, j] * xo[:, None]
                u += wu[e, :, j] * xo[:, None]
    else:
        xr = hblk(xn, 32)
        g = np.einsum("snk,sk->sn", Wg[e], xr[rows])
        u = np.einsum("snk,sk->sn", Wu[e], xr[rows])
    return g, u


def down_ref(m, key, h, idx, w):
    """y (T, D) float64 = sum_slot w * down_e(h_slot)."""
    Wd = _dense(key)["dn"]
    T, k = idx.shape
    e = np.array(idx).reshape(-1)
    if m.dn_fmt == "jangt":
        d = m.down_proj
        oc = np.array(d.jt_ocol)
        mask = np.ones((E, I))
        for ex in range(E):
            mask[ex, oc[ex][oc[ex] >= 0]] = 0.0
        hr = hblk(h * _np(d.jt_cs)[e] * _np(d.jt_su)[None, :] * mask[e], 128)
        y = np.einsum("snk,sk->sn", Wd[e], hr)
        if oc.max() >= 0:
            wo = _np(d.jt_wout)
            for j in range(oc.shape[1]):
                c = oc[e, j]
                ho = np.where(c >= 0, h[np.arange(h.shape[0]), np.maximum(c, 0)], 0.0)
                y += wo[e, :, j] * ho[:, None]
    else:
        y = np.einsum("snk,sk->sn", Wd[e], hblk(h, 32))
    return (y.reshape(T, k, D) * _np(w)[..., None]).sum(axis=1)


def rel(got, ref):
    got = _np(got) if isinstance(got, mx.array) else got
    assert np.isfinite(got).all()
    assert np.isfinite(ref).all()
    nr = float(np.linalg.norm(ref))
    assert nr > 0, "empty reference"
    return float(np.linalg.norm(got - ref)) / nr


def _exercises_limits(g, u):
    """The fixture must clip: a sizeable share of gate values above +L, up values beyond +-L, and gates far negative."""
    assert np.mean(g > LIMIT) > 0.02 and np.mean(np.abs(u) > LIMIT) > 0.04 and np.mean(g < -LIMIT) > 0.02, (
        np.mean(g > LIMIT), np.mean(np.abs(u) > LIMIT), np.mean(g < -LIMIT))


# ------------------------------------------------------------------------------------------------ the reference itself
def test_reference_rotations_match_the_runtime_kernels():
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((3, D)).astype(np.float32))
    s = mx.ones((D,))
    np.testing.assert_allclose(np.array(MK.rotate_rows(x, s)), hblk(np.array(x, np.float64), 128), rtol=0, atol=2e-5)
    np.testing.assert_allclose(np.array(JK.h32_rows(x, mx.float32)), hblk(np.array(x, np.float64), 32), rtol=0, atol=2e-5)


def test_reference_activation_matches_minimax_m3_swiglu():
    """act_ref("oai") is the model's own swiglu (vmlx_engine.models.minimax_m3.minimax_m3.swiglu, float32)."""
    from vmlx_engine.models.minimax_m3.minimax_m3 import swiglu
    rng = np.random.default_rng(6)
    g = rng.standard_normal(4096) * 9
    u = rng.standard_normal(4096) * 9
    got = np.array(swiglu(mx.array(u.astype(np.float32)), mx.array(g.astype(np.float32)), ALPHA, LIMIT))
    ref = act_ref(g.astype(np.float32).astype(np.float64), u.astype(np.float32).astype(np.float64), "oai", LIMIT, ALPHA)
    np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-6)


# ------------------------------------------------------------------------------------------------ decode
@pytest.mark.parametrize("kg,kd", [(2, 2.5), (1.5, 2)])
@pytest.mark.parametrize("T", [1, 3])
def test_decode_gate_up_kernel(kg, kd, T):
    key = ("jangt", "jangt", kg, kd, True, OAI, int(kg * 10) + T)
    m = build(*key)
    x, idx, _ = inputs(T, seed=T)
    g, u = pre_act(m, key, x, idx)
    _exercises_limits(g, u)
    ref = act_ref(g, u, *OAI)
    err = rel(m._decode_h(x, idx), ref)
    assert err < TOL_DECODE, err
    for ctl in CONTROLS:
        c = build(*key[:5], ctl, key[6])
        assert rel(c._decode_h(x, idx), ref) > CTL * TOL_DECODE, ctl


def test_decode_gate_up_rot_out_epilogue():
    """rot_out (the activations Hadamard-32 rotated in the epilogue, for a JANGH down) applies the same variant."""
    key = ("jangt", "jangt", 2, 2.5, True, OAI, 11)
    m = build(*key)
    x, idx, _ = inputs(3, seed=4)
    ref = hblk(act_ref(*pre_act(m, key, x, idx), *OAI), 32)
    U = m._unit()
    got = MK.gather_gate_up_swiglu_v3(MK.rotate_rows(x, U["su_gu"]), x, idx, U, out_dtype=mx.float32, limit=LIMIT,
                                      rot_out=True, act="oai", alpha=ALPHA)
    assert rel(got, ref) < TOL_DECODE
    silu = MK.gather_gate_up_swiglu_v3(MK.rotate_rows(x, U["su_gu"]), x, idx, U, out_dtype=mx.float32, limit=LIMIT,
                                       rot_out=True)
    assert rel(silu, ref) > CTL * TOL_DECODE


@pytest.mark.parametrize("n_out", [True, False])
def test_decode_route_end_to_end(n_out):
    key = ("jangt", "jangt", 2, 2.5, n_out, OAI, 21)
    m = build(*key)
    x, idx, w = inputs(3, seed=7)
    assert idx.size < SW.SORT_THRESHOLD
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *OAI), idx, w)
    assert rel(m.routed(x, idx, w), ref) < TOL_ROUTED
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * TOL_ROUTED, ctl


# ------------------------------------------------------------------------------------------------ prefill (NAX)
def _sorted(idx):
    T, k = idx.shape
    flat = idx.reshape(-1)
    order = mx.argsort(flat)
    return flat[order].astype(mx.uint32), order // k, order


@needs_nax
@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("n_out", [True, False])
def test_prefill_fused_gate_up_kernels(n_out, tiles):
    """gather_gate_up_sorted_jt (outlier columns in the epilogue) / gather_qmm_sorted_jt fused (no outliers)."""
    key = ("jangt", "jangt", 2, 2.5, n_out, OAI, 31)
    m = build(*key)
    x, idx, _ = inputs(37, seed=8, experts=6)                   # experts 6, 7 get no rows
    g, u = pre_act(m, key, x, idx)
    _exercises_limits(g, u)
    ref = act_ref(g, u, *OAI)
    idx_s, tok, order = _sorted(idx)
    U = m._unit()
    plan = JK.expert_tile_plan(idx_s, E) if tiles else None
    xr = MK.rotate_rows(x, U["su_gu"]).astype(mx.bfloat16)[tok]

    def run(kind, limit, alpha):
        if n_out:
            xo = x[:, U["gu_cols"]].astype(mx.float32)[tok]
            h = PF.gather_gate_up_sorted_jt(xr, xo, U["gate"], U["up"], idx_s, limit=limit, plan=plan, act=kind, alpha=alpha)
        else:
            h = PF.gather_qmm_sorted_jt(xr, U["gate"], idx_s, U["up"], limit=limit, plan=plan, act=kind, alpha=alpha)
        return h[mx.argsort(order)]

    assert rel(run("oai", LIMIT, ALPHA), ref) < TOL_PREFILL
    for ctl in CONTROLS:
        assert rel(run(*ctl), ref) > CTL * TOL_PREFILL, ctl


@needs_nax
@pytest.mark.parametrize("path", ["fused_outliers", "fused_plain", "unfused"])
@pytest.mark.parametrize("tiles", ["0", "1"])
def test_prefill_route_end_to_end(path, tiles, monkeypatch):
    """routed() on a prefill batch (148 slots >= SORT_THRESHOLD): every _prefill gate/up branch, both schedules."""
    monkeypatch.setattr(SW, "FUSED_GU_PREFILL", path != "unfused")
    monkeypatch.setenv("JANGT_EXPERT_TILES", tiles)
    key = ("jangt", "jangt", 2, 2.5, path != "fused_plain", OAI, 41)
    m = build(*key)
    x, idx, w = inputs(37, seed=9, experts=6)
    assert idx.size >= SW.SORT_THRESHOLD
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *OAI), idx, w)
    assert rel(m.routed(x, idx, w), ref) < TOL_PREFILL
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * TOL_PREFILL, ctl


@pytest.mark.parametrize("T", [3, 37])
def test_jangt_gate_up_feeding_jangh_down(T):
    """JANGT gate/up + JANGH down: decode (T=3) and the sorted prefill (T=37, H32 store epilogue on NAX)."""
    if T == 37 and not NAX:
        pytest.skip("NAX prefill")
    key = ("jangt", "jangtq2", 2, None, True, OAI, 51)
    m = build(*key)
    x, idx, w = inputs(T, seed=10, experts=6)
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *OAI), idx, w)
    tol = TOL_ROUTED if T == 3 else TOL_PREFILL
    assert rel(m.routed(x, idx, w), ref) < tol
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * tol, ctl


@pytest.mark.parametrize("T", [3, 37])
def test_jangh_gate_up_units(T):
    """JANGH gate/up (its fused epilogue is SiLU only): two single-weight passes + the variant; decode and prefill."""
    if T == 37 and not NAX:
        pytest.skip("NAX prefill")
    key = ("jangtq2", "jangt", None, 2.5, True, OAI, 61)
    m = build(*key)
    x, idx, w = inputs(T, seed=11, experts=6)
    g, u = pre_act(m, key, x, idx)
    _exercises_limits(g, u)
    ref_h = act_ref(g, u, *OAI)
    ref = down_ref(m, key, ref_h, idx, w)
    tol = TOL_ROUTED if T == 3 else TOL_PREFILL
    if T == 3:
        assert rel(m._decode_h(x, idx), ref_h) < 1e-5            # JANGH codes are exact; only the fp32 dot remains
    assert rel(m.routed(x, idx, w), ref) < tol
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * tol, ctl


# ------------------------------------------------------------------------------------------------ fused routed + shared decode
def _mx8(o, i, key):
    lin = nn.QuantizedLinear(i, o, bias=False, group_size=32, bits=8, mode="mxfp8")
    w = (mx.random.normal((o, i), key=key) * 0.3).astype(mx.bfloat16)
    lin.weight, lin.scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    return lin


@pytest.mark.parametrize("R", [1, 3, 8])
def test_fused_moe_decode_oai_bit_identical_to_stock(R):
    """fused_moe (routed + mxfp8 shared expert, both SwiGLU-OAI) == its stock path, row by row, bit for bit."""
    FM._STATE["disabled"] = False
    FM._STATE["checked"].clear()
    m = build("jangt", "jangt", 1.5, 2, True, OAI, 71)
    k1, k2, k3 = mx.random.split(mx.random.key(R), 3)
    shared = SimpleNamespace(w1=_mx8(I, D, k1), w3=_mx8(I, D, k2), w2=_mx8(D, I, k3))
    rng = np.random.default_rng(R)
    x = mx.array((rng.standard_normal((R, D)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.choice(E, TOPK, replace=False) for _ in range(R)]).astype(np.int32))
    w = mx.array(rng.uniform(0.05, 0.5, size=(R, TOPK)).astype(np.float32))
    sg, su = shared.w1(x).astype(mx.float32), shared.w3(x).astype(mx.float32)
    assert float(mx.mean(sg > LIMIT).item()) > 0.02 and float(mx.mean(mx.abs(su) > LIMIT).item()) > 0.04   # shared clamp
    y = FM._run(x, idx, w, m, shared, LIMIT, "oai", ALPHA)
    ref = mx.concatenate([FM._stock_row(x[i:i + 1], idx[i:i + 1], w[i:i + 1], m, shared, LIMIT, "oai", ALPHA)
                          for i in range(R)], 0)
    mx.eval(y, ref)
    assert np.abs(np.array(ref.astype(mx.float32))).max() > 0
    assert mx.array_equal(y, ref).item(), float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item())
    y2 = FM.jt_moe_decode(x, idx, w, m, shared, LIMIT, "oai", ALPHA)
    assert y2 is not None and mx.array_equal(y2, ref).item()
    # control: a SiLU shared expert on the same inputs is a different result
    assert not mx.array_equal(FM._run(x, idx, w, m, shared, LIMIT), ref).item()


def test_fused_moe_routed_part_matches_reference():
    """fused_moe's routed slots apply the experts' own variant (OAI here), whatever the shared expert's is."""
    FM._STATE["disabled"] = False
    FM._STATE["checked"].clear()
    key = ("jangt", "jangt", 1.5, 2, True, OAI, 72)
    m = build(*key)
    k1, k2, k3 = mx.random.split(mx.random.key(0), 3)
    zero = SimpleNamespace(w1=_mx8(I, D, k1), w3=_mx8(I, D, k2), w2=_mx8(D, I, k3))
    zero.w2.scales = mx.zeros_like(zero.w2.scales)                 # e8m0 code 0 = 2^-127: shared output ~1e-37
    x, idx, w = inputs(2, seed=12)
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *OAI), idx, w)
    assert rel(FM._run(x, idx, w, m, zero, LIMIT), ref) < TOL_ROUTED            # SiLU shared, OAI routed
    assert rel(FM._run(x, idx, w, build(*key[:5], CONTROLS[1], key[6]), zero, LIMIT), ref) > CTL * TOL_ROUTED


# ------------------------------------------------------------------------------------------------ kernel identity
def _built(monkeypatch, factory, *args, **kwargs):
    """(name, source + header) a kernel factory hands to mx.fast.metal_kernel (lru_cache bypassed)."""
    seen = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **k: seen.append(k) or object())
    factory.__wrapped__(*args, **kwargs)
    k = seen[-1]
    return k["name"], k["source"] + k.get("header", "")


def test_default_kernels_are_unchanged_and_variants_get_their_own_names(monkeypatch):
    """MLX caches compiled kernels by NAME, so every variant needs its own name; the default (no act given) must be
    exactly the source and name of an explicit SiLU request, with no OAI text in it."""
    cases = [
        (MK._gu_v3, (D, I, 2, 12, 3, 100, "float", "float", 0.0)),
        (MK._gu_v3, (D, I, 2.5, 12, 3, 100, "float", "float", LIMIT, True)),          # clamped, rot_out
        (PF._kernel, (2, 100, True, "bfloat16_t", True, True)),
        (PF._kernel_out, (2, 100, 3, "bfloat16_t", True, False, "half")),
        (FM._gu_kernel, (D, I, 2, 12, 3, 100, 4, LIMIT, True)),
    ]
    for factory, args in cases:
        name, src = _built(monkeypatch, factory, *args)
        assert (name, src) == _built(monkeypatch, factory, *args, act="silu", alpha=0.0), factory
        assert "oai" not in name and "+ 1.0f)" not in src
        oname, osrc = _built(monkeypatch, factory, *args, act="oai", alpha=ALPHA)
        assert oname != name and "oai1p702" in oname and "+ 1.0f)" in osrc
    # fused_moe: routed and shared variants combine into four distinct kernels
    names = {_built(monkeypatch, FM._gu_kernel, D, I, 2, 12, 3, 100, 4, LIMIT, True, a, ALPHA, s_, ALPHA, LIMIT)[0]
             for a in ("silu", "oai") for s_ in ("silu", "oai")}
    assert len(names) == 4
    assert MK.act_tag("silu", 0.0) == "" and MK.act_tag("silu", 10.0) == "_lim10p0"
    assert MK.act_tag("oai", LIMIT, ALPHA) == "_oai1p702_lim7p0"
    with pytest.raises(ValueError):
        MK.check_act("gelu")

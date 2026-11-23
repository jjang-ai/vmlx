"""Inkling-2 JANGHT routed experts: the soft-capped activation and per-group format dispatch.

    h = softcap(g, cg) * sigmoid(g) * softcap(u, cu),   softcap(x, c) = tanh(x / c) * c,   (cg, cu) = (4, 25)

(Inkling-2-Mini's mlp_gate_softcap / mlp_up_softcap; the sigmoid sees the raw gate.) Every JTMixedSwitchGLU path that
applies it is compared with a float64 NumPy reference built from the units' own bits (trellis states unpacked and valued
by jangt.v2l, JANGH codes by jangh.format, native NVFP4 / MXFP8 by mx.dequantize, Hadamard rotations as explicit
Sylvester matrices): the v3 decode gate/up kernel, the decode route end to end, the NAX sorted prefill (fused gate/up
with outlier columns in the epilogue, fused without outliers, unfused trellis matmuls), JANGH gate/up units, and native
groups mixed with JANGT / JANGH groups (the Inkling JANGHT bundle's per-unit dispatch). Shapes are Inkling-2-Mini's
D = 2304, I = 1152: neither is a multiple of the 512-value decode block, so both the gate/up (2304) and the down (1152)
input widths run the masked tail lanes. 37 tokens x 4 slots = 148 prefill rows is not a multiple of the 64-row tile,
and 2 of 8 experts receive no prefill rows.

Controls: the same comparisons with the SiLU and SwiGLU-OAI kernels must miss the tolerance by >= 10x, so the tests
can tell the activations apart. Also: the loader contract (fail closed) and the decode-row invariance a speculative
verify relies on.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from vmlx_engine.jangh import format as JF  # noqa: E402
from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import prefill as PF  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402

D, I, E, TOPK = 2304, 1152, 8, 4
CAPS = (4.0, 25.0)
INK = ("inkling", 0.0, CAPS)
CONTROLS = (("silu", 0.0, 0.0), ("oai", 0.0, 1.702))

# Relative L2 error vs the float64 reference. Measured on M5 Ultra / MLX 0.32.4 (2026-11-23): decode h 8.6-9.0e-4,
# decode route 1.7-3.0e-3 (every group pair; native downs take bf16 h), NAX prefill route 3.9-5.6e-3. Controls (SiLU,
# SwiGLU-OAI on the same weights): >= 3.78 everywhere. Each tolerance is ~3x the measured error, >> 10x below controls.
TOL_DECODE = 3e-3        # fp16 x / half2 trellis dot, fp32 epilogue (gate/up activations)
TOL_ROUTED = 9e-3        # decode route end to end (+ the down projection; bf16 h into a native down)
TOL_PREFILL = 1.5e-2     # bf16 x and weight tiles in the NAX matmul, bf16 h stored between gate/up and down
CTL = 10

NAX = JK.nax_available()
needs_nax = pytest.mark.skipif(not NAX, reason="NAX prefill kernels need macOS 26.2+ on M5")
needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


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
        wo *= (ocol >= 0)[:, None, :]
        p.jt_wout = mx.array(wo.astype(np.float16))


def _fill_tq(p, rng, lo, hi):
    p.tq2_packed = _rand_words(rng, p.tq2_packed.shape)
    p.tq2_scales = mx.array(rng.uniform(lo, hi, size=p.tq2_scales.shape).astype(np.float16))


def _fill_native(m, name, mode, rng, std):
    """Quantize the placeholder the way the loader does (nn.quantize with the entry), then random native bytes."""
    from vmlx_engine.models.inkling2.nvfp4 import replace_scaled_quantized_linears

    gs, bits = SW.NATIVE_MODES[mode]
    nn.quantize(m, class_predicate=lambda p, mod: {"group_size": gs, "bits": bits, "mode": mode} if p == name else False)
    lin = getattr(m, name)
    w = mx.array((rng.standard_normal(lin.weight.shape[:2] + (lin.input_dims,)) * std).astype(np.float32))
    lin.weight, lin.scales = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
    if mode == "nvfp4":
        # the bundle's per-expert MLX-convention global scale (w = e2m1 * e4m3 * g / 2688)
        g = mx.array(rng.uniform(1500.0, 4000.0, size=(E,)).astype(np.float32))
        replace_scaled_quantized_linears(m, {f"{name}.global_scale": g})


GU_COLS = np.array([5, 77, 2001], dtype=np.int32)


def _dn_cols(rng):
    c = np.stack([rng.choice(I, size=2, replace=False) for _ in range(E)]).astype(np.int32)
    c[3, 1] = -1
    return c


ENTRIES = {
    "jangt_gu": lambda k: {"mode": "jangt", "kbits": k}, "jangt_dn": lambda k: {"mode": "jangt", "kbits": k},
    "jangtq2": lambda b: {"mode": "jangtq2", "bits": b, "rotation": "hadamard32"},
    "nvfp4": lambda _: {"mode": "nvfp4", "group_size": 16, "bits": 4},
    "mxfp8": lambda _: {"mode": "mxfp8", "group_size": 32, "bits": 8},
}


def build(gu="jangt", dn="jangt", kg=3, kd=2.5, n_out=True, act=INK, seed=0):
    """A JTMixedSwitchGLU with deterministic random units (same seed -> same weights for every act)."""
    rng = np.random.default_rng(seed)
    gu_e = ENTRIES["jangt_gu"](kg) if gu == "jangt" else ENTRIES[gu](kg)
    dn_e = ENTRIES["jangt_dn"](kd) if dn == "jangt" else ENTRIES[dn](kd)
    no_gu = 3 if (n_out and gu == "jangt") else 0
    no_dn = 2 if (n_out and dn == "jangt") else 0
    kind, limit, alpha = act
    m = SW.JTMixedSwitchGLU(D, I, E, gu_e, dn_e, no_gu, no_dn, limit, kind, alpha)
    # gate rows put |g| well past the gate cap (4) and up rows |u| past the up cap (25) for a sizeable share
    if gu == "jangt":
        cols = np.tile(GU_COLS, (E, 1)) if no_gu else None
        _fill_jt(m.gate_proj, rng, 0.2, 0.45, cols)
        _fill_jt(m.up_proj, rng, 0.8, 1.6, cols)
        m.up_proj.jt_su = m.gate_proj.jt_su
    elif gu == "jangtq2":
        _fill_tq(m.gate_proj, rng, 0.2, 0.45)
        _fill_tq(m.up_proj, rng, 0.8, 1.6)
    else:
        _fill_native(m, "gate_proj", gu, rng, 0.3)
        _fill_native(m, "up_proj", gu, rng, 1.0)
    if dn == "jangt":
        _fill_jt(m.down_proj, rng, 0.004, 0.008, _dn_cols(rng) if no_dn else None)
        m.down_proj.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=m.down_proj.jt_cs.shape).astype(np.float32))
    elif dn == "jangtq2":
        _fill_tq(m.down_proj, rng, 0.004, 0.008)
    else:
        _fill_native(m, "down_proj", dn, rng, 0.006)
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
    with np.errstate(all="ignore"):
        return (x.reshape(*x.shape[:-1], -1, n) @ _sylvester(n)).reshape(x.shape)


def _np(a, dt=np.float64):
    return np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).astype(dt)


def dense_jt(p):
    out = []
    for e in range(p.num_experts):
        st = VL.unpack_states(np.array(p.jt_packed[e]), p.input_dims, 12, p.kbits)
        v = np.array(VL.values_of(mx.array(st), 1.0, 12, p.kbits)).astype(np.float64)
        out.append(v * _np(p.jt_scale[e])[:, None])
    return np.stack(out)


def dense_tq(p):
    q = np.array(JF.unpack_bitstream(p.tq2_packed, p.bits, p.input_dims)).astype(np.int64)
    return JF.codebook(p.bits).astype(np.float64)[q] * _np(p.tq2_scales)[..., None]


def dense_native(p):
    """(E, N, K) float64 raw-basis rows of an MLX quantized switch projection (per-expert global scale for NVFP4)."""
    rows = []
    for e in range(p.num_experts):
        kw = {}
        if hasattr(p, "mlx_global_scale"):
            kw["global_scale"] = p.mlx_global_scale()[e:e + 1]
        rows.append(np.array(mx.dequantize(p.weight[e], p.scales[e], group_size=p.group_size, bits=p.bits, mode=p.mode,
                                           dtype=mx.float32, **kw)).astype(np.float64))
    return np.stack(rows)


def act_ref(g, u, kind, limit, alpha):
    if limit > 0:
        g, u = np.minimum(g, limit), np.clip(u, -limit, limit)
    if kind == "inkling":
        cg, cu = alpha
        return np.tanh(g / cg) * cg / (1.0 + np.exp(-g)) * (np.tanh(u / cu) * cu)
    if kind == "oai":
        return g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)
    return g / (1.0 + np.exp(-g)) * u


_DENSE = {}


def _dense(m, key):
    if key not in _DENSE:
        fn = {"jangt": dense_jt, "jangtq2": dense_tq, "native": dense_native}
        _DENSE[key] = {"gu": (fn[m.gu_fmt](m.gate_proj), fn[m.gu_fmt](m.up_proj)), "dn": fn[m.dn_fmt](m.down_proj)}
    return _DENSE[key]


def _per_expert(W, X, e):
    """out[s] = W[e[s]] @ X[s], grouped by expert (no (S, N, K) temporaries)."""
    out = np.zeros((X.shape[0], W.shape[1]))
    for ex in np.unique(e):
        sel = e == ex
        with np.errstate(all="ignore"):
            out[sel] = X[sel] @ W[ex].T
    return out


def pre_act(m, key, x, idx):
    Wg, Wu = _dense(m, key)["gu"]
    xn = _np(x)
    T, k = idx.shape
    e = np.array(idx).reshape(-1)
    rows = np.repeat(np.arange(T), k)
    if m.gu_fmt == "jangt":
        gp, upp = m.gate_proj, m.up_proj
        oc = np.array(gp.jt_ocol)
        mask = np.ones(D)
        mask[oc[0][oc[0] >= 0]] = 0.0
        xr = hblk(xn * _np(gp.jt_su) * mask, 128)[rows]
        g, u = _per_expert(Wg, xr, e), _per_expert(Wu, xr, e)
        if oc.max() >= 0:
            wg, wu = _np(gp.jt_wout), _np(upp.jt_wout)
            for j in range(oc.shape[1]):
                c = oc[e, j]
                xo = np.where(c >= 0, xn[rows, np.maximum(c, 0)], 0.0)
                g += wg[e, :, j] * xo[:, None]
                u += wu[e, :, j] * xo[:, None]
    elif m.gu_fmt == "jangtq2":
        xr = hblk(xn, 32)[rows]
        g, u = _per_expert(Wg, xr, e), _per_expert(Wu, xr, e)
    else:
        g, u = _per_expert(Wg, xn[rows], e), _per_expert(Wu, xn[rows], e)
    return g, u


def down_ref(m, key, h, idx, w):
    Wd = _dense(m, key)["dn"]
    T, k = idx.shape
    e = np.array(idx).reshape(-1)
    if m.dn_fmt == "jangt":
        d = m.down_proj
        oc = np.array(d.jt_ocol)
        mask = np.ones((E, I))
        for ex in range(E):
            mask[ex, oc[ex][oc[ex] >= 0]] = 0.0
        y = _per_expert(Wd, hblk(h * _np(d.jt_cs)[e] * _np(d.jt_su)[None, :] * mask[e], 128), e)
        if oc.max() >= 0:
            wo = _np(d.jt_wout)
            for j in range(oc.shape[1]):
                c = oc[e, j]
                ho = np.where(c >= 0, h[np.arange(h.shape[0]), np.maximum(c, 0)], 0.0)
                y += wo[e, :, j] * ho[:, None]
    elif m.dn_fmt == "jangtq2":
        y = _per_expert(Wd, hblk(h, 32), e)
    else:
        y = _per_expert(Wd, h, e)
    return (y.reshape(T, k, D) * _np(w)[..., None]).sum(axis=1)


def rel(got, ref):
    got = _np(got) if isinstance(got, mx.array) else got
    assert got.size and ref.size, "INVALID: empty comparison"
    assert np.isfinite(got).all() and np.isfinite(ref).all()
    nr = float(np.linalg.norm(ref))
    assert nr > 0, "INVALID: empty reference"
    return float(np.linalg.norm(got - ref)) / nr


def _exercises_caps(g, u):
    """The fixture must reach the caps: gates past +-4 and ups past +-25 for a sizeable share of rows."""
    assert np.mean(np.abs(g) > CAPS[0]) > 0.2 and np.mean(np.abs(u) > CAPS[1]) > 0.1, (
        np.mean(np.abs(g) > CAPS[0]), np.mean(np.abs(u) > CAPS[1]))


# ------------------------------------------------------------------------------------------------ activation
def test_act_mlx_matches_the_model_activation():
    """act_mlx("inkling") == the Inkling model's own _softcapped_swiglu (float32), and both == the float64 formula."""
    from vmlx_engine.models.inkling2.language import _softcapped_swiglu
    rng = np.random.default_rng(6)
    g = (rng.standard_normal(8192) * 9).astype(np.float32)
    u = (rng.standard_normal(8192) * 40).astype(np.float32)
    model = np.array(_softcapped_swiglu(mx.array(u), mx.array(g), *CAPS))
    ours = np.array(MK.act_mlx(mx.array(g), mx.array(u), "inkling", 0.0, CAPS))
    ref = act_ref(g.astype(np.float64), u.astype(np.float64), *INK)
    np.testing.assert_allclose(ours, ref, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model, ref, rtol=3e-6, atol=3e-6)


def test_act_param_and_kernel_names():
    assert MK.act_param("inkling", [4, 25]) == (4.0, 25.0)
    assert MK.act_param("oai", 1.702) == 1.702 and MK.act_param("silu", 3.0) == 0.0
    for bad in (None, 4.0, (4.0,), (4.0, 0.0), (float("inf"), 25.0), (-4.0, 25.0)):
        with pytest.raises(ValueError):
            MK.act_param("inkling", bad)
    with pytest.raises(ValueError):
        MK.check_act("softcap")
    # every source variant has its own kernel name; the SiLU / OAI names are unchanged
    tags = {MK.act_tag("inkling", 0.0, CAPS), MK.act_tag("inkling", 0.0, (4.0, 30.0)), MK.act_tag("silu", 0.0),
            MK.act_tag("oai", 7.0, 1.702)}
    assert len(tags) == 4
    assert MK.act_tag("silu", 0.0) == "" and MK.act_tag("silu", 10.0) == "_lim10p0"
    assert MK.act_tag("oai", 7.0, 1.702) == "_oai1p702_lim7p0"
    assert MK.act_tag("inkling", 0.0, CAPS) == "_ink40800000_41c80000"


@needs_gpu
def test_metal_expression_matches_float64():
    """The generated Metal expression (precise tanh / exp, caps by bit pattern) on raw values, through a 1-op kernel."""
    rng = np.random.default_rng(7)
    g = (rng.standard_normal(4096) * 9).astype(np.float32)
    u = (rng.standard_normal(4096) * 40).astype(np.float32)
    k = mx.fast.metal_kernel(name="test_inkling_act_expr", input_names=["gi", "ui"], output_names=["out"],
                             source=f"uint i = thread_position_in_grid.x; float g = gi[i], u = ui[i];"
                                    f" out[i] = {MK.act_value_metal('inkling', CAPS)};")
    got = np.array(k(inputs=[mx.array(g), mx.array(u)], grid=(4096, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(4096,)], output_dtypes=[mx.float32])[0])
    np.testing.assert_allclose(got, act_ref(g.astype(np.float64), u.astype(np.float64), *INK), rtol=2e-6, atol=2e-6)


# ------------------------------------------------------------------------------------------------ decode
@needs_gpu
@pytest.mark.parametrize("kg", [2, 3])
@pytest.mark.parametrize("T", [1, 3])
def test_decode_gate_up_kernel(kg, T):
    key = ("jangt", "jangt", kg, 2.5, True, INK, kg * 10 + T)
    m = build(*key)
    x, idx, _ = inputs(T, seed=T)
    g, u = pre_act(m, key, x, idx)
    _exercises_caps(g, u)
    ref = act_ref(g, u, *INK)
    err = rel(m._decode_h(x, idx), ref)
    assert err < TOL_DECODE, err
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6])._decode_h(x, idx), ref) > CTL * TOL_DECODE, ctl


ROUTES = [("jangt", "jangt"), ("jangt", "jangtq2"), ("jangtq2", "jangt"), ("jangtq2", "jangtq2"),
          ("jangtq2", "mxfp8"), ("nvfp4", "jangt"), ("jangt", "nvfp4"), ("mxfp8", "jangtq2")]


def _key(gu, dn, n_out, seed):
    return (gu, dn, 3 if gu == "jangt" else 3, 2.5 if dn == "jangt" else 4, n_out, INK, seed)


@needs_gpu
@pytest.mark.parametrize("gu,dn", ROUTES)
def test_decode_route_end_to_end(gu, dn):
    """routed() on 3 decode rows (12 slots < SORT_THRESHOLD) for every group-format pair the bundle may hold."""
    key = _key(gu, dn, True, 21)
    m = build(*key)
    x, idx, w = inputs(3, seed=7)
    assert idx.size < SW.SORT_THRESHOLD
    g, u = pre_act(m, key, x, idx)
    _exercises_caps(g, u)
    ref = down_ref(m, key, act_ref(g, u, *INK), idx, w)
    err = rel(m.routed(x, idx, w), ref)
    assert err < TOL_ROUTED, err
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * TOL_ROUTED, ctl


@needs_gpu
@pytest.mark.parametrize("gu,dn", [("jangt", "jangtq2"), ("nvfp4", "jangt"), ("jangtq2", "mxfp8")])
def test_verify_rows_equal_decode_rows(gu, dn):
    """A speculative verify block (8 rows x top-4 < SORT_THRESHOLD) gives every row bit-for-bit its decode result."""
    m = build(*_key(gu, dn, True, 25))
    x, idx, w = inputs(8, seed=12)
    block = m.routed(x, idx, w)
    rows = mx.concatenate([m.routed(x[i:i + 1], idx[i:i + 1], w[i:i + 1]) for i in range(8)], axis=0)
    mx.eval(block, rows)
    assert float(mx.abs(block.astype(mx.float32)).max().item()) > 0
    assert mx.array_equal(block, rows).item()


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
    key = ("jangt", "jangt", 3, 2.5, n_out, INK, 31)
    m = build(*key)
    x, idx, _ = inputs(37, seed=8, experts=6)
    g, u = pre_act(m, key, x, idx)
    _exercises_caps(g, u)
    ref = act_ref(g, u, *INK)
    idx_s, tok, order = _sorted(idx)
    U = m._unit()
    plan = JK.expert_tile_plan(idx_s, E) if tiles else None
    xr = MK.rotate_rows(x, U["su_gu"]).astype(mx.bfloat16)[tok]

    def run(kind, limit, alpha):
        if n_out:
            xo = x[:, U["gu_cols"]].astype(mx.float32)[tok]
            h = PF.gather_gate_up_sorted_jt(xr, xo, U["gate"], U["up"], idx_s, limit=limit, plan=plan, act=kind,
                                            alpha=alpha)
        else:
            h = PF.gather_qmm_sorted_jt(xr, U["gate"], idx_s, U["up"], limit=limit, plan=plan, act=kind, alpha=alpha)
        return h[mx.argsort(order)]

    err = rel(run(*INK), ref)
    assert err < TOL_PREFILL, err
    for ctl in CONTROLS:
        assert rel(run(*ctl), ref) > CTL * TOL_PREFILL, ctl


@needs_nax
@pytest.mark.parametrize("path", ["fused_outliers", "fused_plain", "unfused"])
def test_prefill_route_trellis(path, monkeypatch):
    monkeypatch.setattr(SW, "FUSED_GU_PREFILL", path != "unfused")
    key = ("jangt", "jangt", 3, 2.5, path != "fused_plain", INK, 41)
    m = build(*key)
    x, idx, w = inputs(37, seed=9, experts=6)
    assert idx.size >= SW.SORT_THRESHOLD
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *INK), idx, w)
    err = rel(m.routed(x, idx, w), ref)
    assert err < TOL_PREFILL, err
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * TOL_PREFILL, ctl


@needs_nax
@pytest.mark.parametrize("gu,dn", ROUTES[1:])
def test_prefill_route_mixed_groups(gu, dn):
    """The sorted prefill for every mixed pair: JANGT gate/up into a JANGH down (H32 store epilogue), JANGH gate/up,
    native gate/up (sorted gather_qmm) and native downs."""
    key = _key(gu, dn, True, 45)
    m = build(*key)
    x, idx, w = inputs(37, seed=10, experts=6)
    ref = down_ref(m, key, act_ref(*pre_act(m, key, x, idx), *INK), idx, w)
    err = rel(m.routed(x, idx, w), ref)
    assert err < TOL_PREFILL, err
    for ctl in CONTROLS:
        assert rel(build(*key[:5], ctl, key[6]).routed(x, idx, w), ref) > CTL * TOL_PREFILL, ctl


# ------------------------------------------------------------------------------------------------ loader contract
def _config(layers=3, **over):
    quant = {"group_size": 64, "bits": 8, "mode": "affine"}
    fmts = [("jangt", "jangtq2"), ("nvfp4", "nvfp4"), ("jangtq2", "mxfp8")]
    for L in range(layers):
        gu, dn = fmts[L % len(fmts)]
        ent = {"jangt": {"mode": "jangt", "kbits": 3}, "jangtq2": {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"},
               "nvfp4": {"mode": "nvfp4", "group_size": 16, "bits": 4},
               "mxfp8": {"mode": "mxfp8", "group_size": 32, "bits": 8}}
        for p, f in (("gate_proj", gu), ("up_proj", gu), ("down_proj", dn)):
            quant[f"model.layers.{L}.mlp.switch_mlp.{p}"] = dict(ent[f])
    from vmlx_engine.jangh.contract import CUBIC_PARAMS
    books = {}
    for b in (2, 3, 4):
        a, be = CUBIC_PARAMS[b]
        books[str(b)] = {"alpha": a, "beta": be,
                         "levels": [float(np.float32((i - ((1 << b) - 1) / 2) * (a + be * ((i - ((1 << b) - 1) / 2) ** 2))))
                                    for i in range(1 << b)]}
    cfg = {"model_type": "inkling2_loop", "quantization": quant,
           "jangtq": {"version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "codebook_family": "odd-cubic",
                      "rotation": "hadamard32", "codebooks": books},
           "jangt": {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                     "outlier_columns": {"model.layers.0.mlp.switch_mlp.gate_proj": 2,
                                         "model.layers.0.mlp.switch_mlp.up_proj": 2}}}
    cfg.update(over)
    return cfg


def test_contract_accepts_the_bundle_layout():
    from vmlx_engine.models.inkling2.janght import is_janght_config, routed_contract
    cfg = _config()
    assert is_janght_config(cfg)
    c = routed_contract(cfg, 3)
    assert [(c[L]["gu_fmt"], c[L]["dn_fmt"]) for L in range(3)] == [
        ("jangt", "jangtq2"), ("native", "native"), ("jangtq2", "native")]
    assert (c[0]["n_out_gu"], c[0]["n_out_dn"]) == (2, 0)
    assert not is_janght_config({k: v for k, v in cfg.items() if k != "jangt"})     # all-JANGH bundle: generic route


def _mutate(fn):
    cfg = _config()
    fn(cfg)
    return cfg


@pytest.mark.parametrize("label,mutation", [
    ("gate/up differ", lambda c: c["quantization"].update({"model.layers.0.mlp.switch_mlp.up_proj": {"mode": "jangt", "kbits": 2}})),
    ("unknown mode", lambda c: c["quantization"].update({"model.layers.1.mlp.switch_mlp.down_proj": {"mode": "fp8", "bits": 8}})),
    ("nvfp4 group", lambda c: c["quantization"].update({"model.layers.1.mlp.switch_mlp.down_proj": {"mode": "nvfp4", "group_size": 32, "bits": 4}})),
    ("jangt kbits", lambda c: c["quantization"].update({"model.layers.0.mlp.switch_mlp.gate_proj": {"mode": "jangt", "kbits": 5},
                                                        "model.layers.0.mlp.switch_mlp.up_proj": {"mode": "jangt", "kbits": 5}})),
    ("jangtq2 rotation", lambda c: c["quantization"].update({"model.layers.0.mlp.switch_mlp.down_proj": {"mode": "jangtq2", "bits": 3}})),
    ("missing layer", lambda c: c["quantization"].pop("model.layers.2.mlp.switch_mlp.down_proj")),
    ("extra layer", lambda c: c["quantization"].update({"model.layers.3.mlp.switch_mlp.down_proj": {"mode": "mxfp8", "group_size": 32, "bits": 8}})),
    ("jangt block", lambda c: c["jangt"].update({"state_bits": 16})),
    ("outliers on JANGH", lambda c: c["jangt"]["outlier_columns"].update({"model.layers.0.mlp.switch_mlp.down_proj": 2})),
    ("outliers gate only", lambda c: c["jangt"]["outlier_columns"].update({"model.layers.0.mlp.switch_mlp.up_proj": 3})),
    ("codebook", lambda c: c["jangtq"]["codebooks"]["3"].update({"alpha": 0.5})),
])
def test_contract_fails_closed(label, mutation):
    from vmlx_engine.models.inkling2.janght import routed_contract
    with pytest.raises(ValueError):
        routed_contract(_mutate(mutation), 3)


def test_native_switch_entry_fails_closed():
    with pytest.raises(ValueError):
        SW.group_format({"mode": "mxfp8", "group_size": 16, "bits": 8})
    with pytest.raises(ValueError):
        SW.group_format({"mode": "affine", "group_size": 64, "bits": 4})
    assert SW.group_format({"mode": "nvfp4", "group_size": 16, "bits": 4}) == "native"


def test_global_scale_spellings_agree():
    """weight_scale_2 (amax / 2688), weight_global_scale and the MLX-convention global_scale give one MLX factor."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from vmlx_engine.models.inkling2.nvfp4 import ScaledQuantizedSwitchLinear
    g = mx.array([2048.0, 1344.0], mx.float32)
    q = QuantizedSwitchLinear(32, 8, 2, bias=False, group_size=16, bits=4, mode="nvfp4")
    f = [np.array(ScaledQuantizedSwitchLinear(q, v, n).mlx_global_scale()) for n, v in
         (("global_scale", g), ("weight_scale_2", g / 2688.0), ("weight_global_scale", 2688.0 / g))]
    np.testing.assert_allclose(f[0], np.array(g))
    np.testing.assert_allclose(f[1], f[0], rtol=1e-6)
    np.testing.assert_allclose(f[2], f[0], rtol=1e-6)

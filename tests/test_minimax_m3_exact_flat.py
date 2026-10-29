# SPDX-License-Identifier: Apache-2.0
"""E1 exact multi-row verify kernels (``VMLX_M3_EXACT_FLAT``): each weight read and decoded once per tile of rows, every
row on the one-row decode arithmetic.

Two families:
* the fused attention's K1 / K5 multi-row builds (``fused_attn.qkv_e1`` / ``oproj_e1``: attn_xprep + attn_qkv_e1 /
  attn_oproj_e1) must equal the one-row builds (``qkv`` / ``oproj`` at one row: the decode step) bit for bit;
* ``metal/row_exact_e1`` (shared expert, lm_head, dense MLPs) must equal MLX's stock one-row ``quantized_matmul``.

Controls show each check bites: a perturbed E1 build (one association changed) differs from the one-row build, is
rejected by the admission check, and breaks the DSpark verify == decode equality. Exactness is bit-equality.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import tests.test_minimax_m3_dspark as TD  # noqa: E402
import tests.test_minimax_m3_fused_attn as TF  # noqa: E402
from vmlx_engine.dspark import DSparkStats, generate_ar, generate_dspark, load_drafter  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.metal import row_exact_e1 as E1  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import kernels as K  # noqa: E402

LEVERS = ("VMLX_M3_E1_BITS", "VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
          "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK", "VMLX_MINIMAX_MSA",
          "VMLX_MINIMAX_MSA_SELECT", "VMLX_M3_E1_SHAPE", "VMLX_E1_QMV_SHAPE")


@pytest.fixture(autouse=True)
def _levers(monkeypatch):
    for name in LEVERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(F.ENV, "1")
    monkeypatch.setenv(F.EXACT_FLAT_ENV, "1")
    monkeypatch.setenv("VMLX_M3_E1_BITS", "4,6,8")            # every width the kernels cover (6-bit is opt-in)


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _rows(n: int, width: int, seed: int) -> mx.array:
    """Rows with very different magnitudes side by side (1e-3x .. 300x), random signs, a few spikes."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, width)).astype(np.float32)
    x *= np.exp(rng.uniform(np.log(1e-3), np.log(300.0), (n, 1))).astype(np.float32)
    x[rng.random((n, width)) < 0.004] *= 40.0
    return mx.array(x)


# ------------------------------------------------------------------------------------------------ K1 / K5 builds
@pytest.fixture(scope="module", params=[1024, 6144], ids=["hidden1024", "hidden6144"])
def layer(request):
    return TF.Layer(hidden=request.param, seed=5)


def _one_row(fn, *xs):
    return mx.concatenate([fn(*(x[r:r + 1] for x in xs)) for r in range(int(xs[0].shape[0]))], axis=0)


@pytest.mark.parametrize("rows", list(range(2, 17)))
def test_e1_k1_k5_equal_the_one_row_builds(layer, rows, monkeypatch):
    """K1 (input norm + q | k | v affine-4 + index MXFP8) and K5 (o_proj + residual) E1 builds == the one-row builds,
    every row, at every width 2 .. 16 (balanced tiles, zero-padded rows), M3.1 head geometry at hidden 1024 and the
    real 6144."""
    p = F.plan(layer)
    assert p is not None and F._e1_ok(p)
    x2 = _rows(rows, layer.hidden, 100 + rows).astype(mx.bfloat16)
    o = _rows(rows, p.H * p.D, 200 + rows)
    before = dict(F.E1_STATS)
    got_q, got_o = F.qkv_e1(layer, p, x2), F.oproj_e1(layer, p, o, x2)
    assert F.E1_STATS["qkv"] == before["qkv"] + 1 and F.E1_STATS["oproj"] == before["oproj"] + 1
    monkeypatch.setenv(F.EXACT_FLAT_ENV, "0")
    ref_q = _one_row(lambda a: F.qkv(layer, p, a), x2)
    ref_o = _one_row(lambda a, b: F.oproj(layer, p, a, b), o, x2)
    mx.eval(got_q, got_o, ref_q, ref_o)
    assert float(mx.abs(ref_q).max()) > 0 and float(mx.abs(ref_o.astype(mx.float32)).max()) > 0   # fail closed
    assert _eq(got_q, ref_q), int((got_q != ref_q).sum())
    assert _eq(got_o, ref_o), int((got_o != ref_o).sum())


@pytest.mark.parametrize("shape", ["8,4,4", "8,6,4", "4,4,2", "8,4,3", "16,4,4"])
def test_e1_k5_tile_shapes_are_exact(shape, monkeypatch):
    """Every tile shape (simdgroups, outputs per simdgroup, rows per tile) gives the same bits: a row's arithmetic does
    not depend on the tile it lands in."""
    lay = TF.Layer(hidden=1536, seed=9)
    p = F.plan(lay)
    monkeypatch.setenv("VMLX_M3_E1_SHAPE", shape)
    for rows in (5, 12):
        o = _rows(rows, p.H * p.D, rows)
        x2 = _rows(rows, lay.hidden, 7 * rows).astype(mx.bfloat16)
        got = F.oproj_e1(lay, p, o, x2)
        ref = _one_row(lambda a, b: F.oproj(lay, p, a, b), o, x2)
        assert _eq(got, ref), (shape, rows)


def _perturbed_header():
    """The E1 helpers with ONE association changed: the four-value group summed as (c0 + c1) + (c2 + c3) instead of
    m3a_dot's ((c0 + c1) + c2) + c3."""
    src = K.header()
    old = "d[t][o] += x4.x * float(c[o][0]) + x4.y * float(c[o][1]) + x4.z * float(c[o][2]) + x4.w * float(c[o][3]);"
    assert src.count(old) == 1
    return src.replace(old, "d[t][o] += (x4.x * float(c[o][0]) + x4.y * float(c[o][1])) + "
                            "(x4.z * float(c[o][2]) + x4.w * float(c[o][3]));")


def test_control_a_perturbed_e1_build_is_caught(monkeypatch):
    """Control: the same E1 build with one association changed differs from the one-row build (so the equality
    above is not vacuous at these inputs)."""
    lay = TF.Layer(hidden=1024, seed=5)
    p = F.plan(lay)
    x2 = _rows(8, lay.hidden, 3).astype(mx.bfloat16)
    ref = _one_row(lambda a: F.qkv(lay, p, a), x2)
    hdr = _perturbed_header()
    monkeypatch.setattr(K, "header", lambda: hdr)
    monkeypatch.setattr(K, "_KERNELS", {})
    got = F.qkv_e1(lay, p, x2)
    mx.eval(got, ref)
    assert not _eq(got, ref)
    assert float(mx.abs(got - ref).max()) < 1e-2 * float(mx.abs(ref).max())     # same math, other rounding


def test_one_row_decode_build_untouched_by_the_lever(monkeypatch):
    """The lever never reaches a one-row call: K1 / K5 at one row are bit-identical with it on and off, and no E1
    launch is counted."""
    lay = TF.Layer(hidden=1024, seed=6)
    p = F.plan(lay)
    x2 = _rows(1, lay.hidden, 4).astype(mx.bfloat16)
    o = _rows(1, p.H * p.D, 5)
    before = dict(F.E1_STATS)
    on = (F.qkv(lay, p, x2), F.oproj(lay, p, o, x2))
    assert F.E1_STATS == before
    monkeypatch.setenv(F.EXACT_FLAT_ENV, "0")
    off = (F.qkv(lay, p, x2), F.oproj(lay, p, o, x2))
    assert _eq(on[0], off[0]) and _eq(on[1], off[1])


# ------------------------------------------------------------------------------------------------ MLX-arithmetic E1
def _qlin(n: int, k: int, bits: int, seed: int):
    w = mx.array(np.random.default_rng(seed).standard_normal((n, k), dtype=np.float32) * 0.05).astype(mx.bfloat16)
    t = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(t)
    return t


@pytest.mark.parametrize("bits, k, widths", [(4, 6144, (3072, 3072)), (4, 3072, (6144,)), (4, 6144, (12288,)),
                                             (6, 6144, (4096,)), (8, 6144, (1024,)), (4, 1024, (96, 192)),
                                             (6, 512, (1024,)), (8, 512, (128, 128))])
def test_row_exact_e1_equals_stock_one_row_calls(bits, k, widths):
    """Every row of an E1 call (one or two segments sharing the input) == MLX's own one-row quantized_matmul, at every
    width 2 .. 16, 4 / 6 / 8-bit; M3.1's shared expert / dense MLP / down shapes and small ones."""
    segs = [_qlin(n, k, bits, 31 * i + bits) for i, n in enumerate(widths)]
    for rows in range(2, 17):
        x = _rows(rows, k, rows * 13 + bits).astype(mx.bfloat16)
        got = E1.row_exact_e1(x, segs, bits=bits, group_size=64)
        assert got is not None, (bits, k, widths, rows)
        for (w, s, b), g in zip(segs, got):
            want = mx.concatenate([mx.quantized_matmul(x[r:r + 1], w, s, b, transpose=True, group_size=64, bits=bits)
                                   for r in range(rows)], axis=0)
            assert _eq(g, want), (bits, k, widths, rows, int((g != want).sum()))


def test_control_a_perturbed_e1_walk_is_rejected_by_admission(monkeypatch):
    """Control: a walk with one association changed is REJECTED by the bit-equality admission (None: the caller
    keeps its exact path), and its output really differs from the stock one-row calls."""
    old = "res[t][o] += s[o] * d[t][o] + sm * b[o];"
    assert E1._WALK.count(old) == 1
    monkeypatch.setattr(E1, "_WALK", E1._WALK.replace(old, "res[t][o] = (res[t][o] + s[o] * d[t][o]) + sm * b[o];"))
    monkeypatch.setattr(E1, "_ADMITTED", {})
    monkeypatch.setattr(E1, "_DECISIONS", {})
    E1._walk_kernel.cache_clear()
    try:
        segs = [_qlin(3072, 6144, 4, 1)]
        x = _rows(8, 6144, 2).astype(mx.bfloat16)
        before = E1.STATS["rejected"]
        assert E1.row_exact_e1(x, segs, bits=4, group_size=64) is None
        assert E1.STATS["rejected"] == before + 1
        nsg, o, tr = E1.shape(8, (3072,), 4)
        got = E1._launch(x, segs, 4, 64, nsg, o, tr)[0]
        want = mx.concatenate([mx.quantized_matmul(x[r:r + 1], *segs[0], transpose=True, group_size=64, bits=4)
                               for r in range(8)], axis=0)
        assert not _eq(got, want)
    finally:
        E1._walk_kernel.cache_clear()


def test_row_exact_e1_declines_what_it_does_not_cover():
    """None (the caller's exact path) for one row, float32 activations, scales of another dtype, 3-bit."""
    w, s, b = _qlin(256, 512, 4, 3)
    x = _rows(4, 512, 1)
    assert E1.row_exact_e1(x[:1].astype(mx.bfloat16), [(w, s, b)], bits=4, group_size=64) is None
    assert E1.row_exact_e1(x, [(w, s, b)], bits=4, group_size=64) is None                       # float32 x
    assert E1.row_exact_e1(x.astype(mx.bfloat16), [(w, s.astype(mx.float32), b)], bits=4, group_size=64) is None
    w3, s3, b3 = mx.quantize(mx.zeros((256, 512), dtype=mx.bfloat16), group_size=64, bits=3)
    assert E1.row_exact_e1(x.astype(mx.bfloat16), [(w3, s3, b3)], bits=3, group_size=64) is None


@pytest.mark.parametrize("rows", [17, 23, 33])
def test_row_exact_e1_wider_windows_run_in_exact_chunks(rows):
    """Past 16 rows a call runs as balanced chunks of <= 16 rows; every row still equals its one-row call."""
    w, s, b = _qlin(256, 512, 4, 3)
    x = _rows(rows, 512, rows).astype(mx.bfloat16)
    got = E1.row_exact_e1(x, [(w, s, b)], bits=4, group_size=64)
    assert got is not None
    want = mx.concatenate([mx.quantized_matmul(x[r:r + 1], w, s, b, transpose=True, group_size=64, bits=4)
                           for r in range(rows)], axis=0)
    assert _eq(got[0], want)


# ------------------------------------------------------------------------------------------------ DSpark target
@pytest.fixture(scope="module")
def target():
    t = MiniMaxM3DSparkTarget(TF._model())
    assert [F.unsupported_reason(layer) is None for layer in t.nm.layers] == [False, True, True, True]
    return t


def _verify_vs_decode(target, prompt_len, width):
    prompt = mx.array([TD._ids(prompt_len, prompt_len)])
    rows = mx.array([TD._ids(width, 7 * width)])
    taps = TD.DRAFT_TAPS
    target.verify_paths = {k: 0 for k in target.verify_paths}
    cache = target.make_cache()
    target.prefill(prompt, cache, taps, 256)
    logits, feats, _ = target.verify(rows, cache, taps)
    mx.eval(logits, feats)
    paths = dict(target.verify_paths)
    ref = target.make_cache()
    target.prefill(prompt, ref, taps, 256)
    ref_logits, ref_feats = [], []
    for r in range(width):
        lg, f = target.ar_step(rows[0, r:r + 1], ref, taps)
        ref_logits.append(lg)
        ref_feats.append(f)
    mx.eval(ref_logits)
    bad = [r for r in range(width) if not _eq(logits[r], ref_logits[r])]
    return bad, paths, (feats, mx.concatenate(ref_feats, axis=1)), (cache, ref)


@pytest.mark.parametrize("prompt_len, width", [(773, 3), (765, 4), (509, 5), (1021, 8), (771, 12), (300, 16),
                                               (517, 17), (509, 21)])
def test_verify_rows_equal_decode_steps_with_e1(target, prompt_len, width):
    """With the lever on, row r of a T-row verify == the r-th of T decode steps bit for bit (fused attention on its
    E1 K1 / K5 builds, lm_head / dense MLP / every other affine projection on metal/row_exact_e1), widths 3 .. 21
    (past 16 the fused block runs in chunks); taps and caches equal. Fails closed: E1 must have run."""
    q0, o0 = F.E1_STATS["qkv"], F.E1_STATS["oproj"]
    bad, paths, (feats, ref_feats), (cache, ref) = _verify_vs_decode(target, prompt_len, width)
    assert not bad, f"verify rows {bad} differ from decode steps"
    assert _eq(feats, ref_feats)
    assert not TD._cache_problems(cache, ref)
    assert F.E1_STATS["qkv"] > q0 and F.E1_STATS["oproj"] > o0
    assert paths.get("linear_e1", 0) > 0 and paths.get("attention_fused", 0) == 3


def _row_dependent_header():
    """The E1 helpers with tile rows after the first scaled by (1 + 2^-10) in the per-block epilogue (the bug class
    that matters: a row computed differently because of where it sits in the window)."""
    src = K.header()
    old = "for (int o = 0; o < O; ++o) res[t][o] += s[o] * d[t][o] + b[o] * xs;"
    assert src.count(old) == 1
    return src.replace(old, "for (int o = 0; o < O; ++o) res[t][o] += s[o] * (t > 0 ? 1.0009765625f : 1.0f) * d[t][o]"
                            " + b[o] * xs;")


def test_control_a_row_dependent_e1_build_breaks_verify_equality(target, monkeypatch):
    """Control: with a row-dependent (0.1 %) perturbation in the K1 / K5 E1 build the verify rows no longer equal the
    decode steps. (A one-ulp association change, caught at the layer above, can be absorbed by the bf16 cache /
    residual roundings of this tiny model.)"""
    hdr = _row_dependent_header()
    monkeypatch.setattr(K, "header", lambda: hdr)
    monkeypatch.setattr(K, "_KERNELS", {})
    bad, _, _, _ = _verify_vs_decode(target, 773, 8)
    assert bad


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return TD._save_release_drafter(tmp_path_factory.mktemp("m31_e1") / "dspark")


@pytest.mark.parametrize("mode", ["adaptive", 7, 16])
def test_dspark_greedy_equals_ar_with_e1(target, drafter_dir, mode):
    """Greedy DSpark with the lever == greedy AR (drafts wrong at every fifth position: full, partial and zero
    accepts; Adaptive and pinned widths)."""
    prompt = TD._ids(773, 11)
    reference = generate_ar(target, prompt, 40, set(), prefill_step=256)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    stats = DSparkStats()
    width = mode if isinstance(mode, int) else None
    before = F.E1_STATS["qkv"]
    out = generate_dspark(target, drafter, prompt, 40, set(), prefill_step=256, fixed_width=width, stats=stats,
                          max_draft=max(2 * TD.DRAFT_BLOCK, (width or 0)), ar_fallback=TD._WidthOnly())
    assert out == reference
    assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted
    assert F.E1_STATS["qkv"] > before


def test_six_bit_projections_keep_the_row_split_walk_by_default(target, monkeypatch):
    """Default dispatch: 6-bit projections (the M3.1 lm_head) stay on today's exact walk (measured no faster with E1
    on the real bundle); 4-bit ones take E1. The verify stays exact either way."""
    monkeypatch.delenv("VMLX_M3_E1_BITS")
    assert E1.dispatch_bits() == (4, 8)
    head = target.lm_head
    assert int(head.bits) == 6                                   # affine 6, as the M3.1 lm_head (TF._plan)
    x = _rows(5, TD.D, 9).astype(mx.bfloat16)[None]
    assert target._e1(head, x) is None                           # default: today's row-split walk
    up = target.nm.layers[0].mlp.up_proj                         # dense MLP, affine 4
    assert target._e1(up, x) is not None
    bad, _, _, _ = _verify_vs_decode(target, 765, 6)
    assert not bad
    monkeypatch.setenv("VMLX_M3_E1_BITS", "4,6,8")
    assert target._e1(head, x) is not None


def test_decode_logits_identical_with_the_lever_on_and_off(target, monkeypatch):
    """Plain decoding is untouched: greedy tokens and every step's logits are identical with the lever on and off."""
    prompt = TD._ids(509, 4)
    on = generate_ar(target, prompt, 12, set(), prefill_step=256)
    lg_on = TF._decode_logits(target, prompt, on)
    monkeypatch.setenv(F.EXACT_FLAT_ENV, "0")
    off = generate_ar(target, prompt, 12, set(), prefill_step=256)
    lg_off = TF._decode_logits(target, prompt, off)
    assert on == off
    assert all(_eq(a, b) for a, b in zip(lg_on, lg_off))

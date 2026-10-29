# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 flat dense projections (models/minimax_m3/flat_dense.py + kernels/flat_*.metal, VMLX_M3_FLAT_DENSE=1).

Not MLX's qmv arithmetic by design (decode and verify switch together); checked instead:
  * accuracy: every format (affine 4 / 6 / 8-bit g64, MXFP8, several segments in one launch) against a float64
    reference of the same bf16-rounded activations and dequantized weights; the fused norm prep against GemmaRMSNorm;
    the residual / SwiGLU-OAI / dtype epilogues against references;
  * row invariance: a row alone == the same row inside any 2..16-row launch (sizes that are not multiples of 8 / 16,
    input widths whose weight groups do not split evenly over the simdgroups, column counts that are not multiples
    of 64), rows of very different magnitude side by side, another row's input never changes a row;
  * controls: MLX's own multi-row quantized matmul is not row-invariant (so the check bites); a row-count-dependent
    perturbation of the kernel is caught;
  * blocks: the fused attention half and the fused MoE block with the lever: rows alone == rows together (residual
    and K / V / index-key lane rows), fused MoE == the unfused block;
  * model level (tiny M3 with M3.1's formats): DSpark verify rows == sequential decode steps bit for bit with the
    lever on in every combination of the fused / fast-JANGT levers, widths past 16; greedy DSpark == greedy AR;
    control: a decode head that skips the flat kernels while the verify uses them is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

from vmlx_engine.models.minimax_m3 import flat_dense as FL  # noqa: E402

if not FL.available():  # pragma: no cover - needs Metal 4 tensor ops (M5-class GPU)
    pytest.skip("Metal 4 tensor ops (matmul2d) unavailable", allow_module_level=True)

from tests import test_minimax_m3_dspark as TD  # noqa: E402
from vmlx_engine.dspark import DSparkStats, generate_ar, generate_dspark, load_drafter  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402

LEVERS = ("VMLX_M3_FUSED_ATTN", "VMLX_M3_FUSED_MOE", "VMLX_JANGT_FAST", "VMLX_JANGT_GROUPED_FAST")


@pytest.fixture(autouse=True)
def _flat_on(monkeypatch):
    for name in ("VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
                 "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK", "VMLX_MINIMAX_MSA",
                 "VMLX_MINIMAX_MSA_SELECT") + LEVERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(FL.ENV, "1")


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _ql(i, o, bits, rng, mode="affine"):
    w = mx.array(rng.standard_normal((o, i)).astype(np.float32) * (1.5 / np.sqrt(i))).astype(mx.bfloat16)
    if mode == "mxfp8":
        m = nn.QuantizedLinear(i, o, bias=False, group_size=32, bits=8, mode="mxfp8")
        m.weight, m.scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    else:
        m = nn.QuantizedLinear(i, o, bias=False, group_size=64, bits=bits)
        m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(m.parameters())
    return m


def _deq(m) -> np.ndarray:
    """The weights in fp32 (scale * code + bias in fp32, as the kernels apply them; not rounded to bf16)."""
    if m.mode == "mxfp8":
        w = mx.dequantize(m.weight, m.scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.float32)
    else:
        w = mx.dequantize(m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32), group_size=64,
                          bits=m.bits, dtype=mx.float32)
    return np.array(w, np.float64)


def _rows(R, K, seed, extremes=True):
    x = np.random.default_rng(seed).standard_normal((R, K)).astype(np.float32)
    if extremes and R >= 5:
        x[1] *= 300.0
        x[3] *= 1e-3
        x[4] = 0.0
    return mx.array(x).astype(mx.bfloat16)


SEGS = {
    "a4": [(4, "affine", 96)],
    "a6": [(6, "affine", 160)],
    "a8": [(8, "affine", 64)],
    "mxfp8": [(8, "mxfp8", 128)],
    "qkv": [(4, "affine", 256), (4, "affine", 32), (4, "affine", 32), (8, "mxfp8", 64), (8, "mxfp8", 32)],
}


@pytest.mark.parametrize("name", list(SEGS))
@pytest.mark.parametrize("K", [576, 2048])
def test_formats_against_float64_reference(name, K):
    rng = np.random.default_rng(len(name) + K)
    mods = [_ql(K, n, bits, rng, mode) for bits, mode, n in SEGS[name]]
    x = _rows(7, K, 3)
    half16 = any(FL.fmt(m) == 1 for m in mods)
    y = np.array(FL.mm(FL.prep(x, half16=half16), mods, epi=0), np.float64)
    xb = np.array(x.astype(mx.float32), np.float64)
    ref = np.concatenate([xb @ _deq(m).T for m in mods], axis=1)
    scale = np.abs(ref).max(axis=1, keepdims=True) + 1e-30
    rel = np.abs(y - ref) / scale
    assert np.abs(ref).max() > 0 and rel.max() < 2e-5, rel.max()


@pytest.mark.parametrize("K, N", [(576, 96), (2048, 160), (3072, 288)])
def test_rows_alone_equal_rows_together(K, N):
    rng = np.random.default_rng(K)
    mods = [_ql(K, N, 4, rng), _ql(K, 64, 8, rng, "mxfp8")]
    x = _rows(16, K, K)
    full = FL.mm(FL.prep(x, half16=True), mods, epi=0)
    for r in range(16):
        assert _eq(FL.mm(FL.prep(x[r:r + 1], half16=True), mods, epi=0), full[r:r + 1]), r
    for a, b in ((0, 3), (2, 9), (5, 16), (1, 14), (3, 4)):
        assert _eq(FL.mm(FL.prep(x[a:b], half16=True), mods, epi=0), full[a:b]), (a, b)
    # another row's input never changes a row
    x2 = mx.concatenate([x[:6], (x[6:7] * 50.0).astype(mx.bfloat16), x[7:]], axis=0)
    other = FL.mm(FL.prep(x2, half16=True), mods, epi=0)
    keep = [r for r in range(16) if r != 6]
    assert _eq(other[mx.array(keep)], full[mx.array(keep)])


def test_control_mlx_multi_row_quantized_matmul_is_not_row_invariant():
    rng = np.random.default_rng(0)
    m = _ql(4096, 512, 4, rng)
    x = _rows(8, 4096, 1, extremes=False)
    full = m(x)
    assert any(not _eq(m(x[r:r + 1]), full[r:r + 1]) for r in range(8))


def test_control_row_count_dependent_kernel_is_caught(monkeypatch):
    rng = np.random.default_rng(1)
    m = _ql(1024, 128, 4, rng)
    x = _rows(6, 1024, 2)
    real = FL._src

    def bad(name):
        s = real(name)
        return s.replace("C[i] = fma(float(S[sb]), P[i],", "C[i] = fma(float(S[sb]) * (1.0f + 1e-6f * float(R)), P[i],",
                         1) if name == "flat_common" else s

    monkeypatch.setattr(FL, "_src", bad)
    FL._KERNELS.clear()
    try:
        full = FL.mm(FL.prep(x), [m], epi=0)
        assert not all(_eq(FL.mm(FL.prep(x[r:r + 1]), [m], epi=0), full[r:r + 1]) for r in range(6))
    finally:
        FL._KERNELS.clear()


def test_prep_norm_matches_gemma_rms_norm():
    rng = np.random.default_rng(3)
    x = _rows(5, 2048, 4)
    w = mx.array(rng.standard_normal(2048).astype(np.float32) * 0.1).astype(mx.bfloat16)
    A, A16, XS = FL.prep(x, w, 1e-6, half16=True)
    xf = np.array(x.astype(mx.float32), np.float64)
    ref = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + np.array(w.astype(mx.float32), np.float64))
    got = np.array(A.astype(mx.float32), np.float64)
    nz = np.abs(ref).max(-1) > 0
    assert (np.abs(got - ref).max(-1)[nz] / np.abs(ref).max(-1)[nz]).max() < 8e-3      # one bf16 rounding
    assert _eq(A16.astype(mx.float32), A.astype(mx.float32))
    assert np.allclose(np.array(XS)[:5 * 32].reshape(5, 32), got.reshape(5, 32, 64).sum(-1), rtol=1e-5, atol=1e-4)


def test_epilogues():
    rng = np.random.default_rng(5)
    o = _ql(1024, 512, 4, rng)
    xo = mx.array(rng.standard_normal((6, 1024)).astype(np.float32))
    res = _rows(6, 512, 6)
    h = FL.mm(FL.prep(xo), [o], epi=1, out_dtype=mx.bfloat16, resid=res)
    ref = np.array(xo.astype(mx.bfloat16).astype(mx.float32), np.float64) @ _deq(o).T + \
        np.array(res.astype(mx.float32), np.float64)
    ulp = np.abs(ref) * 2.0 ** -8 + 1e-6
    assert (np.abs(np.array(h.astype(mx.float32), np.float64) - ref) / ulp).max() <= 1.0     # one bf16 rounding

    class _MLP:
        pass

    sh = _MLP()
    sh.gate_proj, sh.up_proj, sh.down_proj, sh.alpha, sh.limit = _ql(1024, 512, 4, rng), _ql(1024, 512, 4, rng), \
        _ql(512, 1024, 4, rng), 1.702, 7.0
    assert FL.mlp_supported(sh)
    xs = _rows(5, 1024, 7, extremes=False)
    y = np.array(FL.mlp(sh, xs).astype(mx.float32), np.float64)
    xb = np.array(xs.astype(mx.float32), np.float64)
    g = np.array(mx.array((xb @ _deq(sh.gate_proj).T).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    u = np.array(mx.array((xb @ _deq(sh.up_proj).T).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    g = np.minimum(g, 7.0)
    u = np.clip(u, -7.0, 7.0)
    hs = g / (1 + np.exp(-1.702 * g)) * (u + 1)
    hs = np.array(mx.array(hs.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32), np.float64)
    ref = hs @ _deq(sh.down_proj).T
    assert np.abs(y - ref).max() / np.abs(ref).max() < 1e-2
    assert all(_eq(FL.mlp(sh, xs[r:r + 1]), mx.array(y[r:r + 1].astype(np.float32)).astype(mx.bfloat16))
               for r in range(5))
    lin = FL.linear(sh.gate_proj, xs)
    assert lin.dtype == mx.bfloat16 and lin.shape == (5, 512)


# ------------------------------------------------------------------------------------------- blocks
def test_attention_block_rows_alone_equal_rows_together(monkeypatch):
    from tests import test_minimax_m3_fused_attn as TA

    monkeypatch.setenv(F.ENV, "1")
    layer = TA.Layer(hidden=1024, seed=3)
    assert FL.attn_supported(layer)
    for ctx in (300, 2040, 3001):
        x = TA.rows_input(16, 1024, ctx)
        calls = FL.STATS["mm"]
        together, lanes = TA._run(layer, x, ctx, ctx, per_row=False)
        assert FL.STATS["mm"] > calls
        alone, lanes1 = TA._run(layer, x, ctx, ctx, per_row=True)
        assert _eq(together, alone), ctx
        assert all(_eq(a, b) for a, b in zip(lanes, lanes1)), ctx


@pytest.mark.parametrize("T", [1, 3, 9, 16])
def test_fused_moe_with_flat_equals_unfused(monkeypatch, T):
    import tests.test_minimax_m3_fused_moe as TF
    from vmlx_engine.models.minimax_m3 import fused_moe as FM

    monkeypatch.setenv("VMLX_JANGT_FAST", "1")
    b = TF._block(Kgu=1.5, Kdn=2.5, nog=1, nod=3, seed=7)
    assert FL.mlp_supported(b.shared_experts)
    x, res = TF._rows(T, 1024, seed=T)
    ref = FM.unfused_moe_residual(b, x, res)
    got = FM.fused_moe_residual(b, x, res)
    assert _eq(got, ref)
    assert _eq(FM.fused_moe_residual(b, x[:, :1], res[:, :1]), got[:, :1])


# ------------------------------------------------------------------------------------------- model level
def _m31_model(seed=0):
    """tests/test_minimax_m3_fused_moe.py's 3-layer model (JANGT routed experts) with M3.1's attention / head
    formats: index q / k MXFP8, lm_head affine 6-bit, the rest affine 4-bit."""
    import tests.test_minimax_m3_fused_moe as TF
    from mlx.utils import tree_flatten

    model = TF._model(seed)
    # re-quantize the index projections to MXFP8 and the head to 6-bit from fresh float weights
    rng = np.random.default_rng(seed + 100)
    for layer in model.model.layers:
        ix = layer.self_attn.indexer
        for nm_ in ("index_q_proj", "index_k_proj"):
            old = getattr(ix, nm_)
            o, i = int(old.weight.shape[0]), int(old.weight.shape[1]) * 32 // int(old.bits)
            setattr(ix, nm_, _ql(i, o, 8, rng, "mxfp8"))
    old = model.lm_head
    o, i = int(old.weight.shape[0]), int(old.weight.shape[1]) * 32 // int(old.bits)
    model.lm_head = _ql(i, o, 6, rng)
    mx.eval([v for _, v in tree_flatten(model.parameters())])
    return model


@pytest.fixture(scope="module")
def m31_target():
    return MiniMaxM3DSparkTarget(_m31_model())


@pytest.mark.parametrize("attn, moe, fast", [("1", "1", "1"), ("1", "0", "1"), ("0", "1", "1"), ("1", "1", "0"),
                                             ("0", "0", "0"), ("1", "0", "0")])
def test_verify_rows_equal_decode_steps_every_lever_combo(m31_target, monkeypatch, attn, moe, fast):
    for name, v in zip(LEVERS[:3], (attn, moe, fast)):
        monkeypatch.setenv(name, v)
    tgt = m31_target
    prompt = mx.array([np.random.default_rng(1).integers(0, 511, 77).tolist()])
    for width in (1, 2, 5, 8, 9, 16, 17, 21):
        rows = mx.array([np.random.default_rng(width).integers(0, 511, width).tolist()])
        tgt.verify_paths = {k: 0 for k in tgt.verify_paths}
        calls = FL.STATS["mm"]
        cache = tgt.make_cache()
        tgt.prefill(prompt, cache, (), 32)
        logits, _, _ = tgt.verify(rows, cache, ())
        mx.eval(logits)
        assert FL.STATS["mm"] > calls and tgt.verify_paths.get("head_flat", 0) == 1
        ref = tgt.make_cache()
        tgt.prefill(prompt, ref, (), 32)
        steps = [tgt.decode(rows[0, r:r + 1], ref) for r in range(width)]
        bad = [r for r in range(width) if not _eq(logits[r], steps[r])]
        assert not bad, (attn, moe, fast, width, bad)
        assert not _cache_diffs(cache, ref)


def _cache_diffs(cache, ref) -> list:
    """Differences between two M3 cache lists (any layer count): offsets and the logical K / V / index-key rows."""
    out = []
    assert len(cache) == len(ref) > 0
    for li, (a, b) in enumerate(zip(cache, ref)):
        if int(a.offset) != int(b.offset):
            out.append((li, "offset"))
            continue
        for name, x, y in zip(("keys", "values", "idx_keys"), a.state, b.state):
            assert x is not None and y is not None and x.size > 0 and x.shape[2] == int(a.offset), (li, name)
            if not _eq(x, y):
                out.append((li, name))
    return out


def test_control_decode_head_off_the_flat_kernels_is_caught(m31_target, monkeypatch):
    """Control: decode steps whose head skips the flat kernels (stock qmv) while the verify takes them differ."""
    monkeypatch.setenv("VMLX_M3_FUSED_ATTN", "1")
    tgt = m31_target
    prompt = mx.array([np.random.default_rng(2).integers(0, 511, 70).tolist()])
    rows = mx.array([np.random.default_rng(3).integers(0, 511, 6).tolist()])
    cache = tgt.make_cache()
    tgt.prefill(prompt, cache, (), 32)
    logits, _, _ = tgt.verify(rows, cache, ())
    real = M.lm_head_logits
    monkeypatch.setattr(M, "lm_head_logits", lambda model, out, verify=False: model.lm_head(out))
    ref = tgt.make_cache()
    tgt.prefill(prompt, ref, (), 32)
    steps = [tgt.decode(rows[0, r:r + 1], ref) for r in range(6)]
    monkeypatch.setattr(M, "lm_head_logits", real)
    assert not all(_eq(logits[r], steps[r]) for r in range(6))


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return TD._save_release_drafter(tmp_path_factory.mktemp("m31_flat") / "dspark")


@pytest.fixture(scope="module")
def td_target():
    """tests/test_minimax_m3_fused_attn.py's tiny M3 (the DSpark test drafter's geometry; affine-4 routed experts, so
    its MoE runs the unfused block with the shared expert on the flat kernels; MXFP8 index, 6-bit head)."""
    from tests import test_minimax_m3_fused_attn as TA

    return MiniMaxM3DSparkTarget(TA._model())


@pytest.mark.parametrize("mode", ["adaptive", 7, 17])
def test_dspark_greedy_equals_ar_with_flat(td_target, drafter_dir, monkeypatch, mode):
    monkeypatch.setenv("VMLX_M3_FUSED_ATTN", "1")
    tgt = td_target
    prompt = TD._ids(773, 11)
    calls = FL.STATS["mm"]
    reference = generate_ar(tgt, prompt, 40, set(), prefill_step=256)
    assert FL.STATS["mm"] > calls
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    stats = DSparkStats()
    width = mode if isinstance(mode, int) else None
    out = generate_dspark(tgt, drafter, prompt, 40, set(), prefill_step=256, fixed_width=width, stats=stats,
                          max_draft=max(2 * TD.DRAFT_BLOCK, (width or 0)), ar_fallback=TD._WidthOnly())
    assert out == reference
    assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted

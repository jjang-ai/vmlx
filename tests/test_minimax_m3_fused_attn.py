# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 fused attention half (VMLX_M3_FUSED_ATTN, models/minimax_m3/fused_attn.py + kernels/attn_*.metal).

* Per-kernel error: each of the five launches against a float64 numpy reference of the same math on the same inputs
  (and today's unfused ops against the same reference, for scale). The fused path is NOT bit-identical to today's ops
  (fp32 throughout, fp32 ``1 + w``, one rounding per output); end-to-end quality is judged on the real bundle.
* Row invariance: a row's outputs (residual rows, K / V / index-key lane rows) are bit-identical whether it runs
  alone or with 1 .. 15 other rows, at contexts below, straddling and above the dense threshold (top-k * block keys),
  lengths not multiples of 128 / 256; lazily chained steps across a lane growth; another row's input never changes a
  row. Control: today's multi-row attention forward is not row-invariant, so the comparison bites.
* Lane ownership: in-place appends never write under a view someone else holds (a zero-copy store view after a trim);
  control: forcing in-place writes there corrupts the view.
* Model level (tiny synthetic M3 with M3.1's attention formats): fused vs unfused logits within the model's own noise
  floor (two unfused MSA paths), greedy tokens; DSpark verify rows == sequential decode steps bit for bit with the fused
  path on (including widths past 16 rows), and greedy DSpark == greedy AR.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():  # pragma: no cover - the kernels are Metal only
    pytest.skip("Metal GPU required", allow_module_level=True)

from mlx.utils import tree_flatten  # noqa: E402

from tests import test_minimax_m3_dspark as TD  # noqa: E402
from vmlx_engine.dspark import DSparkStats, generate_ar, generate_dspark, load_drafter  # noqa: E402
from vmlx_engine.dspark.generate import ARFallback  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402
from vmlx_engine.models.minimax_m3 import msa  # noqa: E402
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache  # noqa: E402

MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
A4 = {"group_size": 64, "bits": 4}


@pytest.fixture(autouse=True)
def _fused_on(monkeypatch):
    for name in ("VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
                 "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK", "VMLX_MINIMAX_MSA",
                 "VMLX_MINIMAX_MSA_SELECT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(F.ENV, "1")


# ------------------------------------------------------------------------------------------- one M3.1-shaped layer
class Layer:
    """An attention half with M3.1's head geometry (64 q heads x 128, 4 KV heads, partial RoPE 64, theta 1e7, index
    4 x 128 / 1 x 128, blocks of 128, top-16) and formats (q / k / v / o affine 4-bit g64, index MXFP8); ``hidden``
    is a parameter (6144 on the real model)."""

    def __init__(self, hidden: int = 1024, seed: int = 0, heads: int = 64, kv: int = 4, hd: int = 128,
                 rot: int = 64, block: int = 128, topk: int = 16):
        tc = {"hidden_size": hidden, "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
              "rotary_dim": rot, "rope_theta": 1e7, "rms_norm_eps": 1e-6, "num_hidden_layers": 1,
              "sparse_attention_config": {"sparse_num_index_heads": kv, "sparse_index_dim": hd,
                                          "sparse_block_size": block, "sparse_topk_blocks": topk,
                                          "sparse_local_block": 1, "sparse_attention_freq": [1]}}
        args = M.ModelArgs.from_dict({"text_config": tc})
        rng = np.random.default_rng(seed)
        self.self_attn = attn = M.Attention(args, 0)
        self.input_layernorm = M.GemmaRMSNorm(hidden, 1e-6)
        ix = attn.indexer
        for owner, name, o, i, q in ((attn, "q_proj", heads * hd, hidden, A4), (attn, "k_proj", kv * hd, hidden, A4),
                                     (attn, "v_proj", kv * hd, hidden, A4), (attn, "o_proj", hidden, heads * hd, A4),
                                     (ix, "index_q_proj", kv * hd, hidden, MX8), (ix, "index_k_proj", hd, hidden, MX8)):
            w = mx.array(rng.standard_normal((o, i), dtype=np.float32) * (1.5 / np.sqrt(i))).astype(mx.bfloat16)
            m = nn.QuantizedLinear(i, o, bias=False, **q)
            if q is MX8:
                m.weight, m.scales = mx.quantize(w, mode="mxfp8", group_size=32, bits=8)
            else:
                m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
            setattr(owner, name, m)
        for nm in (attn.q_norm, attn.k_norm, ix.index_q_norm, ix.index_k_norm, self.input_layernorm):
            nm.weight = mx.array(rng.standard_normal(nm.weight.shape, dtype=np.float32) * 0.1).astype(mx.bfloat16)
        mx.eval(attn.parameters(), self.input_layernorm.parameters())
        self.hidden = hidden


def lane_cache(ctx: int, seed: int, *, hd: int = 128, kv: int = 4, spare: int = 0) -> MiniMaxM3SparseCache:
    """A sparse cache at ``ctx`` rows of random K / V / index keys; physical capacity ctx + spare (spare 0: the
    first append grows the lanes)."""
    rng = np.random.default_rng(seed)
    cap = ctx + spare
    lanes = [mx.array(rng.standard_normal((1, h, cap, hd), dtype=np.float32)).astype(mx.bfloat16)
             for h in (kv, kv, 1)]
    c = MiniMaxM3SparseCache()
    c.state = tuple(lanes)
    c.trim(spare)
    mx.eval(c.fused_lanes())
    return c


def rows_input(n: int, hidden: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal((1, n, hidden), dtype=np.float32) * 2.0).astype(
        mx.bfloat16)


@pytest.fixture(scope="module")
def layer():
    return Layer()


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


# ------------------------------------------------------------------------------------------- float64 references
def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _dequant(m) -> np.ndarray:
    if str(getattr(m, "mode", "affine")) == "mxfp8":
        w = mx.dequantize(m.weight, m.scales, mode="mxfp8", group_size=32, bits=8)
    else:
        w = mx.dequantize(m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32),
                          group_size=m.group_size, bits=m.bits)
    return _np(w)


def _gemma_norm(x: np.ndarray, w, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + _np(w))


def _rope(x: np.ndarray, positions: np.ndarray, rot: int, base: float) -> np.ndarray:
    """MLX non-traditional RoPE on the first ``rot`` dims; x [R, heads, D], positions [R]."""
    half = rot // 2
    inv = base ** (-np.arange(half) / half)
    th = positions[:, None, None] * inv[None, None, :]
    c, s = np.cos(th), np.sin(th)
    out = x.copy()
    x1, x2 = x[..., :half], x[..., half:rot]
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rot] = x1 * s + x2 * c
    return out


def _rel(a: np.ndarray, ref: np.ndarray, mask=None) -> float:
    if mask is not None:
        a, ref = a[mask], ref[mask]
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-30))


def _ulps_bf16(a: np.ndarray, ref: np.ndarray) -> float:
    """Max |a - ref| in units of the bf16 spacing at ref (a correctly rounded bf16 result is <= 0.5)."""
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref), 1e-30))) - 7)
    return float((np.abs(a - ref) / ulp).max())


def _top(scores: np.ndarray, n_valid: int, keep: int) -> list:
    """The selection rule of K4: the keep highest of scores[:n_valid], ties to the lower block id, ascending ids."""
    order = sorted(range(n_valid), key=lambda j: (-scores[j], j))
    return sorted(order[:keep])


def kernel_errors(layer: Layer, ctx: int, rows: int, seed: int = 0) -> dict:
    """Max error (relative to the reference tensor's max |value|) of each fused launch and of today's ops for the same
    step, both against a float64 numpy reference of the math on the fused path's own inputs. Also the selection
    agreement of K3 (fused vs reference top-k sets) and of today's selection."""
    p = F.plan(layer)
    attn, ix = layer.self_attn, layer.self_attn.indexer
    H, KVH, D, G = p.H, p.KVH, p.D, p.GI
    x = rows_input(rows, layer.hidden, seed + 1)
    x2 = x.reshape(rows, layer.hidden)
    pos = np.arange(ctx, ctx + rows, dtype=np.float64)
    out = {}

    # K1 ------------------------------------------------------------------------------------------------------------
    y = F.qkv(layer, p, x2)
    mods = (attn.q_proj, attn.k_proj, attn.v_proj, ix.index_q_proj, ix.index_k_proj)
    n_today = layer.input_layernorm(x)
    y_today = mx.concatenate([m(n_today) for m in mods], axis=-1).reshape(rows, -1)
    ref1 = _gemma_norm(_np(x2), layer.input_layernorm.weight, p.eps_in) @ np.concatenate(
        [_dequant(m) for m in mods], axis=0).T
    out["K1"] = (_rel(_np(y), ref1), _rel(_np(y_today), ref1))

    # K2 (out-of-place build: the new K / V / index-key rows come back as arrays) -----------------------------------
    cache = lane_cache(ctx, seed, hd=D, kv=KVH, spare=64)
    dts = F.lane_dtypes(layer, cache, x.dtype)
    nb = -(-(ctx + rows) // p.block)
    meta = mx.array([ctx, rows, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
    q, iq, cnt, new = F.heads(layer, p, y, meta, cache.fused_lanes(), dts, False)
    yn = _np(y)
    seg = np.cumsum([0, H * D, KVH * D, KVH * D, G * D, D])
    parts = [yn[:, seg[i]:seg[i + 1]].reshape(rows, -1, D) for i in range(5)]
    ref_q = _rope(_gemma_norm(parts[0], attn.q_norm.weight, p.eps_head), pos, p.ROT, attn.rope.base)
    ref_k = _rope(_gemma_norm(parts[1], attn.k_norm.weight, p.eps_head), pos, p.ROT, attn.rope.base)
    ref_v = parts[2]
    ref_iq = _rope(_gemma_norm(parts[3], ix.index_q_norm.weight, p.eps_head), pos, p.ROT, attn.rope.base)
    ref_ik = _rope(_gemma_norm(parts[4], ix.index_k_norm.weight, p.eps_head), pos, p.ROT, attn.rope.base)
    yb = y.astype(mx.bfloat16)
    sl = [yb[:, int(seg[i]):int(seg[i + 1])].reshape(1, rows, -1, D) for i in range(5)]
    t_q = attn.rope(attn.q_norm(sl[0]).transpose(0, 2, 1, 3), offset=ctx).transpose(0, 2, 1, 3)[0]
    t_k = attn.rope(attn.k_norm(sl[1]).transpose(0, 2, 1, 3), offset=ctx).transpose(0, 2, 1, 3)[0]
    t_iq = ix.rope(ix.index_q_norm(sl[3]).transpose(0, 2, 1, 3), offset=ctx).transpose(0, 2, 1, 3)[0]
    t_ik = ix.rope(ix.index_k_norm(sl[4]).transpose(0, 2, 1, 3), offset=ctx).transpose(0, 2, 1, 3)[0]
    out["K2 q"] = (_rel(_np(q), ref_q), _rel(_np(t_q), ref_q))
    out["K2 k (lane dtype)"] = (_rel(_np(new[0]), ref_k), _rel(_np(t_k), ref_k))
    out["K2 v (lane dtype)"] = (_rel(_np(new[1]), ref_v), _rel(_np(sl[2][0]), ref_v))
    out["K2 index q"] = (_rel(_np(iq), ref_iq), _rel(_np(t_iq), ref_iq))
    out["K2 index k (lane dtype)"] = (_rel(_np(new[2]), ref_ik), _rel(_np(t_ik), ref_ik))

    # lanes with the new rows (as the fused forward leaves them)
    cache.fused_append(new[0].transpose(1, 0, 2)[None], new[1].transpose(1, 0, 2)[None],
                       new[2].transpose(1, 0, 2)[None])
    kl, vl, il = cache.fused_lanes()
    kn, vn, idn = _np(kl[0, :, :ctx + rows]), _np(vl[0, :, :ctx + rows]), _np(il[0, 0, :ctx + rows])
    qn, iqn = _np(q), _np(iq)

    # K3 ------------------------------------------------------------------------------------------------------------
    sc = F.scores(p, iq, il, meta, nb) if nb > p.keep else None
    sel_f, sel_r, sel_t = {}, {}, {}
    if sc is not None:
        scn = np.array(sc, dtype=np.float64)
        ref3 = np.full((rows, G, nb), -np.inf)
        for r in range(rows):
            qp = ctx + r
            dots = iqn[r] @ idn[:qp + 1].T                                       # [G, qp + 1]
            for b in range(-(-(qp + 1) // p.block)):
                ref3[r, :, b] = dots[:, b * p.block:(b + 1) * p.block].max(axis=1)
            ref3[r, :, qp // p.block] = np.inf
        finite = np.isfinite(ref3)
        assert np.array_equal(np.isfinite(scn), finite)
        t_scores = []
        for r in range(rows):
            qp = ctx + r
            t_scores.append(msa.reference_block_scores(t_iq[r:r + 1].transpose(1, 0, 2)[None],
                                                       il[..., :qp + 1, :], qp, p.block, p.local))
        out["K3 block scores"] = (_rel(scn, ref3, finite),
                                  max(_rel(np.array(t[0, :, 0], dtype=np.float64), ref3[r, :, :t.shape[-1]],
                                           finite[r, :, :t.shape[-1]]) for r, t in enumerate(t_scores)))
        for r in range(rows):
            n_r = (ctx + r) // p.block + 1
            for g in range(G):
                if n_r > p.keep:
                    sel_f[r, g] = _top(scn[r, g], n_r, p.keep)
                    sel_r[r, g] = _top(ref3[r, g], n_r, p.keep)
                    sel_t[r, g] = _top(np.array(t_scores[r][0, g, 0], dtype=np.float64), n_r, p.keep)
        out["K3 selection agreement (pairs)"] = (
            sum(sel_f[k] == sel_r[k] for k in sel_f) / max(1, len(sel_f)),
            sum(sel_t[k] == sel_r[k] for k in sel_t) / max(1, len(sel_t)))

    # K4 (attention over the fused selection; reference in float64) -------------------------------------------------
    o = F.msa(p, q, kl, vl, sc if sc is not None else F._dummy(), cnt, meta)
    ref4 = np.zeros((rows, H, D))
    hg = H // KVH
    for r in range(rows):
        qp = ctx + r
        n_r = qp // p.block + 1
        for g in range(KVH):
            blocks = sel_f[r, g] if (r, g) in sel_f else list(range(n_r))
            keys = np.concatenate([np.arange(b * p.block, min((b + 1) * p.block, qp + 1)) for b in blocks])
            s = qn[r, g * hg:(g + 1) * hg] @ kn[g, keys].T * p.scale
            w = np.exp(s - s.max(axis=1, keepdims=True))
            ref4[r, g * hg:(g + 1) * hg] = (w / w.sum(axis=1, keepdims=True)) @ vn[g, keys]
    on = _np(o)
    t_o = []
    for r in range(rows):
        end = ctx + r + 1
        sel = ix.select(t_iq[r:r + 1].transpose(1, 0, 2)[None], il[..., :end, :], ctx + r)
        t_o.append(attn.attend(t_q[r:r + 1].transpose(1, 0, 2)[None], kl[..., :end, :], vl[..., :end, :], sel,
                               None, None)[0].transpose(1, 0, 2))
    out["K4 attention (fused selection)"] = (_rel(on, ref4), _rel(_np(mx.concatenate(t_o, axis=0)), ref4))

    # K5 ------------------------------------------------------------------------------------------------------------
    h = F.oproj(layer, p, o.reshape(rows, H * D), x2)
    ref5 = _np(x2) + on.reshape(rows, -1) @ _dequant(attn.o_proj).T
    t_h = x2 + attn.o_proj(o.reshape(rows, H * D).astype(mx.bfloat16))
    out["K5 o_proj + residual"] = (_rel(_np(h), ref5), _rel(_np(t_h), ref5))
    out["K5 o_proj + residual (bf16 ulps)"] = (_ulps_bf16(_np(h), ref5), _ulps_bf16(_np(t_h), ref5))
    return out


@pytest.mark.parametrize("ctx, rows", [(700, 3), (3001, 5), (2040, 16)])
def test_kernel_errors_against_float64_reference(layer, ctx, rows):
    """Each launch reproduces its math in fp32: errors against a float64 reference on the same inputs are at fp32
    accumulation level (K2's q / index q also carry MLX's fp32 RoPE angle, ~1e-5 x position), the bf16 lane and
    residual outputs one correct rounding (K5 <= 0.5 bf16 ulp); today's ops on the same step are coarser (bf16
    intermediates) and pick a different block set for some (row, group) pairs."""
    errs = kernel_errors(layer, ctx, rows)
    bounds = {"K1": 2e-6, "K2 q": 1e-3, "K2 index q": 1e-3, "K2 k (lane dtype)": 4.5e-3, "K2 v (lane dtype)": 4.5e-3,
              "K2 index k (lane dtype)": 4.5e-3, "K3 block scores": 1e-5, "K4 attention (fused selection)": 1e-5,
              "K5 o_proj + residual": 4.5e-3, "K5 o_proj + residual (bf16 ulps)": 0.51}
    for name, bound in bounds.items():
        if name not in errs:                                    # no sparse row: K3 does not run
            assert name.startswith("K3") and -(-(ctx + rows) // 128) <= 16
            continue
        fused, today = errs[name]
        assert fused <= bound, (name, fused, today)
    if "K3 selection agreement (pairs)" in errs:
        assert errs["K3 selection agreement (pairs)"][0] >= 0.95
    assert errs["K1"][0] < errs["K1"][1]                       # fp32 norm + one rounding beats today's bf16 chain


# ------------------------------------------------------------------------------------------- row invariance
def _run(layer, x, ctx, seed, *, per_row: bool, spare: int = 0, eager: bool = False):
    c = lane_cache(ctx, seed, spare=spare)
    rows = int(x.shape[1])
    if per_row:
        outs = []
        for r in range(rows):
            outs.append(F.attention_block(layer, x[:, r:r + 1], c))
            if eager:
                mx.eval(outs[-1])
        h = mx.concatenate(outs, axis=1)
    else:
        h = F.attention_block(layer, x, c)
    lanes = [lane[..., :ctx + rows, :] for lane in c.fused_lanes()]
    mx.eval(h, lanes)
    assert int(c.offset) == ctx + rows and int(c._idx_offset) == ctx + rows
    return h, lanes


def _same(a, b) -> bool:
    return _eq(a[0], b[0]) and all(_eq(x, y) for x, y in zip(a[1], b[1]))


@pytest.mark.parametrize("ctx", [300, 2040, 3001])
def test_rows_are_bit_identical_alone_or_together(layer, ctx):
    """1 .. 16 rows in one fused block == the same rows as one-row blocks (residual rows and every lane row), below
    (300), straddling (2040: rows cross 2048 keys, the first sparse rows) and above (3001) the dense threshold; lanes
    start at an exact-length capacity, so the block also grows them (2040 + 16 crosses the 256-row step too)."""
    x = rows_input(16, layer.hidden, ctx)
    for rows in range(1, 17):
        together = _run(layer, x[:, :rows], ctx, ctx, per_row=False)
        alone = _run(layer, x[:, :rows], ctx, ctx, per_row=True, eager=True)
        assert float(mx.abs(together[0].astype(mx.float32)).max()) > 0
        assert _same(together, alone), (ctx, rows)


def test_lazily_chained_steps_across_a_lane_growth(layer):
    """Sixteen one-row steps built as one lazy graph (no eval between them) from 2040 keys: the lanes grow at 2048
    while earlier steps wrote their rows in place. The growth copy must see those rows (the lanes are re-issued behind
    each block's MSA launch); before that ordering, rows 2041 .. 2047 came back as zeros."""
    x = rows_input(16, layer.hidden, 7)
    lazy = _run(layer, x, 2040, 7, per_row=True, spare=8)
    eager = _run(layer, x, 2040, 7, per_row=True, spare=8, eager=True)
    together = _run(layer, x, 2040, 7, per_row=False, spare=8)
    assert _same(lazy, eager) and _same(lazy, together)


def test_another_rows_input_never_changes_a_row(layer):
    """Changing row 6's input changes row 6 (and, causally, the later rows, which attend to its keys) but leaves rows
    0 .. 5 and every lane row before 6 bit-identical."""
    ctx = 3001
    x = rows_input(12, layer.hidden, 3)
    y = mx.concatenate([x[:, :6], rows_input(1, layer.hidden, 99), x[:, 7:]], axis=1)
    a = _run(layer, x, ctx, 5, per_row=False)
    b = _run(layer, y, ctx, 5, per_row=False)
    assert _eq(a[0][:, :6], b[0][:, :6]) and not _eq(a[0][:, 6], b[0][:, 6])
    for la, lb in zip(a[1], b[1]):
        assert _eq(la[..., :ctx + 6, :], lb[..., :ctx + 6, :])


def test_control_todays_multi_row_attention_is_not_row_invariant(layer, monkeypatch):
    """Control: the same comparison on today's multi-row forward (one call for all rows: multi-row qmm, one causal
    attention) finds rows that differ from one-row steps, so the invariance check above can fail."""
    monkeypatch.delenv(F.ENV)
    ctx, rows = 3001, 8
    x = rows_input(rows, layer.hidden, 11)

    def run(per_row):
        c = lane_cache(ctx, 11)
        if per_row:
            outs = [x[:, r:r + 1] + layer.self_attn(layer.input_layernorm(x[:, r:r + 1]), None, c)
                    for r in range(rows)]
            return mx.concatenate(outs, axis=1)
        return x + layer.self_attn(layer.input_layernorm(x), "causal", c)

    together, alone = run(False), run(True)
    assert sum(not _eq(together[:, r], alone[:, r]) for r in range(rows)) >= 2


# ------------------------------------------------------------------------------------------- lane ownership
def test_in_place_appends_never_write_under_a_held_view(layer):
    """A view of the lanes taken from outside (``state``, as the zero-copy prefix store does), then a trim, then fused
    steps over the trimmed rows: the view keeps its contents (the first step after the read copies on write; later
    steps write in place into the cache's own buffers); the cache holds the new rows."""
    ctx = 2100
    c = lane_cache(ctx, 21, spare=200)
    x = rows_input(16, layer.hidden, 21)
    mx.eval(F.attention_block(layer, x[:, :4], c))                  # owned lanes, in-place appends from here
    views = c.state                                                  # an outside holder: [0, ctx + 4)
    snapshot = [v + 0 for v in views]
    mx.eval(snapshot)
    c.trim(4)
    F.STATS.update(inplace=0, copy_appends=0)
    h = [F.attention_block(layer, x[:, 4 + r:5 + r], c) for r in range(6)]
    mx.eval(h)
    assert F.STATS["copy_appends"] == 1 and F.STATS["inplace"] == 5
    assert all(_eq(v, s) for v, s in zip(views, snapshot))
    k, _, _ = c.fused_lanes()
    assert not _eq(k[..., ctx:ctx + 4, :], snapshot[0][..., ctx:ctx + 4, :])


def test_control_forcing_in_place_under_a_view_corrupts_it(layer, monkeypatch):
    """Control: with the ownership check bypassed, the same sequence overwrites rows under the held view."""
    ctx = 2100
    c = lane_cache(ctx, 21, spare=200)
    x = rows_input(16, layer.hidden, 21)
    mx.eval(F.attention_block(layer, x[:, :4], c))
    views = c.state
    snapshot = [v + 0 for v in views]
    mx.eval(snapshot)
    c.trim(4)
    real = MiniMaxM3SparseCache.fused_reserve
    monkeypatch.setattr(MiniMaxM3SparseCache, "fused_reserve", lambda self, *a: real(self, *a) or True)
    mx.eval([F.attention_block(layer, x[:, 4 + r:5 + r], c) for r in range(6)])
    assert not all(_eq(v, s) for v, s in zip(views, snapshot))


def test_steady_decode_appends_in_place(layer):
    c = lane_cache(4000, 3, spare=100)
    x = rows_input(1, layer.hidden, 3)
    F.STATS.update(inplace=0, copy_appends=0)
    for _ in range(5):
        x = F.attention_block(layer, x, c)
        mx.eval(x)
    assert F.STATS == {**F.STATS, "inplace": 4, "copy_appends": 1}
    c.trim(2)                                                        # a rollback keeps ownership
    c.has_index_lane()                                               # and so does the lockstep check
    mx.eval(F.attention_block(layer, x, c))
    assert F.STATS["inplace"] == 5


def test_routing(layer, monkeypatch):
    """The fused path takes B = 1, 1 .. 16 rows, no mask array, an M3 sparse cache; everything else stays on today's
    ops (and the lever off keeps every call there)."""
    c = lane_cache(100, 0)
    assert F.use_fused(layer, rows_input(16, layer.hidden, 0), "causal", c)
    assert not F.use_fused(layer, rows_input(17, layer.hidden, 0), None, c)
    assert not F.use_fused(layer, mx.zeros((2, 1, layer.hidden), dtype=mx.bfloat16), None, c)
    assert not F.use_fused(layer, rows_input(2, layer.hidden, 0), mx.zeros((2, 102)), c)
    from mlx_lm.models.cache import KVCache

    assert not F.use_fused(layer, rows_input(1, layer.hidden, 0), None, KVCache())
    monkeypatch.delenv(F.ENV)
    assert not F.use_fused(layer, rows_input(1, layer.hidden, 0), None, c)
    other = Layer(hidden=512, heads=8, kv=2, hd=64, rot=32, block=32, topk=6)
    other.self_attn.o_proj = nn.Linear(512, 512, bias=False)
    assert F.unsupported_reason(other) == "o_proj not affine 4 / 8-bit"


# ------------------------------------------------------------------------------------------- model level
def _plan(path, module):
    if path.endswith("mlp.gate"):
        return False
    if "switch_mlp" in path:
        return A4 if hasattr(module, "to_quantized") else False
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    if "index_" in path:
        return MX8
    if "self_attn" in path:
        return A4
    if path == "lm_head":
        return {"group_size": 64, "bits": 6}
    if path.endswith("embed_tokens"):
        return {"group_size": 64, "bits": 8}
    return A4


def _model(seed: int = 0, block: int = 32, topk: int = 6):
    """tests/test_minimax_m3_dspark.py's tiny M3 with M3.1's attention formats (q / k / v / o affine 4, index MXFP8):
    layers 1 .. 3 take the fused path (layer 0 is dense attention, today's ops)."""
    model = M.Model(M.ModelArgs.from_dict(TD._config(block, topk)))
    rng = np.random.default_rng(seed)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.1, mx.bfloat16
        elif k.endswith("e_score_correction_bias"):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif k.endswith(".mlp.gate.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.3, mx.float32
        elif "embed_tokens" in k:
            a, dt = rng.standard_normal(v.shape), mx.bfloat16
        else:
            a, dt = rng.standard_normal(v.shape) * 1.5 / np.sqrt(v.shape[-1]), mx.bfloat16
        weights.append((k, mx.array(a.astype(np.float32)).astype(dt)))
    model.load_weights(weights)
    nn.quantize(model, class_predicate=_plan)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def target():
    t = MiniMaxM3DSparkTarget(_model())
    assert [F.unsupported_reason(layer) is None for layer in t.nm.layers] == [False, True, True, True]
    return t


def _decode_logits(target, prompt, tokens):
    c = target.make_cache()
    first, _ = target.prefill(mx.array([prompt]), c, (), 256)
    seq = [first] + [target.decode(mx.array([t]), c) for t in tokens[:-1]]
    mx.eval(seq)
    return [s.astype(mx.float32) for s in seq]


def test_fused_model_logits_within_the_unfused_noise_floor(target, monkeypatch):
    """Teacher-forced along today's greedy tokens: the fused path's logits differ from today's by about what two of
    today's own MSA paths (decode kernel vs dense mask) differ by -- this tiny random MoE amplifies any rounding
    through its expert routing; the argmax agrees wherever today's top-2 margin is not a near-tie."""
    for prompt_len, seed in ((773, 11), (300, 3)):
        prompt = TD._ids(prompt_len, seed)
        monkeypatch.delenv(F.ENV)
        reference = generate_ar(target, prompt, 30, set(), prefill_step=256)
        base = _decode_logits(target, prompt, reference)
        monkeypatch.setenv("VMLX_MINIMAX_MSA", "dense")
        floor = _decode_logits(target, prompt, reference)
        monkeypatch.delenv("VMLX_MINIMAX_MSA")
        monkeypatch.setenv(F.ENV, "1")
        F.STATS.update(calls=0)
        fused = _decode_logits(target, prompt, reference)
        assert F.STATS["calls"] >= 3 * 29                           # decode steps took the fused path
        d_floor = [float(mx.abs(a - b).max()) for a, b in zip(base, floor)]
        d_fused = [float(mx.abs(a - b).max()) for a, b in zip(base, fused)]
        assert np.median(d_fused) <= 3 * max(np.median(d_floor), 0.0625)
        for a, b in zip(base, fused):
            top2 = mx.sort(a)[-2:]
            if float(top2[1] - top2[0]) > 0.25:
                assert int(mx.argmax(a)) == int(mx.argmax(b))


def test_fused_greedy_decoding_diverges_only_at_near_ties(target, monkeypatch):
    """Free-running greedy tokens, fused vs today's ops: they agree up to a step whose top-2 margin (today's logits)
    is a near-tie (this random model has many)."""
    for prompt_len, seed in ((773, 11), (517, 12)):
        prompt = TD._ids(prompt_len, seed)
        monkeypatch.delenv(F.ENV)
        reference = generate_ar(target, prompt, 40, set(), prefill_step=256)
        base = _decode_logits(target, prompt, reference)
        monkeypatch.setenv(F.ENV, "1")
        fused = generate_ar(target, prompt, 40, set(), prefill_step=256)
        first = next((i for i, (a, b) in enumerate(zip(reference, fused)) if a != b), None)
        if first is not None:
            top2 = mx.sort(base[first])[-2:]
            assert float(top2[1] - top2[0]) <= 0.25, (prompt_len, first)


@pytest.mark.parametrize("prompt_len, width", [(773, 2), (765, 8), (1021, 9), (771, 16), (300, 17), (509, 21)])
def test_verify_rows_equal_sequential_decode_steps_with_fused_attention(target, prompt_len, width):
    """With the fused path on, row r of a T-row DSpark verify == the r-th of T one-token decode steps (which also run
    the fused kernels), bit for bit, including widths past 16 rows (the verify chunks the fused block); caches equal."""
    prompt = mx.array([TD._ids(prompt_len, prompt_len)])
    rows = mx.array([TD._ids(width, 7 * width)])
    taps = TD.DRAFT_TAPS
    target.verify_paths = {k: 0 for k in target.verify_paths}
    cache = target.make_cache()
    target.prefill(prompt, cache, taps, 256)
    logits, feats, _ = target.verify(rows, cache, taps)
    mx.eval(logits, feats)
    assert target.verify_paths.get("attention_fused", 0) == 3        # layers 1 .. 3
    ref = target.make_cache()
    target.prefill(prompt, ref, taps, 256)
    F.STATS.update(calls=0)
    ref_logits, ref_feats = [], []
    for r in range(width):
        lg, f = target.ar_step(rows[0, r:r + 1], ref, taps)
        ref_logits.append(lg)
        ref_feats.append(f)
    mx.eval(ref_logits)
    assert F.STATS["calls"] == 3 * width
    bad = [r for r in range(width) if not _eq(logits[r], ref_logits[r])]
    assert not bad, f"verify rows {bad} differ from decode steps"
    assert _eq(feats, mx.concatenate(ref_feats, axis=1))
    assert not TD._cache_problems(cache, ref)


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return TD._save_release_drafter(tmp_path_factory.mktemp("m31_fused") / "dspark")


@pytest.mark.parametrize("mode", ["adaptive", 7, 17])
def test_dspark_greedy_equals_ar_with_fused_attention(target, drafter_dir, mode):
    """Greedy DSpark == greedy AR with the fused path on in both (drafts wrong at every fifth position: full, partial
    and zero accepts; Adaptive widths and pinned widths, one past 16 rows)."""
    prompt = TD._ids(773, 11)
    reference = generate_ar(target, prompt, 40, set(), prefill_step=256)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    stats = DSparkStats()
    width = mode if isinstance(mode, int) else None
    out = generate_dspark(target, drafter, prompt, 40, set(), prefill_step=256, fixed_width=width, stats=stats,
                          max_draft=max(2 * TD.DRAFT_BLOCK, (width or 0)), ar_fallback=TD._WidthOnly())
    assert out == reference
    assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted


def test_released_geometry_past_2048_tokens_with_fused_attention(drafter_dir):
    """Blocks of 128 keys, top-16 (M3.1's geometry) on the tiny model, a 2,300-token prompt: verify == decode steps
    with the fused path and greedy DSpark == AR."""
    t = MiniMaxM3DSparkTarget(_model(seed=1, block=128, topk=16))
    prompt = TD._ids(2300, 5)
    reference = generate_ar(t, prompt, 24, set(), prefill_step=512)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    out = generate_dspark(t, drafter, prompt, 24, set(), prefill_step=512, max_draft=2 * TD.DRAFT_BLOCK,
                          ar_fallback=ARFallback())
    assert out == reference


def test_rope_angle_reference_matches_mlx():
    """The float64 RoPE used by the error references is MLX's non-traditional RoPE (sanity for kernel_errors)."""
    x = np.random.default_rng(0).standard_normal((3, 2, 128))
    rope = nn.RoPE(64, traditional=False, base=1e7)
    got = rope(mx.array(x.astype(np.float32)).transpose(1, 0, 2)[None], offset=5).transpose(0, 2, 1, 3)[0]
    assert np.abs(np.array(got) - _rope(x, np.arange(5, 8, dtype=np.float64), 64, 1e7)).max() < 1e-4

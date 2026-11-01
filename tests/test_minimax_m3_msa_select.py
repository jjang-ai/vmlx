# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 MSA selection: the block-score kernels vs the score-sheet ops they replace.

``msa.reference_block_scores`` is the runtime's selection math as it was (one float32 matmul of the cast index
queries / keys into a [B, G, Sq, Sk] sheet, causal -inf, -inf padding to whole blocks, max per 128-key block,
local block +inf). The kernels (``msa.kernel_block_scores``) must give the same block scores BIT FOR BIT, so the
indexer's top-k (the same argpartition) returns the same block ids in the same order. Every comparison goes through
``_identity``, which fails closed (empty / degenerate references are errors) and which the controls (block ids + 1,
local block not forced, mutated kernel reductions, the other kernel arithmetic) must fail.

Shapes: M3.1 index geometry (index heads = KV groups 4, plus 8; head dim 128; blocks of 128; top-16; local 1).
Lengths 2049 / 2853 / 3011 / 8268 / 32845 are not multiples of the block (128), the K / V step (256) or the chunk
(2048); rows 1 (decode), 2 / 5 / 16 (verify-sized), 512 / 2048 (chunks) and a ragged 1000-row chunk.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.models.minimax_m3 import msa  # noqa: E402
from vmlx_engine.models.minimax_m3.cache import (  # noqa: E402
    MiniMaxM3SparseCache, clone_minimax_m3_sparse, restore_minimax_m3_sparse)
from vmlx_engine.models.minimax_m3.minimax_m3 import Attention, Indexer, ModelArgs  # noqa: E402

H, D, IDX_D, BLOCK, TOPK = 64, 128, 128, 128, 16
METAL = msa._metal()


def _gpu_generation() -> int:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    arch = str(info.get("architecture", ""))
    digits = "".join(ch for ch in arch.split("applegpu_g")[-1] if ch.isdigit()) if arch.startswith("applegpu_g") else ""
    return int(digits) if digits else 0


GPU_GEN = _gpu_generation()
TF32 = os.environ.get("MLX_ENABLE_TF32", "1").strip() not in ("0", "")
needs_metal = pytest.mark.skipif(not METAL, reason="selection kernels run on the Metal GPU only")


@pytest.fixture(autouse=True)
def _auto_routing(monkeypatch):
    monkeypatch.delenv(msa.ENV_PATH, raising=False)
    monkeypatch.delenv(msa.SELECT_ENV, raising=False)


def _args(groups: int) -> ModelArgs:
    return ModelArgs(hidden_size=256, num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=groups,
                     head_dim=D, rotary_dim=64, index_n_heads=groups, index_head_dim=IDX_D,
                     index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1, sparse_attention_freq=[1])


def _index(sk: int, groups: int, rows: int, dtype=mx.bfloat16, seed: int = 0, batch: int = 1):
    """idx_q [B, G, rows, 128], idx_k [B, 1, sk, 128]. Index keys drift slowly (neighbouring blocks score alike, as
    real index keys do) so block maxima sit close together and the top-16 boundary is contested."""
    rng = np.random.default_rng(seed)
    ik = rng.standard_normal((batch, 1, sk, IDX_D)) + np.cumsum(rng.standard_normal((batch, 1, sk, IDX_D)) * 0.05,
                                                                 axis=2)
    iq = rng.standard_normal((batch, groups, rows, IDX_D))
    return mx.array(iq.astype(np.float32)).astype(dtype), mx.array(ik.astype(np.float32)).astype(dtype)


def _top(scores: mx.array) -> mx.array:
    """The indexer's top-k over block scores (Indexer.select's op)."""
    return mx.argpartition(-scores, kth=TOPK - 1, axis=-1)[..., :TOPK]


def _flip_rate(ref_top, got_top) -> float:
    """Fraction of (batch, group, row) whose selected block SET differs."""
    a = np.sort(np.array(ref_top).astype(np.int64), axis=-1)
    b = np.sort(np.array(got_top).astype(np.int64), axis=-1)
    return float((a != b).any(axis=-1).mean())


def _identity(ref_scores, got_scores, ref_top, got_top, offset: int) -> bool:
    """THE identity check: block scores equal bit for bit (NaN where the reference has NaN) and the top-k equal in
    value and order. Fails closed: the reference must be a real selection (rows present, every row's local block
    forced to +inf, finite scores present, and the set changes when the block ids shift by one)."""
    r, g = np.array(ref_scores), np.array(got_scores)
    rt, gt = np.array(ref_top).astype(np.int64), np.array(got_top).astype(np.int64)
    assert r.size and r.ndim == 4 and r.shape == g.shape and rt.size and rt.shape == gt.shape
    B, G, L, nb = r.shape
    assert nb > TOPK, "no selection at this length: every block is visible"
    local = (offset + np.arange(L)) // BLOCK
    assert np.isposinf(r[:, :, np.arange(L), local]).all(), "reference lost the forced local block"
    assert np.isfinite(r).any()
    shifted = np.sort((rt + 1) % nb, axis=-1)
    assert not np.array_equal(np.sort(rt, axis=-1), shifted), "degenerate reference selection"
    return bool(np.array_equal(r, g, equal_nan=True) and np.array_equal(rt, gt))


def _reference(iq, ik, offset, local: int = 1):
    s = msa.reference_block_scores(iq, ik, offset, BLOCK, local)
    t = _top(s)
    mx.eval(s, t)
    return s, t


def _routed(iq, ik, offset, local: int = 1):
    s = msa.block_scores(iq, ik, offset, BLOCK, local)
    t = _top(s)
    mx.eval(s, t)
    return s, t


def _admitted() -> str:
    arith = msa.select_arithmetic(mx.bfloat16, IDX_D)
    assert arith in ("nax", "fma"), "no selection kernel reproduces the reference on this Metal GPU"
    return arith


class _SelectSpy:
    """Records the selection router's picks."""

    def __init__(self, monkeypatch):
        self.picks = []
        real = msa.choose_select_path

        def spy(*a, **kw):
            p = real(*a, **kw)
            self.picks.append(p)
            return p

        monkeypatch.setattr(msa, "choose_select_path", spy)


# ----------------------------------------------------------------------------------------------- admission
@needs_metal
def test_admitted_arithmetic_is_the_one_mlx_runs():
    """MLX's float32 matmul runs on the matrix units with TF32 (its default) on M5-class GPUs, as an fma chain
    otherwise; the self-check admits exactly that kernel family."""
    arith = _admitted()
    if GPU_GEN >= 17:
        assert arith == ("nax" if TF32 else "fma")
    for dt in (mx.float16, mx.float32):
        assert msa.select_arithmetic(dt, IDX_D) == arith


# ----------------------------------------------------------------------------------------------- identity
@needs_metal
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("sk", [2049, 2853, 3011, 8268, 32845])
@pytest.mark.parametrize("rows", [1, 2, 5, 16])
def test_decode_and_verify_sized_selection_is_identical(rows, sk, groups, monkeypatch):
    spy = _SelectSpy(monkeypatch)
    arith = _admitted()
    iq, ik = _index(sk, groups, rows, seed=sk + rows)
    off = sk - rows
    ref = _reference(iq, ik, off)
    spy.picks.clear()
    got = _routed(iq, ik, off)
    assert spy.picks == [arith]                                       # the kernel really ran
    assert _identity(*ref[:1], *got[:1], ref[1], got[1], off)


@needs_metal
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("sk, rows", [(2853, 2048), (3011, 1000), (3011, 512), (8268, 512), (8268, 2048),
                                      (32845, 2048)])
def test_chunk_selection_is_identical(sk, rows, groups, monkeypatch):
    """Chunks, including rows before key 2048 (fewer than 16 causal blocks: the -inf future blocks tie) and a
    ragged 1000-row chunk."""
    spy = _SelectSpy(monkeypatch)
    arith = _admitted()
    iq, ik = _index(sk, groups, rows, seed=sk * 3 + rows)
    off = sk - rows
    ref = _reference(iq, ik, off)
    spy.picks.clear()
    got = _routed(iq, ik, off)
    assert spy.picks == [arith]
    assert _identity(ref[0], got[0], ref[1], got[1], off)


@needs_metal
def test_indexer_select_kernel_equals_reference_path(monkeypatch):
    """Through Indexer.select: VMLX_MINIMAX_MSA_SELECT=reference vs auto give the same MSASelection."""
    ix = Indexer(_args(4))
    for sk, rows in ((2853, 1), (8268, 7), (8268, 600)):
        iq, ik = _index(sk, 4, rows, seed=rows)
        monkeypatch.setenv(msa.SELECT_ENV, "reference")
        a = ix.select(iq, ik, sk - rows)
        monkeypatch.delenv(msa.SELECT_ENV)
        b = ix.select(iq, ik, sk - rows)
        assert a.offset == b.offset and a.sk == b.sk == sk
        assert mx.array_equal(a.top_idx, b.top_idx).item()
    iq, ik = _index(2048, 4, 1)
    assert ix.select(iq, ik, 2047) is None                            # every block visible: no selection


@needs_metal
def test_batch_rows_and_strided_inputs():
    """Batch 2 with several rows (a matrix product in MLX too), non-contiguous queries (a transposed view, the
    DSpark per-row slice) and keys read through a view of a longer buffer whose tail is NaN."""
    iq, ik = _index(3011, 4, 6, batch=2, seed=5)
    ref = _reference(iq, ik, 3011 - 6)
    got = _routed(iq, ik, 3011 - 6)
    assert _identity(ref[0], got[0], ref[1], got[1], 3011 - 6)
    sk, rows = 8268, 9
    iq, ik = _index(sk, 4, rows, seed=6)
    iq_t = mx.contiguous(iq.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)       # [1, G, L, D] view, rows outermost
    buf = mx.concatenate([ik, mx.full((1, 1, 300, IDX_D), mx.nan, dtype=ik.dtype)], axis=2)
    view = buf[:, :, :sk]
    ref = _reference(iq, ik, sk - rows)
    got = _routed(iq_t, view, sk - rows)
    assert _identity(ref[0], got[0], ref[1], got[1], sk - rows)
    assert not bool(mx.isnan(got[0]).any())                          # the NaN tail past sk is never read
    for r in range(rows):                                            # DSpark verify: one row, strided slice
        one_ref = _reference(iq[:, :, r:r + 1], ik[:, :, :sk - rows + r + 1], sk - rows + r)
        one = _routed(iq_t[:, :, r:r + 1], view[:, :, :sk - rows + r + 1], sk - rows + r)
        assert _identity(one_ref[0], one[0], one_ref[1], one[1], sk - rows + r)


@needs_metal
def test_multi_row_selection_equals_one_row_selections():
    """A row of a multi-row call scores exactly like the one-row call a decode step makes at its position."""
    sk, rows = 4095 + 7, 7
    iq, ik = _index(sk, 8, rows, seed=9)
    off = sk - rows
    multi = _routed(iq, ik, off)
    for r in range(rows):
        one = _routed(iq[:, :, r:r + 1], ik[:, :, :off + r + 1], off + r)
        nb = one[0].shape[-1]
        assert mx.array_equal(multi[0][:, :, r:r + 1, :nb], one[0]).item(), r
        # keys past the row's position: -inf in the multi-row call, absent in the one-row call
        assert bool((multi[0][:, :, r, nb:] == -mx.inf).all())


# ----------------------------------------------------------------------------------------------- ties / NaN
@needs_metal
@pytest.mark.parametrize("sk, rows", [(2853, 1), (3011, 5), (3011, 600)])
def test_exact_ties_break_like_the_reference(sk, rows):
    """Index keys repeat every block, so every fully causal block of a row has exactly the same score: the top-k is
    decided by argpartition's tie-break alone, which must see identical inputs."""
    rng = np.random.default_rng(sk + rows)
    one = rng.standard_normal((BLOCK, IDX_D)).astype(np.float32)
    ik = mx.array(np.tile(one, (-(-sk // BLOCK), 1))[:sk])[None, None].astype(mx.bfloat16)
    iq = mx.array(rng.standard_normal((1, 4, rows, IDX_D)).astype(np.float32)).astype(mx.bfloat16)
    off = sk - rows
    ref = _reference(iq, ik, off)
    got = _routed(iq, ik, off)
    s = np.array(ref[0])
    srt = -np.sort(-s, axis=-1)
    full_rows = (off + np.arange(rows)) >= (TOPK + 1) * BLOCK         # rows with > 16 fully causal blocks
    assert full_rows.any()
    assert (srt[:, :, full_rows, TOPK - 1] == srt[:, :, full_rows, TOPK]).all(), "no tie at the top-k boundary"
    assert _identity(ref[0], got[0], ref[1], got[1], off)


@needs_metal
def test_nan_scores_propagate_like_the_reference():
    """A NaN index key inside a row's causal range makes that block's score NaN (MLX's max propagates NaN); one
    past the row's position is masked out."""
    sk, rows = 3011, 5
    iq, ik = _index(sk, 4, rows, seed=11)
    ikn = np.array(ik.astype(mx.float32))
    ikn[0, 0, 700, 3] = np.nan                                        # block 5: causal for every row
    ikn[0, 0, sk - 2, 0] = np.nan                                     # past rows 0 .. 2, causal for rows 3, 4
    ikn = mx.array(ikn).astype(mx.bfloat16)
    off = sk - rows
    ref = _reference(iq, ikn, off)
    got = _routed(iq, ikn, off)
    r = np.array(ref[0])
    assert np.isnan(r[0, :, :, 5]).all()
    assert _identity(ref[0], got[0], ref[1], got[1], off)


# ----------------------------------------------------------------------------------------------- caches
def _attention(groups: int = 4, seed: int = 0) -> Attention:
    a = Attention(_args(groups), 0)
    rng = np.random.default_rng(seed)
    for w in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, a.indexer.index_q_proj, a.indexer.index_k_proj):
        w.weight = mx.array((rng.standard_normal(w.weight.shape) / np.sqrt(w.weight.shape[1])).astype(np.float32))
    a.set_dtype(mx.bfloat16)
    return a


def _x(n: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal((1, n, 256)).astype(np.float32)).astype(mx.bfloat16)


def _prefill(attn, cache, x, step: int = 2048):
    for s in range(0, x.shape[1], step):
        mx.eval(attn(x[:, s:s + step], mask="causal", cache=cache))


def _decode(attn, cache, xs):
    outs = []
    for r in range(xs.shape[1]):
        o = attn(xs[:, r:r + 1], mask=None, cache=cache)
        mx.eval(o)
        outs.append(o)
    return outs


@needs_metal
@pytest.mark.parametrize("prefix", [2853, 3011])
def test_restored_and_cloned_caches_select_like_the_reference(prefix, monkeypatch):
    """Cache offsets not aligned to 128 / 256: live (step-padded index lane), restored (exact-length) and cloned
    caches decode bit-identically with kernel selection and with the reference selection."""
    spy = _SelectSpy(monkeypatch)
    arith = _admitted()
    attn = _attention()
    x = _x(prefix, 1)
    live = MiniMaxM3SparseCache()
    _prefill(attn, live, x)
    assert prefix % 128 and live.idx_keys.shape[2] > live.offset == prefix
    steps = _x(5, 2)
    outs = {}
    for path in ("reference", None):
        if path:
            monkeypatch.setenv(msa.SELECT_ENV, path)
        caches = {"live": clone_minimax_m3_sparse(live), "restored": restore_minimax_m3_sparse(*live.state),
                  "cloned": clone_minimax_m3_sparse(live)}
        spy.picks.clear()
        outs[path] = {name: _decode(attn, c, steps) for name, c in caches.items()}
        assert set(spy.picks) == {"reference" if path else arith}
        monkeypatch.delenv(msa.SELECT_ENV, raising=False)
    base = outs["reference"]["live"]
    for path in outs:
        for name, seq in outs[path].items():
            for r in range(5):
                assert mx.array_equal(seq[r], base[r]).item(), (path, name, r)


@needs_metal
def test_partial_hit_with_poisoned_index_tail_selects_like_a_fresh_cache(monkeypatch):
    """Same prefix, another suffix trimmed off; the stale index-lane tail is NaN. Kernel selection never reads it:
    the suffix chunk and the decode steps after it equal a cache that never saw the first suffix."""
    spy = _SelectSpy(monkeypatch)
    arith = _admitted()
    attn = _attention(seed=3)
    prefix = 2853
    x = _x(prefix, 3)
    hit = MiniMaxM3SparseCache()
    _prefill(attn, hit, x)
    mx.eval(attn(_x(37, 4), mask="causal", cache=hit))
    assert hit.trim(37) == 37 and hit.offset == prefix
    hit.idx_keys[..., prefix:, :] = mx.nan
    hit.keys[..., prefix:, :] = mx.nan
    hit.values[..., prefix:, :] = mx.nan
    fresh = MiniMaxM3SparseCache()
    _prefill(attn, fresh, x)
    b = _x(41, 5)
    spy.picks.clear()
    oh = attn(b, mask="causal", cache=hit)
    of = attn(b, mask="causal", cache=fresh)
    assert mx.array_equal(oh, of).item() and bool(mx.isfinite(oh).all())
    for r, (s1, s2) in enumerate(zip(_decode(attn, hit, _x(4, 6)), _decode(attn, fresh, _x(4, 6)))):
        assert mx.array_equal(s1, s2).item() and bool(mx.isfinite(s1).all()), r
    assert spy.picks and set(spy.picks) == {arith}


# ----------------------------------------------------------------------------------------------- controls
@needs_metal
@pytest.mark.parametrize("sk, rows", [(8268, 1), (8268, 600)])
def test_control_shifted_block_ids_fail(sk, rows):
    iq, ik = _index(sk, 4, rows, seed=21)
    off = sk - rows
    ref = _reference(iq, ik, off)
    got = _routed(iq, ik, off)
    nb = ref[0].shape[-1]
    shifted = (got[1].astype(mx.int32) + 1) % nb
    assert not _identity(ref[0], got[0], ref[1], shifted, off)
    assert _flip_rate(ref[1], shifted) > 0.99


@needs_metal
@pytest.mark.parametrize("sk, rows", [(8268, 1), (8268, 600)])
def test_control_local_block_not_forced_fails(sk, rows):
    """The kernel with local = 0 (no forced block): scores differ at every row's own block; selections lose the
    local block wherever it did not win on its own score."""
    arith = _admitted()
    iq, ik = _index(sk, 4, rows, seed=22)
    off = sk - rows
    ref = _reference(iq, ik, off)
    s = msa.kernel_block_scores(iq, ik, off, BLOCK, 0, arith)
    got = (s, _top(s))
    mx.eval(*got)
    assert not _identity(ref[0], got[0], ref[1], got[1], off)
    assert not np.isposinf(np.array(got[0])).any()


def _mutation_cases(arith: str):
    """(source attribute, original, mutated, shape) per kernel of the admitted family."""
    if arith == "nax":
        return [
            ("_SELECT_NAX_SPLIT_SOURCE", "mm = max(mm, simd_shuffle_xor(mm, ushort(8)));", "", "decode"),
            ("_SELECT_NAX_SPLIT_SOURCE", "for (int tt = 0; tt < NT; ++tt) {", "for (int tt = 0; tt < NT - 1; ++tt) {",
             "decode"),
            ("_SELECT_NAX_SOURCE", "mm = max(mm, simd_shuffle_xor(mm, ushort(8)));", "", "chunk"),
            ("_SELECT_NAX_SOURCE", "if (kp <= qp[u][i] && kp < sk) {", "if (kp <= qp[u][i] + 1 && kp < sk) {",
             "chunk"),
        ]
    return [
        ("_SELECT_FMA_SOURCE", "for (int i = 0; i < NSGB; ++i) {", "for (int i = 0; i < NSGB - 1; ++i) {", "decode"),
        ("_SELECT_FMA_SOURCE", "const float bm = simd_max(x);", "const float bm = simd_max(x) - 1e-3f;", "decode"),
        ("_SELECT_SIMD_SOURCE", "const float m = max(red[tid], red[TP + tid]);", "const float m = red[tid];", "chunk"),
        ("_SELECT_SIMD_SOURCE", "if (kp <= qpos && kp < sk) {", "if (kp <= qpos + 1 && kp < sk) {", "chunk"),
    ]


@needs_metal
@pytest.mark.parametrize("case", range(4))
def test_control_mutated_kernel_fails(case, monkeypatch):
    """Each kernel of the admitted family, mutated (a lane / tile dropped from the block max, the causal limit one
    key late), fails the identity check; the unmutated kernel passes the same case."""
    arith = _admitted()
    attr, old, new, shape = _mutation_cases(arith)[case]
    src = getattr(msa, attr)
    assert old in src
    sk, rows = (8268, 1) if shape == "decode" else (8268, 600)
    iq, ik = _index(sk, 4, rows, seed=30 + case)
    off = sk - rows
    ref = _reference(iq, ik, off)
    msa._SELECT_KERNELS.clear()
    monkeypatch.setattr(msa, attr, src.replace(old, new))
    try:
        s = msa.kernel_block_scores(iq, ik, off, BLOCK, 1, arith)
        mutated = (s, _top(s))
        mx.eval(*mutated)
    finally:
        monkeypatch.setattr(msa, attr, src)
        msa._SELECT_KERNELS.clear()
    assert not _identity(ref[0], mutated[0], ref[1], mutated[1], off)
    good = _routed(iq, ik, off)
    assert _identity(ref[0], good[0], ref[1], good[1], off)


@needs_metal
@pytest.mark.parametrize("sk, rows", [(8268, 1), (32845, 16), (8268, 2048)])
def test_control_other_arithmetic_fails(sk, rows):
    """The non-admitted kernel family computes the same math in another rounding order: its block scores differ
    from the reference, so it is never admitted (its selections may or may not flip at a given row)."""
    arith = _admitted()
    other = "fma" if arith == "nax" else "nax"
    if other == "nax" and not msa.nax_available():
        pytest.skip("no matrix units")
    iq, ik = _index(sk, 4, rows, seed=40)
    off = sk - rows
    ref = _reference(iq, ik, off)
    s = msa.kernel_block_scores(iq, ik, off, BLOCK, 1, other)
    got = (s, _top(s))
    mx.eval(*got)
    assert not _identity(ref[0], got[0], ref[1], got[1], off)


# ----------------------------------------------------------------------------------------------- fma family
def _fma_chain_scores(iq: np.ndarray, ik: np.ndarray, offset: int) -> np.ndarray:
    """Block scores with one float32 multiply-add per head-dim element in ascending order. For bf16 / fp16 inputs
    every product is exact in float32, so separate multiply + add rounds exactly like a fused multiply-add."""
    G, L, Dd = iq.shape
    sk = ik.shape[0]
    s = np.zeros((G * L, sk), dtype=np.float32)
    q2 = iq.reshape(G * L, Dd).astype(np.float32)
    k2 = ik.astype(np.float32)
    for d in range(Dd):
        s = s + np.outer(q2[:, d], k2[:, d]).astype(np.float32)
    s = s.reshape(G, L, sk)
    qpos = offset + np.arange(L)
    s = np.where(np.arange(sk)[None, None, :] > qpos[None, :, None], -np.inf, s)
    nb = -(-sk // BLOCK)
    s = np.concatenate([s, np.full((G, L, nb * BLOCK - sk), -np.inf, dtype=np.float32)], axis=-1)
    out = s.reshape(G, L, nb, BLOCK).max(axis=-1)
    out[:, np.arange(L), qpos // BLOCK] = np.inf
    return out[None]


@needs_metal
@pytest.mark.parametrize("sk, rows", [(2853, 1), (3011, 5), (2853, 40), (3011, 300)])
def test_fma_kernels_equal_a_sequential_fma_chain(sk, rows):
    """The fma family (scalar kernel for decode-sized calls, simdgroup-matrix kernel for chunks) against a NumPy
    float32 chain, whatever this process's TF32 setting: the family is exact on its own terms."""
    iq, ik = _index(sk, 4, rows, seed=50)
    off = sk - rows
    want = _fma_chain_scores(np.array(iq.astype(mx.float32))[0], np.array(ik.astype(mx.float32))[0, 0], off)
    got = msa.kernel_block_scores(iq, ik, off, BLOCK, 1, "fma")
    assert np.array_equal(np.array(got), want)


# ----------------------------------------------------------------------------------------------- routing / memory
def test_selection_routing(monkeypatch):
    iq, ik = _index(3011, 4, 3)
    if METAL:
        arith = _admitted()
        assert msa.choose_select_path(iq, ik, BLOCK) == arith
        assert msa.choose_select_path(iq[:, :, :1], ik, BLOCK) == arith
        iq2, ik2 = _index(3011, 4, 1, batch=2)
        assert msa.choose_select_path(iq2, ik2, BLOCK) == "reference"            # MLX: matrix-vector kernel
        iq2, ik2 = _index(3011, 4, 3, batch=2)
        assert msa.choose_select_path(iq2, ik2, BLOCK) == arith
        assert msa.choose_select_path(iq, ik.astype(mx.float32), BLOCK) == "reference"   # mixed dtypes
        monkeypatch.setenv(msa.ENV_PATH, "dense")                                 # the pre-kernel runtime
        assert msa.choose_select_path(iq, ik, BLOCK) == "reference"
        monkeypatch.setenv(msa.SELECT_ENV, "kernel")
        assert msa.choose_select_path(iq, ik, BLOCK) == arith
        monkeypatch.delenv(msa.ENV_PATH)
        monkeypatch.setenv(msa.SELECT_ENV, "fma")
        assert msa.choose_select_path(iq, ik, BLOCK) == "fma"
        monkeypatch.setenv(msa.SELECT_ENV, "reference")
        assert msa.choose_select_path(iq, ik, BLOCK) == "reference"
    monkeypatch.delenv(msa.SELECT_ENV, raising=False)
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert msa.choose_select_path(iq, ik, BLOCK) == "reference"                  # CPU: the reference ops
    finally:
        mx.set_default_device(device)


@needs_metal
def test_kernel_selection_never_builds_the_score_sheet():
    """2048-row chunk at 32845 keys: the reference's peak includes its [4, 2048, 32845] float32 sheet (1 GiB); the
    kernel's peak is its inputs plus [4, 2048, 257] block scores."""
    sk, rows = 32845, 2048
    iq, ik = _index(sk, 4, rows, seed=60)
    off = sk - rows
    mx.eval(iq, ik)
    peaks = {}
    for name, fn in (("reference", lambda: _reference(iq, ik, off)), ("kernel", lambda: _routed(iq, ik, off))):
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        out = fn()
        peaks[name] = (mx.get_peak_memory() - base) / 2**20
        del out
    assert peaks["reference"] > 1024                                  # the sheet is real (harness sees it)
    assert peaks["kernel"] < 32, peaks

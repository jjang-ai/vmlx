# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 MSA: block-sparse attention paths (decode kernel, chunk kernel, op-level gather) vs the dense path.

Synthetic tensors at the released attention geometry (64 query heads, 4 KV groups, head dim 128, 4 index heads of
128, blocks of 128, top-16, local 1) plus an 8-group variant. The reference is float64 NumPy written from the MSA
rule (key j allowed for row r when its block is selected for (r, group) and j <= offset + r), independent of the
kernels. Lengths do not divide by the block (128) or the K / V step (256): 2049, 2853, 3011, 8191 + 77; 2047 / 2048
check the all-blocks-visible boundary.

Every equivalence check below goes through ``_equivalent``; the negative controls (one selection shared by every
group, block ids off by one, the forced local block dropped) feed wrong selections through the same check and must
fail it. Exactness checks (verify rows vs decode steps, restored / partially hit caches) use mx.array_equal, no
tolerance, and spy on the router so a test cannot pass on the dense path by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.models.minimax_m3 import msa  # noqa: E402
from vmlx_engine.models.minimax_m3.cache import (  # noqa: E402
    MiniMaxM3SparseCache, clone_minimax_m3_sparse, restore_minimax_m3_sparse)
from vmlx_engine.models.minimax_m3.minimax_m3 import Attention, ModelArgs  # noqa: E402

H, D, IDX_D, BLOCK, TOPK = 64, 128, 128, 128, 16


def _gpu_generation() -> tuple:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    arch = str(info.get("architecture", ""))
    digits = "".join(ch for ch in arch.split("applegpu_g")[-1] if ch.isdigit()) if arch.startswith("applegpu_g") else ""
    return arch, int(digits) if digits else 0


# M5-class GPUs (applegpu_g17+) have the matrix units: the bf16 kernel tests run there whether or not the kernel's
# own self-check passed, so a broken kernel fails them instead of being skipped.
ARCH, GPU_GEN = _gpu_generation()
MATRIX_UNITS = GPU_GEN >= 17
FP32_TOL = 2e-5           # per-row relative error vs float64 (measured <= 7.1e-7)
BF16_TOL = 6e-3           # bf16 inputs / output: the float64 answer rounded to bf16 alone is ~1.7e-3


@pytest.fixture(autouse=True)
def _auto_routing(monkeypatch):
    monkeypatch.delenv(msa.ENV_PATH, raising=False)


def _attention(groups: int = 4, hidden: int = 256, seed: int = 0) -> Attention:
    args = ModelArgs(hidden_size=hidden, num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=groups,
                     head_dim=D, rotary_dim=64, index_n_heads=groups, index_head_dim=IDX_D,
                     index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1, sparse_attention_freq=[1])
    a = Attention(args, 0)
    rng = np.random.default_rng(seed)
    params = {}
    for name, w in (("q_proj", a.q_proj), ("k_proj", a.k_proj), ("v_proj", a.v_proj), ("o_proj", a.o_proj)):
        params[name] = rng.standard_normal(w.weight.shape) / np.sqrt(w.weight.shape[1])
    a.q_proj.weight, a.k_proj.weight = mx.array(params["q_proj"], mx.float32), mx.array(params["k_proj"], mx.float32)
    a.v_proj.weight, a.o_proj.weight = mx.array(params["v_proj"], mx.float32), mx.array(params["o_proj"], mx.float32)
    ix = a.indexer
    ix.index_q_proj.weight = mx.array(rng.standard_normal(ix.index_q_proj.weight.shape) / np.sqrt(hidden), mx.float32)
    ix.index_k_proj.weight = mx.array(rng.standard_normal(ix.index_k_proj.weight.shape) / np.sqrt(hidden), mx.float32)
    return a


def _tensors(sk: int, groups: int, dtype, seed: int = 0, rows: int = 0):
    """q [1, H, rows or sk, D], k / v [1, G, sk, D], idx_q [1, G, ., IDX_D], idx_k [1, 1, sk, IDX_D]. Index keys
    drift slowly so neighbouring blocks score alike while groups still pick different blocks."""
    rng = np.random.default_rng(seed)
    n = rows or sk
    ik = rng.standard_normal((sk, IDX_D)) + np.cumsum(rng.standard_normal((sk, IDX_D)) * 0.05, axis=0)
    f = lambda a: mx.array(a.astype(np.float32)).astype(dtype)[None]  # noqa: E731
    return (f(rng.standard_normal((H, n, D)) * 0.5), f(rng.standard_normal((groups, sk, D))),
            f(rng.standard_normal((groups, sk, D))), f(rng.standard_normal((groups, n, IDX_D))), f(ik[None]))


def _reference(q, k, v, top_idx, offset: int, scale: float) -> np.ndarray:
    """float64 MSA attention [rows, H * D] from the selection's block ids (None = every block visible)."""
    Q = np.array(q.astype(mx.float32), dtype=np.float64)[0]                 # [H, rows, D]
    K = np.array(k.astype(mx.float32), dtype=np.float64)[0]
    V = np.array(v.astype(mx.float32), dtype=np.float64)[0]
    G, rows = K.shape[0], Q.shape[1]
    sk = K.shape[1]
    hg = H // G
    pos = np.arange(sk)
    out = np.zeros((rows, H, D))
    top = None if top_idx is None else np.array(top_idx).astype(np.int64)[0]   # [G, rows, keep]
    with np.errstate(all="ignore"):         # Accelerate's matmul raises spurious FP flags on finite inputs
        for r in range(rows):
            q_pos = offset + r
            for g in range(G):
                allowed = pos <= q_pos
                if top is not None:
                    allowed &= np.isin(pos // BLOCK, top[g, r])
                s = Q[g * hg:(g + 1) * hg, r] @ K[g, allowed].T * scale          # [hg, n]
                p = np.exp(s - s.max(-1, keepdims=True))
                out[r, g * hg:(g + 1) * hg] = (p / p.sum(-1, keepdims=True)) @ V[g, allowed]
    assert np.isfinite(out).all()
    return out.reshape(rows, H * D)


def _rows(o) -> np.ndarray:
    """[1, H, rows, D] -> [rows, H * D] float64."""
    a = np.array(o.astype(mx.float32), dtype=np.float64)[0]
    return a.transpose(1, 0, 2).reshape(a.shape[1], -1)


def _rel(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def _equivalent(a: np.ndarray, ref: np.ndarray, tol: float) -> bool:
    """THE equivalence check: every row within ``tol`` relative error, no NaN / inf."""
    return bool(np.isfinite(a).all() and (_rel(a, ref) <= tol).all())


def _run(attn, path, q, k, v, sel, mask=None, monkeypatch=None):
    monkeypatch.setenv(msa.ENV_PATH, path)
    out = attn.attend(q, k, v, sel, mask, None)
    mx.eval(out)
    monkeypatch.delenv(msa.ENV_PATH)
    return out


def _control(sel, kind: str):
    top = np.array(sel.top_idx).astype(np.int64)
    if kind == "shared":
        top = np.broadcast_to(top[:, :1], top.shape).copy()
    elif kind == "off_by_one":
        top = (top + 1) % sel.n_blocks
    elif kind == "no_local":
        for r in range(top.shape[2]):
            local = (sel.offset + r) // sel.block
            for g in range(top.shape[1]):
                row = top[0, g, r]
                spare = next(x for x in range(sel.n_blocks) if x not in set(row.tolist()))
                row[row == local] = spare
    return msa.MSASelection(mx.array(top.astype(np.uint32)), sel.offset, sel.sk, sel.block)


class _Spy:
    """Counts the router's picks."""

    def __init__(self, monkeypatch):
        self.picks = []
        real = msa.choose_path

        def spy(*a, **kw):
            p = real(*a, **kw)
            self.picks.append(p)
            return p

        monkeypatch.setattr(msa, "choose_path", spy)


# ----------------------------------------------------------------------------------------------- selection
def test_dense_bias_is_the_selection_as_a_causal_key_mask():
    """The dense path's mask, rebuilt from the block ids, is exactly 'block selected and key <= query position'."""
    attn = _attention()
    q, k, v, iq, ik = _tensors(2853, 4, mx.float32, rows=40)
    sel = attn.indexer.select(iq, ik, 2853 - 40)
    bias = np.array(sel.dense_bias())[0]                                  # [G, rows, sk]
    top = np.array(sel.top_idx).astype(np.int64)[0]
    pos = np.arange(2853)
    for g in range(4):
        for r in range(40):
            allowed = (pos <= 2853 - 40 + r) & np.isin(pos // BLOCK, top[g, r])
            assert np.array_equal(bias[g, r] == 0, allowed) and np.isneginf(bias[g, r][~allowed]).all()


def test_every_block_visible_up_to_topk_times_block():
    """keep >= n_blocks (<= 2048 keys): no selection, plain causal attention; one key more selects."""
    attn = _attention()
    q, k, v, iq, ik = _tensors(2049, 4, mx.float32)
    for sk in (2047, 2048):
        assert attn.indexer.select(iq[:, :, sk - 1:sk], ik[:, :, :sk], sk - 1) is None
    sel = attn.indexer.select(iq[:, :, 2048:2049], ik, 2048)
    assert sel is not None and sel.n_blocks == 17 and sel.keep == 16


# ----------------------------------------------------------------------------------------------- decode rows
def _decode_case(sk, groups, dtype, n_random=10, seed=0):
    attn = _attention(groups)
    q, k, v, iq, ik = _tensors(sk, groups, dtype, seed)
    rng = np.random.default_rng(seed)
    positions = sorted(p for p in {sk - 1, sk - 2, 2048, *rng.integers(2048, sk, n_random).tolist()} if p >= 2048)
    cases = []
    for p in positions:
        sel = attn.indexer.select(iq[:, :, p:p + 1], ik[:, :, :p + 1], p)
        assert sel is not None and sel.sk == p + 1
        cases.append((p, sel, q[:, :, p:p + 1], k[:, :, :p + 1], v[:, :, :p + 1]))
    return attn, cases


@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("sk", [2049, 2853, 3011, 8191 + 77])
def test_decode_rows_match_the_reference_fp32(sk, groups, monkeypatch):
    attn, cases = _decode_case(sk, groups, mx.float32)
    for path in ("decode", "gather", "dense"):
        for p, sel, qp, kp, vp in cases:
            out = _rows(_run(attn, path, qp, kp, vp, sel, monkeypatch=monkeypatch))
            ref = _reference(qp, kp, vp, sel.top_idx, p, attn.scale)
            assert _equivalent(out, ref, FP32_TOL), (path, p, _rel(out, ref))


@pytest.mark.parametrize("sk", [2049, 3011, 8191 + 77])
def test_decode_rows_match_the_reference_bf16(sk, monkeypatch):
    attn, cases = _decode_case(sk, 4, mx.bfloat16, n_random=6)
    for path in ("decode", "dense"):
        for p, sel, qp, kp, vp in cases:
            out = _rows(_run(attn, path, qp, kp, vp, sel, monkeypatch=monkeypatch))
            ref = _reference(qp, kp, vp, sel.top_idx, p, attn.scale)
            assert _equivalent(out, ref, BF16_TOL), (path, p, _rel(out, ref))


@pytest.mark.parametrize("kind", ["shared", "off_by_one", "no_local"])
@pytest.mark.parametrize("path", ["decode", "gather"])
def test_control_wrong_selection_fails_the_decode_check(kind, path, monkeypatch):
    """A wrong selection run through the sparse path must fail the same equivalence check the real one passes."""
    attn, cases = _decode_case(3011, 4, mx.float32, n_random=4)
    failed = 0
    for p, sel, qp, kp, vp in cases:
        ref = _reference(qp, kp, vp, sel.top_idx, p, attn.scale)
        assert _equivalent(_rows(_run(attn, path, qp, kp, vp, sel, monkeypatch=monkeypatch)), ref, FP32_TOL)
        bad = _rows(_run(attn, path, qp, kp, vp, _control(sel, kind), monkeypatch=monkeypatch))
        failed += not _equivalent(bad, ref, FP32_TOL)
    assert failed == len(cases), f"control {kind} passed at {len(cases) - failed} / {len(cases)} positions"


def test_multi_row_decode_call_equals_single_row_calls_bit_for_bit():
    """The decode kernel's rows do not depend on how many rows share the launch."""
    attn = _attention()
    sk, R = 3011, 9
    q, k, v, iq, ik = _tensors(sk, 4, mx.bfloat16)
    sel = attn.indexer.select(iq[:, :, sk - R:], ik, sk - R)
    many = msa.decode_kernel_attention(q[:, :, sk - R:], k, v, sel, attn.scale)
    for r in range(R):
        one = msa.MSASelection(mx.contiguous(sel.top_idx[:, :, r:r + 1]), sk - R + r, sk - R + r + 1, BLOCK)
        assert mx.array_equal(many[:, :, r:r + 1], msa.decode_kernel_attention(
            q[:, :, sk - R + r:sk - R + r + 1], k[:, :, :sk - R + r + 1], v[:, :, :sk - R + r + 1], one,
            attn.scale)).item(), r


# ----------------------------------------------------------------------------------------------- prefill chunks
@pytest.mark.parametrize("sk, rows", [(2853, 300), (3011, 77), (8191 + 77, 160)])
def test_chunk_kernel_matches_the_reference_fp32(sk, rows, monkeypatch):
    """fp32 runs the simdgroup kernel (the matrix units round fp32 operands); rows straddle blocks and the chunk's
    own causal region, selected future blocks included."""
    attn = _attention()
    q, k, v, iq, ik = _tensors(sk, 4, mx.float32, rows=rows)
    sel = attn.indexer.select(iq, ik, sk - rows)
    ref = _reference(q, k, v, sel.top_idx, sk - rows, attn.scale)
    out = _rows(_run(attn, "kernel", q, k, v, sel, "causal", monkeypatch))
    assert _equivalent(out, ref, FP32_TOL), _rel(out, ref).max()
    for kind in ("shared", "off_by_one", "no_local"):
        bad = _rows(msa.kernel_attention(q, k, v, _control(sel, kind), attn.scale))
        assert not _equivalent(bad, ref, FP32_TOL), kind


@pytest.mark.skipif(not MATRIX_UNITS, reason="no matrix units (M5-class GPU) here")
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("sk, rows", [(2853, 300), (8191 + 77, 160)])
def test_matrix_unit_chunk_kernel_matches_the_reference_bf16(sk, rows, groups, monkeypatch):
    attn = _attention(groups)
    q, k, v, iq, ik = _tensors(sk, groups, mx.bfloat16, rows=rows)
    sel = attn.indexer.select(iq, ik, sk - rows)
    ref = _reference(q, k, v, sel.top_idx, sk - rows, attn.scale)
    out = _rows(msa.kernel_attention(q, k, v, sel, attn.scale, _matrix_units=True))
    assert _equivalent(out, ref, BF16_TOL), _rel(out, ref).max()
    dense = _rows(_run(attn, "dense", q, k, v, sel, "causal", monkeypatch))
    assert _equivalent(dense, ref, BF16_TOL)
    for kind in ("shared", "off_by_one", "no_local"):
        bad = _rows(msa.kernel_attention(q, k, v, _control(sel, kind), attn.scale, _matrix_units=True))
        assert not _equivalent(bad, ref, BF16_TOL), kind


@pytest.mark.skipif(not MATRIX_UNITS, reason="no matrix units (M5-class GPU) here")
def test_matrix_unit_kernel_spikes_on_boundary_keys():
    """Precision-independent masking check for the bf16 kernel: one key gets a huge score and a distinct value. The
    output must follow it exactly when the key is allowed (the row's own key, a partial last block) and ignore it
    when masked (one past the row in its own block, an unselected block)."""
    G, sk, rows = 4, 2853, 4
    rng = np.random.default_rng(5)
    for spike, row, allowed in ((2852, 3, True), (2850, 1, True), (2851, 1, False), (5, 3, None)):
        q = rng.standard_normal((1, H, rows, D)).astype(np.float32) * 0.1
        k = rng.standard_normal((1, G, sk, D)).astype(np.float32) * 0.1
        v = rng.standard_normal((1, G, sk, D)).astype(np.float32)
        k[0, :, spike] = 0.0
        k[0, :, spike, 0] = 2000.0                                      # score ~177 after scaling: decisive
        q[0, :, :, 0] = 1.0
        v[0, :, spike] = 5.0
        top = np.tile(np.arange(TOPK, dtype=np.uint32)[None, None, None], (1, G, rows, 1))
        top[..., 0] = (sk - rows + np.arange(rows)) // BLOCK            # local block first
        top[..., 1:] = np.arange(1, TOPK)[None, None, None] + 1        # blocks 2..16; block 0 / 1 unselected
        sel = msa.MSASelection(mx.array(top), sk - rows, sk, BLOCK)
        if allowed is None:                                              # key 5 sits in unselected block 0
            assert 0 not in top[0, :, row]
        out = np.array(msa.kernel_attention(mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16),
                                            mx.array(v).astype(mx.bfloat16), sel, D ** -0.5,
                                            _matrix_units=True).astype(mx.float32))[0, :, row]
        if allowed:
            assert np.abs(out - 5.0).max() < 0.05, (spike, row)
        else:
            assert np.abs(out).max() < 1.0, (spike, row)


@pytest.mark.skipif(not MATRIX_UNITS, reason="no matrix units (M5-class GPU) here")
def test_matrix_units_are_admitted_on_m5_class_gpus():
    """The matrix-unit kernel self-checks against the fp32 kernel before it is used (a wrong kernel is never
    routed to). On GPUs that have the units that check must pass, so a broken kernel fails here instead of silently
    falling back to the dense path."""
    assert msa.nax_available(), ARCH


# ----------------------------------------------------------------------------------------------- extent / caches
def test_gather_reads_only_rows_below_the_extent():
    """The op-level gather's source rows: every index < sk (clamped lanes masked), allowed == block selected and
    key <= row position, for a partial last block and rows whose selection includes future blocks."""
    attn = _attention()
    sk, rows = 3011, 70
    q, k, v, iq, ik = _tensors(sk, 4, mx.float32, rows=rows)
    sel = attn.indexer.select(iq, ik, sk - rows)
    src, allowed = (np.array(a) for a in msa._key_positions(sel))
    assert src.max() <= sk - 1 and src.min() >= 0
    top = np.sort(np.array(sel.top_idx).astype(np.int64), axis=-1)
    pos = (top[..., None] * BLOCK + np.arange(BLOCK)).reshape(src.shape)
    q_pos = (sk - rows + np.arange(rows)).reshape(1, 1, rows, 1)
    assert np.array_equal(allowed, pos <= q_pos)
    assert np.array_equal(src[allowed], pos[allowed])
    assert (pos >= sk).any() and not allowed[pos >= sk].any()          # the partial last block is exercised
def test_keys_past_the_extent_are_never_read(monkeypatch):
    """Poison K / V rows at and past sk (a step-padded buffer's tail) with NaN: every sparse path stays finite and
    unchanged. A path that read one would turn the row NaN (0 * NaN)."""
    attn = _attention()
    sk, cap, rows = 3011, 3328, 40
    q, k, v, iq, ik = _tensors(sk, 4, mx.bfloat16, rows=rows)
    kb = mx.concatenate([k, mx.full((1, 4, cap - sk, D), mx.nan, dtype=k.dtype)], axis=2)
    vb = mx.concatenate([v, mx.full((1, 4, cap - sk, D), mx.nan, dtype=v.dtype)], axis=2)
    for L in (1, rows):
        qq, off = q[:, :, rows - L:], sk - L
        sel = attn.indexer.select(iq[:, :, rows - L:], ik, off)
        for path in ("decode", "gather", "kernel"):
            clean = _run(attn, path, qq, k, v, sel, "causal" if L > 1 else None, monkeypatch)
            padded = _run(attn, path, qq, kb[:, :, :sk], vb[:, :, :sk], sel, "causal" if L > 1 else None, monkeypatch)
            raw = _run(attn, path, qq, kb, vb, sel, "causal" if L > 1 else None, monkeypatch)
            assert bool(mx.isfinite(raw).all()) and mx.array_equal(raw, clean).item(), (path, L)
            assert mx.array_equal(padded, clean).item(), (path, L)
    with pytest.raises(ValueError):                    # fewer K / V rows than the selection covers: desync
        msa.decode_kernel_attention(q[:, :, -1:], k[:, :, :sk - 1], v[:, :, :sk - 1],
                                    attn.indexer.select(iq[:, :, -1:], ik, sk - 1), attn.scale)


def _x(n: int, seed: int, hidden: int = 256, dtype=mx.bfloat16):
    return mx.array(np.random.default_rng(seed).standard_normal((1, n, hidden)).astype(np.float32)).astype(dtype)


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


@pytest.mark.parametrize("prefix", [2853, 3011])
def test_restored_cache_decodes_like_the_live_cache(prefix, monkeypatch):
    """A cache restored from its serialized state (exact-length buffers) or cloned decodes bit-identically to the
    live, step-padded cache; offsets are not multiples of 128 or of the 256-row step."""
    spy = _Spy(monkeypatch)
    attn = _attention()
    attn.set_dtype(mx.bfloat16)
    x = _x(prefix, 1)
    live = MiniMaxM3SparseCache()
    _prefill(attn, live, x)
    assert prefix % 128 and prefix % 256 and live.keys.shape[2] > live.offset == prefix   # padded buffer
    restored = restore_minimax_m3_sparse(*live.state)
    cloned = clone_minimax_m3_sparse(live)
    assert restored.keys.shape[2] == restored.offset == prefix
    steps = _x(5, 2)
    spy.picks.clear()
    a = _decode(attn, live, steps)
    assert spy.picks and set(spy.picks) == {"decode"}                    # the sparse decode path really ran
    b = _decode(attn, restored, steps)
    c = _decode(attn, cloned, steps)
    for r in range(5):
        assert mx.array_equal(a[r], b[r]).item() and mx.array_equal(a[r], c[r]).item(), r


def test_partial_hit_after_trim_ignores_the_stale_tail(monkeypatch):
    """Same prefix, different suffix: prefix + suffix A, trim back to the prefix (stale A rows stay in the buffer,
    poisoned with NaN here), then suffix B and decode == a cache that only ever saw prefix + B."""
    spy = _Spy(monkeypatch)
    attn = _attention()
    attn.set_dtype(mx.bfloat16)
    prefix, a_len, b_len = 2853, 37, 41
    x = _x(prefix, 3)
    hit = MiniMaxM3SparseCache()
    _prefill(attn, hit, x)
    mx.eval(attn(_x(a_len, 4), mask="causal", cache=hit))
    assert hit.trim(a_len) == a_len and hit.offset == prefix
    tail = slice(prefix, hit.keys.shape[2])
    hit.keys[..., tail, :] = mx.nan
    hit.values[..., tail, :] = mx.nan
    fresh = MiniMaxM3SparseCache()
    _prefill(attn, fresh, x)
    b = _x(b_len, 5)
    for path in ("sparse", None):
        if path:
            monkeypatch.setenv(msa.ENV_PATH, path)
        h, f = MiniMaxM3SparseCache(), MiniMaxM3SparseCache()
        h.state, f.state = hit.state, fresh.state
        h.keys, h.values = mx.array(hit.keys), mx.array(hit.values)       # the poisoned, padded buffers
        spy.picks.clear()
        oh = attn(b, mask="causal", cache=h)
        of = attn(b, mask="causal", cache=f)
        assert mx.array_equal(oh, of).item() and bool(mx.isfinite(oh).all())
        steps = _x(4, 6)
        for r, (s1, s2) in enumerate(zip(_decode(attn, h, steps), _decode(attn, f, steps))):
            assert mx.array_equal(s1, s2).item() and bool(mx.isfinite(s1).all()), r
        assert "decode" in spy.picks
        if path:
            assert "kernel" in spy.picks
        monkeypatch.delenv(msa.ENV_PATH, raising=False)


# ----------------------------------------------------------------------------------------------- DSpark invariant
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("prefix, width", [(2853, 7), (3069, 6), (4095, 4), (2047, 5)])
def test_verify_rows_equal_decode_steps_bit_for_bit(prefix, width, groups, monkeypatch):
    """DSpark's verify (MiniMaxM3DSparkTarget._attention: per-row selection + attend) == the same rows run as
    one-token decode steps, exactly, at M3.1 head geometry; windows straddle block (128) and step (256) boundaries
    and the 2048-key sparse threshold (2047 + 5)."""
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    spy = _Spy(monkeypatch)
    attn = _attention(groups)
    attn.set_dtype(mx.bfloat16)
    x = _x(prefix, 7)
    rows = _x(width, 8)
    dec, ver = MiniMaxM3SparseCache(), MiniMaxM3SparseCache()
    _prefill(attn, dec, x)
    _prefill(attn, ver, x)

    class _Stub:
        def _linear(self, module, t):
            return mx.concatenate([module(t[:, r:r + 1]) for r in range(t.shape[1])], axis=1)

    spy.picks.clear()
    v_out = MiniMaxM3DSparkTarget._attention(_Stub(), attn, rows, ver, prefix)
    mx.eval(v_out)
    verify_picks = list(spy.picks)
    spy.picks.clear()
    d_out = _decode(attn, dec, rows)
    assert verify_picks == spy.picks                                     # same path, row by row
    if prefix + 1 > TOPK * BLOCK:
        assert set(spy.picks) == {"decode"}
    else:
        assert "decode" in spy.picks                                     # crosses into the sparse regime
    for r in range(width):
        assert mx.array_equal(v_out[:, r:r + 1], d_out[r]).item(), r
    assert ver.offset == dec.offset == prefix + width


# ----------------------------------------------------------------------------------------------- routing
def test_routing_picks_by_rows_keys_and_dtype(monkeypatch):
    attn = _attention()
    for dt in (mx.bfloat16, mx.float32):
        q, k, v, iq, ik = _tensors(8191 + 77, 4, dt, rows=1)
        sel = attn.indexer.select(iq, ik, 8191 + 76)
        assert msa.choose_path(sel, q, k, None, 4) == "decode"
        assert msa.choose_path(sel, q, k, mx.zeros((1, 1, 1, sel.sk)), 4) == "dense"     # array mask
        assert msa.choose_path(sel, q, (k, k), None, 4) == "dense"                       # quantized K / V
        monkeypatch.setenv(msa.ENV_PATH, "dense")
        assert msa.choose_path(sel, q, k, None, 4) == "dense"
        monkeypatch.delenv(msa.ENV_PATH)
    q, k, v, iq, ik = _tensors(8191 + 77, 4, mx.bfloat16, rows=600)
    sel = attn.indexer.select(iq, ik, 8191 + 77 - 600)
    expect = "kernel" if msa.matrix_units_supported(q, sel) else "dense"
    assert msa.choose_path(sel, q, k, "causal", 4) == expect
    assert msa.choose_path(sel, q.astype(mx.float32), k.astype(mx.float32), "causal", 4) == "dense"
    q, k, v, iq, ik = _tensors(3011, 4, mx.bfloat16, rows=600)
    sel = attn.indexer.select(iq, ik, 3011 - 600)
    assert msa.choose_path(sel, q, k, "causal", 4) == "dense"            # below the measured crossover
    monkeypatch.setenv(msa.ENV_PATH, "sparse")
    assert msa.choose_path(sel, q, k, "causal", 4) == "kernel"

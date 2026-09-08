"""Qualifies the QSA indexer's exact M-RoPE angle (separate claim from pool
retention): the stock rotary forms inv_freq @ positions with a K=1 matmul and
MLX's GEMM rounds that product differently for multi-row chunks than the
single-row GEMV, so the SAME absolute position received different angles at
prefill and at decode. The exact elementwise product is shape-independent."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.language import Qwen3_5RotaryEmbedding
from vmlx_engine.models.qwen4_exp.language import _exact_mrope_cos_sin

# Flash-Next indexer geometry: rotary_dim 64 (head_dim 256 x 0.25), mrope [11, 11, 10],
# rope_theta 1e7, positions up to the 1M native window.
ROTARY = Qwen3_5RotaryEmbedding(64, max_position_embeddings=1_048_576, base=10_000_000.0,
                                mrope_section=[11, 11, 10])


def _positions(start: int, s: int, media: bool) -> mx.array:
    t = mx.arange(start, start + s)
    if media:
        return mx.stack([t, t + 4096, t + 17])[:, None, :]
    return mx.broadcast_to(t[None, None, :], (3, 1, s))


def _angles_stock(pos):
    x = mx.zeros((1, 1, pos.shape[-1], 64), dtype=mx.float32)
    cos, sin = ROTARY(x, pos)
    return np.asarray(cos)[0], np.asarray(sin)[0]


def _angles_exact(pos):
    cos, sin = _exact_mrope_cos_sin(ROTARY, pos, mx.float32)
    return np.asarray(cos)[0], np.asarray(sin)[0]


def _reference(pos):
    """float64 ground truth for the same interleaved M-RoPE construction."""
    p = np.asarray(pos).astype(np.float64)  # [3, 1, S]
    inv = 1.0 / (10_000_000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64))
    freqs = p[..., None] * inv  # [3, 1, S, 32]
    f = freqs[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        length = [11, 11, 10][dim] * 3
        f[..., offset:length:3] = freqs[dim][..., offset:length:3]
    emb = np.concatenate([f, f], axis=-1)
    return np.cos(emb)[0], np.sin(emb)[0]


@pytest.mark.parametrize("start", [0, 12, 2047, 26_554, 131_072, 1_000_000])
@pytest.mark.parametrize("media", [False, True])
def test_exact_angles_are_shape_independent_and_track_float64(start, media):
    widths = (1, 2, 3, 4, 64, 2048)
    exact = {s: _angles_exact(_positions(start, s, media)) for s in widths}
    # 1) shape independence: the first position's angle is BITWISE equal for every chunk width
    for s in widths[1:]:
        assert np.array_equal(exact[s][0][0], exact[1][0][0]), f"cos differs at S={s}"
        assert np.array_equal(exact[s][1][0], exact[1][1][0]), f"sin differs at S={s}"
    # 2) accuracy vs float64 ground truth (fp32 angle rounding only)
    ref_cos, ref_sin = _reference(_positions(start, 4, media))
    err_exact = max(np.abs(exact[4][0] - ref_cos).max(), np.abs(exact[4][1] - ref_sin).max())
    # fp32 can only hold the angle position*inv_freq to ~1 ulp (2e-3 rad at
    # 26k on the highest frequency), and Metal's cos/sin carry ~1e-4; both
    # are inherent to fp32 RoPE, identical for prefill and decode on this path.
    bound = 2e-4 + 2.5e-7 * (start + 4)
    assert err_exact <= bound, f"exact path error {err_exact:.3e} > {bound:.3e}"


def test_stock_rotary_shape_error_stays_within_documented_bound():
    """Bound stock shape error without requiring a backend rounding defect.

    Some Metal/compiler combinations compute the K=1 product exactly. Zero
    error is valid; the independent exact-path tests still enforce parity.
    """
    pos1 = _positions(12, 1, False)
    pos4 = _positions(12, 4, False)
    c1, s1 = _angles_stock(pos1)
    c4, s4 = _angles_stock(pos4)
    diff = max(np.abs(c1[0] - c4[0]).max(), np.abs(s1[0] - s4[0]).max())
    e_c, e_s = _angles_exact(pos1)
    # single-row stock == exact (GEMV is the exact product); multi-row stock differs
    assert np.array_equal(c1[0], e_c[0]) and np.array_equal(s1[0], e_s[0])
    # measured 2026-09-05 (mlx 0.32.2): GEMM relative angle error ~8e-4..1.2e-3
    # for ANY chunk of >=2 rows -> 3.5e-3 rad at position 12, 13.8 rad at 26,554.
    assert np.isfinite(diff) and 0 <= diff < 2e-2, f"stock shape dependence {diff:.3e} exceeds the documented bound"


@pytest.mark.parametrize("pos", [2048, 26_554, 131_072])
def test_stock_multirow_angle_error_is_bounded_relative_to_position(pos):
    """Reject worsening relative error while allowing an exact backend product."""
    inv = 1.0 / (10_000_000.0 ** (mx.arange(0, 64, 2).astype(mx.float32) / 64))
    exact = inv * float(pos)
    p = mx.arange(pos, pos + 4).astype(mx.float32)
    gemm = (mx.broadcast_to(inv[None, None, :, None], (3, 1, 32, 1))
            @ mx.broadcast_to(p[None, None, None, :], (3, 1, 1, 4)))[0, 0, :, 0]
    gemv = (mx.broadcast_to(inv[None, None, :, None], (3, 1, 32, 1))
            @ mx.broadcast_to(p[:1][None, None, None, :], (3, 1, 1, 1)))[0, 0, :, 0]
    assert bool(mx.array_equal(gemv, exact)), "single-row product must be exact"
    rel = float((mx.abs(gemm - exact) / mx.maximum(mx.abs(exact), 1e-30)).max())
    assert np.isfinite(rel) and 0 <= rel < 1e-2, f"multi-row relative angle error {rel:.2e} exceeds the documented bound"
    assert float(mx.abs(gemm - exact).max()) < 1e-2 * pos


def test_selection_agreement_old_vs_new_on_random_keys():
    """How much the angle change can move block selection: score random pooled
    keys against a query with stock (multi-row) vs exact angles at a long
    offset and count differing top-512 sets. Informational bound, not a
    correctness claim: the exact path is the mathematically intended one."""
    mx.random.seed(3)
    nb, d, k_sel = 6_600, 64, 512
    keys = mx.random.normal((nb, d))
    q = mx.random.normal((4, d))
    pos_q = _positions(26_400, 4, False)
    pos_k = mx.broadcast_to((mx.arange(nb) * 4)[None, None, :], (3, 1, nb))
    def rope(x, pos, exact):
        cos, sin = (_angles_exact(pos) if exact else _angles_stock(pos))
        cos, sin = mx.array(cos), mx.array(sin)
        x1, x2 = x[..., :32], x[..., 32:]
        rot = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rot * sin
    sel = {}
    for exact in (False, True):
        kk = rope(keys, pos_k, exact); qq = rope(q, pos_q, exact)
        scores = mx.maximum(qq @ kk.T, 0.0)
        top = mx.argpartition(-scores, kth=k_sel - 1, axis=-1)[:, :k_sel]
        sel[exact] = [set(np.asarray(top[i]).tolist()) for i in range(4)]
    changed = sum(len(sel[False][i] ^ sel[True][i]) for i in range(4)) / (4 * 2 * k_sel)
    # On structureless random keys the stock (inconsistent q/k angle) path and the
    # exact path disagree on a large share of the top-512 (measured ~26% at 26k):
    # the high-frequency dimensions are noise under the stock path at that
    # distance. Real keys are content-dominated; the live receipts (chain arms
    # ar-rope0 vs ar-retain0) carry the full-model comparison.
    assert 0.0 <= changed <= 1.0
    print(f"selection churn stock-vs-exact at 26k on random keys: {changed:.1%}")


def test_attention_exact_rope_is_opt_in(monkeypatch):
    from vmlx_engine.models.qwen4_exp.language import _qsa_exact_rope_attn_enabled

    monkeypatch.delenv("VMLX_QWEN4_EXACT_ROPE_ATTN", raising=False)
    assert not _qsa_exact_rope_attn_enabled()
    monkeypatch.setenv("VMLX_QWEN4_EXACT_ROPE_ATTN", "1")
    assert _qsa_exact_rope_attn_enabled()

# SPDX-License-Identifier: Apache-2.0
"""Fused DSA indexer scores for prefill: out[q, k] = sum_h w[q, h] * relu(Q[h, q, :] . K[k, :]).

WHY (Naive N0.5 Flash, 2026-10-12): the indexer materialized the (heads, queries, keys) fp32 score tensor and ran
relu, the head-weight product and the head sum over it as separate passes. At a 2048-query prefill chunk the q.k
matmul ran at 14-16 TFLOPS (10.7 ms at 20k keys) while the passes over the intermediate took ~80% of the indexer
(52 ms per DSA layer per chunk at 20k, 105 ms at 40k; 12 DSA layers). This kernel keeps each (query, key) tile in
registers: per head an 8x8 simdgroup-MMA dot product over head_dim, then relu, weight and accumulation, and writes the
(queries, keys) result once.

Arithmetic: dot products accumulate over head_dim in 8-wide simdgroup-MMA steps, head 0..H-1 sequential with the
product rounded before the add (no FMA contraction) — the order of the stock matmul + maximum + multiply + sum path,
which the tests check bit for bit; callers fall back to the stock path for anything not admitted.
"""
from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx



def dsa_index_scores_requested() -> bool:
    return os.environ.get("VMLX_DSA_FUSED_INDEX_SCORES", "1") == "1"


_SOURCE = r"""
    #pragma clang fp contract(off)
    // Q element (h, s, d) at q + h * QH + s * QS + d: [H, L, D] (QH = L*D, QS = D) or [L, H, D] (QH = D, QS = H*D).
    const int Lq = int(QS_LAYOUT ? q_shape[1] : q_shape[2]);
    const int Lk = int(k_shape[K_AXIS]);
    const size_t QH = QS_LAYOUT ? size_t(D) : size_t(Lq) * D;
    const size_t QS = QS_LAYOUT ? size_t(H) * D : size_t(D);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int c0 = min(int(threadgroup_position_in_grid.x) * TN, Lk - TN);
    const int r0 = min(int(threadgroup_position_in_grid.y) * TM, Lq - TM);
    const int sr = r0 + int(sg / 2) * (FI * 8);
    const int sc = c0 + int(sg % 2) * (FJ * 8);
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    float out_acc[FI][FJ][2];
    for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) { out_acc[i][j][0] = 0.0f; out_acc[i][j][1] = 0.0f; }
    for (int h = 0; h < H; ++h) {
        simdgroup_float8x8 acc[FI][FJ];
        for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
        const device float* qh = q + (size_t)h * QH;
        for (int d = 0; d < D; d += 8) {
            simdgroup_float8x8 a[FI];
            simdgroup_float8x8 b[FJ];
            for (int i = 0; i < FI; ++i)
                simdgroup_load(a[i], qh + (size_t)(sr + i * 8) * QS + d, QS);
            for (int j = 0; j < FJ; ++j)
                simdgroup_load(b[j], k + (size_t)(sc + j * 8) * D + d, D, ulong2(0, 0), true);
            for (int i = 0; i < FI; ++i)
                for (int j = 0; j < FJ; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        for (int i = 0; i < FI; ++i) {
            const float wv = w[(size_t)(sr + i * 8 + fm) * H + h];
            for (int j = 0; j < FJ; ++j) {
                thread auto& e = acc[i][j].thread_elements();
                float s0 = e[0];
                float s1 = e[1];
                if (HAS_SCALE) { s0 = s0 * scale[0]; s1 = s1 * scale[0]; }
                float p0 = metal::max(s0, 0.0f) * wv;
                float p1 = metal::max(s1, 0.0f) * wv;
                out_acc[i][j][0] = out_acc[i][j][0] + p0;
                out_acc[i][j][1] = out_acc[i][j][1] + p1;
            }
        }
    }
    for (int i = 0; i < FI; ++i) {
        const size_t row = (size_t)(sr + i * 8 + fm);
        for (int j = 0; j < FJ; ++j) {
            const size_t col = (size_t)(sc + j * 8 + fn);
            out[row * (size_t)Lk + col] = out_acc[i][j][0];
            out[row * (size_t)Lk + col + 1] = out_acc[i][j][1];
        }
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_dsa_fused_index_scores_v1",
        input_names=["q", "k", "w", "scale"],
        output_names=["out"],
        source=_SOURCE,
        header="#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
        ensure_row_contiguous=True,
    )


# Fragments per simdgroup (8x8 each): 4 x 2 -> a 64 x 32 tile per threadgroup of 4 simdgroups. Measured (max2, 16 heads
# x 128, 2048 queries): 2x2 1.6-2.0x of stock, 4x2 2.1-2.5x, 2x4 2.0-2.3x, 4x4 0.25x (register spill).
FI, FJ = 4, 2


def _rows_frags(L: int) -> int:
    """Fragments (8 rows each) per simdgroup row band: 64-row tiles, except 16-row tiles for calls under 32 rows (a warm
    turn's short tail) so they pad to 16 rows instead of 64 — 1.3-2.8x faster per call at 16k-64k keys (M5 Max). 32-row
    tiles measured 4-9% SLOWER than 64-row tiles at 40-63 rows, so there is no middle size. Every output element is the
    same per-element arithmetic whatever the tile height (tests pin rows bit-identical across heights)."""
    return 4 if L >= 32 else 1


def _launch(q, k, w, *, H, L, D, Lk, qs_layout, k_axis, scale, fi=FI):
    TM, TN = 2 * fi * 8, 2 * FJ * 8
    sc = mx.array([float(scale if scale is not None else 1.0)], dtype=mx.float32)
    return _kernel()(
        inputs=[q, k, w, sc],
        template=[("H", int(H)), ("D", int(D)), ("FI", int(fi)), ("FJ", FJ), ("TM", TM), ("TN", TN),
                  ("QS_LAYOUT", int(qs_layout)), ("K_AXIS", int(k_axis)), ("HAS_SCALE", int(scale is not None))],
        grid=(-(-Lk // TN) * 128, -(-L // TM), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, L, Lk)],
        output_dtypes=[mx.float32],
    )[0]


def _admit(L, Lk, D, arrays) -> bool:
    TN = 2 * FJ * 8
    return not (D % 8 or L < 2 or Lk < TN or any(a.dtype != mx.float32 for a in arrays)
                or max(L, 64) * Lk >= 2**31)


def dsa_index_scores(q: mx.array, k: mx.array, w: mx.array, *, enabled: bool = True):
    """Naive layout. q (1, H, L, D) fp32, k (1, 1, Lk, D) fp32, w (1, L, H) fp32 -> (1, L, Lk) fp32 =
    sum_h w * relu(q . k); None when not admitted.

    Admitted: batch 1, one shared key head, fp32 operands, D a multiple of 8, 2 <= L, Lk >= 32 (the last tile
    overlaps the previous one instead of reading past the end; overlapping cells are computed identically; L below
    one tile is zero-padded). Decode (L == 1) returns None."""
    if not enabled or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        return None
    B, H, L, D = q.shape
    if B != 1 or k.shape[0] != 1 or k.shape[1] != 1 or k.shape[3] != D or tuple(w.shape) != (1, L, H):
        return None
    Lk = int(k.shape[2])
    if not _admit(L, Lk, D, (q, k, w)):
        return None                         # decode (L == 1) keeps the stock GEMV path
    fi = _rows_frags(L)
    TM = 2 * fi * 8
    if L < TM:
        # A short prefill tail (a warm turn's few new tokens) takes the same arithmetic as the cold chunk did:
        # zero-pad the queries to one tile and keep the first L rows (each row is computed independently).
        out = _launch(mx.pad(q, [(0, 0), (0, 0), (0, TM - L), (0, 0)]), k, mx.pad(w, [(0, 0), (0, TM - L), (0, 0)]),
                      H=H, L=TM, D=D, Lk=Lk, qs_layout=False, k_axis=2, scale=None, fi=fi)
        return out[:, :L]
    return _launch(q, k, w, H=H, L=L, D=D, Lk=Lk, qs_layout=False, k_axis=2, scale=None, fi=fi)


def dsa_pool_index_scores(q: mx.array, pool_keys: mx.array, head_w: mx.array, scale: float, *, enabled: bool = True,
                          bf16_operands: bool = False):
    """GLM-5.3 layout. q (1, S, H, D) fp32, pool_keys (1, P, D) fp32, head_w (1, S, H) fp32 -> (1, S, P) fp32 =
    sum_h head_w * relu((q . pool_key) * scale) — the indexer's pool scores before the visibility mask; None when not
    admitted (same admission and tail padding as dsa_index_scores).

    bf16_operands=True: the caller states that q and pool_keys hold bf16 values (bf16 arrays, or fp32 arrays upcast
    from bf16 — DeepSeek-V4.1 scores in bf16). The scores then run on the M5 matrix units (dsa_index_scores_mpp.py,
    ~4x faster at 100k keys) when Metal 4 tensor ops are present and VMLX_DSA_MPP_INDEX is not 0; otherwise bf16
    arrays are upcast (exactly) to fp32 for this kernel. Callers that leave it False (GLM-5.3, true fp32) keep this
    kernel bit for bit."""
    if not enabled or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if bf16_operands:
        from .dsa_index_scores_mpp import dsa_pool_index_scores_mpp
        bf = lambda a: a if a.dtype == mx.bfloat16 else a.astype(mx.bfloat16)  # noqa: E731 - exact for bf16 values
        out = dsa_pool_index_scores_mpp(bf(q), bf(pool_keys), head_w, scale)
        if out is not None:
            return out
        f32 = lambda a: a if a.dtype == mx.float32 else a.astype(mx.float32)  # noqa: E731
        q, pool_keys, head_w = f32(q), f32(pool_keys), f32(head_w)
    if q.ndim != 4 or pool_keys.ndim != 3 or head_w.ndim != 3:
        return None
    B, S, H, D = q.shape
    if B != 1 or pool_keys.shape[0] != 1 or pool_keys.shape[2] != D or tuple(head_w.shape) != (1, S, H):
        return None
    P = int(pool_keys.shape[1])
    if not _admit(S, P, D, (q, pool_keys, head_w)):
        return None
    fi = _rows_frags(S)
    TM = 2 * fi * 8
    if S < TM:
        out = _launch(mx.pad(q, [(0, 0), (0, TM - S), (0, 0), (0, 0)]), pool_keys,
                      mx.pad(head_w, [(0, 0), (0, TM - S), (0, 0)]),
                      H=H, L=TM, D=D, Lk=P, qs_layout=True, k_axis=1, scale=scale, fi=fi)
        return out[:, :S]
    return _launch(q, pool_keys, head_w, H=H, L=S, D=D, Lk=P, qs_layout=True, k_axis=1, scale=scale, fi=fi)

"""Per-query sparse DSA attention kernel for Naive N0.5 prefill (vMLX Python, INTERNAL, 2026-10-09).

Real indexer selections are scattered (94-100% of 64-key blocks touched), so block-sparse SDPA cannot skip anything,
and the dense masked SDPA costs O(L x Lk). This kernel costs O(L x top_k): one threadgroup per (query, KV head) covers
the G = H / KV query heads that share that KV head (16 for N0.5). 4 simdgroups x 4 heads each. Per tile of 32 selected
keys: lane j owns key j (16 dot products of width DK against Q held in threadgroup memory, fp32), a per-head online
softmax across the simdgroup, then V accumulation where lane l owns output dims [l*DV/32, (l+1)*DV/32).
Keys beyond the query's causal position (argpartition over -inf rows) are masked. No sink (DSA layers have none)."""
from __future__ import annotations

import functools

import mlx.core as mx

_SRC = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint qi = tg / KV, kvh = tg % KV;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int L = meta[0], Lk = meta[1], qpos = meta[2] + int(qi);
    const float scale = sc[0];
    threadgroup float qs[G * DK];
    for (uint i = tid; i < G * DK; i += 32 * NSG) {
        uint hh = i / DK, d = i % DK;
        qs[i] = float(q[(size_t(kvh * G + hh) * L + qi) * DK + d]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m[HPS], l[HPS], pv[HPS], acc[HPS][DV / 32];
    for (int i = 0; i < HPS; ++i) { m[i] = -INFINITY; l[i] = 0.0f; for (int c = 0; c < DV / 32; ++c) acc[i][c] = 0.0f; }
    const device int* srow = sel + size_t(qi) * TOPK;
    for (int t0 = 0; t0 < TOPK; t0 += 32) {
        const int j = srow[t0 + lane];
        const bool valid = (j >= 0) && (j <= qpos) && (j < Lk);
        float s[HPS];
        for (int i = 0; i < HPS; ++i) s[i] = 0.0f;
        if (valid) {
            const device bfloat16_t* kr = k + (size_t(kvh) * Lk + j) * DK;
            for (int d = 0; d < DK; d += 4) {
                float4 kv4 = float4(float(kr[d]), float(kr[d + 1]), float(kr[d + 2]), float(kr[d + 3]));
                for (int i = 0; i < HPS; ++i) {
                    threadgroup const float* qh = qs + (sg * HPS + i) * DK + d;
                    s[i] += qh[0] * kv4.x + qh[1] * kv4.y + qh[2] * kv4.z + qh[3] * kv4.w;
                }
            }
        }
        for (int i = 0; i < HPS; ++i) {
            const float si = valid ? s[i] : -INFINITY;
            const float mn = metal::max(m[i], simd_max(si));
            if (mn == -INFINITY) { pv[i] = 0.0f; continue; }
            const float p = valid ? metal::exp(si - mn) : 0.0f;
            const float corr = (m[i] == -INFINITY) ? 0.0f : metal::exp(m[i] - mn);
            l[i] = l[i] * corr + simd_sum(p);
            for (int c = 0; c < DV / 32; ++c) acc[i][c] *= corr;
            m[i] = mn; pv[i] = p;
        }
        for (int jj = 0; jj < 32; ++jj) {
            const int jidx = simd_shuffle(j, ushort(jj));
            if (!((jidx >= 0) && (jidx <= qpos) && (jidx < Lk))) continue;          // uniform across the simdgroup
            const device bfloat16_t* vr = v + (size_t(kvh) * Lk + jidx) * DV + lane * (DV / 32);
            float vv[DV / 32];
            for (int c = 0; c < DV / 32; ++c) vv[c] = float(vr[c]);
            for (int i = 0; i < HPS; ++i) {
                const float pp = simd_shuffle(pv[i], ushort(jj));
                for (int c = 0; c < DV / 32; ++c) acc[i][c] += pp * vv[c];
            }
        }
    }
    for (int i = 0; i < HPS; ++i) {
        const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        device bfloat16_t* orow = o + (size_t(kvh * G + sg * HPS + i) * L + qi) * DV + lane * (DV / 32);
        for (int c = 0; c < DV / 32; ++c) orow[c] = bfloat16_t(acc[i][c] * inv);
    }
"""


@functools.lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(name="n05_dsa_sparse_attn", input_names=["q", "k", "v", "sel", "meta", "sc"],
                                output_names=["o"], source=_SRC)


def dsa_sparse_attention_kernel(q, k, v, sel, scale: float, q_pos0: int):
    """q (1, H, L, DK) bf16 (RoPE applied); k (1, KV, Lk, DK) bf16; v (1, KV, Lk, DV) bf16; sel (1, L, TOPK) indices.
    Returns (1, H, L, DV) bf16."""
    _, H, L, DK = q.shape; KV, Lk, DV = k.shape[1], k.shape[2], v.shape[3]; TOPK = sel.shape[-1]
    G = H // KV; NSG = 4; HPS = G // NSG
    assert H % KV == 0 and G % NSG == 0 and DV % 32 == 0 and DK % 4 == 0 and TOPK % 32 == 0
    o = _kernel()(inputs=[q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16), sel.reshape(L, TOPK).astype(mx.int32),
                          mx.array([L, Lk, q_pos0], dtype=mx.int32), mx.array([scale], dtype=mx.float32)],
                  template=[("KV", KV), ("G", G), ("NSG", NSG), ("HPS", HPS), ("DK", DK), ("DV", DV), ("TOPK", TOPK)],
                  grid=(L * KV * 32 * NSG, 1, 1), threadgroup=(32 * NSG, 1, 1),
                  output_shapes=[(H, L, DV)], output_dtypes=[mx.bfloat16])[0]
    return o[None]

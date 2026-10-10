"""GLM-5.3 absorbed-MLA decode attention, dense or DSA-indexed, in two dispatches (2026-10-10).

Decode (one query token) attends 64 absorbed query heads (rank 512) to the shared latent cache. The stock path casts
the latent (dense) or a gathered copy of the selected rows (sparse DSA, 2048 rows) to fp32 every step and runs SDPA
with a materialized mask: fp32 accumulation is REQUIRED (bf16 accumulation over the 512-dim contraction drifts into
repetition loops), the fp32 copies are not. Here bf16 latent rows are read in place, by index, and accumulated in fp32:

  partial: threadgroup = 8 heads (one simdgroup each, 16 rank dims per lane) x one split of SPLIT rows; online softmax
           (m, l, acc[512]) per (split, head). Rows are latent[indices[r]] (dense: r), masked by valid[r] and
           indices[r] <= q_pos, exactly the stock mask.
  combine: per head, rescale the split partials by exp(m_s - M), sum, divide, store in the query dtype.
Same formula as mx.fast.scaled_dot_product_attention in fp32; summation order differs.
VMLX_GLM5_MLA_DECODE_ATTN=0 restores the stock SDPA path.
"""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

SPLIT = 128
_OBSERVED = 0


def _split() -> int:
    return int(os.environ.get("VMLX_GLM5_MLA_SPLIT", str(SPLIT)))


def mla_decode_attn_requested() -> bool:
    return os.environ.get("VMLX_GLM5_MLA_DECODE_ATTN", "1").strip().lower() not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=16)
def _partial(rank: int, indexed: bool, split_rows: int = SPLIT):
    assert rank % 32 == 0
    per = rank // 32
    row = "(uint)indices[r]" if indexed else "r"
    ok = "(valid[r] && (int)indices[r] <= q_pos[0])" if indexed else "true"
    src = f"""
    const uint split = threadgroup_position_in_grid.x;
    const uint hg = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint h = hg * 8u + sg;
    const uint n_heads = (uint)meta[0], n_rows = (uint)meta[1], n_splits = (uint)meta[2];
    if (h >= n_heads) return;
    const float scale = sc[0];
    float qv[{per}], acc[{per}];
    for (uint j = 0; j < {per}u; ++j) {{ qv[j] = (float)q[(size_t)h * {rank}u + lane * {per}u + j] * scale; acc[j] = 0.0f; }}
    float m = -INFINITY, l = 0.0f;
    const uint r0 = split * {split_rows}u, r1 = min(r0 + {split_rows}u, n_rows);
    for (uint r = r0; r < r1; ++r) {{
        if (!({ok})) continue;
        device const T* lr = latent + (size_t)({row}) * {rank}u + lane * {per}u;
        float kv[{per}];
        float d = 0.0f;
        for (uint j = 0; j < {per}u; ++j) {{ kv[j] = (float)lr[j]; d += qv[j] * kv[j]; }}
        d = simd_sum(d);
        float mn = max(m, d);
        float c = metal::exp(m - mn), p = metal::exp(d - mn);
        l = l * c + p;
        for (uint j = 0; j < {per}u; ++j) acc[j] = acc[j] * c + p * kv[j];
        m = mn;
    }}
    const size_t o = (size_t)split * n_heads + h;
    for (uint j = 0; j < {per}u; ++j) pacc[o * {rank}u + lane * {per}u + j] = acc[j];
    if (lane == 0u) {{ pm[o] = m; pl[o] = l; }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_mla_decode_partial_r{rank}_s{split_rows}_{'idx' if indexed else 'dense'}",
        input_names=["q", "latent", "sc", "meta"] + (["indices", "valid", "q_pos"] if indexed else []),
        output_names=["pm", "pl", "pacc"], header="#include <metal_stdlib>\nusing namespace metal;\n", source=src)


@lru_cache(maxsize=4)
def _combine(rank: int):
    src = f"""
    const uint h = threadgroup_position_in_grid.x;
    const uint d = thread_index_in_threadgroup;
    const uint n_heads = (uint)meta[0], n_splits = (uint)meta[2];
    float M = -INFINITY;
    for (uint s = 0; s < n_splits; ++s) M = max(M, pm[(size_t)s * n_heads + h]);
    float L = 0.0f, a = 0.0f;
    for (uint s = 0; s < n_splits; ++s) {{
        const size_t o = (size_t)s * n_heads + h;
        float w = (pm[o] == -INFINITY) ? 0.0f : metal::exp(pm[o] - M);
        L += pl[o] * w;
        a += pacc[o * {rank}u + d] * w;
    }}
    out[(size_t)h * {rank}u + d] = (T)(a / L);
"""
    return mx.fast.metal_kernel(name=f"vmlx_glm5_mla_decode_combine_r{rank}", input_names=["pm", "pl", "pacc", "meta"],
                                output_names=["out"], header="#include <metal_stdlib>\nusing namespace metal;\n", source=src)


def glm5_mla_decode_attn(q_eff, latent, scale: float, indices=None, valid=None, q_pos: int | None = None):
    """q_eff (1, H, 1, R), latent (1, 1, N, R) [both bf16/fp16]; optional DSA selection indices/valid (1, 1, W).
    Returns attended (1, H, 1, R) in q_eff.dtype, or None when the shapes are not the decode geometry."""
    if q_eff.ndim != 4 or q_eff.shape[0] != 1 or q_eff.shape[2] != 1 or latent.ndim != 4 or latent.shape[:2] != (1, 1):
        return None
    H, R = int(q_eff.shape[1]), int(q_eff.shape[3])
    if latent.shape[3] != R or R % 32 or R > 1024 or q_eff.dtype not in (mx.bfloat16, mx.float16) or latent.dtype != q_eff.dtype:
        return None
    indexed = indices is not None
    n = int(indices.shape[-1]) if indexed else int(latent.shape[2])
    if n == 0:
        return None
    sr = _split()
    splits = (n + sr - 1) // sr
    meta = mx.array([H, n, splits], dtype=mx.int32)
    inputs = [q_eff.reshape(H, R), latent.reshape(-1, R), mx.array([scale], mx.float32), meta]
    if indexed:
        if valid is None or q_pos is None or indices.size != n:
            return None
        inputs += [indices.reshape(n).astype(mx.int32), valid.reshape(n).astype(mx.bool_), mx.array([int(q_pos)], mx.int32)]
    pm, pl, pacc = _partial(R, indexed, sr)(
        inputs=inputs, template=[("T", q_eff.dtype)], grid=(splits * 256, (H + 7) // 8, 1), threadgroup=(256, 1, 1),
        output_shapes=[(splits, H), (splits, H), (splits, H, R)], output_dtypes=[mx.float32] * 3)
    out = _combine(R)(inputs=[pm, pl, pacc, meta], template=[("T", q_eff.dtype)], grid=(H * R, 1, 1),
                      threadgroup=(R, 1, 1), output_shapes=[(H, R)], output_dtypes=[q_eff.dtype])[0]
    global _OBSERVED
    _OBSERVED += 1
    return out.reshape(1, H, 1, R)


__all__ = ["glm5_mla_decode_attn", "mla_decode_attn_requested"]

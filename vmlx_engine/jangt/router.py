"""Fused sigmoid-noaux_tc router tail for Naive N0.5 (INTERNAL): logits (T, 256) f32 -> idx (T, 8) int32, w (T, 8) f32.
One threadgroup per token, one thread per expert: sigmoid, +bias for the CHOICE only, 8 rounds of threadgroup argmax
(ties -> lower expert id), gather unbiased scores, normalize, scale. Replaces ~9 MLX launches after the gate matmul."""
import mlx.core as mx

_SRC = r"""
    uint t = threadgroup_position_in_grid.x;
    uint e = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    // one slot per simdgroup: NE <= 1024 (was [8]: NE > 256 wrote out of bounds, GLM-5.3 has 288)
    threadgroup float bv[32]; threadgroup int bi[32]; threadgroup int win[TOPK]; threadgroup float wsc[TOPK];
    float s = 1.0f / (1.0f + metal::exp(-logits[t * NE + e]));
    float c = s + bias[e];
    for (int r = 0; r < TOPK; ++r) {
        float v = c; int i = (int)e;
        for (ushort o = 16; o > 0; o >>= 1) {
            float v2 = simd_shuffle_down(v, o); int i2 = simd_shuffle_down(i, o);
            if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float bb = bv[0]; int ib = bi[0];
            for (int g = 1; g < NE / 32; ++g) if (bv[g] > bb || (bv[g] == bb && bi[g] < ib)) { bb = bv[g]; ib = bi[g]; }
            win[r] = ib;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)e == win[r]) { wsc[r] = s; c = -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e < TOPK) {
        float tot = 0.0f;
        for (int r = 0; r < TOPK; ++r) tot += wsc[r];
        idx[t * TOPK + e] = win[e];
        w[t * TOPK + e] = (NORM ? wsc[e] / (tot + 1e-20f) : wsc[e]) * scaling[0];
    }
"""
_K = mx.fast.metal_kernel(name="n05_router_tail", input_names=["logits", "bias", "scaling"], output_names=["idx", "w"],
                          source=_SRC)


def router_tail(logits, bias, k: int, norm: bool, scaling: float):
    T, NE = logits.shape
    assert NE % 32 == 0 and NE <= 1024 and k <= 32
    idx, w = _K(inputs=[logits.astype(mx.float32), bias.astype(mx.float32), mx.array([scaling], mx.float32)],
                template=[("NE", NE), ("TOPK", k), ("NORM", int(norm))], grid=(T * NE, 1, 1), threadgroup=(NE, 1, 1),
                output_shapes=[(T, k), (T, k)], output_dtypes=[mx.int32, mx.float32])
    return idx, w




# Router logits for decode / short verify slabs (2026-10-10): x (T, D) bf16 . W (E, D) bf16 -> (T, E) fp32.
# One simdgroup per (token, expert) row, each lane accumulates D/32 products in fp32 from coalesced 4-element loads,
# simd_sum. Replaces x.astype(f32) @ W.astype(f32).T, which materialized a fp32 copy of the router weight per call
# (GLM: 288 x 4096 per MoE layer, 42 layers). Summation order differs from MLX's fp32 GEMV (~1e-6 relative).
_LOGITS_SRC = r"""
    uint row = thread_position_in_grid.x / 32u;
    uint lane = thread_index_in_simdgroup;
    if (row >= T_ROWS * NE) return;
    uint t = row / NE, e = row % NE;
    device const T* xr = x + (size_t)t * D;
    device const T* wr = w + (size_t)e * D;
    float acc = 0.0f;
    for (uint k = lane * 4u; k < D; k += 128u) {
        acc += (float)xr[k] * (float)wr[k] + (float)xr[k + 1u] * (float)wr[k + 1u]
             + (float)xr[k + 2u] * (float)wr[k + 2u] + (float)xr[k + 3u] * (float)wr[k + 3u];
    }
    acc = simd_sum(acc);
    if (lane == 0u) logits[(size_t)t * NE + e] = acc;
"""
_KL = mx.fast.metal_kernel(name="jangt_router_logits", input_names=["x", "w"], output_names=["logits"], source=_LOGITS_SRC)


def router_logits(x, weight):
    """x (T, D), weight (E, D), same float dtype, D % 128 == 0 -> fp32 logits (T, E); None if not applicable."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or x.shape[1] % 128 or x.dtype != weight.dtype:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16) or x.shape[0] > 64:
        return None
    T, D = x.shape; E = weight.shape[0]
    return _KL(inputs=[x, weight], template=[("T", x.dtype), ("T_ROWS", T), ("NE", E), ("D", D)],
               grid=(32 * T * E, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(T, E)], output_dtypes=[mx.float32])[0]

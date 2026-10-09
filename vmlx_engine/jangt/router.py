"""Fused sigmoid-noaux_tc router tail for Naive N0.5 (INTERNAL): logits (T, 256) f32 -> idx (T, 8) int32, w (T, 8) f32.
One threadgroup per token, one thread per expert: sigmoid, +bias for the CHOICE only, 8 rounds of threadgroup argmax
(ties -> lower expert id), gather unbiased scores, normalize, scale. Replaces ~9 MLX launches after the gate matmul."""
import mlx.core as mx

_SRC = r"""
    uint t = threadgroup_position_in_grid.x;
    uint e = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float bv[8]; threadgroup int bi[8]; threadgroup int win[TOPK]; threadgroup float wsc[TOPK];
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



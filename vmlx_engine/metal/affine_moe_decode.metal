//
// affine_moe_decode.metal
// Fused affine-dequant + matvec kernels for MoE decode at batch=1.
//
// Problem: MLX's gather_qmm achieves ~2% of peak bandwidth at batch=1/k=6
// because each expert matvec is too small to saturate the GPU. These kernels
// fuse gather + dequant + dot-product into a single pass with no
// intermediate materialization.
//
// Architecture: two-pass split reduction.
//   Pass 1 (these kernels): each thread computes a partial dot-product over
//     a chunk of groups. Grid = (k, out_dim, n_chunks). Output is fp32
//     partial sums of shape [k, out_dim, n_chunks].
//   Pass 2 (host-side mx.sum): reduce partial sums across chunks to [k, out_dim].
//
//   Splitting the group loop across chunks increases GPU occupancy by
//   launching more threadgroups. n_chunks=2 with threadgroup=(1,128,1)
//   gives the best throughput on M4 Max (17.1 tok/s vs 4.9 baseline).
//
// Data layout (matches MLX affine quantization):
//   x:          [hidden]           float16   (single decode token)
//   weights:    [E, out, packed]   uint32    (packed N-bit codes)
//   scales:     [E, out, n_groups] float16   (per-group scale)
//   biases:     [E, out, n_groups] float16   (per-group bias)
//   expert_ids: [k]                uint16    (selected expert indices)
//   partial:    [k, out, n_chunks] float32   (partial sums per chunk)
//
// Variants:
//   b2_g64 — 2-bit codes, group_size=64 (gate_proj, up_proj)
//   b3_g64 — 3-bit codes, group_size=64 (gate_proj on some layers)
//   b2_g32 — 2-bit codes, group_size=32 (down_proj)
//
// Kernel parameters (via mx.fast.metal_kernel):
//   inputs:  x, weights, scales, biases, expert_ids,
//            hidden, out_dim, n_groups, k, n_chunks
//   outputs: partial [k, out_dim, n_chunks] float32
//   grid:    (k, out_dim, n_chunks)
//   threadgroup: (1, 128, 1)  — tuned for M4 Max
//

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// b2_g64: 2-bit codes, group_size=64
// Packing: 64 elements × 2 bits = 128 bits = 4 uint32 per group.
// Each uint32 holds 16 × 2-bit codes.
// ---------------------------------------------------------------------------
kernel void affine_moe_decode_b2_g64(
    device const half*   x          [[buffer(0)]],
    device const uint*   weights    [[buffer(1)]],
    device const half*   scales     [[buffer(2)]],
    device const half*   biases     [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint&       hidden     [[buffer(5)]],
    constant uint&       out_dim    [[buffer(6)]],
    constant uint&       n_groups   [[buffer(7)]],
    constant uint&       k          [[buffer(8)]],
    constant uint&       n_chunks   [[buffer(9)]],
    device float*        partial    [[buffer(10)]],
    uint3 tpig [[thread_position_in_grid]]
) {
    uint expert_local = tpig.x;
    uint out_d = tpig.y;
    uint chunk_id = tpig.z;
    if (expert_local >= k || out_d >= out_dim) return;

    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = n_groups * 4u;
    uint w_row_base = expert_id * out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * out_dim * n_groups + out_d * n_groups;

    uint groups_per_chunk = n_groups / n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;

    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 4u;
        for (uint w = 0; w < 4u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * out_dim * n_chunks + out_d * n_chunks + chunk_id] = sum;
}

// ---------------------------------------------------------------------------
// b3_g64: 3-bit codes, group_size=64
// Packing: 64 elements × 3 bits = 192 bits = 6 uint32 per group.
// Codes may span uint32 boundaries (bit_off > 29).
// ---------------------------------------------------------------------------
kernel void affine_moe_decode_b3_g64(
    device const half*   x          [[buffer(0)]],
    device const uint*   weights    [[buffer(1)]],
    device const half*   scales     [[buffer(2)]],
    device const half*   biases     [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint&       hidden     [[buffer(5)]],
    constant uint&       out_dim    [[buffer(6)]],
    constant uint&       n_groups   [[buffer(7)]],
    constant uint&       k          [[buffer(8)]],
    constant uint&       n_chunks   [[buffer(9)]],
    device float*        partial    [[buffer(10)]],
    uint3 tpig [[thread_position_in_grid]]
) {
    uint expert_local = tpig.x;
    uint out_d = tpig.y;
    uint chunk_id = tpig.z;
    if (expert_local >= k || out_d >= out_dim) return;

    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = n_groups * 6u;
    uint w_row_base = expert_id * out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * out_dim * n_groups + out_d * n_groups;

    uint groups_per_chunk = n_groups / n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;

    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 6u;
        for (uint i = 0; i < 64u; i++) {
            uint bit_start = i * 3u;
            uint uint_idx = bit_start / 32u;
            uint bit_off = bit_start % 32u;
            uint packed = weights[w_base + uint_idx];
            uint code;
            if (bit_off <= 29u) {
                code = (packed >> bit_off) & 0x7u;
            } else {
                uint lo = packed >> bit_off;
                uint hi = weights[w_base + uint_idx + 1u];
                code = (lo | (hi << (32u - bit_off))) & 0x7u;
            }
            sum += (float)x[x_base + i] * ((float)code * scale + bias);
        }
    }
    partial[expert_local * out_dim * n_chunks + out_d * n_chunks + chunk_id] = sum;
}

// ---------------------------------------------------------------------------
// b2_g32: 2-bit codes, group_size=32
// Packing: 32 elements × 2 bits = 64 bits = 2 uint32 per group.
// Used by down_proj (inter=2048, group_size=32).
// x is indexed per-expert: x_row_base = expert_local * hidden.
// ---------------------------------------------------------------------------
kernel void affine_moe_decode_b2_g32(
    device const half*   x          [[buffer(0)]],
    device const uint*   weights    [[buffer(1)]],
    device const half*   scales     [[buffer(2)]],
    device const half*   biases     [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint&       hidden     [[buffer(5)]],
    constant uint&       out_dim    [[buffer(6)]],
    constant uint&       n_groups   [[buffer(7)]],
    constant uint&       k          [[buffer(8)]],
    constant uint&       n_chunks   [[buffer(9)]],
    device float*        partial    [[buffer(10)]],
    uint3 tpig [[thread_position_in_grid]]
) {
    uint expert_local = tpig.x;
    uint out_d = tpig.y;
    uint chunk_id = tpig.z;
    if (expert_local >= k || out_d >= out_dim) return;

    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = n_groups * 2u;
    uint w_row_base = expert_id * out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * out_dim * n_groups + out_d * n_groups;
    uint x_row_base = expert_local * hidden;

    uint groups_per_chunk = n_groups / n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;

    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = x_row_base + g * 32u;
        uint w_base = w_row_base + g * 2u;
        for (uint w = 0; w < 2u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * out_dim * n_chunks + out_d * n_chunks + chunk_id] = sum;
}

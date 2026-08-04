// SPDX-License-Identifier: Apache-2.0
//
// Fused affine MoE decode kernels contributed by Andrew Hornsby (@Hornsan1)
// in jjang-ai/vmlx PR #248. The executable copies live in
// affine_moe_decode.py because mx.fast.metal_kernel accepts kernel bodies.
// Production installation is guarded to the exact DSV4 affine layout there.

#include <metal_stdlib>
using namespace metal;

kernel void affine_moe_decode_b2_g64(
    device const half* x [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    constant uint& n_groups [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    constant uint& n_chunks [[buffer(9)]],
    device float* partial [[buffer(10)]],
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

kernel void affine_moe_decode_b3_g64(
    device const half* x [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    constant uint& n_groups [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    constant uint& n_chunks [[buffer(9)]],
    device float* partial [[buffer(10)]],
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

kernel void affine_moe_decode_b2_g32(
    device const half* x [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device const ushort* expert_ids [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    constant uint& n_groups [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    constant uint& n_chunks [[buffer(9)]],
    device float* partial [[buffer(10)]],
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

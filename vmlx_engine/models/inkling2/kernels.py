"""Inference kernels that retain Inkling 2's FP32 intermediate arithmetic."""

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _short_convolution():
    return mx.fast.metal_kernel(
        name="inkling2_virtual_short_convolution",
        input_names=["state", "weight", "x", "mask", "divisor"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                   + c * x_strides[2]])) / divisor);
        """,
    )


def short_convolution(state, weight, x, mask, width):
    """Read the virtual state/input concatenation without an FP32 input copy."""
    return _short_convolution()(
        inputs=[
            state,
            weight,
            x,
            mask if mask is not None else mx.zeros((1, 1), mx.bool_),
            mx.array(math.sqrt(width + 1), mx.float32),
        ],
        template=[("T", x.dtype), ("W", width), ("HAS_MASK", mask is not None)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]



@lru_cache(maxsize=1)
def _bf16_vocabulary_gemv():
    return mx.fast.metal_kernel(
        name="inkling_head_bf16_read_fp32_dot",
        input_names=["matrix16", "inputs32"],
        output_names=["result"],
        compile_options={"math_mode": "safe"},
        source=r"""        const uint lane = thread_index_in_simdgroup;
        const uint row = (threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup) * 4;
        float accum[Vectors][4] = {};
        for (uint offset = lane * 4; offset < Hidden; offset += 128) {
            float inputs[Vectors][4];
            #pragma unroll
            for (uint v=0; v<Vectors; ++v)
                for (uint i=0; i<4; ++i)
                    inputs[v][i] = inputs32[v * Hidden + offset + i];
            #pragma unroll
            for (uint r=0; r<4; ++r) {
                const auto values = matrix16 + (row + r) * Hidden + offset;
                #pragma unroll
                for (uint i=0; i<4; ++i) {
                    const float entry = float(values[i]);
                    #pragma unroll
                    for (uint v=0; v<Vectors; ++v) {
                        accum[v][r] = fma(entry, inputs[v][i], accum[v][r]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint v=0; v<Vectors; ++v) {
            #pragma unroll
            for (uint r=0; r<4; ++r) {
                float total = accum[v][r];
                #pragma unroll
                for (uint step=16; step>0; step/=2)
                    total += simd_shuffle_down(total, step);
                if (lane == 0) result[v * Vocab + row + r] = total;
            }
        }
        """,
    )


def bf16_vocabulary_gemv(x, weight):
    """Read each BF16 weight once across the rows, with the serial FP32 dot order."""
    vocab, hidden = weight.shape
    return _bf16_vocabulary_gemv()(
        inputs=[weight, x],
        template=[("Hidden", hidden), ("Vocab", vocab), ("Vectors", x.shape[1])],
        grid=(vocab // 32 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, x.shape[1], vocab)],
        output_dtypes=[mx.float32],
    )[0]



@lru_cache(maxsize=1)
def _loop_controller_pair():
    # Match the native RMS kernel's four elements per lane and two SIMD reductions.
    return mx.fast.metal_kernel(
        name="inkling_controller_normalize_pair",
        input_names=["x", "message", "eps"],
        output_names=["pair"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        const uint slot = threadgroup_position_in_grid.y;
        const uint position = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint simdgroup = simdgroup_index_in_threadgroup;
        const uint index = 4 * thread_position_in_threadgroup.x;

        float values[4];
        float squares = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint column = index + i;
            values[i] = slot == 0
                ? float(x[position * x_strides[1] + column * x_strides[2]])
                : float(message[position * message_strides[1] + column * message_strides[2]]);
            squares = fma(values[i], values[i], squares);
        }
        const float partial = simd_sum(squares);
        threadgroup float totals[32];
        threadgroup float scale;
        if (lane == 0) totals[simdgroup] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            const float total = simd_sum(lane < 18 ? totals[lane] : 0.0f);
            if (lane == 0) scale = metal::precise::rsqrt(total / 2304.0f + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint offset = (position * 2 + slot) * 2304 + index;
        for (uint i = 0; i < 4; ++i) {
            const float normalized = values[i] * scale;
            pair[offset + i] = slot == 0 ? P(X(normalized)) : P(M(normalized));
        }
        """,
    )


def loop_controller_pair(x, message, eps, dtype):
    """Write both native RMS-normalized inputs directly to the projection row."""
    length = x.shape[1]
    return _loop_controller_pair()(
        inputs=[x, message, eps],
        template=[("X", x.dtype), ("M", message.dtype), ("P", dtype)],
        grid=(576, 2, length),
        threadgroup=(576, 1, 1),
        output_shapes=[(1, length, 4608)],
        output_dtypes=[dtype],
    )[0]


@lru_cache(maxsize=1)
def _draft_feature_mix():
    return mx.fast.metal_kernel(
        name="inkling_draft_feature_mix_rms",
        input_names=["taps", "alpha", "gamma", "eps"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        const uint position = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint group = simdgroup_index_in_threadgroup;
        const uint index = 4 * thread_position_in_threadgroup.x;

        float values[4];
        float squares = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint column = index + i;
            float value = 0.0f;
            for (uint tap = 0; tap < 4; ++tap) {
                const auto address = position*taps_strides[1] + (tap*2304+column)*taps_strides[2];
                value += float(taps[address]) * float(alpha[tap*alpha_strides[0]]);
            }
            values[i] = value;
            squares = fma(value, value, squares);
        }
        const float partial = simd_sum(squares);
        threadgroup float sums[32];
        threadgroup float scale;
        if (lane == 0) sums[group] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            const float total = simd_sum(lane < 18 ? sums[lane] : 0.0f);
            if (lane == 0) scale = metal::precise::rsqrt(total / 2304.0f + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 4; ++i) {
            const uint column = index + i;
            out[position*2304+column] = bfloat16_t(float(gamma[column*gamma_strides[0]]) * (values[i] * scale));
        }
        """,
    )


def draft_feature_mix(taps, coefficients, norm_weight, eps):
    """Retain the draft head's FP32 mix and RMS reduction in one dispatch."""
    positions = taps.shape[1]
    return _draft_feature_mix()(
        inputs=[taps, coefficients, norm_weight, eps],
        grid=(576 * positions, 1, 1),
        threadgroup=(576, 1, 1),
        output_shapes=[(1, positions, 2304)],
        output_dtypes=[mx.bfloat16],
    )[0]

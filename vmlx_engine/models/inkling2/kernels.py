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
def _joint_dynamic_convolution():
    return mx.fast.metal_kernel(
        name="inkling2_joint_dynamic_convolution",
        input_names=[
            "k", "v", "kcache", "vcache", "kstatic", "vstatic", "kdyn", "vdyn", "valid",
        ],
        output_names=["ko", "vo"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
    uint i = thread_position_in_grid.x;
    const int C = k_shape[2], L = k_shape[1];
    if (i >= k_shape[0] * C * L) return;
    const int c = i % C, t = (i / C) % L, b = i / (C * L);
    const int head = c / (C / Heads);
    const int kbase = b * k_strides[0] + c * k_strides[2];
    const int vbase = b * v_strides[0] + c * v_strides[2];
    const int kb = b * kdyn_strides[0] + t * kdyn_strides[1] + head * kdyn_strides[2];
    const int vb = b * vdyn_strides[0] + t * vdyn_strides[1] + head * vdyn_strides[2];
    float kc = 0.0f, vc = 0.0f;
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const int p = t - lag;
        const bool active = p >= 0 &&
            (!Masked || valid[b * valid_strides[0] + p * valid_strides[1]]);
        const float a = p < 0
            ? float(kcache[b * kcache_strides[0] + (p + 3) * kcache_strides[1] + c * kcache_strides[2]])
            : (active ? float(k[kbase + p * k_strides[1]]) : 0.0f);
        const float z = p < 0
            ? float(vcache[b * vcache_strides[0] + (p + 3) * vcache_strides[1] + c * vcache_strides[2]])
            : (active ? float(v[vbase + p * v_strides[1]]) : 0.0f);
        const float ks = float(kstatic[lag * kstatic_strides[0] + c * kstatic_strides[1]]) +
                         kdyn[kb + lag * kdyn_strides[3]];
        const float vs = float(vstatic[lag * vstatic_strides[0] + c * vstatic_strides[1]]) +
                         vdyn[vb + lag * vdyn_strides[3]];
        kc = kc + a * ks;
        vc = vc + z * vs;
    }
    ko[i] = T((kc + float(k[kbase + t * k_strides[1]])) / 3.0f);
    vo[i] = T((vc + float(v[vbase + t * v_strides[1]])) / 3.0f);
    """,
    )


def joint_dynamic_convolution(keys, values, states, static, dynamic, mask, heads):
    """Fuse equal-shaped K/V dynamic taps while retaining FP32 lag order."""
    return _joint_dynamic_convolution()(
        inputs=[
            keys, values, *states, *static, *dynamic,
            mask if mask is not None else mx.array([[True]]),
        ],
        template=[("T", keys.dtype), ("Heads", heads), ("Masked", mask is not None)],
        grid=(keys.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[keys.shape, values.shape],
        output_dtypes=[keys.dtype, values.dtype],
    )


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


@lru_cache(maxsize=8)
def _attention_residual_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_attention_residual_fused",
        input_names=["blocks", "partial", "query"],
        output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=r"""
        const uint bl = threadgroup_position_in_grid.y;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint Kb = blocks_shape[2];
        const uint K = Kb + 1;
        const uint col_base = tid * ELEMS;

        float q_reg[Q][ELEMS];
        #pragma unroll
        for (uint q = 0; q < Q; ++q) {
            #pragma unroll
            for (uint i = 0; i < ELEMS; ++i) {
                q_reg[q][i] = float(query[q * D + col_base + i]);
            }
        }

        threadgroup float sg_buf[32];
        threadgroup float inv_rms[16];
        threadgroup float scores[Q][16];

        const float inv_sqrt_d = metal::precise::rsqrt(float(D));
        for (uint k = 0; k < K; ++k) {
            float v[ELEMS];
            float sq = 0.0f;
            if (k < Kb) {
                const size_t base = ((size_t)bl * Kb + k) * D + col_base;
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) {
                    v[i] = float(blocks[base + i]);
                    sq = fma(v[i], v[i], sq);
                }
            } else {
                const size_t base = (size_t)bl * D + col_base;
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) {
                    v[i] = float(partial[base + i]);
                    sq = fma(v[i], v[i], sq);
                }
            }
            const float sg_sq = simd_sum(sq);
            if (lane == 0) sg_buf[sg] = sg_sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                const float tot = simd_sum(lane < NSG ? sg_buf[lane] : 0.0f);
                if (lane == 0) inv_rms[k] = metal::precise::rsqrt(tot / float(D) + 1e-6f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float ir = inv_rms[k];
            #pragma unroll
            for (uint q = 0; q < Q; ++q) {
                float dp = 0.0f;
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) {
                    dp = fma(v[i] * ir, q_reg[q][i], dp);
                }
                const float sg_dp = simd_sum(dp);
                if (lane == 0) sg_buf[sg] = sg_dp;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    const float tot = simd_sum(lane < NSG ? sg_buf[lane] : 0.0f);
                    if (lane == 0) scores[q][k] = tot * inv_sqrt_d;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (sg == 0 && lane == 0) {
            #pragma unroll
            for (uint q = 0; q < Q; ++q) {
                float mxv = scores[q][0];
                for (uint k = 1; k < K; ++k) mxv = metal::max(mxv, scores[q][k]);
                float sum_e = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    float e = metal::exp(scores[q][k] - mxv);
                    scores[q][k] = e;
                    sum_e += e;
                }
                float inv_sum = 1.0f / sum_e;
                for (uint k = 0; k < K; ++k) scores[q][k] *= inv_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc[Q][ELEMS];
        #pragma unroll
        for (uint q = 0; q < Q; ++q) {
            #pragma unroll
            for (uint i = 0; i < ELEMS; ++i) acc[q][i] = 0.0f;
        }
        for (uint k = 0; k < K; ++k) {
            const float ir = NORM_VAL ? inv_rms[k] : 1.0f;
            float v[ELEMS];
            if (k < Kb) {
                const size_t base = ((size_t)bl * Kb + k) * D + col_base;
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) v[i] = float(blocks[base + i]) * ir;
            } else {
                const size_t base = (size_t)bl * D + col_base;
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) v[i] = float(partial[base + i]) * ir;
            }
            #pragma unroll
            for (uint q = 0; q < Q; ++q) {
                const float w = scores[q][k];
                #pragma unroll
                for (uint i = 0; i < ELEMS; ++i) acc[q][i] = fma(w, v[i], acc[q][i]);
            }
        }
        #pragma unroll
        for (uint q = 0; q < Q; ++q) {
            const size_t out_base = ((size_t)bl * Q + q) * D + col_base;
            #pragma unroll
            for (uint i = 0; i < ELEMS; ++i) out[out_base + i] = acc[q][i];
        }
        """,
    )


def attention_residual_fused(blocks_buf, partial, query, normalize_values: bool):
    """Fused AttentionResidual over (blocks_buf, partial) in one Metal dispatch."""
    b, length, _, d = blocks_buf.shape
    q_mat = query.reshape(-1, d)
    num_q = q_mat.shape[0]
    tg = 576 if d % 576 == 0 else 256
    elems = d // tg
    nsg = tg // 32
    return _attention_residual_kernel()(
        inputs=[blocks_buf, partial, q_mat],
        template=[
            ("D", d),
            ("Q", num_q),
            ("ELEMS", elems),
            ("NSG", nsg),
            ("NORM_VAL", bool(normalize_values)),
        ],
        grid=(tg, b * length, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(b, length, num_q, d)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=8)
def _short_convolution_with_state_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_short_convolution_state_res",
        input_names=["state", "weight", "x", "res", "divisor"],
        output_names=["out", "nstate"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int L = x_shape[1], C = x_shape[2];
        if ((int)i >= L * C) return;
        int c = (int)(i % (uint)C);
        int t = (int)(i / (uint)C);
        float value = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int p = t + j - 3;
            float xv = p < 0
                ? float(state[(p + 3) * state_strides[1] + c * state_strides[2]])
                : float(x[p * x_strides[1] + c * x_strides[2]]);
            float w = float(weight[c * weight_strides[0] + j * weight_strides[1]]);
            value = value + xv * w;
        }
        bfloat16_t conv_bf16 = bfloat16_t(
            (value + float(x[t * x_strides[1] + c * x_strides[2]])) / divisor[0]
        );
        if (RES_MODE == 0) {
            out[i] = OUT_T(conv_bf16);
        } else if (RES_MODE == 3) {
            out[i] = OUT_T(float(conv_bf16));
        } else if (RES_MODE == 1) {
            float rv = float(res[t * res_strides[1] + c * res_strides[2]]);
            out[i] = OUT_T(float(conv_bf16) + rv);
        } else {
            float rv = float(res[t * res_strides[1] + c * res_strides[2]]);
            out[i] = OUT_T(bfloat16_t(rv + float(conv_bf16)));
        }
        if (t == L - 1) {
            #pragma unroll
            for (int s = 0; s < 3; ++s) {
                int p = L - 3 + s;
                nstate[s * C + c] = p < 0
                    ? float(state[(p + 3) * state_strides[1] + c * state_strides[2]])
                    : float(x[p * x_strides[1] + c * x_strides[2]]);
            }
        }
        """,
    )


def short_convolution_fused(state, weight, x, res=None, res_mode: int = 0):
    """Single-dispatch K=4 depthwise conv + state update + optional residual add (B=1, mask=None)."""
    length, channels = x.shape[1], x.shape[2]
    out_dtype = mx.float32 if res_mode in (1, 3) else mx.bfloat16
    res_in = res if res is not None else x
    return _short_convolution_with_state_kernel()(
        inputs=[state, weight, x, res_in, mx.array([math.sqrt(5.0)], mx.float32)],
        template=[("OUT_T", out_dtype), ("RES_MODE", int(res_mode))],
        grid=(length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, 3, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )


@lru_cache(maxsize=4)
def _joint_dynamic_conv_state_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_joint_dynamic_conv_state",
        input_names=[
            "k", "v", "kcache", "vcache", "kstatic", "vstatic", "kdyn", "vdyn",
        ],
        output_names=["ko", "vo", "knstate", "vnstate"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
    uint i = thread_position_in_grid.x;
    const int C = k_shape[2], L = k_shape[1];
    if ((int)i >= C * L) return;
    const int c = (int)(i % (uint)C), t = (int)(i / (uint)C);
    const int head = c / (C / Heads);
    const int kbase = c * k_strides[2];
    const int vbase = c * v_strides[2];
    const int kb = t * kdyn_strides[1] + head * kdyn_strides[2];
    const int vb = t * vdyn_strides[1] + head * vdyn_strides[2];
    float kc = 0.0f, vc = 0.0f;
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const int p = t - lag;
        const float a = p < 0
            ? float(kcache[(p + 3) * kcache_strides[1] + c * kcache_strides[2]])
            : float(k[kbase + p * k_strides[1]]);
        const float z = p < 0
            ? float(vcache[(p + 3) * vcache_strides[1] + c * vcache_strides[2]])
            : float(v[vbase + p * v_strides[1]]);
        const float ks = float(kstatic[lag * kstatic_strides[0] + c * kstatic_strides[1]]) +
                         kdyn[kb + lag * kdyn_strides[3]];
        const float vs = float(vstatic[lag * vstatic_strides[0] + c * vstatic_strides[1]]) +
                         vdyn[vb + lag * vdyn_strides[3]];
        kc = kc + a * ks;
        vc = vc + z * vs;
    }
    ko[i] = T((kc + float(k[kbase + t * k_strides[1]])) / 3.0f);
    vo[i] = T((vc + float(v[vbase + t * v_strides[1]])) / 3.0f);
    if (t == L - 1) {
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            int p = L - 3 + s;
            knstate[s * C + c] = p < 0
                ? float(kcache[(p + 3) * kcache_strides[1] + c * kcache_strides[2]])
                : float(k[kbase + p * k_strides[1]]);
            vnstate[s * C + c] = p < 0
                ? float(vcache[(p + 3) * vcache_strides[1] + c * vcache_strides[2]])
                : float(v[vbase + p * v_strides[1]]);
        }
    }
    """,
    )


def joint_dynamic_convolution_with_state(keys, values, states, static, dynamic, heads):
    """Fused K/V dynamic short conv + next-state output (B=1, mask=None, non-speculating)."""
    channels = keys.shape[2]
    return _joint_dynamic_conv_state_kernel()(
        inputs=[keys, values, *states, *static, *dynamic],
        template=[("T", keys.dtype), ("Heads", heads)],
        grid=(keys.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[keys.shape, values.shape, (1, 3, channels), (1, 3, channels)],
        output_dtypes=[keys.dtype, values.dtype, mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _loop_controller_norm_down_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_loop_controller_norm_down",
        input_names=["x", "message", "eps", "down_w"],
        output_names=["hidden"],
        compile_options={"math_mode": "safe"},
        source=r"""
        // 16 TGs per token, 4 output rows per TG = 64 rows; 256 threads = 8 simdgroups.
        // Each TG computes normalized [rms(x), rms(message)] into 9 KB shmem once,
        // then 2 simdgroups cooperatively dot each of the 4 rows across K=4608.
        const uint pos = threadgroup_position_in_grid.z;
        const uint row0 = threadgroup_position_in_grid.y * 4;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;

        threadgroup bfloat16_t pair_sm[4608];
        threadgroup float sg_buf[8];
        threadgroup float scales[2];

        float vx[9], vm[9];
        float sqx = 0.0f, sqm = 0.0f;
        const size_t xbase = (size_t)pos * 2304;
        #pragma unroll
        for (uint i = 0; i < 9; ++i) {
            uint c = tid + i * 256;
            vx[i] = float(x[xbase + c]);
            vm[i] = float(message[xbase + c]);
            sqx = fma(vx[i], vx[i], sqx);
            sqm = fma(vm[i], vm[i], sqm);
        }
        float s_sqx = simd_sum(sqx);
        if (lane == 0) sg_buf[sg] = s_sqx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float tot = simd_sum(lane < 8 ? sg_buf[lane] : 0.0f);
            if (lane == 0) scales[0] = metal::precise::rsqrt(tot / 2304.0f + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s_sqm = simd_sum(sqm);
        if (lane == 0) sg_buf[sg] = s_sqm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float tot = simd_sum(lane < 8 ? sg_buf[lane] : 0.0f);
            if (lane == 0) scales[1] = metal::precise::rsqrt(tot / 2304.0f + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sx = scales[0], sm = scales[1];
        #pragma unroll
        for (uint i = 0; i < 9; ++i) {
            uint c = tid + i * 256;
            pair_sm[c] = bfloat16_t(X(vx[i] * sx));
            pair_sm[2304 + c] = bfloat16_t(M(vm[i] * sm));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_r = sg >> 1;   // 0..3
        const uint half_id = sg & 1;    // 0: [0..2303], 1: [2304..4607]
        const uint r = row0 + local_r;
        const size_t wbase = (size_t)r * 4608 + half_id * 2304;
        const uint sm_base = half_id * 2304;
        float acc = 0.0f;
        for (uint k = lane * 4; k < 2304; k += 128) {
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                acc = fma(float(pair_sm[sm_base + k + i]), float(down_w[wbase + k + i]), acc);
            }
        }
        float part = simd_sum(acc);
        if (lane == 0) sg_buf[sg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < 4) {
            float total = sg_buf[lane * 2] + sg_buf[lane * 2 + 1];
            hidden[pos * 64 + row0 + lane] = float(bfloat16_t(total));
        }
        """,
    )


def loop_controller_norm_down(x, message, eps, down_weight):
    """Fused RMSNorm([x, message]) @ down_weight.T -> (1, L, 64) FP32."""
    length = x.shape[1]
    return _loop_controller_norm_down_kernel()(
        inputs=[x, message, eps, down_weight],
        template=[("X", x.dtype), ("M", message.dtype)],
        grid=(256, 16, length),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, 64)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=4)
def _loop_controller_up_blend_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_loop_controller_up_blend",
        input_names=["hidden", "packed", "x", "message"],
        output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=r"""
        const uint c = thread_position_in_grid.x;
        const uint pos = threadgroup_position_in_grid.y;
        const uint tid = thread_position_in_threadgroup.x;
        threadgroup float h_sm[64];
        if (tid < 64) h_sm[tid] = hidden[pos * 64 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c >= 2304) return;
        const device bfloat16_t* wf = packed + (size_t)c * 64;
        const device bfloat16_t* wg = packed + (size_t)(2304 + c) * 64;
        float af = 0.0f, ag = 0.0f;
        #pragma unroll
        for (uint k = 0; k < 64; k += 4) {
            float h0 = h_sm[k], h1 = h_sm[k + 1], h2 = h_sm[k + 2], h3 = h_sm[k + 3];
            af = fma(h0, float(wf[k]), af);
            af = fma(h1, float(wf[k + 1]), af);
            af = fma(h2, float(wf[k + 2]), af);
            af = fma(h3, float(wf[k + 3]), af);
            ag = fma(h0, float(wg[k]), ag);
            ag = fma(h1, float(wg[k + 1]), ag);
            ag = fma(h2, float(wg[k + 2]), ag);
            ag = fma(h3, float(wg[k + 3]), ag);
        }
        const float f = metal::tanh(af);
        const float g = metal::tanh(ag);
        const size_t idx = (size_t)pos * 2304 + c;
        out[idx] = (1.0f - f) * float(x[idx]) + g * float(message[idx]);
        """,
    )


def loop_controller_up_blend(hidden, packed, x, message):
    """Fused (hidden @ [forget; gain].T) + tanh + (1-f)*x + g*msg -> (1, L, 2304) FP32."""
    length = x.shape[1]
    return _loop_controller_up_blend_kernel()(
        inputs=[hidden, packed, x, message],
        grid=(2304, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, 2304)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=8)
def _prep_qkrv_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_prep_qkrv",
        input_names=["query", "k_in", "v_in", "rel", "q_scale", "rel_scale", "tau", "params"],
        output_names=["q_out", "k_out", "v_out", "r_out"],
        compile_options={"math_mode": "safe"},
        source=r"""
        // 1 simdgroup (32 lanes) per (b, l, slot) where slot in [0 .. 2*H + 2*Hkv):
        //   [0, H)             : q head (Dh=128, 4 elems/lane) -> l2_clip + optional tau + transpose (B,H,L,Dh)
        //   [H, H+Hkv)         : k head (Dh=128, 4 elems/lane) -> l2_clip + transpose (B,Hkv,L,Dh)
        //   [H+Hkv, H+2*Hkv)   : v head (Dh=128, 4 elems/lane) -> optional rms_norm + transpose (B,Hkv,L,Dh)
        //   [H+2*Hkv, 2*H+2*Hkv): r head (Dr=16, 1 elem/lane<16) -> l2_clip + optional tau -> (B,L,H,Dr)
        const uint lane = thread_index_in_simdgroup;
        const uint slot = thread_position_in_grid.y;
        const uint bl = threadgroup_position_in_grid.z;
        const uint L = query_shape[1];
        const uint b = bl / L, l = bl % L;
        const uint total_slots = 2 * H + 2 * HKV;
        if (slot >= total_slots) return;
        const float qk_thresh = params[0];
        const float rel_thresh = params[1];
        const float rms_eps = params[2];

        if (slot < H) {
            const uint h = slot;
            const size_t in_base = ((size_t)bl * H + h) * DH + lane * 4;
            const size_t sc_base = (size_t)h * DH + lane * 4;
            float v[4];
            float sq = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                v[i] = float(query[in_base + i]) * float(q_scale[sc_base + i]);
                sq = fma(v[i], v[i], sq);
            }
            float norm = metal::sqrt(simd_sum(sq));
            float factor = metal::min(qk_thresh / (norm + 1e-6f), 1.0f);
            float t_mul = HAS_TAU ? float(tau[l]) : 1.0f;
            const size_t out_base = (((size_t)b * H + h) * L + l) * DH + lane * 4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                T qv = T(v[i] * factor);
                q_out[out_base + i] = HAS_TAU ? T(float(qv) * t_mul) : qv;
            }
        } else if (slot < H + HKV) {
            const uint hk = slot - H;
            const size_t in_base = ((size_t)bl * HKV + hk) * DH + lane * 4;
            float v[4];
            float sq = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                v[i] = float(k_in[in_base + i]);
                sq = fma(v[i], v[i], sq);
            }
            float norm = metal::sqrt(simd_sum(sq));
            float factor = metal::min(qk_thresh / (norm + 1e-6f), 1.0f);
            const size_t out_base = (((size_t)b * HKV + hk) * L + l) * DH + lane * 4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                k_out[out_base + i] = T(v[i] * factor);
            }
        } else if (slot < H + 2 * HKV) {
            const uint hk = slot - H - HKV;
            const size_t in_base = ((size_t)bl * HKV + hk) * DH + lane * 4;
            float v[4];
            float sq = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                v[i] = float(v_in[in_base + i]);
                sq = fma(v[i], v[i], sq);
            }
            float factor = NORM_V ? metal::precise::rsqrt(simd_sum(sq) / float(DH) + rms_eps) : 1.0f;
            const size_t out_base = (((size_t)b * HKV + hk) * L + l) * DH + lane * 4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                v_out[out_base + i] = T(v[i] * factor);
            }
        } else {
            const uint h = slot - H - 2 * HKV;
            float val = 0.0f;
            if (lane < DR) {
                const size_t in_idx = ((size_t)bl * H + h) * DR + lane;
                val = float(rel[in_idx]) * float(rel_scale[(size_t)h * DR + lane]);
            }
            float norm = metal::sqrt(simd_sum(val * val));
            if (lane < DR) {
                float factor = metal::min(rel_thresh / (norm + 1e-6f), 1.0f);
                T rv = T(val * factor);
                float t_mul = HAS_TAU ? float(tau[l]) : 1.0f;
                const size_t out_idx = ((size_t)bl * H + h) * DR + lane;
                r_out[out_idx] = HAS_TAU ? T(float(rv) * t_mul) : rv;
            }
        }
        """,
    )


def prep_qkrv_fused(
    query,
    k_in,
    v_in,
    rel,
    q_scale,
    rel_scale,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    d_rel: int,
    qk_thresh: float,
    rel_thresh: float,
    norm_v: bool,
    rms_eps: float,
    tau=None,
):
    """Fuse q/k/r L2-clip, v RMSNorm, log-scaling tau, and (B,L,H,D)->(B,H,L,D) transpose."""
    b, length = query.shape[0], query.shape[1]
    total_slots = 2 * n_heads + 2 * n_kv
    grid_y = ((total_slots + 3) // 4) * 4
    tau_in = tau.reshape(-1).astype(mx.float32) if tau is not None else mx.ones((1,), mx.float32)
    params = mx.array([float(qk_thresh), float(rel_thresh), float(rms_eps)], mx.float32)
    return _prep_qkrv_kernel()(
        inputs=[query, k_in, v_in, rel, q_scale, rel_scale, tau_in, params],
        template=[
            ("T", query.dtype),
            ("H", n_heads),
            ("HKV", n_kv),
            ("DH", head_dim),
            ("DR", d_rel),
            ("NORM_V", bool(norm_v)),
            ("HAS_TAU", tau is not None),
        ],
        grid=(32, grid_y, b * length),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (b, n_heads, length, head_dim),
            (b, n_kv, length, head_dim),
            (b, n_kv, length, head_dim),
            (b, length, n_heads, d_rel),
        ],
        output_dtypes=[query.dtype, query.dtype, query.dtype, query.dtype],
    )


# ---------------------------------------------------------------------------
# Zero-Mask Tiled M5 MPP FlashAttention Prefill + Split-K FlashDecode for Inkling-2
#
# Eliminates:
#   1. The O(T*S) [1, 24, T, S+1] banded additive mask (49 GB -> 0 B at 1M!)
#   2. Full K/V cache copy to append a dummy zero sink column (0 B!)
#   3. Computing rel @ proj on prefix tiles where dist >= rel_extent (99% of tiles!)
# ---------------------------------------------------------------------------
_INKLING_MPP_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_INKLING_MPP_FA_SRC = r"""
  // TQ=32, TK=128, D=128, NSG=4 (128 threads per TG).
  // Grid: (128, ceil(LQ/32), H).
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const int LQ = int(q_shape[2]);
  const int S  = int(k_shape[2]);
  const int r0 = min(int(threadgroup_position_in_grid.y) * 32, LQ - 32);
  const int h  = int(threadgroup_position_in_grid.z);
  const int kv_h = h / GQA;
  const int q_offset = int(q_off_buf[0]);
  const float sc = scale_buf[0];

  const int k_start = (SLIDING > 0) ? max(0, q_offset + r0 - SLIDING + 1) : 0;
  const int k_end   = min(S, q_offset + r0 + 32);

  threadgroup float S_sh[32 * 128];
  threadgroup bfloat P_sh[32 * 128];
  threadgroup float rel_sh[32 * 16];
  threadgroup float m_sh[32];
  threadgroup float l_sh[32];
  threadgroup float alpha_sh[32];

  // Initialize per-query attention sinks (exact: sink logit = sinks[h, q], sink value = 0)
  if (tid < 32u) {
    m_sh[tid] = float(sinks[(size_t)h * LQ + (r0 + int(tid))]);
    l_sh[tid] = 1.0f;
  }
  // Stage rel[r0..r0+31, h, 0..15] once per TG (2 KB)
  for (uint i = tid; i < 32u * 16u; i += 128u) {
    uint qi = i >> 4;
    uint d  = i & 15u;
    rel_sh[i] = float(rel[((size_t)(r0 + int(qi)) * H + h) * 16u + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc_qk = matmul2d_descriptor(32, 128, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto desc_pv = matmul2d_descriptor(32, 128, 128, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
  matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ(
      (device bfloat*)q + ((size_t)h * LQ + r0) * 128u, dextents<int32_t, 2>(128, 32));
  auto aQ = tQ.slice<128, 32>(0, 0);

  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tP(
      P_sh, dextents<int32_t, 2>(128, 32));
  auto aP = tP.slice<128, 32>(0, 0);

  const size_t kv_head_base = (size_t)kv_h * (size_t)S * 128u;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV0(
      (device bfloat*)v + kv_head_base, dextents<int32_t, 2>(128, 128));
  auto bV0 = tV0.slice<128, 128>(0, 0);

  auto O_acc = op_pv.template get_destination_cooperative_tensor<decltype(aP), decltype(bV0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
    if (O_acc.is_valid_element(i)) O_acc[i] = 0.0f;
  }

  for (int k0 = k_start; k0 < k_end; k0 += 128) {
    const int c0 = (k0 + 128 <= S) ? k0 : (S - 128);
    const bool is_clamped = (c0 < k0) || (c0 + 128 > k_end);
    // Pure zero-bias prefix check: true when EVERY (q, k) pair in this tile has
    // dist >= REL_EXTENT and is unmasked (99% of tiles at 100K-1M context!).
    const bool pure_prefix = (!is_clamped) && (SLIDING == 0) && ((q_offset + r0) - (c0 + 127) >= REL_EXTENT);

    // 1. Q @ K^T on M5 hardware matrix units (32x128x128!)
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK(
        (device bfloat*)k + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, 128));
    auto bK = tK.slice<128, 128>(0, 0);
    auto S_coop = op_qk.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    op_qk.run(aQ, bK, S_coop);

    #pragma unroll
    for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
      if (S_coop.is_valid_element(i)) {
        const auto idx = S_coop.get_multidimensional_index(i);
        S_sh[idx[1] * 128 + idx[0]] = S_coop[i] * sc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Online softmax across 4 simdgroups (8 rows/sg, 4 cols/lane = 128 cols!)
    if (pure_prefix) {
      for (uint r = 0u; r < 8u; ++r) {
        const uint qi = sg * 8u + r;
        const uint rb = qi * 128u;
        const float s0 = S_sh[rb + lane];
        const float s1 = S_sh[rb + lane + 32u];
        const float s2 = S_sh[rb + lane + 64u];
        const float s3 = S_sh[rb + lane + 96u];
        const float rmax = simd_max(metal::max(metal::max(s0, s1), metal::max(s2, s3)));
        const float m_old = m_sh[qi];
        const float m_new = metal::max(m_old, rmax);
        const float alpha = metal::exp(m_old - m_new);
        const float p0 = metal::exp(s0 - m_new);
        const float p1 = metal::exp(s1 - m_new);
        const float p2 = metal::exp(s2 - m_new);
        const float p3 = metal::exp(s3 - m_new);
        const float l_tile = simd_sum((p0 + p1) + (p2 + p3));
        if (lane == 0u) {
          m_sh[qi] = m_new;
          l_sh[qi] = l_sh[qi] * alpha + l_tile;
          alpha_sh[qi] = alpha;
        }
        P_sh[rb + lane]       = bfloat(p0);
        P_sh[rb + lane + 32u] = bfloat(p1);
        P_sh[rb + lane + 64u] = bfloat(p2);
        P_sh[rb + lane + 96u] = bfloat(p3);
      }
    } else {
      for (uint r = 0u; r < 8u; ++r) {
        const uint qi = sg * 8u + r;
        const uint rb = qi * 128u;
        const int q_pos = q_offset + r0 + int(qi);
        const threadgroup float* rq = rel_sh + qi * 16u;
        float sv[4];
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
          const uint ki = lane + c * 32u;
          const int kp = c0 + int(ki);
          float s = S_sh[rb + ki];
          const int d = q_pos - kp;
          if (kp < k0 || kp >= k_end || d < 0 || (SLIDING > 0 && d >= SLIDING)) {
            s = -1e30f;
          } else if (d < REL_EXTENT) {
            const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)d * 16u);
            float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
            float rbias = a0.x * rq[0]  + a0.y * rq[1]  + a0.z * rq[2]  + a0.w * rq[3]
                        + a1.x * rq[4]  + a1.y * rq[5]  + a1.z * rq[6]  + a1.w * rq[7]
                        + a2.x * rq[8]  + a2.y * rq[9]  + a2.z * rq[10] + a2.w * rq[11]
                        + a3.x * rq[12] + a3.y * rq[13] + a3.z * rq[14] + a3.w * rq[15];
            s += float(bfloat(rbias));
          }
          sv[c] = s;
        }
        const float rmax = simd_max(metal::max(metal::max(sv[0], sv[1]), metal::max(sv[2], sv[3])));
        const float m_old = m_sh[qi];
        const float m_new = metal::max(m_old, rmax);
        const float alpha = metal::exp(m_old - m_new);
        float l_acc = 0.0f;
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
          float p = (sv[c] > -1e29f) ? metal::exp(sv[c] - m_new) : 0.0f;
          l_acc += p;
          P_sh[rb + lane + c * 32u] = bfloat(p);
        }
        const float l_tile = simd_sum(l_acc);
        if (lane == 0u) {
          m_sh[qi] = m_new;
          l_sh[qi] = l_sh[qi] * alpha + l_tile;
          alpha_sh[qi] = alpha;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Rescale O_acc and accumulate P @ V on M5 hardware matrix units (32x128x128!)
    #pragma unroll
    for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
      if (O_acc.is_valid_element(i)) {
        O_acc[i] *= alpha_sh[O_acc.get_multidimensional_index(i)[1]];
      }
    }
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV(
        (device bfloat*)v + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, 128));
    auto bV = tV.slice<128, 128>(0, 0);
    op_pv.run(aP, bV, O_acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write normalized BF16 output directly to out[0, h, r0..r0+31, 0..127]
  device bfloat* out_base = (device bfloat*)out + ((size_t)h * LQ + r0) * 128u;
  #pragma unroll
  for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
    if (O_acc.is_valid_element(i)) {
      const auto idx = O_acc.get_multidimensional_index(i);
      out_base[idx[1] * 128 + idx[0]] = bfloat(O_acc[i] / l_sh[idx[1]]);
    }
  }
"""

_INKLING_MPP_COMPILED: dict = {}
_INKLING_MPP_AVAIL = None


def _inkling_mpp_kernel(H: int, GQA: int, sliding: int, rel_extent: int):
    key = (H, GQA, sliding, rel_extent)
    run = _INKLING_MPP_COMPILED.get(key)
    if run is None:
        consts = "".join(
            f"  constexpr int {n} = {int(v)};\n"
            for n, v in (("H", H), ("GQA", GQA), ("SLIDING", sliding), ("REL_EXTENT", rel_extent))
        )
        run = _INKLING_MPP_COMPILED[key] = mx.fast.metal_kernel(
            name=f"inkling2_mpp_fa128_h{H}_g{GQA}_s{sliding}_r{rel_extent}",
            input_names=["q", "k", "v", "rel", "rel_proj_t", "sinks", "scale_buf", "q_off_buf"],
            output_names=["out"],
            header=_INKLING_MPP_HEADER,
            source=consts + _INKLING_MPP_FA_SRC,
            ensure_row_contiguous=True,
        )
    return run


def _inkling_mpp_available() -> bool:
    global _INKLING_MPP_AVAIL
    if _INKLING_MPP_AVAIL is None:
        try:
            from vmlx_engine.metal.dsa_index_scores_mpp import available as mpp_avail
            _INKLING_MPP_AVAIL = bool(mpp_avail())
        except Exception:
            _INKLING_MPP_AVAIL = False
    return _INKLING_MPP_AVAIL


def inkling_flash_attention_prefill(
    q, k, v, relative, rel_proj_t, sinks, *, scale: float, sliding: int, rel_extent: int
):
    """Zero-mask M5 MPP FlashAttention for Inkling-2 prefill (B=1, D=128, d_rel=16, BF16).

    Returns None when not admitted so callers fall back cleanly.
    """
    if (
        not _inkling_mpp_available()
        or q.dtype != mx.bfloat16
        or k.dtype != mx.bfloat16
        or v.dtype != mx.bfloat16
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[3] != 128
        or relative.shape[-1] != 16
    ):
        return None
    _, H, LQ, D = (int(x) for x in q.shape)
    H_kv = int(k.shape[1])
    S = int(k.shape[2])
    if LQ < 2 or S < 128 or H % H_kv != 0:
        return None
    gqa = H // H_kv
    sc = mx.array([float(scale)], dtype=mx.float32)
    q_off = mx.array([S - LQ], dtype=mx.int32)
    sinks_2d = sinks.reshape(1, H, LQ)
    if LQ < 32:
        pad = 32 - LQ
        q_in = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
        r_in = mx.pad(relative, [(0, 0), (0, pad), (0, 0), (0, 0)])
        sk_in = mx.pad(sinks_2d, [(0, 0), (0, 0), (0, pad)])
        out = _inkling_mpp_kernel(H, gqa, int(sliding), int(rel_extent))(
            inputs=[q_in, k, v, r_in, rel_proj_t, sk_in, sc, q_off],
            grid=(128, 1, H),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, H, 32, 128)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return out[:, :, :LQ, :]
    n_qy = (LQ + 31) // 32
    return _inkling_mpp_kernel(H, gqa, int(sliding), int(rel_extent))(
        inputs=[q, k, v, relative, rel_proj_t, sinks_2d, sc, q_off],
        grid=(128, n_qy, H),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, H, LQ, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------
# Zero-Mask Split-K FlashDecode for Inkling-2 (LQ == 1, S >= 2048, D == 128)
# Eliminates allocating a [1, 24, 1, S] mask on every full-attention decode step!
# ---------------------------------------------------------------------------
_INKLING_FLASH_DECODE_S1_SRC = r"""
  // 256 threads (8 simdgroups) per (split, head). Each lane owns 4 dims of D=128.
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint split = threadgroup_position_in_grid.y;
  const uint h = threadgroup_position_in_grid.z;
  const uint kv_h = h / uint(GQA);
  const uint S = uint(k_shape[2]);
  const uint n_splits = uint(params[0]);
  const float sc = scale_buf[0];

  const uint chunk = (S + n_splits - 1u) / n_splits;
  const uint s0 = min(split * chunk, S);
  const uint s1 = min(s0 + chunk, S);

  float4 qv = float4(*(const device vec<bfloat, 4>*)(q + (size_t)h * 128u + lane * 4u));
  threadgroup float rel_sh[16];
  if (tid < 16u) rel_sh[tid] = float(rel[(size_t)h * 16u + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t kv_base = (size_t)kv_h * (size_t)S * 128u;
  float m = -1e30f;
  float l = 0.0f;
  float4 o = float4(0.0f);

  for (uint j = s0 + sg; j < s1; j += 8u) {
    float4 kv = float4(*(const device vec<bfloat, 4>*)(k + kv_base + (size_t)j * 128u + lane * 4u));
    float4 vv = float4(*(const device vec<bfloat, 4>*)(v + kv_base + (size_t)j * 128u + lane * 4u));
    float score = simd_sum(dot(qv, kv)) * sc;
    const uint dist = (S - 1u) - j;
    if (dist < uint(REL_EXTENT)) {
      const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)dist * 16u);
      float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
      float rb = a0.x * rel_sh[0]  + a0.y * rel_sh[1]  + a0.z * rel_sh[2]  + a0.w * rel_sh[3]
               + a1.x * rel_sh[4]  + a1.y * rel_sh[5]  + a1.z * rel_sh[6]  + a1.w * rel_sh[7]
               + a2.x * rel_sh[8]  + a2.y * rel_sh[9]  + a2.z * rel_sh[10] + a2.w * rel_sh[11]
               + a3.x * rel_sh[12] + a3.y * rel_sh[13] + a3.z * rel_sh[14] + a3.w * rel_sh[15];
      score += float(bfloat(rb));
    }
    float m_new = metal::max(m, score);
    float alpha = (l > 0.0f) ? metal::exp(m - m_new) : 0.0f;
    float p = metal::exp(score - m_new);
    o = o * alpha + vv * p;
    l = l * alpha + p;
    m = m_new;
  }

  threadgroup float sg_m[8];
  threadgroup float sg_l[8];
  threadgroup float sg_o[8 * 128];
  if (lane == 0u) {
    sg_m[sg] = m;
    sg_l[sg] = l;
  }
  *(threadgroup float4*)(sg_o + sg * 128u + lane * 4u) = o;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float gm = sg_m[0];
    for (uint i = 1u; i < 8u; ++i) gm = metal::max(gm, sg_m[i]);
    float gl = 0.0f;
    float4 go = float4(0.0f);
    for (uint i = 0u; i < 8u; ++i) {
      if (sg_l[i] > 0.0f) {
        float w = metal::exp(sg_m[i] - gm);
        gl += sg_l[i] * w;
        go += *(const threadgroup float4*)(sg_o + i * 128u + lane * 4u) * w;
      }
    }
    const size_t p_idx = (size_t)h * n_splits + split;
    if (lane == 0u) {
      part_m[p_idx] = gm;
      part_l[p_idx] = gl;
    }
    *(device float4*)(part_o + p_idx * 128u + lane * 4u) = go;
  }
"""

_INKLING_FLASH_DECODE_S2_SRC = r"""
  // Merge NSPLITS partials + attention sink for each head (1 simdgroup per head)
  const uint lane = thread_index_in_simdgroup;
  const uint h = threadgroup_position_in_grid.y;
  const uint n_splits = uint(part_m_shape[1]);

  const device float* pm = part_m + (size_t)h * n_splits;
  const device float* pl = part_l + (size_t)h * n_splits;
  const device float* po = part_o + (size_t)h * n_splits * 128u;

  float gm = float(sinks[h]);
  for (uint s = 0u; s < n_splits; ++s) {
    if (pl[s] > 0.0f) gm = metal::max(gm, pm[s]);
  }
  float gl = metal::exp(float(sinks[h]) - gm);
  float4 go = float4(0.0f);
  for (uint s = 0u; s < n_splits; ++s) {
    if (pl[s] > 0.0f) {
      float w = metal::exp(pm[s] - gm);
      gl += pl[s] * w;
      go += *(const device float4*)(po + (size_t)s * 128u + lane * 4u) * w;
    }
  }
  float inv_l = 1.0f / gl;
  device vec<bfloat, 4>* dst = (device vec<bfloat, 4>*)(out + (size_t)h * 128u + lane * 4u);
  *dst = vec<bfloat, 4>(bfloat(go.x * inv_l), bfloat(go.y * inv_l), bfloat(go.z * inv_l), bfloat(go.w * inv_l));
"""


@lru_cache(maxsize=8)
def _inkling_flash_decode_s1(gqa: int, rel_extent: int):
    return mx.fast.metal_kernel(
        name=f"inkling2_flash_decode_s1_g{gqa}_r{rel_extent}",
        input_names=["q", "k", "v", "rel", "rel_proj_t", "params", "scale_buf"],
        output_names=["part_m", "part_l", "part_o"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_FLASH_DECODE_S1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=2)
def _inkling_flash_decode_s2():
    return mx.fast.metal_kernel(
        name="inkling2_flash_decode_s2",
        input_names=["part_m", "part_l", "part_o", "sinks"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_FLASH_DECODE_S2_SRC,
        ensure_row_contiguous=True,
    )


def inkling_flash_decode(q, k, v, relative, rel_proj_t, sinks, *, scale: float, rel_extent: int):
    """Zero-mask Split-K FlashDecode for Inkling-2 full-attention single-token decode (S >= 2048)."""
    if (
        not mx.metal.is_available()
        or q.dtype != mx.bfloat16
        or k.dtype != mx.bfloat16
        or v.dtype != mx.bfloat16
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[2] != 1
        or q.shape[3] != 128
        or relative.shape[-1] != 16
    ):
        return None
    H = int(q.shape[1])
    H_kv = int(k.shape[1])
    S = int(k.shape[2])
    if S < 2048 or H % H_kv != 0:
        return None
    gqa = H // H_kv
    n_splits = min(64, max(8, (S + 1023) // 1024))
    pm, pl, po = _inkling_flash_decode_s1(gqa, int(rel_extent))(
        inputs=[
            q,
            k,
            v,
            relative,
            rel_proj_t,
            mx.array([n_splits], dtype=mx.uint32),
            mx.array([float(scale)], dtype=mx.float32),
        ],
        template=[("GQA", gqa), ("REL_EXTENT", int(rel_extent))],
        grid=(256, n_splits, H),
        threadgroup=(256, 1, 1),
        output_shapes=[(H, n_splits), (H, n_splits), (H, n_splits, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _inkling_flash_decode_s2()(
        inputs=[pm, pl, po, sinks.reshape(H)],
        grid=(32, H, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, H, 1, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out


# ---------------------------------------------------------------------------
# Multi-Vector Affine 8-bit g=64 GEMV (V = 1..8 rows, reads weights ONCE!)
# Bit-identical per-row reduction for V=1 and V=2..8; optional fused softcap
# and unpadded_vocab_size slice for lm_head (target verify + DFlash2 drafter).
# ---------------------------------------------------------------------------
_AFFINE8_G64_MULTI_GEMV_SRC = r"""
  // 4 simdgroups (128 threads), 2 output rows per simdgroup = 8 output rows/TG.
  // Each lane handles 8 consecutive K elements (2x uint32 = 8 uchar codes)
  // completely inside one group of 64 (g = (k0 >> 6) + (lane >> 3)).
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row0 = threadgroup_position_in_grid.x * 8u + sg * 2u;
  if (row0 >= uint(OUT_N)) return;

  const uint r_valid = metal::min(2u, uint(OUT_N) - row0);
  const uint KW = uint(K) / 4u;
  const uint NG = uint(K) / 64u;
  const uint g_sub = lane >> 3;

  float acc[ROWS][2];
  #pragma unroll
  for (uint v = 0u; v < uint(ROWS); ++v) {
    acc[v][0] = 0.0f;
    acc[v][1] = 0.0f;
  }

  const device uint* w0 = w + size_t(row0) * KW;
  const device ST*   s0 = scales + size_t(row0) * NG;
  const device ST*   b0 = biases + size_t(row0) * NG;

  for (uint k0 = 0u; k0 < uint(K); k0 += 256u) {
    const uint k_lane = k0 + lane * 8u;
    const uint g = (k0 >> 6) + g_sub;
    float4 x_lo[ROWS], x_hi[ROWS];
    #pragma unroll
    for (uint v = 0u; v < uint(ROWS); ++v) {
      const device XT* xp = x + size_t(v) * uint(K) + k_lane;
      x_lo[v] = float4(*(const device vec<XT, 4>*)(xp));
      x_hi[v] = float4(*(const device vec<XT, 4>*)(xp + 4u));
    }

    #pragma unroll
    for (uint r = 0u; r < 2u; ++r) {
      if (r < r_valid) {
        const device uint* wr = w0 + size_t(r) * KW + (k_lane >> 2);
        const uint p0 = wr[0];
        const uint p1 = wr[1];
        const float sc = float(s0[size_t(r) * NG + g]);
        const float bi = float(b0[size_t(r) * NG + g]);
        const float4 w_lo = float4(
            float(p0 & 0xFFu),
            float((p0 >> 8) & 0xFFu),
            float((p0 >> 16) & 0xFFu),
            float((p0 >> 24) & 0xFFu)) * sc + bi;
        const float4 w_hi = float4(
            float(p1 & 0xFFu),
            float((p1 >> 8) & 0xFFu),
            float((p1 >> 16) & 0xFFu),
            float((p1 >> 24) & 0xFFu)) * sc + bi;
        #pragma unroll
        for (uint v = 0u; v < uint(ROWS); ++v) {
          acc[v][r] += dot(x_lo[v], w_lo) + dot(x_hi[v], w_hi);
        }
      }
    }
  }

  const float cap = cap_buf[0];
  #pragma unroll
  for (uint v = 0u; v < uint(ROWS); ++v) {
    #pragma unroll
    for (uint r = 0u; r < 2u; ++r) {
      float total = simd_sum(acc[v][r]);
      if (lane == 0u && r < r_valid) {
        if (HAS_CAP) {
          total = metal::tanh(total / cap) * cap;
        }
        out[size_t(v) * uint(OUT_N) + row0 + r] = OT(total);
      }
    }
  }
"""


@lru_cache(maxsize=64)
def _affine8_g64_multi_kernel(
    rows: int, K: int, out_n: int, x_dn: str, s_dn: str, o_dn: str, has_cap: bool
):
    return mx.fast.metal_kernel(
        name=f"inkling2_aff8g64_mv_r{rows}_k{K}_n{out_n}_{x_dn}_{s_dn}_{o_dn}_c{int(has_cap)}",
        input_names=["x", "w", "scales", "biases", "cap_buf"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_AFFINE8_G64_MULTI_GEMV_SRC,
        ensure_row_contiguous=True,
    )


def affine8_g64_multi_gemv(
    x,
    weight,
    scales,
    biases,
    *,
    out_dtype=None,
    softcap: float = 0.0,
    out_cols: int = 0,
):
    """Multi-row affine 8-bit g=64 GEMV reading weights ONCE across up to 8 rows per pass."""
    lead = x.shape[:-1]
    K = int(x.shape[-1])
    N = int(weight.shape[0])
    out_n = int(out_cols) if (out_cols and 0 < int(out_cols) <= N) else N
    T = int(x.size) // K
    od = out_dtype if out_dtype is not None else x.dtype
    if (
        T < 1
        or K % 256 != 0
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or scales.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or biases.dtype != scales.dtype
        or weight.dtype != mx.uint32
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    x2 = x.reshape(T, K)
    if T > 8:
        parts = [
            affine8_g64_multi_gemv(
                x2[i : i + 8], weight, scales, biases,
                out_dtype=od, softcap=softcap, out_cols=out_n,
            )
            for i in range(0, T, 8)
        ]
        return mx.concatenate(parts, axis=0).reshape(*lead, out_n)
    dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}
    has_cap = bool(softcap and float(softcap) > 0.0)
    cap_buf = mx.array([float(softcap) if has_cap else 1.0], dtype=mx.float32)
    n_tg = (out_n + 7) // 8
    y = _affine8_g64_multi_kernel(T, K, out_n, dn[x2.dtype], dn[scales.dtype], dn[od], has_cap)(
        inputs=[x2, weight, scales, biases, cap_buf],
        template=[
            ("XT", x2.dtype),
            ("ST", scales.dtype),
            ("OT", od),
            ("ROWS", T),
            ("K", K),
            ("OUT_N", out_n),
            ("HAS_CAP", has_cap),
        ],
        grid=(n_tg * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, out_n)],
        output_dtypes=[od],
    )[0]
    return y.reshape(*lead, out_n)


# ---------------------------------------------------------------------------
# Fused Metal Inkling-2 MoE Router + Sigmoid + Bias + Top-8 + Normalize
# Replaces 5-op @mx.compile + argpartition across 48 visits (-2.2 ms/step!).
# ---------------------------------------------------------------------------
_INKLING_ROUTER_SCORES_SRC = r"""
  // 1 TG per expert (128 TGs) x 9 simdgroups (288 threads) splitting K=2304 (256 elems/sg = 8/lane!).
  // Bit-identical per-row reduction across ROWS=1..8.
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row = threadgroup_position_in_grid.x;
  if (row >= uint(E)) return;

  threadgroup float sh[ROWS][9];
  const uint k_lane = sgi * 256u + lane * 8u;
  const device WT* wp = w + size_t(row) * uint(K) + k_lane;
  float4 w_lo = float4(*(const device vec<WT, 4>*)(wp));
  float4 w_hi = float4(*(const device vec<WT, 4>*)(wp + 4u));

  #pragma unroll
  for (uint t = 0u; t < uint(ROWS); ++t) {
    const device XT* xp = x + size_t(t) * uint(K) + k_lane;
    float4 x_lo = float4(*(const device vec<XT, 4>*)(xp));
    float4 x_hi = float4(*(const device vec<XT, 4>*)(xp + 4u));
    float s = simd_sum(dot(x_lo, w_lo) + dot(x_hi, w_hi));
    if (lane == 0u) sh[t][sgi] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u && lane == 0u) {
    #pragma unroll
    for (uint t = 0u; t < uint(ROWS); ++t) {
      float tot = sh[t][0] + sh[t][1] + sh[t][2]
                + sh[t][3] + sh[t][4] + sh[t][5]
                + sh[t][6] + sh[t][7] + sh[t][8];
      scores[size_t(t) * uint(E) + row] = 1.0f / (1.0f + metal::exp(-tot));
    }
  }
"""

_INKLING_ROUTER_TOPK_SRC = r"""
  // 1 TG per token (128 threads = 4 simdgroups, 1 expert per thread for E=128!).
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint row = threadgroup_position_in_grid.y;

  const float raw0 = scores[size_t(row) * 128u + tid];
  float val = raw0 + float(bias[tid]);

  threadgroup float sg_v[4];
  threadgroup uint  sg_i[4];
  threadgroup uint  win_i;
  threadgroup uint  picked_i[TOPK];
  threadgroup float picked_r[TOPK];

  for (uint k = 0u; k < uint(TOPK); ++k) {
    float sm = simd_max(val);
    uint  si = simd_min((val == sm) ? tid : 0xFFFFFFFFu);
    if (lane == 0u) {
      sg_v[sg] = sm;
      sg_i[sg] = si;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float bv = sg_v[0];
      uint  bi = sg_i[0];
      for (uint s = 1u; s < 4u; ++s) {
        if (sg_v[s] > bv || (sg_v[s] == bv && sg_i[s] < bi)) {
          bv = sg_v[s];
          bi = sg_i[s];
        }
      }
      win_i = bi;
      picked_i[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == win_i) {
      picked_r[k] = raw0;
      val = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float r = (lane < uint(TOPK)) ? picked_r[lane] : 0.0f;
    float denom = simd_sum(r) + 1e-20f;
    if (lane < uint(TOPK)) {
      out_ids[size_t(row) * uint(TOPK) + lane] = picked_i[lane];
      out_w[size_t(row) * uint(TOPK) + lane] = (r / denom) * scale_buf[0];
    }
  }
"""


@lru_cache(maxsize=32)
def _inkling_router_scores_kernel(rows: int, E: int, K: int, x_dn: str, w_dn: str):
    return mx.fast.metal_kernel(
        name=f"inkling2_router_scores_r{rows}_e{E}_k{K}_{x_dn}_{w_dn}",
        input_names=["x", "w"],
        output_names=["scores"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_ROUTER_SCORES_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _inkling_router_topk_kernel(top_k: int, b_dn: str):
    return mx.fast.metal_kernel(
        name=f"inkling2_router_topk_k{top_k}_{b_dn}",
        input_names=["scores", "bias", "scale_buf"],
        output_names=["out_ids", "out_w"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_ROUTER_TOPK_SRC,
        ensure_row_contiguous=True,
    )


def inkling_router_topk(x, gate_weight, bias, top_k: int, route_scale: float):
    """Fused Metal Inkling-2 MoE router (E=128, K=2304, T=1..16)."""
    lead = x.shape[:-1]
    K = int(x.shape[-1])
    E = int(gate_weight.shape[0])
    T = int(x.size) // K
    if (
        E != 128
        or K != 2304
        or not (1 <= T <= 16)
        or not (1 <= top_k <= 16)
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or gate_weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or bias.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    x2 = x.reshape(T, K)
    dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}
    if T <= 8:
        scores = _inkling_router_scores_kernel(T, E, K, dn[x2.dtype], dn[gate_weight.dtype])(
            inputs=[x2, gate_weight],
            template=[("XT", x2.dtype), ("WT", gate_weight.dtype), ("ROWS", T), ("E", E), ("K", K)],
            grid=(E * 288, 1, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[(T, E)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        s_a = _inkling_router_scores_kernel(8, E, K, dn[x2.dtype], dn[gate_weight.dtype])(
            inputs=[x2[:8], gate_weight],
            template=[("XT", x2.dtype), ("WT", gate_weight.dtype), ("ROWS", 8), ("E", E), ("K", K)],
            grid=(E * 288, 1, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[(8, E)],
            output_dtypes=[mx.float32],
        )[0]
        rem = T - 8
        s_b = _inkling_router_scores_kernel(rem, E, K, dn[x2.dtype], dn[gate_weight.dtype])(
            inputs=[x2[8:], gate_weight],
            template=[("XT", x2.dtype), ("WT", gate_weight.dtype), ("ROWS", rem), ("E", E), ("K", K)],
            grid=(E * 288, 1, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[(rem, E)],
            output_dtypes=[mx.float32],
        )[0]
        scores = mx.concatenate([s_a, s_b], axis=0)
    ids, weights = _inkling_router_topk_kernel(int(top_k), dn[bias.dtype])(
        inputs=[scores, bias, mx.array([float(route_scale)], dtype=mx.float32)],
        template=[("BT", bias.dtype), ("TOPK", int(top_k))],
        grid=(128, T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, int(top_k)), (T, int(top_k))],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return ids.reshape(*lead, int(top_k)), weights.reshape(*lead, int(top_k))


# ---------------------------------------------------------------------------
# Multi-Row Zero-Mask Split-K Verify Attention for Inkling-2 (2 <= LQ <= 16)
# Reads K and V ONCE for all LQ query rows (replaces L sequential SDPA calls,
# eliminating 336 dispatches/verify and zeroing mask memory up to 1M context!).
# ---------------------------------------------------------------------------
_INKLING_VERIFY_ATTN_S1_SRC = r"""
  // 256 threads (8 simdgroups) per (split, head).
  // Simdgroup sg (0..LQ-1) owns query row r = sg; all 8 simdgroups cooperatively
  // stage BK=8 keys & values into 4 KB threadgroup memory, sharing 100% of KV reads!
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint split = threadgroup_position_in_grid.y;
  const uint h = threadgroup_position_in_grid.z;
  const uint kv_h = h / uint(GQA);
  const int LQ = int(q_shape[2]);
  const int S  = int(k_shape[2]);
  const int n_splits = int(params[0]);
  const int k_min = int(params[1]);
  const float sc = scale_buf[0];

  const int span = S - k_min;
  const int chunk = (span + n_splits - 1) / n_splits;
  const int s0 = min(k_min + int(split) * chunk, S);
  const int s1 = min(s0 + chunk, S);

  const bool active = int(sg) < LQ;
  const int q_pos = (S - LQ) + int(sg);

  float4 qv = float4(0.0f);
  if (active) {
    qv = float4(*(const device vec<bfloat, 4>*)(q + ((size_t)h * LQ + int(sg)) * 128u + lane * 4u));
  }

  threadgroup float rel_sh[8 * 16];
  if (tid < uint(LQ * 16)) {
    uint qi = tid >> 4;
    uint d  = tid & 15u;
    rel_sh[tid] = float(rel[((size_t)qi * H + h) * 16u + d]);
  }
  threadgroup bfloat k_sh[8 * 128];
  threadgroup bfloat v_sh[8 * 128];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t kv_base = (size_t)kv_h * (size_t)S * 128u;
  float m = -1e30f;
  float l = 0.0f;
  float4 o = float4(0.0f);

  for (int k0 = s0; k0 < s1; k0 += 8) {
    const int kp_load = k0 + int(sg);
    if (kp_load < s1) {
      *(threadgroup vec<bfloat, 4>*)(k_sh + sg * 128u + lane * 4u) =
          *(const device vec<bfloat, 4>*)(k + kv_base + (size_t)kp_load * 128u + lane * 4u);
      *(threadgroup vec<bfloat, 4>*)(v_sh + sg * 128u + lane * 4u) =
          *(const device vec<bfloat, 4>*)(v + kv_base + (size_t)kp_load * 128u + lane * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const int b_lim = min(8, s1 - k0);
      const threadgroup float* rq = rel_sh + sg * 16u;
      for (int b = 0; b < b_lim; ++b) {
        const int kp = k0 + b;
        const int dist = q_pos - kp;
        if (dist < 0 || (SLIDING > 0 && dist >= SLIDING)) continue;
        float4 kv = float4(*(const threadgroup vec<bfloat, 4>*)(k_sh + uint(b) * 128u + lane * 4u));
        float4 vv = float4(*(const threadgroup vec<bfloat, 4>*)(v_sh + uint(b) * 128u + lane * 4u));
        float score = simd_sum(dot(qv, kv)) * sc;
        if (dist < REL_EXTENT) {
          const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)dist * 16u);
          float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
          float rb = a0.x * rq[0]  + a0.y * rq[1]  + a0.z * rq[2]  + a0.w * rq[3]
                   + a1.x * rq[4]  + a1.y * rq[5]  + a1.z * rq[6]  + a1.w * rq[7]
                   + a2.x * rq[8]  + a2.y * rq[9]  + a2.z * rq[10] + a2.w * rq[11]
                   + a3.x * rq[12] + a3.y * rq[13] + a3.z * rq[14] + a3.w * rq[15];
          score += float(bfloat(rb));
        }
        float m_new = metal::max(m, score);
        float alpha = (l > 0.0f) ? metal::exp(m - m_new) : 0.0f;
        float p = metal::exp(score - m_new);
        o = o * alpha + vv * p;
        l = l * alpha + p;
        m = m_new;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (active) {
__EPILOGUE__
  }
"""

_INKLING_VERIFY_EPILOGUE_SINGLE = r"""
      const float sk = float(sinks[(size_t)h * LQ + int(sg)]);
      float gm = (l > 0.0f) ? metal::max(m, sk) : sk;
      float w_k = (l > 0.0f) ? metal::exp(m - gm) : 0.0f;
      float w_s = metal::exp(sk - gm);
      float inv_l = 1.0f / (l * w_k + w_s);
      float4 go = o * (w_k * inv_l);
      device vec<bfloat, 4>* dst = (device vec<bfloat, 4>*)(part_o + ((size_t)h * LQ + int(sg)) * 128u + lane * 4u);
      *dst = vec<bfloat, 4>(bfloat(go.x), bfloat(go.y), bfloat(go.z), bfloat(go.w));
"""

_INKLING_VERIFY_EPILOGUE_MULTI = r"""
      const size_t p_idx = ((size_t)h * LQ + int(sg)) * uint(n_splits) + split;
      if (lane == 0u) {
        part_m[p_idx] = m;
        part_l[p_idx] = l;
      }
      *(device float4*)(part_o + p_idx * 128u + lane * 4u) = o;
"""

_INKLING_VERIFY_ATTN_S2_SRC = r"""
  // Merge NSPLITS partials + per-query sink for each (head, row): 1 simdgroup per (h, r)
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.y;
  const uint h = threadgroup_position_in_grid.z;
  const uint LQ = uint(part_m_shape[1]);
  const uint n_splits = uint(part_m_shape[2]);

  const size_t base = ((size_t)h * LQ + r) * n_splits;
  const device float* pm = part_m + base;
  const device float* pl = part_l + base;
  const device float* po = part_o + base * 128u;

  const float sk = float(sinks[(size_t)h * LQ + r]);
  float gm = sk;
  for (uint s = 0u; s < n_splits; ++s) {
    if (pl[s] > 0.0f) gm = metal::max(gm, pm[s]);
  }
  float gl = metal::exp(sk - gm);
  float4 go = float4(0.0f);
  for (uint s = 0u; s < n_splits; ++s) {
    if (pl[s] > 0.0f) {
      float w = metal::exp(pm[s] - gm);
      gl += pl[s] * w;
      go += *(const device float4*)(po + (size_t)s * 128u + lane * 4u) * w;
    }
  }
  float inv_l = 1.0f / gl;
  device vec<bfloat, 4>* dst = (device vec<bfloat, 4>*)(out + ((size_t)h * LQ + r) * 128u + lane * 4u);
  *dst = vec<bfloat, 4>(bfloat(go.x * inv_l), bfloat(go.y * inv_l), bfloat(go.z * inv_l), bfloat(go.w * inv_l));
"""


@lru_cache(maxsize=16)
def _inkling_verify_attn_s1(H: int, gqa: int, sliding: int, rel_extent: int, single_split: bool):
    outs = ["part_o"] if single_split else ["part_m", "part_l", "part_o"]
    epilogue = _INKLING_VERIFY_EPILOGUE_SINGLE if single_split else _INKLING_VERIFY_EPILOGUE_MULTI
    return mx.fast.metal_kernel(
        name=f"inkling2_verify_attn_s1_h{H}_g{gqa}_s{sliding}_r{rel_extent}_ss{int(single_split)}",
        input_names=["q", "k", "v", "rel", "rel_proj_t", "sinks", "params", "scale_buf"],
        output_names=outs,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_VERIFY_ATTN_S1_SRC.replace("__EPILOGUE__", epilogue),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=2)
def _inkling_verify_attn_s2():
    return mx.fast.metal_kernel(
        name="inkling2_verify_attn_s2",
        input_names=["part_m", "part_l", "part_o", "sinks"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_INKLING_VERIFY_ATTN_S2_SRC,
        ensure_row_contiguous=True,
    )


def inkling_verify_attention(
    q, k, v, relative, rel_proj_t, sinks, *, scale: float, sliding: int, rel_extent: int
):
    """Multi-row zero-mask Split-K verify attention (2 <= LQ <= 16, B=1, D=128, d_rel=16)."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or q.dtype != mx.bfloat16
        or k.dtype != mx.bfloat16
        or v.dtype != mx.bfloat16
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[3] != 128
        or relative.shape[-1] != 16
    ):
        return None
    _, H, LQ, _ = (int(x) for x in q.shape)
    H_kv = int(k.shape[1])
    S = int(k.shape[2])
    if not (1 <= LQ <= 16) or H % H_kv != 0:
        return None
    if LQ > 8:
        prefix = S - LQ
        o1 = inkling_verify_attention(
            q[:, :, :8], k[:, :, : prefix + 8], v[:, :, : prefix + 8],
            relative[:, :8], rel_proj_t, sinks[:, :, :8],
            scale=scale, sliding=sliding, rel_extent=rel_extent,
        )
        o2 = inkling_verify_attention(
            q[:, :, 8:], k, v,
            relative[:, 8:], rel_proj_t, sinks[:, :, 8:],
            scale=scale, sliding=sliding, rel_extent=rel_extent,
        )
        return mx.concatenate([o1, o2], axis=2) if (o1 is not None and o2 is not None) else None

    gqa = H // H_kv
    k_min = max(0, (S - LQ) - int(sliding) + 1) if sliding > 0 else 0
    span = S - k_min
    if span <= 128:
        n_splits = 1
    elif span <= 1024:
        n_splits = 4
    else:
        n_splits = min(64, max(8, (span + 511) // 512))
    sinks_2d = sinks.reshape(H, LQ)
    params = mx.array([n_splits, k_min], dtype=mx.int32)
    sc_buf = mx.array([float(scale)], dtype=mx.float32)
    if n_splits == 1:
        return _inkling_verify_attn_s1(H, gqa, int(sliding), int(rel_extent), True)(
            inputs=[q, k, v, relative, rel_proj_t, sinks_2d, params, sc_buf],
            template=[
                ("H", H),
                ("GQA", gqa),
                ("SLIDING", int(sliding)),
                ("REL_EXTENT", int(rel_extent)),
            ],
            grid=(256, 1, H),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, H, LQ, 128)],
            output_dtypes=[mx.bfloat16],
        )[0]
    pm, pl, po = _inkling_verify_attn_s1(H, gqa, int(sliding), int(rel_extent), False)(
        inputs=[q, k, v, relative, rel_proj_t, sinks_2d, params, sc_buf],
        template=[
            ("H", H),
            ("GQA", gqa),
            ("SLIDING", int(sliding)),
            ("REL_EXTENT", int(rel_extent)),
        ],
        grid=(256, n_splits, H),
        threadgroup=(256, 1, 1),
        output_shapes=[(H, LQ, n_splits), (H, LQ, n_splits), (H, LQ, n_splits, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _inkling_verify_attn_s2()(
        inputs=[pm, pl, po, sinks_2d],
        grid=(32, LQ, H),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, H, LQ, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------
# GQA-Grouped M5 MPP Split-K Decode & Verify Attention (LQ = 1..10, S >= 512)
# Packs all M = GQA * LQ query rows sharing one KV head into Q_sh ONCE per TG,
# reading K and V ONCE instead of GQA=6 times (6x less KV DRAM at 32K-1M!)
# while computing Q@K^T and P@V on M5 hardware matrix units!
# ---------------------------------------------------------------------------
_INKLING_GQA_MPP_S1_SRC = r"""
  // Grid: (128, n_splits, H_kv), TG: (128, 1, 1) (4 simdgroups).
  // Compile-time constants: H, GQA, TQ (32 or 64), TK (64 or 32), SLIDING, REL_EXTENT.
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const int split = int(threadgroup_position_in_grid.y);
  const int kv_h = int(threadgroup_position_in_grid.z);

  const int LQ = int(q_shape[2]);
  const int S  = int(k_shape[2]);
  const int n_splits = int(params[0]);
  const int k_min = int(params[1]);
  const float sc = scale_buf[0];

  const int M = GQA * LQ;
  const int h0 = kv_h * GQA;
  const int span = S - k_min;
  const int chunk = (span + n_splits - 1) / n_splits;
  const int s0 = min(k_min + split * chunk, S);
  const int s1 = min(s0 + chunk, S);

  threadgroup bfloat Q_sh[TQ * 128];
  threadgroup float  S_sh[TQ * TK];
  threadgroup bfloat P_sh[TQ * TK];
  threadgroup bfloat rel_sh[TQ * 16];
  threadgroup float  m_sh[TQ];
  threadgroup float  l_sh[TQ];
  threadgroup float  alpha_sh[TQ];

  // Stage Q_sh[0..TQ-1, 0..127] and rel_sh[0..TQ-1, 0..15] ONCE per split!
  // Row m = g * LQ + qi maps to head (h0 + g), query row qi.
  for (uint idx4 = tid; idx4 < uint(TQ * 32); idx4 += 128u) {
    const int m = int(idx4 >> 5);
    const uint d4 = (idx4 & 31u) * 4u;
    vec<bfloat, 4> qv = vec<bfloat, 4>(bfloat(0.0f));
    if (m < M) {
      const int g = m / LQ;
      const int qi = m - g * LQ;
      qv = *(const device vec<bfloat, 4>*)(q + ((size_t)(h0 + g) * LQ + qi) * 128u + d4);
    }
    *(threadgroup vec<bfloat, 4>*)(Q_sh + m * 128 + int(d4)) = qv;
  }
  for (uint i = tid; i < uint(TQ * 16); i += 128u) {
    const int m = int(i >> 4);
    const uint d = i & 15u;
    bfloat rv = bfloat(0.0f);
    if (m < M) {
      const int g = m / LQ;
      const int qi = m - g * LQ;
      rv = rel[((size_t)qi * H + (h0 + g)) * 16u + d];
    }
    rel_sh[i] = rv;
  }
  if (int(tid) < TQ) {
    m_sh[tid] = -1e30f;
    l_sh[tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (s0 >= s1) {
    for (int m = int(tid); m < M; m += 128) {
      const int g = m / LQ;
      const int qi = m - g * LQ;
      const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
      part_m[p_idx] = -1e30f;
      part_l[p_idx] = 0.0f;
    }
    for (uint idx4 = tid; idx4 < uint(M * 32); idx4 += 128u) {
      const int m = int(idx4 >> 5);
      const uint d4 = (idx4 & 31u) * 4u;
      const int g = m / LQ;
      const int qi = m - g * LQ;
      const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
      *(device float4*)(part_o + p_idx * 128u + d4) = float4(0.0f);
    }
    return;
  }

  constexpr auto desc_qk = matmul2d_descriptor(TQ, TK, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto desc_pv = matmul2d_descriptor(TQ, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
  matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tQ(
      Q_sh, dextents<int32_t, 2>(128, TQ));
  auto aQ = tQ.slice<128, TQ>(0, 0);

  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tP(
      P_sh, dextents<int32_t, 2>(TK, TQ));
  auto aP = tP.slice<TK, TQ>(0, 0);

  const size_t kv_head_base = (size_t)kv_h * (size_t)S * 128u;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV0(
      (device bfloat*)v + kv_head_base, dextents<int32_t, 2>(128, TK));
  auto bV0 = tV0.slice<128, TK>(0, 0);

  auto O_acc = op_pv.template get_destination_cooperative_tensor<decltype(aP), decltype(bV0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
    if (O_acc.is_valid_element(i)) O_acc[i] = 0.0f;
  }

  const int q_base_pos = S - LQ;
  constexpr uint ROWS_PER_SG = TQ / 4;
  constexpr uint COLS_PER_LANE = TK / 32;

  for (int k0 = s0; k0 < s1; k0 += TK) {
    const int c0 = (k0 + TK <= S) ? k0 : (S - TK);
    const bool is_clamped = (c0 < k0) || (c0 + TK > s1);
    const bool pure_prefix = (!is_clamped) && (SLIDING == 0) && (q_base_pos - (c0 + TK - 1) >= REL_EXTENT);

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK(
        (device bfloat*)k + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, TK));
    auto bK = tK.slice<128, TK>(0, 0);
    auto S_coop = op_qk.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    op_qk.run(aQ, bK, S_coop);

    #pragma unroll
    for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
      if (S_coop.is_valid_element(i)) {
        const auto idx = S_coop.get_multidimensional_index(i);
        S_sh[idx[1] * TK + idx[0]] = S_coop[i] * sc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (pure_prefix) {
      for (uint r = 0u; r < ROWS_PER_SG; ++r) {
        const uint m = sg * ROWS_PER_SG + r;
        const uint rb = m * TK;
        float sv[COLS_PER_LANE];
        float rmax = -1e30f;
        #pragma unroll
        for (uint c = 0u; c < COLS_PER_LANE; ++c) {
          float s = S_sh[rb + lane + c * 32u];
          sv[c] = s;
          rmax = metal::max(rmax, s);
        }
        rmax = simd_max(rmax);
        const float m_old = m_sh[m];
        const float l_old = l_sh[m];
        const float m_new = (l_old > 0.0f) ? metal::max(m_old, rmax) : rmax;
        const float alpha = (l_old > 0.0f) ? metal::exp(m_old - m_new) : 0.0f;
        float l_acc = 0.0f;
        #pragma unroll
        for (uint c = 0u; c < COLS_PER_LANE; ++c) {
          float p = metal::exp(sv[c] - m_new);
          l_acc += p;
          P_sh[rb + lane + c * 32u] = bfloat(p);
        }
        const float l_tile = simd_sum(l_acc);
        if (lane == 0u) {
          m_sh[m] = m_new;
          l_sh[m] = l_old * alpha + l_tile;
          alpha_sh[m] = alpha;
        }
      }
    } else {
      for (uint r = 0u; r < ROWS_PER_SG; ++r) {
        const int m = int(sg * ROWS_PER_SG + r);
        const uint rb = uint(m) * TK;
        const int qi = (m < M) ? (m % LQ) : 0;
        const int q_pos = q_base_pos + qi;
        const threadgroup bfloat* rq = rel_sh + uint(m) * 16u;
        float sv[COLS_PER_LANE];
        float rmax = -1e30f;
        #pragma unroll
        for (uint c = 0u; c < COLS_PER_LANE; ++c) {
          const uint ki = lane + c * 32u;
          const int kp = c0 + int(ki);
          float s = S_sh[rb + ki];
          const int d = q_pos - kp;
          if (m >= M || kp < k0 || kp >= s1 || d < 0 || (SLIDING > 0 && d >= SLIDING)) {
            s = -1e30f;
          } else if (d < REL_EXTENT) {
            const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)d * 16u);
            float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
            float rbias = a0.x * float(rq[0])  + a0.y * float(rq[1])  + a0.z * float(rq[2])  + a0.w * float(rq[3])
                        + a1.x * float(rq[4])  + a1.y * float(rq[5])  + a1.z * float(rq[6])  + a1.w * float(rq[7])
                        + a2.x * float(rq[8])  + a2.y * float(rq[9])  + a2.z * float(rq[10]) + a2.w * float(rq[11])
                        + a3.x * float(rq[12]) + a3.y * float(rq[13]) + a3.z * float(rq[14]) + a3.w * float(rq[15]);
            s += float(bfloat(rbias));
          }
          sv[c] = s;
          rmax = metal::max(rmax, s);
        }
        rmax = simd_max(rmax);
        const float m_old = m_sh[m];
        const float l_old = l_sh[m];
        const bool has_valid = rmax > -1e29f;
        const float m_new = has_valid ? ((l_old > 0.0f) ? metal::max(m_old, rmax) : rmax) : m_old;
        const float alpha = (l_old > 0.0f && has_valid) ? metal::exp(m_old - m_new) : (l_old > 0.0f ? 1.0f : 0.0f);
        float l_acc = 0.0f;
        #pragma unroll
        for (uint c = 0u; c < COLS_PER_LANE; ++c) {
          float p = (sv[c] > -1e29f) ? metal::exp(sv[c] - m_new) : 0.0f;
          l_acc += p;
          P_sh[rb + lane + c * 32u] = bfloat(p);
        }
        const float l_tile = simd_sum(l_acc);
        if (lane == 0u) {
          m_sh[m] = m_new;
          l_sh[m] = l_old * alpha + l_tile;
          alpha_sh[m] = alpha;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
      if (O_acc.is_valid_element(i)) {
        O_acc[i] *= alpha_sh[O_acc.get_multidimensional_index(i)[1]];
      }
    }
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV(
        (device bfloat*)v + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, TK));
    auto bV = tV.slice<128, TK>(0, 0);
    op_pv.run(aP, bV, O_acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int m = int(tid); m < M; m += 128) {
    const int g = m / LQ;
    const int qi = m - g * LQ;
    const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
    part_m[p_idx] = m_sh[m];
    part_l[p_idx] = l_sh[m];
  }
  #pragma unroll
  for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
    if (O_acc.is_valid_element(i)) {
      const auto idx = O_acc.get_multidimensional_index(i);
      const int m = int(idx[1]);
      if (m < M) {
        const int g = m / LQ;
        const int qi = m - g * LQ;
        const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
        part_o[p_idx * 128u + uint(idx[0])] = O_acc[i];
      }
    }
  }
"""

_INKLING_GQA_MPP_COMPILED: dict = {}


def _inkling_gqa_mpp_s1(H: int, GQA: int, TQ: int, TK: int, sliding: int, rel_extent: int):
    key = (H, GQA, TQ, TK, sliding, rel_extent)
    run = _INKLING_GQA_MPP_COMPILED.get(key)
    if run is None:
        consts = "".join(
            f"  constexpr int {n} = {int(v)};\n"
            for n, v in (
                ("H", H),
                ("GQA", GQA),
                ("TQ", TQ),
                ("TK", TK),
                ("SLIDING", sliding),
                ("REL_EXTENT", rel_extent),
            )
        )
        run = _INKLING_GQA_MPP_COMPILED[key] = mx.fast.metal_kernel(
            name=f"inkling2_gqa_mpp_s1_h{H}_g{GQA}_q{TQ}_k{TK}_s{sliding}_r{rel_extent}",
            input_names=["q", "k", "v", "rel", "rel_proj_t", "params", "scale_buf"],
            output_names=["part_m", "part_l", "part_o"],
            header=_INKLING_MPP_HEADER,
            source=consts + _INKLING_GQA_MPP_S1_SRC,
            ensure_row_contiguous=True,
        )
    return run


def inkling_gqa_mpp_attention(
    q, k, v, relative, rel_proj_t, sinks, *, scale: float, sliding: int, rel_extent: int
):
    """GQA-grouped M5 MPP Split-K Decode/Verify attention reading each KV head ONCE (6x less KV DRAM!)."""
    if (
        not _inkling_mpp_available()
        or q.dtype != mx.bfloat16
        or k.dtype != mx.bfloat16
        or v.dtype != mx.bfloat16
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[3] != 128
        or relative.shape[-1] != 16
    ):
        return None
    _, H, LQ, _ = (int(x) for x in q.shape)
    H_kv = int(k.shape[1])
    S = int(k.shape[2])
    if H % H_kv != 0 or LQ < 1:
        return None
    gqa = H // H_kv
    M = gqa * LQ
    if M <= 32:
        TQ, TK = 32, 64
    elif M <= 64:
        TQ, TK = 64, 32
    else:
        return None
    if S < TK:
        return None
    k_min = max(0, (S - LQ) - int(sliding) + 1) if sliding > 0 else 0
    span = S - k_min
    if span < 512:
        return None
    # Saturate M5 Max (40 cores) / M5 Ultra (80 cores) across H_kv * n_splits threadgroups!
    n_splits = min(64, max(8, (span + 511) // 512))
    params = mx.array([n_splits, k_min], dtype=mx.int32)
    sc_buf = mx.array([float(scale)], dtype=mx.float32)
    pm, pl, po = _inkling_gqa_mpp_s1(H, gqa, TQ, TK, int(sliding), int(rel_extent))(
        inputs=[q, k, v, relative, rel_proj_t, params, sc_buf],
        grid=(128, n_splits, H_kv),
        threadgroup=(128, 1, 1),
        output_shapes=[(H, LQ, n_splits), (H, LQ, n_splits), (H, LQ, n_splits, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _inkling_verify_attn_s2()(
        inputs=[pm, pl, po, sinks.reshape(H, LQ)],
        grid=(32, LQ, H),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, H, LQ, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]

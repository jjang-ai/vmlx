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

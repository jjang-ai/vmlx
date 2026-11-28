"""Fused single-request NVFP4 expert decode & verify for the official Inkling Mini layout."""

from functools import lru_cache

import mlx.core as mx

from .masked_quantized import _HEADER, _global_scale, supports_masked_qmv
from .nvfp4 import ScaledQuantizedSwitchLinear


def prepare_fused_nvfp4_experts(language_model):
    """Prepare small scale vectors once so fused NVFP4 decode uses contiguous weights."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return 0

    prepared = 0
    for layer in language_model.model.layers:
        moe = layer.mlp
        switch = moe.switch_mlp
        gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
        if moe.top_k != 8 or not all(
            type(module) is ScaledQuantizedSwitchLinear
            and supports_masked_qmv(module, input_alignment=128)
            and (module.input_dims, module.output_dims) == dims
            for module, dims in (
                (gate, (2304, 1152)), (up, (2304, 1152)), (down, (1152, 2304)),
            )
        ):
            continue

        caps = mx.array([switch.activation.gate_cap, switch.activation.up_cap], mx.float32)
        g_sc = _global_scale(gate)
        u_sc = _global_scale(up)
        d_sc = _global_scale(down)
        mx.eval(g_sc, u_sc, d_sc, caps)
        moe._fused_nvfp4_constants = (g_sc, u_sc, d_sc, caps)
        prepared += 1
    return prepared


def fused_nvfp4_experts(switch, x, indices, route_weights, constants):
    """Fused NVFP4 MoE: 1-pass QMV gate+up+softcapped-SwiGLU + weighted down.

    For multi-row verify (T > 1), schedules route/position on the fastest grid axis (y)
    so consecutive threadgroups share the same expert weight tile in L2 cache (bit-identical!).
    """
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    gate_scale, up_scale, down_scale, caps = constants[:4]
    t = x.dtype
    hidden = x.shape[-1]
    inner = gate.output_dims
    slot_fastest = int(indices.size) > 8
    active = _gate_up_and_activation(slot_fastest)(
        inputs=[
            x, gate.weight, gate.scales, gate_scale,
            up.weight, up.scales, up_scale, indices, caps,
        ],
        template=[("T", t), ("K", hidden), ("N", inner), ("R", 8)],
        grid=(32, indices.size, inner // 4) if slot_fastest else (32, inner // 4, indices.size),
        threadgroup=(32, 1, 2) if slot_fastest else (32, 2, 1),
        output_shapes=[(*indices.shape, 1, inner)],
        output_dtypes=[t],
    )[0]
    return _down_and_weighted_sum(slot_fastest)(
        inputs=[active, down.weight, down.scales, down_scale, indices, route_weights],
        template=[("T", t), ("K", inner), ("N", hidden)],
        grid=(32, indices.size, hidden // 4) if slot_fastest else (32, hidden // 4, indices.size),
        threadgroup=(32, 8, 1) if slot_fastest else (32, 1, 8),
        output_shapes=[(*indices.shape[:-1], hidden)],
        output_dtypes=[t],
    )[0]


_GATE_UP_SRC = r"""
const uint route = threadgroup_position_in_grid.z;
const uint lane = thread_index_in_simdgroup;
const uint row0 = 8 * threadgroup_position_in_grid.y + 4 * simdgroup_index_in_threadgroup;
// One input load feeds both independent FP32 projections.
const uint expert = ids[route];
const device T* input = x + (route / R) * K;
constexpr int P = K % 512 == 0 ? 16 : 8;
constexpr int BLOCK = 32 * P;
const uint base = expert * N + row0;
const device uchar* gw = (const device uchar*)gate_w + base * (K / 2) + lane * (P / 2);
const device uchar* uw = (const device uchar*)up_w + base * (K / 2) + lane * (P / 2);
const device uchar* gs = gate_s + base * (K / 16) + lane / (16 / P);
const device uchar* us = up_s + base * (K / 16) + lane / (16 / P);
float acc_gate[4] = {};
float acc_up[4] = {};
thread float values[P];
for (uint k = 0; k < K; k += BLOCK) {
    load_vector<T,float,P>(input + k + lane * P, values);
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        uint weight_offset = r * (K / 2) + k / 2;
        uint scale_offset = r * (K / 16) + k / 16;
        float scale_g = dequantize_scale<float,16>(gs[scale_offset]);
        float scale_u = dequantize_scale<float,16>(us[scale_offset]);
        acc_gate[r] += qdot<float,P,4>(gw + weight_offset, values, scale_g);
        acc_up[r] += qdot<float,P,4>(uw + weight_offset, values, scale_u);
    }
}
const float global_g = gate_g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
const float global_u = up_g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
#pragma unroll
for (uint r = 0; r < 4; ++r) {
    uint row = row0 + r;
    const T g = T(simd_sum(acc_gate[r]) * global_g);
    const T u = T(simd_sum(acc_up[r]) * global_u);

    if (lane == 0) {
        const float gate_value = float(g);
        const float up_value = float(u);
        const float gsoft = metal::precise::tanh(gate_value / caps[0]) * caps[0];
        const float y = 1.0f / (1.0f + metal::precise::exp(abs(gate_value)));
        const float sig = gate_value < 0.0f ? y : 1.0f - y;
        activated[route * N + row] = T(gsoft * sig * metal::precise::tanh(up_value / caps[1]) * caps[1]);
    }
}
"""


_DOWN_SRC = r"""
const uint position = threadgroup_position_in_grid.z;
const uint within = simdgroup_index_in_threadgroup;
const uint route = position * 8 + within;
const uint expert = ids[route];
const uint lane = thread_index_in_simdgroup;
const uint row0 = 4 * threadgroup_position_in_grid.y;
const uint base = expert * N + row0;
const device T* input = middle + route * K;
const device uchar* ws = (const device uchar*)weight + base * (K/2) + lane * 4;
const device uchar* ss = scales + base * (K/16) + lane / 2;
thread float values[8];
float accum[4] = {};
threadgroup float products[8][4];
for (uint k=0; k<K; k+=256) {
    if (k + lane * 8 >= K) break;
    load_vector<T,float,8>(input + k + lane * 8, values);
    #pragma unroll
    for (uint r=0; r<4; ++r) {
        uint scale_offset = r * (K/16) + k/16;
        uint weight_offset = r * (K/2) + k/2;
        const float value = dequantize_scale<float,16>(ss[scale_offset]);
        accum[r] += qdot<float,8,4>(ws + weight_offset, values, value);
    }
}
const float factor = global_scales[expert] / (F8E4M3_MAX * F4E2M1_MAX);
#pragma unroll
for (uint r=0; r<4; ++r) {
    const T projected = T(simd_sum(accum[r]) * factor);
    if (lane==0) products[within][r] = float(projected) * route_weights[route];
}
// Both BF16 projection and FP32 route weighting precede the native-order sum.
threadgroup_barrier(mem_flags::mem_threadgroup);
if (within==0 && lane<4) {
    float value = 0.0f;
    #pragma unroll
    for (uint e=0; e<8; ++e) value += products[e][lane];
    out[position*N+row0+lane]=T(value);
}
"""


def _swap_yz(src: str) -> str:
    return (
        src.replace("threadgroup_position_in_grid.y", "__TGP_Y__")
        .replace("threadgroup_position_in_grid.z", "threadgroup_position_in_grid.y")
        .replace("__TGP_Y__", "threadgroup_position_in_grid.z")
    )


@lru_cache(maxsize=2)
def _gate_up_and_activation(slot_fastest: bool = False):
    return mx.fast.metal_kernel(
        name="inkling_nvfp4_gate_up_activation_yz" if slot_fastest else "inkling_nvfp4_gate_up_activation",
        input_names=["x", "gate_w", "gate_s", "gate_g", "up_w", "up_s", "up_g", "ids", "caps"],
        output_names=["activated"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=_swap_yz(_GATE_UP_SRC) if slot_fastest else _GATE_UP_SRC,
    )


@lru_cache(maxsize=2)
def _down_and_weighted_sum(slot_fastest: bool = False):
    return mx.fast.metal_kernel(
        name="inkling_nvfp4_down_weighted_routes_yz" if slot_fastest else "inkling_nvfp4_down_weighted_routes",
        input_names=["middle", "weight", "scales", "global_scales", "ids", "route_weights"],
        output_names=["out"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=_swap_yz(_DOWN_SRC) if slot_fastest else _DOWN_SRC,
    )

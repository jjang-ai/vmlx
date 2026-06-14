'''MiniMax-M3 affine-2 SwitchGLU fast path.'''

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU


_KERNEL_CACHE: dict[tuple[int, int, int, int, int], Any] = {}
_DISABLE_ENV_NAMES = ('VMLINUX_M3_AFFINE2_SWITCH', 'VMLX_M3_AFFINE2_SWITCH')


def _disabled() -> bool:
    for name in _DISABLE_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in {'0', 'false', 'off', 'no'}
    return False


def _is_affine2_projection(proj: Any) -> bool:
    return (
        getattr(proj, 'bits', None) == 2
        and getattr(proj, 'mode', 'affine') == 'affine'
        and getattr(proj, 'group_size', None) is not None
        and getattr(proj, 'scales', None) is not None
        and getattr(proj, 'biases', None) is not None
    )


def _has_single_decode_row(x: mx.array, indices: mx.array) -> bool:
    if len(x.shape) < 2 or len(indices.shape) < 1:
        return False
    leading = 1
    for dim in x.shape[:-1]:
        leading *= dim
    return leading == 1 and indices.size == indices.shape[-1]


def can_use_affine2_switchglu(switch: Any, x: mx.array, indices: mx.array) -> bool:
    '''Return true when the decode fast path is safe for this SwitchGLU call.'''
    if _disabled() or not _has_single_decode_row(x, indices):
        return False
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        if not _is_affine2_projection(getattr(switch, name, None)):
            return False
    return (
        switch.gate_proj.input_dims == x.shape[-1]
        and switch.up_proj.input_dims == x.shape[-1]
        and switch.down_proj.input_dims == switch.gate_proj.output_dims
        and switch.down_proj.input_dims == switch.up_proj.output_dims
    )

def _make_affine2_gather(
    in_features: int,
    out_features: int,
    group_size: int,
    top_k: int,
    opt: int,
):
    key = (in_features, out_features, group_size, top_k, opt)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    packed_cols = (in_features * 2 + 31) // 32
    scale_cols = (in_features + group_size - 1) // group_size
    meta = mx.array(
        [top_k, in_features, out_features, packed_cols, scale_cols, group_size],
        dtype=mx.uint32,
    )
    source = f'''
        uint gx = thread_position_in_grid.x;
        uint dispatch_idx = thread_position_in_grid.y;
        uint out_group = gx / 32u;
        uint lane = gx % 32u;
        uint out0 = out_group * {opt}u;

        uint K = meta[0];
        uint in_features = meta[1];
        uint out_features = meta[2];
        uint packed_cols = meta[3];
        uint scale_cols = meta[4];
        uint group_size = meta[5];
        if (dispatch_idx >= K || out0 >= out_features) return;

        uint expert = rhs_indices[dispatch_idx];
        float acc[{opt}];
        #pragma unroll
        for (uint o = 0; o < {opt}u; o++) acc[o] = 0.0f;

        uint n_outs = {opt}u;
        if (out0 + {opt}u > out_features) n_outs = out_features - out0;
        uint expert_w_base = expert * out_features * packed_cols;
        uint expert_s_base = expert * out_features * scale_cols;
        uint x_base = dispatch_idx * in_features;

        for (uint pc = lane; pc < packed_cols; pc += 32u) {{
            uint i_base = pc * 16u;
            uint g = i_base / group_size;
            uint pv[{opt}];
            float sc[{opt}];
            float bs[{opt}];

            #pragma unroll
            for (uint o = 0; o < {opt}u; o++) {{
                if (o < n_outs) {{
                    uint row = out0 + o;
                    pv[o] = weight[expert_w_base + row * packed_cols + pc];
                    sc[o] = static_cast<float>(scales[expert_s_base + row * scale_cols + g]);
                    bs[o] = static_cast<float>(biases[expert_s_base + row * scale_cols + g]);
                }} else {{
                    pv[o] = 0u;
                    sc[o] = 0.0f;
                    bs[o] = 0.0f;
                }}
            }}

            #pragma unroll
            for (uint kk = 0; kk < 16u; kk++) {{
                uint i = i_base + kk;
                if (i >= in_features) break;
                float xv = static_cast<float>(x[x_base + i]);
                uint shift = kk * 2u;

                #pragma unroll
                for (uint o = 0; o < {opt}u; o++) {{
                    float q = static_cast<float>((pv[o] >> shift) & 3u);
                    acc[o] += xv * (q * sc[o] + bs[o]);
                }}
            }}
        }}

        #pragma unroll
        for (uint o = 0; o < {opt}u; o++) acc[o] = simd_sum(acc[o]);

        if (lane == 0u) {{
            uint base = dispatch_idx * out_features;
            for (uint o = 0; o < n_outs; o++) out[base + out0 + o] = acc[o];
        }}
    '''
    kernel = mx.fast.metal_kernel(
        name=f'm3_affine2_i{in_features}_o{out_features}_g{group_size}_k{top_k}_o{opt}',
        input_names=['x', 'weight', 'scales', 'biases', 'rhs_indices', 'meta'],
        output_names=['out'],
        source=source,
    )
    grid_x = ((out_features + opt - 1) // opt) * 32

    def gather(x_rows, proj, rhs_indices):
        (out,) = kernel(
            inputs=[x_rows, proj.weight, proj.scales, proj.biases, rhs_indices, meta],
            output_shapes=[(top_k, out_features)],
            output_dtypes=[mx.float32],
            grid=(grid_x, top_k, 1),
            threadgroup=(min(grid_x, 256), 1, 1),
        )
        return out

    _KERNEL_CACHE[key] = gather
    return gather


def _make_affine2_gather_pair(
    in_features: int,
    out_features: int,
    group_size: int,
    top_k: int,
    opt: int,
):
    key = (in_features, out_features, group_size, top_k, opt, 2)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    packed_cols = (in_features * 2 + 31) // 32
    scale_cols = (in_features + group_size - 1) // group_size
    meta = mx.array(
        [top_k, in_features, out_features, packed_cols, scale_cols, group_size],
        dtype=mx.uint32,
    )
    source = f'''
        uint gx = thread_position_in_grid.x;
        uint dispatch_idx = thread_position_in_grid.y;
        uint out_group = gx / 32u;
        uint lane = gx % 32u;
        uint out0 = out_group * {opt}u;

        uint K = meta[0];
        uint in_features = meta[1];
        uint out_features = meta[2];
        uint packed_cols = meta[3];
        uint scale_cols = meta[4];
        uint group_size = meta[5];
        if (dispatch_idx >= K || out0 >= out_features) return;

        uint expert = rhs_indices[dispatch_idx];
        float gate_acc[{opt}];
        float up_acc[{opt}];
        #pragma unroll
        for (uint o = 0; o < {opt}u; o++) {{
            gate_acc[o] = 0.0f;
            up_acc[o] = 0.0f;
        }}

        uint n_outs = {opt}u;
        if (out0 + {opt}u > out_features) n_outs = out_features - out0;
        uint expert_w_base = expert * out_features * packed_cols;
        uint expert_s_base = expert * out_features * scale_cols;
        uint x_base = dispatch_idx * in_features;

        for (uint pc = lane; pc < packed_cols; pc += 32u) {{
            uint i_base = pc * 16u;
            uint g = i_base / group_size;
            uint gate_pv[{opt}];
            uint up_pv[{opt}];
            float gate_sc[{opt}];
            float gate_bs[{opt}];
            float up_sc[{opt}];
            float up_bs[{opt}];

            #pragma unroll
            for (uint o = 0; o < {opt}u; o++) {{
                if (o < n_outs) {{
                    uint row = out0 + o;
                    uint w_off = expert_w_base + row * packed_cols + pc;
                    uint s_off = expert_s_base + row * scale_cols + g;
                    gate_pv[o] = gate_weight[w_off];
                    up_pv[o] = up_weight[w_off];
                    gate_sc[o] = static_cast<float>(gate_scales[s_off]);
                    gate_bs[o] = static_cast<float>(gate_biases[s_off]);
                    up_sc[o] = static_cast<float>(up_scales[s_off]);
                    up_bs[o] = static_cast<float>(up_biases[s_off]);
                }} else {{
                    gate_pv[o] = 0u;
                    up_pv[o] = 0u;
                    gate_sc[o] = 0.0f;
                    gate_bs[o] = 0.0f;
                    up_sc[o] = 0.0f;
                    up_bs[o] = 0.0f;
                }}
            }}

            #pragma unroll
            for (uint kk = 0; kk < 16u; kk++) {{
                uint i = i_base + kk;
                if (i >= in_features) break;
                float xv = static_cast<float>(x[x_base + i]);
                uint shift = kk * 2u;

                #pragma unroll
                for (uint o = 0; o < {opt}u; o++) {{
                    float gate_q = static_cast<float>((gate_pv[o] >> shift) & 3u);
                    float up_q = static_cast<float>((up_pv[o] >> shift) & 3u);
                    gate_acc[o] += xv * (gate_q * gate_sc[o] + gate_bs[o]);
                    up_acc[o] += xv * (up_q * up_sc[o] + up_bs[o]);
                }}
            }}
        }}

        #pragma unroll
        for (uint o = 0; o < {opt}u; o++) {{
            gate_acc[o] = simd_sum(gate_acc[o]);
            up_acc[o] = simd_sum(up_acc[o]);
        }}

        if (lane == 0u) {{
            uint base = dispatch_idx * out_features;
            for (uint o = 0; o < n_outs; o++) {{
                uint off = base + out0 + o;
                gate_out[off] = gate_acc[o];
                up_out[off] = up_acc[o];
            }}
        }}
    '''
    kernel = mx.fast.metal_kernel(
        name=f'm3_affine2_pair_i{in_features}_o{out_features}_g{group_size}_k{top_k}_o{opt}',
        input_names=[
            'x',
            'gate_weight',
            'gate_scales',
            'gate_biases',
            'up_weight',
            'up_scales',
            'up_biases',
            'rhs_indices',
            'meta',
        ],
        output_names=['gate_out', 'up_out'],
        source=source,
    )
    grid_x = ((out_features + opt - 1) // opt) * 32

    def gather_pair(x_rows, gate_proj, up_proj, rhs_indices):
        gate_out, up_out = kernel(
            inputs=[
                x_rows,
                gate_proj.weight,
                gate_proj.scales,
                gate_proj.biases,
                up_proj.weight,
                up_proj.scales,
                up_proj.biases,
                rhs_indices,
                meta,
            ],
            output_shapes=[(top_k, out_features), (top_k, out_features)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(grid_x, top_k, 1),
            threadgroup=(min(grid_x, 256), 1, 1),
        )
        return gate_out, up_out

    _KERNEL_CACHE[key] = gather_pair
    return gather_pair


def affine2_switchglu_decode(switch: Any, x: mx.array, indices: mx.array, opt: int = 8) -> mx.array:
    '''Run affine-2 SwitchGLU for a small decode top-k without generic gather_qmm.'''
    top_k = indices.size
    hidden = x.shape[-1]

    idx = indices.reshape(top_k).astype(mx.uint32)
    x_rows = mx.broadcast_to(x.reshape(1, hidden), (top_k, hidden))

    down_kernel = _make_affine2_gather(
        switch.down_proj.input_dims,
        switch.down_proj.output_dims,
        switch.down_proj.group_size,
        top_k,
        opt,
    )

    if (
        switch.gate_proj.output_dims == switch.up_proj.output_dims
        and switch.gate_proj.group_size == switch.up_proj.group_size
    ):
        pair_kernel = _make_affine2_gather_pair(
            hidden, switch.gate_proj.output_dims, switch.gate_proj.group_size, top_k, opt
        )
        gate, up = pair_kernel(x_rows, switch.gate_proj, switch.up_proj, idx)
    else:
        gate_kernel = _make_affine2_gather(
            hidden, switch.gate_proj.output_dims, switch.gate_proj.group_size, top_k, opt
        )
        up_kernel = _make_affine2_gather(
            hidden, switch.up_proj.output_dims, switch.up_proj.group_size, top_k, opt
        )
        gate = gate_kernel(x_rows, switch.gate_proj, idx)
        up = up_kernel(x_rows, switch.up_proj, idx)
    act = switch.activation(up, gate)
    down = down_kernel(act, switch.down_proj, idx)
    return down.reshape(*x.shape[:-1], top_k, switch.down_proj.output_dims)


class MiniMaxM3Affine2SwitchGLU(SwitchGLU):
    'SwitchGLU with a guarded MiniMax-M3 decode fast path.'

    def __call__(self, x, indices):
        if can_use_affine2_switchglu(self, x, indices):
            return affine2_switchglu_decode(self, x, indices)
        return super().__call__(x, indices)

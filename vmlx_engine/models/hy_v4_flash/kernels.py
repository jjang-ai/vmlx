# SPDX-License-Identifier: Apache-2.0
"""Fused Apple Silicon Metal kernels for Hy4-flash (iHC, Clamped SwiGLU, Router Top-16, MXFP4 SwitchGLU)."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# 1. Fused FP32 Identity Hyper-Connections (iHC) Pre-Reduce & Post-Scatter
#    Exact match to Hy4-flash's 4-stream iHC (NO Sinkhorn-Knopp, NO comb matrix).
# ---------------------------------------------------------------------------
_IHC_PRE_SRC = r"""
  // Grid: (256, B * L, 1), TG: (256, 1, 1) (8 simdgroups).
  // x: [BL, 4, D] (T), hc_fn: [8, 4*D] (float), hc_base: [8] (float), hc_scale: [2] (float)
  // Computes:
  //   r = rsqrt(mean(fp32(x)^2) + rms_eps)
  //   mix[m] = dot(fp32(x_flat), hc_fn[m]) * r   (m = 0..7)
  //   g_pre[c]  = sigmoid(mix[c]   * hc_scale[0] + hc_base[c])   + hc_eps
  //   g_post[c] = hc_mag * sigmoid(mix[4+c] * hc_scale[1] + hc_base[4+c]) + hc_eps
  //   u[d] = T(sum_c g_pre[c] * fp32(x[c, d]))
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint row  = threadgroup_position_in_grid.y;

  const uint D4 = uint(D) * 4u;
  const float rms_eps = params[0];
  const float hc_eps  = params[1];
  const float hc_mag  = params[2];

  const size_t x_base = size_t(row) * size_t(D4);

  // Pass 1: sum of squares over 4*D elements + dot products with all 8 rows of hc_fn!
  float sq = 0.0f;
  float dot8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint i = tid; i < D4; i += 256u) {
    float xv = float(x[x_base + i]);
    sq += xv * xv;
    #pragma unroll
    for (uint m = 0u; m < 8u; ++m) {
      dot8[m] = fma(xv, hc_fn[size_t(m) * size_t(D4) + i], dot8[m]);
    }
  }

  threadgroup float sg_sq[8];
  threadgroup float sg_dot[8][8];
  float s_sq = simd_sum(sq);
  if (lane == 0u) sg_sq[sg] = s_sq;
  #pragma unroll
  for (uint m = 0u; m < 8u; ++m) {
    float sd = simd_sum(dot8[m]);
    if (lane == 0u) sg_dot[m][sg] = sd;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float g_pre_sh[4];
  threadgroup float g_post_sh[4];
  if (tid == 0u) {
    float tot_sq = sg_sq[0] + sg_sq[1] + sg_sq[2] + sg_sq[3]
                 + sg_sq[4] + sg_sq[5] + sg_sq[6] + sg_sq[7];
    float r = 1.0f / metal::sqrt(tot_sq / float(D4) + rms_eps);
    float sc_pre  = hc_scale[0];
    float sc_post = hc_scale[1];
    for (uint c = 0u; c < 4u; ++c) {
      float m_pre = (sg_dot[c][0] + sg_dot[c][1] + sg_dot[c][2] + sg_dot[c][3]
                   + sg_dot[c][4] + sg_dot[c][5] + sg_dot[c][6] + sg_dot[c][7]) * r;
      float gp = 1.0f / (1.0f + metal::exp(-(m_pre * sc_pre + hc_base[c]))) + hc_eps;
      g_pre_sh[c] = gp;
      float m_post = (sg_dot[4u + c][0] + sg_dot[4u + c][1] + sg_dot[4u + c][2] + sg_dot[4u + c][3]
                    + sg_dot[4u + c][4] + sg_dot[4u + c][5] + sg_dot[4u + c][6] + sg_dot[4u + c][7]) * r;
      float go = hc_mag / (1.0f + metal::exp(-(m_post * sc_post + hc_base[4u + c]))) + hc_eps;
      g_post_sh[c] = go;
      g_post_out[size_t(row) * 4u + c] = go;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 2: collapse 4 streams -> u[row, 0..D-1]
  const float gp0 = g_pre_sh[0], gp1 = g_pre_sh[1], gp2 = g_pre_sh[2], gp3 = g_pre_sh[3];
  device T* u_row = u_out + size_t(row) * uint(D);
  for (uint d = tid; d < uint(D); d += 256u) {
    float acc = gp0 * float(x[x_base + 0u * uint(D) + d])
              + gp1 * float(x[x_base + 1u * uint(D) + d])
              + gp2 * float(x[x_base + 2u * uint(D) + d])
              + gp3 * float(x[x_base + 3u * uint(D) + d]);
    u_row[d] = T(acc);
  }
"""

_IHC_POST_SRC = r"""
  // Grid: (ceil(D/4/256)*256, B*L, 1), TG: (256, 1, 1).
  // out[row, c, d] = T(fp32(x[row, c, d]) + g_post[row, c] * fp32(y[row, d]))
  const uint d4  = thread_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  const uint D4  = uint(D) >> 2;
  if (d4 >= D4) return;

  const uint d0 = d4 << 2;
  const size_t y_base = size_t(row) * uint(D) + d0;
  const float4 yv = float4(float(y[y_base]), float(y[y_base + 1u]), float(y[y_base + 2u]), float(y[y_base + 3u]));
  const size_t gp_base = size_t(row) * 4u;
  const size_t x_base = size_t(row) * size_t(4u * uint(D)) + d0;
  device T* o_row = out + size_t(row) * size_t(4u * uint(D)) + d0;

  #pragma unroll
  for (uint c = 0u; c < 4u; ++c) {
    const size_t xb = x_base + size_t(c) * uint(D);
    const float4 xv = float4(float(x[xb]), float(x[xb + 1u]), float(x[xb + 2u]), float(x[xb + 3u]));
    const float4 rv = fma(float4(g_post[gp_base + c]), yv, xv);
    device T* oc = o_row + size_t(c) * uint(D);
    oc[0] = T(rv.x); oc[1] = T(rv.y); oc[2] = T(rv.z); oc[3] = T(rv.w);
  }
"""


@lru_cache(maxsize=8)
def _ihc_pre_kernel(dtype_name: str, D: int):
    return mx.fast.metal_kernel(
        name=f"hy4_ihc_pre_{dtype_name}_d{D}",
        input_names=["x", "hc_fn", "hc_base", "hc_scale", "params"],
        output_names=["u_out", "g_post_out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_IHC_PRE_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _ihc_post_kernel(dtype_name: str, D: int):
    return mx.fast.metal_kernel(
        name=f"hy4_ihc_post_{dtype_name}_d{D}",
        input_names=["x", "y", "g_post"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_IHC_POST_SRC,
        ensure_row_contiguous=True,
    )


def hy4_ihc_pre(
    x: mx.array,
    hc_fn: mx.array,
    hc_base: mx.array,
    hc_scale: mx.array,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    hc_magnitude: float = 2.0,
) -> Tuple[mx.array, mx.array]:
    """Single-dispatch FP32 4-stream iHC pre-reduce + g_post computation."""
    B, L, C, D = x.shape
    if (
        C == 4
        and D % 4 == 0
        and 1 <= B * L <= 65535
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
        params = mx.array([float(rms_eps), float(hc_eps), float(hc_magnitude)], dtype=mx.float32)
        u, gp = _ihc_pre_kernel(dn, int(D))(
            inputs=[
                x.reshape(B * L, 4, D),
                hc_fn.astype(mx.float32),
                hc_base.astype(mx.float32),
                hc_scale.astype(mx.float32),
                params,
            ],
            template=[("T", x.dtype), ("D", int(D))],
            grid=(256, B * L, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B * L, D), (B * L, 4)],
            output_dtypes=[x.dtype, mx.float32],
        )
        return u.reshape(B, L, D), gp.reshape(B, L, 4)

    xf = x.astype(mx.float32)
    x_flat = xf.reshape(B, L, C * D)
    r = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + rms_eps)
    mixes = (x_flat @ hc_fn.astype(mx.float32).T) * r
    g_pre = mx.sigmoid(mixes[..., :C] * hc_scale[0].astype(mx.float32) + hc_base[:C].astype(mx.float32)) + hc_eps
    g_post = hc_magnitude * mx.sigmoid(
        mixes[..., C:] * hc_scale[1].astype(mx.float32) + hc_base[C:].astype(mx.float32)
    ) + hc_eps
    u = mx.sum(g_pre[..., None] * xf, axis=2).astype(x.dtype)
    return u, g_post


def hy4_ihc_post(x: mx.array, y: mx.array, g_post: mx.array) -> mx.array:
    """Single-dispatch FP32 4-stream iHC post-scatter residual update."""
    B, L, C, D = x.shape
    if (
        C == 4
        and D % 4 == 0
        and 1 <= B * L <= 65535
        and x.dtype == y.dtype
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
        grid_x = (((D // 4) + 255) // 256) * 256
        out = _ihc_post_kernel(dn, int(D))(
            inputs=[x.reshape(B * L, 4, D), y.reshape(B * L, D), g_post.astype(mx.float32).reshape(B * L, 4)],
            template=[("T", x.dtype), ("D", int(D))],
            grid=(grid_x, B * L, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B * L, 4, D)],
            output_dtypes=[x.dtype],
        )[0]
        return out.reshape(B, L, 4, D)
    return (x.astype(mx.float32) + g_post[..., None] * y.astype(mx.float32)[..., None, :]).astype(x.dtype)


# ---------------------------------------------------------------------------
# 2. Fused Hy4-flash Clamped SwiGLU (silu(clamp(gate, max=10)) * clamp(up, -10, 10))
# ---------------------------------------------------------------------------
_CLAMPED_SWIGLU_SRC = r"""
  const uint idx = thread_position_in_grid.x;
  const uint H = uint(gate_up_shape[1]) >> 1;
  const uint total = uint(gate_up_shape[0]) * H;
  if (idx >= total) return;
  const uint row = idx / H;
  const uint col = idx - row * H;
  const float lim = limit_buf[0];
  const size_t r_base = size_t(row) * size_t(2u * H);
  float g = metal::min(float(gate_up[r_base + col]), lim);
  float u = metal::clamp(float(gate_up[r_base + H + col]), -lim, lim);
  float sg = g / (1.0f + metal::exp(-g));
  out[idx] = T(sg * u);
"""


@lru_cache(maxsize=4)
def _clamped_swiglu_kernel(dtype_name: str):
    return mx.fast.metal_kernel(
        name=f"hy4_clamped_swiglu_{dtype_name}",
        input_names=["gate_up", "limit_buf"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_CLAMPED_SWIGLU_SRC,
        ensure_row_contiguous=True,
    )


def hy4_clamped_swiglu(gate_up: mx.array, limit: float = 10.0) -> mx.array:
    """Exact Hy4-flash routed-expert clamped SwiGLU on fused [..., 2*H] gate_up tensor."""
    lead = gate_up.shape[:-1]
    H = int(gate_up.shape[-1]) // 2
    if (
        gate_up.size > 0
        and gate_up.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        rows = int(gate_up.size) // (2 * H)
        dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[gate_up.dtype]
        out = _clamped_swiglu_kernel(dn)(
            inputs=[gate_up.reshape(rows, 2 * H), mx.array([float(limit)], dtype=mx.float32)],
            template=[("T", gate_up.dtype)],
            grid=(((rows * H + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, H)],
            output_dtypes=[gate_up.dtype],
        )[0]
        return out.reshape(*lead, H)
    g = mx.minimum(gate_up[..., :H].astype(mx.float32), limit)
    u = mx.clip(gate_up[..., H:].astype(mx.float32), -limit, limit)
    return (nn.silu(g) * u).astype(gate_up.dtype)


# ---------------------------------------------------------------------------
# 3. Fused Hy4-flash MoE Router Top-16 (E=512, sigmoid + bias + renorm * 3.998)
# ---------------------------------------------------------------------------
_HY4_ROUTER_TOPK_SRC = r"""
  // 256 threads (8 simdgroups) per token row; each thread holds 2 experts for E=512!
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint row  = threadgroup_position_in_grid.y;

  const size_t s_base = size_t(row) * 512u;
  float r0 = scores[s_base + tid];
  float r1 = scores[s_base + tid + 256u];
  float v0 = r0 + bias[tid];
  float v1 = r1 + bias[tid + 256u];

  threadgroup float sg_v[8];
  threadgroup uint  sg_i[8];
  threadgroup uint  win_i;
  threadgroup uint  picked_i[TOPK];
  threadgroup float picked_r[TOPK];

  for (uint k = 0u; k < uint(TOPK); ++k) {
    float bv = (v0 >= v1) ? v0 : v1;
    uint  bi = (v0 >= v1) ? tid : (tid + 256u);
    float sm = simd_max(bv);
    uint  si = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
    if (lane == 0u) {
      sg_v[sg] = sm;
      sg_i[sg] = si;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float gv = sg_v[0];
      uint  gi = sg_i[0];
      for (uint s = 1u; s < 8u; ++s) {
        if (sg_v[s] > gv || (sg_v[s] == gv && sg_i[s] < gi)) {
          gv = sg_v[s];
          gi = sg_i[s];
        }
      }
      win_i = gi;
      picked_i[k] = gi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = win_i;
    if (tid == w) {
      picked_r[k] = r0;
      v0 = -INFINITY;
    } else if (tid + 256u == w) {
      picked_r[k] = r1;
      v1 = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float r = (lane < uint(TOPK)) ? picked_r[lane] : 0.0f;
    float denom = simd_sum(r) + 1e-20f;
    if (lane < uint(TOPK)) {
      float w = NORM ? (r / denom) : r;
      out_ids[size_t(row) * uint(TOPK) + lane] = picked_i[lane];
      out_w[size_t(row) * uint(TOPK) + lane] = w * scale_buf[0];
    }
  }
"""


@lru_cache(maxsize=8)
def _hy4_router_topk_kernel(top_k: int, norm: bool):
    return mx.fast.metal_kernel(
        name=f"hy4_router_topk_k{top_k}_n{int(norm)}",
        input_names=["scores", "bias", "scale_buf"],
        output_names=["out_ids", "out_w"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_HY4_ROUTER_TOPK_SRC,
        ensure_row_contiguous=True,
    )


def hy4_router_topk(
    x: mx.array,
    gate_weight: mx.array,
    bias: mx.array,
    *,
    top_k: int = 16,
    routed_scaling_factor: float = 3.998,
    norm_topk_prob: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Exact Hy4-flash FP32 sigmoid router + bias top-K + pre-bias renorm * routed_scaling_factor."""
    lead = x.shape[:-1]
    K = int(x.shape[-1])
    E = int(gate_weight.shape[0])
    x2 = x.reshape(-1, K).astype(mx.float32)
    T = int(x2.shape[0])
    scores = mx.sigmoid(x2 @ gate_weight.astype(mx.float32).T)
    if (
        E == 512
        and 1 <= top_k <= 32
        and 1 <= T <= 65535
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        ids, w = _hy4_router_topk_kernel(int(top_k), bool(norm_topk_prob))(
            inputs=[
                scores,
                bias.astype(mx.float32),
                mx.array([float(routed_scaling_factor)], dtype=mx.float32),
            ],
            template=[("TOPK", int(top_k)), ("NORM", bool(norm_topk_prob))],
            grid=(256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, int(top_k)), (T, int(top_k))],
            output_dtypes=[mx.uint32, mx.float32],
        )
        return ids.reshape(*lead, int(top_k)), w.reshape(*lead, int(top_k))
    ids = mx.argpartition(-(scores + bias.astype(mx.float32)), kth=top_k - 1, axis=-1)[..., :top_k]
    w = mx.take_along_axis(scores, ids, axis=-1)
    if norm_topk_prob:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    w = w * float(routed_scaling_factor)
    return ids.astype(mx.uint32).reshape(*lead, int(top_k)), w.reshape(*lead, int(top_k))


# ---------------------------------------------------------------------------
# 4. Native MXFP4 / BF16 Routed SwitchGLU for Hy4-flash LatentMoE
#    Latent width 1536 -> 2*1536 gate_up -> clamped SwiGLU(10.0) -> 1536 down
# ---------------------------------------------------------------------------
class HyV4MXFP4SwitchGLU(nn.Module):
    """Native MXFP4 (bits=4, group_size=32, UE8M0) fused routed SwitchGLU for Hy4-flash."""

    SORT_THRESHOLD = 160

    def __init__(
        self,
        num_experts: int,
        latent_dim: int,
        intermediate_dim: int,
        *,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.latent_dim = int(latent_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.swiglu_limit = float(swiglu_limit)
        self.gu_weight = mx.zeros((num_experts, 2 * intermediate_dim, latent_dim // 8), dtype=mx.uint32)
        self.gu_scales = mx.zeros((num_experts, 2 * intermediate_dim, latent_dim // 32), dtype=mx.uint8)
        self.down_weight = mx.zeros((num_experts, latent_dim, intermediate_dim // 8), dtype=mx.uint32)
        self.down_scales = mx.zeros((num_experts, latent_dim, intermediate_dim // 32), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x: mx.array, indices: mx.array, route_weights: mx.array) -> mx.array:
        dtype = x.dtype
        lead = x.shape[:-1]
        K_in = int(x.shape[-1])
        top_k = int(indices.shape[-1])
        T = int(x.size) // K_in

        x2 = x.reshape(T, K_in)
        idx_2d = indices.reshape(T, top_k).astype(mx.uint32)
        rw_2d = route_weights.reshape(T, top_k).astype(mx.float32)

        if T * top_k >= self.SORT_THRESHOLD:
            order = mx.argsort(idx_2d.reshape(-1))
            inv_order = mx.argsort(order)
            tok_idx = order // top_k
            sorted_exp = idx_2d.reshape(-1)[order]
            x_sorted = x2[tok_idx][:, None, :]
            gu = mx.gather_qmm(
                x_sorted,
                self.gu_weight,
                scales=self.gu_scales,
                rhs_indices=sorted_exp,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=True,
            )[:, 0, :]
            act = hy4_clamped_swiglu(gu, self.swiglu_limit)
            down = mx.gather_qmm(
                act[:, None, :],
                self.down_weight,
                scales=self.down_scales,
                rhs_indices=sorted_exp,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=True,
            )[:, 0, :]
            down = down[inv_order].reshape(T, top_k, K_in)
        else:
            x_exp = x2[:, None, None, :]
            gu = mx.gather_qmm(
                x_exp,
                self.gu_weight,
                scales=self.gu_scales,
                rhs_indices=idx_2d,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=False,
            )[:, :, 0, :]
            act = hy4_clamped_swiglu(gu, self.swiglu_limit)
            down = mx.gather_qmm(
                act[:, :, None, :],
                self.down_weight,
                scales=self.down_scales,
                rhs_indices=idx_2d,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=False,
            )[:, :, 0, :]

        out = mx.sum(down.astype(mx.float32) * rw_2d[..., None], axis=1).astype(dtype)
        return out.reshape(*lead, K_in)


class HyV4DenseSwitchGLU(nn.Module):
    """BF16 / FP8-repacked SwitchGLU for Hy4-flash (`tencent/Hy4-flash` / `-FP8` / unit tests)."""

    def __init__(
        self,
        num_experts: int,
        latent_dim: int,
        intermediate_dim: int,
        *,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.latent_dim = int(latent_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.swiglu_limit = float(swiglu_limit)
        self.gu_weight = mx.zeros((num_experts, 2 * intermediate_dim, latent_dim), dtype=mx.bfloat16)
        self.down_weight = mx.zeros((num_experts, latent_dim, intermediate_dim), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, indices: mx.array, route_weights: mx.array) -> mx.array:
        dtype = x.dtype
        lead = x.shape[:-1]
        K_in = int(x.shape[-1])
        top_k = int(indices.shape[-1])
        T = int(x.size) // K_in

        x2 = x.reshape(T, K_in)
        idx_2d = indices.reshape(T, top_k).astype(mx.uint32)
        rw_2d = route_weights.reshape(T, top_k).astype(mx.float32)

        gu = mx.gather_mm(
            x2[:, None, None, :],
            self.gu_weight.swapaxes(-1, -2),
            rhs_indices=idx_2d,
        )[:, :, 0, :]
        act = hy4_clamped_swiglu(gu, self.swiglu_limit)
        down = mx.gather_mm(
            act[:, :, None, :],
            self.down_weight.swapaxes(-1, -2),
            rhs_indices=idx_2d,
        )[:, :, 0, :]
        out = mx.sum(down.astype(mx.float32) * rw_2d[..., None], axis=1).astype(dtype)
        return out.reshape(*lead, K_in)

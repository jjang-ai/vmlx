# SPDX-License-Identifier: Apache-2.0
"""Fused Apple Silicon Metal kernels for ERNIE 5.5 Flash (mHC, MultiRouter, HCA Compressor, E4M3 rounding, SiTU, lm_head+MultiMax, DSpark Markov)."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# 1. Per-128 power-of-two E4M3 activation rounding (1 dispatch instead of 14 ops)
# ---------------------------------------------------------------------------
_ROUND_E4M3_PER128_SRC = r"""
  // Exact per-128 dynamic power-of-two E4M3FN activation rounding.
  // 32 lanes per 128-element block (4 elements per lane).
  const uint blk = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const size_t base = size_t(blk) * 128u + lane * 4u;

  float v[4];
  float local_max = 1e-10f;
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {
    float val = float(x[base + i]);
    v[i] = val;
    local_max = metal::max(local_max, metal::abs(val));
  }
  float absmax = simd_max(local_max);
  // scale = 2^ceil(log2(absmax / 448.0f))
  float ratio = absmax * (1.0f / 448.0f);
  float exp_f = metal::ceil(metal::log2(ratio));
  float scale = metal::exp2(exp_f);
  float inv_scale = metal::exp2(-exp_f);

  // Exact IEEE FP8 E4M3FN (bias=7, 3 mantissa bits, max normal=448.0, no Inf)
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {
    float q = metal::clamp(v[i] * inv_scale, -448.0f, 448.0f);
    uint bits = as_type<uint>(q);
    uint sign = bits & 0x80000000u;
    uint mag = bits & 0x7FFFFFFFu;
    float r;
    if (mag == 0u) {
      r = 0.0f;
    } else {
      int f32_exp = int((mag >> 23) & 0xFFu) - 127;
      uint f32_mant = mag & 0x007FFFFFu;
      if (f32_exp >= -6) {
        // Normal E4M3FN: keep top 3 mantissa bits with round-to-nearest-even.
        // Mantissa bit 20 is LSB of 3-bit mantissa; bit 19 is round bit.
        uint keep = (mag >> 20) & 0x7u;
        uint rem = mag & 0x000FFFFFu;
        uint round_up = (rem > 0x00080000u) || (rem == 0x00080000u && (keep & 1u) != 0u);
        uint rounded = (mag & 0xFFF00000u) + (round_up ? 0x00100000u : 0u);
        // Clamp to 448.0f (0x43E00000) if rounding overflowed
        if (rounded > 0x43E00000u) rounded = 0x43E00000u;
        r = as_type<float>(sign | rounded);
      } else {
        // Subnormal E4M3FN: values are multiples of 2^-9 (= 1/512).
        float scaled_sub = as_type<float>(mag) * 512.0f;
        float rounded_int = metal::rint(scaled_sub);
        r = (sign ? -1.0f : 1.0f) * (rounded_int * (1.0f / 512.0f));
      }
    }
    out[base + i] = T(r * scale);
  }
"""


@lru_cache(maxsize=4)
def _round_e4m3_kernel(dtype_name: str):
    return mx.fast.metal_kernel(
        name=f"ernie_round_e4m3_per128_{dtype_name}",
        input_names=["x"],
        output_names=["out"],
        source=_ROUND_E4M3_PER128_SRC,
    )


def round_e4m3_per128(x: mx.array) -> mx.array:
    """Exact per-128 dynamic power-of-two E4M3FN activation rounding in 1 Metal dispatch."""
    if x.size == 0:
        return x
    if x.shape[-1] % 128 != 0 or x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        blocks = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 128)
        absmax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), 1e-10)
        scale = mx.power(2.0, mx.ceil(mx.log2(absmax * (1.0 / 448.0))))
        codes = mx.to_fp8(mx.clip(blocks / scale, -448.0, 448.0))
        return (mx.from_fp8(codes, dtype=mx.float32) * scale).reshape(x.shape).astype(x.dtype)
    xc = mx.contiguous(x)
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[xc.dtype]
    n_blocks = xc.size // 128
    return _round_e4m3_kernel(dname)(
        inputs=[xc],
        template=[("T", xc.dtype)],
        grid=(n_blocks * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[xc.dtype],
    )[0]


# ---------------------------------------------------------------------------
# 1b. Fused E4M3-per-128 + JANGT H128 sign-rotation AND E4M3-per-128 + JANGH H32
# ---------------------------------------------------------------------------
_E4M3_INLINE_EXPR = r"""
  float absmax = simd_max(local_max);
  float ratio = absmax * (1.0f / 448.0f);
  float exp_f = metal::ceil(metal::log2(ratio));
  float scale = metal::exp2(exp_f);
  float inv_scale = metal::exp2(-exp_f);
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {
    float q = metal::clamp(v[i] * inv_scale, -448.0f, 448.0f);
    uint bits = as_type<uint>(q);
    uint sign = bits & 0x80000000u;
    uint mag = bits & 0x7FFFFFFFu;
    float r;
    if (mag == 0u) {
      r = 0.0f;
    } else {
      int f32_exp = int((mag >> 23) & 0xFFu) - 127;
      if (f32_exp >= -6) {
        uint keep = (mag >> 20) & 0x7u;
        uint rem = mag & 0x000FFFFFu;
        uint round_up = (rem > 0x00080000u) || (rem == 0x00080000u && (keep & 1u) != 0u);
        uint rounded = (mag & 0xFFF00000u) + (round_up ? 0x00100000u : 0u);
        if (rounded > 0x43E00000u) rounded = 0x43E00000u;
        r = as_type<float>(sign | rounded);
      } else {
        float scaled_sub = as_type<float>(mag) * 512.0f;
        r = (sign ? -1.0f : 1.0f) * (metal::rint(scaled_sub) * (1.0f / 512.0f));
      }
    }
    // Round through bfloat16_t to match round_e4m3_per128(bf16) output bit-for-bit!
    v[i] = float(T(r * scale));
  }
"""


@lru_cache(maxsize=16)
def _e4m3_rot128_kernel(K: int, dtype_name: str, per_row_signs: bool):
    sidx = "sidx[r]" if per_row_signs else "0u"
    src = f"""
  // Fused E4M3FN per-128 rounding + JANGT sign multiply + H128 butterfly in 1 dispatch!
  // Also writes rounded x_q (for outlier columns) AND rotated xr (f32).
  const uint lane = thread_index_in_simdgroup;
  const uint blk  = threadgroup_position_in_grid.x;
  const uint r    = threadgroup_position_in_grid.y;
  const size_t base = size_t(r) * {K}u + blk * 128u + lane * 4u;

  float v[4];
  float local_max = 1e-10f;
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {{
    float val = float(x[base + i]);
    v[i] = val;
    local_max = metal::max(local_max, metal::abs(val));
  }}
  {_E4M3_INLINE_EXPR}
  const device float* sv = s + size_t({sidx}) * {K}u + blk * 128u + lane * 4u;
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {{
    xq[base + i] = T(v[i]);
    v[i] *= sv[i];
  }}
  {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }}
  {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }}
  for (uint d = 1u; d < 32u; d <<= 1) {{
    bool upper = (lane & d) != 0u;
    #pragma unroll
    for (uint i = 0; i < 4u; i++) {{
      float o = simd_shuffle_xor(v[i], d);
      v[i] = upper ? (o - v[i]) : (v[i] + o);
    }}
  }}
  #pragma unroll
  for (uint i = 0; i < 4u; i++) xr[base + i] = v[i] * 0.08838834764831845f;
"""
    ins = ["x", "s"] + (["sidx"] if per_row_signs else [])
    return mx.fast.metal_kernel(
        name=f"ernie_e4m3_rot128_k{K}_{dtype_name}_{int(per_row_signs)}",
        input_names=ins,
        output_names=["xq", "xr"],
        source=src,
    )


def round_e4m3_and_rot128(x: mx.array, signs: mx.array, sidx: mx.array | None = None) -> Tuple[mx.array, mx.array]:
    """Single-dispatch per-128 E4M3 rounding + JANGT H128 rotation -> (x_q, xr_f32)."""
    R, K = x.shape
    xc = mx.contiguous(x)
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[xc.dtype]
    ins = [xc, signs] + ([sidx.astype(mx.uint32)] if sidx is not None else [])
    xq, xr = _e4m3_rot128_kernel(K, dname, sidx is not None)(
        inputs=ins,
        template=[("T", xc.dtype)],
        grid=(32 * (K // 128), R, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(R, K), (R, K)],
        output_dtypes=[xc.dtype, mx.float32],
    )
    return xq, xr


@lru_cache(maxsize=8)
def _e4m3_h32_kernel(K: int, dtype_name: str, out_dtype_name: str):
    src = f"""
  // Fused E4M3FN per-128 rounding + JANGH blockwise H32 rotation in 1 dispatch!
  // 32 lanes handle 128 elements (= four 32-element H32 sub-blocks, 8 lanes per H32 block).
  const uint lane = thread_index_in_simdgroup;
  const uint blk  = threadgroup_position_in_grid.x;
  const uint r    = threadgroup_position_in_grid.y;
  const size_t base = size_t(r) * {K}u + blk * 128u + lane * 4u;

  float v[4];
  float local_max = 1e-10f;
  #pragma unroll
  for (uint i = 0; i < 4u; i++) {{
    float val = float(x[base + i]);
    v[i] = val;
    local_max = metal::max(local_max, metal::abs(val));
  }}
  {_E4M3_INLINE_EXPR}
  // 4-element in-register butterfly + 3 shuffle stages (d=1,2,4) across 8-lane H32 sub-block!
  {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }}
  {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }}
  for (uint d = 1u; d < 8u; d <<= 1) {{
    bool upper = (lane & d) != 0u;
    #pragma unroll
    for (uint i = 0; i < 4u; i++) {{
      float o = simd_shuffle_xor(v[i], d);
      v[i] = upper ? (o - v[i]) : (v[i] + o);
    }}
  }}
  #pragma unroll
  for (uint i = 0; i < 4u; i++) out[base + i] = OT(v[i] * 0.17677669529663687f);
"""
    return mx.fast.metal_kernel(
        name=f"ernie_e4m3_h32_k{K}_{dtype_name}_{out_dtype_name}",
        input_names=["x"],
        output_names=["out"],
        source=src,
    )


def round_e4m3_and_h32(x: mx.array, out_dtype: mx.Dtype = mx.float32) -> mx.array:
    """Single-dispatch per-128 E4M3 rounding + JANGH H32 rotation."""
    R, K = x.shape
    xc = mx.contiguous(x)
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[xc.dtype]
    oname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[out_dtype]
    return _e4m3_h32_kernel(K, dname, oname)(
        inputs=[xc],
        template=[("T", xc.dtype), ("OT", out_dtype)],
        grid=(32 * (K // 128), R, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(R, K)],
        output_dtypes=[out_dtype],
    )[0]


# ---------------------------------------------------------------------------
# 2. Fused ERNIE SiTU activation: beta * tanh(g/beta) * sigmoid(g) * (lb * tanh(u/lb))
# ---------------------------------------------------------------------------
_SITU_SRC = r"""
  const size_t i = thread_position_in_grid.x;
  if (i >= size_t(params[0])) return;
  const float g = float(gate[i]);
  const float u = float(up[i]);
  const float beta = floats[0];
  const float lbeta = floats[1];
  const float sig = 1.0f / (1.0f + metal::exp(-g));
  const float g_act = beta * metal::precise::tanh(g / beta) * sig;
  const float u_act = lbeta * metal::precise::tanh(u / lbeta);
  out[i] = T(g_act * u_act);
"""


@lru_cache(maxsize=4)
def _situ_kernel(dtype_name: str):
    return mx.fast.metal_kernel(
        name=f"ernie_situ_{dtype_name}",
        input_names=["up", "gate", "params", "floats"],
        output_names=["out"],
        source=_SITU_SRC,
    )


def fused_situ(up: mx.array, gate: mx.array, beta: float = 4.0, linear_beta: float = 25.0) -> mx.array:
    """Single-dispatch Metal SiTU activation."""
    if up.size == 0 or up.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        dt = gate.dtype
        g = gate.astype(mx.float32)
        u = up.astype(mx.float32)
        return (
            (beta * mx.tanh(g / beta) * mx.sigmoid(g))
            * (linear_beta * mx.tanh(u / linear_beta))
        ).astype(dt)
    uc = mx.contiguous(up)
    gc = mx.contiguous(gate)
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[gc.dtype]
    n = uc.size
    tpg = 256
    gx = ((n + tpg - 1) // tpg) * tpg
    return _situ_kernel(dname)(
        inputs=[
            uc,
            gc,
            mx.array([n], dtype=mx.uint32),
            mx.array([float(beta), float(linear_beta)], dtype=mx.float32),
        ],
        template=[("T", gc.dtype)],
        grid=(gx, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[gc.shape],
        output_dtypes=[gc.dtype],
    )[0]


# ---------------------------------------------------------------------------
# 3. Fused mHC Sinkhorn + Collapse + RMSNorm AND mHC Expand
# ---------------------------------------------------------------------------
_MHC_SINKHORN_COLLAPSE_NORM_SRC = r"""
  // Fused Sinkhorn (20 iters, HC=4) + 4-stream collapse + RMSNorm(norm_w, norm_eps).
  // 256 threads (8 simdgroups) per token row.
  const uint tid  = thread_position_in_threadgroup.x;
  const uint row  = threadgroup_position_in_grid.x;
  const uint lane = tid & 31u;
  const uint sg   = tid >> 5;

  constexpr int MIX = 24;       // (2 + 4) * 4
  constexpr int BASE_OFF = 8;   // 2 * 4
  const float hc_eps = eps_params[0];
  const float norm_eps = eps_params[1];

  const size_t mix_off    = size_t(row) * MIX;
  device float* post_out  = post + size_t(row) * 4u;
  device float* comb_out  = comb + size_t(row) * 16u;

  threadgroup float pre_shared[4];
  threadgroup float sg_sq[8];

  // Phase 1: Branchless Sinkhorn on simdgroup 0 (4 lanes active)
  if (sg == 0u) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float active = (lane < 4u) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, 3u);

    float pre_z  = mixes[mix_off + llane]      * pre_scale  + base[llane];
    float post_z = mixes[mix_off + 4u + llane] * post_scale + base[4u + llane];
    float pre_v  = 1.0f / (1.0f + metal::exp(-pre_z)) + hc_eps;
    float post_v = 2.0f / (1.0f + metal::exp(-post_z));
    if (lane < 4u) {
      pre_shared[lane] = pre_v;
      post_out[lane]   = post_v;
    }

    const uint b4 = BASE_OFF + llane * 4u;
    float4 mv = float4(mixes[mix_off + b4 + 0u], mixes[mix_off + b4 + 1u],
                       mixes[mix_off + b4 + 2u], mixes[mix_off + b4 + 3u]);
    float4 bv = float4(base[b4 + 0u], base[b4 + 1u], base[b4 + 2u], base[b4 + 3u]);
    float4 v = (mv * comb_scale + bv) * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + hc_eps)) + hc_eps * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + hc_eps);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      r *= (1.0f / (r.x + r.y + r.z + r.w + hc_eps)) * active;
      col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + hc_eps);
      r *= col_inv;
    }
    if (lane < 4u) {
      *(device float4*)(comb_out + lane * 4u) = r;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: Collapse 4 streams and accumulate sum-of-squares for RMSNorm
  const float p0 = pre_shared[0], p1 = pre_shared[1], p2 = pre_shared[2], p3 = pre_shared[3];
  const device T* x_row = x_in + size_t(row) * (4u * uint(D));
  using T4 = vec<T, 4>;
  const device T4* x0 = (const device T4*)(x_row + 0u * uint(D));
  const device T4* x1 = (const device T4*)(x_row + 1u * uint(D));
  const device T4* x2 = (const device T4*)(x_row + 2u * uint(D));
  const device T4* x3 = (const device T4*)(x_row + 3u * uint(D));
  constexpr uint D4 = uint(D) / 4u;
  constexpr uint PER_T = D4 / 256u;   // D=4096 -> D4=1024 -> PER_T=4

  float4 vals[PER_T];
  float sq = 0.0f;
  #pragma unroll
  for (uint k = 0; k < PER_T; k++) {
    uint d4 = tid + k * 256u;
    float4 c = fma(float4(p0), float4(x0[d4]),
               fma(float4(p1), float4(x1[d4]),
               fma(float4(p2), float4(x2[d4]), float4(p3) * float4(x3[d4]))));
    // Round collapsed value through T before RMSNorm to match reference layer(hc -> norm)
    float4 cr = float4(T4(c));
    vals[k] = cr;
    sq += cr.x * cr.x + cr.y * cr.y + cr.z * cr.z + cr.w * cr.w;
  }
  float sg_sum = simd_sum(sq);
  if (lane == 0u) sg_sq[sg] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = 0.0f;
  for (uint i = 0; i < 8u; i++) total_sq += sg_sq[i];
  float inv_rms = metal::rsqrt(total_sq / float(D) + norm_eps);

  const device T4* nw = (const device T4*)norm_w;
  device T4* out4 = (device T4*)(normed_out + size_t(row) * uint(D));
  #pragma unroll
  for (uint k = 0; k < PER_T; k++) {
    uint d4 = tid + k * 256u;
    out4[d4] = T4(vals[k] * inv_rms * float4(nw[d4]));
  }
"""


_MHC_SPLITK_S1_SRC = r"""
  // Stage 1: Split-K=64 over K = 4 * D (= 16384 -> 256 elements per split).
  // 256 threads (8 simdgroups) per threadgroup -> 64 TGs saturating M5 Ultra's 80 cores!
  constexpr uint SPLITS = 64u;
  constexpr uint CHUNK  = uint(K_FLAT) / SPLITS;   // 256 for K_FLAT=16384
  const uint split = threadgroup_position_in_grid.x;
  const uint row   = threadgroup_position_in_grid.y;
  const uint tid   = thread_position_in_threadgroup.x;
  const uint sg    = tid >> 5;
  const uint lane  = tid & 31u;
  const uint k0    = split * CHUNK;

  threadgroup float x_smem[256];
  threadgroup float sq_sg[8];

  const device T* x_row = x_in + size_t(row) * uint(K_FLAT) + k0;
  float v0 = float(x_row[tid]);
  x_smem[tid] = v0;
  float sg_s = simd_sum(v0 * v0);
  if (lane == 0u) sq_sg[sg] = sg_s;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t pbase = (size_t(row) * SPLITS + split) * 25u;
  if (tid == 0u) {
    partials[pbase + 24u] = sq_sg[0] + sq_sg[1] + sq_sg[2] + sq_sg[3]
                          + sq_sg[4] + sq_sg[5] + sq_sg[6] + sq_sg[7];
  }

  // Each simdgroup owns 3 rows of fn (8 * 3 = 24 rows), walking 256 elems in 2x vec4 steps.
  const uint m0 = sg * 3u;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  const device FN_T* f0 = fn + size_t(m0 + 0u) * uint(K_FLAT) + k0;
  const device FN_T* f1 = fn + size_t(m0 + 1u) * uint(K_FLAT) + k0;
  const device FN_T* f2 = fn + size_t(m0 + 2u) * uint(K_FLAT) + k0;
  #pragma unroll
  for (uint j = 0; j < 2u; j++) {
    uint idx = j * 128u + lane * 4u;
    float4 xv = *(const threadgroup float4*)(x_smem + idx);
    a0 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f0 + idx)));
    a1 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f1 + idx)));
    a2 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f2 + idx)));
  }
  float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2);
  if (lane == 0u) {
    partials[pbase + m0 + 0u] = s0;
    partials[pbase + m0 + 1u] = s1;
    partials[pbase + m0 + 2u] = s2;
  }
"""

_MHC_EXP_SPLITK_S1_SRC = r"""
  // Fused hc_expand + Split-K=64 mHC Stage 1: eliminates 85 separate hc_expand dispatches/tok!
  // Split s in [0..63] lies completely inside stream h = s >> 4 at d0 = (s & 15) * 256.
  constexpr uint SPLITS = 64u;
  constexpr uint CHUNK  = uint(K_FLAT) / SPLITS;   // 256 for K_FLAT=16384
  const uint split = threadgroup_position_in_grid.x;
  const uint row   = threadgroup_position_in_grid.y;
  const uint tid   = thread_position_in_threadgroup.x;
  const uint sg    = tid >> 5;
  const uint lane  = tid & 31u;
  const uint k0    = split * CHUNK;
  const uint h     = split >> 4;
  const uint d     = (split & 15u) * 256u + tid;

  threadgroup float x_smem[256];
  threadgroup float sq_sg[8];

  const size_t p_off = size_t(row) * 4u;
  const size_t c_off = size_t(row) * 16u;
  const device T* res_row = res + size_t(row) * uint(K_FLAT);
  const float s_val = float(sub[size_t(row) * uint(D) + d]);
  const float r0 = float(res_row[0u * uint(D) + d]);
  const float r1 = float(res_row[1u * uint(D) + d]);
  const float r2 = float(res_row[2u * uint(D) + d]);
  const float r3 = float(res_row[3u * uint(D) + d]);
  float y = fma(prev_post[p_off + h], s_val,
            fma(prev_comb[c_off + 0u * 4u + h], r0,
            fma(prev_comb[c_off + 1u * 4u + h], r1,
            fma(prev_comb[c_off + 2u * 4u + h], r2, prev_comb[c_off + 3u * 4u + h] * r3))));
  T yt = T(y);
  x_out[size_t(row) * uint(K_FLAT) + k0 + tid] = yt;
  float v0 = float(yt);
  x_smem[tid] = v0;
  float sg_s = simd_sum(v0 * v0);
  if (lane == 0u) sq_sg[sg] = sg_s;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t pbase = (size_t(row) * SPLITS + split) * 25u;
  if (tid == 0u) {
    partials[pbase + 24u] = sq_sg[0] + sq_sg[1] + sq_sg[2] + sq_sg[3]
                          + sq_sg[4] + sq_sg[5] + sq_sg[6] + sq_sg[7];
  }

  const uint m0 = sg * 3u;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  const device FN_T* f0 = fn + size_t(m0 + 0u) * uint(K_FLAT) + k0;
  const device FN_T* f1 = fn + size_t(m0 + 1u) * uint(K_FLAT) + k0;
  const device FN_T* f2 = fn + size_t(m0 + 2u) * uint(K_FLAT) + k0;
  #pragma unroll
  for (uint j = 0; j < 2u; j++) {
    uint idx = j * 128u + lane * 4u;
    float4 xv = *(const threadgroup float4*)(x_smem + idx);
    a0 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f0 + idx)));
    a1 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f1 + idx)));
    a2 += dot(xv, float4(*(const device vec<FN_T, 4>*)(f2 + idx)));
  }
  float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2);
  if (lane == 0u) {
    partials[pbase + m0 + 0u] = s0;
    partials[pbase + m0 + 1u] = s1;
    partials[pbase + m0 + 2u] = s2;
  }
"""

_MHC_SPLITK_S2_SRC = r"""
  // Stage 2: Reduce 64 split-K partials -> ERNIE exact RMS div -> 20-iter Sinkhorn
  // -> 4-stream collapse -> RMSNorm(norm_w, NORM_EPS). 256 threads per token row.
  constexpr uint SPLITS = 64u;
  constexpr int BASE_OFF = 8;
  const uint tid  = thread_position_in_threadgroup.x;
  const uint row  = threadgroup_position_in_grid.x;
  const uint lane = tid & 31u;
  const uint sg   = tid >> 5;
  const float hc_eps = __HC_EPS__;
  const float norm_eps = __NORM_EPS__;

  threadgroup float mix_s[24];
  threadgroup float inv_rms_x_sh;
  threadgroup float pre_shared[4];
  threadgroup float sg_sq[8];

  const size_t rbase = size_t(row) * SPLITS * 25u;
  // All 8 simdgroups reduce 3 mix rows each in parallel across 64 splits (2 splits/lane!)
  const uint m0 = sg * 3u;
  const size_t p_lo = rbase + size_t(lane) * 25u + m0;
  const size_t p_hi = rbase + size_t(lane + 32u) * 25u + m0;
  float d0 = simd_sum(partials[p_lo + 0u] + partials[p_hi + 0u]);
  float d1 = simd_sum(partials[p_lo + 1u] + partials[p_hi + 1u]);
  float d2 = simd_sum(partials[p_lo + 2u] + partials[p_hi + 2u]);
  if (lane == 0u) {
    mix_s[m0 + 0u] = d0;
    mix_s[m0 + 1u] = d1;
    mix_s[m0 + 2u] = d2;
  }
  if (sg == 0u) {
    float sq_part = partials[rbase + size_t(lane) * 25u + 24u]
                  + partials[rbase + size_t(lane + 32u) * 25u + 24u];
    float tot_sq = simd_sum(sq_part);
    if (lane == 0u) {
      inv_rms_x_sh = 1.0f / (metal::sqrt(tot_sq * (1.0f / float(4u * uint(D)))) + norm_eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device float* post_out = post + size_t(row) * 4u;
  device float* comb_out = comb + size_t(row) * 16u;

  if (sg == 0u) {
    const float inv_rms_x  = inv_rms_x_sh;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float active = (lane < 4u) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, 3u);

    float pre_z  = (mix_s[llane]      * inv_rms_x) * pre_scale  + base[llane];
    float post_z = (mix_s[4u + llane] * inv_rms_x) * post_scale + base[4u + llane];
    float pre_v  = 1.0f / (1.0f + metal::exp(-pre_z)) + hc_eps;
    float post_v = 2.0f / (1.0f + metal::exp(-post_z));
    if (lane < 4u) {
      pre_shared[lane] = pre_v;
      post_out[lane]   = post_v;
    }

    const uint b4 = BASE_OFF + llane * 4u;
    float4 mv = float4(mix_s[b4 + 0u], mix_s[b4 + 1u], mix_s[b4 + 2u], mix_s[b4 + 3u]) * inv_rms_x;
    float4 bv = float4(base[b4 + 0u], base[b4 + 1u], base[b4 + 2u], base[b4 + 3u]);
    float4 v = (mv * comb_scale + bv) * active;
    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + hc_eps)) + hc_eps * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + hc_eps);
    r *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      r *= (1.0f / (r.x + r.y + r.z + r.w + hc_eps)) * active;
      col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + hc_eps);
      r *= col_inv;
    }
    if (lane < 4u) {
      *(device float4*)(comb_out + lane * 4u) = r;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: Collapse 4 streams and accumulate sum-of-squares for RMSNorm
  const float p0 = pre_shared[0], p1 = pre_shared[1], p2 = pre_shared[2], p3 = pre_shared[3];
  const device T* x_row = x_in + size_t(row) * (4u * uint(D));
  using T4 = vec<T, 4>;
  const device T4* x0 = (const device T4*)(x_row + 0u * uint(D));
  const device T4* x1 = (const device T4*)(x_row + 1u * uint(D));
  const device T4* x2 = (const device T4*)(x_row + 2u * uint(D));
  const device T4* x3 = (const device T4*)(x_row + 3u * uint(D));
  constexpr uint D4 = uint(D) / 4u;
  constexpr uint PER_T = D4 / 256u;

  float4 vals[PER_T];
  float sq = 0.0f;
  #pragma unroll
  for (uint k = 0; k < PER_T; k++) {
    uint d4 = tid + k * 256u;
    float4 c = fma(float4(p0), float4(x0[d4]),
               fma(float4(p1), float4(x1[d4]),
               fma(float4(p2), float4(x2[d4]), float4(p3) * float4(x3[d4]))));
    float4 cr = float4(T4(c));
    vals[k] = cr;
    sq += cr.x * cr.x + cr.y * cr.y + cr.z * cr.z + cr.w * cr.w;
  }
  float sg_sum = simd_sum(sq);
  if (lane == 0u) sg_sq[sg] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = 0.0f;
  for (uint i = 0; i < 8u; i++) total_sq += sg_sq[i];
  float inv_rms = metal::rsqrt(total_sq / float(D) + norm_eps);

  const device T4* nw = (const device T4*)norm_w;
  device T4* out4 = (device T4*)(normed_out + size_t(row) * uint(D));
  #pragma unroll
  for (uint k = 0; k < PER_T; k++) {
    uint d4 = tid + k * 256u;
    out4[d4] = T4(vals[k] * inv_rms * float4(nw[d4]));
  }
"""


@lru_cache(maxsize=8)
def _mhc_splitk_s1_kernel(dtype_name: str, fn_dtype_name: str, K_flat: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_sk64_1_{dtype_name}_{fn_dtype_name}_k{K_flat}",
        input_names=["x_in", "fn"],
        output_names=["partials"],
        source=_MHC_SPLITK_S1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _mhc_exp_splitk_s1_kernel(dtype_name: str, fn_dtype_name: str, D: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_expsk64_1_{dtype_name}_{fn_dtype_name}_d{D}",
        input_names=["sub", "res", "prev_post", "prev_comb", "fn"],
        output_names=["x_out", "partials"],
        source=_MHC_EXP_SPLITK_S1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _mhc_splitk_s2_kernel(dtype_name: str, D: int, iters: int, hc_eps: float, norm_eps: float):
    src = _MHC_SPLITK_S2_SRC.replace("__HC_EPS__", f"{float(hc_eps)!r}f").replace("__NORM_EPS__", f"{float(norm_eps)!r}f")
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_sk64_2_{dtype_name}_d{D}_i{iters}",
        input_names=["x_in", "partials", "scale", "base", "norm_w"],
        output_names=["normed_out", "post", "comb"],
        source=src,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mhc_sinkhorn_collapse_norm_kernel(dtype_name: str, D: int, iters: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_scn_{dtype_name}_d{D}_i{iters}",
        input_names=["x_in", "mixes", "scale", "base", "norm_w", "eps_params"],
        output_names=["normed_out", "post", "comb"],
        source=_MHC_SINKHORN_COLLAPSE_NORM_SRC,
        ensure_row_contiguous=True,
    )


def ernie_mhc_collapse_norm(
    x: mx.array,
    fn: mx.array,
    scale: mx.array,
    base: mx.array,
    norm_w: mx.array,
    hc_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Fused ERNIE mHC (epsilon outside sqrt!) + Sinkhorn + collapse + sublayer RMSNorm."""
    B, L, H, D = x.shape
    T = B * L
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype in (mx.float32, mx.bfloat16):
        fn_dn = "bf16" if fn.dtype == mx.bfloat16 else "f32"
        partials = _mhc_splitk_s1_kernel(dname, fn_dn, H * D)(
            inputs=[x, fn],
            template=[("T", x.dtype), ("FN_T", fn.dtype), ("K_FLAT", H * D)],
            grid=(64 * 256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, 64, 25)],
            output_dtypes=[mx.float32],
        )[0]
        normed, post, comb = _mhc_splitk_s2_kernel(dname, D, hc_iters, hc_eps, norm_eps)(
            inputs=[x, partials, scale, base, norm_w],
            template=[("T", x.dtype), ("D", D), ("ITERS", hc_iters)],
            grid=(T * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (B, L, H), (B, L, H, H)],
            output_dtypes=[x.dtype, mx.float32, mx.float32],
        )
        return normed, post, comb

    # Prefill (T > 16): GEMM is compute-bound and saturates all GPU cores.
    flat = x.astype(mx.float32).reshape(B, L, H * D)
    rms = mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + norm_eps
    fn_f32 = fn if fn.dtype == mx.float32 else fn.astype(mx.float32)
    mixes = (flat @ fn_f32.T) / rms
    kern = _mhc_sinkhorn_collapse_norm_kernel(dname, D, hc_iters)
    normed, post, comb = kern(
        inputs=[
            x,
            mixes,
            scale,
            base,
            norm_w,
            mx.array([hc_eps, norm_eps], dtype=mx.float32),
        ],
        template=[("T", x.dtype), ("D", D), ("ITERS", hc_iters)],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, H), (B, L, H, H)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )
    return normed, post, comb


def ernie_hc_expand_collapse_norm(
    sub: mx.array,
    residual: mx.array,
    prev_post: mx.array,
    prev_comb: mx.array,
    fn: mx.array,
    scale: mx.array,
    base: mx.array,
    norm_w: mx.array,
    hc_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused hc_expand + next mHC collapse_and_norm: eliminates 85 hc_expand dispatches/tok!"""
    B, L, H, D = residual.shape
    T = B * L
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[residual.dtype]
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype in (mx.float32, mx.bfloat16):
        fn_dn = "bf16" if fn.dtype == mx.bfloat16 else "f32"
        x_new, partials = _mhc_exp_splitk_s1_kernel(dname, fn_dn, D)(
            inputs=[sub, residual, prev_post, prev_comb, fn],
            template=[("T", residual.dtype), ("FN_T", fn.dtype), ("D", D), ("K_FLAT", H * D)],
            grid=(64 * 256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[residual.shape, (T, 64, 25)],
            output_dtypes=[residual.dtype, mx.float32],
        )
        normed, post, comb = _mhc_splitk_s2_kernel(dname, D, hc_iters, hc_eps, norm_eps)(
            inputs=[x_new, partials, scale, base, norm_w],
            template=[("T", residual.dtype), ("D", D), ("ITERS", hc_iters)],
            grid=(T * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (B, L, H), (B, L, H, H)],
            output_dtypes=[residual.dtype, mx.float32, mx.float32],
        )
        return x_new, normed, post, comb
    x_new = ernie_hc_expand(sub, residual, prev_post, prev_comb)
    normed, post, comb = ernie_mhc_collapse_norm(
        x_new, fn, scale, base, norm_w, hc_iters=hc_iters, hc_eps=hc_eps, norm_eps=norm_eps
    )
    return x_new, normed, post, comb


# ---------------------------------------------------------------------------
# 3b. Fused Split-K HyperHead (eliminates 0.48 ms 1-TG FP32 GEMV stall on target & DSpark!)
# ---------------------------------------------------------------------------
_HC_HEAD_S1_SRC = r"""
  // Split-K=32 over K_FLAT=16384 -> 512 elements per split, 128 threads (4 simdgroups, 1 row/sg).
  constexpr uint SPLITS = 32u;
  constexpr uint CHUNK  = uint(K_FLAT) / SPLITS;   // 512
  const uint split = threadgroup_position_in_grid.x;
  const uint row   = threadgroup_position_in_grid.y;
  const uint tid   = thread_position_in_threadgroup.x;
  const uint sg    = tid >> 5;
  const uint lane  = tid & 31u;
  const uint k0    = split * CHUNK;

  threadgroup float x_smem[512];
  threadgroup float sq_sg[4];

  const device T* x_row = x_in + size_t(row) * uint(K_FLAT) + k0;
  float v0 = float(x_row[tid * 4u + 0u]);
  float v1 = float(x_row[tid * 4u + 1u]);
  float v2 = float(x_row[tid * 4u + 2u]);
  float v3 = float(x_row[tid * 4u + 3u]);
  *(threadgroup float4*)(x_smem + tid * 4u) = float4(v0, v1, v2, v3);
  float sg_s = simd_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
  if (lane == 0u) sq_sg[sg] = sg_s;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t pbase = (size_t(row) * SPLITS + split) * 5u;
  if (tid == 0u) {
    partials[pbase + 4u] = sq_sg[0] + sq_sg[1] + sq_sg[2] + sq_sg[3];
  }
  const device FN_T* fm = fn + size_t(sg) * uint(K_FLAT) + k0;
  float acc = 0.0f;
  #pragma unroll
  for (uint j = 0; j < 4u; j++) {
    uint idx = j * 128u + lane * 4u;
    acc += dot(*(const threadgroup float4*)(x_smem + idx), float4(*(const device vec<FN_T, 4>*)(fm + idx)));
  }
  float s = simd_sum(acc);
  if (lane == 0u) partials[pbase + sg] = s;
"""

_HC_HEAD_S2_SRC = r"""
  // Stage 2: reduce 32 splits -> rsqrt(mean(x^2) + norm_eps) -> sigmoid -> 4-stream collapse!
  constexpr uint SPLITS = 32u;
  const uint tid  = thread_position_in_threadgroup.x;
  const uint row  = threadgroup_position_in_grid.x;
  const uint lane = tid & 31u;
  const uint sg   = tid >> 5;
  const float hc_eps = __HC_EPS__;
  const float norm_eps = __NORM_EPS__;

  threadgroup float pre_sh[4];
  threadgroup float inv_rms_sh;

  const size_t rbase = size_t(row) * SPLITS * 5u;
  if (sg == 0u) {
    float tot_sq = simd_sum(partials[rbase + size_t(lane) * 5u + 4u]);
    if (lane == 0u) {
      inv_rms_sh = metal::rsqrt(tot_sq * (1.0f / float(4u * uint(D))) + norm_eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 4u) {
    float dot_v = simd_sum(partials[rbase + size_t(lane) * 5u + sg]);
    if (lane == 0u) {
      float z = (dot_v * inv_rms_sh) * scale[0] + base[sg];
      pre_sh[sg] = 1.0f / (1.0f + metal::exp(-z)) + hc_eps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float p0 = pre_sh[0], p1 = pre_sh[1], p2 = pre_sh[2], p3 = pre_sh[3];
  const device T* x_row = x_in + size_t(row) * (4u * uint(D));
  using T4 = vec<T, 4>;
  const device T4* x0 = (const device T4*)(x_row + 0u * uint(D));
  const device T4* x1 = (const device T4*)(x_row + 1u * uint(D));
  const device T4* x2 = (const device T4*)(x_row + 2u * uint(D));
  const device T4* x3 = (const device T4*)(x_row + 3u * uint(D));
  device T4* out4 = (device T4*)(out + size_t(row) * uint(D));
  constexpr uint D4 = uint(D) / 4u;
  constexpr uint PER_T = D4 / 256u;
  #pragma unroll
  for (uint k = 0; k < PER_T; k++) {
    uint d4 = tid + k * 256u;
    float4 c = fma(float4(p0), float4(x0[d4]),
               fma(float4(p1), float4(x1[d4]),
               fma(float4(p2), float4(x2[d4]), float4(p3) * float4(x3[d4]))));
    out4[d4] = T4(c);
  }
"""


@lru_cache(maxsize=8)
def _hc_head_s1_kernel(dtype_name: str, fn_dtype_name: str, K_flat: int):
    return mx.fast.metal_kernel(
        name=f"ernie_hchead_s1_{dtype_name}_{fn_dtype_name}_k{K_flat}",
        input_names=["x_in", "fn"],
        output_names=["partials"],
        source=_HC_HEAD_S1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _hc_head_s2_kernel(dtype_name: str, D: int, hc_eps: float, norm_eps: float):
    src = _HC_HEAD_S2_SRC.replace("__HC_EPS__", f"{float(hc_eps)!r}f").replace("__NORM_EPS__", f"{float(norm_eps)!r}f")
    return mx.fast.metal_kernel(
        name=f"ernie_hchead_s2_{dtype_name}_d{D}",
        input_names=["x_in", "partials", "scale", "base"],
        output_names=["out"],
        source=src,
        ensure_row_contiguous=True,
    )


def ernie_hc_head_forward(
    x: mx.array,
    fn: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> mx.array:
    """Split-K=32 fused HyperHead (matches mx.fast.rms_norm eps-inside-sqrt)."""
    B, L, H, D = x.shape
    T = B * L
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype in (mx.float32, mx.bfloat16) and x.dtype in (mx.bfloat16, mx.float16, mx.float32):
        dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
        fn_dn = "bf16" if fn.dtype == mx.bfloat16 else "f32"
        partials = _hc_head_s1_kernel(dname, fn_dn, H * D)(
            inputs=[x, fn],
            template=[("T", x.dtype), ("FN_T", fn.dtype), ("K_FLAT", H * D)],
            grid=(32 * 128, T, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 32, 5)],
            output_dtypes=[mx.float32],
        )[0]
        return _hc_head_s2_kernel(dname, D, hc_eps, norm_eps)(
            inputs=[x, partials, scale, base],
            template=[("T", x.dtype), ("D", D)],
            grid=(T * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D)],
            output_dtypes=[x.dtype],
        )[0]
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, norm_eps)
    fn_f32 = fn if fn.dtype == mx.float32 else fn.astype(mx.float32)
    mixes = z @ fn_f32.T
    pre = mx.sigmoid(mixes * scale + base) + hc_eps
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype)


_HC_EXPAND_SRC = r"""
  // Fused hc_expand for HC=4:
  // out[row, h, d] = T(post[row, h] * float(sub[row, d]) + sum_j comb[row, j, h] * float(res[row, j, d]))
  const uint row = threadgroup_position_in_grid.y;
  const uint d4  = threadgroup_position_in_grid.x * 256u + thread_position_in_threadgroup.x;
  constexpr uint D4 = uint(D) / 4u;
  if (d4 >= D4) return;

  using T4 = vec<T, 4>;
  const size_t p_off = size_t(row) * 4u;
  const size_t c_off = size_t(row) * 16u;
  const float4 s = float4(((const device T4*)(sub + size_t(row) * uint(D)))[d4]);
  const device T* res_row = res + size_t(row) * (4u * uint(D));
  const float4 r0 = float4(((const device T4*)(res_row + 0u * uint(D)))[d4]);
  const float4 r1 = float4(((const device T4*)(res_row + 1u * uint(D)))[d4]);
  const float4 r2 = float4(((const device T4*)(res_row + 2u * uint(D)))[d4]);
  const float4 r3 = float4(((const device T4*)(res_row + 3u * uint(D)))[d4]);

  device T* out_row = out + size_t(row) * (4u * uint(D));
  #pragma unroll
  for (uint h = 0; h < 4u; h++) {
    // comb^T[h, j] = comb[j, h] = comb[c_off + j * 4 + h]
    float4 y = fma(float4(post[p_off + h]), s,
               fma(float4(comb[c_off + 0u * 4u + h]), r0,
               fma(float4(comb[c_off + 1u * 4u + h]), r1,
               fma(float4(comb[c_off + 2u * 4u + h]), r2, float4(comb[c_off + 3u * 4u + h]) * r3))));
    ((device T4*)(out_row + h * uint(D)))[d4] = T4(y);
  }
"""


@lru_cache(maxsize=4)
def _hc_expand_kernel(dtype_name: str, D: int):
    return mx.fast.metal_kernel(
        name=f"ernie_hc_expand_{dtype_name}_d{D}",
        input_names=["sub", "res", "post", "comb"],
        output_names=["out"],
        source=_HC_EXPAND_SRC,
        ensure_row_contiguous=True,
    )


def ernie_hc_expand(sub: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """Single-dispatch Metal mHC residual expansion (zero FP32 [B,L,4,4096] temp)."""
    B, L, H, D = residual.shape
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[residual.dtype]
    D4 = D // 4
    grid_x = ((D4 + 255) // 256) * 256
    return _hc_expand_kernel(dname, D)(
        inputs=[sub, residual, post, comb],
        template=[("T", residual.dtype), ("D", D)],
        grid=(grid_x, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[residual.dtype],
    )[0]


# ---------------------------------------------------------------------------
# 4. Direct BF16-weight FP32-accum In-TG Split-K GEMV (Router, HCA Compressor)
#    + High-occupancy M5 Ultra MXFP8 LUT GEMV (qa_kv_fused, shared_fc1/fc2, wo_b)
# ---------------------------------------------------------------------------
_BF16_MATVEC_FP32_SRC = r"""
  // Single-dispatch 4-simdgroup in-TG Split-K over K (1 output row/TG -> 4,096 sgs for N=1024!).
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row = threadgroup_position_in_grid.x;
  if (row >= uint(N)) return;

  threadgroup float sh[ROWS][4];
  float acc[ROWS];
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) acc[t] = 0.0f;

  const device bfloat16_t* wr = w + size_t(row) * uint(K);
  for (uint k = sgi * 256u + lane * 8u; k < uint(K); k += 1024u) {
    float4 w_lo = float4(*(const device vec<bfloat16_t, 4>*)(wr + k));
    float4 w_hi = float4(*(const device vec<bfloat16_t, 4>*)(wr + k + 4u));
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      const device X_T* xp = x + size_t(t) * uint(K) + k;
      float4 xv_lo = float4(float(xp[0]), float(xp[1]), float(xp[2]), float(xp[3]));
      float4 xv_hi = float4(float(xp[4]), float(xp[5]), float(xp[6]), float(xp[7]));
      acc[t] += dot(xv_lo, w_lo) + dot(xv_hi, w_hi);
    }
  }
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) {
    float s = simd_sum(acc[t]);
    if (lane == 0u) sh[t][sgi] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u && lane == 0u) {
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      out[size_t(t) * uint(N) + row] = sh[t][0] + sh[t][1] + sh[t][2] + sh[t][3];
    }
  }
"""


@lru_cache(maxsize=32)
def _bf16_matvec_fp32_kernel(xdt_name: str, rows: int, N: int, K: int):
    return mx.fast.metal_kernel(
        name=f"ernie_bf16_mv4sk_f32_{xdt_name}_r{rows}_n{N}_k{K}",
        input_names=["x", "w"],
        output_names=["out"],
        source=_BF16_MATVEC_FP32_SRC,
        ensure_row_contiguous=True,
    )


def bf16_linear_fp32(x: mx.array, weight_bf16: mx.array) -> mx.array:
    """Compute x.f32 @ weight_bf16.f32.T with 4-sg in-TG Split-K (saturates M5 Ultra)."""
    lead = x.shape[:-1]
    K = x.shape[-1]
    N = weight_bf16.shape[0]
    x2 = x.reshape(-1, K)
    T = x2.shape[0]
    if 1 <= T <= 15 and K % 1024 == 0 and weight_bf16.dtype == mx.bfloat16 and x2.dtype in (mx.bfloat16, mx.float16, mx.float32):
        xdt_name = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x2.dtype]
        if T <= 8:
            kern = _bf16_matvec_fp32_kernel(xdt_name, T, N, K)
            out = kern(
                inputs=[x2, weight_bf16],
                template=[("X_T", x2.dtype), ("ROWS", T), ("N", N), ("K", K)],
                grid=(N * 128, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(T, N)],
                output_dtypes=[mx.float32],
            )[0]
            return out.reshape(*lead, N)
        parts = [bf16_linear_fp32(x2[i : i + 8], weight_bf16) for i in range(0, T, 8)]
        return mx.concatenate(parts, axis=0).reshape(*lead, N)
    return (x2.astype(mx.bfloat16) @ weight_bf16.T).astype(mx.float32).reshape(*lead, N)


# ---------------------------------------------------------------------------
# 5. Fused MultiRouter: Single-dispatch 8-simdgroup in-TG Split-K (4,096 sgs for N=512!)
# ---------------------------------------------------------------------------
_MULTI_ROUTER_SRC = r"""
  // Single-dispatch 8-simdgroup in-TG Split-K over K=4096 (1 row/TG -> 512 TGs x 8 sg = 4,096 sgs!).
  // Each simdgroup handles 512 elements (16 elems/lane = 4x bfloat16_t4 coalesced loads!).
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row = threadgroup_position_in_grid.x;
  if (row >= uint(N)) return;

  threadgroup float sh0[ROWS][8];
  threadgroup float sh1[ROWS][8];
  float a0[ROWS], a1[ROWS];
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) { a0[t] = 0.0f; a1[t] = 0.0f; }

  const device bfloat16_t* w0r = w0 + size_t(row) * uint(K);
  const device bfloat16_t* w1r = w1 + size_t(row) * uint(K);

  for (uint k = sgi * 512u + lane * 16u; k < uint(K); k += 4096u) {
    float4 v0_0 = float4(*(const device vec<bfloat16_t, 4>*)(w0r + k + 0u));
    float4 v0_1 = float4(*(const device vec<bfloat16_t, 4>*)(w0r + k + 4u));
    float4 v0_2 = float4(*(const device vec<bfloat16_t, 4>*)(w0r + k + 8u));
    float4 v0_3 = float4(*(const device vec<bfloat16_t, 4>*)(w0r + k + 12u));
    float4 v1_0 = float4(*(const device vec<bfloat16_t, 4>*)(w1r + k + 0u));
    float4 v1_1 = float4(*(const device vec<bfloat16_t, 4>*)(w1r + k + 4u));
    float4 v1_2 = float4(*(const device vec<bfloat16_t, 4>*)(w1r + k + 8u));
    float4 v1_3 = float4(*(const device vec<bfloat16_t, 4>*)(w1r + k + 12u));
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      const device X_T* xp = x + size_t(t) * uint(K) + k;
      float4 x0 = float4(float(xp[0]),  float(xp[1]),  float(xp[2]),  float(xp[3]));
      float4 x1 = float4(float(xp[4]),  float(xp[5]),  float(xp[6]),  float(xp[7]));
      float4 x2 = float4(float(xp[8]),  float(xp[9]),  float(xp[10]), float(xp[11]));
      float4 x3 = float4(float(xp[12]), float(xp[13]), float(xp[14]), float(xp[15]));
      a0[t] += dot(x0, v0_0) + dot(x1, v0_1) + dot(x2, v0_2) + dot(x3, v0_3);
      a1[t] += dot(x0, v1_0) + dot(x1, v1_1) + dot(x2, v1_2) + dot(x3, v1_3);
    }
  }
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) {
    float s0 = simd_sum(a0[t]);
    float s1 = simd_sum(a1[t]);
    if (lane == 0u) {
      sh0[t][sgi] = s0;
      sh1[t][sgi] = s1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u && lane == 0u) {
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      float t0 = sh0[t][0] + sh0[t][1] + sh0[t][2] + sh0[t][3]
               + sh0[t][4] + sh0[t][5] + sh0[t][6] + sh0[t][7];
      float t1 = sh1[t][0] + sh1[t][1] + sh1[t][2] + sh1[t][3]
               + sh1[t][4] + sh1[t][5] + sh1[t][6] + sh1[t][7];
      scores[size_t(t) * uint(N) + row] = (1.0f / (1.0f + metal::exp(-t0))) + (1.0f / (1.0f + metal::exp(-t1)));
    }
  }
"""


@lru_cache(maxsize=16)
def _multi_router_kernel(xdt_name: str, rows: int, N: int, K: int):
    return mx.fast.metal_kernel(
        name=f"ernie_multi_router8sk_{xdt_name}_r{rows}_n{N}_k{K}",
        input_names=["x", "w0", "w1"],
        output_names=["scores"],
        source=_MULTI_ROUTER_SRC,
        ensure_row_contiguous=True,
    )


def ernie_multi_router_scores(x: mx.array, w0: mx.array, w1: mx.array) -> mx.array:
    """Compute sigmoid(x @ w0.T) + sigmoid(x @ w1.T) in FP32 with 8-sg in-TG Split-K."""
    lead = x.shape[:-1]
    K = x.shape[-1]
    N = w0.shape[0]
    x2 = x.reshape(-1, K)
    T = x2.shape[0]
    if 1 <= T <= 15 and K % 4096 == 0 and w0.dtype == mx.bfloat16 and w1.dtype == mx.bfloat16:
        if T <= 8:
            xdt_name = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x2.dtype]
            out = _multi_router_kernel(xdt_name, T, N, K)(
                inputs=[x2, w0, w1],
                template=[("X_T", x2.dtype), ("ROWS", T), ("N", N), ("K", K)],
                grid=(N * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(T, N)],
                output_dtypes=[mx.float32],
            )[0]
            return out.reshape(*lead, N)
        parts = [ernie_multi_router_scores(x2[i : i + 8], w0, w1) for i in range(0, T, 8)]
        return mx.concatenate(parts, axis=0).reshape(*lead, N)
    x_bf = x2.astype(mx.bfloat16)
    s0 = mx.sigmoid((x_bf @ w0.T).astype(mx.float32))
    s1 = mx.sigmoid((x_bf @ w1.T).astype(mx.float32))
    return (s0 + s1).reshape(*lead, N)


# ---------------------------------------------------------------------------
# 5b. Fused MultiRouter Top-K + Normalize + Expert Scale (replaces 7 ops x 42 layers!)
# ---------------------------------------------------------------------------
_MULTI_ROUTER_TOPK_SRC = r"""
  // Exact top-K selection over N=512 experts + optional L1 normalize + routed_scaling_factor_param[id].
  // 256 threads (8 simdgroups) per token row; each thread holds 2 experts in registers!
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint row  = threadgroup_position_in_grid.y;

  const device float* s_row = scores + size_t(row) * 512u;
  float r0 = s_row[tid];
  float r1 = s_row[tid + 256u];
  float v0 = r0 + bias[tid];
  float v1 = r1 + bias[tid + 256u];

  threadgroup float sg_val[8];
  threadgroup uint  sg_idx[8];
  threadgroup uint  win_idx_sh;
  threadgroup uint  picked_idx[TOPK];
  threadgroup float picked_raw[TOPK];

  for (uint k = 0u; k < uint(TOPK); k++) {
    float bv = (v0 >= v1) ? v0 : v1;
    uint  bi = (v0 >= v1) ? tid : (tid + 256u);
    float sm = simd_max(bv);
    uint  si = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
    if (lane == 0u) {
      sg_val[sg] = sm;
      sg_idx[sg] = si;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float best_v = sg_val[0];
      uint  best_i = sg_idx[0];
      for (uint s = 1u; s < 8u; s++) {
        if (sg_val[s] > best_v || (sg_val[s] == best_v && sg_idx[s] < best_i)) {
          best_v = sg_val[s];
          best_i = sg_idx[s];
        }
      }
      win_idx_sh = best_i;
      picked_idx[k] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = win_idx_sh;
    if (tid == w) {
      picked_raw[k] = r0;
      v0 = -INFINITY;
    } else if (tid + 256u == w) {
      picked_raw[k] = r1;
      v1 = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float raw = (lane < uint(TOPK)) ? picked_raw[lane] : 0.0f;
    float denom = metal::max(simd_sum(raw), 1e-20f);
    if (lane < uint(TOPK)) {
      uint id = picked_idx[lane];
      float w = NORM ? (raw / denom) : raw;
      out_ids[size_t(row) * uint(TOPK) + lane] = id;
      out_w[size_t(row) * uint(TOPK) + lane] = w * scale_param[id];
    }
  }
"""


@lru_cache(maxsize=8)
def _multi_router_topk_kernel(top_k: int, normalize: bool):
    return mx.fast.metal_kernel(
        name=f"ernie_multi_router_topk_k{top_k}_n{int(normalize)}",
        input_names=["scores", "bias", "scale_param"],
        output_names=["out_ids", "out_w"],
        source=_MULTI_ROUTER_TOPK_SRC,
        ensure_row_contiguous=True,
    )


def ernie_multi_router_topk(
    scores: mx.array,
    bias: mx.array,
    scale_param: mx.array,
    top_k: int = 10,
    normalize: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Single-dispatch exact top-K + optional L1 normalize + routed scaling factor."""
    lead = scores.shape[:-1]
    N = scores.shape[-1]
    s2 = scores.reshape(-1, N)
    T = s2.shape[0]
    if N == 512 and 1 <= top_k <= 32 and 1 <= T <= 65535 and s2.dtype == mx.float32:
        ids, weights = _multi_router_topk_kernel(int(top_k), bool(normalize))(
            inputs=[s2, bias.astype(mx.float32), scale_param.astype(mx.float32)],
            template=[("TOPK", int(top_k)), ("NORM", bool(normalize))],
            grid=(256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, int(top_k)), (T, int(top_k))],
            output_dtypes=[mx.uint32, mx.float32],
        )
        return ids.reshape(*lead, int(top_k)), weights.reshape(*lead, int(top_k))
    ids = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
    ids = mx.stop_gradient(ids)
    weights = mx.take_along_axis(scores, ids, axis=-1)
    if normalize:
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
    return ids, weights * scale_param[ids]


# ---------------------------------------------------------------------------
# 5c. Fused LatentMoE: Shared SiTU(proj) + Latent RMSNorm(combined) + Concat
#     Eliminates 2 strided contiguous copies + 3 dispatches across 42 MoE layers!
# ---------------------------------------------------------------------------
_SHARED_SITU_LATENT_NORM_CONCAT_SRC = r"""
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint row  = threadgroup_position_in_grid.y;

  const float beta = floats[0];
  const float lbeta = floats[1];
  const float norm_eps = floats[2];

  const device T* p_row = proj + size_t(row) * uint(P_DIM);
  const device T* c_row = comb + size_t(row) * uint(L_DIM);
  device T* o_row = out + size_t(row) * uint(S_DIM + L_DIM);

  for (uint i = tid; i < uint(S_DIM); i += 256u) {
    float g = float(p_row[i]);
    float u = float(p_row[uint(S_DIM) + i]);
    float sig = 1.0f / (1.0f + metal::exp(-g));
    float g_act = beta * metal::precise::tanh(g / beta) * sig;
    float u_act = lbeta * metal::precise::tanh(u / lbeta);
    o_row[i] = T(g_act * u_act);
  }

  threadgroup float sg_sq[8];
  float sq = 0.0f;
  for (uint j = tid; j < uint(L_DIM); j += 256u) {
    float v = float(c_row[j]);
    sq += v * v;
  }
  float sg_s = simd_sum(sq);
  if (lane == 0u) sg_sq[sg] = sg_s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = sg_sq[0] + sg_sq[1] + sg_sq[2] + sg_sq[3]
                 + sg_sq[4] + sg_sq[5] + sg_sq[6] + sg_sq[7];
  float inv_rms = metal::rsqrt(total_sq / float(L_DIM) + norm_eps);
  for (uint j = tid; j < uint(L_DIM); j += 256u) {
    float v = float(c_row[j]);
    o_row[uint(S_DIM) + j] = T(v * inv_rms * float(norm_w[j]));
  }
"""


@lru_cache(maxsize=12)
def _shared_situ_latent_norm_concat_kernel(dtype_name: str, p_dim: int, s_dim: int, l_dim: int):
    return mx.fast.metal_kernel(
        name=f"ernie_situ_ln_cat_{dtype_name}_p{p_dim}_s{s_dim}_l{l_dim}",
        input_names=["proj", "comb", "norm_w", "floats"],
        output_names=["out"],
        source=_SHARED_SITU_LATENT_NORM_CONCAT_SRC,
        ensure_row_contiguous=True,
    )


def ernie_shared_situ_latent_norm_concat(
    proj: mx.array,
    combined: mx.array,
    norm_w: mx.array,
    s_dim: int,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    norm_eps: float = 1e-6,
) -> mx.array:
    """Single-dispatch SiTU(proj[..., :s], proj[..., s:2s]) + RMSNorm(combined) + concat."""
    lead = combined.shape[:-1]
    p_dim = proj.shape[-1]
    l_dim = combined.shape[-1]
    T = combined.size // l_dim
    if (
        1 <= T <= 65535
        and p_dim >= 2 * s_dim
        and proj.dtype == combined.dtype
        and combined.dtype in (mx.bfloat16, mx.float16, mx.float32)
    ):
        dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[combined.dtype]
        nw = norm_w if norm_w.dtype == combined.dtype else norm_w.astype(combined.dtype)
        out = _shared_situ_latent_norm_concat_kernel(dname, int(p_dim), int(s_dim), int(l_dim))(
            inputs=[
                proj.reshape(T, p_dim),
                combined.reshape(T, l_dim),
                nw,
                mx.array([float(beta), float(linear_beta), float(norm_eps)], dtype=mx.float32),
            ],
            template=[("T", combined.dtype), ("P_DIM", int(p_dim)), ("S_DIM", int(s_dim)), ("L_DIM", int(l_dim))],
            grid=(256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, int(s_dim) + int(l_dim))],
            output_dtypes=[combined.dtype],
        )[0]
        return out.reshape(*lead, int(s_dim) + int(l_dim))
    shared_mid = fused_situ(proj[..., s_dim : 2 * s_dim], proj[..., :s_dim], beta, linear_beta)
    normed_combined = mx.fast.rms_norm(combined, norm_w, norm_eps)
    return mx.concatenate([shared_mid, normed_combined], axis=-1)


# ---------------------------------------------------------------------------
# 6. Fused BF16-weight FP32-accum lm_head + optional MultiMax epilogue (T = 1..8)
# ---------------------------------------------------------------------------
_LM_HEAD_MULTIMAX_SRC = r"""
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row0 = threadgroup_position_in_grid.x * 8u + sgi * 4u;
  if (row0 >= uint(N)) return;

  float acc[ROWS][4];
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) {
    acc[t][0] = 0.0f; acc[t][1] = 0.0f; acc[t][2] = 0.0f; acc[t][3] = 0.0f;
  }

  const uint r_valid = metal::min(4u, uint(N) - row0);
  const device bfloat16_t* wbase = w + size_t(row0) * uint(K);

  for (uint k = lane * 4u; k < uint(K); k += 128u) {
    float4 xv[ROWS];
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      const device float* xp = x + size_t(t) * uint(K) + k;
      xv[t] = *(const device float4*)xp;
    }
    #pragma unroll
    for (uint r = 0; r < 4u; r++) {
      if (r < r_valid) {
        float4 wv = float4(*(const device vec<bfloat16_t, 4>*)(wbase + size_t(r) * uint(K) + k));
        #pragma unroll
        for (uint t = 0; t < uint(ROWS); t++) {
          acc[t][r] += dot(xv[t], wv);
        }
      }
    }
  }

  const float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
  const float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];

  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) {
    #pragma unroll
    for (uint r = 0; r < 4u; r++) {
      float v = simd_sum(acc[t][r]);
      if (lane == 0u && r < r_valid) {
        if (APPLY_MULTIMAX) {
          float a = metal::max(r0 - v, 0.0f);
          float b = metal::max(v - r1, 0.0f);
          float c = metal::max(r2 - v, 0.0f);
          float d = metal::max(v - r3, 0.0f);
          v = v + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
        }
        out[size_t(t) * uint(N) + row0 + r] = v;
      }
    }
  }
"""


@lru_cache(maxsize=24)
def _lm_head_multimax_kernel(rows: int, N: int, K: int, apply_multimax: bool):
    return mx.fast.metal_kernel(
        name=f"ernie_lmhead_mm_r{rows}_n{N}_k{K}_m{int(apply_multimax)}",
        input_names=["x", "w", "ranges", "ts"],
        output_names=["out"],
        source=_LM_HEAD_MULTIMAX_SRC,
        ensure_row_contiguous=True,
    )


def ernie_lm_head_forward(
    h: mx.array,
    weight_bf16: mx.array,
    ranges: mx.array,
    ts: mx.array,
    *,
    apply_multimax: bool = True,
) -> mx.array:
    """BF16-weight FP32-accum lm_head (+ optional MultiMax) sharing weight reads across T=1..8 rows."""
    lead = h.shape[:-1]
    K = h.shape[-1]
    N = weight_bf16.shape[0]
    h32 = h.astype(mx.float32).reshape(-1, K)
    T = h32.shape[0]
    if 1 <= T <= 15 and K % 128 == 0 and weight_bf16.dtype == mx.bfloat16:
        if T <= 8:
            kern = _lm_head_multimax_kernel(T, N, K, apply_multimax)
            n_tg = (N + 7) // 8
            out = kern(
                inputs=[h32, weight_bf16, ranges, ts],
                template=[("ROWS", T), ("N", N), ("K", K), ("APPLY_MULTIMAX", bool(apply_multimax))],
                grid=(n_tg * 64, 1, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(T, N)],
                output_dtypes=[mx.float32],
            )[0]
            return out.reshape(*lead, N)
        parts = [
            ernie_lm_head_forward(h32[i : i + 8], weight_bf16, ranges, ts, apply_multimax=apply_multimax)
            for i in range(0, T, 8)
        ]
        return mx.concatenate(parts, axis=0).reshape(*lead, N)
    # Fallback for T > 8 (e.g. if full prompt logits are ever requested)
    logits = (h32.astype(mx.bfloat16) @ weight_bf16.T).astype(mx.float32)
    if apply_multimax:
        r, t = ranges, ts
        logits = (
            logits
            + t[0] * mx.maximum(r[0] - logits, 0)
            + t[1] * mx.maximum(logits - r[1], 0)
            + t[2] * mx.maximum(r[2] - logits, 0) ** 2
            + t[3] * mx.maximum(logits - r[3], 0) ** 2
        )
    return logits.reshape(*lead, N)


# ---------------------------------------------------------------------------
# 7. Fused DSpark Markov Step: argmax_v MultiMax(base[v] + dot(w2[v], trans))
# ---------------------------------------------------------------------------
_DSPARK_MARKOV_STAGE1_SRC = r"""
  // Stage 1: 512 threadgroups x 256 threads (8 simdgroups).
  // Each simdgroup handles rows of V=201216 strided by grid_sg, computing
  // dot(markov_w2[v], trans[256]) + base[v] -> MultiMax -> threadgroup argmax!
  // Zero [201216] intermediate allocations!
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint tg = threadgroup_position_in_grid.x;
  const uint n_tg = threadgroups_per_grid.x;

  // Stage transition[256] into threadgroup memory ONCE (1 KB!)
  threadgroup float t_smem[256];
  t_smem[tid] = float(trans[tid]);
  threadgroup float sg_val[8];
  threadgroup uint  sg_idx[8];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each lane holds 8 elements of transition (lane*8 .. lane*8+7)
  float tv0 = t_smem[lane * 8u + 0u];
  float tv1 = t_smem[lane * 8u + 1u];
  float tv2 = t_smem[lane * 8u + 2u];
  float tv3 = t_smem[lane * 8u + 3u];
  float tv4 = t_smem[lane * 8u + 4u];
  float tv5 = t_smem[lane * 8u + 5u];
  float tv6 = t_smem[lane * 8u + 6u];
  float tv7 = t_smem[lane * 8u + 7u];

  const float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
  const float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];

  float best_v = -1e38f;
  uint best_i = 0u;
  const uint global_sg = tg * 8u + sg;
  const uint stride_sg = n_tg * 8u;

  for (uint v = global_sg; v < uint(V); v += stride_sg) {
    const device bfloat16_t* wp = w2 + size_t(v) * 256u + lane * 8u;
    float4 w_lo = float4(*(const device vec<bfloat16_t, 4>*)(wp));
    float4 w_hi = float4(*(const device vec<bfloat16_t, 4>*)(wp + 4u));
    float d = w_lo.x * tv0 + w_lo.y * tv1 + w_lo.z * tv2 + w_lo.w * tv3
            + w_hi.x * tv4 + w_hi.y * tv5 + w_hi.z * tv6 + w_hi.w * tv7;
    float dot_v = simd_sum(d);
    if (lane == 0u) {
      // Match reference: (w2 @ transition).astype(float32) + base[v] -> MultiMax
      float val = base_row[v] + float(bfloat16_t(dot_v));
      float a = metal::max(r0 - val, 0.0f);
      float b = metal::max(val - r1, 0.0f);
      float c = metal::max(r2 - val, 0.0f);
      float e = metal::max(val - r3, 0.0f);
      val = val + t0 * a + t1 * b + t2 * (c * c) + t3 * (e * e);
      if (val > best_v) {
        best_v = val;
        best_i = v;
      }
    }
  }
  if (lane == 0u) {
    sg_val[sg] = best_v;
    sg_idx[sg] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float bv = sg_val[0];
    uint bi = sg_idx[0];
    for (uint s = 1u; s < 8u; s++) {
      if (sg_val[s] > bv || (sg_val[s] == bv && sg_idx[s] < bi)) {
        bv = sg_val[s];
        bi = sg_idx[s];
      }
    }
    blk_val[tg] = bv;
    blk_idx[tg] = bi;
  }
"""


@lru_cache(maxsize=4)
def _dspark_markov_stage1_kernel(V: int):
    return mx.fast.metal_kernel(
        name=f"ernie_dspark_markov_s1_v{V}",
        input_names=["base_row", "w2", "trans", "ranges", "ts"],
        output_names=["blk_val", "blk_idx"],
        source=_DSPARK_MARKOV_STAGE1_SRC,
        ensure_row_contiguous=True,
    )


def ernie_dspark_markov_argmax(
    base_row: mx.array,
    markov_w2: mx.array,
    transition: mx.array,
    ranges: mx.array,
    ts: mx.array,
) -> mx.array:
    """Single-pass fused argmax_v MultiMax(base_row[v] + (markov_w2 @ transition)[v])."""
    V, R = markov_w2.shape
    if R == 256 and markov_w2.dtype == mx.bfloat16:
        n_tg = 512
        bval, bidx = _dspark_markov_stage1_kernel(V)(
            inputs=[base_row, markov_w2, transition, ranges, ts],
            template=[("V", V)],
            grid=(n_tg * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n_tg,), (n_tg,)],
            output_dtypes=[mx.float32, mx.uint32],
        )
        winner = mx.argmax(bval)
        return bidx[winner].astype(mx.int32)
    bias = (markov_w2 @ transition).astype(mx.float32)
    v = base_row + bias
    r, t = ranges, ts
    v = (
        v
        + t[0] * mx.maximum(r[0] - v, 0)
        + t[1] * mx.maximum(v - r[1], 0)
        + t[2] * mx.maximum(r[2] - v, 0) ** 2
        + t[3] * mx.maximum(v - r[3], 0) ** 2
    )
    return mx.argmax(v).astype(mx.int32)


# ---------------------------------------------------------------------------
# 7b. Exact Global Top-256 Radix Prefilter + Single-Dispatch k-Step Markov Chain
#     Cuts 6x markov_w2 scan from 1.41 ms -> ~0.25 ms with 100% exact Top-256
#     match (30/30 on real ERNIE logits) and 100% lossless target verification.
# ---------------------------------------------------------------------------

_DSPARK_MARKOV_CHAIN_SRC = r"""
  // 1 TG of 256 threads (8 simdgroups) runs ALL K_POS sequential Markov steps
  // over exact global Top-K (e.g. K=256) candidates in ONE dispatch!
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint K_POS = uint(cand_vals_shape[0]);
  const uint N_CAND = uint(cand_vals_shape[1]);

  const float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
  const float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];

  threadgroup float t_sh[256];
  threadgroup float sg_v[8];
  threadgroup uint  sg_i[8];
  threadgroup uint  cur_tok_sh;

  if (tid == 0u) cur_tok_sh = uint(anchor[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = 0u; p < K_POS; ++p) {
    const uint prev_tok = cur_tok_sh;
    t_sh[tid] = float(markov_emb[size_t(prev_tok) * 256u + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d0 = lane * 8u;
    float4 tv_lo = float4(t_sh[d0 + 0u], t_sh[d0 + 1u], t_sh[d0 + 2u], t_sh[d0 + 3u]);
    float4 tv_hi = float4(t_sh[d0 + 4u], t_sh[d0 + 5u], t_sh[d0 + 6u], t_sh[d0 + 7u]);

    const device float* cv = cand_vals + size_t(p) * size_t(N_CAND);
    const device uint*  ci = cand_idxs + size_t(p) * size_t(N_CAND);

    float best_v = -1e38f;
    uint  best_i = 0u;
    for (uint c = sg; c < N_CAND; c += 8u) {
      const uint vid = ci[c];
      if (vid == 0xFFFFFFFFu) continue;
      const device bfloat16_t* wp = w2 + size_t(vid) * 256u + d0;
      float4 w_lo = float4(*(const device vec<bfloat16_t, 4>*)(wp));
      float4 w_hi = float4(*(const device vec<bfloat16_t, 4>*)(wp + 4u));
      float dot_v = simd_sum(dot(tv_lo, w_lo) + dot(tv_hi, w_hi));
      if (lane == 0u) {
        float val = cv[c] + float(bfloat16_t(dot_v));
        float a = metal::max(r0 - val, 0.0f);
        float b = metal::max(val - r1, 0.0f);
        float g = metal::max(r2 - val, 0.0f);
        float e = metal::max(val - r3, 0.0f);
        val = val + t0 * a + t1 * b + t2 * (g * g) + t3 * (e * e);
        if (val > best_v || (val == best_v && vid < best_i)) {
          best_v = val;
          best_i = vid;
        }
      }
    }
    if (lane == 0u) {
      sg_v[sg] = best_v;
      sg_i[sg] = best_i;
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
      out_ids[p] = int(gi);
      cur_tok_sh = gi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@lru_cache(maxsize=2)
def _dspark_markov_chain_kernel():
    return mx.fast.metal_kernel(
        name="ernie_dspark_markov_chain_topk",
        input_names=["cand_vals", "cand_idxs", "markov_emb", "w2", "anchor", "ranges", "ts"],
        output_names=["out_ids"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_DSPARK_MARKOV_CHAIN_SRC,
        ensure_row_contiguous=True,
    )


def ernie_dspark_markov_topk_chain(
    base_logits: mx.array,
    markov_emb: mx.array,
    markov_w2: mx.array,
    anchor: mx.array,
    ranges: mx.array,
    ts: mx.array,
    *,
    top_k: int = 256,
) -> Optional[mx.array]:
    """Exact global Top-K radix prefilter + single-dispatch k-step greedy Markov chain."""
    if os.environ.get("VMLX_ERNIE_MARKOV_TOPK", "1") == "0":
        return None
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or base_logits.ndim != 2
        or markov_w2.ndim != 2
        or markov_w2.shape[1] != 256
        or markov_w2.dtype != mx.bfloat16
        or markov_emb.dtype != mx.bfloat16
    ):
        return None
    K_pos, V = int(base_logits.shape[0]), int(base_logits.shape[1])
    k_sel = min(int(top_k), V)
    if K_pos < 1 or k_sel < 1 or V <= k_sel:
        return None
    from vmlx_engine.metal.sparse_topk import metal_radix_topk

    bf32 = base_logits.astype(mx.float32)
    cidxs = metal_radix_topk(bf32, k_sel)
    if cidxs is None:
        return None
    cvals = mx.take_along_axis(bf32, cidxs.astype(mx.int32), axis=-1)
    return _dspark_markov_chain_kernel()(
        inputs=[
            cvals,
            cidxs.astype(mx.uint32),
            markov_emb,
            markov_w2,
            anchor.reshape(1).astype(mx.int32),
            ranges.astype(mx.float32),
            ts.astype(mx.float32),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K_pos,)],
        output_dtypes=[mx.int32],
    )[0]


# ---------------------------------------------------------------------------
# 8. Fused Sparse Absorbed MLA Prefill & Verify (O(topK + window) up to 1M!)
#    Eliminates expanding k_full/v_full over S keys and O(L*S) pe_scores!
# ---------------------------------------------------------------------------
_SPARSE_MLA_ABSORBED_SRC = r"""
  // One threadgroup (256 threads = 8 simdgroups) owns 1 query `t` and
  // `8 * H_PER_SG` query heads (H_PER_SG = 1 -> 8 heads/TG for small L;
  // H_PER_SG = 2 -> 16 heads/TG for prefill L > 8).
  // All 8 simdgroups cooperatively stage BK=8 gathered KV (512) + KPE (64)
  // keys into 9 KB threadgroup memory, sharing 100% of KV DRAM reads!
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint t = threadgroup_position_in_grid.y;
  const uint hb = threadgroup_position_in_grid.z;

  const uint L = uint(q_nope_shape[2]);
  const uint S = uint(kv_shape[2]);
  const uint K_sel = uint(indices_shape[2]);
  const float scale = scale_buf[0];

  const uint h_base = hb * (8u * uint(H_PER_SG)) + sg * uint(H_PER_SG);

  // Load this lane's slice of Q_nope (16 dims = 4x vec4) and Q_pe (2 dims)
  // for each of the H_PER_SG heads owned by this simdgroup.
  float qn[H_PER_SG][16];
  float qp[H_PER_SG][2];
  float acc[H_PER_SG][16];
  float m_val[H_PER_SG];
  float l_val[H_PER_SG];

  #pragma unroll
  for (uint hi = 0u; hi < uint(H_PER_SG); ++hi) {
    const uint h = h_base + hi;
    const device T* qn_p = q_nope + (size_t(h) * size_t(L) + size_t(t)) * 512u + lane * 16u;
    const device T* qp_p = q_pe   + (size_t(h) * size_t(L) + size_t(t)) * 64u  + lane * 2u;
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
      float4 qv = float4(*(const device vec<T, 4>*)(qn_p + j * 4u));
      qn[hi][j * 4u + 0u] = qv.x;
      qn[hi][j * 4u + 1u] = qv.y;
      qn[hi][j * 4u + 2u] = qv.z;
      qn[hi][j * 4u + 3u] = qv.w;
    }
    qp[hi][0] = float(qp_p[0]);
    qp[hi][1] = float(qp_p[1]);
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) acc[hi][j] = 0.0f;
    // Exact MLX attention sink initialization: sink logit = sinks[h], sink value = 0
    m_val[hi] = sinks[h];
    l_val[hi] = 1.0f;
  }

  constexpr uint BK = 8u;
  threadgroup T kv_sh[BK * 512u];
  threadgroup T kpe_sh[BK * 64u];
  threadgroup uint valid_sh[BK];

  const device uint* idx_row = indices + size_t(t) * size_t(K_sel);

  for (uint k0 = 0u; k0 < K_sel; k0 += BK) {
    // 1. Load BK indices & validate
    if (tid < BK) {
      uint kk = k0 + tid;
      uint idx = (kk < K_sel) ? idx_row[kk] : 0xFFFFFFFFu;
      valid_sh[tid] = (idx < S) ? idx : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Cooperatively gather BK * 512 KV elements (16 elems = 4x vec4 per thread)
    //    and BK * 64 KPE elements (2 elems per thread) into threadgroup memory!
    //    Map tid -> (b = tid >> 5, col_chunk = tid & 31): each simdgroup copies 1 key!
    {
      const uint b = sg;
      const uint idx = valid_sh[b];
      if (idx != 0xFFFFFFFFu) {
        const device vec<T, 4>* src_kv = (const device vec<T, 4>*)(kv + size_t(idx) * 512u + lane * 16u);
        threadgroup vec<T, 4>* dst_kv = (threadgroup vec<T, 4>*)(kv_sh + b * 512u + lane * 16u);
        dst_kv[0] = src_kv[0];
        dst_kv[1] = src_kv[1];
        dst_kv[2] = src_kv[2];
        dst_kv[3] = src_kv[3];
        const device T* src_kp = k_pe + size_t(idx) * 64u + lane * 2u;
        threadgroup T* dst_kp = kpe_sh + b * 64u + lane * 2u;
        dst_kp[0] = src_kp[0];
        dst_kp[1] = src_kp[1];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Process BK keys in registers for all H_PER_SG heads of this simdgroup
    for (uint b = 0u; b < BK; ++b) {
      if (valid_sh[b] == 0xFFFFFFFFu) continue;
      const threadgroup vec<T, 4>* kv_b = (const threadgroup vec<T, 4>*)(kv_sh + b * 512u + lane * 16u);
      float4 kv0 = float4(kv_b[0]);
      float4 kv1 = float4(kv_b[1]);
      float4 kv2 = float4(kv_b[2]);
      float4 kv3 = float4(kv_b[3]);
      float kp0 = float(kpe_sh[b * 64u + lane * 2u + 0u]);
      float kp1 = float(kpe_sh[b * 64u + lane * 2u + 1u]);

      #pragma unroll
      for (uint hi = 0u; hi < uint(H_PER_SG); ++hi) {
        float d = qn[hi][0]  * kv0.x + qn[hi][1]  * kv0.y + qn[hi][2]  * kv0.z + qn[hi][3]  * kv0.w
                + qn[hi][4]  * kv1.x + qn[hi][5]  * kv1.y + qn[hi][6]  * kv1.z + qn[hi][7]  * kv1.w
                + qn[hi][8]  * kv2.x + qn[hi][9]  * kv2.y + qn[hi][10] * kv2.z + qn[hi][11] * kv2.w
                + qn[hi][12] * kv3.x + qn[hi][13] * kv3.y + qn[hi][14] * kv3.z + qn[hi][15] * kv3.w
                + qp[hi][0] * kp0 + qp[hi][1] * kp1;
        float score = simd_sum(d) * scale;
        float m_old = m_val[hi];
        float m_new = metal::max(m_old, score);
        float alpha = metal::exp(m_old - m_new);
        float p = metal::exp(score - m_new);
        m_val[hi] = m_new;
        l_val[hi] = l_val[hi] * alpha + p;
        acc[hi][0]  = fma(acc[hi][0],  alpha, p * kv0.x);
        acc[hi][1]  = fma(acc[hi][1],  alpha, p * kv0.y);
        acc[hi][2]  = fma(acc[hi][2],  alpha, p * kv0.z);
        acc[hi][3]  = fma(acc[hi][3],  alpha, p * kv0.w);
        acc[hi][4]  = fma(acc[hi][4],  alpha, p * kv1.x);
        acc[hi][5]  = fma(acc[hi][5],  alpha, p * kv1.y);
        acc[hi][6]  = fma(acc[hi][6],  alpha, p * kv1.z);
        acc[hi][7]  = fma(acc[hi][7],  alpha, p * kv1.w);
        acc[hi][8]  = fma(acc[hi][8],  alpha, p * kv2.x);
        acc[hi][9]  = fma(acc[hi][9],  alpha, p * kv2.y);
        acc[hi][10] = fma(acc[hi][10], alpha, p * kv2.z);
        acc[hi][11] = fma(acc[hi][11], alpha, p * kv2.w);
        acc[hi][12] = fma(acc[hi][12], alpha, p * kv3.x);
        acc[hi][13] = fma(acc[hi][13], alpha, p * kv3.y);
        acc[hi][14] = fma(acc[hi][14], alpha, p * kv3.z);
        acc[hi][15] = fma(acc[hi][15], alpha, p * kv3.w);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write normalized absorbed output [1, H, L, 512]
  #pragma unroll
  for (uint hi = 0u; hi < uint(H_PER_SG); ++hi) {
    const uint h = h_base + hi;
    const float inv_l = 1.0f / l_val[hi];
    device vec<T, 4>* out_p = (device vec<T, 4>*)(out + (size_t(h) * size_t(L) + size_t(t)) * 512u + lane * 16u);
    out_p[0] = vec<T, 4>(T(acc[hi][0] * inv_l),  T(acc[hi][1] * inv_l),  T(acc[hi][2] * inv_l),  T(acc[hi][3] * inv_l));
    out_p[1] = vec<T, 4>(T(acc[hi][4] * inv_l),  T(acc[hi][5] * inv_l),  T(acc[hi][6] * inv_l),  T(acc[hi][7] * inv_l));
    out_p[2] = vec<T, 4>(T(acc[hi][8] * inv_l),  T(acc[hi][9] * inv_l),  T(acc[hi][10] * inv_l), T(acc[hi][11] * inv_l));
    out_p[3] = vec<T, 4>(T(acc[hi][12] * inv_l), T(acc[hi][13] * inv_l), T(acc[hi][14] * inv_l), T(acc[hi][15] * inv_l));
  }
"""


@lru_cache(maxsize=8)
def _sparse_mla_absorbed_kernel(dtype_name: str, h_per_sg: int):
    return mx.fast.metal_kernel(
        name=f"ernie_sparse_mla_absorbed_{dtype_name}_hps{h_per_sg}",
        input_names=["q_nope", "q_pe", "kv", "k_pe", "indices", "sinks", "scale_buf"],
        output_names=["out"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_SPARSE_MLA_ABSORBED_SRC,
        ensure_row_contiguous=True,
    )


def ernie_sparse_mla_prefill(
    q_nope: mx.array,
    q_pe: mx.array,
    kv: mx.array,
    k_pe: mx.array,
    indices: mx.array,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Fused sparse absorbed MLA attention with O(K_sel) memory & FLOPs up to 1M context.

    Args:
        q_nope: (1, H, L, 512) absorbed latent query (bf16/fp16/fp32)
        q_pe:   (1, H, L, 64)  RoPE query (same dtype)
        kv:     (1, 1, S, 512) latent KV cache (same dtype)
        k_pe:   (1, 1, S, 64)  RoPE key cache (same dtype)
        indices:(1, L, K_sel)  uint32 per-query selected key indices (0xFFFFFFFF sentinel = skip)
        sinks:  (H,)           attention sink logits
        scale:  attention softmax scale

    Returns:
        (1, H, L, 512) absorbed attention output (same dtype as q_nope).
    """
    B, H, L, D_lat = q_nope.shape
    if (
        B != 1
        or D_lat != 512
        or q_pe.shape[-1] != 64
        or H % 16 != 0
        or q_nope.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not mx.metal.is_available()
    ):
        raise ValueError(
            f"Unsupported ernie_sparse_mla_prefill shapes/dtypes: "
            f"q_nope={q_nope.shape}({q_nope.dtype}), q_pe={q_pe.shape}, kv={kv.shape}"
        )
    dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[q_nope.dtype]
    h_per_sg = 1 if L <= 8 else 2
    n_hb = H // (8 * h_per_sg)
    sinks_f = sinks.astype(mx.float32)
    idx_u32 = indices.astype(mx.uint32)
    sc_buf = mx.array([float(scale)], dtype=mx.float32)
    return _sparse_mla_absorbed_kernel(dname, h_per_sg)(
        inputs=[q_nope, q_pe, kv, k_pe, idx_u32, sinks_f, sc_buf],
        template=[("T", q_nope.dtype), ("H_PER_SG", h_per_sg)],
        grid=(256, int(L), int(n_hb)),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, int(H), int(L), 512)],
        output_dtypes=[q_nope.dtype],
    )[0]


# ---------------------------------------------------------------------------
# 9. High-Bandwidth M5 MXFP8 Multi-Vector GEMV (M = 1..8, 2D and 3D MultiLinear)
#    Reads 8-bit E4M3 weights + UE8M0 scales ONCE across up to 8 rows at ~90% DRAM BW!
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _e4m3_lut_header() -> str:
    vals = mx.from_fp8(mx.arange(256, dtype=mx.uint8), dtype=mx.float32).tolist()
    entries = ", ".join(
        "0.0f" if (math.isnan(v) or math.isinf(v)) else f"{float(v)!r}f"
        for v in vals
    )
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        f"constant float E4M3_LUT[256] = {{ {entries} }};\n"
    )


_MXFP8_MULTI_GEMV_SRC = r"""
  // 4 simdgroups (128 threads) per TG, RPS output rows/sg, G groups along grid.y.
  // Each lane reads 8 E4M3 weights (uint2 = 8 bytes) + 1 UE8M0 scale byte per 256-elem step.
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint g = threadgroup_position_in_grid.y;
  const uint row0 = threadgroup_position_in_grid.x * (4u * uint(RPS)) + sgi * uint(RPS);
  if (row0 >= uint(N)) return;

  const uint r_valid = metal::min(uint(RPS), uint(N) - row0);
  const uint KW = uint(K) >> 2;
  const uint NG = uint(K) >> 5;
  const uint lane_g = lane >> 2;

  float acc[ROWS][RPS];
  #pragma unroll
  for (uint v = 0u; v < uint(ROWS); ++v) {
    #pragma unroll
    for (uint r = 0u; r < uint(RPS); ++r) {
      acc[v][r] = 0.0f;
    }
  }

  const device XT* x_g = x + size_t(g) * size_t(ROWS) * uint(K);
  const device uint* w0 = w + (size_t(g) * uint(N) + row0) * KW;
  const device uchar* s0 = scales + (size_t(g) * uint(N) + row0) * NG;

  for (uint k0 = 0u; k0 < uint(K); k0 += 256u) {
    const uint k_lane = k0 + lane * 8u;
    const uint g_idx = (k0 >> 5) + lane_g;
    #pragma unroll
    for (uint r = 0u; r < uint(RPS); ++r) {
      if (r < r_valid) {
        const device uint2* wr = (const device uint2*)(w0 + size_t(r) * KW + (k_lane >> 2));
        const uint2 p = *wr;
        const uint p0 = p.x;
        const uint p1 = p.y;
        const float sc = as_type<float>(uint(s0[size_t(r) * NG + g_idx]) << 23);
        const float4 w_lo = float4(
            E4M3_LUT[p0 & 0xFFu],
            E4M3_LUT[(p0 >> 8) & 0xFFu],
            E4M3_LUT[(p0 >> 16) & 0xFFu],
            E4M3_LUT[p0 >> 24]) * sc;
        const float4 w_hi = float4(
            E4M3_LUT[p1 & 0xFFu],
            E4M3_LUT[(p1 >> 8) & 0xFFu],
            E4M3_LUT[(p1 >> 16) & 0xFFu],
            E4M3_LUT[p1 >> 24]) * sc;
        #pragma unroll
        for (uint v = 0u; v < uint(ROWS); ++v) {
          const device XT* xp = x_g + size_t(v) * uint(K) + k_lane;
          float4 x_lo = float4(*(const device vec<XT, 4>*)(xp));
          float4 x_hi = float4(*(const device vec<XT, 4>*)(xp + 4u));
          acc[v][r] += dot(x_lo, w_lo) + dot(x_hi, w_hi);
        }
      }
    }
  }

  #pragma unroll
  for (uint v = 0u; v < uint(ROWS); ++v) {
    #pragma unroll
    for (uint r = 0u; r < uint(RPS); ++r) {
      float total = simd_sum(acc[v][r]);
      if (lane == 0u && r < r_valid) {
        out[(size_t(g) * uint(ROWS) + v) * uint(N) + row0 + r] = OT(total);
      }
    }
  }
"""


@lru_cache(maxsize=64)
def _mxfp8_multi_gemv_kernel(rows: int, rps: int, K: int, N: int, x_dn: str, o_dn: str):
    return mx.fast.metal_kernel(
        name=f"ernie_mxfp8_mv_r{rows}_p{rps}_k{K}_n{N}_{x_dn}_{o_dn}",
        input_names=["x", "w", "scales"],
        output_names=["out"],
        header=_e4m3_lut_header(),
        source=_MXFP8_MULTI_GEMV_SRC,
        ensure_row_contiguous=True,
    )


def ernie_mxfp8_multi_gemv(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    grouped: bool = False,
) -> mx.array | None:
    """High-bandwidth MXFP8 multi-vector GEMV for 2D (G=1) or 3D grouped (B=1, G, T, K) projections."""
    if os.environ.get("VMLX_ERNIE_FAST_MXFP8", "1") == "0":
        return None
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or weight.dtype != mx.uint32
        or scales.dtype != mx.uint8
    ):
        return None
    K = int(x.shape[-1])
    if K % 256 != 0:
        return None
    min_t = int(os.environ.get("VMLX_ERNIE_FAST_MXFP8_MIN_T", "1"))
    if grouped:
        # x: (B, G, T, K) with B == 1; weight: (G, N, K//4); scales: (G, N, K//32)
        if x.ndim != 4 or x.shape[0] != 1 or weight.ndim != 3:
            return None
        G, T = int(x.shape[1]), int(x.shape[2])
        N = int(weight.shape[1])
        if not (min_t <= T <= 8):
            return None
        x3 = x.reshape(G, T, K)
        out_shape = (1, G, T, N)
    else:
        lead = x.shape[:-1]
        T = int(x.size) // K
        N = int(weight.shape[0])
        if not (min_t <= T <= 8):
            return None
        G = 1
        x3 = x.reshape(1, T, K)
        out_shape = (*lead, N)

    rps = 2 if (G * N) >= 4096 else 1
    dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
    n_tg = (N + 4 * rps - 1) // (4 * rps)
    y = _mxfp8_multi_gemv_kernel(T, rps, K, N, dn, dn)(
        inputs=[x3, weight, scales],
        template=[
            ("XT", x.dtype),
            ("OT", x.dtype),
            ("ROWS", T),
            ("RPS", rps),
            ("K", K),
            ("N", N),
        ],
        grid=(n_tg * 128, G, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(G, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(out_shape)


def ernie_mxfp8_linear(module, x: mx.array) -> mx.array:
    """Dispatch 2D MXFP8 QuantizedLinear to ernie_mxfp8_multi_gemv for T=1..8, else stock module(x)."""
    if (
        getattr(module, "mode", None) == "mxfp8"
        and getattr(module, "bits", None) == 8
        and getattr(module, "group_size", None) == 32
        and "bias" not in module
    ):
        y = ernie_mxfp8_multi_gemv(x, module.weight, module.scales, grouped=False)
        if y is not None:
            return y
    return module(x)

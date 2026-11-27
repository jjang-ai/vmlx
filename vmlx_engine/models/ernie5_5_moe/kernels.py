# SPDX-License-Identifier: Apache-2.0
"""Fused Apple Silicon Metal kernels for ERNIE 5.5 Flash (mHC, MultiRouter, HCA Compressor, E4M3 rounding, SiTU, lm_head+MultiMax, DSpark Markov)."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

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

  // Each simdgroup owns 3 rows of fn (8 * 3 = 24 rows), walking 256 elems in 2x float4 steps.
  const uint m0 = sg * 3u;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  const device float* f0 = fn + size_t(m0 + 0u) * uint(K_FLAT) + k0;
  const device float* f1 = fn + size_t(m0 + 1u) * uint(K_FLAT) + k0;
  const device float* f2 = fn + size_t(m0 + 2u) * uint(K_FLAT) + k0;
  #pragma unroll
  for (uint j = 0; j < 2u; j++) {
    uint idx = j * 128u + lane * 4u;
    float4 xv = *(const threadgroup float4*)(x_smem + idx);
    a0 += dot(xv, *(const device float4*)(f0 + idx));
    a1 += dot(xv, *(const device float4*)(f1 + idx));
    a2 += dot(xv, *(const device float4*)(f2 + idx));
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
  const device float* f0 = fn + size_t(m0 + 0u) * uint(K_FLAT) + k0;
  const device float* f1 = fn + size_t(m0 + 1u) * uint(K_FLAT) + k0;
  const device float* f2 = fn + size_t(m0 + 2u) * uint(K_FLAT) + k0;
  #pragma unroll
  for (uint j = 0; j < 2u; j++) {
    uint idx = j * 128u + lane * 4u;
    float4 xv = *(const threadgroup float4*)(x_smem + idx);
    a0 += dot(xv, *(const device float4*)(f0 + idx));
    a1 += dot(xv, *(const device float4*)(f1 + idx));
    a2 += dot(xv, *(const device float4*)(f2 + idx));
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


@lru_cache(maxsize=4)
def _mhc_splitk_s1_kernel(dtype_name: str, K_flat: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_sk64_1_{dtype_name}_k{K_flat}",
        input_names=["x_in", "fn"],
        output_names=["partials"],
        source=_MHC_SPLITK_S1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mhc_exp_splitk_s1_kernel(dtype_name: str, D: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mhc_expsk64_1_{dtype_name}_d{D}",
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
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype == mx.float32:
        partials = _mhc_splitk_s1_kernel(dname, H * D)(
            inputs=[x, fn],
            template=[("T", x.dtype), ("K_FLAT", H * D)],
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
    mixes = (flat @ fn.T) / rms
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
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype == mx.float32:
        x_new, partials = _mhc_exp_splitk_s1_kernel(dname, D)(
            inputs=[sub, residual, prev_post, prev_comb, fn],
            template=[("T", residual.dtype), ("D", D), ("K_FLAT", H * D)],
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
  const device float* fm = fn + size_t(sg) * uint(K_FLAT) + k0;
  float acc = 0.0f;
  #pragma unroll
  for (uint j = 0; j < 4u; j++) {
    uint idx = j * 128u + lane * 4u;
    acc += dot(*(const threadgroup float4*)(x_smem + idx), *(const device float4*)(fm + idx));
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


@lru_cache(maxsize=4)
def _hc_head_s1_kernel(dtype_name: str, K_flat: int):
    return mx.fast.metal_kernel(
        name=f"ernie_hchead_s1_{dtype_name}_k{K_flat}",
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
    if 1 <= T <= 16 and H == 4 and D == 4096 and fn.dtype == mx.float32 and x.dtype in (mx.bfloat16, mx.float16, mx.float32):
        dname = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[x.dtype]
        partials = _hc_head_s1_kernel(dname, H * D)(
            inputs=[x, fn],
            template=[("T", x.dtype), ("K_FLAT", H * D)],
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
    mixes = z @ fn.T
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
# 5b. High-occupancy M5 Ultra MXFP8 LUT GEMV (1 lane = 32-elem MXFP8 block, 4-sg Split-K)
# ---------------------------------------------------------------------------
def _build_e4m3_lut_header() -> str:
    vals = []
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0:
            v = s * (mant / 512.0)
        elif exp == 15 and mant == 7:
            v = 0.0
        else:
            v = s * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        vals.append(f"half({v!r})")
    return "constant half E4M3_LUT[256] = {" + ", ".join(vals) + "};\n"


_MXFP8_LUT_HEADER = _build_e4m3_lut_header()

_MXFP8_GEMV_4SG_SRC = r"""
  // High-occupancy M5 Ultra/Max MXFP8 GEMV (small N < 2048, K % 4096 == 0):
  // 1 lane = one 32-elem MXFP8 block; 4 simdgroups (128 threads) = 4096 elems/step; 1 row/TG.
  // Shares weight loads across ALL T=1..8 verify rows (identical FP32 reduction order for T=1..8!).
  const uint sgi  = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row  = threadgroup_position_in_grid.x;
  if (row >= uint(N)) return;

  threadgroup float sh[ROWS][4];
  float acc[ROWS];
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) acc[t] = 0.0f;

  const device uint32_t* wr = w + size_t(row) * (uint(K) / 4u);
  const device uint8_t*  sr = scales + size_t(row) * (uint(K) / 32u);

  for (uint blk = sgi * 32u + lane; blk < (uint(K) / 32u); blk += 128u) {
    const float sf = as_type<float>(uint(sr[blk]) << 23u);
    const device uint4* wp = (const device uint4*)(wr + blk * 8u);
    uint4 w0 = wp[0];
    uint4 w1 = wp[1];
    uint words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    float4 wf[8];
    #pragma unroll
    for (uint j = 0; j < 8u; j++) {
      uint u = words[j];
      wf[j] = float4(
        float(E4M3_LUT[u & 255u]),
        float(E4M3_LUT[(u >> 8u) & 255u]),
        float(E4M3_LUT[(u >> 16u) & 255u]),
        float(E4M3_LUT[(u >> 24u) & 255u])
      );
    }
    const uint k0 = blk * 32u;
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + size_t(t) * uint(K) + k0);
      float d = 0.0f;
      #pragma unroll
      for (uint j = 0; j < 8u; j++) {
        d += dot(float4(xp[j]), wf[j]);
      }
      acc[t] += d * sf;
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
      out[size_t(t) * uint(N) + row] = T(sh[t][0] + sh[t][1] + sh[t][2] + sh[t][3]);
    }
  }
"""

_MXFP8_GEMV_1SG_SRC = r"""
  // High-throughput barrier-free MXFP8 multi-row GEMV (N >= 2048 or K == 1024, K % 1024 == 0):
  // 1 simdgroup = 2 output rows (reuses x across 2 rows!), 1 lane = 32-elem MXFP8 block (1024 elems/step!).
  // 4 simdgroups (128 threads) per TG = 8 output rows/TG -> ZERO TG barriers, flat T=1..8 time!
  const uint sgi  = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint grp  = threadgroup_position_in_grid.y;   // 0 for 2D, 0..G-1 for 3D MultiLinear!
  const uint row0 = threadgroup_position_in_grid.x * 8u + sgi * 2u;
  if (row0 >= uint(N)) return;

  float acc0[ROWS], acc1[ROWS];
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

  const bool has1 = (row0 + 1u) < uint(N);
  const size_t g_w_off = size_t(grp) * uint(N) * (uint(K) / 4u);
  const size_t g_s_off = size_t(grp) * uint(N) * (uint(K) / 32u);
  const size_t g_x_off = size_t(grp) * uint(ROWS) * uint(K);
  const device uint32_t* w0r = w + g_w_off + size_t(row0) * (uint(K) / 4u);
  const device uint32_t* w1r = w0r + (uint(K) / 4u);
  const device uint8_t*  s0r = scales + g_s_off + size_t(row0) * (uint(K) / 32u);
  const device uint8_t*  s1r = s0r + (uint(K) / 32u);

  for (uint blk = lane; blk < (uint(K) / 32u); blk += 32u) {
    const uint k0 = blk * 32u;
    // Load x[t] ONCE and reuse across BOTH output rows (halves x reads)!
    float4 xv[ROWS][8];
    #pragma unroll
    for (uint t = 0; t < uint(ROWS); t++) {
      const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + g_x_off + size_t(t) * uint(K) + k0);
      #pragma unroll
      for (uint j = 0; j < 8u; j++) xv[t][j] = float4(xp[j]);
    }
    // Row 0
    {
      const float sf = as_type<float>(uint(s0r[blk]) << 23u);
      const device uint4* wp = (const device uint4*)(w0r + blk * 8u);
      uint4 p0 = wp[0], p1 = wp[1];
      uint words[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float4 wf[8];
      #pragma unroll
      for (uint j = 0; j < 8u; j++) {
        uint u = words[j];
        wf[j] = float4(
          float(E4M3_LUT[u & 255u]), float(E4M3_LUT[(u >> 8u) & 255u]),
          float(E4M3_LUT[(u >> 16u) & 255u]), float(E4M3_LUT[(u >> 24u) & 255u])
        );
      }
      #pragma unroll
      for (uint t = 0; t < uint(ROWS); t++) {
        float d = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) d += dot(xv[t][j], wf[j]);
        acc0[t] += d * sf;
      }
    }
    // Row 1
    if (has1) {
      const float sf = as_type<float>(uint(s1r[blk]) << 23u);
      const device uint4* wp = (const device uint4*)(w1r + blk * 8u);
      uint4 p0 = wp[0], p1 = wp[1];
      uint words[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float4 wf[8];
      #pragma unroll
      for (uint j = 0; j < 8u; j++) {
        uint u = words[j];
        wf[j] = float4(
          float(E4M3_LUT[u & 255u]), float(E4M3_LUT[(u >> 8u) & 255u]),
          float(E4M3_LUT[(u >> 16u) & 255u]), float(E4M3_LUT[(u >> 24u) & 255u])
        );
      }
      #pragma unroll
      for (uint t = 0; t < uint(ROWS); t++) {
        float d = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) d += dot(xv[t][j], wf[j]);
        acc1[t] += d * sf;
      }
    }
  }
  const size_t g_o_off = size_t(grp) * uint(ROWS) * uint(N);
  #pragma unroll
  for (uint t = 0; t < uint(ROWS); t++) {
    float s0 = simd_sum(acc0[t]);
    float s1 = simd_sum(acc1[t]);
    if (lane == 0u) {
      out[g_o_off + size_t(t) * uint(N) + row0] = T(s0);
      if (has1) out[g_o_off + size_t(t) * uint(N) + row0 + 1u] = T(s1);
    }
  }
"""


@lru_cache(maxsize=32)
def _mxfp8_gemv_4sg_kernel(dt_name: str, rows: int, N: int, K: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mxfp8_mv4sk_{dt_name}_r{rows}_n{N}_k{K}",
        input_names=["x", "w", "scales"],
        output_names=["out"],
        header=_MXFP8_LUT_HEADER,
        source=_MXFP8_GEMV_4SG_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _mxfp8_gemv_1sg_kernel(dt_name: str, rows: int, N: int, K: int):
    return mx.fast.metal_kernel(
        name=f"ernie_mxfp8_mv1sg_{dt_name}_r{rows}_n{N}_k{K}",
        input_names=["x", "w", "scales"],
        output_names=["out"],
        header=_MXFP8_LUT_HEADER,
        source=_MXFP8_GEMV_1SG_SRC,
        ensure_row_contiguous=True,
    )


def mxfp8_linear_fast(linear: Any, x: mx.array) -> mx.array:
    """High-occupancy multi-row MXFP8 GEMV (shares weight loads across T=1..8 verify rows!)."""
    if (
        getattr(linear, "mode", None) == "mxfp8"
        and getattr(linear, "bits", None) == 8
        and getattr(linear, "group_size", None) == 32
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        K = x.shape[-1]
        N = int(linear.weight.shape[0])
        T = x.size // K
        if 1 <= T <= 15 and K % 1024 == 0:
            lead = x.shape[:-1]
            x2 = x.reshape(T, K)
            dt_name = "bf16" if x.dtype == mx.bfloat16 else "f16"
            if T <= 8:
                if N < 2048 and K % 4096 == 0:
                    out = _mxfp8_gemv_4sg_kernel(dt_name, T, N, K)(
                        inputs=[x2, linear.weight, linear.scales],
                        template=[("T", x.dtype), ("ROWS", T), ("N", N), ("K", K)],
                        grid=(N * 128, 1, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(T, N)],
                        output_dtypes=[x.dtype],
                    )[0]
                else:
                    n_tg = (N + 7) // 8
                    out = _mxfp8_gemv_1sg_kernel(dt_name, T, N, K)(
                        inputs=[x2, linear.weight, linear.scales],
                        template=[("T", x.dtype), ("ROWS", T), ("N", N), ("K", K)],
                        grid=(n_tg * 128, 1, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(T, N)],
                        output_dtypes=[x.dtype],
                    )[0]
                return out.reshape(*lead, N)
            parts = [mxfp8_linear_fast(linear, x2[i : i + 8]) for i in range(0, T, 8)]
            return mx.concatenate(parts, axis=0).reshape(*lead, N)
    return linear(x)


def mxfp8_multi_linear_fast(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    """3-D grouped MXFP8 multi-row GEMV (`wo_a`: [B=1, G, T, K] @ [G, N, K/4]) sharing weights for T=1..8."""
    if x.ndim == 4 and x.shape[0] == 1 and x.dtype in (mx.bfloat16, mx.float16):
        _, G, T, K = x.shape
        N = int(weight.shape[1])
        if 1 <= T <= 8 and K % 1024 == 0:
            dt_name = "bf16" if x.dtype == mx.bfloat16 else "f16"
            n_tg = (N + 7) // 8
            out = _mxfp8_gemv_1sg_kernel(dt_name, T, N, K)(
                inputs=[x[0], weight, scales],
                template=[("T", x.dtype), ("ROWS", T), ("N", N), ("K", K)],
                grid=(n_tg * 128, G, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(G, T, N)],
                output_dtypes=[x.dtype],
            )[0]
            return out[None]
    return mx.quantized_matmul(
        x, weight, scales=scales, transpose=True, group_size=32, bits=8, mode="mxfp8"
    )


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

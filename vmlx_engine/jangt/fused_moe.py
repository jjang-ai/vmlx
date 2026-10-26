"""Fused JANGT routed + mxfp8 shared-expert MoE decode (<= 8 rows) in four Metal dispatches.

WHY: a DeepSeek-V4.1 JANGHT decode layer ran the routed experts in 4 dispatches (rotate x, trellis gate/up/SwiGLU,
rotate h, trellis down) plus the mxfp8 shared expert as ~10 more (w1, w3, two casts, clip, minimum, silu, multiply,
cast, w2, cast, add): ~14 small dispatches per layer x 40 layers per token, the dispatch-bound part of decode (the
affine bundles got the same treatment in jang_tools.dsv41.metal_decode.moe_decode, mlx-serve MOE_GU / MOE_DN).

HOW (per layer, R rows, k routed slots):
  1. rotate:  xr = H128(x * su_gu)                                   (moe_kernels.rotate_rows, unchanged)
  2. GU:      grid z = R x (k + 1). Slot < k: the v3 trellis gate+up+SwiGLU of the slot's expert on xr (+ outlier
              columns from the raw x), exactly moe_kernels._gu_v3. Slot k: the shared expert's w1 / w3 rows on the raw
              bf16 x with MLX's mxfp8 qmv_fast arithmetic (8 values per lane per 256-value block, E8M0 group scale,
              simd_sum) rounded to bf16 as quantized_matmul returns them, then the stock clamp and
              silu(g) * u in f32 (MLX's sigmoid: 1 / (1 + precise::exp(|g|)), sign select) rounded to bf16.
  3. rotate:  hr = H128(h * sv_dn[expert])                           (unchanged)
  4. DN:      per row: the v3 trellis down over the k slots (fma(w, y, acc) in slot order), rounded to bf16 as the
              routed output was; the shared w2 rows on the bf16 shared h with the qmv_fast arithmetic, rounded to
              bf16; their f32 sum rounded to bf16 (stock: f32(bf16 routed) + f32(bf16 shared) -> bf16).
Every row is computed exactly as its 1-row decode. Bit-identity with the stock path (each row run alone) is checked on
the first call per shape; a mismatch disables the fused path (logged) and the caller keeps the stock path.
"""
from __future__ import annotations

import functools
import logging
import os

import mlx.core as mx

from vmlx_engine.jangt import moe_kernels as MK
from vmlx_engine.jangt import v2l as VL

logger = logging.getLogger(__name__)

MAX_ROWS = 8
_U = MK._UNROLL
_STATE = {"disabled": False, "checked": set()}

_HEADER = r"""
inline float jt_e4m3(uint8_t b) {
  uint16_t v = b & 127;
  uint16_t sign_bit = ((uint16_t)((b >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted * 256.0);
}
inline float jt_e8m0(uint8_t b) {
  uint32_t out = (b == 0 ? 0x400000 : (static_cast<uint16_t>(b) << 23));
  return as_type<float>(out);
}
inline float jt_sigmoid(float x) {
  float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""


def requested() -> bool:
    """VMLX_JANGT_FUSED_MOE=0 keeps the stock routed + shared path (read per call: in-process A/B)."""
    return os.environ.get("VMLX_JANGT_FUSED_MOE", "1").strip().lower() not in {"0", "false", "off", "no"}


def _fp8_rows4(K: int, xexpr: str, w: str, s: str, res: str) -> str:
    """MLX fp_qmv_fast_impl for this simdgroup's 4 output rows (row0..row0+3) of an mxfp8 [N, K] weight against one
    activation row read through `xexpr(i)` (i = absolute input index): res[r] (lane partials, simd_sum after)."""
    return f"""
    {{ const device uint8_t* wb = (const device uint8_t*){w} + (size_t)row0 * {K}u + lane * 8u;
      const device uint8_t* sb = {s} + (size_t)row0 * {K // 32}u + lane / 4u;
      for (uint k = 0; k < {K}u; k += 256u) {{
        float xt[8];
        {_U} for (uint i = 0; i < 8u; i++) xt[i] = float({xexpr("k + lane * 8u + i")});
        {_U} for (uint r = 0; r < 4u; r++) {{
          const device uint8_t* wl = wb + r * {K}u;
          float sc = jt_e8m0(sb[r * {K // 32}u]);
          float accum = 0.0f;
          {_U} for (uint i = 0; i < 8u; i++) accum += xt[i] * jt_e4m3(wl[i]);
          {res}[r] += sc * accum;
        }}
        wb += 256u; sb += 8u;
      }} }}"""


@functools.lru_cache(maxsize=None)
def _gu_kernel(D: int, F: int, K, L: int, n_out: int, W: int, topk: int, limit: float, guard: bool):
    mu, isd = VL.code_stats(L, K); PL, adv, _ = MK._lane_geom(K, L)
    gu_body = f"""{MK._xload3(guard)}
      {_U} for (uint r = 0; r < 4u; r++) {{
        half2 ag = half2(0.0h), au = half2(0.0h);
        const device uint32_t* wr = wp + r * {W}u; {MK._tdot3(K, L, "wr", "ag")}
        const device uint32_t* ur = up + r * {W}u; {MK._tdot3(K, L, "ur", "au")}
        g0[r] += float(ag.x) * xi; g1[r] += float(ag.y) * xi; u0[r] += float(au.x) * xi; u1[r] += float(au.y) * xi;
      }}"""
    outl_pre = "" if n_out == 0 else f"""
      float og[4] = {{0.0f, 0.0f, 0.0f, 0.0f}}, ou[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float xo = float(x[(size_t)t * {D}u + uint(c)]);
          {_U} for (uint r = 0; r < 4u; r++) {{ og[r] = xo * float(woutg[((size_t)e * {F}u + row0 + r) * {n_out}u + lane]);
                                                 ou[r] = xo * float(woutu[((size_t)e * {F}u + row0 + r) * {n_out}u + lane]); }} }} }}
      {_U} for (uint r = 0; r < 4u; r++) {{ og[r] = simd_sum(og[r]); ou[r] = simd_sum(ou[r]); }}"""
    outl = "" if n_out == 0 else " g += og[r]; u += ou[r];"
    clamp = "" if not limit > 0 else f" g = metal::min(g, {float(limit)}f); u = metal::clamp(u, {-float(limit)}f, {float(limit)}f);"
    shared_clamp = "" if not limit > 0 else f" su_v = metal::clamp(su_v, {-float(limit)}f, {float(limit)}f); sg_v = metal::min(sg_v, {float(limit)}f);"
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint z = threadgroup_position_in_grid.z;
    uint t = z / {topk + 1}u; uint slot = z - t * {topk + 1}u;
    if (slot < {topk}u) {{
      uint ts = t * {topk}u + slot; uint e = idx[ts];{outl_pre}
      const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
      const device uint32_t* wp = wg + ((size_t)e * {F}u + row0) * {W}u + (lb >> 5);
      const device uint32_t* up = wu + ((size_t)e * {F}u + row0) * {W}u + (lb >> 5);
      const device float* xp = xr + (size_t)t * {D}u + lane * 16u;
      float g0[4], g1[4], u0[4], u1[4];
      {_U} for (uint r = 0; r < 4u; r++) {{ g0[r] = 0.0f; g1[r] = 0.0f; u0[r] = 0.0f; u1[r] = 0.0f; }}
      float se = 0.0f, so = 0.0f;
      {MK._blocks(D, gu_body, f"wp += {adv}u; up += {adv}u; xp += 512u;")}
      float SE = simd_sum(se), SO = simd_sum(so);
      {_U} for (uint r = 0; r < 4u; r++) {{
        float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
        if (lane == 0u) {{
          float g = ((G0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (G1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sg[(size_t)e * {F}u + row0 + r];
          float u = ((U0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (U1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * su_[(size_t)e * {F}u + row0 + r];{outl}{clamp}
          hr_out[(size_t)ts * {F}u + row0 + r] = (g / (1.0f + metal::fast::exp(-g))) * u;
        }}
      }}
    }} else {{
      float rg[4] = {{0.0f, 0.0f, 0.0f, 0.0f}}, ru[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      {_fp8_rows4(D, lambda i: f"x[(size_t)t * {D}u + {i}]", "w1", "s1", "rg")}
      {_fp8_rows4(D, lambda i: f"x[(size_t)t * {D}u + {i}]", "w3", "s3", "ru")}
      {_U} for (uint r = 0; r < 4u; r++) {{
        float G = simd_sum(rg[r]), U = simd_sum(ru[r]);
        if (lane == 0u) {{
          float sg_v = float(static_cast<bfloat16_t>(G)), su_v = float(static_cast<bfloat16_t>(U));{shared_clamp}
          float silu = sg_v * jt_sigmoid(sg_v);
          hs_out[(size_t)t * {F}u + row0 + r] = static_cast<bfloat16_t>(silu * su_v);
        }}
      }}
    }}
    """
    ins = ["xr", "x", "idx", "wg", "sg", "wu", "su_", "w1", "s1", "w3", "s3"] + (["ocol", "woutg", "woutu"] if n_out else [])
    lim = "" if not limit > 0 else f"_lim{str(float(limit)).replace('.', 'p')}"
    return mx.fast.metal_kernel(name=f"jtf_gu_d{D}_f{F}_K{MK._kname(K)}_o{n_out}_w{W}_k{topk}{lim}{'' if guard else '_ng'}",
                                input_names=ins, output_names=["hr_out", "hs_out"], source=src, header=_HEADER)


@functools.lru_cache(maxsize=None)
def _dn_kernel(F: int, D: int, K, L: int, n_out: int, topk: int, W: int, guard: bool):
    mu, isd = VL.code_stats(L, K); PL, adv, _ = MK._lane_geom(K, L)
    dn_body = f"""{MK._xload3(guard)}
        {_U} for (uint r = 0; r < 4u; r++) {{
          half2 ad = half2(0.0h);
          const device uint32_t* wr = wp + r * {W}u; {MK._tdot3(K, L, "wr", "ad")}
          d0[r] += float(ad.x) * xi; d1[r] += float(ad.y) * xi;
        }}"""
    outl_pre = "" if n_out == 0 else f"""
      float od[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts * {F}u + uint(c)]);
          {_U} for (uint r = 0; r < 4u; r++) od[r] = ho * float(wout[((size_t)e * {D}u + row0 + r) * {n_out}u + lane]); }} }}
      {_U} for (uint r = 0; r < 4u; r++) od[r] = simd_sum(od[r]);"""
    outl = "" if n_out == 0 else " y += od[r];"
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint t = threadgroup_position_in_grid.z;
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint sl = 0; sl < {topk}u; sl++) {{
      uint ts = t * {topk}u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);{outl_pre}
      const device uint32_t* wp = wd + ((size_t)e * {D}u + row0) * {W}u + (lb >> 5);
      const device float* xp = hr + (size_t)ts * {F}u + lane * 16u;
      float d0[4], d1[4];
      {_U} for (uint r = 0; r < 4u; r++) {{ d0[r] = 0.0f; d1[r] = 0.0f; }}
      float se = 0.0f, so = 0.0f;
      {MK._blocks(F, dn_body, f"wp += {adv}u; xp += 512u;")}
      float SE = simd_sum(se), SO = simd_sum(so);
      {_U} for (uint r = 0; r < 4u; r++) {{
        float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
        float y = ((D0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (D1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sd[(size_t)e * {D}u + row0 + r];{outl}
        acc[r] = fma(wgt, y, acc[r]);
      }}
    }}
    float rs[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    {_fp8_rows4(F, lambda i: f"hs[(size_t)t * {F}u + {i}]", "w2", "s2", "rs")}
    {_U} for (uint r = 0; r < 4u; r++) {{
      float S = simd_sum(rs[r]);
      if (lane == 0u) {{
        float routed = float(static_cast<bfloat16_t>(acc[r]));
        float shared = float(static_cast<bfloat16_t>(S));
        out[(size_t)t * {D}u + row0 + r] = static_cast<bfloat16_t>(routed + shared);
      }}
    }}
    """
    ins = ["hr", "h", "idx", "rw", "wd", "sd", "hs", "w2", "s2"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jtf_dn_f{F}_d{D}_K{MK._kname(K)}_o{n_out}_k{topk}_w{W}{'' if guard else '_ng'}",
                                input_names=ins, output_names=["out"], source=src, header=_HEADER)


def _eligible(x, idx, experts, shared) -> bool:
    if x.ndim != 2 or idx.ndim != 2 or x.dtype != mx.bfloat16 or x.shape[0] > MAX_ROWS:
        return False
    if getattr(experts, "gu_fmt", None) != "jangt" or getattr(experts, "dn_fmt", None) != "jangt":
        return False
    lin = (getattr(shared, "w1", None), getattr(shared, "w3", None), getattr(shared, "w2", None))
    if any(getattr(m, "mode", None) != "mxfp8" or getattr(m, "bits", None) != 8 or getattr(m, "group_size", None) != 32
           for m in lin):
        return False
    D, F = experts.D, experts.I
    return (D % 256 == 0 and F % 256 == 0 and D % 8 == 0 and F % 8 == 0 and x.shape[1] == D
            and tuple(shared.w1.weight.shape) == (F, D // 4) and tuple(shared.w2.weight.shape) == (D, F // 4))


def _run(x, idx, w, experts, shared, limit: float):
    U = experts._unit()
    R, D = x.shape
    k = idx.shape[1]
    F = experts.I
    idx_u = idx.reshape(-1).astype(mx.uint32)
    guard = MK._fp16_guard()
    g, u, d = U["gate"], U["up"], U["down"]
    xr = MK.rotate_rows(x, U["su_gu"])
    gu = _gu_kernel(D, F, U["K"], U["L"], g["n_out"], int(g["packed"].shape[-1]), k, float(limit), guard)
    ins = [xr, x, idx_u, g["packed"], g["scale"], u["packed"], u["scale"],
           shared.w1.weight, shared.w1.scales, shared.w3.weight, shared.w3.scales]
    if g["n_out"]:
        ins += [g["ocol"], g["wout"], u["wout"]]
    h, hs = gu(inputs=ins, grid=(64, F // 8, R * (k + 1)), threadgroup=(64, 1, 1),
               output_shapes=[(R * k, F), (R, F)], output_dtypes=[mx.float32, mx.bfloat16],
               )
    hr = MK.rotate_rows(h, U["sv_dn"], idx_u)
    dn = _dn_kernel(F, D, U["K_dn"], U["L"], d["n_out"], k, int(d["packed"].shape[-1]), guard)
    ins = [hr, h, idx_u, w.reshape(-1).astype(mx.float32), d["packed"], d["scale"], hs, shared.w2.weight, shared.w2.scales]
    if d["n_out"]:
        ins += [d["ocol"], d["wout"]]
    return dn(inputs=ins, grid=(64, D // 8, R), threadgroup=(64, 1, 1),
              output_shapes=[(R, D)], output_dtypes=[mx.bfloat16])[0]


def _stock_row(x1, idx1, w1, experts, shared, limit: float):
    """The stock JANGT routed + mxfp8 shared path for ONE row (jang_tools.dsv41.model.Block.moe / _shared)."""
    y = experts.routed(x1, idx1, w1).astype(mx.float32)
    sg = shared.w1(x1).astype(mx.float32)
    su = shared.w3(x1).astype(mx.float32)
    if limit > 0:
        su = mx.clip(su, -limit, limit)
        sg = mx.minimum(sg, limit)
    import mlx.nn as nn
    y = y + shared.w2((nn.silu(sg) * su).astype(mx.bfloat16)).astype(mx.float32)
    return y.astype(x1.dtype)


def jt_moe_decode(x, idx, w, experts, shared, limit: float):
    """y [R, D] bf16 for x [R, D] bf16 (R <= 8), idx [R, k], w [R, k] f32 (normalized, scaled), a JTMixedSwitchGLU and
    a shared expert of mxfp8 Linears w1/w3/w2; None when not eligible / disabled / the self-check failed."""
    if _STATE["disabled"] or not requested() or not _eligible(x, idx, experts, shared):
        return None
    y = _run(x, idx, w, experts, shared, float(limit))
    key = (x.shape[1], experts.I, idx.shape[1], float(limit), experts._unit()["K"], experts._unit()["K_dn"])
    if key not in _STATE["checked"]:
        ref = mx.concatenate([_stock_row(x[i:i + 1], idx[i:i + 1], w[i:i + 1], experts, shared, float(limit))
                              for i in range(x.shape[0])], axis=0)
        ok = bool(mx.array_equal(y, ref).item())
        if not ok:
            diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item())
            logger.warning("JANGT fused MoE decode: not bit-identical to the stock path (max |d| %.3g); disabled", diff)
            _STATE["disabled"] = True
            return None
        _STATE["checked"].add(key)
    return y

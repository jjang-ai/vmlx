"""JANGT MoE decode kernels (gathered experts, M = few tokens) — reference implementation for the vMLX runtime.

One MoE layer, T tokens x k selected experts ("slots"), JANGT units with ONE width K per unit (B93/B97 layout):
  1. xr  = rotate_rows(x * su_gu)                    one launch: H128 of every token's input (signs per LAYER,
                                                     0 on the layer's outlier columns) — shared by all experts
  2. h   = gather_gate_up_swiglu(xr, x, idx)         one launch, grid z = (token, slot): trellis decode of gate AND
                                                     up rows of expert idx[slot], SwiGLU epilogue, outlier cols
  3. hr  = rotate_rows(h * (su_dn * cs[idx]))        one launch over all (token, slot) rows; per-EXPERT column
                                                     scales cs (down colnorm + LS refit) folded into the signs
  4. y   = gather_down_weighted(hr, h, idx, w)       one launch: every threadgroup owns 8 output rows of one token
                                                     and loops over that token's k slots, accumulating
                                                     w_slot * (down_e . hr_slot) in registers -> no reduction op
Bit extraction: generic 96-bit window per lane (any K pattern: 2, 2.5, 3), lane = 16 consecutive weights = 8 steps.
All weights of a unit are stored (E, N, W) uint32 (ragged files with uniform K reshape to this at load).
"""
from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from vmlx_engine.jangt import v2l as VL
from vmlx_engine.jangt.v2core import MASK, ORB


def _dt(a: mx.array) -> str:
    return {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[a.dtype]


def _kname(K) -> str:
    return str(K).replace(".", "p")


import os as _os
HALF_ACC = _os.environ.get("JANGT_HALF_ACC", "1") == "1"
CODE_VARIANT = _os.environ.get("JANGT_CODE_PROBE", "mul32")   # mul32 = the real code; others are timing probes only   # packed half2 FMA per 16 weights (fp32 across chunks)


def _decode_lane(K, L: int, W: int, wexpr: str, acc0: str, acc1: str, xt: str) -> str:
    """Metal snippet: decode this lane's 8 steps of row `wexpr` (uint* to the row) and FMA into acc0/acc1.
    32-bit only (Apple GPUs have no native 64-bit integer ALU; the first version used ulong shifts and ran at
    ~60 % of DRAM bandwidth): the lane's window is 2 (K=2) or 3 words; word select by compile-time step offset +
    runtime sub-word shift, funnel shift with 32-bit ops."""
    so = VL.step_offsets(K, 9); PL = int(VL.step_offsets(K, 17)[16]) // 2; mk = (1 << L) - 1; MU = VL.MULTS[K]
    nwin = 2 if K == 2 else 3
    # lane segments are contiguous: lane+1 starts exactly PL bits later. K=2 (PL=32): the lookahead word is the
    # next lane's own word -> simd_shuffle_down, lane 31 loads it. Other widths keep explicit loads (PL != 32).
    if K == 2:
        loads = "uint w0 = wr[wi]; uint w1 = simd_shuffle_down(w0, 1u); if (lane == 31u) w1 = wr[wi + 1u];"
    else:
        loads = "uint w0 = wr[wi], w1 = wr[wi + 1u], w2 = wr[wi + 2u];"
    steps = []
    for j in range(8):
        o = int(so[j])
        if K == 2:   # PL = 32: sh == 0 always -> fully static
            wi_, sh_ = o >> 5, o & 31
            lo, hi = f"w{wi_}", f"w{wi_ + 1}"
            ex = lo if sh_ == 0 else f"(({lo} >> {sh_}u) | ({hi} << {32 - sh_}u))"
        else:
            # b = sh + o, sh in [0, 31]; o static -> word index is o>>5 or (o>>5)+1 depending on sh + (o & 31)
            base = o >> 5; r = o & 31
            ex = (f"({{ uint bb = sh + {r}u; uint lo = (bb < 32u) ? w{base} : w{min(base + 1, nwin - 1)}; "
                  f"uint hi = (bb < 32u) ? w{min(base + 1, nwin - 1)} : w{min(base + 2, nwin - 1)}; uint s5 = bb & 31u; "
                  f"(s5 == 0u) ? lo : ((lo >> s5) | (hi << (32u - s5))); }})")
        if CODE_VARIANT == "mul32":
            hp = f"uint p = ((s * {MU}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;"
        elif CODE_VARIANT == "mul16":   # timing probe: two 16-bit multiplies (would need a new code family)
            hp = (f"ushort s16 = ushort(s); ushort lo = ushort(s16 * ushort({MU & 0xFFFF}u)); ushort hi = ushort(s16 * ushort({(MU >> 16) | 1}u));"
                  f" uint p = ((uint(lo) | (uint(hi) << 16)) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;")
        else:                           # timing probe: no hash (memory + extraction ceiling)
            hp = f"uint p = ((s | (s << 16)) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;"
        steps.append(f"""{{ uint s = ({ex}) & {mk}u;
            {hp} half2 v = as_type<half2>(p);
            ACC }}""".replace("ACC", (f"ah = fma({xt}h[{j}], v, ah);" if HALF_ACC else
                                        f"a0 = fma({xt}[{2*j}], float(v.x), a0); a1 = fma({xt}[{2*j+1}], float(v.y), a1);")))
    body = "\n            ".join(steps)
    init = "half2 ah = half2(0.0h);" if HALF_ACC else "float a0 = 0.0f, a1 = 0.0f;"
    fin = f"{acc0} += float(ah.x); {acc1} += float(ah.y);" if HALF_ACC else f"{acc0} += a0; {acc1} += a1;"
    return f"""
        {{ const device uint* wr = {wexpr};
          uint bit0 = (k0 / 2u + lane * 8u) / 8u * {PL}u; uint wi = bit0 >> 5, sh = bit0 & 31u; (void)sh;   // 32-bit
          {loads}
          {init}
            {body}
          {fin} }}"""


# ------------------------------------------------------------------------------------------- 1/3: row rotation
@functools.lru_cache(maxsize=None)
def _rot_rows_kernel(K: int, xdt: str, per_row_signs: bool):
    """out[r] = H128_blockwise(x[r] * s) (f32). s = su (K,) shared, or s = sv[idx_row] (E, K) gathered per row."""
    sidx = "sidx[r]" if per_row_signs else "0u"
    src = f"""
    uint lane = thread_index_in_simdgroup; uint blk = threadgroup_position_in_grid.x; uint r = threadgroup_position_in_grid.y;
    size_t base = (size_t)r * {K}u + blk * 128u + lane * 4u;
    const device float* sv = s + (size_t)({sidx}) * {K}u;
    float v[4];
    for (uint i = 0; i < 4u; i++) v[i] = float(x[base + i]) * sv[blk * 128u + lane * 4u + i];
    {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }}
    {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }}
    for (uint d = 1u; d < 32u; d <<= 1) {{
      bool upper = (lane & d) != 0u;
      for (uint i = 0; i < 4u; i++) {{ float o = simd_shuffle_xor(v[i], d); v[i] = upper ? (o - v[i]) : (v[i] + o); }}
    }}
    for (uint i = 0; i < 4u; i++) out[base + i] = v[i] * 0.08838834764831845f;
    """
    ins = ["x", "s"] + (["sidx"] if per_row_signs else [])
    return mx.fast.metal_kernel(name=f"jt_rotrows_k{K}_{xdt}_{int(per_row_signs)}", input_names=ins, output_names=["out"], source=src)


def rotate_rows(x: mx.array, s: mx.array, sidx: mx.array | None = None) -> mx.array:
    R, K = x.shape
    ins = [x, s] + ([sidx.astype(mx.uint32)] if sidx is not None else [])
    return _rot_rows_kernel(K, _dt(x), sidx is not None)(inputs=ins, grid=(32 * (K // 128), R, 1), threadgroup=(32, 1, 1),
                                                         output_shapes=[(R, K)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- 2: gate+up+SwiGLU
@functools.lru_cache(maxsize=None)
def _gu_kernel(kin: int, N: int, K, L: int, n_out: int, xdt: str, odt: str, RS: int = 0):
    W = RS or VL.words_per_row(kin, L, K); mu, isd = VL.code_stats(L, K)
    outl = "" if n_out == 0 else f"""
          for (uint j = 0; j < {n_out}u; j++) {{ int c = ocol[(size_t)e * {n_out}u + j]; if (c < 0) continue;
            float xo = float(x[(size_t)t * {kin}u + uint(c)]);
            g = fma(xo, float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), g);
            u = fma(xo, float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), u); }}"""
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint ts = threadgroup_position_in_grid.z;
    uint t = ts / kslots[0]; uint e = idx[ts];
    const device float* xp = xr + (size_t)t * {kin}u;
    const device uint* wgE = wg + ((size_t)e * {N}u + row0) * {W}u; const device uint* wuE = wu + ((size_t)e * {N}u + row0) * {W}u;
    const device uint* rg[4]; const device uint* ru[4];
    for (uint r = 0; r < 4u; r++) {{ rg[r] = wgE + r * {W}u; ru[r] = wuE + r * {W}u; }}      // 64-bit math ONCE
    float g0[4] = {{0,0,0,0}}, g1[4] = {{0,0,0,0}}, u0[4] = {{0,0,0,0}}, u1[4] = {{0,0,0,0}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      float xt[16];
      for (uint i = 0; i < 4u; i++) {{ float4 q = ((const device float4*)xp)[(k0 + lane * 16u) / 4u + i];
        xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      half2 xth[8]; for (uint i = 0; i < 8u; i++) xth[i] = half2(half(xt[2u*i]), half(xt[2u*i+1u]));
      for (uint r = 0; r < 4u; r++) {{
        {_decode_lane(K, L, W, "rg[r]", "g0[r]", "g1[r]", "xt")}
        {_decode_lane(K, L, W, "ru[r]", "u0[r]", "u1[r]", "xt")}
      }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < 4u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {{
        float g = ((G0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (G1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sg[(size_t)e * {N}u + row0 + r];
        float u = ((U0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (U1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * su_[(size_t)e * {N}u + row0 + r];{outl}
        out[(size_t)ts * {N}u + row0 + r] = {odt}((g / (1.0f + metal::fast::exp(-g))) * u);
      }}
    }}
    """
    ins = ["xr", "x", "idx", "kslots", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt_gu_k{kin}_n{N}_K{_kname(K)}_o{n_out}_{xdt}_{odt}_h{int(HALF_ACC)}_rs{W}_{CODE_VARIANT}_v2", input_names=ins, output_names=["out"], source=src)


def gather_gate_up_swiglu(xr, x, idx, U, out_dtype=mx.bfloat16):
    """xr (T, kin) f32 rotated, x (T, kin) raw, idx (T, k) expert ids; U = unit dict (see load_unit). -> (T*k, N)."""
    T, kin = x.shape; k = idx.shape[1]; N = U["gate"]["scale"].shape[1]
    n_out = U["gate"]["n_out"]
    ins = [xr, x, idx.reshape(-1).astype(mx.uint32), mx.array([k], dtype=mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    kern = _gu_kernel(kin, N, U["K"], U["L"], n_out, _dt(x), {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[out_dtype],
                      int(U["gate"]["packed"].shape[-1]))
    return kern(inputs=ins, grid=(64, N // 8, T * k), threadgroup=(64, 1, 1), output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


# ------------------------------------------------------------------------------------------- 4: weighted down
@functools.lru_cache(maxsize=None)
def _down_kernel(kin: int, N: int, K, L: int, n_out: int, kslots: int, hdt: str, RS: int = 0):
    W = RS or VL.words_per_row(kin, L, K); mu, isd = VL.code_stats(L, K)
    outl = "" if n_out == 0 else f"""
            for (uint j = 0; j < {n_out}u; j++) {{ int c = ocol[(size_t)e * {n_out}u + j]; if (c < 0) continue;
              y = fma(float(h[(size_t)ts * {kin}u + uint(c)]), float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), y); }}"""
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint t = threadgroup_position_in_grid.z;
    float acc[4] = {{0,0,0,0}};
    for (uint sl = 0; sl < {kslots}u; sl++) {{
      uint ts = t * {kslots}u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);
      const device float* xp = hr + (size_t)ts * {kin}u;
      const device uint* wE = wd + ((size_t)e * {N}u + row0) * {W}u;
      const device uint* rd[4]; for (uint r = 0; r < 4u; r++) rd[r] = wE + r * {W}u;
      float d0[4] = {{0,0,0,0}}, d1[4] = {{0,0,0,0}}; float se = 0.0f, so = 0.0f;
      for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
        float xt[16];
        for (uint i = 0; i < 4u; i++) {{ float4 q = ((const device float4*)xp)[(k0 + lane * 16u) / 4u + i];
          xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
        for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
        half2 xth[8]; for (uint i = 0; i < 8u; i++) xth[i] = half2(half(xt[2u*i]), half(xt[2u*i+1u]));
        for (uint r = 0; r < 4u; r++) {{
          {_decode_lane(K, L, W, "rd[r]", "d0[r]", "d1[r]", "xt")}
        }}
      }}
      float SE = simd_sum(se), SO = simd_sum(so);
      for (uint r = 0; r < 4u; r++) {{
        float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
        float y = ((D0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (D1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sd[(size_t)e * {N}u + row0 + r];{outl}
        acc[r] = fma(wgt, y, acc[r]);
      }}
    }}
    if (lane == 0u) for (uint r = 0; r < 4u; r++) out[(size_t)t * {N}u + row0 + r] = acc[r];
    """
    ins = ["hr", "h", "idx", "rw", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt_dn_k{kin}_n{N}_K{_kname(K)}_o{n_out}_s{kslots}_{hdt}_h{int(HALF_ACC)}_rs{W}_{CODE_VARIANT}_v2", input_names=ins, output_names=["out"], source=src)


def gather_down_weighted(hr, h, idx, rw, U):
    """hr (T*k, I) f32 rotated (with column scales), h (T*k, I) raw, idx (T, k), rw (T, k) router weights -> (T, D) f32."""
    T, k = idx.shape; kin = h.shape[1]; N = U["down"]["scale"].shape[1]; n_out = U["down"]["n_out"]
    ins = [hr, h, idx.reshape(-1).astype(mx.uint32), rw.reshape(-1).astype(mx.float32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    kern = _down_kernel(kin, N, U["K_dn"], U["L"], n_out, k, _dt(h), int(U["down"]["packed"].shape[-1]))
    return kern(inputs=ins, grid=(64, N // 8, T), threadgroup=(64, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- loading + full layer
def _proj(sf: dict, m: str, kin: int, L: int) -> dict:
    kb = np.array(sf[m + ".jt_kbits"]).reshape(-1); assert (kb == kb[0]).all(), f"{m}: per-expert widths need the ragged kernel"
    K = float(kb[0]); K = int(K) if K.is_integer() else K
    scale = sf[m + ".jt_scale"]; E, N = scale.shape; W = VL.words_per_row(kin, L, K)
    flat = sf[m + ".jt_packed"]; assert flat.size == E * N * W, (m, flat.size, E * N * W)
    oc = sf[m + ".jt_ocol"]; n_out = int(oc.shape[-1]) if int(mx.max(oc).item()) >= 0 else 0
    return {"K": K, "kin": kin, "packed": flat.reshape(E, N, W), "scale": scale.astype(mx.float32), "su": sf[m + ".jt_su"].astype(mx.float32),
            "ocol": oc.astype(mx.int32), "wout": sf[m + ".jt_wout"].astype(mx.float16), "n_out": n_out,
            "cs": sf.get(m + ".jt_cs"), "N": N}


def load_unit(sf: dict, layer: int, D: int = 4096, I: int = 2048, L: int = 12) -> dict:
    """The JANGT groups of one layer (gate_up and/or down). Groups stored as JANGH elsewhere are absent (None);
    the runtime composes them with the JANGH kernels."""
    b = f"model.layers.{layer}.mlp.switch_mlp."
    has = lambda p: b + p + "_proj.jt_packed" in sf
    U = {"L": L, "gate": None, "up": None, "down": None, "K": None, "K_dn": None}
    if has("gate"):
        g, u = _proj(sf, b + "gate_proj", D, L), _proj(sf, b + "up_proj", D, L); assert g["K"] == u["K"]
        U.update(gate=g, up=u, K=g["K"], su_gu=g["su"] * mx.array(_mask(D, g["ocol"]))); mx.eval(U["su_gu"])
    if has("down"):
        d = _proj(sf, b + "down_proj", I, L)
        cs = d["cs"].astype(mx.float32) if d["cs"] is not None else mx.ones((d["scale"].shape[0], I))
        U.update(down=d, K_dn=d["K"], sv_dn=cs * d["su"][None, :] * mx.array(_mask_rows(I, d["ocol"]))); mx.eval(U["sv_dn"])
    return U


def _mask(K: int, ocol: mx.array) -> np.ndarray:
    m = np.ones(K, np.float32); c = np.array(ocol)[0]; m[c[c >= 0]] = 0.0; return m


def _mask_rows(K: int, ocol: mx.array) -> np.ndarray:
    oc = np.array(ocol); m = np.ones((oc.shape[0], K), np.float32)
    for e in range(oc.shape[0]):
        c = oc[e][oc[e] >= 0]; m[e, c] = 0.0
    return m


def moe_decode(x: mx.array, idx: mx.array, rw: mx.array, U: dict) -> mx.array:
    """x (T, D) bf16, idx (T, k), rw (T, k) -> (T, D) bf16. 4 launches."""
    xr = rotate_rows(x, U["su_gu"])
    h = gather_gate_up_swiglu(xr, x, idx, U, out_dtype=x.dtype)
    hr = rotate_rows(h, U["sv_dn"], idx.reshape(-1))
    return gather_down_weighted(hr, h, idx, rw, U).astype(x.dtype)


# ------------------------------------------------------------------------------------------- word-aligned fast path
WPL = {2: 16, 3: 32, 2.5: 64}          # weights per lane so a lane's bit segment is whole 32-bit words


def _lane_decode_static(K, L: int, rows: list[tuple[str, str, str]], xt: str) -> str:
    """Fully unrolled decode of one lane segment for several rows sharing the x registers.
    rows: (word_ptr_expr, acc0, acc1). Bit offsets are compile-time constants (segment starts on a word)."""
    wpl = WPL[K]; steps = wpl // 2; so = VL.step_offsets(K, steps + 1)
    nw = int(so[steps - 1] + L + 31) // 32                       # words touched by this lane (incl. lookahead)
    mk = (1 << L) - 1; MU = VL.MULTS[K]; M2 = MASK | (MASK << 16); O2 = ORB | (ORB << 16)
    out = []
    for (wp, a0, a1) in rows:
        out.append("{ " + " ".join(f"uint w{i} = {wp}[{i}];" for i in range(nw)) + " float s0 = 0.0f, s1 = 0.0f;")
        for j in range(steps):
            b = int(so[j]); wi, sh = b >> 5, b & 31
            ex = f"w{wi}" if sh == 0 else f"((w{wi} >> {sh}u) | (w{wi + 1} << {32 - sh}u))"
            out.append(f"{{ uint p = ((({ex}) & {mk}u) * {MU}u & {M2}u) | {O2}u; half2 v = as_type<half2>(p);"
                       f" s0 = fma({xt}[{2*j}], float(v.x), s0); s1 = fma({xt}[{2*j+1}], float(v.y), s1); }}")
        out.append(f"{a0} += s0; {a1} += s1; }}")
    return "\n        ".join(out)


@functools.lru_cache(maxsize=None)
def _gu_fast(kin: int, N: int, K, L: int, n_out: int, xdt: str, odt: str):
    W = VL.words_per_row(kin, L, K); mu, isd = VL.code_stats(L, K); wpl = WPL[K]; CH = 32 * wpl
    assert kin % CH == 0, (kin, CH)
    lane_words = wpl * (5 if K == 2.5 else 2 * K) // 2 // 32 * 2 if K != 2.5 else 5   # 32-bit words per lane segment
    lane_words = {2: 1, 3: 3, 2.5: 5}[K]
    outl = "" if n_out == 0 else f"""
          for (uint j = 0; j < {n_out}u; j++) {{ int c = ocol[(size_t)e * {n_out}u + j]; if (c < 0) continue;
            float xo = float(x[(size_t)t * {kin}u + uint(c)]);
            g = fma(xo, float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), g);
            u = fma(xo, float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), u); }}"""
    rows = []
    for r in range(4):
        rows.append((f"(wgE + (size_t)(row0 + {r}u) * {W}u + seg)", f"g0[{r}]", f"g1[{r}]"))
        rows.append((f"(wuE + (size_t)(row0 + {r}u) * {W}u + seg)", f"u0[{r}]", f"u1[{r}]"))
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint ts = threadgroup_position_in_grid.z;
    uint t = ts / kslots[0]; uint e = idx[ts];
    const device float* xp = xr + (size_t)t * {kin}u;
    const device uint* wgE = wg + (size_t)e * {N}u * {W}u; const device uint* wuE = wu + (size_t)e * {N}u * {W}u;
    float g0[4] = {{0,0,0,0}}, g1[4] = {{0,0,0,0}}, u0[4] = {{0,0,0,0}}, u1[4] = {{0,0,0,0}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += {CH}u) {{
      uint seg = (k0 / {CH}u * 32u + lane) * {lane_words}u;
      float xt[{wpl}];
      for (uint i = 0; i < {wpl // 4}u; i++) {{ float4 q = ((const device float4*)xp)[(k0 + lane * {wpl}u) / 4u + i];
        xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
      for (uint i = 0; i < {wpl // 2}u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
        {_lane_decode_static(K, L, rows, "xt")}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < 4u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {{
        float g = ((G0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (G1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sg[(size_t)e * {N}u + row0 + r];
        float u = ((U0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (U1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * su_[(size_t)e * {N}u + row0 + r];{outl}
        out[(size_t)ts * {N}u + row0 + r] = {odt}((g / (1.0f + metal::fast::exp(-g))) * u);
      }}
    }}
    """
    ins = ["xr", "x", "idx", "kslots", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt_guf_k{kin}_n{N}_K{_kname(K)}_o{n_out}_{xdt}_{odt}", input_names=ins, output_names=["out"], source=src)


@functools.lru_cache(maxsize=None)
def _down_fast(kin: int, N: int, K, L: int, n_out: int, kslots: int, hdt: str):
    W = VL.words_per_row(kin, L, K); mu, isd = VL.code_stats(L, K); wpl = WPL[K]; CH = 32 * wpl
    assert kin % CH == 0, (kin, CH)
    lane_words = {2: 1, 3: 3, 2.5: 5}[K]
    outl = "" if n_out == 0 else f"""
            for (uint j = 0; j < {n_out}u; j++) {{ int c = ocol[(size_t)e * {n_out}u + j]; if (c < 0) continue;
              y = fma(float(h[(size_t)ts * {kin}u + uint(c)]), float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + j]), y); }}"""
    rows = [(f"(wE + (size_t)(row0 + {r}u) * {W}u + seg)", f"d0[{r}]", f"d1[{r}]") for r in range(4)]
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint t = threadgroup_position_in_grid.z;
    float acc[4] = {{0,0,0,0}};
    for (uint sl = 0; sl < {kslots}u; sl++) {{
      uint ts = t * {kslots}u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);
      const device float* xp = hr + (size_t)ts * {kin}u;
      const device uint* wE = wd + (size_t)e * {N}u * {W}u;
      float d0[4] = {{0,0,0,0}}, d1[4] = {{0,0,0,0}}; float se = 0.0f, so = 0.0f;
      for (uint k0 = 0; k0 < {kin}u; k0 += {CH}u) {{
        uint seg = (k0 / {CH}u * 32u + lane) * {lane_words}u;
        float xt[{wpl}];
        for (uint i = 0; i < {wpl // 4}u; i++) {{ float4 q = ((const device float4*)xp)[(k0 + lane * {wpl}u) / 4u + i];
          xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
        for (uint i = 0; i < {wpl // 2}u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
        {_lane_decode_static(K, L, rows, "xt")}
      }}
      float SE = simd_sum(se), SO = simd_sum(so);
      for (uint r = 0; r < 4u; r++) {{
        float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
        float y = ((D0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (D1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sd[(size_t)e * {N}u + row0 + r];{outl}
        acc[r] = fma(wgt, y, acc[r]);
      }}
    }}
    if (lane == 0u) for (uint r = 0; r < 4u; r++) out[(size_t)t * {N}u + row0 + r] = acc[r];
    """
    ins = ["hr", "h", "idx", "rw", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt_dnf_k{kin}_n{N}_K{_kname(K)}_o{n_out}_s{kslots}_{hdt}", input_names=ins, output_names=["out"], source=src)


def gather_gate_up_swiglu_fast(xr, x, idx, U, out_dtype=mx.bfloat16):
    T, kin = x.shape; k = idx.shape[1]; N = U["gate"]["scale"].shape[1]; n_out = U["gate"]["n_out"]
    ins = [xr, x, idx.reshape(-1).astype(mx.uint32), mx.array([k], dtype=mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    kern = _gu_fast(kin, N, U["K"], U["L"], n_out, _dt(x), {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[out_dtype])
    return kern(inputs=ins, grid=(64, N // 8, T * k), threadgroup=(64, 1, 1), output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


def gather_down_weighted_fast(hr, h, idx, rw, U):
    T, k = idx.shape; kin = h.shape[1]; N = U["down"]["scale"].shape[1]; n_out = U["down"]["n_out"]
    ins = [hr, h, idx.reshape(-1).astype(mx.uint32), rw.reshape(-1).astype(mx.float32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    kern = _down_fast(kin, N, U["K_dn"], U["L"], n_out, k, _dt(h))
    return kern(inputs=ins, grid=(64, N // 8, T), threadgroup=(64, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


def pad_rows(U: dict, align_words: int) -> dict:
    """Re-stride every JANGT projection's rows to a multiple of align_words (experiment: row-alignment cost)."""
    import copy
    V = dict(U)
    for p in ("gate", "up", "down"):
        if U[p] is None:
            continue
        P = U[p]["packed"]; E, N, W = P.shape; Wp = -(-W // align_words) * align_words
        V[p] = dict(U[p]); V[p]["packed"] = mx.concatenate([P, mx.zeros((E, N, Wp - W), dtype=P.dtype)], axis=-1) if Wp > W else P
        mx.eval(V[p]["packed"])
    return V


# ------------------------------------------------------------------------------------------- v3: JANGH qmv structure
# Measured 2026-10-09: the SAME trellis decode inside MLX/JANGH's qmv_fast skeleton (row pointers formed once, pointer
# increments per 512-value block, unrolled x loads, no per-iteration index math) runs at 519 GB/s on N0.5 gate+up K=2
# (65 us vs 111-146 us for the v1/v2 kernels above) — the decode never was the bottleneck, the loop structure was.
_UNROLL = '_Pragma("clang loop unroll(full)")'


def _lane_geom(K):
    """bits per lane per 512-value block, words advanced per block, words the lane window needs."""
    PL = int(VL.step_offsets(K, 17)[16]) // 2          # bits per lane (8 steps)
    return PL, PL * 32 // 32, 2 if K == 2 else 3


def _tdot3(K, L: int, wr: str, acc: str) -> str:
    """Trellis dot of one row's lane chunk -> half2 accumulator `acc` (x in xh[8]). K=2: one word + lookahead word,
    static shifts. K=2.5/3: the lane's sub-word shift `sh` is constant for the whole loop, so the 3-word window is
    realigned ONCE per load (two funnel shifts) and every step is a static shift of a0/a1 (no per-step selects)."""
    so = VL.step_offsets(K, 9); mk = (1 << L) - 1; MU = VL.MULTS[K]; M2 = MASK | (MASK << 16); O2 = ORB | (ORB << 16)
    if K == 2:
        ld = f"uint a0 = {wr}[0], a1 = {wr}[1];"
    else:
        ld = (f"uint w0 = {wr}[0], w1 = {wr}[1], w2 = {wr}[2]; "
              f"uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));")
    st = []
    for j in range(8):
        o = int(so[j]); assert o + L <= 64, (K, o)
        if o + L <= 32:
            ex = "a0" if o == 0 else f"(a0 >> {o}u)"
        elif o >= 32:
            ex = "a1" if o == 32 else f"(a1 >> {o - 32}u)"
        else:
            ex = f"((a0 >> {o}u) | (a1 << {32 - o}u))"
        st.append(f"{{ uint s = ({ex}) & {mk}u; uint p = ((s * {MU}u) & {M2}u) | {O2}u; {acc} = fma(xh[{j}], as_type<half2>(p), {acc}); }}")
    return "{ " + ld + " " + " ".join(st) + " }"


def _xload3() -> str:
    return f"""float xt[16];
      {_UNROLL} for (uint i = 0; i < 16u; i++) xt[i] = xp[i];
      half2 xh[8];
      {_UNROLL} for (uint i = 0; i < 8u; i++) {{ xh[i] = half2(half(xt[2u*i]), half(xt[2u*i+1u])); se += xt[2u*i]; so += xt[2u*i+1u]; }}"""


@functools.lru_cache(maxsize=None)
def _gu_v3(kin: int, N: int, K, L: int, n_out: int, W: int, xdt: str, odt: str):
    mu, isd = VL.code_stats(L, K); PL, adv, _ = _lane_geom(K)
    # outlier columns: lane j < n_out owns column j for the simdgroup's 4 rows, one simd_sum per row (the first
    # version looped them on lane 0: -21 us per layer, 515 -> 389 GB/s)
    outl_pre = "" if n_out == 0 else f"""
    float og[4] = {{0.0f, 0.0f, 0.0f, 0.0f}}, ou[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
      if (c >= 0) {{ float xo = float(x[(size_t)t * {kin}u + uint(c)]);
        {_UNROLL} for (uint r = 0; r < 4u; r++) {{ og[r] = xo * float(woutg[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]);
                                                   ou[r] = xo * float(woutu[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }} }}
    {_UNROLL} for (uint r = 0; r < 4u; r++) {{ og[r] = simd_sum(og[r]); ou[r] = simd_sum(ou[r]); }}"""
    outl = "" if n_out == 0 else " g += og[r]; u += ou[r];"
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint ts = threadgroup_position_in_grid.z;
    uint t = ts / kslots[0]; uint e = idx[ts];{outl_pre}
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wp = wg + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    const device uint32_t* up = wu + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
    const device float* xp = xr + (size_t)t * {kin}u + lane * 16u;
    float g0[4], g1[4], u0[4], u1[4];
    {_UNROLL} for (uint r = 0; r < 4u; r++) {{ g0[r] = 0.0f; g1[r] = 0.0f; u0[r] = 0.0f; u1[r] = 0.0f; }}
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      {_xload3()}
      {_UNROLL} for (uint r = 0; r < 4u; r++) {{
        half2 ag = half2(0.0h), au = half2(0.0h);
        const device uint32_t* wr = wp + r * {W}u; {_tdot3(K, L, "wr", "ag")}
        const device uint32_t* ur = up + r * {W}u; {_tdot3(K, L, "ur", "au")}
        g0[r] += float(ag.x); g1[r] += float(ag.y); u0[r] += float(au.x); u1[r] += float(au.y);
      }}
      wp += {adv}u; up += {adv}u; xp += 512u;
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    {_UNROLL} for (uint r = 0; r < 4u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {{
        float g = ((G0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (G1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sg[(size_t)e * {N}u + row0 + r];
        float u = ((U0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (U1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * su_[(size_t)e * {N}u + row0 + r];{outl}
        out[(size_t)ts * {N}u + row0 + r] = {odt}((g / (1.0f + metal::fast::exp(-g))) * u);
      }}
    }}
    """
    ins = ["xr", "x", "idx", "kslots", "wg", "sg", "wu", "su_"] + (["ocol", "woutg", "woutu"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt4_gu_k{kin}_n{N}_K{_kname(K)}_o{n_out}_w{W}_{xdt}_{odt}", input_names=ins, output_names=["out"], source=src)


@functools.lru_cache(maxsize=None)
def _down_v3(kin: int, N: int, K, L: int, n_out: int, kslots: int, W: int, hdt: str):
    mu, isd = VL.code_stats(L, K); PL, adv, _ = _lane_geom(K)
    outl_pre = "" if n_out == 0 else f"""
      float od[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
      if (lane < {n_out}u) {{ int c = ocol[(size_t)e * {n_out}u + lane];
        if (c >= 0) {{ float ho = float(h[(size_t)ts * {kin}u + uint(c)]);
          {_UNROLL} for (uint r = 0; r < 4u; r++) od[r] = ho * float(wout[((size_t)e * {N}u + row0 + r) * {n_out}u + lane]); }} }}
      {_UNROLL} for (uint r = 0; r < 4u; r++) od[r] = simd_sum(od[r]);"""
    outl = "" if n_out == 0 else " y += od[r];"
    src = f"""
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint t = threadgroup_position_in_grid.z;
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint sl = 0; sl < {kslots}u; sl++) {{
      uint ts = t * {kslots}u + sl; uint e = idx[ts]; float wgt = float(rw[ts]);{outl_pre}
      const device uint32_t* wp = wd + ((size_t)e * {N}u + row0) * {W}u + (lb >> 5);
      const device float* xp = hr + (size_t)ts * {kin}u + lane * 16u;
      float d0[4], d1[4];
      {_UNROLL} for (uint r = 0; r < 4u; r++) {{ d0[r] = 0.0f; d1[r] = 0.0f; }}
      float se = 0.0f, so = 0.0f;
      for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
        {_xload3()}
        {_UNROLL} for (uint r = 0; r < 4u; r++) {{
          half2 ad = half2(0.0h);
          const device uint32_t* wr = wp + r * {W}u; {_tdot3(K, L, "wr", "ad")}
          d0[r] += float(ad.x); d1[r] += float(ad.y);
        }}
        wp += {adv}u; xp += 512u;
      }}
      float SE = simd_sum(se), SO = simd_sum(so);
      {_UNROLL} for (uint r = 0; r < 4u; r++) {{
        float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
        float y = ((D0 - {float(mu[0])}f * SE) * {float(isd[0])}f + (D1 - {float(mu[1])}f * SO) * {float(isd[1])}f) * sd[(size_t)e * {N}u + row0 + r];{outl}
        acc[r] = fma(wgt, y, acc[r]);
      }}
    }}
    if (lane == 0u) {{ {_UNROLL} for (uint r = 0; r < 4u; r++) out[(size_t)t * {N}u + row0 + r] = acc[r]; }}
    """
    ins = ["hr", "h", "idx", "rw", "wd", "sd"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jt4_dn_k{kin}_n{N}_K{_kname(K)}_o{n_out}_s{kslots}_w{W}_{hdt}", input_names=ins, output_names=["out"], source=src)


def gather_gate_up_swiglu_v3(xr, x, idx, U, out_dtype=mx.bfloat16):
    T, kin = x.shape; k = idx.shape[1]; N = U["gate"]["scale"].shape[1]; n_out = U["gate"]["n_out"]; W = int(U["gate"]["packed"].shape[-1])
    ins = [xr, x, idx.reshape(-1).astype(mx.uint32), mx.array([k], dtype=mx.uint32), U["gate"]["packed"], U["gate"]["scale"],
           U["up"]["packed"], U["up"]["scale"]] + ([U["gate"]["ocol"], U["gate"]["wout"], U["up"]["wout"]] if n_out else [])
    kern = _gu_v3(kin, N, U["K"], U["L"], n_out, W, _dt(x), {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[out_dtype])
    return kern(inputs=ins, grid=(64, N // 8, T * k), threadgroup=(64, 1, 1), output_shapes=[(T * k, N)], output_dtypes=[out_dtype])[0]


def gather_down_weighted_v3(hr, h, idx, rw, U):
    T, k = idx.shape; kin = h.shape[1]; N = U["down"]["scale"].shape[1]; n_out = U["down"]["n_out"]; W = int(U["down"]["packed"].shape[-1])
    ins = [hr, h, idx.reshape(-1).astype(mx.uint32), rw.reshape(-1).astype(mx.float32), U["down"]["packed"], U["down"]["scale"]] + \
          ([U["down"]["ocol"], U["down"]["wout"]] if n_out else [])
    kern = _down_v3(kin, N, U["K_dn"], U["L"], n_out, k, W, _dt(h))
    return kern(inputs=ins, grid=(64, N // 8, T), threadgroup=(64, 1, 1), output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- prefill (bring-up)
@functools.lru_cache(maxsize=None)
def _dequant_rot_kernel(kin: int, N: int, K, L: int, W: int):
    """Packed rows of a contiguous expert range -> bf16 (n_e, N, kin) in the ROTATED basis (value * row scale).
    One simdgroup per row and 512-value block; lane = 16 consecutive weights (same window logic as decode)."""
    mu, isd = VL.code_stats(L, K); PL, adv, _ = _lane_geom(K)
    so = VL.step_offsets(K, 9); mk = (1 << L) - 1; MU = VL.MULTS[K]; M2 = MASK | (MASK << 16); O2 = ORB | (ORB << 16)
    ld = ("uint a0 = wr[0], a1 = wr[1];" if K == 2 else
          "uint w0 = wr[0], w1 = wr[1], w2 = wr[2]; uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); "
          "uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));")
    st = []
    for j in range(8):
        o = int(so[j])
        ex = ("a0" if o == 0 else f"(a0 >> {o}u)") if o + L <= 32 else (("a1" if o == 32 else f"(a1 >> {o - 32}u)") if o >= 32 else f"((a0 >> {o}u) | (a1 << {32 - o}u))")
        st.append(f"{{ uint s = ({ex}) & {mk}u; half2 v = as_type<half2>(((s * {MU}u) & {M2}u) | {O2}u);"
                  f" o_[{2*j}] = bfloat16_t((float(v.x) - {float(mu[0])}f) * {float(isd[0])}f * sc);"
                  f" o_[{2*j+1}] = bfloat16_t((float(v.y) - {float(mu[1])}f) * {float(isd[1])}f * sc); }}")
    src = f"""
    uint lane = thread_index_in_simdgroup; uint blk = threadgroup_position_in_grid.x; uint row = threadgroup_position_in_grid.y;
    uint e = threadgroup_position_in_grid.z + e0[0];
    const uint lb = lane * {PL}u; const uint sh = lb & 31u; (void)sh;
    const device uint32_t* wr = w + ((size_t)e * {N}u + row) * {W}u + (size_t)blk * {adv}u + (lb >> 5);
    float sc = scale[(size_t)e * {N}u + row];
    device bfloat16_t* o_ = out + ((size_t)threadgroup_position_in_grid.z * {N}u + row) * {kin}u + blk * 512u + lane * 16u;
    {ld}
    {" ".join(st)}
    """
    return mx.fast.metal_kernel(name=f"jt_deq_k{kin}_n{N}_K{_kname(K)}_w{W}", input_names=["w", "scale", "e0"], output_names=["out"], source=src)


def dequant_rot(P: dict, e0: int, e1: int, L: int = 12) -> mx.array:
    """Experts e0..e1-1 of one JANGT projection -> (e1-e0, N, kin) bf16, rotated basis, row scales applied."""
    E, N, W = P["packed"].shape; kin = P["kin"]
    kern = _dequant_rot_kernel(kin, N, P["K"], L, W)
    return kern(inputs=[P["packed"], P["scale"], mx.array([e0], dtype=mx.uint32)], grid=(32 * (kin // 512), N, e1 - e0), threadgroup=(32, 1, 1),
                output_shapes=[(e1 - e0, N, kin)], output_dtypes=[mx.bfloat16])[0]

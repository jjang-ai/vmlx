"""JANGT v1 with a configurable state length L (default 12): the whole Viterbi cost array of a row lives in threadgroup
memory (2^L floats, 16 KB at L=12), so the encoder never touches device memory inside its inner loop.

Why: at L=16 the 65,536-entry cost array must live in device memory; the encoder then moves ~512 KB per row per step
and is bandwidth-bound (measured 3.5 M weights/s on M5 Max — ~24 h for N0.5's 303 B expert weights). L=12 costs +4.5 %
Gaussian MSE (0.0740 vs 0.0708 at K=2, V=2; still 37 % below scalar Lloyd-Max).

Conventions (K=2, V=2): 4 new bits per step enter at the top: s' = (s >> 4) | (c << (L-4)); predecessors of s' are
((s' & (2^(L-4)-1)) << 4) | h. Row stream: state_t = (S >> 4t) & (2^L - 1); the first L bits are state_0.
Code: same v2 bit-pattern family (h = s * MULT, (h & MASK) | OR -> half2), statistics over the 2^L states."""
from __future__ import annotations
import functools
import os
import numpy as np
import mlx.core as mx
from .v2core import MULT, MASK, ORB

KV = 4
# K bits per weight -> hash multiplier (mask / or shared). K=3: CPU code search 2026-10-08, Gaussian MSE 0.0239 at L=12
# (scalar Lloyd-Max 3-bit 0.0345). Each step shifts in 2K bits (V=2).
MULTS = {2: MULT, 3: 668265263, 2.5: MULT}


def kpat(K) -> tuple[int, int]:
    """New bits entering at transition t (t >= 1): pat[t & 1]. Integer K: (2K, 2K). K = 2.5: transitions alternate
    6 (t even) / 4 (t odd) bits -> 10 bits per 4 weights; a lane's first step is always even (k0/2 + 8*lane)."""
    if float(K).is_integer():
        return (2 * int(K), 2 * int(K))
    assert K == 2.5, f"bitrate {K}"
    return (6, 4)


def step_offsets(K, T: int) -> np.ndarray:
    """Bit offset of state t in the row stream (state 0 at 0)."""
    p0, p1 = kpat(K)
    kb = np.array([0] + [p1 if t & 1 else p0 for t in range(1, T)], dtype=np.int64)
    return np.cumsum(kb)


@functools.lru_cache(maxsize=None)
def code_stats(L: int, K=2):
    s = np.arange(1 << L, dtype=np.uint64)
    h = (s * np.uint64(MULTS[K])) & np.uint64(0xFFFFFFFF)
    p = (h & np.uint64(MASK | (MASK << 16))) | np.uint64(ORB | (ORB << 16))
    lo = (p & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (p >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    raw = np.stack([lo, hi], 1)
    return raw.mean(0).astype(np.float32), (1.0 / raw.std(0)).astype(np.float32)


def words_per_row(kin: int, L: int, K=2) -> int:
    if float(K).is_integer():
        K = int(K); return (K * kin + (L - 2 * K) + 31) // 32 + 1
    bits = int(step_offsets(K, kin // 2)[-1]) + L
    return (bits + 31) // 32 + 1


def pack_states(states: np.ndarray, kin: int, L: int, K=2) -> np.ndarray:
    N, T = states.shape; W = words_per_row(kin, L, K); off = step_offsets(K, T); p0, p1 = kpat(K)
    bits = np.zeros((N, W * 32), dtype=np.uint8)
    s0 = states[:, 0].astype(np.uint32)
    for b in range(L):
        bits[:, b] = (s0 >> b) & 1
    for t in range(1, T):
        KB = p1 if t & 1 else p0
        top = states[:, t].astype(np.uint32) >> (L - KB)
        for b in range(KB):
            bits[:, off[t] + L - KB + b] = (top >> b) & 1
    w = np.zeros((N, W), dtype=np.uint64)
    for b in range(32):
        w |= bits[:, b::32].astype(np.uint64) << np.uint64(b)
    return w.astype(np.uint32)


def unpack_states(packed: np.ndarray, kin: int, L: int, K=2) -> np.ndarray:
    N, W = packed.shape; T = kin // 2; p64 = packed.astype(np.uint64); mask = np.uint64((1 << L) - 1)
    out = np.zeros((N, T), dtype=np.uint32)
    off = step_offsets(K, T)
    for t in range(T):
        bit = int(off[t]); wi, sh = bit // 32, bit % 32
        lo = p64[:, wi] >> np.uint64(sh)
        hi = (p64[:, wi + 1] << np.uint64(32 - sh)) if sh else np.zeros(N, dtype=np.uint64)
        out[:, t] = ((lo | hi) & mask).astype(np.uint32)
    return out.astype(np.uint16)


@functools.lru_cache(maxsize=None)
def _viterbi_tg(T: int, L: int, TPR: int = 256, K=2):
    """One threadgroup of TPR threads per row; costs (2^L floats) in threadgroup memory. KB_t new bits at transition t
    (kpat): 2^KB_t predecessors per state group (16 at K=2, 64 at K=3; alternating 64/16 at K=2.5)."""
    P0, P1 = kpat(K); NSL = 1 << L; GM = NSL >> min(P0, P1); per = NSL // TPR; MU = MULTS[K]
    G0, G1 = NSL >> P0, NSL >> P1
    src = f"""
    uint r = threadgroup_position_in_grid.x; uint tid = thread_position_in_threadgroup.x;
    threadgroup float cA[{NSL}]; threadgroup float minc[{GM}];   // single buffer: minc holds all that phase B reads
    device uchar* bp = back + (size_t)r * {T}u * {GM}u;
    uint st = start[r];
    for (uint k = 0; k < {per}u; k++) {{ uint s = k * {TPR}u + tid; cA[s] = (st == 0xFFFFFFFFu || s == st) ? 0.0f : 3.0e38f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g0 = gain[0];
    threadgroup float* a = cA;
    for (uint t = 0; t < {T}u; t++) {{
      uint KB = (t & 1u) ? {P1}u : {P0}u; uint G = (t & 1u) ? {G1}u : {G0}u;
      for (uint m = tid; m < G; m += {TPR}u) {{
        float bv = a[m << KB]; uint bh = 0u;
        for (uint h = 1u; h < (1u << KB); h++) {{ float c = a[(m << KB) | h]; if (c < bv) {{ bv = c; bh = h; }} }}
        minc[m] = bv; bp[t * {GM}u + m] = uchar(bh);
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float x0 = tgt[(size_t)r * {2 * T}u + 2u * t], x1 = tgt[(size_t)r * {2 * T}u + 2u * t + 1u];
      for (uint k = 0; k < {per}u; k++) {{
        uint s = k * {TPR}u + tid;
        uint p = ((s * {MU}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
        half2 v = as_type<half2>(p);
        float d0 = x0 - g0 * (float(v.x) - mu[0]) * isd[0];
        float d1 = x1 - g0 * (float(v.y) - mu[1]) * isd[1];
        a[s] = minc[s & (G - 1u)] + d0 * d0 + d1 * d1;
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float bv = 3.0e38f; uint bs = 0u;
    for (uint k = 0; k < {per}u; k++) {{ uint s = k * {TPR}u + tid; float c = a[s]; if (c < bv) {{ bv = c; bs = s; }} }}
    threadgroup float rv[{TPR}]; threadgroup uint rs[{TPR}];
    rv[tid] = bv; rs[tid] = bs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = {TPR // 2}u; w > 0u; w >>= 1) {{
      if (tid < w && (rv[tid + w] < rv[tid] || (rv[tid + w] == rv[tid] && rs[tid + w] < rs[tid]))) {{ rv[tid] = rv[tid + w]; rs[tid] = rs[tid + w]; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
      uint s = rs[0]; err[r] = rv[0];
      for (int t = {T - 1}; t >= 0; t--) {{
        states[(size_t)r * {T}u + uint(t)] = ushort(s);
        if (t > 0) {{ uint KB = (uint(t) & 1u) ? {P1}u : {P0}u; uint G = (uint(t) & 1u) ? {G1}u : {G0}u;
          s = ((s & (G - 1u)) << KB) | uint(bp[uint(t) * {GM}u + (s & (G - 1u))]); }}
      }}
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_viterbi_tg_T{T}_L{L}_K{str(K).replace('.', 'p')}", input_names=["tgt", "start", "gain", "mu", "isd"],
                                output_names=["states", "err", "back"], source=src)


def viterbi_rows(tgt: mx.array, start: mx.array, gain: float, L: int = 12, K=2):
    R, n = tgt.shape; T = n // 2; mu, isd = code_stats(L, K); TPR = 256
    states, err, _ = _viterbi_tg(T, L, TPR, K)(
        inputs=[tgt, start, mx.array([gain], dtype=mx.float32), mx.array(mu), mx.array(isd)],
        grid=(R * TPR, 1, 1), threadgroup=(TPR, 1, 1),
        output_shapes=[(R, T), (R,), (R * T * ((1 << L) >> min(kpat(K))),)], output_dtypes=[mx.uint16, mx.float32, mx.uint8])
    return states, err


def values_of(states: mx.array, gain: float, L: int = 12, K=2) -> mx.array:
    mu, isd = code_stats(L, K)
    s = states.astype(mx.uint32); p = ((s * MULTS[K]) & (MASK | (MASK << 16))) | (ORB | (ORB << 16))
    lo = (p & 0xFFFF).astype(mx.uint16).view(mx.float16).astype(mx.float32)
    hi = (p >> 16).astype(mx.uint16).view(mx.float16).astype(mx.float32)
    v = mx.stack([(lo - float(mu[0])) * float(isd[0]), (hi - float(mu[1])) * float(isd[1])], axis=-1)
    return gain * v.reshape(states.shape[0], -1)


# ------------------------------------------------------------------------------------------- fused decode GEMV
@functools.lru_cache(maxsize=None)
def _gemv_fused(kin: int, L: int, n_out: int, xdt: str):
    """One kernel: x * su_m (signs; 0 on outlier columns) -> blockwise H128 in registers (4 in-lane butterfly stages +
    3 cross-lane simd_shuffle_xor stages, lanes 8j..8j+7 form one 128-block) -> trellis decode (one state + one
    multiply per two weights, bit-pattern values) -> fp32 accumulate -> offsets via sums of x -> row scale ->
    + outlier columns (<= 8 dense fp16 values per row) in the epilogue. 2 simdgroups x 4 rows per threadgroup."""
    W = words_per_row(kin, L); mk = (1 << L) - 1
    outl = "" if n_out == 0 else f"""
        float o = 0.0f;
        for (uint j = 0; j < {n_out}u; j++) o = fma(float(xp[ocol[j]]), float(wout[(size_t)(row0 + r) * {n_out}u + j]), o);
        y += o;"""
    src = f"""
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
    uint m = threadgroup_position_in_grid.z;
    const device {xdt}* xp = x + (size_t)m * {kin}u;
    float acc0[4] = {{0.0f,0.0f,0.0f,0.0f}}, acc1[4] = {{0.0f,0.0f,0.0f,0.0f}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint gl = k0 / 16u + lane;
      float xt[16];
      for (uint i = 0; i < 16u; i++) xt[i] = float(xp[k0 + lane * 16u + i]) * su[k0 + lane * 16u + i];
      for (uint hb = 1u; hb < 16u; hb <<= 1) {{
        for (uint i = 0; i < 16u; i++) {{
          if ((i & hb) == 0u) {{ float a0 = xt[i], b0 = xt[i + hb]; xt[i] = a0 + b0; xt[i + hb] = a0 - b0; }}
        }}
      }}
      for (uint d = 1u; d < 8u; d <<= 1) {{
        bool upper = (lane & d) != 0u;
        for (uint i = 0; i < 16u; i++) {{ float other = simd_shuffle_xor(xt[i], d); xt[i] = upper ? (other - xt[i]) : (xt[i] + other); }}
      }}
      for (uint i = 0; i < 16u; i++) xt[i] *= 0.08838834764831845f;     // 1/sqrt(128)
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < 4u; r++) {{
        const device uint* wr = w + (size_t)(row0 + r) * {W}u;
        uint w0 = wr[gl], w1 = wr[gl + 1u];
        float a0 = 0.0f, a1 = 0.0f;
        for (uint j = 0; j < 8u; j++) {{
          uint sh = 4u * j;
          uint s = (sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)))) & {mk}u;
          uint p = ((s * {MULT}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
          half2 v = as_type<half2>(p);
          a0 = fma(xt[2u*j], float(v.x), a0); a1 = fma(xt[2u*j+1u], float(v.y), a1);
        }}
        acc0[r] += a0; acc1[r] += a1;
      }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < 4u; r++) {{
      float A0 = simd_sum(acc0[r]), A1 = simd_sum(acc1[r]);
      if (lane == 0u && row0 + r < N) {{
        float y = ((A0 - mu[0] * SE) * isd[0] + (A1 - mu[1] * SO) * isd[1]) * scale[row0 + r];{outl}
        out[(size_t)m * N + row0 + r] = y;
      }}
    }}
    """
    ins = ["x", "su", "w", "scale", "mu", "isd", "meta"] + (["ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jangt_gemvf_k{kin}_L{L}_o{n_out}_{xdt}", input_names=ins, output_names=["out"], source=src)


def gemv_fused(x: mx.array, su_m: mx.array, packed: mx.array, scale: mx.array, L: int, ocol=None, wout=None) -> mx.array:
    """x (M, kin) RAW activations (bf16/f16/f32); su_m (kin,) f32 signs with 0 on outlier columns; returns (M, N) f32."""
    M, kin = x.shape; N = packed.shape[0]; mu, isd = code_stats(L)
    n_out = 0 if ocol is None else int(ocol.size)
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[x.dtype]
    ins = [x, su_m, packed, scale, mx.array(mu), mx.array(isd), mx.array([N], dtype=mx.uint32)] + ([ocol, wout] if n_out else [])
    return _gemv_fused(kin, L, n_out, xdt)(inputs=ins, grid=(64, N // 8, M), threadgroup=(64, 1, 1),
                                           output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- rotate-once structure
@functools.lru_cache(maxsize=None)
def _consts(L: int, N: int):
    """Per-(L, N) constant inputs, built once: rebuilding three tiny mx.arrays per call cost ~1.7 us of host time per
    kernel (measured), i.e. ~3 % of Qwen3-1.7B decode."""
    mu, isd = code_stats(L); c = (mx.array(mu), mx.array(isd), mx.array([N], dtype=mx.uint32)); mx.eval(*c); return c
@functools.lru_cache(maxsize=None)
def _gemv_rot(kin: int, L: int, n_out: int, xdt: str, R: int = 4):
    """GEMV on an ALREADY rotated input xr (f32), plus the outlier columns read from the raw input in the epilogue.
    (Rotating inside every threadgroup — _gemv_fused — repeats the H128 work N/8 times per call: measured 105 tok/s vs
    218 unfused on Qwen3-1.7B. Rotate once per input instead, shared by gate and up.)"""
    W = words_per_row(kin, L); mk = (1 << L) - 1
    outl = "" if n_out == 0 else f"""
        float o = 0.0f;
        for (uint j = 0; j < {n_out}u; j++) o = fma(float(xraw[(size_t)m * {kin}u + ocol[j]]), float(wout[(size_t)(row0 + r) * {n_out}u + j]), o);
        y += o;"""
    src = f"""
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * {2*R}u + sgi * {R}u;
    uint m = threadgroup_position_in_grid.z;
    const device float4* xp = (const device float4*)(xr + (size_t)m * {kin}u);
    float acc0[{R}], acc1[{R}];
    for (uint r = 0; r < {R}u; r++) {{ acc0[r] = 0.0f; acc1[r] = 0.0f; }}
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint gl = k0 / 16u + lane;
      float xt[16];
      for (uint i = 0; i < 4u; i++) {{ float4 q = xp[(k0 + lane * 16u) / 4u + i]; xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < {R}u; r++) {{
        const device uint* wr = w + (size_t)(row0 + r) * {W}u;
        uint w0 = wr[gl], w1 = wr[gl + 1u];
        float a0 = 0.0f, a1 = 0.0f;
        for (uint j = 0; j < 8u; j++) {{
          uint sh = 4u * j;
          uint s = (sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)))) & {mk}u;
          uint p = ((s * {MULT}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
          half2 v = as_type<half2>(p);
          a0 = fma(xt[2u*j], float(v.x), a0); a1 = fma(xt[2u*j+1u], float(v.y), a1);
        }}
        acc0[r] += a0; acc1[r] += a1;
      }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < {R}u; r++) {{
      float A0 = simd_sum(acc0[r]), A1 = simd_sum(acc1[r]);
      if (lane == 0u && row0 + r < N) {{
        float y = ((A0 - mu[0] * SE) * isd[0] + (A1 - mu[1] * SO) * isd[1]) * scale[row0 + r];{outl}
        out[(size_t)m * N + row0 + r] = y;
      }}
    }}
    """
    ins = ["xr", "w", "scale", "mu", "isd", "meta"] + (["xraw", "ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jangt_gemvr_k{kin}_L{L}_o{n_out}_{xdt}_r{R}", input_names=ins, output_names=["out"], source=src)


def rotate(x: mx.array, su_m: mx.array, hb: int = 128) -> mx.array:
    """x (M, K) raw -> f32 (x * su_m) with blockwise Hadamard (MLX's fast hadamard_transform)."""
    M, K = x.shape
    import math
    return mx.hadamard_transform((x.astype(mx.float32) * su_m).reshape(M, K // hb, hb), scale=1.0 / math.sqrt(hb)).reshape(M, K)


def gemv_rot(xr: mx.array, xraw: mx.array, packed: mx.array, scale: mx.array, L: int, ocol=None, wout=None, R: int = 0) -> mx.array:
    M, kin = xr.shape; N = packed.shape[0]; mu, isd = code_stats(L); n_out = 0 if ocol is None else int(ocol.size)
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[xraw.dtype]
    R = R or int(os.environ.get("JANGT_R", "4"))
    ins = [xr, packed, scale, *_consts(L, N)] + ([xraw, ocol, wout] if n_out else [])
    return _gemv_rot(kin, L, n_out, xdt, R)(inputs=ins, grid=(64, N // (2 * R), M), threadgroup=(64, 1, 1),
                                         output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- production decode set
@functools.lru_cache(maxsize=None)
def _rot_kernel(K: int, xdt: str):
    """One launch: out = H128_blockwise(x * su) in f32. One simdgroup per 128-block: lane holds 4 consecutive values
    (2 in-lane butterfly stages) + 5 cross-lane simd_shuffle_xor stages (1..16)."""
    src = f"""
    uint lane = thread_index_in_simdgroup; uint blk = threadgroup_position_in_grid.x; uint m = threadgroup_position_in_grid.y;
    size_t base = (size_t)m * {K}u + blk * 128u + lane * 4u;
    float v[4];
    for (uint i = 0; i < 4u; i++) v[i] = float(x[base + i]) * su[blk * 128u + lane * 4u + i];
    {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }}
    {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }}
    for (uint d = 1u; d < 32u; d <<= 1) {{
      bool upper = (lane & d) != 0u;
      for (uint i = 0; i < 4u; i++) {{ float o = simd_shuffle_xor(v[i], d); v[i] = upper ? (o - v[i]) : (v[i] + o); }}
    }}
    for (uint i = 0; i < 4u; i++) out[base + i] = v[i] * 0.08838834764831845f;
    """
    return mx.fast.metal_kernel(name=f"jangt_rot128_k{K}_{xdt}", input_names=["x", "su"], output_names=["out"], source=src)


def rotate1(x: mx.array, su_m: mx.array) -> mx.array:
    M, K = x.shape
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[x.dtype]
    return _rot_kernel(K, xdt)(inputs=[x, su_m], grid=(32 * (K // 128), M, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(M, K)], output_dtypes=[mx.float32])[0]


@functools.lru_cache(maxsize=None)
def _gu_kernel(kin: int, L: int, n_out: int, xdt: str, odt: str):
    """Fused gate+up decode with SwiGLU epilogue: one pass over the shared rotated input xr decodes BOTH matrices'
    rows; out = silu(g) * u. Outlier columns (shared by gate and up: same input Hessian) read from the raw input."""
    W = words_per_row(kin, L); mk = (1 << L) - 1
    def dot(wn, acc0, acc1):
        return f"""
        {{ const device uint* wr = {wn} + (size_t)(row0 + r) * {W}u; uint w0 = wr[gl], w1 = wr[gl + 1u];
          float a0 = 0.0f, a1 = 0.0f;
          for (uint j = 0; j < 8u; j++) {{
            uint sh = 4u * j; uint s = (sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)))) & {mk}u;
            uint p = ((s * {MULT}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u; half2 v = as_type<half2>(p);
            a0 = fma(xt[2u*j], float(v.x), a0); a1 = fma(xt[2u*j+1u], float(v.y), a1); }}
          {acc0}[r] += a0; {acc1}[r] += a1; }}"""
    outl = "" if n_out == 0 else f"""
        for (uint j = 0; j < {n_out}u; j++) {{ float xo = float(xraw[(size_t)m * {kin}u + ocol[j]]);
          g = fma(xo, float(woutg[(size_t)(row0 + r) * {n_out}u + j]), g); u = fma(xo, float(woutu[(size_t)(row0 + r) * {n_out}u + j]), u); }}"""
    src = f"""
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u; uint m = threadgroup_position_in_grid.z;
    const device float* xp = xr + (size_t)m * {kin}u;
    float g0[4] = {{0.0f,0.0f,0.0f,0.0f}}, g1[4] = {{0.0f,0.0f,0.0f,0.0f}}, u0[4] = {{0.0f,0.0f,0.0f,0.0f}}, u1[4] = {{0.0f,0.0f,0.0f,0.0f}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint gl = k0 / 16u + lane; float xt[16];
      for (uint i = 0; i < 16u; i++) xt[i] = xp[k0 + lane * 16u + i];
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < 4u; r++) {{ {dot("wg", "g0", "g1")} {dot("wu", "u0", "u1")} }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < 4u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u && row0 + r < N) {{
        float g = ((G0 - mu[0] * SE) * isd[0] + (G1 - mu[1] * SO) * isd[1]) * sg[row0 + r];
        float u = ((U0 - mu[0] * SE) * isd[0] + (U1 - mu[1] * SO) * isd[1]) * su_[row0 + r];{outl}
        out[(size_t)m * N + row0 + r] = {odt}((g / (1.0f + metal::fast::exp(-g))) * u);
      }}
    }}
    """
    ins = ["xr", "wg", "sg", "wu", "su_", "mu", "isd", "meta"] + (["xraw", "ocol", "woutg", "woutu"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jangt_gu_k{kin}_L{L}_o{n_out}_{xdt}_{odt}", input_names=ins, output_names=["out"], source=src)


def gate_up_swiglu(xr, xraw, pg, sg, pu, su_, L, ocol=None, woutg=None, woutu=None, out_dtype=mx.bfloat16):
    M, kin = xr.shape; N = pg.shape[0]; mu, isd = code_stats(L); n_out = 0 if ocol is None else int(ocol.size)
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[xraw.dtype]
    odt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[out_dtype]
    ins = [xr, pg, sg, pu, su_, *_consts(L, N)] + ([xraw, ocol, woutg, woutu] if n_out else [])
    return _gu_kernel(kin, L, n_out, xdt, odt)(inputs=ins, grid=(64, N // 8, M), threadgroup=(64, 1, 1),
                                               output_shapes=[(M, N)], output_dtypes=[out_dtype])[0]


# ------------------------------------------------------------------------------------------- 2-launch MLP decode
_H128_TG = """
    {{ // H128 on threadgroup buffer {buf} block {blk}: lane holds 4 consecutive values, 2 in-lane + 5 shuffle stages
      uint b0 = {blk} * 128u + lane * 4u; float v[4];
      for (uint i = 0; i < 4u; i++) v[i] = {buf}[b0 + i];
      {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }}
      {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }}
      for (uint d = 1u; d < 32u; d <<= 1) {{
        bool upper = (lane & d) != 0u;
        for (uint i = 0; i < 4u; i++) {{ float o = simd_shuffle_xor(v[i], d); v[i] = upper ? (o - v[i]) : (v[i] + o); }}
      }}
      for (uint i = 0; i < 4u; i++) {buf}[b0 + i] = v[i] * 0.08838834764831845f;
    }}"""


@functools.lru_cache(maxsize=None)
def _gu2_kernel(kin: int, L: int, n_out: int, xdt: str, odt: str):
    """Gate+up+SwiGLU with BOTH rotations fused. One threadgroup = 32 simdgroups x 4 rows = 128 output rows:
    (1) x*su -> threadgroup memory, H128 there (once per threadgroup: N/128 copies, not N/8);
    (2) trellis dot products for gate and up reading the rotated x from threadgroup memory;
    (3) h = silu(g)*u for the 128 rows -> hraw (for down's outlier columns) and, since the 128 rows are exactly one
        H128 block of down's input, hr = H128(h*su_d) straight from threadgroup memory.
    MLP decode = this kernel + the down GEMV: 2 launches."""
    W = words_per_row(kin, L); mk = (1 << L) - 1; nb = kin // 128
    def dot(wn, acc0, acc1):
        return f"""
        {{ const device uint* wr = {wn} + (size_t)(row0 + r) * {W}u; uint w0 = wr[gl], w1 = wr[gl + 1u];
          float a0 = 0.0f, a1 = 0.0f;
          for (uint j = 0; j < 8u; j++) {{
            uint sh = 4u * j; uint s = (sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)))) & {mk}u;
            uint p = ((s * {MULT}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u; half2 v = as_type<half2>(p);
            a0 = fma(xt[2u*j], float(v.x), a0); a1 = fma(xt[2u*j+1u], float(v.y), a1); }}
          {acc0}[r] += a0; {acc1}[r] += a1; }}"""
    outl = "" if n_out == 0 else f"""
          for (uint j = 0; j < {n_out}u; j++) {{ float xo = float(x[(size_t)m * {kin}u + ocol[j]]);
            g = fma(xo, float(woutg[(size_t)(row0 + r) * {n_out}u + j]), g); u = fma(xo, float(woutu[(size_t)(row0 + r) * {n_out}u + j]), u); }}"""
    src = f"""
    threadgroup float xs[{kin}];
    threadgroup float hs[128];
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, tid = thread_position_in_threadgroup.x;
    uint tg = threadgroup_position_in_grid.y; uint m = threadgroup_position_in_grid.z;
    uint row0 = tg * 128u + sgi * 4u;
    for (uint i = tid; i < {kin}u; i += 1024u) xs[i] = float(x[(size_t)m * {kin}u + i]) * sux[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bb = sgi; bb < {nb}u; bb += 32u) {_H128_TG.format(buf="xs", blk="bb")}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g0[4] = {{0.0f,0.0f,0.0f,0.0f}}, g1[4] = {{0.0f,0.0f,0.0f,0.0f}}, u0[4] = {{0.0f,0.0f,0.0f,0.0f}}, u1[4] = {{0.0f,0.0f,0.0f,0.0f}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint gl = k0 / 16u + lane; float xt[16];
      for (uint i = 0; i < 16u; i++) xt[i] = xs[k0 + lane * 16u + i];
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < 4u; r++) {{ {dot("wg", "g0", "g1")} {dot("wu", "u0", "u1")} }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < 4u; r++) {{
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {{
        float g = ((G0 - mu[0] * SE) * isd[0] + (G1 - mu[1] * SO) * isd[1]) * sg[row0 + r];
        float u = ((U0 - mu[0] * SE) * isd[0] + (U1 - mu[1] * SO) * isd[1]) * su_[row0 + r];{outl}
        {odt} h = {odt}((g / (1.0f + metal::fast::exp(-g))) * u);
        hraw[(size_t)m * N + row0 + r] = h;
        hs[sgi * 4u + r] = float(h) * sud[row0 + r];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0u) {{
      {_H128_TG.format(buf="hs", blk="0u")}
      for (uint i = 0; i < 4u; i++) hr[(size_t)m * N + tg * 128u + lane * 4u + i] = hs[lane * 4u + i];
    }}
    """
    ins = ["x", "sux", "wg", "sg", "wu", "su_", "sud", "mu", "isd", "meta"] + (["ocol", "woutg", "woutu"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jangt_gu2_k{kin}_L{L}_o{n_out}_{xdt}_{odt}", input_names=ins,
                                output_names=["hraw", "hr"], source=src)


def gate_up_swiglu_rot(x, sux, pg, sg, pu, su_, sud, L, ocol=None, woutg=None, woutu=None, out_dtype=mx.bfloat16):
    """x RAW (M, kin); sux/sud input signs (0 on outlier cols) of gate/up and of down. Returns (hraw, hr):
    hraw (M, N) out_dtype = SwiGLU output, hr (M, N) f32 = H128(hraw*sud), ready for gemv_rot of down."""
    M, kin = x.shape; N = pg.shape[0]; n_out = 0 if ocol is None else int(ocol.size)
    assert N % 128 == 0 and kin % 512 == 0
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[x.dtype]
    odt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[out_dtype]
    ins = [x, sux, pg, sg, pu, su_, sud, *_consts(L, N)] + ([ocol, woutg, woutu] if n_out else [])
    return _gu2_kernel(kin, L, n_out, xdt, odt)(inputs=ins, grid=(1024, N // 128, M), threadgroup=(1024, 1, 1),
                                                output_shapes=[(M, N), (M, N)], output_dtypes=[out_dtype, mx.float32])


# ------------------------------------------------------------------------------------------- K>2 decode GEMV
@functools.lru_cache(maxsize=None)
def _gemv_rot_k(kin: int, L: int, n_out: int, xdt: str, K, R: int = 4):
    """gemv_rot for K bits/weight (2K-bit steps). Each lane owns 16 consecutive weights = 8 steps; its bits start at
    (k0/2 + 8*lane) * 2K and span 8*2K + L bits (60 at K=3) -> a 96-bit window of 3 words, steps extracted with
    64-bit shifts. K=2 keeps the 1-word fast path of _gemv_rot."""
    W = words_per_row(kin, L, K); mk = (1 << L) - 1; MU = MULTS[K]
    so = step_offsets(K, 9); offs = [int(so[j]) for j in range(8)]          # lane's first step is even
    per16 = int(step_offsets(K, 17)[16])                                      # bits per 16 steps (2 lanes) / 2 -> per lane
    assert per16 % 2 == 0; PL = per16 // 2
    outl = "" if n_out == 0 else f"""
        float o = 0.0f;
        for (uint j = 0; j < {n_out}u; j++) o = fma(float(xraw[(size_t)m * {kin}u + ocol[j]]), float(wout[(size_t)(row0 + r) * {n_out}u + j]), o);
        y += o;"""
    src = f"""
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * {2*R}u + sgi * {R}u;
    uint m = threadgroup_position_in_grid.z;
    const device float4* xp = (const device float4*)(xr + (size_t)m * {kin}u);
    float acc0[{R}], acc1[{R}];
    for (uint r = 0; r < {R}u; r++) {{ acc0[r] = 0.0f; acc1[r] = 0.0f; }}
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint bit0 = (k0 / 2u + lane * 8u) / 8u * {PL}u; uint wi = bit0 >> 5, sh = bit0 & 31u;
      const uint OFF[8] = {{{", ".join(f"{o}u" for o in offs)}}};
      float xt[16];
      for (uint i = 0; i < 4u; i++) {{ float4 q = xp[(k0 + lane * 16u) / 4u + i]; xt[4u*i] = q.x; xt[4u*i+1u] = q.y; xt[4u*i+2u] = q.z; xt[4u*i+3u] = q.w; }}
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < {R}u; r++) {{
        const device uint* wr = w + (size_t)(row0 + r) * {W}u;
        ulong lo = ulong(wr[wi]) | (ulong(wr[wi + 1u]) << 32); ulong hi = ulong(wr[wi + 2u]);
        float a0 = 0.0f, a1 = 0.0f;
        for (uint j = 0; j < 8u; j++) {{
          uint b = sh + OFF[j];
          ulong v64 = (lo >> b) | (b == 0u ? 0ul : (hi << (64u - b)));
          uint s = uint(v64) & {mk}u;
          uint p = ((s * {MU}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
          half2 v = as_type<half2>(p);
          a0 = fma(xt[2u*j], float(v.x), a0); a1 = fma(xt[2u*j+1u], float(v.y), a1);
        }}
        acc0[r] += a0; acc1[r] += a1;
      }}
    }}
    float SE = simd_sum(se), SO = simd_sum(so);
    for (uint r = 0; r < {R}u; r++) {{
      float A0 = simd_sum(acc0[r]), A1 = simd_sum(acc1[r]);
      if (lane == 0u && row0 + r < N) {{
        float y = ((A0 - mu[0] * SE) * isd[0] + (A1 - mu[1] * SO) * isd[1]) * scale[row0 + r];{outl}
        out[(size_t)m * N + row0 + r] = y;
      }}
    }}
    """
    ins = ["xr", "w", "scale", "mu", "isd", "meta"] + (["xraw", "ocol", "wout"] if n_out else [])
    return mx.fast.metal_kernel(name=f"jangt_gemvrk_k{kin}_L{L}_K{str(K).replace('.', 'p')}_o{n_out}_{xdt}_r{R}", input_names=ins, output_names=["out"], source=src)


@functools.lru_cache(maxsize=None)
def _consts_k(L: int, N: int, K):
    mu, isd = code_stats(L, K); c = (mx.array(mu), mx.array(isd), mx.array([N], dtype=mx.uint32)); mx.eval(*c); return c


def gemv_rot_k(xr: mx.array, xraw: mx.array, packed: mx.array, scale: mx.array, L: int, K, ocol=None, wout=None, R: int = 4) -> mx.array:
    if K == 2:
        return gemv_rot(xr, xraw, packed, scale, L, ocol, wout, R)
    M, kin = xr.shape; N = packed.shape[0]; n_out = 0 if ocol is None else int(ocol.size)
    xdt = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[xraw.dtype]
    ins = [xr, packed, scale, *_consts_k(L, N, K)] + ([xraw, ocol, wout] if n_out else [])
    return _gemv_rot_k(kin, L, n_out, xdt, K, R)(inputs=ins, grid=(64, N // (2 * R), M), threadgroup=(64, 1, 1),
                                                 output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]

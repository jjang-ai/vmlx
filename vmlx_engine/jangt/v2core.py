"""JANGT v1 core for the small-model test (K=2, V=2, L=16): code, bit layout, exact Metal encoder + decoder.

Trellis convention (state shifts RIGHT, new bits enter at the top):
  state_{t+1} = (state_t >> 4) | (c_{t+1} << 12)          (K=2, V=2 -> 4 new bits per step)
  stream bits S (LSB-first), state_t = (S >> 4t) & 0xFFFF  -> the first 16 bits of a row are state_0 (free start)
  predecessors of s' : ((s' & 0xFFF) << 4) | h, h in [0, 16)   (contiguous groups of 16)
Code (our own, from jangt.codesearch_v2): h = s * MULT (mod 2^32); p = (h & MASK) | OR; the two fp16 halves of p are
the raw values of the two weights of the step; value = (raw - mu_i) * inv_sd_i (per half, from the 65536-state table).
Row layout: `words_per_row = (2*Kin + 12 + 31)//32 + 1` uint32 (one spare word so the last lane's window is in bounds).
"""
from __future__ import annotations
import functools
import numpy as np
import mlx.core as mx

MULT, MASK, ORB = 636231159, 36863, 12288          # best v2_halfbits from codesearch_v2 (K2 MSE 0.0708)
NS, KV = 1 << 16, 4


def code_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(raw (65536, 2) float32, mu (2,), inv_sd (2,)) for the code."""
    s = np.arange(NS, dtype=np.uint64)
    h = (s * np.uint64(MULT)) & np.uint64(0xFFFFFFFF)
    m32 = np.uint64(MASK | (MASK << 16)); o32 = np.uint64(ORB | (ORB << 16))
    p = (h & m32) | o32
    lo = (p & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (p >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    raw = np.stack([lo, hi], 1)
    mu = raw.mean(0); inv_sd = 1.0 / raw.std(0)
    return raw, mu.astype(np.float32), inv_sd.astype(np.float32)


RAW, MU, INV_SD = code_table()
VALUES = (RAW - MU) * INV_SD                         # normalized code values per state (65536, 2)


def words_per_row(kin: int) -> int:
    return (2 * kin + 12 + 31) // 32 + 1


def pack_states(states: np.ndarray, kin: int) -> np.ndarray:
    """states (N, T) uint16 with T = kin/2 -> packed (N, words_per_row) uint32 stream."""
    N, T = states.shape
    nbits = 16 + 4 * (T - 1); W = words_per_row(kin)
    bits = np.zeros((N, W * 32), dtype=np.uint8)
    s0 = states[:, 0].astype(np.uint32)
    for b in range(16):
        bits[:, b] = (s0 >> b) & 1
    top = (states[:, 1:].astype(np.uint32) >> 12)    # the 4 new bits of each later step
    for t in range(1, T):
        for b in range(4):
            bits[:, 16 + 4 * (t - 1) + b] = (top[:, t - 1] >> b) & 1
    w = np.zeros((N, W), dtype=np.uint64)
    for b in range(32):
        w |= bits[:, b::32].astype(np.uint64) << np.uint64(b)
    assert nbits <= W * 32
    return w.astype(np.uint32)


def unpack_states(packed: np.ndarray, kin: int) -> np.ndarray:
    """Oracle decoder: packed (N, W) -> states (N, kin/2)."""
    N, W = packed.shape; T = kin // 2
    stream = np.zeros(N, dtype=object)
    out = np.zeros((N, T), dtype=np.uint32)
    p64 = packed.astype(np.uint64)
    for t in range(T):
        bit = 4 * t; wi, sh = bit // 32, bit % 32
        lo = p64[:, wi] >> np.uint64(sh)
        hi = (p64[:, wi + 1] << np.uint64(32 - sh)) if sh else np.zeros(N, dtype=np.uint64)
        out[:, t] = ((lo | hi) & np.uint64(0xFFFF)).astype(np.uint32)
    return out.astype(np.uint16)


# ------------------------------------------------------------------------------------------- Metal encoder
@functools.lru_cache(maxsize=None)
def _viterbi_kernel(T: int):
    """One threadgroup (1024 threads) per row. Device scratch: two cost buffers of 65536 floats per row and per-step
    back-pointers for the 4096 predecessor groups. Phase A: min over each 16-wide predecessor group -> threadgroup
    minc/arg; Phase B: new cost of every state = minc[s & 0xFFF] + local error of its two values."""
    src = f"""
    uint r = threadgroup_position_in_grid.x; uint tid = thread_position_in_threadgroup.x;
    device float* cA = scratch + (size_t)r * 131072u; device float* cB = cA + 65536u;
    device uchar* bp = back + (size_t)r * {T}u * 4096u;
    threadgroup float minc[4096]; threadgroup uchar arg[4096];
    uint st = start[r];
    for (uint i = 0; i < 64u; i++) {{ uint s = tid * 64u + i; cA[s] = (st == 0xFFFFFFFFu || s == st) ? 0.0f : 3.0e38f; }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float g0 = gain[0];
    for (uint t = 0; t < {T}u; t++) {{
      // phase A: predecessor groups m = tid*4 .. tid*4+3
      for (uint q = 0; q < 4u; q++) {{
        uint m = tid * 4u + q; float b = 3.0e38f; uint ba = 0u;
        for (uint h = 0; h < 16u; h++) {{ float c = cA[(m << 4) | h]; if (c < b) {{ b = c; ba = h; }} }}
        minc[m] = b; arg[m] = uchar(ba); bp[t * 4096u + m] = uchar(ba);
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float x0 = tgt[(size_t)r * {2 * T}u + 2u * t], x1 = tgt[(size_t)r * {2 * T}u + 2u * t + 1u];
      for (uint i = 0; i < 64u; i++) {{
        uint s = tid * 64u + i;
        uint h32 = s * {MULT}u; uint p = (h32 & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
        half2 v = as_type<half2>(p);
        float d0 = x0 - g0 * (float(v.x) - mu[0]) * isd[0];
        float d1 = x1 - g0 * (float(v.y) - mu[1]) * isd[1];
        cB[s] = minc[s & 0xFFFu] + d0 * d0 + d1 * d1;
      }}
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      device float* tmp = cA; cA = cB; cB = tmp;
    }}
    // argmin over the final costs (each thread its 64, then threadgroup reduce)
    float bv = 3.0e38f; uint bs = 0u;
    for (uint i = 0; i < 64u; i++) {{ uint s = tid * 64u + i; float c = cA[s]; if (c < bv) {{ bv = c; bs = s; }} }}
    threadgroup float rv[1024]; threadgroup uint rs[1024];
    rv[tid] = bv; rs[tid] = bs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 512u; w > 0u; w >>= 1) {{
      if (tid < w && rv[tid + w] < rv[tid]) {{ rv[tid] = rv[tid + w]; rs[tid] = rs[tid + w]; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
      uint s = rs[0]; err[r] = rv[0];
      for (int t = {T - 1}; t >= 0; t--) {{
        states[(size_t)r * {T}u + uint(t)] = ushort(s);
        if (t > 0) s = ((s & 0xFFFu) << 4) | uint(bp[uint(t) * 4096u + (s & 0xFFFu)]);
      }}
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_viterbi_v2k2_T{T}", input_names=["tgt", "start", "gain", "mu", "isd"],
                                output_names=["states", "err", "scratch", "back"], source=src)


def viterbi_rows(tgt: mx.array, start: mx.array, gain: float, scratch=None, back=None):
    """tgt (R, 2T) float32 normalized targets; start (R,) uint32 (0xFFFFFFFF = free). Returns states (R, T) uint16.
    Scratch (2x65536 floats per row) and back-pointers are kernel outputs (MLX inputs are read-only); the allocator
    reuses them between calls. `scratch`/`back` args are accepted for API compatibility and ignored."""
    R, n = tgt.shape; T = n // 2
    k = _viterbi_kernel(T)
    states, err, _, _ = k(inputs=[tgt, start, mx.array([gain], dtype=mx.float32), mx.array(MU), mx.array(INV_SD)],
                          grid=(R * 1024, 1, 1), threadgroup=(1024, 1, 1),
                          output_shapes=[(R, T), (R,), (R * 131072,), (R * T * 4096,)],
                          output_dtypes=[mx.uint16, mx.float32, mx.float32, mx.uint8])
    return states, err


def values_of(states: mx.array, gain: float) -> mx.array:
    """states (R, T) -> reconstructed normalized values (R, 2T) float32 (mlx, exact same math as the kernel)."""
    s = states.astype(mx.uint32)
    h = s * MULT
    p = (h & (MASK | (MASK << 16))) | (ORB | (ORB << 16))
    lo = (p & 0xFFFF).astype(mx.uint16).view(mx.float16).astype(mx.float32)
    hi = (p >> 16).astype(mx.uint16).view(mx.float16).astype(mx.float32)
    v = mx.stack([(lo - float(MU[0])) * float(INV_SD[0]), (hi - float(MU[1])) * float(INV_SD[1])], axis=-1)
    return gain * v.reshape(states.shape[0], -1)


# ------------------------------------------------------------------------------------------- Metal decoder (GEMV)
@functools.lru_cache(maxsize=None)
def _gemv_kernel(kin: int):
    """Exact decode GEMV: y[m, n] = sum_k x[m, k] * W_hat[n, k] with W_hat = gain * (raw - mu) * isd per half.
    Lane = 16 weights = 8 steps; its states live in bits [32*gl, 32*gl + 44) of the row stream (gl = global lane index
    = block*32 + lane), i.e. words gl and gl+1. Offsets leave the inner loop: sum x*raw - mu0*sum(x_even) - mu1*sum(x_odd).
    2 simdgroups x 4 rows per threadgroup (JANGH skeleton)."""
    W = words_per_row(kin)
    src = f"""
    uint N = meta[0];
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
    uint m = threadgroup_position_in_grid.z;
    const device float* xp = x + (size_t)m * {kin}u;
    float acc0[4] = {{0.0f,0.0f,0.0f,0.0f}}, acc1[4] = {{0.0f,0.0f,0.0f,0.0f}};
    float se = 0.0f, so = 0.0f;
    for (uint k0 = 0; k0 < {kin}u; k0 += 512u) {{
      uint gl = k0 / 16u + lane;
      float xt[16];
      for (uint i = 0; i < 16u; i++) xt[i] = xp[k0 + lane * 16u + i];
      for (uint i = 0; i < 8u; i++) {{ se += xt[2u*i]; so += xt[2u*i+1u]; }}
      for (uint r = 0; r < 4u; r++) {{
        const device uint* wr = w + (size_t)(row0 + r) * {W}u;
        uint w0 = wr[gl], w1 = wr[gl + 1u];
        float a0 = 0.0f, a1 = 0.0f;
        for (uint j = 0; j < 8u; j++) {{
          uint sh = 4u * j;
          uint s = (sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)))) & 0xFFFFu;
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
        float y = ((A0 - mu[0] * SE) * isd[0] + (A1 - mu[1] * SO) * isd[1]) * scale[row0 + r];
        out[(size_t)m * N + row0 + r] = y;
      }}
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_v2k2_gemv_k{kin}", input_names=["x", "w", "scale", "mu", "isd", "meta"],
                                output_names=["out"], source=src)


def gemv(x: mx.array, packed: mx.array, scale: mx.array) -> mx.array:
    """x (M, kin) float32 (already rotated); packed (N, W) uint32; scale (N,) float32 = row_scale * gain."""
    M, kin = x.shape; N = packed.shape[0]
    assert kin % 512 == 0 and N % 8 == 0
    return _gemv_kernel(kin)(inputs=[x, packed, scale, mx.array(MU), mx.array(INV_SD), mx.array([N], dtype=mx.uint32)],
                             grid=(64, N // 8, M), threadgroup=(64, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------- coalesced encoder
@functools.lru_cache(maxsize=None)
def _viterbi_kernel_c(T: int):
    """Same trellis as _viterbi_kernel, memory-coalesced: thread tid handles states g = k*1024 + tid (k = 0..63), so a
    simdgroup touches 32 consecutive floats; each 16-lane half owns one predecessor group (16 consecutive states) and
    reduces min/argmin with simd shuffles (xor 1,2,4,8)."""
    src = f"""
    uint r = threadgroup_position_in_grid.x; uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    device float* cA = scratch + (size_t)r * 131072u; device float* cB = cA + 65536u;
    device uchar* bp = back + (size_t)r * {T}u * 4096u;
    threadgroup float minc[4096];
    uint st = start[r];
    for (uint k = 0; k < 64u; k++) {{ uint s = k * 1024u + tid; cA[s] = (st == 0xFFFFFFFFu || s == st) ? 0.0f : 3.0e38f; }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float g0 = gain[0];
    for (uint t = 0; t < {T}u; t++) {{
      for (uint k = 0; k < 64u; k++) {{
        uint g = k * 1024u + tid;                       // predecessor state; group m = g >> 4, h = g & 15
        float c = cA[g]; uint h = g & 15u;
        for (uint off = 1u; off < 16u; off <<= 1) {{
          float oc = simd_shuffle_xor(c, off); uint oh = simd_shuffle_xor(h, off);
          if (oc < c || (oc == c && oh < h)) {{ c = oc; h = oh; }}
        }}
        if ((lane & 15u) == 0u) {{ uint m = g >> 4; minc[m] = c; bp[t * 4096u + m] = uchar(h); }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float x0 = tgt[(size_t)r * {2 * T}u + 2u * t], x1 = tgt[(size_t)r * {2 * T}u + 2u * t + 1u];
      for (uint k = 0; k < 64u; k++) {{
        uint s = k * 1024u + tid;
        uint p = ((s * {MULT}u) & {MASK | (MASK << 16)}u) | {ORB | (ORB << 16)}u;
        half2 v = as_type<half2>(p);
        float d0 = x0 - g0 * (float(v.x) - mu[0]) * isd[0];
        float d1 = x1 - g0 * (float(v.y) - mu[1]) * isd[1];
        cB[s] = minc[s & 0xFFFu] + d0 * d0 + d1 * d1;
      }}
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      device float* tmp = cA; cA = cB; cB = tmp;
    }}
    float bv = 3.0e38f; uint bs = 0u;
    for (uint k = 0; k < 64u; k++) {{ uint s = k * 1024u + tid; float c = cA[s]; if (c < bv) {{ bv = c; bs = s; }} }}
    threadgroup float rv[1024]; threadgroup uint rs[1024];
    rv[tid] = bv; rs[tid] = bs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 512u; w > 0u; w >>= 1) {{
      if (tid < w && (rv[tid + w] < rv[tid] || (rv[tid + w] == rv[tid] && rs[tid + w] < rs[tid]))) {{ rv[tid] = rv[tid + w]; rs[tid] = rs[tid + w]; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
      uint s = rs[0]; err[r] = rv[0];
      for (int t = {T - 1}; t >= 0; t--) {{
        states[(size_t)r * {T}u + uint(t)] = ushort(s);
        if (t > 0) s = ((s & 0xFFFu) << 4) | uint(bp[uint(t) * 4096u + (s & 0xFFFu)]);
      }}
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_viterbi_v2k2c_T{T}", input_names=["tgt", "start", "gain", "mu", "isd"],
                                output_names=["states", "err", "scratch", "back"], source=src)


def viterbi_rows_c(tgt: mx.array, start: mx.array, gain: float, *_):
    R, n = tgt.shape; T = n // 2
    states, err, _, _ = _viterbi_kernel_c(T)(
        inputs=[tgt, start, mx.array([gain], dtype=mx.float32), mx.array(MU), mx.array(INV_SD)],
        grid=(R * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(R, T), (R,), (R * 131072,), (R * T * 4096,)], output_dtypes=[mx.uint16, mx.float32, mx.float32, mx.uint8])
    return states, err

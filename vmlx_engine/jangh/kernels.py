"""JANGTQ v2 Metal kernels  .

Design rule: every kernel is MLX's own affine kernel with ONLY the weight decode swapped for
`scale[row] * level(q)`. v2 bitstream packing == MLX affine packing, so the byte walk is identical.

  decode  : gather_qmv in MLX qmv_fast layout (2 simdgroups x 4 rows, 16 values per lane, x held in registers and
            reused across rows, row scale applied once), + fused gate/up/clamped-SwiGLU, + fused router-weighted down
  prefill : gather_qmm_rhs on expert-sorted rows, NAX (M5 tensor_ops via MLX steel/gemm/nax.h) with a TQBlockLoader
            writing the decoded tile to threadgroup memory; fused gate/up/SwiGLU variant; MLX 0.32 sg_active skip.

Codebook: v2 odd-cubic family level(q) = u*(A + B*u^2), u = q - (2^b-1)/2, constants BAKED into the Metal source.
  2-bit: register select tree over the 4 levels (== Lloyd-Max exactly)  -> decode 0.925x MLX affine2 (measured)
  3/6-bit: 3-byte groups (MLX 3-bit layout) + cubic arithmetic          -> decode 1.095x affine3
  4/8-bit: nibble/byte + cubic arithmetic                               -> decode 0.985x affine4
  (tables indexed by q were 1.6-2.2x slower at 3/4-bit: thread arrays with dynamic index spill.)

MLX custom kernels cannot #include MLX headers, so they are inlined from the RUNNING MLX's include/ directory
(utils.h is already in the custom-kernel preamble and is skipped). Inputs with < 8 elements are placed in
`constant` space by MLX; the prefill index array is padded to 8.
"""
from __future__ import annotations

import functools
import math
import os
import re
import threading

import mlx.core as mx

from .format import SUPPORTED_BITS, codebook, cubic_params

NSG, RPS = 2, 4          # decode: simdgroups per threadgroup, rows per simdgroup (MLX qmv_fast)
VPT = 16                 # decode: values per lane per block at every bit width (lane chunk = 2*bits bytes)
BLOCK = VPT * 32         # decode: values per block per simdgroup
BM = BN = BK = 64        # prefill tile (MLX default for gather_qmm_rhs)
WM = WN = 2
UNROLL = '_Pragma("clang loop unroll(full)")'
_TNAME = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}


# ------------------------------------------------------------------ header inlining
@functools.lru_cache(maxsize=None)
def _include_dir() -> str:
    return os.path.join(os.path.dirname(mx.__file__), "include")


def _expand(rel: str, seen: set) -> str:
    if rel in seen:
        return ""
    seen.add(rel)
    with open(os.path.join(_include_dir(), rel)) as f:
        src = f.read()
    out = []
    for line in src.splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_expand(m.group(1), seen))
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "\n".join(out)


def _mlx_headers(files) -> str:
    seen: set = set()
    _expand("mlx/backend/metal/kernels/utils.h", seen)  # already in the preamble
    return "".join(_expand(f"mlx/backend/metal/kernels/{p}", seen) + "\n" for p in files)


# ------------------------------------------------------------------ codebook (baked constants)
def _lit(v: float) -> str:
    return f"{float(v)!r}f"


@functools.lru_cache(maxsize=None)
def _cb_header() -> str:
    """tq_level<bits>(q) with the codebook as literal immediates (program-scope `constant` variables measured slower:
    they can be emitted as constant-buffer loads instead of folded immediates)."""
    lv = codebook(2)
    out = ["template <int bits> METAL_FUNC float tq_level(uint q);",
           "template <> METAL_FUNC float tq_level<2>(uint q) {"
           f" float lo = (q & 1u) ? {_lit(lv[1])} : {_lit(lv[0])}; float hi = (q & 1u) ? {_lit(lv[3])} : {_lit(lv[2])};"
           " return (q & 2u) ? hi : lo; }"]
    for b in SUPPORTED_BITS:
        if b == 2:
            continue
        a, c = cubic_params(b)
        out.append(f"template <> METAL_FUNC float tq_level<{b}>(uint q) {{ float u = float(q) - {_lit(((1 << b) - 1) / 2)};"
                   f" return u * fma({_lit(c)}, u * u, {_lit(a)}); }}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ prefill (NAX)
_LOADER = r'''
// v2 tile loader: identical byte walk to MLX QuantizedBlockLoader (bitstream packing); decode = scale[row]*level(q)
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, short bits>
struct TQBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short NCB = 1 << bits;
  const int src_ld; const int tile_stride; const short thread_idx; const short bi; const short bj;
  threadgroup T* dst; const device uint8_t* src; float s_row;
  TQBlockLoader(const device uint8_t* src_, const device half* scales_, const int src_ld_,
                threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id)
      : src_ld(src_ld_), tile_stride(BCOLS_PACKED * bytes_per_pack),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED), bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack) {
    s_row = (bi < BROWS) ? float(scales_[bi]) : 0.0f;
  }
  METAL_FUNC void decode_(short i) const {
    uint v = src[i * bytes_per_pack];
    if (bytes_per_pack > 1) v |= uint(src[i * bytes_per_pack + 1]) << 8;
    if (bytes_per_pack > 2) v |= uint(src[i * bytes_per_pack + 2]) << 16;
    for (short j = 0; j < pack_factor; j++)
      dst[i * pack_factor + j] = T(s_row * tq_level<bits>((v >> (j * bits)) & (NCB - 1)));
  }
  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void load_safe(short2 src_tile_dim) const {   // K % BK == 0 is enforced host-side; only rows can be ragged
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) return;
    if (bi >= src_tile_dim.y) { for (short i = 0; i < n_reads * pack_factor; i++) dst[i] = T(0); return; }
    for (short i = 0; i < n_reads; i++) decode_(i);
  }
  void next() { src += tile_stride; }
};
'''

_NAX_IMPL = r'''
using namespace mlx::steel;
// Copyright © 2024 Apple Inc.
// Adapted from MLX 0.32.2 hadamard.h; preserve its FP32 addition order.
template <short R>
METAL_FUNC void jangh_h32_radix(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < logR; s++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

METAL_FUNC float tq_act(float g, float u, float lim) {
  if (lim > 0.0f) { g = metal::min(g, lim); u = metal::clamp(u, -lim, lim); }
  return (g / (1.0f + metal::fast::exp(-g))) * u;
}
// FUSED=false: y = x W^T for one weight.  FUSED=true: y = act(x Wg^T, x Wu^T).
template <typename T, int bits, bool FUSED, bool EXPERT_ALIGNED = false, bool H32 = false>
METAL_FUNC void tq_gather_qmm_nax(
    const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* wu, const device half* su,
    const device uint32_t* indices, device T* y, const int M, const int N, const int K, const float lim,
    threadgroup T* Wg, threadgroup T* Wu, uint3 tid, uint simd_group_id, uint simd_lane_id,
    const device int* expert_offsets = nullptr, const device int* tile_offsets = nullptr, const int E = 0, const float h32_scale = 0.0f) {
  static_assert(!H32 || (FUSED && EXPERT_ALIGNED && metal::is_same<T, bfloat16_t>::value),
                "H32 requires BF16 fused expert tiles");
  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
  const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  int y_row = tid.y * BM;
  int row_end = M;
  if constexpr (EXPERT_ALIGNED) {
    // Uniform exit before any dereference or barrier; the grid is an upper bound.
    if (int(tid.y) >= tile_offsets[E]) return;
    int lo = 0, hi = E;
    // Upper bound skips repeated offsets belonging to empty experts.
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (tile_offsets[mid + 1] <= int(tid.y)) lo = mid + 1; else hi = mid;
    }
    y_row = expert_offsets[lo] + (int(tid.y) - tile_offsets[lo]) * BM;
    row_end = expert_offsets[lo + 1];
  }
  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, row_end - y_row));
  const short tgp_bn = short(min(BN, N - y_col));
  auto wgl = (const device uint8_t*)wg; auto wul = (const device uint8_t*)wu;
  x += size_t(y_row) * K; y += size_t(y_row) * N + y_col;
  wgl += size_t(y_col) * K_w; if (FUSED) wul += size_t(y_col) * K_w;
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  const short tm = SM * (simd_group_id / WN); const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = short(min(int(SM), max(0, row_end - (y_row + tm))));
  const short sgp_sn = short(min(int(SN), max(0, N - (y_col + tn))));
  uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
  while (n < tgp_bm) {
    n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
    for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
    threadgroup_barrier(mem_flags::mem_none);
    NAXTile<float, TM, TN> Gt; Gt.clear();
    NAXTile<float, TM, TN> Ut; if (FUSED) Ut.clear();
    const device T* xn = x + tm * K;
    thread loader_w_t lg(wgl + index * stride_w, sg + size_t(index) * N + y_col, K, Wg, simd_group_id, simd_lane_id);
    thread loader_w_t lu(FUSED ? wul + index * stride_w : wgl, FUSED ? su + size_t(index) * N + y_col : sg, K,
                         FUSED ? Wu : Wg, simd_group_id, simd_lane_id);
    const bool full_n = (tgp_bn == BN);
    // MLX 0.32 optimization: a simdgroup whose rows lie outside this expert segment skips the MMA
    // (it still helps load the shared weight tile and joins every barrier).
    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = (m_hi_lim > m_lo_lim) && (sgp_sn > 0);
    dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (full_n) { lg.load_unsafe(); if (FUSED) lu.load_unsafe(); }
        else { lg.load_safe(short2(BK, tgp_bn)); if (FUSED) lu.load_safe(short2(BK, tgp_bn)); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_active) {
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile; NAXTile<T, TN, TK> Bg;
            volatile int compiler_barrier;
            if constexpr (kAlignedM.value) Atile.load(xn + kk1, K); else Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            Bg.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
            tile_matmad_nax(Gt, Atile, metal::bool_constant<false>{}, Bg, metal::bool_constant<true>{});
            if (FUSED) {
              NAXTile<T, TN, TK> Bu;
              Bu.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
              tile_matmad_nax(Ut, Atile, metal::bool_constant<false>{}, Bu, metal::bool_constant<true>{});
            }
            (void)compiler_barrier;
          }
        }
        xn += BK; lg.next(); if (FUSED) lu.next();
      }
    });
    if (FUSED) {
      for (short i = 0; i < decltype(Gt)::kNumFrags; i++)
        for (short e = 0; e < decltype(Gt)::kElemsPerFrag; e++)
          Gt.val_frags[i][e] = tq_act(Gt.val_frags[i][e], Ut.val_frags[i][e], lim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (H32) {
      // All final-K weight reads precede the barrier above. Reuse dead Wg.
      // Match native output-store rounding before the FP32 transform.
      if (sg_active) {
        for (short f = 0; f < decltype(Gt)::kNumFrags; ++f) {
          for (short e = 0; e < decltype(Gt)::kElemsPerFrag; ++e) {
            const short2 pos = BaseNAXFrag::get_coord(e);
            const int row = tm + (f / TN) * 16 + pos.y;
            const int col = tn + (f % TN) * 16 + pos.x;
            if (row >= offset && row < offset_next && col < tgp_bn)
              Wg[row * BN + col] = T(Gt.val_frags[f][e]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint linear = simd_group_id * 32 + simd_lane_id;
      const uint hrow = linear / 2;
      const uint hcol = (linear % 2) * 32;
      // Invalid rows do not read scratch. Every lane still joins both barriers.
      if (hrow < uint(tgp_bm)) {
        float v[32];
        for (uint j = 0; j < 32; ++j) v[j] = float(Wg[hrow * BN + hcol + j]);
        jangh_h32_radix<16>(v);
        jangh_h32_radix<16>(v + 16);
        for (uint j = 0; j < 16; ++j) {
          float a = v[j], b = v[j + 16];
          v[j] = a + b;
          v[j + 16] = a - b;
        }
        for (uint j = 0; j < 32; ++j)
          y[hrow * N + hcol + j] = T(v[j] * h32_scale);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      if (sg_active) {
        if (m_lo_lim == 0 && m_hi_lim == SM && sgp_sn == SN) Gt.store(y + tm * N + tn, N);
        else Gt.store_slice(y + tm * N + tn, N, short2(0, m_lo_lim), short2(sgp_sn, m_hi_lim));
      }
    }
  }
}
'''


@functools.lru_cache(maxsize=None)
def _nax_header() -> str:
    return (_mlx_headers(["steel/gemm/gemm.h", "steel/gemm/nax.h", "steel/gemm/loader.h", "quantized_nax.h"])
            + _cb_header() + _LOADER + _NAX_IMPL)


@functools.lru_cache(maxsize=None)
def _nax_kernel(bits: int, fused: bool, tname: str):
    src = f'''
  constexpr int BK_padded = (64 + 16 / sizeof({tname}));
  threadgroup {tname} Wg[64 * BK_padded];
  threadgroup {tname} Wu[{'64' if fused else '1'} * BK_padded];
  tq_gather_qmm_nax<{tname}, {bits}, {'true' if fused else 'false'}>(
      x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], lim[0], Wg, Wu,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
'''
    return mx.fast.metal_kernel(
        name=f"jangtq2i_qmm_nax_b{bits}_{'fused' if fused else 'single'}_{tname}",
        input_names=["x", "wg", "sg", "wu", "su", "indices", "meta", "lim"],
        output_names=["y"], header=_nax_header(), source=src)


@functools.lru_cache(maxsize=None)
def _expert_nax_kernel(bits: int, fused: bool, tname: str, h32: bool = False):
    src = f'''
  constexpr int BK_padded = (64 + 16 / sizeof({tname}));
  threadgroup {tname} Wg[64 * BK_padded];
  threadgroup {tname} Wu[{'64' if fused else '1'} * BK_padded];
  tq_gather_qmm_nax<{tname}, {bits}, {'true' if fused else 'false'}, true, {'true' if h32 else 'false'}>(
      x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], lim[0], Wg, Wu,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
      expert_offsets, tile_offsets, meta[3]{", scale[0]" if h32 else ""});
'''
    return mx.fast.metal_kernel(
        name=f"jangh_expert_qmm_nax_b{bits}_{fused}_{tname}" + ("_h32" if h32 else ""),
        input_names=["x", "wg", "sg", "wu", "su", "indices", "expert_offsets", "tile_offsets", "meta", "lim"] + (["scale"] if h32 else []),
        output_names=["y"], header=_nax_header(), source=src)


def expert_tile_plan(idx_sorted, experts):
    """GPU-only offsets shared by gate/up and down; no host synchronization."""
    starts = mx.searchsorted(idx_sorted, mx.arange(experts + 1, dtype=mx.uint32)).astype(mx.int32)
    counts = (starts[1:] - starts[:-1] + BM - 1) // BM
    tiles = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts).astype(mx.int32)])
    return starts, tiles


@functools.lru_cache(maxsize=1)
def _h32_scale():
    return mx.array([float(1.0 / math.sqrt(32))], dtype=mx.float32)


def gather_qmm_expert_sorted(x, packed, scales, idx, bits, plan, *, packed_u=None, scales_u=None, limit=0.0, rotate_output=False):
    """Internal NAX route, admitted only for qualified geometry by the switch."""
    M, width = x.shape
    experts, columns = packed.shape[:2]
    fused = packed_u is not None
    starts, tiles = plan
    if rotate_output:
        expected_packed = (288, 2048, 4096 * bits // 32)
        if not (fused and scales_u is not None and bits in (2, 3) and x.dtype == mx.bfloat16
                and (width, columns, experts) == (4096, 2048, 288) and M >= 64
                and packed.shape == packed_u.shape == expected_packed
                and packed.dtype == packed_u.dtype == mx.uint32
                and scales.shape == scales_u.shape == (288, 2048)
                and scales.dtype == scales_u.dtype == mx.float16
                and idx.dtype == mx.uint32 and idx.size == M
                and starts.shape == tiles.shape == (289,)
                and starts.dtype == tiles.dtype == mx.int32 and nax_available()):
            raise ValueError("JANGH H32 requires qualified BF16 GLM expert tiles")
    return _expert_nax_kernel(bits, fused, _TNAME[x.dtype], rotate_output)(
        inputs=[x, packed, scales, packed_u if fused else packed, scales_u if fused else scales,
                idx, starts, tiles, mx.array([M, columns, width, experts], dtype=mx.int32),
                _consts(float(limit), dtype=mx.float32)] + ([_h32_scale()] if rotate_output else []),
        grid=(((columns + BN - 1) // BN) * 128, (M + BM - 1) // BM + experts, 1),
        threadgroup=(128, 1, 1), output_shapes=[(M, columns)], output_dtypes=[x.dtype])[0]


# ------------------------------------------------------------------ prefill fallback (no NAX: M1-M4, macOS < 26.2)
_STEEL_IMPL = r'''
template <typename T, int bits>
METAL_FUNC void tq_gather_qmm_steel(
    const device T* x, const device uint32_t* wg, const device half* sg, const device uint32_t* indices, device T* y,
    const int M, const int N, const int K, threadgroup T* Xs, threadgroup T* Ws,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  // MLX affine_gather_qmm_rhs (non-NAX, transpose=true) with the TQ tile loader. Tiles: 16x32x32, 1x2 simdgroups.
  constexpr int BM = 16, BN = 32, BK = 32, WM = 1, WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;
  const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const int y_row = tid.y * BM; const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = short(min(BN, N - y_col));
  auto wl = (const device uint8_t*)wg;
  x += size_t(y_row) * K; y += size_t(y_row) * N + y_col; wl += size_t(y_col) * K_w;
  uint32_t index; short offset; uint32_t index_next = indices[y_row]; short offset_next = 0; int n = 0;
  while (n < tgp_bm) {
    n++; offset = offset_next; index = index_next; offset_next = tgp_bm;
    for (; n < tgp_bm; n++) { if (indices[y_row + n] != index) { offset_next = n; index_next = indices[y_row + n]; break; } }
    threadgroup_barrier(mem_flags::mem_none);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(wl + index * stride_w, sg + size_t(index) * N + y_col, K, Ws, simd_group_id, simd_lane_id);
    if (tgp_bm == BM && tgp_bn == BN) gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    else if (tgp_bn == BN) gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else if (tgp_bm == BM) gemm_loop_unaligned<true, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    else gemm_loop_unaligned<false, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, (short)BK);
    if (offset_next - offset == BM && tgp_bn == BN) mma_op.store_result(y, N);
    else mma_op.store_result_slice(y, N, short2(0, offset), short2(tgp_bn, offset_next));
  }
}
'''


@functools.lru_cache(maxsize=None)
def _steel_header() -> str:
    return (_mlx_headers(["steel/gemm/gemm.h", "quantized_utils.h", "quantized.h"])
            + _cb_header() + _LOADER + _STEEL_IMPL)


@functools.lru_cache(maxsize=None)
def _steel_kernel(bits: int, tname: str):
    src = f'''
  constexpr int BK_padded = (32 + 16 / sizeof({tname}));
  threadgroup {tname} Xs[16 * BK_padded];
  threadgroup {tname} Ws[32 * BK_padded];
  tq_gather_qmm_steel<{tname}, {bits}>(x, wg, sg, indices, y, meta[0], meta[1], meta[2], Xs, Ws,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
'''
    return mx.fast.metal_kernel(name=f"jangtq2_qmm_steel_b{bits}_{tname}",
                                input_names=["x", "wg", "sg", "indices", "meta"], output_names=["y"],
                                header=_steel_header(), source=src)


@functools.lru_cache(maxsize=None)
def nax_available() -> bool:
    """Mirror of MLX metal::is_nax_available(): macOS >= 26.2 and GPU generation >= 17 (>= 18 for 'p' parts).
    Env override JANGTQ2_PREFILL=steel|nax for A/B tests only (never needed in production)."""
    from .runtime_identity import PREFILL

    forced = PREFILL
    if forced in ("steel", "nax"):
        return forced == "nax"
    import platform
    try:
        ver = tuple(int(v) for v in platform.mac_ver()[0].split(".")[:2])
        # Ask the GPU explicitly: mx.device_info() describes the DEFAULT device, which a process (a test child, a
        # CPU-side tool) may set to the CPU ("arm64"), and the answer feeds the persisted-cache identity
        # (jangt.switch.jangt_runtime_identity) as well as kernel routing.
        arch = (mx.device_info(mx.gpu) if hasattr(mx, "device_info") else mx.metal.device_info())["architecture"]
        m = re.search(r"g(\d+)([a-z])", arch)
        gen, kind = int(m.group(1)), m.group(2)
        return ver >= (26, 2) and gen >= (18 if kind == "p" else 17)
    except Exception:
        return False


def _gather_qmm_sorted_steel(x, packed, scales, idx, bits):
    M, K = x.shape
    N = packed.shape[1]
    if K % 32:
        raise ValueError(f"jangtq2 steel prefill requires K % 32 == 0 (K={K})")
    return _steel_kernel(bits, _TNAME[x.dtype])(
        inputs=[x, packed, scales, idx, mx.array([M, N, K], dtype=mx.int32)],
        grid=(((N + 31) // 32) * 64, (M + 15) // 16, 1), threadgroup=(64, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


@functools.lru_cache(maxsize=4096)
def _consts(*vals, dtype=None):
    return mx.array(list(vals), dtype=dtype)


def gather_qmm_sorted(x, packed, scales, cb_unused, idx_sorted, bits, *, packed_u=None, scales_u=None, limit=0.0):
    """Prefill: x (M, K) rows sorted by expert (idx_sorted (M,) uint32).
    single: returns x W_e^T (M, N).  fused (packed_u given): returns act(x Wg^T, x Wu^T) with clamped SwiGLU."""
    M, K = x.shape
    N = packed.shape[1]
    if K % BK:
        raise ValueError(f"jangtq2 prefill requires K % {BK} == 0 (K={K})")
    fused = packed_u is not None
    idx = idx_sorted.astype(mx.uint32).reshape(-1)
    if idx.size < 8:   # MLX custom kernels put inputs with < 8 elements in `constant` space; the impl takes device*
        idx = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (8 - idx.size,))])
    if not nax_available():
        g = _gather_qmm_sorted_steel(x, packed, scales, idx, bits)
        if not fused:
            return g
        u = _gather_qmm_sorted_steel(x, packed_u, scales_u, idx, bits).astype(mx.float32)
        g = g.astype(mx.float32)
        if limit > 0:
            g = mx.minimum(g, limit); u = mx.clip(u, -limit, limit)
        return (g * mx.sigmoid(g) * u).astype(x.dtype)
    k = _nax_kernel(bits, fused, _TNAME[x.dtype])
    return k(inputs=[x, packed, scales, packed_u if fused else packed, scales_u if fused else scales, idx,
                     mx.array([M, N, K], dtype=mx.int32), _consts(float(limit), dtype=mx.float32)],
             grid=(((N + BN - 1) // BN) * WM * WN * 32, (M + BM - 1) // BM, 1), threadgroup=(WM * WN * 32, 1, 1),
             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


# ------------------------------------------------------------------ activation rotation (one pass)
@functools.lru_cache(maxsize=None)
def _h32_kernel(xname: str, oname: str):
    """One simdgroup per 32-wide block: lane i holds x[block*32 + i]; five butterfly stages across the lanes."""
    src = f'''
    uint K = meta[0];
    uint lane = thread_index_in_simdgroup;
    size_t off = (size_t)threadgroup_position_in_grid.z * K + (size_t)threadgroup_position_in_grid.y * 32u + lane;
    float v = float(x[off]);
    {UNROLL} for (uint h = 1u; h < 32u; h <<= 1) {{
      float o = simd_shuffle_xor(v, ushort(h));
      v = (lane & h) ? (o - v) : (v + o);
    }}
    out[off] = static_cast<{oname}>(v * 0.17677669529663687f);
'''
    return mx.fast.metal_kernel(name=f"jangh_h32_{xname}_{oname}", input_names=["x", "meta"], output_names=["out"], source=src)


def h32_rows(x, out_dtype=None):
    """Blockwise normalized Hadamard-32 over the last axis (== mx.hadamard_transform on (..., K/32, 32) blocks),
    ONE kernel launch instead of cast + transform + cast. x (..., K), K % 32 == 0."""
    out_dtype = out_dtype or x.dtype
    if x.ndim == 0 or x.dtype not in _TNAME or out_dtype not in _TNAME:
        raise ValueError("JANGH h32 requires floating activation rows")
    K = x.shape[-1]
    if K == 0 or K % 32:
        raise ValueError("jangtq2 h32 requires K % 32 == 0")
    M = x.size // K
    if M == 0:
        return x.astype(out_dtype)
    k = _h32_kernel(_TNAME[x.dtype], _TNAME[out_dtype])
    return k(inputs=[x.reshape(M, K), _consts(K, 0, 0, 0, 0, 0, 0, 0, dtype=mx.uint32)], grid=(32, K // 32, M), threadgroup=(32, 1, 1),
             output_shapes=[(M, K)], output_dtypes=[out_dtype])[0].reshape(x.shape)


# ------------------------------------------------------------------ decode (qmv)
def slot_fastest_enabled() -> bool:
    """Multi-row decode MoE kernels schedule the expert slot as the FASTEST grid axis (VMLX_MOE_SLOT_FASTEST, default
    on, read per call for in-process A/B). With the slot on the slowest axis (z) the GPU ran every row tile of slot 0
    before slot 1, so an expert chosen by several rows of a speculative verify block was fetched from DRAM once per
    row; with the slot on y, consecutive threadgroups compute the same weight rows for different slots and repeated
    experts hit the cache. Per-threadgroup arithmetic is unchanged: outputs are bit-identical. 1-row decode keeps
    the original order."""
    return os.environ.get("VMLX_MOE_SLOT_FASTEST", "1").strip().lower() not in {"0", "false", "off", "no"}


def swap_grid_yz(src: str) -> str:
    """Kernel source with threadgroup grid axes y and z exchanged (row tile <-> slot)."""
    return (src.replace("threadgroup_position_in_grid.y", "__TGP_Y__")
               .replace("threadgroup_position_in_grid.z", "threadgroup_position_in_grid.y")
               .replace("__TGP_Y__", "threadgroup_position_in_grid.z"))


def _qdot(bits: int, wr: str, acc: str) -> str:
    """Metal: accumulate dot(xt[0..15], level(q)) over one row's 16-value lane chunk at byte pointer `wr`.
    Wide loads: the lane chunk is 2*bits bytes, 4-byte aligned for 2/4/6/8-bit (uint32 words) and 2-byte aligned for
    3-bit (ushort). Byte-by-byte loads measured ~20% slower (4 memory instructions instead of 1 at 2-bit)."""
    mask = (1 << bits) - 1
    if bits in (2, 4, 8):
        nw = bits // 2              # words per lane chunk: 2-bit 1, 4-bit 2, 8-bit 4
        per = 32 // bits            # values per word
        return f'''
        {{ {UNROLL} for (uint j = 0; j < {nw}u; j++) {{ uint ww = {wr}[j];
            {UNROLL} for (uint t = 0; t < {per}u; t++)
              {acc} = fma(xt[j * {per}u + t], tq_level<{bits}>((ww >> ({bits}u * t)) & {mask}u), {acc}); }} }}'''
    if bits == 6:                   # 12 bytes = 3 words = 16 values; values may straddle words -> 64-bit window
        return f'''
        {{ uint w0 = {wr}[0], w1 = {wr}[1], w2 = {wr}[2];
          ulong lo = ulong(w0) | (ulong(w1) << 32); ulong hi = ulong(w1) | (ulong(w2) << 32);
          {UNROLL} for (uint t = 0; t < 16u; t++) {{
            uint bp = 6u * t;
            uint q = bp < 32u ? uint((lo >> bp) & {mask}ul) : uint((hi >> (bp - 32u)) & {mask}ul);
            {acc} = fma(xt[t], tq_level<6>(q), {acc}); }} }}'''
    # 3-bit: 6 bytes = 3 ushorts = two 24-bit groups of 8 values
    return f'''
        {{ uint s0 = {wr}[0], s1 = {wr}[1], s2 = {wr}[2];
          uint g0 = s0 | ((s1 & 0xffu) << 16); uint g1 = (s1 >> 8) | (s2 << 8);
          {UNROLL} for (uint t = 0; t < 8u; t++) {acc} = fma(xt[t], tq_level<3>((g0 >> (3u * t)) & 7u), {acc});
          {UNROLL} for (uint t = 0; t < 8u; t++) {acc} = fma(xt[8u + t], tq_level<3>((g1 >> (3u * t)) & 7u), {acc}); }}'''


# Unguarded main blocks + one guarded tail block when K % BLOCK != 0 but rows are aligned (qwen4_exp down K=640).
# JANGH_TAIL_SPLIT=0 restores the fully guarded form for A/B.
from .runtime_identity import TAIL_SPLIT as _TAIL_SPLIT

TAIL_SPLIT = _TAIL_SPLIT == "1"


def _ptype(bits: int) -> tuple[str, int]:
    """Element type used to address packed weights (alignment-provable) and its size in bytes."""
    return ("uint16_t", 2) if bits == 3 else ("uint32_t", 4)


def _wptr(bits: int, base: str, e: str) -> str:
    """Typed pointer to row0's lane chunk of expert `e` (never derived from a uint8 cast)."""
    t, sz = _ptype(bits)
    cast = "" if t == "uint32_t" else f"(const device {t}*)"
    return f"{cast}{base} + ((size_t){e} * N + row0) * (RB / {sz}u) + lane * (LB / {sz}u)"


def _prologue(bits: int) -> str:
    return f'''
    const uint RB = K * {bits}u / 8u;       // bytes per row (multiple of 4: K % 32 == 0)
    const uint LB = {2 * bits}u;             // bytes per lane chunk (16 values)
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * {NSG * RPS}u + sgi * {RPS}u;'''


_HAD32 = f'''
      // blockwise normalized Hadamard-32 of the activation (v2 rotation "hadamard32"): lane holds 16 contiguous
      // values -> 4 in-register butterfly stages, then one stage across the lane pair (lane ^ 1), then 1/sqrt(32).
      {UNROLL} for (uint h = 1u; h < 16u; h <<= 1) {{
        {UNROLL} for (uint i = 0; i < 16u; i++) {{
          if ((i & h) == 0u) {{ float a0 = xt[i], b0 = xt[i + h]; xt[i] = a0 + b0; xt[i + h] = a0 - b0; }}
        }}
      }}
      {{ const bool upper = (lane & 1u) != 0u;
        {UNROLL} for (uint i = 0; i < 16u; i++) {{
          float other = simd_shuffle_xor(xt[i], 1u);
          xt[i] = (upper ? (other - xt[i]) : (xt[i] + other)) * 0.17677669529663687f;
        }}
      }}'''


def _load_x(aligned: bool, rotate: bool = False) -> str:
    """aligned: K % BLOCK == 0 and N % 8 == 0 -> unconditional loads the compiler can vectorize.
    A predicated load per value (the guarded form) measured 1.26x vs 0.95x for the same kernel.
    rotate: apply the v2 blockwise Hadamard-32 to the lane pair's 32 values (inactive lanes hold zeros; K % 32 == 0
    keeps every 32-block entirely active or inactive, and both lanes of a pair always take the shuffle)."""
    if aligned:
        base = f'''
      const bool active = true;
      float xt[{VPT}];
      {UNROLL} for (uint i = 0; i < {VPT}u; i++) xt[i] = float(xp[i]);'''
    else:
        base = f'''
      bool active = (k0 + lane * {VPT}u) < K;
      float xt[{VPT}];
      {UNROLL} for (uint i = 0; i < {VPT}u; i++) xt[i] = active ? float(xp[i]) : 0.0f;'''
    return base + (_HAD32 if rotate else "")


def _tail_mode(K: int, N: int) -> str:
    """"al": K % BLOCK == 0 and rows aligned (unguarded); "tl": rows aligned, K % BLOCK != 0 -> unguarded main blocks
    + ONE guarded tail block (qwen4_exp down_proj K=640 = 512 + 128); "gd": rows not aligned (fully guarded)."""
    if N % (NSG * RPS):
        return "gd"
    return "al" if K % BLOCK == 0 else "tl"


def _k_loop(mode: str, rotate: bool, body_al: str, body_gd: str, advance: str) -> str:
    if mode == "al":
        return f"for (uint k0 = 0; k0 < K; k0 += {BLOCK}u) {{ {_load_x(True, rotate)} {body_al} {advance} }}"
    if mode == "gd":
        return f"for (uint k0 = 0; k0 < K; k0 += {BLOCK}u) {{ {_load_x(False, rotate)} {body_gd} {advance} }}"
    return (f"uint Kmain = K - (K % {BLOCK}u);\n"
            f"    for (uint k0 = 0; k0 < Kmain; k0 += {BLOCK}u) {{ {_load_x(True, rotate)} {body_al} {advance} }}\n"
            f"    if (Kmain < K) {{ uint k0 = Kmain; {_load_x(False, rotate)} {body_gd} }}")


@functools.lru_cache(maxsize=None)
def _qmv_kernel(bits: int, fused: bool, xname: str, aligned, rotate: bool = False, yz: bool = False):
    mode = aligned if isinstance(aligned, str) else ("al" if aligned else "gd")
    return _qmv_kernel_mode(bits, fused, xname, mode, rotate, yz)


def _qmv_rows(bits: int, fused: bool, guard: str) -> str:
    rows = f'''
      {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
        {guard}
        const device {_ptype(bits)[0]}* wr = wp + r * (RB / {_ptype(bits)[1]}u); float a = 0.0f;
        {_qdot(bits, "wr", "a")}
        accg[r] += a;'''
    if fused:
        rows += f'''
        const device {_ptype(bits)[0]}* ur = up + r * (RB / {_ptype(bits)[1]}u); float b2 = 0.0f;
        {_qdot(bits, "ur", "b2")}
        accu[r] += b2;'''
    return rows + "\n      }"


@functools.lru_cache(maxsize=None)
def _qmv_kernel_mode(bits: int, fused: bool, xname: str, mode: str, rotate: bool = False, yz: bool = False):
    aligned = mode == "al"
    guard_rows = "if (!active) continue;" if mode == "tl" else "if (row0 + r >= N || !active) continue;"
    loop = _k_loop(mode, rotate, _qmv_rows(bits, fused, ""), _qmv_rows(bits, fused, guard_rows),
                   f"wp += {BLOCK * bits // 8 // _ptype(bits)[1]}u; up += {BLOCK * bits // 8 // _ptype(bits)[1]}u; xp += {BLOCK}u;")
    guard = "" if aligned else "if (row0 + r >= N || !active) continue;"
    rows = f'''
      {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
        {guard}
        const device {_ptype(bits)[0]}* wr = wp + r * (RB / {_ptype(bits)[1]}u); float a = 0.0f;
        {_qdot(bits, "wr", "a")}
        accg[r] += a;'''
    if fused:
        rows += f'''
        const device {_ptype(bits)[0]}* ur = up + r * (RB / {_ptype(bits)[1]}u); float b2 = 0.0f;
        {_qdot(bits, "ur", "b2")}
        accu[r] += b2;'''
    rows += "\n      }"
    if fused:
        store = f'''
    float L = lim[0];
    {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
      float g = simd_sum(accg[r]); float u = simd_sum(accu[r]);
      if (lane == 0 && row0 + r < N) {{
        g *= float(sg[(size_t)e * N + row0 + r]); u *= float(su[(size_t)e * N + row0 + r]);
        if (L > 0.0f) {{ g = metal::min(g, L); u = metal::clamp(u, -L, L); }}
        out[(size_t)disp * N + row0 + r] = (g / (1.0f + metal::fast::exp(-g))) * u;
      }}
    }}'''
    else:
        store = f'''
    {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
      float s = simd_sum(accg[r]);
      if (lane == 0 && row0 + r < N) out[(size_t)disp * N + row0 + r] = s * float(sg[(size_t)e * N + row0 + r]);
    }}'''
    src = f'''
    uint K = meta[0], N = meta[1], xdiv = meta[2];
    {_prologue(bits)}
    uint disp = threadgroup_position_in_grid.z;
    uint e = idx[disp];
    const device {_ptype(bits)[0]}* wp = {_wptr(bits, "wg", "e")};
    const device {_ptype(bits)[0]}* up = {_wptr(bits, "wu", "e")};
    auto xp = x + (size_t)(disp / xdiv) * K + lane * {VPT}u;
    float accg[{RPS}]; float accu[{RPS}];
    {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{ accg[r] = 0.0f; accu[r] = 0.0f; }}
    {loop}
    {store}
'''
    return mx.fast.metal_kernel(
        name=f"jangtq2t_qmv_b{bits}_{'fused' if fused else 'single'}_{xname}_{mode}{'_h32' if rotate else ''}{'_yz' if yz else ''}",
        input_names=["x", "wg", "sg", "wu", "su", "idx", "meta", "lim"],
        output_names=["out"], header=_cb_header(), source=swap_grid_yz(src) if yz else src)


def gather_qmv(x, packed, scales, cb_unused, idx, bits, *, x_per_dispatch: bool, packed_u=None, scales_u=None, limit=0.0,
               rotate: bool = False, slot_fastest: bool = False):
    """Decode: x (n_x, K) any float dtype. idx (ndisp,) expert per dispatch.
    x row for dispatch d = d if x_per_dispatch else d // (ndisp // n_x).  Returns float32 (ndisp, N).
    Single: W_e x.  Fused (packed_u given): act(Wg x, Wu x) with clamped SwiGLU."""
    ndisp = idx.size
    nx, K = x.shape
    N = packed.shape[1]
    if K % 32:
        raise ValueError("jangtq2 requires K % 32 == 0")
    xdiv = 1 if x_per_dispatch else ndisp // nx
    fused = packed_u is not None
    yz = bool(slot_fastest) and slot_fastest_enabled()      # callers pass it only for multi-row calls
    k = _qmv_kernel(bits, fused, _TNAME[x.dtype], _tail_mode(K, N) if TAIL_SPLIT else (K % BLOCK == 0 and N % (NSG * RPS) == 0), rotate, yz)
    tiles = (N + NSG * RPS - 1) // (NSG * RPS)
    return k(inputs=[x, packed, scales, packed_u if fused else packed, scales_u if fused else scales,
                     idx.astype(mx.uint32).reshape(-1), _consts(K, N, xdiv, dtype=mx.uint32),
                     _consts(float(limit), dtype=mx.float32)],
             grid=(NSG * 32, ndisp, tiles) if yz else (NSG * 32, tiles, ndisp), threadgroup=(NSG * 32, 1, 1),
             output_shapes=[(ndisp, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------ dense multi-row decode (verify windows)
#
# One dense JANGTQ2 projection over M activation rows that share every weight read (gather_qmv with
# x_per_dispatch=True re-reads the whole weight once per row: Qwen3.8-27B JANGH2's 2-bit MLP cost +15 ms per verify
# row, so DFlash2 widths 2-5 paid 53/68/77/78 ms per forward against 37 ms for one row).  Per (row m, output r) the
# accumulation is exactly gather_qmv's: a_m starts at 0 per 512-value block, fma over the lane's 16 values in
# ascending t, folded into accg_m[r], then simd_sum - so every row equals the per-row kernel bit for bit, and the
# routing between the two is a speed decision only (tests/test_jangh_dense_rows.py).
# Registers bound the design: ROWS_PER_SG output rows per simdgroup (RPSM) and XCH activation values held per row,
# the 16-value lane chunk walked in XCH-wide pieces (t ascending, so the order is unchanged).  Measured on M5 Max
# (32 distinct 21 MB weights, 27B MLP shapes, 2-bit): 1 row 0.057 ms with RPSM 8 + in-kernel rotation vs 0.065-0.091
# per-row; 2/3/4/5 rows 0.065/0.09/0.12-0.15/0.15-0.19 ms with RPSM 4 + XCH 8 on fp32 pre-rotated rows vs per-row
# 0.13/0.18/0.25/0.32 and the sorted tile path 0.17-0.20 (flat); from 6 rows the sorted path wins.
# Rotation: at 1 row in-kernel (the per-row kernel's _HAD32 on each row's 16 values); at 2+ rows h32_rows(x, float32)
# once per activation (the same fp32 butterfly stages in the same order, verified bit-identical), reused across the
# gate/up pair that reads the same activation (``_h32_cached``).
ROWS_KERNEL_BITS = (2, 4, 8)
ROWS_MAX_LAUNCH = 8


def _rows_chunk_dot(bits: int, mr: int, xch: int, had: str) -> str:
    mask = (1 << bits) - 1
    per = 32 // bits
    out = []
    for c in range(0, VPT, xch):
        loads = "\n".join(f"            xt{m}[i] = active ? float(xp{m}[{c}u + i]) : 0.0f;" for m in range(mr))
        fmas = "\n".join(f"            a{m} = fma(xt{m}[t - {c}u], lv, a{m});" for m in range(mr))
        out.append(f'''
          {UNROLL} for (uint i = 0; i < {xch}u; i++) {{
{loads}
          }}
          {had if c == 0 else ""}
          {UNROLL} for (uint t = {c}u; t < {c + xch}u; t++) {{
            float lv = tq_level<{bits}>((ww[t / {per}u] >> ({bits}u * (t % {per}u))) & {mask}u);
{fmas}
          }}''')
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def _rows_kernel(bits: int, mr: int, rpsm: int, xch: int, xname: str, guarded: bool, rotate: bool):
    if bits not in ROWS_KERNEL_BITS or VPT % xch or (rotate and xch != VPT):
        raise ValueError("jangtq2 rows kernel: unsupported bits / chunk / rotation combination")
    nw = bits // 2
    psz = _ptype(bits)[1]
    had = "\n".join(_HAD32.replace("xt[", f"xt{m}[") for m in range(mr)) if rotate else ""
    acc_decl = " ".join(f"float accg{m}[{rpsm}];" for m in range(mr))
    acc_zero = " ".join(f"accg{m}[r] = 0.0f;" for m in range(mr))
    xt_decl = " ".join(f"float xt{m}[{xch}];" for m in range(mr))
    a_decl = " ".join(f"float a{m} = 0.0f;" for m in range(mr))
    a_fold = " ".join(f"accg{m}[r] += a{m};" for m in range(mr))
    xptr = "\n    ".join(f"auto xp{m} = x + (size_t){m} * K + lane * {VPT}u;" for m in range(mr))
    xadv = " ".join(f"xp{m} += {BLOCK}u;" for m in range(mr))
    guard = "if (row0 + r >= N || !active) continue;" if guarded else ""
    active = f"bool active = (k0 + lane * {VPT}u) < K;" if guarded else "const bool active = true;"
    store = "\n".join(f'''
      {{ float s = simd_sum(accg{m}[r]);
        if (lane == 0 && row0 + r < N) out[(size_t){m} * N + row0 + r] = s * float(sg[(size_t)e * N + row0 + r]); }}'''
                      for m in range(mr))
    src = f'''
    uint K = meta[0], N = meta[1];
    const uint RB = K * {bits}u / 8u;
    const uint LB = {2 * bits}u;
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * {NSG * rpsm}u + sgi * {rpsm}u;
    uint e = idx[0];
    const device uint32_t* wp = wg + ((size_t)e * N + row0) * (RB / {psz}u) + lane * (LB / {psz}u);
    {xptr}
    {acc_decl} {xt_decl}
    {UNROLL} for (uint r = 0; r < {rpsm}u; r++) {{ {acc_zero} }}
    for (uint k0 = 0; k0 < K; k0 += {BLOCK}u) {{
      {active}
      {UNROLL} for (uint r = 0; r < {rpsm}u; r++) {{
        {guard}
        const device uint32_t* wr = wp + r * (RB / {psz}u);
        uint ww[{nw}];
        {UNROLL} for (uint j = 0; j < {nw}u; j++) ww[j] = wr[j];
        {a_decl}
{_rows_chunk_dot(bits, mr, xch, had)}
        {a_fold}
      }}
      wp += {BLOCK * bits // 8 // psz}u; {xadv}
    }}
    {UNROLL} for (uint r = 0; r < {rpsm}u; r++) {{
{store}
    }}
'''
    return mx.fast.metal_kernel(
        name=f"jangtq2t_qmv_rows_m{mr}_r{rpsm}_c{xch}_b{bits}_{xname}_{'gd' if guarded else 'al'}{'_h32' if rotate else ''}",
        input_names=["x", "wg", "sg", "idx", "meta"], output_names=["out"], header=_cb_header(), source=src)


def rows_kernel_supports(bits: int) -> bool:
    return bits in ROWS_KERNEL_BITS


def gather_qmv_rows(x, packed, scales, idx, bits, *, rotate: bool = False, rows_per_simdgroup: int | None = None,
                    chunk: int | None = None, rows_per_launch: int = ROWS_MAX_LAUNCH):
    """Dense decode over M rows that share each weight read: x (M, K) -> float32 (M, N) of expert idx[0].
    rotate: apply the v2 Hadamard-32 in the kernel (1-row calls); multi-row callers pass rows already rotated by
    h32_rows(x, float32).  Every row's bits equal gather_qmv(x_per_dispatch=True)'s."""
    M, Kd = x.shape
    N = packed.shape[1]
    if Kd % 32:
        raise ValueError("jangtq2 requires K % 32 == 0")
    if rows_per_simdgroup is None:
        rows_per_simdgroup = 8 if M == 1 else 4
    if chunk is None:
        chunk = VPT if (M == 1 or rotate) else 8
    guarded = not (Kd % BLOCK == 0 and N % (NSG * rows_per_simdgroup) == 0)
    tiles = (N + NSG * rows_per_simdgroup - 1) // (NSG * rows_per_simdgroup)
    outs = []
    for m0 in range(0, M, rows_per_launch):
        mr = min(rows_per_launch, M - m0)
        k = _rows_kernel(bits, mr, rows_per_simdgroup, chunk, _TNAME[x.dtype], guarded, rotate)
        outs.append(k(inputs=[x[m0:m0 + mr] if M > mr else x, packed, scales, idx.astype(mx.uint32).reshape(-1),
                             _consts(Kd, N, dtype=mx.uint32)],
                      grid=(NSG * 32, tiles, 1), threadgroup=(NSG * 32, 1, 1),
                      output_shapes=[(mr, N)], output_dtypes=[mx.float32])[0])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


_H32_LAST = threading.local()


def h32_rows_cached(x):
    """h32_rows(x, float32), reused while the caller passes the SAME array object (a layer's gate and up projections
    read one activation).  Holding x keeps its id from being recycled, so `is` cannot match another array."""
    slot = getattr(_H32_LAST, "slot", None)
    if slot is not None and slot[0] is x:
        return slot[1]
    r = h32_rows(x, mx.float32)
    _H32_LAST.slot = (x, r)
    return r


def _wdown_rows(bits: int, guard: str) -> str:
    return f'''
        {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
          {guard}
          const device {_ptype(bits)[0]}* wr = wp + r * (RB / {_ptype(bits)[1]}u); float a = 0.0f;
          {_qdot(bits, "wr", "a")}
          acc[r] += a;
        }}'''


@functools.lru_cache(maxsize=None)
def _qmv_weighted_down_kernel(bits: int, tname: str, xname: str, aligned, rotate: bool = False, yz: bool = False):
    """Decode down projection fused with the router-weighted sum over the k selected experts:
    y[t, r] = sum_k w[t,k] * scale[e_k, r] * dot(h[t,k,:], level(q[e_k, r, :]))."""
    mode = aligned if isinstance(aligned, str) else ("al" if aligned else "gd")
    aligned = mode == "al"
    wloop = _k_loop(mode, rotate, _wdown_rows(bits, ""),
                    _wdown_rows(bits, "if (!active) continue;" if mode == "tl" else "if (row0 + r >= N || !active) continue;"),
                    f"wp += {BLOCK * bits // 8 // _ptype(bits)[1]}u; xp += {BLOCK}u;")
    src = f'''
    uint K = meta[0], N = meta[1], KT = meta[2];
    {_prologue(bits)}
    uint t = threadgroup_position_in_grid.z;
    float out_acc[{RPS}]; {UNROLL} for (uint r = 0; r < {RPS}u; r++) out_acc[r] = 0.0f;
    for (uint kk = 0; kk < KT; kk++) {{
      uint disp = t * KT + kk;
      uint e = idx[disp];
      float wk = float(wts[disp]);
      const device {_ptype(bits)[0]}* wp = {_wptr(bits, "wg", "e")};
      auto xp = x + (size_t)disp * K + lane * {VPT}u;
      float acc[{RPS}]; {UNROLL} for (uint r = 0; r < {RPS}u; r++) acc[r] = 0.0f;
      {wloop}
      {UNROLL} for (uint r = 0; r < {RPS}u; r++) {{
        float sres = simd_sum(acc[r]);
        if (row0 + r < N) out_acc[r] += wk * sres * float(sg[(size_t)e * N + row0 + r]);
      }}
    }}
    {UNROLL} for (uint r = 0; r < {RPS}u; r++)
      if (lane == 0 && row0 + r < N) out[(size_t)t * N + row0 + r] = static_cast<{tname}>(out_acc[r]);
'''
    return mx.fast.metal_kernel(name=f"jangtq2t_qmv_wdown_b{bits}_{tname}_{xname}_{mode}{'_h32' if rotate else ''}{'_yz' if yz else ''}",
                                input_names=["x", "wg", "sg", "idx", "wts", "meta"],
                                output_names=["out"], header=_cb_header(), source=swap_grid_yz(src) if yz else src)


def gather_qmv_weighted_down(h, packed, scales, cb_unused, idx, weights, bits, out_dtype, rotate: bool = False,
                             slot_fastest: bool = False):
    """Decode: h (T*k, K) activations per dispatch, idx/weights (T, k). Returns (T, N) = sum_k w * (W_e h)."""
    T, kt = idx.shape
    K = h.shape[-1]
    N = packed.shape[1]
    yz = bool(slot_fastest) and T > 1 and slot_fastest_enabled()
    k = _qmv_weighted_down_kernel(bits, _TNAME[out_dtype], _TNAME[h.dtype],
                                  _tail_mode(K, N) if TAIL_SPLIT else (K % BLOCK == 0 and N % (NSG * RPS) == 0), rotate, yz)
    tiles = (N + NSG * RPS - 1) // (NSG * RPS)
    return k(inputs=[h, packed, scales, idx.astype(mx.uint32).reshape(-1), weights.astype(mx.float32).reshape(-1),
                     _consts(K, N, kt, dtype=mx.uint32)],
             grid=(NSG * 32, T, tiles) if yz else (NSG * 32, tiles, T), threadgroup=(NSG * 32, 1, 1),
             output_shapes=[(T, N)], output_dtypes=[out_dtype])[0]


# ------------------------------------------------------------------ sorted prefill weighted epilogue
@functools.lru_cache(maxsize=1)
def prefill_weighted_unsort_available() -> bool:
    """Only measured M5 Max / MLX 0.32.2 and 0.32.3 reductions are admitted."""
    import importlib.metadata
    try:
        return (importlib.metadata.version("mlx") in ("0.32.2", "0.32.3")
                and mx.metal.is_available()
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


@functools.lru_cache(maxsize=1)
def _prefill_weighted_unsort_kernel():
    # MLX 0.32.2 col_reduce_small: eight BF16 leaves, then sequential
    # BF16 adds in original route order. FP32 FMA/reduction is NOT equivalent.
    return mx.fast.metal_kernel(
        name="vmlx_jangh_prefill_weighted_unsort",
        input_names=["Y", "INV", "S"], output_names=["OUT"],
        compile_options={"math_mode": "safe"},
        source=r'''
const uint i = thread_position_in_grid.x;
const uint d = i % 4096;
const uint r = i / 4096;
if (r >= ROWS) return;
bfloat total = bfloat(0.0f);
for (uint k=0; k<8; ++k) {
  const uint row = INV[r*8+k];
  // Internal caller passes argsort(argsort(...)); guard corrupted indices
  // defensively so even malformed input can never read beyond Y.
  if (row >= ROWS*8) { OUT[i] = bfloat(NAN); return; }
  const bfloat w = bfloat(S[r*8+k]);
  const bfloat p = bfloat(float(Y[size_t(row)*4096+d]) * float(w));
  const bfloat leaf = bfloat(float(p) + 0.0f);
  if (k == 0) total = leaf;
  else total = bfloat(float(leaf) + float(total));
}
OUT[i] = total;
''')


def prefill_weighted_unsort(y, inverse, scores, *, enabled=False):
    """Exact qualified BF16 top-eight epilogue, or None for stock fallback.

    The inverse is an internal permutation of sorted dispatch rows. Shape and
    dtype checks precede dispatch; custom kernels materialize strided inputs.
    No model/cache arrays are retained. Decode and unweighted calls do not use
    this helper. The startup flag is owned by the caller and cache identity.
    """
    if not enabled:
        return None
    if (y.ndim != 2 or y.dtype != mx.bfloat16 or y.shape[1] != 4096
            or scores.ndim != 2 or scores.shape[1] != 8 or scores.shape[0] < 8
            or scores.dtype not in (mx.float32, mx.bfloat16)
            or inverse.ndim != 1 or inverse.dtype != mx.uint32
            or y.shape[0] != scores.size or inverse.size != scores.size
            or y.size >= 2**32 or mx.default_device() != mx.gpu
            or not prefill_weighted_unsort_available()):
        return None
    return _prefill_weighted_unsort_kernel()(
        inputs=[y, inverse, scores], template=[("ROWS", scores.shape[0])],
        grid=(scores.shape[0]*4096, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(scores.shape[0], 4096)], output_dtypes=[mx.bfloat16])[0]


# ------------------------------------------------------------------ generic prefill weighted unsort (any D, any k)
@functools.lru_cache(maxsize=None)
def _weighted_unsort_kernel(tname: str):
    """OUT[t, d] = sum_k S[t,k] * Y[INV[t*k + j], d], fp32 accumulation, one thread per (t, d4) handling 4 columns.
    Replaces the unfused y[inv].reshape(T,k,D) * scores -> sum (three passes over T*k*D) with one pass.
    Not bit-identical to MLX's bf16 reduction order (fp32 accumulate): exactness gate = greedy byte identity."""
    return mx.fast.metal_kernel(
        name=f"vmlx_jangh_weighted_unsort_{tname}",
        input_names=["Y", "INV", "S"], output_names=["OUT"],
        source=f'''
const uint d4 = thread_position_in_grid.x;            // column quad
const uint t = thread_position_in_grid.y;             // token
if (d4 * 4u >= D || t >= ROWS) return;
float4 acc = float4(0.0f);
for (uint j = 0; j < KK; ++j) {{
  const uint row = INV[t * KK + j];
  const float w = float(S[t * KK + j]);
  const device {tname}* yr = Y + (size_t)row * D + d4 * 4u;
  acc += w * float4(float(yr[0]), float(yr[1]), float(yr[2]), float(yr[3]));
}}
device {tname}* o = OUT + (size_t)t * D + d4 * 4u;
o[0] = {tname}(acc.x); o[1] = {tname}(acc.y); o[2] = {tname}(acc.z); o[3] = {tname}(acc.w);
''')


def weighted_unsort(y, inv, scores, out_dtype=None):
    """y (T*k, D) sorted expert outputs, inv (T*k,) inverse permutation, scores (T, k) -> (T, D)."""
    T, kk = scores.shape
    D = y.shape[-1]
    if D % 4:
        return None
    dt = out_dtype or y.dtype
    k = _weighted_unsort_kernel(_TNAME[y.dtype])
    return k(inputs=[y, inv.astype(mx.uint32).reshape(-1), scores.astype(mx.float32).reshape(-1)],
             template=[("D", D), ("KK", kk), ("ROWS", T)],
             grid=(D // 4, T, 1), threadgroup=(min(256, D // 4), 1, 1),
             output_shapes=[(T, D)], output_dtypes=[y.dtype])[0].astype(dt)

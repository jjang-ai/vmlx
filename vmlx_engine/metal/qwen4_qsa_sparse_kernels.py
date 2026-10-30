# SPDX-License-Identifier: Apache-2.0
"""Sparse QSA attention kernels for Qwen4 (Flash-Next) prefill on M5-class GPUs.

Qwen4's QSA indexer gives every query row up to 512 four-token blocks plus a
zero-to-three-token causal tail.  MLX's stock attention sees that selection as
a dense additive mask over the whole context, so it still multiplies every
query against every cached key.  The two kernels here touch only the selected
keys and reproduce MLX's arithmetic bit for bit:

``sparse_scores``
    The QK^T a stock unfused GQA attention would compute, but only for the
    selected keys: a ``[1, 2, 12, S, T]`` score tensor in the input dtype, -inf
    everywhere else.  Each score is a 16x32x16 matrix-unit MMA chain over the
    head dim in ascending 16-wide slices, the same chain MLX's NAX GEMM runs.
    Softmax and the probability-value product stay with MLX.

``sparse_online_attention``
    The whole attention in one pass with an online softmax, reproducing MLX's
    head-dim-split NAX attention for 256-wide heads: the scores are two
    128-dim partial sums added together, and 32-key tiles are visited in
    absolute order.  Tiles that hold no selected key are skipped; in the stock
    kernel those tiles are fully masked and change nothing.

Selection contract (both kernels): ``ids`` is a contiguous ``[1, 1, S, 512]``
uint32 buffer whose first ``min(512, (pos + 1) // 4)`` entries per row are
strictly ascending block ids (``pos`` = absolute query position); later
entries are ignored.  K/V are the logical cache views (length == context) and
may be strided slices of a larger buffer; Q may be a transposed view.

Fragment layout and the ``matmul2d`` descriptor follow MLX's own NAX helpers
(``mlx/backend/metal/kernels/steel/gemm/nax.h``).
"""

from __future__ import annotations

import hashlib
from functools import cache

import mlx.core as mx

QUERY_HEADS = 24
KV_HEADS = 2
GROUP = QUERY_HEADS // KV_HEADS        # 12 query heads per KV head
HEAD_DIM = 256
BLOCK = 4                               # tokens per QSA block
TOP_K = 512                             # selected blocks per row
CANDIDATES = TOP_K * BLOCK + BLOCK - 1  # 2051 slots: blocks then causal tail
SCORE_SIMDGROUPS = 4
ONLINE_SIMDGROUPS = 4                   # 2 form QK halves; each owns 256/4 output dims

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Fixed-trip loops over fragment arrays must unroll so the arrays stay in registers.
#define QSA_UNROLL _Pragma("clang loop unroll(full)")

// Lane -> (column, row) of a 16x16 matrix-unit fragment; element e of the
// lane's 8 sits at row (y + (e / 4) * 8), column (x + e % 4).
inline short2 qsa_frag_coord(uint lane) {
  const short quad = short(lane >> 2);
  const short row = (quad & 4) | ((short(lane) >> 1) & 3);
  const short col = ((quad & 2) | (short(lane) & 1)) * 4;
  return short2(col, row);
}

// C[16x32] += A[16x16] * B, B given as two 16x16 fragments (columns 0-15 and
// 16-31; stored transposed when TRANS_B).  MLX's NAX fragment MMA descriptor.
template <bool TRANS_B, typename A, typename B>
inline void qsa_mma(thread metal::vec<float, 8>& c0, thread metal::vec<float, 8>& c1,
                    thread const metal::vec<A, 8>& a,
                    thread const metal::vec<B, 8>& b0, thread const metal::vec<B, 8>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TRANS_B, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto lhs = op.template get_left_input_cooperative_tensor<A, B, float>();
  auto rhs = op.template get_right_input_cooperative_tensor<A, B, float>();
  auto acc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(lhs)>,
      metal::remove_addrspace_t<decltype(rhs)>, float>();
  QSA_UNROLL for (short i = 0; i < 8; ++i) {
    lhs[i] = a[i];
    rhs[i] = b0[i];
    rhs[8 + i] = b1[i];
    acc[i] = c0[i];
    acc[8 + i] = c1[i];
  }
  op.run(lhs, rhs, acc);
  QSA_UNROLL for (short i = 0; i < 8; ++i) {
    c0[i] = acc[i];
    c1[i] = acc[8 + i];
  }
}

// One 16x16 input fragment from a row-major device matrix: this lane's two
// rows (y, y + 8) x four contiguous columns; rows at or past `rows_left`
// read as zero (MLX's load_rows).
template <typename T>
inline metal::vec<T, 8> qsa_load_frag(const device T* base, size_t row_stride, short2 fc,
                                      int rows_left) {
  // four contiguous elements at an offset that is a multiple of 4: one 8-byte load
  const device T* p = base + (size_t)fc.y * row_stride + fc.x;
  const metal::vec<T, 4> lo = fc.y < rows_left
      ? *(const device metal::vec<T, 4>*)p : metal::vec<T, 4>(T(0));
  const metal::vec<T, 4> hi = fc.y + 8 < rows_left
      ? *(const device metal::vec<T, 4>*)(p + 8 * row_stride) : metal::vec<T, 4>(T(0));
  return metal::vec<T, 8>(lo, hi);
}

// MLX fragment row reduction: in-lane pairs, then lanes ^1 and ^8.
template <bool IS_MAX>
inline void qsa_row_fold(thread const metal::vec<float, 8>& v, thread float* rows) {
  QSA_UNROLL for (short r = 0; r < 2; ++r) {
    float a = IS_MAX ? metal::max(v[4 * r], v[4 * r + 1]) : v[4 * r] + v[4 * r + 1];
    float b = IS_MAX ? metal::max(v[4 * r + 2], v[4 * r + 3]) : v[4 * r + 2] + v[4 * r + 3];
    float x = IS_MAX ? metal::max(a, b) : a + b;
    float y = simd_shuffle_xor(x, ushort(1));
    x = IS_MAX ? metal::max(x, y) : x + y;
    y = simd_shuffle_xor(x, ushort(8));
    x = IS_MAX ? metal::max(x, y) : x + y;
    rows[r] = IS_MAX ? metal::max(rows[r], x) : rows[r] + x;
  }
}
"""

# One threadgroup per (query row, KV head).  The 12 query rows (padded to 16)
# are staged once; each simdgroup then takes every SG-th group of 32 candidate
# slots, resolves their tokens lane-parallel and shares them with shuffles.
_SCORES = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint flat = thread_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.x);
  const int kv = int(threadgroup_position_in_grid.y);
  const int rows = int(q_shape[2]);
  const int context = int(k_shape[2]);
  const int pos = q_offset + row;
  const int complete = (pos + 1) / 4;
  const int usable = min(512, complete);
  const short2 fc = qsa_frag_coord(lane);

  threadgroup T q_rows[16 * 256];
  for (uint i = flat; i < 16u * 256u; i += 32u * SG) {
    const int h = int(i / 256u), d = int(i % 256u);
    q_rows[i] = h < 12
        ? q[(size_t)(kv * 12 + h) * q_strides[1] + (size_t)row * q_strides[2] + d]
        : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const T* keys = k + (size_t)kv * k_strides[1];
  device const uint* row_ids = ids + (size_t)row * 512;

  for (int grp = int(sg); grp * 32 < 2051; grp += SG) {
    const int slot = grp * 32 + int(lane);
    int token = -1;
    if (slot < 2048) {
      if (slot / 4 < usable) {
        const ulong cand = ulong(row_ids[slot / 4]) * 4ul + ulong(slot % 4);
        if (cand < ulong(context) && cand <= ulong(pos)) token = int(cand);
      }
    } else if (slot < 2051) {
      const int cand = complete * 4 + slot - 2048;
      if (cand < context && cand <= pos) token = cand;
    }
    // tokens of the four key rows this lane feeds into the B fragments
    int key_tok[4];
    QSA_UNROLL for (short r = 0; r < 4; ++r) key_tok[r] = simd_shuffle(token, ushort(fc.y + 8 * r));

    const size_t kstride = (size_t)k_strides[2];
    metal::vec<float, 8> lo = 0.0f, hi = 0.0f;
    QSA_UNROLL for (int c = 0; c < 16; ++c) {
      const int d = c * 16 + fc.x;
      const metal::vec<T, 8> a(
          *(const threadgroup metal::vec<T, 4>*)(q_rows + fc.y * 256 + d),
          *(const threadgroup metal::vec<T, 4>*)(q_rows + (fc.y + 8) * 256 + d));
      metal::vec<T, 4> part[4];
      QSA_UNROLL for (short r = 0; r < 4; ++r)
        part[r] = key_tok[r] >= 0
            ? *(const device metal::vec<T, 4>*)(keys + (size_t)key_tok[r] * kstride + d)
            : metal::vec<T, 4>(T(0));
      qsa_mma<true>(lo, hi, a, metal::vec<T, 8>(part[0], part[1]), metal::vec<T, 8>(part[2], part[3]));
    }

    QSA_UNROLL for (short j = 0; j < 4; ++j) {
      const int t_lo = simd_shuffle(token, ushort(fc.x + j));
      const int t_hi = simd_shuffle(token, ushort(16 + fc.x + j));
      QSA_UNROLL for (short r = 0; r < 2; ++r) {
        const int h = fc.y + r * 8;
        if (h >= 12) continue;
        device T* dst = scores + ((size_t)(kv * 12 + h) * rows + row) * context;
        if (t_lo >= 0) dst[t_lo] = static_cast<T>(lo[r * 4 + j]);
        if (t_hi >= 0) dst[t_hi] = static_cast<T>(hi[r * 4 + j]);
      }
    }
  }
"""

# One threadgroup per (query row, KV head), four simdgroups.  Simdgroups 0 and
# 1 each form the partial QK^T over one 128-dim half (the split MLX uses for
# 256-wide heads); every simdgroup then reads both partials, runs the same
# online-softmax update and accumulates P @ V for its own 64 output dims, which
# keeps the per-thread accumulator at four fragments.  All simdgroups walk the
# same tile sequence, recomputed from the ids lane-parallel.
_ONLINE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint half_id = sg & 1u;            // QK half for simdgroups 0 and 1
  const bool forms_qk = sg < 2u;
  constexpr int FR = 16 / NSG;             // output fragments per simdgroup
  const int out0 = int(sg) * 16 * FR;      // this simdgroup's output dims
  const int row = int(threadgroup_position_in_grid.x);
  const int kv = int(threadgroup_position_in_grid.y);
  const int rows = int(q_shape[2]);
  const int context = int(k_shape[2]);
  const int pos = q_offset + row;
  const int complete = (pos + 1) / 4;
  const int usable = min(512, complete);
  const int tail_tokens = (pos + 1) % 4;
  const int tail_tile = (complete * 4) / 32 * 32;
  const uint tail_bits = ((1u << tail_tokens) - 1u) << ((complete * 4) % 32);
  const float scale2 = scale * 1.44269504089f;
  const int dim0 = int(half_id) * 128;
  const short2 fc = qsa_frag_coord(lane);

  // [tile parity][QK half]: alternating buffers let one barrier per tile suffice
  threadgroup float partial[2][2][32 * 16];
  uint parity = 0;

  device const T* keys = k + (size_t)kv * k_strides[1];
  device const T* values = v + (size_t)kv * v_strides[1];
  device const uint* row_ids = ids + (size_t)row * 512;

  metal::vec<T, 8> qf[8];
  QSA_UNROLL for (int c = 0; c < 8; ++c)
    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      const int h = fc.y + (e / 4) * 8;
      qf[c][e] = h < 12
          ? q[(size_t)(kv * 12 + h) * q_strides[1] + (size_t)row * q_strides[2]
              + dim0 + c * 16 + fc.x + e % 4]
          : T(0);
    }

  metal::vec<float, 8> acc[FR];
  QSA_UNROLL for (int i = 0; i < FR; ++i) acc[i] = 0.0f;
  float run_max[2] = {-metal::numeric_limits<float>::max(), -metal::numeric_limits<float>::max()};
  float denom[2] = {0.0f, 0.0f};

  int cursor = 0;
  bool tail_open = tail_tokens != 0;
  while (true) {
    int base;
    uint bits;
    if (cursor < usable) {
      const int s = cursor + int(lane);
      const uint block = s < usable ? row_ids[s] : 0xffffffffu;
      const uint lead = simd_broadcast(block, ushort(0)) / 8u;
      const bool here = s < usable && block / 8u == lead;
      bits = simd_or(here ? (15u << ((block % 8u) * 4u)) : 0u);
      cursor += int(simd_sum(here ? 1u : 0u));
      base = int(lead) * 32;
      if (tail_open && base == tail_tile) {
        bits |= tail_bits;
        tail_open = false;
      }
    } else if (tail_open) {
      base = tail_tile;
      bits = tail_bits;
      tail_open = false;
    } else {
      break;
    }

    // partial QK^T over this simdgroup's 128 dims; key rows past the context read as 0
    const int left = context - base;      // key rows of this tile inside the context
    const size_t kstride = (size_t)k_strides[2];
    const device T* tile_keys = keys + (size_t)base * kstride + dim0;
    metal::vec<float, 8> s0 = 0.0f, s1 = 0.0f;
    if (forms_qk) {
    QSA_UNROLL for (int c = 0; c < 8; ++c) {
      const metal::vec<T, 8> b0 = qsa_load_frag(tile_keys + c * 16, kstride, fc, left);
      const metal::vec<T, 8> b1 = qsa_load_frag(tile_keys + 16 * kstride + c * 16, kstride, fc, left - 16);
      qsa_mma<true>(s0, s1, qf[c], b0, b1);
    }
    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      partial[parity][half_id][lane * 16 + e] = s0[e];
      partial[parity][half_id][lane * 16 + 8 + e] = s1[e];
    }
    }
    // this simdgroup's V block for the tile, fetched while the partials land
    const size_t vstride = (size_t)v_strides[2];
    const device T* tile_values = values + (size_t)base * vstride + out0;
    metal::vec<T, 8> vf[FR][2];
    QSA_UNROLL for (int i = 0; i < FR; ++i) {
      vf[i][0] = qsa_load_frag(tile_values + i * 16, vstride, fc, left);
      vf[i][1] = qsa_load_frag(tile_values + 16 * vstride + i * 16, vstride, fc, left - 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      s0[e] = partial[parity][0][lane * 16 + e] + partial[parity][1][lane * 16 + e];
      s1[e] = partial[parity][0][lane * 16 + 8 + e] + partial[parity][1][lane * 16 + 8 + e];
    }
    parity ^= 1u;

    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      const int col = fc.x + e % 4;
      const int t0 = base + col, t1 = base + 16 + col;
      const bool v0 = ((bits >> col) & 1u) && t0 < context && t0 <= pos;
      const bool v1 = ((bits >> (16 + col)) & 1u) && t1 < context && t1 <= pos;
      s0[e] = v0 ? s0[e] * scale2 : -INFINITY;
      s1[e] = v1 ? s1[e] * scale2 : -INFINITY;
    }

    float next_max[2] = {run_max[0], run_max[1]};
    qsa_row_fold<true>(s0, next_max);
    qsa_row_fold<true>(s1, next_max);
    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      s0[e] = fast::exp2(s0[e] - next_max[e / 4]);
      s1[e] = fast::exp2(s1[e] - next_max[e / 4]);
    }
    float factor[2];
    QSA_UNROLL for (short r = 0; r < 2; ++r) {
      factor[r] = fast::exp2(run_max[r] - next_max[r]);
      run_max[r] = next_max[r];
      denom[r] *= factor[r];
    }
    qsa_row_fold<false>(s0, denom);
    qsa_row_fold<false>(s1, denom);
    QSA_UNROLL for (int i = 0; i < FR; ++i)
      QSA_UNROLL for (short e = 0; e < 8; ++e) acc[i][e] *= factor[e / 4];

    QSA_UNROLL for (int i = 0; i < FR; i += 2) {
      qsa_mma<false>(acc[i], acc[i + 1], s0, vf[i][0], vf[i + 1][0]);
      qsa_mma<false>(acc[i], acc[i + 1], s1, vf[i][1], vf[i + 1][1]);
    }
  }

  const float inv[2] = {1.f / denom[0], 1.f / denom[1]};
  QSA_UNROLL for (int i = 0; i < FR; ++i)
    QSA_UNROLL for (short e = 0; e < 8; ++e) {
      const int h = fc.y + (e / 4) * 8;
      if (h < 12)
        out[((size_t)(kv * 12 + h) * rows + row) * 256 + out0 + i * 16 + fc.x + e % 4] =
            static_cast<T>(acc[i][e] * inv[e / 4]);
    }
"""

# Identity of this file's kernel math for prefix-cache namespaces.
MATH_ABI = "qsa-sparse-nax-v1-" + hashlib.sha256(
    (_HEADER + _SCORES + _ONLINE).encode()).hexdigest()[:12]


def matrix_units_available() -> bool:
    """Whether MLX targets a GPU whose matmuls run on the NAX matrix units.

    Mirrors MLX's architecture gate: Apple GPU generation 17+ (18+ for the
    'p' variants).  These kernels only exist in that form.
    """
    try:
        arch = str(mx.device_info().get("architecture", ""))
    except (AttributeError, RuntimeError):
        return False
    prefix = "applegpu_g"
    if not arch.startswith(prefix) or len(arch) <= len(prefix) + 1:
        return False
    generation = arch[len(prefix):-1]
    if not generation.isdigit():
        return False
    return int(generation) >= (18 if arch.endswith("p") else 17)


@cache
def _scores_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_qsa_sparse_scores",
        input_names=["q", "k", "ids", "q_offset"],
        output_names=["scores"],
        header=_HEADER,
        source=_SCORES,
        ensure_row_contiguous=False,
    )


@cache
def _online_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_qsa_sparse_online",
        input_names=["q", "k", "v", "ids", "scale", "q_offset"],
        output_names=["out"],
        header=_HEADER,
        source=_ONLINE,
        ensure_row_contiguous=False,
    )


def sparse_scores(q: mx.array, k: mx.array, ids: mx.array, q_offset: int,
                  scale: float) -> mx.array:
    """Scaled selected-key scores ``[1, 2, 12, S, T]``; -inf where not selected.

    ``q`` is scaled in its own dtype first (an MLX multiply), exactly as the
    stock unfused attention scales queries before its QK^T matmul.
    """
    rows, context = int(q.shape[2]), int(k.shape[2])
    scaled = q * mx.array(scale, dtype=q.dtype)
    return _scores_kernel()(
        inputs=[scaled, k, ids, int(q_offset)],
        template=[("T", q.dtype), ("SG", SCORE_SIMDGROUPS)],
        grid=(32 * SCORE_SIMDGROUPS * rows, KV_HEADS, 1),
        threadgroup=(32 * SCORE_SIMDGROUPS, 1, 1),
        output_shapes=[(1, KV_HEADS, GROUP, rows, context)],
        output_dtypes=[q.dtype],
        init_value=-float("inf"),
    )[0]


def sparse_online_attention(q: mx.array, k: mx.array, v: mx.array, ids: mx.array,
                            q_offset: int, scale: float) -> mx.array:
    """Fused sparse attention ``[1, 24, S, 256]`` with an online softmax."""
    rows = int(q.shape[2])
    return _online_kernel()(
        inputs=[q, k, v, ids, float(scale), int(q_offset)],
        template=[("T", q.dtype), ("NSG", ONLINE_SIMDGROUPS)],
        grid=(32 * ONLINE_SIMDGROUPS * rows, KV_HEADS, 1),
        threadgroup=(32 * ONLINE_SIMDGROUPS, 1, 1),
        output_shapes=[(1, QUERY_HEADS, rows, HEAD_DIM)],
        output_dtypes=[q.dtype],
    )[0]

// SPDX-License-Identifier: Apache-2.0
//
// Vendored into MTPLX from oMLX (https://github.com/jundot/omlx), PR #3244,
// revision dc312e6e905e03d21ef0c4a86289cbfa2cf857cc.
//
// MTPLX ships only the one measured production specialization
// (BK=64, DC=64) in fp16 and bf16. The oMLX file also instantiates
// (128,32), (256,32) and (128,64); packaging kernels that no MTPLX caller
// can request only widens the Metal-compilation surface that must be
// qualified on the first M3 run.

// Include order is load-bearing: Steel's attention header provides Limits
// used by the specialized Qwen kernel. The params struct is defined here
// (Metal has no access to the C++ translation unit) and MUST match the
// layout in qwen4_qsa_sparse_gqa.cpp byte-for-byte.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

struct Qwen4QSASparseGQAParams {
  int B;
  int q_heads;
  int kv_heads;
  int qL;
  int kL;
  int topk;
  int gqa_factor;
  int q_offset;

  float scale;

  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t Topk_strides[3];
  int64_t O_strides[3];
};

#include "kernels/steel_qwen4_qsa_sparse_gqa.h"
// clang-format on

#define instantiate_qwen4_sparse_gqa(tname, dtype, bk, dc)                     \
  instantiate_kernel("qwen4_qsa_sparse_gqa_" #tname "_bk" #bk "_dc" #dc        \
                     "_gqa12_hp16_d256_wm2",                                   \
                     qwen4_qsa_sparse_gqa_attention, dtype, bk, dc, 12, 16,    \
                     256, 2, uint, float)

instantiate_qwen4_sparse_gqa(float16, half, 64, 64);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 64, 64);

#define instantiate_qwen4_sparse_scores(tname, dtype) \
  instantiate_kernel("qwen4_qsa_sparse_scores_" #tname "_bk64_dc64_gqa12_hp16_d256_wm2", \
                     qwen4_qsa_sparse_gqa_scores, dtype, 64, 64, 12, 16, 256, 2, uint, float)
instantiate_qwen4_sparse_scores(float16, half);
instantiate_qwen4_sparse_scores(bfloat16, bfloat16_t);

template <typename T>
[[kernel]] void qwen4_sparse_scores_fill(
    device T* output [[buffer(0)]],
    constant int& tokens [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  for (int token = int(lane); token < tokens; token += 256) {
    output[size_t(group.x) * tokens + token] = T(-INFINITY);
  }
}
instantiate_kernel("qwen4_sparse_scores_fill_float16", qwen4_sparse_scores_fill, half);
instantiate_kernel("qwen4_sparse_scores_fill_bfloat16", qwen4_sparse_scores_fill, bfloat16_t);

// Match MLX 0.32.2 NAX GEMM's 16-wide accumulation order. Inputs are already
// scaled in their own dtype by the C++ wrapper. One simdgroup computes the
// twelve query heads against 32 selected keys; masked positions stay -inf
// from the preceding fill dispatch. No compact-PV reduction is introduced.
#if defined(MTPLX_QSA_HAS_NAX)
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen4_sparse_scores_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device uint* Ids [[buffer(3)]],
    device T* O [[buffer(4)]],
    constant Qwen4QSASparseGQAParams* params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;
  (void)V;
  const int row = int(group.x), kv = int(group.y), tile = int(group.z);
  const int absolute = params->q_offset + row;
  const int complete = (absolute + 1) / 4;
  const int valid_blocks = min(512, complete);
  threadgroup int selected[32];
  const int slot = tile * 32 + int(lane);
  int token = -1;
  if (slot < 2048 && slot / 4 < valid_blocks) {
    const ulong candidate =
        ulong(Ids[size_t(row) * params->Topk_strides[2] + slot / 4]) * 4
        + ulong(slot % 4);
    if (candidate < ulong(params->kL) && candidate <= ulong(absolute))
      token = int(candidate);
  } else if (slot >= 2048 && slot < 2051) {
    const int candidate = complete * 4 + slot - 2048;
    if (candidate < params->kL && candidate <= absolute) token = candidate;
  }
  selected[lane] = token;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 coord = BaseNAXFrag::get_coord();
  NAXTile<float, 1, 2> accum;
  accum.clear();
  STEEL_PRAGMA_NO_UNROLL
  for (int d = 0; d < 256; d += 32) {
    NAXTile<T, 1, 2> a;
    NAXTile<T, 2, 2> b;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 2; ++j) {
      thread auto& frag = a.frag_at(0, j);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 8; ++e) {
        const int h = coord.y + (e / 4) * 8;
        const int dim = d + j * 16 + coord.x + e % 4;
        frag[e] = h < 12
            ? Q[size_t(kv * 12 + h) * params->Q_strides[1]
                + size_t(row) * params->Q_strides[2] + dim] : T(0);
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 2; ++j) {
        thread auto& frag = b.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          const int key = selected[i * 16 + coord.y + (e / 4) * 8];
          const int dim = d + j * 16 + coord.x + e % 4;
          frag[e] = key >= 0
              ? K[size_t(kv) * params->K_strides[1]
                  + size_t(key) * params->K_strides[2] + dim] : T(0);
        }
      }
    }
    tile_matmad_nax(accum, a, metal::bool_constant<false>{},
                   b, metal::bool_constant<true>{});
  }
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < 2; ++j) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
      const int h = coord.y + (e / 4) * 8;
      const int key = selected[j * 16 + coord.x + e % 4];
      if (h < 12 && key >= 0)
        O[(size_t(kv * 12 + h) * params->qL + row) * params->kL + key]
            = T(accum.frag_at(0, j)[e]);
    }
  }
}
instantiate_kernel("qwen4_sparse_scores_nax_float16", qwen4_sparse_scores_nax, half);
instantiate_kernel("qwen4_sparse_scores_nax_bfloat16", qwen4_sparse_scores_nax, bfloat16_t);
#endif

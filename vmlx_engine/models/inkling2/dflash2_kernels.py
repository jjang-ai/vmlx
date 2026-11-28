# SPDX-License-Identifier: Apache-2.0
"""Fused Apple Silicon Metal kernels for Inkling-2 DFlash2 candidate selection.

Eliminates two bottlenecks in the DFlash2 draft loop:
1. ``dflash2_topk``: 2-stage exact top-K (K <= 32) over unpadded vocab (~200K)
   replacing ``mx.argpartition`` + ``mx.take_along_axis`` (~1.0 ms -> ~0.06 ms).
2. ``dflash2_markov_select_greedy``: single-threadgroup multi-position Markov
   edge scorer + post-correction softcap + greedy path selector running all P
   positions inside ONE Metal dispatch (~1.5 ms Python loop -> ~0.03 ms).
3. ``dflash2_markov_step_scores``: single-dispatch edge scorer + post-correction
   softcap for sampled (temperature > 0) candidate selection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# 1. Exact 2-Stage Metal Top-K (K <= 32, e.g. K=16) over V (~200,058)
# ---------------------------------------------------------------------------
_TOPK_STAGE1_SRC = r"""
  // Grid: (256 * N_BLOCKS, P, 1), TG: (256, 1, 1) (8 simdgroups).
  // Each TG scans a contiguous slice of [V] and extracts its exact top-K.
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint blk  = threadgroup_position_in_grid.x;
  const uint pos  = threadgroup_position_in_grid.y;

  const uint V = uint(logits_shape[1]);
  const uint n_blk = threadgroups_per_grid.x;
  const uint chunk = (V + n_blk - 1u) / n_blk;
  const uint base = blk * chunk;
  const uint end  = min(base + chunk, V);

  // Max chunk for V <= 262144 and n_blk = 256 is 1024 -> <= 4 elements/thread!
  constexpr uint MAX_E = 6u;
  float vals[MAX_E];
  uint  idxs[MAX_E];
  uint count = 0u;
  const device float* row = logits + size_t(pos) * size_t(V);
  for (uint j = base + tid; j < end && count < MAX_E; j += 256u) {
    vals[count] = row[j];
    idxs[count] = j;
    count++;
  }
  for (uint c = count; c < MAX_E; ++c) {
    vals[c] = -INFINITY;
    idxs[c] = 0xFFFFFFFFu;
  }

  threadgroup float sg_v[8];
  threadgroup uint  sg_i[8];
  threadgroup uint  win_idx_sh;

  const size_t out_off = (size_t(pos) * size_t(n_blk) + size_t(blk)) * uint(TOPK);
  for (uint k = 0u; k < uint(TOPK); ++k) {
    float bv = vals[0];
    uint  bi = idxs[0];
    uint  bc = 0u;
    #pragma unroll
    for (uint c = 1u; c < MAX_E; ++c) {
      if (vals[c] > bv || (vals[c] == bv && idxs[c] < bi)) {
        bv = vals[c];
        bi = idxs[c];
        bc = c;
      }
    }
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
      win_idx_sh = gi;
      blk_vals[out_off + k] = gv;
      blk_idxs[out_off + k] = gi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bi == win_idx_sh && bi != 0xFFFFFFFFu) {
      vals[bc] = -INFINITY;
      idxs[bc] = 0xFFFFFFFFu;
    }
  }
"""

_TOPK_STAGE2_SRC = r"""
  // Grid: (256, P, 1), TG: (256, 1, 1).
  // Merges M = N_BLOCKS * TOPK (e.g. 256 * 16 = 4096) candidates -> exact global top-K!
  // Each thread owns TOPK elements (16 floats in registers).
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const uint pos  = threadgroup_position_in_grid.y;

  const uint M = uint(blk_vals_shape[1]);
  const device float* rv = blk_vals + size_t(pos) * size_t(M);
  const device uint*  ri = blk_idxs + size_t(pos) * size_t(M);

  float vals[TOPK];
  uint  idxs[TOPK];
  #pragma unroll
  for (uint c = 0u; c < uint(TOPK); ++c) {
    uint j = tid + c * 256u;
    if (j < M) {
      vals[c] = rv[j];
      idxs[c] = ri[j];
    } else {
      vals[c] = -INFINITY;
      idxs[c] = 0xFFFFFFFFu;
    }
  }

  threadgroup float sg_v[8];
  threadgroup uint  sg_i[8];
  threadgroup uint  win_idx_sh;

  const size_t out_off = size_t(pos) * uint(TOPK);
  for (uint k = 0u; k < uint(TOPK); ++k) {
    float bv = vals[0];
    uint  bi = idxs[0];
    uint  bc = 0u;
    #pragma unroll
    for (uint c = 1u; c < uint(TOPK); ++c) {
      if (vals[c] > bv || (vals[c] == bv && idxs[c] < bi)) {
        bv = vals[c];
        bi = idxs[c];
        bc = c;
      }
    }
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
      win_idx_sh = gi;
      out_vals[out_off + k] = gv;
      out_idxs[out_off + k] = gi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bi == win_idx_sh && bi != 0xFFFFFFFFu) {
      vals[bc] = -INFINITY;
      idxs[bc] = 0xFFFFFFFFu;
    }
  }
"""


@lru_cache(maxsize=8)
def _topk_stage1_kernel(top_k: int):
    return mx.fast.metal_kernel(
        name=f"vmlx_dflash2_topk_s1_k{top_k}",
        input_names=["logits"],
        output_names=["blk_vals", "blk_idxs"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_TOPK_STAGE1_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _topk_stage2_kernel(top_k: int):
    return mx.fast.metal_kernel(
        name=f"vmlx_dflash2_topk_s2_k{top_k}",
        input_names=["blk_vals", "blk_idxs"],
        output_names=["out_vals", "out_idxs"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_TOPK_STAGE2_SRC,
        ensure_row_contiguous=True,
    )


def dflash2_topk(logits: mx.array, top_k: int) -> Optional[Tuple[mx.array, mx.array]]:
    """Exact 2-stage Metal top-K over [1, P, V] float32 logits (returns (vals, idxs) of shape [1, P, K])."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not (1 <= top_k <= 24)
    ):
        return None
    P = int(logits.shape[1])
    V = int(logits.shape[2])
    # Stage 1 holds up to 6 elems/thread across 256 threads * 256 blocks = 393,216 vocab max!
    n_blocks = 256
    if P < 1 or V <= top_k or V > n_blocks * 256 * 6:
        return None
    l2 = logits.reshape(P, V).astype(mx.float32)
    bvals, bidxs = _topk_stage1_kernel(int(top_k))(
        inputs=[l2],
        template=[("TOPK", int(top_k))],
        grid=(n_blocks * 256, P, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(P, n_blocks * int(top_k)), (P, n_blocks * int(top_k))],
        output_dtypes=[mx.float32, mx.uint32],
    )
    vals, idxs = _topk_stage2_kernel(int(top_k))(
        inputs=[bvals, bidxs],
        template=[("TOPK", int(top_k))],
        grid=(256, P, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(P, int(top_k)), (P, int(top_k))],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return vals.reshape(1, P, int(top_k)), idxs.reshape(1, P, int(top_k)).astype(mx.int32)


# ---------------------------------------------------------------------------
# 2. Single-Dispatch Multi-Position Markov Path Selector (Greedy)
#    Runs ALL P positions inside 1 threadgroup (zero inter-step dispatch gap!)
# ---------------------------------------------------------------------------
_MARKOV_SELECT_GREEDY_SRC = r"""
  // 1 TG of 512 threads (16 simdgroups) or 256 threads (8 simdgroups) over K candidates.
  // For K <= 16 and R == 256: 8 simdgroups (256 threads), each sg evaluates up to 2 candidates!
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;

  const uint P = uint(unary_shape[0]);
  const float cap = cap_buf[0];

  threadgroup float ph_sh[256];
  threadgroup float sc_sh[32];
  threadgroup uint  cur_pred_sh;

  if (tid == 0u) {
    cur_pred_sh = uint(anchor[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = 0u; p < P; ++p) {
    const uint pred_id = cur_pred_sh;
    // Cooperatively load pred_cb[pred_id, 0..255] * hidden_proj[p, 0..255] into shared memory!
    const float pv = float(pred_cb[size_t(pred_id) * 256u + tid]);
    const float hv = float(h_proj[size_t(p) * 256u + tid]);
    ph_sh[tid] = pv * hv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load lane's 8 elements of ph_sh into registers
    const uint d0 = lane * 8u;
    float4 ph_lo = float4(ph_sh[d0 + 0u], ph_sh[d0 + 1u], ph_sh[d0 + 2u], ph_sh[d0 + 3u]);
    float4 ph_hi = float4(ph_sh[d0 + 4u], ph_sh[d0 + 5u], ph_sh[d0 + 6u], ph_sh[d0 + 7u]);

    // Each simdgroup evaluates candidates k = sg, sg + 8, ...
    for (uint k = sg; k < uint(TOPK); k += 8u) {
      const uint cid = uint(cands[size_t(p) * uint(TOPK) + k]);
      const device CT* sp = succ_cb + size_t(cid) * 256u + d0;
      float4 s_lo = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
      float4 s_hi = float4(float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7]));
      float edge = simd_sum(dot(ph_lo, s_lo) + dot(ph_hi, s_hi));
      if (lane == 0u) {
        float sc = float(unary[size_t(p) * uint(TOPK) + k]) + edge;
        if (HAS_CAP) {
          sc = cap * metal::tanh(sc / cap);
        }
        sc_sh[k] = sc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      float best_s = sc_sh[0];
      uint  best_k = 0u;
      for (uint k = 1u; k < uint(TOPK); ++k) {
        if (sc_sh[k] > best_s) {
          best_s = sc_sh[k];
          best_k = k;
        }
      }
      uint winner = uint(cands[size_t(p) * uint(TOPK) + best_k]);
      path[p] = int(winner);
      cur_pred_sh = winner;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@lru_cache(maxsize=16)
def _markov_select_greedy_kernel(top_k: int, has_cap: bool, c_dn: str, h_dn: str):
    return mx.fast.metal_kernel(
        name=f"vmlx_dflash2_mkv_sel_k{top_k}_c{int(has_cap)}_{c_dn}_{h_dn}",
        input_names=["unary", "cands", "h_proj", "pred_cb", "succ_cb", "anchor", "cap_buf"],
        output_names=["path"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_MARKOV_SELECT_GREEDY_SRC,
        ensure_row_contiguous=True,
    )


def dflash2_markov_select_greedy(
    unary: mx.array,
    candidates: mx.array,
    hidden_proj: mx.array,
    pred_weight: mx.array,
    succ_weight: mx.array,
    anchor_ids: mx.array,
    cap: float = 0.0,
) -> Optional[mx.array]:
    """Single-dispatch greedy Markov candidate selector over [1, P, K] (R == 256, K <= 32)."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or unary.ndim != 3
        or unary.shape[0] != 1
        or hidden_proj.shape[-1] != 256
        or pred_weight.dtype != succ_weight.dtype
        or pred_weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or hidden_proj.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return None
    P = int(unary.shape[1])
    K = int(unary.shape[2])
    if P < 1 or not (1 <= K <= 32):
        return None
    dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}
    has_cap = bool(cap and float(cap) > 0.0)
    cap_buf = mx.array([float(cap) if has_cap else 1.0], dtype=mx.float32)
    path = _markov_select_greedy_kernel(K, has_cap, dn[pred_weight.dtype], dn[hidden_proj.dtype])(
        inputs=[
            unary.reshape(P, K).astype(mx.float32),
            candidates.reshape(P, K).astype(mx.int32),
            hidden_proj.reshape(P, 256),
            pred_weight,
            succ_weight,
            anchor_ids.reshape(1).astype(mx.int32),
            cap_buf,
        ],
        template=[
            ("CT", pred_weight.dtype),
            ("HT", hidden_proj.dtype),
            ("TOPK", K),
            ("HAS_CAP", has_cap),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(P,)],
        output_dtypes=[mx.int32],
    )[0]
    return path.reshape(1, P)


# ---------------------------------------------------------------------------
# 3. Single-Dispatch Sampled Step Edge Scorer (for temperature > 0)
# ---------------------------------------------------------------------------
_MARKOV_STEP_SCORES_SRC = r"""
  const uint tid  = thread_position_in_threadgroup.x;
  const uint sg   = tid >> 5;
  const uint lane = tid & 31u;
  const float cap = cap_buf[0];

  threadgroup float ph_sh[256];
  const uint pred_id = uint(pred_arr[0]);
  ph_sh[tid] = float(pred_cb[size_t(pred_id) * 256u + tid]) * float(h_row[tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint d0 = lane * 8u;
  float4 ph_lo = float4(ph_sh[d0 + 0u], ph_sh[d0 + 1u], ph_sh[d0 + 2u], ph_sh[d0 + 3u]);
  float4 ph_hi = float4(ph_sh[d0 + 4u], ph_sh[d0 + 5u], ph_sh[d0 + 6u], ph_sh[d0 + 7u]);

  for (uint k = sg; k < uint(TOPK); k += 8u) {
    const uint cid = uint(cand_row[k]);
    const device CT* sp = succ_cb + size_t(cid) * 256u + d0;
    float4 s_lo = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
    float4 s_hi = float4(float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7]));
    float edge = simd_sum(dot(ph_lo, s_lo) + dot(ph_hi, s_hi));
    if (lane == 0u) {
      float sc = float(unary_row[k]) + edge;
      if (HAS_CAP) {
        sc = cap * metal::tanh(sc / cap);
      }
      scores[k] = sc;
    }
  }
"""


@lru_cache(maxsize=16)
def _markov_step_scores_kernel(top_k: int, has_cap: bool, c_dn: str, h_dn: str):
    return mx.fast.metal_kernel(
        name=f"vmlx_dflash2_mkv_step_k{top_k}_c{int(has_cap)}_{c_dn}_{h_dn}",
        input_names=["unary_row", "cand_row", "h_row", "pred_cb", "succ_cb", "pred_arr", "cap_buf"],
        output_names=["scores"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_MARKOV_STEP_SCORES_SRC,
        ensure_row_contiguous=True,
    )


def dflash2_markov_step_scores(
    unary_row: mx.array,
    cand_row: mx.array,
    h_row: mx.array,
    pred_weight: mx.array,
    succ_weight: mx.array,
    pred_id: mx.array,
    cap: float = 0.0,
) -> Optional[mx.array]:
    """Single-dispatch Markov edge scorer + post-correction softcap for 1 position [1, K]."""
    K = int(unary_row.shape[-1])
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or h_row.shape[-1] != 256
        or not (1 <= K <= 32)
        or pred_weight.dtype != succ_weight.dtype
        or pred_weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or h_row.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return None
    dn = {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}
    has_cap = bool(cap and float(cap) > 0.0)
    cap_buf = mx.array([float(cap) if has_cap else 1.0], dtype=mx.float32)
    sc = _markov_step_scores_kernel(K, has_cap, dn[pred_weight.dtype], dn[h_row.dtype])(
        inputs=[
            unary_row.reshape(K).astype(mx.float32),
            cand_row.reshape(K).astype(mx.int32),
            h_row.reshape(256),
            pred_weight,
            succ_weight,
            pred_id.reshape(1).astype(mx.int32),
            cap_buf,
        ],
        template=[
            ("CT", pred_weight.dtype),
            ("HT", h_row.dtype),
            ("TOPK", K),
            ("HAS_CAP", has_cap),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K,)],
        output_dtypes=[mx.float32],
    )[0]
    return sc.reshape(1, K)

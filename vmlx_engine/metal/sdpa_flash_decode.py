# SPDX-License-Identifier: Apache-2.0
"""Decode / verify attention for a few query rows over a long KV cache, split over key ranges.

Used by MiMo-V2.6 full attention (64 query heads over 4 KV heads, q/k 192, v 128) for 1..8 query rows, and by the
long-context Qwen3.5 decode / verify route (``qwen35_flash_decode``).

Why: MLX's vector SDPA scores every (query row, head, key) with a 32-lane reduction and runs the online softmax on
all 32 lanes. With 16 query heads per KV head that is ALU-bound well below DRAM bandwidth for one row, and each extra
row costs as much again. Here all query heads of one KV head (times the query rows) are scored together with 8x8
fp32 simdgroup matrix products, so one pass over K and V serves every head that reads them.

Layout of the work
------------------
* "qrows" of a KV head: ``row * GQA + head_in_group``; 8 consecutive qrows form a band (one 8-row fragment), and a
  band never straddles two query rows because GQA is a multiple of 8.
* Pass 1: one threadgroup per (KV head, batch, key split). Each split is a contiguous chunk of ``chunk_size(S)``
  keys; the threadgroup stages TILE-key slices of K and V in threadgroup memory and every band runs an fp32 online
  softmax over them. One-row calls give each band two simdgroups (one per half of the head dims); multi-row calls
  give it one simdgroup that keeps the two halves in separate accumulators. Either way a score is
  ``(first-half dot product) + (second-half dot product)``.
* Pass 2: merge the splits per output element, in split order (the two-pass recipe of MLX's ``sdpa_vector_2pass``).

Numerics: q is scaled in fp32 (``scale * float(q)``), K/V widen exactly from bf16/fp16, every product and sum is fp32
and the per-split partials stay fp32. A qrow's sequence of operations depends only on the key count (through the
chunk size), never on the number of rows in the call, so a verify row equals the one-row decode at the same
position bit for bit unless the two key counts straddle a chunk-size step.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx

MAX_ROWS = 8
TILE_KEYS = 8

_SPLIT_PASS = r"""
  // ---- geometry ----
  constexpr int FRAG = 8;
  constexpr int BANDS = (GQA * ROWS) / FRAG;      // 8-qrow bands per KV head
  constexpr int QK_BLOCKS = QK_DIM / FRAG;
  constexpr int HALF_BLOCKS = QK_BLOCKS / 2;      // head-dim blocks per half
  constexpr int V_BLOCKS = V_DIM / FRAG / HALVES; // value blocks owned by this simdgroup
  constexpr int Q_BLOCKS = QK_BLOCKS / HALVES;    // query blocks held by this simdgroup
  constexpr int K_PITCH = QK_DIM + 8;             // padded rows of the staged tiles
  constexpr int V_PITCH = V_DIM + 8;
  constexpr int THREADS = BANDS * HALVES * 32;
  constexpr int K_CHUNKS = TILE * (QK_DIM / 8);    // 8-element chunks per staged tile
  constexpr int V_CHUNKS = TILE * (V_DIM / 8);
  constexpr int MY_CHUNKS = (K_CHUNKS + V_CHUNKS + THREADS - 1) / THREADS;
  constexpr int FRAGS = TILE / FRAG;
  typedef simdgroup_matrix<float, 8, 8> facc;
  typedef simdgroup_matrix<T, 8, 8> fin;
  // The two fp32 elements of an 8x8 fragment held by this lane.
  auto elems = [](thread facc& m) -> thread vec<float, 2>& {
    return reinterpret_cast<thread vec<float, 2>&>(m.thread_elements());
  };

  threadgroup T k_tile[TILE * K_PITCH];
  threadgroup T v_tile[TILE * V_PITCH];
  threadgroup float half_scores[HALVES > 1 ? BANDS * 2 * FRAGS * 64 : 1];

  const int kv_head = threadgroup_position_in_grid.x;
  const int batch = threadgroup_position_in_grid.y;
  const int split = threadgroup_position_in_grid.z;
  const int n_splits = threadgroups_per_grid.z;
  const int band = simdgroup_index_in_threadgroup % BANDS;
  const int half_id = simdgroup_index_in_threadgroup / BANDS;
  const int lane = thread_index_in_simdgroup;
  const int tid = thread_index_in_threadgroup;
  const int n_keys = keys_shape[2];
  const int key_begin = split * params[0];
  const int key_end = min(n_keys, key_begin + params[0]);
  const int qrow0 = band * FRAG;
  const int row = qrow0 / GQA;                    // query row of this band
  const int head0 = kv_head * GQA + qrow0 % GQA;  // first query head of this band
  const int n_heads = KV_HEADS * GQA;
  // MLX's 8x8 fragment layout: this lane holds row frow, columns fcol and fcol + 1.
  const int frow = ((lane / 4) & 4) + ((lane / 2) % 4);
  const int fcol = ((lane / 4) & 2) * 2 + (lane % 2) * 2;

  // ---- query fragments, pre-scaled in fp32 ----
  facc q_frag[Q_BLOCKS];
  {
    const float s = scale[0];
    const device T* qp = queries + batch * queries_strides[0] + (head0 + frow) * queries_strides[1] +
        row * queries_strides[2];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < Q_BLOCKS; i++) {
      const int d = (half_id * Q_BLOCKS + i) * FRAG + fcol;
      thread auto& e = elems(q_frag[i]);
      e[0] = s * float(qp[d * queries_strides[3]]);
      e[1] = s * float(qp[(d + 1) * queries_strides[3]]);
    }
  }

  // ---- running softmax state of row frow ----
  float run_max = Limits<float>::finite_min;
  float run_sum = 0;
  if (HAS_SINKS && split == 0) {
    run_max = float(sinks[head0 + frow]);
    run_sum = 1;
  }
  facc acc[V_BLOCKS];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < V_BLOCKS; i++) {
    acc[i] = facc(0);
  }

  // ---- staging plan: this thread's chunks of each tile ----
  const int64_t k_row = keys_strides[2], v_row = values_strides[2];
  const int64_t k_el = keys_strides[3], v_el = values_strides[3];
  const device T* k_src = keys + batch * keys_strides[0] + kv_head * keys_strides[1];
  const device T* v_src = values + batch * values_strides[0] + kv_head * values_strides[1];
  const bool wide_loads = k_el == 1 && v_el == 1 && k_row % 8 == 0 && v_row % 8 == 0 &&
      keys_strides[1] % 8 == 0 && values_strides[1] % 8 == 0 && keys_strides[0] % 8 == 0 &&
      values_strides[0] % 8 == 0;
  int chunk_row[MY_CHUNKS];         // key row within the tile, TILE when this slot is unused
  const device T* chunk_src[MY_CHUNKS];  // the chunk's source in the current tile (advanced one tile per step)
  int64_t chunk_step[MY_CHUNKS];
  threadgroup T* chunk_dst[MY_CHUNKS];
  bool chunk_is_k[MY_CHUNKS];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < MY_CHUNKS; j++) {
    const int c = tid + j * THREADS;
    const bool is_k = c < K_CHUNKS;
    const int cc = is_k ? c : c - K_CHUNKS;
    const int per_row = (is_k ? QK_DIM : V_DIM) / 8;
    const int r = cc / per_row;
    const int col = (cc % per_row) * 8;
    chunk_is_k[j] = is_k;
    chunk_row[j] = c < K_CHUNKS + V_CHUNKS ? r : TILE;
    chunk_dst[j] = is_k ? k_tile + r * K_PITCH + col : v_tile + r * V_PITCH + col;
    chunk_src[j] = is_k ? k_src + (key_begin + r) * k_row + col * k_el : v_src + (key_begin + r) * v_row + col * v_el;
    chunk_step[j] = TILE * (is_k ? k_row : v_row);
  }

  const int last_visible = n_keys - ROWS + row;   // causal: last key this row may see

  for (int t0 = key_begin; t0 < key_end; t0 += TILE) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool whole_tile = t0 + TILE <= key_end;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MY_CHUNKS; j++) {
      if (chunk_row[j] >= TILE) {
        continue;
      }
      const bool live = whole_tile || t0 + chunk_row[j] < key_end;
      const device T* src = chunk_src[j];
      chunk_src[j] += chunk_step[j];
      if (wide_loads) {
        *reinterpret_cast<threadgroup vec<T, 8>*>(chunk_dst[j]) =
            live ? *reinterpret_cast<const device vec<T, 8>*>(src) : vec<T, 8>(0);
      } else {
        const int64_t el = chunk_is_k[j] ? k_el : v_el;
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 8; e++) {
          chunk_dst[j][e] = live ? src[e * el] : T(0);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: S = q' K^T, summed as (first half of the head dims) + (second half).
    facc score[FRAGS];
    if (HALVES == 1) {
      facc upper[FRAGS];
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < FRAGS; f++) {
        score[f] = facc(0);
        upper[f] = facc(0);
      }
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < HALF_BLOCKS; i++) {
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < FRAGS; f++) {
          fin kt;
          simdgroup_load(kt, k_tile + f * 8 * K_PITCH + i * 8, K_PITCH, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(score[f], q_frag[i], kt, score[f]);
          fin kt_hi;
          simdgroup_load(kt_hi, k_tile + f * 8 * K_PITCH + (HALF_BLOCKS + i) * 8, K_PITCH, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(upper[f], q_frag[HALF_BLOCKS + i], kt_hi, upper[f]);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < FRAGS; f++) {
        elems(score[f]) += elems(upper[f]);
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < FRAGS; f++) {
        score[f] = facc(0);
      }
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < HALF_BLOCKS; i++) {
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < FRAGS; f++) {
          fin kt;
          simdgroup_load(kt, k_tile + f * 8 * K_PITCH + (half_id * HALF_BLOCKS + i) * 8, K_PITCH,
                         ulong2(0, 0), true);
          simdgroup_multiply_accumulate(score[f], q_frag[i], kt, score[f]);
        }
      }
      // Swap halves through threadgroup memory; both simdgroups then hold lower + upper.
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < FRAGS; f++) {
        simdgroup_store(score[f], half_scores + ((band * 2 + half_id) * FRAGS + f) * 64, 8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < FRAGS; f++) {
        const int at = frow * 8 + fcol;
        vec<float, 2> lower = *reinterpret_cast<const threadgroup vec<float, 2>*>(
            half_scores + ((band * 2) * FRAGS + f) * 64 + at);
        lower += *reinterpret_cast<const threadgroup vec<float, 2>*>(
            half_scores + ((band * 2 + 1) * FRAGS + f) * 64 + at);
        elems(score[f]) = lower;
      }
    }

    // Masking (MLX's rules) and the tile max of row frow.
    float tile_max = Limits<float>::finite_min;
    const bool unmasked = MASK_KIND == 0 && whole_tile && (!CAUSAL || t0 + TILE - 1 <= last_visible);
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FRAGS; f++) {
      thread auto& e = elems(score[f]);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 2; j++) {
        float s = e[j];
        if (!unmasked) {
          const int key = t0 + f * 8 + fcol + j;
          bool keep = key < key_end;
          const size_t at = (size_t(batch) * ROWS + row) * n_keys + key;
          if (CAUSAL) {
            keep = keep && key <= last_visible;
          } else if (MASK_KIND == 1) {
            keep = keep && bool(mask[at]);
          } else if (MASK_KIND == 2) {
            keep = keep && mask[at] >= Limits<T>::finite_min;
          }
          if (MASK_KIND == 2 && keep) {
            s += mask[at];
          }
          s = keep ? s : -INFINITY;
          e[j] = s;
        }
        tile_max = max(tile_max, s);
      }
    }
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));

    // Online softmax update.
    const float new_max = max(run_max, tile_max);
    const float rescale = fast::exp(run_max - new_max);
    float tile_sum = 0;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FRAGS; f++) {
      thread auto& e = elems(score[f]);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 2; j++) {
        const float p = fast::exp(e[j] - new_max);
        e[j] = p;
        tile_sum += p;
      }
    }
    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 8);
    const bool any_moved = simd_any(new_max != run_max);  // rescaling by exactly 1 is skippable
    run_sum = run_sum * rescale + tile_sum;
    run_max = new_max;
    if (any_moved) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < V_BLOCKS; i++) {
        elems(acc[i]) *= rescale;
      }
    }

    // acc += P V.
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < FRAGS; f++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < V_BLOCKS; i++) {
        fin vt;
        simdgroup_load(vt, v_tile + f * 8 * V_PITCH + (half_id * V_BLOCKS + i) * 8, V_PITCH);
        simdgroup_multiply_accumulate(acc[i], score[f], vt, acc[i]);
      }
    }
  }

  // ---- partials of this split: unnormalized output, max, sum ----
  const int out_row0 = (batch * ROWS + row) * n_heads + head0;
  device float* op = o_part + (size_t(out_row0) * n_splits + split) * V_DIM + half_id * V_BLOCKS * 8;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < V_BLOCKS; i++) {
    simdgroup_store(acc[i], op + i * 8, ulong(n_splits) * V_DIM);
  }
  if (half_id == 0 && fcol == 0) {
    const size_t at = size_t(out_row0 + frow) * n_splits + split;
    m_part[at] = run_max;
    l_part[at] = run_sum;
  }
"""

_MERGE_PASS = r"""
  const int out_row = threadgroup_position_in_grid.x;  // (batch * ROWS + row) * H + head
  const int d = thread_position_in_threadgroup.x;
  const int n_splits = m_part_shape[1];
  const device float* mp = m_part + size_t(out_row) * n_splits;
  const device float* lp = l_part + size_t(out_row) * n_splits;
  const device float* op = o_part + size_t(out_row) * n_splits * V_DIM + d;
  float top = Limits<float>::finite_min;
  for (int s = 0; s < n_splits; s++) {
    top = max(top, mp[s]);
  }
  float total = 0;
  float acc = 0;
  for (int s = 0; s < n_splits; s++) {
    const float w = fast::exp(mp[s] - top);
    total += w * lp[s];
    acc += w * op[s * V_DIM];
  }
  acc = total == 0 ? acc : (acc / total);
  out[size_t(out_row) * V_DIM + d] = static_cast<T>(acc);
"""


@lru_cache(maxsize=None)
def _split_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_split_key_attention",
        input_names=["queries", "keys", "values", "scale", "params", "mask", "sinks"],
        output_names=["o_part", "m_part", "l_part"],
        source=_SPLIT_PASS,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="vmlx_split_key_merge",
        input_names=["o_part", "m_part", "l_part"],
        output_names=["out"],
        source=_MERGE_PASS,
    )


def chunk_size(n_keys: int) -> int:
    """Keys per split: ``n_keys / 128`` rounded up to a power of two, clamped to [256, 1024]. The same for every row
    count, so a verify row matches the one-row decode at its position unless the key counts straddle a step."""
    c = 1 << max(0, math.ceil(math.log2(max(1, n_keys) / 128)))
    return max(256, min(1024, c))


@lru_cache(maxsize=None)
def _int_scalar(value: int):
    return mx.array([value], dtype=mx.int32)


@lru_cache(maxsize=None)
def _float_scalar(value: float):
    return mx.array([value], dtype=mx.float32)


_PLACEHOLDERS: dict = {}


def _placeholder(dtype):
    arr = _PLACEHOLDERS.get(dtype)
    if arr is None:
        arr = _PLACEHOLDERS[dtype] = mx.zeros((1,), dtype=dtype)
    return arr


def _mask_operand(mask, B: int, L: int, S: int, dtype):
    """(kind, array): kind 0 none, 1 bool, 2 additive (in q's dtype); array laid out (B, L, S). None if unusable."""
    if mask is None or isinstance(mask, str):
        return 0, _placeholder(mx.bool_)
    if not isinstance(mask, mx.array) or not 1 <= mask.ndim <= 4:
        return None
    m = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
    if m.shape[1] != 1:
        return None
    try:
        m = mx.broadcast_to(m, (B, 1, L, S))
    except ValueError:
        return None
    kind = 1 if m.dtype == mx.bool_ else 2
    if kind == 2:
        m = m.astype(dtype)
    return kind, mx.contiguous(m.reshape(B, L, S))


def sdpa_flash(q, k, v, scale: float, mask, sinks, tile: int = TILE_KEYS) -> Optional[mx.array]:
    """Attention of ``q (B, H, L, D)`` over ``k (B, Hk, S, D)`` / ``v (B, Hk, S, Dv)`` for ``L <= MAX_ROWS`` rows,
    returned as ``(B, L, H * Dv)``.

    ``mask`` is ``None``, ``"causal"`` or a bool / additive array broadcastable to ``(B, 1, L, S)``; ``sinks`` per
    query head, like MLX's SDPA. Returns ``None`` when the call is outside this kernel's contract.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    B, H, L, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    Dv = v.shape[3]
    if not 1 <= L <= MAX_ROWS or k.shape[0] != B or v.shape[0] != B:
        return None
    if v.shape[1] != Hk or v.shape[2] != S or k.shape[3] != D or Hk == 0 or H % Hk:
        return None
    if D % 16 or Dv % 16 or D > 256 or Dv > 256 or L > S:
        return None
    dtype = q.dtype
    if dtype not in (mx.bfloat16, mx.float16) or k.dtype != dtype or v.dtype != dtype:
        return None
    gqa = H // Hk
    if gqa % 8:
        return None
    causal = isinstance(mask, str)
    if causal and mask != "causal":
        return None
    operand = _mask_operand(mask, B, L, S, dtype)
    if operand is None:
        return None
    mask_kind, mask_arr = operand
    if sinks is not None and (sinks.ndim != 1 or sinks.shape[0] != H):
        return None
    sinks_arr = _placeholder(dtype) if sinks is None else sinks.astype(dtype)

    chunk = chunk_size(S)
    splits = -(-S // chunk)
    bands = gqa * L // 8
    halves = 2 if L == 1 else 1     # one-row calls: two simdgroups per band (more threads per key tile)
    o_part, m_part, l_part = _split_kernel()(
        inputs=[q, k, v, _float_scalar(float(scale)), _int_scalar(int(chunk)), mask_arr, sinks_arr],
        template=[
            ("T", dtype),
            ("QK_DIM", int(D)),
            ("V_DIM", int(Dv)),
            ("GQA", int(gqa)),
            ("KV_HEADS", int(Hk)),
            ("ROWS", int(L)),
            ("TILE", int(tile)),
            ("HALVES", halves),
            ("CAUSAL", bool(causal and L > 1)),
            ("MASK_KIND", int(mask_kind)),
            ("HAS_SINKS", sinks is not None),
        ],
        grid=(32 * bands * halves * Hk, B, splits),
        threadgroup=(32 * bands * halves, 1, 1),
        output_shapes=[(B * L * H, splits, Dv), (B * L * H, splits), (B * L * H, splits)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _merge_kernel()(
        inputs=[o_part, m_part, l_part],
        template=[("T", dtype), ("V_DIM", int(Dv))],
        grid=(Dv * B * L * H, 1, 1),
        threadgroup=(Dv, 1, 1),
        output_shapes=[(B, L, H * Dv)],
        output_dtypes=[dtype],
    )
    return out

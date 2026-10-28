# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA (block-sparse attention): the selection and the attention paths that consume it.

The Lightning indexer picks, for every query row and KV group, ``keep`` (16) key blocks of ``block`` (128) keys:
the top block scores, the query's own (local) block forced in. Attention then sees only those blocks, causally.

``MSASelection`` carries that pick as block ids ([B, G, Sq, keep]) plus the positions it was made at. Paths
(``choose_path`` routes by measured crossovers; ``VMLX_MINIMAX_MSA`` forces one for comparisons):

* ``dense``: the additive [B, G, Sq, Sk] float32 key mask the runtime used before (``dense_bias``, same ops) and
  one SDPA over all Sk keys. Kept as the reference / fallback path (and, for now, the prefill path).
* ``decode``: decode-sized calls (decode steps, DSpark verify rows). A Metal kernel splits each (row, group)'s
  keep * block selected keys into 32-key chunks across threadgroups and merges the partial softmaxes; keys are read
  in place from the cache. Rows are independent: a row's output is the same alone or with other rows.
* ``gather``: pure MLX ops (gather the selected keys, SDPA over them). Not routed to; kept as an op-level reference.

Positions: query row r sits at ``offset + r``; key j is allowed when its block is selected and ``j <= offset + r``.
Keys at or past ``sk`` (the logical extent, i.e. the cache offset) are never read: the decode kernel stops at the
row's position (< sk) and the gather clamps to ``sk - 1`` and masks the clamped lanes. A partial last block, a
query's own block (causal inside it) and selected blocks that lie entirely in the future (possible when fewer than
``keep`` blocks precede a prefill row) are all handled by that one positional rule.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import mlx.core as mx

# Path override for comparisons and diagnostics: "auto" (default) routes by the measured crossovers below,
# "dense" forces the [B, G, Sq, Sk] mask path, "sparse" the decode kernel for decode-sized calls, "gather" /
# "decode" that path wherever its shapes are supported.
ENV_PATH = "VMLX_MINIMAX_MSA"

# Metal element type of each activation dtype the kernels take.
_ELEM = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}


class MSASelection:
    """Top-k key blocks per (query row, KV group) for queries at ``offset`` .. ``offset + Sq - 1`` over keys
    ``0 .. sk - 1``. ``top_idx`` [B, G, Sq, keep] holds distinct block ids per row (argpartition order)."""

    __slots__ = ("top_idx", "offset", "sk", "block")

    def __init__(self, top_idx: mx.array, offset: int, sk: int, block: int):
        self.top_idx = top_idx
        self.offset = int(offset)
        self.sk = int(sk)
        self.block = int(block)

    @property
    def shape(self):
        """[B, G, Sq, Sk]: the shape of the equivalent key mask."""
        B, G, Sq, _ = self.top_idx.shape
        return (B, G, Sq, self.sk)

    @property
    def n_blocks(self) -> int:
        return -(-self.sk // self.block)

    @property
    def keep(self) -> int:
        return int(self.top_idx.shape[-1])

    def dense_bias(self) -> mx.array:
        """The additive key mask [B, G, Sq, Sk] (float32 0 / -inf, causal) -- the pre-block-sparse runtime's ops."""
        B, G, Sq, _ = self.top_idx.shape
        q_pos = mx.arange(self.offset, self.offset + Sq).reshape(1, 1, Sq, 1)
        k_pos = mx.arange(self.sk).reshape(1, 1, 1, self.sk)
        block_keep = mx.full((B, G, Sq, self.n_blocks), -mx.inf, dtype=mx.float32)
        block_keep = mx.put_along_axis(block_keep, self.top_idx, mx.array(0.0, dtype=mx.float32), axis=-1)
        key_bias = mx.repeat(block_keep, self.block, axis=-1)[..., : self.sk]
        return mx.where(k_pos > q_pos, -mx.inf, key_bias)


def _kname(prefix: str, source: str) -> str:
    """Kernel name carrying a hash of its full source: MLX caches compiled kernels by name."""
    return f"{prefix}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _metal() -> bool:
    """The custom kernels run on the Metal GPU only (CPU default device: the dense path)."""
    try:
        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


def requested_path() -> str:
    raw = os.environ.get(ENV_PATH, "").strip().lower()
    return raw if raw in ("dense", "sparse", "gather", "decode") else "auto"


def _key_positions(sel: MSASelection):
    """(source index [B, G, Sq, N] clamped below sk, allowed [B, G, Sq, N]) for the N = keep * block selected keys of
    every row, blocks in ascending order (keys in position order, as the dense path visits them)."""
    B, G, Sq, keep = sel.top_idx.shape
    blocks = mx.sort(sel.top_idx.astype(mx.int32), axis=-1)
    pos = blocks[..., None] * sel.block + mx.arange(sel.block, dtype=mx.int32)
    pos = pos.reshape(B, G, Sq, keep * sel.block)
    q_pos = mx.arange(sel.offset, sel.offset + Sq, dtype=mx.int32).reshape(1, 1, Sq, 1)
    allowed = pos <= q_pos
    return mx.minimum(pos, sel.sk - 1), allowed


def _check_extent(k: mx.array, v: mx.array, sel: MSASelection) -> None:
    if int(k.shape[2]) < sel.sk or int(v.shape[2]) < sel.sk:
        # The selection was made over more keys than the attention sees: the index lane and the K / V lane have
        # desynchronised. Reading on would gather rows that were never written.
        raise ValueError(
            f"MSA selection covers {sel.sk} keys but K / V hold {k.shape[2]} / {v.shape[2]}")
    if sel.offset + int(sel.top_idx.shape[2]) > sel.sk:
        raise ValueError(
            f"MSA query rows end at {sel.offset + int(sel.top_idx.shape[2])} past the {sel.sk} selected keys")


def gathered_attention(q: mx.array, k: mx.array, v: mx.array, sel: MSASelection, scale: float) -> mx.array:
    """Attention [B, H, L, D] of q over only the selected key blocks of each (row, KV group).

    q [B, H, L, D] (query heads contiguous per KV group); k / v [B, G, >= sk, D]. Every row is one SDPA batch
    element over its own keep * block gathered keys, so row r's output is the same whether it runs alone (a decode
    step) or with other rows (a verify)."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    Hg = H // G
    _check_extent(k, v, sel)
    src, allowed = _key_positions(sel)                                  # [B, G, L, N]
    N = int(src.shape[-1])
    bias = mx.where(allowed, mx.array(0.0, dtype=q.dtype), mx.array(-mx.inf, dtype=q.dtype))
    if L == 1:
        idx = src.reshape(B, G, N, 1)
        ks = mx.take_along_axis(k, idx, axis=2)                         # [B, G, N, D]
        vs = mx.take_along_axis(v, idx, axis=2)
        o = mx.fast.scaled_dot_product_attention(
            q.reshape(B * G, Hg, 1, D), ks.reshape(B * G, 1, N, D), vs.reshape(B * G, 1, N, D),
            scale=scale, mask=bias.reshape(B * G, 1, 1, N))
        return o.reshape(B, H, 1, D)
    idx = src[..., None]                                                # [B, G, L, N, 1]
    ks = mx.take_along_axis(k[:, :, None], idx, axis=3)                 # [B, G, L, N, D]
    vs = mx.take_along_axis(v[:, :, None], idx, axis=3)
    qr = q.reshape(B, G, Hg, L, D).transpose(0, 1, 3, 2, 4).reshape(B * G * L, Hg, 1, D)
    o = mx.fast.scaled_dot_product_attention(
        qr, ks.reshape(B * G * L, 1, N, D), vs.reshape(B * G * L, 1, N, D),
        scale=scale, mask=bias.reshape(B * G * L, 1, 1, N))
    return o.reshape(B, G, L, Hg, D).transpose(0, 1, 3, 2, 4).reshape(B, H, L, D)


# --------------------------------------------------------------------------------------------- decode kernel
# Few query rows (decode steps, verify rows): latency matters, so the keep * block selected keys of every
# (row, KV group) are split into 32-key chunks, one threadgroup per chunk, and the group's query heads into slices of
# HS heads, one simdgroup per slice. Lane j owns key j of the chunk: it scores that key against the slice's heads
# (queries pre-scaled, in threadgroup memory), the simdgroup takes each head's max / sum over the chunk, then lane j
# accumulates P V for head dims [j * DPL, (j + 1) * DPL). Partials (max, sum, output) per chunk and head go to a
# second launch that merges a row's chunks (log-sum-exp). All fp32. Keys past the row's position are masked and never
# read (so nothing at or past sk is read). Every threadgroup touches one row only: a row's result does not depend on
# how many rows share the launch.
_DECODE_PARTIAL_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);            // head slice
  const uint tid = thread_position_in_threadgroup.x;
  const int chunk = int(threadgroup_position_in_grid.x);         // 32-key chunk of the row's selection
  const int rg = int(threadgroup_position_in_grid.y);            // (b * G + g) * L + l
  const int G = int(blocks_shape[1]);
  const int L = int(blocks_shape[2]);
  const int H = int(q_shape[2]);
  const int l = rg % L;
  const int bgi = rg / L;
  const int b = bgi / G;
  const int g = bgi - b * G;
  const int q_pos = meta[0] + l;
  const float qscale = scale[0] * 1.4426950408889634f;
  const float NEG = -1.0e30f;
  constexpr int CPB = BLOCK / 32;                                // chunks per block

  const int start = int(blocks[(size_t)rg * KEEP + chunk / CPB]) * BLOCK + (chunk % CPB) * 32;
  const size_t pbase = ((size_t)rg * NCHUNK + chunk) * HG + sg * HS;
  if (start > q_pos) {                                           // whole chunk in the future: empty partial
    for (int h = 0; h < HS; ++h) {
      for (int e = 0; e < DPL; ++e) po[(pbase + h) * D + lane * DPL + e] = 0.0f;
      if (lane == 0) { pm[pbase + h] = NEG; pl[pbase + h] = 0.0f; }
    }
    return;
  }
  threadgroup float qs[HG * D];
  threadgroup float ps[HG * 32];
  device const elem_t* qrow = q + ((size_t)(b * L + l) * H + g * HG) * D;
  for (int i = int(tid); i < HG * D; i += (HG / HS) * 32) qs[i] = float(qrow[i]) * qscale;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const elem_t* kb = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
  device const elem_t* vb = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
  const size_t kss = k_strides[2], vss = v_strides[2], kds = k_strides[3], vds = v_strides[3];
  threadgroup const float* qh = qs + sg * HS * D;
  threadgroup float* pp = ps + sg * HS * 32;

  const int kp = start + int(lane);
  const bool valid = kp <= q_pos;                                // q_pos < sk, so valid keys are inside the extent
  float s[HS];
  #pragma unroll
  for (int h = 0; h < HS; ++h) s[h] = 0.0f;
  // 4-wide loads when the head dim is contiguous and rows stay 4-element aligned (always true for a cache view).
  const bool vec_ok = kds == 1 && vds == 1 && (kss % 4) == 0 && (vss % 4) == 0
      && ((ulong)kb % (4 * sizeof(elem_t))) == 0 && ((ulong)vb % (4 * sizeof(elem_t))) == 0;
  if (valid) {
    device const elem_t* krow = kb + (size_t)kp * kss;
    if (vec_ok) {
      for (int d = 0; d < D; d += 4) {
        const vec<elem_t, 4> k4 = *((device const vec<elem_t, 4>*)(krow + d));
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float kd = float(k4[e]);
          #pragma unroll
          for (int h = 0; h < HS; ++h) s[h] += qh[h * D + d + e] * kd;
        }
      }
    } else {
      for (int d = 0; d < D; ++d) {
        const float kd = float(krow[(size_t)d * kds]);
        #pragma unroll
        for (int h = 0; h < HS; ++h) s[h] += qh[h * D + d] * kd;
      }
    }
  }
  #pragma unroll
  for (int h = 0; h < HS; ++h) {
    const float x = valid ? s[h] : NEG;
    const float mh = simd_max(x);
    const float p = valid ? fast::exp2(x - mh) : 0.0f;
    const float lh = simd_sum(p);
    pp[h * 32 + lane] = p;
    if (lane == 0) {
      pm[pbase + h] = mh;
      pl[pbase + h] = lh;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float o[HS][DPL];
  #pragma unroll
  for (int h = 0; h < HS; ++h) {
    #pragma unroll
    for (int e = 0; e < DPL; ++e) o[h][e] = 0.0f;
  }
  const int nvalid = min(32, q_pos - start + 1);
  for (int j = 0; j < nvalid; ++j) {
    device const elem_t* vrow = vb + (size_t)(start + j) * vss + (size_t)(lane * DPL) * vds;
    float vv[DPL];
    if (vec_ok && DPL % 4 == 0) {
      #pragma unroll
      for (int e = 0; e < DPL; e += 4) {
        const vec<elem_t, 4> v4 = *((device const vec<elem_t, 4>*)(vrow + e));
        vv[e] = float(v4[0]); vv[e + 1] = float(v4[1]); vv[e + 2] = float(v4[2]); vv[e + 3] = float(v4[3]);
      }
    } else {
      #pragma unroll
      for (int e = 0; e < DPL; ++e) vv[e] = float(vrow[(size_t)e * vds]);
    }
    #pragma unroll
    for (int h = 0; h < HS; ++h) {
      const float p = pp[h * 32 + j];
      #pragma unroll
      for (int e = 0; e < DPL; ++e) o[h][e] += p * vv[e];
    }
  }
  #pragma unroll
  for (int h = 0; h < HS; ++h) {
    #pragma unroll
    for (int e = 0; e < DPL; ++e) po[(pbase + h) * D + lane * DPL + e] = o[h][e];
  }
"""

_DECODE_COMBINE_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int h = int(threadgroup_position_in_grid.y);
  const int rg = int(threadgroup_position_in_grid.z);
  const int G = int(blocks_shape[1]);
  const int L = int(blocks_shape[2]);
  const int H = G * HG;
  const int l = rg % L;
  const int bgi = rg / L;
  const int b = bgi / G;
  const int g = bgi - b * G;
  float mmax = -1.0e30f;
  for (int c = 0; c < NCHUNK; ++c) mmax = max(mmax, pm[((size_t)rg * NCHUNK + c) * HG + h]);
  float lsum = 0.0f;
  float o[DPL];
  for (int e = 0; e < DPL; ++e) o[e] = 0.0f;
  for (int c = 0; c < NCHUNK; ++c) {
    const size_t pidx = ((size_t)rg * NCHUNK + c) * HG + h;
    const float w = fast::exp2(pm[pidx] - mmax);
    lsum += w * pl[pidx];
    for (int e = 0; e < DPL; ++e) o[e] += w * po[pidx * D + lane * DPL + e];
  }
  const float inv = 1.0f / lsum;
  device elem_t* orow = out + ((size_t)(b * L + l) * H + g * HG + h) * D + lane * DPL;
  for (int e = 0; e < DPL; ++e) orow[e] = elem_t(o[e] * inv);
"""

_DECODE_KERNELS: dict = {}


def _decode_kernels(elem: str, hg: int, d: int, keep: int, block: int):
    key = (elem, hg, d, keep, block)
    kerns = _DECODE_KERNELS.get(key)
    if kerns is None:
        hs = _decode_head_slice(hg)
        nchunk = keep * (block // 32)
        consts = {"HG": hg, "HS": hs, "D": d, "DPL": d // 32, "KEEP": keep, "BLOCK": block, "NCHUNK": nchunk}
        head = f"  using elem_t = {elem};\n" + "".join(f"  constexpr int {n} = {int(v)};\n" for n, v in consts.items())
        tag = f"{elem}_h{hg}_d{d}_k{keep}_b{block}"
        partial = mx.fast.metal_kernel(
            name=_kname(f"vmlx_minimax_msa_decode_partial_{tag}", head + _DECODE_PARTIAL_SOURCE),
            input_names=["q", "k", "v", "blocks", "meta", "scale"], output_names=["po", "pm", "pl"],
            source=head + _DECODE_PARTIAL_SOURCE, ensure_row_contiguous=False)
        combine = mx.fast.metal_kernel(
            name=_kname(f"vmlx_minimax_msa_decode_combine_{tag}", head + _DECODE_COMBINE_SOURCE),
            input_names=["po", "pm", "pl", "blocks"], output_names=["out"], source=head + _DECODE_COMBINE_SOURCE)
        kerns = _DECODE_KERNELS[key] = (partial, combine, nchunk, hg // hs)
    return kerns


def _decode_head_slice(hg: int) -> int:
    """Query heads per simdgroup in the decode kernel (4, or all of them when the group has fewer / not a multiple)."""
    return 4 if hg % 4 == 0 else hg


def decode_kernel_supported(q: mx.array, k: mx.array, sel: MSASelection) -> bool:
    """Head dim a multiple of 32 (lane j owns D / 32 output dims), at most 32 query heads per KV group, blocks a
    multiple of the 32-key chunk, and the group's queries plus the chunk probabilities in threadgroup memory."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    if (not _metal() or q.dtype not in _ELEM or k.dtype != q.dtype or H % G or D % 32 or sel.block % 32
            or sel.block > 1024):
        return False
    hg = H // G
    return hg <= 32 and 4 * (hg * D + hg * 32) <= 32768


def decode_kernel_attention(q: mx.array, k: mx.array, v: mx.array, sel: MSASelection, scale: float) -> mx.array:
    """Attention [B, H, L, D] over each (row, KV group)'s selected blocks for decode-sized calls (split keys + merge)."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    hg = H // G
    _check_extent(k, v, sel)
    # Chunks may come in any block order (the merge does not depend on it), so the selection is used as picked.
    # (argpartition's [..., :keep] is a strided view; the kernel indexes it densely.)
    blocks = mx.contiguous(sel.top_idx)
    qc = mx.contiguous(q.transpose(0, 2, 1, 3))                     # [B, L, H, D]
    meta = mx.array([sel.offset, sel.sk], dtype=mx.int32)
    partial, combine, nchunk, n_slices = _decode_kernels(_ELEM[q.dtype], hg, D, sel.keep, sel.block)
    rows = B * G * L
    po, pm, pl = partial(
        inputs=[qc, k, v, blocks, meta, mx.array([scale], dtype=mx.float32)],
        grid=(32 * n_slices * nchunk, rows, 1), threadgroup=(32 * n_slices, 1, 1),
        output_shapes=[(rows, nchunk, hg, D), (rows, nchunk, hg), (rows, nchunk, hg)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    out = combine(
        inputs=[po, pm, pl, blocks], grid=(32, hg, rows), threadgroup=(32, 1, 1),
        output_shapes=[(B, L, H, D)], output_dtypes=[q.dtype])[0]
    return out.transpose(0, 2, 1, 3)


def choose_path(sel: MSASelection, q: mx.array, k, mask, n_kv: int) -> str:
    """'dense', 'gather' or 'decode' for this call. Only picks a path; every path computes the same attention (the
    sparse ones over the selected blocks only). The pick depends on (rows, keys, dtype, shapes) only, so a one-row
    verify call takes exactly the path a one-row decode step takes."""
    if not isinstance(k, mx.array) or int(sel.top_idx.shape[1]) != n_kv or isinstance(mask, mx.array):
        # Quantized K / V, indexer heads that do not line up with KV groups, or an extra array mask (batch
        # padding): the dense mask path covers these as before.
        return "dense"
    want = requested_path()
    if want == "dense":
        return "dense"
    if want == "gather":
        return "gather"
    if want == "decode":
        return "decode" if decode_kernel_supported(q, k, sel) else "dense"
    L = int(q.shape[2])
    if L <= DECODE_MAX_ROWS and decode_kernel_supported(q, k, sel):
        if want == "sparse" or L <= DECODE_ALWAYS_ROWS or sel.sk >= DECODE_MIN_KEYS:
            return "decode"
    return "dense"


# Routing constants. They only pick a path; measured on an M5 Max, bf16, M3.1 geometry (64 heads, 4 KV groups,
# head dim 128, blocks of 128, top-16), bench/minimax_m3_msa_bench.py, latency of dependent calls, medians of 7:
# * one row (decode step / verify row): the decode kernel beat the dense mask path at every sparse length
#   (attention 0.039 vs 0.052 ms at 2049 keys, 0.039 vs 0.088 at 8k, 0.040 vs 0.226 at 32k, 0.045 vs 0.754 at 128k).
# * 2 .. 16 rows: the decode kernel costs ~16 us a row; dense (MLX's unfused path at these widths) ~0.25 ms at
#   2.5k keys and ~0.8 ms at 8k. Up to 8 rows the decode kernel won everywhere; at 16 rows dense was 6 % faster at
#   2.5k keys and the decode kernel 2.9x faster at 8k.
#   fp32: the decode kernel still won (0.12 vs 0.28 ms at 8k, 0.14 vs 0.90 at 32k).
DECODE_MAX_ROWS = 16
DECODE_ALWAYS_ROWS = 8
DECODE_MIN_KEYS = 4096

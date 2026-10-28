# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA (block-sparse attention): the selection and the attention paths that consume it.

The Lightning indexer picks, for every query row and KV group, ``keep`` (16) key blocks of ``block`` (128) keys:
the top block scores, the query's own (local) block forced in. Attention then sees only those blocks, causally.

``MSASelection`` carries that pick as block ids ([B, G, Sq, keep]) plus the positions it was made at. Paths
(``choose_path`` routes by measured crossovers; ``VMLX_MINIMAX_MSA`` forces one for comparisons):

* ``dense``: the additive [B, G, Sq, Sk] float32 key mask the runtime used before (``dense_bias``, same ops) and
  one SDPA over all Sk keys. Kept as the reference / fallback path.
* ``decode``: decode-sized calls (decode steps, DSpark verify rows). A Metal kernel splits each (row, group)'s
  keep * block selected keys into 32-key chunks across threadgroups and merges the partial softmaxes; keys are read
  in place from the cache. Rows are independent: a row's output is the same alone or with other rows.
* ``kernel``: prefill chunks. One simdgroup per (row, group) walks the row's selected blocks with an online
  softmax, on the matrix units (Metal 4 tensor ops) for bf16 / fp16, else on 8x8 simdgroup matrices. No mask, no
  copy of K / V.
* ``gather``: pure MLX ops (gather the selected keys, SDPA over them). Not routed to; kept as an op-level reference.

Positions: query row r sits at ``offset + r``; key j is allowed when its block is selected and ``j <= offset + r``.
Keys at or past ``sk`` (the logical extent, i.e. the cache offset) are never read: the kernels stop at the row's
position (< sk) and the gather clamps to ``sk - 1`` and masks the clamped lanes. A partial last block, a query's own
block (causal inside it) and selected blocks that lie entirely in the future (possible when fewer than ``keep``
blocks precede a prefill row) are all handled by that one positional rule.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import mlx.core as mx

# Path override for comparisons and diagnostics: "auto" (default) routes by the measured crossovers below,
# "dense" forces the [B, G, Sq, Sk] mask path, "sparse" the block-sparse path for the call's size (the decode kernel
# for decode-sized calls, the chunk kernel for longer ones), "gather" / "decode" / "kernel" that path wherever its
# shapes are supported.
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
    return raw if raw in ("dense", "sparse", "gather", "decode", "kernel") else "auto"


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


# --------------------------------------------------------------------------------------------- prefill kernel
# One threadgroup per (batch, KV group, query row): the HG query heads of the group (2 simdgroups x 8 heads) attend
# that row's selected blocks, read in place from K / V (any sequence / group / batch stride; the head dim must be
# contiguous). Online softmax in fp32 over 32-key tiles staged in threadgroup memory; S = Q K^T and O += P V on 8x8
# simdgroup matrices with fp32 accumulators. Blocks are visited in ascending order; tiles that start past the row's
# position are skipped (never read), keys past the row's position are masked. Ints that vary per call (offset, sk)
# ride an input buffer, so one compiled kernel serves every chunk.
_PREFILL_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_position_in_threadgroup.x;
  const int l = int(threadgroup_position_in_grid.y);
  const int bg = int(threadgroup_position_in_grid.z);
  const int G = int(blocks_shape[1]);
  const int L = int(blocks_shape[2]);
  const int H = int(q_shape[2]);
  const int b = bg / G;
  const int g = bg - b * G;
  const int offset = meta[0];
  const int sk = meta[1];
  const int q_pos = offset + l;
  const float qscale = scale[0] * 1.4426950408889634f;   // softmax in base 2
  const float NEG = -1.0e30f;

  threadgroup elem_t ks[TK * KPAD];
  threadgroup elem_t vs[TK * KPAD];

  // Lane layout of an 8x8 simdgroup matrix: this lane holds (row fm, cols fn and fn + 1).
  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;

  const int head = g * HG + int(sg) * 8 + fm;
  device const elem_t* qrow = q + ((size_t)(b * L + l) * H + head) * D;
  simdgroup_matrix<float, 8, 8> Qf[D / 8];
  simdgroup_matrix<float, 8, 8> Of[D / 8];
  for (int d = 0; d < D / 8; ++d) {
    Qf[d].thread_elements()[0] = float(qrow[d * 8 + fn]) * qscale;
    Qf[d].thread_elements()[1] = float(qrow[d * 8 + fn + 1]) * qscale;
    Of[d] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m_run = NEG;
  float l_run = 0.0f;

  device const int* blk = blocks + ((size_t)(b * G + g) * L + l) * KEEP;
  device const elem_t* kbase = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
  device const elem_t* vbase = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
  const size_t kss = k_strides[2];
  const size_t vss = v_strides[2];
  constexpr int CE = 16 / sizeof(elem_t);                 // elements per 16-byte chunk
  const bool vec_ok = k_strides[3] == 1 && v_strides[3] == 1 && (kss % CE) == 0 && (vss % CE) == 0
      && ((ulong)kbase % 16) == 0 && ((ulong)vbase % 16) == 0;

  for (int i = 0; i < KEEP; ++i) {
    const int start = blk[i] * BLOCK;
    if (start > q_pos) break;                               // ascending: every later block is in the future too
    for (int t0 = 0; t0 < BLOCK; t0 += TK) {
      const int ts = start + t0;
      if (ts > q_pos) break;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (vec_ok) {
        for (int c = int(tid); c < TK * (D / CE); c += NT) {
          const int r = c / (D / CE);
          const int cc = c - r * (D / CE);
          const int kp = ts + r;
          uint4 kv = uint4(0), vv = uint4(0);
          if (kp < sk) {
            kv = *((device const uint4*)(kbase + (size_t)kp * kss) + cc);
            vv = *((device const uint4*)(vbase + (size_t)kp * vss) + cc);
          }
          *((threadgroup uint4*)(ks + r * KPAD) + cc) = kv;
          *((threadgroup uint4*)(vs + r * KPAD) + cc) = vv;
        }
      } else {
        for (int c = int(tid); c < TK * D; c += NT) {
          const int r = c / D;
          const int dd = c - r * D;
          const int kp = ts + r;
          ks[r * KPAD + dd] = kp < sk ? kbase[(size_t)kp * kss + (size_t)dd * k_strides[3]] : elem_t(0);
          vs[r * KPAD + dd] = kp < sk ? vbase[(size_t)kp * vss + (size_t)dd * v_strides[3]] : elem_t(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      simdgroup_matrix<float, 8, 8> Sf[TK / 8];
      for (int f = 0; f < TK / 8; ++f) {
        Sf[f] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (int d = 0; d < D / 8; ++d) {
          simdgroup_matrix<float, 8, 8> Kt;
          Kt.thread_elements()[0] = float(ks[(f * 8 + fn) * KPAD + d * 8 + fm]);
          Kt.thread_elements()[1] = float(ks[(f * 8 + fn + 1) * KPAD + d * 8 + fm]);
          simdgroup_multiply_accumulate(Sf[f], Qf[d], Kt, Sf[f]);
        }
      }
      float tmax = NEG;
      for (int f = 0; f < TK / 8; ++f) {
        for (int e = 0; e < 2; ++e) {
          const int kp = ts + f * 8 + fn + e;
          float s = Sf[f].thread_elements()[e];
          s = kp <= q_pos ? s : NEG;
          Sf[f].thread_elements()[e] = s;
          tmax = max(tmax, s);
        }
      }
      tmax = max(tmax, simd_shuffle_xor(tmax, 1));
      tmax = max(tmax, simd_shuffle_xor(tmax, 8));
      const float m_new = max(m_run, tmax);
      const float factor = fast::exp2(m_run - m_new);
      float psum = 0.0f;
      for (int f = 0; f < TK / 8; ++f) {
        for (int e = 0; e < 2; ++e) {
          const float p = fast::exp2(Sf[f].thread_elements()[e] - m_new);
          Sf[f].thread_elements()[e] = p;
          psum += p;
        }
      }
      psum += simd_shuffle_xor(psum, 1);
      psum += simd_shuffle_xor(psum, 8);
      l_run = l_run * factor + psum;
      m_run = m_new;
      for (int d = 0; d < D / 8; ++d) {
        Of[d].thread_elements()[0] *= factor;
        Of[d].thread_elements()[1] *= factor;
      }
      for (int f = 0; f < TK / 8; ++f) {
        for (int d = 0; d < D / 8; ++d) {
          simdgroup_matrix<float, 8, 8> Vf;
          Vf.thread_elements()[0] = float(vs[(f * 8 + fm) * KPAD + d * 8 + fn]);
          Vf.thread_elements()[1] = float(vs[(f * 8 + fm) * KPAD + d * 8 + fn + 1]);
          simdgroup_multiply_accumulate(Of[d], Sf[f], Vf, Of[d]);
        }
      }
    }
  }
  const float inv = 1.0f / l_run;
  device elem_t* orow = out + ((size_t)(b * L + l) * H + head) * D;
  for (int d = 0; d < D / 8; ++d) {
    orow[d * 8 + fn] = elem_t(Of[d].thread_elements()[0] * inv);
    orow[d * 8 + fn + 1] = elem_t(Of[d].thread_elements()[1] * inv);
  }
"""

_PREFILL_KERNELS: dict = {}


def _prefill_tile(elem: str) -> int:
    """Keys per staged K / V tile: two [TK][D + 8] tiles fit the 32 KB of threadgroup memory."""
    return 16 if elem == "float" else 32


def _prefill_kernel(elem: str, hg: int, d: int, keep: int, block: int):
    key = (elem, hg, d, keep, block)
    kern = _PREFILL_KERNELS.get(key)
    if kern is None:
        consts = {"HG": hg, "D": d, "KEEP": keep, "BLOCK": block, "TK": _prefill_tile(elem), "KPAD": d + 8,
                  "NT": 32 * (hg // 8)}
        source = f"  using elem_t = {elem};\n" + "".join(
            f"  constexpr int {n} = {int(v)};\n" for n, v in consts.items()) + _PREFILL_SOURCE
        kern = _PREFILL_KERNELS[key] = mx.fast.metal_kernel(
            name=_kname(f"vmlx_minimax_msa_prefill_{elem}_h{hg}_d{d}_k{keep}_b{block}", source),
            input_names=["q", "k", "v", "blocks", "meta", "scale"], output_names=["out"], source=source,
            ensure_row_contiguous=False)
    return kern


# --------------------------------------------------------------------------------------------- matrix-unit kernel
# Same contract as the kernel above, on the M5 matrix units (Metal 4 tensor ops, per-simdgroup 16x32x16 matmul2d on
# register fragments). One simdgroup owns one (batch, KV group, query row): the group's query heads are the 16 rows of
# every fragment (fewer heads leave the remaining rows zero), so S = Q K^T is [16 heads x 32 keys] per step and O is
# [16 x D] in registers. Fragment layout of a 16x16 tile: this lane holds rows fm and fm + 8, columns fn .. fn + 3.
# K / V are read straight from device memory (no staging); NSG consecutive query rows share a threadgroup so blocks
# the rows have in common are read while still in cache. Q K^T multiplies the stored (bf16 / fp16) values exactly
# with fp32 accumulation; P (fp32) times V runs at the op's relaxed precision (P rounded to ~11 significant bits).
_NAX_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

template <typename AT, typename BT, bool TB>
METAL_FUNC void msa_mma_16x32(thread float* c, thread const AT* a, thread const BT* b0, thread const BT* b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  #pragma unroll
  for (short i = 0; i < 8; ++i) { ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; }
  #pragma unroll
  for (short i = 0; i < 16; ++i) cc[i] = c[i];
  op.run(ca, cb, cc);
  #pragma unroll
  for (short i = 0; i < 16; ++i) c[i] = cc[i];
}

// dst[0..3] = row[col .. col + 3] (head-dim stride ds); zeros when the row is past the logical extent.
template <typename T>
METAL_FUNC void msa_load4(thread T* dst, device const T* row, int col, size_t ds, bool vec, bool valid) {
  if (!valid) {
    dst[0] = T(0); dst[1] = T(0); dst[2] = T(0); dst[3] = T(0);
  } else if (vec) {
    const metal::vec<T, 4> x = *((device const metal::vec<T, 4>*)(row + col));
    dst[0] = x[0]; dst[1] = x[1]; dst[2] = x[2]; dst[3] = x[3];
  } else {
    for (short j = 0; j < 4; ++j) dst[j] = row[(size_t)(col + j) * ds];
  }
}
"""

_NAX_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int L = int(blocks_shape[2]);
  const int l = int(threadgroup_position_in_grid.y) * NSG + sg;
  if (l >= L) return;                                      // per-simdgroup ops only: no threadgroup barrier below
  const int G = int(blocks_shape[1]);
  const int H = int(q_shape[2]);
  const int bg = int(threadgroup_position_in_grid.z);
  const int b = bg / G;
  const int g = bg - b * G;
  const int offset = meta[0];
  const int sk = meta[1];
  const int q_pos = offset + l;
  const float qscale = scale[0] * 1.4426950408889634f;
  const float NEG = -1.0e30f;

  const short qid = short(lane >> 2);
  const short fm = (qid & 4) | short((lane >> 1) & 3);
  const short fn = short(((qid & 2) | (lane & 1)) * 4);

  // Q fragments: rows = heads of the group, columns = 16-wide head-dim chunks.
  elem_t qf[D / 16][8];
  device const elem_t* qrow = q + ((size_t)(b * L + l) * H + g * HG) * D;
  for (int c = 0; c < D / 16; ++c) {
    for (short i = 0; i < 2; ++i) {
      const short r = fm + 8 * i;
      for (short j = 0; j < 4; ++j) {
        qf[c][i * 4 + j] = r < HG ? qrow[(size_t)r * D + c * 16 + fn + j] : elem_t(0);
      }
    }
  }
  float o[D / 32][16];
  for (int dp = 0; dp < D / 32; ++dp) {
    for (short e = 0; e < 16; ++e) o[dp][e] = 0.0f;
  }
  float m_run[2] = {NEG, NEG};
  float l_run[2] = {0.0f, 0.0f};

  device const int* blk = blocks + ((size_t)(b * G + g) * L + l) * KEEP;
  device const elem_t* kbase = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
  device const elem_t* vbase = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
  const size_t kss = k_strides[2];
  const size_t vss = v_strides[2];
  const size_t kds = k_strides[3];
  const size_t vds = v_strides[3];
  // 8-byte vector loads when the head dim is contiguous and every 4-element chunk is 8-byte aligned.
  const bool vec_ok = kds == 1 && vds == 1 && (kss % 4) == 0 && (vss % 4) == 0
      && ((ulong)kbase % 8) == 0 && ((ulong)vbase % 8) == 0;

  for (int bi = 0; bi < KEEP; ++bi) {
    const int start = blk[bi] * BLOCK;
    if (start > q_pos) break;                              // ascending: every later block is in the future too
    for (int t0 = 0; t0 < BLOCK; t0 += 32) {
      const int ts = start + t0;
      if (ts > q_pos) break;
      float s[16];
      for (short e = 0; e < 16; ++e) s[e] = 0.0f;
      for (int c = 0; c < D / 16; ++c) {
        elem_t kb0[8], kb1[8];
        for (short i = 0; i < 2; ++i) {
          const int kp0 = ts + fm + 8 * i;
          const int kp1 = kp0 + 16;
          msa_load4(kb0 + i * 4, kbase + (size_t)kp0 * kss, c * 16 + fn, kds, vec_ok, kp0 < sk);
          msa_load4(kb1 + i * 4, kbase + (size_t)kp1 * kss, c * 16 + fn, kds, vec_ok, kp1 < sk);
        }
        msa_mma_16x32<elem_t, elem_t, true>(s, qf[c], kb0, kb1);
      }
      // s[n * 8 + i * 4 + j] = S[head fm + 8i][key ts + n * 16 + fn + j]
      float rmax[2] = {NEG, NEG};
      for (short n = 0; n < 2; ++n) {
        for (short i = 0; i < 2; ++i) {
          for (short j = 0; j < 4; ++j) {
            const int kp = ts + n * 16 + fn + j;
            const short e = n * 8 + i * 4 + j;
            const float x = kp <= q_pos ? s[e] * qscale : NEG;
            s[e] = x;
            rmax[i] = max(rmax[i], x);
          }
        }
      }
      float factor[2];
      for (short i = 0; i < 2; ++i) {
        rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(1)));
        rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(8)));
        const float m_new = max(m_run[i], rmax[i]);
        factor[i] = fast::exp2(m_run[i] - m_new);
        m_run[i] = m_new;
      }
      float rsum[2] = {0.0f, 0.0f};
      for (short n = 0; n < 2; ++n) {
        for (short i = 0; i < 2; ++i) {
          for (short j = 0; j < 4; ++j) {
            const short e = n * 8 + i * 4 + j;
            const float p = fast::exp2(s[e] - m_run[i]);
            s[e] = p;
            rsum[i] += p;
          }
        }
      }
      for (short i = 0; i < 2; ++i) {
        rsum[i] += simd_shuffle_xor(rsum[i], ushort(1));
        rsum[i] += simd_shuffle_xor(rsum[i], ushort(8));
        l_run[i] = l_run[i] * factor[i] + rsum[i];
      }
      for (int dp = 0; dp < D / 32; ++dp) {
        for (short h = 0; h < 2; ++h) {
          for (short i = 0; i < 2; ++i) {
            for (short j = 0; j < 4; ++j) o[dp][h * 8 + i * 4 + j] *= factor[i];
          }
        }
      }
      for (short n = 0; n < 2; ++n) {
        for (int dp = 0; dp < D / 32; ++dp) {
          elem_t vb0[8], vb1[8];
          for (short i = 0; i < 2; ++i) {
            const int kp = ts + n * 16 + fm + 8 * i;
            msa_load4(vb0 + i * 4, vbase + (size_t)kp * vss, dp * 32 + fn, vds, vec_ok, kp < sk);
            msa_load4(vb1 + i * 4, vbase + (size_t)kp * vss, dp * 32 + 16 + fn, vds, vec_ok, kp < sk);
          }
          msa_mma_16x32<float, elem_t, false>(o[dp], s + n * 8, vb0, vb1);
        }
      }
    }
  }
  device elem_t* orow = out + ((size_t)(b * L + l) * H + g * HG) * D;
  for (short i = 0; i < 2; ++i) {
    const short r = fm + 8 * i;
    if (r >= HG) continue;
    const float inv = 1.0f / l_run[i];
    for (int dp = 0; dp < D / 32; ++dp) {
      for (short h = 0; h < 2; ++h) {
        for (short j = 0; j < 4; ++j) {
          orow[(size_t)r * D + dp * 32 + h * 16 + fn + j] = elem_t(o[dp][h * 8 + i * 4 + j] * inv);
        }
      }
    }
  }
"""

_NAX_NSG = 4
_NAX_KERNELS: dict = {}
_NAX_OK: Optional[bool] = None


def _nax_kernel(elem: str, hg: int, d: int, keep: int, block: int):
    key = (elem, hg, d, keep, block)
    kern = _NAX_KERNELS.get(key)
    if kern is None:
        consts = {"HG": hg, "D": d, "KEEP": keep, "BLOCK": block, "NSG": _NAX_NSG}
        source = f"  using elem_t = {elem};\n" + "".join(
            f"  constexpr int {n} = {int(v)};\n" for n, v in consts.items()) + _NAX_SOURCE
        kern = _NAX_KERNELS[key] = mx.fast.metal_kernel(
            name=_kname(f"vmlx_minimax_msa_prefill_nax_{elem}_h{hg}_d{d}_k{keep}_b{block}", _NAX_HEADER + source),
            input_names=["q", "k", "v", "blocks", "meta", "scale"], output_names=["out"], source=source,
            header=_NAX_HEADER, ensure_row_contiguous=False)
    return kern


def _nax_launch(q, k, v, sel, scale):
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    hg = H // G
    blocks = mx.sort(sel.top_idx.astype(mx.int32), axis=-1)
    qc = mx.contiguous(q.transpose(0, 2, 1, 3))                     # [B, L, H, D]
    meta = mx.array([sel.offset, sel.sk], dtype=mx.int32)
    kern = _nax_kernel(_ELEM[q.dtype], hg, D, sel.keep, sel.block)
    out = kern(
        inputs=[qc, k, v, blocks, meta, mx.array([scale], dtype=mx.float32)],
        grid=(32 * _NAX_NSG, -(-L // _NAX_NSG), B * G), threadgroup=(32 * _NAX_NSG, 1, 1),
        output_shapes=[(B, L, H, D)], output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def nax_available() -> bool:
    """Metal 4 tensor ops compile here and the matrix-unit kernel matches the fp32 kernel on a small case with a
    partial last block and causal rows inside it (checked once per process)."""
    global _NAX_OK
    if _NAX_OK is None:
        try:
            if not _metal():
                raise RuntimeError("no Metal GPU")
            key = mx.random.key(11)
            k1, k2, k3, k4 = mx.random.split(key, 4)
            G, HG, D, sk, L, blk, keep = 2, 16, 64, 300, 40, 32, 4
            q = mx.random.normal((1, G * HG, L, D), key=k1).astype(mx.bfloat16)
            kk = mx.random.normal((1, G, sk, D), key=k2).astype(mx.bfloat16)
            vv = mx.random.normal((1, G, sk, D), key=k3).astype(mx.bfloat16)
            nb = -(-sk // blk)
            pick = mx.argsort(mx.random.uniform(shape=(1, G, L, nb), key=k4), axis=-1)[..., :keep - 1]
            local = mx.broadcast_to(mx.arange(sk - L, sk).reshape(1, 1, L, 1) // blk, (1, G, L, 1))
            pick = mx.where(pick == local, (local + 1) % nb, pick)
            sel = MSASelection(mx.concatenate([local, pick], axis=-1).astype(mx.uint32), sk - L, sk, blk)
            got = _nax_launch(q, kk, vv, sel, D ** -0.5).astype(mx.float32)
            ref = kernel_attention(q.astype(mx.float32), kk.astype(mx.float32), vv.astype(mx.float32), sel,
                                   D ** -0.5, _matrix_units=False)
            err = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
            _NAX_OK = bool(err < 2e-2)
        except Exception:
            _NAX_OK = False
    return _NAX_OK


def kernel_supported(q: mx.array, k: mx.array, sel: MSASelection) -> bool:
    """Shapes the prefill kernel is written for: head dim a multiple of 8 (<= 256), whole simdgroups of 8 query
    heads per KV group, blocks a multiple of the 32-key tile, K / V in a dtype the kernel stages."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    if not _metal() or q.dtype not in _ELEM or k.dtype != q.dtype or H % G:
        return False
    hg = H // G
    tile = _prefill_tile(_ELEM[q.dtype])
    return (D % 8 == 0 and 2 * tile * (D + 8) * q.dtype.size <= 32768 and hg % 8 == 0 and hg <= 32
            and sel.block % tile == 0)


def matrix_units_supported(q: mx.array, sel: MSASelection) -> bool:
    """The matrix-unit kernel's shapes: bf16 / fp16, at most 16 query heads per KV group (one fragment of rows),
    head dim a multiple of 32, blocks a multiple of 32 keys, Metal 4 tensor ops present and self-checked."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    return (q.dtype in (mx.bfloat16, mx.float16) and H % G == 0 and H // G <= 16 and D % 32 == 0 and D <= 256
            and sel.block % 32 == 0 and nax_available())


def kernel_attention(q: mx.array, k: mx.array, v: mx.array, sel: MSASelection, scale: float, *,
                     _matrix_units: Optional[bool] = None) -> mx.array:
    """Block-sparse attention [B, H, L, D] for an L-row chunk, reading only the selected blocks (no key mask).
    bf16 / fp16 run on the matrix units when admitted (``matrix_units_supported``), everything else (fp32 included)
    on the simdgroup kernel."""
    B, H, L, D = q.shape
    G = int(sel.top_idx.shape[1])
    hg = H // G
    _check_extent(k, v, sel)
    use_nax = matrix_units_supported(q, sel) if _matrix_units is None else _matrix_units
    if use_nax:
        return _nax_launch(q, k, v, sel, scale)
    blocks = mx.sort(sel.top_idx.astype(mx.int32), axis=-1)
    qc = mx.contiguous(q.transpose(0, 2, 1, 3))                     # [B, L, H, D]
    meta = mx.array([sel.offset, sel.sk], dtype=mx.int32)
    kern = _prefill_kernel(_ELEM[q.dtype], hg, D, sel.keep, sel.block)
    out = kern(
        inputs=[qc, k, v, blocks, meta, mx.array([scale], dtype=mx.float32)],
        grid=(32 * (hg // 8), L, B * G), threadgroup=(32 * (hg // 8), 1, 1),
        output_shapes=[(B, L, H, D)], output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def choose_path(sel: MSASelection, q: mx.array, k, mask, n_kv: int) -> str:
    """'dense', 'gather', 'decode' or 'kernel' for this call. Only picks a path; every path computes the same
    attention (the sparse ones over the selected blocks only). The pick depends on (rows, keys, dtype, shapes) only,
    so a one-row verify call takes exactly the path a one-row decode step takes."""
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
    if want == "kernel":
        return "kernel" if kernel_supported(q, k, sel) else "dense"
    L = int(q.shape[2])
    if L <= DECODE_MAX_ROWS:
        if not decode_kernel_supported(q, k, sel):
            return "dense"
        if want == "sparse" or L <= DECODE_ALWAYS_ROWS or sel.sk >= DECODE_MIN_KEYS:
            return "decode"
        return "dense"
    if not kernel_supported(q, k, sel):
        return "dense"
    if want == "sparse":
        return "kernel"
    # Auto: the chunk kernel only on the matrix units (fp32 / no Metal 4: the simdgroup version measured slower).
    return "kernel" if sel.sk >= KERNEL_MIN_KEYS and matrix_units_supported(q, sel) else "dense"


# Routing constants. They only pick a path; measured on an M5 Max, bf16, M3.1 geometry (64 heads, 4 KV groups,
# head dim 128, blocks of 128, top-16), bench/minimax_m3_msa_bench.py, latency of dependent calls, medians of 7:
# * one row (decode step / verify row): the decode kernel beat the dense mask path at every sparse length
#   (attention 0.039 vs 0.052 ms at 2049 keys, 0.039 vs 0.088 at 8k, 0.040 vs 0.226 at 32k, 0.045 vs 0.754 at 128k).
# * 2 .. 16 rows: the decode kernel costs ~16 us a row; dense (MLX's unfused path at these widths) ~0.25 ms at
#   2.5k keys and ~0.8 ms at 8k. Up to 8 rows the decode kernel won everywhere; at 16 rows dense was 6 % faster at
#   2.5k keys and the decode kernel 2.9x faster at 8k.
# * longer chunks: the chunk kernel's cost is flat in context (~7.7 ms for 2048 rows), dense grows with it; dense won
#   below ~4.8k (2048 rows) / ~5.1k (512 rows) / ~5k (128 rows) keys, the kernel above (1.7x at 8k, 4.9x at 32k).
#   fp32 (simdgroup kernel, 512 rows): 18.8 vs 5.9 ms dense at 8k, 30.6 vs 32.0 at 32k, so fp32 chunks stay dense;
#   the fp32 decode kernel still won (0.12 vs 0.28 ms at 8k, 0.14 vs 0.90 at 32k).
DECODE_MAX_ROWS = 16
DECODE_ALWAYS_ROWS = 8
DECODE_MIN_KEYS = 4096
KERNEL_MIN_KEYS = 5120

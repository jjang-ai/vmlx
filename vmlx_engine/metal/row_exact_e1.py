"""Exact multi-row affine QMV for DSpark / MTP verify: each weight read and decoded ONCE per tile of rows (E1).

Contract (the same as ``row_exact_qmv``): every row of a call equals MLX's own one-row ``quantized_matmul`` of that row
(``qmv_fast``) bit for bit, so a verify row equals the decode step that would compute it. The difference is cost: the
``row_exact_qmv`` transcriptions decode each weight word per <= 4-row block inside MLX's per-row loop and redo
``load_vector`` per output tile; here

* one PREP launch runs MLX's own ``load_vector`` (taken verbatim from the installed MLX headers) once per (row, block,
  lane) and stores ``x_thread`` and its ``sum`` (6-bit: with the two ``x * 256`` terms ``qdot`` forms, so the walk
  only reads them), rows zero-padded to whole tiles;
* the WALK gives each simdgroup O output rows and a tile of TR rows: per block and output row it loads the packed
  word(s) once, decodes one four-value group at a time into half registers (4 / 6 / 8-bit codes times their place
  value are exact in half and ``float(half)`` is the float ``qdot`` multiplies by), and applies the group to every row
  of the tile with ``qdot``'s own statements for that width (4-bit ``accum += (x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3)``;
  6-bit / 8-bit one ``accum += c * x`` per term), ``accum`` starting at 0 per block; then
  ``result += scale * accum + sum * bias`` per block and ``static_cast<T>(simd_sum(result))``, as ``qmv_fast``.

So the per-element operation sequence is ``qmv_fast``'s; only the loop nest around it changed. Each (rows, K, output
widths, bits, group, dtype) is admitted only after a bit-equality check against stock one-row calls on the real
weights (zero-mean and shifted / spiky probes); a failed check returns None (the caller keeps its exact path) and logs
it. Several projections with the same input (a SwiGLU's gate and up) can share one prep and one walk (``segments``).

Measured on the M5 Max (MLX 0.32.3, MiniMax-M3.1 shapes; see docs ``14-PY-EXACT-FLAT-VERIFY.md``). Lever:
``VMLX_M3_EXACT_FLAT`` (the callers check it; default off).

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import Optional, Sequence

import mlx.core as mx

logger = logging.getLogger(__name__)

MIN_ROWS, MAX_ROWS = 2, 16
# Callers take E1 from this many rows: at 2 rows it ties today's builds (no gain for the extra prep launch).
DISPATCH_MIN_ROWS = 3
TILE_ROWS = 4
_ADMITTED: dict = {}
_DECISIONS: dict = {}
STATS = {"calls": 0, "admitted": 0, "rejected": 0}

_HEADER_EXTRA = r"""
// --------------------------------------------------------------------------------------------- E1 walk helpers
// Values per lane chunk (VPT), prepared floats per chunk (XW) and packed bytes per chunk of qmv_fast for BITS
// (packs_per_thread 2).
template <int BITS> inline constexpr int e1_vpt() { return BITS == 4 ? 16 : 8; }
template <int BITS> inline constexpr int e1_xw() { return BITS == 6 ? 12 : e1_vpt<BITS>(); }
template <int BITS> inline constexpr int e1_bytes() { return e1_vpt<BITS>() * BITS / 8; }

// Decoded codes of four-value group g of one lane chunk (MLX qdot's masks, the place value kept): 4 codes for 4 / 8-bit,
// 6 partial codes for 6-bit (two values span two bytes). Exact in half.
template <int BITS>
METAL_FUNC void e1_codes(const thread uint* wb, int g, thread half* c) {
  if (BITS == 4) {
    const uint h16 = (wb[g >> 1] >> (16 * (g & 1))) & 0xffffu;
    c[0] = half(h16 & 0x000fu);
    c[1] = half(h16 & 0x00f0u);
    c[2] = half(h16 & 0x0f00u);
    c[3] = half(h16 & 0xf000u);
  } else if (BITS == 8) {
    const uint wd = wb[g];
    for (int j = 0; j < 4; ++j) c[j] = half((wd >> (8 * j)) & 0xffu);
  } else {  // 6-bit: bytes 3 g .. 3 g + 2 of the chunk
    const int o = 3 * g;
    const uint b0 = (wb[o >> 2] >> (8 * (o & 3))) & 0xffu;
    const uint b1 = (wb[(o + 1) >> 2] >> (8 * ((o + 1) & 3))) & 0xffu;
    const uint b2 = (wb[(o + 2) >> 2] >> (8 * ((o + 2) & 3))) & 0xffu;
    c[0] = half(b0 & 0x3fu);
    c[1] = half(b0 & 0xc0u);
    c[2] = half(b1 & 0x0fu);
    c[3] = half(b1 & 0xf0u);
    c[4] = half(b2 & 0x03u);
    c[5] = half(b2 & 0xfcu);
  }
}

// qdot's accumulation of one four-value group for one row: x = the row's prepared floats of the group (4-bit / 8-bit:
// x0..x3; 6-bit: x0, x1, x1 * 256, x2, x2 * 256, x3).
template <int BITS>
METAL_FUNC void e1_group(thread float& accum, const thread float* x, const thread half* c) {
  if (BITS == 4) {
    accum += (x[0] * float(c[0]) + x[1] * float(c[1]) + x[2] * float(c[2]) + x[3] * float(c[3]));
  } else if (BITS == 8) {
    accum += x[0] * float(c[0]);
    accum += x[1] * float(c[1]);
    accum += x[2] * float(c[2]);
    accum += x[3] * float(c[3]);
  } else {
    accum += float(c[0]) * x[0];
    accum += float(c[1]) * x[1];
    accum += float(c[2]) * x[2];
    accum += float(c[3]) * x[3];
    accum += float(c[4]) * x[4];
    accum += float(c[5]) * x[5];
  }
}
"""

_PREP = r"""
  // MLX load_vector once per (row, block, lane): x_thread (6-bit: expanded with the two x * 256 terms) and its sum.
  constexpr int VPT = e1_vpt<BITS>();
  constexpr int XW = e1_xw<BITS>();
  constexpr int BLK = VPT * 32;
  constexpr int NCH = K / VPT;
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  device float* xr = xp + (size_t)r * NCH * XW;
  if (r >= int(x_shape[0])) {
    for (int k = 0; k < K; k += BLK) {
      device float* dst = xr + (size_t)(k / VPT + int(lane)) * XW;
      for (int i = 0; i < XW; ++i) dst[i] = 0.0f;
      xs[(size_t)r * NCH + k / VPT + int(lane)] = 0.0f;
    }
    return;
  }
  for (int k = 0; k < K; k += BLK) {
    float xt_[VPT];
    const float sm = load_vector<T, float, VPT, BITS>(x + (size_t)r * K + k + int(lane) * VPT, xt_);
    device float* dst = xr + (size_t)(k / VPT + int(lane)) * XW;
    if (BITS == 6) {
      for (int i = 0; i < VPT / 4; ++i) {
        dst[6 * i + 0] = xt_[4 * i];
        dst[6 * i + 1] = xt_[4 * i + 1];
        dst[6 * i + 2] = xt_[4 * i + 1] * 256.0f;
        dst[6 * i + 3] = xt_[4 * i + 2];
        dst[6 * i + 4] = xt_[4 * i + 2] * 256.0f;
        dst[6 * i + 5] = xt_[4 * i + 3];
      }
    } else {
      for (int i = 0; i < VPT; ++i) dst[i] = xt_[i];
    }
    xs[(size_t)r * NCH + k / VPT + int(lane)] = sm;
  }
"""

_WALK = r"""
  // Per simdgroup O output rows, per threadgroup one row tile of TR rows (tiles of one weight tile adjacent).
  constexpr int VPT = e1_vpt<BITS>();
  constexpr int XW = e1_xw<BITS>();
  constexpr int BLK = VPT * 32;
  constexpr int NCH = K / VPT;
  constexpr int BYTES = e1_bytes<BITS>();
  constexpr int WN = (BYTES + 3) / 4;
  constexpr int NG = VPT / 4;
  constexpr int GW = BITS == 6 ? 6 : 4;            // prepared floats / codes per four-value group
  constexpr int ROWB = K * BITS / 8;
  constexpr int NGRP = K / GS;
  constexpr int NTOT = N0 + N1;
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(rv_shape[0]);
  const int NRT = int(xs_shape[0]) / TR;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * TR;
  const int orow = (tg * NSG + sg) * O;            // output row in the concatenated segments
  if (orow >= NTOT) return;
  const bool seg1 = orow >= N0;
  const device uint8_t* wseg = (const device uint8_t*)(seg1 ? w1 : w0);
  const device T* sseg = seg1 ? s1 : s0;
  const device T* bseg = seg1 ? b1 : b0;
  const int lrow = seg1 ? orow - N0 : orow;
  const device uint8_t* wl = wseg + (size_t)lrow * ROWB + int(lane) * BYTES;
  const int gl = (int(lane) * VPT) / GS;
  float res[TR][O];
  for (int t = 0; t < TR; ++t)
    for (int o = 0; o < O; ++o) res[t][o] = 0.0f;
  const device float* xr0 = xp + (size_t)r0 * NCH * XW + (size_t)int(lane) * XW;
  for (int k = 0; k < K; k += BLK) {
    const int gi = k / GS + gl;
    uint wb[O][WN];
    float s[O], b[O];
    for (int o = 0; o < O; ++o) {
      const device uint8_t* wp = wl + (size_t)o * ROWB + (size_t)k * BITS / 8;
      if (BITS == 6) {
        const device ushort* w16 = (const device ushort*)wp;   // 6 bytes, 2-byte aligned
        wb[o][0] = uint(w16[0]) | (uint(w16[1]) << 16);
        wb[o][1] = uint(w16[2]);
      } else {
        const uint2 v2 = *((const device uint2*)wp);
        wb[o][0] = v2.x;
        wb[o][WN - 1] = v2.y;
      }
      s[o] = float(sseg[(size_t)(lrow + o) * NGRP + gi]);
      b[o] = float(bseg[(size_t)(lrow + o) * NGRP + gi]);
    }
    float d[TR][O];
    for (int t = 0; t < TR; ++t)
      for (int o = 0; o < O; ++o) d[t][o] = 0.0f;
    for (int g = 0; g < NG; ++g) {
      half c[O][GW];
      for (int o = 0; o < O; ++o) e1_codes<BITS>(wb[o], g, c[o]);
      for (int t = 0; t < TR; ++t) {
        const device float* xg = xr0 + (size_t)t * NCH * XW + (size_t)(k / VPT) * XW + g * GW;
        float xv[GW];
        if (GW == 4) {
          const float4 v = *((const device float4*)xg);
          xv[0] = v.x; xv[1] = v.y; xv[2] = v.z; xv[3] = v.w;
        } else {
          const float2 v0 = *((const device float2*)xg);
          const float2 v1 = *((const device float2*)(xg + 2));
          const float2 v2 = *((const device float2*)(xg + 4));
          xv[0] = v0.x; xv[1] = v0.y; xv[2] = v1.x; xv[3] = v1.y; xv[4] = v2.x; xv[5] = v2.y;
        }
        for (int o = 0; o < O; ++o) e1_group<BITS>(d[t][o], xv, c[o]);
      }
    }
    for (int t = 0; t < TR; ++t) {
      const float sm = xs[(size_t)(r0 + t) * NCH + k / VPT + int(lane)];
      for (int o = 0; o < O; ++o) res[t][o] += s[o] * d[t][o] + sm * b[o];
    }
  }
  device T* y = seg1 ? y1 : y0;
  const int NSEG = seg1 ? N1 : N0;
  for (int t = 0; t < TR; ++t) {
    if (r0 + t >= R) break;
    for (int o = 0; o < O; ++o) {
      const float v = simd_sum(res[t][o]);
      if (int(lane) == o) y[(size_t)(r0 + t) * NSEG + lrow + o] = static_cast<T>(v);
    }
  }
"""


def enabled() -> bool:
    """``VMLX_M3_EXACT_FLAT`` (default off; read per call)."""
    return os.environ.get("VMLX_M3_EXACT_FLAT", "").strip().lower() in {"1", "true", "on", "yes"}


@lru_cache(maxsize=1)
def _header() -> Optional[str]:
    from .row_exact_qmv import _mlx_qmv_header

    base = _mlx_qmv_header()
    return None if base is None else base + "\n" + _HEADER_EXTRA


_MTYPE = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t"}


def _consts(pairs) -> str:
    """Constants baked into the source (a templated call costs ~2.5 us more per launch in MLX 0.32.3)."""
    return "".join(f"  using {n} = {_MTYPE[v]};\n" if n == "T" else f"  constexpr int {n} = {int(v)};\n"
                   for n, v in pairs)


@lru_cache(maxsize=64)
def _prep_kernel(consts: tuple):
    header = _header()
    if header is None:
        return None
    src = _consts(consts) + _PREP
    tag = hashlib.sha256((header + src).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name=f"vmlx_row_exact_e1_prep_{tag}", input_names=["x"], output_names=["xp", "xs"],
                                header=header, source=src)


@lru_cache(maxsize=256)
def _walk_kernel(consts: tuple):
    header = _header()
    if header is None:
        return None
    src = _consts(consts) + _WALK
    tag = hashlib.sha256((header + src).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name=f"vmlx_row_exact_e1_walk_{tag}",
                                input_names=["xp", "xs", "rv", "w0", "s0", "b0", "w1", "s1", "b1"],
                                output_names=["y0", "y1"], header=header, source=src)


def _vpt(bits: int) -> int:
    return 16 if bits == 4 else 8


def _xw(bits: int) -> int:
    return 12 if bits == 6 else _vpt(bits)


_RV: dict = {}


def _rv(rows: int) -> mx.array:
    a = _RV.get(rows)
    if a is None:
        a = _RV[rows] = mx.zeros((rows,), dtype=mx.int32)
        mx.eval(a)
    return a


_DUMMY: dict = {}


def _dummy(dtype) -> mx.array:
    d = _DUMMY.get(dtype)
    if d is None:
        d = _DUMMY[dtype] = mx.zeros((8,), dtype=dtype)
        mx.eval(d)
    return d


def shape(rows: int, widths: Sequence[int], bits: int) -> tuple:
    """(NSG, O, TR): balanced row tiles of <= TILE_ROWS rows; 6 output rows per simdgroup where every segment
    divides into whole tiles of NSG * 6, else 4 (M5 Max, M3.1 shapes, 8 rows: 4-bit 6 x 4 1.38x the row-split walk,
    6-bit head 1.07x; 8 x 4 / 4 x 8 spill registers). ``VMLX_E1_QMV_SHAPE=nsg,o,tr`` pins it (A/B)."""
    pin = os.environ.get("VMLX_E1_QMV_SHAPE", "").strip()
    if pin:
        nsg, o, tr = (int(v) for v in pin.split(","))
        return nsg, o, tr
    nsg = 8
    n_tiles = -(-rows // TILE_ROWS)
    tr = -(-rows // n_tiles)
    o = 6 if all(w % (nsg * 6) == 0 for w in widths) else 4
    return nsg, o, tr


def _supported(x2d, segs, bits: int, group_size: int) -> bool:
    if bits not in (4, 6, 8) or group_size not in (32, 64, 128):
        return False
    if x2d.dtype not in (mx.float16, mx.bfloat16):
        return False
    rows, k = int(x2d.shape[0]), int(x2d.shape[1])
    if not MIN_ROWS <= rows <= MAX_ROWS or k % (_vpt(bits) * 32) or k % group_size or group_size % _vpt(bits):
        return False
    for w, s, b in segs:
        if w.dtype != mx.uint32 or b is None or s.dtype != x2d.dtype or b.dtype != x2d.dtype:
            return False
        if int(w.shape[-1]) * 32 // bits != k or w.ndim != 2:
            return False
    return 1 <= len(segs) <= 2


def _plan(rows: int, k: int, widths: tuple, bits: int, group_size: int, dtype, nsg: int, o: int, tr: int) -> tuple:
    """Everything a launch needs except the arrays (cached per shape: a verify makes ~120 of these calls)."""
    key = (rows, k, widths, bits, group_size, dtype, nsg, o, tr)
    p = _PLANS.get(key)
    if p is None:
        nrt = -(-rows // tr)
        rp = nrt * tr
        nch = k // _vpt(bits)
        n0 = widths[0]
        n1 = widths[1] if len(widths) > 1 else 0
        p = _PLANS[key] = (
            (("T", dtype), ("BITS", bits), ("K", k)),
            dict(grid=(32, rp, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(rp * nch * _xw(bits),), (rp, nch)], output_dtypes=[mx.float32, mx.float32]),
            (("T", dtype), ("BITS", bits), ("GS", group_size), ("K", k), ("N0", n0), ("N1", n1), ("O", o), ("TR", tr),
             ("NSG", nsg)),
            dict(grid=(32 * nsg * ((n0 + n1) // (nsg * o)) * nrt, 1, 1), threadgroup=(32 * nsg, 1, 1),
                 output_shapes=[(rows, n0), (rows, n1) if n1 else (1,)], output_dtypes=[dtype, dtype]),
            _rv(rows))
    return p


_PLANS: dict = {}


def _launch(x2d, segs, bits: int, group_size: int, nsg: int, o: int, tr: int):
    rows, k = int(x2d.shape[0]), int(x2d.shape[1])
    widths = tuple(int(w.shape[0]) for w, _, _ in segs)
    pc, prep, wc, walk, rv = _plan(rows, k, widths, bits, group_size, x2d.dtype, nsg, o, tr)
    xp, xs = _prep_kernel(pc)(inputs=[x2d], **prep)
    (w0, s0, b0) = segs[0]
    (w1, s1, b1) = segs[1] if len(segs) > 1 else (_dummy(mx.uint32), _dummy(x2d.dtype), _dummy(x2d.dtype))
    y0, y1 = _walk_kernel(wc)(inputs=[xp, xs, rv, w0, s0, b0, w1, s1, b1], **walk)
    return [y0] if len(segs) == 1 else [y0, y1]


def _probes(rows: int, k: int, dtype) -> list:
    zero_mean = mx.random.normal((rows, k), key=mx.random.key(4321)) * 0.5
    shift = mx.random.uniform(-3.0, 3.0, (rows, 1), key=mx.random.key(4322))
    spikes = (mx.random.uniform(shape=(rows, k), key=mx.random.key(4323)) < 0.01) * 30.0
    scale = mx.exp(mx.random.uniform(-4.0, 4.0, (rows, 1), key=mx.random.key(4324)))
    return [zero_mean.astype(dtype), ((zero_mean + shift + spikes) * scale).astype(dtype)]


def _admit(x2d, segs, bits: int, group_size: int):
    """(nsg, o, tr) when this shape's E1 call reproduces stock one-row calls on these weights, else False."""
    rows, k = int(x2d.shape[0]), int(x2d.shape[1])
    if _header() is None or not _supported(x2d, segs, bits, group_size):
        return False
    widths = tuple(int(w.shape[0]) for w, _, _ in segs)
    nsg, o, tr = shape(rows, widths, bits)
    if any(wd % (nsg * o) for wd in widths):
        return False
    ok = True
    for probe in _probes(rows, k, x2d.dtype):
        got = _launch(probe, segs, bits, group_size, nsg, o, tr)
        for (w, s, b), g in zip(segs, got):
            want = mx.concatenate([
                mx.quantized_matmul(probe[r:r + 1], w, s, b, transpose=True, group_size=group_size, bits=bits,
                                    mode="affine") for r in range(rows)], axis=0)
            ok = ok and bool(mx.array_equal(g, want).item())
    STATS["admitted" if ok else "rejected"] += 1
    (logger.info if ok else logger.warning)(
        "row-exact E1 QMV %s rows=%d K=%d N=%s bits=%d gs=%d dtype=%s shape=%s",
        "admitted" if ok else "REJECTED (stock one-row arithmetic not reproduced; caller keeps its path)",
        rows, k, widths, bits, group_size, x2d.dtype, (nsg, o, tr))
    return (nsg, o, tr) if ok else False


def row_exact_e1(x, segments, *, bits: int, group_size: int) -> Optional[list]:
    """``[x @ dequant(W_i).T for each segment]`` (segments: (weight, scales, biases) sharing x, bits and group),
    every row bit-equal to MLX's one-row call, or None (unsupported / not admitted: keep the caller's exact path).
    The decision (supported + admitted, tile shape) is cached per (rows, K, widths, bits, group, x / scale dtype, tile
    pin): a verify makes ~120 of these calls, so the repeat path is one lookup and two launches."""
    shp = x.shape
    if len(shp) < 2:
        return None
    k = int(shp[-1])
    rows = 1
    for dim in shp[:-1]:
        rows *= int(dim)
    if rows > MAX_ROWS:
        # wider windows: balanced chunks of <= MAX_ROWS rows, each its own walk (every row still exact)
        x2d = x.reshape(rows, k)
        n = -(-rows // MAX_ROWS)
        base, extra = divmod(rows, n)
        parts, start = [], 0
        for c in range(n):
            size = base + (1 if c < extra else 0)
            got = row_exact_e1(x2d[start:start + size], segments, bits=bits, group_size=group_size)
            if got is None:
                return None
            parts.append(got)
            start += size
        lead = tuple(shp[:-1])
        return [mx.concatenate([p[i] for p in parts], axis=0).reshape(*lead, -1) for i in range(len(segments))]
    if rows < MIN_ROWS:
        return None
    segs = segments if isinstance(segments[0], tuple) else [tuple(t) for t in segments]
    key = (rows, k, tuple(int(w.shape[0]) for w, _, _ in segs), bits, group_size, x.dtype, segs[0][1].dtype,
           os.environ.get("VMLX_E1_QMV_SHAPE", ""))
    dec = _DECISIONS.get(key)
    x2d = x if len(shp) == 2 else x.reshape(rows, k)
    if dec is None:
        dec = _DECISIONS[key] = _admit(x2d, segs, bits, group_size)
        _ADMITTED[key] = bool(dec)
    if not dec:
        return None
    STATS["calls"] += 1
    outs = _launch(x2d, segs, bits, group_size, *dec)
    if len(shp) == 2:
        return outs
    lead = tuple(shp[:-1])
    return [y.reshape(*lead, int(y.shape[-1])) for y in outs]

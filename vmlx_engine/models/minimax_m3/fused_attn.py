# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 fused attention half for decode and DSpark verify (1 .. 16 rows): five Metal launches per layer.

``VMLX_M3_FUSED_ATTN=1`` (default off) routes a decoder layer's attention half -- ``x + o_proj(MSA(norm(x)))`` -- to

* K1 ``attn_qkv``: input RMSNorm fused into ONE projection launch for q | k | v (affine) and index q | k (MXFP8 or
  affine): threadgroups partitioned by weight, the norm's sum of squares accumulated once per row on the fly;
* K2 ``attn_heads``: q / k / index-q / index-k head norms + partial RoPE + casts + the K / V / index-key appends into
  the ``MiniMaxM3SparseCache`` lanes (in place while the cache owns its lane buffers, else MLX slice updates);
* K3 ``attn_scores``: indexer block scores of every (row, KV group), only when some row is past the dense threshold
  (``index_topk_blocks * index_block_size`` keys);
* K4 ``attn_msa``: per-(row, group) top-k selection + MSA attention over the selected blocks + the merge;
* K5 ``attn_oproj``: o_proj + residual add.

Every row's arithmetic is independent of the other rows of the launch, so a decode step (one row) and the same row in
a DSpark verify (up to 16 rows) are bit-identical; they are NOT bit-identical to the unfused ops (fp32 throughout,
fp32 ``1 + w``, one rounding per output). The fused path covers B = 1, 1 .. 16 rows, sparse layers (an indexer whose
heads are the KV groups) with affine q / k / v / o and MXFP8 or affine index projections; everything else (prefill
chunks, batches, dense layers, other formats) runs today's ops. Metal sources: ``kernels/attn_*.metal``.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from . import kernels as K

ENV = "VMLX_M3_FUSED_ATTN"
MAX_ROWS = 16
# K1 / K5 threadgroup shapes. Row tiles ride the grid (threadgroup = weight tile x row tile) and every output element
# runs the same per-chunk arithmetic whatever the shape (kernels/attn_common.metal m3a_rowsN), so a row's output does
# not depend on the shape or the row count. One row: NSG 2 simdgroups x RPS 4 output rows (MLX qmv's shape). Several
# rows: RT 2 rows per threadgroup (each output chunk's codes decoded once for both; 4 runs out of registers), 4 x 8
# (each row's input loaded by fewer threadgroups, its per-chunk load amortized over more outputs). Measured on the
# M5 Max at M3.1 shapes, 2 .. 16 rows: (4, 8) ~ (8, 8) < (8, 4) < (2, 4).
PROJ_NSG, PROJ_RPS = 2, 4
PROJ_RT = 2
MULTI_NSG, MULTI_RPS = 4, 8


def _proj_shape(rows: int) -> tuple:
    """(NSG, RPS, RT) of the K1 / K5 build for ``rows``."""
    return (PROJ_NSG, PROJ_RPS, 1) if rows == 1 else (MULTI_NSG, MULTI_RPS, PROJ_RT)


# E1 multi-row K1 / K5 (``VMLX_M3_EXACT_FLAT=1``, default off until measured on the real bundle): for 2 .. 16 rows a
# prep launch computes each row's input chunks once (attn_xprep: the one-row build's own m3a_load_chunk), then the
# projection walks each weight ONCE per tile of up to E1_TR rows, decoding every four-value code group once into half
# registers and applying it to each row with the one-row build's per-element sequence (attn_common.metal m3e_rows).
# Verify rows stay bit-identical to decode steps; the one-row builds (decode) are not touched. Shapes measured on the
# M5 Max at M3.1 shapes (o_proj 8192 -> 6144, rows 4 / 8 / 16 vs the 2-row build): 6 outputs x 4 rows per lane 1.4 /
# 1.44 / 1.6x faster, 4 x 4 1.3x; wider tiles (4 x 8, 8 x 4) spill registers and run 2x slower.
EXACT_FLAT_ENV = "VMLX_M3_EXACT_FLAT"
E1_NSG, E1_TR = 8, 4
E1_STATS = {"qkv": 0, "oproj": 0}


def exact_flat() -> bool:
    return os.environ.get(EXACT_FLAT_ENV, "").strip().lower() in {"1", "true", "on", "yes"}


def _e1_ok(p) -> bool:
    """E1 builds cover 4-bit affine q / k / v / o and an MXFP8 or 4-bit affine index (16-value lane chunks)."""
    return p.bits_a == 4 and (p.fmt_b == 1 or p.bits_b == 4) and p.bits_o == 4


def _e1_shape(rows: int, widths) -> tuple:
    """(NSG, O, TR) of an E1 build: balanced row tiles of <= E1_TR rows; 6 outputs per simdgroup where every segment
    divides into whole tiles of NSG * 6, else 4. ``VMLX_M3_E1_SHAPE=nsg,o,tr`` pins it (A/B)."""
    pin = os.environ.get("VMLX_M3_E1_SHAPE", "").strip()
    if pin:
        nsg, o, tr = (int(v) for v in pin.split(","))
        return nsg, o, tr
    n_tiles = -(-rows // E1_TR)
    tr = -(-rows // n_tiles)
    o = 6 if all(w % (E1_NSG * 6) == 0 for w in widths) else 4
    return E1_NSG, o, tr


# Diagnostics: launches, rows, appends written in place vs through MLX slice updates, launches that ran K3.
STATS = {"calls": 0, "rows": 0, "inplace": 0, "copy_appends": 0, "sparse_calls": 0}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in {"1", "true", "on", "yes"}


def _metal() -> bool:
    try:
        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


def _promote(a, b):
    return (mx.zeros((1,), dtype=a) + mx.zeros((1,), dtype=b)).dtype


_DUMMY: dict = {}


def _dummy(dtype=mx.float32) -> mx.array:
    d = _DUMMY.get(dtype)
    if d is None:
        d = _DUMMY[dtype] = mx.zeros((8,), dtype=dtype)
        mx.eval(d)
    return d


# ------------------------------------------------------------------------------------------------ per-layer plan
class Plan:
    """Static facts of one layer's attention the kernels are built for (``plan(layer)``)."""

    __slots__ = ("hidden", "H", "KVH", "GI", "D", "ROT", "log2base", "eps_in", "eps_head", "block", "keep", "local",
                 "bits_a", "gs_a", "fmt_b", "bits_b", "gs_b", "bits_o", "gs_o", "scale", "HG", "HS")


def _affine(m) -> Optional[tuple]:
    if not isinstance(m, nn.QuantizedLinear) or "bias" in m or "biases" not in m:
        return None
    if str(getattr(m, "mode", "affine") or "affine") != "affine":
        return None
    bits, gs = int(m.bits), int(m.group_size)
    if bits not in (4, 8) or gs not in (32, 64, 128) or m["scales"].dtype != m["biases"].dtype:
        return None
    return bits, gs


def _mxfp8(m) -> bool:
    return (isinstance(m, nn.QuantizedLinear) and "bias" not in m and str(getattr(m, "mode", "")) == "mxfp8"
            and int(m.bits) == 8 and int(m.group_size) == 32)


def _out_in(m) -> tuple:
    """(output rows, input dims) of a quantized linear."""
    w = m["weight"]
    bits = int(m.bits)
    return int(w.shape[0]), int(w.shape[1]) * 32 // bits


def _build_plan(layer) -> tuple:
    """(Plan, None) or (None, reason)."""
    attn = getattr(layer, "self_attn", None)
    norm = getattr(layer, "input_layernorm", None)
    ix = getattr(attn, "indexer", None)
    if attn is None or norm is None or ix is None:
        return None, "no sparse attention (indexer) on this layer"
    p = Plan()
    try:
        p.H, p.KVH, p.D = int(attn.n_heads), int(attn.n_kv), int(attn.hd)
        p.hidden = int(norm.weight.shape[0])
        p.eps_in = float(norm.eps)
        p.GI, p.block, p.keep, p.local = int(ix.nh), int(ix.block), int(ix.topk), int(ix.local)
        p.scale = float(attn.scale)
    except Exception as exc:  # pragma: no cover - foreign module layout
        return None, f"layout: {exc}"
    qkv = [_affine(m) for m in (attn.q_proj, attn.k_proj, attn.v_proj)]
    if any(f is None for f in qkv) or len(set(qkv)) != 1:
        return None, "q / k / v not one affine (4 / 8-bit) format"
    p.bits_a, p.gs_a = qkv[0]
    if _mxfp8(ix.index_q_proj) and _mxfp8(ix.index_k_proj):
        p.fmt_b, p.bits_b, p.gs_b = 1, 8, 32
    else:
        ib = [_affine(m) for m in (ix.index_q_proj, ix.index_k_proj)]
        if any(f is None for f in ib) or len(set(ib)) != 1:
            return None, "index q / k neither MXFP8 nor one affine format"
        p.fmt_b, (p.bits_b, p.gs_b) = 2, ib[0]
    o = _affine(attn.o_proj)
    if o is None:
        return None, "o_proj not affine 4 / 8-bit"
    p.bits_o, p.gs_o = o
    shapes = {
        "q": (_out_in(attn.q_proj), (p.H * p.D, p.hidden)),
        "k": (_out_in(attn.k_proj), (p.KVH * p.D, p.hidden)),
        "v": (_out_in(attn.v_proj), (p.KVH * p.D, p.hidden)),
        "iq": (_out_in(ix.index_q_proj), (p.GI * p.D, p.hidden)),
        "ik": (_out_in(ix.index_k_proj), (p.D, p.hidden)),
        "o": (_out_in(attn.o_proj), (p.hidden, p.H * p.D)),
    }
    for name, (got, want) in shapes.items():
        if got != want:
            return None, f"{name} shape {got} != {want} (index head dim must equal the head dim)"
    blk_a = 512 if p.bits_a == 4 else 256
    blk_b = 512 if p.fmt_b == 1 or p.bits_b == 4 else 256
    blk_o = 512 if p.bits_o == 4 else 256
    rows_tg = max(_proj_shape(1)[0] * _proj_shape(1)[1], _proj_shape(2)[0] * _proj_shape(2)[1])
    if (p.hidden % blk_a or p.hidden % blk_b or (p.H * p.D) % blk_o or p.hidden % rows_tg
            or any(n % rows_tg for n in (p.H * p.D, p.KVH * p.D, p.GI * p.D, p.D))):
        return None, "projection widths not multiples of the kernel blocks"
    if p.D % 32 or p.D > 256 or p.KVH < 1 or p.H % p.KVH or p.GI != p.KVH:
        return None, "head geometry (head dim multiple of 32 <= 256, index heads = KV groups)"
    p.HG = p.H // p.KVH
    p.HS = 4 if p.HG % 4 == 0 else p.HG
    if p.HS > 8 or p.HG * p.D * 4 > 16384:
        return None, "query heads per KV group"
    cpb = p.block // 32
    if (p.block % 32 or cpb & (cpb - 1) or p.keep < 1 or p.local < 0 or (p.HG // p.HS) * cpb > 32
            or 4 * (p.HG * p.D + max(1, cpb // 2) * p.HG * (p.D + 2)) + 1400 + 4 * p.keep > 32768):
        return None, "index block geometry (32-key chunks per block: a power of two, threadgroup memory)"
    ropes = (attn.rope, ix.rope)
    for rope in ropes:
        if bool(getattr(rope, "traditional", True)) or float(getattr(rope, "scale", 1.0)) != 1.0:
            return None, "RoPE not the non-traditional unscaled form"
    p.ROT = int(attn.rope.dims)
    if int(ix.rope.dims) != p.ROT or float(ix.rope.base) != float(attn.rope.base):
        return None, "index RoPE differs from the attention RoPE"
    if p.ROT % 2 or p.ROT > p.D or not ((p.ROT // 2) % 32 == 0 or p.ROT <= 32):
        return None, "rotary dims"
    p.log2base = float(math.log2(float(attn.rope.base)))
    eps = {float(m.eps) for m in (attn.q_norm, attn.k_norm, ix.index_q_norm, ix.index_k_norm)}
    if len(eps) != 1:
        return None, "head norms with different eps"
    p.eps_head = eps.pop()
    return p, None


def plan(layer) -> Optional[Plan]:
    attn = layer.self_attn
    cached = attn.__dict__.get("_m3_fused_plan")
    if cached is None:
        built, reason = _build_plan(layer)
        cached = built if built is not None else reason
        attn.__dict__["_m3_fused_plan"] = cached
    return cached if isinstance(cached, Plan) else None


def unsupported_reason(layer) -> Optional[str]:
    """Why ``layer`` cannot take the fused path (None: it can, shape permitting)."""
    if plan(layer) is not None:
        return None
    return layer.self_attn.__dict__.get("_m3_fused_plan")


def use_fused(layer, x: mx.array, mask: Any, cache: Any, *, force: bool = False) -> bool:
    """True when this call takes the fused attention: the lever is on (or ``force``), one sequence of 1 .. 16 rows,
    no explicit mask array, an M3 sparse cache that supports in-lane appends, a supported layer, a Metal GPU."""
    if not (force or enabled()):
        return False
    if x.ndim != 3 or int(x.shape[0]) != 1 or not 1 <= int(x.shape[1]) <= MAX_ROWS:
        return False
    if isinstance(mask, mx.array) or cache is None or not hasattr(cache, "fused_reserve"):
        return False
    if isinstance(getattr(cache, "offset", None), mx.array) or x.dtype not in K.METAL_TYPE:
        return False
    if not _metal():
        return False
    return plan(layer) is not None


# ------------------------------------------------------------------------------------------------ launches
def _t(a: mx.array) -> str:
    return K.METAL_TYPE[a.dtype]


def qkv(layer, p: Plan, x2: mx.array) -> mx.array:
    """K1: [R, hidden] residual rows -> [R, (H + 2 KVH + GI + 1) * D] float32 (q | k | v | index q | index k)."""
    attn, ix = layer.self_attn, layer.self_attn.indexer
    R = int(x2.shape[0])
    if R > 1 and exact_flat() and _e1_ok(p):
        return qkv_e1(layer, p, x2)
    q, k, v, iq, ik = attn.q_proj, attn.k_proj, attn.v_proj, ix.index_q_proj, ix.index_k_proj
    nq, nk, nv, niq, nik = p.H * p.D, p.KVH * p.D, p.KVH * p.D, p.GI * p.D, p.D
    nsg, rps, rt = _proj_shape(R)
    sbt = _t(iq["scales"]) if p.fmt_b == 2 else "float"
    kern = K.kernel(
        "qkv",
        (("xt", _t(x2)), ("nwt", _t(layer.input_layernorm.weight)), ("sat", _t(q["scales"])), ("sbt", sbt)),
        (("KIN", p.hidden), ("NQ", nq), ("NK", nk), ("NV", nv), ("NIQ", niq), ("NIK", nik), ("BITS_A", p.bits_a),
         ("GS_A", p.gs_a), ("FMT_B", p.fmt_b), ("BITS_B", p.bits_b), ("GS_B", p.gs_b), ("NSG", nsg), ("RPS", rps),
         ("RT", rt)),
        (("EPS", p.eps_in),))
    if p.fmt_b == 1:
        iqb, ikb = _dummy(), _dummy()
    else:
        iqb, ikb = iq["biases"], ik["biases"]
    ntot = nq + nk + nv + niq + nik
    return kern(
        inputs=[x2, layer.input_layernorm.weight, q["weight"], q["scales"], q["biases"], k["weight"], k["scales"],
                k["biases"], v["weight"], v["scales"], v["biases"], iq["weight"], iq["scales"], iqb, ik["weight"],
                ik["scales"], ikb],
        grid=(32 * nsg * (ntot // (nsg * rps)) * -(-R // rt), 1, 1), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(R, ntot)], output_dtypes=[mx.float32])[0]


def xprep(x2: mx.array, nw: Optional[mx.array], eps: float, two: bool, rp: int):
    """(xa, xb or None, xs, inv): each row's input chunks for the E1 builds (kernels/attn_xprep.metal), zero-padded to
    ``rp`` rows (whole row tiles)."""
    kin = int(x2.shape[1])
    norm = nw is not None
    kern = K.kernel("xprep", (("xt", _t(x2)), ("nwt", _t(nw) if norm else "float")),
                    (("KIN", kin), ("NORM", int(norm)), ("TWO", int(two))), (("EPS", float(eps)),))
    xa, xb, xs, inv = kern(inputs=[x2, nw if norm else _dummy()], grid=(32, rp, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(rp, kin), (rp, kin) if two else (1,), (rp, kin // 16), (max(8, rp),)],
                           output_dtypes=[mx.float32] * 4)
    return xa, (xb if two else None), xs, inv


_RV: dict = {}


def _rv(rows: int) -> mx.array:
    """A cached [rows] array: the E1 K1 build reads only its length (the valid row count)."""
    a = _RV.get(rows)
    if a is None:
        a = _RV[rows] = mx.zeros((rows,), dtype=mx.int32)
        mx.eval(a)
    return a


def qkv_e1(layer, p: Plan, x2: mx.array) -> mx.array:
    """K1 for 2 .. 16 rows, E1 build: bit-identical per row to ``qkv``'s one-row build."""
    attn, ix = layer.self_attn, layer.self_attn.indexer
    R = int(x2.shape[0])
    q, k, v, iq, ik = attn.q_proj, attn.k_proj, attn.v_proj, ix.index_q_proj, ix.index_k_proj
    nq, nk, nv, niq, nik = p.H * p.D, p.KVH * p.D, p.KVH * p.D, p.GI * p.D, p.D
    nsg, o, tr = _e1_shape(R, (nq, nk, nv, niq, nik))
    nrt = -(-R // tr)
    xa, xb, xs, inv = xprep(x2, layer.input_layernorm.weight, p.eps_in, two=p.fmt_b == 1, rp=nrt * tr)
    sbt = _t(iq["scales"]) if p.fmt_b == 2 else "float"
    kern = K.kernel(
        "qkv_e1", (("sat", _t(q["scales"])), ("sbt", sbt)),
        (("KIN", p.hidden), ("NQ", nq), ("NK", nk), ("NV", nv), ("NIQ", niq), ("NIK", nik), ("GS_A", p.gs_a),
         ("FMT_B", p.fmt_b), ("GS_B", p.gs_b), ("NSG", nsg), ("O", o), ("TR", tr)))
    iqb, ikb = (_dummy(), _dummy()) if p.fmt_b == 1 else (iq["biases"], ik["biases"])
    ntot = nq + nk + nv + niq + nik
    E1_STATS["qkv"] += 1
    return kern(
        inputs=[xa, xb if xb is not None else xa, xs, inv, _rv(R), q["weight"], q["scales"], q["biases"], k["weight"],
                k["scales"], k["biases"], v["weight"], v["scales"], v["biases"], iq["weight"], iq["scales"], iqb,
                ik["weight"], ik["scales"], ikb],
        grid=(32 * nsg * (ntot // (nsg * o)) * nrt, 1, 1), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(R, ntot)], output_dtypes=[mx.float32])[0]


def oproj_e1(layer, p: Plan, o: mx.array, x2: mx.array) -> mx.array:
    """K5 for 2 .. 16 rows, E1 build: bit-identical per row to ``oproj``'s one-row build."""
    m = layer.self_attn.o_proj
    R = int(o.shape[0])
    nsg, oo, tr = _e1_shape(R, (p.hidden,))
    nrt = -(-R // tr)
    xa, _, xs, _ = xprep(o, None, 0.0, two=False, rp=nrt * tr)
    kern = K.kernel(
        "oproj_e1", (("xt", _t(x2)), ("st", _t(m["scales"]))),
        (("KIN", p.H * p.D), ("NOUT", p.hidden), ("GS", p.gs_o), ("NSG", nsg), ("O", oo), ("TR", tr)))
    E1_STATS["oproj"] += 1
    return kern(inputs=[xa, xs, x2, m["weight"], m["scales"], m["biases"]],
                grid=(32 * nsg * (p.hidden // (nsg * oo)) * nrt, 1, 1), threadgroup=(32 * nsg, 1, 1),
                output_shapes=[(R, p.hidden)], output_dtypes=[x2.dtype])[0]


def heads(layer, p: Plan, y: mx.array, meta: mx.array, lanes, lane_dtypes, inplace: bool):
    """K2: (q [R, H, D] f32, index q [R, GI, D] f32, counters, new K / V / index rows or None)."""
    attn, ix = layer.self_attn, layer.self_attn.indexer
    R = int(y.shape[0])
    kdt, vdt, idt = lane_dtypes
    cnt = max(8, R * p.KVH)
    kern = K.kernel(
        "heads",
        (("kt", K.METAL_TYPE[kdt]), ("vt", K.METAL_TYPE[vdt]), ("it", K.METAL_TYPE[idt]),
         ("qnt", _t(attn.q_norm.weight)), ("knt", _t(attn.k_norm.weight)), ("iqnt", _t(ix.index_q_norm.weight)),
         ("iknt", _t(ix.index_k_norm.weight))),
        (("H", p.H), ("KVH", p.KVH), ("GI", p.GI), ("D", p.D), ("ROT", p.ROT), ("INPLACE", int(inplace)),
         ("CNT", cnt)),
        (("EPS", p.eps_head), ("LOG2BASE", p.log2base)))
    ht = p.H + 2 * p.KVH + p.GI + 1
    threads = 32 * R * ht
    if inplace:
        kc, vc, ic = lanes
        shapes = [(R, p.H, p.D), (R, p.GI, p.D), (cnt,), (1,), (1,), (1,)]
    else:
        kc = vc = ic = _dummy()
        shapes = [(R, p.H, p.D), (R, p.GI, p.D), (cnt,), (R, p.KVH, p.D), (R, p.KVH, p.D), (R, 1, p.D)]
    out = kern(
        inputs=[y, attn.q_norm.weight, attn.k_norm.weight, ix.index_q_norm.weight, ix.index_k_norm.weight, meta,
                kc, vc, ic],
        grid=(-(-threads // 128) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=shapes,
        output_dtypes=[mx.float32, mx.float32, mx.uint32, kdt, vdt, idt])
    return out[0], out[1], out[2], (None if inplace else (out[3], out[4], out[5]))


def scores(p: Plan, iq: mx.array, idx_lane: mx.array, meta: mx.array, nb: int) -> mx.array:
    """K3: block scores [R, GI, nb] float32."""
    R = int(iq.shape[0])
    pairs = R * p.GI
    pt = 4 if pairs <= 4 else (8 if pairs <= 8 else 16)    # pairs per pass: a pair's arithmetic does not depend on it
    kern = K.kernel(
        "scores", (("it", K.METAL_TYPE[idx_lane.dtype]),),
        (("GI", p.GI), ("D", p.D), ("BLOCK", p.block), ("LOCAL", p.local), ("PT", pt)))
    return kern(inputs=[iq, idx_lane, meta], grid=(p.block * nb, 1, 1), threadgroup=(p.block, 1, 1),
                output_shapes=[(R, p.GI, nb)], output_dtypes=[mx.float32])[0]


def msa(p: Plan, q: mx.array, k_lane: mx.array, v_lane: mx.array, sc: mx.array, cnt: mx.array,
        meta: mx.array, partials: bool = False):
    """K4: attention output [R, H, D] float32 over each row's selected blocks (``partials``: also its block partials
    (po, pm, pl), for tests)."""
    R = int(q.shape[0])
    nsg = p.HG // p.HS
    threads = 32 * nsg * (p.block // 32)
    kern = K.kernel(
        "msa", (("kt", K.METAL_TYPE[k_lane.dtype]), ("vt", K.METAL_TYPE[v_lane.dtype])),
        (("G", p.KVH), ("HG", p.HG), ("D", p.D), ("BLOCK", p.block), ("KEEP", p.keep), ("HS", p.HS),
         ("NSG", nsg)))
    rg = R * p.KVH
    out = kern(
        inputs=[q, k_lane, v_lane, sc, cnt, meta, mx.array([p.scale], dtype=mx.float32)],
        grid=(threads * p.keep, rg, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(R, p.H, p.D), (rg * p.keep * p.HG * p.D,), (rg * p.keep * p.HG,), (rg * p.keep * p.HG,)],
        output_dtypes=[mx.float32] * 4)
    return (out[0], tuple(out[1:])) if partials else out[0]


def oproj(layer, p: Plan, o: mx.array, x2: mx.array, ssp: bool = False):
    """K5: x2 + o_proj(o) [R, hidden] in x2's dtype (and the sum-of-squares partials [R, hidden / 8] when ssp)."""
    m = layer.self_attn.o_proj
    R = int(o.shape[0])
    if R > 1 and not ssp and exact_flat() and _e1_ok(p):
        return oproj_e1(layer, p, o, x2)
    nsg, rps, rt = _proj_shape(R)
    kern = K.kernel(
        "oproj", (("xt", _t(x2)), ("st", _t(m["scales"]))),
        (("KIN", p.H * p.D), ("NOUT", p.hidden), ("BITS", p.bits_o), ("GS", p.gs_o), ("NSG", nsg), ("RPS", rps),
         ("RT", rt), ("SSP", int(ssp))))
    out = kern(inputs=[o, x2, m["weight"], m["scales"], m["biases"]],
               grid=(32 * nsg * (p.hidden // (nsg * rps)) * -(-R // rt), 1, 1), threadgroup=(32 * nsg, 1, 1),
               output_shapes=[(R, p.hidden), (R, p.hidden // 8) if ssp else (1,)],
               output_dtypes=[x2.dtype, mx.float32])
    return (out[0], out[1]) if ssp else out[0]


def lane_dtypes(layer, cache, x_dtype):
    """(K, V, index-key) lane dtypes: the lanes' own when they exist, else what today's ops would append."""
    keys, values, idx = cache.fused_lanes()
    attn, ix = layer.self_attn, layer.self_attn.indexer
    kdt = keys.dtype if keys is not None else _promote(x_dtype, attn.k_norm.weight.dtype)
    vdt = values.dtype if values is not None else x_dtype
    idt = idx.dtype if idx is not None else _promote(x_dtype, ix.index_k_norm.weight.dtype)
    return kdt, vdt, idt


# ------------------------------------------------------------------------------------------------ forward
def attention_block(layer, x: mx.array, cache) -> mx.array:
    """``x + self_attn(input_layernorm(x))`` for one sequence of 1 .. 16 rows at ``cache.offset`` (appends the rows'
    K / V / index keys to ``cache``). Call only when ``use_fused`` said so."""
    p = plan(layer)
    _, R, hidden = x.shape
    x2 = x.reshape(R, hidden)
    offset = int(cache.offset)
    y = qkv(layer, p, x2)
    dts = lane_dtypes(layer, cache, x.dtype)
    inplace = cache.fused_reserve(R, p.KVH, p.D, dts[0], dts[1], p.D, dts[2])
    nb = -(-(offset + R) // p.block)                     # key blocks of the last row (the widest extent)
    meta = mx.array([offset, R, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
    q, iq, cnt, rows = heads(layer, p, y, meta, cache.fused_lanes(), dts, inplace)
    if rows is not None:
        k_new, v_new, i_new = rows
        cache.fused_append(k_new.transpose(1, 0, 2)[None], v_new.transpose(1, 0, 2)[None],
                           i_new.transpose(1, 0, 2)[None])
        STATS["copy_appends"] += 1
    else:
        STATS["inplace"] += 1
    k_lane, v_lane, i_lane = cache.fused_lanes()
    if nb > p.keep:
        sc = scores(p, iq, i_lane, meta, nb)
        STATS["sparse_calls"] += 1
    else:
        sc = _dummy()
    o = msa(p, q, k_lane, v_lane, sc, cnt, meta)
    # K4 ran behind the barrier MLX puts after K2 (it reads K2's outputs): later readers of the lanes are ordered
    # after it, so they see the rows K2 wrote in place.
    cache.fused_commit(R, after=o)
    h = oproj(layer, p, o.reshape(R, p.H * p.D), x2)
    STATS["calls"] += 1
    STATS["rows"] += R
    return h.reshape(1, R, hidden)

# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 / M3.1 KV cache formats: the K / V lanes and the Lightning-indexer key lane stored as packed rows.

Levers (default off = today's bf16 lanes, byte for byte):

* ``VMLX_M3_KV_DTYPE``       main K / V lanes:      bf16 | mxfp8 | nvfp4 | affine8 | affine4
* ``VMLX_M3_INDEX_KV_DTYPE`` index-key lane (K only): bf16 | mxfp8 | nvfp4

MiniMax's own serving recipes keep both lanes in a 4-bit float format (FP8 on GPUs without FP4 units), with scales
computed on the fly (the checkpoint ships no KV scales). This module gives MLX the same two classes: ``mxfp8`` (e4m3
codes, a power-of-two scale per 32 dims) and ``nvfp4`` (e2m1 codes, an e4m3 scale per 16 dims, plus a power-of-two
exponent per row that keeps every block scale in the e4m3 normal range); ``affine8`` / ``affine4`` (64-dim groups,
bf16 scale + bias) are references. Layout and rounding: ``kernels/kv_quant.metal``.

A quantized lane is a uint8 array [B, heads, capacity, RB] with one packed row per (token, head): RB bytes of codes +
scales. Groups run along the head dim, so a token's row depends on that token alone: decode steps, DSpark verify rows
and prefill chunks produce the same bytes for the same values, and every token-axis operation the cache tiers do
(slice, trim, concatenate, store / restore blocks) is unchanged. Readers decode only what they read: the unfused path
decodes the visible extent into the nominal dtype (``dequantize``); the fused decode / verify kernels decode the rows
they touch in registers (K3 every index key, K4 only the selected blocks). Format identity: a lane's dtype says
whether it is quantized; the cache object (``MiniMaxM3SparseCache``) names the format, and persistent tiers are
namespaced by the configured formats (``cache_identity``), so rows written under one format are never read under
another.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

KV_ENV = "VMLX_M3_KV_DTYPE"
INDEX_ENV = "VMLX_M3_INDEX_KV_DTYPE"


@dataclass(frozen=True)
class Fmt:
    name: str
    code: int          # the F constant of kernels/kv_quant.metal (0 = unquantized)
    group: int         # dims per scale group (0: unquantized)

    @property
    def quantized(self) -> bool:
        return self.code != 0

    def code_bytes(self, d: int) -> int:
        return d if self.code in (1, 3) else d // 2

    def row_bytes(self, d: int) -> int:
        """Packed bytes per (token, head) row of head dim ``d`` (0 when unquantized)."""
        if not self.quantized:
            return 0
        meta = {1: d // 32, 2: d // 16 + 1}.get(self.code, 4 * (d // 64))
        return -(-(self.code_bytes(d) + meta) // 4) * 4

    def supports(self, d: int) -> bool:
        return not self.quantized or (d % 32 == 0 and d % max(self.group, 1) == 0 and d <= 1024)

    def head_dim(self, row_bytes: int) -> Optional[int]:
        """The head dim whose packed row is ``row_bytes`` wide (None when no supported head dim gives it)."""
        return _head_dim(self, int(row_bytes))


BF16 = Fmt("bf16", 0, 0)
MXFP8 = Fmt("mxfp8", 1, 32)
NVFP4 = Fmt("nvfp4", 2, 16)
AFFINE8 = Fmt("affine8", 3, 64)
AFFINE4 = Fmt("affine4", 4, 64)
FORMATS = {f.name: f for f in (BF16, MXFP8, NVFP4, AFFINE8, AFFINE4)}
KV_FORMATS = ("bf16", "mxfp8", "nvfp4", "affine8", "affine4")
INDEX_FORMATS = ("bf16", "mxfp8", "nvfp4")
_ALIASES = {"": "bf16", "none": "bf16", "off": "bf16", "0": "bf16", "bfloat16": "bf16", "native": "bf16",
            "fp8": "mxfp8", "e4m3": "mxfp8", "fp4": "nvfp4", "q8": "affine8", "q4": "affine4"}


@lru_cache(maxsize=None)
def _head_dim(fmt: Fmt, row_bytes: int) -> Optional[int]:
    for d in range(32, 1025, 32):
        if fmt.supports(d) and fmt.row_bytes(d) == row_bytes:
            return d
    return None


def get_code(code: int) -> Fmt:
    for f in FORMATS.values():
        if f.code == int(code):
            return f
    raise KeyError(code)


def get(name) -> Fmt:
    if isinstance(name, Fmt):
        return name
    key = str(name or "").strip().lower()
    return FORMATS[_ALIASES.get(key, key)]


_WARNED: set = set()


def _parse(env: str, allowed: Tuple[str, ...]) -> Fmt:
    raw = os.environ.get(env, "")
    key = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
    if key in allowed:
        return FORMATS[key]
    if (env, raw) not in _WARNED:
        _WARNED.add((env, raw))
        logger.warning("%s=%r is not one of %s; keeping bf16 cache lanes", env, raw, ", ".join(allowed))
    return BF16


def configured() -> Tuple[Fmt, Fmt]:
    """(K / V format, index-key format) from the levers (read on every call; the process sets them at start)."""
    return _parse(KV_ENV, KV_FORMATS), _parse(INDEX_ENV, INDEX_FORMATS)


def cache_identity() -> str:
    """The persistent-cache namespace part for the configured formats ('' for the default bf16 lanes, so existing
    namespaces are unchanged)."""
    kv, idx = configured()
    if not kv.quantized and not idx.quantized:
        return ""
    return f"minimax_m3_kv={kv.name},minimax_m3_index_kv={idx.name},kvq_layout=v1"


def lane_format(lane, fmt: Fmt, head_dim: Optional[int] = None) -> Fmt:
    """The format a lane array holds: BF16 for a floating lane, else ``fmt`` (the cache's, or the configured one when
    the cache names none), checked against the lane's row width. Raises ValueError on a width the format cannot
    produce (rows written under another layout)."""
    if lane is None or lane.dtype != mx.uint8:
        return BF16
    if not fmt.quantized:
        raise ValueError("uint8 cache lane without a quantized format")
    rb = int(lane.shape[-1])
    d = fmt.head_dim(rb)
    if d is None or (head_dim is not None and fmt.row_bytes(int(head_dim)) != rb):
        raise ValueError(f"cache lane of {rb}-byte rows is not {fmt.name} (head dim {head_dim})")
    return fmt


# ------------------------------------------------------------------------------------------------ host kernels
def _kernels():
    from . import kernels as K

    return K


def quantize(x: mx.array, fmt: Fmt) -> mx.array:
    """Packed rows [..., RB] uint8 of x [B, H, L, D] (any strides) in ``fmt``."""
    fmt = get(fmt)
    K = _kernels()
    if x.ndim != 4:
        raise ValueError(f"quantize expects [B, H, L, D], got {x.shape}")
    B, H, L, D = (int(s) for s in x.shape)
    if not fmt.supports(D):
        raise ValueError(f"{fmt.name} cannot pack head dim {D}")
    rb = fmt.row_bytes(D)
    if B * H * L == 0:
        return mx.zeros((B, H, L, rb), dtype=mx.uint8)
    kern = K.kernel("kvq_quant", (("tn", K.METAL_TYPE[x.dtype]),), (("F", fmt.code), ("D", D)))
    rows = B * H * L
    return kern(inputs=[x], grid=(-(-32 * rows // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(B, H, L, rb)], output_dtypes=[mx.uint8])[0]


def dequantize(lane: mx.array, n: int, fmt: Fmt, dtype=mx.bfloat16) -> mx.array:
    """Rows [0, n) of a packed lane [B, H, capacity, RB] decoded to [B, H, n, D] ``dtype``."""
    fmt = get(fmt)
    K = _kernels()
    B, H = int(lane.shape[0]), int(lane.shape[1])
    D = fmt.head_dim(int(lane.shape[3]))
    if D is None:
        raise ValueError(f"lane rows of {lane.shape[3]} bytes are not {fmt.name}")
    n = int(n)
    if n == 0:
        return mx.zeros((B, H, 0, D), dtype=dtype)
    view = lane[:, :, :n, :] if int(lane.shape[2]) != n else lane
    kern = K.kernel("kvq_dequant", (("tn", K.METAL_TYPE[dtype]),), (("F", fmt.code), ("D", D)))
    threads = B * H * n * (D // 8)
    return kern(inputs=[view], grid=(-(-threads // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(B, H, n, D)], output_dtypes=[dtype])[0]


def bytes_per_token(fmt: Fmt, heads: int, head_dim: int, nominal_bytes: int = 2) -> int:
    """Bytes one token occupies in one lane of ``heads`` heads."""
    fmt = get(fmt)
    return heads * (fmt.row_bytes(head_dim) if fmt.quantized else head_dim * nominal_bytes)


# ------------------------------------------------------------------------------------------------ numpy reference
# An independent host implementation of the encoders / decoders (tests compare the Metal kernels against it bit for
# bit). Inputs are float32 values (already rounded to the nominal dtype).
def _np():
    import numpy as np

    return np


def ref_e4m3(x):
    """float32 -> e4m3 codes (round to nearest even, saturate 448, NaN -> 0x7f)."""
    np = _np()
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    s = ((u >> 24) & 0x80).astype(np.uint32)
    a = u & 0x7FFFFFFF
    out = np.zeros(x.shape, dtype=np.uint32)
    nan = a > 0x7F800000
    sat = (a >= 0x43E00000) & ~nan
    sub = a < 0x3C800000
    norm = ~(nan | sat | sub)
    af = a.view(np.float32)
    out[sub] = np.rint(af[sub].astype(np.float64) * 512.0).astype(np.uint32)
    mant = a & 0x7FFFFF
    q = (((a >> 23) - 120) << 3) | (mant >> 20)
    rest = mant & 0xFFFFF
    q = q + ((rest > 0x80000) | ((rest == 0x80000) & ((q & 1) == 1))).astype(np.uint32)
    out[norm] = np.minimum(q[norm], 0x7E)
    out[sat] = 0x7E
    out[nan] = 0x7F
    return (out | s).astype(np.uint8)


def ref_fp8(c):
    np = _np()
    c = np.asarray(c, dtype=np.uint32)
    e = (c >> 3) & 15
    m = c & 7
    v = np.where(e == 0, m.astype(np.float64) * 2.0 ** -9, (8 + m).astype(np.float64) * 2.0 ** (e.astype(np.float64) - 10))
    return np.where(c & 0x80, -v, v).astype(np.float32)


def ref_e2m1(x):
    np = _np()
    x = np.asarray(x, dtype=np.float32)
    s = np.signbit(x).astype(np.uint32) << 3
    a = np.abs(x)
    m = np.select([a > 5.0, a >= 3.5, a > 2.5, a >= 1.75, a > 1.25, a >= 0.75, a > 0.25],
                  [7, 6, 5, 4, 3, 2, 1], 0).astype(np.uint32)
    return (s | m).astype(np.uint8)


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def ref_fp4(c):
    np = _np()
    c = np.asarray(c, dtype=np.uint32)
    v = np.asarray(_FP4, dtype=np.float32)[c & 7]
    return np.where(c & 8, -v, v).astype(np.float32)


def _bf16(x):
    """float32 -> nearest bf16 (round to nearest even), returned as float32 values."""
    np = _np()
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def _pow2(i):
    np = _np()
    return np.ldexp(np.float32(1.0), np.asarray(i, dtype=np.int32)).astype(np.float32)


def _exp_of(a):
    np = _np()
    return ((np.asarray(a, dtype=np.float32).view(np.uint32) >> 23) & 0xFF).astype(np.int64) - 127


def ref_quantize(x, fmt) -> "object":
    """Packed rows [..., RB] uint8 of float32 rows x [..., D]: the bytes the Metal encoder must produce."""
    np = _np()
    fmt = get(fmt)
    x = np.asarray(x, dtype=np.float32)
    D = x.shape[-1]
    rows = x.reshape(-1, D)
    rb = fmt.row_bytes(D)
    cb = fmt.code_bytes(D)
    out = np.zeros((rows.shape[0], rb), dtype=np.uint8)
    if fmt.code == 1:
        g = rows.reshape(rows.shape[0], D // 32, 32)
        amax = np.abs(g).max(axis=-1)
        e = np.minimum(_exp_of(amax), 127)
        n = e - 8
        bump = amax * _pow2(-np.clip(n, -127, 127)) > 448.0
        n = np.where(n < -127, -127, np.where(bump, n + 1, n))
        n = np.where(amax > 0, np.minimum(n, 127), -127)
        codes = ref_e4m3((g * _pow2(-n)[..., None]).astype(np.float32))
        out[:, :D] = codes.reshape(rows.shape[0], D)
        out[:, cb:cb + D // 32] = (n + 127).astype(np.uint8)
    elif fmt.code == 2:
        ram = np.abs(rows).max(axis=-1)
        e = np.minimum(_exp_of(ram), 127)
        k = e - 11
        bump = ram * _pow2(-np.clip(k, -126, 127)) > 2688.0
        k = np.where(k < -126, -126, np.where(bump, k + 1, k))
        k = np.where(ram > 0, np.minimum(k, 127), -126)
        xs = (rows * _pow2(-k)[:, None]).astype(np.float32)
        g = xs.reshape(rows.shape[0], D // 16, 16)
        bm = np.abs(g).max(axis=-1)
        sb = ref_e4m3((bm.astype(np.float64) / 6.0).astype(np.float32))
        s = ref_fp8(sb)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (g.astype(np.float64) / np.where(s > 0, s, 1.0)[..., None]).astype(np.float32)
        c = np.where((s > 0)[..., None], ref_e2m1(q), 0).astype(np.uint8).reshape(rows.shape[0], D)
        out[:, :cb] = c[:, 0::2] | (c[:, 1::2] << 4)
        out[:, cb:cb + D // 16] = sb
        out[:, cb + D // 16] = (k + 127).astype(np.uint8)
    else:
        lv = 255 if fmt.code == 3 else 15
        ng = D // 64
        g = rows.reshape(rows.shape[0], ng, 64)
        mn, mxv = g.min(axis=-1), g.max(axis=-1)
        s = _bf16(((mxv - mn).astype(np.float32).astype(np.float64) / lv).astype(np.float32))
        b = _bf16(mn)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.rint(((g - b[..., None]).astype(np.float32).astype(np.float64) / np.where(s > 0, s, 1.0)[..., None])
                        .astype(np.float32))
        q = np.where((s > 0)[..., None], np.clip(q, 0, lv), 0).astype(np.uint8).reshape(rows.shape[0], D)
        if fmt.code == 3:
            out[:, :D] = q
        else:
            out[:, :cb] = q[:, 0::2] | (q[:, 1::2] << 4)
        sbits = s.view(np.uint32) >> 16
        bbits = b.view(np.uint32) >> 16
        out[:, cb:cb + 2 * ng] = sbits.astype(np.uint16).view(np.uint8).reshape(rows.shape[0], 2 * ng)
        out[:, cb + 2 * ng:cb + 4 * ng] = bbits.astype(np.uint16).view(np.uint8).reshape(rows.shape[0], 2 * ng)
    return out.reshape(x.shape[:-1] + (rb,))


def ref_dequantize(packed, fmt, d: int, nominal_bf16: bool = True):
    """float32 values [..., D] of packed rows (rounded to bf16 for the affine formats when ``nominal_bf16``)."""
    np = _np()
    fmt = get(fmt)
    p = np.asarray(packed, dtype=np.uint8)
    rows = p.reshape(-1, p.shape[-1])
    cb = fmt.code_bytes(d)
    if fmt.code == 1:
        s = np.ldexp(np.float32(1.0), rows[:, cb:cb + d // 32].astype(np.int32) - 127).astype(np.float32)
        s = np.where(rows[:, cb:cb + d // 32] == 0, np.float32(2.0 ** -127), s)
        v = ref_fp8(rows[:, :d]).reshape(-1, d // 32, 32) * s[..., None]
    elif fmt.code == 2:
        sb = ref_fp8(rows[:, cb:cb + d // 16])
        k = rows[:, cb + d // 16].astype(np.int32) - 127
        s = sb * _pow2(k)[:, None]
        c = np.stack([rows[:, :cb] & 15, rows[:, :cb] >> 4], axis=-1).reshape(-1, d)
        v = ref_fp4(c).reshape(-1, d // 16, 16) * s[..., None]
    else:
        ng = d // 64
        s = (rows[:, cb:cb + 2 * ng].copy().view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        b = (rows[:, cb + 2 * ng:cb + 4 * ng].copy().view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        if fmt.code == 3:
            q = rows[:, :d].astype(np.float32)
        else:
            q = np.stack([rows[:, :cb] & 15, rows[:, :cb] >> 4], axis=-1).reshape(-1, d).astype(np.float32)
        v = (s[..., None].astype(np.float64) * q.reshape(-1, ng, 64) + b[..., None]).astype(np.float32)
        if nominal_bf16:
            v = _bf16(v)
    return np.asarray(v, dtype=np.float32).reshape(p.shape[:-1] + (d,))

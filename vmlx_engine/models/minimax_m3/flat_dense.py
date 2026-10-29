# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 "flat" dense projections: decode steps and DSpark verify rows on the matrix units (``VMLX_M3_FLAT_DENSE``).

Why: a DSpark verify must give every row exactly the arithmetic of a one-row decode step. With MLX's one-row qmv
arithmetic (the fused kernels' K1 / K5 / shared expert, ``metal/row_exact_qmv``) that costs ~2-4 ALU ops per weight
PER ROW, so the dense projections of a W-row verify cost ~W/4 .. W/2 one-row steps (measured on the M5 Max at M3.1
shapes: q_proj 63 us at 1 row, 188 us at 8, 370 us at 16). Here every projection runs ONE 16-row matrix-unit tile per
weight group whatever the row count (``kernels/flat_mm.metal``): a row's bits never depend on the other rows or on
how many there are, and 1..16 rows cost about the same (q_proj ~81 us at 1 .. 16 rows). Decode steps run the same
kernels, so verify rows == decode steps bit for bit; the price is the one-row step itself (~81 vs 63 us for q_proj:
the matrix unit computes all 16 rows of the tile).

Arithmetic (NOT MLX's; decode and verify switch together): the activation is rounded once to bf16 (half for MXFP8),
as the stock unfused path rounds a norm output; per weight group the matrix unit's fp32 dot with the raw codes, then
``acc = fma(scale, dot, fma(bias, XS, acc))`` (XS = the group's sum of the bf16 activations), groups in order, split-K
partials added in simdgroup order.

Covered when ``VMLX_M3_FLAT_DENSE=1`` (default off; read per call), B = 1, rows 1 .. 16 (wider verifies in chunks of
16): the fused attention half's q | k | v | index q | index k and o_proj (``fused_attn``), the fused MoE block's shared
expert (``fused_moe``), the dense MLP layers and the lm_head (decode: the model's own calls; verify: the DSpark
target). Prefill chunks wider than 16 rows keep MLX's kernels.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

ENV = "VMLX_M3_FLAT_DENSE"
MAX_ROWS = 16
NT = 32
DIR = Path(__file__).resolve().parent / "kernels"
STATS = {"mm": 0, "rows": 0}
_TYPE = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float", mx.uint8: "uint8_t",
         mx.uint32: "uint32_t"}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in {"1", "true", "on", "yes"}


_AVAILABLE: dict = {}


def available() -> bool:
    """Metal 4 tensor ops compile on this GPU and give the right answer (checked once per process)."""
    hit = _AVAILABLE.get("ok")
    if hit is None:
        try:
            hit = mx.default_device() == mx.gpu and mx.metal.is_available() and _self_check()
        except Exception:
            hit = False
        _AVAILABLE["ok"] = hit
    return hit


def active() -> bool:
    return enabled() and available()


# ------------------------------------------------------------------------------------------------ kernels
@lru_cache(maxsize=None)
def _src(name: str) -> str:
    return (DIR / f"{name}.metal").read_text()


_SPECS = {
    "flat_prep": (("X", "NW"), ("A", "A16", "XS")),
    "flat_mm": (("A", "A16", "XS", "w0", "s0", "b0", "w1", "s1", "b1", "w2", "s2", "b2", "w3", "s3", "b3", "w4", "s4",
                 "b4", "RS"), ("Y",)),
}
_KERNELS: dict = {}


def _kernel(name: str, types: tuple, ints: tuple, floats: tuple = ()):
    key = (name, types, ints, floats)
    k = _KERNELS.get(key)
    if k is None:
        pre = "".join(f"  using {n} = {t};\n" for n, t in types)
        pre += "".join(f"  constexpr int {n} = {int(v)};\n" for n, v in ints)
        pre += "".join(f"  constexpr float {n} = {float(v)!r}f;\n" for n, v in floats)
        header = _src("flat_common")
        source = pre + _src(name)
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
        ins, outs = _SPECS[name]
        k = _KERNELS[key] = mx.fast.metal_kernel(name=f"vmlx_m3_{name}_{digest}", input_names=list(ins),
                                                 output_names=list(outs), source=source, header=header)
    return k


_DUMMY: dict = {}


def _dummy(dtype=mx.float32) -> mx.array:
    d = _DUMMY.get(dtype)
    if d is None:
        d = _DUMMY[dtype] = mx.zeros((8,), dtype=dtype)
        mx.eval(d)
    return d


# ------------------------------------------------------------------------------------------------ formats
def fmt(m) -> Optional[int]:
    """flat_mm format of a quantized linear: 0 affine 4-bit g64, 1 MXFP8 g32, 2 affine 8-bit g64, 3 affine 6-bit
    g64 (bf16 / fp16 scales, no bias); None when unsupported."""
    if not isinstance(m, nn.QuantizedLinear) or "bias" in m:
        return None
    mode = str(getattr(m, "mode", "affine") or "affine")
    bits, gs = int(m.bits), int(m.group_size)
    if mode == "mxfp8":
        return 1 if bits == 8 and gs == 32 else None
    if mode != "affine" or "biases" not in m or gs != 64 or m["scales"].dtype not in (mx.bfloat16, mx.float16):
        return None
    return {4: 0, 8: 2, 6: 3}.get(bits)


def _in_out(m) -> tuple:
    w = m["weight"]
    return int(w.shape[1]) * 32 // int(m.bits), int(w.shape[0])


def supported(mods: Sequence, k: Optional[int] = None) -> bool:
    if not mods or len(mods) > 5:
        return False
    ks = set()
    for m in mods:
        f = fmt(m)
        if f is None:
            return False
        kin, n = _in_out(m)
        if n % NT or kin % 64:
            return False
        ks.add(kin)
    return len(ks) == 1 and (k is None or ks == {k})


def split_k(ntiles: int, K: int) -> int:
    """Simdgroups splitting the weight groups: fixed by the shape (never by the row count)."""
    return 8 if ntiles <= 128 and K // 64 >= 64 else 4


# ------------------------------------------------------------------------------------------------ launches
def prep(x2: mx.array, nw: Optional[mx.array] = None, eps: float = 1e-6, half16: bool = False):
    """(A bf16, A16 half or None, XS f32) of R rows (flat_prep)."""
    R, K = int(x2.shape[0]), int(x2.shape[1])
    kern = _kernel("flat_prep", (("xt", _TYPE[x2.dtype]), ("nwt", _TYPE[nw.dtype] if nw is not None else "float")),
                   (("K", K), ("NORM", int(nw is not None)), ("H16", int(half16))), (("EPS", float(eps)),))
    # XS flat [R * K / 64] (at least 8 elements: smaller arrays would be bound as constant-space buffers)
    A, A16, XS = kern(inputs=[x2, nw if nw is not None else _dummy()], grid=(256, R, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(R, K), (R, K) if half16 else (1,), (max(8, R * (K // 64)),)],
                      output_dtypes=[mx.bfloat16, mx.float16, mx.float32])
    return A, (A16 if half16 else None), XS


def mm(pr, mods: Sequence, epi: int = 0, out_dtype=mx.float32, resid: Optional[mx.array] = None,
       alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    """flat_mm over the concatenated segments ``mods`` (see kernels/flat_mm.metal for EPI)."""
    A, A16, XS = pr
    R, K = int(A.shape[0]), int(A.shape[1])
    segs = list(mods) + [None] * (5 - len(mods))
    types, ints, ins = [], [("K", K)], [A, A16 if A16 is not None else _dummy(mx.float16), XS]
    ns = []
    for i, m in enumerate(segs):
        if m is None:
            types.append((f"st{i}", "float"))
            ints += [(f"N{i}", 0), (f"F{i}", 0)]
            ins += [_dummy(), _dummy(), _dummy()]
            ns.append(0)
            continue
        f = fmt(m)
        _, n = _in_out(m)
        ns.append(n)
        types.append((f"st{i}", _TYPE[m["scales"].dtype]))
        ints += [(f"N{i}", n), (f"F{i}", f)]
        ins += [m["weight"], m["scales"], m["biases"] if f != 1 else _dummy(m["scales"].dtype)]
    ntiles = (ns[0] if epi == 2 else sum(ns)) // NT
    sk = split_k(ntiles, K)
    out_dtype = mx.float32 if epi == 0 else out_dtype
    types += [("ot", _TYPE[out_dtype]), ("rt", _TYPE[resid.dtype] if resid is not None else "float")]
    ints += [("NT", NT), ("SK", sk), ("EPI", epi)]
    ins.append(resid if resid is not None else _dummy())
    kern = _kernel("flat_mm", tuple(types), tuple(ints), (("ALPHA", float(alpha)), ("LIMIT", float(limit))))
    nout = ns[0] if epi in (1, 2) else sum(ns)
    STATS["mm"] += 1
    STATS["rows"] += R
    return kern(inputs=ins, grid=(32 * sk * ntiles, 1, 1), threadgroup=(32 * sk, 1, 1),
                output_shapes=[(R, nout)], output_dtypes=[mx.float32 if epi == 0 else out_dtype])[0]


def _chunks(x2: mx.array, fn):
    R = int(x2.shape[0])
    if R <= MAX_ROWS:
        return fn(x2)
    return mx.concatenate([fn(x2[a:a + MAX_ROWS]) for a in range(0, R, MAX_ROWS)], axis=0)


def linear(m, x: mx.array) -> mx.array:
    """``m(x)`` for (..., K) rows on the flat kernels (in chunks of 16 rows), output in x's dtype."""
    lead, K = x.shape[:-1], int(x.shape[-1])
    x2 = x.reshape(-1, K)
    y = _chunks(x2, lambda xc: mm(prep(xc, half16=fmt(m) == 1), [m], epi=3, out_dtype=x.dtype))
    return y.reshape(*lead, int(y.shape[-1]))


def mlp(m, x: mx.array) -> mx.array:
    """SwiGLU-OAI MLP (gate / up / down: the shared expert or a dense MLP layer) of (..., D) rows: gate and up in ONE
    launch with the SwiGLU epilogue, then down; output in x's dtype."""
    lead, K = x.shape[:-1], int(x.shape[-1])
    x2 = x.reshape(-1, K)

    def run(xc):
        hs = mm(prep(xc), [m.gate_proj, m.up_proj], epi=2, out_dtype=x.dtype, alpha=float(m.alpha),
                limit=float(m.limit))
        return mm(prep(hs), [m.down_proj], epi=3, out_dtype=x.dtype)

    y = _chunks(x2, run)
    return y.reshape(*lead, int(y.shape[-1]))


def mlp_supported(m) -> bool:
    g, u, d = (getattr(m, n, None) for n in ("gate_proj", "up_proj", "down_proj"))
    if g is None or u is None or d is None or not supported([g, u]) or not supported([d]):
        return False
    if fmt(g) != fmt(u) or fmt(g) == 1 or _in_out(g) != _in_out(u) or _in_out(d)[0] != _in_out(g)[1]:
        return False
    return hasattr(m, "alpha") and hasattr(m, "limit")


def linear_supported(m) -> bool:
    return supported([m])


# ------------------------------------------------------------------------------------------------ attention half
def attn_supported(layer) -> bool:
    attn = layer.self_attn
    ix = getattr(attn, "indexer", None)
    if ix is None:
        return False
    mods = [attn.q_proj, attn.k_proj, attn.v_proj, ix.index_q_proj, ix.index_k_proj]
    return supported(mods) and supported([attn.o_proj])


def attn_qkv(layer, x2: mx.array) -> mx.array:
    """fused_attn K1's output on the flat kernels: input RMSNorm, then q | k | v | index q | index k, [R, ...] f32."""
    attn, ix = layer.self_attn, layer.self_attn.indexer
    norm = layer.input_layernorm
    mods = [attn.q_proj, attn.k_proj, attn.v_proj, ix.index_q_proj, ix.index_k_proj]
    half16 = any(fmt(m) == 1 for m in mods)
    return mm(prep(x2, norm.weight, float(norm.eps), half16=half16), mods, epi=0)


def attn_oproj(layer, o: mx.array, x2: mx.array) -> mx.array:
    """fused_attn K5 on the flat kernels: x2 + o_proj(o), one rounding to x2's dtype."""
    return mm(prep(o), [layer.self_attn.o_proj], epi=1, out_dtype=x2.dtype, resid=x2)


# ------------------------------------------------------------------------------------------------ self check
def _self_check() -> bool:
    import numpy as np

    rng = np.random.default_rng(0)
    w = mx.array(rng.standard_normal((64, 512), dtype=np.float32) * 0.05).astype(mx.bfloat16)
    m = nn.QuantizedLinear(512, 64, bias=False, group_size=64, bits=4)
    m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
    x = mx.array(rng.standard_normal((3, 512), dtype=np.float32)).astype(mx.bfloat16)
    y = mm(prep(x), [m], epi=0)
    ref = x.astype(mx.float32) @ mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=4).astype(
        mx.float32).T
    return bool(mx.allclose(y, ref, atol=2e-2, rtol=2e-2).item())

# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax-M3 / M3.1 MoE block for decode and DSpark verify (1..16 rows), behind ``VMLX_M3_FUSED_MOE=1``.

Four launches per MoE layer instead of ~20 (router ops, rotate, gate/up, rotate, down, shared gate / up / SwiGLU /
down, adds and casts):

  K6  m3_router        fp32 router logits + sigmoid for all rows; Hadamard-128 rotation of the routed input
  K7  m3_moe_gate_up   routing (top-k of sigmoid + bias, normalize, scale); routed JANGT gate / up + SwiGLU-OAI per
                       (row, slot); shared expert (affine 4-bit) gate / up + SwiGLU-OAI per row
  R   rotate_rows      the routed down input, rotated with each slot's expert column scales (moe_kernels)
  K8  m3_moe_down      routed JANGT down + routing-weighted combine, shared expert down, residual add -> new residual

Every kernel computes a row's values with threads and an order that do not depend on how many rows share the launch,
so decode steps and verify rows run identical arithmetic (greedy DSpark == plain decoding with the fused path on).
Beyond that, each kernel follows today's unfused ops op by op (MLX's one-row gemv / qmv_fast / Sigmoid / compiled
SwiGLU, the JANGT v3 decode kernels, the same bf16 roundings), so the fused block is bit-identical to the unfused
block: tests/test_minimax_m3_fused_moe.py checks it at 1..16 rows.

Metal sources: kernels/m3_moe_common.metal (header) + one body file per kernel (shared with vmlx-swift via
MLXFast.metalKernel; the template parameters are listed in each file and in ``template_args`` below).
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

KDIR = Path(__file__).parent / "kernels"
MAX_ROWS = 16
ENV = "VMLX_M3_FUSED_MOE"
# Rows up to which the shared expert runs inside K7 / K8 (one threadgroup per row and output tile, MLX's one-row qmv).
# Wider calls run it as row-exact 4-row qmv launches (metal/row_exact_qmv; the same per-row arithmetic), which walk
# the weights a quarter as often: measured 16 rows, 57 layers, fused shared ~30 ms vs ~10 ms. VMLX_M3_FUSED_SHARED_ROWS.
SHARED_IN_KERNEL_ROWS = 2


def _shared_rows() -> int:
    try:
        return int(os.environ.get("VMLX_M3_FUSED_SHARED_ROWS", "") or SHARED_IN_KERNEL_ROWS)
    except ValueError:
        return SHARED_IN_KERNEL_ROWS

_INPUTS = {
    "m3_router": ["x", "gw", "su"],
    "m3_moe_gate_up": ["fc", "xr", "x", "sig", "bias", "wg", "sg", "wu", "su_", "ocol", "woutg", "woutu",
                       "sgw", "sgs", "sgb", "suw", "sus", "sub"],
    "m3_moe_down": ["fc", "hr", "h", "inds", "scores", "wd", "sd", "ocol", "wout", "hs", "sdw", "sds", "sdb", "resid",
                    "sdo", "ry"],
}
_OUTPUTS = {
    "m3_router": ["sig", "xr"],
    "m3_moe_gate_up": ["h", "hs", "inds", "scores"],
    "m3_moe_down": ["out"],
}


def enabled() -> bool:
    """``VMLX_M3_FUSED_MOE=1`` turns the fused block on (default off; read per call)."""
    return os.environ.get(ENV, "0").strip().lower() in {"1", "true", "on", "yes"}


def metal_source(name: str) -> tuple[str, str]:
    """(header, body) of a kernel, exactly as handed to mx.fast.metal_kernel (and MLXFast.metalKernel)."""
    return (KDIR / "m3_moe_common.metal").read_text(), (KDIR / f"{name}.metal").read_text()


@functools.lru_cache(maxsize=None)
def _kernel(name: str):
    header, body = metal_source(name)
    tag = hashlib.sha256((header + body).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(name=f"vmlx_{name}_{tag}", input_names=_INPUTS[name], output_names=_OUTPUTS[name],
                                source=body, header=header)


def fbits(v: float) -> int:
    """A positive float32 constant as the int32 bit pattern the kernels read back with as_type<float> (template
    arguments must be non-negative: MLX puts them into the kernel's host name)."""
    b = int(np.array(v, np.float32).view(np.int32))
    if b < 0:
        raise ValueError(f"fused M3 MoE: template float {v} must be positive")
    return b


def _kb2(K) -> int:
    k2 = float(K) * 2
    if k2 not in (2.0, 3.0, 4.0, 5.0, 6.0):
        raise ValueError(f"fused M3 MoE: JANGT width {K} not supported")
    return int(k2)


def _stats(L: int, K) -> list:
    """(mu0, mu1, isd0, isd1): the trellis code statistics of width K (moe_kernels code_stats)."""
    from vmlx_engine.jangt import v2l as VL

    mu, isd = VL.code_stats(L, K)
    return [float(mu[0]), float(mu[1]), float(isd[0]), float(isd[1])]


def _affine4(q) -> bool:
    return (isinstance(q, nn.QuantizedLinear) and str(getattr(q, "mode", "affine") or "affine") == "affine"
            and int(q.bits) == 4 and int(q.group_size) == 64 and "biases" in q and "bias" not in q)


def supported(block, x: mx.array) -> bool:
    """The fused kernels' contract: a JANGT (gate/up and down) SwiGLU-OAI routed stack, an affine-4 / group-64
    shared expert, an fp32 router with top-k > 1 normalized, n_group 1, 1..16 rows, bf16 / fp16 activations."""
    sw = getattr(block, "switch_mlp", None)
    if sw is None or not getattr(sw, "is_jangt", False) or sw.gu_fmt != "jangt" or sw.dn_fmt != "jangt":
        return False
    if sw.act != "oai" or not sw.limit > 0:
        return False
    sh = getattr(block, "shared_experts", None)
    if sh is None or not all(_affine4(getattr(sh, n, None)) for n in ("gate_proj", "up_proj", "down_proj")):
        return False
    if float(getattr(sh, "alpha", -1)) != float(sw.alpha) or float(getattr(sh, "limit", -1)) != float(sw.limit):
        return False
    if int(sh.gate_proj.weight.shape[0]) != sw.I or int(sh.down_proj.weight.shape[0]) != sw.D:
        return False                       # the shared expert shares K7's intermediate tiles
    if sw.I % 512 or sw.D % 512:
        return False                       # MLX's one-row qmv_fast traversal (the shared expert's arithmetic)
    if sw.D % 1024 or sw.D < 16 * sw.E:
        return False                       # MLX's one-row gemv configuration K6 transcribes (BN 8, K >= 16 N)
    gate = getattr(block, "gate", None)
    if not isinstance(gate, nn.Linear) or isinstance(gate, nn.QuantizedLinear) or "bias" in gate:
        return False
    if gate.weight.dtype != mx.float32 or block.e_score_correction_bias.dtype != mx.float32:
        return False
    if not block.norm_topk_prob or block.top_k < 2:
        return False
    rows = int(np.prod(x.shape[:-1]))
    D, I = sw.D, sw.I
    return (1 <= rows <= MAX_ROWS and x.dtype in (mx.bfloat16, mx.float16) and x.shape[-1] == D
            and D % 512 == 0 and I % 128 == 0 and D % 128 == 0 and I % 8 == 0 and D % 8 == 0
            and sh.gate_proj.scales.dtype == x.dtype)


def router(x2, gw, su, rows: int, rot: bool = True):
    """K6: (sig [rows, E] f32, xr [rows, D] f32 -- not written with rot=False)."""
    E, D = int(gw.shape[0]), int(gw.shape[1])
    n_tg = E + (rows * ((D + 1023) // 1024) if rot else 0)
    return _kernel("m3_router")(
        inputs=[x2, gw, su], template=[("T", x2.dtype), ("D", D), ("E", E), ("ROWS", rows), ("ROT", int(rot))],
        grid=(256 * n_tg, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, E), (rows, D)], output_dtypes=[mx.float32, mx.float32])


def template_args(block, rows: int, x_dtype, shared: bool = True, routed: bool = True) -> dict:
    """Every template parameter K7 / K8 take for this layer, and their float constant buffers ``fc`` (also what
    vmlx-swift passes)."""
    sw = block.switch_mlp
    U = sw._unit()
    E, I, D = sw.E, sw.I, sw.D
    common = [("T", x_dtype), ("D", D), ("I", I), ("E", E), ("ROWS", rows), ("KS", int(block.top_k)),
              ("L", int(U["L"]))]
    gu = common + [("KB2", _kb2(U["K"])), ("W", int(U["gate"]["packed"].shape[-1])), ("NOUT", int(U["gate"]["n_out"])),
                   ("ROUTED", int(routed)), ("SHARED", int(shared)), ("ALPHA", fbits(sw.alpha)),
                   ("LIMIT", fbits(sw.limit)), ("RSF", fbits(block.routed_scaling_factor))]
    dn = common + [("KB2", _kb2(U["K_dn"])), ("W", int(U["down"]["packed"].shape[-1])), ("NOUT", int(U["down"]["n_out"])),
                   ("SHARED", int(shared)), ("ROUTED", int(routed))]
    fc_gu = _stats(U["L"], U["K"])
    fc_dn = _stats(U["L"], U["K_dn"])
    return {"gate_up": gu, "down": dn, "fc_gate_up": fc_gu, "fc_down": fc_dn}


@functools.lru_cache(maxsize=None)
def _fc(values: tuple) -> mx.array:
    """The float constants as a device buffer: zero-padded to 16 elements -- mx.fast.metal_kernel binds inputs of
    fewer than 8 elements as constant-address-space buffers, and reads of such a 7-float buffer came back wrong in the
    shared-expert branch of K7 (alpha 0.404, limit 32768) while the same reads in the routed branch were right."""
    return mx.array(np.pad(np.array(values, np.float32), (0, 16 - len(values))))


def gate_up(block, x2, sig, xr, shared: bool = True, routed: bool = True):
    """K7: (h [rows * KS, I] f32, hs [rows, I] (written only with ``shared``), inds [rows, KS] u32, scores [rows, KS]
    f32)."""
    sw, sh = block.switch_mlp, block.shared_experts
    U = sw._unit()
    rows, I, KS = int(x2.shape[0]), sw.I, int(block.top_k)
    targs = template_args(block, rows, x2.dtype, shared, routed)
    g, u = U["gate"], U["up"]
    return _kernel("m3_moe_gate_up")(
        inputs=[_fc(tuple(targs["fc_gate_up"])), sig if xr is None else xr, x2, sig, block.e_score_correction_bias, g["packed"], g["scale"],
                u["packed"], u["scale"], g["ocol"], g["wout"], u["wout"], sh.gate_proj.weight, sh.gate_proj.scales,
                sh.gate_proj.biases, sh.up_proj.weight, sh.up_proj.scales, sh.up_proj.biases],
        template=targs["gate_up"], grid=(64, rows * KS + (rows if shared else 0), I // 8), threadgroup=(64, 1, 1),
        output_shapes=[(rows * KS, I), (rows, I), (rows, KS), (rows, KS)],
        output_dtypes=[mx.float32, x2.dtype, mx.uint32, mx.float32])


def down_input(block, h, inds):
    """R: the routed down input, rotated with each slot's expert column scales (moe_kernels.rotate_rows)."""
    from vmlx_engine.jangt import moe_kernels as MK

    return MK.rotate_rows(h, block.switch_mlp._unit()["sv_dn"], inds.reshape(-1))


def down(block, hr, h, inds, scores, hs, r2, sdo=None, ry=None):
    """K8: the new residual rows [rows, D]. ``sdo``: the shared expert's output computed by the caller (else K8
    computes it from ``hs``)."""
    sw, sh = block.switch_mlp, block.shared_experts
    U = sw._unit()
    rows, D = int(r2.shape[0]), sw.D
    shared = sdo is None
    targs = template_args(block, rows, r2.dtype, shared, ry is None)
    d = U["down"]
    return _kernel("m3_moe_down")(
        inputs=[_fc(tuple(targs["fc_down"])), h if hr is None else hr, h, inds, scores, d["packed"], d["scale"], d["ocol"], d["wout"],
                r2 if hs is None else hs, sh.down_proj.weight, sh.down_proj.scales, sh.down_proj.biases, r2,
                r2 if sdo is None else sdo, ry if ry is not None else _fc((0.0,))],
        template=targs["down"], grid=(64, rows, D // 8), threadgroup=(64, 1, 1),
        output_shapes=[(rows, D)], output_dtypes=[r2.dtype])[0]


def fused_moe_residual(block, x_normed: mx.array, resid: mx.array) -> mx.array:
    """``resid + block(x_normed)`` for a SparseMoeBlock with the fused kernels; shapes (..., D), 1..16 rows."""
    sw = block.switch_mlp
    U = sw._unit()
    lead = x_normed.shape[:-1]
    D = sw.D
    x2 = mx.contiguous(x_normed.reshape(-1, D))
    r2 = mx.contiguous(resid.reshape(-1, D).astype(x2.dtype))
    rows = int(x2.shape[0])
    shared = rows <= _shared_rows()
    sdo = None if shared else _shared_rows_exact(block.shared_experts, x2)
    from vmlx_engine.jangt import fast_decode as FD

    if FD.enabled() and FD.supported(U, D, sw.I):
        # VMLX_JANGT_FAST=1: routing + shared expert in K6 / K7, the routed experts on the fast JANGT kernels (the
        # same ones JTMixedSwitchGLU.routed runs with the lever), K8 adds shared + residual
        sig, _ = router(x2, block.gate.weight, U["su_gu"], rows, rot=False)
        h, hs, inds, scores = gate_up(block, x2, sig, None, shared, routed=False)
        ry = FD.routed_fast(x2, inds, scores, U, sw.limit, sw.act, sw.alpha)
        out = down(block, None, h, inds, scores, hs if shared else None, r2, sdo, ry=ry)
        return out.reshape(*lead, D)
    sig, xr = router(x2, block.gate.weight, U["su_gu"], rows)
    h, hs, inds, scores = gate_up(block, x2, sig, xr, shared)
    out = down(block, down_input(block, h, inds), h, inds, scores, hs if shared else None, r2, sdo)
    return out.reshape(*lead, D)


def _shared_rows_exact(sh, x2):
    """The shared expert of several rows, every row on MLX's one-row qmv arithmetic (one row-split walk for 5..16
    rows, 4-row launches otherwise, one-row calls where a shape is not admitted) and the model's own SwiGLU-OAI:
    [rows, D]."""
    from vmlx_engine.metal.row_exact_qmv import row_exact_qmv, row_exact_qmv_rowsplit, rowsplit_enabled

    def lin(q, x):
        rows = int(x.shape[0])
        if rows > 4 and rowsplit_enabled():
            y = row_exact_qmv_rowsplit(x, q["weight"], q["scales"], q["biases"], group_size=int(q.group_size),
                                       bits=int(q.bits))
            if y is not None:
                return y
        outs = []
        for a in range(0, rows, 4):
            xb = x[a:a + 4]
            y = row_exact_qmv(xb, q["weight"], q["scales"], q["biases"], group_size=int(q.group_size),
                              bits=int(q.bits)) if int(xb.shape[0]) > 1 else None
            outs.append(y if y is not None else
                        mx.concatenate([q(xb[r:r + 1]) for r in range(int(xb.shape[0]))], axis=0))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    from vmlx_engine.models.minimax_m3.minimax_m3 import swiglu

    return lin(sh.down_proj, swiglu(lin(sh.up_proj, x2), lin(sh.gate_proj, x2), sh.alpha, sh.limit))


def unfused_moe_residual(block, x_normed: mx.array, resid: mx.array) -> mx.array:
    """The reference: today's decode ops, one row at a time (each row exactly a decode step's MoE block)."""
    rows = [x_normed[..., r:r + 1, :] for r in range(int(x_normed.shape[-2]))]
    outs = [resid[..., r:r + 1, :] + block(xr) for r, xr in enumerate(rows)]
    return mx.concatenate(outs, axis=-2)

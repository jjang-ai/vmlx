"""TQSwitchGLU: drop-in for mlx_lm SwitchGLU on JANGTQ v2 experts  .

Tensor names use the `tq2_` prefix ON PURPOSE: released vMLX routes any bundle with `.tq_packed` tensors to the v1
JANGTQ loader, which would silently mis-decode v2. With `tq2_*` names an old runtime fails closed (unknown tensors).

Weights (loaded lazily from the bundle, never repacked at load):
  {gate,up,down}_proj.tq2_packed  uint32 (E, N, K*bits/32)
  {gate,up,down}_proj.tq2_scales  float16 (E, N)
Per-module rotation (config entry "rotation"):
  none       : weights quantized as is
  hadamard32 : weights quantized as W R^T with R = blockwise normalized Walsh-Hadamard over 32-wide input blocks
               (no random signs). Decode rotates activation rows before the expert kernels by default.
Paths (same switch point as MLX SwitchGLU: sort when indices.size >= 64):
  decode : fused gate/up/SwiGLU qmv (f32) -> down qmv with the router-weighted sum fused (x.dtype)
  prefill: argsort experts (GPU) -> fused gate/up/SwiGLU NAX qmm -> down NAX qmm -> unsort -> weighted sum
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import kernels as K
from .format import codebook

import logging
import os

SORT_THRESHOLD = 64
# One-shot log witness per process of which prefill kernel path the routed experts took (effective kernel
# evidence for served proofs; the env/identity flags alone cannot prove a path armed).
_PREFILL_PATH_WITNESSED: dict = {}
logger = logging.getLogger(__name__)
# generic fused weighted unsort for prefill (any D / k; fp32 accumulate). JANGH_WEIGHTED_UNSORT=0 to disable.
from .runtime_identity import WEIGHTED_UNSORT  # default off; Qwen opts in per module
ROTATIONS = ("none", "hadamard32")
# Where decode applies the Hadamard-32: "host" = once per activation row (x once per token, h once per expert-token)
# then the unrotated fast kernels; "kernel" = in-register inside every threadgroup (redundant: measured 1.05-1.15x).
from .runtime_identity import DECODE_ROT, EXPERT_TILES, GATEUP_H32, H32_ROWS, PREFILL_REDUCE


class TQSwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int, bits: int, rotation: str = "none"):
        super().__init__()
        if rotation not in ROTATIONS:
            raise ValueError(f"jangtq2: unknown rotation {rotation!r}")
        if rotation == "hadamard32" and input_dims % 32:
            raise ValueError("jangtq2: hadamard32 requires input_dims % 32 == 0")
        self.bits = bits
        self.rotation = rotation
        self.input_dims, self.output_dims, self.num_experts = input_dims, output_dims, num_experts
        self.tq2_packed = mx.zeros((num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.tq2_scales = mx.zeros((num_experts, output_dims), dtype=mx.float16)
        self._cb = mx.array(codebook(bits))

    @property
    def rotated(self) -> bool:
        return self.rotation == "hadamard32"

    def to_quantized(self, **kwargs):
        """Keep packed JANGH modules intact in generic text-model loading."""
        if (
            kwargs.get("mode", "jangtq2") != "jangtq2"
            or kwargs.get("bits", self.bits) != self.bits
            or kwargs.get("rotation", self.rotation) != self.rotation
        ):
            raise ValueError("jangtq2: quantization entry differs from installed module")
        return self


def rotate_rows(x: mx.array, lin: TQSwitchLinear) -> mx.array:
    """Host-side activation rotation (prefill path): blockwise normalized Hadamard-32 in float32, back to x.dtype."""
    if not lin.rotated:
        return x
    if H32_ROWS == "1" or getattr(lin, "use_h32_rows", False):
        # Preserve the existing output dtype and rounding boundary.
        return K.h32_rows(x, x.dtype)
    shp = x.shape
    return mx.hadamard_transform(x.astype(mx.float32).reshape(*shp[:-1], shp[-1] // 32, 32)).reshape(shp).astype(x.dtype)


class TQSwitchGLU(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int, bits_gate_up: int, bits_down: int,
                 swiglu_limit: float = 0.0, rotation_gate_up: str = "none", rotation_down: str = "none"):
        super().__init__()
        self.gate_proj = TQSwitchLinear(input_dims, hidden_dims, num_experts, bits_gate_up, rotation_gate_up)
        self.up_proj = TQSwitchLinear(input_dims, hidden_dims, num_experts, bits_gate_up, rotation_gate_up)
        self.down_proj = TQSwitchLinear(hidden_dims, input_dims, num_experts, bits_down, rotation_down)
        self.limit = float(swiglu_limit)

    def _rot_in_kernel(self) -> bool:
        """Decode H32 rotation inside the expert kernels (process-wide JANGTQ2_DECODE_ROT=kernel, or per module: a
        family loader sets ``decode_rot_kernel`` where it measured faster, e.g. MiniCPM-V-4.7's D=2048 / I=512)."""
        return DECODE_ROT == "kernel" or bool(getattr(self, "decode_rot_kernel", False))

    def _decode_h(self, xf, idx):
        g, u = self.gate_proj, self.up_proj
        in_kernel = g.rotated and self._rot_in_kernel()
        if g.rotated and not in_kernel:
            xf = rotate_rows(xf, g)
        return K.gather_qmv(xf, g.tq2_packed, g.tq2_scales, g._cb, idx, g.bits, x_per_dispatch=False,
                            packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit, rotate=in_kernel,
                            slot_fastest=xf.shape[0] > 1)

    def _down_in(self, h):
        """(h, rotate_in_kernel) for the down projection."""
        d = self.down_proj
        if d.rotated and not self._rot_in_kernel():
            return rotate_rows(h, d), False
        return h, d.rotated

    def _use_expert_tiles(self, x, kk):
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        if not (EXPERT_TILES == "1" or getattr(self, "use_expert_tiles", False)):
            return False
        D, I = g.input_dims, g.output_dims
        if (D, I) == (4096, 2048):                 # GLM-5.3 (E=288) / MiMo-V2.6 (E=256)
            down_bits = (2, 3, 4) if g.num_experts == 256 else (2, 3)
            experts = (256, 288)
        elif (D, I) == (3072, 1024):               # Ornith-1.5-122B (qwen3_5_moe, E=256)
            down_bits = (2, 3)
            experts = (256,)
        else:
            return False
        return (
            x.dtype == mx.bfloat16 and x.shape[-1] == D and kk == 8
            and g.num_experts in experts
            and (u.input_dims, u.output_dims, u.num_experts) == (D, I, g.num_experts)
            and (d.input_dims, d.output_dims, d.num_experts) == (I, D, g.num_experts)
            and g.bits == u.bits and g.bits in (2, 3)
            and d.bits in down_bits
            and g.rotated and u.rotated and d.rotated and K.nax_available()
        )

    def _prefill(self, x, idx, kk, scores=None):
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        order = mx.argsort(idx)
        inv = mx.argsort(order)
        idx_s = idx[order]
        xs = rotate_rows(x, g)[order // kk]
        if self._use_expert_tiles(x, kk):
            if not _PREFILL_PATH_WITNESSED.get("tiles"):
                _PREFILL_PATH_WITNESSED["tiles"] = True
                logger.info("JANGH prefill path witness: expert tiles E=%d bits=%d/%d h32_epilogue=%s (rows=%d)",
                            g.num_experts, g.bits, d.bits,
                            (GATEUP_H32 == "1" or getattr(self, "use_gateup_h32", False)) and g.num_experts == 288, int(x.shape[0]))
            plan = K.expert_tile_plan(idx_s, g.num_experts)
            # Fused output rotation is separately qualified for GLM only (E=288); the glm5_next
            # loader opts its modules in (runtime_identity.GLM_FUSED_TILES), env is the diagnostic switch.
            rotate_output = (GATEUP_H32 == "1" or getattr(self, "use_gateup_h32", False)) and g.num_experts == 288
            h = K.gather_qmm_expert_sorted(
                xs, g.tq2_packed, g.tq2_scales, idx_s, g.bits, plan,
                packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit,
                rotate_output=rotate_output)
            y = K.gather_qmm_expert_sorted(
                h if rotate_output else rotate_rows(h, d), d.tq2_packed, d.tq2_scales, idx_s, d.bits, plan)
        else:
            if not _PREFILL_PATH_WITNESSED.get("sorted"):
                _PREFILL_PATH_WITNESSED["sorted"] = True
                logger.info("JANGH prefill path witness: generic sorted NAX/steel E=%d bits=%d/%d host_rotation=%s (rows=%d)",
                            g.num_experts, g.bits, d.bits, bool(d.rotated), int(x.shape[0]))
            h = K.gather_qmm_sorted(xs, g.tq2_packed, g.tq2_scales, g._cb, idx_s, g.bits,
                                    packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit)
            y = K.gather_qmm_sorted(rotate_rows(h, d), d.tq2_packed, d.tq2_scales, d._cb, idx_s, d.bits)
        if scores is not None:
            fused = K.prefill_weighted_unsort(y, inv, scores, enabled=PREFILL_REDUCE == "1")
            if fused is not None:
                return fused
            if WEIGHTED_UNSORT == "1" or getattr(self, "use_weighted_unsort", False):
                fused = K.weighted_unsort(y, inv, scores)
                if fused is not None:
                    return fused
            y = y[inv].reshape(x.shape[0], kk, y.shape[-1])
            return (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        return y[inv]

    def _use_sorted(self, routed_rows: int, kk: int) -> bool:
        """Choose the sorted NAX prefill path (True) or the per-row gather decode path (False).

        Legacy rule: sorted when routed rows >= SORT_THRESHOLD (64), i.e. >= 8 tokens at k=8
        (GLM/Naive geometry, qualified). A family loader may set ``decode_max_tokens`` from a
        measured crossover; the rule is then by TOKENS. Measured on M5 Max / MLX 0.32.3 for
        qwen4_exp (D=2560, I=640, E=512, k=10, JANGH 4/6-bit): the gather path wins through
        96 tokens (3.5 ms vs 5.6 ms at 64, 5.0 vs 6.0 at 96) and loses from 128 (6.5 vs 6.3);
        the sorted path has ~5.5 ms of fixed argsort/gather/tile cost. With the row rule a
        continuous-batching decode step of >= 7 sequences (70 rows) ran the prefill kernels:
        measured 8 concurrent = 35 tok/s aggregate, below single-stream (producer 05 playbook).
        """
        limit = getattr(self, "decode_max_tokens", None)
        if limit is None:
            return routed_rows >= SORT_THRESHOLD
        return (routed_rows // max(1, int(kk))) > int(limit)

    def _experts(self, x, indices):
        """x (..., D), indices (..., k) -> (..., k, D) in x.dtype (unweighted, like SwitchGLU)."""
        d = self.down_proj
        lead, kk, D = x.shape[:-1], indices.shape[-1], x.shape[-1]
        xf = x.reshape(-1, D)
        idx = indices.reshape(-1).astype(mx.uint32)
        if not self._use_sorted(idx.size, kk):
            h, rk = self._down_in(self._decode_h(xf, idx))
            y = K.gather_qmv(h, d.tq2_packed, d.tq2_scales, d._cb, idx, d.bits, x_per_dispatch=True, rotate=rk,
                             slot_fastest=xf.shape[0] > 1)
            return y.astype(x.dtype).reshape(*lead, kk, D)
        return self._prefill(xf, idx, kk).reshape(*lead, kk, D)

    def __call__(self, x, indices):
        return self._experts(x, indices)

    def routed(self, x, indices, scores):
        """Weighted routed output (..., D) = sum_k scores[...,k] * expert_k(x). Decode fuses the weighted sum
        into the down kernel; prefill uses the sorted NAX path then a weighted reduction."""
        d = self.down_proj
        lead, kk, D = x.shape[:-1], indices.shape[-1], x.shape[-1]
        if self._use_sorted(indices.size, kk):
            if (PREFILL_REDUCE == "1" and x.ndim == 3 and x.shape[0] == 1
                    and x.shape[1] >= 8 and kk == 8 and D == 4096
                    and x.dtype == mx.bfloat16):
                if tuple(indices.shape) != (*lead, kk) or tuple(scores.shape) != tuple(indices.shape):
                    raise ValueError("JANGH routed indices/scores shape differs from input rows")
                y = self._prefill(x.reshape(-1, D), indices.reshape(-1).astype(mx.uint32),
                                  kk, scores.reshape(-1, kk))
                return y.reshape(*lead, D)
            if WEIGHTED_UNSORT == "1" or getattr(self, "use_weighted_unsort", False):
                y = self._prefill(x.reshape(-1, D), indices.reshape(-1).astype(mx.uint32), kk, scores.reshape(-1, kk))
                return y.reshape(*lead, D)
            y = self._experts(x, indices)
            return (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        idx2 = indices.reshape(-1, kk).astype(mx.uint32)
        h, rk = self._down_in(self._decode_h(x.reshape(-1, D), idx2.reshape(-1)))
        y = K.gather_qmv_weighted_down(h, d.tq2_packed, d.tq2_scales, d._cb, idx2, scores.reshape(-1, kk), d.bits,
                                       x.dtype, rotate=rk, slot_fastest=idx2.shape[0] > 1)
        return y.reshape(*lead, D)


def relu2(x: mx.array) -> mx.array:
    return mx.square(mx.maximum(x, 0))


class TQSwitchMLP(nn.Module):
    """Drop-in for mlx_lm SwitchMLP with gate-less relu2 experts on JANGTQ v2 weights (Nemotron-H: fc1 -> relu2 -> fc2,
    e.g. Nemotron-3.5-Bolt's 1024-wide latent experts). Same kernels as TQSwitchGLU, single-projection form:
      decode : fc1 gather qmv (f32) -> relu2 -> fc2 gather qmv          (rows < SORT_THRESHOLD routed pairs)
      prefill: argsort experts -> fc1 sorted qmm -> relu2 -> fc2 sorted qmm -> unsort
    Unweighted output (..., k, D) like SwitchMLP; the caller applies the router weights."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int, bits_fc1: int, bits_fc2: int,
                 rotation_fc1: str = "none", rotation_fc2: str = "none"):
        super().__init__()
        self.fc1 = TQSwitchLinear(input_dims, hidden_dims, num_experts, bits_fc1, rotation_fc1)
        self.fc2 = TQSwitchLinear(hidden_dims, input_dims, num_experts, bits_fc2, rotation_fc2)

    def _use_sorted(self, routed_rows: int, kk: int) -> bool:
        limit = getattr(self, "decode_max_tokens", None)
        if limit is None:
            return routed_rows >= SORT_THRESHOLD
        return (routed_rows // max(1, int(kk))) > int(limit)

    def __call__(self, x, indices):
        f1, f2 = self.fc1, self.fc2
        lead, kk, D = x.shape[:-1], indices.shape[-1], x.shape[-1]
        xf = x.reshape(-1, D)
        idx = indices.reshape(-1).astype(mx.uint32)
        if not self._use_sorted(idx.size, kk):
            h = K.gather_qmv(rotate_rows(xf, f1), f1.tq2_packed, f1.tq2_scales, f1._cb, idx, f1.bits, x_per_dispatch=False,
                             slot_fastest=xf.shape[0] > 1)
            h = relu2(h).astype(x.dtype)
            y = K.gather_qmv(rotate_rows(h, f2), f2.tq2_packed, f2.tq2_scales, f2._cb, idx, f2.bits, x_per_dispatch=True,
                             slot_fastest=xf.shape[0] > 1)
            return y.astype(x.dtype).reshape(*lead, kk, D)
        order = mx.argsort(idx)
        inv = mx.argsort(order)
        idx_s = idx[order]
        xs = rotate_rows(xf, f1)[order // kk]
        h = relu2(K.gather_qmm_sorted(xs, f1.tq2_packed, f1.tq2_scales, f1._cb, idx_s, f1.bits))
        y = K.gather_qmm_sorted(rotate_rows(h, f2), f2.tq2_packed, f2.tq2_scales, f2._cb, idx_s, f2.bits)
        return y[inv].astype(x.dtype).reshape(*lead, kk, D)

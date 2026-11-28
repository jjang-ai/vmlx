# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 Flash JANGHT2.4 loader: exact MXFP8 trunk + fused sibling projections + fused JANGT/JANGH SiTU MoE."""

from __future__ import annotations

import gc
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vmlx_engine.jangh import kernels as jh_kernels
from vmlx_engine.jangh.switch import TQSwitchLinear, rotate_rows as jh_rotate_rows
from vmlx_engine.jangt import moe_kernels as jt_kernels
from vmlx_engine.jangt.prefill import gather_gate_up_sorted_jt, gather_qmm_sorted_jt
from vmlx_engine.jangt.switch import JTSwitchLinear, nax_prefill

from .config import ModelConfig
from .kernels import (
    fused_situ,
    round_e4m3_and_h32,
    round_e4m3_and_rot128,
    round_e4m3_per128,
)
from .language import (
    DecoderLayer,
    HCAAttention,
    LanguageModel,
    LatentMoE,
    MLAAttention,
    MLP,
    MXFP8MultiLinear,
)

logger = logging.getLogger(__name__)

# All ERNIE 128x128 FP8 trunk linears that repack losslessly into MLX MXFP8 (group_size=32, bits=8)
DIRECT_MXFP8_LINEARS = (
    "self_attn.o_proj.weight",
    "self_attn.linear_q_up_proj.weight",
    "self_attn.gate_proj.weight",
    "self_attn.linear_q_down_proj.weight",
    "self_attn.linear_kv_proj.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.core_attention.indexer.wq_b.weight",
    "block_sparse_moe.fc1_latent_proj.weight",
    "block_sparse_moe.fc2_latent_proj.weight",
    "block_sparse_moe.shared_experts.w1.weight",
    "block_sparse_moe.shared_experts.w2.weight",
    "block_sparse_moe.shared_experts.w3.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def mxfp8_view(weight: mx.array, scale: mx.array) -> Tuple[mx.array, mx.array]:
    """Exact repack of ERNIE 128x128 power-of-two FP8 E4M3 weights into MLX MXFP8."""
    if weight.dtype != mx.uint8 or weight.ndim != 2 or scale.dtype != mx.float32:
        raise ValueError("Expected 2-D E4M3 uint8 weight and FP32 block scales")
    rows, columns = weight.shape
    if columns % 32 or scale.shape != ((rows + 127) // 128, (columns + 127) // 128):
        raise ValueError(f"Invalid shape for ERNIE 128x128 FP8 projection: {weight.shape} vs {scale.shape}")
    raw = scale.view(mx.uint32)
    exponent = (raw >> 23) & 255
    valid = ((raw & 0x807FFFFF) == 0) & (exponent > 0) & (exponent < 255)
    if not bool(mx.all(valid).item()):
        raise ValueError("Non power-of-two FP8 block scale encountered")
    packed_scale = mx.repeat(mx.repeat(exponent.astype(mx.uint8), 128, 0), 4, 1)
    return mx.contiguous(weight).view(mx.uint32), mx.contiguous(packed_scale[:rows, : columns // 32])


def keep_fp8_linears(prefix: str, tensors: Dict[str, mx.array]) -> None:
    """Convert eligible FP8 trunk linears (including 3D linear_o_group_proj) to MXFP8 in-place."""
    for name in DIRECT_MXFP8_LINEARS:
        key = prefix + name
        scale_key = key + "_scale_inv"
        if key in tensors and scale_key in tensors:
            try:
                codes, scales = mxfp8_view(tensors[key], tensors[scale_key])
            except ValueError:
                continue
            tensors.pop(scale_key)
            tensors[key] = codes
            tensors[key.removesuffix("weight") + "scales"] = scales
    # Also convert 3-D grouped HCA output projection `self_attn.linear_o_group_proj` (1.242 GB!)
    wo_key = prefix + "self_attn.linear_o_group_proj"
    wo_skey = wo_key + "_scale_inv"
    if wo_key in tensors and wo_skey in tensors:
        try:
            codes, scales = mxfp8_view(tensors[wo_key], tensors[wo_skey])
        except ValueError:
            return
        tensors.pop(wo_skey)
        tensors[wo_key] = codes
        tensors[wo_key + ".scales"] = scales


def install_direct_mxfp8_linears(block: nn.Module, parameters: Dict[str, mx.array]) -> Tuple[int, int]:
    """Replace nn.Linear / MultiLinear with exact-value MXFP8 QuantizedLinear / MXFP8MultiLinear."""
    count = 0
    reclaimed = 0
    for name, scales in list(parameters.items()):
        if not name.endswith(".scales") or scales.dtype != mx.uint8:
            continue
        scope = name.removesuffix(".scales")
        ancestors = scope.split(".")
        parent = block
        for comp in ancestors[:-1]:
            parent = getattr(parent, comp)
        target = getattr(parent, ancestors[-1])
        packed = parameters[scope + ".weight"]
        if isinstance(target, nn.Linear):
            out, width = target.weight.shape
            if packed.shape != (out, width // 4) or scales.shape != (out, width // 32):
                raise ValueError(f"MXFP8 shape mismatch at {scope}: {packed.shape} vs {(out, width // 4)}")
            rep = nn.QuantizedLinear(width, out, bias=False, group_size=32, bits=8, mode="mxfp8")
            setattr(parent, ancestors[-1], rep)
            count += 1
            reclaimed += out * width * 2 - packed.nbytes - scales.nbytes
        elif hasattr(target, "weight") and target.weight.ndim == 3 and packed.ndim == 3:
            g, out, width = target.weight.shape
            if packed.shape != (g, out, width // 4) or scales.shape != (g, out, width // 32):
                raise ValueError(f"3D MXFP8 shape mismatch at {scope}: {packed.shape}")
            rep3 = MXFP8MultiLinear(g, out, width)
            setattr(parent, ancestors[-1], rep3)
            count += 1
            reclaimed += g * out * width * 2 - packed.nbytes - scales.nbytes
    return count, reclaimed


def _concat_mxfp8_rows(linears: Sequence[nn.QuantizedLinear]) -> Optional[nn.QuantizedLinear]:
    """Concatenate same-input MXFP8 QuantizedLinears along output rows (zero extra RAM)."""
    if not all(isinstance(m, nn.QuantizedLinear) and m.mode == "mxfp8" and m.bits == 8 and m.group_size == 32 for m in linears):
        return None
    w = mx.concatenate([m.weight for m in linears], axis=0)
    s = mx.concatenate([m.scales for m in linears], axis=0)
    mx.eval(w, s)
    out_dims = int(w.shape[0])
    in_dims = int(w.shape[1]) * 4
    fused = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=32, bits=8, mode="mxfp8")
    fused.weight = w
    fused.scales = s
    fused.freeze()
    return fused


def _concat_mxfp8_cols(linears: Sequence[nn.QuantizedLinear]) -> Optional[nn.QuantizedLinear]:
    """Concatenate same-output MXFP8 QuantizedLinears along input columns (K % 32 == 0)."""
    if not all(isinstance(m, nn.QuantizedLinear) and m.mode == "mxfp8" and m.bits == 8 and m.group_size == 32 for m in linears):
        return None
    w = mx.concatenate([m.weight for m in linears], axis=1)
    s = mx.concatenate([m.scales for m in linears], axis=1)
    mx.eval(w, s)
    out_dims = int(w.shape[0])
    in_dims = int(w.shape[1]) * 4
    fused = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=32, bits=8, mode="mxfp8")
    fused.weight = w
    fused.scales = s
    fused.freeze()
    return fused


def prepare_ernie_runtime(layers: Sequence[DecoderLayer]) -> Dict[str, int]:
    """Fuse sibling MXFP8 projections, fold VirtualHeads, cast mHC fn to BF16, and fuse HCA compressors."""
    counts = {"vha_folded": 0, "hca_comp": 0, "attn_fused": 0, "moe_fused": 0, "dense_mlp_fused": 0, "mhc_bf16": 0}
    mhc_bf16 = os.environ.get("VMLX_ERNIE_MHC_BF16", "1") != "0"
    for layer in layers:
        if mhc_bf16:
            for hc in (getattr(layer, "attn_hc", None), getattr(layer, "ffn_hc", None)):
                if hc is not None and getattr(hc, "fn", None) is not None and hc.fn.dtype == mx.float32:
                    fn_b = hc.fn.astype(mx.bfloat16)
                    mx.eval(fn_b)
                    hc.fn = fn_b
                    counts["mhc_bf16"] += 1
        attn = layer.attn
        if hasattr(attn, "vha"):
            attn.vha.fold()
            counts["vha_folded"] += 1
        if hasattr(attn, "attn_sink"):
            sink_bf16 = attn.attn_sink.astype(mx.bfloat16)
            mx.eval(sink_bf16)
            attn._attn_sink_bf16 = sink_bf16
        if isinstance(attn, HCAAttention):
            attn.compressor.prepare_fused()
            counts["hca_comp"] += 1
            f1 = _concat_mxfp8_rows((attn.wq_a, attn.wkv))
            if f1 is not None:
                attn._qa_kv_fused = f1
                attn.wq_a = None
                attn.wkv = None
                counts["attn_fused"] += 1
            f2 = _concat_mxfp8_rows((attn.wq_b, attn.o_gate))
            if f2 is not None:
                attn._qb_ogate_fused = f2
                attn.wq_b = None
                attn.o_gate = None
                counts["attn_fused"] += 1
        elif isinstance(attn, MLAAttention):
            f1 = _concat_mxfp8_rows((attn.q_a_proj, attn.kv_a_proj_with_mqa))
            if f1 is not None:
                attn._qa_kva_fused = f1
                attn.q_a_proj = None
                attn.kv_a_proj_with_mqa = None
                counts["attn_fused"] += 1
            f2 = _concat_mxfp8_rows((attn.q_b_proj, attn.o_gate, attn.indexer.wq_b))
            if f2 is not None:
                attn._qb_og_idx_fused = f2
                attn.q_b_proj = None
                attn.o_gate = None
                attn.indexer.wq_b = None
                counts["attn_fused"] += 1
        elif hasattr(attn, "wq_a") and hasattr(attn, "wq_b") and hasattr(attn, "o_gate"):
            # DraftAttention in DSpark layers
            f2 = _concat_mxfp8_rows((attn.wq_b, attn.o_gate))
            if f2 is not None:
                attn._qb_ogate_fused = f2
                attn.wq_b = None
                attn.o_gate = None
                counts["attn_fused"] += 1

        ffn = layer.ffn
        if isinstance(ffn, LatentMoE):
            f_in = _concat_mxfp8_rows(
                (ffn.shared_experts.gate_proj, ffn.shared_experts.up_proj, ffn.fc1_latent_proj)
            )
            if f_in is not None:
                g_dim = int(ffn.shared_experts.gate_proj.weight.shape[0])
                u_dim = int(ffn.shared_experts.up_proj.weight.shape[0])
                ffn._shared_fc1_fused = f_in
                ffn._shared_fc1_splits = (g_dim, g_dim + u_dim)
                ffn.shared_experts.gate_proj = None
                ffn.shared_experts.up_proj = None
                ffn.fc1_latent_proj = None
                counts["moe_fused"] += 1
            f_out = _concat_mxfp8_cols((ffn.shared_experts.down_proj, ffn.fc2_latent_proj))
            if f_out is not None:
                ffn._shared_fc2_fused = f_out
                ffn.shared_experts.down_proj = None
                ffn.fc2_latent_proj = None
                counts["moe_fused"] += 1
        elif isinstance(ffn, MLP):
            f_gu = _concat_mxfp8_rows((ffn.gate_proj, ffn.up_proj))
            if f_gu is not None:
                ffn._gate_up_fused = f_gu
                ffn.gate_proj = None
                ffn.up_proj = None
                counts["dense_mlp_fused"] += 1
    mx.clear_cache()
    return counts


# ---------------------------------------------------------------------------
# Fused ERNIE JANGHT SwitchGLU (supports JANGT + JANGH + exact per-128 E4M3 + SiTU + weighted down!)
# ---------------------------------------------------------------------------
class ErnieSwitchGLU(nn.Module):
    """High-speed mixed JANGT / JANGH routed SwitchGLU for ERNIE 5.5 Flash.

    - Rounds `latent` ONCE via single-dispatch `round_e4m3_per128` (not 20x per expert!).
    - Runs paired JANGH or fused JANGT `gate_proj + up_proj` + SiTU (`beta=4.0, linear_beta=25.0`).
    - Rounds intermediate ONCE via `round_e4m3_per128`.
    - Runs fused weighted `down_proj` reduction in-kernel (`gather_down_weighted_v3` for JANGT,
      `gather_qmv_weighted_down` for JANGH) for decode & DSpark verify, and NAX `gather_qmm_sorted`
      / `gather_qmm_sorted_jt` for prefill!
    """

    accepts_route_weights = True
    SORT_THRESHOLD = 160  # T*top_k >= 160 (T >= 16 at top_k=10) -> NAX sorted prefill; T <= 15 stays row-exact decode

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        *,
        beta: float = 4.0,
        linear_beta: float = 25.0,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)
        self._gu_is_jt = isinstance(gate_proj, JTSwitchLinear)
        self._dn_is_jt = isinstance(down_proj, JTSwitchLinear)
        self._jt_gu_bundle: Optional[Dict[str, Any]] = None
        self._jt_dn_bundle: Optional[Dict[str, Any]] = None
        self._jt_g_signs: Optional[mx.array] = None
        self._jt_u_signs: Optional[mx.array] = None
        self._jt_gu_same_signs: bool = False
        self._jt_gu_layer_su: Optional[mx.array] = None
        self._jt_gu_cols: Optional[mx.array] = None
        self._jt_gu_fused_ok: bool = False
        self._jt_dn_signs: Optional[mx.array] = None

    @staticmethod
    def _build_jt_signs(proj: JTSwitchLinear) -> Tuple[Dict[str, Any], mx.array]:
        unit = proj.unit()
        cols = unit["ocol"]
        targets = mx.arange(proj.input_dims)[None, :, None]
        valid = ~mx.any(targets == cols[:, None, :], axis=-1)
        signs = proj.jt_su[None, :] * valid
        if unit["cs"] is not None:
            signs = signs * unit["cs"]
        mx.eval(signs)
        return unit, signs

    def prepare(self) -> None:
        import numpy as np
        if self._gu_is_jt:
            gu_g, sg = self._build_jt_signs(self.gate_proj)
            gu_u, su = self._build_jt_signs(self.up_proj)
            self._jt_g_signs = sg
            self._jt_u_signs = su
            self._jt_gu_same_signs = bool(mx.array_equal(sg, su).item())
            cg, cu = np.array(gu_g["ocol"]), np.array(gu_u["ocol"])
            same_ocol = (cg == cu).all() and (cg == cg[0]).all()
            same_su = bool(mx.array_equal(gu_g["su"], gu_u["su"]).item())
            same_k = gu_g["K"] == gu_u["K"] and gu_g["packed"].shape == gu_u["packed"].shape
            self._jt_gu_fused_ok = bool(same_ocol and same_su and same_k)
            if self._jt_gu_fused_ok:
                m = np.ones(self.gate_proj.input_dims, np.float32)
                c0 = cg[0]
                m[c0[c0 >= 0]] = 0.0
                self._jt_gu_layer_su = gu_g["su"] * mx.array(m)
                mx.eval(self._jt_gu_layer_su)
                if gu_g["n_out"] > 0 and (c0 >= 0).all():
                    self._jt_gu_cols = mx.array(c0.astype(np.int32))
                    mx.eval(self._jt_gu_cols)
            self._jt_gu_bundle = {
                "gate": gu_g,
                "up": gu_u,
                "K": self.gate_proj.kbits,
                "L": self.gate_proj.state_bits,
            }
        if self._dn_is_jt:
            dn_u, sd = self._build_jt_signs(self.down_proj)
            self._jt_dn_signs = sd
            self._jt_dn_bundle = {
                "down": dn_u,
                "K_dn": self.down_proj.kbits,
                "L": self.down_proj.state_bits,
            }

    def __call__(self, x: mx.array, indices: mx.array, route_weights: mx.array) -> mx.array:
        dtype = x.dtype
        lead = x.shape[:-1]
        K_in = x.shape[-1]
        top_k = indices.shape[-1]
        T = x.size // K_in

        idx_2d = indices.reshape(T, top_k).astype(mx.uint32)
        rw_2d = route_weights.reshape(T, top_k).astype(mx.float32)
        routes = idx_2d.reshape(-1)
        use_sorted = (routes.size >= self.SORT_THRESHOLD) and (
            (not self._gu_is_jt or nax_prefill()) and (not self._dn_is_jt or nax_prefill())
        )

        x2 = x.reshape(T, K_in)
        if use_sorted:
            return self._forward_prefill(x2, idx_2d, rw_2d, dtype, lead)
        return self._forward_decode(x2, idx_2d, rw_2d, dtype, lead)

    def _forward_decode(
        self, x2: mx.array, idx_2d: mx.array, rw_2d: mx.array, dtype: mx.Dtype, lead: Tuple[int, ...]
    ) -> mx.array:
        T, K_in = x2.shape
        top_k = idx_2d.shape[1]
        routes = idx_2d.reshape(-1)
        slot_fastest = T > 1
        act_caps = (self.beta, self.linear_beta)

        # --- Gate + Up + SiTU (FUSED E4M3+ROT + 1-PASS JANGT/JANGH!) ---
        if self._gu_is_jt:
            U = self._jt_gu_bundle
            assert U is not None
            if self._jt_gu_fused_ok and self._jt_gu_layer_su is not None:
                x_q, xr = round_e4m3_and_rot128(x2, self._jt_gu_layer_su)
                mid = jt_kernels.gather_gate_up_swiglu_v3(
                    xr, x_q, idx_2d, U, out_dtype=dtype,
                    slot_fastest=slot_fastest, act="inkling", alpha=act_caps,
                )
            else:
                x_q = round_e4m3_per128(x2)
                x_rep = mx.repeat(x_q, top_k, axis=0)
                xr_g = jt_kernels.rotate_rows(x_rep, self._jt_g_signs, routes)
                xr_u = xr_g if self._jt_gu_same_signs else jt_kernels.rotate_rows(x_rep, self._jt_u_signs, routes)
                ones = mx.ones((routes.size, 1), dtype=mx.float32)
                sel = routes[:, None]
                g_out = jt_kernels.gather_down_weighted_v3(
                    xr_g, x_rep, sel, ones, {"down": U["gate"], "K_dn": U["K"], "L": U["L"]}
                ).astype(dtype)
                u_out = jt_kernels.gather_down_weighted_v3(
                    xr_u, x_rep, sel, ones, {"down": U["up"], "K_dn": U["K"], "L": U["L"]}
                ).astype(dtype)
                mid = fused_situ(u_out, g_out, self.beta, self.linear_beta)
        else:
            gp, up = self.gate_proj, self.up_proj
            # Fuse per-128 E4M3 rounding + H32 rotation in ONE dispatch on T token rows!
            xin = round_e4m3_and_h32(x2, mx.float32) if gp.rotated else round_e4m3_per128(x2)
            mid = jh_kernels.gather_qmv(
                xin, gp.tq2_packed, gp.tq2_scales, gp._cb, routes, gp.bits,
                x_per_dispatch=False, packed_u=up.tq2_packed, scales_u=up.tq2_scales,
                rotate=False, slot_fastest=slot_fastest, act="inkling", alpha=act_caps,
            ).astype(dtype)

        # --- Weighted Down Combine (FUSED E4M3+ROT + WEIGHTED DOWN!) ---
        if self._dn_is_jt:
            Ud = self._jt_dn_bundle
            assert Ud is not None
            mid_q, hr = round_e4m3_and_rot128(mid, self._jt_dn_signs, routes)
            out = jt_kernels.gather_down_weighted_v3(hr, mid_q, idx_2d, rw_2d, Ud, slot_fastest=False)
            return out.astype(dtype).reshape(*lead, self.down_proj.output_dims)
        else:
            dp = self.down_proj
            din = round_e4m3_and_h32(mid, mx.float32) if dp.rotated else round_e4m3_per128(mid)
            out = jh_kernels.gather_qmv_weighted_down(
                din, dp.tq2_packed, dp.tq2_scales, dp._cb, idx_2d, rw_2d, dp.bits,
                dtype, rotate=False, slot_fastest=slot_fastest,
            )
            return out.reshape(*lead, dp.output_dims)

    def _forward_prefill(
        self, x2: mx.array, idx_2d: mx.array, rw_2d: mx.array, dtype: mx.Dtype, lead: Tuple[int, ...]
    ) -> mx.array:
        T, K_in = x2.shape
        top_k = idx_2d.shape[1]
        routes = idx_2d.reshape(-1)
        order = mx.argsort(routes)
        inv = mx.argsort(order)
        s_routes = routes[order]
        tok_idx = order // top_k
        use_nax = jh_kernels.nax_available()
        plan = jh_kernels.expert_tile_plan(s_routes, self.gate_proj.num_experts) if use_nax else None

        # --- Gate + Up + SiTU (separate single-weight NAX tiles double occupancy on cold DRAM!) ---
        if self._gu_is_jt:
            U = self._jt_gu_bundle
            assert U is not None
            if self._jt_gu_fused_ok and self._jt_gu_layer_su is not None:
                x_q, xr_f32 = round_e4m3_and_rot128(x2, self._jt_gu_layer_su)
                xr_s = xr_f32.astype(mx.bfloat16)[tok_idx]
                x_sorted = x_q[tok_idx] if U["gate"]["n_out"] > 0 else x_q
                jt_gu_plan = None if (U["gate"]["n_out"] == 0 and routes.size >= 8192) else plan
                g_out = self._jt_sorted_proj(xr_s, x_sorted, U["gate"], s_routes, plan=jt_gu_plan).astype(dtype)
                u_out = self._jt_sorted_proj(xr_s, x_sorted, U["up"], s_routes, plan=jt_gu_plan).astype(dtype)
                mid = fused_situ(u_out, g_out, self.beta, self.linear_beta)
            else:
                x_q = round_e4m3_per128(x2)
                x_sorted = x_q[tok_idx]
                xr_g = jt_kernels.rotate_rows(x_sorted, self._jt_g_signs, s_routes).astype(mx.bfloat16)
                g_out = self._jt_sorted_proj(xr_g, x_sorted, U["gate"], s_routes, plan=plan).astype(dtype)
                xr_u = xr_g if self._jt_gu_same_signs else jt_kernels.rotate_rows(x_sorted, self._jt_u_signs, s_routes).astype(mx.bfloat16)
                u_out = self._jt_sorted_proj(xr_u, x_sorted, U["up"], s_routes, plan=plan).astype(dtype)
                mid = fused_situ(u_out, g_out, self.beta, self.linear_beta)
        else:
            gp, up = self.gate_proj, self.up_proj
            x_rot = (round_e4m3_and_h32(x2, dtype) if gp.rotated else round_e4m3_per128(x2))[tok_idx]
            if plan is not None:
                g_out = jh_kernels.gather_qmm_expert_sorted(
                    x_rot, gp.tq2_packed, gp.tq2_scales, s_routes, gp.bits, plan,
                )
                u_out = jh_kernels.gather_qmm_expert_sorted(
                    x_rot, up.tq2_packed, up.tq2_scales, s_routes, up.bits, plan,
                )
            else:
                g_out = jh_kernels.gather_qmm_sorted(
                    x_rot, gp.tq2_packed, gp.tq2_scales, gp._cb, s_routes, gp.bits,
                )
                u_out = jh_kernels.gather_qmm_sorted(
                    x_rot, up.tq2_packed, up.tq2_scales, up._cb, s_routes, up.bits,
                )
            mid = fused_situ(u_out, g_out, self.beta, self.linear_beta).astype(dtype)

        # --- Down (FUSED E4M3+ROT + EXPERT-ALIGNED NAX PREFILL!) ---
        if self._dn_is_jt:
            Ud = self._jt_dn_bundle
            assert Ud is not None
            mid_q, hr_f32 = round_e4m3_and_rot128(mid, self._jt_dn_signs, s_routes)
            dn_sorted = self._jt_sorted_proj(hr_f32.astype(mx.bfloat16), mid_q, Ud["down"], s_routes, plan=plan).astype(dtype)
        else:
            dp = self.down_proj
            din = round_e4m3_and_h32(mid, dtype) if dp.rotated else round_e4m3_per128(mid)
            if plan is not None:
                dn_sorted = jh_kernels.gather_qmm_expert_sorted(din, dp.tq2_packed, dp.tq2_scales, s_routes, dp.bits, plan)
            else:
                dn_sorted = jh_kernels.gather_qmm_sorted(din, dp.tq2_packed, dp.tq2_scales, dp._cb, s_routes, dp.bits)

        fused_out = jh_kernels.weighted_unsort(dn_sorted, inv, rw_2d, dtype)
        if fused_out is not None:
            return fused_out.reshape(*lead, self.down_proj.output_dims)
        raw = dn_sorted[inv].reshape(T, top_k, self.down_proj.output_dims)
        out = (raw.astype(mx.float32) * rw_2d[..., None]).sum(axis=1)
        return out.astype(dtype).reshape(*lead, self.down_proj.output_dims)

    @staticmethod
    def _jt_sorted_proj(xr_bf16: mx.array, x_orig: mx.array, unit: Dict[str, Any], s_routes: mx.array, plan=None) -> mx.array:
        out = gather_qmm_sorted_jt(xr_bf16, unit, s_routes, plan=plan).astype(mx.float32)
        if unit["n_out"]:
            cols = unit["ocol"][s_routes]
            orig = mx.take_along_axis(x_orig, mx.maximum(cols, 0), axis=-1)
            orig = mx.where(cols >= 0, orig, 0).astype(mx.float32)
            outliers = unit["wout"].astype(mx.float32).swapaxes(-1, -2)
            out = out + mx.gather_mm(orig[:, None, :], outliers, rhs_indices=s_routes, sorted_indices=True)[:, 0]
        return out


# ---------------------------------------------------------------------------
# Native Full MXFP4 SwitchGLU for ERNIE 5.5 Flash (M5 Ultra / 256 GB+)
# Fuses gate_proj (w1) + up_proj (w3) into ONE MXFP4 gather_qmm dispatch!
# ---------------------------------------------------------------------------
class ErnieMXFP4SwitchGLU(nn.Module):
    """Native MXFP4 (bits=4, group_size=32, UE8M0) routed SwitchGLU for ERNIE 5.5 Flash."""

    accepts_route_weights = True
    SORT_THRESHOLD = 160

    def __init__(
        self,
        num_experts: int,
        input_dims: int,
        hidden_dims: int,
        *,
        beta: float = 4.0,
        linear_beta: float = 25.0,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.input_dims = int(input_dims)
        self.hidden_dims = int(hidden_dims)
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)
        self.gu_weight = mx.zeros((num_experts, 2 * hidden_dims, input_dims // 8), dtype=mx.uint32)
        self.gu_scales = mx.zeros((num_experts, 2 * hidden_dims, input_dims // 32), dtype=mx.uint8)
        self.down_weight = mx.zeros((num_experts, input_dims, hidden_dims // 8), dtype=mx.uint32)
        self.down_scales = mx.zeros((num_experts, input_dims, hidden_dims // 32), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x: mx.array, indices: mx.array, route_weights: mx.array) -> mx.array:
        dtype = x.dtype
        lead = x.shape[:-1]
        K_in = x.shape[-1]
        top_k = indices.shape[-1]
        T = x.size // K_in
        H = self.hidden_dims

        x2 = x.reshape(T, K_in)
        idx_2d = indices.reshape(T, top_k).astype(mx.uint32)
        rw_2d = route_weights.reshape(T, top_k).astype(mx.float32)

        if T * top_k >= self.SORT_THRESHOLD:
            routes = idx_2d.reshape(-1)
            order = mx.argsort(routes)
            inv = mx.argsort(order)
            s_routes = routes[order]
            tok_idx = order // top_k

            gu = mx.gather_qmm(
                x2[tok_idx][:, None, :],
                self.gu_weight,
                self.gu_scales,
                None,
                rhs_indices=s_routes,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=True,
            )[:, 0, :]
            mid = fused_situ(gu[:, H:], gu[:, :H], self.beta, self.linear_beta)
            dn_sorted = mx.gather_qmm(
                mid[:, None, :],
                self.down_weight,
                self.down_scales,
                None,
                rhs_indices=s_routes,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=True,
            )[:, 0, :]
            fused_out = jh_kernels.weighted_unsort(dn_sorted, inv, rw_2d, dtype)
            if fused_out is not None:
                return fused_out.reshape(*lead, K_in)
            raw = dn_sorted[inv].reshape(T, top_k, K_in)
            return (raw.astype(mx.float32) * rw_2d[..., None]).sum(axis=1).astype(dtype).reshape(*lead, K_in)

        gu = mx.gather_qmm(
            x2[:, None, None, :],
            self.gu_weight,
            self.gu_scales,
            None,
            rhs_indices=idx_2d,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )[:, :, 0, :]
        mid = fused_situ(gu[..., H:], gu[..., :H], self.beta, self.linear_beta)
        dn = mx.gather_qmm(
            mid[:, :, None, :],
            self.down_weight,
            self.down_scales,
            None,
            rhs_indices=idx_2d,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )[:, :, 0, :]
        out = (dn.astype(mx.float32) * rw_2d[..., None]).sum(axis=1)
        return out.astype(dtype).reshape(*lead, K_in)


def pack_mxfp4_experts(
    expert_tensors: Dict[str, mx.array],
    expert_prefix: str,
    num_experts: int,
    input_dims: int,
    hidden_dims: int,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> ErnieMXFP4SwitchGLU:
    """Stack 512 per-expert MXFP4 (I8 weight + UE8M0 scale) tensors into fused ErnieMXFP4SwitchGLU."""
    gu_w_list: List[mx.array] = []
    gu_s_list: List[mx.array] = []
    dn_w_list: List[mx.array] = []
    dn_s_list: List[mx.array] = []
    for e in range(num_experts):
        ep = f"{expert_prefix}{e}."
        w1 = mx.contiguous(expert_tensors.pop(ep + "w1.weight")).view(mx.uint32)
        s1 = mx.contiguous(expert_tensors.pop(ep + "w1.weight_scale_inv")).view(mx.uint8)
        w3 = mx.contiguous(expert_tensors.pop(ep + "w3.weight")).view(mx.uint32)
        s3 = mx.contiguous(expert_tensors.pop(ep + "w3.weight_scale_inv")).view(mx.uint8)
        w2 = mx.contiguous(expert_tensors.pop(ep + "w2.weight")).view(mx.uint32)
        s2 = mx.contiguous(expert_tensors.pop(ep + "w2.weight_scale_inv")).view(mx.uint8)
        gu_w_list.append(mx.concatenate([w1, w3], axis=0))
        gu_s_list.append(mx.concatenate([s1, s3], axis=0))
        dn_w_list.append(w2)
        dn_s_list.append(s2)

    mod = ErnieMXFP4SwitchGLU(
        num_experts, input_dims, hidden_dims, beta=beta, linear_beta=linear_beta
    )
    mod.gu_weight = mx.stack(gu_w_list, axis=0)
    mod.gu_scales = mx.stack(gu_s_list, axis=0)
    mod.down_weight = mx.stack(dn_w_list, axis=0)
    mod.down_scales = mx.stack(dn_s_list, axis=0)
    mx.eval(mod.gu_weight, mod.gu_scales, mod.down_weight, mod.down_scales)
    return mod


def _read_tensors(root: Path, weight_map: Dict[str, str], names: Sequence[str]) -> Dict[str, mx.array]:
    requested: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        sf = weight_map[name]
        if Path(sf).name != sf or not sf.endswith(".safetensors"):
            raise ValueError(f"Unsupported shard name: {sf}")
        requested[sf].append(name)
    out: Dict[str, mx.array] = {}
    for sf, keys in requested.items():
        arrays = mx.load(str(root / sf))
        picked = {k: arrays[k] for k in keys}
        # Materialize raw safetensors on the CPU Load stream before queuing any
        # downstream GPU transforms so slow external-SSD reads never trigger
        # Metal's command-buffer watchdog timeout.
        if picked:
            mx.eval(*picked.values())
        out.update(picked)
    return out


def _make_expert_proj(
    n_in: int, n_out: int, n_exp: int, key: str, entry: Dict[str, Any], packed: Dict[str, mx.array], trellis: Optional[Dict[str, Any]]
) -> nn.Module:
    mode = entry["mode"]
    if mode == "jangtq2":
        if entry.get("rotation") != "hadamard32" or entry.get("bits") not in (2, 3):
            raise ValueError(f"Unsupported ERNIE JANGH layout on {key}: {entry}")
        return TQSwitchLinear(n_in, n_out, n_exp, int(entry["bits"]), "hadamard32")
    if mode == "jangt":
        if not trellis or trellis.get("state_bits") != 12 or trellis.get("rotation") != "hadamard128-signs" or entry.get("kbits") != 2:
            raise ValueError(f"Unsupported ERNIE JANGT layout on {key}: {entry}")
        n_ocol = int(trellis["outlier_columns"][key])
        if packed[key + ".jt_ocol"].shape[-1] != n_ocol:
            raise ValueError(f"Outlier column count disagrees with tensor on {key}")
        return JTSwitchLinear(n_in, n_out, n_exp, 2, n_ocol, False)
    raise ValueError(f"Unsupported ERNIE expert mode on {key}: {mode}")


def load_ernie5_5_moe(model_path: str | Path, *, keep_exact_mxfp8: bool = True) -> LanguageModel:
    """Load an ERNIE 5.5 Flash bundle (JANGHT2.4 OR native MXFP4) with exact MXFP8 trunk and fused kernels."""
    root = Path(model_path)
    settings = json.loads((root / "config.json").read_text())
    config = ModelConfig.from_dict(settings)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    declared = settings.get("quantization") or {}
    trellis = settings.get("jangt")
    qcfg = settings.get("quantization_config") or {}
    is_mxfp4 = (
        str(qcfg.get("expert_dtype", "")).lower() == "mxfp4"
        or "model.layers.1.block_sparse_moe.experts.0.w1.weight" in weight_map
    )

    model = LanguageModel(config)

    # 1. Load global parameters (embed_tokens, hc_head, norm, lm_head, multimax)
    # Exclude both decoder layers and inline `model.mtp.*` DSpark drafter weights.
    global_keys = [
        k for k in weight_map
        if not k.startswith("model.layers.") and not k.startswith("model.mtp.")
    ]
    global_mapped = LanguageModel.sanitize_dict(config, _read_tensors(root, weight_map, global_keys))
    model.load_weights(list(global_mapped.items()), strict=False)
    mx.eval(*global_mapped.values())
    del global_mapped

    total_mxfp8 = 0
    total_saved = 0
    # 2. Load decoder layers one by one to keep peak load memory flat
    for num, block in enumerate(model.layers):
        prefix = f"model.layers.{num}."
        packed_prefix = prefix + "mlp.switch_mlp."
        mxfp4_expert_prefix = prefix + "block_sparse_moe.experts."
        layer_keys = [k for k in weight_map if k.startswith(prefix)]
        raw = _read_tensors(root, weight_map, layer_keys)
        if is_mxfp4:
            expert_raw = {k: v for k, v in raw.items() if k.startswith(mxfp4_expert_prefix)}
            packed = {}
            trunk = {k: v for k, v in raw.items() if k not in expert_raw}
        else:
            expert_raw = {}
            packed = {k: v for k, v in raw.items() if k.startswith(packed_prefix)}
            trunk = {k: v for k, v in raw.items() if k not in packed}
        del raw

        switch_glu: Optional[ErnieSwitchGLU] = None
        if num >= config.first_k_dense_replace:
            n_lat = config.moe_latent_size
            n_int = config.moe_intermediate_size
            n_exp = config.n_routed_experts
            if is_mxfp4:
                block.ffn.switch_mlp = pack_mxfp4_experts(
                    expert_raw,
                    mxfp4_expert_prefix,
                    n_exp,
                    n_lat,
                    n_int,
                    beta=config.activation_situ_beta,
                    linear_beta=config.activation_situ_linear_beta,
                )
                del expert_raw
            else:
                gp = _make_expert_proj(n_lat, n_int, n_exp, packed_prefix + "gate_proj", declared[packed_prefix + "gate_proj"], packed, trellis)
                up = _make_expert_proj(n_lat, n_int, n_exp, packed_prefix + "up_proj", declared[packed_prefix + "up_proj"], packed, trellis)
                dp = _make_expert_proj(n_int, n_lat, n_exp, packed_prefix + "down_proj", declared[packed_prefix + "down_proj"], packed, trellis)
                switch_glu = ErnieSwitchGLU(
                    gp, up, dp, beta=config.activation_situ_beta, linear_beta=config.activation_situ_linear_beta
                )
                block.ffn.switch_mlp = switch_glu

        if keep_exact_mxfp8:
            keep_fp8_linears(prefix, trunk)
        sanitized = LanguageModel.sanitize_dict(config, trunk)
        layer_weights = {k.removeprefix(prefix): v for k, v in sanitized.items()}
        del trunk, sanitized

        if keep_exact_mxfp8:
            c_mx, s_mx = install_direct_mxfp8_linears(block, layer_weights)
            total_mxfp8 += c_mx
            total_saved += s_mx

        for k, v in packed.items():
            layer_weights[k.replace(packed_prefix, "ffn.switch_mlp.", 1)] = v
        del packed

        block.load_weights(list(layer_weights.items()), strict=not is_mxfp4)
        mx.eval(block.parameters())
        if switch_glu is not None:
            switch_glu.prepare()
        del layer_weights
        gc.collect()
        mx.clear_cache()

    model.eval()
    prep = prepare_ernie_runtime(model.layers)
    if os.environ.get("VMLX_ERNIE_MHC_BF16", "1") != "0":
        hc_head = getattr(model.model, "hc_head", None)
        if hc_head is not None and getattr(hc_head, "fn", None) is not None and hc_head.fn.dtype == mx.float32:
            fn_b = hc_head.fn.astype(mx.bfloat16)
            mx.eval(fn_b)
            hc_head.fn = fn_b
            prep["mhc_bf16"] = prep.get("mhc_bf16", 0) + 1
    gc.collect()
    mx.clear_cache()
    logger.info(
        "ERNIE 5.5 Flash (%s) loaded: mxfp8_linears=%d saved=%.2f GiB prep=%s active=%.2f GiB",
        "MXFP4" if is_mxfp4 else "JANGHT",
        total_mxfp8, total_saved / 2**30, prep, mx.get_active_memory() / 2**30,
    )
    return model

# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer loader for Hy4-flash (`hy_v4_flash`) — MXFP4, FP8, and BF16 checkpoints.

Key properties:
- **Zero-copy disk-backed Engram** (`118.02 GiB` across shards `00001..00004` stays on NVMe SSD
  via `HyV4DiskEngramTable`, using `0 GiB` GPU RAM!).
- **CPU-first shard materialization** (`mx.eval(*picked.values())` on the CPU Load stream before
  queuing any GPU dequantize/pack transforms so slow external-SSD reads never hit Metal's GPU
  command-buffer watchdog timeout!).
- **Native MXFP4 experts**: packs `512` routed experts + shared experts directly into
  `HyV4MXFP4SwitchGLU` & `nn.QuantizedLinear(mode="mxfp4", bits=4, group_size=32)` (~84.3 GiB
  total GPU RAM including inline DFly2!).
- **Block-128 FP8 attention**: repacks power-of-two `128x128` FP8 scales into exact MXFP8
  `QuantizedLinear(mode="mxfp8", bits=8, group_size=32)` (or exact BF16 dequantization for
  non-power-of-two scales).
- **Auto-loads inline `model.dfly_layers.*`** unless `VMLX_DFLY2=0`.
"""

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
import numpy as np

from .config import HyV4FlashConfig
from .dfly2 import HyV4DFly2Drafter
from .engram import HyV4DiskEngramTable, HyV4EngramHasher
from .kernels import HyV4MXFP4SwitchGLU
from .language import HyV4FlashForCausalLM

logger = logging.getLogger(__name__)


def _read_tensors(root: Path, weight_map: Dict[str, str], names: Sequence[str]) -> Dict[str, mx.array]:
    """Read requested tensors shard-by-shard and materialize on CPU Load stream before GPU ops."""
    requested: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        sf = weight_map[name]
        if Path(sf).name != sf or not sf.endswith(".safetensors"):
            raise ValueError(f"Unsupported shard name: {sf}")
        requested[sf].append(name)
    out: Dict[str, mx.array] = {}
    for sf, keys in requested.items():
        arrays = mx.load(str(root / sf))
        picked = {k: arrays[k] for k in keys if k in arrays}
        if picked:
            mx.eval(*picked.values())
        out.update(picked)
    return out


def _dequantize_fp8_128(weight_u8: mx.array, scale_inv: mx.array) -> mx.array:
    """Dequantize 128x128 block-FP8 E4M3 weight with FP32/BF16 block scale."""
    rows, cols = int(weight_u8.shape[0]), int(weight_u8.shape[1])
    w_f32 = mx.from_fp8(weight_u8, dtype=mx.float32)
    s_rep = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), 128, axis=0), 128, axis=1)[:rows, :cols]
    return (w_f32 * s_rep).astype(mx.bfloat16)


def _try_mxfp8_view(weight_u8: mx.array, scale_inv: mx.array) -> Optional[Tuple[mx.array, mx.array]]:
    """Exact lossless repack of power-of-two 128x128 FP8 E4M3 weights into MLX MXFP8 (group=32)."""
    if weight_u8.dtype != mx.uint8 or weight_u8.ndim != 2:
        return None
    rows, cols = int(weight_u8.shape[0]), int(weight_u8.shape[1])
    if cols % 32 != 0:
        return None
    s_f32 = scale_inv.astype(mx.float32)
    if s_f32.shape != ((rows + 127) // 128, (cols + 127) // 128):
        return None
    raw = s_f32.view(mx.uint32)
    exponent = (raw >> 23) & 255
    valid = ((raw & 0x807FFFFF) == 0) & (exponent > 0) & (exponent < 255)
    if not bool(mx.all(valid).item()):
        return None
    packed_scale = mx.repeat(mx.repeat(exponent.astype(mx.uint8), 128, axis=0), 4, axis=1)
    return mx.contiguous(weight_u8).view(mx.uint32), mx.contiguous(packed_scale[:rows, : cols // 32])


def _install_mxfp4_linear(w_packed: mx.array, w_scale: mx.array) -> nn.QuantizedLinear:
    """Wrap packed OCP MXFP4 `[out, in//2]` uint8 + `[out, in//32]` uint8 scales into QuantizedLinear."""
    w_u32 = mx.contiguous(w_packed).view(mx.uint32) if w_packed.dtype == mx.uint8 else mx.contiguous(w_packed)
    s_u8 = mx.contiguous(w_scale).astype(mx.uint8)
    out_d = int(w_u32.shape[0])
    in_d = int(w_u32.shape[1]) * 8
    lin = nn.QuantizedLinear(in_d, out_d, bias=False, group_size=32, bits=4, mode="mxfp4")
    lin.weight = w_u32
    lin.scales = s_u8
    lin.freeze()
    return lin


def _install_fp8_or_bf16_linear(
    weight: mx.array,
    scale: Optional[mx.array],
    *,
    keep_exact_mxfp8: bool = True,
) -> nn.Module:
    """Install either exact MXFP8 QuantizedLinear or BF16 Linear for an attention/draft projection."""
    if scale is not None and weight.dtype == mx.uint8:
        if keep_exact_mxfp8:
            repacked = _try_mxfp8_view(weight, scale)
            if repacked is not None:
                codes, scales = repacked
                out_d, in_d = int(weight.shape[0]), int(weight.shape[1])
                lin = nn.QuantizedLinear(in_d, out_d, bias=False, group_size=32, bits=8, mode="mxfp8")
                lin.weight = codes
                lin.scales = scales
                lin.freeze()
                return lin
        bf = _dequantize_fp8_128(weight, scale)
        lin_b = nn.Linear(int(bf.shape[1]), int(bf.shape[0]), bias=False)
        lin_b.weight = bf
        return lin_b
    bf = weight.astype(mx.bfloat16) if weight.dtype != mx.bfloat16 else weight
    lin_b = nn.Linear(int(bf.shape[1]), int(bf.shape[0]), bias=False)
    lin_b.weight = bf
    return lin_b


def _pack_hy4_mxfp4_experts(
    expert_raw: Dict[str, mx.array],
    expert_prefix: str,
    n_experts: int,
    latent_dim: int,
    intermediate_dim: int,
    *,
    swiglu_limit: float = 10.0,
) -> HyV4MXFP4SwitchGLU:
    """Pack either 3-D fused (`experts.gate_up_proj_packed`) or per-expert (`experts.{e}.*`) MXFP4 weights."""
    mod = HyV4MXFP4SwitchGLU(n_experts, latent_dim, intermediate_dim, swiglu_limit=swiglu_limit)

    def _to_u32(t: mx.array) -> mx.array:
        return mx.contiguous(t).view(mx.uint32) if t.dtype == mx.uint8 else mx.contiguous(t)

    # Case A: already stacked 3-D fused keys (e.g. `experts.gate_up_proj_packed` / `weight_packed`)
    for gu_w_k, gu_s_k, dn_w_k, dn_s_k in (
        ("gate_up_proj.weight_packed", "gate_up_proj.weight_scale", "down_proj.weight_packed", "down_proj.weight_scale"),
        ("gate_up_proj_packed", "gate_up_proj_scale", "down_proj_packed", "down_proj_scale"),
        ("gate_up_proj", "gate_up_proj_scale", "down_proj", "down_proj_scale"),
    ):
        if (expert_prefix + gu_w_k) in expert_raw and (expert_prefix + dn_w_k) in expert_raw:
            mod.gu_weight = _to_u32(expert_raw[expert_prefix + gu_w_k])
            mod.gu_scales = mx.contiguous(expert_raw[expert_prefix + gu_s_k]).astype(mx.uint8)
            mod.down_weight = _to_u32(expert_raw[expert_prefix + dn_w_k])
            mod.down_scales = mx.contiguous(expert_raw[expert_prefix + dn_s_k]).astype(mx.uint8)
            mx.eval(mod.gu_weight, mod.gu_scales, mod.down_weight, mod.down_scales)
            return mod

    # Case B: per-expert keys `experts.{e}.{gate_proj, up_proj, down_proj}.{weight_packed, weight_scale}`
    # OR `experts.{e}.{gate_up_proj, down_proj}.{weight_packed, weight_scale}`
    gu_ws, gu_ss, dn_ws, dn_ss = [], [], [], []
    for e in range(n_experts):
        ep = f"{expert_prefix}{e}."
        if (ep + "gate_up_proj.weight_packed") in expert_raw:
            gu_ws.append(_to_u32(expert_raw[ep + "gate_up_proj.weight_packed"]))
            gu_ss.append(mx.contiguous(expert_raw[ep + "gate_up_proj.weight_scale"]).astype(mx.uint8))
        else:
            gw = _to_u32(expert_raw[ep + "gate_proj.weight_packed"])
            gs = mx.contiguous(expert_raw[ep + "gate_proj.weight_scale"]).astype(mx.uint8)
            uw = _to_u32(expert_raw[ep + "up_proj.weight_packed"])
            us = mx.contiguous(expert_raw[ep + "up_proj.weight_scale"]).astype(mx.uint8)
            gu_ws.append(mx.concatenate([gw, uw], axis=0))
            gu_ss.append(mx.concatenate([gs, us], axis=0))
        dn_ws.append(_to_u32(expert_raw[ep + "down_proj.weight_packed"]))
        dn_ss.append(mx.contiguous(expert_raw[ep + "down_proj.weight_scale"]).astype(mx.uint8))

    mod.gu_weight = mx.stack(gu_ws, axis=0)
    mod.gu_scales = mx.stack(gu_ss, axis=0)
    mod.down_weight = mx.stack(dn_ws, axis=0)
    mod.down_scales = mx.stack(dn_ss, axis=0)
    mx.eval(mod.gu_weight, mod.gu_scales, mod.down_weight, mod.down_scales)
    return mod


def _load_inline_dfly2(
    root: Path,
    weight_map: Dict[str, str],
    config: HyV4FlashConfig,
    *,
    keep_exact_mxfp8: bool = True,
) -> Optional[HyV4DFly2Drafter]:
    """Load inline `model.dfly_layers.*` + `mtp_fusion` + `hidden_correction` + `confidence_head`."""
    dfly_keys = [
        k for k in weight_map
        if (
            k.startswith("model.dfly_layers.")
            or k.startswith("model.mtp_fusion.")
            or k.startswith("model.hidden_correction.")
            or k.startswith("model.dfly_final_norm.")
            or k.startswith("model.dfly_confidence_head.")
            or k.startswith("model.confidence_head.")
            or k.startswith("dfly_layers.")
        )
    ]
    if not dfly_keys:
        return None

    raw = _read_tensors(root, weight_map, dfly_keys)
    # Infer intermediate_size directly from gate_proj weight shape (9216 for inline B6)
    inter_dim = 9216
    for k, v in raw.items():
        if "dfly_layers.0.mlp.gate_proj.weight" in k and not k.endswith("_scale") and not k.endswith("_scale_inv"):
            inter_dim = int(v.shape[0])
            break
    has_conf = any("confidence_head" in k for k in raw)
    drafter = HyV4DFly2Drafter(config, intermediate_size=inter_dim, has_confidence_head=has_conf)

    # 1. Load each DFly2 layer
    for idx, dlayer in enumerate(drafter.layers):
        for pfx in (f"model.dfly_layers.{idx}.", f"dfly_layers.{idx}."):
            if (pfx + "input_layernorm.weight") in raw:
                break
        else:
            continue
        dlayer.input_layernorm.weight = raw[pfx + "input_layernorm.weight"].astype(mx.bfloat16)
        dlayer.post_attention_layernorm.weight = raw[pfx + "post_attention_layernorm.weight"].astype(mx.bfloat16)

        sa = dlayer.self_attn
        sp = pfx + "self_attn."
        sa.q_a_proj.weight = raw[sp + "q_a_proj.weight"].astype(mx.bfloat16)
        sa.q_a_layernorm.weight = raw[sp + "q_a_layernorm.weight"].astype(mx.bfloat16)
        sa.kv_a_proj_with_mqa.weight = raw[sp + "kv_a_proj_with_mqa.weight"].astype(mx.bfloat16)
        sa.kv_a_layernorm.weight = raw[sp + "kv_a_layernorm.weight"].astype(mx.bfloat16)
        sa.set_kv_b_weight(raw[sp + "kv_b_proj.weight"].astype(mx.bfloat16))
        sa.learnable_sink_param = raw[sp + "learnable_sink_param"].astype(mx.float32)
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
        sa.post_proj_norm.weight = raw[sp + "post_proj_norm.weight"].astype(mx.bfloat16)

        for attr in ("q_b_proj", "linear_gate", "o_proj"):
            wk = sp + f"{attr}.weight"
            sk = raw.get(wk + "_scale_inv", raw.get(wk + "_scale"))
            setattr(sa, attr, _install_fp8_or_bf16_linear(raw[wk], sk, keep_exact_mxfp8=keep_exact_mxfp8))

        mp = pfx + "mlp."
        for attr in ("gate_proj", "up_proj", "down_proj"):
            wk = mp + f"{attr}.weight"
            sk = raw.get(wk + "_scale_inv", raw.get(wk + "_scale"))
            setattr(dlayer.mlp, attr, _install_fp8_or_bf16_linear(raw[wk], sk, keep_exact_mxfp8=keep_exact_mxfp8))

    # 2. Aux heads: mtp_fusion, hidden_correction, final_norm, confidence_head
    def _pick(suffix: str) -> Optional[mx.array]:
        for k, v in raw.items():
            if k.endswith(suffix):
                return v
        return None

    for name, mod_attr in (
        ("mtp_fusion.enorm.weight", ("mtp_fusion", "enorm", "weight")),
        ("mtp_fusion.hnorm.weight", ("mtp_fusion", "hnorm", "weight")),
        ("mtp_fusion.eh_proj.weight", ("mtp_fusion", "eh_proj", "weight")),
        ("mtp_fusion.gate", ("mtp_fusion", "gate")),
    ):
        t = _pick(name)
        if t is not None:
            obj: Any = drafter
            for a in mod_attr[:-1]:
                obj = getattr(obj, a)
            setattr(obj, mod_attr[-1], t.astype(mx.bfloat16))

    if drafter.hidden_correction is not None:
        hc = drafter.hidden_correction
        for name, target_pair in (
            ("hidden_correction.hidden_norm.weight", (hc.hidden_norm, "weight")),
            ("hidden_correction.embed_norm.weight", (hc.embed_norm, "weight")),
            ("hidden_correction.gate_proj.weight", (hc.gate_proj, "weight")),
            ("hidden_correction.up_proj.weight", (hc.up_proj, "weight")),
            ("hidden_correction.down_proj.weight", (hc.down_proj, "weight")),
        ):
            t = _pick(name)
            if t is not None:
                setattr(target_pair[0], target_pair[1], t.astype(mx.bfloat16))

    fn = _pick("dfly_final_norm.weight") or _pick("final_norm.weight")
    if fn is not None:
        drafter.final_norm.weight = fn.astype(mx.bfloat16)

    if drafter.confidence_head is not None:
        cw = _pick("confidence_head.proj.weight")
        cb = _pick("confidence_head.proj.bias")
        if cw is not None:
            drafter.confidence_head.weight = cw.astype(mx.bfloat16)
        if cb is not None:
            drafter.confidence_head.bias = cb.astype(mx.bfloat16)

    mx.eval(drafter.parameters())
    drafter.eval()
    del raw
    gc.collect()
    mx.clear_cache()
    return drafter


def load_hy_v4_flash(
    model_path: str | Path,
    *,
    keep_exact_mxfp8: bool = True,
    load_dfly2: bool = True,
) -> HyV4FlashForCausalLM:
    """Load Hy4-flash (`hy_v4_flash`) with zero-copy disk-backed Engram (0 GiB GPU RAM) & inline DFly2."""
    root = Path(model_path)
    raw_cfg = json.loads((root / "config.json").read_text())
    config = HyV4FlashConfig.from_dict(raw_cfg)
    weight_map: Dict[str, str] = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    # Detect MXFP4 experts strictly from checkpoint keys / quantization_config (zero model-name regex!)
    qcfg_str = json.dumps(config.quantization_config).lower()
    is_mxfp4 = (
        "mxfp4" in qcfg_str
        or any(k.endswith(".weight_packed") and ".mlp." in k for k in weight_map)
        or any("gate_up_proj_packed" in k for k in weight_map)
    )

    model = HyV4FlashForCausalLM(config, mxfp4=is_mxfp4)

    # 1. Load global trunk tensors (embed_tokens, hc_head, norm, lm_head) — exclude layers, vit, dfly
    global_keys = [
        k for k in weight_map
        if not k.startswith("model.layers.")
        and not k.startswith("vit.")
        and not k.startswith("visual.")
        and "dfly" not in k
        and "mtp_fusion" not in k
        and "hidden_correction" not in k
        and "confidence_head" not in k
    ]
    g_raw = _read_tensors(root, weight_map, global_keys)
    model.model.embed_tokens.weight = g_raw["model.embed_tokens.weight"].astype(mx.bfloat16)
    model.model.norm.weight = g_raw["model.norm.weight"].astype(mx.bfloat16)
    model.lm_head.weight = g_raw["lm_head.weight"].astype(mx.bfloat16)
    model.model.hc_head.hc_head_fn = g_raw["model.hc_head.hc_head_fn"].astype(mx.float32)
    model.model.hc_head.hc_head_base = g_raw["model.hc_head.hc_head_base"].astype(mx.float32)
    model.model.hc_head.hc_head_scale = g_raw["model.hc_head.hc_head_scale"].astype(mx.float32)
    mx.eval(
        model.model.embed_tokens.parameters(),
        model.model.norm.parameters(),
        model.lm_head.parameters(),
        model.model.hc_head.parameters(),
    )
    del g_raw
    gc.collect()
    mx.clear_cache()

    # 2. Load decoder layers one by one; bind Engram embedding tables strictly via disk mmap (0 GiB GPU RAM!)
    engram_lookup_np: Optional[np.ndarray] = None
    disk_engram_bytes = 0

    for num, block in enumerate(model.layers):
        prefix = f"model.layers.{num}."
        eng_big_key = prefix + "engram_layer.multi_head_embedding.embedding.weight"
        eng_scale_key = prefix + "engram_layer.multi_head_embedding.embedding.weight_scale"
        if eng_big_key in weight_map and block.engram_layer is not None:
            shard_path = root / weight_map[eng_big_key]
            sc_name = eng_scale_key if eng_scale_key in weight_map else None
            tbl = HyV4DiskEngramTable(shard_path, eng_big_key, scale_tensor_name=sc_name)
            block.engram_layer.bind_table(tbl)
            disk_engram_bytes += int(tbl.rows.nbytes) + (int(tbl.scales.nbytes) if tbl.scales is not None else 0)

        # Read all non-big-Engram tensors for this layer on CPU stream first
        layer_keys = [
            k for k in weight_map
            if k.startswith(prefix) and k != eng_big_key and k != eng_scale_key
        ]
        raw = _read_tensors(root, weight_map, layer_keys)

        # 2a. iHC weights (forced FP32)
        for hc_attr, hc_mod in (("hc_attn_layer", block.hc_attn_layer), ("hc_mlp_layer", block.hc_mlp_layer)):
            hp = f"{prefix}{hc_attr}.hc_pre."
            hc_mod.hc_fn = raw[hp + "hc_fn"].astype(mx.float32)
            hc_mod.hc_base = raw[hp + "hc_base"].astype(mx.float32)
            hc_mod.hc_scale = raw[hp + "hc_scale"].astype(mx.float32)

        block.input_layernorm.weight = raw[prefix + "input_layernorm.weight"].astype(mx.bfloat16)
        block.post_attention_layernorm.weight = raw[prefix + "post_attention_layernorm.weight"].astype(mx.bfloat16)

        # 2b. Engram layer projections, norms & frozen lookup_table
        if block.engram_layer is not None:
            ep = prefix + "engram_layer."
            eng = block.engram_layer
            if engram_lookup_np is None and (ep + "lookup_table") in raw:
                engram_lookup_np = np.array(raw[ep + "lookup_table"], dtype=np.int64)
            for h in range(eng.n_heads):
                eng.lookup_per_head_norms[h].weight = raw[f"{ep}lookup_per_head_norms.{h}.weight"].astype(mx.bfloat16)
            for c in range(eng.hc_mult):
                eng.key_projs[c].weight = raw[f"{ep}key_projs.{c}.weight"].astype(mx.bfloat16)
                eng.key_projs[c].bias = raw[f"{ep}key_projs.{c}.bias"].astype(mx.bfloat16)
                eng.norm_key[c].weight = raw[f"{ep}norm_key.{c}.weight"].astype(mx.bfloat16)
                eng.norm_query[c].weight = raw[f"{ep}norm_query.{c}.weight"].astype(mx.bfloat16)
                eng.unified_per_hc_scales[c] = raw[f"{ep}unified_per_hc_scales.{c}"].astype(mx.bfloat16)
            eng.value_proj.weight = raw[ep + "value_proj.weight"].astype(mx.bfloat16)
            eng.value_proj.bias = raw[ep + "value_proj.bias"].astype(mx.bfloat16)
            eng.unified_pre_gamma = raw[ep + "unified_pre_gamma"].astype(mx.bfloat16)
            eng.prepare_runtime()

        # 2c. Self-Attention (DSA or SWA Gated MLA)
        sa = block.self_attn
        sp = prefix + "self_attn."
        sa.q_a_proj.weight = raw[sp + "q_a_proj.weight"].astype(mx.bfloat16)
        sa.q_a_layernorm.weight = raw[sp + "q_a_layernorm.weight"].astype(mx.bfloat16)
        sa.kv_a_proj_with_mqa.weight = raw[sp + "kv_a_proj_with_mqa.weight"].astype(mx.bfloat16)
        sa.kv_a_layernorm.weight = raw[sp + "kv_a_layernorm.weight"].astype(mx.bfloat16)
        sa.set_kv_b_weight(raw[sp + "kv_b_proj.weight"].astype(mx.bfloat16))
        sa.learnable_sink_param = raw[sp + "learnable_sink_param"].astype(mx.float32)
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
        sa.post_proj_norm.weight = raw[sp + "post_proj_norm.weight"].astype(mx.bfloat16)

        for attr in ("q_b_proj", "linear_gate", "o_proj"):
            wk = sp + f"{attr}.weight"
            sk = raw.get(wk + "_scale_inv", raw.get(wk + "_scale"))
            setattr(sa, attr, _install_fp8_or_bf16_linear(raw[wk], sk, keep_exact_mxfp8=keep_exact_mxfp8))

        if sa.compressed_indexer is not None:
            ci = sa.compressed_indexer
            cp = sp + "compressed_indexer."
            ci.query_projection.weight = raw[cp + "query_projection.weight"].astype(mx.bfloat16)
            ci.head_weight_projection.weight = raw[cp + "head_weight_projection.weight"].astype(mx.bfloat16)
            ci.token_key_projection.weight = raw[cp + "token_key_projection.weight"].astype(mx.bfloat16)
            ci.token_key_norm.weight = raw[cp + "token_key_norm.weight"].astype(mx.bfloat16)
            ci.key_value_projection.weight = raw[cp + "key_value_projection.weight"].astype(mx.float32)
            ci.key_gate_projection.weight = raw[cp + "key_gate_projection.weight"].astype(mx.float32)
            ci.key_norm.weight = raw[cp + "key_norm.weight"].astype(mx.float32)
            ci.intra_group_ape = raw[cp + "intra_group_ape"].astype(mx.float32)

        # 2d. MLP (Dense Layer 0 vs LatentMoE Layers 1..41)
        mp = prefix + "mlp."
        if not block.is_moe:
            block.mlp.gate_proj.weight = raw[mp + "gate_proj.weight"].astype(mx.bfloat16)
            block.mlp.up_proj.weight = raw[mp + "up_proj.weight"].astype(mx.bfloat16)
            block.mlp.down_proj.weight = raw[mp + "down_proj.weight"].astype(mx.bfloat16)
            block.mlp.post_fc2_norm.weight = raw[mp + "post_fc2_norm.weight"].astype(mx.bfloat16)
        else:
            moe = block.mlp
            moe.gate.weight = raw[mp + "gate.weight"].astype(mx.bfloat16)
            moe.e_score_correction_bias = raw[mp + "gate.e_score_correction_bias"].astype(mx.float32)
            moe.routed_expert_down_proj.weight = raw[mp + "routed_expert_down_proj.weight"].astype(mx.bfloat16)
            moe.routed_expert_up_proj.weight = raw[mp + "routed_expert_up_proj.weight"].astype(mx.bfloat16)
            moe.combine_output_norm.weight = raw[mp + "combine_output_norm.weight"].astype(mx.bfloat16)

            # Shared experts (MXFP4 or BF16)
            sh = moe.shared_experts
            for attr in ("gate_proj", "up_proj", "down_proj"):
                pk = f"{mp}shared_experts.{attr}.weight_packed"
                if pk in raw:
                    setattr(sh, attr, _install_mxfp4_linear(raw[pk], raw[f"{mp}shared_experts.{attr}.weight_scale"]))
                else:
                    wk = f"{mp}shared_experts.{attr}.weight"
                    sk = raw.get(wk + "_scale_inv", raw.get(wk + "_scale"))
                    setattr(sh, attr, _install_fp8_or_bf16_linear(raw[wk], sk, keep_exact_mxfp8=keep_exact_mxfp8))

            # Routed experts (MXFP4 or FP8/BF16)
            exp_pfx = mp + "experts."
            exp_raw = {k: v for k, v in raw.items() if k.startswith(exp_pfx)}
            if is_mxfp4:
                moe.switch_mlp = _pack_hy4_mxfp4_experts(
                    exp_raw,
                    exp_pfx,
                    config.n_routed_experts,
                    config.routed_expert_hidden_size,
                    config.moe_intermediate_size,
                    swiglu_limit=config.swiglu_limit,
                )
            else:
                # FP8 block-128 or BF16 3-D stacked experts
                if (exp_pfx + "gate_up_proj") in exp_raw:
                    gu = exp_raw[exp_pfx + "gate_up_proj"]
                    gus = exp_raw.get(exp_pfx + "gate_up_proj_scale_inv")
                    dn = exp_raw[exp_pfx + "down_proj"]
                    dns = exp_raw.get(exp_pfx + "down_proj_scale_inv")
                    if gus is not None and gu.dtype == mx.uint8:
                        gu = mx.stack([_dequantize_fp8_128(gu[e], gus[e]) for e in range(gu.shape[0])], axis=0)
                    if dns is not None and dn.dtype == mx.uint8:
                        dn = mx.stack([_dequantize_fp8_128(dn[e], dns[e]) for e in range(dn.shape[0])], axis=0)
                    moe.switch_mlp.gu_weight = gu.astype(mx.bfloat16)
                    moe.switch_mlp.down_weight = dn.astype(mx.bfloat16)
                else:
                    gu_list, dn_list = [], []
                    for e in range(config.n_routed_experts):
                        ep = f"{exp_pfx}{e}."
                        if (ep + "gate_up_proj") in exp_raw:
                            gu_e = exp_raw[ep + "gate_up_proj"]
                            gus_e = exp_raw.get(ep + "gate_up_proj_scale_inv")
                            gu_list.append(_dequantize_fp8_128(gu_e, gus_e) if gus_e is not None else gu_e.astype(mx.bfloat16))
                        else:
                            gw = exp_raw[ep + "gate_proj.weight"].astype(mx.bfloat16)
                            uw = exp_raw[ep + "up_proj.weight"].astype(mx.bfloat16)
                            gu_list.append(mx.concatenate([gw, uw], axis=0))
                        dn_e = exp_raw.get(ep + "down_proj", exp_raw.get(ep + "down_proj.weight"))
                        dns_e = exp_raw.get(ep + "down_proj_scale_inv")
                        dn_list.append(_dequantize_fp8_128(dn_e, dns_e) if dns_e is not None else dn_e.astype(mx.bfloat16))
                    moe.switch_mlp.gu_weight = mx.stack(gu_list, axis=0)
                    moe.switch_mlp.down_weight = mx.stack(dn_list, axis=0)
            del exp_raw

        mx.eval(block.parameters())
        del raw
        gc.collect()
        mx.clear_cache()

    if engram_lookup_np is not None and config.engram_layer_ids:
        model.model.set_engram_hasher(HyV4EngramHasher(config, engram_lookup_np))

    # 3. Auto-attach inline DFly2 drafter unless VMLX_DFLY2=0
    if load_dfly2 and os.environ.get("VMLX_DFLY2", "1") != "0":
        model._dfly_drafter = _load_inline_dfly2(root, weight_map, config, keep_exact_mxfp8=keep_exact_mxfp8)

    model.eval()
    gc.collect()
    mx.clear_cache()
    logger.info(
        "Hy4-flash (%s) loaded: resident_gpu=%.2f GiB | disk_engram_mmap=%.2f GiB | inline_dfly2=%s",
        "MXFP4" if is_mxfp4 else "FP8/BF16",
        mx.get_active_memory() / 2**30,
        disk_engram_bytes / 2**30,
        model._dfly_drafter is not None,
    )
    return model

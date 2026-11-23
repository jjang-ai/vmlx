"""Inkling-2 JANGHT bundles: every routed projection bank keeps its own format.

Bundle contract (config.json; the shipped JANGH2 bundle plus a per-unit format choice):
  quantization["model.layers.L.mlp.switch_mlp.<proj>"] for every layer L and proj in gate_proj / up_proj / down_proj:
    {"mode": "jangtq2", "bits": b, "rotation": "hadamard32"}          JANGH codebook (config "jangtq" declaration)
    {"mode": "jangt", "kbits": K}                                       JANGT trellis (config "jangt" block)
    {"mode": "nvfp4", "group_size": 16, "bits": 4}                     the source's NVFP4 bytes + per-expert global_scale
    {"mode": "mxfp8", "group_size": 32, "bits": 8}                     the source's MXFP8 bytes
  gate_proj and up_proj share one entry; down may differ. Everything else (attention / embeddings / lm_head affine
  8-bit g64, router fp32, loop and attn-res tensors, vision tower, native drafter) is the JANGH2 layout.

A layer whose three banks are all native keeps the native Inkling SwitchGLU (fused NVFP4 decode where enabled, the
masked verify kernels). Any other layer becomes a jangt.switch.JTMixedSwitchGLU whose gate/up and down groups are each
JANGT, JANGH or native, with the model's own soft-capped activation (moe_kernels "inkling"; caps from the model config,
never from the bundle). Every unsupported or inconsistent entry fails the load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig
from .inkling2 import Model
from .processing_inkling2 import Inkling2Processor

logger = logging.getLogger(__name__)

ROUTED = ("gate_proj", "up_proj", "down_proj")
_SWITCH = ".mlp.switch_mlp."
JANGT_BLOCK = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs"}
# routed tensors and their stored dtypes, per group format (the bundle contract)
_ROUTED_DTYPES = {
    "tq2_packed": mx.uint32, "tq2_scales": mx.float16,
    "jt_packed": mx.uint32, "jt_scale": mx.float32, "jt_su": mx.float32, "jt_ocol": mx.int32,
    "jt_wout": mx.float16, "jt_cs": mx.float32,
    "weight": mx.uint32, "scales": mx.uint8, "global_scale": mx.float32,
}


def is_janght_config(config: dict) -> bool:
    """An Inkling bundle whose routed banks follow the per-unit JANGHT contract (config "jangt" block present)."""
    return str(config.get("model_type", "")) == "inkling2_loop" and config.get("jangt") is not None


def _fail(message: str):
    raise ValueError("Inkling JANGHT: " + message)


def routed_contract(config: dict, num_layers: int) -> dict[int, dict]:
    """Validate the per-unit routed contract. Returns {layer: {"gu", "dn", "gu_fmt", "dn_fmt", "n_out_gu",
    "n_out_dn"}}; fails closed on any missing, extra, inconsistent or unsupported entry."""
    from vmlx_engine.jangt import v2l as VL
    from vmlx_engine.jangt.switch import group_format

    jt = config.get("jangt")
    if not isinstance(jt, dict) or any(jt.get(k) != v for k, v in JANGT_BLOCK.items()):
        _fail(f"unsupported jangt block {jt}")
    outliers = jt.get("outlier_columns", {})
    if not isinstance(outliers, dict):
        _fail("jangt outlier_columns must be an object")
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        _fail("per-module quantization is required")
    if any(isinstance(v, dict) and v.get("mode") == "jangtq2" for v in quant.values()):
        from vmlx_engine.jangh.contract import validate_format

        if not validate_format(config):          # raises on a bad codebook declaration
            _fail("jangtq2 entries require the jangtq v2 declaration")

    routed_keys = {k for k in quant if isinstance(k, str) and _SWITCH in k}
    expected = {f"model.layers.{L}.mlp.switch_mlp.{p}" for L in range(num_layers) for p in ROUTED}
    if routed_keys != expected:
        _fail(f"routed entries differ from the {num_layers}-layer stack: missing {sorted(expected - routed_keys)[:4]}, "
              f"extra {sorted(routed_keys - expected)[:4]}")
    stray = set(outliers) - expected
    if stray:
        _fail(f"outlier columns for unknown modules {sorted(stray)[:4]}")

    contract, used_outliers = {}, set()
    for L in range(num_layers):
        key = f"model.layers.{L}.mlp.switch_mlp"
        gate, up, down = (quant[f"{key}.{p}"] for p in ROUTED)
        if not isinstance(gate, dict) or gate != up:
            _fail(f"gate/up entries differ for {key}: {gate} vs {up}")
        if not isinstance(down, dict):
            _fail(f"down entry is not an object for {key}")
        unit = {"gu": dict(gate), "dn": dict(down)}
        for group, entry in (("gu", gate), ("dn", down)):
            fmt = group_format(entry)                       # jangt | jangtq2 | native (validates native group/bits)
            if fmt == "jangt":
                kbits = entry.get("kbits")
                if type(kbits) not in (int, float) or kbits not in VL.MULTS or set(entry) - {"mode", "kbits"}:
                    _fail(f"unsupported jangt entry {entry} for {key}")
            elif fmt == "jangtq2":
                if entry.get("bits") not in (2, 3, 4) or entry.get("rotation") != "hadamard32":
                    _fail(f"unsupported jangtq2 entry {entry} for {key}")
            unit[f"{group}_fmt"] = fmt
        projections = {"gu": ("gate_proj", "up_proj"), "dn": ("down_proj",)}
        for group in ("gu", "dn"):
            counts = set()
            for p in projections[group]:
                name = f"{key}.{p}"
                n = outliers.get(name, 0)
                if type(n) is not int or n < 0:
                    _fail(f"invalid outlier column count {n!r} for {name}")
                if n and unit[f"{group}_fmt"] != "jangt":
                    _fail(f"outlier columns declared for non-trellis unit {name}")
                if name in outliers:
                    used_outliers.add(name)
                counts.add(n)
            if len(counts) != 1:
                _fail(f"gate/up outlier column counts differ for {key}")
            unit[f"n_out_{group}"] = counts.pop()
        contract[L] = unit
    return contract


def install_routed_units(language_model, contract: dict, gate_cap: float, up_cap: float) -> dict:
    """Replace the routed SwitchGLU of every layer that is not all-native with a JTMixedSwitchGLU (before quantize)."""
    from vmlx_engine.jangt.switch import JTMixedSwitchGLU

    counts = {}
    for L, layer in enumerate(language_model.model.layers):
        unit = contract[L]
        tag = f"{unit['gu_fmt']}/{unit['dn_fmt']}"
        counts[tag] = counts.get(tag, 0) + 1
        if unit["gu_fmt"] == unit["dn_fmt"] == "native":
            continue
        sw = layer.mlp.switch_mlp
        E, I, D = sw.gate_proj.weight.shape
        layer.mlp.switch_mlp = JTMixedSwitchGLU(
            D, I, E, unit["gu"], unit["dn"], unit["n_out_gu"], unit["n_out_dn"], 0.0, "inkling", (gate_cap, up_cap),
        )
    return counts


def _bundle_key(path: str) -> str:
    """Runtime module path -> its on-disk quantization key."""
    return path[len("language_model."):] if path.startswith("language_model.") else path


def load_janght_inkling2(path: Path, config: dict):
    path = Path(path)
    model = Model(ModelConfig.from_dict(config))
    text = model.config.text_config
    contract = routed_contract(config, text.num_hidden_layers)
    uses_nvfp4 = any(contract[L][g].get("mode") == "nvfp4" for L in contract for g in ("gu", "dn"))
    if uses_nvfp4 and "global_scale" not in (mx.gather_qmm.__doc__ or ""):
        raise RuntimeError("Inkling Mini NVFP4 requires MLX 0.32.4 or newer")
    units = install_routed_units(model.language_model, contract, text.mlp_gate_softcap, text.mlp_up_softcap)

    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        shard = Path(name)
        if shard.name != name or shard.suffix != ".safetensors":
            raise ValueError(f"Invalid Inkling target shard name: {name!r}")
        weights.update(mx.load(path / shard))
    weights = model.sanitize(weights)          # drops mtp_drafter.* (DSpark loads it separately)

    # A trellis unit without outlier columns may omit (or ship zero-width) jt_ocol / jt_wout: keep the -1 / 0 fill.
    optional = set()
    for L, unit in contract.items():
        for group, projections in (("gu", ("gate_proj", "up_proj")), ("dn", ("down_proj",))):
            if unit[f"{group}_fmt"] == "jangt" and unit[f"n_out_{group}"] == 0:
                for p in projections:
                    for leaf in ("jt_ocol", "jt_wout"):
                        k = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{leaf}"
                        optional.add(k)
                        if k in weights and weights[k].shape[-1] == 0:
                            del weights[k]

    quant = config["quantization"]
    root = {"group_size": int(quant.get("group_size", 64)), "bits": int(quant.get("bits", 8)),
            "mode": str(quant.get("mode", "affine"))}
    consumed = set()

    def predicate(name, module):
        if not hasattr(module, "to_quantized"):
            return False
        key = _bundle_key(name)
        entry = quant.get(key)
        if isinstance(entry, dict):
            consumed.add(key)
            mode = entry.get("mode", root["mode"])
            if mode in ("jangt", "jangtq2"):
                return dict(entry)                  # the installed unit validates its own entry (to_quantized)
            if f"{name}.scales" not in weights:
                _fail(f"{key} is declared {mode} but the bundle has no {key}.scales")
            return {"group_size": int(entry.get("group_size", root["group_size"])),
                    "bits": int(entry.get("bits", root["bits"])), "mode": mode}
        if f"{name}.scales" in weights:
            return dict(root)
        return False

    nn.quantize(model, class_predicate=predicate)
    unused = sorted(k for k, v in quant.items() if isinstance(v, dict) and k not in consumed)
    if unused:
        _fail(f"quantization entries match no module: {unused[:6]} ({len(unused)} total)")

    from .nvfp4 import replace_scaled_quantized_linears

    replace_scaled_quantized_linears(model, weights)

    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    allowed_missing = set(missing) & optional
    unexpected = sorted(set(weights) - set(expected))
    if set(missing) - allowed_missing or unexpected:
        _fail(f"weights differ from the runtime: missing={sorted(set(missing) - allowed_missing)[:8]}, "
              f"unexpected={unexpected[:8]}")
    for k, v in weights.items():
        ref = expected[k]
        if tuple(v.shape) != tuple(ref.shape):
            _fail(f"{k} has shape {tuple(v.shape)}, the runtime expects {tuple(ref.shape)}")
        if _SWITCH in k:
            want = _ROUTED_DTYPES.get(k.rsplit(".", 1)[1])
            if want is None or v.dtype != want:
                _fail(f"{k} has dtype {v.dtype}, the contract stores {want}")
    model.load_weights(list(weights.items()), strict=False)
    del weights
    mx.eval(model.parameters())
    model.eval()

    from .fused_nvfp4_experts import prepare_fused_nvfp4_experts
    from .jangh import prepare_jangh_decode

    prepare_jangh_decode(model)                 # packed affine attention decode (same as the JANGH2 bundle)
    fused_layers = prepare_fused_nvfp4_experts(model.language_model)
    if fused_layers:
        logger.info("Inkling NVFP4 fused expert decode enabled for %d native layers", fused_layers)
        for layer in model.language_model.model.layers:
            for controller in getattr(layer, "controllers", {}).values():
                controller.prepare_fused_decode()
    model._inkling_janght_units = dict(units)
    logger.info("Inkling JANGHT routed units (gate_up/down): %s", ", ".join(f"{k} x{v}" for k, v in sorted(units.items())))
    return model, Inkling2Processor.from_pretrained(path)

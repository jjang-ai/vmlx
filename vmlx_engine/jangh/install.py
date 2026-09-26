"""vMLX runtime hook: install JANGTQ v2 routed experts into a constructed model before nn.quantize.

Contract (fail closed):
  * config["jangtq"]["version"] == 2 and per-module entries {"mode": "jangtq2", "bits": b} for every
    <layer>.mlp.switch_mlp.{gate,up,down}_proj of every routed-MoE layer.
  * gate_proj and up_proj bits must match (fused kernel), rotation must be absent or declared per module.
  * Every SwitchGLU that has a jangtq2 entry is replaced; any jangtq2 entry that matches no module is an error,
    and any MoE SwitchGLU left without an entry in a jangtq2 bundle is an error (no silent mixed stacks).
"""
from __future__ import annotations

import logging

import mlx.nn as nn

from .switch import TQSwitchGLU

logger = logging.getLogger(__name__)


def _cfg_key(module_path: str) -> str:
    """Runtime module path -> on-disk config key (the bundle uses model.layers.N... naming)."""
    for pre in ("language_model.model.", "model.language_model.", "language_model."):
        if module_path.startswith(pre):
            return "model." + module_path[len(pre):]
    return module_path


from .contract import (
    alias_runtime_quant_keys,
    projection_contract,
    validate_format,
)


def is_jangh(config: dict) -> bool:
    return validate_format(config)


def install_jangh(model: nn.Module, config: dict) -> int:
    from mlx_lm.models.switch_layers import SwitchGLU

    contracts = projection_contract(config)
    if not contracts:
        return 0
    limit = float(config.get("text_config", config).get("swiglu_limit", 0.0) or 0.0)
    inventory = {}
    for path, mod in list(model.named_modules()):
        sw = getattr(mod, "switch_mlp", None)
        if not isinstance(sw, (SwitchGLU, TQSwitchGLU)):
            continue
        if "mtp" in path.split("."):
            continue
        key = _cfg_key(f"{path}.switch_mlp")
        if key in inventory:
            raise ValueError(f"jangtq2: duplicate routed module {key}")
        inventory[key] = (mod, sw)
    if set(inventory) != set(contracts):
        raise ValueError("jangtq2: routed module inventory differs from quantization entries")
    prepared = []
    for key, (mod, sw) in inventory.items():
        gu, dn = contracts[key]["gate_proj"], contracts[key]["down_proj"]
        if isinstance(sw, TQSwitchGLU):
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(sw, name)
                expected = contracts[key][name]
                if linear.bits != expected["bits"] or linear.rotation != expected["rotation"]:
                    raise ValueError(f"jangtq2: installed contract differs for {key}.{name}")
            if sw.limit != limit:
                raise ValueError("jangtq2: installed SwiGLU limit differs")
            continue
        gate, up, down = sw.gate_proj.weight.shape, sw.up_proj.weight.shape, sw.down_proj.weight.shape
        if len(gate) != 3 or up != gate or len(down) != 3 or down != (gate[0], gate[2], gate[1]):
            raise ValueError(f"jangtq2: inconsistent expert dimensions for {key}")
        E, I, D = gate
        for width, spec in ((D, gu), (I, dn)):
            if width % 32:
                raise ValueError(f"jangtq2: unsupported packed/rotation dimensions for {key}")
        prepared.append((mod, D, I, E, gu, dn))
    replacements = []
    for mod, D, I, E, gu, dn in prepared:
        replacement = TQSwitchGLU(D, I, E, gu["bits"], dn["bits"], limit,
                                  rotation_gate_up=gu["rotation"], rotation_down=dn["rotation"])
        replacement.is_jangh = True
        replacement.is_jangtq2 = True
        replacements.append((mod, replacement))
    for mod, replacement in replacements:
        mod.switch_mlp = replacement
    logger.info("JANGTQ v2: installed %d TQSwitchGLU modules", len(replacements))
    return len(replacements)

# Existing integrations retain the same callable, not a second implementation.
is_jangtq2 = is_jangh
install_jangtq2 = install_jangh

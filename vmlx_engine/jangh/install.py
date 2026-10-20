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

from .switch import TQSwitchGLU, TQSwitchMLP

logger = logging.getLogger(__name__)

# Gate-less relu2 stacks (Nemotron-H, e.g. Nemotron-3.5-Bolt: D=1024 latent, I=1536-2560, E=240-496, k=9): per-row gather
# (decode) vs sorted (prefill) crossover, measured on the whole Bolt forward at a warm 1.5k-token state, M5 Max / MLX
# 0.32.3, median of 7: gather 38 / 50 / 58 / 70 / 77 ms vs sorted 113 / 102 / 103 / 102 / 100 ms at 8 / 12 / 16 / 24 / 32
# tokens, tie at 48 (117 vs 116), sorted wins from 64 (120 vs 140). The generic rule (sorted from 64 routed rows = 7
# tokens at k=9) sent every 8-40-token forward (short cached follow-ups, chunk tails, speculative verify) to the sorted
# path, which carries ~100 ms of fixed cost here. JANGH_RELU2_DECODE_MAX_TOKENS overrides.
RELU2_DECODE_MAX_TOKENS = 40


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
    try:
        from mlx_lm.models.switch_layers import SwitchMLP
    except ImportError:
        SwitchMLP = None
    # gate-less relu2 stacks (Nemotron-H). Resolved through globals: the installer contract tests execute this body in
    # a namespace that provides only TQSwitchGLU, with a switch_layers double that defines only SwitchGLU.
    tq_mlp = globals().get("TQSwitchMLP")
    MLP_TYPES = tuple(t for t in (tq_mlp, SwitchMLP) if t is not None)
    contracts = projection_contract(config)
    if not contracts:
        return 0
    limit = float(config.get("text_config", config).get("swiglu_limit", 0.0) or 0.0)
    inventory = {}
    for path, mod in list(model.named_modules()):
        sw = getattr(mod, "switch_mlp", None)
        if not isinstance(sw, (SwitchGLU, TQSwitchGLU) + MLP_TYPES):
            continue
        if "mtp" in path.split("."):
            continue
        key = _cfg_key(f"{path}.switch_mlp")
        if key in inventory:
            raise ValueError(f"jangtq2: duplicate routed module {key}")
        inventory[key] = (mod, sw)
    if set(inventory) != set(contracts):
        raise ValueError("jangtq2: routed module inventory differs from quantization entries")
    prepared, mlp_prepared = [], []
    for key, (mod, sw) in inventory.items():
        if isinstance(sw, MLP_TYPES):
            # gate-less relu2 stack (Nemotron-H): fc1 (E, I, D) / fc2 (E, D, I), independent widths
            if set(contracts[key]) != {"fc1", "fc2"}:
                raise ValueError(f"jangtq2: {key} is a fc1/fc2 expert stack but its entries are {sorted(contracts[key])}")
            c1, c2 = contracts[key]["fc1"], contracts[key]["fc2"]
            if tq_mlp is not None and isinstance(sw, tq_mlp):
                for name, expected in (("fc1", c1), ("fc2", c2)):
                    linear = getattr(sw, name)
                    if linear.bits != expected["bits"] or linear.rotation != expected["rotation"]:
                        raise ValueError(f"jangtq2: installed contract differs for {key}.{name}")
                continue
            w1, w2 = sw.fc1.weight.shape, sw.fc2.weight.shape
            if len(w1) != 3 or len(w2) != 3 or w2 != (w1[0], w1[2], w1[1]):
                raise ValueError(f"jangtq2: inconsistent expert dimensions for {key}")
            E, I, D = w1
            if D % 32 or I % 32:
                raise ValueError(f"jangtq2: unsupported packed/rotation dimensions for {key}")
            mlp_prepared.append((mod, D, I, E, c1, c2))
            continue
        if set(contracts[key]) != set(("gate_proj", "up_proj", "down_proj")):
            raise ValueError(f"jangtq2: {key} is a SwiGLU expert stack but its entries are {sorted(contracts[key])}")
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
        import os as _os

        if (D, I, E) == (3072, 1024, 256) and gu["rotation"] == dn["rotation"] == "hadamard32" \
                and _os.environ.get("JANGH_ORNITH_EXPERT_TILES", "1").strip() != "0":
            # Expert-aligned NAX prefill tiles at this geometry (Ornith-1.5-122B): same tq_gather_qmm_nax arithmetic as
            # the generic sorted route, only the tile -> row plan differs, so outputs are bit-identical (measured,
            # tests/test_jangh_expert_tiles_d3072.py); expert GEMMs 1.10-1.29x faster at 2-4k-row prefill segments.
            # switch._use_expert_tiles re-checks shapes, bits, rotation and NAX on every call.
            replacement.use_expert_tiles = True
        replacements.append((mod, replacement))
    if mlp_prepared:
        # resolved only when relu2 stacks exist: the installer contract tests run this body in a namespace that
        # provides only TQSwitchGLU (see the tq_mlp lookup above)
        import os

        raw_limit = os.environ.get("JANGH_RELU2_DECODE_MAX_TOKENS", "").strip()
        relu2_limit = int(raw_limit) if raw_limit.isdigit() else RELU2_DECODE_MAX_TOKENS
    for mod, D, I, E, c1, c2 in mlp_prepared:
        replacement = tq_mlp(D, I, E, c1["bits"], c2["bits"], rotation_fc1=c1["rotation"], rotation_fc2=c2["rotation"])
        replacement.is_jangh = True
        replacement.is_jangtq2 = True
        replacement.decode_max_tokens = relu2_limit
        replacements.append((mod, replacement))
    for mod, replacement in replacements:
        mod.switch_mlp = replacement
    logger.info("JANGTQ v2: installed %d routed modules (%d TQSwitchMLP relu2)", len(replacements), len(mlp_prepared))
    return len(replacements)

# Existing integrations retain the same callable, not a second implementation.
is_jangtq2 = is_jangh
install_jangtq2 = install_jangh


def enable_minicpm_fused(model: nn.Module) -> int:
    """MiniCPM-V-4.7-35B-A3B routed JANGH banks: fused fp32 weighted unsort + single-launch H32 rows (prefill) and the
    in-kernel H32 rotation (decode). Measured faster at E=256 / D=2048 / I=512 / k=8 (runtime_identity.MINICPM_FUSED);
    JANGH_MINICPM_FUSED=0 keeps the generic paths. Returns the number of banks configured."""
    from .runtime_identity import MINICPM_FUSED

    if MINICPM_FUSED != "1":
        return 0
    n = 0
    for _path, mod in model.named_modules():
        if isinstance(mod, TQSwitchGLU):
            mod.use_weighted_unsort = True
            mod.decode_rot_kernel = True
            for lin in (mod.gate_proj, mod.up_proj, mod.down_proj):
                lin.use_h32_rows = True
            n += 1
    if n:
        logger.info("MiniCPM-V-4.7 JANGH: fused prefill unsort + H32 rows and in-kernel decode rotation on %d banks", n)
    return n

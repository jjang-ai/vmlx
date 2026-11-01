"""Bundled DSpark drafter -> serving runtime (target adapter + drafter + width ladder), shared by every lane.

Contract: a bundle's ``dspark/`` directory (or ``jang_config.json["speculative"]["drafter_path"]``) holding a drafter
``config.json`` that declares DSpark plus its safetensors. On/off follows the FILES (Eric 2026-10-13): a shipped
drafter is used by default, a bundle without one serves plain decoding (never an error), ``VMLX_DSPARK=0`` keeps any
bundle on plain decoding. The stamp's ``speculative.default`` is informational only.

Targets: MiniCPM-V-4.7 (``minicpmv4_7``, the multimodal lane) and MiniMax-M3 / M3.1 (``minimax_m3`` /
``minimax_m3_vl``, the text lane). Drafters: openbmb ``Qwen3DSparkModel`` / SpecForge ``DSparkDraftModel`` and
MiniMax ``DSparkMiniMaxDraftModel`` (as released with MXFP8 linears, or pre-quantized).

Draft width (Eric's rule: no hard draft depth): Adaptive picks the width of every round (``generate.ARFallback``:
acceptance history / confidence head, measured AR fallback). The ladder's top is the target's verify coverage
(``target.max_verify_rows``) and otherwise twice the drafter's trained block (the DFlash2 lane's ceiling); a round
drafting past the trained block runs a longer block. Bundle stamps (``speculative.max_draft``, a drafter's
``block_size`` / ``dspark_block_size``, any recommended depth) never lower it. Diagnostic env levers only:
``VMLX_DSPARK_MAX_DRAFT=<n>`` (ladder top) and ``VMLX_DSPARK_WIDTH=<n>`` (pin every round's width).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# model_type -> (module under vmlx_engine.dspark, target class). Looked up at call time.
TARGETS = {
    "minicpmv4_7": ("minicpm_target", "MiniCPMDSparkTarget"),
    "minimax_m3": ("minimax_target", "MiniMaxM3DSparkTarget"),
    "minimax_m3_vl": ("minimax_target", "MiniMaxM3DSparkTarget"),
}
TEXT_LANE_MODEL_TYPES = frozenset({"minimax_m3", "minimax_m3_vl"})


def config_declares_dspark(cfg: Any) -> bool:
    """openbmb MiniCPM-V-4.7 drafters (``Qwen3DSparkModel``, ``dflash_config.projector_type: dspark``), SpecForge
    ``DSparkDraftModel`` and MiniMax ``DSparkMiniMaxDraftModel``. Mirrors
    panel/src/main/dsparkBundle.ts::configDeclaresDspark."""
    if not isinstance(cfg, dict):
        return False
    if any("dspark" in str(a).lower() for a in cfg.get("architectures") or []):
        return True
    projector = (cfg.get("dflash_config") or {}).get("projector_type") or cfg.get("projector_type")
    return str(projector or "").lower() == "dspark"


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignored", name, raw)
        return None
    return value if value >= 1 else None


def ladder_top(target, drafter) -> int:
    """The widest draft Adaptive may choose: the target's verify coverage, else twice the trained block."""
    top = 2 * int(drafter.block)
    cover = getattr(target, "max_verify_rows", None)
    if cover is not None:
        top = min(top, int(cover) - 1)
    pinned = _env_int("VMLX_DSPARK_MAX_DRAFT")
    if pinned is not None:
        logger.warning("DSpark: VMLX_DSPARK_MAX_DRAFT=%d limits the draft ladder (diagnostic lever)", pinned)
        top = pinned if cover is None else min(pinned, int(cover) - 1)
    return max(1, top)


def _advisory_stamps(spec: dict, drafter_cfg: dict, drafter, top: int) -> None:
    text = drafter_cfg.get("text_config") or drafter_cfg
    stamps = {
        "speculative.max_draft": spec.get("max_draft"),
        "speculative.recommended_num_drafts": spec.get("recommended_num_drafts"),
        "drafter block": drafter.block,
        "drafter recommended_num_drafts": text.get("recommended_num_drafts"),
    }
    present = {k: v for k, v in stamps.items() if v is not None}
    logger.info("DSpark: stamps %s are advisory (start points only); Adaptive picks each round's width up to %d",
                present, top)


def load_dspark_runtime(model: Any, model_path: str | Path, *, model_types=None) -> dict | None:
    """The DSpark runtime for a bundle that ships a drafter, or None (plain decoding).

    ``model_types``: the target families this caller serves (default: every family in ``TARGETS``). Raises when the
    drafter is present but unusable (corrupt weights, geometry that does not fit the target); callers log and serve
    plain decoding."""
    override = os.environ.get("VMLX_DSPARK", "").strip().lower()
    if override in {"0", "false", "off", "no"}:
        return None
    root = Path(model_path)
    try:
        cfg = json.loads((root / "config.json").read_text())
    except Exception:
        return None
    model_type = str(cfg.get("model_type", "")).lower()
    allowed = TARGETS.keys() if model_types is None else set(model_types) & TARGETS.keys()
    if model_type not in allowed:
        return None
    spec = {}
    stamp_path = root / "jang_config.json"
    if stamp_path.is_file():
        try:
            spec = json.loads(stamp_path.read_text()).get("speculative") or {}
        except Exception:
            spec = {}
    if spec and str(spec.get("method", "")).lower() not in ("", "dspark"):
        return None
    drafter_dir = root / str(spec.get("drafter_path") or "dspark")
    if not (drafter_dir / "config.json").is_file():
        if spec:
            logger.info("DSpark declared in jang_config but the bundle ships no drafter (%s); serving AR", drafter_dir)
        return None
    # Same rule as the panel (dsparkBundle.configDeclaresDspark): a sidecar config that does not declare DSpark is
    # not a DSpark drafter.
    try:
        dcfg_raw = json.loads((drafter_dir / "config.json").read_text())
    except Exception:
        dcfg_raw = {}
    if not config_declares_dspark(dcfg_raw):
        logger.info("%s/config.json does not declare a DSpark drafter; serving AR", drafter_dir)
        return None
    from ..bundled_draft import validate_bundled_draft

    # Admit real, target-compatible weights before allocating a drafter or attaching hooks to the target.
    validate_bundled_draft(drafter_dir, root)
    from . import load_drafter

    module_name, class_name = TARGETS[model_type]
    target_cls = getattr(importlib.import_module(f"{__package__}.{module_name}"), class_name)
    target = target_cls(model)
    drafter = load_drafter(drafter_dir)
    text_cfg = cfg.get("text_config") or cfg
    dcfg = drafter.cfg
    if int(dcfg["hidden_size"]) != int(text_cfg.get("hidden_size", -1)):
        raise ValueError("DSpark drafter hidden_size does not match the target")
    if int(dcfg["vocab_size"]) > int(text_cfg.get("vocab_size", 0)):
        raise ValueError("DSpark drafter vocabulary is larger than the target's")
    if max(drafter.target_layer_ids) >= int(text_cfg.get("num_hidden_layers", 0)):
        raise ValueError("DSpark drafter taps layers the target does not have")
    max_draft = ladder_top(target, drafter)
    _advisory_stamps(spec, dcfg_raw, drafter, max_draft)
    fixed_width = _env_int("VMLX_DSPARK_WIDTH")
    if fixed_width is not None:
        logger.warning("DSpark: VMLX_DSPARK_WIDTH=%d pins every round's width (diagnostic lever)", fixed_width)
    logger.info("DSpark drafter loaded from %s: %s target, block %d, taps %s, draft ladder 1..%d (Adaptive)",
                drafter_dir, model_type, drafter.block, drafter.target_layer_ids, max_draft)
    return {"target": target, "drafter": drafter, "path": str(drafter_dir), "max_draft": max_draft,
            "fixed_width": fixed_width, "cost_per_row": float(spec.get("cost_per_row", 0.12)),
            "model_type": model_type}


def attach_text_lane(model: Any, model_name: str | Path) -> dict | None:
    """Attach the DSpark runtime of a text-lane bundle (MiniMax-M3 / M3.1 shipping ``dspark/``) to the loaded model as
    ``model._vmlx_dspark_runtime`` (outside the module's parameter tree), where the scheduler picks the DSpark text
    generator from. A missing or unusable drafter leaves the model on plain decoding (logged, never an error)."""
    try:
        from ..utils.model_inspector import resolve_model_path

        root = Path(resolve_model_path(str(model_name)))
    except Exception:
        return None
    try:
        cfg = json.loads((root / "config.json").read_text())
    except Exception:
        return None
    if str(cfg.get("model_type", "")).lower() not in TEXT_LANE_MODEL_TYPES:
        return None
    try:
        runtime = load_dspark_runtime(model, root, model_types=TEXT_LANE_MODEL_TYPES)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSpark drafter not loaded for %s: %s; serving plain decoding", root, exc)
        return None
    if runtime is not None:
        object.__setattr__(model, "_vmlx_dspark_runtime", runtime)
    return runtime

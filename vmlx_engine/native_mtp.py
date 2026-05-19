# SPDX-License-Identifier: Apache-2.0
"""Native MTP autodetect and mlx-lm runtime activation.

This module is intentionally metadata-first. A bundle is treated as native-MTP
ready only when config/sidecar metadata and the real safetensor index agree
that MTP tensors are present. Runtime activation is then narrowed to families
with a vMLX-backed draft/verify path.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DISABLE_ENV_VALUES = {"0", "false", "FALSE", "no", "NO", "off", "OFF"}

_FAMILY_ALIAS = {
    "qwen3_5_text": "qwen3_5",
    "qwen3_6": "qwen3_5",
    "qwen3_6_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",
}

# Keep this narrow. The copied mlx-lm patch currently wires Qwen3.5/3.6 MTP
# through GenerationBatch. DeepSeek-V4 has model-side code copied for later
# work, but vMLX's DSV4 runtime uses a custom generator, so do not advertise
# it as native-MTP active yet.
_RUNTIME_SUPPORTED_FAMILIES = {
    "qwen3_5",
    "qwen3_5_moe",
}

_ENABLE_ENV_VALUES = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_ACTIVE_NATIVE_MTP_MODEL_PATH: Path | None = None


def _read_json(bundle_path: str | Path | None, name: str) -> dict[str, Any]:
    if not bundle_path:
        return {}
    try:
        path = Path(bundle_path) / name
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_index(bundle_path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if not bundle_path:
        return None, None
    try:
        path = Path(bundle_path) / "model.safetensors.index.json"
        if not path.is_file():
            return None, None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None, None
    except Exception as exc:
        return None, f"model.safetensors.index.json could not be read: {exc}"


def _bundle_weight_keys(
    bundle_path: str | Path | None,
) -> tuple[list[str], str, str | None]:
    """Return real tensor keys from the bundle index or safetensors headers.

    MTP/VL detection must be artifact driven. The fast path reads the
    Hugging Face safetensors index when present; single-shard MLX/JANG/MXFP4
    bundles often omit that index, so the fallback scans safetensors headers
    with ``safe_open``. This does not materialize tensor data.
    """
    if not bundle_path:
        return [], "none", None

    index, error = _read_index(bundle_path)
    if error:
        return [], "index", error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if isinstance(weight_map, dict):
        return [str(key) for key in weight_map], "index", None
    if index is not None and not isinstance(weight_map, dict):
        return [], "index", "model.safetensors.index.json has no weight_map object"

    try:
        from safetensors import safe_open
    except Exception as exc:
        return [], "safetensors", f"safetensors header reader unavailable: {exc}"

    keys: list[str] = []
    errors: list[str] = []
    try:
        paths = sorted(Path(bundle_path).glob("*.safetensors"))
    except Exception as exc:
        return [], "safetensors", f"safetensors files could not be listed: {exc}"
    for shard in paths:
        try:
            with safe_open(str(shard), framework="numpy") as handle:
                keys.extend(str(key) for key in handle.keys())
        except Exception as exc:
            errors.append(f"{shard.name}: {exc}")
    if errors:
        return keys, "safetensors", "safetensors header read failed: " + "; ".join(errors)
    return keys, "safetensors", None


def _normalize_family(name: str | None) -> str | None:
    if not name:
        return None
    value = str(name).strip().lower()
    return _FAMILY_ALIAS.get(value, value)


def _bundle_family(cfg: dict[str, Any], jang_cfg: dict[str, Any]) -> str | None:
    capabilities = jang_cfg.get("capabilities") or {}
    for raw in (
        capabilities.get("family") if isinstance(capabilities, dict) else None,
        cfg.get("model_type"),
        (cfg.get("text_config") or {}).get("model_type")
        if isinstance(cfg.get("text_config"), dict)
        else None,
    ):
        normalized = _normalize_family(raw)
        if normalized and normalized != "unknown":
            return normalized
    return None


def _coerce_non_negative_int(raw: Any) -> tuple[int | None, bool]:
    if raw is None:
        return None, False
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, True
    return value, value < 0


def _coerce_native_mtp_depth(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(3, value))


def _model_tuning_depth(
    bundle_path: str | Path | None,
) -> tuple[int | None, str | None]:
    """Read a model-local measured MTP depth hint without probing the model."""
    tuning = _read_json(bundle_path, "vmlx_mtp_tuning.json")
    if not tuning:
        return None, None

    candidates: list[tuple[str, Any]] = []
    native_mtp = tuning.get("native_mtp")
    if isinstance(native_mtp, dict):
        native_mtp_depth_allowed = (
            native_mtp.get("blocked") is not True
            and native_mtp.get("validated") is not False
            and native_mtp.get("output_equivalent") is not False
        )
        if native_mtp_depth_allowed:
            candidates.append(
                (
                    "vmlx_mtp_tuning.json:native_mtp.best_depth",
                    native_mtp.get("best_depth"),
                )
            )

    sweep_result = tuning.get("best_native_mtp_depth")
    if isinstance(sweep_result, dict):
        candidates.append(
            (
                "vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth",
                sweep_result.get("best_depth"),
            )
        )

    candidates.append(("vmlx_mtp_tuning.json:best_depth", tuning.get("best_depth")))

    for source, raw_depth in candidates:
        depth = _coerce_native_mtp_depth(raw_depth)
        if depth is not None:
            return depth, source
    return None, None


def _config_mtp_layer_count(
    cfg: dict[str, Any], jang_cfg: dict[str, Any]
) -> tuple[int | None, str | None, list[str], int | None]:
    issues: list[str] = []
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    runtime = jang_cfg.get("runtime") if isinstance(jang_cfg.get("runtime"), dict) else {}
    mtp = jang_cfg.get("mtp") if isinstance(jang_cfg.get("mtp"), dict) else {}

    candidates = [
        ("config.num_nextn_predict_layers", cfg.get("num_nextn_predict_layers")),
        ("config.mtp_num_hidden_layers", cfg.get("mtp_num_hidden_layers")),
        (
            "config.text_config.num_nextn_predict_layers",
            text_cfg.get("num_nextn_predict_layers"),
        ),
        (
            "config.text_config.mtp_num_hidden_layers",
            text_cfg.get("mtp_num_hidden_layers"),
        ),
    ]

    selected_value: int | None = None
    selected_source: str | None = None
    for source, raw in candidates:
        value, invalid = _coerce_non_negative_int(raw)
        if invalid:
            issues.append(f"{source} is invalid; expected a non-negative integer")
            continue
        if value is not None:
            selected_value = value
            selected_source = source
            if value > 0:
                break

    jang_value = None
    for source, raw in (
        ("jang_config.runtime.mtp_layers", runtime.get("mtp_layers")),
        ("jang_config.mtp.num_layers", mtp.get("num_layers")),
    ):
        value, invalid = _coerce_non_negative_int(raw)
        if invalid:
            issues.append(f"{source} is invalid; expected a non-negative integer")
            continue
        if value is not None:
            jang_value = value
            break

    if selected_value in (None, 0) and jang_value and jang_value > 0:
        selected_value = jang_value
        selected_source = "jang_config.runtime.mtp_layers"
    if (
        selected_value is not None
        and selected_value > 0
        and jang_value is not None
        and jang_value > 0
        and selected_value != jang_value
    ):
        issues.append(
            f"{selected_source}={selected_value} but jang_config reports "
            f"{jang_value} MTP layer(s)"
        )

    return selected_value, selected_source, issues, jang_value


def _index_mtp_keys(bundle_path: str | Path | None) -> tuple[list[str], str | None]:
    weight_keys, _source, error = _bundle_weight_keys(bundle_path)
    if error:
        return [], error
    return _mtp_keys_from_weight_keys(weight_keys), None


def _mtp_keys_from_weight_keys(weight_keys: list[str]) -> list[str]:
    return [
        str(key)
        for key in weight_keys
        if re.search(r"(^|\.)mtp(\.|$)", str(key))
    ]


_MTP_LAYER_PATTERNS = (
    re.compile(r"^mtp\.(\d+)(?:\.|$)"),
    re.compile(r"^mtp\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)mtp\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)mtp_layers\.(\d+)(?:\.|$)"),
)


def _mtp_layer_count_from_keys(keys: list[str]) -> int | None:
    indexes: set[int] = set()
    for key in keys:
        for pattern in _MTP_LAYER_PATTERNS:
            match = pattern.search(key)
            if match:
                indexes.add(int(match.group(1)))
                break
    if not indexes:
        return None
    return max(indexes) + 1


def _vision_weight_keys(bundle_path: str | Path | None) -> tuple[list[str], str | None]:
    weight_keys, _source, error = _bundle_weight_keys(bundle_path)
    if error:
        return [], error
    return _vision_keys_from_weight_keys(weight_keys), None


def _vision_keys_from_weight_keys(weight_keys: list[str]) -> list[str]:
    vision_re = re.compile(
        r"(^|\.)(vision_tower|vision_model|visual|patch_embed|"
        r"multi_modal_projector|mm_projector|image_newline)(\.|$)"
    )
    return [str(key) for key in weight_keys if vision_re.search(str(key))]


def bundle_index_mtp_layer_count(bundle_path: str | Path | None) -> int | None:
    """Return highest indexed MTP layer + 1 from supported MTP key layouts."""
    keys, _error = _index_mtp_keys(bundle_path)
    if not keys:
        return None
    return _mtp_layer_count_from_keys(keys)


def _runtime_enabled_by_env() -> bool:
    return os.environ.get("VMLINUX_NATIVE_MTP", "1") not in _DISABLE_ENV_VALUES


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "") in _ENABLE_ENV_VALUES for name in names)


def _env_disabled(*names: str) -> bool:
    return any(os.environ.get(name, "") in _DISABLE_ENV_VALUES for name in names)


def _runtime_validation_block_reason(jang_cfg: dict[str, Any]) -> str | None:
    """Block native MTP for artifact profiles that failed live validation.

    The JANG_2K profile still needs the native-MTP VL loader route for image
    input, but the 2026-05-17 six-variant packaged gate showed native MTP was
    slower and less coherent for this artifact profile. Keep this as a runtime
    acceleration block, not an artifact metadata failure.
    """
    quantization = (
        jang_cfg.get("quantization")
        if isinstance(jang_cfg.get("quantization"), dict)
        else {}
    )
    profile = str(
        quantization.get("profile")
        or jang_cfg.get("profile")
        or ""
    ).strip().upper()
    if profile != "JANG_2K":
        return None
    if _env_enabled("VMLINUX_NATIVE_MTP_ALLOW_JANG2K", "VMLX_NATIVE_MTP_ALLOW_JANG2K"):
        return None
    return (
        "JANG_2K native MTP acceleration is blocked by validation policy: "
        "the 2026-05-17 packaged six-variant gate showed low MTP acceptance, "
        "MTP speed regression, and coherence failures. Set "
        "VMLINUX_NATIVE_MTP_ALLOW_JANG2K=1 to force the experimental path."
    )


def native_mtp_effective_depth(
    model_path: str | Path | None = None,
) -> tuple[int, str]:
    """Resolve native-MTP runtime draft depth.

    Qwen3.6 ships one trained MTP head; depth here means recursive runtime
    drafting through that head, clamped to the verifier implementation's D3
    support.

    Validated model-local tuning files are honored before the generic D3
    fallback. That lets measured bundles such as 27B MXFP4 use their proven
    D2 policy while preserving D3 for artifacts without a sidecar.
    """
    env_name = None
    raw = None
    for candidate in ("VMLINUX_NATIVE_MTP_DEPTH", "VMLX_NATIVE_MTP_DEPTH"):
        if candidate in os.environ:
            env_name = candidate
            raw = os.environ.get(candidate)
            break
    if raw is None:
        raw = "3"
        source = "default"
    else:
        source = env_name or "env"
    try:
        depth = int(raw)
    except (TypeError, ValueError):
        depth = 3
        source = f"invalid:{source}"
    if raw is not None and source != "default":
        return max(1, min(3, depth)), source

    if not _env_disabled("VMLINUX_NATIVE_MTP_USE_TUNING", "VMLX_NATIVE_MTP_USE_TUNING"):
        tuned_path = model_path or _ACTIVE_NATIVE_MTP_MODEL_PATH
        tuned_depth, tuned_source = _model_tuning_depth(tuned_path)
        if tuned_depth is not None and tuned_source is not None:
            return tuned_depth, tuned_source
    return max(1, min(3, depth)), source


def inspect_native_mtp_bundle(bundle_path: str | Path | None) -> dict[str, Any]:
    cfg = _read_json(bundle_path, "config.json")
    jang_cfg = _read_json(bundle_path, "jang_config.json")
    family = _bundle_family(cfg, jang_cfg)
    config_layers, layer_source, issues, jang_layers = _config_mtp_layer_count(
        cfg, jang_cfg
    )

    drop_mtp_raw = jang_cfg.get("drop_mtp")
    drop_mtp_invalid_type = (
        drop_mtp_raw is not None and not isinstance(drop_mtp_raw, bool)
    )
    drop_mtp = drop_mtp_raw if isinstance(drop_mtp_raw, bool) else None
    if drop_mtp_invalid_type:
        issues.append(
            "jang_config.drop_mtp must be a boolean; got "
            f"{type(drop_mtp_raw).__name__}: {drop_mtp_raw!r}"
        )

    mtp_sidecar = jang_cfg.get("mtp") if isinstance(jang_cfg.get("mtp"), dict) else {}
    if mtp_sidecar.get("enabled") is False or mtp_sidecar.get("kept") is False:
        drop_mtp = True

    weight_keys, _weight_key_source, weight_key_error = _bundle_weight_keys(bundle_path)
    if weight_key_error:
        issues.append(weight_key_error)
    mtp_keys = _mtp_keys_from_weight_keys(weight_keys) if not weight_key_error else []
    has_mtp_tensors = bool(mtp_keys)
    indexed_layer_count = _mtp_layer_count_from_keys(mtp_keys)

    if (
        config_layers is not None
        and config_layers > 0
        and drop_mtp is not True
        and not has_mtp_tensors
    ):
        issues.append(
            "config expects MTP next-token prediction layers, but the bundle "
            "index has no mtp.* tensors"
        )
    if config_layers in (None, 0) and drop_mtp is not True and has_mtp_tensors:
        issues.append("bundle indexes mtp.* tensors but config disables MTP runtime")
    if drop_mtp is True and config_layers not in (None, 0):
        issues.append(
            "jang_config.drop_mtp=true but config declares "
            f"{config_layers} MTP layer(s)"
        )
    if drop_mtp is True and has_mtp_tensors:
        issues.append("jang_config.drop_mtp=true but bundle still indexes mtp.* tensors")
    if (
        config_layers is not None
        and config_layers > 0
        and indexed_layer_count is not None
        and indexed_layer_count != config_layers
        and drop_mtp is not True
    ):
        issues.append(
            f"{layer_source or 'config MTP layer count'}={config_layers} but "
            f"bundle index has {indexed_layer_count} distinct MTP layer(s)"
        )

    artifact_available = bool(
        config_layers
        and config_layers > 0
        and drop_mtp is not True
        and has_mtp_tensors
        and not issues
    )
    runtime_supported = bool(
        artifact_available and _normalize_family(family) in _RUNTIME_SUPPORTED_FAMILIES
    )
    runtime_env_enabled = _runtime_enabled_by_env()
    runtime_validation_block_reason = (
        _runtime_validation_block_reason(jang_cfg) if runtime_supported else None
    )
    runtime_available = bool(
        runtime_supported
        and runtime_env_enabled
        and runtime_validation_block_reason is None
    )
    effective_depth, effective_depth_source = native_mtp_effective_depth(bundle_path)

    has_vision_config = bool(cfg.get("vision_config")) or bool(
        isinstance(cfg.get("text_config"), dict)
        and (cfg.get("text_config") or {}).get("vision_config")
    )
    vision_keys = _vision_keys_from_weight_keys(weight_keys) if not weight_key_error else []
    has_vision_weights = bool(vision_keys)
    has_vision = bool(has_vision_config and has_vision_weights)
    capabilities = jang_cfg.get("capabilities") if isinstance(jang_cfg.get("capabilities"), dict) else {}
    cache_type = capabilities.get("cache_type") if isinstance(capabilities, dict) else None

    vl_runtime_available = bool(runtime_available and has_vision and runtime_supported)
    runtime_scope = (
        "text+vl"
        if runtime_supported and has_vision
        else "text"
        if runtime_supported
        else None
    )

    if issues:
        status = "metadata_inconsistent"
        runtime_reason = "metadata_inconsistent"
    elif drop_mtp is True:
        status = "dropped"
        runtime_reason = "jang_config.drop_mtp=true"
    elif runtime_available:
        status = "native_runtime_ready"
        runtime_reason = (
            "native MTP runtime will be enabled for supported text and VL "
            "Qwen3.5/3.6 sessions"
            if vl_runtime_available
            else "native MTP runtime will be enabled for supported text "
            "BatchGenerator sessions"
        )
    elif artifact_available and runtime_supported and not runtime_env_enabled:
        status = "runtime_disabled"
        runtime_reason = "VMLINUX_NATIVE_MTP disables native MTP runtime"
    elif artifact_available and runtime_supported and runtime_validation_block_reason:
        status = "runtime_validation_blocked"
        runtime_reason = runtime_validation_block_reason
    elif artifact_available:
        status = "weights_present_runtime_unwired"
        runtime_reason = (
            f"MTP metadata is present for family '{family or 'unknown'}', but "
            "this family is not currently on the JangMTP support map / native "
            "MTP runtime map yet"
        )
    elif config_layers:
        status = "configured_without_runtime"
        runtime_reason = "config requests MTP but runtime requirements are incomplete"
    else:
        status = "not_configured"
        runtime_reason = "config does not request MTP"

    return {
        "config_num_nextn_predict_layers": config_layers,
        "config_mtp_layer_source": layer_source,
        "jang_mtp_layers": jang_layers,
        "jang_drop_mtp": drop_mtp,
        "index_has_mtp_tensors": has_mtp_tensors,
        "index_mtp_layer_count": indexed_layer_count,
        "mtp_tensor_count": len(mtp_keys),
        "artifact_available": artifact_available,
        "family": family,
        "has_vision_config": has_vision_config,
        "has_vision_weights": has_vision_weights,
        "vision_tensor_count": len(vision_keys),
        "cache_type": cache_type,
        "runtime_supported": runtime_supported,
        "runtime_available": runtime_available,
        "runtime_active": False,
        "runtime_validation_blocked": bool(runtime_validation_block_reason),
        "effective_depth": effective_depth if runtime_available else None,
        "effective_depth_source": effective_depth_source if runtime_available else None,
        "runtime_scope": runtime_scope,
        "vl_runtime_available": vl_runtime_available,
        "runtime_reason": runtime_reason,
        "status": status,
        "issues": issues,
    }


def _apply_mlx_lm_mtp_patch() -> bool:
    from .patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch
    from .patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch

    lm_ok = apply_mlx_lm_mtp_patch()
    vl_ok = apply_mlx_vlm_mtp_patch()
    return bool(lm_ok and vl_ok)


def _set_mtp_active(active: bool) -> None:
    from .patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(active)


def deactivate_native_mtp() -> None:
    global _ACTIVE_NATIVE_MTP_MODEL_PATH
    _ACTIVE_NATIVE_MTP_MODEL_PATH = None
    try:
        _set_mtp_active(False)
    except Exception:
        pass


def maybe_apply_native_mtp(
    model_path: str | Path,
    *,
    allow_runtime: bool = True,
    reason: str | None = None,
) -> dict[str, Any]:
    """Apply sanitize/runtime patches before loading a native-MTP bundle."""
    global _ACTIVE_NATIVE_MTP_MODEL_PATH
    status = inspect_native_mtp_bundle(model_path)
    should_patch_for_sanitize = bool(
        status["artifact_available"] and status["runtime_supported"]
    )
    runtime_active = bool(
        should_patch_for_sanitize and status["runtime_available"] and allow_runtime
    )

    if should_patch_for_sanitize:
        if _apply_mlx_lm_mtp_patch():
            _set_mtp_active(runtime_active)
            _ACTIVE_NATIVE_MTP_MODEL_PATH = Path(model_path) if runtime_active else None
            status["runtime_active"] = runtime_active
            if runtime_active:
                logger.info(
                    "Native MTP runtime active for %s "
                    "(family=%s layers=%s tensors=%s cache=%s depth=%s/%s)",
                    model_path,
                    status.get("family"),
                    status.get("config_num_nextn_predict_layers"),
                    status.get("mtp_tensor_count"),
                    status.get("cache_type"),
                    status.get("effective_depth"),
                    status.get("effective_depth_source"),
                )
            else:
                status["runtime_available"] = False
                status["runtime_active"] = False
                if status.get("status") == "native_runtime_ready":
                    status["status"] = "runtime_disabled"
                    status["runtime_reason"] = (
                        reason or "native MTP patch applied for sanitize only"
                    )
                elif reason:
                    status["runtime_reason"] = reason
                logger.info(
                    "Native MTP patch applied sanitize-only for %s (%s)",
                    model_path,
                    status["runtime_reason"],
                )
        else:
            _ACTIVE_NATIVE_MTP_MODEL_PATH = None
            try:
                _set_mtp_active(False)
            except Exception:
                pass
            status["runtime_available"] = False
            status["runtime_active"] = False
            status["status"] = "runtime_patch_failed"
            status["runtime_reason"] = "native MTP patch failed to apply"
    else:
        deactivate_native_mtp()
    return status


def model_has_native_mtp_runtime(model: Any) -> bool:
    """True when a loaded model instance has an attached native MTP head."""
    seen: set[int] = set()

    def _walk(obj: Any, depth: int = 0) -> bool:
        if obj is None or depth > 6:
            return False
        ident = id(obj)
        if ident in seen:
            return False
        seen.add(ident)
        has_forward = callable(getattr(obj, "mtp_forward", None))
        has_cache_builder = callable(getattr(obj, "make_mtp_cache", None))
        has_head = getattr(obj, "mtp", None) is not None
        if has_head and has_forward and has_cache_builder:
            return True
        for attr in ("_model", "model", "language_model"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and _walk(child, depth + 1):
                return True
        return False

    return _walk(model)

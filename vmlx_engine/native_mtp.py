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
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Native-MTP depth ceiling. The product default is 3 and does NOT change: this
# is a MEASUREMENT lever so a depth sweep can probe D4/D5 on bundles+lanes that
# can verify them (VLM lane with partial rollback), never a shipped raise.
# Correctness gates still apply per lane — the text lane hard-caps hybrids at
# D1 and qwen4_exp sharded-GDN rollback is not implemented, so raising this
# does not by itself make those lanes run deeper.
_NATIVE_MTP_DEPTH_HARD_CEILING = 8
_NATIVE_MTP_DEFAULT_MAX_DEPTH = 3


def native_mtp_max_depth() -> int:
    """Resolve the configured native-MTP depth ceiling (default 3)."""
    raw = os.environ.get(
        "VMLINUX_NATIVE_MTP_MAX_DEPTH",
        os.environ.get("VMLX_NATIVE_MTP_MAX_DEPTH", ""),
    ).strip()
    if not raw:
        return _NATIVE_MTP_DEFAULT_MAX_DEPTH
    try:
        value = int(raw)
    except ValueError:
        return _NATIVE_MTP_DEFAULT_MAX_DEPTH
    return max(1, min(_NATIVE_MTP_DEPTH_HARD_CEILING, value))


_DISABLE_ENV_VALUES = {"0", "false", "FALSE", "no", "NO", "off", "OFF"}

_FAMILY_ALIAS = {
    "qwen3_5_text": "qwen3_5",
    "qwen3_6": "qwen3_5",
    "qwen3_6_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",
    "qwen4_exp_text": "qwen4_exp",
    # config.model_type for the ERNIE-4.5 MoE checkpoints; registry family is ernie4_5
    "ernie4_5_moe": "ernie4_5",
}

# Keep this narrow. The copied mlx-lm patch currently wires Qwen3.5/3.6 MTP
# through GenerationBatch. DeepSeek-V4 has model-side code copied for later
# work, but vMLX's DSV4 runtime uses a custom generator, so do not advertise
# it as native-MTP active yet.
# hy_v3 (Tencent Hy3): model-side hooks live in jang_tools.hy3.model
# (Hy3MTPLayer + mtp_forward/make_mtp_cache); patches/mlx_lm_mtp/hy_v3_model.py
# gates head attachment. Bundle ships the DSV3-style head as mtp.0.* tensors.
_RUNTIME_SUPPORTED_FAMILIES = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen4_exp",
    "hy_v3",
    # ernie4_5: vMLX-owned models/ernie4_5 runtime exposes the full contract
    # (model.mtp module, mtp_forward taking the backbone's pre-norm hidden and
    # applying model.norm itself, make_mtp_cache = one KVCache). Head parity
    # gated against a plain-torch reference (see the introducing PR).
    "ernie4_5",
    # glm5_next: the vendored model owns the layer-45 draft head, pre-output
    # hidden handoff, KDA accepted-prefix snapshots, and trimmable MLA cache.
    # It runs through the same GenerationBatch verifier as Qwen.
    "glm5_next",
    # dots3_note: the vendored runtime exposes the full contract (non-null
    # mtp module = SWA-geometry layer 46, mtp_forward with recursive
    # drafting through the DEDICATED model.mtp.embed_tokens table,
    # make_mtp_cache). MLLM lane. Stamp recommends 1 draft; unmeasured on
    # this artifact until a live depth sweep writes the tuning sidecar.
    "dots3_note",
}
_EAGLE3_NATIVE_MTP_FAMILIES = {
    "minimax_m3",
    "minimax_m3_vl",
}

_ENABLE_ENV_VALUES = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_ACTIVE_NATIVE_MTP_MODEL_PATH: Path | None = None


def _read_json(bundle_path: str | Path | None, name: str) -> dict[str, Any]:
    if not bundle_path:
        return {}
    try:
        path = Path(bundle_path) / name
        if not path.is_file():
            if name != "jang_config.json":
                return {}
            config_path = Path(bundle_path) / "config.json"
            if not config_path.is_file():
                return {}
            config = json.loads(config_path.read_text())
            if not isinstance(config, dict):
                return {}
            embedded = config.get("jang_config")
            if embedded is None:
                embedded = config.get("jang")
            return embedded if isinstance(embedded, dict) else {}
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
        keys = [str(key) for key in weight_map]
        indexed_files = {
            Path(str(filename)).name
            for filename in weight_map.values()
            if isinstance(filename, str)
        }
        try:
            from safetensors import safe_open
        except Exception as exc:
            return keys, "index", f"safetensors header reader unavailable: {exc}"

        supplemental_count = 0
        errors: list[str] = []
        try:
            paths = sorted(
                p
                for p in Path(bundle_path).glob("*.safetensors")
                if not p.name.startswith("._")
            )
        except Exception as exc:
            return keys, "index", f"safetensors files could not be listed: {exc}"
        for shard in paths:
            if shard.name in indexed_files:
                continue
            try:
                with safe_open(str(shard), framework="numpy") as handle:
                    supplemental_keys = [str(key) for key in handle.keys()]
                if supplemental_keys:
                    keys.extend(supplemental_keys)
                    supplemental_count += 1
            except Exception as exc:
                # An unreadable file advertised as MTP must fail closed.
                # Unrelated custom sidecars do not invalidate a sound index.
                if "mtp" in shard.name.lower():
                    errors.append(f"{shard.name}: {exc}")
        if errors:
            return keys, "index+supplemental_safetensors", (
                "supplemental MTP safetensors header read failed: " + "; ".join(errors)
            )
        return list(dict.fromkeys(keys)), (
            "index+supplemental_safetensors" if supplemental_count else "index"
        ), None
    if index is not None and not isinstance(weight_map, dict):
        return [], "index", "model.safetensors.index.json has no weight_map object"

    try:
        from safetensors import safe_open
    except Exception as exc:
        return [], "safetensors", f"safetensors header reader unavailable: {exc}"

    keys: list[str] = []
    errors: list[str] = []
    try:
        paths = sorted(
            p
            for p in Path(bundle_path).glob("*.safetensors")
            if not p.name.startswith("._")
        )
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


def _bundle_name_declares_mtp(bundle_path: str | Path | None) -> bool:
    """Return whether the user-facing bundle name explicitly opts into MTP.

    Some upstream architecture configs retain ``mtp_num_hidden_layers`` even
    when a converted runtime bundle intentionally does not ship an MTP head.
    Treat that nested field as an architecture hint unless the bundle name,
    JANG runtime sidecar, or tensor index independently declares MTP.
    """
    if not bundle_path:
        return False
    name = Path(bundle_path).name
    return bool(re.search(r"(?:^|[-_.])mtp(?:$|[-_.])", name, re.IGNORECASE))


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
    return max(1, min(native_mtp_max_depth(), value))


def _positive_finite_number(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) and value > 0 else None


def _validated_flat_tuning_depth(tuning: dict[str, Any]) -> int | None:
    """Trust flat producer tuning only when its wall-speed evidence agrees."""
    if tuning.get("blocked") is True or tuning.get("output_equivalent") is False:
        return None
    depth = _coerce_native_mtp_depth(tuning.get("best_depth"))
    if depth is None:
        return None
    # D1 is the conservative legacy seed used by unmeasured bundles. A deeper
    # producer recommendation changes runtime cost and therefore needs real
    # matched-output wall-speed evidence, not tokens-per-cycle acceptance.
    if depth == 1:
        return 1
    if tuning.get("validated") is not True:
        return None
    if any(
        _positive_finite_number(tuning.get(key)) is None
        for key in ("baseline_tok_s", "best_tok_s", "speedup_vs_baseline")
    ):
        return None
    if float(tuning["speedup_vs_baseline"]) <= 1.0:
        return None

    raw_speeds = tuning.get("measured_tok_s_by_depth")
    if not isinstance(raw_speeds, dict):
        return None
    speeds: dict[int, float] = {}
    for raw_key, raw_speed in raw_speeds.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            continue
        speed = _positive_finite_number(raw_speed)
        if 1 <= key <= native_mtp_max_depth() and speed is not None:
            speeds[key] = speed

    # "best" is meaningful only across the producer's complete supported
    # depth surface.  A 4M Flash-Next sidecar stamped D2 after measuring only
    # D1/D2 and explicitly saying "D3 NOT measured" remained authoritative
    # after newer kernels made D3 the matched wall winner.  Require D1..D3 by
    # default; a genuinely narrower runtime may declare the ceiling it
    # actually supports and measured.
    raw_ceiling = tuning.get("measured_depth_ceiling", tuning.get("depth_ceiling", 3))
    try:
        measured_ceiling = max(1, min(native_mtp_max_depth(), int(raw_ceiling)))
    except (TypeError, ValueError):
        return None
    required_depths = set(range(1, measured_ceiling + 1))
    if depth > measured_ceiling or not required_depths.issubset(speeds):
        return None
    fastest_speed = max(speeds[key] for key in required_depths)
    fastest_depth = min(
        key for key in required_depths if speeds[key] == fastest_speed
    )
    if fastest_depth != depth:
        return None
    return depth


def _attested_block_depth(block: dict[str, Any]) -> int | None:
    """Depth from a nested ``native_mtp`` / ``best_native_mtp_depth`` block.

    Those blocks carry an attestation, not a speed table, so they are held to
    the same rule as the flat format's evidence gate: depth 1 is the
    conservative seed and is accepted as long as the block is not blocked or
    invalidated; any deeper recommendation needs ``validated`` to be exactly
    ``True``.  A block that merely omits ``validated`` used to pass (the
    check was ``is not False``), which let an unvalidated D2/D3 through here
    while the flat format demanded matched wall-speed evidence.
    """
    if not isinstance(block, dict):
        return None
    if block.get("blocked") is True or block.get("output_equivalent") is False:
        return None
    depth = _coerce_native_mtp_depth(block.get("best_depth"))
    if depth is None:
        return None
    if depth == 1:
        return 1
    if block.get("validated") is not True:
        return None
    return depth


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
        if (
            native_mtp.get("blocked") is True
            or native_mtp.get("validated") is False
            or native_mtp.get("output_equivalent") is False
        ):
            # A tuning file that declares its native_mtp measurement
            # blocked/invalid must not leak a depth through the legacy
            # top-level fallback keys either (2026-07-10 audit High-4).
            return None, None
        candidates.append(
            (
                "vmlx_mtp_tuning.json:native_mtp.best_depth",
                _attested_block_depth(native_mtp),
            )
        )

    sweep_result = tuning.get("best_native_mtp_depth")
    if isinstance(sweep_result, dict):
        candidates.append(
            (
                "vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth",
                _attested_block_depth(sweep_result),
            )
        )

    flat_depth = _validated_flat_tuning_depth(tuning)
    if flat_depth is not None:
        candidates.append(("vmlx_mtp_tuning.json:best_depth", flat_depth))

    for source, raw_depth in candidates:
        depth = _coerce_native_mtp_depth(raw_depth)
        if depth is not None:
            return depth, source
    return None, None


def _read_eagle3_sidecar(
    bundle_path: str | Path | None,
) -> tuple[dict[str, Any], list[str]]:
    cfg = _read_json(bundle_path, "eagle3_config.json")
    if not cfg:
        return {}, []

    issues: list[str] = []
    if str(cfg.get("method", "")).lower() != "eagle3":
        issues.append("eagle3_config.json method must be 'eagle3'")

    weights_file = cfg.get("weights_file")
    if not isinstance(weights_file, str) or not weights_file:
        issues.append("eagle3_config.json weights_file must be a non-empty string")
    elif bundle_path and not (Path(bundle_path) / weights_file).is_file():
        issues.append(f"eagle3 weights file is missing: {weights_file}")

    aux_layers = cfg.get("aux_hidden_state_layers")
    if (
        not isinstance(aux_layers, list)
        or len(aux_layers) != 3
        or not all(isinstance(layer, int) for layer in aux_layers)
    ):
        issues.append(
            "eagle3_config.json aux_hidden_state_layers must be a list of 3 integers"
        )

    if cfg.get("draft_to_target_id_map") not in (None, "identity"):
        issues.append("MiniMax-M3 EAGLE3 runtime currently requires identity token map")

    return cfg, issues


def _eagle3_native_mtp_status(
    bundle_path: str | Path | None,
    cfg: dict[str, Any],
    jang_cfg: dict[str, Any],
    family: str | None,
) -> dict[str, Any] | None:
    eagle_cfg, issues = _read_eagle3_sidecar(bundle_path)
    if not eagle_cfg:
        return None

    normalized_family = _normalize_family(family)
    family_supported = normalized_family in _EAGLE3_NATIVE_MTP_FAMILIES
    if not family_supported:
        issues.append(
            "eagle3_config.json is present but the bundle family is not MiniMax-M3"
        )

    runtime_env_enabled = _runtime_enabled_by_env()
    artifact_available = bool(not issues)
    measured_accept = eagle_cfg.get("measured_accept")
    if not isinstance(measured_accept, dict):
        measured_accept = {}

    has_vision_config = bool(cfg.get("vision_config")) or bool(
        isinstance(cfg.get("text_config"), dict)
        and (cfg.get("text_config") or {}).get("vision_config")
    )
    capabilities = (
        jang_cfg.get("capabilities")
        if isinstance(jang_cfg.get("capabilities"), dict)
        else {}
    )
    cache_type = capabilities.get("cache_type") if isinstance(capabilities, dict) else None

    if issues:
        status = "metadata_inconsistent"
        runtime_reason = "metadata_inconsistent"
    elif not runtime_env_enabled:
        status = "runtime_disabled"
        runtime_reason = (
            "VMLINUX_NATIVE_MTP/VMLX_NATIVE_MTP disables native MTP runtime"
        )
    else:
        status = "weights_present_runtime_unwired"
        runtime_reason = (
            "MiniMax-M3 EAGLE3 native-MTP sidecar is present; draft loader "
            "scaffold is available, but the target verify loop and MSA cache "
            "rollback are not wired into live decode yet"
        )

    return {
        "config_num_nextn_predict_layers": None,
        "config_mtp_layer_source": "eagle3_config.json",
        "jang_mtp_layers": None,
        "jang_drop_mtp": None,
        "index_has_mtp_tensors": False,
        "index_mtp_layer_count": None,
        "mtp_tensor_count": 0,
        "artifact_available": artifact_available,
        "family": family,
        "has_vision_config": has_vision_config,
        "has_vision_weights": False,
        "vision_tensor_count": 0,
        "cache_type": cache_type,
        "runtime_supported": bool(family_supported and artifact_available),
        "runtime_available": False,
        "runtime_active": False,
        "runtime_validation_blocked": False,
        "effective_depth": None,
        "effective_depth_source": None,
        "runtime_scope": "text" if family_supported else None,
        "vl_runtime_available": False,
        "runtime_bundle_has_mtp": True if artifact_available else None,
        "runtime_mtp_mode": "eagle3_sidecar",
        "runtime_adapter": "minimax_m3_eagle3",
        "native_mtp_method": "eagle3",
        "eagle3_weights_file": eagle_cfg.get("weights_file"),
        "eagle3_aux_hidden_state_layers": eagle_cfg.get("aux_hidden_state_layers"),
        "eagle3_draft_to_target_id_map": eagle_cfg.get("draft_to_target_id_map"),
        "eagle3_tokens_per_target_forward": measured_accept.get(
            "tokens_per_target_forward"
        ),
        "runtime_reason": runtime_reason,
        "status": status,
        "issues": issues,
    }


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
        # v3 capability-schema stamps (Qwen3.6-27B D-series onward) spell it
        # `num_hidden_layers` inside the mtp block. Without this entry the
        # inspection reported the artifact READY while the layer count stayed
        # None, and model construction then built no head at all.
        ("jang_config.mtp.num_hidden_layers", mtp.get("num_hidden_layers")),
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
    # `mtp.` covers Qwen/DeepSeek-style heads. ERNIE-4.5 ships its head as
    # `model.mtp_block.N.*`, `model.mtp_emb_norm.N`, `model.mtp_hidden_norm.N`,
    # `model.mtp_linear_proj.N` (underscore, no `mtp.` segment), so without the
    # second alternative a preserved ERNIE head counts as 0 tensors and the
    # status reads "not declared" although config.num_nextn_predict_layers=1.
    return [
        str(key)
        for key in weight_keys
        if re.search(r"(^|\.)mtp(\.|$)", str(key))
        or re.search(r"(^|\.)mtp_(block|emb_norm|hidden_norm|linear_proj)(\.|$)", str(key))
    ]


_MTP_LAYER_PATTERNS = (
    re.compile(r"^mtp\.(\d+)(?:\.|$)"),
    re.compile(r"^mtp\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)mtp\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)mtp_layers\.(\d+)(?:\.|$)"),
    # ERNIE-4.5: one decoder block per draft step under model.mtp_block.<n>
    re.compile(r"(?:^|\.)mtp_block\.(\d+)(?:\.|$)"),
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
    # The kill switch must read BOTH prefixes like every other knob in this
    # file (FORCE/DEPTH/USE_TUNING all go through _env_enabled/_env_disabled
    # pairs, and mtp_runtime_common.py documents the pair). Reading only
    # VMLINUX_NATIVE_MTP made VMLX_NATIVE_MTP=0 a silent no-op.
    return not _env_disabled("VMLINUX_NATIVE_MTP", "VMLX_NATIVE_MTP")


def native_mtp_disabled_by_env() -> bool:
    """Whether the environment turns native MTP off, for every reader.

    The per-request MLLM gate and the /health status bit each grew their own
    copy of this check that only knew the legacy VMLINUX_ spelling, so
    ``VMLX_NATIVE_MTP=0`` switched the runtime off here while those two carried
    on as if it were on — an A/B run that way compares MTP against itself and
    /health reports the wrong state. One function so a new spelling can only
    ever be added in one place.
    """
    return not _runtime_enabled_by_env()


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "") in _ENABLE_ENV_VALUES for name in names)


def _env_disabled(*names: str) -> bool:
    return any(os.environ.get(name, "") in _DISABLE_ENV_VALUES for name in names)


def _native_mtp_explicitly_requested() -> bool:
    """Return whether the operator explicitly opted into native MTP.

    Most supported families retain the historical auto-on runtime policy. A
    GLM-5.3 bundle is the measured exception: its verifier is currently slower
    than AR on first-turn and sustained decode, so an unset environment must
    not silently activate it. An explicit enable/force flag or a valid fixed
    D1-D3 selection remains an opt-in for diagnostics and future tuning.
    """
    if _env_enabled(
        "VMLINUX_NATIVE_MTP",
        "VMLX_NATIVE_MTP",
        "VMLINUX_NATIVE_MTP_FORCE",
        "VMLX_NATIVE_MTP_FORCE",
    ):
        return True
    for name in ("VMLINUX_NATIVE_MTP_DEPTH", "VMLX_NATIVE_MTP_DEPTH"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            if 1 <= int(raw) <= native_mtp_max_depth():
                return True
        except (TypeError, ValueError):
            continue
    return False


def _runtime_default_enabled_for_family(family: str | None) -> bool:
    """Resolve the family default without changing explicit MTP controls."""
    if _normalize_family(family) != "glm5_next":
        return True
    return _native_mtp_explicitly_requested()


def _runtime_validation_block_reason(
    bundle_path: str | Path | None,
    jang_cfg: dict[str, Any],
    family: str | None,
) -> str | None:
    """Block native MTP for artifacts that failed live validation.

    This is a runtime *acceleration* block, never an artifact metadata failure
    (the MTP weights stay valid and present). It can come from either:

    ``jang_config["runtime"]["native_mtp_blocked"] = "<measured reason>"``.

    or, for Hy3's affine multi-row verifier, a missing/failed exact-output
    attestation in ``vmlx_mtp_tuning.json``. Hy3's standard greedy decode runs
    one token at a time, while the native-MTP verifier runs two tokens through
    the affine backbone. MLX affine matmul is not bit-identical across those
    shapes, so a speed/quality-only tuning pass cannot establish speculative
    decode correctness. The tuning sidecar must explicitly stamp
    ``output_equivalent: true`` after comparing real-artifact token IDs.

    A bundle that measured MTP as a net slowdown declares it here, rather
    than the engine hardcoding profile names. Override with
    ``VMLX_NATIVE_MTP_FORCE=1`` to re-run the experiment.

    A profile name alone is not a block. The gate is the model-local measured
    attestation, so a future Hy3 artifact can re-enable the fast verifier after
    it proves exact greedy identity on its real quantized weights.
    """
    forced = _env_enabled("VMLINUX_NATIVE_MTP_FORCE", "VMLX_NATIVE_MTP_FORCE")
    runtime = (
        jang_cfg.get("runtime") if isinstance(jang_cfg.get("runtime"), dict) else {}
    )
    declared = runtime.get("native_mtp_blocked")
    if isinstance(declared, str) and declared.strip():
        if forced:
            return None
        return (
            f"bundle declares native MTP blocked: {declared.strip()} "
            "(set VMLX_NATIVE_MTP_FORCE=1 to force the experimental path)"
        )

    if _normalize_family(family) == "hy_v3":
        tuning = _read_json(bundle_path, "vmlx_mtp_tuning.json")
        native_mtp = (
            tuning.get("native_mtp")
            if isinstance(tuning.get("native_mtp"), dict)
            else {}
        )
        output_equivalent = native_mtp.get("output_equivalent")
        if output_equivalent is not True and not forced:
            detail = (
                "explicitly failed"
                if output_equivalent is False
                else "is missing"
            )
            return (
                "Hy3 affine native MTP is validation-blocked because "
                f"vmlx_mtp_tuning.json native_mtp.output_equivalent {detail}; "
                "the two-token affine verifier must prove token-identical "
                "greedy output against one-token autoregressive decode "
                "(set VMLX_NATIVE_MTP_FORCE=1 only for measurement)"
            )
    return None


def native_mtp_effective_depth(
    model_path: str | Path | None = None,
) -> tuple[int, str]:
    """Resolve native-MTP runtime draft depth.

    Qwen3.6 ships one trained MTP head; depth here means recursive runtime
    drafting through that head, clamped to the verifier implementation's D3
    support.

    Validated model-local tuning files are honored before the family/generic
    fallback. That lets measured bundles such as 27B MXFP4 use their proven
    D2 policy while preserving D3 for Qwen artifacts without a sidecar.

    Family fallback: hy_v3 defaults to depth 1 — the 2026-07-10 sweep on
    Hy3-JANG_2K-MTP measured d1 +10% vs baseline (~full acceptance) but
    d2 -6% (24.5%) and d3 -31% (1.9%); deeper drafts collapse acceptance
    against the hy3 head. Qwen3.6 keeps the D3 default.
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
        # An invalid explicit override must NOT silently become the generic
        # D3 default — that bypasses measured tuning sidecars and family
        # fallbacks (2026-07-10 audit High-4). Ignore it and resolve as if
        # unset.
        if source != "default":
            logger.warning(
                "Ignoring invalid %s=%r; resolving native MTP depth from "
                "tuning/family instead",
                source,
                raw,
            )
        depth = 3
        source = "default"
    if source != "default":
        return max(1, min(native_mtp_max_depth(), depth)), source

    tuned_path = model_path or _ACTIVE_NATIVE_MTP_MODEL_PATH
    if not _env_disabled("VMLINUX_NATIVE_MTP_USE_TUNING", "VMLX_NATIVE_MTP_USE_TUNING"):
        tuned_depth, tuned_source = _model_tuning_depth(tuned_path)
        if tuned_depth is not None and tuned_source is not None:
            return tuned_depth, tuned_source

    # v3 capability-schema stamps carry a bundle recommendation in
    # jang_config.mtp.recommended_num_drafts — DRAFT tokens per cycle, which
    # maps 1:1 onto this engine's depth. It ranks like a tuning sidecar:
    # below explicit env/tuning overrides, above the generic default.
    #
    # ⚠️ NEVER read `upstream_num_speculative_tokens` for this. A first
    # version of this branch did, and it is a terminology collision: vLLM's
    # field counts DRAFTS, the kit's D-notation counts drafts+verified, so
    # upstream's "2" is 2 drafts — the kit's forbidden D3, the exact config
    # the Nemotron sweep measured at 0.48x. The stamp's `depth_notation`
    # field exists precisely because an agent (me) copied the only number
    # present and mapped it to the nearest-looking label. The stamper keeps
    # the upstream value for provenance, flagged do-not-copy.
    try:
        _v3_mtp = _read_json(tuned_path, "jang_config.json").get("mtp")
        if isinstance(_v3_mtp, dict):
            _v3_drafts, _v3_invalid = _coerce_non_negative_int(
                _v3_mtp.get("recommended_num_drafts")
            )
            if not _v3_invalid and _v3_drafts:
                return max(1, min(native_mtp_max_depth(), _v3_drafts)), "bundle:recommended_num_drafts"
    except Exception:
        pass

    try:
        family = _bundle_family(
            _read_json(tuned_path, "config.json"),
            _read_json(tuned_path, "jang_config.json"),
        )
    except Exception:
        family = None
    if family == "hy_v3":
        return 1, "family_default:hy_v3"
    if family == "ernie4_5":
        # D1 is the conservative seed; deeper chaining has prompt-dependent
        # speedups and requires an explicit override or validated tuning.
        return 1, "family_default:ernie4_5"
    return max(1, min(native_mtp_max_depth(), depth)), source


def inspect_native_mtp_bundle(bundle_path: str | Path | None) -> dict[str, Any]:
    cfg = _read_json(bundle_path, "config.json")
    jang_cfg = _read_json(bundle_path, "jang_config.json")
    family = _bundle_family(cfg, jang_cfg)
    eagle3_status = _eagle3_native_mtp_status(bundle_path, cfg, jang_cfg, family)
    if eagle3_status is not None:
        return eagle3_status

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
    stamped_mtp_mode = str(mtp_sidecar.get("mtp_mode") or "").strip().lower()
    stamped_mtp_absent = stamped_mtp_mode in {"none", "absent", "disabled", "off"}
    if mtp_sidecar.get("enabled") is False or mtp_sidecar.get("kept") is False:
        drop_mtp = True
    if stamped_mtp_absent:
        drop_mtp = True
    runtime_sidecar = (
        jang_cfg.get("runtime") if isinstance(jang_cfg.get("runtime"), dict) else {}
    )
    runtime_bundle_has_mtp = runtime_sidecar.get("bundle_has_mtp")
    if runtime_bundle_has_mtp is None and stamped_mtp_absent:
        runtime_bundle_has_mtp = False
    runtime_mtp_mode = runtime_sidecar.get("mtp_mode") or (
        stamped_mtp_mode if stamped_mtp_mode else None
    )
    runtime_declares_dropped_mtp = (
        runtime_bundle_has_mtp is False
        and isinstance(runtime_mtp_mode, str)
        and (
            "drop" in runtime_mtp_mode.lower()
            or runtime_mtp_mode.lower() in {"none", "absent", "disabled", "off"}
        )
    )
    openpangu_included_mtp_runtime_unwired = (
        _normalize_family(family) == "openpangu_v2"
        and runtime_bundle_has_mtp is True
        and isinstance(runtime_mtp_mode, str)
        and runtime_mtp_mode.lower() == "included"
    )
    if runtime_declares_dropped_mtp:
        drop_mtp = True

    weight_keys, _weight_key_source, weight_key_error = _bundle_weight_keys(bundle_path)
    if weight_key_error:
        issues.append(weight_key_error)
    mtp_keys = _mtp_keys_from_weight_keys(weight_keys) if not weight_key_error else []
    glm_appended_layer_mtp = False
    # glm5_next stores its MTP block as model.layers.<num_hidden_layers>.*
    # (the block appended after the base stack), NOT under an `mtp.` prefix.
    # Without this the tensor counter reports 0 on a preserved-MTP bundle and
    # the status falsely reads "metadata_inconsistent" (mode=preserved_enabled
    # vs 0 tensors). Count the layer-N block so health and the native runtime
    # agree on the preserved draft head.
    if family == "glm5_next" and not mtp_keys and not weight_key_error:
        base_layers = (cfg.get("text_config") or cfg).get("num_hidden_layers")
        if isinstance(base_layers, int) and base_layers > 0:
            mtp_prefix = f"model.layers.{base_layers}."
            mtp_keys = [k for k in weight_keys if str(k).startswith(mtp_prefix)]
            glm_appended_layer_mtp = bool(mtp_keys)
    has_mtp_tensors = bool(mtp_keys)
    indexed_layer_count = _mtp_layer_count_from_keys(mtp_keys)
    if glm_appended_layer_mtp:
        indexed_layer_count = 1

    bundle_name_declares_mtp = _bundle_name_declares_mtp(bundle_path)
    sidecar_declares_mtp = bool(
        runtime_bundle_has_mtp is True
        or mtp_sidecar.get("enabled") is True
        or mtp_sidecar.get("kept") is True
        or (jang_layers is not None and jang_layers > 0)
    )
    nested_architecture_hint_only = bool(
        config_layers is not None
        and config_layers > 0
        and layer_source
        in {
            "config.text_config.num_nextn_predict_layers",
            "config.text_config.mtp_num_hidden_layers",
        }
        and not bundle_name_declares_mtp
        and not sidecar_declares_mtp
        and not has_mtp_tensors
    )
    mtp_declared = bool(
        bundle_name_declares_mtp
        or sidecar_declares_mtp
        or has_mtp_tensors
        or (config_layers and not nested_architecture_hint_only)
    )

    if (
        config_layers is not None
        and config_layers > 0
        and drop_mtp is not True
        and not has_mtp_tensors
        and not openpangu_included_mtp_runtime_unwired
        and not nested_architecture_hint_only
    ):
        issues.append(
            "config expects MTP next-token prediction layers, but the bundle "
            "index has no mtp.* tensors"
        )
    if config_layers in (None, 0) and drop_mtp is not True and has_mtp_tensors:
        issues.append("bundle indexes mtp.* tensors but config disables MTP runtime")
    if (
        drop_mtp is True
        and config_layers not in (None, 0)
        and not runtime_declares_dropped_mtp
    ):
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
    runtime_family_default_enabled = _runtime_default_enabled_for_family(family)
    runtime_validation_block_reason = (
        _runtime_validation_block_reason(bundle_path, jang_cfg, family)
        if runtime_supported
        else None
    )
    runtime_available = bool(
        runtime_supported
        and runtime_env_enabled
        and runtime_family_default_enabled
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

    normalized_family = _normalize_family(family)
    text_only_runtime = False
    if normalized_family == "glm5_next":
        try:
            from .models.glm5_next.register import glm5_next_vlm_runtime_available

            text_only_runtime = not glm5_next_vlm_runtime_available()
        except Exception:
            text_only_runtime = True
    vl_runtime_available = bool(
        runtime_available
        and has_vision
        and runtime_supported
        and not text_only_runtime
    )
    runtime_scope = (
        "text+vl"
        if runtime_supported and has_vision and not text_only_runtime
        else "text"
        if runtime_supported
        else None
    )

    if issues:
        status = "metadata_inconsistent"
        runtime_reason = "metadata_inconsistent"
    elif drop_mtp is True:
        status = "dropped"
        runtime_reason = (
            "jang_config.mtp.mtp_mode=" + stamped_mtp_mode
            if stamped_mtp_absent
            else "jang_config.runtime.bundle_has_mtp=false"
            if runtime_declares_dropped_mtp
            else "jang_config.drop_mtp=true"
        )
    elif runtime_available:
        status = "native_runtime_ready"
        runtime_reason = (
            "native MTP runtime will be enabled for supported text and VL "
            "sessions"
            if vl_runtime_available
            else "native MTP runtime will be enabled for the GLM text "
            "runtime; the bundled visual tower is not wired"
            if text_only_runtime and has_vision
            else "native MTP runtime will be enabled for supported text "
            "BatchGenerator sessions"
        )
    elif artifact_available and runtime_supported and not runtime_env_enabled:
        status = "runtime_disabled"
        runtime_reason = (
            "VMLINUX_NATIVE_MTP/VMLX_NATIVE_MTP disables native MTP runtime"
        )
    elif artifact_available and runtime_supported and not runtime_family_default_enabled:
        status = "runtime_disabled"
        runtime_reason = (
            "glm5_next defaults to autoregressive decode because its native "
            "MTP verifier has not beaten AR; explicitly select D1-D3 or set "
            "VMLX_NATIVE_MTP=1 to opt into MTP"
        )
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
    elif openpangu_included_mtp_runtime_unwired and config_layers:
        status = "weights_present_runtime_unwired"
        runtime_reason = (
            "openPangu MTP heads are stored as extra model.layers entries and "
            "are intentionally dropped by the current openpangu_v2 runtime"
        )
        runtime_mtp_mode = "included_but_dropped_for_runtime"
    elif nested_architecture_hint_only:
        status = "not_configured"
        runtime_reason = (
            "nested architecture MTP field is inactive because the bundle "
            "name, JANG runtime sidecar, and tensor index do not declare MTP"
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
        "runtime_default_mode": (
            "off" if _normalize_family(family) == "glm5_next" else "auto"
        ),
        "runtime_active": False,
        "runtime_validation_blocked": bool(runtime_validation_block_reason),
        "effective_depth": effective_depth if runtime_available else None,
        "effective_depth_source": effective_depth_source if runtime_available else None,
        "runtime_scope": runtime_scope,
        "vl_runtime_available": vl_runtime_available,
        "runtime_bundle_has_mtp": runtime_bundle_has_mtp,
        "runtime_mtp_mode": runtime_mtp_mode,
        "bundle_name_declares_mtp": bundle_name_declares_mtp,
        "mtp_declared": mtp_declared,
        "architecture_mtp_hint_layers": (
            config_layers if nested_architecture_hint_only else None
        ),
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


def native_mtp_active_model_path() -> Path | None:
    """Bundle path of the currently activated native-MTP runtime, if any.

    Set by maybe_apply_native_mtp before model construction in every lane
    that can draft; None in sanitize-only/doctor/deactivated states — i.e.
    exactly when drafting cannot happen. Callers treat None as "skip any
    bundle-sidecar work" (fail-open).
    """

    return _ACTIVE_NATIVE_MTP_MODEL_PATH


def deactivate_native_mtp() -> None:
    global _ACTIVE_NATIVE_MTP_MODEL_PATH
    _ACTIVE_NATIVE_MTP_MODEL_PATH = None
    try:
        _set_mtp_active(False)
    except Exception:
        pass
    # Clear the published v3 bundle layer count, or the NEXT model loaded in
    # this process would inherit the previous bundle's head declaration.
    try:
        from .patches.mlx_vlm_mtp.qwen35_vl import set_active_bundle_mtp_layers
        set_active_bundle_mtp_layers(0)
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
    runtime_adapter = status.get("runtime_adapter", "mlx_lm_mtp_patch")
    should_patch_for_sanitize = bool(
        status["artifact_available"] and status["runtime_supported"]
        and runtime_adapter == "mlx_lm_mtp_patch"
    )
    runtime_active = bool(
        should_patch_for_sanitize and status["runtime_available"] and allow_runtime
    )

    if should_patch_for_sanitize:
        if _apply_mlx_lm_mtp_patch():
            _set_mtp_active(runtime_active)
            _ACTIVE_NATIVE_MTP_MODEL_PATH = Path(model_path) if runtime_active else None
            # v3 capability-schema bundles declare the head in jang_config's
            # mtp block; publish the bundle-declared count for the qwen35_vl
            # construction patch to consume as a fallback. The publisher does
            # NOT try to decide whether config.json already carries the count —
            # review proved that check was self-defeating (the inspection
            # PROMOTES the jang value into `config_num_nextn_predict_layers`,
            # so "config is silent" was never true and this always published
            # 0; the shipped D-series worked only because their config.json
            # stamps `text_config.mtp_num_hidden_layers` after all). The
            # consumer already prefers config.json and touches the published
            # value ONLY when the config is genuinely silent, which is the one
            # place that judgement can be made correctly.
            try:
                from .patches.mlx_vlm_mtp.qwen35_vl import set_active_bundle_mtp_layers
                set_active_bundle_mtp_layers(
                    int(status.get("jang_mtp_layers") or 0) if runtime_active else 0
                )
            except Exception as _pub_err:
                logger.debug(f"MTP bundle layer publish skipped: {_pub_err}")
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
        # A bundle that DECLARES MTP but is not runtime-supported used to
        # deactivate in total silence, so the model ran plain autoregressive
        # with nothing in the log to say why. MEASURED: Nemotron 3.5 Lightning
        # (JANG_2L/4M/6M, 34 mtp.layers.0.* tensors, num_nextn_predict_layers=1)
        # and Inkling both hit this — the only surfaces telling the truth were
        # /health.mtp and the CLI startup banner. Decode-time ineligibility is
        # DEBUG-only, so nothing at INFO ever mentioned it.
        #
        # Say it once, at INFO, when the bundle declared MTP. Silence about a
        # feature the bundle advertises is the defect.
        if status.get("mtp_declared") or status.get("artifact_available"):
            logger.info(
                "Native MTP NOT active for %s: %s (family=%s, status=%s). "
                "The bundle declares MTP weights; generation will run "
                "autoregressive.",
                model_path,
                status.get("runtime_reason") or "runtime not supported",
                status.get("family"),
                status.get("status"),
            )
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

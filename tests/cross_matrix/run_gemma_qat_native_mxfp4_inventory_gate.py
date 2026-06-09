#!/usr/bin/env python3
"""Inventory and proof-boundary gate for Gemma QAT/native MXFP4 rows.

This is intentionally no-heavy: it does not download models or load weights.
It records which Gemma QAT/native-MXFP4 release rows are present locally and
which live proof surfaces remain open, so older Gemma4 JANG_4M evidence cannot
be reused for new QAT/native rows by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
    Path("/Users/example/models"),
    Path("/Users/example/.mlxstudio/models"),
)
DEFAULT_OUT = Path(
    "build/current-gemma-qat-native-mxfp4-local-inventory-after-source-smoke-map-20260609.json"
)

REQUIRED_QAT_ROWS = {
    "gemma4_e2b_qat_native_mxfp4": {
        "display": "Gemma 4 E2B QAT/native MXFP4",
        "path_markers": ("gemma-4-e2b", "qat", "mxfp4"),
        "expected_model_type": "gemma4",
        "tool_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "requires": ("text", "vision", "audio"),
    },
    "gemma4_e4b_qat_native_mxfp4": {
        "display": "Gemma 4 E4B QAT/native MXFP4",
        "path_markers": ("gemma-4-e4b", "qat", "mxfp4"),
        "expected_model_type": "gemma4",
        "tool_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "requires": ("text", "vision", "audio"),
    },
    "gemma4_12b_native_mxfp4": {
        "display": "Gemma 4 12B native MXFP4/QAT-style",
        "path_markers": ("gemma-4-12b", "qat", "mxfp4"),
        "expected_model_type": "gemma4_unified",
        "tool_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "requires": ("text", "vision", "audio", "video"),
    },
    "gemma4_26b_vl": {
        "display": "Gemma 4 26B QAT/native MXFP4 VL/video",
        "path_markers": ("gemma-4-26b", "qat", "mxfp4"),
        "expected_model_type": "gemma4",
        "tool_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "requires": ("text", "vision", "video"),
    },
    "gemma4_31v_or_31b_vl": {
        "display": "Gemma 4 31V/31B QAT/native MXFP4 VL/video",
        "path_markers": ("gemma-4-31", "qat", "mxfp4"),
        "expected_model_type": "gemma4",
        "tool_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "requires": ("text", "vision", "video"),
    },
}

REQUIRED_LIVE_PROOF_SURFACES = (
    "model_owned_generation_config_defaults",
    "chat_completions_nonstream_visible",
    "chat_completions_stream_content_delta",
    "responses_nonstream_visible",
    "responses_stream_content_delta",
    "responses_function_call_arguments_delta_done_output_item",
    "required_tool_call",
    "auto_tool_call",
    "tool_result_continuation",
    "no_tool_plain_answer",
    "multi_turn_recall",
    "raw_tool_markup_leak_check",
    "hidden_reasoning_leak_check",
    "json_xml_code_whitespace_exactness",
    "vision_if_advertised",
    "audio_if_advertised_or_honestly_gated",
    "video_if_advertised_or_honestly_gated",
    "post_media_text_recovery",
    "media_salted_cache_no_cross_reuse",
    "prefix_cache_first_miss_second_hit",
    "paged_or_native_cache_telemetry",
    "mixed_swa_component_state",
    "turboquant_kv_encode_decode_boundary_where_valid",
    "block_disk_l2_write",
    "fresh_process_l2_restore",
    "async_rederive_or_clean_partial_hit_policy",
    "cli_ui_parser_reasoning_cache_max_token_parity",
    "installed_app_startup_and_settings_parity",
)

SOURCE_LIVE_SMOKE_PROOFS = {
    "gemma4_e2b_qat_native_mxfp4": Path(
        "build/current-all-local-model-smoke-gemma4-e2b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json"
    ),
    "gemma4_e4b_qat_native_mxfp4": Path(
        "build/current-all-local-model-smoke-gemma4-e4b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json"
    ),
    "gemma4_12b_native_mxfp4": Path(
        "build/current-all-local-model-smoke-gemma4-12b-qat-mxfp4-fullmedia-tools-l2-after-tool-result-quoted-target-20260609/summary.json"
    ),
    "gemma4_26b_vl": Path(
        "build/current-all-local-model-smoke-gemma4-26b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json"
    ),
    "gemma4_31v_or_31b_vl": Path(
        "build/current-all-local-model-smoke-gemma4-31b-qat-mxfp4-tools-l2-after-audio-capability-gate-20260609/summary.json"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"_error": str(exc)}


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _weight_map(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    index = _load_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
    return {str(key): str(value) for key, value in weight_map.items()}


def _modality_backing(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    weight_map = _weight_map(path)
    keys = tuple(weight_map)
    lower_keys = tuple(key.lower() for key in keys)
    audio_advertised = bool(
        config.get("audio_config")
        or config.get("audio_token_id")
        or config.get("audio_token_ids")
        or config.get("audio_token_index")
        or config.get("audio_tokenizer_config")
    )
    vision_advertised = bool(config.get("vision_config") or config.get("image_token_id"))
    video_advertised = bool(config.get("video_config") or config.get("video_token_id"))
    audio_tower_count = sum(1 for key in keys if key.startswith("audio_tower."))
    audio_embed_count = sum(1 for key in lower_keys if "embed_audio" in key)
    vision_weight_count = sum(
        1
        for key in lower_keys
        if key.startswith("vision_tower.")
        or key.startswith("vision_embedder.")
        or key.startswith("embed_vision.")
        or key.startswith("visual.")
        or key.startswith("vision_model.")
    )
    video_weight_count = sum(
        1
        for key in lower_keys
        if key.startswith("video_tower.")
        or key.startswith("video_model.")
        or key.startswith("video_encoder.")
    )
    return {
        "weight_map_present": bool(weight_map),
        "audio_advertised_by_config": audio_advertised,
        "audio_weight_backed": audio_tower_count > 0,
        "audio_tower_weight_count": audio_tower_count,
        "audio_embed_only": audio_advertised and audio_tower_count == 0 and audio_embed_count > 0,
        "audio_embed_weight_count": audio_embed_count,
        "vision_advertised_by_config": vision_advertised,
        "vision_weight_backed": vision_weight_count > 0,
        "vision_weight_count": vision_weight_count,
        "video_advertised_by_config": video_advertised,
        "video_weight_backed": video_weight_count > 0,
        "video_weight_count": video_weight_count,
        "video_runtime_proof_required": video_advertised and video_weight_count == 0,
    }


def _row_from_config(path: Path) -> dict[str, Any]:
    config = _load_json(path / "config.json")
    jang = _load_json(path / "jang_config.json") if (path / "jang_config.json").exists() else {}
    quant = config.get("quantization") if isinstance(config.get("quantization"), dict) else {}
    architecture = jang.get("architecture") if isinstance(jang.get("architecture"), dict) else {}
    backing = _modality_backing(path, config)

    return {
        "path": str(path),
        "name": path.name,
        "model_type": config.get("model_type"),
        "text_model_type": _nested(config, "text_config", "model_type"),
        "vision": backing["vision_advertised_by_config"],
        "audio": backing["audio_advertised_by_config"],
        "video": backing["video_advertised_by_config"],
        "modality_backing": backing,
        "weight_format": (
            jang.get("weight_format")
            or config.get("weight_format")
            or quant.get("mode")
            or quant.get("method")
        ),
        "format": jang.get("format"),
        "quantization_mode": quant.get("mode"),
        "quantization_method": quant.get("method"),
        "quantization_bits": quant.get("bits"),
        "quantization_group_size": quant.get("group_size"),
        "jang_has_vision": architecture.get("has_vision", jang.get("has_vision")),
        "jang_has_audio": architecture.get("has_audio", jang.get("has_audio")),
        "has_generation_config": (path / "generation_config.json").is_file(),
        "has_tokenizer_config": (path / "tokenizer_config.json").is_file(),
        "has_processor_config": (path / "processor_config.json").is_file(),
        "has_jang_config": (path / "jang_config.json").is_file(),
    }


def inventory_gemma_models(roots: tuple[Path, ...] = DEFAULT_ROOTS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for config_path in root.glob("**/config.json"):
            model_dir = config_path.parent
            if model_dir in seen:
                continue
            if "gemma" not in str(model_dir).lower():
                continue
            seen.add(model_dir)
            rows.append(_row_from_config(model_dir))
    return sorted(rows, key=lambda row: row["path"])


def _matches_required(row: dict[str, Any], markers: tuple[str, ...]) -> bool:
    path = str(row.get("path") or "").lower()
    return all(marker.lower() in path for marker in markers)


def _source_live_smoke_status(row_key: str, proof_root: Path) -> dict[str, Any]:
    rel = SOURCE_LIVE_SMOKE_PROOFS.get(row_key)
    if rel is None:
        return {"status": "missing", "artifact": None}
    path = proof_root / rel
    proof = _load_json(path)
    status = "pass" if proof.get("status") == "pass" and int(proof.get("failed") or 0) == 0 else "missing"
    if path.exists() and status != "pass":
        status = "open"
    return {
        "status": status,
        "artifact": str(rel),
        "summary_status": proof.get("status"),
        "failed": proof.get("failed"),
        "completed": proof.get("completed"),
        "row_count": proof.get("row_count"),
    }


def classify_required_rows(
    rows: list[dict[str, Any]],
    *,
    proof_root: Path = Path("."),
) -> dict[str, dict[str, Any]]:
    classified: dict[str, dict[str, Any]] = {}
    for key, spec in REQUIRED_QAT_ROWS.items():
        matches = [row for row in rows if _matches_required(row, spec["path_markers"])]
        proof_status = "missing" if not matches else "open"
        notes: list[str] = []
        if not matches:
            notes.append("bundle_not_present_in_local_model_roots")
        else:
            for row in matches:
                if row.get("model_type") != spec["expected_model_type"]:
                    notes.append(
                        f"{row['name']}: model_type={row.get('model_type')!r} "
                        f"expected {spec['expected_model_type']!r}"
                    )
                for modality in spec["requires"]:
                    if modality == "text":
                        continue
                    if not row.get(modality):
                        notes.append(f"{row['name']}: advertised {modality}=false")
                    backing = row.get("modality_backing") if isinstance(row.get("modality_backing"), dict) else {}
                    if (
                        modality == "audio"
                        and backing.get("weight_map_present")
                        and backing.get("audio_advertised_by_config")
                        and not backing.get("audio_weight_backed")
                    ):
                        notes.append(f"{row['name']}: audio metadata present without audio_tower weights")
                    if (
                        modality == "video"
                        and backing.get("video_advertised_by_config")
                        and backing.get("video_runtime_proof_required")
                    ):
                        notes.append(f"{row['name']}: video metadata requires live frame-through-vision proof")
                if not row.get("has_generation_config"):
                    notes.append(f"{row['name']}: missing generation_config.json")
                if not row.get("has_tokenizer_config"):
                    notes.append(f"{row['name']}: missing tokenizer_config.json")
                if "mxfp4" in key and row.get("weight_format") != "mxfp4":
                    notes.append(
                        f"{row['name']}: weight_format={row.get('weight_format')!r} "
                        "does not prove native MXFP4"
                    )

        source_live_smoke = _source_live_smoke_status(key, proof_root)
        classified[key] = {
            "display": spec["display"],
            "status": proof_status,
            "expected_model_type": spec["expected_model_type"],
            "tool_parser": spec["tool_parser"],
            "reasoning_parser": spec["reasoning_parser"],
            "required_modalities": list(spec["requires"]),
            "matching_paths": [row["path"] for row in matches],
            "matching_rows": matches,
            "notes": notes,
            "live_proof_required": list(REQUIRED_LIVE_PROOF_SURFACES),
            "live_proof_status": "missing",
            "source_live_smoke": source_live_smoke,
        }
    return classified


def build_artifact(
    roots: tuple[Path, ...] = DEFAULT_ROOTS,
    *,
    proof_root: Path = Path("."),
) -> dict[str, Any]:
    rows = inventory_gemma_models(roots)
    classified = classify_required_rows(rows, proof_root=proof_root)
    def backing(row_key: str) -> dict[str, Any]:
        matches = classified.get(row_key, {}).get("matching_rows")
        if not isinstance(matches, list) or not matches:
            return {}
        first = matches[0]
        return first.get("modality_backing") if isinstance(first.get("modality_backing"), dict) else {}

    gemma12b_backing = backing("gemma4_12b_native_mxfp4")
    gemma26b_backing = backing("gemma4_26b_vl")
    gemma31b_backing = backing("gemma4_31v_or_31b_vl")
    gemma12b_audio_honestly_gated = (
        gemma12b_backing.get("audio_advertised_by_config") is True
        and gemma12b_backing.get("audio_weight_backed") is not True
        and gemma12b_backing.get("audio_embed_only") is True
    )
    missing = [key for key, row in classified.items() if row["status"] == "missing"]
    open_rows = [key for key, row in classified.items() if row["status"] == "open"]
    source_smoke_open = [
        key
        for key, row in classified.items()
        if row["status"] != "missing"
        and row.get("source_live_smoke", {}).get("status") != "pass"
    ]
    return {
        "status": "open" if missing or open_rows else "pass",
        "roots": [str(root) for root in roots],
        "count": len(rows),
        "rows": rows,
        "required_rows": classified,
        "checks": {
            "gemma4_e2b_qat_native_mxfp4_present": (
                classified["gemma4_e2b_qat_native_mxfp4"]["status"] != "missing"
            ),
            "gemma4_e4b_qat_native_mxfp4_present": (
                classified["gemma4_e4b_qat_native_mxfp4"]["status"] != "missing"
            ),
            "gemma4_12b_native_mxfp4_present": classified["gemma4_12b_native_mxfp4"]["status"] != "missing",
            "gemma4_26b_present": classified["gemma4_26b_vl"]["status"] != "missing",
            "gemma4_31v_or_31b_present": classified["gemma4_31v_or_31b_vl"]["status"] != "missing",
            "all_required_source_live_smokes_present": not source_smoke_open,
            "all_required_live_proofs_present": False,
            "gemma4_12b_audio_weight_backed": gemma12b_backing.get("audio_weight_backed") is True,
            "gemma4_12b_audio_honestly_gated": gemma12b_audio_honestly_gated,
            "gemma4_12b_vision_weight_backed": gemma12b_backing.get("vision_weight_backed") is True,
            "gemma4_12b_video_runtime_proof_required": (
                gemma12b_backing.get("video_runtime_proof_required") is True
            ),
            "gemma4_26b_video_runtime_proof_required": (
                gemma26b_backing.get("video_runtime_proof_required") is True
            ),
            "gemma4_31v_or_31b_video_runtime_proof_required": (
                gemma31b_backing.get("video_runtime_proof_required") is True
            ),
        },
        "missing_required_rows": missing,
        "open_required_rows": open_rows,
        "source_live_smoke_open_rows": source_smoke_open,
        "boundary": (
            "No-heavy inventory only. A present row remains open until live "
            "media/cache/tool/Responses/UI/installed-app proof artifacts cover "
            "the row's advertised modalities and parser/cache requirements. "
            "source_live_smoke records narrower current-source all_local_model_smoke "
            "evidence and is not installed-app or release clearance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root", action="append", type=Path)
    args = parser.parse_args()

    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    artifact = build_artifact(roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"count={artifact['count']}")
    print(f"missing_required_rows={artifact['missing_required_rows']}")
    print(f"open_required_rows={artifact['open_required_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-heavy MiMo V2.5 local bundle metadata contract.

This verifies the local release-candidate MiMo bundles fail closed for media:
weights/config sidecars may be preserved, but runtime-advertised modalities must
stay text-only until a real MiMo multimodal forward path is implemented and
live-tested.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("build/current-mimo-v2-local-bundle-metadata-contract-20260607.json")
DEFAULT_MANIFEST_OUT = Path("build/current-mimo-jangtq2-local-manifest-20260607.tsv")
DEFAULT_STRUCTURAL_OUT = Path("build/current-mimo-jang2l-local-structural-verify-20260606.json")

MIMO_LOCAL_BUNDLES = {
    "jangtq2": Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2"),
    "jang2l": Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L"),
}

EXPECTED_PRESERVED_MODALITIES = ["vision", "audio"]
EXPECTED_RUNTIME_STATUS = "weights_preserved_text_runtime"
REQUIRED_BOOKEND_KEYS = {
    "lm_head.weight",
    "lm_head.scales",
    "lm_head.biases",
    "model.embed_tokens.weight",
    "model.embed_tokens.scales",
    "model.embed_tokens.biases",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weight_map(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    return weight_map if isinstance(weight_map, dict) else {}


def _generation_config_status(path: Path) -> dict[str, Any]:
    generation_path = path / "generation_config.json"
    if not generation_path.exists():
        return {
            "exists": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": None,
        }
    generation_config = _load_json(generation_path)
    return {
        "exists": True,
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "max_new_tokens": generation_config.get("max_new_tokens"),
        "do_sample": generation_config.get("do_sample"),
    }


def _cache_topology(runtime: dict[str, Any]) -> dict[str, Any]:
    cache_topology = runtime.get("cache_topology")
    if not isinstance(cache_topology, dict):
        cache_topology = {}
    return {
        "family": cache_topology.get("family"),
        "prefix_cache": cache_topology.get("prefix_cache"),
        "l2_disk_cache": cache_topology.get("l2_disk_cache"),
        "turboquant_kv": cache_topology.get("turboquant_kv"),
        "swa_layers": cache_topology.get("swa_layers"),
    }


def _structural_status(name: str, path: Path) -> dict[str, Any]:
    failures: list[str] = []
    config_path = path / "config.json"
    jang_config_path = path / "jang_config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    jang_config = _load_json(jang_config_path) if jang_config_path.exists() else {}
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    capabilities = (
        config.get("capabilities") if isinstance(config.get("capabilities"), dict) else {}
    )
    tools = capabilities.get("tools") if isinstance(capabilities.get("tools"), dict) else {}
    reasoning = (
        capabilities.get("reasoning")
        if isinstance(capabilities.get("reasoning"), dict)
        else {}
    )
    weight_map = _weight_map(path)
    mapped_files = sorted(set(weight_map.values()))
    missing_mapped_files = [rel for rel in mapped_files if not (path / rel).exists()]
    bookend_missing = sorted(REQUIRED_BOOKEND_KEYS - set(weight_map))
    switch_mlp_keys = [key for key in weight_map if ".mlp.switch_mlp." in key]
    legacy_expert_keys = [key for key in weight_map if ".mlp.experts." in key]
    generation_config = _generation_config_status(path)
    cache_topology = _cache_topology(runtime)

    if not weight_map:
        failures.append("weight_map_missing")
    if missing_mapped_files:
        failures.append("mapped_safetensor_files_missing")
    if bookend_missing:
        failures.append("bookend_affine_sidecars_missing")
    if not switch_mlp_keys:
        failures.append("stacked_switch_mlp_keys_missing")
    if legacy_expert_keys:
        failures.append("legacy_per_expert_layout_present")
    if tools.get("parser") != "xml_function" or tools.get("supported") is not True:
        failures.append("tool_parser_metadata_missing")
    if reasoning.get("parser") != "think_xml" or reasoning.get("supported") is not True:
        failures.append("reasoning_parser_metadata_missing")
    if cache_topology.get("prefix_cache") is not True:
        failures.append("prefix_cache_metadata_missing")
    if cache_topology.get("l2_disk_cache") is not True:
        failures.append("l2_disk_cache_metadata_missing")
    if not cache_topology.get("turboquant_kv"):
        failures.append("turboquant_kv_boundary_missing")
    if not generation_config["exists"]:
        failures.append("generation_config_missing")

    if name == "jangtq2":
        if not (path / "jangtq_runtime.safetensors").exists():
            failures.append("jangtq_runtime_safetensors_missing")
        if runtime.get("tq_layout") != "prestacked_switch_mlp":
            failures.append("jangtq_prestacked_layout_metadata_missing")
    if name == "jang2l":
        if jang_config.get("expert_layout") not in {
            "stacked_switch_mlp",
            "stacked_affine_switch_mlp",
        } or jang_config.get("runtime_expert_module") != "switch_mlp":
            failures.append("jang2l_stacked_layout_metadata_missing")
        if jang_config.get("bundle_has_mtp") is not False:
            failures.append("jang2l_mtp_absence_metadata_missing")

    return {
        "name": name,
        "path": str(path),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "model_type": config.get("model_type"),
        "profile": jang_config.get("profile") or runtime.get("quantization_profile"),
        "weight_count": len(weight_map),
        "mapped_file_count": len(mapped_files),
        "missing_mapped_files": missing_mapped_files[:20],
        "bookend_missing": bookend_missing,
        "switch_mlp_key_count": len(switch_mlp_keys),
        "legacy_expert_key_count": len(legacy_expert_keys),
        "generation_config": generation_config,
        "cache_topology": cache_topology,
        "tools": {
            "supported": tools.get("supported"),
            "parser": tools.get("parser"),
        },
        "reasoning": {
            "supported": reasoning.get("supported"),
            "parser": reasoning.get("parser"),
            "default": reasoning.get("default"),
        },
        "runtime": {
            "attention_impl": runtime.get("attention_impl"),
            "bundle_has_mtp": runtime.get("bundle_has_mtp"),
            "mtp_mode": runtime.get("mtp_mode"),
            "tq_layout": runtime.get("tq_layout"),
        },
    }


def _bundle_status(name: str, path: Path) -> dict[str, Any]:
    failures: list[str] = []
    config_path = path / "config.json"
    jang_config_path = path / "jang_config.json"
    if not path.exists():
        return {
            "name": name,
            "path": str(path),
            "status": "fail",
            "failures": ["bundle_missing"],
        }
    if not config_path.exists():
        failures.append("config_json_missing")
        config: dict[str, Any] = {}
    else:
        config = _load_json(config_path)
    if not jang_config_path.exists():
        failures.append("jang_config_json_missing")

    capabilities = config.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
        failures.append("capabilities_missing")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        failures.append("runtime_missing")

    modalities = capabilities.get("modalities")
    preserved = capabilities.get("preserved_modalities")
    unwired = capabilities.get("unwired_modalities")
    multimodal_status = capabilities.get("multimodal_status")
    runtime_mode = runtime.get("multimodal_mode")

    if modalities != ["text"]:
        failures.append("runtime_modalities_not_text_only")
    if preserved != EXPECTED_PRESERVED_MODALITIES:
        failures.append("preserved_modalities_not_recorded")
    if unwired != EXPECTED_PRESERVED_MODALITIES:
        failures.append("unwired_modalities_not_recorded")
    if multimodal_status != EXPECTED_RUNTIME_STATUS:
        failures.append("multimodal_status_not_text_runtime")
    if runtime_mode != EXPECTED_RUNTIME_STATUS:
        failures.append("runtime_multimodal_mode_not_text_runtime")
    if "vision_config" not in config:
        failures.append("vision_config_missing")
    if "audio_config" not in config:
        failures.append("audio_config_missing")
    if not (path / "preprocessor_config.json").exists():
        failures.append("preprocessor_config_missing")
    if not (path / "audio_tokenizer").exists():
        failures.append("audio_tokenizer_missing")

    return {
        "name": name,
        "path": str(path),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "config_path": str(config_path),
        "jang_config_path": str(jang_config_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "capabilities": {
            "modalities": modalities,
            "preserved_modalities": preserved,
            "unwired_modalities": unwired,
            "multimodal_status": multimodal_status,
        },
        "runtime": {
            "multimodal_mode": runtime_mode,
            "quantization_profile": runtime.get("quantization_profile"),
            "cache_topology": runtime.get("cache_topology"),
        },
        "sidecars": {
            "vision_config": "vision_config" in config,
            "audio_config": "audio_config" in config,
            "preprocessor_config": (path / "preprocessor_config.json").exists(),
            "audio_tokenizer": (path / "audio_tokenizer").exists(),
        },
    }


def build_artifact() -> dict[str, Any]:
    bundles = {name: _bundle_status(name, path) for name, path in MIMO_LOCAL_BUNDLES.items()}
    structural = {
        name: _structural_status(name, path)
        for name, path in MIMO_LOCAL_BUNDLES.items()
        if path.exists()
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(row["status"] == "pass" for row in bundles.values()) else "fail",
        "classification": "mimo_local_bundles_media_metadata_text_runtime_contract",
        "release_boundary": (
            "This proves local MiMo JANGTQ_2 and JANG_2L metadata honesty only. "
            "It does not implement or clear MiMo VL/audio/video runtime; media remains "
            "preserved_unwired until a real multimodal forward path and media cache proof exist."
        ),
        "expected_runtime_modalities": ["text"],
        "expected_preserved_modalities": EXPECTED_PRESERVED_MODALITIES,
        "bundles": bundles,
        "structural": structural,
    }


def build_structural_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    structural = artifact.get("structural")
    structural = structural if isinstance(structural, dict) else {}
    status = "pass" if structural and all(
        isinstance(row, dict) and row.get("status") == "pass"
        for row in structural.values()
    ) else "fail"
    return {
        "created_at": artifact["created_at"],
        "status": status,
        "classification": "mimo_local_bundle_noheavy_structural_contract",
        "release_boundary": (
            "No-heavy local bundle proof only. This verifies local MiMo JANGTQ_2 "
            "and JANG_2L config/index/sidecar/layout metadata; it does not prove "
            "live text exactness, tool-result continuation, media E2E, or app UI."
        ),
        "bundles": structural,
    }


def write_manifest(model_parent: Path, bundle: Path, out: Path) -> None:
    rows: list[tuple[int, str]] = []
    for file_path in sorted(path for path in bundle.rglob("*") if path.is_file()):
        rel = file_path.relative_to(model_parent)
        rows.append((file_path.stat().st_size, rel.as_posix()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(f"{size}\t{rel}\n" for size, rel in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--structural-out", type=Path, default=DEFAULT_STRUCTURAL_OUT)
    args = parser.parse_args()
    artifact = build_artifact()
    structural_artifact = build_structural_artifact(artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.structural_out.parent.mkdir(parents=True, exist_ok=True)
    args.structural_out.write_text(
        json.dumps(structural_artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        MIMO_LOCAL_BUNDLES["jangtq2"].parent,
        MIMO_LOCAL_BUNDLES["jangtq2"],
        args.manifest_out,
    )
    print(args.out)
    print(args.structural_out)
    print(args.manifest_out)
    print(f"status={artifact['status']}")
    print(f"structural_status={structural_artifact['status']}")
    for name, row in artifact["bundles"].items():
        print(f"{name}: {row['status']} failures={row['failures']}")
    return 0 if artifact["status"] == "pass" and structural_artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

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

MIMO_LOCAL_BUNDLES = {
    "jangtq2": Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2"),
    "jang2l": Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L"),
}

EXPECTED_PRESERVED_MODALITIES = ["vision", "audio"]
EXPECTED_RUNTIME_STATUS = "weights_preserved_text_runtime"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    artifact = build_artifact()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    for name, row in artifact["bundles"].items():
        print(f"{name}: {row['status']} failures={row['failures']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

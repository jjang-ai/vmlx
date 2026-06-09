#!/usr/bin/env python3
"""No-heavy local memory/index preflight for Nex/N2 JANG_1L.

This deliberately does not import or load model weights. It reads the safetensor
index/config metadata, records explicit GiB math, and decides whether a live
proof should be scheduled on the current host.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/Users/example/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANG_1L")
DEFAULT_OUT = REPO / "build/current-n2-pro-jang1l-local-memory-preflight-20260609.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _round_gib(byte_count: int) -> float:
    return round(byte_count / (1024**3), 2)


def memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "unit": "GiB",
            "total_gib": round(vm.total / (1024**3), 2),
            "available_gib": round(vm.available / (1024**3), 2),
            "percent": vm.percent,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact
        return {"unit": "GiB", "error": f"{type(exc).__name__}: {exc}"}


def _weight_counts(weight_map: dict[str, str]) -> dict[str, int]:
    keys = tuple(weight_map)
    return {
        "total_tensors": len(keys),
        "vision_tensors": sum("vision" in key for key in keys),
        "audio_tensors": sum("audio" in key for key in keys),
        "video_tensors": sum("video" in key for key in keys),
        "linear_attention_tensors": sum("linear_attn" in key for key in keys),
        "gated_delta_tensors": sum(("gated_delta" in key) or ("gdn" in key) for key in keys),
        "expert_tensors": sum(("expert" in key) or ("experts" in key) for key in keys),
        "mtp_tensors": sum("mtp" in key for key in keys),
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    index_path = model / "model.safetensors.index.json"
    config_path = model / "config.json"
    jang_config_path = model / "jang_config.json"
    index = _load_json(index_path)
    config = _load_json(config_path)
    jang_config = _load_json(jang_config_path)
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        weight_map = {}

    metadata = index.get("metadata") or {}
    payload_bytes = _as_int(metadata.get("total_size"))
    if payload_bytes <= 0:
        payload_bytes = sum(path.stat().st_size for path in model.glob("*.safetensors"))

    payload_gib = _round_gib(payload_bytes)
    required_headroom = float(args.required_extra_headroom_gib)
    required_available = round(payload_gib + required_headroom, 2)
    memory = memory_snapshot()
    available = memory.get("available_gib")
    launch_safe = (
        model.exists()
        and index_path.exists()
        and isinstance(available, (int, float))
        and available >= required_available
    )
    gap = (
        round(max(0.0, required_available - float(available)), 2)
        if isinstance(available, (int, float))
        else None
    )

    quant = jang_config.get("quantization") or config.get("quantization") or {}
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else None
    decision = "schedule_live_proof" if launch_safe else "do_not_launch"
    reason = (
        "payload plus required runtime headroom fits current available memory"
        if launch_safe
        else "payload plus required runtime headroom exceeds current available memory"
    )
    if not model.exists():
        decision = "missing_model"
        reason = "model directory is missing"
    elif not index_path.exists():
        decision = "missing_index"
        reason = "model.safetensors.index.json is missing"

    return {
        "artifact": str(args.out) if getattr(args, "out", None) else str(DEFAULT_OUT),
        "status": "open",
        "timestamp_unix": time.time(),
        "model_path": str(model),
        "model_exists": model.exists(),
        "index_exists": index_path.exists(),
        "config_exists": config_path.exists(),
        "jang_config_exists": jang_config_path.exists(),
        "no_load": True,
        "unit": "GiB",
        "decision": decision,
        "launch_safe": launch_safe,
        "classification": "careful_ram_live_proof_pending",
        "reason": reason,
        "indexed_payload_bytes": payload_bytes,
        "indexed_payload_gib": payload_gib,
        "indexed_payload_gb_decimal": round(payload_bytes / 1_000_000_000, 2),
        "required_extra_headroom_gib": required_headroom,
        "required_available_gib": required_available,
        "available_gib": available,
        "memory_gap_gib": gap,
        "system_memory": memory,
        "model_type": config.get("model_type"),
        "architecture": architecture,
        "format": jang_config.get("format") or metadata.get("format"),
        "format_version": metadata.get("format_version"),
        "artifact_profile": quant.get("profile"),
        "quantization": quant,
        "weights": _weight_counts(weight_map),
        "release_boundary": (
            "This is a no-heavy scheduling and artifact/index proof only. It is "
            "not live N2 JANG_1L runtime/cache/API/UI proof, and it must not be "
            "used to claim release support."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--required-extra-headroom-gib", type=float, default=4.0)
    args = parser.parse_args(argv)

    result = build_preflight(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

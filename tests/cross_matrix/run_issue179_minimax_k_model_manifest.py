#!/usr/bin/env python3
"""Write a no-load file manifest for the local MiniMax-K artifact.

The issue #179 reporter evidence cannot be closed unless the reporter's model
artifact is proven identical to the local artifact. This script records the
local side without importing MLX or loading weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K")
DEFAULT_OUT = Path(
    "build/current-issue179-minimax-k-local-model-manifest-20260527.json"
)

IMPORTANT_FILES = (
    "config.json",
    "generation_config.json",
    "jang_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "modeling_minimax_m2.py",
    "configuration_minimax_m2.py",
    "jangtq_runtime.safetensors",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path, root: Path, *, hash_contents: bool) -> dict[str, Any]:
    stat = path.stat()
    row: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": stat.st_size,
    }
    if hash_contents:
        row["sha256"] = sha256_file(path)
    return row


def build_manifest(model_path: Path, *, hash_shards: bool) -> dict[str, Any]:
    if not model_path.exists():
        return {
            "status": "missing",
            "model_path": str(model_path),
            "error": "model_path_missing",
        }

    all_files = sorted(path for path in model_path.iterdir() if path.is_file())
    safetensors = [path for path in all_files if path.name.endswith(".safetensors")]
    model_shards = [
        path
        for path in safetensors
        if path.name.startswith("model-") and path.name.endswith(".safetensors")
    ]
    important = [
        model_path / name for name in IMPORTANT_FILES if (model_path / name).exists()
    ]
    sidecars = [
        path
        for path in all_files
        if not path.name.endswith(".safetensors") or path.name == "jangtq_runtime.safetensors"
    ]
    hashed_paths = set(important)
    if hash_shards:
        hashed_paths.update(model_shards)

    rows = [
        file_row(path, model_path, hash_contents=path in hashed_paths)
        for path in all_files
    ]
    checks = {
        "has_config": (model_path / "config.json").exists(),
        "has_generation_config": (model_path / "generation_config.json").exists(),
        "has_jang_config": (model_path / "jang_config.json").exists(),
        "has_model_index": (model_path / "model.safetensors.index.json").exists(),
        "has_tokenizer": (model_path / "tokenizer.json").exists(),
        "has_jangtq_runtime": (model_path / "jangtq_runtime.safetensors").exists(),
        "model_shard_count_is_67": len(model_shards) == 67,
        "important_files_hashed": all(path in hashed_paths for path in important),
    }
    return {
        "status": "pass" if all(checks.values()) else "open",
        "model_path": str(model_path),
        "total_file_count": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "safetensors_count": len(safetensors),
        "model_shard_count": len(model_shards),
        "hash_shards": hash_shards,
        "checks": checks,
        "files": rows,
        "summary": {
            "sidecar_count": len(sidecars),
            "hashed_file_count": sum(1 for row in rows if "sha256" in row),
            "unhashed_safetensors_count": sum(
                1
                for row in rows
                if row["path"].endswith(".safetensors") and "sha256" not in row
            ),
        },
    }


def write_manifest(model_path: Path, out_path: Path, *, hash_shards: bool) -> dict[str, Any]:
    manifest = build_manifest(model_path, hash_shards=hash_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--hash-shards",
        action="store_true",
        help="Also hash every model-*.safetensors shard. This streams the full artifact.",
    )
    args = parser.parse_args()
    manifest = write_manifest(args.model_path, args.out, hash_shards=args.hash_shards)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": manifest["status"],
                "model_shard_count": manifest.get("model_shard_count"),
                "hashed_file_count": manifest.get("summary", {}).get("hashed_file_count"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

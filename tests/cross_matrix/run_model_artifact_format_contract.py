#!/usr/bin/env python3
"""Run no-heavy model artifact format detection contracts.

This gate protects the release row for JANG/JANGTQ/MXTQ, MXFP4, MXFP8, and
native-MTP artifact classification. It intentionally stays source/static only:
no model is loaded, and no path/name-only inference is accepted as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-model-artifact-format-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/model_configs.py",
    "vmlx_engine/native_mtp.py",
    "vmlx_engine/server.py",
    "vmlx_engine/utils/jang_loader.py",
    "tests/test_model_config_registry.py",
    "tests/test_jang_loader.py",
    "tests/test_native_mtp_autodetect.py",
    "tests/test_cross_matrix_audit_runner.py",
)

COMMANDS: dict[str, list[str]] = {
    "model_artifact_format_pytest": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_model_config_registry.py",
        "tests/test_jang_loader.py",
        "tests/test_native_mtp_autodetect.py",
        "tests/test_cross_matrix_audit_runner.py",
        "-k",
        "jang or JANG or mxfp or mxp or mtp or MTP or cache_profile",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    deselected = None
    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "deselected": deselected}


def _run(root: Path, name: str, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {name: _run(root, name, cmd) for name, cmd in COMMANDS.items()}
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    checks = {
        "jang_and_jangtq_detection": not failed,
        "mxfp4_detection": not failed,
        "mxfp8_detection": not failed,
        "dropped_mtp_detection": not failed,
        "preserved_mtp_detection": not failed,
        "cache_profile_detection": not failed,
        "not_path_name_only": not failed,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("failed=" + json.dumps(artifact["failed"]))
    for name, result in artifact["results"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run no-heavy cache architecture/family classification contracts.

This gate protects the release row that spans cache-family behavior: DSV4
native composite cache, ZAYA CCA, hybrid SSM/Mamba, MLA, MLLM media-salted
caches, TurboQuant KV runtime, and L2/disk serialization.
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


DEFAULT_OUT = Path("build/current-cache-architecture-contract-20260521.json")

CACHE_PATTERN = (
    "dsv4 or hybrid_ssm or cache_detail or MLA or mla or TurboQuant "
    "or turboquant or TQ or zaya or media_salt or cache_profile"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/scheduler.py",
    "vmlx_engine/prefix_cache.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/tq_disk_cache.py",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_hybrid_batching.py",
    "tests/test_hybrid_prefix_cache.py",
    "tests/test_kimi_k25_mla_patch.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_turboquant_cache_contract.py",
    "tests/test_tq_disk_cache.py",
)

COMMANDS: dict[str, list[str]] = {
    "api_cache_status_contracts": [
        sys.executable,
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "--out",
        "build/current-api-cache-contract-cache-architecture-check-20260521.json",
    ],
    "cache_family_pytest": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_batching.py",
        "tests/test_hybrid_batching.py",
        "tests/test_hybrid_prefix_cache.py",
        "tests/test_kimi_k25_mla_patch.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_turboquant_cache_contract.py",
        "tests/test_tq_disk_cache.py",
        "-k",
        CACHE_PATTERN,
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


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


def _load_api_cache_artifact(root: Path) -> dict[str, Any]:
    path = root / "build/current-api-cache-contract-cache-architecture-check-20260521.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact(root: Path) -> dict[str, Any]:
    results = {name: _run(root, name, cmd) for name, cmd in COMMANDS.items()}
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    api_artifact = _load_api_cache_artifact(root)
    api_checks = api_artifact.get("checks", {})
    cache_passed = results["cache_family_pytest"]["counts"]["passed"] or 0
    checks = {
        "dsv4_native_composite_cache_status": (
            not failed and api_checks.get("dsv4_native_cache_status") is True
        ),
        "zaya_typed_cca_status": (
            not failed and api_checks.get("zaya_typed_cca_status") is True
        ),
        "hybrid_ssm_partial_reuse": (
            not failed and api_checks.get("hybrid_ssm_partial_reuse") is True and cache_passed >= 55
        ),
        "generic_tq_not_applied_to_hybrid_ssm": (
            not failed and api_checks.get("no_generic_tq_on_hybrid_ssm") is True
        ),
        "turboquant_kv_runtime_contract": (
            not failed and api_checks.get("turboquant_kv_runtime_contract") is True and cache_passed >= 55
        ),
        "turboquant_disk_roundtrip": (
            not failed and api_checks.get("turboquant_disk_roundtrip") is True and cache_passed >= 55
        ),
        "mla_cache_shape_contracts": not failed and cache_passed >= 55,
        "cache_detail_telemetry_contracts": not failed and cache_passed >= 55,
        "mllm_media_salt_cache_contracts": not failed and cache_passed >= 55,
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
        "api_cache_artifact_status": api_artifact.get("status"),
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
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

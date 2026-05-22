#!/usr/bin/env python3
"""Run no-publish packaged integrity contracts.

This gate does not build, tag, upload, notarize, or update release feeds. It
checks the release-gate script contracts, bundled Python verifier, and the dry
release gate behavior. The current dry gate is allowed to fail only for the
known DSV4 long-output/code quality objective row.
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


DEFAULT_OUT = Path("build/current-packaged-integrity-contract-20260521.json")
EXPECTED_OPEN_REQUIREMENT = "DSV4 long-output/code/file-generation quality is release-cleared"

SOURCE_HASH_FILES = (
    "panel/scripts/release-gate-python-app.py",
    "panel/scripts/verify-bundled-python.sh",
    "panel/scripts/bundle-python.sh",
    "panel/package.json",
    "pyproject.toml",
    "tests/test_release_gate_python_app.py",
    "tests/cross_matrix/summarize_objective_proof.py",
    "tests/test_objective_proof_digest.py",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "release_gate_unit_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_release_gate_python_app.py",
        ],
    ),
    "bundled_python_verifier": (
        Path("panel"),
        [
            "npm",
            "run",
            "verify-bundled",
        ],
    ),
    "release_gate_skip_app": (
        Path("."),
        [
            "panel/scripts/release-gate-python-app.py",
            "--skip-app",
            "--skip-gui",
        ],
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    match = re.search(r"Tests\s+(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def _run(root: Path, name: str, cwd_rel: Path, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root / cwd_rel,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd_rel),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout_tail": proc.stdout.splitlines()[-100:],
    }


def release_gate_failure_is_expected(step: dict[str, Any]) -> bool:
    if step["returncode"] == 0:
        return True
    text = "\n".join(step.get("stdout_tail", []))
    fail_lines = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    expected = f"[FAIL] objective proof digest: {EXPECTED_OPEN_REQUIREMENT}"
    forbidden = (
        "bundled python import gate: FAIL",
        "panel typecheck: FAIL",
        "panel request/type tests: FAIL",
        "version triple: FAIL",
    )
    return fail_lines == [expected] and not any(item in text for item in forbidden)


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    release_gate_ok = release_gate_failure_is_expected(results["release_gate_skip_app"])
    unit_passed = results["release_gate_unit_contracts"]["counts"]["passed"] or 0
    verifier_output = "\n".join(results["bundled_python_verifier"]["stdout_tail"])
    checks = {
        "release_gate_unit_contracts_pass": (
            results["release_gate_unit_contracts"]["returncode"] == 0 and unit_passed >= 38
        ),
        "bundled_python_verify_passes": results["bundled_python_verifier"]["returncode"] == 0,
        "bundled_engine_version_matches_package_json": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled vmlx_engine version matches package.json" in verifier_output
        ),
        "bundled_engine_hash_parity": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled critical vmlx_engine files match source content" in verifier_output
        ),
        "bundled_jang_tools_hash_parity": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled critical jang_tools files match source content" in verifier_output
        ),
        "bundled_console_scripts_relocatable": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "console-script shebangs are relocatable" in verifier_output
        ),
        "bundled_media_and_jang_dependencies_import": (
            results["bundled_python_verifier"]["returncode"] == 0
            and "bundled-python: all critical imports ok" in verifier_output
        ),
        "dry_release_gate_fails_only_on_known_dsv4_objective": release_gate_ok,
    }
    failed = [
        name
        for name, result in results.items()
        if result["returncode"] != 0 and not (name == "release_gate_skip_app" and release_gate_ok)
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) and not failed else "fail",
        "checks": checks,
        "failed": failed,
        "known_expected_release_gate_open_requirement": EXPECTED_OPEN_REQUIREMENT,
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
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

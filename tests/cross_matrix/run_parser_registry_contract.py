#!/usr/bin/env python3
"""Run no-heavy parser registry parity contracts.

This gate protects the app/engine boundary that previously regressed MiniMax:
the panel must not emit tool or reasoning parser ids that the engine CLI cannot
accept, and family autodetection must keep parser ids canonical.
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


DEFAULT_OUT = Path("build/current-parser-registry-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/cli.py",
    "vmlx_engine/reasoning/__init__.py",
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/tool_parsers/__init__.py",
    "panel/src/main/model-config-registry.ts",
    "panel/src/shared/reasoningParserAliases.ts",
    "panel/src/shared/toolParserAliases.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/tests/model-config-registry.test.ts",
    "panel/tests/settings-flow.test.ts",
    "tests/cross_matrix/run_decode_speed_gate.py",
    "tests/test_engine_audit.py",
    "tests/test_model_config_registry.py",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_parser_registry": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_engine_audit.py",
            "tests/test_model_config_registry.py",
            "-k",
            "parser or minimax or reasoning_parser",
        ],
    ),
    "panel_parser_registry": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/model-config-registry.test.ts",
            "tests/settings-flow.test.ts",
            "--testNamePattern",
            "parser|Parser|reasoning|Reasoning|minimax|MiniMax",
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
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    engine_passed = results["engine_parser_registry"]["counts"]["passed"] or 0
    panel_passed = results["panel_parser_registry"]["counts"]["passed"] or 0
    checks = {
        "engine_accepts_registered_reasoning_parsers": not failed and engine_passed >= 30,
        "engine_accepts_registered_tool_parsers": not failed and engine_passed >= 30,
        "panel_emitted_reasoning_parsers_are_engine_valid": not failed and panel_passed >= 35,
        "panel_emitted_tool_parsers_are_engine_valid": not failed and panel_passed >= 35,
        "minimax_m2_reasoning_parser_regression": not failed and engine_passed >= 30 and panel_passed >= 35,
        "parser_aliases_are_canonical_before_cli": not failed and panel_passed >= 35,
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
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

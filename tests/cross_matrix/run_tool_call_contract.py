#!/usr/bin/env python3
"""Run no-heavy tool-call parser, loop, and panel security contracts.

This gate protects the DSV4/DSML tool-call path and the app-side built-in tool
loop controls. It intentionally does not load a model; live default-cache DSV4
proof is tracked separately by ``run_dsv4_default_cache_tool_loop_gate.py``.
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


DEFAULT_OUT = Path("build/current-tool-call-contract-20260521.json")

DSML_PATTERN = (
    "streaming_rejects_schema_residue_arguments "
    "or streaming_rejects_value_attribute_residue_arguments "
    "or streaming_accepts_schema_valid_dsml_arguments "
    "or server_rejects_dsv4_placeholder_residue_tool_arguments "
    "or server_rejects_dsv4_value_residue_tool_arguments "
    "or dsml_parser_rejects_placeholder_attribute_residue_as_argument_value "
    "or server_cleans_suppressed_dsv4_dsml_without_schema_or_call "
    "or responses_tool_choice_none_does_not_fallback_to_raw_dsml "
    "or test_repairs_live_dsv4_default_cache_nested_name_inv_shape "
    "or test_repairs_live_dsv4_default_cache_invue_typo_shape "
    "or test_repairs_live_dsv4_inv_par_value_attribute_shape "
    "or test_responses_dsml_tool_call_suppresses_planning_preamble "
    "or test_dsv4_responses_stream_buffers_tool_preamble"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_tool_format.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/coding-tools.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/tests/tool-executor-security.test.ts",
    "panel/tests/tool-auto-continue.test.ts",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_dsv4_dsml_tool_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_dsml_tool_parser.py",
            "tests/test_tool_format.py",
            "tests/test_engine_audit.py",
            "-k",
            DSML_PATTERN,
        ],
    ),
    "panel_tool_loop_security": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/tool-executor-security.test.ts",
            "tests/tool-auto-continue.test.ts",
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
    engine_passed = results["engine_dsv4_dsml_tool_contracts"]["counts"]["passed"] or 0
    panel_passed = results["panel_tool_loop_security"]["counts"]["passed"] or 0
    live_default_cache_artifact = root / "build/current-dsv4-default-cache-tool-loop/result.json"
    checks = {
        "tool_parser_residue_rejected_instead_of_executed": not failed and engine_passed >= 13,
        "schema_valid_dsml_tool_call_preserved": not failed and engine_passed >= 13,
        "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid": not failed and engine_passed >= 13,
        "dsv4_tool_preamble_suppressed_and_not_stored_without_call": not failed and engine_passed >= 13,
        "tool_choice_none_does_not_fallback_to_raw_dsml": not failed and engine_passed >= 13,
        "panel_tool_executor_blocks_unsafe_paths_and_commands": not failed and panel_passed >= 12,
        "panel_max_tool_iterations_caps_tool_loops": not failed and panel_passed >= 12,
        "live_default_cache_dsv4_tool_loop_artifact_present": live_default_cache_artifact.exists(),
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
        "live_default_cache_artifact": str(live_default_cache_artifact),
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

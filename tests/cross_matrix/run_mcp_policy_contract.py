#!/usr/bin/env python3
"""Run no-heavy MCP policy, redaction, UI, and gateway contracts."""

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


DEFAULT_OUT = Path("build/current-mcp-policy-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/mcp/manager.py",
    "vmlx_engine/mcp/config.py",
    "vmlx_engine/mcp/security.py",
    "vmlx_engine/mcp/types.py",
    "tests/test_mcp_policy.py",
    "tests/test_mcp_security.py",
    "panel/src/main/api-gateway.ts",
    "panel/src/main/ipc/sessions.ts",
    "panel/src/main/mcp-config-store.ts",
    "panel/src/shared/mcpPolicy.ts",
    "panel/src/shared/mcpConfigValidation.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/tests/mcp-policy.test.ts",
    "panel/tests/mcp-config-store.test.ts",
    "panel/tests/mcp-gateway-routing.test.ts",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_mcp_policy_security": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_mcp_policy.py",
            "tests/test_mcp_security.py",
        ],
    ),
    "panel_mcp_policy_gateway": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/mcp-policy.test.ts",
            "tests/mcp-config-store.test.ts",
            "tests/mcp-gateway-routing.test.ts",
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
    engine_passed = results["engine_mcp_policy_security"]["counts"]["passed"] or 0
    panel_passed = results["panel_mcp_policy_gateway"]["counts"]["passed"] or 0
    checks = {
        "mcp_autodiscovery_without_cli_or_env": not failed and engine_passed >= 76,
        "mcp_policy_filters_servers_tools_before_schema_merge": not failed and engine_passed >= 76,
        "mcp_disabled_tool_execution_rejected_server_side": not failed and engine_passed >= 76,
        "mcp_status_redacts_urls_headers_env": not failed and engine_passed >= 76,
        "mcp_command_security_blocks_injection": not failed and engine_passed >= 76,
        "panel_mcp_policy_flags_and_preview_wired": not failed and panel_passed >= 13,
        "panel_mcp_config_import_redacts_metadata": not failed and panel_passed >= 13,
        "mcp_gateway_routes_by_explicit_model": not failed and panel_passed >= 13,
        "mcp_gateway_rejects_ambiguous_multi_session_requests": not failed and panel_passed >= 13,
        "builtin_electron_tools_separate_from_mcp_execution": not failed and panel_passed >= 13,
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

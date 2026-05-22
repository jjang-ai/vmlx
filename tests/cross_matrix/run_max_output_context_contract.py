#!/usr/bin/env python3
"""Run no-heavy max-output/max-context separation contracts.

This gate protects the release regression Eric called out: server default
output caps, per-chat/API output overrides, and prompt/context caps must stay
separate across panel launch, request building, persisted-session migration,
and server request resolution.
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


DEFAULT_OUT = Path("build/current-max-output-context-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "tests/test_engine_audit.py",
    "panel/src/main/chat-override-policy.ts",
    "panel/src/main/sessions.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/renderer/src/i18n/locales/en.json",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/database-migrations.test.ts",
    "panel/tests/request-builder.test.ts",
    "panel/tests/chat-override-policy.test.ts",
    "panel/tests/dsv4-request-budget.test.ts",
)

PANEL_PATTERN = (
    "maxTokens|Max Tokens|Max Output|Max Context|max context|max output|"
    "max_tokens|max_output_tokens|32768|server default output|per-chat|"
    "omits max_tokens|max token budget|output budgets"
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_output_context_resolution": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_request_output_caps_override_server_default_without_touching_context_cap",
        ],
    ),
    "panel_output_context_wiring": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/settings-flow.test.ts",
            "tests/chat-settings-compatibility.test.ts",
            "tests/database-migrations.test.ts",
            "tests/request-builder.test.ts",
            "tests/chat-override-policy.test.ts",
            "tests/dsv4-request-budget.test.ts",
            "--testNamePattern",
            PANEL_PATTERN,
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
    engine_passed = results["engine_output_context_resolution"]["counts"]["passed"] or 0
    panel_passed = results["panel_output_context_wiring"]["counts"]["passed"] or 0
    checks = {
        "server_default_output_cap_uses_max_tokens": not failed and engine_passed >= 1,
        "chat_max_tokens_overrides_server_default_per_request": not failed and engine_passed >= 1,
        "responses_max_output_tokens_overrides_server_default_per_request": not failed and engine_passed >= 1,
        "prompt_context_caps_do_not_rewrite_output_cap": not failed and engine_passed >= 1,
        "panel_server_default_output_maps_to_max_tokens": not failed and panel_passed >= 20,
        "panel_max_context_maps_to_max_prompt_tokens": not failed and panel_passed >= 20,
        "stale_32768_session_output_caps_are_migrated": not failed and panel_passed >= 20,
        "chat_output_cap_remains_per_chat_override": not failed and panel_passed >= 20,
        "request_builders_omit_auto_output_cap": not failed and panel_passed >= 20,
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

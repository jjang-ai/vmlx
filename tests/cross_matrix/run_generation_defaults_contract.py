#!/usr/bin/env python3
"""Run no-heavy generation-defaults/no-hidden-forcing contracts.

This gate protects model-owned sampling defaults across the app/engine
boundary. Bundles may declare generation defaults in ``generation_config.json``
or JANG chat metadata in ``jang_config.json``; the app must surface those
values without copying them into sticky hidden overrides, and request/API
parameters must remain explicit.
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


DEFAULT_OUT = Path("build/current-generation-defaults-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/cli.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/models.ts",
    "panel/src/main/sessions.ts",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/tests/generation-defaults.test.ts",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "panel_generation_defaults": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/generation-defaults.test.ts",
        ],
    ),
    "engine_generation_defaults": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_engine_audit.py",
            "-k",
            "generation_config or max_tokens_resolution_contract or sampling",
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
    panel_passed = results["panel_generation_defaults"]["counts"]["passed"] or 0
    engine_passed = results["engine_generation_defaults"]["counts"]["passed"] or 0
    checks = {
        "generation_config_defaults_are_surfaced": not failed and panel_passed >= 7,
        "jang_config_sampling_defaults_override_generation_config": not failed and panel_passed >= 7,
        "disabled_top_k_sentinels_normalize_to_off": not failed and panel_passed >= 7,
        "mode_specific_jang_repetition_penalty_is_metadata_owned": not failed and panel_passed >= 7,
        "request_api_overrides_win_over_startup_defaults": not failed and engine_passed >= 25,
        "bundle_max_new_tokens_preserved_when_omitted": not failed and engine_passed >= 25,
        "omitted_max_tokens_without_bundle_default_is_bounded": not failed and engine_passed >= 25,
        "no_hidden_sampler_forcing_or_repetition_floor": not failed and engine_passed >= 25,
        "panel_does_not_emit_default_sampler_cli_flags": not failed and engine_passed >= 25,
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

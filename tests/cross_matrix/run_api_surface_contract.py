#!/usr/bin/env python3
"""Run no-heavy API surface parity contracts.

This gate protects the API surfaces Eric called out directly: OpenAI Chat
Completions, OpenAI Responses, Anthropic messages, Ollama gateway behavior, and
panel request builders. It intentionally reuses the lower-level API/cache
contract for server-side routing while adding a panel-side request translation
slice so the app boundary is covered by a single artifact.
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


DEFAULT_OUT = Path("build/current-api-surface-contract-20260521.json")
NESTED_OUT = Path("build/current-api-cache-contract-api-surface-check-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/chat.ts",
    "panel/src/main/sessions.ts",
    "panel/src/main/server/api-gateway.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/tests/request-builder.test.ts",
    "panel/tests/api-gateway-ollama-behavior.test.ts",
    "panel/tests/chat-override-policy.test.ts",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "server_api_surface": (
        Path("."),
        [
            sys.executable,
            "tests/cross_matrix/run_noheavy_api_cache_contract.py",
            "--out",
            str(NESTED_OUT),
        ],
    ),
    "panel_api_request_builders": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/request-builder.test.ts",
            "tests/api-gateway-ollama-behavior.test.ts",
            "tests/chat-override-policy.test.ts",
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
    match = re.search(r"passed=(\d+)", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"deselected=(\d+)", output)
    if match:
        deselected = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match and deselected is None:
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


def _load_nested(root: Path) -> dict[str, Any]:
    path = root / NESTED_OUT
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    nested = _load_nested(root)
    nested_checks = nested.get("checks", {})
    panel_passed = results["panel_api_request_builders"]["counts"]["passed"] or 0
    checks = {
        "openai_chat_completions_sampling_defaults": (
            not failed and nested_checks.get("openai_chat_sampling_kwargs") is True
        ),
        "openai_responses_sampling_defaults": (
            not failed and nested_checks.get("responses_sampling_kwargs") is True
        ),
        "chat_and_responses_output_caps_override_server_default": (
            not failed
            and nested_checks.get("request_output_caps_override_server_default") is True
        ),
        "prompt_context_caps_stay_separate_from_output_caps": (
            not failed
            and nested_checks.get("prompt_context_caps_stay_separate_from_output_caps") is True
        ),
        "anthropic_adapter_bundle_defaults": (
            not failed and nested_checks.get("anthropic_bundle_defaults") is True
        ),
        "ollama_adapter_streaming_done_behavior": (
            not failed and nested_checks.get("ollama_adapter_surface") is True
        ),
        "panel_request_builder_sampling_and_output_overrides": not failed and panel_passed >= 53,
        "panel_ollama_gateway_omits_disabled_sentinels": not failed and panel_passed >= 53,
        "panel_chat_override_policy_preserves_explicit_values": not failed and panel_passed >= 53,
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
        "nested_api_cache_status": nested.get("status"),
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

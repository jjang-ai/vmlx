#!/usr/bin/env python3
"""Run no-heavy VLM media/cache/tool-followup contracts.

This gate protects the VLM side of the current release hardening: image/video
request serialization, media-salted cache keys, hybrid SSM cache follow-up, and
tool-result media replay must stay covered without implying live Omni/video
quality clearance.
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


DEFAULT_OUT = Path("build/current-vl-media-cache-contract-20260521.json")

PYTEST_PATTERN = (
    "video_url or video_fallback or media_salt or mediaSalt or tool_replay "
    "or mllm_chat_template_receives_effective_tool_schemas "
    "or forwards_effective_tools or hybrid_ssm or qwen36 or zaya or content_part"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/models/mllm.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/mllm_cache.py",
    "vmlx_engine/engine/batched.py",
    "vmlx_engine/openai_models.py",
    "panel/src/main/ipc/chat.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/tools/registry.ts",
    "panel/tests/tool-media-followup.test.ts",
    "panel/tests/image-display-consistency.test.ts",
    "tests/test_vl_video_regression.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_mllm_tool_replay.py",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_vl_media_cache_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_vl_video_regression.py",
            "tests/test_mllm_scheduler_cache.py",
            "tests/test_mllm_tool_replay.py",
            "-k",
            PYTEST_PATTERN,
        ],
    ),
    "panel_vl_media_followup_contracts": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/tool-media-followup.test.ts",
            "tests/image-display-consistency.test.ts",
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
    engine_passed = results["engine_vl_media_cache_contracts"]["counts"]["passed"] or 0
    panel_passed = results["panel_vl_media_followup_contracts"]["counts"]["passed"] or 0
    checks = {
        "video_url_request_schema": not failed and engine_passed >= 22,
        "video_fallback_processing": not failed and engine_passed >= 22,
        "media_cache_salt_separates_modal_inputs": not failed and engine_passed >= 22,
        "hybrid_ssm_vlm_cache_contracts": not failed and engine_passed >= 22,
        "mllm_tool_replay_preserves_effective_tools": not failed and engine_passed >= 22,
        "panel_read_video_builtin_tool": not failed and panel_passed >= 12,
        "panel_media_tool_followup_content_parts": not failed and panel_passed >= 12,
        "panel_image_display_consistency": not failed and panel_passed >= 12,
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

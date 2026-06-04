#!/usr/bin/env python3
"""Guarded max2 DSV4 real-UI proof launcher.

This is the safe entrypoint for the real Electron UI half of the DSV4 release
blocker. It checks the current-source proof worktree, model, memory, Node, and
staged app path before it can launch the UI proof. In unsafe states it writes a
skipped artifact and does not start DSV4.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.run_remote_max2_dsv4_readiness import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_REQUIRED_AVAILABLE_GB,
    DEFAULT_REQUIRED_VERSION,
    DEFAULT_WORKTREE,
    build_readiness,
)


DEFAULT_OUT = Path("build/current-remote-max2-dsv4-real-ui-guard-20260604.json")
DEFAULT_PROOF_BASENAME = "current-real-ui-live-model-dsv4-max2-proof-worktree-20260604"
DEFAULT_APP = DEFAULT_WORKTREE / "panel/release/sequoia-app/mac-arm64/vMLX.app"
DEFAULT_NODE = Path("/opt/homebrew/bin/node")
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_TOOL_ITERATIONS = 8
DEFAULT_CACHE_EXPECT_REGEX = (
    "Prefix Cache|Paged KV Cache|Block Disk Cache|Stored Cache Quantization"
)


def remote_real_ui_command(
    *,
    worktree: Path,
    node: Path,
    app: Path,
    model: Path,
    proof_basename: str = DEFAULT_PROOF_BASENAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    cache_expect_regex: str = DEFAULT_CACHE_EXPECT_REGEX,
) -> str:
    env = {
        "VMLINUX_REAL_UI_APP_PATH": str(app),
        "VMLINUX_REAL_UI_MODEL_PATH": str(model),
        "VMLINUX_REAL_UI_PROOF_BASENAME": proof_basename,
        "VMLINUX_REAL_UI_WIRE_API": "responses",
        "VMLINUX_REAL_UI_BUILTIN_TOOLS": "1",
        "VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS": str(max_tool_iterations),
        "VMLINUX_REAL_UI_MAX_TOKENS": str(max_tokens),
        "VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS": "1",
        "VMLINUX_REAL_UI_CHECK_MEDIA": "0",
        "VMLINUX_REAL_UI_CACHE_EXPECT_REGEX": cache_expect_regex,
    }
    exports = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())
    )
    return (
        f"cd {shlex.quote(str(worktree))} && "
        f"{exports} {shlex.quote(str(node))} panel/scripts/live-real-ui-model-proof.mjs"
    )


def _ssh(host: str, command: str, *, connect_timeout: int) -> str:
    return subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            host,
            command,
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )


def remote_ui_prereqs(
    *,
    host: str,
    worktree: Path,
    node: Path,
    app: Path,
    connect_timeout: int,
) -> dict[str, Any]:
    command = f"""
cd {shlex.quote(str(worktree))}
node_path={shlex.quote(str(node))}
node_present=no
node_version=""
if [ -x "$node_path" ]; then
  node_present=yes
  node_version="$("$node_path" --version 2>/dev/null || true)"
fi
script_present=no
app_present=no
if [ -f panel/scripts/live-real-ui-model-proof.mjs ]; then script_present=yes; fi
if [ -d {shlex.quote(str(app))} ]; then app_present=yes; fi
NODE_PRESENT="$node_present" NODE_VERSION="$node_version" SCRIPT_PRESENT="$script_present" APP_PRESENT="$app_present" python3 - <<'PY'
import json
import os

print(json.dumps({{
    "node_present": os.environ.get("NODE_PRESENT") == "yes",
    "node_version": os.environ.get("NODE_VERSION", ""),
    "script_present": os.environ.get("SCRIPT_PRESENT") == "yes",
    "app_present": os.environ.get("APP_PRESENT") == "yes",
}}))
PY
"""
    text = _ssh(host, command, connect_timeout=connect_timeout).strip().splitlines()[-1]
    return json.loads(text)


def build_guard_result(
    *,
    readiness: dict[str, Any],
    prereqs: dict[str, Any],
    command: str,
    dry_run: bool,
) -> dict[str, Any]:
    blockers = list(readiness.get("launch_blockers") or [])
    if prereqs.get("node_present") is not True or not prereqs.get("node_version"):
        blockers.append("node_missing")
    if prereqs.get("script_present") is not True:
        blockers.append("ui_proof_script_missing")
    if prereqs.get("app_present") is not True:
        blockers.append("staged_app_missing")
    if dry_run:
        status = "dry_run"
        launch_decision = "dry_run_no_launch"
    elif blockers:
        status = "skipped_not_ready"
        launch_decision = "do_not_launch"
    else:
        status = "ready_to_launch"
        launch_decision = "launch_allowed"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "launch_decision": launch_decision,
        "launch_allowed": launch_decision == "launch_allowed",
        "launch_blockers": blockers,
        "readiness": readiness,
        "ui_prereqs": prereqs,
        "remote_command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--node", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--required-version", default=DEFAULT_REQUIRED_VERSION)
    parser.add_argument("--required-available-gb", type=float, default=DEFAULT_REQUIRED_AVAILABLE_GB)
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--proof-basename", default=DEFAULT_PROOF_BASENAME)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-tool-iterations", type=int, default=DEFAULT_MAX_TOOL_ITERATIONS)
    parser.add_argument("--cache-expect-regex", default=DEFAULT_CACHE_EXPECT_REGEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    readiness = build_readiness(
        host=args.host,
        worktree=args.worktree,
        python=args.python,
        model=args.model,
        required_version=args.required_version,
        required_available_gb=args.required_available_gb,
        connect_timeout=args.connect_timeout,
    )
    prereqs = remote_ui_prereqs(
        host=args.host,
        worktree=args.worktree,
        node=args.node,
        app=args.app,
        connect_timeout=args.connect_timeout,
    )
    command = remote_real_ui_command(
        worktree=args.worktree,
        node=args.node,
        app=args.app,
        model=args.model,
        proof_basename=args.proof_basename,
        max_tokens=args.max_tokens,
        max_tool_iterations=args.max_tool_iterations,
        cache_expect_regex=args.cache_expect_regex,
    )
    result = build_guard_result(
        readiness=readiness,
        prereqs=prereqs,
        command=command,
        dry_run=args.dry_run,
    )
    if result["launch_decision"] == "launch_allowed":
        result["remote_stdout"] = _ssh(args.host, command, connect_timeout=args.connect_timeout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.out} status={result['status']} "
        f"launch_decision={result['launch_decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

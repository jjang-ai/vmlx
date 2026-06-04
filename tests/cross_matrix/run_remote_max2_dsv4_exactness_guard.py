#!/usr/bin/env python3
"""Guarded max2 DSV4 route-mode exactness launcher.

This is the safe entrypoint for the remaining DSV4 release blocker. It checks
the current-source max2 proof worktree readiness first, refuses to launch when
version/import/model/memory gates are not clear, and records the exact command
that will run once the host is ready.
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


DEFAULT_OUT = Path("build/current-remote-max2-dsv4-exactness-guard-20260604.json")
DEFAULT_REMOTE_OUT = Path(
    "build/current-dsv4-route-mode-code-exactness-max2-proof-worktree-20260604.json"
)
DEFAULT_CASES = (
    "chat_off_rep1,chat_off_no_punct_rep1,responses_off_rep1,"
    "responses_off_no_punct_rep1"
)
DEFAULT_PORT = 8897
DEFAULT_TIMEOUT = 420
DEFAULT_REQUEST_TIMEOUT = 300


def remote_exactness_command(
    *,
    worktree: Path,
    python: Path,
    model: Path,
    remote_out: Path,
    cases: str = DEFAULT_CASES,
    port: int = DEFAULT_PORT,
    timeout: int = DEFAULT_TIMEOUT,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    min_free_gb: float = DEFAULT_REQUIRED_AVAILABLE_GB,
) -> str:
    parts = [
        "cd",
        shlex.quote(str(worktree)),
        "&&",
        shlex.quote(str(python)),
        "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
        "--python",
        shlex.quote(str(python)),
        "--model",
        shlex.quote(str(model)),
        "--port",
        str(port),
        "--timeout",
        str(timeout),
        "--request-timeout",
        str(request_timeout),
        "--cases",
        shlex.quote(cases),
        "--min-free-gb",
        str(min_free_gb),
        "--out",
        shlex.quote(str(remote_out)),
    ]
    return " ".join(parts)


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


def build_guard_result(
    *,
    readiness: dict[str, Any],
    command: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        status = "dry_run"
        launch_decision = "dry_run_no_launch"
    elif readiness.get("launch_allowed") is True:
        status = "ready_to_launch"
        launch_decision = "launch_allowed"
    else:
        status = "skipped_not_ready"
        launch_decision = "do_not_launch"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "launch_decision": launch_decision,
        "dry_run": dry_run,
        "readiness": readiness,
        "remote_command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--required-version", default=DEFAULT_REQUIRED_VERSION)
    parser.add_argument("--required-available-gb", type=float, default=DEFAULT_REQUIRED_AVAILABLE_GB)
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--remote-out", type=Path, default=DEFAULT_REMOTE_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
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
    command = remote_exactness_command(
        worktree=args.worktree,
        python=args.python,
        model=args.model,
        remote_out=args.remote_out,
        cases=args.cases,
        port=args.port,
        timeout=args.timeout,
        request_timeout=args.request_timeout,
        min_free_gb=args.required_available_gb,
    )
    result = build_guard_result(readiness=readiness, command=command, dry_run=args.dry_run)
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

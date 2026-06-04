#!/usr/bin/env python3
"""Aggregate guarded max2 DSV4 release proof.

Runs the max2 readiness check plus the guarded exactness and real-UI launchers
and writes one release-proof status artifact. The child guards own the actual
launch safety decisions; in the current memory-blocked state this command is
expected to report skipped child proofs and no DSV4 launch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.run_remote_max2_dsv4_exactness_guard import (
    DEFAULT_OUT as DEFAULT_EXACTNESS_OUT,
)
from tests.cross_matrix.run_remote_max2_dsv4_readiness import (
    DEFAULT_OUT as DEFAULT_READINESS_OUT,
)
from tests.cross_matrix.run_remote_max2_dsv4_real_ui_guard import (
    DEFAULT_OUT as DEFAULT_REAL_UI_OUT,
)


DEFAULT_OUT = Path("build/current-remote-max2-dsv4-release-proof-guard-20260604.json")


def aggregate_status(children: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statuses = {
        name: str(child.get("status") or "")
        for name, child in children.items()
        if isinstance(child, dict)
    }
    launch_decisions = {
        name: str(child.get("launch_decision") or "")
        for name, child in children.items()
        if isinstance(child, dict)
    }
    if statuses.get("readiness") == "ready" and all(
        status in {"ready_to_launch", "pass"}
        for name, status in statuses.items()
        if name != "readiness"
    ):
        status = "ready_or_pass"
    elif any(
        decision == "launch_allowed"
        for name, decision in launch_decisions.items()
        if name != "readiness"
    ):
        status = "partial_launch_attempted"
    elif any(status.startswith("skipped") or status in {"prepared_but_blocked"} for status in statuses.values()):
        status = "blocked_no_launch"
    elif any(status in {"dry_run"} for status in statuses.values()):
        status = "dry_run"
    else:
        status = "unknown"
    return {
        "status": status,
        "child_statuses": statuses,
        "child_launch_decisions": launch_decisions,
        "release_ready": status == "ready_or_pass",
    }


def _run_json_command(cmd: list[str], artifact: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    meta = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "artifact": str(artifact),
    }
    if artifact.exists():
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report malformed artifact
            payload = {"status": "artifact_read_error", "error": f"{type(exc).__name__}: {exc}"}
    else:
        payload = {"status": "artifact_missing"}
    return payload, meta


def build_commands(
    *,
    readiness_out: Path,
    exactness_out: Path,
    real_ui_out: Path,
    dry_run: bool,
) -> dict[str, list[str]]:
    python = sys.executable
    exactness_cmd = [
        python,
        "tests/cross_matrix/run_remote_max2_dsv4_exactness_guard.py",
        "--out",
        str(exactness_out),
    ]
    real_ui_cmd = [
        python,
        "tests/cross_matrix/run_remote_max2_dsv4_real_ui_guard.py",
        "--out",
        str(real_ui_out),
    ]
    if dry_run:
        exactness_cmd.append("--dry-run")
        real_ui_cmd.append("--dry-run")
    return {
        "readiness": [
            python,
            "tests/cross_matrix/run_remote_max2_dsv4_readiness.py",
            "--out",
            str(readiness_out),
        ],
        "exactness": exactness_cmd,
        "real_ui": real_ui_cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--readiness-out", type=Path, default=DEFAULT_READINESS_OUT)
    parser.add_argument("--exactness-out", type=Path, default=DEFAULT_EXACTNESS_OUT)
    parser.add_argument("--real-ui-out", type=Path, default=DEFAULT_REAL_UI_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_commands(
        readiness_out=args.readiness_out,
        exactness_out=args.exactness_out,
        real_ui_out=args.real_ui_out,
        dry_run=args.dry_run,
    )
    children: dict[str, dict[str, Any]] = {}
    command_meta: dict[str, dict[str, Any]] = {}
    artifact_by_name = {
        "readiness": args.readiness_out,
        "exactness": args.exactness_out,
        "real_ui": args.real_ui_out,
    }
    for name, cmd in commands.items():
        payload, meta = _run_json_command(cmd, artifact_by_name[name])
        children[name] = payload
        command_meta[name] = meta

    aggregate = aggregate_status(children)
    result = {
        **aggregate,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "children": children,
        "commands": command_meta,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out} status={result['status']} release_ready={result['release_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""No-heavy DSV4 readiness probe for the max2 proof worktree.

This helper exists so the remaining DSV4 release blocker is repeatable instead
of a hand-written SSH transcript. It does not launch the model. It verifies the
remote proof worktree/source import path and records whether the remote memory
state is safe enough for the later heavy real-UI/exactness proof.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HOST = "test-host.local"
DEFAULT_WORKTREE = Path("/Users/example/mlx/vllm-mlx-codex-proof-1554")
DEFAULT_PYTHON = Path("/Users/example/mlx/vllm-mlx/.venv/bin/python")
DEFAULT_MODEL = Path("/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K")
DEFAULT_REQUIRED_VERSION = "1.5.54"
DEFAULT_REQUIRED_AVAILABLE_GB = 120.0
DEFAULT_OUT = Path("build/current-remote-max2-dsv4-proof-worktree-readiness-20260604.json")
DEFAULT_CONNECT_TIMEOUT = 8

HEAVY_PROCESS_PATTERN = (
    "vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|"
    "run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|"
    "run_dsv4_default_cache_tool_loop_gate"
)


def parse_vm_stat(text: str) -> dict[str, Any]:
    page_size_match = re.search(r"page size of (\d+) bytes", text)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384
    pages: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"Pages ([^:]+):\s+([0-9]+)\.", line.strip())
        if not match:
            continue
        key = match.group(1).strip().replace(" ", "_")
        pages[key] = int(match.group(2))
    return {"page_size": page_size, "pages": pages}


def free_plus_speculative_purgeable_gb(vm_stat: dict[str, Any]) -> float:
    pages = vm_stat.get("pages") if isinstance(vm_stat.get("pages"), dict) else {}
    page_size = int(vm_stat.get("page_size") or 16384)
    total_pages = (
        int(pages.get("free", 0))
        + int(pages.get("speculative", 0))
        + int(pages.get("purgeable", 0))
    )
    return round(total_pages * page_size / (1024**3), 2)


def launch_decision(
    *,
    version_ok: bool,
    import_ok: bool,
    model_present: bool,
    available_gb: float,
    required_available_gb: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not version_ok:
        blockers.append("version_mismatch")
    if not import_ok:
        blockers.append("source_import_failed")
    if not model_present:
        blockers.append("model_missing")
    if available_gb < required_available_gb:
        blockers.append("insufficient_memory")
    return {
        "launch_allowed": not blockers,
        "launch_decision": "launch_allowed" if not blockers else "do_not_launch",
        "launch_blockers": blockers,
    }


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


def _remote_json_probe(
    *,
    host: str,
    worktree: Path,
    python: Path,
    model: Path,
    required_version: str,
    connect_timeout: int,
) -> dict[str, Any]:
    command = f"""
set -e
cd {shlex.quote(str(worktree))}
{shlex.quote(str(python))} - <<'PY'
import json
import pathlib
import subprocess
import sys

worktree = pathlib.Path({str(worktree)!r})
model = pathlib.Path({str(model)!r})
required_version = {required_version!r}

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

version = None
for line in (worktree / "pyproject.toml").read_text().splitlines():
    if line.startswith("version = "):
        version = line.split("=", 1)[1].strip().strip('"')
        break

import_error = None
import_version = None
import_file = None
try:
    import vmlx_engine
    import_version = getattr(vmlx_engine, "__version__", None)
    import_file = getattr(vmlx_engine, "__file__", None)
except Exception as exc:
    import_error = f"{{type(exc).__name__}}: {{exc}}"

payload = {{
    "head": run(["git", "log", "-1", "--oneline", "--decorate"]),
    "status_short": run(["git", "status", "-sb"]),
    "version": version,
    "required_version": required_version,
    "version_ok": version == required_version,
    "import_version": import_version,
    "import_file": import_file,
    "import_error": import_error,
    "import_ok": import_version == required_version and bool(import_file) and str(worktree) in str(import_file),
    "model_present": model.exists(),
    "model_du": run(["du", "-sh", str(model)]).split()[0] if model.exists() else None,
}}
print(json.dumps(payload))
PY
"""
    return json.loads(_ssh(host, command, connect_timeout=connect_timeout))


def build_readiness(
    *,
    host: str = DEFAULT_HOST,
    worktree: Path = DEFAULT_WORKTREE,
    python: Path = DEFAULT_PYTHON,
    model: Path = DEFAULT_MODEL,
    required_version: str = DEFAULT_REQUIRED_VERSION,
    required_available_gb: float = DEFAULT_REQUIRED_AVAILABLE_GB,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
) -> dict[str, Any]:
    source = _remote_json_probe(
        host=host,
        worktree=worktree,
        python=python,
        model=model,
        required_version=required_version,
        connect_timeout=connect_timeout,
    )
    vm_text = _ssh(host, "vm_stat", connect_timeout=connect_timeout)
    vm = parse_vm_stat(vm_text)
    available_gb = free_plus_speculative_purgeable_gb(vm)
    memory_gap_gb = round(max(0.0, required_available_gb - available_gb), 2)
    heavy_processes = _ssh(
        host,
        f"pgrep -af {shlex.quote(HEAVY_PROCESS_PATTERN)} || true",
        connect_timeout=connect_timeout,
    ).splitlines()
    top_rss = _ssh(
        host,
        "ps -axo pid,rss,command -r | head -n 12",
        connect_timeout=connect_timeout,
    ).splitlines()
    decision = launch_decision(
        version_ok=bool(source.get("version_ok")),
        import_ok=bool(source.get("import_ok")),
        model_present=bool(source.get("model_present")),
        available_gb=available_gb,
        required_available_gb=required_available_gb,
    )
    status = "ready" if decision["launch_allowed"] else "prepared_but_blocked"
    return {
        "status": status,
        "host": host,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "proof_worktree": str(worktree),
        "python": str(python),
        "model": str(model),
        "source": source,
        "memory": {
            "unit": "GiB",
            "free_plus_speculative_purgeable_gb": available_gb,
            "free_plus_speculative_purgeable_gib": available_gb,
            "required_available_gb": required_available_gb,
            "required_available_gib": required_available_gb,
            "memory_gap_gb": memory_gap_gb,
            "memory_gap_gib": memory_gap_gb,
            "vm_stat": vm,
        },
        "heavy_processes": heavy_processes,
        "top_rss": top_rss,
        **decision,
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = build_readiness(
        host=args.host,
        worktree=args.worktree,
        python=args.python,
        model=args.model,
        required_version=args.required_version,
        required_available_gb=args.required_available_gb,
        connect_timeout=args.connect_timeout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.out} status={result['status']} "
        f"launch_decision={result['launch_decision']} "
        f"available_gb={result['memory']['free_plus_speculative_purgeable_gb']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

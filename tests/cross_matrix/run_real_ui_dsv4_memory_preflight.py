#!/usr/bin/env python3
"""Refresh the no-heavy DSV4 real-UI memory preflight artifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K")
DEFAULT_OUT = Path(
    "build/current-real-ui-dsv4-memory-preflight-20260601-local-refresh.json"
)
DEFAULT_REQUIRED_AVAILABLE_GB = 120.0
DEFAULT_REQUIRED_MODEL_MARGIN_GB = 40.0
TOP_MEMORY_PROCESS_COUNT = 10
HEAVY_PROCESS_PATTERNS = (
    "vmlx_engine.cli serve",
    "mlx_lm.server",
    "run_runtime_memory_stress_probe",
    "run_decode_speed_gate",
    "live-real-ui-model-proof",
    "run_dsv4_route_mode_code_exactness",
    "run_dsv4_default_cache_tool_loop_gate",
)


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


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


def gb_from_pages(count: int, page_size: int) -> float:
    return round((count * page_size) / (1024**3), 2)


def parse_memory_pressure(text: str) -> dict[str, Any]:
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", text)
    return {
        "free_percent": int(match.group(1)) if match else None,
        "raw_tail": text.splitlines()[-20:],
    }


def psutil_memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "available_gb": round(vm.available / (1024**3), 2),
        "total_gb": round(vm.total / (1024**3), 2),
        "percent": vm.percent,
    }


def model_size_gb(model_path: Path) -> float | None:
    if not model_path.exists():
        return None
    total = 0
    for path in model_path.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return round(total / (1024**3), 2)


def parse_top_memory_processes(text: str, limit: int = TOP_MEMORY_PROCESS_COUNT) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("PID "):
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        processes.append(
            {
                "pid": pid,
                "rss_gb": round(rss_kb / (1024**2), 2),
                "command": parts[2],
            }
        )
    processes.sort(key=lambda item: item["rss_gb"], reverse=True)
    return processes[:limit]


def parse_active_heavy_processes(text: str) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if "pgrep -af" in command:
            continue
        if not any(pattern in command for pattern in HEAVY_PROCESS_PATTERNS):
            continue
        processes.append({"pid": pid, "command": command})
    return processes


def build_preflight(
    *,
    model_path: Path,
    required_available_gb: float,
    vm_stat_text: str | None = None,
    memory_pressure_text: str | None = None,
    top_processes_text: str | None = None,
    active_processes_text: str | None = None,
) -> dict[str, Any]:
    vm = parse_vm_stat(vm_stat_text if vm_stat_text is not None else _run_text(["vm_stat"]))
    page_size = int(vm["page_size"])
    pages = vm["pages"]
    free_gb = gb_from_pages(int(pages.get("free", 0)), page_size)
    speculative_gb = gb_from_pages(int(pages.get("speculative", 0)), page_size)
    purgeable_gb = gb_from_pages(int(pages.get("purgeable", 0)), page_size)
    inactive_gb = gb_from_pages(int(pages.get("inactive", 0)), page_size)
    free_plus_speculative_purgeable = round(free_gb + speculative_gb + purgeable_gb, 2)
    psutil_snapshot = psutil_memory_snapshot()
    memory_pressure_command = "memory_pressure"
    if memory_pressure_text is None:
        try:
            memory_pressure_text = _run_text([memory_pressure_command])
        except Exception:
            memory_pressure_text = ""
    memory_pressure_snapshot = parse_memory_pressure(memory_pressure_text)
    psutil_available_gb = psutil_snapshot.get("available_gb")
    psutil_memory_gap_gb = (
        round(max(0.0, required_available_gb - float(psutil_available_gb)), 2)
        if isinstance(psutil_available_gb, (int, float))
        else None
    )
    size_gb = model_size_gb(model_path)
    safety_margin_gb = (
        round(required_available_gb - size_gb, 2) if size_gb is not None else None
    )
    floor_valid = (
        size_gb is not None
        and safety_margin_gb is not None
        and safety_margin_gb >= DEFAULT_REQUIRED_MODEL_MARGIN_GB
    )
    active_process_command = (
        "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|"
        "run_runtime_memory_stress_probe|run_decode_speed_gate|"
        "live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|"
        "run_dsv4_default_cache_tool_loop_gate'"
    )
    if active_processes_text is None:
        try:
            active_processes_text = _run_text(["sh", "-c", active_process_command])
        except subprocess.CalledProcessError:
            active_processes_text = ""
        except Exception:
            active_processes_text = ""
    active_heavy_processes = parse_active_heavy_processes(active_processes_text)
    if not floor_valid:
        status = "invalid_preflight_floor"
    elif active_heavy_processes:
        status = "skipped_active_heavy_process"
    elif free_plus_speculative_purgeable >= required_available_gb:
        status = "ready_to_launch"
    else:
        status = "skipped_insufficient_memory"
    memory_gap_gb = round(max(0.0, required_available_gb - free_plus_speculative_purgeable), 2)
    top_process_command = "ps -axo pid,rss,command -r | head -n 11"
    if top_processes_text is None:
        try:
            top_processes_text = _run_text(["sh", "-c", top_process_command])
        except Exception:
            top_processes_text = ""
    return {
        "status": status,
        "reason": (
            "local memory meets DSV4 real-UI preflight floor; run the live row instead of leaving dsv4 missing"
            if status == "ready_to_launch"
            else "another heavy model/proof process is active; do not launch DSV4 concurrently"
            if status == "skipped_active_heavy_process"
            else "DSV4 preflight floor is below model-size plus safety headroom; do not use this artifact as a launch/skip decision"
            if status == "invalid_preflight_floor"
            else "local memory below DSV4 real-UI preflight floor; leave matrix missing dsv4 until enough headroom is available"
        ),
        "did_not_launch": True,
        "launch_decision": "launch_allowed" if status == "ready_to_launch" else "do_not_launch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_size_gb": size_gb,
        "required_available_gb": required_available_gb,
        "required_model_margin_gb": DEFAULT_REQUIRED_MODEL_MARGIN_GB,
        "safety_margin_gb": safety_margin_gb,
        "floor_valid": floor_valid,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "free_plus_speculative_purgeable_gb": free_plus_speculative_purgeable,
        "memory_gap_gb": memory_gap_gb,
        "inactive_file_cache_gb": inactive_gb,
        "psutil_available_gb": psutil_available_gb,
        "psutil_total_gb": psutil_snapshot.get("total_gb"),
        "psutil_memory_percent": psutil_snapshot.get("percent"),
        "psutil_memory_gap_gb": psutil_memory_gap_gb,
        "memory_pressure_free_percent": memory_pressure_snapshot.get("free_percent"),
        "memory_pressure_raw_tail": memory_pressure_snapshot.get("raw_tail"),
        "top_memory_processes": parse_top_memory_processes(top_processes_text),
        "active_heavy_processes": active_heavy_processes,
        "active_heavy_process_count": len(active_heavy_processes),
        "launch_blockers": [
            blocker
            for blocker, blocked in (
                ("invalid_preflight_floor", status == "invalid_preflight_floor"),
                ("active_heavy_process", bool(active_heavy_processes)),
                ("insufficient_memory", free_plus_speculative_purgeable < required_available_gb),
            )
            if blocked
        ],
        "memory_breakdown_gb": {
            "free": free_gb,
            "speculative": speculative_gb,
            "purgeable": purgeable_gb,
            "inactive": inactive_gb,
        },
        "commands": {
            "memory": "vm_stat",
            "memory_pressure": memory_pressure_command,
            "model_size": f"walk {model_path}",
            "top_memory_processes": top_process_command,
            "active_heavy_processes": active_process_command,
        },
    }


def write_preflight(
    *,
    model_path: Path,
    out_path: Path,
    required_available_gb: float,
) -> dict[str, Any]:
    payload = build_preflight(
        model_path=model_path,
        required_available_gb=required_available_gb,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--required-available-gb", type=float, default=DEFAULT_REQUIRED_AVAILABLE_GB)
    args = parser.parse_args()
    payload = write_preflight(
        model_path=args.model_path,
        out_path=args.out,
        required_available_gb=args.required_available_gb,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": payload["status"],
                "free_plus_speculative_purgeable_gb": payload[
                    "free_plus_speculative_purgeable_gb"
                ],
                "required_available_gb": payload["required_available_gb"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

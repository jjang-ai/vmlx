#!/usr/bin/env python3
"""Installed-app runtime parity audit for issues #175-#177.

This is intentionally no-heavy: it proves the packaged runtime carries the
memory/cache fixes and telemetry surfaces without loading a large model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-issue175-177-installed-runtime-audit-20260601-local-refresh.json"
)
INSTALLED_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)


def _run_installed_python(code: str) -> dict[str, Any]:
    if not INSTALLED_PYTHON.exists():
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": f"missing installed python: {INSTALLED_PYTHON}",
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [str(INSTALLED_PYTHON), "-B", "-s", "-c", code],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _load_json_line(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("returncode") != 0:
        return {}
    stdout = str(result.get("stdout") or "")
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def build_audit(root: Path) -> dict[str, Any]:
    del root  # Installed-app import isolation is the point of this gate.
    code = r'''
import ast
import json
from pathlib import Path

from vmlx_engine.mlx_memory import clear_mlx_memory_cache
import vmlx_engine.mlx_memory as mlx_memory
import vmlx_engine.mllm_batch_generator as mllm_batch_generator
import vmlx_engine.mllm_scheduler as mllm_scheduler
import vmlx_engine.paged_cache as paged_cache
import vmlx_engine.prefix_cache as prefix_cache
import vmlx_engine.scheduler as scheduler
import vmlx_engine.server as server

runtime_modules = [
    "vmlx_engine.engine.simple",
    "vmlx_engine.image_gen",
    "vmlx_engine.mllm_scheduler",
    "vmlx_engine.models.mllm",
    "vmlx_engine.reranker",
    "vmlx_engine.scheduler",
    "vmlx_engine.server",
]

offenders = []
for module_name in runtime_modules:
    module = __import__(module_name, fromlist=["__file__"])
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "clear_memory_cache":
            offenders.append(f"{module_name}:{node.lineno}")

memory_method = clear_mlx_memory_cache()
prefix_source = Path(prefix_cache.__file__).read_text(encoding="utf-8")
paged_source = Path(paged_cache.__file__).read_text(encoding="utf-8")
scheduler_source = Path(scheduler.__file__).read_text(encoding="utf-8")
mllm_batch_source = Path(mllm_batch_generator.__file__).read_text(encoding="utf-8")
mllm_scheduler_source = Path(mllm_scheduler.__file__).read_text(encoding="utf-8")
server_source = Path(server.__file__).read_text(encoding="utf-8")
soft_start = server_source.index("async def admin_soft_sleep")
deep_start = server_source.index("async def admin_deep_sleep")
wake_start = server_source.index("async def admin_wake")
soft_sleep_source = server_source[soft_start:deep_start]
deep_sleep_source = server_source[deep_start:wake_start]

payload = {
    "mlx_memory_file": mlx_memory.__file__,
    "paged_cache_file": paged_cache.__file__,
    "prefix_cache_file": prefix_cache.__file__,
    "scheduler_file": scheduler.__file__,
    "server_file": server.__file__,
    "memory_method": memory_method,
    "raw_clear_memory_cache_offenders": offenders,
    "promoted_disk_block_markers": {
        "paged_has_disk_marker": "cache_data_from_disk" in paged_source,
        "prefix_drops_cache_data": "block.cache_data = None" in prefix_source,
        "prefix_drops_disk_marker": "block.cache_data_from_disk = False" in prefix_source,
        "block_disk_store_has_readable_probe": "def has_block" in Path(__import__("vmlx_engine.block_disk_store", fromlist=["__file__"]).__file__).read_text(encoding="utf-8"),
        "prefix_tracks_l2_readable_blocks": "l2_readable_block_ids" in prefix_source,
        "prefix_checks_disk_has_block": "has_block(block.block_hash)" in prefix_source,
    },
    "cache_selection_markers": {
        "scheduler_last_cache_selection": "last_cache_selection" in scheduler_source,
        "scheduler_last_cache_execution": "last_cache_execution" in scheduler_source,
        "scheduler_worker_cache_seconds": "total_worker_cache_seconds" in scheduler_source,
        "scheduler_reconstruction_seconds": "reconstruction_seconds" in scheduler_source,
        "scheduler_dequantization_seconds": "dequantization_seconds" in scheduler_source,
        "mllm_last_cache_execution": "last_cache_execution" in mllm_batch_source,
        "mllm_worker_cache_seconds": "total_worker_cache_seconds" in mllm_batch_source,
        "mllm_scheduler_projects_cache_execution": "last_cache_execution" in mllm_scheduler_source,
        "scheduler_cold_paged_reason": "cold_paged_reconstruction_cost" in scheduler_source,
        "server_health_last_cache_selection": "last_cache_selection" in server_source,
        "server_health_last_cache_execution": "last_cache_execution" in server_source,
    },
    "admin_sleep_memory_clear_markers": {
        "soft_sleep_uses_helper": "clear_mlx_memory_cache(log=logger)" in soft_sleep_source,
        "deep_sleep_uses_helper": "clear_mlx_memory_cache(log=logger)" in deep_sleep_source,
        "soft_sleep_avoids_top_level_clear_cache": "mx.clear_cache()" not in soft_sleep_source,
        "deep_sleep_avoids_top_level_clear_cache": "mx.clear_cache()" not in deep_sleep_source,
        "soft_sleep_avoids_direct_metal_clear_cache": "mx.metal.clear_cache()" not in soft_sleep_source,
        "deep_sleep_avoids_direct_metal_clear_cache": "mx.metal.clear_cache()" not in deep_sleep_source,
    },
}
print(json.dumps(payload, sort_keys=True))
'''
    result = _run_installed_python(code)
    payload = _load_json_line(result)
    installed_prefix = "/Applications/vMLX.app/Contents/Resources/bundled-python"

    files = [
        str(payload.get("mlx_memory_file", "")),
        str(payload.get("paged_cache_file", "")),
        str(payload.get("prefix_cache_file", "")),
        str(payload.get("scheduler_file", "")),
        str(payload.get("server_file", "")),
    ]
    installed_files = all(path.startswith(installed_prefix) for path in files)
    disk_markers = payload.get("promoted_disk_block_markers")
    if not isinstance(disk_markers, dict):
        disk_markers = {}
    cache_markers = payload.get("cache_selection_markers")
    if not isinstance(cache_markers, dict):
        cache_markers = {}
    admin_sleep_markers = payload.get("admin_sleep_memory_clear_markers")
    if not isinstance(admin_sleep_markers, dict):
        admin_sleep_markers = {}

    checks = {
        "installed_python_exists": INSTALLED_PYTHON.exists(),
        "memory_clear_helper_imports_from_installed_app": installed_files,
        "memory_clear_uses_available_mlx_api": payload.get("memory_method")
        in {"mx.metal.clear_cache", "mx.clear_cache"},
        "runtime_paths_avoid_removed_clear_memory_cache": payload.get(
            "raw_clear_memory_cache_offenders"
        )
        == [],
        "admin_sleep_uses_memory_clear_helper": all(
            admin_sleep_markers.get(name) is True
            for name in (
                "soft_sleep_uses_helper",
                "deep_sleep_uses_helper",
                "soft_sleep_avoids_top_level_clear_cache",
                "deep_sleep_avoids_top_level_clear_cache",
                "soft_sleep_avoids_direct_metal_clear_cache",
                "deep_sleep_avoids_direct_metal_clear_cache",
            )
        ),
        "promoted_disk_blocks_drop_parent_mirror": all(
            disk_markers.get(name) is True
            for name in (
                "paged_has_disk_marker",
                "prefix_drops_cache_data",
                "prefix_drops_disk_marker",
            )
        ),
        "l2_readable_write_through_blocks_drop_parent_mirror": all(
            disk_markers.get(name) is True
            for name in (
                "block_disk_store_has_readable_probe",
                "prefix_tracks_l2_readable_blocks",
                "prefix_checks_disk_has_block",
                "prefix_drops_cache_data",
                "prefix_drops_disk_marker",
            )
        ),
        "cache_selection_telemetry_installed": all(
            cache_markers.get(name) is True
            for name in (
                "scheduler_last_cache_selection",
                "scheduler_cold_paged_reason",
                "server_health_last_cache_selection",
            )
        ),
        "cache_execution_timing_telemetry_installed": all(
            cache_markers.get(name) is True
            for name in (
                "scheduler_last_cache_execution",
                "scheduler_worker_cache_seconds",
                "scheduler_reconstruction_seconds",
                "scheduler_dequantization_seconds",
                "mllm_last_cache_execution",
                "mllm_worker_cache_seconds",
                "mllm_scheduler_projects_cache_execution",
                "server_health_last_cache_execution",
            )
        ),
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "artifact": "",
        "status": "pass" if not failures else "fail",
        "installed_python": str(INSTALLED_PYTHON),
        "checks": checks,
        "failures": failures,
        "import_returncode": result.get("returncode"),
        "import_stderr_tail": str(result.get("stderr") or "").splitlines()[-20:],
        "payload": payload,
        "boundary": (
            "No-heavy installed-app proof for #175/#176/#177. Live model stress "
            "still remains separate from this packaged runtime parity proof."
        ),
    }


def write_audit(root: Path, out: Path) -> dict[str, Any]:
    audit = build_audit(root)
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    audit = write_audit(args.root, args.out)
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

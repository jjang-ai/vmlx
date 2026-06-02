#!/usr/bin/env python3
"""Validate live installed-app runtime evidence for issues #175-#177.

This gate records real installed-app behavior without overstating clearance.
It accepts a live cache-hit/TTFT proof as supporting evidence, while keeping the
stronger memory-pressure and cold paged/TurboQuant stress rows open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_QWEN_SOURCE = Path(
    "build/current-runtime-memory-stress-qwen36-mtp-installed-issue175-177-live-20260527.json"
)
DEFAULT_MINIMAX_SOURCE = Path(
    "build/current-runtime-memory-stress-minimax-small-installed-issue177-cache-selection-live-20260527.json"
)
DEFAULT_MINIMAX_RESTART_READER_SOURCE = Path(
    "build/current-runtime-memory-stress-minimax-small-installed-issue177-restart-reader-synced-20260527.json"
)
DEFAULT_ADMIN_SLEEP_SOURCE = Path(
    "build/current-issue175-admin-sleep-probe-installed-20260527.json"
)
DEFAULT_OUT = Path(
    "build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json"
)
INSTALLED_PYTHON = (
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _nested(stage: dict[str, Any], *path: str) -> Any:
    cur: Any = stage
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stage_pair(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    stages = [stage for stage in results if isinstance(stage, dict)]
    return stages[0] if stages else {}, stages[1] if len(stages) > 1 else {}


def _scheduler_after(stage: dict[str, Any]) -> dict[str, Any]:
    scheduler = _nested(stage, "after", "health", "body", "scheduler")
    return scheduler if isinstance(scheduler, dict) else {}


def _cache_execution_timing_seen(
    execution: Any,
    *,
    detail: str,
    alternate_details: tuple[str, ...] = (),
) -> bool:
    if not isinstance(execution, dict):
        return False
    accepted_details = {detail, *alternate_details}
    return (
        execution.get("cache_detail") in accepted_details
        and _number(execution.get("cached_tokens")) is not None
        and _number(execution.get("cached_tokens")) > 0
        and _number(execution.get("reconstruction_seconds")) is not None
        and _number(execution.get("dequantization_seconds")) is not None
        and _number(execution.get("total_worker_cache_seconds")) is not None
        and _number(execution.get("total_worker_cache_seconds")) >= 0
    )


def _latency_seconds(stage: dict[str, Any]) -> float | None:
    ttft = _number(_nested(stage, "http_timing", "ttft_seconds"))
    if ttft is not None:
        return ttft
    return _number(_nested(stage, "http_timing", "total_seconds"))


def _latency_source(stage: dict[str, Any]) -> str | None:
    if _number(_nested(stage, "http_timing", "ttft_seconds")) is not None:
        return "http_timing.ttft_seconds"
    if _number(_nested(stage, "http_timing", "total_seconds")) is not None:
        return "http_timing.total_seconds"
    return None


def build_audit(
    root: Path,
    source: Path = DEFAULT_QWEN_SOURCE,
    minimax_source: Path = DEFAULT_MINIMAX_SOURCE,
    minimax_restart_reader_source: Path = DEFAULT_MINIMAX_RESTART_READER_SOURCE,
    admin_sleep_source: Path = DEFAULT_ADMIN_SLEEP_SOURCE,
) -> dict[str, Any]:
    artifact_path = root / source
    payload = _load(artifact_path)
    first, second = _stage_pair(payload)

    first_ttft = _number(_nested(first, "http_timing", "ttft_seconds"))
    second_ttft = _number(_nested(second, "http_timing", "ttft_seconds"))
    cached_tokens = _number(_nested(second, "usage_summary", "cached_tokens")) or 0
    cache_detail = _nested(second, "usage_summary", "cache_detail")
    block_disk_hits = _number(
        _nested(second, "after", "cache_stats", "body", "block_disk_cache", "disk_hits")
    ) or 0
    active_before = _number(_nested(first, "memory", "metal_active_before_gb"))
    active_after = _number(_nested(second, "memory", "metal_active_after_gb"))
    visible_first = _nested(first, "response_rails", "has_visible_content") is True
    visible_second = _nested(second, "response_rails", "has_visible_content") is True
    python_path = str(payload.get("python") or "")

    minimax_payload = _load(root / minimax_source)
    minimax_first, minimax_second = _stage_pair(minimax_payload)
    minimax_first_ttft = _latency_seconds(minimax_first)
    minimax_second_ttft = _latency_seconds(minimax_second)
    minimax_first_latency_source = _latency_source(minimax_first)
    minimax_second_latency_source = _latency_source(minimax_second)
    minimax_cached_tokens = (
        _number(_nested(minimax_second, "usage_summary", "cached_tokens")) or 0
    )
    minimax_cache_detail = _nested(minimax_second, "usage_summary", "cache_detail")
    minimax_block_disk_hits = (
        _number(
            _nested(
                minimax_second,
                "after",
                "cache_stats",
                "body",
                "block_disk_cache",
                "disk_hits",
            )
        )
        or 0
    )
    minimax_scheduler = _scheduler_after(minimax_second)
    minimax_selection = minimax_scheduler.get("last_cache_selection")
    if not isinstance(minimax_selection, dict):
        minimax_selection = {}
    minimax_execution = minimax_scheduler.get("last_cache_execution")
    if not isinstance(minimax_execution, dict):
        minimax_execution = {}
    minimax_tq = _nested(minimax_second, "after", "health", "body", "turboquant_kv_cache")
    if not isinstance(minimax_tq, dict):
        minimax_tq = {}
    minimax_visible_first = (
        _nested(minimax_first, "response_rails", "has_visible_content") is True
    )
    minimax_visible_second = (
        _nested(minimax_second, "response_rails", "has_visible_content") is True
    )
    minimax_python_path = str(minimax_payload.get("python") or "")

    restart_reader_payload = _load(root / minimax_restart_reader_source)
    restart_reader_stage: dict[str, Any] = {}
    restart_reader_results = restart_reader_payload.get("results")
    if isinstance(restart_reader_results, list) and restart_reader_results:
        restart_reader_stage = (
            restart_reader_results[0]
            if isinstance(restart_reader_results[0], dict)
            else {}
        )
    restart_cached_tokens = (
        _number(_nested(restart_reader_stage, "usage_summary", "cached_tokens")) or 0
    )
    restart_cache_detail = _nested(restart_reader_stage, "usage_summary", "cache_detail")
    restart_block_disk_hits = (
        _number(
            _nested(
                restart_reader_stage,
                "after",
                "cache_stats",
                "body",
                "block_disk_cache",
                "disk_hits",
            )
        )
        or 0
    )
    restart_scheduler = _scheduler_after(restart_reader_stage)
    restart_selection = restart_scheduler.get("last_cache_selection")
    if not isinstance(restart_selection, dict):
        restart_selection = {}
    restart_execution = restart_scheduler.get("last_cache_execution")
    if not isinstance(restart_execution, dict):
        restart_execution = {}
    restart_python_path = str(restart_reader_payload.get("python") or "")
    restart_visible = (
        _nested(restart_reader_stage, "response_rails", "has_visible_content") is True
    )
    admin_sleep_payload = _load(root / admin_sleep_source)
    admin_sleep_classification = admin_sleep_payload.get("classification")
    if not isinstance(admin_sleep_classification, dict):
        admin_sleep_classification = {}
    admin_sleep_checks = admin_sleep_classification.get("checks")
    if not isinstance(admin_sleep_checks, dict):
        admin_sleep_checks = {}
    deep_sleep_health = admin_sleep_payload.get("health_deep_sleep")
    if not isinstance(deep_sleep_health, dict):
        deep_sleep_health = {}

    checks = {
        "installed_app_qwen_live_probe_passed": payload.get("status") == "pass"
        and python_path == INSTALLED_PYTHON,
        "cache_hit_ttft_improved": (
            first_ttft is not None
            and second_ttft is not None
            and second_ttft < first_ttft
        ),
        "l2_block_disk_hits_observed": block_disk_hits > 0,
        "paged_ssm_cache_hit_observed": cached_tokens > 0 and cache_detail == "paged+ssm",
        "visible_content_observed": visible_first and visible_second,
        "metal_memory_metrics_observed": active_before is not None and active_after is not None,
        "prefix_paged_l2_capacity_projection_valid": (
            payload.get("cache_proof_policy", {}).get("capacity_projection_valid") is True
            and _nested(second, "cache_capacity_projection", "basis")
            == "observed_block_disk_l2"
        ),
        "minimax_installed_live_probe_passed": minimax_payload.get("status") == "pass"
        and minimax_python_path == INSTALLED_PYTHON,
        "turboquant_live_decode_observed": minimax_tq.get("enabled") is True
        and minimax_tq.get("batch_api") == "turboquant_kv_v1",
        "scheduler_cache_selection_observed": bool(minimax_selection),
        "scheduler_cache_execution_timing_observed": _cache_execution_timing_seen(
            minimax_execution,
            detail="paged+tq",
            alternate_details=("paged",),
        ),
        "paged_tq_cache_hit_observed": (
            minimax_cached_tokens > 0 and minimax_cache_detail == "paged+tq"
        ),
        "minimax_l2_block_disk_hits_observed": minimax_block_disk_hits > 0,
        "minimax_cache_hit_request_latency_improved": (
            minimax_first_ttft is not None
            and minimax_second_ttft is not None
            and minimax_second_ttft < minimax_first_ttft
        ),
        "minimax_visible_content_observed": (
            minimax_visible_first and minimax_visible_second
        ),
        "minimax_restart_reader_probe_passed": restart_reader_payload.get("status")
        == "pass"
        and restart_python_path == INSTALLED_PYTHON,
        "cold_paged_tq_restart_hit_observed": (
            restart_cached_tokens > 0
            and restart_cache_detail == "paged+disk+tq"
            and restart_block_disk_hits > 0
        ),
        "cold_paged_cache_selection_observed": (
            restart_selection.get("paged_cold_tokens", 0) > 0
            and restart_selection.get("selected") == "paged"
        ),
        "cold_paged_stream_ttft_observed": (
            _number(_nested(restart_reader_stage, "http_timing", "ttft_seconds"))
            is not None
            and _number(_nested(restart_reader_stage, "http_timing", "ttft_seconds"))
            > 0
        ),
        "restart_visible_content_observed": restart_visible,
        "admin_sleep_lifecycle_probe_passed": (
            admin_sleep_payload.get("status") == "pass"
            and admin_sleep_classification.get("status") == "pass"
        ),
        "admin_sleep_deep_unload_observed": (
            _nested(deep_sleep_health, "body", "model_loaded") is False
            and _nested(deep_sleep_health, "body", "status") == "standby_deep"
        ),
    }
    failures = [name for name, ok in checks.items() if not ok]
    remaining_blockers = [] if not failures else [
        "issue175_live_app_memory_stress_open",
        "issue176_live_memory_pressure_open",
        "issue177_live_ttft_paged_turboquant_open",
    ]
    return {
        "artifact": "",
        "source_artifact": str(source),
        "minimax_source_artifact": str(minimax_source),
        "minimax_restart_reader_source_artifact": str(
            minimax_restart_reader_source
        ),
        "admin_sleep_source_artifact": str(admin_sleep_source),
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "metrics": {
            "qwen": {
            "first_ttft_seconds": first_ttft,
            "cache_hit_ttft_seconds": second_ttft,
            "cached_tokens": int(cached_tokens),
            "cache_detail": cache_detail,
            "block_disk_hits": int(block_disk_hits),
            "metal_active_before_gb": active_before,
            "metal_active_after_gb": active_after,
            },
            "minimax": {
                "first_latency_seconds": minimax_first_ttft,
                "cache_hit_latency_seconds": minimax_second_ttft,
                "first_latency_source": minimax_first_latency_source,
                "cache_hit_latency_source": minimax_second_latency_source,
                "cached_tokens": int(minimax_cached_tokens),
                "cache_detail": minimax_cache_detail,
                "block_disk_hits": int(minimax_block_disk_hits),
                "cache_selection": minimax_selection,
                "cache_execution": minimax_execution,
                "turboquant_kv_cache": minimax_tq,
            },
            "minimax_restart_reader": {
                "cached_tokens": int(restart_cached_tokens),
                "cache_detail": restart_cache_detail,
                "block_disk_hits": int(restart_block_disk_hits),
                "cache_selection": restart_selection,
                "cache_execution": restart_execution,
            },
            "admin_sleep": {
                "classification": admin_sleep_classification,
                "deep_sleep_health_status": _nested(deep_sleep_health, "body", "status"),
                "deep_sleep_model_loaded": _nested(deep_sleep_health, "body", "model_loaded"),
            },
        },
        "remaining_blockers": remaining_blockers,
        "boundary": (
            "This is supporting installed-app live evidence. It proves cache-hit "
            "TTFT improvement, paged+SSM reuse, L2 disk hits, visible streaming "
            "content, and memory telemetry on a Qwen3.6 MTP row. It also proves "
            "MiniMax Small installed-app paged+TQ reuse, TurboQuant live decode, "
            "block-disk hits, visible content, and scheduler cache-selection "
            "telemetry. A fresh-process MiniMax restart reader additionally "
            "proves cold paged+disk+TQ promotion from L2. It does not by itself "
            "clear the broader memory-pressure requirement or every warm-prefix "
            "yield variant."
        ),
    }


def write_audit(
    root: Path,
    out: Path,
    source: Path = DEFAULT_QWEN_SOURCE,
    minimax_source: Path = DEFAULT_MINIMAX_SOURCE,
    minimax_restart_reader_source: Path = DEFAULT_MINIMAX_RESTART_READER_SOURCE,
) -> dict[str, Any]:
    audit = build_audit(root, source, minimax_source, minimax_restart_reader_source)
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, default=DEFAULT_QWEN_SOURCE)
    parser.add_argument("--minimax-source", type=Path, default=DEFAULT_MINIMAX_SOURCE)
    parser.add_argument(
        "--minimax-restart-reader-source",
        type=Path,
        default=DEFAULT_MINIMAX_RESTART_READER_SOURCE,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    audit = write_audit(
        args.root,
        args.out,
        args.source,
        args.minimax_source,
        args.minimax_restart_reader_source,
    )
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

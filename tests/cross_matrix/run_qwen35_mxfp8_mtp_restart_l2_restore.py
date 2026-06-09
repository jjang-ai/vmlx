#!/usr/bin/env python3
"""Live Qwen3.6 35B MXFP8-MTP fresh-process L2 restore gate.

This runner writes the two-phase schema consumed by the full release objective
checklist. Phase 1 starts a Qwen35 server, sends a short cacheable chat request,
and verifies block+SSM L2 state is written. Phase 2 starts a fresh process with
the same cache dir and verifies the same prompt is served from disk-backed
hybrid SSM cache while native MTP remains active.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.cross_matrix.run_qwen27_mxfp4_mtp_api_parity import (  # noqa: E402
    _as_int,
    build_command,
    chat_cached_tokens,
    chat_payload,
    memory_preflight,
    post_json,
    resource_snapshot,
    start_server,
    stop_server,
)
from tests.cross_matrix.run_n2_chat_cache_gate import get_json  # noqa: E402


DEFAULT_MODEL = Path("/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP")
DEFAULT_OUT = (
    REPO
    / "build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json"
)
DEFAULT_CACHE_DIR = (
    REPO / "build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/block-cache"
)


def _get(data: Any, *keys: str, default: Any = None) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _cache_stats_from_health(health: dict[str, Any] | None) -> dict[str, Any]:
    cache = health.get("cache") if isinstance(health, dict) else {}
    cache = cache if isinstance(cache, dict) else {}
    ssm = cache.get("ssm_companion") if isinstance(cache.get("ssm_companion"), dict) else {}
    return {
        "block_disk_cache": (
            cache.get("block_disk_cache")
            if isinstance(cache.get("block_disk_cache"), dict)
            else {}
        ),
        "ssm_companion": {
            "disk": ssm.get("disk") if isinstance(ssm.get("disk"), dict) else {}
        },
    }


def _prompt_details(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details")
    return details if isinstance(details, dict) else {}


def _phase_response(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    return {
        "code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "error": response.get("error"),
        "usage": body.get("usage") if isinstance(body, dict) else None,
    }


def _phase_summary(
    *,
    label: str,
    health_initial: dict[str, Any],
    response: dict[str, Any],
    health_after: dict[str, Any],
    server_log: Path,
    server_exit: int | None,
) -> dict[str, Any]:
    cache_stats = _cache_stats_from_health(health_after)
    details = _prompt_details(response)
    return {
        "label": label,
        "server_log": str(server_log),
        "server_exit": server_exit,
        "health_initial": health_initial,
        "health": health_after,
        "response": _phase_response(response),
        "usage": {"prompt_tokens_details": details},
        "cache_hit_tokens": details.get("cached_tokens"),
        "cache_stats": cache_stats,
        "block_disk_cache": cache_stats["block_disk_cache"],
        "ssm_companion_disk": cache_stats["ssm_companion"]["disk"],
        "native_cache": health_after.get("native_cache") if isinstance(health_after, dict) else {},
        "mtp": health_after.get("mtp") if isinstance(health_after, dict) else {},
    }


def _run_phase(args: argparse.Namespace, *, label: str, log_path: Path) -> dict[str, Any]:
    proc = None
    health_initial: dict[str, Any] = {}
    health_after: dict[str, Any] = {}
    response: dict[str, Any] = {}
    server_exit: int | None = None
    try:
        proc, health_initial = start_server(args, log_path)
        response = post_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            chat_payload(args),
            args.request_timeout_s,
        )
        health_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=5)
    finally:
        server_exit = stop_server(proc)
    return _phase_summary(
        label=label,
        health_initial=health_initial,
        response=response,
        health_after=health_after,
        server_log=log_path,
        server_exit=server_exit,
    )


def _native_hybrid_ok(native: dict[str, Any]) -> bool:
    return bool(
        native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
    )


def _mtp_active(mtp: dict[str, Any]) -> bool:
    return bool(
        mtp.get("runtime_active") is True
        and mtp.get("status") == "native_runtime_active"
        and _as_int(mtp.get("effective_depth")) > 0
    )


def _passed(summary: dict[str, Any]) -> bool:
    phase1 = _get(summary, "phases", "phase1", default={})
    phase2 = _get(summary, "phases", "phase2", default={})
    phase2_detail = _get(phase2, "usage", "prompt_tokens_details", default={}) or {}
    return bool(
        _get(phase1, "response", "code") == 200
        and _get(phase2, "response", "code") == 200
        and _as_int(_get(phase1, "block_disk_cache", "disk_writes")) > 0
        and _as_int(_get(phase1, "ssm_companion_disk", "stores")) > 0
        and _as_int(phase2_detail.get("cached_tokens")) > 0
        and "disk" in str(phase2_detail.get("cache_detail") or "")
        and _as_int(_get(phase2, "block_disk_cache", "disk_hits")) > 0
        and _as_int(_get(phase2, "ssm_companion_disk", "hits")) > 0
        and _native_hybrid_ok(_get(phase2, "native_cache", default={}))
        and _mtp_active(_get(phase2, "mtp", default={}))
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        return {"status": "skipped", "reason": "model_missing", "model": str(args.model)}
    preflight = memory_preflight(args.min_available_gb)
    if preflight is not None:
        preflight.update({"model": str(args.model), "artifact": str(args.out)})
        return preflight

    telemetry = [resource_snapshot("before_phase1")]
    phase1 = _run_phase(
        args,
        label="phase1_store",
        log_path=args.out.with_name("phase1_server.log"),
    )
    telemetry.append(resource_snapshot("after_phase1"))
    phase2 = _run_phase(
        args,
        label="phase2_fresh_process_restore",
        log_path=args.out.with_name("phase2_server.log"),
    )
    telemetry.append(resource_snapshot("after_phase2"))
    summary = {
        "schema": "vmlx-qwen35-restart-l2-restore-v1",
        "status": "fail",
        "artifact": str(args.out),
        "model": str(args.model),
        "served_model_name": args.served_model_name,
        "cache_dir": str(args.cache_dir),
        "cmd": build_command(args),
        "phases": {"phase1": phase1, "phase2": phase2},
        "telemetry": telemetry,
        "checks": {},
    }
    summary["checks"] = {
        "phase1_http_ok": _get(phase1, "response", "code") == 200,
        "phase1_block_disk_writes": _as_int(_get(phase1, "block_disk_cache", "disk_writes")) > 0,
        "phase1_ssm_stores": _as_int(_get(phase1, "ssm_companion_disk", "stores")) > 0,
        "phase2_http_ok": _get(phase2, "response", "code") == 200,
        "phase2_disk_cache_detail": "disk"
        in str(_get(phase2, "usage", "prompt_tokens_details", "cache_detail") or ""),
        "phase2_cached_tokens": _as_int(
            _get(phase2, "usage", "prompt_tokens_details", "cached_tokens")
        )
        > 0,
        "phase2_block_disk_hits": _as_int(_get(phase2, "block_disk_cache", "disk_hits")) > 0,
        "phase2_ssm_hits": _as_int(_get(phase2, "ssm_companion_disk", "hits")) > 0,
        "phase2_native_cache": _native_hybrid_ok(_get(phase2, "native_cache", default={})),
        "phase2_mtp_active": _mtp_active(_get(phase2, "mtp", default={})),
    }
    summary["status"] = "pass" if _passed(summary) else "fail"
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8906)
    parser.add_argument("--served-model-name", default="qwen3.6-35b-mxfp8-mtp-restart-l2")
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=240)
    parser.add_argument("--min-available-gb", type=float, default=72.0)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--completion-batch-size", type=int, default=128)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=8192)
    parser.add_argument("--server-max-tokens", type=int, default=64)
    parser.add_argument("--paged-cache-block-size", type=int, default=64)
    parser.add_argument("--max-cache-blocks", type=int, default=1000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=4.0)
    parser.add_argument("--cache-prompt", default="Reply exactly ACK.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    summary = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary.get("status"), "artifact": str(args.out)}, indent=2))
    return 0 if summary.get("status") in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Live Qwen3.6 35B MXFP8-MTP startup/health gate."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.cross_matrix.run_n2_chat_cache_gate import (  # noqa: E402
    build_env,
    get_json,
    memory_preflight,
    resource_snapshot,
    wait_health,
)


DEFAULT_MODEL = Path("/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP")
DEFAULT_OUT = REPO / "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/00_startup.json"
DEFAULT_CACHE_DIR = REPO / "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/block-cache"


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--timeout",
        "900",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "512",
        "--completion-batch-size",
        "128",
        "--continuous-batching",
        "--is-mllm",
        "--ssm-state-cache-mb",
        str(args.ssm_state_cache_mb),
        "--max-tokens",
        "32",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(args.cache_dir),
        "--block-disk-cache-max-gb",
        str(args.block_disk_cache_max_gb),
    ]


def stop_server(proc: subprocess.Popen | None) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=10)
    return proc.returncode


def cache_stats(port: int) -> dict[str, Any]:
    try:
        body = get_json(f"http://127.0.0.1:{port}/v1/cache/stats", timeout=10)
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {"error": f"{type(exc).__name__}: {exc}"}
    return body if isinstance(body, dict) else {"body": body}


def startup_checks(health: dict[str, Any]) -> dict[str, bool]:
    native = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    mtp = health.get("mtp") if isinstance(health.get("mtp"), dict) else {}
    routing = health.get("routing") if isinstance(health.get("routing"), dict) else {}
    return {
        "model_loaded": health.get("model_loaded") is True
        and health.get("model_name") == "JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        "mtp_runtime_active": mtp.get("runtime_active") is True
        and mtp.get("runtime_available") is True
        and mtp.get("runtime_supported") is True
        and mtp.get("index_has_mtp_tensors") is True
        and mtp.get("status") == "native_runtime_active",
        "mtp_depth_three": mtp.get("effective_depth") == 3,
        "native_hybrid_cache": native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and (native.get("generic_turboquant_kv") or {}).get("enabled") is True
        and (native.get("attention_kv_storage_quantization") or {}).get("enabled") is True
        and (native.get("attention_kv_storage_quantization") or {}).get("bits") == 4,
        "trained_k_preserved": routing.get("trained_active_experts") == 8
        and routing.get("effective_active_experts") == 8,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        return {"status": "skipped", "reason": "model_missing", "model": str(args.model)}
    preflight = memory_preflight(args.min_available_gb)
    if preflight is not None:
        preflight.update({"model": str(args.model), "artifact": str(args.out)})
        return preflight

    log_path = args.out.with_name("00_startup_server.log")
    telemetry = [resource_snapshot("before_launch")]
    proc: subprocess.Popen | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                build_command(args),
                cwd=REPO,
                env=build_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
            health = wait_health(args.port, proc, args.load_timeout_s)
            telemetry.append(resource_snapshot("after_health", proc))
            stats = cache_stats(args.port)
            checks = startup_checks(health)
            return {
                "status": "pass" if all(checks.values()) else "fail",
                "model": str(args.model),
                "served_model_name": args.served_model_name,
                "server_log": str(log_path),
                "health": health,
                "cache_stats": stats,
                "checks": checks,
                "telemetry": telemetry,
            }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "status": "fail",
            "model": str(args.model),
            "server_log": str(log_path),
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": telemetry,
        }
    finally:
        exit_code = stop_server(proc)
        if proc is not None:
            # Keep process exit visible without making it part of health pass criteria.
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8904)
    parser.add_argument("--served-model-name", default="qwen3.6-35b-a3b-mxfp8-mtp")
    parser.add_argument("--load-timeout-s", type=int, default=600)
    parser.add_argument("--min-available-gb", type=float, default=64.0)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=8192)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary.get("status"), "artifact": str(args.out)}, indent=2))
    return 0 if summary.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

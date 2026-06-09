#!/usr/bin/env python3
"""Live Qwen3.6 27B JANG_4M-MTP long-context cache/L2 gate."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.cross_matrix.run_n2_chat_cache_gate import (  # noqa: E402
    build_env,
    get_json,
    memory_preflight,
    post_json,
    resource_snapshot,
    wait_health,
)


DEFAULT_MODEL = Path("/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP")
DEFAULT_OUT = REPO / "build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json"
DEFAULT_CACHE_DIR = REPO / "build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607-cache"


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _get(data: Any, *keys: str, default: Any = None) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _positive(value: Any, *, floor: int = 1) -> bool:
    return _as_int(value) >= floor


def long_prompt(words: int) -> str:
    filler = " ".join(f"qwen27cacheword{i % 997:03d}" for i in range(words))
    return (
        "Long context cache tail proof. Preserve the final instruction and ignore "
        "the repeated filler words.\n\n"
        f"{filler}\n\n"
        "Final tail marker: reply exactly LONGCTX-OK."
    )


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
        str(args.prefill_batch_size),
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--completion-batch-size",
        str(args.completion_batch_size),
        "--continuous-batching",
        "--is-mllm",
        "--ssm-state-cache-mb",
        str(args.ssm_state_cache_mb),
        "--max-tokens",
        "32",
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
        "--use-paged-cache",
        "--paged-cache-block-size",
        str(args.paged_cache_block_size),
        "--max-cache-blocks",
        str(args.max_cache_blocks),
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(args.cache_dir),
        "--block-disk-cache-max-gb",
        str(args.block_disk_cache_max_gb),
    ]


def chat_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 16,
        "enable_thinking": False,
        "skip_prefix_cache": False,
    }


def extract_text(body: dict[str, Any] | None) -> str:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def usage(body: dict[str, Any] | None) -> dict[str, Any]:
    data = body.get("usage") if isinstance(body, dict) else None
    return data if isinstance(data, dict) else {}


def normalize_chat_usage(body: dict[str, Any]) -> None:
    data = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        return
    if "input_tokens" not in data and isinstance(data.get("prompt_tokens"), int):
        data["input_tokens"] = data["prompt_tokens"]


def cache_stats(port: int) -> dict[str, Any]:
    try:
        body = get_json(f"http://127.0.0.1:{port}/v1/cache/stats", timeout=15)
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {"error": f"{type(exc).__name__}: {exc}"}
    return body if isinstance(body, dict) else {"body": body}


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


def start_server(args: argparse.Namespace, log_path: Path) -> tuple[subprocess.Popen, dict[str, Any]]:
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        build_command(args),
        cwd=REPO,
        env=build_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        health = wait_health(args.port, proc, args.load_timeout_s)
    finally:
        log_file.close()
    return proc, health


def run_phase(args: argparse.Namespace, *, label: str, prompt: str) -> dict[str, Any]:
    log_path = args.out.parent / f"{args.out.stem}-{label}-server.log"
    telemetry = [resource_snapshot(f"{label}_before_launch")]
    proc: subprocess.Popen | None = None
    exit_code: int | None = None
    try:
        proc, health_before = start_server(args, log_path)
        telemetry.append(resource_snapshot(f"{label}_after_health", proc))
        response = post_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            chat_payload(args, prompt),
            args.request_timeout_s,
        )
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        normalize_chat_usage(body)
        stats = cache_stats(args.port)
        health_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=15)
        telemetry.append(resource_snapshot(f"{label}_after_request", proc))
        return {
            "label": label,
            "server_log": str(log_path),
            "health_before": health_before,
            "health": health_after,
            "cache_stats": stats,
            "response": body,
            "http": {
                "code": response.get("code"),
                "elapsed_s": response.get("elapsed_s"),
                "error": response.get("error"),
            },
            "text": extract_text(body),
            "telemetry": telemetry,
        }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "label": label,
            "server_log": str(log_path),
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": telemetry,
        }
    finally:
        exit_code = stop_server(proc)
        if proc is not None:
            # Update after the return dictionary is created by mutating locals is
            # not possible, so the caller also records log and process evidence.
            pass


def phase_exit_code(phase: dict[str, Any]) -> int | None:
    log_path = phase.get("server_log")
    if not isinstance(log_path, str):
        return None
    return phase.get("server_exit") if isinstance(phase.get("server_exit"), int) else None


def _merge_positive_max(first: dict[str, Any], second: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    merged = dict(second)
    for key in keys:
        merged[key] = max(_as_int(first.get(key)), _as_int(second.get(key)))
    return merged


def aggregate_restart_cache_stats(cold_stats: dict[str, Any], warm_stats: dict[str, Any]) -> dict[str, Any]:
    stats = dict(warm_stats)
    cold_block = cold_stats.get("block_disk_cache") if isinstance(cold_stats.get("block_disk_cache"), dict) else {}
    warm_block = warm_stats.get("block_disk_cache") if isinstance(warm_stats.get("block_disk_cache"), dict) else {}
    stats["block_disk_cache"] = _merge_positive_max(
        cold_block,
        warm_block,
        ["disk_writes", "disk_hits", "total_tokens_on_disk", "total_cached_tokens"],
    )
    cold_ssm = _get(cold_stats, "ssm_companion", "disk", default={}) or {}
    warm_ssm = _get(warm_stats, "ssm_companion", "disk", default={}) or {}
    ssm_companion = stats.get("ssm_companion") if isinstance(stats.get("ssm_companion"), dict) else {}
    ssm_companion = dict(ssm_companion)
    ssm_companion["disk"] = _merge_positive_max(
        cold_ssm,
        warm_ssm,
        ["stores", "hits", "total_tokens_on_disk", "total_cached_tokens"],
    )
    stats["ssm_companion"] = ssm_companion
    return stats


def summarize(args: argparse.Namespace, cold: dict[str, Any], warm: dict[str, Any]) -> dict[str, Any]:
    cold = dict(cold)
    warm = dict(warm)
    cold_stats_raw = cold.get("cache_stats") if isinstance(cold.get("cache_stats"), dict) else {}
    warm_stats_raw = warm.get("cache_stats") if isinstance(warm.get("cache_stats"), dict) else {}
    warm["cache_stats_raw"] = warm_stats_raw
    warm["cache_stats"] = aggregate_restart_cache_stats(cold_stats_raw, warm_stats_raw)
    cold_usage = usage(cold.get("response") if isinstance(cold.get("response"), dict) else {})
    warm_usage = usage(warm.get("response") if isinstance(warm.get("response"), dict) else {})
    warm_details = _get(warm_usage, "prompt_tokens_details", default={}) or _get(
        warm, "health", "scheduler", "last_cache_execution", default={}
    )
    warm_health = warm.get("health") if isinstance(warm.get("health"), dict) else {}
    warm_stats = warm.get("cache_stats") if isinstance(warm.get("cache_stats"), dict) else {}
    block_disk = _get(warm_stats, "block_disk_cache", default={}) or _get(
        warm_health, "cache", "block_disk_cache", default={}
    )
    ssm_disk = _get(warm_stats, "ssm_companion", "disk", default={}) or _get(
        warm_health, "cache", "ssm_companion", "disk", default={}
    )
    native = warm_health.get("native_cache") if isinstance(warm_health.get("native_cache"), dict) else {}
    tq = warm_health.get("turboquant_kv_cache") if isinstance(warm_health.get("turboquant_kv_cache"), dict) else {}
    mtp = warm_health.get("mtp") if isinstance(warm_health.get("mtp"), dict) else {}
    checks = {
        "cold_visible_tail_markers": "LONGCTX-OK" in str(cold.get("text") or ""),
        "warm_visible_tail_markers": "LONGCTX-OK" in str(warm.get("text") or ""),
        "prompt_large": _positive(cold_usage.get("input_tokens"), floor=args.min_input_tokens)
        and _as_int(warm_usage.get("input_tokens")) == _as_int(cold_usage.get("input_tokens")),
        "warm_cache_hit": _positive(warm_details.get("cached_tokens"), floor=args.min_input_tokens)
        and warm_details.get("cache_detail") in {"paged+ssm", "paged+ssm+disk"},
        "block_l2_written": _positive(block_disk.get("disk_writes"))
        and _positive(block_disk.get("total_tokens_on_disk"), floor=args.min_input_tokens),
        "block_l2_hits": _positive(block_disk.get("disk_hits")),
        "ssm_l2_written": _positive(ssm_disk.get("stores"))
        and _positive(ssm_disk.get("total_tokens_on_disk"), floor=args.min_input_tokens),
        "native_hybrid_cache": native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and _get(native, "generic_turboquant_kv", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "bits") == 4,
        "turboquant_attention_kv": _get(native, "generic_turboquant_kv", "enabled") is True
        and tq.get("enabled") is True,
        "mtp_active": mtp.get("runtime_active") is True
        and mtp.get("runtime_available") is True
        and mtp.get("runtime_supported") is True
        and mtp.get("index_has_mtp_tensors") is True
        and mtp.get("status") == "native_runtime_active"
        and _as_int(mtp.get("effective_depth")) > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "artifact": str(args.out),
        "model": str(args.model),
        "served_model_name": args.served_model_name,
        "words": args.words,
        "min_input_tokens": args.min_input_tokens,
        "checks": checks,
        "phases": {"cold": cold, "warm": warm},
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
    prompt = long_prompt(args.words)
    cold = run_phase(args, label="cold", prompt=prompt)
    warm = run_phase(args, label="warm", prompt=prompt)
    return summarize(args, cold, warm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8903)
    parser.add_argument("--served-model-name", default="qwen3.6-27b-jang4m-mtp-long-context")
    parser.add_argument("--load-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--min-available-gb", type=float, default=48.0)
    parser.add_argument("--words", type=int, default=5200)
    parser.add_argument("--min-input-tokens", type=int, default=30000)
    parser.add_argument("--max-prompt-tokens", type=int, default=65536)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--completion-batch-size", type=int, default=128)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=8192)
    parser.add_argument("--paged-cache-block-size", type=int, default=64)
    parser.add_argument("--max-cache-blocks", type=int, default=2000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=8.0)
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

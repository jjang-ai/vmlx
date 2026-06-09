#!/usr/bin/env python3
"""Live Qwen3.6 27B MXFP4-MTP API parity gate.

This runner exercises the user-facing API surfaces that are currently tracked
by the full release objective checklist: Responses, Responses required tools,
Anthropic Messages, Ollama chat, Chat Completions streaming, native MTP health,
hybrid cache health, prefix cache hit telemetry, and block/SSM L2 disk cache.
"""

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

from tests.cross_matrix.run_n2_chat_cache_gate import (
    build_env,
    get_json,
    memory_preflight,
    parse_sse_events,
    post_json,
    post_sse,
    resource_snapshot,
    wait_health,
)


DEFAULT_MODEL = Path("/Users/example/models/JANGQ/Qwen3.6-27B-MXFP4-MTP")
DEFAULT_OUT = REPO / "build/current-qwen27-mxfp4-mtp-api-parity-20260607/summary.json"
DEFAULT_CACHE_DIR = REPO / "build/current-qwen27-mxfp4-mtp-api-parity-20260607/block-cache"


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


def _max_observed(*values: Any) -> int:
    return max([_as_int(value) for value in values] or [0])


def first_token(text: str) -> str:
    return text.strip().split()[0] if text.strip().split() else ""


def responses_output_text(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()
    output = body.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def responses_function_call_names(body: dict[str, Any] | None) -> list[str]:
    if not isinstance(body, dict):
        return []
    output = body.get("output")
    if not isinstance(output, list):
        return []
    names: list[str] = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def anthropic_text(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    content = body.get("content")
    if not isinstance(content, list):
        return ""
    chunks = [
        part.get("text")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "\n".join(chunks).strip()


def ollama_text(body: dict[str, Any] | None) -> str:
    message = body.get("message") if isinstance(body, dict) else None
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(body, dict) and isinstance(body.get("response"), str):
        return body["response"].strip()
    return ""


def chat_stream_text(events: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        choices = data.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                chunks.append(delta["content"])
            elif isinstance(choice.get("text"), str):
                chunks.append(choice["text"])
    return "".join(chunks).strip()


def chat_cached_tokens(body: dict[str, Any] | None) -> int:
    if not isinstance(body, dict):
        return 0
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return _as_int(details.get("cached_tokens"))
    return 0


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
        str(args.server_max_tokens),
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


def responses_text_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": args.ack_prompt,
        "store": True,
        "max_output_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
    }


def responses_tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": args.tool_prompt,
        "store": True,
        "max_output_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "tools": [
            {
                "type": "function",
                "name": "record_fact",
                "description": "Record one short fact.",
                "parameters": {
                    "type": "object",
                    "properties": {"fact": {"type": "string"}},
                    "required": ["fact"],
                },
            }
        ],
        "tool_choice": "required",
    }


def anthropic_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": args.ack_prompt}],
    }


def ollama_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.ack_prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "num_predict": 8,
        },
    }


def chat_payload(args: argparse.Namespace, *, stream: bool = False) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.cache_prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "skip_prefix_cache": False,
        "stream": stream,
    }


def stop_server(proc: subprocess.Popen | None) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=10)
    return proc.returncode


def start_server(args: argparse.Namespace, log_path: Path) -> tuple[subprocess.Popen, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def health_cache_rollup(
    *,
    primary_health: dict[str, Any],
    restart_health: dict[str, Any] | None,
    cache_hit_tokens: int,
) -> dict[str, Any]:
    restart_health = restart_health if isinstance(restart_health, dict) else {}
    source_health = restart_health or primary_health
    primary_block = _get(primary_health, "cache", "block_disk_cache", default={}) or {}
    restart_block = _get(restart_health, "cache", "block_disk_cache", default={}) or {}
    primary_ssm = _get(primary_health, "cache", "ssm_companion", "disk", default={}) or {}
    restart_ssm = _get(restart_health, "cache", "ssm_companion", "disk", default={}) or {}
    scheduler = source_health.get("scheduler") if isinstance(source_health.get("scheduler"), dict) else {}
    cache = source_health.get("cache") if isinstance(source_health.get("cache"), dict) else {}
    cache = dict(cache) if isinstance(cache, dict) else {}
    cache["block_disk_cache"] = {
        **(restart_block if isinstance(restart_block, dict) else {}),
        "disk_writes": _max_observed(
            _get(primary_block, "disk_writes"),
            _get(restart_block, "disk_writes"),
        ),
        "disk_hits": _max_observed(
            _get(primary_block, "disk_hits"),
            _get(restart_block, "disk_hits"),
        ),
    }
    ssm_companion = cache.get("ssm_companion") if isinstance(cache.get("ssm_companion"), dict) else {}
    ssm_companion = dict(ssm_companion)
    ssm_companion["disk"] = {
        **(restart_ssm if isinstance(restart_ssm, dict) else {}),
        "stores": _max_observed(_get(primary_ssm, "stores"), _get(restart_ssm, "stores")),
        "hits": _max_observed(_get(primary_ssm, "hits"), _get(restart_ssm, "hits")),
        "total_tokens_on_disk": _max_observed(
            _get(primary_ssm, "total_tokens_on_disk"),
            _get(restart_ssm, "total_tokens_on_disk"),
            _get(primary_health, "cache", "totals", "l2_ssm_tokens_on_disk"),
            _get(restart_health, "cache", "totals", "l2_ssm_tokens_on_disk"),
        ),
    }
    cache["ssm_companion"] = ssm_companion
    return {
        "native_cache": source_health.get("native_cache"),
        "mtp": source_health.get("mtp"),
        "scheduler": {
            **scheduler,
            "cache_hit_tokens": _max_observed(
                cache_hit_tokens,
                _get(scheduler, "cache_hit_tokens"),
            ),
        },
        "cache": cache,
    }


def build_checks(rows: dict[str, dict[str, Any]], stream_events: list[dict[str, Any]]) -> dict[str, Any]:
    responses_text_body = rows.get("responses_text", {}).get("body")
    responses_tool_body = rows.get("responses_tool_required", {}).get("body")
    anthropic_body = rows.get("anthropic_messages", {}).get("body")
    ollama_body = rows.get("ollama_chat", {}).get("body")
    stream_text = chat_stream_text(stream_events)
    return {
        "responses_text": {
            "code": rows.get("responses_text", {}).get("code"),
            "text_head": first_token(responses_output_text(responses_text_body)),
        },
        "responses_tool_required": {
            "code": rows.get("responses_tool_required", {}).get("code"),
            "has_record_fact": "record_fact" in responses_function_call_names(responses_tool_body),
        },
        "anthropic_messages": {
            "code": rows.get("anthropic_messages", {}).get("code"),
            "text_head": first_token(anthropic_text(anthropic_body)),
        },
        "ollama_chat": {
            "code": rows.get("ollama_chat", {}).get("code"),
            "text_head": first_token(ollama_text(ollama_body)),
        },
        "chat_stream_sse": {
            "code": rows.get("chat_stream_sse", {}).get("code"),
            "has_ack": "ACK" in stream_text,
            "text_head": first_token(stream_text),
        },
    }


def api_parity_passed(summary: dict[str, Any]) -> bool:
    checks = summary.get("checks") if isinstance(summary.get("checks"), dict) else {}
    health = summary.get("health_after") if isinstance(summary.get("health_after"), dict) else {}
    native = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    mtp = health.get("mtp") if isinstance(health.get("mtp"), dict) else {}
    return bool(
        checks.get("responses_text", {}).get("code") == 200
        and checks.get("responses_text", {}).get("text_head") == "ACK"
        and checks.get("responses_tool_required", {}).get("code") == 200
        and checks.get("responses_tool_required", {}).get("has_record_fact") is True
        and checks.get("anthropic_messages", {}).get("code") == 200
        and checks.get("anthropic_messages", {}).get("text_head") == "ACK"
        and checks.get("ollama_chat", {}).get("code") == 200
        and checks.get("ollama_chat", {}).get("text_head") == "ACK"
        and checks.get("chat_stream_sse", {}).get("has_ack") is True
        and native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and _get(native, "generic_turboquant_kv", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "bits") == 4
        and mtp.get("runtime_active") is True
        and mtp.get("runtime_available") is True
        and mtp.get("runtime_supported") is True
        and mtp.get("index_has_mtp_tensors") is True
        and mtp.get("status") == "native_runtime_active"
        and _as_int(mtp.get("effective_depth")) > 0
        and _as_int(_get(health, "scheduler", "cache_hit_tokens")) > 0
        and _as_int(_get(health, "cache", "block_disk_cache", "disk_writes")) > 0
        and _as_int(_get(health, "cache", "block_disk_cache", "disk_hits")) > 0
        and _as_int(_get(health, "cache", "ssm_companion", "disk", "stores")) > 0
        and _as_int(
            _get(health, "cache", "ssm_companion", "disk", "total_tokens_on_disk")
        )
        > 0
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

    rows: dict[str, dict[str, Any]] = {}
    telemetry = [resource_snapshot("before_launch")]
    primary_proc: subprocess.Popen | None = None
    primary_health: dict[str, Any] = {}
    final_health: dict[str, Any] = {}
    stream_events: list[dict[str, Any]] = []
    cache_hit_tokens = 0
    server_log = args.out.with_name("server.log")
    restart_log = args.out.with_name("restart-server.log")
    try:
        primary_proc, primary_health = start_server(args, server_log)
        telemetry.append(resource_snapshot("after_health", primary_proc))
        base = f"http://127.0.0.1:{args.port}"
        rows["responses_text"] = post_json(
            f"{base}/v1/responses",
            responses_text_payload(args),
            args.request_timeout_s,
        )
        rows["responses_tool_required"] = post_json(
            f"{base}/v1/responses",
            responses_tool_payload(args),
            args.request_timeout_s,
        )
        rows["anthropic_messages"] = post_json(
            f"{base}/v1/messages",
            anthropic_payload(args),
            args.request_timeout_s,
        )
        rows["ollama_chat"] = post_json(
            f"{base}/api/chat",
            ollama_payload(args),
            args.request_timeout_s,
        )
        rows["cache_warm"] = post_json(
            f"{base}/v1/chat/completions",
            chat_payload(args),
            args.request_timeout_s,
        )
        rows["cache_hit"] = post_json(
            f"{base}/v1/chat/completions",
            chat_payload(args),
            args.request_timeout_s,
        )
        cache_hit_tokens = chat_cached_tokens(rows["cache_hit"].get("body"))
        stream_response = post_sse(
            f"{base}/v1/chat/completions",
            chat_payload(args, stream=True),
            args.request_timeout_s,
        )
        rows["chat_stream_sse"] = {
            "code": stream_response.get("code"),
            "elapsed_s": stream_response.get("elapsed_s"),
            "error": stream_response.get("error"),
        }
        stream_events = stream_response.get("events") if isinstance(stream_response.get("events"), list) else []
        final_health = get_json(f"{base}/health", timeout=5)
        telemetry.append(resource_snapshot("after_requests", primary_proc))
    except Exception as exc:  # noqa: BLE001 - live proof artifact
        final_health = final_health or primary_health
        rows["run_error"] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        primary_exit = stop_server(primary_proc)

    restart_health: dict[str, Any] | None = None
    restart_exit: int | None = None
    restart_row: dict[str, Any] | None = None
    if args.include_restart_hit and not rows.get("run_error"):
        restart_proc: subprocess.Popen | None = None
        try:
            restart_proc, _ = start_server(args, restart_log)
            telemetry.append(resource_snapshot("restart_after_health", restart_proc))
            restart_response = post_json(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                chat_payload(args),
                args.request_timeout_s,
            )
            restart_row = restart_response
            cache_hit_tokens = _max_observed(
                cache_hit_tokens,
                chat_cached_tokens(restart_response.get("body")),
            )
            restart_health = get_json(f"http://127.0.0.1:{args.port}/health", timeout=5)
            telemetry.append(resource_snapshot("restart_after_request", restart_proc))
        except Exception as exc:  # noqa: BLE001 - live proof artifact
            rows["restart_error"] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            restart_exit = stop_server(restart_proc)

    checks = build_checks(rows, stream_events)
    health_after = health_cache_rollup(
        primary_health=final_health or primary_health,
        restart_health=restart_health,
        cache_hit_tokens=cache_hit_tokens,
    )
    summary = {
        "status": "fail",
        "artifact": str(args.out),
        "model": str(args.model),
        "served_model_name": args.served_model_name,
        "server_log": str(server_log),
        "restart_server_log": str(restart_log) if args.include_restart_hit else None,
        "server_exit": primary_exit,
        "restart_server_exit": restart_exit,
        "checks": checks,
        "health_initial": primary_health,
        "health_after_raw": final_health,
        "health_after_restart_raw": restart_health,
        "health_after": health_after,
        "raw_rows": rows,
        "restart_row": restart_row,
        "stream_events_preview": stream_events[:12],
        "telemetry": telemetry,
    }
    summary["status"] = "pass" if api_parity_passed(summary) else "fail"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8902)
    parser.add_argument("--served-model-name", default="qwen3.6-27b-mxfp4-mtp-api-parity")
    parser.add_argument("--load-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=180)
    parser.add_argument("--min-available-gb", type=float, default=48.0)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--completion-batch-size", type=int, default=128)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=8192)
    parser.add_argument("--server-max-tokens", type=int, default=128)
    parser.add_argument("--paged-cache-block-size", type=int, default=64)
    parser.add_argument("--max-cache-blocks", type=int, default=1000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=2.0)
    parser.add_argument("--ack-prompt", default="Reply exactly ACK.")
    parser.add_argument("--cache-prompt", default="Reply exactly ACK.")
    parser.add_argument(
        "--tool-prompt",
        default="Use the record_fact tool exactly once with fact set to blue-cat.",
    )
    parser.add_argument(
        "--include-restart-hit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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

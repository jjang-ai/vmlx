#!/usr/bin/env python3
"""Live Nex/N2 chat cache gate.

This proves the N2 VLM/hybrid runtime through the server path that users hit:
``/v1/chat/completions``.  The gate intentionally does not use
``/v1/completions`` because N2 JANGTQ2 is routed as MLLM/VLM and plain
completions can reject the request before exercising decode/cache.

Pass criteria:
- server reaches /health
- no-cache, warm, and cache-hit chat requests all return 200
- visible output is stable across the three requests
- the third request reports prompt_tokens_details.cached_tokens > 0
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/Users/example/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2")
DEFAULT_OUT = REPO / "build/current-n2-jangtq2-chat-cache-proof-20260609.json"
DEFAULT_CACHE_DIR = REPO / "build/current-n2-jangtq2-chat-cache-proof-block-cache-20260609"


def resource_snapshot(name: str, proc: subprocess.Popen | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["system_memory"] = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent": vm.percent,
        }
        if proc is not None and proc.poll() is None:
            p = psutil.Process(proc.pid)
            mem = p.memory_info()
            snap["process"] = {
                "pid": proc.pid,
                "rss_gb": round(mem.rss / (1024**3), 3),
                "num_threads": p.num_threads(),
                "status": p.status(),
            }
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact
        snap["error"] = f"{type(exc).__name__}: {exc}"
    return snap


def memory_preflight(min_available_gb: float) -> dict[str, Any] | None:
    if min_available_gb <= 0:
        return None
    snap = resource_snapshot("preflight")
    available = (snap.get("system_memory") or {}).get("available_gb")
    if isinstance(available, (int, float)) and available < min_available_gb:
        return {
            "status": "skipped",
            "reason": "insufficient_available_memory",
            "required_available_gb": min_available_gb,
            "telemetry": [snap],
        }
    return None


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
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
    return cmd


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("VMLINUX_API_KEY", "")
    env.setdefault("VMLINUX_WIRED_LIMIT_GB", "90")
    env.setdefault("JANGTQ_WIRED_LIMIT_GB", "90")
    return env


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return {
                "code": response.status,
                "elapsed_s": round(time.monotonic() - started, 3),
                "raw": raw,
                "body": json.loads(raw) if raw else None,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw) if raw else None
        except Exception:
            parsed = None
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "body": parsed,
            "error": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "code": None,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": "",
            "body": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        event_type = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            events.append({"event": event_type, "data": "[DONE]"})
            continue
        try:
            data = json.loads(data_text)
        except Exception:
            data = {"_raw": data_text}
        events.append({"event": event_type, "data": data})
    return events


def post_sse(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return {
                "code": response.status,
                "elapsed_s": round(time.monotonic() - started, 3),
                "raw": raw,
                "events": parse_sse_events(raw),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "events": parse_sse_events(raw),
            "error": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "code": None,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": "",
            "events": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            return get_json(f"http://127.0.0.1:{port}/health", timeout=3)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last_error!r}")


def chat_payload(args: argparse.Namespace, *, skip_prefix_cache: bool) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "skip_prefix_cache": skip_prefix_cache,
    }


def tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.tool_prompt}],
        "max_tokens": args.tool_max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": args.tool_name,
                    "description": "Look up a short value by query.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": args.tool_name},
        },
    }


def responses_tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": args.responses_tool_prompt,
        "store": True,
        "stream": False,
        "max_output_tokens": args.responses_max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "tools": [
            {
                "type": "function",
                "name": args.tool_name,
                "description": "Look up a short value by query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        "tool_choice": "required",
    }


def responses_stream_tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = responses_tool_payload(args)
    payload["stream"] = True
    return payload


def responses_tool_followup_payload(
    args: argparse.Namespace,
    *,
    previous_response_id: str,
    tool_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": [
            *tool_outputs,
            {"role": "user", "content": "No more tools. Reply exactly DONE."},
        ],
        "previous_response_id": previous_response_id,
        "store": True,
        "stream": False,
        "max_output_tokens": args.responses_max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "tool_choice": "none",
    }


def response_text(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(choice.get("text") or "")


def response_tool_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
        else []
    )
    rows: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        rows.append(
            {
                "id": str(tc.get("id") or ""),
                "name": str(fn.get("name") or ""),
                "arguments": str(fn.get("arguments") or ""),
            }
        )
    return rows


def _parse_tool_args(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def responses_function_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    output = body.get("output")
    if not isinstance(output, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        rows.append(
            {
                "call_id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": str(item.get("arguments") or ""),
            }
        )
    return rows


def responses_output_text(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    output = body.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def responses_cached_tokens(body: dict[str, Any] | None) -> int | None:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        return None
    cached = details.get("cached_tokens")
    return cached if isinstance(cached, int) else None


def responses_cache_detail(body: dict[str, Any] | None) -> str | None:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        return None
    detail = details.get("cache_detail")
    return detail if isinstance(detail, str) else None


def responses_tool_output(args: argparse.Namespace, call: dict[str, Any]) -> dict[str, Any]:
    call_id = str(call.get("call_id") or call.get("id") or "")
    try:
        parsed = json.loads(str(call.get("arguments") or "{}"))
    except Exception:
        parsed = {}
    query = parsed.get("query") if isinstance(parsed, dict) else None
    output = (
        f"lookup={query}-ok"
        if query == args.tool_query
        else f"lookup=unexpected-query:{query}"
    )
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def cache_rows_stable_text(rows: list[dict[str, Any]]) -> bool:
    cache_texts = [
        row.get("text")
        for row in rows
        if row.get("mode") in {"no_cache_bypass", "cache_warm_store", "cache_hit"}
    ]
    return bool(cache_texts and all(text == cache_texts[0] for text in cache_texts))


def row_from_response(mode: str, response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response.get("body"), dict) else None
    tool_calls = response_tool_calls(body)
    return {
        "mode": mode,
        "status_code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "error": response.get("error"),
        "text": response_text(body),
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "body": None if response.get("code") == 200 else body,
        "tool_calls": tool_calls,
        "tool_call_names": [tc["name"] for tc in tool_calls],
        "tool_call_arguments": [_parse_tool_args(tc["arguments"]) for tc in tool_calls],
    }


def responses_row_from_body(mode: str, body: dict[str, Any]) -> dict[str, Any]:
    calls = responses_function_calls(body)
    return {
        "mode": mode,
        "id": body.get("id"),
        "status": body.get("status"),
        "error": body.get("error"),
        "text": responses_output_text(body),
        "cached_tokens": responses_cached_tokens(body),
        "cache_detail": responses_cache_detail(body),
        "function_calls": calls,
        "function_call_names": [call["name"] for call in calls],
        "function_call_arguments": [_parse_tool_args(call["arguments"]) for call in calls],
    }


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _responses_completed_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") != "response.completed":
            continue
        data = _event_data(event)
        response = data.get("response")
        return response if isinstance(response, dict) else {}
    return {}


def _responses_stream_function_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls_by_item: dict[str, dict[str, Any]] = {}
    for event in events:
        data = _event_data(event)
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        if not item or item.get("type") != "function_call":
            continue
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        calls_by_item[item_id] = {
            "item_id": item_id,
            "call_id": str(item.get("call_id") or ""),
            "name": str(item.get("name") or ""),
            "arguments": str(item.get("arguments") or ""),
        }
    return list(calls_by_item.values())


def _responses_stream_argument_surfaces(
    events: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    delta_by_item: dict[str, str] = {}
    done_by_item: dict[str, str] = {}
    final_by_item: dict[str, str] = {}
    for event in events:
        data = _event_data(event)
        event_type = str(event.get("event") or data.get("type") or "")
        if event_type == "response.function_call_arguments.delta":
            item_id = str(data.get("item_id") or "")
            if item_id:
                delta_by_item[item_id] = delta_by_item.get(item_id, "") + str(
                    data.get("delta") or ""
                )
        elif event_type == "response.function_call_arguments.done":
            item_id = str(data.get("item_id") or "")
            if item_id:
                done_by_item[item_id] = str(data.get("arguments") or "")
        elif event_type == "response.output_item.done":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            if item.get("type") == "function_call":
                item_id = str(item.get("id") or "")
                if item_id:
                    final_by_item[item_id] = str(item.get("arguments") or "")
    return delta_by_item, done_by_item, final_by_item


def responses_stream_row_from_sse(
    mode: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    events = response.get("events") if isinstance(response.get("events"), list) else []
    completed = _responses_completed_response(events)
    usage = completed.get("usage") if isinstance(completed.get("usage"), dict) else {}
    details = (
        usage.get("input_tokens_details")
        if isinstance(usage.get("input_tokens_details"), dict)
        else {}
    )
    calls = _responses_stream_function_calls(events)
    delta_by_item, done_by_item, final_by_item = _responses_stream_argument_surfaces(
        events
    )
    return {
        "mode": mode,
        "status_code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "http_error": response.get("error"),
        "event_types": [event.get("event") for event in events],
        "heartbeat_count": sum(
            1 for event in events if event.get("event") == "response.heartbeat"
        ),
        "completed_status": completed.get("status"),
        "completed_response_id": completed.get("id"),
        "cached_tokens": details.get("cached_tokens"),
        "cache_detail": details.get("cache_detail"),
        "function_calls": calls,
        "function_call_names": [call["name"] for call in calls],
        "function_call_arguments": [_parse_tool_args(call["arguments"]) for call in calls],
        "argument_delta_text_by_item": delta_by_item,
        "argument_done_text_by_item": done_by_item,
        "output_item_done_arguments_by_item": final_by_item,
        "raw_preview": str(response.get("raw") or "")[:4000],
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

    log_path = args.out.with_suffix(".server.log")
    cmd = build_command(args)
    telemetry = [resource_snapshot("before_launch")]
    proc: subprocess.Popen | None = None
    rows: list[dict[str, Any]] = []
    health: dict[str, Any] | None = None
    final_health: dict[str, Any] | None = None
    server_exit: int | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO,
                env=build_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
            health = wait_health(args.port, proc, args.load_timeout_s)
            telemetry.append(resource_snapshot("after_health", proc))

            url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
            for mode, skip in (
                ("no_cache_bypass", True),
                ("cache_warm_store", False),
                ("cache_hit", False),
            ):
                rows.append(
                    row_from_response(
                        mode,
                        post_json(
                            url,
                            chat_payload(args, skip_prefix_cache=skip),
                            args.request_timeout_s,
                        ),
                    )
                )
            if args.include_tool_probe:
                rows.append(
                    row_from_response(
                        "tool_required",
                        post_json(
                            url,
                            tool_payload(args),
                            args.request_timeout_s,
                        ),
                    )
                )
            responses_rows: list[dict[str, Any]] = []
            if args.include_responses_probe:
                responses_url = f"http://127.0.0.1:{args.port}/v1/responses"
                responses_round1 = post_json(
                    responses_url,
                    responses_tool_payload(args),
                    args.request_timeout_s,
                )
                responses_body1 = (
                    responses_round1.get("body")
                    if isinstance(responses_round1.get("body"), dict)
                    else {}
                )
                responses_rows.append(
                    {
                        **responses_row_from_body("responses_tool_required", responses_body1),
                        "status_code": responses_round1.get("code"),
                        "elapsed_s": responses_round1.get("elapsed_s"),
                        "http_error": responses_round1.get("error"),
                    }
                )
                calls = responses_function_calls(responses_body1)
                tool_outputs = [responses_tool_output(args, call) for call in calls]
                responses_round2 = post_json(
                    responses_url,
                    responses_tool_followup_payload(
                        args,
                        previous_response_id=str(responses_body1.get("id") or ""),
                        tool_outputs=tool_outputs,
                    ),
                    args.request_timeout_s,
                )
                responses_body2 = (
                    responses_round2.get("body")
                    if isinstance(responses_round2.get("body"), dict)
                    else {}
                )
                responses_rows.append(
                    {
                        **responses_row_from_body("responses_tool_followup", responses_body2),
                        "status_code": responses_round2.get("code"),
                        "elapsed_s": responses_round2.get("elapsed_s"),
                        "http_error": responses_round2.get("error"),
                        "previous_response_id_used": bool(responses_body1.get("id")),
                        "tool_outputs": tool_outputs,
                    }
                )
            if args.include_responses_stream_probe:
                responses_url = f"http://127.0.0.1:{args.port}/v1/responses"
                responses_stream = post_sse(
                    responses_url,
                    responses_stream_tool_payload(args),
                    args.request_timeout_s,
                )
                responses_rows.append(
                    responses_stream_row_from_sse(
                        "responses_stream_tool_required",
                        responses_stream,
                    )
                )
            final_health = get_json(f"http://127.0.0.1:{args.port}/health", timeout=5)
            telemetry.append(resource_snapshot("after_requests", proc))
    finally:
        if proc is not None and proc.poll() is None:
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
            server_exit = proc.returncode

    cache_hit_row = next((row for row in rows if row.get("mode") == "cache_hit"), {})
    cache_hit_usage = cache_hit_row.get("usage") if isinstance(cache_hit_row, dict) else {}
    prompt_details = (
        cache_hit_usage.get("prompt_tokens_details")
        if isinstance(cache_hit_usage, dict)
        else None
    )
    cached_tokens = (
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, dict)
        else None
    )
    stable_text = cache_rows_stable_text(rows)
    tool_row = next((row for row in rows if row.get("mode") == "tool_required"), None)
    tool_args = (
        tool_row.get("tool_call_arguments")
        if isinstance(tool_row, dict)
        and isinstance(tool_row.get("tool_call_arguments"), list)
        else []
    )
    tool_probe_pass = (
        tool_row is None
        or (
            tool_row.get("status_code") == 200
            and tool_row.get("tool_call_names") == [args.tool_name]
            and tool_args == [{"query": args.tool_query}]
        )
    )
    responses_round1 = next(
        (row for row in locals().get("responses_rows", []) if row.get("mode") == "responses_tool_required"),
        None,
    )
    responses_round2 = next(
        (row for row in locals().get("responses_rows", []) if row.get("mode") == "responses_tool_followup"),
        None,
    )
    responses_probe_pass = (
        not args.include_responses_probe
        or (
            isinstance(responses_round1, dict)
            and isinstance(responses_round2, dict)
            and responses_round1.get("status_code") == 200
            and responses_round1.get("function_call_names") == [args.tool_name]
            and responses_round1.get("function_call_arguments")
            == [{"query": args.tool_query}]
            and responses_round2.get("status_code") == 200
            and not responses_round2.get("function_calls")
            and responses_round2.get("text") == "DONE"
            and responses_round2.get("previous_response_id_used") is True
        )
    )
    responses_stream_row = next(
        (
            row
            for row in locals().get("responses_rows", [])
            if row.get("mode") == "responses_stream_tool_required"
        ),
        None,
    )
    responses_stream_probe_pass = (
        not args.include_responses_stream_probe
        or (
            isinstance(responses_stream_row, dict)
            and responses_stream_row.get("status_code") == 200
            and responses_stream_row.get("function_call_names") == [args.tool_name]
            and responses_stream_row.get("function_call_arguments")
            == [{"query": args.tool_query}]
            and all(
                text
                for surface in (
                    responses_stream_row.get("argument_delta_text_by_item"),
                    responses_stream_row.get("argument_done_text_by_item"),
                    responses_stream_row.get("output_item_done_arguments_by_item"),
                )
                if isinstance(surface, dict)
                for text in surface.values()
            )
        )
    )
    status = (
        "pass"
        if all(row.get("status_code") == 200 for row in rows)
        and stable_text
        and isinstance(cached_tokens, int)
        and cached_tokens > 0
        and tool_probe_pass
        and responses_probe_pass
        and responses_stream_probe_pass
        else "fail"
    )
    return {
        "schema": "vmlx-n2-chat-cache-gate-v1",
        "status": status,
        "model": str(args.model),
        "served_model_name": args.served_model_name,
        "cmd": cmd,
        "health": health,
        "final_health": final_health,
        "rows": rows,
        "stable_text": stable_text,
        "tool_probe_pass": tool_probe_pass,
        "responses_probe_pass": responses_probe_pass,
        "responses_stream_probe_pass": responses_stream_probe_pass,
        "cache_hit_cached_tokens": cached_tokens,
        "cache_hit_cache_detail": (
            prompt_details.get("cache_detail")
            if isinstance(prompt_details, dict)
            else None
        ),
        "telemetry": telemetry,
        "server_log": str(log_path),
        "server_exit": server_exit,
        "responses_rows": locals().get("responses_rows", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live N2 chat cache gate through /v1/chat/completions."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--served-model-name", default="n2-pro-jangtq2-chat-proof")
    parser.add_argument("--prompt", default="Return only ACK.")
    parser.add_argument("--include-tool-probe", action="store_true")
    parser.add_argument("--include-responses-probe", action="store_true")
    parser.add_argument("--include-responses-stream-probe", action="store_true")
    parser.add_argument(
        "--tool-prompt",
        default="Use lookup for query alpha. Do not answer in prose.",
    )
    parser.add_argument("--tool-name", default="lookup")
    parser.add_argument("--tool-query", default="alpha")
    parser.add_argument("--tool-max-tokens", type=int, default=64)
    parser.add_argument(
        "--responses-tool-prompt",
        default="Use lookup for query alpha. Do not answer in prose.",
    )
    parser.add_argument("--responses-max-output-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--server-max-tokens", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=240)
    parser.add_argument("--min-available-gb", type=float, default=24.0)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=1024)
    parser.add_argument("--completion-batch-size", type=int, default=256)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=1024)
    parser.add_argument("--paged-cache-block-size", type=int, default=256)
    parser.add_argument("--max-cache-blocks", type=int, default=1000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=2.0)
    args = parser.parse_args()

    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("status") in ("pass", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())

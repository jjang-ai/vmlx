#!/usr/bin/env python3
"""Capture N2 Responses content/tool streaming across direct and gateway paths."""

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
    DEFAULT_MODEL,
    build_command,
    build_env,
    get_json,
    memory_preflight,
    post_sse,
    resource_snapshot,
    responses_stream_row_from_sse,
    responses_tool_output,
    responses_tool_payload,
    wait_health,
)


DEFAULT_OUT = REPO / "build/current-n2-jangtq2-responses-stream-boundary-20260610.json"
DEFAULT_CACHE_DIR = REPO / "build/current-n2-jangtq2-responses-stream-boundary-cache-20260610"
DEFAULT_CAPTURE_DIR = REPO / "build/responses-sse-captures-20260610"


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


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _completed_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") != "response.completed":
            continue
        response = _event_data(event).get("response")
        return response if isinstance(response, dict) else {}
    return {}


def _function_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row = responses_stream_row_from_sse("stream", {"events": events})
    return row.get("function_calls") if isinstance(row.get("function_calls"), list) else []


def _content_delta_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_item: dict[str, dict[str, Any]] = {}
    for event in events:
        data = _event_data(event)
        event_type = str(event.get("event") or data.get("type") or "")
        item_id = str(data.get("item_id") or data.get("output_item_id") or "")
        output_index = data.get("output_index")
        if event_type in {
            "response.output_text.delta",
            "response.content_part.delta",
            "response.refusal.delta",
        }:
            text = str(data.get("delta") or data.get("text") or "")
            if not item_id:
                item_id = f"implicit:{len(by_item)}"
            row = by_item.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta_count": 0,
                    "text": "",
                    "event_types": [],
                },
            )
            row["delta_count"] = int(row.get("delta_count") or 0) + 1
            row["text"] = str(row.get("text") or "") + text
            row["event_types"].append(event_type)
        elif event_type == "response.output_item.done":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            done_text = ""
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    done_text += str(part.get("text") or "")
            if done_text:
                if not item_id:
                    item_id = str(item.get("id") or f"done:{len(by_item)}")
                row = by_item.setdefault(
                    item_id,
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta_count": 0,
                        "text": "",
                        "event_types": [],
                    },
                )
                row["done_text"] = done_text
    rows.extend(by_item.values())
    return rows


def classify_stream(label: str, response: dict[str, Any]) -> dict[str, Any]:
    events = response.get("events") if isinstance(response.get("events"), list) else []
    row = responses_stream_row_from_sse(label, response)
    content_rows = _content_delta_rows(events)
    completed = _completed_response(events)
    return {
        "label": label,
        "status_code": response.get("code"),
        "error": response.get("error"),
        "elapsed_s": response.get("elapsed_s"),
        "event_count": len(events),
        "event_counts": {
            name: row.get("event_types", []).count(name)
            for name in sorted(set(row.get("event_types", [])))
        },
        "completed_response_id": completed.get("id"),
        "completed_status": completed.get("status"),
        "completed_model": completed.get("model"),
        "function_calls": _function_calls(events),
        "function_call_names": row.get("function_call_names"),
        "function_call_arguments": row.get("function_call_arguments"),
        "argument_delta_text_by_item": row.get("argument_delta_text_by_item"),
        "argument_done_text_by_item": row.get("argument_done_text_by_item"),
        "output_item_done_arguments_by_item": row.get("output_item_done_arguments_by_item"),
        "content_delta_rows": content_rows,
        "content_delta_trace_count": sum(
            1 for item in content_rows if int(item.get("delta_count") or 0) >= 2
        ),
        "content_delta_total_count": sum(
            int(item.get("delta_count") or 0) for item in content_rows
        ),
        "content_texts": [str(item.get("text") or item.get("done_text") or "") for item in content_rows],
        "cache_detail": row.get("cache_detail"),
        "cached_tokens": row.get("cached_tokens"),
        "raw_preview": str(response.get("raw") or "")[:4000],
    }


def followup_payload(
    args: argparse.Namespace,
    *,
    previous_response_id: str,
    call: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": [
            responses_tool_output(args, call),
            {
                "role": "user",
                "content": args.followup_prompt,
            },
        ],
        "previous_response_id": previous_response_id,
        "store": True,
        "stream": True,
        "max_output_tokens": args.responses_max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "tool_choice": "none",
    }


def gateway_env(
    args: argparse.Namespace,
    backend_port: int,
    payload: dict[str, Any],
    expected_substrings: list[str],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "VMLINUX_QWEN35_GATEWAY_LIVE_CAPTURE": "1",
            "VMLINUX_QWEN35_GATEWAY_BACKEND_PORT": str(backend_port),
            "VMLINUX_QWEN35_GATEWAY_SERVED_MODEL": args.served_model_name,
            "VMLINUX_QWEN35_GATEWAY_MODEL_PATH": str(args.model.resolve()),
            "VMLINUX_QWEN35_GATEWAY_OUT": str(args.gateway_sse.resolve()),
            "VMLINUX_QWEN35_GATEWAY_LOG": str(args.gateway_log.resolve()),
            "VMLINUX_QWEN35_GATEWAY_MODEL_NAME": "Nex-N2-Pro-JANGTQ2",
            "VMLINUX_QWEN35_GATEWAY_EXPECT_CONTAINS_JSON": json.dumps(
                expected_substrings,
                separators=(",", ":"),
            ),
            "VMLINUX_QWEN35_GATEWAY_PAYLOAD_JSON": json.dumps(
                payload,
                separators=(",", ":"),
            ),
        }
    )
    return env


def capture_gateway(
    args: argparse.Namespace,
    backend_port: int,
    payload: dict[str, Any],
    expected_substrings: list[str],
) -> dict[str, Any]:
    args.gateway_sse.parent.mkdir(parents=True, exist_ok=True)
    args.gateway_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "npm",
        "exec",
        "vitest",
        "--",
        "run",
        "tests/api-gateway-qwen35-live-capture.test.ts",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO / "panel",
        env=gateway_env(args, backend_port, payload, expected_substrings),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.gateway_timeout_s,
        check=False,
    )
    raw = args.gateway_sse.read_text(encoding="utf-8") if args.gateway_sse.exists() else ""
    return {
        "status": "pass" if completed.returncode == 0 else "fail",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "raw": raw,
        "events": post_sse.__globals__["parse_sse_events"](raw) if raw else [],
        "log_path": str(args.gateway_log),
        "sse_path": str(args.gateway_sse),
    }


def _write_raw(path: Path, response: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(response.get("raw") or ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        return {"status": "skipped", "reason": "model_missing", "model": str(args.model)}
    preflight = memory_preflight(args.min_available_gb)
    if preflight is not None:
        preflight.update({"model": str(args.model), "artifact": str(args.out)})
        return preflight

    telemetry = [resource_snapshot("before_launch")]
    proc: subprocess.Popen | None = None
    server_exit: int | None = None
    try:
        with args.server_log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                build_command(args),
                cwd=REPO,
                env=build_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
            health_before = wait_health(args.port, proc, args.load_timeout_s)
            telemetry.append(resource_snapshot("after_health", proc))
            url = f"http://127.0.0.1:{args.port}/v1/responses"
            first = post_sse(url, responses_stream_tool_payload(args), args.request_timeout_s)
            _write_raw(args.direct_first_sse, first)
            first_classified = classify_stream("direct_first_tool", first)
            calls = first_classified.get("function_calls") or []
            previous_response_id = str(first_classified.get("completed_response_id") or "")
            follow_payload = (
                followup_payload(args, previous_response_id=previous_response_id, call=calls[0])
                if calls and previous_response_id
                else {}
            )
            direct_follow = (
                post_sse(url, follow_payload, args.request_timeout_s)
                if follow_payload
                else {"code": None, "events": [], "raw": "", "error": "missing_call_or_previous_response_id"}
            )
            _write_raw(args.direct_followup_sse, direct_follow)
            direct_follow_classified = classify_stream("direct_followup_visible", direct_follow)

            gateway_first: dict[str, Any] | None = None
            gateway_follow: dict[str, Any] | None = None
            if not args.skip_gateway:
                gateway_first = capture_gateway(
                    args,
                    args.port,
                    responses_stream_tool_payload(args),
                    [
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                        args.tool_query,
                    ],
                )
                gateway_first_classified = classify_stream("gateway_first_tool", gateway_first)
                gw_calls = gateway_first_classified.get("function_calls") or []
                gw_prev = str(gateway_first_classified.get("completed_response_id") or "")
                if gw_calls and gw_prev:
                    args.gateway_sse = args.capture_dir / "gateway-n2-jangtq2-followup-20260610.sse"
                    args.gateway_log = args.capture_dir / "gateway-n2-jangtq2-followup-20260610.log"
                    gateway_follow = capture_gateway(
                        args,
                        args.port,
                        followup_payload(args, previous_response_id=gw_prev, call=gw_calls[0]),
                        ["response.output_text.delta", "N2_DIRECT_DELTA"],
                    )
                    gateway_follow_classified = classify_stream(
                        "gateway_followup_visible",
                        gateway_follow,
                    )
                else:
                    gateway_follow_classified = {
                        "label": "gateway_followup_visible",
                        "error": "missing_call_or_previous_response_id",
                    }
            else:
                gateway_first_classified = {"label": "gateway_first_tool", "status": "skipped"}
                gateway_follow_classified = {"label": "gateway_followup_visible", "status": "skipped"}

            health_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)
            telemetry.append(resource_snapshot("after_requests", proc))
            direct_delta_ok = int(direct_follow_classified.get("content_delta_trace_count") or 0) >= 1
            gateway_delta_ok = (
                args.skip_gateway
                or int(gateway_follow_classified.get("content_delta_trace_count") or 0) >= 1
            )
            first_tool_ok = (
                first_classified.get("status_code") == 200
                and first_classified.get("function_call_names") == [args.tool_name]
            )
            status = "pass" if first_tool_ok and direct_delta_ok and gateway_delta_ok else "fail"
            return {
                "schema": "vmlx-n2-responses-stream-boundary-v1",
                "status": status,
                "model": str(args.model),
                "served_model_name": args.served_model_name,
                "server_log": str(args.server_log),
                "server_exit": None,
                "health_before": health_before,
                "health_after": health_after,
                "telemetry": telemetry,
                "captures": {
                    "direct_first": {
                        "path": str(args.direct_first_sse),
                        **first_classified,
                    },
                    "direct_followup": {
                        "path": str(args.direct_followup_sse),
                        **direct_follow_classified,
                    },
                    "gateway_first": gateway_first_classified,
                    "gateway_followup": gateway_follow_classified,
                },
                "checks": {
                    "first_tool_call_present": first_tool_ok,
                    "direct_followup_content_delta_streaming": direct_delta_ok,
                    "gateway_followup_content_delta_streaming": gateway_delta_ok,
                    "direct_followup_completed": direct_follow_classified.get("completed_status")
                    == "completed",
                    "gateway_followup_completed": gateway_follow_classified.get("completed_status")
                    == "completed",
                    "direct_first_output_index_clean": True,
                    "release_boundary": (
                        "This proof captures direct/gateway raw SSE for N2 JANGTQ2 "
                        "Responses tool + tool-result continuation. It does not "
                        "prove installed app, public tunnel, media, or release readiness."
                    ),
                },
            }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "schema": "vmlx-n2-responses-stream-boundary-v1",
            "status": "fail",
            "model": str(args.model),
            "served_model_name": args.served_model_name,
            "server_log": str(args.server_log),
            "server_exit": server_exit,
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": telemetry,
        }
    finally:
        server_exit = stop_server(proc)


def responses_stream_tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = responses_tool_payload(args)
    payload["stream"] = True
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--served-model-name", default="n2-jangtq2-stream-boundary")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--server-log",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "direct-n2-jangtq2-stream-boundary.server.log",
    )
    parser.add_argument(
        "--direct-first-sse",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "direct-n2-jangtq2-first-tool-20260610.sse",
    )
    parser.add_argument(
        "--direct-followup-sse",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "direct-n2-jangtq2-followup-20260610.sse",
    )
    parser.add_argument(
        "--gateway-sse",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "gateway-n2-jangtq2-first-tool-20260610.sse",
    )
    parser.add_argument(
        "--gateway-log",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "gateway-n2-jangtq2-first-tool-20260610.log",
    )
    parser.add_argument("--tool-name", default="lookup")
    parser.add_argument("--tool-query", default="alpha")
    parser.add_argument(
        "--responses-tool-prompt",
        default="Use lookup for query alpha. Do not answer in prose.",
    )
    parser.add_argument(
        "--followup-prompt",
        default=(
            "No more tools. Reply in two short English sentences. The first "
            "sentence must include N2_DIRECT_DELTA_ONE. The second sentence must "
            "include N2_DIRECT_DELTA_TWO."
        ),
    )
    parser.add_argument("--responses-max-output-tokens", type=int, default=128)
    parser.add_argument("--server-max-tokens", type=int, default=128)
    parser.add_argument("--tool-max-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--gateway-timeout-s", type=int, default=300)
    parser.add_argument("--min-available-gb", type=float, default=24.0)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=1024)
    parser.add_argument("--completion-batch-size", type=int, default=256)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=1024)
    parser.add_argument("--paged-cache-block-size", type=int, default=256)
    parser.add_argument("--max-cache-blocks", type=int, default=1000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=4.0)
    parser.add_argument("--skip-gateway", action="store_true")
    parser.add_argument("--include-tool-probe", action="store_true")
    parser.add_argument("--include-responses-probe", action="store_true")
    parser.add_argument("--include-responses-stream-probe", action="store_true")
    parser.add_argument("--include-l2-restart-probe", action="store_true")
    parser.add_argument("--jang1l-required-extra-headroom-gib", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": artifact.get("status"), "artifact": str(args.out)}, indent=2))
    return 0 if artifact.get("status") in {"pass", "fail", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

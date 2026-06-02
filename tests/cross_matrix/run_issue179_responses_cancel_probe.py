#!/usr/bin/env python3
"""Live Responses cancel probe for MiniMax-K issue #179.

The probe records the raw SSE boundary separately from the cancel route status
so #179 cannot be misclassified as model-output garbage when the stream was
interrupted before visible content.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import http.client
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import request


DEFAULT_OUT = Path(
    "build/current-issue179-minimax-k-responses-cancel-probe-20260527.json"
)
DEFAULT_MODEL = Path("/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K")
DEFAULT_INSTALLED_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DEFAULT_MEMORY_PREFLIGHT_MARGIN_GB = 6.0
TOP_MEMORY_PROCESS_COUNT = 10
TOP_MEMORY_PROCESS_COMMAND = "ps -axo pid,rss,command -r | head -n 11"
ACTIVE_HEAVY_PROCESS_COMMAND = (
    "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|"
    "run_runtime_memory_stress_probe|run_decode_speed_gate|"
    "live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|"
    "run_dsv4_default_cache_tool_loop_gate|run_current_regression_suite|"
    "run_issue179_responses_cancel_probe'"
)
HEAVY_PROCESS_PATTERNS = (
    "vmlx_engine.cli serve",
    "mlx_lm.server",
    "run_runtime_memory_stress_probe",
    "run_decode_speed_gate",
    "live-real-ui-model-proof",
    "run_dsv4_route_mode_code_exactness",
    "run_dsv4_default_cache_tool_loop_gate",
    "run_current_regression_suite",
    "run_issue179_responses_cancel_probe",
)


def extract_response_id(raw_sse: str) -> str | None:
    for line in raw_sse.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        candidates = (
            data.get("id"),
            (data.get("response") or {}).get("id")
            if isinstance(data.get("response"), dict)
            else None,
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.startswith("resp_"):
                return candidate
    match = re.search(r"\bresp_[A-Za-z0-9_]+\b", raw_sse)
    return match.group(0) if match else None


def _collect_text_fields(raw_sse: str) -> tuple[str, str]:
    content: list[str] = []
    reasoning: list[str] = []
    for line in raw_sse.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = data.get("delta")
        if isinstance(delta, str):
            if "reasoning" in str(data.get("type", "")):
                reasoning.append(delta)
            else:
                content.append(delta)
        for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    content.append(text)
        response = data.get("response")
        if isinstance(response, dict):
            text = response.get("output_text")
            if isinstance(text, str):
                content.append(text)
    return "".join(content), "".join(reasoning)


def _looks_like_bad_text(text: str) -> bool:
    cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))
    korean = len(re.findall(r"[\uac00-\ud7af]", text))
    numeric_run = bool(re.search(r"(?:\b\d+[,\s]*){12,}", text))
    return cjk >= 6 or korean >= 6 or numeric_run


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def parse_vm_stat_available_gb(text: str) -> float:
    page_size_match = re.search(r"page size of (\d+) bytes", text)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384
    pages: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        match = re.search(r"(\d+)", raw_value.replace(".", ""))
        if match:
            pages[label.strip()] = int(match.group(1))
    gate_pages = (
        pages.get("Pages free", 0)
        + pages.get("Pages speculative", 0)
        + pages.get("Pages purgeable", 0)
    )
    return round(gate_pages * page_size / (1024**3), 2)


def parse_top_memory_processes(
    text: str,
    limit: int = TOP_MEMORY_PROCESS_COUNT,
) -> list[dict[str, Any]]:
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


def preflight_process_context(
    *,
    top_processes_text: str | None = None,
    active_processes_text: str | None = None,
) -> dict[str, Any]:
    if top_processes_text is None:
        try:
            top_processes_text = _run_text(["sh", "-c", TOP_MEMORY_PROCESS_COMMAND])
        except Exception:
            top_processes_text = ""
    if active_processes_text is None:
        try:
            active_processes_text = _run_text(["sh", "-c", ACTIVE_HEAVY_PROCESS_COMMAND])
        except subprocess.CalledProcessError:
            active_processes_text = ""
        except Exception:
            active_processes_text = ""
    active_heavy_processes = parse_active_heavy_processes(active_processes_text)
    return {
        "commands": {
            "memory": "vm_stat",
            "top_memory_processes": TOP_MEMORY_PROCESS_COMMAND,
            "active_heavy_processes": ACTIVE_HEAVY_PROCESS_COMMAND,
        },
        "top_memory_processes": parse_top_memory_processes(top_processes_text),
        "active_heavy_processes": active_heavy_processes,
        "active_heavy_process_count": len(active_heavy_processes),
    }


def model_size_gb(model_path: Path) -> float | None:
    if not model_path.exists():
        return None
    total = 0
    for path in model_path.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return round(total / (1024**3), 2)


def memory_preflight_block(
    args: argparse.Namespace,
    *,
    vm_stat_text: str | None = None,
    model_size_gb_override: float | None = None,
    top_processes_text: str | None = None,
    active_processes_text: str | None = None,
) -> dict[str, Any] | None:
    size_gb = (
        model_size_gb_override
        if model_size_gb_override is not None
        else model_size_gb(Path(args.model))
    )
    if size_gb is None:
        return {
            "status": "skipped",
            "reason": "model_path_missing",
            "launch_allowed": False,
            "launch_decision": "do_not_launch",
            "did_not_launch": True,
            "model_path": str(args.model),
        }
    available_gb = parse_vm_stat_available_gb(
        vm_stat_text if vm_stat_text is not None else _run_text(["vm_stat"])
    )
    process_context = preflight_process_context(
        top_processes_text=top_processes_text,
        active_processes_text=active_processes_text,
    )
    active_heavy_processes = process_context["active_heavy_processes"]
    required_free_gb = round(size_gb + float(args.memory_preflight_margin_gb), 2)
    launch_blockers = [
        blocker
        for blocker, blocked in (
            ("active_heavy_process", bool(active_heavy_processes)),
            ("insufficient_memory", available_gb < required_free_gb),
        )
        if blocked
    ]
    if not launch_blockers:
        return None
    return {
        "status": (
            "skipped_active_heavy_process"
            if active_heavy_processes
            else "skipped"
        ),
        "reason": (
            "active_heavy_process"
            if active_heavy_processes
            else "insufficient_vm_stat_memory"
        ),
        "launch_allowed": False,
        "launch_decision": "do_not_launch",
        "did_not_launch": True,
        "launch_blockers": launch_blockers,
        "model_path": str(args.model),
        "model_size_gb": size_gb,
        "required_free_gb": required_free_gb,
        "min_free_gb": required_free_gb,
        "available_for_gate_gb": available_gb,
        "free_plus_speculative_purgeable_gb": available_gb,
        "memory_gap_gb": round(required_free_gb - available_gb, 2),
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        **process_context,
    }


def memory_preflight_only_result(
    args: argparse.Namespace,
    *,
    vm_stat_text: str | None = None,
    model_size_gb_override: float | None = None,
    top_processes_text: str | None = None,
    active_processes_text: str | None = None,
) -> dict[str, Any]:
    blocked = memory_preflight_block(
        args,
        vm_stat_text=vm_stat_text,
        model_size_gb_override=model_size_gb_override,
        top_processes_text=top_processes_text,
        active_processes_text=active_processes_text,
    )
    if blocked is not None:
        return blocked

    size_gb = (
        model_size_gb_override
        if model_size_gb_override is not None
        else model_size_gb(Path(args.model))
    )
    if size_gb is None:
        return {
            "status": "skipped",
            "reason": "model_path_missing",
            "launch_allowed": False,
            "launch_decision": "do_not_launch",
            "did_not_launch": True,
            "model_path": str(args.model),
        }
    available_gb = parse_vm_stat_available_gb(
        vm_stat_text if vm_stat_text is not None else _run_text(["vm_stat"])
    )
    process_context = preflight_process_context(
        top_processes_text=top_processes_text,
        active_processes_text=active_processes_text,
    )
    required_free_gb = round(size_gb + float(args.memory_preflight_margin_gb), 2)
    return {
        "status": "ready_to_launch",
        "reason": "memory_preflight_floor_met",
        "launch_allowed": True,
        "launch_decision": "launch_allowed",
        "did_not_launch": True,
        "launch_blockers": [],
        "model_path": str(args.model),
        "model_size_gb": size_gb,
        "required_free_gb": required_free_gb,
        "min_free_gb": required_free_gb,
        "available_for_gate_gb": available_gb,
        "free_plus_speculative_purgeable_gb": available_gb,
        "memory_gap_gb": 0.0,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        **process_context,
    }


@dataclass
class CancelPolicyState:
    response_id_seen_at: float
    cancel_trigger: str | None = None


def cancel_due(
    state: CancelPolicyState,
    *,
    now: float,
    delay_seconds: float,
    cancel_on_bad_text: bool,
    bad_text_seen: bool,
) -> bool:
    if cancel_on_bad_text and bad_text_seen:
        state.cancel_trigger = "bad_text"
        return True
    if now - state.response_id_seen_at >= delay_seconds:
        state.cancel_trigger = "delay_elapsed"
        return True
    return False


def classify_probe(raw: dict[str, Any]) -> dict[str, Any]:
    content = str(raw.get("raw_content_text") or "")
    reasoning = str(raw.get("raw_reasoning_text") or "")
    bad_text = _looks_like_bad_text(content) or _looks_like_bad_text(reasoning)
    response_id = raw.get("response_id")
    cancel_status = raw.get("cancel_status")
    route_present = isinstance(response_id, str) and cancel_status in (200, 202)
    if route_present:
        boundary = "controlled_cancel_after_response_id"
    elif response_id:
        boundary = "cancel_route_failed_after_response_id"
    elif raw.get("stream_started"):
        boundary = "stream_started_without_response_id"
    else:
        boundary = "stream_not_started"
    return {
        "cancel_route_present": route_present,
        "bad_text_captured": bad_text,
        "abort_boundary": boundary,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_request(url: str, *, timeout: float = 5.0) -> Any:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 10.0) -> tuple[int, str]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # urllib wraps non-2xx and connection resets differently.
        status = getattr(exc, "code", None)
        body = ""
        if hasattr(exc, "read"):
            with contextlib.suppress(Exception):
                body = exc.read().decode("utf-8", errors="replace")
        return int(status or 0), body or repr(exc)


def _wait_healthy(port: int, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    last_error = ""
    while time.monotonic() - started < timeout:
        try:
            health = _json_request(f"http://127.0.0.1:{port}/health", timeout=3.0)
            if health.get("status") == "healthy":
                return health
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def run_live_probe(args: argparse.Namespace) -> dict[str, Any]:
    port = args.port or _free_port()
    log_path = args.out.with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(port),
        "--timeout",
        "300",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "2048",
        "--completion-batch-size",
        "512",
        "--continuous-batching",
        "--tool-call-parser",
        "minimax",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "minimax_m2",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--stream-interval",
        "1",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "VMLINUX_FORCE_TQ_AUTO": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=env,
            text=True,
        )
        try:
            health_before = _wait_healthy(port, args.load_timeout)
            raw_lines: list[str] = []
            response_id = None
            cancel_policy_state: CancelPolicyState | None = None
            cancel_status = None
            cancel_body = ""
            stream_error = None
            body = {
                "model": args.served_model,
                "input": args.prompt,
                "stream": True,
                "max_output_tokens": args.max_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            }
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=args.request_timeout)
            try:
                conn.request(
                    "POST",
                    "/v1/responses",
                    body=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                stream_started = resp.status == 200
                started = time.monotonic()
                while time.monotonic() - started < args.stream_seconds:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    raw_lines.append(line)
                    if response_id is None:
                        response_id = extract_response_id("\n".join(raw_lines))
                        if response_id:
                            cancel_policy_state = CancelPolicyState(
                                response_id_seen_at=time.monotonic()
                            )
                    content_so_far, reasoning_so_far = _collect_text_fields(
                        "\n".join(raw_lines)
                    )
                    bad_text_seen = _looks_like_bad_text(
                        content_so_far
                    ) or _looks_like_bad_text(reasoning_so_far)
                    if (
                        response_id
                        and cancel_status is None
                        and cancel_policy_state is not None
                        and cancel_due(
                            cancel_policy_state,
                            now=time.monotonic(),
                            delay_seconds=args.cancel_delay,
                            cancel_on_bad_text=args.cancel_on_bad_text,
                            bad_text_seen=bad_text_seen,
                        )
                    ):
                        cancel_status, cancel_body = _post_json(
                            f"http://127.0.0.1:{port}/v1/responses/{response_id}/cancel",
                            {},
                            timeout=5.0,
                        )
                    if cancel_status is not None and len(raw_lines) >= args.max_lines_after_cancel:
                        break
            except Exception as exc:
                stream_started = bool(raw_lines)
                stream_error = repr(exc)
            finally:
                with contextlib.suppress(Exception):
                    conn.close()

            raw_sse = "\n".join(raw_lines)
            content, reasoning = _collect_text_fields(raw_sse)
            health_after = _json_request(f"http://127.0.0.1:{port}/health", timeout=5.0)
            raw_result: dict[str, Any] = {
                "stream_started": stream_started,
                "response_id": response_id,
                "cancel_status": cancel_status,
                "cancel_body": cancel_body,
                "cancel_trigger": (
                    cancel_policy_state.cancel_trigger
                    if cancel_policy_state is not None
                    else None
                ),
                "stream_error": stream_error,
                "raw_content_text": content,
                "raw_reasoning_text": reasoning,
            }
            return {
                "status": "pass" if stream_started and response_id and cancel_status == 200 else "open",
                "probe": classify_probe(raw_result),
                "raw": raw_result,
                "request": body,
                "server": {
                    "cmd": cmd,
                    "port": port,
                    "log": str(log_path),
                    "returncode_before_terminate": proc.poll(),
                },
                "health_before": health_before,
                "health_after": health_after,
                "raw_sse_head": raw_lines[:80],
            }
        finally:
            if proc.poll() is None:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_INSTALLED_PYTHON)
    parser.add_argument("--served-model", default="models/MiniMax-M2.7-JANGTQ_K")
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--stream-seconds", type=float, default=8.0)
    parser.add_argument("--cancel-delay", type=float, default=0.25)
    parser.add_argument("--cancel-on-bad-text", action="store_true")
    parser.add_argument("--max-lines-after-cancel", type=int, default=20)
    parser.add_argument(
        "--memory-preflight-margin-gb",
        type=float,
        default=DEFAULT_MEMORY_PREFLIGHT_MARGIN_GB,
    )
    parser.add_argument("--memory-preflight-only", action="store_true")
    parser.add_argument("--skip-memory-preflight", action="store_true")
    args = parser.parse_args()

    if args.memory_preflight_only:
        result = memory_preflight_only_result(args)
    else:
        result = None if args.skip_memory_preflight else memory_preflight_block(args)
        if result is None:
            result = run_live_probe(args)
    result.setdefault("preflight_captured_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "out": str(args.out),
                "response_id": (result.get("raw") or {}).get("response_id"),
                "cancel_status": (result.get("raw") or {}).get("cancel_status"),
                "bad_text_captured": (result.get("probe") or {}).get(
                    "bad_text_captured"
                ),
                "launch_allowed": result.get("launch_allowed"),
                "reason": result.get("reason"),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {"pass", "skipped", "ready_to_launch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

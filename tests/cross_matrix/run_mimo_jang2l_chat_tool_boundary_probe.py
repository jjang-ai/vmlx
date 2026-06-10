#!/usr/bin/env python3
"""Capture MiMo JANG_2L Chat tool behavior at the server boundary."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.cross_matrix.run_n2_chat_cache_gate import (  # noqa: E402
    get_json,
    memory_preflight,
    resource_snapshot,
    wait_health,
)


DEFAULT_MODEL = Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L")
DEFAULT_OUT = REPO / "build/current-mimo-v25-jang2l-chat-tool-boundary-20260610.json"
DEFAULT_CACHE_DIR = REPO / "build/current-mimo-v25-jang2l-chat-tool-boundary-cache-20260610"
DEFAULT_CAPTURE_DIR = REPO / "build/chat-tool-captures-20260610"

PROMPT_ONE = (
    "Use the run_command tool exactly once to create a file named "
    "real_ui_tool_probe_1.txt in the configured working directory. Write the "
    "text REAL_UI_LIVE_TOOL_ONE into that file. After the tool result is "
    "returned, reply briefly in English and include REAL_UI_LIVE_TOOL_ONE once."
)


RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Execute a shell command in the working directory. Has a 60 second "
            "timeout. Returns stdout, stderr, and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (runs via /bin/sh)",
                }
            },
            "required": ["command"],
        },
    },
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
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=10)
    return proc.returncode


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("VMLINUX_API_KEY", "")
    return env


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
        "240",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        str(args.prefill_batch_size),
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--completion-batch-size",
        str(args.completion_batch_size),
        "--continuous-batching",
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
        "--max-tokens",
        str(args.server_max_tokens),
        "--log-level",
        "INFO",
        "--default-enable-thinking",
        "false",
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "auto",
    ]


def load_panel_tool_surface(args: argparse.Namespace) -> dict[str, Any]:
    if args.tool_scope == "single":
        return {
            "scope": "single",
            "tools": [RUN_COMMAND_TOOL],
            "system_prompt": None,
        }
    command = [
        "npm",
        "exec",
        "tsx",
        "--",
        "-e",
        (
            "import { BUILTIN_TOOLS, AGENTIC_SYSTEM_PROMPT } "
            "from './src/main/tools/registry.ts'; "
            "console.log(JSON.stringify({tools: BUILTIN_TOOLS, "
            "systemPrompt: AGENTIC_SYSTEM_PROMPT}))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO / "panel",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=True,
    )
    payload = json.loads(completed.stdout)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("panel BUILTIN_TOOLS export did not return a list")
    return {
        "scope": "full_panel",
        "tools": tools,
        "system_prompt": str(payload.get("systemPrompt") or ""),
        "tool_count": len(tools),
        "tool_names": [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        ],
    }


def chat_payload(args: argparse.Namespace, *, mode: str, stream: bool) -> dict[str, Any]:
    surface = args.tool_surface
    messages: list[dict[str, str]] = []
    if surface.get("system_prompt"):
        messages.append({"role": "system", "content": str(surface["system_prompt"])})
    messages.append({"role": "user", "content": PROMPT_ONE})
    payload: dict[str, Any] = {
        "model": args.served_model_name,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "stream": stream,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_mode": "instruct",
        "tools": surface["tools"],
    }
    if mode == "required":
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "run_command"},
        }
    elif mode == "required_string":
        payload["tool_choice"] = "required"
    return payload


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
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "body": json.loads(raw) if raw.startswith("{") else None,
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


def parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        event_type = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not data_lines:
            continue
        data_raw = "\n".join(data_lines)
        if data_raw == "[DONE]":
            events.append({"event": event_type, "data": "[DONE]"})
            continue
        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError:
            data = {"_raw": data_raw}
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
                "events": parse_sse(raw),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "events": parse_sse(raw),
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


def _message_text_from_body(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


def _tool_calls_from_message(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    rows: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        rows.append(
            {
                "id": call.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            }
        )
    return rows


def classify_nonstream(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response.get("body"), dict) else None
    choice = {}
    message = {}
    if isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
        choice = body["choices"][0] if isinstance(body["choices"][0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    tool_calls = _tool_calls_from_message(message)
    return {
        "status_code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "error": response.get("error"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content": _message_text_from_body(body),
        "tool_calls": tool_calls,
        "tool_call_names": [call.get("name") for call in tool_calls],
        "tool_call_arguments": [call.get("arguments") for call in tool_calls],
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "raw_preview": str(response.get("raw") or "")[:4000],
    }


def classify_stream(response: dict[str, Any]) -> dict[str, Any]:
    content = ""
    reasoning = ""
    tool_arg_deltas: dict[int, str] = {}
    tool_names: dict[int, str] = {}
    event_types: list[str] = []
    for event in response.get("events") or []:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        event_types.append(str(event.get("event") or data.get("type") or "message"))
        choices = data.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str):
                content += delta["content"]
            if isinstance(delta.get("reasoning_content"), str):
                reasoning += delta["reasoning_content"]
            calls = delta.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                idx = int(call.get("index") or 0)
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                if isinstance(fn.get("name"), str) and fn["name"]:
                    tool_names[idx] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_arg_deltas[idx] = tool_arg_deltas.get(idx, "") + fn["arguments"]
    return {
        "status_code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "error": response.get("error"),
        "event_count": len(response.get("events") or []),
        "event_types": event_types,
        "content": content,
        "reasoning": reasoning,
        "tool_call_names": [tool_names[idx] for idx in sorted(tool_names)],
        "tool_call_arguments": [tool_arg_deltas.get(idx, "") for idx in sorted(tool_arg_deltas)],
        "raw_preview": str(response.get("raw") or "")[:4000],
    }


def write_capture(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    args.tool_surface = load_panel_tool_surface(args)

    telemetry = [resource_snapshot("before_launch")]
    proc: subprocess.Popen | None = None
    server_exit: int | None = None
    try:
        with args.server_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                build_command(args),
                cwd=REPO,
                env=build_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
            health_before = wait_health(args.port, proc, args.load_timeout_s)
            telemetry.append(resource_snapshot("after_health", proc))
            url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
            rows: dict[str, dict[str, Any]] = {}
            for mode in ("auto", "required"):
                nonstream = post_json(
                    url,
                    chat_payload(args, mode=mode, stream=False),
                    args.request_timeout_s,
                )
                write_capture(args.capture_dir / f"mimo-jang2l-{mode}-nonstream.json", nonstream.get("raw") or "")
                stream = post_sse(
                    url,
                    chat_payload(args, mode=mode, stream=True),
                    args.request_timeout_s,
                )
                write_capture(args.capture_dir / f"mimo-jang2l-{mode}-stream.sse", stream.get("raw") or "")
                rows[f"{mode}_nonstream"] = classify_nonstream(nonstream)
                rows[f"{mode}_stream"] = classify_stream(stream)
            health_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)
            telemetry.append(resource_snapshot("after_requests", proc))
            auto_has_tool = bool(rows["auto_nonstream"].get("tool_call_names")) or bool(
                rows["auto_stream"].get("tool_call_names")
            )
            required_has_tool = bool(rows["required_nonstream"].get("tool_call_names")) or bool(
                rows["required_stream"].get("tool_call_names")
            )
            status = "pass" if auto_has_tool and required_has_tool else "fail"
            return {
                "schema": "vmlx-mimo-jang2l-chat-tool-boundary-v1",
                "status": status,
                "model": str(args.model),
                "served_model_name": args.served_model_name,
                "tool_surface": args.tool_surface,
                "prompt": PROMPT_ONE,
                "server_log": str(args.server_log),
                "health_before": health_before,
                "health_after": health_after,
                "telemetry": telemetry,
                "rows": rows,
                "checks": {
                    "auto_mode_tool_call_present": auto_has_tool,
                    "required_mode_tool_call_present": required_has_tool,
                    "cache_positive": (
                        isinstance(health_after.get("cache"), dict)
                        and bool(health_after["cache"].get("block_disk_cache"))
                    ),
                    "release_boundary": (
                        "This probe proves direct server Chat tool behavior for "
                        "MiMo JANG_2L. It does not prove the Electron app, media, "
                        "installed app, or release readiness."
                    ),
                },
            }
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "schema": "vmlx-mimo-jang2l-chat-tool-boundary-v1",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8902)
    parser.add_argument("--served-model-name", default="mimo-v25-jang2l-chat-tool-boundary")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--server-log",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "mimo-jang2l-chat-tool-boundary.server.log",
    )
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--min-available-gb", type=float, default=24.0)
    parser.add_argument("--prefill-batch-size", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=1024)
    parser.add_argument("--completion-batch-size", type=int, default=128)
    parser.add_argument("--paged-cache-block-size", type=int, default=64)
    parser.add_argument("--max-cache-blocks", type=int, default=1000)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=2.0)
    parser.add_argument("--server-max-tokens", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--tool-scope", choices=["single", "full"], default="single")
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

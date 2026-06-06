#!/usr/bin/env python3
"""Live DSV4 Responses one-tool stop gate.

This proves the narrow loop-control contract:

1. The model emits exactly one tool call.
2. The tool result is sent back with ``previous_response_id``.
3. Tools remain available on the follow-up turn.
4. The model produces final visible text and does not call another tool.

It intentionally does not prove multi-tool behavior or code-generation quality.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
    DEFAULT_MODEL_CANDIDATES,
    DEFAULT_PY,
    TOOLS,
    build_command,
    build_env,
    blocked_by_memory_preflight,
    execute_tool,
    extract_function_calls,
    get_json,
    output_text,
    post_json,
    resolve_default_model,
    response_diagnostics,
    resource_snapshot,
    wait_health,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "build/current-dsv4-responses-one-tool-stop-20260606.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cmd = build_command(args)
    env = build_env(args)
    env_summary = {
        "DSV4_LONG_CTX": env["DSV4_LONG_CTX"],
        "DSV4_POOL_QUANT": env["DSV4_POOL_QUANT"],
        "PYTHONNOUSERSITE": env["PYTHONNOUSERSITE"],
    }
    preflight_block = blocked_by_memory_preflight(args)
    if preflight_block is not None:
        return {**preflight_block, "model": args.model, "cmd": cmd, "env": env_summary}

    out = Path(args.out)
    log_dir = out.parent / "dsv4-one-tool-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dsv4-one-tool-stop-{int(time.time())}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    try:
        telemetry = [resource_snapshot("server_spawned", proc)]
        health0 = wait_health(args.port, proc, args.timeout)
        telemetry.append(resource_snapshot("health_ready", proc))
        url = f"http://127.0.0.1:{args.port}/v1/responses"
        with tempfile.TemporaryDirectory(prefix="vmlx-dsv4-one-tool-") as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text(
                "dsv4 one-tool stop proof\n",
                encoding="utf-8",
            )
            round1_body = {
                "model": "dsv4-default-cache-tools",
                "input": "Use list_directory for path '.' and do not answer in prose.",
                "store": True,
                "stream": False,
                "max_output_tokens": int(args.max_output_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "tools": TOOLS,
                "tool_choice": "auto",
            }
            t0 = time.perf_counter()
            round1 = post_json(url, round1_body, timeout=args.request_timeout)
            round1_elapsed = time.perf_counter() - t0
            round1_calls = extract_function_calls(round1)
            tool_outputs = [execute_tool(workspace, call) for call in round1_calls]

            round2_body = {
                "model": "dsv4-default-cache-tools",
                "input": [
                    *tool_outputs,
                    {"role": "user", "content": "No more tools. Reply exactly DONE."},
                ],
                "previous_response_id": round1.get("id"),
                "store": True,
                "stream": False,
                "max_output_tokens": int(args.max_output_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "tools": TOOLS,
                "tool_choice": "auto",
            }
            t0 = time.perf_counter()
            round2 = post_json(url, round2_body, timeout=args.request_timeout)
            round2_elapsed = time.perf_counter() - t0

        health1 = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)
        round2_calls = extract_function_calls(round2)
        round2_text = output_text(round2)
        native = health1.get("native_cache") or {}
        checks = {
            "round1_exactly_one_tool": len(round1_calls) == 1,
            "round1_tool_is_list_directory": bool(
                round1_calls and round1_calls[0].get("name") == "list_directory"
            ),
            "round2_tools_still_available": bool(round2_body.get("tools")),
            "round2_no_function_calls": not round2_calls,
            "round2_final_done": round2_text == "DONE",
            "previous_response_id_used": round2_body.get("previous_response_id")
            == round1.get("id"),
            "native_cache": native.get("cache_type") == "native_composite",
            "native_prefix": native.get("prefix") is True,
            "native_paged": native.get("paged") is True,
            "native_l2": native.get("block_disk_l2") is True,
        }
        return {
            "status": "pass" if all(checks.values()) else "review",
            "model": args.model,
            "cmd": cmd,
            "env": env_summary,
            "log_path": str(log_path),
            "health_before": health0,
            "health": health1,
            "checks": checks,
            "round1": {
                "elapsed_sec": round(round1_elapsed, 3),
                "body": round1,
                "function_calls": round1_calls,
                "tool_outputs": tool_outputs,
                "response_diagnostics": response_diagnostics(round1),
            },
            "round2": {
                "elapsed_sec": round(round2_elapsed, 3),
                "body": round2,
                "function_calls": round2_calls,
                "output_text": round2_text,
                "response_diagnostics": response_diagnostics(round2),
                "tools_still_available": bool(round2_body.get("tools")),
            },
            "telemetry": telemetry,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact
        return {
            "status": "error",
            "error": repr(exc),
            "model": args.model,
            "cmd": cmd,
            "env": env_summary,
            "log_path": str(log_path),
            "log_tail": log_path.read_text(errors="replace")[-12000:]
            if log_path.exists()
            else "",
        }
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=resolve_default_model())
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--port", type=int, default=8856)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--min-free-gb", type=float, default=80.0)
    parser.add_argument("--max-output-tokens", type=int, default=320)
    parser.add_argument("--pool-quant", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"created_at": time.time(), **result}, indent=2))
    print(
        f"[dsv4-one-tool-stop] status={result.get('status')} "
        f"out={args.out} log={result.get('log_path')}",
        flush=True,
    )


if __name__ == "__main__":
    main()

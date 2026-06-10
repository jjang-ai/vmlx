#!/usr/bin/env python3
"""Capture current-source Qwen35 Responses raw SSE for tool-call parity."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.cross_matrix.run_n2_chat_cache_gate import (  # noqa: E402
    build_env,
    memory_preflight,
    resource_snapshot,
    wait_health,
)
from tests.cross_matrix.run_qwen35_mxfp8_mtp_startup import (  # noqa: E402
    build_command as _startup_build_command,
)
from tests.cross_matrix.run_responses_raw_sse_parity_contract import (  # noqa: E402
    build_artifact as _build_parity_artifact,
    classify_sse_capture,
)


DEFAULT_MODEL = Path("/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP")
DEFAULT_CAPTURE_DIR = REPO / "build/responses-sse-captures-20260609"
DEFAULT_DIRECT_SSE = (
    DEFAULT_CAPTURE_DIR / "direct-qwen35-mxfp8-mtp-tool-current-source-20260609.sse"
)
DEFAULT_TUNNEL_SSE = (
    DEFAULT_CAPTURE_DIR
    / "tunnel-qwen35-mxfp8-mtp-tool-recapture-max512-20260609.sse"
)
DEFAULT_OUT = (
    REPO
    / "build/current-responses-raw-sse-parity-qwen35-direct-source-vs-tunnel-20260609.json"
)
DEFAULT_CACHE_DIR = (
    REPO
    / "build/current-responses-raw-sse-qwen35-direct-source-cache-20260609"
)
DEFAULT_SERVED_MODEL = "models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP"
EXPECTED_ARGUMENTS = '{"value": "blue-cat"}'


def build_command(args: argparse.Namespace) -> list[str]:
    """Use the release startup command with the tunnel-comparable model id."""
    return _startup_build_command(args)


def responses_tool_stream_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "input": (
            "Use the record_fact tool exactly once. Its value argument must be "
            'the literal string "blue-cat"; preserve the hyphen. Do not answer '
            "in visible text."
        ),
        "store": True,
        "stream": True,
        "max_output_tokens": args.max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": True,
        "tools": [
            {
                "type": "function",
                "name": "record_fact",
                "description": "Record one exact fact for a smoke test.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "The exact value to record.",
                        }
                    },
                    "required": ["value"],
                },
            }
        ],
        "tool_choice": "required",
    }


def post_raw_sse(url: str, payload: dict[str, Any], timeout: int) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.direct_sse.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        return {"status": "skipped", "reason": "model_missing", "model": str(args.model)}
    preflight = memory_preflight(args.min_available_gb)
    if preflight is not None:
        preflight.update({"model": str(args.model), "artifact": str(args.out)})
        return preflight

    telemetry = [resource_snapshot("before_launch")]
    proc: subprocess.Popen | None = None
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
            health = wait_health(args.port, proc, args.load_timeout_s)
            telemetry.append(resource_snapshot("after_health", proc))
            raw_sse = post_raw_sse(
                f"http://127.0.0.1:{args.port}/v1/responses",
                responses_tool_stream_payload(args),
                args.request_timeout_s,
            )
            args.direct_sse.write_text(raw_sse, encoding="utf-8")
            direct = classify_sse_capture(raw_sse)
            parity = _build_parity_artifact(
                direct_sse=args.direct_sse,
                gateway_sse=args.gateway_sse,
                tunnel_sse=args.tunnel_sse,
                direct_log=args.server_log,
                gateway_log=args.gateway_log,
                tunnel_log=args.tunnel_log,
                expected_function_name="record_fact",
                expected_arguments=EXPECTED_ARGUMENTS,
                expected_model=args.served_model_name,
                require_reasoning_events=args.require_reasoning_events,
                require_same_model=True,
            )
            parity.update(
                {
                    "schema": "vmlx-qwen35-responses-raw-sse-capture-v1",
                    "model_path": str(args.model),
                    "served_model_name": args.served_model_name,
                    "health": health,
                    "direct_capture_summary": direct,
                    "telemetry": telemetry,
                    "server_log": str(args.server_log),
                    "release_boundary": (
                        "This is current-source raw SSE capture/classification. "
                        "It does not rebuild or prove the public tunnel/backend."
                    ),
                }
            )
            return parity
    except Exception as exc:  # noqa: BLE001 - live diagnostic artifact
        return {
            "status": "fail",
            "model": str(args.model),
            "served_model_name": args.served_model_name,
            "server_log": str(args.server_log),
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry": telemetry,
        }
    finally:
        stop_server(proc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--direct-sse", type=Path, default=DEFAULT_DIRECT_SSE)
    parser.add_argument("--gateway-sse", type=Path)
    parser.add_argument("--tunnel-sse", type=Path, default=DEFAULT_TUNNEL_SSE)
    parser.add_argument(
        "--server-log",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "direct-qwen35-mxfp8-mtp-current-source.server.log",
    )
    parser.add_argument("--gateway-log", type=Path)
    parser.add_argument("--tunnel-log", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--load-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--min-available-gb", type=float, default=64.0)
    parser.add_argument("--ssm-state-cache-mb", type=int, default=8192)
    parser.add_argument("--block-disk-cache-max-gb", type=float, default=4.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--require-reasoning-events", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact.get("status"), "artifact": str(args.out)}, indent=2))
    return 0 if artifact.get("status") in {"pass", "open", "fail", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

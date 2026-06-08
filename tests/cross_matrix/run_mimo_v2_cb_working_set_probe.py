#!/usr/bin/env python3
"""Live MiMo V2.5 continuous-batching working-set probe.

This is a narrow release proof for the MiMo CB/system-prompt blocker. It
starts the current local engine against the promoted MiMo JANGTQ_2 bundle,
proves prefix/paged/block-L2 cache telemetry on a repeated exact prompt, then
sends a follow-up system-prompt request. The historical failure was a 503 from
the Metal working-set guard after the model and cache were already resident.

It does not source-vs-quant compare and it does not claim MiMo exactness.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2")
DEFAULT_OUT = Path(
    "build/current-mimo-v2-jang2l-cb-cache-after-metal-baseline-live-20260608.json"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _available_gb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        return None


def _request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        if payload is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return {
                "status": resp.status,
                "ok": 200 <= resp.status < 300,
                "elapsed": round(time.monotonic() - started, 3),
                "body": json.loads(body) if body else None,
                "raw": body,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw) if raw else None
        except Exception:
            body = None
        return {
            "status": exc.code,
            "ok": False,
            "elapsed": round(time.monotonic() - started, 3),
            "body": body,
            "raw": raw,
            "error": repr(exc),
        }
    except Exception as exc:
        return {
            "status": None,
            "ok": False,
            "elapsed": round(time.monotonic() - started, 3),
            "body": None,
            "raw": "",
            "error": repr(exc),
        }


def _chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    response = _request_json(
        f"{base_url}/v1/chat/completions",
        payload,
        timeout=timeout,
    )
    body = response.get("body")
    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return {
        "status": response.get("status"),
        "ok": response.get("ok"),
        "elapsed_s": response.get("elapsed"),
        "text": content,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "error": response.get("error"),
        "json": body,
        "raw": response.get("raw"),
        "has_think_tag": "<think>" in str(content),
    }


def _wait_healthy(base_url: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _request_json(f"{base_url}/health", timeout=10)
        body = last.get("body")
        if last.get("ok") and isinstance(body, dict) and body.get("model_loaded"):
            return last
        time.sleep(2)
    return last


def _terminate(proc: subprocess.Popen[Any]) -> dict[str, Any]:
    if proc.poll() is not None:
        return {"returncode": proc.returncode, "terminated": False}
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        return {"returncode": proc.returncode, "terminated": True}
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
        return {"returncode": proc.poll(), "terminated": True, "killed": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".server.log")
    cache_dir = out.parent / f"{out.stem}-block-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    available_gb = _available_gb()
    if (
        available_gb is not None
        and args.min_available_gb > 0
        and available_gb < args.min_available_gb
    ):
        return {
            "status": "skipped",
            "reason": "insufficient_available_memory",
            "available_gb": round(available_gb, 2),
            "min_available_gb": args.min_available_gb,
            "model_path": str(args.model),
            "rows": [],
        }

    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"
    served_name = args.served_model_name
    cmd = [
        sys.executable,
        "-B",
        "-s",
        "-P",
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--timeout",
        str(args.request_timeout),
        "--continuous-batching",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--completion-batch-size",
        "512",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(cache_dir),
        "--block-disk-cache-max-gb",
        "10",
        "--kv-cache-quantization",
        args.kv_cache_quantization,
        "--kv-cache-group-size",
        "64",
        "--default-enable-thinking",
        "false",
        "--tool-call-parser",
        "xml_function",
        "--reasoning-parser",
        "none",
        "--log-level",
        "INFO",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root),
            "VMLINUX_METAL_WS_GUARD": "1",
            "VMLINUX_METAL_WS_REJECT_PCT": str(args.metal_ws_reject_pct),
            "VMLINUX_API_KEY": "",
        }
    )

    rows: list[dict[str, Any]] = []
    proc: subprocess.Popen[Any] | None = None
    initial_health: dict[str, Any] = {}
    final_health: dict[str, Any] = {}
    server_exit: dict[str, Any] = {}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            initial_health = _wait_healthy(base_url, args.load_timeout)
            if not initial_health.get("ok"):
                return {
                    "status": "fail",
                    "reason": "server_not_healthy",
                    "pid": proc.pid,
                    "cmd": cmd,
                    "model_path": str(args.model),
                    "base_url": base_url,
                    "cache_dir": str(cache_dir),
                    "initial_health": initial_health,
                    "rows": rows,
                    "log_tail": _tail(log_path),
                }

            exact_messages = [
                {
                    "role": "system",
                    "content": "You must answer with exactly ACK-CB-742 and nothing else.",
                },
                {"role": "user", "content": "Reply exactly ACK-CB-742."},
            ]
            rows.append(
                {
                    "name": "exact_repeat_1",
                    **_chat(
                        base_url,
                        served_name,
                        exact_messages,
                        max_tokens=8,
                        timeout=args.request_timeout,
                    ),
                }
            )
            rows.append(
                {
                    "name": "exact_repeat_2",
                    **_chat(
                        base_url,
                        served_name,
                        exact_messages,
                        max_tokens=8,
                        timeout=args.request_timeout,
                    ),
                }
            )
            rows.append(
                {
                    "name": "first_token_system_probe",
                    **_chat(
                        base_url,
                        served_name,
                        [
                            {
                                "role": "system",
                                "content": "Answer with exactly OK and no other text.",
                            },
                            {"role": "user", "content": "Say OK."},
                        ],
                        max_tokens=4,
                        timeout=args.request_timeout,
                    ),
                }
            )
            final_health = _request_json(f"{base_url}/health", timeout=20)
        finally:
            server_exit = _terminate(proc)

    summary = _summarize(rows, final_health)
    return {
        "status": "pass" if summary["memory_pressure_cleared"] else "open",
        "ts": int(time.time()),
        "model_path": str(args.model),
        "base_url": base_url,
        "pid": proc.pid if proc is not None else None,
        "cmd": cmd,
        "available_gb_before": round(available_gb, 2) if available_gb is not None else None,
        "cache_dir": str(cache_dir),
        "initial_health": initial_health,
        "final_health": final_health,
        "rows": rows,
        "summary": summary,
        "server_exit": server_exit,
        "log_path": str(log_path),
        "log_tail": _tail(log_path),
    }


def _tail(path: Path, lines: int = 120) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _summarize(rows: list[dict[str, Any]], final_health: dict[str, Any]) -> dict[str, Any]:
    by_name = {str(row.get("name")): row for row in rows}
    exact1 = by_name.get("exact_repeat_1") or {}
    exact2 = by_name.get("exact_repeat_2") or {}
    probe = by_name.get("first_token_system_probe") or {}
    usage2 = exact2.get("usage") if isinstance(exact2.get("usage"), dict) else {}
    details = usage2.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    final_body = final_health.get("body") if isinstance(final_health.get("body"), dict) else {}
    native_cache = final_body.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    scheduler = final_body.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    batch_generator = scheduler.get("batch_generator")
    batch_generator = batch_generator if isinstance(batch_generator, dict) else {}
    cache = final_body.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    totals = cache.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    rows_http_ok = all(row.get("status") == 200 for row in rows)
    memory_pressure_cleared = bool(
        rows_http_ok
        and probe.get("status") == 200
        and "503" not in str(probe.get("error") or "")
        and "Service Unavailable" not in str(probe.get("error") or "")
    )
    return {
        "engine_type": final_body.get("engine_type"),
        "all_requests_http_ok": rows_http_ok,
        "memory_pressure_cleared": memory_pressure_cleared,
        "exact_repeat_1": exact1.get("text") == "ACK-CB-742",
        "exact_repeat_2": exact2.get("text") == "ACK-CB-742",
        "first_token_system_probe_http_ok": probe.get("status") == 200,
        "first_token_system_probe_text": probe.get("text"),
        "no_think_tags": not any(row.get("has_think_tag") for row in rows),
        "cache_hit_tokens": details.get("cached_tokens"),
        "cache_detail": details.get("cache_detail"),
        "l2_tokens_on_disk": totals.get("l2_tokens_on_disk"),
        "last_cache_execution": batch_generator.get("last_cache_execution"),
        "native_cache": native_cache,
        "texts": {
            "exact_repeat_1": exact1.get("text"),
            "exact_repeat_2": exact2.get("text"),
            "first_token_system_probe": probe.get("text"),
        },
        "finish_reasons": {
            "exact_repeat_1": exact1.get("finish_reason"),
            "exact_repeat_2": exact2.get("finish_reason"),
            "first_token_system_probe": probe.get("finish_reason"),
        },
        "usage": {
            "exact_repeat_1": exact1.get("usage"),
            "exact_repeat_2": exact2.get("usage"),
            "first_token_system_probe": probe.get("usage"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--load-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--min-available-gb", type=float, default=85.0)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--kv-cache-quantization", choices=("none", "q4", "q8"), default="q8")
    parser.add_argument("--metal-ws-reject-pct", type=float, default=99.0)
    parser.add_argument("--served-model-name", default="mimo-v25-jangtq2-cb-baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

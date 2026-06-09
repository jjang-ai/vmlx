#!/usr/bin/env python3
"""Live MiMo V2.5 cache-vs-no-cache next-token logprob proof.

This closes exactly one release question: does the current MiMo JANGTQ2 text
runtime return the same next-token distribution when prefix/paged/L2 cache is
bypassed versus when the same prompt is served from cache?

It intentionally does not score literal exactness, media, JANG_2L, UI, or N2.
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
DEFAULT_MODEL = Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2")
DEFAULT_OUT = REPO / "build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-20260609.json"
DEFAULT_CACHE_DIR = REPO / "build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-block-cache-20260609"
DEFAULT_MIN_AVAILABLE_GB = 90.0


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
        "512",
        "--prefill-step-size",
        "1024",
        "--completion-batch-size",
        "256",
        "--continuous-batching",
        "--ssm-state-cache-mb",
        "1024",
        "--max-tokens",
        "16",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
    ]
    if args.cache_mode == "paged":
        cmd.extend(
            [
                "--use-paged-cache",
                "--paged-cache-block-size",
                "256",
                "--max-cache-blocks",
                "1000",
                "--enable-block-disk-cache",
                "--block-disk-cache-dir",
                str(args.cache_dir),
                "--block-disk-cache-max-gb",
                "2",
            ]
        )
    elif args.cache_mode == "legacy":
        cmd.append("--no-memory-aware-cache")
    return cmd


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("VMLINUX_API_KEY", "")
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
            body = json.loads(raw) if raw else None
        except Exception:
            body = None
        return {
            "code": exc.code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "raw": raw,
            "body": body,
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


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            return get_json(f"http://127.0.0.1:{port}/health", timeout=3)
        except Exception as exc:  # noqa: BLE001 - wait loop records last error
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last_error!r}")


def completion_payload(args: argparse.Namespace, *, skip_prefix_cache: bool) -> dict[str, Any]:
    return {
        "model": args.served_model_name,
        "prompt": args.prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "logprobs": 10,
        "skip_prefix_cache": skip_prefix_cache,
    }


def row_from_response(mode: str, response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprobs = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
    top_logprobs = logprobs.get("top_logprobs") if isinstance(logprobs.get("top_logprobs"), list) else []
    first_top = top_logprobs[0] if top_logprobs and isinstance(top_logprobs[0], dict) else {}
    top10 = [
        {
            "id": None,
            "text": str(token),
            "logprob": float(value),
        }
        for token, value in first_top.items()
    ]
    tokens = logprobs.get("tokens") if isinstance(logprobs.get("tokens"), list) else []
    return {
        "mode": mode,
        "status_code": response.get("code"),
        "elapsed_s": response.get("elapsed_s"),
        "error": response.get("error"),
        "text": str(choice.get("text") or ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "token": tokens[0] if tokens else None,
        "token_logprob": (
            logprobs.get("token_logprobs", [None])[0]
            if isinstance(logprobs.get("token_logprobs"), list)
            and logprobs.get("token_logprobs")
            else None
        ),
        "top10": top10,
    }


def top10_signature(row: dict[str, Any]) -> list[tuple[Any, Any]]:
    return [(item.get("id"), item.get("text")) for item in row.get("top10") or []]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        return {
            "status": "skipped",
            "reason": "model_missing",
            "model": str(args.model),
        }

    preflight = memory_preflight(args.min_available_gb)
    if preflight is not None:
        preflight.update({"model": str(args.model), "artifact": str(args.out)})
        return preflight

    log_path = args.out.with_suffix(".server.log")
    cmd = build_command(args)
    telemetry: list[dict[str, Any]] = [resource_snapshot("before_launch")]
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

            url = f"http://127.0.0.1:{args.port}/v1/completions"
            response_nocache = post_json(
                url,
                completion_payload(args, skip_prefix_cache=True),
                args.request_timeout_s,
            )
            rows.append(row_from_response("no_cache_bypass", response_nocache))

            response_cache_warm = post_json(
                url,
                completion_payload(args, skip_prefix_cache=False),
                args.request_timeout_s,
            )
            rows.append(row_from_response("cache_warm_store", response_cache_warm))

            response_cache_hit = post_json(
                url,
                completion_payload(args, skip_prefix_cache=False),
                args.request_timeout_s,
            )
            rows.append(row_from_response("cache_hit", response_cache_hit))
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

    signatures = [top10_signature(row) for row in rows if row.get("top10")]
    top10_match = bool(signatures and all(sig == signatures[0] for sig in signatures[1:]))
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
    cache_detail = (
        prompt_details.get("cache_detail") if isinstance(prompt_details, dict) else None
    )
    status = (
        "pass"
        if top10_match
        and all(row.get("status_code") == 200 for row in rows)
        and all(row.get("top10") for row in rows)
        and isinstance(cached_tokens, int)
        and cached_tokens > 0
        else "fail"
    )
    return {
        "schema": "vmlx-mimo-v2-cache-vs-nocache-next-token-logprobs-v1",
        "status": status,
        "model": str(args.model),
        "served_model_name": args.served_model_name,
        "cmd": cmd,
        "port": args.port,
        "cache_mode": args.cache_mode,
        "prompt": args.prompt,
        "health": health,
        "final_health": final_health,
        "telemetry": telemetry,
        "server_log": str(log_path),
        "server_exit": server_exit,
        "top10_match": top10_match,
        "cache_hit_cached_tokens": cached_tokens,
        "cache_hit_cache_detail": cache_detail,
        "rows": rows,
        "release_boundary": (
            "This proves MiMo JANGTQ2 cache/no-cache next-token distribution "
            "equivalence only; it does not clear literal exactness, media, "
            "JANG_2L, N2, UI, or speed rows."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--port", type=int, default=8874)
    parser.add_argument("--served-model-name", default="mimo-v25-jangtq2-cache-proof")
    parser.add_argument("--prompt", default="Return exactly one word: ACK")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cache-mode",
        choices=("paged", "memory", "legacy"),
        default="paged",
        help="Cache implementation to compare against skip_prefix_cache bypass.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument("--min-available-gb", type=float, default=DEFAULT_MIN_AVAILABLE_GB)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Installed-app admin sleep/wake lifecycle probe for issue #175."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-issue175-admin-sleep-probe-installed-20260527.json")
DEFAULT_MODEL = Path("/Users/example/models/JANGQ/ZAYA1-8B-MXFP4")
DEFAULT_INSTALLED_PYTHON = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)


def classify_probe(payload: dict[str, Any]) -> dict[str, Any]:
    def wake_ok(name: str) -> bool:
        body = (payload.get(name) or {}).get("body", {})
        return (payload.get(name) or {}).get("code") == 200 and body.get("status") in {
            "active",
            "awake",
        }

    checks = {
        "initial_visible": (payload.get("initial_request") or {}).get("visible") is True,
        "soft_sleep_entered": (payload.get("soft_sleep") or {}).get("code") == 200
        and (payload.get("soft_sleep") or {}).get("body", {}).get("status") == "soft_sleep",
        "soft_wake_succeeded": wake_ok("soft_wake"),
        "visible_after_soft_wake": (payload.get("after_soft_request") or {}).get("visible")
        is True,
        "deep_sleep_entered": (payload.get("deep_sleep") or {}).get("code") == 200
        and (payload.get("deep_sleep") or {}).get("body", {}).get("status") == "deep_sleep",
        "deep_wake_succeeded": wake_ok("deep_wake"),
        "visible_after_deep_wake": (payload.get("after_deep_request") or {}).get("visible")
        is True,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {"status": "pass" if not failures else "fail", "checks": checks, "failures": failures}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {"code": int(resp.status), "body": json.loads(raw) if raw else None}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        with contextlib.suppress(Exception):
            return {"code": int(exc.code), "body": json.loads(raw)}
        return {"code": int(exc.code), "body": raw}


def _wait_health(port: int, proc: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health rc={proc.returncode}")
        try:
            result = _request_json("GET", f"http://127.0.0.1:{port}/health", timeout=3)
            if result["code"] == 200:
                return result
            last = result
        except Exception as exc:
            last = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"health timeout: {last!r}")


def _visible_from_chat_response(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    return isinstance(content, str) and bool(content.strip())


def _chat(base: str, prompt: str, timeout: float) -> dict[str, Any]:
    result = _request_json(
        "POST",
        f"{base}/v1/chat/completions",
        {
            "model": "ZAYA1-8B-MXFP4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
            "top_p": 1,
        },
        timeout=timeout,
    )
    return {**result, "ok": result["code"] == 200, "visible": _visible_from_chat_response(result.get("body"))}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    port = args.port or _free_port()
    base = f"http://127.0.0.1:{port}"
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
        "128",
        "--prefill-step-size",
        "1024",
        "--completion-batch-size",
        "128",
        "--continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "256",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "2",
        "--wake-timeout",
        str(args.wake_timeout),
    ]
    env = os.environ.copy()
    env.update({"PYTHONPATH": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/tmp", env=env, text=True)
        try:
            health_ready = _wait_health(port, proc, args.load_timeout)
            payload: dict[str, Any] = {
                "health_ready": health_ready,
                "initial_request": _chat(base, "Reply with the word READY.", args.request_timeout),
                "cache_before_sleep": _request_json("GET", f"{base}/v1/cache/stats", timeout=5),
                "soft_sleep": _request_json("POST", f"{base}/admin/soft-sleep", {}, timeout=20),
                "health_soft_sleep": _request_json("GET", f"{base}/health", timeout=5),
                "soft_wake": _request_json("POST", f"{base}/admin/wake", {}, timeout=args.wake_timeout),
                "after_soft_request": _chat(base, "Reply with the word SOFTWAKE.", args.request_timeout),
                "deep_sleep": _request_json("POST", f"{base}/admin/deep-sleep", {}, timeout=30),
                "health_deep_sleep": _request_json("GET", f"{base}/health", timeout=5),
                "deep_wake": _request_json("POST", f"{base}/admin/wake", {}, timeout=args.wake_timeout),
                "after_deep_request": _chat(base, "Reply with the word DEEPWAKE.", args.request_timeout),
                "cache_after_wake": _request_json("GET", f"{base}/v1/cache/stats", timeout=5),
            }
            payload["classification"] = classify_probe(payload)
            payload["status"] = payload["classification"]["status"]
            payload["server"] = {"cmd": cmd, "log": str(log_path), "port": port}
            return payload
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
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--load-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--wake-timeout", type=int, default=120)
    args = parser.parse_args()
    result = run_probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Live Gemma 4 12B speed gate.

This gate measures sustained decode throughput through the real vMLX HTTP
server. It is intentionally separate from the broad Gemma4 runtime audit
because short 7-20 token smoke rows include fixed request overhead and are not a
valid proof of the model's target decode speed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M"
DEFAULT_OUT = REPO / "build/current-gemma4-12b-speed-gate-20260604.json"
SAFE_SERVER_CWD = Path("/tmp")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw
            return resp.status, body, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return exc.code, body, raw
    except Exception as exc:  # noqa: BLE001 - artifact captures transport failures
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, ""


def wait_for_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "returncode": proc.returncode, "last": last}
        code, body, raw = request_json("GET", f"{base_url}/health", timeout=5)
        last = {"code": code, "body": body, "raw_tail": raw[-1000:]}
        if code == 200 and isinstance(body, dict) and body.get("status") == "healthy":
            return {"ok": True, "body": body}
        time.sleep(2)
    return {"ok": False, "last": last}


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(args.python).absolute()),
        "-B",
        "-s",
        "-P",
        "-m",
        "vmlx_engine.cli",
        "serve",
        args.model,
        "--served-model-name",
        args.served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
        "--continuous-batching",
        "--max-num-seqs",
        "1",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--enable-block-disk-cache",
        "--kv-cache-quantization",
        "q8",
        "--kv-cache-group-size",
        "64",
        "--disable-native-mtp",
    ]


def speed_prompt() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Write exactly 180 short comma-separated API testing checklist "
                "items. Do not number them. Keep going until complete."
            ),
        }
    ]


def speed_payload(model: str, variant: str, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": speed_prompt(),
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "skip_prefix_cache": True,
    }
    if variant == "temp0_topk0":
        payload.update({"temperature": 0, "top_p": 1, "top_k": 0})
    elif variant == "temp1_topk0":
        payload.update({"temperature": 1, "top_p": 0.95, "top_k": 0})
    elif variant == "temp1_topk64":
        payload.update({"temperature": 1, "top_p": 0.95, "top_k": 64})
    elif variant != "default_bundle_sampling":
        raise ValueError(f"unknown speed variant: {variant}")
    return payload


def chat_content(body: Any) -> str:
    try:
        return str(body["choices"][0]["message"].get("content") or "")
    except Exception:
        return ""


def run_speed_request(
    *,
    base_url: str,
    model: str,
    variant: str,
    max_tokens: int,
    timeout: float,
    round_index: int,
) -> dict[str, Any]:
    payload = speed_payload(model, variant, max_tokens)
    started = time.monotonic()
    code, body, raw = request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        payload,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    usage = body.get("usage") if isinstance(body, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "round": round_index,
        "variant": variant,
        "http_code": code,
        "elapsed_sec": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "wall_tps": round(completion_tokens / elapsed, 3)
        if elapsed > 0 and completion_tokens
        else 0.0,
        "finish_reason": (
            body.get("choices", [{}])[0].get("finish_reason")
            if isinstance(body, dict)
            else None
        ),
        "tail": chat_content(body)[-200:],
        "raw_tail": raw[-1000:],
    }


def summarize_results(results: list[dict[str, Any]], target_tps: float) -> dict[str, Any]:
    by_variant: dict[str, list[float]] = {}
    for row in results:
        if row.get("http_code") != 200:
            continue
        tps = float(row.get("wall_tps") or 0.0)
        if tps > 0:
            by_variant.setdefault(str(row.get("variant")), []).append(tps)
    variant_summary = {
        variant: {
            "count": len(values),
            "median_tps": round(statistics.median(values), 3),
            "min_tps": round(min(values), 3),
            "max_tps": round(max(values), 3),
            "target_tps": target_tps,
            "passes_target": statistics.median(values) >= target_tps,
        }
        for variant, values in sorted(by_variant.items())
        if values
    }
    default_summary = variant_summary.get("default_bundle_sampling")
    status = (
        "pass"
        if default_summary and default_summary.get("passes_target") is True
        else "fail"
    )
    topk_summaries = [
        variant_summary.get("temp1_topk0"),
        variant_summary.get("temp1_topk64"),
    ]
    topk_known = all(isinstance(item, dict) for item in topk_summaries)
    topk_regression = False
    if topk_known:
        topk0 = float(topk_summaries[0]["median_tps"])  # type: ignore[index]
        topk64 = float(topk_summaries[1]["median_tps"])  # type: ignore[index]
        topk_regression = abs(topk64 - topk0) > 5.0
    return {
        "status": status,
        "target_tps": target_tps,
        "variant_summary": variant_summary,
        "default_median_tps": default_summary.get("median_tps")
        if default_summary
        else None,
        "top_k_primary_regression": topk_regression,
    }


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.model).is_dir():
        return {"status": "missing", "model": args.model}
    cmd = build_command(args)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SAFE_SERVER_CWD),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        health = wait_for_health(base_url, proc, args.load_timeout)
        if not health.get("ok"):
            return {
                "status": "server_unhealthy",
                "command": cmd,
                "health": health,
                "server_log": str(log_path),
            }
        request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            {
                "model": args.served_name,
                "messages": [{"role": "user", "content": "Say READY and stop."}],
                "max_tokens": 8,
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "enable_thinking": False,
                "skip_prefix_cache": True,
            },
            timeout=120,
        )
        variants = [
            "default_bundle_sampling",
            "temp0_topk0",
            "temp1_topk0",
            "temp1_topk64",
        ]
        results: list[dict[str, Any]] = []
        for round_index in range(args.rounds):
            for variant in variants:
                results.append(
                    run_speed_request(
                        base_url=base_url,
                        model=args.served_name,
                        variant=variant,
                        max_tokens=args.max_tokens,
                        timeout=args.request_timeout,
                        round_index=round_index,
                    )
                )
        _, cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=20)
        _, capabilities, _ = request_json(
            "GET",
            f"{base_url}/v1/models/{args.served_name}/capabilities",
            timeout=20,
        )
        summary = summarize_results(results, args.target_tps)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        return {
            **summary,
            "schema": "vmlx-gemma4-12b-speed-gate-v1",
            "model": args.model,
            "served_name": args.served_name,
            "command": cmd,
            "target_tps": args.target_tps,
            "health": health.get("body"),
            "capabilities": capabilities,
            "cache_stats": cache_stats,
            "results": results,
            "server_log": str(log_path),
            "server_log_tail": log_text[-20000:],
            "diagnosis": {
                "top_k_is_primary_cause": summary["top_k_primary_regression"],
                "cache_hits_during_speed_rows": (
                    (cache_stats or {})
                    .get("scheduler_stats", {})
                    .get("cache_hit_requests", 0)
                    if isinstance(cache_stats, dict)
                    else None
                ),
                "native_cache_family": (
                    (cache_stats or {}).get("native_cache", {}).get("family")
                    if isinstance(cache_stats, dict)
                    else None
                ),
                "native_cache_schema": (
                    (cache_stats or {}).get("native_cache", {}).get("schema")
                    if isinstance(cache_stats, dict)
                    else None
                ),
            },
        }
    finally:
        terminate(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(REPO / ".venv/bin/python"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-name", default="gemma4-12b-speed-gate")
    parser.add_argument("--port", type=int, default=9032)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--target-tps", type=float, default=45.0)
    parser.add_argument(
        "--log",
        default=str(REPO / "build/current-gemma4-12b-speed-gate-server-20260604.log"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.out} status={result.get('status')} "
        f"default_median_tps={result.get('default_median_tps')}"
    )
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

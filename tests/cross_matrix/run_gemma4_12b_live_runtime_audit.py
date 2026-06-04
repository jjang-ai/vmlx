#!/usr/bin/env python3
"""Run live Gemma 4 12B vMLX runtime rows.

This is intentionally heavier than the artifact contract: it starts
``vmlx-engine serve`` and probes the real HTTP API path. It keeps modes
separate so a conservative load/generation failure is not confused with a
prefix-cache, paged-cache, L2, or TurboQuant-KV failure.
"""

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "build/current-gemma4-12b-live-runtime-audit.json"
SAFE_SERVER_CWD = Path("/tmp")


@dataclass(frozen=True)
class Row:
    name: str
    path: str
    served_name: str


@dataclass(frozen=True)
class Mode:
    name: str
    extra_args: tuple[str, ...]
    expect_cache_reuse: bool = False
    expect_l2: bool = False
    expect_tq_kv: bool = False


ROWS: dict[str, Row] = {
    "mxfp4": Row(
        "mxfp4",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP4",
        "gemma4-12b-mxfp4-live",
    ),
    "mxfp8": Row(
        "mxfp8",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8",
        "gemma4-12b-mxfp8-live",
    ),
    "mxfp8_attnfp16": Row(
        "mxfp8_attnfp16",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE",
        "gemma4-12b-mxfp8-attnfp16-live",
    ),
    "mxfp8_attnfp16_l024": Row(
        "mxfp8_attnfp16_l024",
        "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE",
        "gemma4-12b-mxfp8-attnfp16-l024-live",
    ),
    "jang4m": Row(
        "jang4m",
        "/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
        "gemma4-12b-jang4m-live",
    ),
}

MODES: dict[str, Mode] = {
    "conservative": Mode(
        "conservative",
        (
            "--no-continuous-batching",
            "--disable-prefix-cache",
            "--kv-cache-quantization",
            "none",
            "--disable-native-mtp",
        ),
    ),
    "prefix_paged_l2": Mode(
        "prefix_paged_l2",
        (
            "--continuous-batching",
            "--max-num-seqs",
            "1",
            "--enable-prefix-cache",
            "--use-paged-cache",
            "--paged-cache-block-size",
            "64",
            "--enable-block-disk-cache",
            "--kv-cache-quantization",
            "none",
            "--disable-native-mtp",
        ),
        expect_cache_reuse=True,
        expect_l2=True,
    ),
    "prefix_paged_tq_q8": Mode(
        "prefix_paged_tq_q8",
        (
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
        ),
        expect_cache_reuse=True,
        expect_l2=True,
        expect_tq_kv=True,
    ),
}


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> tuple[int, Any, str]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
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
    except Exception as exc:  # noqa: BLE001 - audit captures transport failures
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, ""


def _wait_for_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        if proc.poll() is not None:
            return {
                "ok": False,
                "returncode": proc.returncode,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "last": last,
            }
        code, body, _ = _request_json("GET", f"{base_url}/health", timeout=5)
        last = {"code": code, "body": body}
        if code == 200 and isinstance(body, dict):
            return {
                "ok": True,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "body": body,
            }
        time.sleep(2)
    return {
        "ok": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "last": last,
    }


def _content_from_chat(body: Any) -> str:
    try:
        return str(body["choices"][0]["message"].get("content") or "")
    except Exception:
        return ""


def _tool_calls_from_chat(body: Any) -> list[Any]:
    try:
        calls = body["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        return []
    return calls if isinstance(calls, list) else []


def _coherence_checks(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "has_visible_text": bool(text.strip()),
        "mentions_paris": "paris" in lowered,
        "no_replacement_char": "\ufffd" not in text,
        "no_channel_leak": "<|channel>" not in text and "<channel|>" not in text,
        "no_tool_markup_leak": "<|tool_call>" not in text and "<tool_call|>" not in text,
        "length": len(text),
    }


def _run_chat(base_url: str, model: str, messages: list[dict[str, Any]], max_tokens: int = 80, **extra: Any) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    started = time.monotonic()
    code, body, raw = _request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        payload=payload,
        timeout=180,
    )
    return {
        "code": code,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "body": body,
        "raw_tail": raw[-4000:],
        "content": _content_from_chat(body),
        "tool_calls": _tool_calls_from_chat(body),
    }


def _run_cache_warm(base_url: str, model: str) -> dict[str, Any]:
    payload = {
        "prompts": ["Cache warm probe. Remember codeword ORCHID-4729."],
    }
    code, body, raw = _request_json("POST", f"{base_url}/v1/cache/warm", payload=payload, timeout=180)
    return {"code": code, "body": body, "raw_tail": raw[-2000:]}


def _cache_warm_endpoint_ok(warm: dict[str, Any]) -> bool:
    body = warm.get("body")
    if warm.get("code") not in {200, 202} or not isinstance(body, dict):
        return False
    if body.get("error"):
        return False
    try:
        return int(body.get("warmed", 0)) >= 1
    except (TypeError, ValueError):
        return False


def _server_log_tail(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def build_serve_command(
    *,
    python: Path,
    row: Row,
    mode: Mode,
    port: int,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        str(python.absolute()),
        "-B",
        "-s",
        "-P",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row.path,
        "--served-model-name",
        row.served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
        *mode.extra_args,
        *extra_args,
    ]
    return cmd


def run_row(
    *,
    row: Row,
    mode: Mode,
    python: Path,
    port: int,
    load_timeout_s: float,
    request_extra_args: list[str],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / f"{row.name}_{mode.name}_server.log"
    cmd = build_serve_command(
        python=python,
        row=row,
        mode=mode,
        port=port,
        extra_args=request_extra_args,
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result: dict[str, Any] = {
        "row": row.name,
        "mode": mode.name,
        "model_path": row.path,
        "served_name": row.served_name,
        "command": cmd,
        "server_log": str(log_path),
        "status": "open",
        "checks": {},
        "failures": [],
    }
    if not Path(row.path).is_dir():
        result["status"] = "missing"
        result["failures"].append(f"model path missing: {row.path}")
        return result

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
    base_url = f"http://127.0.0.1:{port}"
    try:
        health = _wait_for_health(base_url, proc, load_timeout_s)
        result["health_initial"] = health
        if not health.get("ok"):
            result["failures"].append("server_did_not_become_healthy")
            return result

        code, models, raw_models = _request_json("GET", f"{base_url}/v1/models", timeout=20)
        result["models"] = {"code": code, "body": models, "raw_tail": raw_models[-2000:]}
        cap_code, caps, raw_caps = _request_json(
            "GET",
            f"{base_url}/v1/models/{row.served_name}/capabilities",
            timeout=20,
        )
        result["capabilities"] = {"code": cap_code, "body": caps, "raw_tail": raw_caps[-4000:]}
        stats0_code, stats0, raw_stats0 = _request_json("GET", f"{base_url}/v1/cache/stats", timeout=20)
        result["cache_stats_initial"] = {
            "code": stats0_code,
            "body": stats0,
            "raw_tail": raw_stats0[-3000:],
        }

        chat1 = _run_chat(
            base_url,
            row.served_name,
            [{"role": "user", "content": "Answer in one short sentence: what is the capital of France?"}],
            max_tokens=48,
        )
        result["chat_basic"] = chat1
        result["chat_basic_checks"] = _coherence_checks(chat1.get("content") or "")

        chat2_messages = [
            {"role": "user", "content": "Remember this exact codeword for the next turn: ORCHID-4729."},
        ]
        chat2a = _run_chat(base_url, row.served_name, chat2_messages, max_tokens=72)
        chat2b = _run_chat(
            base_url,
            row.served_name,
            chat2_messages
            + [{"role": "assistant", "content": chat2a.get("content") or ""}]
            + [{"role": "user", "content": "What exact codeword did I give you?"}],
            max_tokens=72,
        )
        result["chat_memory_turn1"] = chat2a
        result["chat_memory_turn2"] = chat2b

        reasoning = _run_chat(
            base_url,
            row.served_name,
            [{"role": "user", "content": "Think briefly if needed, then answer only the number: 19 + 26."}],
            max_tokens=96,
            extra_body={"enable_thinking": True},
        )
        result["chat_reasoning_on"] = reasoning

        tool = _run_chat(
            base_url,
            row.served_name,
            [{"role": "user", "content": "Use the available tool to record the codeword ORCHID-4729."}],
            max_tokens=128,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "record_fact",
                        "description": "Record a fact.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "codeword": {"type": "string"},
                            },
                            "required": ["codeword"],
                        },
                    },
                }
            ],
        )
        result["chat_tool_required"] = tool

        warm = _run_cache_warm(base_url, row.served_name)
        result["cache_warm"] = warm
        repeat1 = _run_chat(
            base_url,
            row.served_name,
            [{"role": "user", "content": "Repeatable cache probe. Say exactly: CACHE-ACK ORCHID-4729."}],
            max_tokens=64,
        )
        repeat2 = _run_chat(
            base_url,
            row.served_name,
            [{"role": "user", "content": "Repeatable cache probe. Say exactly: CACHE-ACK ORCHID-4729."}],
            max_tokens=64,
        )
        result["cache_repeat_1"] = repeat1
        result["cache_repeat_2"] = repeat2
        stats1_code, stats1, raw_stats1 = _request_json("GET", f"{base_url}/v1/cache/stats", timeout=20)
        result["cache_stats_final"] = {
            "code": stats1_code,
            "body": stats1,
            "raw_tail": raw_stats1[-5000:],
        }
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=20)
        result["server_returncode"] = proc.returncode
        result["server_log_tail"] = _server_log_tail(log_path)

    checks = result["checks"]
    checks["health_ok"] = bool((result.get("health_initial") or {}).get("ok"))
    checks["models_ok"] = result.get("models", {}).get("code") == 200
    checks["capabilities_ok"] = result.get("capabilities", {}).get("code") == 200
    checks["cache_stats_ok"] = result.get("cache_stats_final", {}).get("code") == 200
    checks["basic_chat_ok"] = (
        result.get("chat_basic", {}).get("code") == 200
        and (result.get("chat_basic_checks") or {}).get("has_visible_text") is True
        and (result.get("chat_basic_checks") or {}).get("mentions_paris") is True
        and (result.get("chat_basic_checks") or {}).get("no_channel_leak") is True
        and (result.get("chat_basic_checks") or {}).get("no_tool_markup_leak") is True
    )
    memory_text = str(result.get("chat_memory_turn2", {}).get("content") or "")
    checks["multi_turn_recall_ok"] = (
        result.get("chat_memory_turn2", {}).get("code") == 200
        and "ORCHID-4729" in memory_text
    )
    reasoning_text = str(result.get("chat_reasoning_on", {}).get("content") or "")
    checks["reasoning_request_visible_answer_ok"] = (
        result.get("chat_reasoning_on", {}).get("code") == 200
        and "45" in reasoning_text
        and "<|channel>" not in reasoning_text
    )
    calls = result.get("chat_tool_required", {}).get("tool_calls") or []
    checks["tool_required_structured_ok"] = (
        result.get("chat_tool_required", {}).get("code") == 200
        and len(calls) >= 1
        and str(calls[0]).find("record_fact") >= 0
        and str(calls[0]).find("ORCHID-4729") >= 0
    )
    checks["cache_warm_endpoint_ok"] = (
        True
        if not mode.expect_cache_reuse
        else _cache_warm_endpoint_ok(result.get("cache_warm", {}))
    )
    final_stats = result.get("cache_stats_final", {}).get("body")
    final_stats_text = json.dumps(final_stats, sort_keys=True) if isinstance(final_stats, dict) else str(final_stats)
    checks["cache_reuse_observed_if_expected"] = (
        True
        if not mode.expect_cache_reuse
        else ("cached_tokens" in final_stats_text or "tokens_saved" in final_stats_text or "cache_hits" in final_stats_text)
    )
    checks["block_l2_observed_if_expected"] = (
        True
        if not mode.expect_l2
        else ("disk" in final_stats_text.lower() or "l2" in final_stats_text.lower())
    )
    checks["tq_kv_observed_if_expected"] = (
        True
        if not mode.expect_tq_kv
        else ("turboquant" in final_stats_text.lower() or "kv_cache_quantization" in final_stats_text.lower())
    )

    for name, ok in checks.items():
        if not ok:
            result["failures"].append(name)
    result["status"] = "pass" if not result["failures"] else "fail"
    return result


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    rows = [ROWS[name] for name in args.rows]
    modes = [MODES[name] for name in args.modes]
    artifact_dir = args.out.parent / (args.out.stem + "_artifacts")
    results: dict[str, Any] = {}
    port = args.port
    for row in rows:
        for mode in modes:
            key = f"{row.name}_{mode.name}"
            results[key] = run_row(
                row=row,
                mode=mode,
                python=args.python,
                port=port,
                load_timeout_s=args.load_timeout,
                request_extra_args=args.serve_extra_arg or [],
                artifact_dir=artifact_dir,
            )
            port += 1
            if results[key].get("status") != "pass" and args.stop_on_failure:
                break
    checks = {
        "all_rows_passed": all(result.get("status") == "pass" for result in results.values()),
        "at_least_one_row_ran": bool(results),
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "rows": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--rows", nargs="+", choices=sorted(ROWS), default=["mxfp4"])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["conservative"])
    parser.add_argument("--port", type=int, default=8934)
    parser.add_argument("--load-timeout", type=float, default=420)
    parser.add_argument("--serve-extra-arg", action="append")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    for key, result in artifact["rows"].items():
        print(f"{key}: status={result.get('status')} failures={result.get('failures')}")
        if result.get("server_log"):
            print(f"  log={result['server_log']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

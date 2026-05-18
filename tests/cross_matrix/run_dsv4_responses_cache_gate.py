#!/usr/bin/env python3
"""Live DSV4 Flash Responses/cache gate.

This is the Responses API sibling of ``run_dsv4_long_context_gate.py``. It
keeps DSV4-specific cache proof reusable instead of leaving it as ad-hoc JSON:

- real source/app server process
- ``/v1/responses`` with ``previous_response_id``
- per-run nonce so stale block-disk cache cannot fake a fresh store turn
- explicit no-cache full-prompt control
- DSV4 native composite health/status capture
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = (
    REPO
    / "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DEFAULT_MODEL = (
    "/Users/example/models/JANGQ/"
    "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
)
DEFAULT_OUT = REPO / "docs/internal/release-gates/dsv4_responses_cache_gate_latest.json"


def post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def iter_sse_json_lines(lines: Iterable[bytes | str]) -> Iterable[dict[str, Any]]:
    """Yield JSON payloads from an SSE byte/string line iterator."""
    data_lines: list[str] = []
    for raw_line in lines:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, (bytes, bytearray))
            else str(raw_line)
        )
        line = line.rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[5:].lstrip()
            if data == "[DONE]":
                continue
            data_lines.append(data)
            continue
        if line == "" and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            yield json.loads(payload)
    if data_lines:
        yield json.loads("\n".join(data_lines))


def stream_responses(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    """Run a streaming Responses request and capture real TTFT plus final usage."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage_obj: dict[str, Any] = {}
    completed_status: str | None = None
    event_count = 0
    error_events: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for event in iter_sse_json_lines(response):
            event_count += 1
            event_type = str(event.get("type") or "")
            delta = ""
            if event_type == "response.output_text.delta":
                delta = str(event.get("delta") or "")
                if delta:
                    content_parts.append(delta)
            elif event_type == "response.reasoning_summary_text.delta":
                delta = str(event.get("delta") or "")
                if delta:
                    reasoning_parts.append(delta)
            elif event_type == "response.usage":
                usage_obj = event.get("usage") or usage_obj
            elif event_type == "response.completed":
                completed = event.get("response") or {}
                completed_status = completed.get("status")
                usage_obj = completed.get("usage") or usage_obj
                if not content_parts and isinstance(completed.get("output_text"), str):
                    content_parts.append(completed["output_text"])
            elif event_type == "error":
                error_events.append(event)

            if ttft is None and delta:
                ttft = time.perf_counter() - t0

    wall = time.perf_counter() - t0
    text = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    out_tokens = int(usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0)
    return {
        "status": completed_status or ("error" if error_events else None),
        "wall_seconds": wall,
        "ttft_seconds": ttft,
        "output_tokens": out_tokens,
        "tok_s_wall": out_tokens / wall if wall else 0.0,
        "content": text,
        "reasoning": reasoning,
        "content_head": text[:900],
        "content_tail": text[-900:],
        "reasoning_head": reasoning[:900],
        "reasoning_tail": reasoning[-900:],
        "has_cerulean": "CERULEAN" in text.upper(),
        "has_45": "45" in text,
        "has_ada": "ADA" in text.upper() and "LOVELACE" in text.upper(),
        "loopish": re.search(r"(.{24,160})\1\1", text + "\n" + reasoning, re.S)
        is not None,
        "usage": usage_obj,
        "event_count": event_count,
        "errors": error_events,
    }


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            return get_json(f"http://127.0.0.1:{port}/health", timeout=2)
        except Exception as exc:  # noqa: BLE001 - live diagnostic script
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last_error!r}")


def make_long_context(target_words: int = 3400) -> str:
    anchors = (
        "ANCHOR COLOR = CERULEAN. ANCHOR NUMBER = 45. "
        "ANCHOR PERSON = ADA LOVELACE. ANCHOR CITY = KYOTO."
    )
    subjects = [
        "archives",
        "compilers",
        "weather stations",
        "matrix ledgers",
        "router traces",
        "sliding windows",
        "compressed pools",
        "local attention spans",
        "prefix records",
        "storage journals",
    ]
    verbs = ["summarize", "compare", "index", "calibrate", "describe", "sequence"]
    parts = [anchors]
    i = 0
    while len(" ".join(parts).split()) < target_words:
        parts.append(
            f"Section {i:03d}: The notes {verbs[i % len(verbs)]} "
            f"{subjects[i % len(subjects)]} for diagnostic passage {1000 + i}. "
            "This is context, not an instruction, and it does not modify the "
            f"anchor facts. Cross-reference {i % 17}-{(i * 7) % 29}."
        )
        i += 1
    parts.append(anchors)
    return "\n".join(parts)


def extract_output_text(obj: dict[str, Any]) -> str:
    if isinstance(obj.get("output_text"), str):
        return obj["output_text"]
    parts: list[str] = []
    for item in obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "output_text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") not in {"output_text", "text"}:
                continue
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def extract_reasoning(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"reasoning", "reasoning_summary"}:
            continue
        for key in ("summary", "text", "content"):
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                        parts.append(entry["text"])
    return "\n".join(parts)


def usage(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("usage") or {}


def cached_tokens(obj: dict[str, Any]) -> int:
    details = (
        usage(obj).get("input_tokens_details")
        or usage(obj).get("prompt_tokens_details")
        or {}
    )
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cache_detail(obj: dict[str, Any]) -> str:
    details = (
        usage(obj).get("input_tokens_details")
        or usage(obj).get("prompt_tokens_details")
        or {}
    )
    return str(details.get("cache_detail") or "")


def response_case(obj: dict[str, Any], wall: float) -> dict[str, Any]:
    text = extract_output_text(obj)
    reasoning = extract_reasoning(obj)
    out_tokens = int(usage(obj).get("output_tokens") or usage(obj).get("completion_tokens") or 0)
    return {
        "id": obj.get("id"),
        "status": obj.get("status"),
        "wall_seconds": wall,
        "output_tokens": out_tokens,
        "tok_s_wall": out_tokens / wall if wall else 0.0,
        "content": text,
        "reasoning": reasoning,
        "content_head": text[:900],
        "content_tail": text[-900:],
        "reasoning_head": reasoning[:900],
        "reasoning_tail": reasoning[-900:],
        "has_cerulean": "CERULEAN" in text.upper(),
        "has_45": "45" in text,
        "has_ada": "ADA" in text.upper() and "LOVELACE" in text.upper(),
        "loopish": re.search(r"(.{24,160})\1\1", text + "\n" + reasoning, re.S)
        is not None,
        "usage": usage(obj),
        "raw_status": obj.get("status"),
    }


def responses(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    t0 = time.perf_counter()
    obj = post_json(url, payload, timeout=timeout)
    return response_case(obj, time.perf_counter() - t0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    logs = out.parent / "dsv4-responses-logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"dsv4-responses-{int(time.time())}.log"
    model_name = "dsv4_responses"
    cmd = [
        str(args.python),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--timeout",
        "600",
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
        "deepseek",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_r1",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "256",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--stream-interval",
        "1",
        "--max-tokens",
        "32768",
        "--served-model-name",
        model_name,
    ]
    env = dict(os.environ)
    for key in (
        "JANGTQ_MPP_NAX",
        "JANGTQ_MPP_NAX_DISABLE",
        "JANGTQ_MPP_NAX_STRICT",
        "JANGTQ_MPP_DENSE",
        "JANGTQ_MPP_DENSE_STRICT",
        "JANGTQ_DISABLE_DSV4_STREAM_LOAD",
        "JANGTQ_DISABLE_DSV4_FAST_LOAD",
        "VMLX_DSV4_HARD_REP_BLOCK",
        "VMLINUX_DSV4_HARD_REP_BLOCK",
        "VMLX_DSV4_FORCE_DIRECT_RAIL",
        "VMLX_DSV4_RAW_MAX",
    ):
        env.pop(key, None)
    env["DSV4_LONG_CTX"] = "1"
    env["DSV4_POOL_QUANT"] = "1" if args.pool_quant else "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("VMLX_METAL_WS_REJECT_PCT", "98")

    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    try:
        health0 = wait_health(args.port, proc, args.timeout)
        url = f"http://127.0.0.1:{args.port}/v1/responses"
        run_id = f"dsv4-responses-gate-{int(time.time() * 1000)}-{os.getpid()}"
        long_context = (
            make_long_context(args.words)
            + f"\n\nGATE RUN ID = {run_id}. "
            "This nonce is diagnostic-only and does not modify the anchor facts."
        )
        store_prompt = long_context + "\n\nStore the anchor facts. Reply exactly STORED."
        follow_prompt = "Recall the anchors. Answer exactly: COLOR / SUM / PERSON."
        full_prompt = (
            long_context
            + "\n\n"
            + follow_prompt
            + " Use CERULEAN, 45, and ADA LOVELACE if those are the anchors."
        )

        store_turn = responses(
            url,
            {
                "model": model_name,
                "input": store_prompt,
                "store": True,
                "stream": False,
                "max_output_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        previous_response_follow = responses(
            url,
            {
                "model": model_name,
                "input": follow_prompt,
                "previous_response_id": store_turn["id"],
                "store": True,
                "stream": False,
                "max_output_tokens": 192,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        stream_previous_response_follow = stream_responses(
            url,
            {
                "model": model_name,
                "input": follow_prompt,
                "previous_response_id": store_turn["id"],
                "store": False,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_output_tokens": 192,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        explicit_no_cache_full_prompt = responses(
            url,
            {
                "model": model_name,
                "input": full_prompt,
                "store": False,
                "stream": False,
                "max_output_tokens": 192,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_prefix_cache": True,
            },
            timeout=600,
        )
        health1 = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)

        result = {
            "status": "pass",
            "model": args.model,
            "cmd": cmd,
            "env": {
                "DSV4_LONG_CTX": env["DSV4_LONG_CTX"],
                "DSV4_POOL_QUANT": env["DSV4_POOL_QUANT"],
                "VMLX_METAL_WS_REJECT_PCT": env.get("VMLX_METAL_WS_REJECT_PCT"),
            },
            "run_id": run_id,
            "log_path": str(log_path),
            "health_before": health0,
            "health_after": health1,
            "cases": {
                "store_turn": store_turn,
                "previous_response_follow": previous_response_follow,
                "stream_previous_response_follow": stream_previous_response_follow,
                "explicit_no_cache_full_prompt": explicit_no_cache_full_prompt,
            },
        }

        notes: list[str] = []
        native = health1.get("native_cache") or (
            (health1.get("cache") or {}).get("native") or {}
        )
        if native.get("family") != "deepseek_v4":
            notes.append("health: missing deepseek_v4 native cache family")
        if native.get("schema") != "deepseek_v4_v7":
            notes.append("health: missing deepseek_v4_v7 schema")
        if native.get("cache_type") != "native_composite":
            notes.append("health: missing native_composite cache_type")
        if native.get("generic_turboquant_kv", {}).get("enabled") is not False:
            notes.append("health: generic TurboQuant KV not disabled")
        if native.get("pool_quant", {}).get("enabled") is not bool(args.pool_quant):
            notes.append("health: pool_quant setting did not match gate input")
        if cached_tokens(store_turn) or cache_detail(store_turn):
            notes.append(
                "store_turn: unexpected prefix cache hit; per-run nonce should make store fresh"
            )
        if cached_tokens(previous_response_follow) <= 0:
            notes.append("previous_response_follow: missing cached_tokens evidence")
        if "dsv4" not in cache_detail(previous_response_follow):
            notes.append("previous_response_follow: missing dsv4 cache_detail")
        if stream_previous_response_follow.get("ttft_seconds") is None:
            notes.append("stream_previous_response_follow: missing TTFT")
        if cached_tokens(stream_previous_response_follow) <= 0:
            notes.append("stream_previous_response_follow: missing cached_tokens evidence")
        if "dsv4" not in cache_detail(stream_previous_response_follow):
            notes.append("stream_previous_response_follow: missing dsv4 cache_detail")
        if cached_tokens(explicit_no_cache_full_prompt) or cache_detail(
            explicit_no_cache_full_prompt
        ):
            notes.append("explicit_no_cache_full_prompt: unexpected cache usage")
        for name, case in result["cases"].items():
            if name == "store_turn":
                continue
            if case["loopish"]:
                notes.append(f"{name}: loopish")
            if not (case["has_cerulean"] and case["has_45"] and case["has_ada"]):
                notes.append(f"{name}: missing anchor")
        if notes:
            result["status"] = "review"
            result["notes"] = notes
        return result
    except Exception as exc:  # noqa: BLE001 - diagnostic gate
        return {
            "status": "error",
            "error": repr(exc),
            "model": args.model,
            "cmd": cmd,
            "log_path": str(log_path),
            "log_tail": log_path.read_text(errors="replace")[-16000:]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--port", type=int, default=8841)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--words", type=int, default=3400)
    parser.add_argument("--pool-quant", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"created_at": time.time(), **result}, indent=2))
    print(
        f"[dsv4-responses] status={result.get('status')} notes={result.get('notes')} "
        f"log={result.get('log_path')} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

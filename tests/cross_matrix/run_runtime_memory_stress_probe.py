#!/usr/bin/env python3
"""Measured live RAM/cache stress probe for release runtime rows.

This harness is intentionally about resource accounting, not quality scoring. It
starts one vMLX engine row, sends staged requests, and writes health, cache,
process RSS, system memory, and abort-threshold evidence after each stage.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SAFE_SERVER_CWD = Path("/tmp")
DEFAULT_PY = REPO / ".venv/bin/python"


@dataclass(frozen=True)
class Row:
    name: str
    path: str
    is_mllm: bool = False
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    cache_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


COMMON_PAGED_CACHE = [
    "--use-paged-cache",
    "--paged-cache-block-size",
    "64",
    "--max-cache-blocks",
    "501",
    "--enable-block-disk-cache",
    "--block-disk-cache-max-gb",
    "10",
]


ROWS: dict[str, Row] = {
    "dsv4_jangtq_k": Row(
        "dsv4_jangtq_k",
        "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        tool_parser="deepseek",
        reasoning_parser="deepseek_r1",
        cache_args=[
            "--dsv4-enable-prefix-cache",
            "--use-paged-cache",
            "--paged-cache-block-size",
            "256",
            "--enable-block-disk-cache",
            "--block-disk-cache-max-gb",
            "10",
        ],
        env={"DSV4_POOL_QUANT": "1"},
    ),
    "qwen36_27b_mxfp4_mtp": Row(
        "qwen36_27b_mxfp4_mtp",
        "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP4-MTP",
        is_mllm=True,
        tool_parser="qwen",
        reasoning_parser="qwen3",
        cache_args=COMMON_PAGED_CACHE
        + ["--native-mtp-depth", "3", "--native-mtp-sampling-policy", "deterministic-defaults"],
    ),
    "gemma4_26b_jang4m": Row(
        "gemma4_26b_jang4m",
        "/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        is_mllm=True,
        tool_parser="gemma4",
        reasoning_parser="gemma4",
        cache_args=COMMON_PAGED_CACHE,
    ),
    "gemma4_31b_jang4m_mtp": Row(
        "gemma4_31b_jang4m_mtp",
        "/Users/example/models/JANGQ/Gemma-4-31B-it-JANG_4M-MTP",
        is_mllm=True,
        tool_parser="gemma4",
        reasoning_parser="gemma4",
        cache_args=COMMON_PAGED_CACHE
        + ["--native-mtp-depth", "3", "--native-mtp-sampling-policy", "deterministic-defaults"],
    ),
    "minimax_m27_small_jangtq": Row(
        "minimax_m27_small_jangtq",
        "/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        tool_parser="minimax",
        reasoning_parser="minimax_m2",
        cache_args=COMMON_PAGED_CACHE,
    ),
}


def build_clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    if extra:
        env.update(extra)
    return env


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --serve-env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --serve-env {item!r}; empty key")
        env[key] = value
    return env


def _round_http_seconds(value: float) -> float:
    return round(max(value, 0.0), 6)


def http_json_timed(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any, dict[str, float]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            opened = time.perf_counter()
            raw_bytes = resp.read()
            read_done = time.perf_counter()
            raw = raw_bytes.decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw else None
            parsed_done = time.perf_counter()
            return resp.status, parsed, {
                "open_seconds": _round_http_seconds(opened - started),
                "read_seconds": _round_http_seconds(read_done - opened),
                "json_seconds": _round_http_seconds(parsed_done - read_done),
                "total_seconds": _round_http_seconds(parsed_done - started),
            }
    except urllib.error.HTTPError as exc:
        opened = time.perf_counter()
        raw = exc.read().decode("utf-8", errors="replace")
        read_done = time.perf_counter()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        parsed_done = time.perf_counter()
        return exc.code, parsed, {
            "open_seconds": _round_http_seconds(opened - started),
            "read_seconds": _round_http_seconds(read_done - opened),
            "json_seconds": _round_http_seconds(parsed_done - read_done),
            "total_seconds": _round_http_seconds(parsed_done - started),
        }


def _merge_stream_chunk(target: dict[str, Any], chunk: dict[str, Any]) -> bool:
    text_delta = ""
    event_type = str(chunk.get("type") or "")
    delta = chunk.get("delta")
    if event_type in {"response.output_text.delta", "response.refusal.delta"} and isinstance(delta, str):
        text_delta += delta
    choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
    if choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        content = delta.get("content")
        if isinstance(content, str):
            text_delta += content
    if text_delta:
        target.setdefault("choices", [{"message": {"content": ""}}])
        message = target["choices"][0].setdefault("message", {})
        message["content"] = str(message.get("content") or "") + text_delta
    output_text = chunk.get("output_text")
    if isinstance(output_text, str):
        target["output_text"] = str(target.get("output_text") or "") + output_text
        text_delta += output_text
    elif event_type in {"response.output_text.delta", "response.refusal.delta"} and text_delta:
        target["output_text"] = str(target.get("output_text") or "") + text_delta
    output = chunk.get("output")
    if isinstance(output, list):
        target.setdefault("output", [])
        target["output"].extend(output)
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        target["usage"] = usage
    response = chunk.get("response")
    if event_type == "response.completed" and isinstance(response, dict):
        completed_usage = response.get("usage")
        if isinstance(completed_usage, dict):
            target["usage"] = completed_usage
        completed_text = response.get("output_text")
        if isinstance(completed_text, str) and not target.get("output_text"):
            target["output_text"] = completed_text
        completed_output = response.get("output")
        if isinstance(completed_output, list) and not target.get("output"):
            target["output"] = completed_output
        completed_status = response.get("status")
        if completed_status is not None:
            target["status"] = completed_status
    return bool(text_delta)


def http_json_stream_timed(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any, dict[str, float | int]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    parsed: dict[str, Any] = {}
    first_token_time: float | None = None
    last_text_time: float | None = None
    last_event_time: float | None = None
    chunk_count = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        opened = time.perf_counter()
        for raw_line in resp:
            now = time.perf_counter()
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                last_event_time = now
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            had_text = _merge_stream_chunk(parsed, chunk)
            if had_text:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = now
                last_text_time = now
            last_event_time = now
        done = time.perf_counter()
    if first_token_time is None:
        first_token_time = done
    if last_text_time is None:
        last_text_time = first_token_time
    if last_event_time is None:
        last_event_time = done
    return resp.status, parsed, {
        "open_seconds": _round_http_seconds(opened - started),
        "ttft_seconds": _round_http_seconds(first_token_time - started),
        "stream_read_seconds": _round_http_seconds(done - opened),
        "post_first_token_seconds": _round_http_seconds(done - first_token_time),
        "post_first_visible_token_seconds": _round_http_seconds(
            last_text_time - first_token_time
        ),
        "post_last_visible_to_done_seconds": _round_http_seconds(done - last_text_time),
        "total_seconds": _round_http_seconds(done - started),
        "chunk_count": chunk_count,
    }


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    code, parsed, _timing = http_json_timed(method, url, body, timeout)
    return code, parsed


def wait_health(port: int, proc: subprocess.Popen[str], timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health: rc={proc.returncode}")
        try:
            code, body = http_json("GET", f"http://127.0.0.1:{port}/health", timeout=2)
            if code == 200 and isinstance(body, dict):
                return body
            last = body
        except Exception as exc:  # noqa: BLE001 - diagnostic polling only
            last = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"health timeout on port {port}: {last!r}")


def process_tree_rss_kb(root_pid: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,ppid,rss,comm,args"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}
    children: dict[int, list[int]] = {}
    rows: dict[int, dict[str, Any]] = {}
    for line in proc.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss = int(parts[2])
        except ValueError:
            continue
        rows[pid] = {
            "pid": pid,
            "ppid": ppid,
            "rss_kb": rss,
            "comm": parts[3],
            "args": parts[4] if len(parts) > 4 else "",
        }
        children.setdefault(ppid, []).append(pid)
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    selected = [rows[pid] for pid in sorted(seen) if pid in rows]
    return {
        "root_pid": root_pid,
        "pids": [row["pid"] for row in selected],
        "rss_kb": sum(int(row["rss_kb"]) for row in selected),
        "rss_mb": round(sum(int(row["rss_kb"]) for row in selected) / 1024, 1),
        "processes": selected,
    }


def vm_stat_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["vm_stat"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}
    page_size = 16384
    out: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                pass
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            out[key.strip()] = int(digits)
    out["page_size"] = page_size
    for key in ("Pages free", "Pages active", "Pages inactive", "Pages wired down", "Pages occupied by compressor"):
        if key in out:
            out[f"{key.lower().replace(' ', '_')}_mb"] = round(out[key] * page_size / (1024**2), 1)
    return out


def make_prompt(
    approx_tokens: int,
    *,
    mode: str = "default",
    max_tokens: int = 0,
) -> str:
    if mode == "speed_floor":
        target = max(int(max_tokens or 0), 64)
        seed = (
            f"Write exactly {target} numbered audit lines. Do not stop early. "
            "Each line must be concise English and must include one of these "
            "exact identifiers without changing spelling: DSV4CacheBudget, "
            "qwen_mtp_media_history, GemmaCacheProbe. Continue until the token "
            "budget stops you. "
        )
        words = seed.split()
        repeated: list[str] = []
        while len(repeated) < approx_tokens:
            repeated.extend(words)
        return " ".join(repeated[:approx_tokens])
    seed = (
        "Audit this runtime memory test. Preserve exact identifiers like "
        "DSV4CacheBudget, qwen_mtp_media_history, and GemmaCacheProbe. "
    )
    words = seed.split()
    if approx_tokens <= len(words):
        return " ".join(words[:approx_tokens])
    repeated = []
    while len(repeated) < approx_tokens:
        repeated.extend(words)
    return " ".join(repeated[:approx_tokens])


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_request(
    row: Row,
    prompt: str,
    max_tokens: int,
    route: str,
    image_path: Path | None,
    retained_image_turns: int,
    enable_thinking: bool | None = None,
    max_thinking_tokens: int | None = None,
    cache_salt: str | None = None,
    skip_prefix_cache: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if image_path:
        data_url = image_data_url(image_path)
        for idx in range(max(1, retained_image_turns)):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image history turn {idx + 1}. {prompt}"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
    else:
        messages.append({"role": "user", "content": prompt})
    if route == "responses":
        request = {
            "model": Path(row.path).name,
            "input": messages,
            "stream": stream,
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    else:
        request = {
            "model": Path(row.path).name,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    if stream:
        request["stream_options"] = {"include_usage": True}
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        request["enable_thinking"] = enable_thinking
        template_kwargs["enable_thinking"] = enable_thinking
    if max_thinking_tokens is not None:
        request["max_thinking_tokens"] = int(max_thinking_tokens)
        template_kwargs["thinking_budget"] = int(max_thinking_tokens)
    if cache_salt:
        request["cache_salt"] = cache_salt
    if skip_prefix_cache:
        request["skip_prefix_cache"] = True
    if template_kwargs:
        request["chat_template_kwargs"] = template_kwargs
    return request


def redact_large_payloads(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("data:image/") or value.startswith("data:video/") or value.startswith("data:audio/"):
            prefix = value.split(",", 1)[0]
            return f"{prefix},<redacted chars={len(value)}>"
        if len(value) > 4000:
            return value[:2000] + f"...<redacted chars={len(value)}>" + value[-500:]
        return value
    if isinstance(value, list):
        return [redact_large_payloads(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_large_payloads(item) for key, item in value.items()}
    return value


def build_serve_cmd(row: Row, python: Path, port: int, timeout: int, extra_args: list[str]) -> list[str]:
    cmd = [
        str(python),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row.path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        str(timeout),
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "129",
        "--prefill-step-size",
        "2048",
        "--completion-batch-size",
        "256",
        "--continuous-batching",
        "--stream-interval",
        "1",
    ]
    if row.is_mllm:
        cmd.append("--is-mllm")
    if row.tool_parser:
        cmd += ["--tool-call-parser", row.tool_parser, "--enable-auto-tool-choice"]
    if row.reasoning_parser:
        cmd += ["--reasoning-parser", row.reasoning_parser]
    cmd += row.cache_args
    cmd += extra_args
    return cmd


def snapshot(label: str, port: int, proc: subprocess.Popen[str]) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "label": label,
        "time": time.time(),
        "process": process_tree_rss_kb(proc.pid),
        "vm_stat": vm_stat_snapshot(),
    }
    for name, path in (("health", "/health"), ("cache_stats", "/v1/cache/stats")):
        try:
            code, body = http_json("GET", f"http://127.0.0.1:{port}{path}", timeout=5)
            snap[name] = {"code": code, "body": body}
        except Exception as exc:  # noqa: BLE001
            snap[name] = {"error": repr(exc)}
    return snap


def threshold_exceeded(snap: dict[str, Any], abort_rss_gb: float, abort_active_gb: float) -> str | None:
    rss_mb = ((snap.get("process") or {}).get("rss_mb") or 0)
    if abort_rss_gb > 0 and rss_mb / 1024 >= abort_rss_gb:
        return f"process RSS {rss_mb / 1024:.1f}GB exceeded abort threshold {abort_rss_gb:.1f}GB"
    health = ((snap.get("health") or {}).get("body") or {})
    mem = health.get("memory") or {}
    active_mb = mem.get("active_mb") or 0
    if abort_active_gb > 0 and active_mb / 1024 >= abort_active_gb:
        return f"MLX active {active_mb / 1024:.1f}GB exceeded abort threshold {abort_active_gb:.1f}GB"
    return None


def extract_usage(resp: Any) -> dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return {}
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input_tokens_total")
        or 0
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("output_tokens_total")
        or 0
    )
    cached_tokens = 0
    details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    if isinstance(details, dict):
        cached_tokens = int(details.get("cached_tokens") or 0)
    try:
        prompt_tokens = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        prompt_tokens = 0
    try:
        completion_tokens = int(completion_tokens or 0)
    except (TypeError, ValueError):
        completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_detail": details.get("cache_detail") if isinstance(details, dict) else None,
    }


def add_speed_metrics(stage: dict[str, Any]) -> None:
    elapsed = float(stage.get("elapsed_s") or 0.0)
    if elapsed <= 0:
        return
    usage = extract_usage(stage.get("response"))
    stage["usage_summary"] = usage
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
    stage["speed"] = {
        "wall_seconds": elapsed,
        "prompt_tok_s_wall": round(prompt_tokens / elapsed, 3) if prompt_tokens else 0.0,
        "uncached_prompt_tok_s_wall": (
            round(uncached_prompt_tokens / elapsed, 3) if uncached_prompt_tokens else 0.0
        ),
        "decode_tok_s_wall": round(completion_tokens / elapsed, 3) if completion_tokens else 0.0,
        "cached_tokens": cached_tokens,
        "cache_detail": usage.get("cache_detail"),
    }


def add_stream_speed_metrics(stage: dict[str, Any]) -> None:
    timing = stage.get("http_timing") if isinstance(stage.get("http_timing"), dict) else {}
    post_first = float(
        timing.get("post_first_visible_token_seconds")
        or timing.get("post_first_token_seconds")
        or 0.0
    )
    if post_first <= 0:
        return
    usage = extract_usage(stage.get("response"))
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    if completion_tokens <= 1:
        decode_tps = 0.0
    else:
        decode_tps = round((completion_tokens - 1) / post_first, 3)
    stage["stream_speed"] = {
        "ttft_seconds": timing.get("ttft_seconds"),
        "post_first_token_seconds": timing.get("post_first_token_seconds"),
        "post_first_visible_token_seconds": timing.get(
            "post_first_visible_token_seconds"
        ),
        "post_last_visible_to_done_seconds": timing.get(
            "post_last_visible_to_done_seconds"
        ),
        "decode_tok_s_stream": decode_tps,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def summarize_response_rails(resp: Any) -> dict[str, Any]:
    """Summarize visible/reasoning rails without storing another full response copy."""
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    status: str | None = None
    if isinstance(resp, dict):
        status_value = resp.get("status")
        if status_value is not None:
            status = str(status_value)
        output_text = resp.get("output_text")
        if isinstance(output_text, str):
            visible_parts.append(output_text)
        output = resp.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text")
                        if not isinstance(text, str):
                            continue
                        part_type = str(part.get("type") or "")
                        if item_type == "reasoning" or part_type == "reasoning":
                            reasoning_parts.append(text)
                        elif item_type == "message" or part_type in {"output_text", "text"}:
                            visible_parts.append(text)
                text = item.get("text")
                if isinstance(text, str):
                    if item_type == "reasoning":
                        reasoning_parts.append(text)
                    elif item_type in {"message", "output_text", "text"}:
                        visible_parts.append(text)
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message") if isinstance(first.get("message"), dict) else {}
            content = msg.get("content")
            if isinstance(content, str):
                visible_parts.append(content)
            reasoning = (
                msg.get("reasoning_content")
                or msg.get("reasoning")
                or first.get("reasoning")
            )
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
    visible = "\n".join(part for part in visible_parts if part)
    reasoning = "\n".join(part for part in reasoning_parts if part)
    return {
        "status": status,
        "visible_chars": len(visible),
        "reasoning_chars": len(reasoning),
        "has_visible_content": bool(visible.strip()),
        "has_reasoning_content": bool(reasoning.strip()),
        "visible_preview": visible[:500],
        "reasoning_preview": reasoning[:500],
    }


def add_response_contract_metrics(
    stage: dict[str, Any],
    *,
    expect_visible_content: bool = False,
) -> None:
    summary = summarize_response_rails(stage.get("response"))
    stage["response_rails"] = summary
    failures: list[str] = []
    if expect_visible_content and not summary.get("has_visible_content"):
        failures.append("missing_visible_content")
    if failures:
        stage["response_contract_failures"] = failures
        if stage.get("status") == "ok":
            stage["status"] = "response_contract_failed"


def _nested_number(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = value
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, bool):
        return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _mb_to_gb(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value / 1024, 3)


def _bytes_to_gb(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 3)


def add_memory_metrics(stage: dict[str, Any]) -> None:
    before = stage.get("before") if isinstance(stage.get("before"), dict) else {}
    after = stage.get("after") if isinstance(stage.get("after"), dict) else {}

    rss_before = _nested_number(before, ("process", "rss_mb"))
    rss_after = _nested_number(after, ("process", "rss_mb"))
    active_before = _nested_number(before, ("health", "body", "memory", "active_mb"))
    active_after = _nested_number(after, ("health", "body", "memory", "active_mb"))
    peak_after = _nested_number(after, ("health", "body", "memory", "peak_mb"))
    cache_after = _nested_number(after, ("health", "body", "memory", "cache_mb"))

    memory: dict[str, Any] = {}
    if rss_before is not None:
        memory["process_rss_before_gb"] = _mb_to_gb(rss_before)
    if rss_after is not None:
        memory["process_rss_after_gb"] = _mb_to_gb(rss_after)
    if rss_before is not None and rss_after is not None:
        memory["process_rss_delta_gb"] = _mb_to_gb(rss_after - rss_before)
    if active_before is not None:
        memory["metal_active_before_gb"] = _mb_to_gb(active_before)
    if active_after is not None:
        memory["metal_active_after_gb"] = _mb_to_gb(active_after)
    if active_before is not None and active_after is not None:
        memory["metal_active_delta_gb"] = _mb_to_gb(active_after - active_before)
    if peak_after is not None:
        memory["metal_peak_after_gb"] = _mb_to_gb(peak_after)
    if cache_after is not None:
        memory["metal_cache_after_gb"] = _mb_to_gb(cache_after)
    if memory:
        stage["memory"] = memory


def extract_block_disk_max_gb(args: list[str]) -> float | None:
    value: float | None = None
    for idx, item in enumerate(args):
        if item == "--block-disk-cache-max-gb" and idx + 1 < len(args):
            try:
                value = float(args[idx + 1])
            except (TypeError, ValueError):
                value = None
        elif item.startswith("--block-disk-cache-max-gb="):
            try:
                value = float(item.split("=", 1)[1])
            except (TypeError, ValueError):
                value = None
    return value


def add_cache_capacity_projection(
    stage: dict[str, Any],
    block_disk_max_gb: float | None,
    cache_bypassed: bool = False,
) -> None:
    after = stage.get("after") if isinstance(stage.get("after"), dict) else {}
    cache_stats = after.get("cache_stats") if isinstance(after.get("cache_stats"), dict) else {}
    body = cache_stats.get("body") if isinstance(cache_stats.get("body"), dict) else {}
    block_disk = body.get("block_disk_cache") if isinstance(body.get("block_disk_cache"), dict) else {}
    try:
        disk_size_bytes = int(block_disk.get("disk_size_bytes") or 0)
        tokens_on_disk = int(block_disk.get("total_tokens_on_disk") or 0)
        blocks_on_disk = int(block_disk.get("blocks_on_disk") or 0)
    except (TypeError, ValueError):
        return
    if disk_size_bytes <= 0 or tokens_on_disk <= 0:
        return

    if cache_bypassed:
        stage["cache_capacity_projection"] = {
            "basis": "invalid_for_capacity_proof",
            "reason": "cache_salt_or_skip_prefix_cache_bypasses_prefix_paged_l2",
            "tokens_on_disk": tokens_on_disk,
            "blocks_on_disk": blocks_on_disk,
            "disk_size_gb": _bytes_to_gb(float(disk_size_bytes)),
        }
        return

    bytes_per_token = disk_size_bytes / tokens_on_disk
    projection: dict[str, Any] = {
        "basis": "observed_block_disk_l2",
        "tokens_on_disk": tokens_on_disk,
        "blocks_on_disk": blocks_on_disk,
        "disk_size_gb": _bytes_to_gb(float(disk_size_bytes)),
        "bytes_per_token": round(bytes_per_token, 1),
        "projected_gb_at_1m_tokens": _bytes_to_gb(bytes_per_token * 1_000_000),
    }
    if block_disk_max_gb is not None and block_disk_max_gb > 0:
        projection["projected_tokens_at_l2_max_gb"] = int((block_disk_max_gb * (1024**3)) // bytes_per_token)
        projection["block_disk_max_gb"] = block_disk_max_gb
    stage["cache_capacity_projection"] = projection


def extract_response_error_code(resp: Any) -> str | None:
    if not isinstance(resp, dict):
        return None
    error = resp.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return str(code) if code is not None else None


def classify_http_stage_status(
    http_code: int,
    resp: Any,
    expect_http_code: int = 0,
    expect_error_code: str | None = None,
) -> str:
    if 200 <= http_code < 300:
        return "ok"
    if expect_http_code and http_code == expect_http_code:
        actual_error_code = extract_response_error_code(resp)
        if not expect_error_code or actual_error_code == expect_error_code:
            return "expected_http_error"
    return "http_error"


def probe_status_from_results(results: list[dict[str, Any]]) -> tuple[str, str | None]:
    bad_stages = [
        f"{idx}:{stage.get('status')}"
        for idx, stage in enumerate(results)
        if stage.get("status") not in {"ok", "expected_http_error"}
    ]
    if bad_stages:
        return "fail", "non-ok stage(s): " + ", ".join(bad_stages)
    return "pass", None


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    row = ROWS[args.row]
    python = Path(args.python).expanduser()
    if not python.is_absolute():
        python = (REPO / python).resolve()
    block_disk_max_gb = extract_block_disk_max_gb(row.cache_args + args.serve_extra_arg)
    out: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row": row.name,
        "model_path": row.path,
        "python": str(python),
        "request_route": args.route,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "max_thinking_tokens": args.max_thinking_tokens,
        "image_path": str(args.image_path) if args.image_path else None,
        "retained_image_turns": args.retained_image_turns,
        "abort_thresholds": {
            "rss_gb": args.abort_rss_gb,
            "metal_active_gb": args.abort_metal_active_gb,
        },
        "block_disk_max_gb": block_disk_max_gb,
        "cache_proof_policy": {
            "cache_salt": args.cache_salt,
            "skip_prefix_cache": bool(args.skip_prefix_cache),
            "prefix_paged_l2_bypassed": bool(args.cache_salt or args.skip_prefix_cache),
            "capacity_projection_valid": not bool(args.cache_salt or args.skip_prefix_cache),
        },
        "response_contract": {
            "expect_visible_content": bool(getattr(args, "expect_visible_content", False)),
        },
        "results": [],
    }
    if not Path(row.path).is_dir():
        out["status"] = "skipped"
        out["reason"] = "model path missing"
        return out
    if not python.exists():
        out["status"] = "error"
        out["reason"] = "python missing"
        return out

    cmd = build_serve_cmd(row, python, args.port, args.timeout, args.serve_extra_arg)
    out["cmd"] = cmd
    serve_env = dict(row.env)
    serve_env.update(parse_env_overrides(args.serve_env))
    proc = subprocess.Popen(
        cmd,
        cwd=str(SAFE_SERVER_CWD),
        env=build_clean_env(serve_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_lines: list[str] = []

    def read_log() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))

    threading.Thread(target=read_log, daemon=True).start()

    try:
        out["health_ready"] = wait_health(args.port, proc, args.timeout)
        out["snapshots"] = [snapshot("health_ready", args.port, proc)]
        base_url = f"http://127.0.0.1:{args.port}"
        endpoint = "/v1/responses" if args.route == "responses" else "/v1/chat/completions"
        for target in args.prompt_tokens:
            stage: dict[str, Any] = {
                "prompt_tokens_approx": target,
                "before": snapshot(f"before_{target}", args.port, proc),
            }
            reason = threshold_exceeded(stage["before"], args.abort_rss_gb, args.abort_metal_active_gb)
            if reason:
                stage["status"] = "aborted_before_request"
                stage["abort_reason"] = reason
                out["results"].append(stage)
                out["status"] = "aborted"
                break
            body = build_request(
                row,
                make_prompt(
                    target,
                    mode=args.prompt_mode,
                    max_tokens=args.max_tokens,
                ),
                args.max_tokens,
                args.route,
                Path(args.image_path) if args.image_path else None,
                args.retained_image_turns,
                args.enable_thinking,
                args.max_thinking_tokens,
                args.cache_salt,
                args.skip_prefix_cache,
                args.stream,
            )
            stage["request"] = redact_large_payloads(body)
            started = time.time()
            try:
                if args.stream:
                    code, resp, http_timing = http_json_stream_timed(
                        "POST", f"{base_url}{endpoint}", body, timeout=args.request_timeout
                    )
                else:
                    code, resp, http_timing = http_json_timed(
                        "POST", f"{base_url}{endpoint}", body, timeout=args.request_timeout
                    )
                stage["http_code"] = code
                stage["response"] = redact_large_payloads(resp)
                stage["http_timing"] = http_timing
                stage["elapsed_s"] = round(time.time() - started, 3)
                add_speed_metrics(stage)
                if args.stream:
                    add_stream_speed_metrics(stage)
                stage["error_code"] = extract_response_error_code(resp)
                stage["status"] = classify_http_stage_status(
                    code,
                    resp,
                    args.expect_http_code,
                    args.expect_error_code,
                )
                add_response_contract_metrics(
                    stage,
                    expect_visible_content=bool(getattr(args, "expect_visible_content", False)),
                )
            except Exception as exc:  # noqa: BLE001
                stage["status"] = "request_exception"
                stage["error"] = repr(exc)
                stage["elapsed_s"] = round(time.time() - started, 3)
            stage["after"] = snapshot(f"after_{target}", args.port, proc)
            add_memory_metrics(stage)
            add_cache_capacity_projection(
                stage,
                block_disk_max_gb,
                cache_bypassed=bool(args.cache_salt or args.skip_prefix_cache),
            )
            reason = threshold_exceeded(stage["after"], args.abort_rss_gb, args.abort_metal_active_gb)
            if reason:
                stage["abort_reason"] = reason
                out["status"] = "aborted"
                out["results"].append(stage)
                break
            out["results"].append(stage)
        if "status" not in out:
            status, reason = probe_status_from_results(out["results"])
            out["status"] = status
            if reason:
                out["reason"] = reason
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = repr(exc)
    finally:
        out["returncode_before_terminate"] = proc.poll()
        terminate(proc)
        out["returncode"] = proc.poll()
        out["log_tail"] = log_lines[-240:]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=sorted(ROWS), required=True)
    parser.add_argument("--python", default=str(DEFAULT_PY))
    parser.add_argument("--port", type=int, default=8815)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--prompt-tokens", type=lambda s: [int(x) for x in s.split(",") if x], default=[512, 4096])
    parser.add_argument("--prompt-mode", choices=("default", "speed_floor"), default="default")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--route", choices=("chat", "responses"), default="chat")
    parser.add_argument("--stream", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--max-thinking-tokens", type=int, default=None)
    parser.add_argument("--cache-salt", default=None)
    parser.add_argument("--skip-prefix-cache", action="store_true")
    parser.add_argument("--image-path")
    parser.add_argument("--retained-image-turns", type=int, default=1)
    parser.add_argument("--abort-rss-gb", type=float, default=0.0)
    parser.add_argument("--abort-metal-active-gb", type=float, default=0.0)
    parser.add_argument("--expect-http-code", type=int, default=0)
    parser.add_argument("--expect-error-code", default=None)
    parser.add_argument("--expect-visible-content", action="store_true")
    parser.add_argument("--serve-extra-arg", action="append", default=[])
    parser.add_argument("--serve-env", action="append", default=[])
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{out} status={result.get('status')}")
    return 0 if result.get("status") in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

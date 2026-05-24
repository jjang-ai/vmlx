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


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed


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


def make_prompt(approx_tokens: int) -> str:
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
            "stream": False,
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    else:
        request = {
            "model": Path(row.path).name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    if enable_thinking is not None:
        request["enable_thinking"] = enable_thinking
        request["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
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


def probe_status_from_results(results: list[dict[str, Any]]) -> tuple[str, str | None]:
    bad_stages = [
        f"{idx}:{stage.get('status')}"
        for idx, stage in enumerate(results)
        if stage.get("status") != "ok"
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
    out: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row": row.name,
        "model_path": row.path,
        "python": str(python),
        "request_route": args.route,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "image_path": str(args.image_path) if args.image_path else None,
        "retained_image_turns": args.retained_image_turns,
        "abort_thresholds": {
            "rss_gb": args.abort_rss_gb,
            "metal_active_gb": args.abort_metal_active_gb,
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
                make_prompt(target),
                args.max_tokens,
                args.route,
                Path(args.image_path) if args.image_path else None,
                args.retained_image_turns,
                args.enable_thinking,
            )
            started = time.time()
            try:
                code, resp = http_json("POST", f"{base_url}{endpoint}", body, timeout=args.request_timeout)
                stage["http_code"] = code
                stage["response"] = redact_large_payloads(resp)
                stage["elapsed_s"] = round(time.time() - started, 3)
                add_speed_metrics(stage)
                stage["status"] = "ok" if 200 <= code < 300 else "http_error"
            except Exception as exc:  # noqa: BLE001
                stage["status"] = "request_exception"
                stage["error"] = repr(exc)
                stage["elapsed_s"] = round(time.time() - started, 3)
            stage["after"] = snapshot(f"after_{target}", args.port, proc)
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
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--route", choices=("chat", "responses"), default="chat")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--image-path")
    parser.add_argument("--retained-image-turns", type=int, default=1)
    parser.add_argument("--abort-rss-gb", type=float, default=0.0)
    parser.add_argument("--abort-metal-active-gb", type=float, default=0.0)
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

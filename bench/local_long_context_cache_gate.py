#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENV = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable)).expanduser()
PYTHON = _PYTHON_ENV if _PYTHON_ENV.is_absolute() else (ROOT / _PYTHON_ENV).resolve()
AUX_NAMES = {"audio_tokenizer", "audio_encoder", "visual", "vision_tower", "processor"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-._")[:96] or "model"


def model_type(config: dict[str, Any]) -> str:
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return str(config.get("model_type") or text_config.get("model_type") or "unknown")


def has_key_recursive(obj: Any, keys: set[str]) -> bool:
    if isinstance(obj, dict):
        return any(k in keys or has_key_recursive(v, keys) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_key_recursive(v, keys) for v in obj)
    return False


def discover(models_root: Path, only: str | None) -> list[dict[str, Any]]:
    filters = [s.strip().lower() for s in (only or "").split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    for config_path in sorted(models_root.rglob("config.json")):
        model_dir = config_path.parent
        if model_dir.name in AUX_NAMES:
            continue
        has_weights = (model_dir / "model.safetensors.index.json").is_file() or any(model_dir.glob("*.safetensors"))
        if not has_weights:
            continue
        config = read_json(config_path)
        mt = model_type(config)
        haystack = f"{model_dir} {model_dir.name} {mt}".lower()
        if filters and not any(f in haystack for f in filters):
            continue
        is_mllm = (
            mt == "mimo_v2"
            or (
                mt != "step3p7"
                and has_key_recursive(config, {"vision_config", "visual", "image_token_id", "video_token_id"})
            )
        )
        rows.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "served_name": sanitize(model_dir.name.lower()),
                "model_type": mt,
                "is_mllm": is_mllm,
            }
        )
    return rows


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> tuple[int, Any, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            elapsed = time.monotonic() - start
            try:
                return resp.status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return resp.status, raw, elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        elapsed = time.monotonic() - start
        try:
            return exc.code, json.loads(raw), elapsed
        except json.JSONDecodeError:
            return exc.code, raw, elapsed


def wait_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float, log_path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}: {log_path.read_text(errors='replace')[-4000:]}")
        try:
            code, body, _ = request_json("GET", f"{base_url}/health", timeout=5)
            if code == 200 and isinstance(body, dict) and body.get("model_loaded") is True:
                return body
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy: {last_error}: {log_path.read_text(errors='replace')[-4000:]}")


def serve_cmd(row: dict[str, Any], port: int, row_dir: Path, max_prompt_tokens: int) -> list[str]:
    cmd = [
        str(PYTHON),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        row["served_name"],
        "--timeout",
        "240",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "1024",
        "--completion-batch-size",
        "128",
        "--continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "4000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str((row_dir / "block_cache").resolve()),
        "--block-disk-cache-max-gb",
        "4",
        "--ssm-state-cache-mb",
        "1024",
        "--max-prompt-tokens",
        str(max_prompt_tokens),
        "--max-tokens",
        "64",
        "--log-level",
        "INFO",
        "--default-enable-thinking",
        "false",
    ]
    if row.get("is_mllm"):
        cmd.append("--is-mllm")
    extra = os.environ.get("VMLINUX_BENCH_EXTRA_SERVE_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def long_prompt(word_count: int) -> str:
    filler = " ".join(f"cacheword{i % 251:03d}" for i in range(word_count))
    return (
        "Long context cache probe. Read the filler, ignore its repeated words, "
        "and answer exactly LONGCTX-OK.\n\n"
        f"{filler}\n\nAnswer exactly LONGCTX-OK."
    )


def payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16,
        "enable_thinking": False,
    }


def extract_text(resp: Any) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else ""
    return content if isinstance(content, str) else ""


def cached_tokens(resp: Any) -> int:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    try:
        return int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    except (TypeError, ValueError):
        return 0


def cache_detail(resp: Any) -> str:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    return str(details.get("cache_detail") or "") if isinstance(details, dict) else ""


def run_row(row: dict[str, Any], port: int, out: Path, load_timeout_s: float, request_timeout_s: float, words: int, max_prompt_tokens: int) -> dict[str, Any]:
    row_dir = out / sanitize(row["name"])
    row_dir.mkdir(parents=True, exist_ok=True)
    log_path = row_dir / "server.log"
    cmd = serve_cmd(row, port, row_dir, max_prompt_tokens)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    result: dict[str, Any] = {"row": row, "command": cmd, "word_count": words, "max_prompt_tokens": max_prompt_tokens}
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        result["health_before"] = wait_health(base, proc, load_timeout_s, log_path)
        prompt = long_prompt(words)
        turns = []
        for label in ("first", "second"):
            code, resp, elapsed = request_json("POST", f"{base}/v1/chat/completions", payload(row["served_name"], prompt), timeout=request_timeout_s)
            cache_code, cache_body, _ = request_json("GET", f"{base}/v1/cache/stats", timeout=30)
            text = extract_text(resp)
            turns.append(
                {
                    "label": label,
                    "code": code,
                    "elapsed_sec": elapsed,
                    "response": resp,
                    "text": text,
                    "text_head": text[:240],
                    "cached_tokens": cached_tokens(resp),
                    "cache_detail": cache_detail(resp),
                    "usage": resp.get("usage") if isinstance(resp, dict) else None,
                    "cache_stats": {"code": cache_code, "body": cache_body},
                }
            )
        result["turns"] = turns
        result["health_after"] = request_json("GET", f"{base}/health", timeout=30)[1]
        second = turns[1]
        cache_hit_ok = second["code"] == 200 and second["cached_tokens"] >= max(64, words // 4)
        output_ok = "LONGCTX-OK" in second["text"]
        result["classification"] = {
            "status": "pass" if cache_hit_ok else "fail",
            "cache_status": "pass" if cache_hit_ok else "fail",
            "output_status": "pass" if output_ok else "review",
            "checks": {
                "second_http_ok": second["code"] == 200,
                "second_cached_tokens": second["cached_tokens"],
                "second_cache_detail": second["cache_detail"],
                "second_cached_tokens_meets_floor": second["cached_tokens"] >= max(64, words // 4),
                "second_output_contains_longctx_ok": output_ok,
            },
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        result["returncode"] = proc.returncode
        result["log_tail"] = log_path.read_text(errors="replace")[-6000:]
    result["status"] = result.get("classification", {}).get("status", "error")
    write_json(row_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Local long-context repeated-prefix cache gate.")
    parser.add_argument("--models-root", default=str(Path.home() / ".mlxstudio/models"))
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--words", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--load-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = discover(Path(args.models_root).expanduser(), args.only)
    summary: dict[str, Any] = {"status": "pass", "row_count": len(rows), "rows": rows, "results": []}
    write_json(out / "inventory.json", {"row_count": len(rows), "rows": rows})
    print(json.dumps({"out": str(out), "row_count": len(rows)}, indent=2), flush=True)
    for row in rows:
        try:
            result = run_row(row, args.port, out, args.load_timeout_s, args.request_timeout_s, args.words, args.max_prompt_tokens)
        except Exception as exc:
            result = {"row": row, "status": "error", "error": repr(exc)}
        if result.get("status") != "pass":
            summary["status"] = "fail"
        summary["results"].append(result)
        write_json(out / "summary.json", summary)
        print(json.dumps({"model": row["name"], "status": result.get("status")}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

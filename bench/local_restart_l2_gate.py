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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENV = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable)).expanduser()
PYTHON = _PYTHON_ENV if _PYTHON_ENV.is_absolute() else (ROOT / _PYTHON_ENV).resolve()


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


def raw_model_type(config: dict[str, Any]) -> str:
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return str(config.get("model_type") or text_config.get("model_type") or "unknown")


def has_key_recursive(obj: Any, keys: set[str]) -> bool:
    if isinstance(obj, dict):
        return any(k in keys or has_key_recursive(v, keys) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_key_recursive(v, keys) for v in obj)
    return False


def discover(models_root: Path, only: str | None) -> list[dict[str, Any]]:
    dirs = {p.parent for p in models_root.rglob("config.json")}
    rows: list[dict[str, Any]] = []
    filters = [s.strip().lower() for s in (only or "").split(",") if s.strip()]
    aux_names = {"audio_tokenizer", "audio_encoder", "visual", "vision_tower", "processor"}
    for model_dir in sorted(dirs):
        if model_dir.name in aux_names:
            continue
        has_weights = (model_dir / "model.safetensors.index.json").is_file() or any(
            model_dir.glob("*.safetensors")
        )
        if not has_weights:
            continue
        config = read_json(model_dir / "config.json")
        model_type = raw_model_type(config)
        name = model_dir.name
        haystack = f"{model_dir} {name} {model_type}".lower()
        if filters and not any(f in haystack for f in filters):
            continue
        is_mllm = (
            model_type == "mimo_v2"
            or (
                model_type != "step3p7"
                and has_key_recursive(config, {"vision_config", "visual", "image_token_id", "video_token_id"})
            )
        )
        rows.append(
            {
                "name": name,
                "path": str(model_dir),
                "served_name": sanitize(name.lower()),
                "model_type": model_type,
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


def serve_cmd(row: dict[str, Any], port: int, block_dir: Path) -> list[str]:
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
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(block_dir.resolve()),
        "--block-disk-cache-max-gb",
        "2",
        "--ssm-state-cache-mb",
        "1024",
        "--max-tokens",
        "128",
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


def chat_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Cache restore probe nonce L2-RESTORE-20260607. Reply exactly: ACK",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
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
    if not isinstance(resp, dict):
        return 0
    usage = resp.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cache_detail(resp: Any) -> str:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    return str(details.get("cache_detail") or "") if isinstance(details, dict) else ""


def nested_number(obj: Any, path: list[str]) -> int:
    cur = obj
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
    try:
        return int(cur or 0)
    except (TypeError, ValueError):
        return 0


def run_phase(row: dict[str, Any], phase: str, port: int, row_dir: Path, block_dir: Path, timeout_s: float) -> dict[str, Any]:
    log_path = row_dir / f"{phase}_server.log"
    cmd = serve_cmd(row, port, block_dir)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{port}"
    result: dict[str, Any] = {
        "phase": phase,
        "command": cmd,
        "log": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        result["health_before"] = wait_health(base_url, proc, timeout_s, log_path)
        code, before_cache, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        result["cache_before"] = {"code": code, "body": before_cache}
        code, resp, elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(row["served_name"]),
            timeout=180,
        )
        code_after, after_cache, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30)
        result.update(
            {
                "code": code,
                "elapsed_sec": elapsed,
                "response": resp,
                "text": extract_text(resp),
                "cached_tokens": cached_tokens(resp),
                "cache_detail": cache_detail(resp),
                "cache_after": {"code": code_after, "body": after_cache},
                "health_after": request_json("GET", f"{base_url}/health", timeout=30)[1],
            }
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        result["returncode"] = proc.returncode
        result["log_tail"] = log_path.read_text(errors="replace")[-6000:]
    return result


def classify(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    second_cache = second.get("cache_after", {}).get("body", {})
    second_health = second.get("health_after", {})
    scheduler = second_health.get("scheduler") if isinstance(second_health, dict) else {}
    last_exec = scheduler.get("last_cache_execution") if isinstance(scheduler, dict) else {}
    checks = {
        "first_http_ok": first.get("code") == 200,
        "second_http_ok": second.get("code") == 200,
        "first_visible_ack": "ACK" in first.get("text", ""),
        "second_visible_ack": "ACK" in second.get("text", ""),
        "second_cached_tokens_positive": (second.get("cached_tokens") or 0) > 0,
        "second_cache_detail_mentions_disk": "disk" in str(second.get("cache_detail") or ""),
        "block_disk_hit_positive": nested_number(second_cache, ["block_disk_cache", "disk_hits"]) > 0,
        "last_execution_disk_hit": isinstance(last_exec, dict) and last_exec.get("disk_hit") is True,
    }
    checks["restart_l2_restore_observed"] = (
        checks["second_http_ok"]
        and checks["second_cached_tokens_positive"]
        and (
            checks["second_cache_detail_mentions_disk"]
            or checks["block_disk_hit_positive"]
            or checks["last_execution_disk_hit"]
        )
    )
    checks["output_exact_ack_observed"] = checks["first_visible_ack"] and checks["second_visible_ack"]
    return {
        "checks": checks,
        "cache_status": "pass" if checks["restart_l2_restore_observed"] else "fail",
        "output_status": "pass" if checks["output_exact_ack_observed"] else "review",
        "status": "pass" if checks["restart_l2_restore_observed"] else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local model restart/L2 restore gate.")
    parser.add_argument("--models-root", default=str(Path.home() / ".mlxstudio/models"))
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8860)
    parser.add_argument("--load-timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = discover(Path(args.models_root).expanduser(), args.only)
    summary: dict[str, Any] = {"rows": rows, "results": [], "status": "pass"}
    write_json(out / "inventory.json", {"rows": rows, "row_count": len(rows)})
    print(json.dumps({"out": str(out), "row_count": len(rows)}, indent=2), flush=True)
    for row in rows:
        row_dir = out / sanitize(row["name"])
        row_dir.mkdir(parents=True, exist_ok=True)
        block_dir = row_dir / "shared_block_cache"
        try:
            first = run_phase(row, "first", args.port, row_dir, block_dir, args.load_timeout_s)
            second = run_phase(row, "second", args.port, row_dir, block_dir, args.load_timeout_s)
            classification = classify(first, second)
            result = {"row": row, "first": first, "second": second, "classification": classification, "status": classification["status"]}
        except Exception as exc:
            result = {"row": row, "status": "error", "error": repr(exc)}
        if result.get("status") != "pass":
            summary["status"] = "fail"
        summary["results"].append(result)
        write_json(row_dir / "result.json", result)
        write_json(out / "summary.json", summary)
        print(json.dumps({"model": row["name"], "status": result.get("status")}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

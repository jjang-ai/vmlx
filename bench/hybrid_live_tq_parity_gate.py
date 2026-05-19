#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable))
OUT_BASE = ROOT / "docs/internal/release-gates/2026-05-19-task8-hybrid-live-tq"

ROWS: dict[str, dict[str, str]] = {
    "mxfp4": {
        "path": "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP4-MTP",
        "served": "qwen36-27b-mxfp4-task8",
    },
    "mxfp8": {
        "path": "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP8-MTP",
        "served": "qwen36-27b-mxfp8-task8",
    },
    "jang4m": {
        "path": "/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        "served": "qwen36-27b-jang4m-task8",
    },
}


PROBES: list[dict[str, Any]] = [
    {
        "name": "exact_phrase",
        "expected_contains": "BLUE-17",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly this text and nothing else: BLUE-17",
                }
            ],
            "temperature": 0,
            "max_tokens": 12,
            "stream": False,
            "enable_thinking": False,
        },
    },
    {
        "name": "arithmetic",
        "expected_contains": "45",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 + 28? Reply with only the number.",
                }
            ],
            "temperature": 0,
            "max_tokens": 12,
            "stream": False,
            "enable_thinking": False,
        },
    },
    {
        "name": "multiturn_fact",
        "expected_contains": "ORCHID-731",
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": "Store this exact code for the next turn: ORCHID-731. Reply only noted.",
                },
                {"role": "assistant", "content": "noted"},
                {
                    "role": "user",
                    "content": "What exact code did I give you? Reply with only the code.",
                },
            ],
            "temperature": 0,
            "max_tokens": 24,
            "stream": False,
            "enable_thinking": False,
        },
    },
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def request_json(method: str, url: str, body: Any | None = None, timeout: float = 120.0) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - t0
            if not raw:
                return response.status, None, elapsed
            try:
                return response.status, json.loads(raw), elapsed
            except Exception:
                return response.status, raw.decode("utf-8", "replace"), elapsed
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def extract_text(resp: Any) -> tuple[str, str, dict[str, Any], str]:
    if not isinstance(resp, dict):
        return "", "", {}, ""
    choices = resp.get("choices") or []
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    if choices:
        choice = choices[0] or {}
        msg = choice.get("message") or {}
        content = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
        finish = str(choice.get("finish_reason") or "")
        return content, reasoning, usage, finish
    return "", "", usage, ""


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def log_tail(path: Path, n: int = 80) -> list[str]:
    try:
        return path.read_text(errors="ignore").splitlines()[-n:]
    except Exception:
        return []


def wait_ready(base: str, timeout_s: float, proc: subprocess.Popen[Any], log_path: Path) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during load rc={proc.returncode}; log tail:\n"
                + "\n".join(log_tail(log_path))
            )
        code, body, _ = request_json("GET", f"{base}/health", timeout=3.0)
        last = body
        if code == 200 and isinstance(body, dict) and body.get("model_loaded"):
            return body
        time.sleep(2)
    raise TimeoutError(f"server did not become ready; last health={last!r}")


def start_server(
    *,
    row_id: str,
    row: dict[str, str],
    mode: str,
    port: int,
    out_dir: Path,
    load_timeout_s: float,
) -> tuple[subprocess.Popen[Any], dict[str, Any], Path]:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    log_path = mode_dir / "server.log"
    block_cache_dir = mode_dir / "block_cache"
    cmd = [
        str(PYTHON),
        "-B",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        "300",
        "--continuous-batching",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "2048",
        "--completion-batch-size",
        "512",
        "--stream-interval",
        "1",
        "--max-tokens",
        "512",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "512",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(block_cache_dir),
        "--block-disk-cache-max-gb",
        "2",
        "--ssm-state-cache-mb",
        "1024",
        "--served-model-name",
        row["served"],
        "--is-mllm",
        "--disable-native-mtp",
        "--tool-call-parser",
        "qwen",
        "--reasoning-parser",
        "qwen3",
        "--default-enable-thinking",
        "false",
        "--log-level",
        "INFO",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("JANGTQ_TOPK_OVERRIDE", None)
    env.pop("VMLINUX_DISABLE_TQ_KV", None)
    env.pop("VMLX_DISABLE_TQ_KV", None)
    if mode == "baseline_full_precision_live_kv":
        env["VMLX_DISABLE_TQ_KV"] = "1"
    elif mode != "selective_live_tq_kv":
        raise ValueError(f"unknown mode: {mode}")

    write_json(
        mode_dir / "start.json",
        {
            "row_id": row_id,
            "mode": mode,
            "command": cmd,
            "env_overrides": {
                key: env[key]
                for key in ("PYTHONPATH", "VMLX_DISABLE_TQ_KV")
                if key in env
            },
        },
    )
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    health = wait_ready(f"http://127.0.0.1:{port}", load_timeout_s, proc, log_path)
    write_json(mode_dir / "health_initial.json", health)
    return proc, health, log_path


def cache_status(health: dict[str, Any]) -> dict[str, Any]:
    native_top = health.get("native_cache")
    if isinstance(native_top, dict):
        return native_top
    cache = health.get("cache") if isinstance(health.get("cache"), dict) else {}
    native = cache.get("native") if isinstance(cache.get("native"), dict) else {}
    return native if isinstance(native, dict) else {}


def run_mode(
    *,
    row_id: str,
    row: dict[str, str],
    mode: str,
    port: int,
    out_dir: Path,
    load_timeout_s: float,
    request_timeout_s: float,
    keep_server: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_id": row_id,
        "mode": mode,
        "path": row["path"],
        "served": row["served"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probes": [],
    }
    proc: subprocess.Popen[Any] | None = None
    log_path: Path | None = None
    try:
        proc, health, log_path = start_server(
            row_id=row_id,
            row=row,
            mode=mode,
            port=port,
            out_dir=out_dir,
            load_timeout_s=load_timeout_s,
        )
        result["pid"] = proc.pid
        result["health_initial"] = health
        base = f"http://127.0.0.1:{port}"
        for probe in PROBES:
            body = dict(probe["body"])
            body["model"] = row["served"]
            code, resp, elapsed = request_json(
                "POST",
                f"{base}/v1/chat/completions",
                body,
                timeout=request_timeout_s,
            )
            content, reasoning, usage, finish = extract_text(resp)
            probe_result = {
                "name": probe["name"],
                "code": code,
                "elapsed_sec": elapsed,
                "finish": finish,
                "content": content,
                "reasoning": reasoning,
                "usage": usage,
                "expected_contains": probe["expected_contains"],
                "expected_ok": probe["expected_contains"].lower() in content.lower(),
                "raw_response": resp,
            }
            result["probes"].append(probe_result)
            write_json(out_dir / mode / f"{probe['name']}.json", probe_result)
        code, health_after, _ = request_json("GET", f"{base}/health", timeout=30.0)
        result["health_after_code"] = code
        result["health_after"] = health_after
        result["native_cache"] = cache_status(health_after if isinstance(health_after, dict) else health)
        result["log_markers"] = [
            line
            for line in log_tail(log_path, 240)
            if "Runtime cache layout:" in line
            or "TurboQuant" in line
            or "Hybrid/path-dependent" in line
            or "VLM KV cache quantization" in line
        ][-80:]
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if log_path is not None:
            result["log_tail"] = log_tail(log_path)
    finally:
        if proc is not None and not keep_server:
            terminate_process(proc)
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(out_dir / mode / "SUMMARY.json", result)
    return result


def compare_modes(baseline: dict[str, Any], tq: dict[str, Any]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    baseline_by_name = {probe["name"]: probe for probe in baseline.get("probes", [])}
    tq_by_name = {probe["name"]: probe for probe in tq.get("probes", [])}
    for name in [probe["name"] for probe in PROBES]:
        left = baseline_by_name.get(name, {})
        right = tq_by_name.get(name, {})
        left_text = normalize(str(left.get("content") or ""))
        right_text = normalize(str(right.get("content") or ""))
        left_tokens = (left.get("usage") or {}).get("completion_tokens")
        right_tokens = (right.get("usage") or {}).get("completion_tokens")
        token_delta = None
        if isinstance(left_tokens, int) and isinstance(right_tokens, int):
            token_delta = abs(left_tokens - right_tokens)
        pairs.append(
            {
                "name": name,
                "baseline_text": left_text,
                "tq_text": right_text,
                "normalized_equal": left_text == right_text,
                "both_expected_ok": bool(left.get("expected_ok")) and bool(right.get("expected_ok")),
                "completion_token_delta": token_delta,
                "char_len_delta": abs(len(left_text) - len(right_text)),
                "baseline_finish": left.get("finish"),
                "tq_finish": right.get("finish"),
            }
        )
    tq_native = tq.get("native_cache") or {}
    base_native = baseline.get("native_cache") or {}
    tq_live = tq_native.get("live_attention_tq_kv") or {}
    base_live = base_native.get("live_attention_tq_kv") or {}
    all_exact = all(pair["normalized_equal"] for pair in pairs)
    all_expected = all(pair["both_expected_ok"] for pair in pairs)
    token_delta_ok = all(
        pair["completion_token_delta"] is None or pair["completion_token_delta"] <= 1
        for pair in pairs
    )
    tq_active = (
        tq_native.get("generic_turboquant_kv", {}).get("enabled") is True
        and tq_live.get("enabled") is True
        and tq_live.get("applies_to") == "attention_kv_layers_only"
    )
    baseline_inactive = (
        base_native.get("generic_turboquant_kv", {}).get("enabled") is False
        and base_live.get("enabled") is False
    )
    return {
        "pairs": pairs,
        "tolerance": {
            "output": "normalized exact match for deterministic probes",
            "completion_tokens": "delta <= 1 token",
            "content": "both modes must satisfy expected substring",
        },
        "tq_active": tq_active,
        "baseline_inactive": baseline_inactive,
        "all_outputs_exact": all_exact,
        "all_expected_ok": all_expected,
        "token_delta_ok": token_delta_ok,
        "status": (
            "PASS"
            if tq.get("status") == "PASS"
            and baseline.get("status") == "PASS"
            and tq_active
            and baseline_inactive
            and all_exact
            and all_expected
            and token_delta_ok
            else "FAIL"
        ),
    }


def run_row(
    row_id: str,
    row: dict[str, str],
    *,
    port_base: int,
    out_base: Path,
    load_timeout_s: float,
    request_timeout_s: float,
    keep_server: bool,
) -> dict[str, Any]:
    out_dir = out_base / row_id
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_mode(
        row_id=row_id,
        row=row,
        mode="baseline_full_precision_live_kv",
        port=port_base,
        out_dir=out_dir,
        load_timeout_s=load_timeout_s,
        request_timeout_s=request_timeout_s,
        keep_server=keep_server,
    )
    if keep_server:
        raise RuntimeError("--keep-server can only be used for a single mode during manual debugging")
    tq = run_mode(
        row_id=row_id,
        row=row,
        mode="selective_live_tq_kv",
        port=port_base,
        out_dir=out_dir,
        load_timeout_s=load_timeout_s,
        request_timeout_s=request_timeout_s,
        keep_server=keep_server,
    )
    comparison = compare_modes(baseline, tq)
    result = {
        "row_id": row_id,
        "row": row,
        "baseline": baseline,
        "selective_live_tq": tq,
        "comparison": comparison,
        "status": comparison["status"],
    }
    write_json(out_dir / "PARITY_SUMMARY.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3.6 hybrid live TQ-KV parity gate."
    )
    parser.add_argument(
        "--rows",
        default="mxfp4,mxfp8,jang4m",
        help="Comma-separated row ids: mxfp4,mxfp8,jang4m.",
    )
    parser.add_argument("--port-base", type=int, default=8190)
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_BASE)
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    selected = [item.strip() for item in args.rows.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows: {unknown}; valid={sorted(ROWS)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": str(PYTHON),
        "root": str(ROOT),
        "rows": [],
    }
    for index, row_id in enumerate(selected):
        row = ROWS[row_id]
        if not Path(row["path"]).is_dir():
            summary["rows"].append(
                {"row_id": row_id, "status": "FAIL", "reason": "model path missing", "path": row["path"]}
            )
            continue
        result = run_row(
            row_id,
            row,
            port_base=args.port_base + index,
            out_base=args.out_dir,
            load_timeout_s=args.load_timeout_s,
            request_timeout_s=args.request_timeout_s,
            keep_server=args.keep_server,
        )
        summary["rows"].append(
            {
                "row_id": row_id,
                "status": result["status"],
                "summary": str(args.out_dir / row_id / "PARITY_SUMMARY.json"),
            }
        )
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["status"] = (
        "PASS"
        if summary["rows"] and all(row.get("status") == "PASS" for row in summary["rows"])
        else "FAIL"
    )
    write_json(args.out_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Live DSV4 Flash Responses restart/L2 block-disk cache gate.

This proves the cache path that same-process cache gates cannot prove:

- start a real source/app server with native DSV4 prefix+paged+block-disk L2
- write a fresh per-run prefix into an isolated block-disk cache directory
- stop the server
- restart a new server process using the same block-disk cache directory
- replay the same terminal prompt and require disk hits plus ``paged+dsv4`` cached
  token accounting after restart

The per-run nonce and isolated block cache directory are deliberate. A global
old cache hit must not be able to fake this proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = (
    REPO
    / "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DEFAULT_MODEL_CANDIDATES = (
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
    "/Users/example/models/JANGQ/"
    "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP",
)
DEFAULT_OUT = REPO / "build/current-dsv4-responses-restart-l2-gate-20260606.json"
DEFAULT_MIN_FREE_GB = 80.0


def resolve_default_model() -> str:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url: str, timeout: int = 10) -> dict[str, Any]:
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


def usage_details(obj: dict[str, Any]) -> dict[str, Any]:
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    details = (
        usage.get("input_tokens_details")
        or usage.get("prompt_tokens_details")
        or {}
    )
    return details if isinstance(details, dict) else {}


def cached_tokens(obj: dict[str, Any]) -> int:
    try:
        return int(usage_details(obj).get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cache_detail(obj: dict[str, Any]) -> str:
    return str(usage_details(obj).get("cache_detail") or "")


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
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def block_disk_stats(cache_stats: dict[str, Any]) -> dict[str, Any]:
    stats = cache_stats.get("block_disk_cache")
    return stats if isinstance(stats, dict) else {}


def native_cache(health: dict[str, Any]) -> dict[str, Any]:
    native = health.get("native_cache")
    if isinstance(native, dict):
        return native
    nested = (health.get("cache") or {}).get("native") if isinstance(health.get("cache"), dict) else {}
    return nested if isinstance(nested, dict) else {}


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
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        snap["error"] = f"{type(exc).__name__}: {exc}"
    return snap


def blocked_by_memory_preflight(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.min_free_gb <= 0:
        return None
    snap = resource_snapshot("preflight")
    available = (snap.get("system_memory") or {}).get("available_gb")
    if isinstance(available, (int, float)) and available < args.min_free_gb:
        return {
            "status": "skipped",
            "reason": "insufficient_free_memory",
            "required_available_gb": args.min_free_gb,
            "telemetry": [snap],
        }
    return None


def build_command(args: argparse.Namespace, cache_dir: Path, model_name: str) -> list[str]:
    return [
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
        "900",
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
        "dsml",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_r1",
        "--dsv4-enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "256",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(cache_dir),
        "--block-disk-cache-max-gb",
        "10",
        "--stream-interval",
        "1",
        "--max-tokens",
        str(args.max_output_tokens),
        "--served-model-name",
        model_name,
    ]


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "JANGTQ_MPP_NAX",
        "JANGTQ_MPP_NAX_DISABLE",
        "JANGTQ_MPP_NAX_STRICT",
        "JANGTQ_MPP_DENSE",
        "JANGTQ_MPP_DENSE_STRICT",
        "JANGTQ_DISABLE_DSV4_STREAM_LOAD",
        "JANGTQ_DISABLE_DSV4_FAST_LOAD",
        "VMLX_DSV4_ENABLE_PREFIX_CACHE",
        "VMLINUX_DSV4_ENABLE_PREFIX_CACHE",
        "VMLX_DSV4_FORCE_DIRECT_RAIL",
        "VMLINUX_DSV4_FORCE_DIRECT_RAIL",
        "VMLX_DSV4_RAW_MAX",
        "VMLINUX_DSV4_RAW_MAX",
        "VMLX_DSV4_HARD_REP_BLOCK",
        "VMLINUX_DSV4_HARD_REP_BLOCK",
    ):
        env.pop(key, None)
    env["DSV4_LONG_CTX"] = "1"
    env["DSV4_POOL_QUANT"] = "1" if args.pool_quant else "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("VMLINUX_METAL_WS_REJECT_PCT", "98")
    env.setdefault("VMLX_METAL_WS_REJECT_PCT", "98")
    return env


def stop_server(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:
        return True
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        return False


def start_server(
    args: argparse.Namespace,
    cache_dir: Path,
    model_name: str,
    log_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen, list[str]]:
    cmd = build_command(args, cache_dir, model_name)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, cmd


def response_payload(model_name: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": prompt,
        "store": True,
        "stream": False,
        "max_output_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_block = blocked_by_memory_preflight(args)
    if preflight_block is not None:
        return {**preflight_block, "model": args.model}

    out = Path(args.out)
    run_id = f"dsv4-restart-l2-{int(time.time() * 1000)}-{os.getpid()}"
    base_dir = out.parent / "dsv4-restart-l2"
    cache_dir = Path(args.cache_dir) if args.cache_dir else base_dir / run_id / "block-cache"
    log_dir = out.parent / "dsv4-restart-l2-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists() and not args.keep_existing_cache_dir:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    model_name = "dsv4-restart-l2"
    long_context = (
        make_long_context(args.words)
        + f"\n\nGATE RUN ID = {run_id}. "
        "This nonce is diagnostic-only and does not modify the anchor facts."
    )
    store_prompt = long_context + "\n\nStore the anchor facts. Reply exactly STORED."
    restart_prompt = store_prompt

    proc1: subprocess.Popen | None = None
    proc2: subprocess.Popen | None = None
    first_stopped_cleanly = False
    try:
        log1 = log_dir / f"{run_id}-before-restart.log"
        proc1, cmd1 = start_server(args, cache_dir, model_name, log1, env)
        health1_before = wait_health(args.port, proc1, args.timeout)
        url = f"http://127.0.0.1:{args.port}/v1/responses"
        t0 = time.perf_counter()
        store_response = post_json(
            url,
            response_payload(model_name, store_prompt, args.max_output_tokens),
            timeout=args.request_timeout,
        )
        store_wall = time.perf_counter() - t0
        stats_before_restart = get_json(
            f"http://127.0.0.1:{args.port}/v1/cache/stats", timeout=10
        )
        health1_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)
        first_pid = proc1.pid
        first_stopped_cleanly = stop_server(proc1)

        log2 = log_dir / f"{run_id}-after-restart.log"
        proc2, cmd2 = start_server(args, cache_dir, model_name, log2, env)
        health2_before = wait_health(args.port, proc2, args.timeout)
        t1 = time.perf_counter()
        restart_response = post_json(
            url,
            response_payload(model_name, restart_prompt, args.max_output_tokens),
            timeout=args.request_timeout,
        )
        restart_wall = time.perf_counter() - t1
        stats_after_restart = get_json(
            f"http://127.0.0.1:{args.port}/v1/cache/stats", timeout=10
        )
        health2_after = get_json(f"http://127.0.0.1:{args.port}/health", timeout=10)

        before_block = block_disk_stats(stats_before_restart)
        after_block = block_disk_stats(stats_after_restart)
        native = native_cache(health2_after)
        restart_detail = cache_detail(restart_response)
        restart_cached = cached_tokens(restart_response)
        checks = {
            "native_cache": native.get("cache_type") == "native_composite",
            "native_prefix": native.get("prefix") is True,
            "native_paged": native.get("paged") is True,
            "native_l2": native.get("block_disk_l2") is True,
            "generic_tq_kv_off": (native.get("generic_turboquant_kv") or {}).get("enabled")
            is False,
            "disk_write_before_restart": int(before_block.get("disk_writes") or 0) > 0
            and int(before_block.get("blocks_on_disk") or 0) > 0,
            "restart_l2_disk_hit": int(after_block.get("disk_hits") or 0) > 0,
            "restart_dsv4_cache_hit": restart_cached > 0 and "dsv4" in restart_detail,
            "same_block_disk_cache_dir": "--block-disk-cache-dir" in cmd1
            and "--block-disk-cache-dir" in cmd2
            and cmd1[cmd1.index("--block-disk-cache-dir") + 1]
            == cmd2[cmd2.index("--block-disk-cache-dir") + 1]
            == str(cache_dir),
            "fresh_run_nonce": run_id in store_prompt and run_id in restart_prompt,
            "restart_replayed_same_terminal_prompt": restart_prompt == store_prompt,
            "server_restarted": first_pid != proc2.pid and first_stopped_cleanly,
            "store_turn_fresh": cached_tokens(store_response) == 0
            and not cache_detail(store_response),
            "restart_visible_output_ok": extract_output_text(restart_response) == "STORED",
        }
        notes = [key for key, ok in checks.items() if ok is not True]
        status = "pass" if not notes else "review"
        return {
            "status": status,
            "notes": notes,
            "model": args.model,
            "run_id": run_id,
            "cache_dir": str(cache_dir),
            "env": {
                "DSV4_LONG_CTX": env["DSV4_LONG_CTX"],
                "DSV4_POOL_QUANT": env["DSV4_POOL_QUANT"],
                "PYTHONNOUSERSITE": env["PYTHONNOUSERSITE"],
                "VMLINUX_METAL_WS_REJECT_PCT": env.get("VMLINUX_METAL_WS_REJECT_PCT"),
                "VMLX_METAL_WS_REJECT_PCT": env.get("VMLX_METAL_WS_REJECT_PCT"),
            },
            "before_restart": {
                "cmd": cmd1,
                "pid": first_pid,
                "log_path": str(log1),
                "health_before": health1_before,
                "health_after": health1_after,
                "response": {
                    **store_response,
                    "wall_seconds": store_wall,
                    "output_text_extracted": extract_output_text(store_response),
                },
                "cache_stats": stats_before_restart,
                "stopped_cleanly": first_stopped_cleanly,
            },
            "after_restart": {
                "cmd": cmd2,
                "pid": proc2.pid,
                "log_path": str(log2),
                "health_before": health2_before,
                "health_after": health2_after,
                "response": {
                    **restart_response,
                    "wall_seconds": restart_wall,
                    "output_text_extracted": extract_output_text(restart_response),
                },
                "cache_stats": stats_after_restart,
            },
            "checks": checks,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic gate
        log_tail = ""
        for proc, suffix in ((proc1, "before"), (proc2, "after")):
            if proc is None:
                continue
            candidate = log_dir / f"{run_id}-{suffix}-restart.log"
            if candidate.exists():
                log_tail += f"\n--- {candidate} ---\n{candidate.read_text(errors='replace')[-12000:]}"
        return {
            "status": "error",
            "error": repr(exc),
            "model": args.model,
            "run_id": run_id,
            "cache_dir": str(cache_dir),
            "log_tail": log_tail,
        }
    finally:
        if proc1 is not None and proc1.poll() is None:
            stop_server(proc1)
        if proc2 is not None and proc2.poll() is None:
            stop_server(proc2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=resolve_default_model())
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--port", type=int, default=8843)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--words", type=int, default=3400)
    parser.add_argument("--max-output-tokens", type=int, default=192)
    parser.add_argument("--pool-quant", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--keep-existing-cache-dir", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"created_at": time.time(), **result}, indent=2))
    print(
        f"[dsv4-restart-l2] status={result.get('status')} "
        f"notes={result.get('notes')} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

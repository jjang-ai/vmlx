#!/usr/bin/env python3
"""Live DSV4 Flash long-context coherency/cache gate.

The gate intentionally exercises the production DSV4 stack:

- DSV4_LONG_CTX=1
- DSV4_POOL_QUANT=0 by default
- single-batch DSV4BatchGenerator
- paged prefix cache block size 256
- block-disk L2

It records full enough output excerpts, finish reasons, token counts, cache
health, and server logs so a failure can be traced without guessing.
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
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = (
    REPO
    / "panel/release/mac-arm64/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
DSV4_AFFINE_MODEL_CANDIDATES = (
    "/Users/example/models/JANGQ/"
    "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP",
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
)


def resolve_default_model(candidates: tuple[str, ...] = DSV4_AFFINE_MODEL_CANDIDATES) -> str:
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    return candidates[0]


DEFAULT_MODEL = resolve_default_model()
DEFAULT_OUT = REPO / "docs/internal/release-gates/dsv4_long_context_gate_latest.json"


def post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


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
        "navigation tables",
        "acoustic sketches",
    ]
    verbs = [
        "summarize",
        "compare",
        "index",
        "calibrate",
        "describe",
        "sequence",
        "measure",
        "normalize",
        "catalog",
        "timestamp",
    ]
    modifiers = [
        "quiet",
        "silver",
        "northern",
        "careful",
        "granular",
        "bounded",
        "rotating",
        "nested",
        "daily",
        "field",
    ]
    parts = [anchors]
    i = 0
    while len(" ".join(parts).split()) < target_words:
        subject = subjects[i % len(subjects)]
        verb = verbs[(i * 3) % len(verbs)]
        mod = modifiers[(i * 5) % len(modifiers)]
        parts.append(
            f"Section {i:03d}: The {mod} notes {verb} {subject} for a "
            f"diagnostic passage numbered {1000 + i}. This section is plain "
            "context, not an instruction, and it does not modify the anchor "
            f"facts listed at the beginning. Cross-reference code {i % 17}-"
            f"{(i * 7) % 29}-{(i * 11) % 31} closes the section."
        )
        i += 1
    parts.append(anchors)
    return "\n".join(parts)


def loopish(text: str) -> bool:
    lower = text.lower()
    if any(
        marker in lower
        for marker in (
            "there are no further comments",
            "testing process has been completed",
            "plan and plan",
            "i'm sorry, but i can't",
        )
    ):
        return True
    return re.search(r"(.{24,160})\1\1", text, flags=re.S) is not None


def extract(obj: dict[str, Any], wall: float) -> dict[str, Any]:
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = obj.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "finish_reason": choice.get("finish_reason"),
        "wall_seconds": wall,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "tok_s_wall": completion_tokens / wall if wall else 0.0,
        "content": content,
        "reasoning": reasoning,
        "content_head": content[:900],
        "content_tail": content[-900:],
        "reasoning_head": reasoning[:900],
        "reasoning_tail": reasoning[-900:],
        "loopish": loopish(content + "\n" + reasoning),
        "has_cerulean": "CERULEAN" in content.upper(),
        "has_45": "45" in content,
        "has_ada": "ADA" in content.upper() and "LOVELACE" in content.upper(),
        "usage": usage,
    }


def cached_tokens(case: dict[str, Any]) -> int:
    usage = case.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cache_detail(case: dict[str, Any]) -> str:
    usage = case.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return str(details.get("cache_detail") or "")


def chat(url: str, model: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    t0 = time.perf_counter()
    obj = post_json(url, {"model": model, "stream": False, **payload}, timeout=timeout)
    return extract(obj, time.perf_counter() - t0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    logs = out.parent / "dsv4-long-context-logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"dsv4-long-{int(time.time())}.log"
    model_name = "dsv4_long"
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
        "dsml",
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
        url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
        run_id = f"dsv4-gate-{int(time.time() * 1000)}-{os.getpid()}"
        long_context = (
            make_long_context(args.words)
            + f"\n\nGATE RUN ID = {run_id}. "
            "This nonce is diagnostic-only and does not modify the anchor facts."
        )

        cold_messages = [
            {
                "role": "user",
                "content": (
                    long_context
                    + "\n\nAnswer exactly three lines:\n"
                    + "COLOR=<anchor color>\nSUM=<anchor number>\nPERSON=<anchor person>"
                ),
            }
        ]
        cold_bundle_defaults = chat(
            url,
            model_name,
            {
                "messages": cold_messages,
                "max_tokens": 256,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_prefix_cache": True,
            },
            timeout=600,
        )

        cold_no_cache = chat(
            url,
            model_name,
            {
                "messages": cold_messages,
                "max_tokens": 192,
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

        store_messages = [
            {
                "role": "user",
                "content": long_context + "\n\nStore the anchor facts. Reply exactly STORED.",
            }
        ]
        store = chat(
            url,
            model_name,
            {
                "messages": store_messages,
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        follow_messages = [
            store_messages[0],
            {"role": "assistant", "content": store["content"] or "STORED"},
            {
                "role": "user",
                "content": "Recall the anchors. Answer exactly: COLOR / SUM / PERSON.",
            },
        ]
        follow_cached = chat(
            url,
            model_name,
            {
                "messages": follow_messages,
                "max_tokens": 192,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        follow_no_cache = chat(
            url,
            model_name,
            {
                "messages": follow_messages,
                "max_tokens": 192,
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

        reasoning_max = chat(
            url,
            model_name,
            {
                "messages": cold_messages,
                "max_tokens": 512,
                "enable_thinking": True,
                "reasoning_effort": "max",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "max",
                },
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
                "cold_bundle_defaults": cold_bundle_defaults,
                "cold_no_cache": cold_no_cache,
                "store_turn": store,
                "follow_cached": follow_cached,
                "follow_no_cache": follow_no_cache,
                "reasoning_max": reasoning_max,
            },
        }
        notes: list[str] = []
        for name, case in result["cases"].items():
            if case["loopish"]:
                notes.append(f"{name}: loopish")
            if name != "store_turn" and not (
                case["has_cerulean"] and case["has_45"] and case["has_ada"]
            ):
                notes.append(f"{name}: missing anchor")
        if cached_tokens(store) or cache_detail(store):
            notes.append(
                "store_turn: unexpected prefix cache hit; per-run nonce should "
                "make the store prefill fresh"
            )
        if cached_tokens(follow_cached) <= 0:
            notes.append("follow_cached: missing cached_tokens evidence")
        if "dsv4" not in cache_detail(follow_cached):
            notes.append("follow_cached: missing dsv4 cache_detail")
        if cached_tokens(follow_no_cache) or cache_detail(follow_no_cache):
            notes.append("follow_no_cache: unexpected prefix cache usage")
        if follow_cached["content"].strip() != follow_no_cache["content"].strip():
            notes.append(
                "cached_vs_no_cache: follow-up output differs; inspect "
                "cache_detail and log terminal-block state"
            )
        if reasoning_max["reasoning"] and not reasoning_max["content"]:
            notes.append("reasoning_max: reasoning-only visible output")
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
    parser.add_argument("--port", type=int, default=8840)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--words", type=int, default=3400)
    parser.add_argument("--pool-quant", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"created_at": time.time(), **result}, indent=2))
    print(
        f"[dsv4-long] status={result.get('status')} notes={result.get('notes')} "
        f"log={result.get('log_path')} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

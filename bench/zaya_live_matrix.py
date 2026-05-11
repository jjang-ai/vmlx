#!/usr/bin/env python3
"""Live ZAYA/ZAYA1-VL coherence, cache, and speed matrix.

This is intentionally product-path oriented:
- starts one vmlx-engine server per local bundle;
- uses the full single-user cache stack;
- sends OpenAI Chat Completions requests through HTTP;
- records health/cache stats, full streamed text, usage, and basic coherence
  checks.

It is not a replacement for the broader model-production gate. It is a focused
ZAYA release-blocker probe for text/VL, MXFP4/JANGTQ variants.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"
ENGINE = REPO / ".venv/bin/vmlx-engine"


@dataclass
class Row:
    name: str
    model_path: str
    port: int
    is_vl: bool


MATRIX = [
    Row("zaya_text_mxfp4", "/Users/example/models/JANGQ/ZAYA1-8B-MXFP4", 8110, False),
    Row("zaya_text_jangtq4", "/Users/example/jang/models/Zyphra/ZAYA1-8B-JANGTQ4", 8111, False),
    Row("zaya_text_jangtq2", "/Users/example/models/JANGQ/ZAYA1-8B-JANGTQ2", 8112, False),
    Row("zaya_vl_mxfp4", "/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4", 8113, True),
    Row("zaya_vl_jangtq2", "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ2", 8114, True),
]


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def wait_health(base_url: str, log_path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            health = request_json(f"{base_url}/health", timeout=2)
            if isinstance(health, dict) and health.get("status") == "healthy":
                return health
        except Exception as exc:  # noqa: BLE001 - artifact captures the boundary.
            last_error = repr(exc)
        time.sleep(1.0)
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
    raise RuntimeError(f"server did not become healthy: {last_error}\n{tail}")


def stream_chat(base_url: str, model: str, messages: list[dict[str, Any]], *,
                max_tokens: int = 96) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    chunks: list[str] = []
    usage: dict[str, Any] | None = None
    cache_detail = ""
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
                details = usage.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cache_detail = details.get("cache_detail") or cache_detail
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks.append(text)
    end = time.perf_counter()
    text = "".join(chunks)
    completion_tokens = (usage or {}).get("completion_tokens") or len(chunks)
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    visible_window = max((last or end) - (first or end), 1e-9)
    stream_close = max(end - (first or end), 1e-9)
    return {
        "request": payload,
        "text": text,
        "usage": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached,
        "cache_detail": cache_detail,
        "ttft_ms": ((first or end) - t0) * 1000,
        "visible_tps": completion_tokens / visible_window,
        "stream_close_tps": completion_tokens / stream_close,
        "wall_seconds": end - t0,
    }


def image_data_url() -> str:
    fixture = REPO / "docs/internal/release-gates/20260510_zaya1_vl_jangtq2_live_repro/image_ocr_app_like_request.json"
    try:
        data = json.loads(fixture.read_text())
        content = data["messages"][0]["content"]
        for item in content:
            if item.get("type") == "image_url":
                return item["image_url"]["url"]
    except Exception:
        pass
    # 1x1 PNG fallback. This will not pass OCR but keeps the failure explicit.
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    return f"data:image/png;base64,{b64}"


def classify_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    words = lower.replace("\n", " ").split()
    repeat = False
    if len(words) >= 12:
        for n in (1, 2, 3):
            grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
            if grams and max(grams.count(g) for g in set(grams)) >= 6:
                repeat = True
                break
    bad_markers = [
        "generation failed",
        "traceback",
        "runtimeerror",
        "<|object_ref_start|>",
        "<zyphra_tool_call>",
    ]
    return {
        "non_empty": bool(text.strip()),
        "repeat_risk": repeat,
        "bad_marker": next((m for m in bad_markers if m in lower), ""),
        "preview": text.strip()[:240],
        "tail": text.strip()[-240:],
    }


def simple_answer(text: str) -> str:
    return text.strip().upper().strip(" \t\r\n.!:")


def block_disk_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    root = stats.get("block_disk_cache")
    if isinstance(root, dict):
        return root
    nested = (stats.get("cache") or {}).get("block_disk_cache")
    return nested if isinstance(nested, dict) else {}


def start_server(row: Row, log_path: Path, block_dir: Path) -> subprocess.Popen:
    cmd = [
        str(ENGINE),
        "serve",
        row.model_path,
        "--host", "127.0.0.1",
        "--port", str(row.port),
        "--continuous-batching",
        "--max-num-seqs", "1",
        "--prefill-batch-size", "512",
        "--prefill-step-size", "2048",
        "--completion-batch-size", "512",
        "--use-paged-cache",
        "--enable-prefix-cache",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir", str(block_dir),
        "--max-cache-blocks", "1000",
        "--tool-call-parser", "zaya_xml",
        "--reasoning-parser", "auto",
        "--default-enable-thinking", "false",
        "--served-model-name", row.name,
        "--log-level", "INFO",
    ]
    env = os.environ.copy()
    env.pop("VMLINUX_ZAYA_ENABLE_TYPED_CCA_CACHE", None)
    env.pop("VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE", None)
    env["PYTHONUNBUFFERED"] = "1"
    log = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO),
        env=env,
        start_new_session=True,
    )
    log.close()
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_row(row: Row, root: Path) -> dict[str, Any]:
    row_dir = root / row.name
    row_dir.mkdir(parents=True, exist_ok=True)
    block_dir = row_dir / "block_l2"
    if block_dir.exists():
        shutil.rmtree(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    log_path = row_dir / "server.log"
    base_url = f"http://127.0.0.1:{row.port}"
    proc: subprocess.Popen | None = start_server(row, log_path, block_dir)
    try:
        health_before = wait_health(base_url, log_path, 180)
        (row_dir / "health_before.json").write_text(json.dumps(health_before, indent=2), encoding="utf-8")
        try:
            cache_before = request_json(f"{base_url}/v1/cache/stats", timeout=10)
        except Exception as exc:  # noqa: BLE001
            cache_before = {"error": repr(exc)}
        (row_dir / "cache_before.json").write_text(json.dumps(cache_before, indent=2), encoding="utf-8")

        messages: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        messages.append({"role": "user", "content": "Remember the code word CERULEAN. Reply exactly READY."})
        first = stream_chat(base_url, row.name, messages, max_tokens=48)
        turns.append(first)
        messages.append({"role": "assistant", "content": first["text"]})
        messages.append({"role": "user", "content": "What code word did I ask you to remember? Answer one word."})

        second = stream_chat(base_url, row.name, messages, max_tokens=48)
        turns.append(second)
        messages.append({"role": "assistant", "content": second["text"]})
        messages.append({"role": "user", "content": "Compute 17 + 28. Answer only the number."})
        third = stream_chat(base_url, row.name, messages, max_tokens=48)
        turns.append(third)

        image_turn = None
        if row.is_vl:
            image_messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                    {"type": "text", "text": "What uppercase word is shown in the image? Answer just the word."},
                ],
            }]
            image_turn = stream_chat(base_url, row.name, image_messages, max_tokens=48)

        speed_messages = [{"role": "user", "content": "Write one coherent paragraph of about 120 words explaining why cache correctness matters for multimodal models."}]
        speed = stream_chat(base_url, row.name, speed_messages, max_tokens=160)
        time.sleep(2.0)

        health_after = request_json(f"{base_url}/health", timeout=10)
        cache_after = request_json(f"{base_url}/v1/cache/stats", timeout=10)
        (row_dir / "health_after.json").write_text(json.dumps(health_after, indent=2), encoding="utf-8")
        (row_dir / "cache_after.json").write_text(json.dumps(cache_after, indent=2), encoding="utf-8")

        for i, turn in enumerate(turns, start=1):
            (row_dir / f"turn{i}.json").write_text(json.dumps(turn, indent=2), encoding="utf-8")
        if image_turn is not None:
            (row_dir / "image_turn.json").write_text(json.dumps(image_turn, indent=2), encoding="utf-8")
        (row_dir / "speed.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")

        stop_server(proc)
        proc = None

        restart_log = row_dir / "restart_server.log"
        proc = start_server(row, restart_log, block_dir)
        wait_health(base_url, restart_log, 180)
        restart_messages = [{"role": "user", "content": "Remember the code word CERULEAN. Reply exactly READY."}]
        restart_turn = stream_chat(base_url, row.name, restart_messages, max_tokens=48)
        restart_health = request_json(f"{base_url}/health", timeout=10)
        restart_cache = request_json(f"{base_url}/v1/cache/stats", timeout=10)
        (row_dir / "restart_turn.json").write_text(json.dumps(restart_turn, indent=2), encoding="utf-8")
        (row_dir / "health_restart.json").write_text(json.dumps(restart_health, indent=2), encoding="utf-8")
        (row_dir / "cache_restart.json").write_text(json.dumps(restart_cache, indent=2), encoding="utf-8")

        block_after = block_disk_stats(cache_after)
        block_restart = block_disk_stats(restart_cache)
        native_cache = (health_after or {}).get("native_cache") or {}

        classifications = [classify_text(t["text"]) for t in turns]
        if image_turn is not None:
            classifications.append(classify_text(image_turn["text"]))
        speed_classification = classify_text(speed["text"])
        checks = {
            "all_non_empty": all(c["non_empty"] for c in classifications),
            "no_repeat_risk": not any(c["repeat_risk"] for c in classifications),
            "no_bad_markers": not any(c["bad_marker"] for c in classifications + [speed_classification]),
            "first_ready_ok": simple_answer(first["text"]) == "READY",
            "recall_ok": simple_answer(second["text"]) == "CERULEAN",
            "math_ok": simple_answer(third["text"]) == "45",
            "cache_hit_seen": any((t.get("cached_tokens") or 0) > 0 for t in turns[1:]),
            "typed_cache_seen": any("zaya_cca" in (t.get("cache_detail") or "") for t in turns[1:]),
            "native_cache_schema_ok": native_cache.get("schema") == "zaya_cca_v1",
            "native_cache_paged_ok": native_cache.get("paged") is True,
            "native_cache_l2_ok": native_cache.get("block_disk_l2") is True,
            "generic_tq_kv_off_for_cca": ((health_after or {}).get("turboquant_kv_cache") or {}).get("enabled") is False,
            "block_l2_write_seen": (block_after.get("disk_writes") or 0) > 0,
            "restart_disk_hit_seen": (block_restart.get("disk_hits") or 0) > 0 or (restart_turn.get("cached_tokens") or 0) > 0,
            "speed_non_empty": speed_classification["non_empty"],
            "speed_no_repeat_risk": not speed_classification["repeat_risk"],
        }
        if row.is_vl:
            image_compact = (image_turn["text"] if image_turn else "").upper().replace(" ", "")
            checks["image_ok"] = image_turn is not None and (
                "VMLX" in image_compact or "VMLINUX" in image_compact
            )
        checks["pass"] = all(checks.values())
        summary = {
            "row": asdict(row),
            "checks": checks,
            "turns": turns,
            "image_turn": image_turn,
            "restart_turn": restart_turn,
            "turn_classifications": classifications,
            "speed": speed,
            "speed_classification": speed_classification,
            "health_runtime": {
                "model_type": health_after.get("model_type"),
                "engine_type": health_after.get("engine_type"),
                "native_cache": health_after.get("native_cache"),
                "scheduler": health_after.get("scheduler"),
                "quantization": health_after.get("quantization"),
            },
        }
        (row_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if proc is not None:
            stop_server(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(REPO / "docs/internal/release-gates/20260510_zaya_live_matrix_current"))
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = [r for r in MATRIX if not args.only or r.name in args.only]
    results = []
    for row in rows:
        if not Path(row.model_path).exists():
            results.append({"row": asdict(row), "error": "missing model path"})
            continue
        print(f"=== {row.name} ===", flush=True)
        try:
            result = run_row(row, root)
            results.append(result)
            checks = result["checks"]
            speed = result["speed"]
            print(
                f"{row.name}: pass={checks['pass']} recall={checks['recall_ok']} "
                f"math={checks['math_ok']} cache={checks['typed_cache_seen']} "
                f"speed_visible={speed['visible_tps']:.1f} speed_closed={speed['stream_close_tps']:.1f}",
                flush=True,
            )
            print(f"  t1={result['turns'][0]['text'][:120]!r}", flush=True)
            print(f"  t2={result['turns'][1]['text'][:120]!r}", flush=True)
            print(f"  t3={result['turns'][2]['text'][:120]!r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"row": asdict(row), "error": repr(exc)})
            print(f"{row.name}: ERROR {exc!r}", flush=True)

    summary = {"schema": "zaya_live_matrix.v1", "results": results}
    (root / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    passed = sum(1 for r in results if isinstance(r, dict) and r.get("checks", {}).get("pass"))
    print(f"matrix_passed={passed}/{len(results)} artifact_dir={root}", flush=True)
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

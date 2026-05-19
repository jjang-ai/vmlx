#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable))


def solid_png_base64(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw))
    data += chunk(b"IEND", b"")
    return base64.b64encode(data).decode("ascii")


BLUE_PNG = solid_png_base64(16, 16, (0, 0, 255))
RED_PNG = solid_png_base64(16, 16, (255, 0, 0))


def blue_video_data_url() -> str:
    import cv2
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            2.0,
            (32, 32),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open mp4v VideoWriter")
        for _ in range(8):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)  # BGR blue.
            writer.write(frame)
        writer.release()
        data = Path(tmp.name).read_bytes()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def build_probe_payloads(
    *,
    model: str,
    include_video: bool,
    max_tokens: int,
    long_probes: bool = False,
) -> list[dict[str, Any]]:
    image_prompt = (
        "First answer Blue. Then write two concise sentences about the image color "
        "so generation continues beyond a one-word reply."
        if long_probes
        else "What is the dominant color in this image? Reply one word."
    )
    no_image_prompt = (
        "First answer no. Then write one concise sentence explaining that the current message has no attached image."
        if long_probes
        else "Do you currently see an image in this message? Reply exactly yes or no."
    )
    image_history_prompt = (
        "Only consider this final user message, not earlier turns. First answer no. Then write one concise sentence."
        if long_probes
        else "In my current message, did I attach a new image? Reply exactly yes or no."
    )
    video_prompt = (
        "First answer Blue. Then write two concise sentences about the video color "
        "so generation continues beyond a one-word reply."
        if long_probes
        else "What is the dominant color in this video? Reply one word."
    )
    no_video_prompt = (
        "First answer no. Then write one concise sentence explaining that the current message has no attached video."
        if long_probes
        else "Do you currently see a video in this message? Reply exactly yes or no."
    )
    image_content = [
        {
            "type": "text",
            "text": image_prompt,
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + BLUE_PNG},
        },
    ]

    probes: list[dict[str, Any]] = [
        {
            "label": "vl_blue_image",
            "payload": {
                "model": model,
                "messages": [{"role": "user", "content": image_content}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
        {
            "label": "text_no_media_after_image",
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": no_image_prompt,
                    }
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
        {
            "label": "vl_image_history_then_text",
            "payload": {
                "model": model,
                "messages": [
                    {"role": "user", "content": image_content},
                    {"role": "assistant", "content": "blue"},
                    {
                        "role": "user",
                        "content": image_history_prompt,
                    },
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
    ]

    if include_video:
        video_content = [
            {
                "type": "text",
                "text": video_prompt,
            },
            {"type": "video_url", "video_url": {"url": blue_video_data_url()}},
        ]
        probes.extend(
            [
                {
                    "label": "vl_blue_video",
                    "payload": {
                        "model": model,
                        "messages": [{"role": "user", "content": video_content}],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "enable_thinking": False,
                        "video_fps": 2,
                        "video_max_frames": 4,
                    },
                },
                {
                    "label": "text_no_media_after_video",
                    "payload": {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": no_video_prompt,
                            }
                        ],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "enable_thinking": False,
                    },
                },
            ]
        )
    return probes


def build_repeat_media_cache_payloads(
    *,
    model: str,
    include_video: bool,
    max_tokens: int,
) -> list[dict[str, Any]]:
    image_prompt = (
        "What is the dominant color in this image? Answer one word, then one short sentence."
    )
    blue_image_content = [
        {"type": "text", "text": image_prompt},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + BLUE_PNG},
        },
    ]
    red_image_content = [
        {"type": "text", "text": image_prompt},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + RED_PNG},
        },
    ]
    probes: list[dict[str, Any]] = [
        {
            "label": "vl_blue_image_repeat_1",
            "payload": {
                "model": model,
                "messages": [{"role": "user", "content": blue_image_content}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
        {
            "label": "vl_blue_image_repeat_2",
            "payload": {
                "model": model,
                "messages": [{"role": "user", "content": blue_image_content}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
        {
            "label": "vl_red_image_changed",
            "payload": {
                "model": model,
                "messages": [{"role": "user", "content": red_image_content}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
    ]

    if include_video:
        video_prompt = (
            "What is the dominant color in this video? Answer one word, then one short sentence."
        )
        blue_video_url = blue_video_data_url()
        video_content = [
            {"type": "text", "text": video_prompt},
            {"type": "video_url", "video_url": {"url": blue_video_url}},
        ]
        for label, video_fps, video_max_frames in [
            ("vl_blue_video_repeat_1", 2, 4),
            ("vl_blue_video_repeat_2", 2, 4),
            ("vl_blue_video_changed_settings", 1, 2),
        ]:
            probes.append(
                {
                    "label": label,
                    "payload": {
                        "model": model,
                        "messages": [{"role": "user", "content": video_content}],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "enable_thinking": False,
                        "video_fps": video_fps,
                        "video_max_frames": video_max_frames,
                    },
                }
            )
    return probes


def build_env(base_env: dict[str, str], *, mode: str, args: argparse.Namespace) -> dict[str, str]:
    env = dict(base_env)
    enabled = "1" if mode == "mtp" else "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("VMLX_DISABLE_TQ_KV", None)
    env.pop("VMLINUX_DISABLE_TQ_KV", None)
    env["VMLINUX_NATIVE_MTP"] = enabled
    env["VMLX_NATIVE_MTP"] = enabled
    depth = max(1, min(3, int(getattr(args, "depth", 3) or 3)))
    env["VMLINUX_NATIVE_MTP_DEPTH"] = str(depth)
    env["VMLX_NATIVE_MTP_DEPTH"] = str(depth)
    trace = bool(getattr(args, "trace_mtp", False)) or (
        mode == "mtp" and bool(getattr(args, "cost_fallback", False))
    )
    env["VMLINUX_NATIVE_MTP_TRACE"] = "1" if trace else "0"
    if mode == "mtp" and bool(getattr(args, "cost_fallback", False)):
        env["VMLINUX_NATIVE_MTP_COST_FALLBACK"] = "1"
        ar_step_ms = getattr(args, "cost_ar_step_ms", None)
        if ar_step_ms is not None:
            env["VMLINUX_NATIVE_MTP_AR_STEP_MS"] = f"{float(ar_step_ms):.6f}"
        threshold = getattr(args, "cost_ratio_threshold", None)
        if threshold is not None:
            env["VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD"] = f"{float(threshold):.6f}"
    if bool(getattr(args, "media_prefix_cache", False)):
        env["VMLINUX_MLLM_MEDIA_PREFIX_CACHE"] = "1"
    return env


def request_json(
    method: str,
    url: str,
    body: Any | None = None,
    timeout: float = 300.0,
) -> tuple[Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else None, time.perf_counter() - t0


def wait_ready(base_url: str, deadline_s: float) -> dict[str, Any]:
    end = time.monotonic() + deadline_s
    last_error: str | None = None
    while time.monotonic() < end:
        try:
            health, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
            if isinstance(health, dict) and health.get("model_loaded"):
                return health
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            last_error = repr(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy before timeout: {last_error}")


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=10)


def log_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []


def start_server(
    *,
    model_path: Path,
    served_name: str,
    port: int,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[Any], dict[str, Any], Path]:
    env = build_env(os.environ.copy(), mode=mode, args=args)
    block_cache_dir = out_dir / "block_cache"
    log_path = out_dir / "server.log"
    cmd = [
        str(PYTHON),
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--is-mllm",
        "--max-num-seqs",
        "1",
        "--max-tokens",
        "512",
        "--log-level",
        "INFO",
        "--use-paged-cache",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(block_cache_dir),
        "--block-disk-cache-max-gb",
        "2",
        "--ssm-state-cache-mb",
        "1024",
    ]
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    try:
        health = wait_ready(f"http://127.0.0.1:{port}", float(args.load_timeout_s))
        return proc, health, log_path
    except Exception:
        terminate_process(proc)
        tail = "\n".join(log_lines(log_path)[-80:])
        raise RuntimeError(f"server failed for {mode}; log tail:\n{tail}")


def post_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        response, elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=300.0,
        )
        choice = ((response or {}).get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = (response or {}).get("usage") or {}
        content = message.get("content") or ""
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("reasoning_text")
            or ""
        )
        tokens = int(usage.get("completion_tokens") or 0)
        return {
            "ok": True,
            "elapsed_sec": elapsed,
            "completion_tps": tokens / elapsed if tokens and elapsed > 0 else None,
            "finish_reason": choice.get("finish_reason"),
            "content": content,
            "content_head": content[:240],
            "content_tail": content[-240:],
            "reasoning_content": reasoning,
            "reasoning_head": reasoning[:240],
            "usage": usage,
            "response": response,
        }
    except Exception as exc:  # noqa: BLE001 - persisted gate artifact
        return {
            "ok": False,
            "elapsed_sec": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cache_hit_tokens_by_detail(stats: dict[str, Any]) -> dict[str, Any]:
    scheduler = stats.get("scheduler_cache") or {}
    return scheduler.get("cache_hit_tokens_by_detail") or {}


def cache_counters(stats: dict[str, Any]) -> dict[str, Any]:
    scheduler = stats.get("scheduler_cache") or {}
    block = stats.get("block_disk_cache") or {}
    ssm = stats.get("ssm_companion") or {}
    return {
        "scheduler_hits": scheduler.get("hits"),
        "scheduler_misses": scheduler.get("misses"),
        "cache_hit_tokens_by_detail": scheduler.get("cache_hit_tokens_by_detail") or {},
        "block_disk_hits": block.get("disk_hits"),
        "block_disk_writes": block.get("disk_writes"),
        "ssm_entries": ssm.get("entries"),
        "ssm_stores": ssm.get("stores"),
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    requests = result.get("requests") or []
    by_label = {row.get("label"): row for row in requests}
    cache_after = result.get("cache_after") or {}
    block = cache_after.get("block_disk_cache") or {}
    ssm = cache_after.get("ssm_companion") or {}
    health = result.get("health_after") or result.get("health_before") or {}
    mtp = health.get("mtp") or {}
    native_cache = health.get("native_cache") or {}
    return {
        "mode": result.get("mode"),
        "mtp_runtime_active": mtp.get("runtime_active"),
        "mtp_effective_depth": mtp.get("effective_depth"),
        "generic_turboquant_kv": (native_cache.get("generic_turboquant_kv") or {}),
        "live_attention_tq_kv": (native_cache.get("live_attention_tq_kv") or {}),
        "all_http_ok": all(bool(row.get("ok")) for row in requests) if requests else False,
        "visible_nonempty": all(
            bool(str(row.get("content") or "").strip())
            for row in requests
            if row.get("ok")
        )
        if requests
        else False,
        "image_answer": str((by_label.get("vl_blue_image") or {}).get("content") or "").strip(),
        "no_media_after_image_answer": str(
            (by_label.get("text_no_media_after_image") or {}).get("content") or ""
        ).strip(),
        "image_history_then_text_answer": str(
            (by_label.get("vl_image_history_then_text") or {}).get("content") or ""
        ).strip(),
        "video_answer": str((by_label.get("vl_blue_video") or {}).get("content") or "").strip(),
        "no_media_after_video_answer": str(
            (by_label.get("text_no_media_after_video") or {}).get("content") or ""
        ).strip(),
        "repeat_media_answers": {
            str(row.get("label")): str(row.get("content") or "").strip()
            for row in requests
            if str(row.get("label") or "").startswith("vl_")
            and (
                "repeat" in str(row.get("label") or "")
                or "changed" in str(row.get("label") or "")
            )
        },
        "cache_counters": cache_counters(cache_after),
        "per_request_cache": {
            row.get("label"): cache_counters(row.get("cache_after") or {})
            for row in requests
            if row.get("cache_after")
        },
        "cache_hit_tokens_by_detail": cache_hit_tokens_by_detail(cache_after),
        "block_disk_hits": block.get("disk_hits"),
        "block_disk_writes": block.get("disk_writes"),
        "ssm_entries": ssm.get("entries"),
    }


def run_mode(
    *,
    model_path: Path,
    served_name: str,
    mode: str,
    port: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    proc, health_before, log_path = start_server(
        model_path=model_path,
        served_name=served_name,
        port=port,
        mode=mode,
        out_dir=mode_dir,
        args=args,
    )
    base_url = f"http://127.0.0.1:{port}"
    result: dict[str, Any] = {
        "mode": mode,
        "port": port,
        "model_path": str(model_path),
        "served_name": served_name,
        "health_before": health_before,
        "requests": [],
    }
    try:
        cache_before, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
        result["cache_before"] = cache_before
        probes = (
            build_repeat_media_cache_payloads(
                model=served_name,
                include_video=bool(args.include_video),
                max_tokens=int(args.max_tokens),
            )
            if bool(args.repeat_media_cache)
            else build_probe_payloads(
                model=served_name,
                include_video=bool(args.include_video),
                max_tokens=int(args.max_tokens),
                long_probes=bool(args.long_probes),
            )
        )
        for probe in probes:
            row = post_chat(base_url, probe["payload"])
            row["label"] = probe["label"]
            row["request"] = probe["payload"]
            try:
                row_cache_after, _ = request_json(
                    "GET", f"{base_url}/v1/cache/stats", timeout=30.0
                )
                row["cache_after"] = row_cache_after
                row["cache_counters_after"] = cache_counters(row_cache_after)
            except Exception as exc:  # noqa: BLE001 - persisted diagnostic only
                row["cache_after_error"] = f"{type(exc).__name__}: {exc}"
            result["requests"].append(row)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "label": row["label"],
                        "ok": row.get("ok"),
                        "tps": row.get("completion_tps"),
                        "content": str(row.get("content") or "")[:120],
                        "usage": row.get("usage"),
                        "cache": row.get("cache_counters_after"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        health_after, _ = request_json("GET", f"{base_url}/health", timeout=30.0)
        cache_after, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
        result["health_after"] = health_after
        result["cache_after"] = cache_after
    finally:
        terminate_process(proc)
        full_log = log_lines(log_path)
        result["log_signals"] = [
            line
            for line in full_log
            if "MLLM MTP[" in line
            or "native MTP" in line
            or "TurboQuant" in line
            or "Runtime cache layout:" in line
            or "VLM HYBRID cache" in line
            or "VLM prefix cache" in line
            or "prefix cache store" in line
            or "mllm_media" in line
            or "SSM" in line
        ][-120:]
        result["summary"] = summarize_result(result)
        (mode_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
    return result


def compare_rows(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {row.get("mode"): row for row in results}
    ar = by_mode.get("ar")
    mtp = by_mode.get("mtp")
    comparisons: list[dict[str, Any]] = []
    if not ar or not mtp:
        return {"comparisons": comparisons}
    ar_by_label = {row.get("label"): row for row in ar.get("requests") or []}
    for mtp_row in mtp.get("requests") or []:
        label = mtp_row.get("label")
        ar_row = ar_by_label.get(label)
        if not ar_row:
            continue
        comparisons.append(
            {
                "label": label,
                "content_equal": ar_row.get("content") == mtp_row.get("content"),
                "ar_ok": ar_row.get("ok"),
                "mtp_ok": mtp_row.get("ok"),
                "ar_content": ar_row.get("content"),
                "mtp_content": mtp_row.get("content"),
                "ar_usage": ar_row.get("usage"),
                "mtp_usage": mtp_row.get("usage"),
            }
        )
    return {
        "comparisons": comparisons,
        "all_content_equal": bool(comparisons)
        and all(row["content_equal"] for row in comparisons),
        "all_http_ok": bool(comparisons)
        and all(row["ar_ok"] and row["mtp_ok"] for row in comparisons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Private Qwen3.6 native-MTP VL gate.")
    parser.add_argument("model_path")
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--port", type=int, default=8340)
    parser.add_argument("--mode", choices=["ar", "mtp", "both"], default="mtp")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument(
        "--long-probes",
        action="store_true",
        help="Use longer image/video/no-media prompts so MTP verify cycles are exercised.",
    )
    parser.add_argument(
        "--repeat-media-cache",
        action="store_true",
        help="Send identical and changed media payloads to probe media cache hit/miss behavior.",
    )
    parser.add_argument(
        "--media-prefix-cache",
        action="store_true",
        help="Enable experimental media-keyed MLLM prefix cache for this gate.",
    )
    parser.add_argument("--trace-mtp", action="store_true")
    parser.add_argument("--cost-fallback", action="store_true")
    parser.add_argument("--cost-ar-step-ms", type=float, default=None)
    parser.add_argument("--cost-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    served_name = args.served_name or model_path.name.lower().replace("/", "-")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "docs/internal/release-gates" / f"{stamp}_native_mtp_vl_gate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["ar", "mtp"] if args.mode == "both" else [args.mode]
    results = []
    for idx, mode in enumerate(modes):
        results.append(
            run_mode(
                model_path=model_path,
                served_name=served_name,
                mode=mode,
                port=args.port + idx,
                out_dir=out_dir,
                args=args,
            )
        )
        time.sleep(2.0)

    summary = {
        "model_path": str(model_path),
        "served_name": served_name,
        "include_video": bool(args.include_video),
        "repeat_media_cache": bool(args.repeat_media_cache),
        "media_prefix_cache": bool(args.media_prefix_cache),
        "modes": {row["mode"]: row.get("summary") for row in results},
        "comparison": compare_rows(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

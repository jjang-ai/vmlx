#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable))
DEFAULT_SKIP_TERMS = ("kimi",)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-._")
    return cleaned[:96] or "model"


def _raw_model_type(config: dict[str, Any]) -> str:
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return str(config.get("model_type") or text_config.get("model_type") or "unknown")


def _has_key_recursive(obj: Any, key_names: set[str]) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_names:
                return True
            if _has_key_recursive(value, key_names):
                return True
    elif isinstance(obj, list):
        return any(_has_key_recursive(item, key_names) for item in obj)
    return False


def _positive_int_recursive(obj: Any, key_names: set[str]) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_names:
                try:
                    if int(value) > 0:
                        return True
                except (TypeError, ValueError):
                    pass
            if _positive_int_recursive(value, key_names):
                return True
    elif isinstance(obj, list):
        return any(_positive_int_recursive(item, key_names) for item in obj)
    return False


def _index_has_mtp_tensors(model_dir: Path) -> bool:
    index = read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
    return any(str(key).startswith("mtp.") for key in weight_map)


def classify_model_dir(model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    jang = read_json(model_dir / "jang_config.json")
    name = model_dir.name
    model_type = _raw_model_type(config)
    lower_blob = json.dumps({"name": name, "config": config, "jang": jang}, sort_keys=True).lower()

    is_mllm = (
        _has_key_recursive(config, {"vision_config", "visual", "image_token_id", "video_token_id"})
        or "-vl" in name.lower()
        or "vl-" in name.lower()
        or "omni" in name.lower()
        or model_type in {"gemma4", "zaya1_vl", "qwen2_5_vl", "qwen3_5_vl"}
    )
    name_lower = name.lower()
    name_has_mtp = "mtp" in name_lower and "nomtp" not in name_lower and "no-mtp" not in name_lower
    has_mtp = (
        jang.get("drop_mtp") is not True
        and (
            name_has_mtp
            or _index_has_mtp_tensors(model_dir)
            or _positive_int_recursive(
                config,
                {"mtp_num_hidden_layers", "num_nextn_predict_layers", "num_nextn_predict_layers"},
            )
        )
    )
    supports_video = bool(is_mllm and ("video" in lower_blob or "qwen" in lower_blob or "vl" in lower_blob or "omni" in lower_blob))
    supports_thinking = (
        "qwen" in lower_blob
        or "deepseek" in lower_blob
        or "gemma4" in lower_blob
        or "hy3" in lower_blob
        or "hy_v3" in lower_blob
    )
    if "ling" in lower_blob or "zaya" in lower_blob or "minimax" in lower_blob:
        supports_thinking = False

    if "deepseek_v4" in lower_blob or "deepseek-v4" in name.lower():
        cache_family = "deepseek_v4_composite"
    elif "zaya" in lower_blob:
        cache_family = "zaya_cca"
    elif "laguna" in lower_blob:
        cache_family = "swa_rotating"
    elif any(key in lower_blob for key in ("qwen3.6", "qwen3_5", "ling", "nemotron", "mamba", "ssm")):
        cache_family = "hybrid_ssm"
    elif any(key in lower_blob for key in ("jangtq", "mxtq", "turboquant", "tq_packed")):
        cache_family = "turboquant_kv"
    else:
        cache_family = "generic_kv"

    rel = model_dir
    served_name = sanitize_name(name.lower())
    return {
        "path": str(model_dir),
        "served_name": served_name,
        "name": name,
        "relative_path": str(rel),
        "model_type": model_type,
        "is_mllm": is_mllm,
        "supports_video": supports_video,
        "supports_thinking": supports_thinking,
        "has_mtp": has_mtp,
        "cache_family": cache_family,
        "has_jang_config": bool(jang),
        "namespace": "dealign.ai" if "dealign.ai" in str(model_dir) else ("JANGQ" if "JANGQ" in str(model_dir) else "other"),
    }


def discover_model_dirs(models_root: Path) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    for path in models_root.rglob("config.json"):
        candidates.add(path.parent)
    for path in models_root.rglob("jang_config.json"):
        candidates.add(path.parent)
    rows = [classify_model_dir(path) for path in sorted(candidates)]
    return sorted(rows, key=lambda row: (row["namespace"], row["name"], row["path"]))


def build_serve_command(
    row: dict[str, Any],
    *,
    port: int,
    out_dir: Path,
    py: str | None = None,
) -> list[str]:
    python = py or str(PYTHON)
    cmd = [
        python,
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
        "256",
        "--continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str(out_dir / "block_cache"),
        "--block-disk-cache-max-gb",
        "2",
        "--ssm-state-cache-mb",
        "1024",
        "--max-tokens",
        "512",
        "--log-level",
        "INFO",
        "--default-enable-thinking",
        "false",
    ]
    if row.get("is_mllm"):
        cmd.append("--is-mllm")
    return cmd


def build_server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if env.get("VMLINUX_BENCH_ISOLATED") == "1":
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
    else:
        env["PYTHONPATH"] = str(ROOT)
    return env


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


def blue_video_data_url() -> str | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

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
            return None
        for _ in range(8):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)
            writer.write(frame)
        writer.release()
        data = Path(tmp.name).read_bytes()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def _text_payload(model: str, prompt: str, max_tokens: int, *, thinking: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": bool(thinking),
    }


def build_probe_payloads(
    row: dict[str, Any],
    *,
    max_tokens: int,
    include_reasoning: bool,
    include_media: bool = True,
    include_video: bool = True,
) -> list[dict[str, Any]]:
    model = row["served_name"]
    cache_prompt = (
        "Cache probe fixed prefix. Read these stable words: alpha beta gamma delta "
        "epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau. "
        "Reply exactly: ACK"
    )
    probes: list[dict[str, Any]] = [
        {
            "label": "text_cache_repeat_1",
            "payload": _text_payload(model, cache_prompt, max_tokens, thinking=False),
        },
        {
            "label": "text_cache_repeat_2",
            "payload": _text_payload(model, cache_prompt, max_tokens, thinking=False),
        },
        {
            "label": "text_multiturn_recall",
            "payload": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Remember color=blue and animal=cat. Reply exactly: noted."},
                    {"role": "assistant", "content": "noted"},
                    {"role": "user", "content": "What color and animal did I ask you to remember?"},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "enable_thinking": False,
            },
        },
    ]
    if include_reasoning and row.get("supports_thinking"):
        probes.append(
            {
                "label": "reasoning_on",
                "payload": _text_payload(
                    model,
                    "Think briefly, then answer visibly with exactly: FINAL=OK",
                    max(256, max_tokens),
                    thinking=True,
                ),
            }
        )
    if include_media and row.get("is_mllm"):
        image_content = [
            {"type": "text", "text": "What is the dominant color in this image? Reply one word."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + BLUE_PNG}},
        ]
        red_image_content = [
            {"type": "text", "text": "What is the dominant color in this image? Reply one word."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + RED_PNG}},
        ]
        probes.extend(
            [
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
                    "payload": _text_payload(
                        model,
                        "Do you currently see an image in this message? Reply exactly yes or no.",
                        max_tokens,
                        thinking=False,
                    ),
                },
                {
                    "label": "vl_blue_image_repeat",
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
        )
        if include_video and row.get("supports_video"):
            video_url = blue_video_data_url()
            if video_url:
                video_content = [
                    {"type": "text", "text": "What is the dominant color in this video? Reply one word."},
                    {"type": "video_url", "video_url": {"url": video_url}},
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
                            "payload": _text_payload(
                                model,
                                "Do you currently see a video in this message? Reply exactly yes or no.",
                                max_tokens,
                                thinking=False,
                            ),
                        },
                    ]
                )
    return probes


def probe_options_from_capabilities(
    row: dict[str, Any],
    capabilities: dict[str, Any] | None,
    *,
    include_media: bool,
    include_video: bool,
) -> dict[str, bool]:
    if not include_media:
        return {"include_media": False, "include_video": False}
    modalities_raw = (capabilities or {}).get("modalities")
    modalities = {
        str(item).lower()
        for item in modalities_raw
        if isinstance(item, str)
    } if isinstance(modalities_raw, list) else set()
    if modalities:
        allows_image = bool(modalities & {"vision", "image", "images"})
        allows_video = bool(modalities & {"video", "videos"})
    else:
        allows_image = bool(row.get("is_mllm"))
        allows_video = bool(row.get("supports_video"))
    return {
        "include_media": bool(include_media and allows_image),
        "include_video": bool(include_media and include_video and allows_image and allows_video),
    }


def request_json(
    method: str,
    url: str,
    body: Any | None = None,
    *,
    timeout: float = 180.0,
) -> tuple[int, Any, float]:
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
                return response.status, json.loads(raw.decode("utf-8")), elapsed
            except Exception:
                return response.status, raw.decode("utf-8", "replace"), elapsed
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def extract_text(resp: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(resp, dict):
        return "", "", {}
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    choices = resp.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("reasoning_text")
            or ""
        )
        return content, reasoning, usage
    return "", "", usage


_CONTROL_FRAGMENT_RE = re.compile(r"(?:}<\?|<\?){2,}")
_TEMPLATE_FRAGMENT_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9_])(?:codeline|promcodeline|prep|instructions)(?:[^a-z0-9_]|$)"
)


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def validate_probe_response(
    label: str,
    code: int,
    content: str,
    reasoning: str = "",
) -> list[dict[str, Any]]:
    """Apply semantic gates so nonempty incoherent output cannot pass."""
    visible = str(content or "")
    stripped = visible.strip()
    failures: list[dict[str, Any]] = []
    if code != 200:
        failures.append({"label": label, "reason": "http_status", "code": code})
    if not stripped:
        failures.append(
            {
                "label": label,
                "reason": "empty_visible",
                "reasoning_chars": len(str(reasoning or "")),
            }
        )
        return failures

    control_fragments = len(re.findall(r"}<\?|<\?", stripped))
    if control_fragments >= 2 or _CONTROL_FRAGMENT_RE.search(stripped) or _TEMPLATE_FRAGMENT_RE.search(stripped):
        failures.append(
            {
                "label": label,
                "reason": "incoherent_visible_text",
                "content_head": stripped[:160],
            }
        )

    lower = stripped.lower()
    compact_lower = re.sub(r"\s+", "", lower)
    words = _word_set(stripped)
    if label.startswith("text_cache_repeat") and not ({"ack", "acknowledged"} & words):
        failures.append({"label": label, "reason": "expected_ack_missing", "missing": ["ACK"]})
    elif label == "text_multiturn_recall":
        missing = [term for term in ("blue", "cat") if term not in words]
        if missing:
            failures.append(
                {
                    "label": label,
                    "reason": "expected_recall_missing",
                    "missing": missing,
                }
            )
    elif label == "reasoning_on":
        has_final_ok = "final=ok" in compact_lower or ("final" in words and "ok" in words)
        if not has_final_ok:
            failures.append(
                {
                    "label": label,
                    "reason": "expected_final_ok_missing",
                    "missing": ["FINAL=OK"],
                }
            )
    elif label in {"vl_blue_image", "vl_blue_image_repeat", "vl_blue_video"}:
        if "blue" not in words:
            failures.append({"label": label, "reason": "expected_color_missing", "expected": "blue"})
    elif label == "vl_red_image_changed":
        if "red" not in words:
            failures.append({"label": label, "reason": "expected_color_missing", "expected": "red"})
    elif label in {"text_no_media_after_image", "text_no_media_after_video"}:
        says_no = "no" in words or "do not" in lower or "don't" in lower
        if not says_no:
            failures.append({"label": label, "reason": "expected_no_media_missing", "expected": "no"})
    return failures


def collect_probe_failures(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for req in requests:
        label = str(req.get("label") or "unknown")
        code = int(req.get("code") or 0)
        content = str(req.get("content") or "")
        reasoning = str(req.get("reasoning") or req.get("reasoning_head") or "")
        for failure in validate_probe_response(label, code, content, reasoning):
            failure.setdefault("content_head", req.get("content_head", content[:280]))
            if "reasoning_chars" not in failure and "reasoning_chars" in req:
                failure["reasoning_chars"] = req["reasoning_chars"]
            failures.append(failure)
    return failures


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def wait_ready(base_url: str, load_timeout_s: float, proc: subprocess.Popen[Any]) -> dict[str, Any]:
    end = time.monotonic() + load_timeout_s
    last: Any = None
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during load rc={proc.returncode}")
        code, health, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
        last = health
        if code == 200 and isinstance(health, dict) and health.get("model_loaded"):
            return health
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready: {last}")


def summarize_cache(cache_stats: Any) -> dict[str, Any]:
    text = json.dumps(cache_stats, sort_keys=True).lower() if cache_stats is not None else ""
    totals = {
        "cache_hit_tokens": 0,
        "cache_hits": 0,
        "disk_hits": 0,
        "hybrid_kv_without_ssm_hits": 0,
        "hybrid_kv_without_ssm_tokens": 0,
    }

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in totals and isinstance(value, (int, float)):
                    totals[key] += int(value)
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(cache_stats)
    return {
        **totals,
        "has_cache_hit": (
            totals["cache_hit_tokens"] > 0
            or totals["cache_hits"] > 0
            or totals["disk_hits"] > 0
        ),
        "mentions_turboquant": "turboquant" in text or "tq" in text,
        "mentions_ssm": "ssm" in text or "mamba" in text,
        "mentions_media_salt": "media" in text or "vision" in text,
        "mentions_rotating": "rotating" in text or "swa" in text,
        "has_hybrid_kv_without_ssm": (
            totals["hybrid_kv_without_ssm_hits"] > 0
            or totals["hybrid_kv_without_ssm_tokens"] > 0
        ),
    }


def run_model_row(
    row: dict[str, Any],
    *,
    port: int,
    out_root: Path,
    load_timeout_s: float,
    request_timeout_s: float,
    include_reasoning: bool,
    include_media: bool,
    include_video: bool,
) -> dict[str, Any]:
    row_dir = out_root / sanitize_name(row["namespace"] + "_" + row["name"])
    row_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_serve_command(row, port=port, out_dir=row_dir)
    log_path = row_dir / "server.log"
    result: dict[str, Any] = {
        "row": row,
        "port": port,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requests": [],
        "status": "starting",
    }
    write_json(row_dir / "start.json", result)

    env = build_server_env()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{port}"
    try:
        try:
            health_before = wait_ready(base_url, load_timeout_s, proc)
        except Exception as exc:
            result["status"] = "load_failed"
            result["error"] = repr(exc)
            result["server_log_tail"] = log_path.read_text(errors="replace")[-6000:]
            write_json(row_dir / "result.json", result)
            return result

        result["status"] = "loaded"
        result["health_before"] = health_before
        write_json(row_dir / "health_before.json", health_before)
        code, capabilities, _ = request_json(
            "GET",
            f"{base_url}/v1/models/{urllib.parse.quote(row['served_name'], safe='')}/capabilities",
            timeout=30.0,
        )
        result["capabilities"] = {"code": code, "body": capabilities}
        write_json(row_dir / "capabilities.json", result["capabilities"])
        code, cache_before, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
        result["cache_before"] = {"code": code, "body": cache_before, "summary": summarize_cache(cache_before)}
        write_json(row_dir / "cache_before.json", result["cache_before"])

        capability_body = capabilities if isinstance(capabilities, dict) else None
        probe_options = probe_options_from_capabilities(
            row,
            capability_body,
            include_media=include_media,
            include_video=include_video,
        )
        result["probe_options"] = probe_options
        for probe in build_probe_payloads(
            row,
            max_tokens=48,
            include_reasoning=include_reasoning,
            include_media=probe_options["include_media"],
            include_video=probe_options["include_video"],
        ):
            label = probe["label"]
            payload = probe["payload"]
            code, resp, elapsed = request_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                payload,
                timeout=request_timeout_s,
            )
            content, reasoning, usage = extract_text(resp)
            code_stats, cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
            request_record = {
                "label": label,
                "code": code,
                "elapsed_sec": elapsed,
                "content": content,
                "content_head": content[:280],
                "reasoning_head": reasoning[:280],
                "reasoning_chars": len(reasoning),
                "usage": usage,
                "cache_stats_code": code_stats,
                "cache_summary": summarize_cache(cache_stats),
            }
            request_record["validation_failures"] = validate_probe_response(
                label, code, content, reasoning
            )
            result["requests"].append(request_record)
            write_json(
                row_dir / f"{sanitize_name(label)}.json",
                {
                    "request": payload,
                    "response": resp,
                    "request_record": request_record,
                    "cache_stats": cache_stats,
                },
            )

        code, health_after, _ = request_json("GET", f"{base_url}/health", timeout=30.0)
        code_cache, cache_after, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
        result["health_after"] = {"code": code, "body": health_after}
        result["cache_after"] = {"code": code_cache, "body": cache_after, "summary": summarize_cache(cache_after)}
        failures = collect_probe_failures(result["requests"])
        result["status"] = "pass" if not failures else "probe_failed"
        result["failures"] = failures
        result["server_log_tail"] = log_path.read_text(errors="replace")[-6000:]
        write_json(row_dir / "result.json", result)
        return result
    finally:
        terminate_process(proc)


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    only: str | None,
    skip: str | None,
    include_dealign: bool,
    include_sources: bool,
    include_aux: bool,
) -> list[dict[str, Any]]:
    only_terms = [term.strip().lower() for term in (only or "").split(",") if term.strip()]
    skip_terms = [
        *DEFAULT_SKIP_TERMS,
        *[term.strip().lower() for term in (skip or "").split(",") if term.strip()],
    ]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join([row["path"], row["name"], row["model_type"], row["namespace"]]).lower()
        if not include_dealign and row["namespace"] == "dealign.ai":
            continue
        if not include_sources and "/sources/" in haystack:
            continue
        if not include_aux and (
            "/mtp_assistant" in haystack
            or "_upload_prep" in haystack
            or "/stage_jangq" in haystack
            or "/stage_osaurus" in haystack
        ):
            continue
        if only_terms and not any(term in haystack for term in only_terms):
            continue
        if skip_terms and any(term in haystack for term in skip_terms):
            continue
        filtered.append(row)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential local-model vMLX engine smoke matrix. Private/internal gate."
    )
    parser.add_argument("--models-root", default=str(Path.home() / "models"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--port", type=int, default=8840)
    parser.add_argument("--load-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--only", default=None, help="Comma-separated substring filters.")
    parser.add_argument(
        "--skip",
        default=None,
        help=f"Comma-separated substring filters to skip. Built-in skips: {', '.join(DEFAULT_SKIP_TERMS)}.",
    )
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--no-dealign", action="store_true")
    parser.add_argument("--include-sources", action="store_true")
    parser.add_argument("--include-aux", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else ROOT / "docs/internal/release-gates" / f"{stamp}_all_local_model_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = filter_rows(
        discover_model_dirs(Path(args.models_root).expanduser()),
        only=args.only,
        skip=args.skip,
        include_dealign=not args.no_dealign,
        include_sources=args.include_sources,
        include_aux=args.include_aux,
    )
    if args.max_models is not None:
        rows = rows[: args.max_models]
    inventory = {
        "created_at": stamp,
        "models_root": str(Path(args.models_root).expanduser()),
        "row_count": len(rows),
        "rows": rows,
    }
    write_json(out_root / "inventory.json", inventory)
    print(json.dumps({"out": str(out_root), "row_count": len(rows)}, indent=2), flush=True)
    if args.dry_run:
        return 0

    results = []
    any_failed = False
    for index, row in enumerate(rows):
        result = run_model_row(
            row,
            port=args.port,
            out_root=out_root,
            load_timeout_s=args.load_timeout_s,
            request_timeout_s=args.request_timeout_s,
            include_reasoning=not args.no_reasoning,
            include_media=not args.no_media,
            include_video=not args.no_video,
        )
        results.append(result)
        if result.get("status") != "pass":
            any_failed = True
        write_json(
            out_root / "summary.json",
            {
                "created_at": stamp,
                "completed": len(results),
                "row_count": len(rows),
                "results": results,
            },
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "model": row["name"],
                    "status": result.get("status"),
                    "failures": len(result.get("failures") or []),
                }
            ),
            flush=True,
        )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

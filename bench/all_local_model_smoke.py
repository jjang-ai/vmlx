#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
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
_PYTHON_ENV = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable)).expanduser()
PYTHON = _PYTHON_ENV if _PYTHON_ENV.is_absolute() else (ROOT / _PYTHON_ENV).resolve()
DEFAULT_SKIP_TERMS = ("kimi",)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _bundle_template_has_thinking_control(model_dir: Path) -> bool:
    """Return True if the local bundle owns an explicit thinking template rail."""
    candidates: list[str] = []
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str):
        candidates.append(template)
    for rel in ("chat_template.jinja", "chat_template.json"):
        path = model_dir / rel
        try:
            if path.is_file():
                candidates.append(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    joined = "\n".join(candidates).lower()
    return bool(
        "enable_thinking" in joined
        or "reasoning_effort" in joined
        or "<think" in joined
        or "</think" in joined
    )


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


def _bundle_declares_native_video(model_dir: Path, config: dict[str, Any], lower_blob: str) -> bool:
    if config.get("video_config") is not None:
        return True
    if (model_dir / "video_preprocessor_config.json").is_file():
        return True
    if "qwen" in lower_blob and _has_key_recursive(
        config,
        {"video_token_id", "video_token_index"},
    ):
        return True
    return False


def classify_model_dir(model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    jang = read_json(model_dir / "jang_config.json")
    name = model_dir.name
    model_type = _raw_model_type(config)
    lower_blob = json.dumps({"name": name, "config": config, "jang": jang}, sort_keys=True).lower()
    jang_capabilities = jang.get("capabilities") if isinstance(jang.get("capabilities"), dict) else {}
    config_capabilities = (
        config.get("capabilities") if isinstance(config.get("capabilities"), dict) else {}
    )
    capabilities = jang_capabilities or config_capabilities
    runtime = (
        jang.get("runtime")
        if isinstance(jang.get("runtime"), dict)
        else config.get("runtime")
        if isinstance(config.get("runtime"), dict)
        else {}
    )

    is_mllm = (
        _has_key_recursive(config, {"vision_config", "visual", "image_token_id", "video_token_id"})
        or "-vl" in name.lower()
        or "vl-" in name.lower()
        or "omni" in name.lower()
        or model_type in {"gemma4", "zaya1_vl", "qwen2_5_vl", "qwen3_5_vl"}
    )
    if model_type == "step3p7":
        is_mllm = False
    name_lower = name.lower()
    name_has_mtp = "mtp" in name_lower and "nomtp" not in name_lower and "no-mtp" not in name_lower
    has_mtp = (
        jang.get("drop_mtp") is not True
        and (
            name_has_mtp
            or runtime.get("bundle_has_mtp") is True
            or _index_has_mtp_tensors(model_dir)
            or _positive_int_recursive(
                config,
                {"mtp_num_hidden_layers", "num_nextn_predict_layers", "num_nextn_predict_layers"},
            )
        )
    )
    supports_video = bool(
        is_mllm
        and (
            _bundle_declares_native_video(model_dir, config, lower_blob)
        )
    )
    supports_thinking = (
        "qwen" in lower_blob
        or "deepseek" in lower_blob
        or "gemma4" in lower_blob
        or "hy3" in lower_blob
        or "hy_v3" in lower_blob
        or "minimax" in lower_blob
        or model_type == "mimo_v2"
        or model_type == "zaya1_vl"
    )
    non_reasoning_family_blob = " ".join((name_lower, model_type.lower()))
    if "ling" in non_reasoning_family_blob or model_type == "zaya":
        supports_thinking = False
    reasoning_capabilities = (
        capabilities.get("reasoning")
        if isinstance(capabilities.get("reasoning"), dict)
        else {}
    )
    if isinstance(capabilities.get("supports_thinking"), bool):
        supports_thinking = bool(capabilities["supports_thinking"])
    elif isinstance(reasoning_capabilities.get("supported"), bool):
        supports_thinking = bool(reasoning_capabilities["supported"])
    if model_type == "zaya1_vl" and not _bundle_template_has_thinking_control(model_dir):
        supports_thinking = False

    supports_tools = True
    tool_capabilities = (
        capabilities.get("tools")
        if isinstance(capabilities.get("tools"), dict)
        else {}
    )
    if isinstance(capabilities.get("supports_tools"), bool):
        supports_tools = bool(capabilities["supports_tools"])
    elif isinstance(tool_capabilities.get("supported"), bool):
        supports_tools = bool(tool_capabilities["supported"])
    elif model_type == "mimo_v2":
        supports_tools = True

    if model_type == "mimo_v2":
        cache_family = "mimo_v2_hybrid_swa"
    elif model_type in {"gemma4", "gemma4_unified"}:
        cache_family = "swa_rotating"
    elif "deepseek_v4" in lower_blob or "deepseek-v4" in name.lower():
        cache_family = "deepseek_v4_composite"
    elif "zaya" in lower_blob:
        cache_family = "zaya_cca"
    elif "laguna" in lower_blob:
        cache_family = "swa_rotating"
    elif model_type == "step3p7":
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
        "supports_tools": supports_tools,
        "has_mtp": has_mtp,
        "cache_family": cache_family,
        "capabilities": capabilities,
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
    dsv4_active = row.get("cache_family") == "deepseek_v4_composite"
    paged_block_size = "256" if dsv4_active else "64"
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
        paged_block_size,
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str((out_dir / "block_cache").resolve()),
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
    if dsv4_active:
        cmd.append("--dsv4-enable-prefix-cache")
    if row.get("is_mllm"):
        cmd.append("--is-mllm")
    extra_serve_args = os.environ.get("VMLINUX_BENCH_EXTRA_SERVE_ARGS", "").strip()
    if extra_serve_args:
        cmd.extend(shlex.split(extra_serve_args))
    return cmd


def build_server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if env.get("VMLINUX_BENCH_ISOLATED") == "1":
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
    else:
        pythonpath_parts = [str(ROOT)]
        extra_pythonpath = os.environ.get("VMLINUX_BENCH_EXTRA_PYTHONPATH", "")
        pythonpath_parts.extend(
            part for part in extra_pythonpath.split(os.pathsep) if part
        )
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def server_process_cwd(row_dir: Path) -> Path:
    if os.environ.get("VMLINUX_BENCH_ISOLATED") == "1":
        return row_dir.resolve()
    return ROOT


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


def _is_zaya_row(row: dict[str, Any]) -> bool:
    blob = " ".join(
        str(row.get(key, "")).lower()
        for key in ("name", "served_name", "model_type", "path", "cache_family")
    )
    return "zaya" in blob


def _exact_ack_cache_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = _text_payload(model, prompt, max_tokens, thinking=False)
    payload["messages"] = [
        {
            "role": "system",
            "content": "Output exactly ACK and nothing else. Do not explain or prefix it.",
        },
        {"role": "user", "content": prompt},
    ]
    return payload


def _zaya_cache_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return _text_payload(
        model,
        prompt,
        max_tokens,
        thinking=False,
    )


def _required_tool_payload(model: str, max_tokens: int, *, prompt_style: str = "natural") -> dict[str, Any]:
    if prompt_style == "json_call":
        tool_prompt = (
            "Call function record_fact with exactly these JSON arguments and "
            'no other value: {"value":"blue-cat"}. The string blue-cat is a '
            "literal value; include the hyphen and the letters cat. Return no prose."
        )
    else:
        tool_prompt = (
            "Use the record_fact tool exactly once. Its value argument must be "
            'the literal string "blue-cat"; include the hyphen and the letters '
            "cat. Do not answer in visible text."
        )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": tool_prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "description": "Record one exact fact for a smoke test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The exact value to record.",
                            }
                        },
                        "required": ["value"],
                    },
                },
            }
        ],
    }


def _tool_result_continuation_payload(model: str, max_tokens: int) -> dict[str, Any]:
    call_id = "call_smoke_record_fact"
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the record_fact tool exactly once. Its value argument "
                    'must be the literal string "blue-cat"; include the hyphen '
                    "and the letters cat. Do not answer in visible text."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "record_fact",
                            "arguments": "{\"value\":\"blue-cat\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "record_fact",
                "content": "{\"ok\":true,\"stored\":\"blue-cat\"}",
            },
            {
                "role": "user",
                "content": (
                    "The tool has completed. Do not call another tool. "
                    'Reply with exactly: STORED blue-cat. Include the hyphen '
                    "and the letters cat."
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "tool_choice": "none",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "description": "Record one exact fact for a smoke test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The exact value to record.",
                            }
                        },
                        "required": ["value"],
                    },
                },
            }
        ],
    }


def _strict_json_payload(model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON object and nothing else, preserving "
                    "the literal string blue-cat including the hyphen and the "
                    "letters cat: "
                    "{\"status\":\"ok\",\"value\":\"blue-cat\",\"count\":3}"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }


def _exact_code_payload(model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Output exactly these three Python lines, preserving spaces, "
                    "newlines, punctuation, and capitalization. Do not use markdown.\n"
                    "def add(a, b):\n"
                    "    return a + b\n"
                    "print(add(2, 3))"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }


def _is_lfm_row(row: dict[str, Any]) -> bool:
    blob = " ".join(
        str(row.get(key, "")).lower()
        for key in ("name", "served_name", "model_type", "path")
    )
    return "lfm2" in blob


def _is_nemotron_row(row: dict[str, Any]) -> bool:
    blob = " ".join(
        str(row.get(key, "")).lower()
        for key in ("name", "served_name", "model_type", "path")
    )
    return "nemotron" in blob or "nemo" in blob


def _is_qwen36_moe_mtp_row(row: dict[str, Any]) -> bool:
    blob = " ".join(
        str(row.get(key, "")).lower()
        for key in ("name", "served_name", "model_type", "path", "cache_family")
    )
    return (
        ("qwen3.6" in blob or "qwen36" in blob or "qwen3_5_moe" in blob)
        and ("35b" in blob or "moe" in blob or "a3b" in blob)
        and "mtp" in blob
    )


def _lfm_strict_probe_max_tokens(row: dict[str, Any], max_tokens: int) -> int:
    # LFM2 often emits an internal direct-answer prelude before the final terse
    # answer even when thinking is disabled. The API parser returns the final
    # visible answer cleanly once generation reaches it, but 48/96-token probes
    # can stop mid-prelude and create a false runtime failure. Keep the output
    # contract strict; only give LFM enough budget to reach its final answer.
    return max(512, max_tokens) if _is_lfm_row(row) else max_tokens


def _reasoning_probe_max_tokens(row: dict[str, Any], max_tokens: int) -> int:
    # ZAYA/Nemotron/Qwen3.6 MoE MTP thinking rails are valid, but a 256-token
    # probe can stop before the closing </think> and visible final answer.
    # Keep validation strict; give these families enough budget to prove the
    # advertised thinking mode instead of misclassifying them as empty-visible.
    if _is_zaya_row(row) or _is_nemotron_row(row) or _is_qwen36_moe_mtp_row(row):
        return max(512, max_tokens)
    return max(256, max_tokens)


def _tool_prompt_style(row: dict[str, Any]) -> str:
    blob = " ".join(
        str(row.get(key, "")).lower()
        for key in ("name", "served_name", "model_type", "path")
    )
    if "lfm2" in blob and "mxfp8" in blob:
        return "json_call"
    return "natural"


def _cache_probe_prompt(row: dict[str, Any]) -> str:
    if _is_zaya_row(row):
        return (
            "ZAYA typed CCA cache probe. Read the stable sequence: "
            "blue green blue amber violet. Which color word repeats? "
            "Answer exactly one lowercase word."
        )
    base = (
        "Cache probe fixed prefix. Read these stable words: alpha beta gamma delta "
        "epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau."
    )
    if row.get("cache_family") != "deepseek_v4_composite":
        return base
    anchors = " ".join(f"dsv4-native-cache-anchor-{index:03d}" for index in range(360))
    return (
        "DSV4 native cache probe. Keep this deterministic prefix unchanged and "
        f"answer according to the system instruction only. {anchors}"
    )


def _cache_probe_expected(row: dict[str, Any]) -> str:
    return "blue" if _is_zaya_row(row) else "ACK"


def _cache_probe_payload(row: dict[str, Any], model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    if _is_zaya_row(row):
        return _zaya_cache_payload(model, prompt, max_tokens)
    return _exact_ack_cache_payload(model, prompt, max_tokens)


def _no_media_payload(
    model: str,
    media_kind: str,
    max_tokens: int,
    *,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if row is not None and _is_zaya_row(row):
        return _text_payload(
            model,
            (
                f"If a {media_kind} file is attached to this current message, "
                "answer ATTACHED. Otherwise answer NONE."
            ),
            max_tokens,
            thinking=False,
        )
    sentinel = media_kind.upper()
    payload = _text_payload(
        model,
        (
            f"Look only at attachments in this current user message. If there "
            f"are no {media_kind} attachments, answer exactly NONE. Otherwise "
            f"answer exactly {sentinel}."
        ),
        max_tokens,
        thinking=False,
    )
    payload["messages"] = [
        {
            "role": "system",
            "content": (
                f"For this request, the current user message has zero {media_kind} "
                f"attachments. Answer exactly NONE if there is no {media_kind} "
                f"attachment; answer exactly {sentinel} only if a {media_kind} "
                "attachment is actually included in the current user message."
            ),
        },
        payload["messages"][0],
    ]
    return payload


def build_probe_payloads(
    row: dict[str, Any],
    *,
    max_tokens: int,
    include_reasoning: bool,
    include_media: bool = True,
    include_video: bool = True,
    include_tools: bool = False,
) -> list[dict[str, Any]]:
    model = row["served_name"]
    cache_prompt = _cache_probe_prompt(row)
    cache_expected = _cache_probe_expected(row)
    strict_max_tokens = _lfm_strict_probe_max_tokens(row, max_tokens)
    probes: list[dict[str, Any]] = [
        {
            "label": "text_cache_repeat_1",
            "payload": _cache_probe_payload(row, model, cache_prompt, strict_max_tokens),
            "expected_content": cache_expected,
        },
        {
            "label": "text_cache_repeat_2",
            "payload": _cache_probe_payload(row, model, cache_prompt, strict_max_tokens),
            "expected_content": cache_expected,
        },
        {
            "label": "text_multiturn_recall",
            "payload": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Remember color=blue and animal=cat. Reply exactly: noted."},
                    {"role": "assistant", "content": "noted"},
                    {
                        "role": "user",
                        "content": (
                            "What color and animal did I ask you to remember? "
                            "Reply with exactly the remembered color word, a "
                            "space, then the remembered animal word."
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": strict_max_tokens,
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
                    _reasoning_probe_max_tokens(row, max_tokens),
                    thinking=True,
                ),
            }
        )
    if include_tools and row.get("supports_tools", True):
        probes.append(
            {
                "label": "tool_required",
                "payload": _required_tool_payload(
                    model,
                    max(_lfm_strict_probe_max_tokens(row, max_tokens), 96),
                    prompt_style=_tool_prompt_style(row),
                ),
            }
        )
        probes.append(
            {
                "label": "tool_result_continuation",
                "payload": _tool_result_continuation_payload(
                    model,
                    max(_lfm_strict_probe_max_tokens(row, max_tokens), 96),
                ),
            }
        )
    probes.append(
        {
            "label": "structured_json_exact",
            "payload": _strict_json_payload(
                model,
                max(_lfm_strict_probe_max_tokens(row, max_tokens), 96),
            ),
        }
    )
    probes.append(
        {
            "label": "exact_code_whitespace",
            "payload": _exact_code_payload(
                model,
                max(_lfm_strict_probe_max_tokens(row, max_tokens), 96),
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
                    "payload": _no_media_payload(model, "image", max_tokens, row=row),
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
        if include_video:
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
                            "payload": _no_media_payload(
                                model,
                                "video",
                                max_tokens,
                                row=row,
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
    media = (capabilities or {}).get("media") if isinstance(capabilities, dict) else None
    modalities_raw = None
    if isinstance(media, dict) and isinstance(media.get("runtime_modalities"), list):
        modalities_raw = media.get("runtime_modalities")
    if modalities_raw is None:
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


def extract_tool_calls(resp: Any) -> list[dict[str, Any]]:
    if not isinstance(resp, dict):
        return []
    choices = resp.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls")
    return tool_calls if isinstance(tool_calls, list) else []


_CONTROL_FRAGMENT_RE = re.compile(r"(?:}<\?|<\?){2,}")
_TEMPLATE_FRAGMENT_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9_])(?:codeline|promcodeline|prep|instructions)(?:[^a-z0-9_]|$)"
)
_CJK_RE = re.compile(
    r"[\u1100-\u11ff\u3040-\u309f\u30a0-\u30ff\u3130-\u318f"
    r"\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7af"
    r"\ud7b0-\ud7ff\uf900-\ufaff\uff66-\uff9f"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf"
    r"\U0002ceb0-\U0002ebef\U0002f800-\U0002fa1f"
    r"\U00030000-\U0003134f]"
)
MAX_REASONING_ON_CHARS = 4096


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def validate_probe_response(
    label: str,
    code: int,
    content: str,
    reasoning: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    expected_content: str | None = None,
) -> list[dict[str, Any]]:
    """Apply semantic gates so nonempty incoherent output cannot pass."""
    visible = str(content or "")
    stripped = visible.strip()
    failures: list[dict[str, Any]] = []
    if code != 200:
        failures.append({"label": label, "reason": "http_status", "code": code})
    if not stripped and label != "tool_required":
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

    cjk_chars = len(_CJK_RE.findall(stripped))
    if cjk_chars:
        failures.append(
            {
                "label": label,
                "reason": "unexpected_cjk_visible_text",
                "cjk_chars": cjk_chars,
            }
        )

    lower = stripped.lower()
    compact_lower = re.sub(r"\s+", "", lower)
    words = _word_set(stripped)
    if label == "tool_required":
        expected_tool = "record_fact"
        expected_value = "blue-cat"
        if stripped:
            failures.append(
                {
                    "label": label,
                    "reason": "tool_visible_text_leak",
                    "content": stripped[:160],
                }
            )
        matching_call = None
        for call in tool_calls or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            if function.get("name") == expected_tool:
                matching_call = function
                break
        if matching_call is None:
            failures.append(
                {
                    "label": label,
                    "reason": "expected_tool_call_missing",
                    "expected_tool": expected_tool,
                }
            )
        else:
            arguments = matching_call.get("arguments")
            try:
                parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except Exception:
                parsed_args = None
            value = parsed_args.get("value") if isinstance(parsed_args, dict) else None
            if value != expected_value:
                failures.append(
                    {
                        "label": label,
                        "reason": "expected_tool_argument_missing",
                        "expected": {"value": expected_value},
                        "actual": parsed_args,
                    }
                )
    elif label == "tool_result_continuation":
        if tool_calls:
            failures.append(
                {
                    "label": label,
                    "reason": "unexpected_tool_call_after_tool_result",
                    "tool_calls": tool_calls,
                }
            )
        if stripped != "STORED blue-cat":
            failures.append(
                {
                    "label": label,
                    "reason": "expected_tool_result_summary_missing",
                    "expected": "STORED blue-cat",
                }
            )
        raw_markup_needles = (
            "<tool_call",
            "</tool_call",
            "<function",
            "</function",
            "tool_calls",
            "record_fact(",
        )
        if any(needle in stripped for needle in raw_markup_needles):
            failures.append(
                {
                    "label": label,
                    "reason": "raw_tool_markup_leak",
                    "content_head": stripped[:160],
                }
            )
    elif label == "structured_json_exact":
        expected_json = {"status": "ok", "value": "blue-cat", "count": 3}
        try:
            parsed_json = json.loads(stripped)
        except Exception as exc:
            parsed_json = None
            failures.append(
                {
                    "label": label,
                    "reason": "json_parse_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "content_head": stripped[:160],
                }
            )
        if parsed_json is not None and parsed_json != expected_json:
            failures.append(
                {
                    "label": label,
                    "reason": "json_exact_object_mismatch",
                    "expected": expected_json,
                    "actual": parsed_json,
                }
            )
    elif label == "exact_code_whitespace":
        expected_code = "def add(a, b):\n    return a + b\nprint(add(2, 3))"
        if visible != expected_code:
            failures.append(
                {
                    "label": label,
                    "reason": "exact_code_whitespace_mismatch",
                    "expected": expected_code,
                    "actual_head": visible[:160],
                }
            )
    elif label.startswith("text_cache_repeat") and stripped != (expected_content or "ACK"):
        failures.append(
            {
                "label": label,
                "reason": "expected_exact_ack_missing",
                "expected": expected_content or "ACK",
            }
        )
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
        reasoning_chars = len(str(reasoning or ""))
        if reasoning_chars > MAX_REASONING_ON_CHARS:
            failures.append(
                {
                    "label": label,
                    "reason": "reasoning_loop_too_long",
                    "reasoning_chars": reasoning_chars,
                    "max_reasoning_chars": MAX_REASONING_ON_CHARS,
                }
            )
    elif label in {"vl_blue_image", "vl_blue_image_repeat", "vl_blue_video"}:
        if "blue" not in words:
            failures.append({"label": label, "reason": "expected_color_missing", "expected": "blue"})
    elif label == "vl_red_image_changed":
        if "red" not in words:
            failures.append({"label": label, "reason": "expected_color_missing", "expected": "red"})
    elif label in {"text_no_media_after_image", "text_no_media_after_video"}:
        media_kind = "video" if label.endswith("_video") else "image"
        says_no = (
            "no" in words
            or "none" in words
            or "zero" in words
            or "text_only" in compact_lower
            or "textonly" in compact_lower
            or "text-only" in lower
            or "not included" in lower
            or "not provided" in lower
            or "not attached" in lower
            or "without" in words
            or "do not" in lower
            or "don't" in lower
        )
        says_affirmative_media = (
            lower in {media_kind, f"{media_kind}."}
            or lower.startswith(f"yes")
            or f"{media_kind} is included" in lower
            or f"{media_kind} attachment is included" in lower
        )
        if not says_no:
            failures.append({"label": label, "reason": "expected_no_media_missing", "expected": "no-or-none"})
        if says_affirmative_media:
            failures.append(
                {
                    "label": label,
                    "reason": "unexpected_media_carryover_claim",
                    "media_kind": media_kind,
                }
            )
    return failures


def collect_probe_failures(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for req in requests:
        label = str(req.get("label") or "unknown")
        code = int(req.get("code") or 0)
        content = str(req.get("content") or "")
        reasoning = str(req.get("reasoning") or req.get("reasoning_head") or "")
        tool_calls = req.get("tool_calls") if isinstance(req.get("tool_calls"), list) else []
        expected_content = (
            str(req.get("expected_content"))
            if req.get("expected_content") is not None
            else None
        )
        for failure in validate_probe_response(
            label,
            code,
            content,
            reasoning,
            tool_calls=tool_calls,
            expected_content=expected_content,
        ):
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
    include_tools: bool,
) -> dict[str, Any]:
    row_dir = out_root / sanitize_name(row["namespace"] + "_" + row["name"])
    row_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_serve_command(row, port=port, out_dir=row_dir)
    log_path = row_dir / "server.log"
    result: dict[str, Any] = {
        "row": row,
        "port": port,
        "command": cmd,
        "server_cwd": str(server_process_cwd(row_dir)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requests": [],
        "status": "starting",
    }
    write_json(row_dir / "start.json", result)

    env = build_server_env()
    server_cwd = server_process_cwd(row_dir)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=server_cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
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
        if not (
            isinstance(capability_body, dict)
            and (
                isinstance(capability_body.get("modalities"), list)
                or isinstance(
                    (capability_body.get("media") or {}).get("runtime_modalities")
                    if isinstance(capability_body.get("media"), dict)
                    else None,
                    list,
                )
            )
        ):
            row_capabilities = row.get("capabilities")
            capability_body = (
                row_capabilities if isinstance(row_capabilities, dict) else capability_body
            )
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
            include_tools=include_tools,
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
            tool_calls = extract_tool_calls(resp)
            code_stats, cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
            request_record = {
                "label": label,
                "code": code,
                "elapsed_sec": elapsed,
                "content": content,
                "content_head": content[:280],
                "reasoning_head": reasoning[:280],
                "reasoning_chars": len(reasoning),
                "tool_calls": tool_calls,
                "usage": usage,
                "cache_stats_code": code_stats,
                "cache_summary": summarize_cache(cache_stats),
            }
            if probe.get("expected_content") is not None:
                request_record["expected_content"] = probe["expected_content"]
            request_record["validation_failures"] = validate_probe_response(
                label,
                code,
                content,
                reasoning,
                tool_calls=tool_calls,
                expected_content=probe.get("expected_content"),
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
            or "/audio_tokenizer" in haystack
            or "/audio-tokenizer" in haystack
            or "/video_processor" in haystack
            or "/video-processor" in haystack
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
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="Add explicit Chat Completions tool_choice=required and tool-result continuation probes.",
    )
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
            include_tools=args.include_tools,
        )
        results.append(result)
        if result.get("status") != "pass":
            any_failed = True
        failed_count = sum(1 for item in results if item.get("status") != "pass")
        write_json(
            out_root / "summary.json",
            {
                "created_at": stamp,
                "status": "fail" if failed_count else "pass",
                "completed": len(results),
                "row_count": len(rows),
                "failed": failed_count,
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

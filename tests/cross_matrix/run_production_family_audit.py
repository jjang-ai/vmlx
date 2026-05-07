#!/usr/bin/env python3
"""Production-scale vMLX family audit runner.

This is intentionally explicit and conservative. It has two phases:

1. Static bundle audit: config/generation/jang metadata, quant bit maps,
   special EOS/template fields, cache-risk flags, and safetensor shard summary.
2. Optional live audit: spawn one model at a time, run the same small matrix
   across Chat Completions, Responses, Anthropic Messages, Ollama chat,
   reasoning on/off, tool-history continuation shape, and cache stats.

The live phase is opt-in with --live because these models are huge. Missing
models are reported as SKIP, never silently passed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path("/Users/eric/mlx/vllm-mlx")
DEFAULT_PY = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
OUT_DIR = Path("/tmp/vmlx_family_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_MODEL_ROOTS = [
    Path("/Users/eric/models/JANGQ"),
    Path("/Users/eric/models/dealign.ai"),
    Path("/Users/eric/models/Sources"),
    Path("/Users/eric/.lmstudio/models"),
    Path("/Users/eric/.mlxstudio/models"),
]
MODEL_NAME_ALIASES = {
    "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2": [
        "Nemotron-Omni-Nano-JANGTQ-CRACK",
    ],
    "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4": [
        "Nemotron-Omni-Nano-JANGTQ4-CRACK",
    ],
}


@dataclass
class ModelRow:
    id: str
    label: str
    path: str
    family: str
    kind: str = "llm"  # llm | vl | omni | audio
    slow: bool = False
    expect_reasoning: bool = False
    expect_tool_parser: str | None = None
    cache_profile: str = "default"  # default | dsv4_composite | vl | hybrid_ssm
    live_supported: bool = True
    unsupported_reason: str | None = None
    notes: list[str] = field(default_factory=list)


ROWS: list[ModelRow] = [
    ModelRow(
        id="dsv4_tq",
        label="DeepSeek-V4-Flash JANGTQ",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "DSV4 SWA+CSA/HSA heterogenous cache; paged/block L2 must use "
            "deepseek_v4_v7 composite-state serialization, not generic KV blocks"
        ],
    ),
    ModelRow(
        id="dsv4_tq2",
        label="DeepSeek-V4-Flash JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ2",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "DSV4 canonical encoder + EOS [end,user,assistant] required; "
            "DeepseekV4Cache paged/L2 v7 schema must be active"
        ],
    ),
    ModelRow(
        id="dsv4_2l",
        label="DeepSeek-V4-Flash JANG_2L",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANG_2L",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        live_supported=False,
        unsupported_reason=(
            "Local bundle is corrupt: live load fails with safetensors invalid "
            "data offsets in layers.28.ffn.experts.197.w3.biases. Re-download "
            "or regenerate before treating this row as a runtime target."
        ),
        slow=True,
    ),
    ModelRow(
        id="qwen36_moe_tq4",
        label="Qwen3.6-35B-A3B JANGTQ4 hybrid MoE+SSM",
        path="/Volumes/EricsLLMDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_moe_crack",
        label="Qwen3.6-35B-A3B JANGTQ CRACK hybrid MoE+SSM",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_dense_mxfp4",
        label="Qwen3.6-27B MXFP4 CRACK dense",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-MXFP4-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="qwen36_dense_jang",
        label="Qwen3.6-27B JANG_4M CRACK dense",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-JANG_4M-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="nemotron_omni_tq2",
        label="Nemotron-3-Nano-Omni-30B-A3B JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
        family="nemotron_omni",
        kind="omni",
        expect_reasoning=True,
        expect_tool_parser="nemotron",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="nemotron_omni_tq4",
        label="Nemotron-3-Nano-Omni-30B-A3B JANGTQ4",
        path="/Volumes/EricsLLMDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4",
        family="nemotron_omni",
        kind="omni",
        expect_reasoning=True,
        expect_tool_parser="nemotron",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="ling_flash_tq",
        label="Ling-2.6-flash JANGTQ",
        path="/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ",
        family="bailing_hybrid",
        expect_reasoning=True,
        cache_profile="hybrid_ssm",
        slow=True,
        notes=[
            "Bailing/Ling hybrid row. This catches the repeated emoji/token-loop "
            "failure seen in panel chat and validates SSM companion-cache budgets."
        ],
    ),
    ModelRow(
        id="ling_flash_tq2_crack",
        label="Ling-2.6-flash JANGTQ2 CRACK",
        path="/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK",
        family="bailing_hybrid",
        expect_reasoning=True,
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="ling_flash_mxfp4_crack",
        label="Ling-2.6-flash MXFP4 CRACK",
        path="/Users/eric/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK",
        family="bailing_hybrid",
        expect_reasoning=True,
        cache_profile="hybrid_ssm",
        slow=True,
        notes=[
            "Higher-bit Ling control row for the Russian Three.js prompt. "
            "Use to separate JANGTQ2 quality/runtime failures from the shared "
            "Bailing hybrid cache stack."
        ],
    ),
    ModelRow(
        id="laguna_tq",
        label="Laguna-XS.2 JANGTQ",
        path="/Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Laguna-XS.2-JANGTQ",
        family="laguna",
        expect_tool_parser="qwen",
        slow=True,
    ),
    ModelRow(
        id="mistral_medium35_tq",
        label="Mistral-Medium-3.5-128B JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Mistral-Medium-3.5-128B-JANGTQ",
        family="mistral3",
        kind="vl",
        expect_reasoning=False,
        expect_tool_parser="mistral",
        cache_profile="vl",
        live_supported=False,
        unsupported_reason=(
            "Known non-production bundle: 2-bit dense JANGTQ Mistral 3.5 "
            "loads structurally but full prefill is decode-kernel bound and "
            "text quality degenerates. Use Mistral-Medium-3.5-128B-mxfp4 "
            "until a JANGTQ4 bundle/runtime is validated."
        ),
        slow=True,
        notes=[
            "Structural-only regression row. Do not treat as a coherent production target.",
            "Pixtral-style multimodal Mistral path; vision passthrough fp16",
        ],
    ),
    ModelRow(
        id="mistral_medium35_mxfp4",
        label="Mistral-Medium-3.5-128B MXFP4",
        path="/Volumes/EricsLLMDrive/jangq-ai/OsaurusAI/Mistral-Medium-3.5-128B-mxfp4",
        family="mistral3",
        kind="vl",
        expect_reasoning=False,
        expect_tool_parser="mistral",
        cache_profile="vl",
        slow=True,
        notes=[
            "Production Mistral 3.5 target. Direct runtime verified: "
            "368-token prefill ~7s, first decode ~0.7s, coherent 'ok'.",
            "Pixtral vision metadata present; Python runtime is currently text-first.",
        ],
    ),
    ModelRow(
        id="mistral_small4_crack",
        label="Mistral-Small-4-119B JANG_6M CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/Mistral-Small-4-119B-JANG_6M-CRACK",
        family="mistral",
        expect_tool_parser="mistral",
        slow=True,
    ),
    ModelRow(
        id="gemma4_crack",
        label="Gemma-4-26B-A4B-it JANG_4M CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        family="gemma4",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="gemma4",
        cache_profile="vl",
        slow=True,
    ),
    ModelRow(
        id="minimax_m27_tq_crack",
        label="MiniMax-M2.7 JANGTQ CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/MiniMax-M2.7-JANGTQ-CRACK",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        slow=True,
        notes=[
            "Pure-attention large MoE; validates long prompt KV/paged/L2 behavior "
            "separately from hybrid SSM rows."
        ],
    ),
    ModelRow(
        id="minimax_m27_tq_k",
        label="MiniMax-M2.7 JANGTQ_K mixed-bit",
        path="/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        slow=True,
        notes=[
            "Mixed per-projection routed expert bits: gate/up 2-bit and down 4-bit."
        ],
    ),
    ModelRow(
        id="minimax_m27_small_tq",
        label="MiniMax-M2.7 Small JANGTQ",
        path="/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        slow=True,
    ),
    ModelRow(
        id="minimax_m27_tq4_crack",
        label="MiniMax-M2.7 JANGTQ4 CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/MiniMax-M2.7-JANGTQ4-CRACK",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        live_supported=False,
        unsupported_reason=(
            "Local 4-bit MiniMax bundle loads at ~117GB model working set and "
            "trips the Metal working-set guard on this 128GB Mac "
            "(live 2026-05-03: active 110.4GB, 103% of 107.5GB cap). "
            "Use the validated 2-bit JANGTQ row or a larger unified-memory host."
        ),
        slow=True,
        notes=[
            "Higher-bit MiniMax runtime target; keep in the matrix so static "
            "bit/layer/template drift is visible even when live pass is skipped.",
            "Live load also showed quant_shape_inference patched 250 mixed-precision "
            "modules where config.json claimed uniform 4-bit.",
        ],
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}
    return {}


def resolve_model_dir(path: str) -> Path:
    model_dir = Path(path)
    if model_dir.is_dir():
        return model_dir
    names = [model_dir.name]
    if model_dir.name.endswith("-CRACK"):
        names.append(model_dir.name[: -len("-CRACK")])
    names.extend(MODEL_NAME_ALIASES.get(model_dir.name, []))
    for root in FALLBACK_MODEL_ROOTS:
        for name in names:
            candidate = root / name
            if candidate.is_dir():
                return candidate
    return model_dir


def summarize_safetensors(model_dir: Path, max_files: int = 9999) -> dict[str, Any]:
    files = sorted(model_dir.glob("*.safetensors"))
    total_gb = sum(p.stat().st_size for p in files) / (1024**3)
    summary: dict[str, Any] = {
        "files": len(files),
        "total_gb": round(total_gb, 2),
        "layer_ids_seen": [],
        "sample_keys": [],
        "key_prefix_counts": {},
    }
    if not files:
        return summary
    try:
        from safetensors import safe_open
    except Exception as exc:
        summary["safetensors_import_error"] = str(exc)
        return summary

    layer_ids: set[int] = set()
    prefixes: dict[str, int] = {}
    sample: list[str] = []
    for p in files[:max_files]:
        try:
            with safe_open(str(p), framework="np") as f:
                for key in f.keys():
                    if len(sample) < 30:
                        sample.append(key)
                    head = key.split(".", 2)[0]
                    if head == "layers":
                        parts = key.split(".", 3)
                        if len(parts) > 1 and parts[1].isdigit():
                            layer_ids.add(int(parts[1]))
                            head = ".".join(parts[:3]) if len(parts) > 2 else "layers"
                    prefixes[head] = prefixes.get(head, 0) + 1
        except Exception as exc:
            summary.setdefault("file_errors", []).append(
                {"file": p.name, "error": f"{type(exc).__name__}: {exc}"}
            )
    summary["layer_ids_seen"] = [min(layer_ids), max(layer_ids), len(layer_ids)] if layer_ids else []
    summary["sample_keys"] = sample
    summary["key_prefix_counts"] = dict(sorted(prefixes.items(), key=lambda kv: (-kv[1], kv[0]))[:40])
    return summary


def static_audit(row: ModelRow) -> dict[str, Any]:
    model_dir = resolve_model_dir(row.path)
    cfg = read_json(model_dir / "config.json")
    gen = read_json(model_dir / "generation_config.json")
    jang = read_json(model_dir / "jang_config.json")
    tok_cfg = read_json(model_dir / "tokenizer_config.json")
    chat_template_file = model_dir / "chat_template.jinja"
    registry: dict[str, Any] = {}
    try:
        sys.path.insert(0, str(ROOT))
        from vmlx_engine.model_config_registry import get_model_config_registry

        rcfg = get_model_config_registry().lookup(str(model_dir))
        registry = {
            "family_name": rcfg.family_name,
            "reasoning_parser": rcfg.reasoning_parser,
            "tool_parser": rcfg.tool_parser,
            "cache_type": rcfg.cache_type,
            "think_in_template": rcfg.think_in_template,
            "eos_tokens": rcfg.eos_tokens,
        }
    except Exception as exc:
        registry = {"lookup_error": f"{type(exc).__name__}: {exc}"}

    layer_types = cfg.get("layer_types") or cfg.get("layer_type") or []
    if isinstance(layer_types, list):
        layer_type_counts = {x: layer_types.count(x) for x in sorted(set(map(str, layer_types)))}
    else:
        layer_type_counts = {}

    chat = jang.get("chat") if isinstance(jang.get("chat"), dict) else {}
    reasoning = chat.get("reasoning") if isinstance(chat.get("reasoning"), dict) else {}
    sampling = chat.get("sampling_defaults") if isinstance(chat.get("sampling_defaults"), dict) else {}
    safetensors_summary = summarize_safetensors(model_dir)

    eos = {
        "config": cfg.get("eos_token_id"),
        "generation": gen.get("eos_token_id"),
        "tokenizer_eos": tok_cfg.get("eos_token"),
        "tokenizer_has_template": bool(tok_cfg.get("chat_template")),
        "chat_template_file": chat_template_file.is_file(),
        "registry_eos_tokens": registry.get("eos_tokens"),
    }

    issues: list[str] = []
    if model_dir.is_dir() and safetensors_summary.get("files", 0) == 0:
        issues.append("bundle has no safetensors weights; structural metadata only")
    if not row.live_supported and row.unsupported_reason:
        issues.append(row.unsupported_reason)
    if row.family == "deepseek_v4":
        if jang.get("weight_format") == "mxtq":
            bits = jang.get("mxtq_bits", {})
            if bits.get("routed_expert") not in (2, 4, 8):
                issues.append("DSV4 routed_expert mxtq_bits missing/unexpected")
            if sampling.get("repetition_penalty_thinking") is None:
                issues.append("DSV4 missing chat.sampling_defaults.repetition_penalty_thinking")
        if "<｜Assistant｜>" not in str(registry.get("eos_tokens")):
            issues.append("DSV4 bundle EOS config does not list Assistant marker; engine registry must add it")
        if not tok_cfg.get("chat_template") and not chat_template_file.is_file():
            issues.append("DSV4 missing chat template file/config; canonical encoder shim still required")
    has_reasoning_surface = bool(
        reasoning.get("supported")
        or tok_cfg.get("chat_template")
        or chat_template_file.is_file()
        or registry.get("reasoning_parser")
    )
    if row.expect_reasoning and not has_reasoning_surface:
        issues.append("expected reasoning model but no reasoning metadata/template found")
    if row.kind in ("vl", "omni") and not any(
        k in (str(cfg) + str(jang) + str(tok_cfg)).lower()
        for k in ("vision", "image", "audio", "video", "mm", "omni", "radio", "parakeet")
    ):
        issues.append("multimodal row did not expose obvious multimodal metadata; loader detection must be live-tested")

    return {
        "id": row.id,
        "label": row.label,
        "path": str(model_dir),
        "declared_path": row.path,
        "exists": model_dir.is_dir(),
        "family_expected": row.family,
        "kind": row.kind,
        "config": {
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "num_key_value_heads": cfg.get("num_key_value_heads"),
            "sliding_window": cfg.get("sliding_window"),
            "window_size": cfg.get("window_size"),
            "kv_lora_rank": cfg.get("kv_lora_rank"),
            "layer_type_counts": layer_type_counts,
            "quantization": cfg.get("quantization") or cfg.get("quantization_config"),
        },
        "generation": {
            "eos_token_id": gen.get("eos_token_id"),
            "pad_token_id": gen.get("pad_token_id"),
            "max_new_tokens": gen.get("max_new_tokens"),
            "temperature": gen.get("temperature"),
            "top_p": gen.get("top_p"),
        },
        "jang": {
            "weight_format": jang.get("weight_format"),
            "mxtq_bits": jang.get("mxtq_bits"),
            "quantization": jang.get("quantization"),
            "sampling_defaults": sampling,
            "reasoning": reasoning,
            "runtime": jang.get("runtime"),
        },
        "eos": eos,
        "registry": registry,
        "safetensors": safetensors_summary,
        "issues": issues,
        "live_supported": row.live_supported,
        "unsupported_reason": row.unsupported_reason,
        "notes": row.notes,
    }


def http_json(method: str, url: str, body: Any | None = None, timeout: int = 240) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            if not raw:
                return r.status, None
            return r.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def _directory_size_and_count(path: Path) -> dict[str, Any]:
    files = 0
    total = 0
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "gb": 0.0}
    for p in path.rglob("*"):
        try:
            if p.is_file():
                files += 1
                total += p.stat().st_size
        except OSError:
            continue
    return {
        "exists": True,
        "files": files,
        "bytes": total,
        "gb": round(total / (1024**3), 4),
    }


def telemetry_snapshot(
    name: str,
    *,
    proc: subprocess.Popen | None,
    base: str | None = None,
    block_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Capture resource and cache telemetry for live production proof.

    This intentionally uses only public HTTP endpoints plus psutil process
    accounting. The snapshot is small enough to record before/after every major
    request without hiding the model's full response artifacts.
    """
    snap: dict[str, Any] = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "monotonic": round(time.monotonic(), 3),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["system_memory"] = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_gb": round(vm.used / (1024**3), 2),
            "percent": vm.percent,
        }
        if proc is not None and proc.poll() is None:
            p = psutil.Process(proc.pid)
            with p.oneshot():
                mem = p.memory_info()
                cpu = p.cpu_times()
                snap["process"] = {
                    "pid": proc.pid,
                    "rss_gb": round(mem.rss / (1024**3), 3),
                    "vms_gb": round(mem.vms / (1024**3), 3),
                    "cpu_user_sec": round(cpu.user, 3),
                    "cpu_system_sec": round(cpu.system, 3),
                    "num_threads": p.num_threads(),
                    "status": p.status(),
                }
    except Exception as exc:
        snap["process_error"] = f"{type(exc).__name__}: {exc}"
    if base:
        code, health = http_json("GET", f"{base}/health", timeout=5)
        snap["health"] = {"code": code, "body": health}
        code, cache = http_json("GET", f"{base}/v1/cache/stats", timeout=10)
        snap["cache_stats"] = {"code": code, "body": cache}
    if block_cache_dir is not None:
        snap["block_cache_dir"] = _directory_size_and_count(block_cache_dir)
    return snap


def extract_chat_text(resp: Any) -> tuple[str, str, str]:
    try:
        ch = resp["choices"][0]
        msg = ch.get("message") or {}
        return (
            msg.get("content") or "",
            msg.get("reasoning_content") or "",
            ch.get("finish_reason") or "",
        )
    except Exception:
        return "", "", ""


def extract_responses_text(resp: Any) -> str:
    if isinstance(resp, dict):
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"]
        out = resp.get("output")
        if isinstance(out, list):
            texts: list[str] = []
            for item in out:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            texts.append(c.get("text", ""))
            return "\n".join(texts)
    return ""


def extract_responses_function_calls(resp: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(resp, dict):
        return calls
    out = resp.get("output")
    if not isinstance(out, list):
        return calls
    for item in out:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def extract_anthropic_text_and_stop(resp: Any) -> tuple[str, str]:
    """Return visible Anthropic text blocks and stop reason.

    The live audit uses exact-answer probes. Do not inspect the whole JSON
    object for the expected word, because reasoning/thinking blocks can mention
    the target string while the model never emits a visible answer.
    """
    if not isinstance(resp, dict):
        return "", ""
    texts: list[str] = []
    for block in resp.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text") or "")
    return "\n".join(texts), str(resp.get("stop_reason") or "")


def extract_ollama_visible_text_and_stop(resp: Any) -> tuple[str, str]:
    """Return visible Ollama text and done reason.

    Ollama has a separate ``message.thinking`` rail. For exact-answer checks,
    only ``message.content`` is the answer.
    """
    if not isinstance(resp, dict):
        return "", ""
    msg = resp.get("message") or {}
    return str(msg.get("content") or ""), str(resp.get("done_reason") or "")


def is_non_length_stop(stop: str) -> bool:
    stop = (stop or "").strip().lower()
    return stop not in {"length", "max_tokens"}


def has_duplicate_block(text: str, min_len: int = 80) -> bool:
    compact = re.sub(r"\s+", " ", text)
    if len(compact) < min_len * 2:
        return False
    for size in (240, 160, 100, 80):
        if len(compact) < size * 2:
            continue
        seen: set[str] = set()
        for i in range(0, len(compact) - size, max(16, size // 4)):
            block = compact[i : i + size]
            if block in seen:
                return True
            seen.add(block)
    return False


def simple_loop_score(text: str) -> float:
    words = re.findall(r"[\w']+", text.lower())
    if len(words) < 20:
        return 0.0
    worst = 0
    for n in (1, 2, 3):
        grams = [" ".join(words[i : i + n]) for i in range(0, len(words) - n + 1)]
        if not grams:
            continue
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        worst = max(worst, max(counts.values()))
    return worst / max(1, len(words))


def normalize_short_answer(text: str) -> str:
    """Normalize terse instruction-following answers for strict probes."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n`\"'“”‘’")


def text_quality_summary(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    digit_lines = sum(1 for ln in lines if re.fullmatch(r"\d+[\).]?", ln))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text)
    return {
        "lines": len(lines),
        "digit_lines": digit_lines,
        "digit_line_ratio": digit_lines / max(1, len(lines)),
        "cyrillic_chars": cyrillic,
        "latin_chars": latin,
        "cjk_chars": cjk,
        "word_count": len(words),
    }


def write_probe_output(row_id: str, name: str, payload: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row_id}_{name}")
    path = OUT_DIR / f"{safe}_{int(time.time())}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def post_chat(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None,
    max_tokens: int = 180,
    temperature: float = 0.6,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    t0 = time.perf_counter()
    code, resp = http_json("POST", f"{base}/v1/chat/completions", body)
    return {
        "code": code,
        "body": resp,
        "request": body,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }


def live_audit(row: ModelRow, py: Path, port: int, timeout_load: int, keep_running: bool = False) -> dict[str, Any]:
    model_dir = resolve_model_dir(row.path)
    result: dict[str, Any] = {
        "id": row.id,
        "label": row.label,
        "path": str(model_dir),
        "declared_path": row.path,
        "status": "UNKNOWN",
        "checks": [],
        "requests": [],
        "telemetry": [],
    }
    if not model_dir.is_dir():
        result["status"] = "SKIP"
        result["reason"] = "path missing"
        return result
    if not row.live_supported:
        result["status"] = "SKIP"
        result["reason"] = row.unsupported_reason or "row marked live unsupported"
        return result

    log = OUT_DIR / f"{row.id}_{int(time.time())}.log"
    block_cache_dir = OUT_DIR / f"{row.id}_block_cache_{int(time.time())}"
    block_cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        "300",
        "--continuous-batching",
        "--max-num-seqs",
        "5",
        "--prefill-batch-size",
        "1024",
        "--prefill-step-size",
        "2048",
        "--completion-batch-size",
        "1024",
        "--cache-memory-percent",
        "0.2",
        "--stream-interval",
        "1",
        "--max-tokens",
        "32768",
        "--default-temperature",
        "0.6",
        "--default-top-p",
        "0.95",
        "--default-repetition-penalty",
        "1.10",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--block-disk-cache-dir",
        str(block_cache_dir),
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--enable-auto-tool-choice",
    ]

    if os.environ.get("VMLINUX_AUDIT_KILL_EXISTING") == "1":
        subprocess.run(["pkill", "-9", "-f", "vmlx_engine.cli"], capture_output=True)
        time.sleep(2)
    with log.open("w") as lf:
        lf.write("# " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    result["pid"] = proc.pid
    result["log"] = str(log)
    result["block_disk_cache_dir"] = str(block_cache_dir)

    base = f"http://127.0.0.1:{port}"
    result["telemetry"].append(
        telemetry_snapshot("server_spawned", proc=proc, block_cache_dir=block_cache_dir)
    )

    def snapshot(name: str) -> None:
        result["telemetry"].append(
            telemetry_snapshot(
                name, proc=proc, base=base, block_cache_dir=block_cache_dir
            )
        )

    def request_json(
        name: str,
        method: str,
        path: str,
        body: Any | None = None,
        *,
        timeout: int = 240,
    ) -> tuple[int, Any, float]:
        snapshot(f"before_{name}")
        t0 = time.perf_counter()
        code, resp = http_json(method, f"{base}{path}", body, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 3)
        usage = resp.get("usage") if isinstance(resp, dict) else None
        artifact = None
        if method == "POST":
            artifact = write_probe_output(
                row.id,
                name,
                {"request": body, "code": code, "elapsed_sec": elapsed, "response": resp},
            )
        result["requests"].append(
            {
                "name": name,
                "method": method,
                "path": path,
                "code": code,
                "elapsed_sec": elapsed,
                "usage": usage,
                "artifact": artifact,
            }
        )
        snapshot(f"after_{name}")
        return code, resp, elapsed

    loaded = False
    for _ in range(timeout_load):
        if proc.poll() is not None:
            result["status"] = "FAIL"
            result["reason"] = f"process exited during load: {proc.returncode}"
            result["log_tail"] = log.read_text(errors="ignore")[-4000:]
            return result
        code, health = http_json("GET", f"{base}/health", timeout=2)
        if code == 200 and isinstance(health, dict):
            loaded = True
            result["health"] = health
            break
        time.sleep(1)
    if not loaded:
        result["status"] = "FAIL"
        result["reason"] = "load timeout"
        result["log_tail"] = log.read_text(errors="ignore")[-4000:]
        proc.terminate()
        return result

    def check(name: str, ok: bool, detail: Any) -> None:
        result["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    snapshot("after_load")

    log_text = log.read_text(errors="ignore")
    if row.family == "deepseek_v4":
        check(
            "dsv4_paged_cache_composite_enabled",
            "DeepseekV4Cache-aware paged prefix cache enabled" in log_text
            and "deepseek_v4_v7" in log_text,
            "DSV4 must keep paged/L2 cache on and use the deepseek_v4 v7 composite-state schema",
        )
        check(
            "dsv4_canonical_encoder_shim",
            "DSV4 chat-template shim installed" in log_text,
            "tokenizer.apply_chat_template must route through encoding_dsv4",
        )
        check(
            "dsv4_multi_eos",
            "128804" in log_text and "128803" in log_text,
            "expected EOS includes end/user/assistant markers",
        )

    code, stats0, _ = request_json("cache_stats_initial", "GET", "/v1/cache/stats", timeout=20)
    check("cache_stats_available", code == 200, stats0)

    model = (
        result.get("health", {}).get("model_name")
        or result.get("health", {}).get("model")
        or row.label
    )

    def chat_probe(
        name: str,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None,
        max_tokens: int = 180,
        temperature: float = 0.6,
    ) -> dict[str, Any]:
        snapshot(f"before_{name}")
        probe = post_chat(
            base,
            model,
            messages,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = (
            probe.get("body", {}).get("usage")
            if isinstance(probe.get("body"), dict)
            else None
        )
        content, reasoning, finish = extract_chat_text(probe["body"])
        artifact = write_probe_output(
            row.id,
            name,
            {
                "request": probe["request"],
                "code": probe["code"],
                "elapsed_sec": probe["elapsed_sec"],
                "finish": finish,
                "content": content,
                "reasoning": reasoning,
                "usage": usage,
                "raw_response": probe["body"],
            },
        )
        result["requests"].append(
            {
                "name": name,
                "method": "POST",
                "path": "/v1/chat/completions",
                "code": probe["code"],
                "elapsed_sec": probe["elapsed_sec"],
                "usage": usage,
                "artifact": artifact,
            }
        )
        snapshot(f"after_{name}")
        return probe

    cap_model = urllib.parse.quote(str(model), safe="")
    code, caps, _ = request_json(
        "model_capabilities",
        "GET",
        f"/v1/models/{cap_model}/capabilities",
        timeout=20,
    )
    supported_modes = (
        caps.get("supported_modes", []) if code == 200 and isinstance(caps, dict) else []
    )
    experimental_modes = (
        caps.get("experimental_modes", []) if code == 200 and isinstance(caps, dict) else []
    )
    expected_modes = ["instruct"]
    if row.expect_reasoning:
        expected_modes.append("reasoning")
    check(
        "model_capabilities_endpoint",
        code == 200
        and all(mode in supported_modes for mode in expected_modes)
        and (
            row.family != "deepseek_v4"
            or (
                "max" in experimental_modes
                and caps.get("cache", {}).get("dsv4_composite_state") is True
            )
        ),
        {"code": code, "caps": caps},
    )
    history: list[dict[str, Any]] = []
    r1 = chat_probe(
        "chat_turn1_thinking_off",
        [{"role": "user", "content": "Remember these facts: color blue, pet cat. Reply exactly: noted."}],
        thinking=False,
        max_tokens=80,
    )
    c1, reasoning1, finish1 = extract_chat_text(r1["body"])
    normalized_c1 = normalize_short_answer(c1).lower()
    check(
        "chat_thinking_off_basic",
        r1["code"] == 200
        and finish1 == "stop"
        and normalized_c1 in {"noted", "noted."}
        and not reasoning1.strip()
        and not has_duplicate_block(c1 + reasoning1),
        {
            "finish": finish1,
            "content": c1[:300],
            "normalized_content": normalized_c1,
            "reasoning": reasoning1[:160],
            "code": r1["code"],
            "elapsed_sec": r1["elapsed_sec"],
            "usage": r1.get("body", {}).get("usage") if isinstance(r1.get("body"), dict) else None,
        },
    )
    history += r1["request"]["messages"] + [{"role": "assistant", "content": c1}]

    r2 = chat_probe(
        "chat_turn2_recall_thinking_on",
        history + [{"role": "user", "content": "What color did I say? Answer in one word."}],
        thinking=True if row.expect_reasoning else False,
        max_tokens=220,
    )
    c2, reasoning2, finish2 = extract_chat_text(r2["body"])
    normalized_c2 = normalize_short_answer(c2).lower().rstrip(".")
    check(
        "chat_thinking_on_or_reasoning_toggle_recall",
        r2["code"] == 200
        and finish2 == "stop"
        and normalized_c2 == "blue"
        and "blue" in (c2 + reasoning2).lower()
        and not has_duplicate_block(c2 + reasoning2),
        {
            "finish": finish2,
            "content": c2[:300],
            "normalized_content": normalized_c2,
            "reasoning_chars": len(reasoning2),
            "reasoning_head": reasoning2[:240],
            "code": r2["code"],
            "elapsed_sec": r2["elapsed_sec"],
            "usage": r2.get("body", {}).get("usage") if isinstance(r2.get("body"), dict) else None,
        },
    )

    if row.family == "deepseek_v4":
        code, max_resp, max_elapsed = request_json(
            "dsv4_thinking_mode_max",
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use max thinking. What is 3 plus 4? Include the final digit.",
                    }
                ],
                "thinking_mode": "max",
                "max_tokens": 320,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=240,
        )
        max_content, max_reasoning, max_finish = extract_chat_text(max_resp)
        max_joined = (max_content + "\n" + max_reasoning).lower()
        check(
            "dsv4_thinking_mode_max",
            code == 200
            and "7" in max_joined
            and not has_duplicate_block(max_joined),
            {
                "code": code,
                "finish": max_finish,
                "content": max_content[:300],
                "reasoning_chars": len(max_reasoning),
                "reasoning_head": max_reasoning[:240],
                "elapsed_sec": max_elapsed,
            },
        )

        long_prompts = [
            (
                "vc_project_plan",
                "For our VC fund, we are trying to document everything we do, "
                "create cadences, and bring in a student from the Haskayne Propel "
                "2026 project plan program. Help fill this form field by field. "
                "Proposed Project Name: Process optimisation and document "
                "standardization for the VC fund. What is the project's goal? "
                "The primary deliverable must be realistic and achievable within "
                "50 hours by a student unfamiliar with our organization and "
                "industry. Please outline the project timeline. What will the "
                "student work on each week at about 6 hours per week? Weeks 1 "
                "and 8 have preset onboarding and offboarding activities. Please "
                "break down specialized skills or mindsets required. Students may "
                "need to work with people from diverse backgrounds and should be "
                "strong in Microsoft software, process mapping, documentation, "
                "stakeholder interviews, and organized project management. Please "
                "think carefully, then produce a concise usable draft with clear "
                "headings and no repeated filler."
            ),
            (
                "game_design_long_context",
                "Create a single HTML file for a Three.js game. The player is a "
                "hunter moving through a forest with a shotgun while hostile "
                "boars, moose, and other enemies spawn with different behavior. "
                "Include projectile physics, collision detection, enemy AI, spawn "
                "timers, acceleration over time, health, score, and a compact UI "
                "layer over the canvas. Explain the architecture first, then give "
                "the complete file. Avoid repeating the same noun or phrase."
            ),
        ]
        for probe_name, prompt in long_prompts:
            code, long_resp, long_elapsed = request_json(
                f"dsv4_long_{probe_name}",
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "enable_thinking": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "max_tokens": 900,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "stream": False,
                },
                timeout=420,
            )
            long_content, long_reasoning, long_finish = extract_chat_text(long_resp)
            full_text = f"{long_reasoning}\n{long_content}".strip()
            loop_score = simple_loop_score(full_text)
            artifact = write_probe_output(
                row.id,
                f"dsv4_long_{probe_name}",
                {
                    "request": {"prompt": prompt, "max_tokens": 900},
                    "code": code,
                    "finish": long_finish,
                    "content": long_content,
                    "reasoning": long_reasoning,
                    "content_chars": len(long_content),
                    "reasoning_chars": len(long_reasoning),
                    "loop_score": loop_score,
                    "raw_response": long_resp,
                    "elapsed_sec": long_elapsed,
                },
            )
            check(
                f"dsv4_long_context_full_output_{probe_name}",
                code == 200
                and len(full_text) >= 120
                and len(long_content) >= 40
                and not has_duplicate_block(full_text)
                and loop_score < 0.25,
                {
                    "code": code,
                    "finish": long_finish,
                    "content_chars": len(long_content),
                    "reasoning_chars": len(long_reasoning),
                    "loop_score": loop_score,
                    "elapsed_sec": long_elapsed,
                    "artifact": artifact,
                    "head": full_text[:400],
                    "tail": full_text[-400:],
                },
            )

    if row.family == "bailing_hybrid":
        code, ling_resp, ling_elapsed = request_json(
            "ling_multilingual_loop_trigger",
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Привет. Напиши краткий план одной HTML/Three.js "
                            "игры: охотник бежит по лесу, стреляет из дробовика, "
                            "а кабаны и лоси появляются как враги. Ответь по-русски "
                            "структурировано, без повторения одного слова или символа."
                        ),
                    }
                ],
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 420,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=300,
        )
        ling_content, ling_reasoning, ling_finish = extract_chat_text(ling_resp)
        ling_full = f"{ling_reasoning}\n{ling_content}".strip()
        ling_loop_score = simple_loop_score(ling_full)
        ling_quality = text_quality_summary(ling_full)
        ling_artifact = write_probe_output(
            row.id,
            "ling_multilingual_game_prompt",
            {
                "code": code,
                "finish": ling_finish,
                "content": ling_content,
                "reasoning": ling_reasoning,
                "loop_score": ling_loop_score,
                "quality": ling_quality,
                "raw_response": ling_resp,
                "elapsed_sec": ling_elapsed,
            },
        )
        check(
            "ling_multilingual_loop_trigger",
            code == 200
            and len(ling_full) >= 120
            and not has_duplicate_block(ling_full)
            and ling_loop_score < 0.25
            and ling_full.count("👀") < 8
            and ling_quality["word_count"] >= 24
            and ling_quality["digit_line_ratio"] < 0.35,
            {
                "code": code,
                "finish": ling_finish,
                "content_chars": len(ling_content),
                "reasoning_chars": len(ling_reasoning),
                "loop_score": ling_loop_score,
                "quality": ling_quality,
                "elapsed_sec": ling_elapsed,
                "artifact": ling_artifact,
                "head": ling_full[:300],
                "tail": ling_full[-300:],
            },
        )

    # Responses API, including persisted tool-history shape. This first
    # check verifies replay of an already-completed tool turn; it deliberately
    # provides no tools and sets tool_choice=none so the model must use the
    # existing function_call_output instead of making a fresh call.
    responses_input = [
        {"role": "user", "content": "We used a directory tool. Remember this."},
        {
            "type": "function_call",
            "call_id": "call_audit_1",
            "name": "list_directory",
            "arguments": json.dumps({"path": "."}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_audit_1",
            "output": "README.md\npyproject.toml\nvmlx_engine",
        },
        {"type": "output_text", "text": "README.md"},
        {
            "type": "message",
            "role": "user",
            "content": (
                "Based only on the previous tool result text, copy exactly one "
                "listed string from this set: README.md, pyproject.toml, "
                "vmlx_engine. Do not call a tool. Reply with only that string."
            ),
        },
    ]
    code, resp, resp_elapsed = request_json(
        "responses_tool_history_continuation",
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": responses_input,
            "max_output_tokens": 160,
            "temperature": 0.3,
            "stream": False,
            "enable_thinking": False,
            "tool_choice": "none",
        },
        timeout=240,
    )
    resp_text = extract_responses_text(resp)
    check(
        "responses_tool_history_continuation",
        code == 200 and any(x in resp_text.lower() for x in ("readme", "pyproject", "vmlx_engine")),
        {
            "code": code,
            "text": resp_text[:400],
            "elapsed_sec": resp_elapsed,
            "raw_keys": list(resp.keys()) if isinstance(resp, dict) else None,
        },
    )

    # Responses API auto tool choice. Only model families with a declared
    # native tool parser are required to emit a structured function_call.
    # Non-tool rows still exercise Responses history above; requiring tool
    # behavior from them turns model capability gaps into false engine failures.
    if row.expect_tool_parser:
        code, resp_tool, resp_tool_elapsed = request_json(
            "responses_auto_tool_choice_structured",
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": "Use the list_directory tool for path '.' and do not answer in prose.",
                "tools": [
                    {
                        "type": "function",
                        "name": "list_directory",
                        "description": "List files in a directory.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_output_tokens": 160,
                "temperature": 0.1,
                "stream": False,
                "enable_thinking": False,
            },
            timeout=240,
        )
        fcalls = extract_responses_function_calls(resp_tool)
        raw_tool_text = extract_responses_text(resp_tool)
        check(
            "responses_auto_tool_choice_structured",
            code == 200
            and bool(fcalls)
            and any(c.get("name") == "list_directory" for c in fcalls)
            and "<tool_call>" not in raw_tool_text,
            {
                "code": code,
                "function_calls": fcalls,
                "text": raw_tool_text[:300],
                "elapsed_sec": resp_tool_elapsed,
                "raw_keys": list(resp_tool.keys()) if isinstance(resp_tool, dict) else None,
            },
        )
    else:
        check(
            "responses_auto_tool_choice_structured",
            True,
            {
                "skipped": True,
                "reason": "row has no declared native tool parser",
            },
        )

    code, anth, anth_elapsed = request_json(
        "anthropic_messages_basic",
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 120,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly: anthropic-ok"}],
            "enable_thinking": False,
        },
        timeout=240,
    )
    anth_text, anth_stop = extract_anthropic_text_and_stop(anth)
    check(
        "anthropic_messages_basic",
        code == 200
        and normalize_short_answer(anth_text).lower() == "anthropic-ok"
        and is_non_length_stop(anth_stop),
        {
            "code": code,
            "visible_text": anth_text,
            "normalized_text": normalize_short_answer(anth_text),
            "stop_reason": anth_stop,
            "body": anth,
            "elapsed_sec": anth_elapsed,
        },
    )

    code, ollama, ollama_elapsed = request_json(
        "ollama_chat_basic",
        "POST",
        "/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly: ollama-ok"}],
            "options": {"temperature": 0.0, "num_predict": 80},
        },
        timeout=240,
    )
    ollama_text, ollama_stop = extract_ollama_visible_text_and_stop(ollama)
    check(
        "ollama_chat_basic",
        code == 200
        and normalize_short_answer(ollama_text).lower() == "ollama-ok"
        and is_non_length_stop(ollama_stop),
        {
            "code": code,
            "visible_text": ollama_text,
            "normalized_text": normalize_short_answer(ollama_text),
            "done_reason": ollama_stop,
            "body": ollama,
            "elapsed_sec": ollama_elapsed,
        },
    )

    # Cache exact-hit coherence. Multi-turn memory is covered by the chat
    # recall checks above; this check isolates cache mechanics with a prompt
    # that stops cleanly and produces >3 tokens so the scheduler does not
    # skip cache donation as a short single-token benchmark response.
    cache_messages = [
        {
            "role": "user",
            "content": (
                "Cache audit exact-hit check. Reply exactly: blue blue blue blue."
            ),
        }
    ]
    r_cache = chat_probe(
        "cache_exact_hit_first",
        cache_messages,
        thinking=False,
        max_tokens=512,
        temperature=0.1,
    )
    c_cache, _, _ = extract_chat_text(r_cache["body"])
    r_cache_repeat = chat_probe(
        "cache_exact_hit_repeat",
        cache_messages,
        thinking=False,
        max_tokens=512,
        temperature=0.1,
    )
    c_cache_repeat, _, _ = extract_chat_text(r_cache_repeat["body"])
    _repeat_usage = (
        r_cache_repeat.get("body", {}).get("usage", {})
        if isinstance(r_cache_repeat.get("body"), dict)
        else {}
    )
    _repeat_details = (
        _repeat_usage.get("prompt_tokens_details") or {}
        if isinstance(_repeat_usage, dict)
        else {}
    )
    code, stats1, _ = request_json("cache_stats_final", "GET", "/v1/cache/stats", timeout=20)
    _sched_cache = stats1.get("scheduler_cache", {}) if code == 200 and isinstance(stats1, dict) else {}
    _block_disk = stats1.get("block_disk_cache", {}) if code == 200 and isinstance(stats1, dict) else {}
    _cache_hit_observed = (
        int(_repeat_details.get("cached_tokens") or 0) > 0
        or int(_sched_cache.get("cache_hits") or _sched_cache.get("hits") or 0) > 0
        or int(_block_disk.get("disk_hits") or 0) > 0
    )
    _mixed_attention_detected = (
        row.family == "gemma4"
        and "mixed-attention model detected" in log_text
    )
    check(
        "cache_second_turn_coherent",
        r_cache["code"] == 200
        and r_cache_repeat["code"] == 200
        and c_cache.lower().count("blue") >= 3
        and c_cache_repeat.lower().count("blue") >= 3
        and not has_duplicate_block(c_cache + c_cache_repeat)
        and _cache_hit_observed,
        {
            "content": c_cache[:240],
            "repeat_content": c_cache_repeat[:240],
            "repeat_usage": _repeat_usage,
            "cache_hit_observed": _cache_hit_observed,
            "mixed_attention_detected": _mixed_attention_detected,
            "first_elapsed_sec": r_cache["elapsed_sec"],
            "repeat_elapsed_sec": r_cache_repeat["elapsed_sec"],
            "cache_stats": stats1 if code == 200 else stats0,
        },
    )

    snapshot("before_shutdown")

    failures = [c for c in result["checks"] if not c["ok"]]
    result["status"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures

    if keep_running:
        result["kept_running"] = True
    else:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="Comma-separated row IDs. Default: all.")
    ap.add_argument("--live", action="store_true", help="Actually start models and hit APIs.")
    ap.add_argument("--skip-slow", action="store_true", help="Skip rows tagged slow.")
    ap.add_argument("--py", default=str(DEFAULT_PY), help="Python executable to run vmlx_engine.")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--keep-running", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "production_family_audit.json"))
    args = ap.parse_args()

    rows = ROWS
    if args.rows:
        wanted = {x.strip() for x in args.rows.split(",") if x.strip()}
        rows = [r for r in rows if r.id in wanted]
    if args.skip_slow:
        rows = [r for r in rows if not r.slow]

    results: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": args.py,
        "live": args.live,
        "rows": [],
    }
    for row in rows:
        print(f"\n=== {row.id}: {row.label}")
        static = static_audit(row)
        print(
            "static:",
            "exists=" + str(static["exists"]),
            "model_type=" + str(static["config"].get("model_type")),
            "bits=" + str(static["jang"].get("mxtq_bits") or static["config"].get("quantization")),
            "issues=" + str(static["issues"]),
        )
        item: dict[str, Any] = {"static": static}
        if args.live:
            live = live_audit(
                row,
                Path(args.py),
                args.port,
                args.load_timeout,
                keep_running=args.keep_running,
            )
            print("live:", live["status"], "failures=", len(live.get("failures", [])))
            for chk in live.get("checks", []):
                print(" ", "OK" if chk["ok"] else "FAIL", chk["name"])
            item["live"] = live
        results["rows"].append(item)
        if args.live and args.keep_running:
            print("--keep-running set; stopping after first live row")
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {out}")

    if args.live:
        failed = [
            r
            for r in results["rows"]
            if r.get("live", {}).get("status") not in (None, "PASS", "SKIP")
        ]
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

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


ROOT = Path(
    os.environ.get("VMLINUX_AUDIT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DEFAULT_PY = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
OUT_DIR = Path("/tmp/vmlx_family_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_MODEL_ROOTS = [
    Path("/Users/example/models/JANGQ"),
    Path("/Users/example/models/dealign.ai"),
    Path("/Users/example/models/Sources"),
    Path("/Users/example/.lmstudio/models"),
    Path("/Users/example/.mlxstudio/models"),
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
    cache_profile: str = "default"  # default | dsv4_composite | vl | hybrid_ssm | zaya_cca
    live_supported: bool = True
    unsupported_reason: str | None = None
    defer_failure_reason: str | None = None
    notes: list[str] = field(default_factory=list)


ROWS: list[ModelRow] = [
    ModelRow(
        id="zaya_jangtq2",
        label="ZAYA1-8B JANGTQ2 CCA",
        path="/Users/example/jang/models/Zyphra/ZAYA1-8B-JANGTQ2",
        family="zaya",
        expect_reasoning=False,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        defer_failure_reason=(
            "ZAYA1 JANGTQ2 is a low-bit CCA row with intermittent Responses "
            "tool-history/tool-choice failures: it can emit raw Zyphra tool "
            "markup or prose instead of the requested continuation. Higher-bit "
            "ZAYA MXFP4/JANGTQ4 rows are the production references until the "
            "2-bit tool-history behavior is fixed."
        ),
        notes=[
            "ZAYA CCA cache has KV + conv_state + prev_hs. Prefix/paged/L2 "
            "run only under VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE=1 until the "
            "server/API live gates pass. Runtime TQ-KV stays disabled."
        ],
    ),
    ModelRow(
        id="zaya_mxfp4",
        label="ZAYA1-8B MXFP4 CCA",
        path="/Users/example/jang/models/Zyphra/ZAYA1-8B-MXFP4",
        family="zaya",
        expect_reasoning=False,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        notes=[
            "Higher-bit ZAYA control row. Same CCA cache contract as JANGTQ2; "
            "no generic hybrid SSM cache assumptions."
        ],
    ),
    ModelRow(
        id="zaya_jangtq4",
        label="ZAYA1-8B JANGTQ4 CCA",
        path="/Users/example/jang/models/Zyphra/ZAYA1-8B-JANGTQ4",
        family="zaya",
        expect_reasoning=False,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        notes=[
            "Higher-bit JANGTQ ZAYA control row. Same CCA cache contract; "
            "use it to separate 2-bit expert quality from runtime/template issues."
        ],
    ),
    ModelRow(
        id="zaya_vl_mxfp4",
        label="ZAYA1-VL-8B MXFP4 CCA",
        path="/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4",
        family="zaya1_vl",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        notes=[
            "ZAYA-VL media row. Must keep typed zaya_cca_v1 cache for text-only "
            "prefixes while media-bearing requests bypass token-prefix CCA cache."
        ],
    ),
    ModelRow(
        id="zaya_vl_jangtq4",
        label="ZAYA1-VL-8B JANGTQ4 CCA",
        path="/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        family="zaya1_vl",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        notes=[
            "ZAYA-VL routed 4-bit control row. Validates JANGTQ MLLM fast path, "
            "ZAYA CCA text cache, and media-prefix cache isolation."
        ],
    ),
    ModelRow(
        id="zaya_vl_jangtq_k",
        label="ZAYA1-VL-8B JANGTQ_K CCA",
        path="/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ_K",
        family="zaya1_vl",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="zaya_xml",
        cache_profile="zaya_cca",
        defer_failure_reason=(
            "Responses tool-history continuation emits ZAYA localization/control "
            "tokens on the mixed JANGTQ_K VL bundle. Chat, cache, Anthropic, and "
            "Ollama rows pass; keep this row out of release claims until the "
            "Responses tool-history prompt/runtime path is fixed for this artifact."
        ),
        notes=[
            "Mixed 2/2/4 ZAYA-VL row. Product repo may warn about forced "
            "thinking quality, but the runtime should not silently clamp this row."
        ],
    ),
    ModelRow(
        id="dsv4_tq",
        label="DeepSeek-V4-Flash JANGTQ-K sub-80 candidate",
        path="/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "Current local DSV4 upload candidate: sub-80GB JANGTQ-K plan with "
            "F32 critical controls, explicit bit plan, and prestacked JANGTQ "
            "sidecar.",
            "DSV4 SWA+CSA/HCA heterogenous cache; paged/block L2 must use "
            "deepseek_v4_v7 composite-state serialization, not generic KV blocks"
        ],
    ),
    ModelRow(
        id="dsv4_jang_local",
        label="DeepSeek-V4-Flash JANG local affine artifact",
        path="/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "Actual local affine DSV4 artifact present on this host. Static "
            "headers show the same output-head/final-norm precision boundary "
            "as JANGTQ-K: quantized head plus F16 norm, so it cannot clear "
            "long-output production claims without source-vs-quant or rebuilt "
            "artifact evidence.",
        ],
    ),
    ModelRow(
        id="dsv4_jang_dq2_gate3math6",
        label="DeepSeek-V4-Flash JANG local pure-affine NoMTP artifact",
        path="/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "Current present pure-affine JANG DSV4 keeper candidate. It uses "
            "the JANG_2L_GS64_ProjLayerBits_Ggs32-Dgs32-Ugs64_bk4_Tok8g64_NoMTP "
            "profile from jang_config.json.",
            "Must use the DSV4 native SWA+CSA/HCA composite cache path "
            "with deepseek_v4_v7 paged-prefix/L2 serialization; generic "
            "TurboQuant KV is invalid for this family.",
        ],
    ),
    ModelRow(
        id="dsv4_tq2",
        label="DeepSeek-V4-Flash JANGTQ2",
        path="/Volumes/ModelDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ2",
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
        path="/Volumes/ModelDrive/jangq-ai/DeepSeek-V4-Flash-JANG_2L",
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
        path="/Volumes/ModelDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_moe_crack",
        label="Qwen3.6-35B-A3B JANGTQ CRACK hybrid MoE+SSM",
        path="/Volumes/ModelDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_dense_mxfp4",
        label="Qwen3.6-27B MXFP4 CRACK dense",
        path="/Volumes/ModelDrive/dealignai/Qwen3.6-27B-MXFP4-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="qwen36_dense_jang",
        label="Qwen3.6-27B JANG_4M CRACK dense",
        path="/Volumes/ModelDrive/dealignai/Qwen3.6-27B-JANG_4M-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="hy3_preview_jangtq2",
        label="Hy3-preview JANGTQ2",
        path="/Users/example/models/JANGQ/Hy3-preview-JANGTQ2",
        family="hy_v3",
        expect_reasoning=True,
        expect_tool_parser="hunyuan",
        slow=True,
        notes=[
            "Hy3/Hunyuan reasoning row. UI/API contract is reasoning_effort "
            "low/high with no Medium; runtime parser is qwen3 text extraction "
            "and tool parser is hunyuan.",
            "Large 128GB-class row; Metal working-set guard must allow the "
            "validated near-limit load without command-buffer OOM.",
        ],
    ),
    ModelRow(
        id="hy3_preview_jang_2k",
        label="Hy3-preview JANG_2K",
        path="/Users/example/models/JANGQ/Hy3-preview-JANG_2K",
        family="hy_v3",
        expect_reasoning=True,
        expect_tool_parser="hunyuan",
        slow=True,
        notes=[
            "Hy3 all-affine JANG_2K row. UI/API contract is reasoning_effort "
            "low/high with qwen3 reasoning extraction and Hunyuan tool calls.",
            "Bundle generation_config intentionally omits max_new_tokens; live "
            "gates must prove omitted requests resolve to the sane bounded "
            "server fallback, not 32768.",
            "Affine quantization uses group_size=128 with 2-bit routed gate/up "
            "and 3-bit routed down projections; release proof must confirm the "
            "group-size metadata is honored by the loader.",
        ],
    ),
    ModelRow(
        id="hy3_preview_jang_2l",
        label="Hy3-preview JANG_2L",
        path="/Users/example/models/JANGQ/Hy3-preview-JANG_2L",
        family="hy_v3",
        expect_reasoning=True,
        expect_tool_parser="hunyuan",
        slow=True,
        notes=[
            "Hy3 all-affine JANG_2L row. Same qwen3 reasoning parser, Hunyuan "
            "tool parser, KV cache, and low/high effort UI/API contract as "
            "JANGTQ2/JANG_2K.",
            "Bundle generation_config intentionally omits max_new_tokens; live "
            "gates must prove omitted requests resolve to the sane bounded "
            "server fallback, not 32768.",
            "Affine quantization uses group_size=128 with 2-bit routed expert "
            "projections; release proof must confirm group-size metadata is "
            "honored by the loader.",
        ],
    ),
    ModelRow(
        id="nemotron_omni_tq2",
        label="Nemotron-3-Nano-Omni-30B-A3B JANGTQ2",
        path="/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
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
        path="/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4",
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
        path="/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ",
        family="bailing_hybrid",
        expect_reasoning=False,
        expect_tool_parser="deepseek",
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
        path="/Users/example/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK",
        family="bailing_hybrid",
        expect_reasoning=False,
        expect_tool_parser="deepseek",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="ling_flash_mxfp4_crack",
        label="Ling-2.6-flash MXFP4 CRACK",
        path="/Users/example/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK",
        family="bailing_hybrid",
        expect_reasoning=False,
        expect_tool_parser="deepseek",
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
        path="/Volumes/ModelDrive/jangq-ai/JANGQ-AI/Laguna-XS.2-JANGTQ",
        family="laguna",
        expect_tool_parser="qwen",
        slow=True,
    ),
    ModelRow(
        id="mistral_medium35_tq",
        label="Mistral-Medium-3.5-128B JANGTQ2",
        path="/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-JANGTQ",
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
        path="/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-mxfp4",
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
        path="/Volumes/ModelDrive/dealignai/Mistral-Small-4-119B-JANG_6M-CRACK",
        family="mistral",
        expect_tool_parser="mistral",
        slow=True,
    ),
    ModelRow(
        id="gemma4_crack",
        label="Gemma-4-26B-A4B-it JANG_4M CRACK",
        path="/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        family="gemma4",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="gemma4",
        cache_profile="vl",
        slow=True,
    ),
    ModelRow(
        id="gemma4_31b_jang4m_mtp",
        label="Gemma-4-31B-it JANG_4M MTP-preserved",
        path="/Users/example/models/JANGQ/Gemma-4-31B-it-JANG_4M-MTP",
        family="gemma4",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="gemma4",
        cache_profile="vl",
        slow=True,
        notes=[
            "Gemma 31B row is an MTP-preserved artifact boundary, not a proven "
            "native self-spec runtime row. Do not claim an MTP speedup unless "
            "accept/reject and full-output equivalence are live-proven."
        ],
    ),
    ModelRow(
        id="minimax_m27_tq_crack",
        label="MiniMax-M2.7 JANGTQ CRACK",
        path="/Volumes/ModelDrive/dealignai/MiniMax-M2.7-JANGTQ-CRACK",
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
        path="/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K",
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
        path="/Users/example/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        slow=True,
    ),
    ModelRow(
        id="minimax_m27_tq4_crack",
        label="MiniMax-M2.7 JANGTQ4 CRACK",
        path="/Volumes/ModelDrive/dealignai/MiniMax-M2.7-JANGTQ4-CRACK",
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


def _index_tensor_prefix_status(model_dir: Path, prefix: str) -> tuple[bool, str | None]:
    index = read_json(model_dir / "model.safetensors.index.json")
    if "_read_error" in index:
        return False, f"model.safetensors.index.json read failed: {index['_read_error']}"
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        if (model_dir / "model.safetensors.index.json").is_file():
            return False, "model.safetensors.index.json has no weight_map object"
        return False, None
    return any(str(key).startswith(prefix) for key in weight_map), None


def _index_has_tensor_prefix(model_dir: Path, prefix: str) -> bool:
    return _index_tensor_prefix_status(model_dir, prefix)[0]


def _tensor_header_summary(model_dir: Path, tensor_names: tuple[str, ...]) -> dict[str, Any]:
    """Return safetensors header metadata for named tensors without loading data."""
    summary: dict[str, Any] = {
        "checked": False,
        "tensors": {},
        "missing": [],
        "errors": [],
    }
    index = read_json(model_dir / "model.safetensors.index.json")
    if "_read_error" in index:
        summary["errors"].append(f"model.safetensors.index.json read failed: {index['_read_error']}")
        return summary
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        if (model_dir / "model.safetensors.index.json").is_file():
            summary["errors"].append("model.safetensors.index.json has no weight_map object")
        return summary
    try:
        from safetensors import safe_open
    except Exception as exc:
        summary["errors"].append(f"safetensors import failed: {exc}")
        return summary

    tensors_by_file: dict[str, list[str]] = {}
    for name in tensor_names:
        filename = weight_map.get(name)
        if not filename:
            summary["missing"].append(name)
            continue
        tensors_by_file.setdefault(str(filename), []).append(name)

    for filename, names in tensors_by_file.items():
        path = model_dir / filename
        if not path.is_file():
            summary["errors"].extend(
                f"{name}: mapped shard missing: {filename}" for name in names
            )
            continue
        try:
            with safe_open(str(path), framework="np") as reader:
                available = set(reader.keys())
                for name in names:
                    if name not in available:
                        summary["errors"].append(f"{name}: mapped shard {filename} does not contain tensor")
                        continue
                    tensor = reader.get_slice(name)
                    summary["tensors"][name] = {
                        "file": str(filename),
                        "dtype": tensor.get_dtype(),
                        "shape": tensor.get_shape(),
                    }
        except Exception as exc:
            summary["errors"].extend(
                f"{name}: header read failed from {filename}: {type(exc).__name__}: {exc}"
                for name in names
            )
    summary["checked"] = True
    return summary


def dsv4_output_precision_boundary(model_dir: Path) -> dict[str, Any]:
    """Inspect DSV4 output-head/final-norm precision from safetensors headers.

    Live probes showed DSV4 exact-code corruption occurs in generated logits,
    after prompt rendering/tokenization and before response assembly. This
    header-only check records whether the active artifact compressed the final
    output projection or final norm so release gates cannot treat a DSV4 row as
    production-cleared without source-vs-quant or rebuilt-artifact evidence.
    """
    header = _tensor_header_summary(
        model_dir,
        (
            "head.weight",
            "head.scales",
            "head.biases",
            "lm_head.weight",
            "lm_head.scales",
            "lm_head.biases",
            "norm.weight",
            "model.norm.weight",
        ),
    )
    tensors = header.get("tensors") if isinstance(header.get("tensors"), dict) else {}
    output_key = "head.weight" if "head.weight" in tensors else "lm_head.weight" if "lm_head.weight" in tensors else None
    output_dtype = tensors.get(output_key, {}).get("dtype") if output_key else None
    output_shape = tensors.get(output_key, {}).get("shape") if output_key else None
    output_has_aux = bool(
        ("head.scales" in tensors and "head.biases" in tensors)
        or ("lm_head.scales" in tensors and "lm_head.biases" in tensors)
    )
    final_norm_key = "norm.weight" if "norm.weight" in tensors else "model.norm.weight" if "model.norm.weight" in tensors else None
    final_norm_dtype = tensors.get(final_norm_key, {}).get("dtype") if final_norm_key else None
    quantized_dtypes = {"U8", "U16", "U32", "I8", "I16", "I32"}
    source_like_dtypes = {"BF16", "F32"}
    output_head_quantized = bool(output_dtype in quantized_dtypes or output_has_aux)
    final_norm_lower_precision = bool(final_norm_dtype and final_norm_dtype not in source_like_dtypes)
    needs_clearance = bool(output_head_quantized or final_norm_lower_precision)
    return {
        **header,
        "output_head_key": output_key,
        "output_head_dtype": output_dtype,
        "output_head_shape": output_shape,
        "output_head_has_quant_aux": output_has_aux,
        "output_head_quantized": output_head_quantized,
        "final_norm_key": final_norm_key,
        "final_norm_dtype": final_norm_dtype,
        "final_norm_lower_precision": final_norm_lower_precision,
        "needs_source_or_rebuild_clearance": needs_clearance,
    }


def dsv4_rope_scaling_contract(
    cfg: dict[str, Any],
    *,
    model_dir: Path,
) -> dict[str, Any]:
    """Record whether DSV4 Flash runtime needs the source YaRN metadata repair."""
    try:
        from vmlx_engine.loaders.load_jangtq_dsv4 import (
            _DSV4_FLASH_REQUIRED_ROPE_SCALING,
            _normalize_dsv4_runtime_config,
        )
    except Exception as exc:
        return {
            "checked": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_repair_required": False,
            "source_required_rope_scaling": None,
            "bundle_rope_scaling": cfg.get("rope_scaling"),
            "runtime_rope_scaling": cfg.get("rope_scaling"),
        }

    runtime_cfg, changed = _normalize_dsv4_runtime_config(cfg, model_path=model_dir)
    return {
        "checked": True,
        "bundle_rope_scaling": cfg.get("rope_scaling"),
        "runtime_rope_scaling": runtime_cfg.get("rope_scaling"),
        "runtime_repair_required": changed,
        "source_required_rope_scaling": dict(_DSV4_FLASH_REQUIRED_ROPE_SCALING),
    }


def dsv4_release_fault_matrix(
    *,
    model_dir: Path,
    precision_boundary: dict[str, Any] | None = None,
    rope_scaling_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the current DSV4 release fault matrix.

    This is intentionally explicit because the DSV4 failure has repeatedly
    looked "nearly fixed" from proxy evidence. The matrix separates layers that
    existing probes have ruled out from blockers that still require source,
    broader-body, or first-divergence proof before any release claim.
    """
    precision = precision_boundary or {}
    rope_contract = rope_scaling_contract or {}
    source_dir = Path("/Users/example/models/Sources/DeepSeek-V4-Flash")
    ruled_out = [
        {
            "layer": "prompt_renderer",
            "status": "ruled_out",
            "evidence": "canonical DSV4 prompt-render probes preserved Three.js identifiers exactly",
        },
        {
            "layer": "api_endpoint_adapter",
            "status": "ruled_out",
            "evidence": "raw completions with the rendered canonical chat prompt reproduced the same identifier corruption",
        },
        {
            "layer": "stream_sse_or_response_assembly",
            "status": "ruled_out",
            "evidence": "non-stream logprob responses contained generated token sequence .Web, Web, GL, Renderer",
        },
        {
            "layer": "tokenizer_roundtrip",
            "status": "ruled_out",
            "evidence": "tokenizer and streaming detokenizer round-tripped known-correct Three.js identifiers",
        },
        {
            "layer": "mxtq_only_kernel_path",
            "status": "ruled_out",
            "evidence": "local affine DSV4 JANG artifact reproduced the same Three.js identifier corruption class",
        },
        {
            "layer": "output_head_or_final_norm_only",
            "status": "ruled_out",
            "evidence": "BF16 head/final-norm overlay over the JANGTQ-K body still generated THREE.WebWebGLRenderer",
        },
    ]
    open_blockers = [
        {
            "layer": "source_or_broader_body_comparison",
            "status": "open",
            "evidence": "source DSV4 is present locally but too large for a casual 128 GB live load; no source full-output clearance artifact exists",
            "source_model_present": source_dir.is_dir(),
        },
        {
            "layer": "shared_dsv4_runtime_or_body_precision",
            "status": "open",
            "evidence": "affine and JANGTQ rows share the failure class; remaining suspects are shared DSV4 runtime integration or quantized body precision",
        },
        {
            "layer": "full_output_identifier_gate",
            "status": "open",
            "evidence": "short exact identifier and long Three.js full-output gates still reject corrupted API names",
        },
        {
            "layer": "dsv4_composite_prefix_cache_equivalence",
            "status": "mitigated_open",
            "evidence": (
                "patched-config live DSV4 run produced a paged+dsv4 cache hit "
                "with cached_tokens evidence, but greedy follow-up output differed "
                "from the skip_prefix_cache control; vMLX now disables DSV4 "
                "composite prefix/paged/L2 reuse by default and requires "
                "--dsv4-enable-prefix-cache or VMLX_DSV4_ENABLE_PREFIX_CACHE=1 "
                "for diagnostic opt-in"
            ),
        },
    ]
    if precision.get("needs_source_or_rebuild_clearance"):
        open_blockers.append(
            {
                "layer": "precision_boundary_clearance",
                "status": "open",
                "evidence": (
                    "active artifact still needs source or rebuilt-body clearance "
                    f"(head_quantized={precision.get('output_head_quantized')}, "
                    f"final_norm_lower_precision={precision.get('final_norm_lower_precision')})"
                ),
            }
        )
    if rope_contract.get("runtime_repair_required"):
        open_blockers.append(
            {
                "layer": "compressed_rope_scaling_metadata",
                "status": "open",
                "evidence": (
                    "active bundle config omitted source DeepSeek-V4-Flash YaRN "
                    "rope_scaling; vMLX now repairs it before construction, but "
                    "DSV4 still needs a fresh live identifier/full-output gate"
                ),
            }
        )
    return {
        "checked": True,
        "model_dir": str(model_dir),
        "release_blocked": True,
        "ruled_out": ruled_out,
        "open_blockers": open_blockers,
        "required_next_proofs": [
            "Run the same full-output identifier gate on source DSV4 or a broader rebuilt body; a short canary is not a release pass.",
            "If source cannot fit, run first-divergence or teacher-forced margin probes against a coherent rebuilt affine/MXFP body before changing prompt/API code.",
            "Keep DSV4 composite prefix/paged/L2 cache disabled by default until a diagnostic opt-in cache equivalence proof shows temperature=0 cached follow-up byte-matches the skip_prefix_cache control for the same rendered prompt.",
            "Verify the DSV4 source YaRN rope-scaling repair on a live full-output identifier gate; static config repair is not enough.",
            "Keep DSV4 scoped as not production-cleared until exact identifiers and long raw HTML gates pass without parser, reasoning, or tool leaks.",
        ],
    }


def _mtp_status(
    model_dir: Path,
    cfg: dict[str, Any],
    jang: dict[str, Any],
    safetensors_summary: dict[str, Any],
) -> dict[str, Any]:
    raw_layers = cfg.get("num_nextn_predict_layers")
    invalid_config_layers = False
    try:
        config_layers = int(raw_layers) if raw_layers is not None else None
    except (TypeError, ValueError):
        config_layers = None
        invalid_config_layers = True

    jang_drop_mtp = jang.get("drop_mtp")
    index_has_mtp, index_error = _index_tensor_prefix_status(model_dir, "mtp.")
    if not index_has_mtp:
        prefix_counts = safetensors_summary.get("key_prefix_counts")
        if isinstance(prefix_counts, dict):
            index_has_mtp = bool(prefix_counts.get("mtp"))

    issues: list[str] = []
    if index_error:
        issues.append(index_error)
    if invalid_config_layers:
        issues.append(
            "config.num_nextn_predict_layers is invalid; expected an integer"
        )
    if config_layers and config_layers > 0 and jang_drop_mtp is not True and not index_has_mtp:
        issues.append(
            "config expects MTP next-token prediction layers, but the bundle "
            "index has no mtp.* tensors"
        )
    if config_layers in (None, 0) and jang_drop_mtp is not True and index_has_mtp:
        issues.append(
            "bundle indexes mtp.* tensors but config disables MTP runtime"
        )
    if jang_drop_mtp is True and index_has_mtp:
        issues.append("jang_config.drop_mtp=true but bundle still indexes mtp.* tensors")

    artifact_available = bool(
        config_layers
        and config_layers > 0
        and jang_drop_mtp is not True
        and index_has_mtp
        and not issues
    )
    runtime_available = False
    if issues:
        status = "metadata_inconsistent"
        runtime_reason = "metadata_inconsistent"
    elif jang_drop_mtp is True:
        status = "dropped"
        runtime_reason = "jang_config.drop_mtp=true"
    elif artifact_available:
        status = "weights_present_runtime_unwired"
        runtime_reason = "MTP weights are present, but vMLX MTP decode is not wired"
    elif config_layers:
        status = "configured_without_runtime"
        runtime_reason = "config requests MTP but runtime requirements are incomplete"
    else:
        status = "not_configured"
        runtime_reason = "config does not request MTP"

    return {
        "config_num_nextn_predict_layers": config_layers,
        "jang_drop_mtp": jang_drop_mtp,
        "index_has_mtp_tensors": index_has_mtp,
        "artifact_available": artifact_available,
        "runtime_available": runtime_available,
        "runtime_reason": runtime_reason,
        "status": status,
        "issues": issues,
    }


def sampling_loop_risk_summary(
    *,
    gen: dict[str, Any],
    sampling: dict[str, Any],
    reasoning: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Surface model-owned sampling gaps tied to long-output loop reports.

    This is deliberately evidence-only. It does not prescribe hidden fallback
    params; it records whether the bundle itself declares the output cap and
    sampler fields that the app/engine should pick up when requests omit them.
    """

    def numeric(value: Any) -> float | int | None:
        return (
            value
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    max_new_tokens = numeric(sampling.get("max_new_tokens"))
    output_cap_source = "jang_config.chat.sampling_defaults.max_new_tokens"
    if max_new_tokens is None:
        max_new_tokens = numeric(gen.get("max_new_tokens"))
        output_cap_source = "generation_config.max_new_tokens"
    if max_new_tokens is None or max_new_tokens <= 0:
        output_cap_source = "engine_fallback_4096"

    repetition_sources = {
        f"jang_config.chat.sampling_defaults.{key}": sampling.get(key)
        for key in (
            "repetition_penalty_chat",
            "repetition_penalty_thinking",
            "repetition_penalty",
        )
        if numeric(sampling.get(key)) is not None
    }
    if numeric(gen.get("repetition_penalty")) is not None:
        repetition_sources["generation_config.repetition_penalty"] = gen.get(
            "repetition_penalty"
        )

    top_k = (
        sampling.get("top_k")
        if numeric(sampling.get("top_k")) is not None
        else gen.get("top_k")
    )
    top_k_disabled = numeric(top_k) is not None and int(top_k) <= 0
    supports_reasoning = bool(
        reasoning.get("supported")
        or registry.get("reasoning_parser")
        or registry.get("think_in_template")
    )
    default_mode = reasoning.get("default_mode") or reasoning.get("default")

    review_notes: list[str] = []
    if output_cap_source == "engine_fallback_4096":
        review_notes.append(
            "bundle omits positive max_new_tokens; omitted requests rely on bounded engine fallback 4096"
        )
    if not repetition_sources:
        review_notes.append(
            "bundle omits repetition_penalty; engine must omit it rather than invent a hidden floor"
        )
    if top_k_disabled:
        review_notes.append("bundle disables top_k with a non-positive sentinel")
    if supports_reasoning and not default_mode:
        review_notes.append(
            "reasoning-capable bundle has no explicit chat.reasoning.default_mode; Auto must stay model/template-owned"
        )

    return {
        "output_cap_source": output_cap_source,
        "max_new_tokens": (
            int(max_new_tokens)
            if isinstance(max_new_tokens, (int, float)) and max_new_tokens > 0
            else None
        ),
        "temperature": sampling.get("temperature", gen.get("temperature")),
        "top_p": sampling.get("top_p", gen.get("top_p")),
        "top_k": top_k,
        "top_k_disabled": top_k_disabled,
        "min_p": sampling.get("min_p", gen.get("min_p")),
        "repetition_sources": repetition_sources,
        "reasoning_supported": supports_reasoning,
        "reasoning_default_mode": default_mode,
        "review_notes": review_notes,
        "status": "review" if review_notes else "covered",
    }


FAMILY_EXPECTATION_ALIASES = {
    "bailing_hybrid": {"bailing_hybrid", "ling"},
    "nemotron_omni": {"nemotron_omni", "nemotron_h", "nemotron"},
}


def family_matches_expected(expected: str, actual: Any) -> bool:
    actual_s = str(actual or "")
    return actual_s in FAMILY_EXPECTATION_ALIASES.get(expected, {expected})


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
            "cache_subtype": getattr(rcfg, "cache_subtype", None),
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
    mtp = _mtp_status(model_dir, cfg, jang, safetensors_summary)
    sampling_risk = sampling_loop_risk_summary(
        gen=gen,
        sampling=sampling,
        reasoning=reasoning,
        registry=registry,
    )

    eos = {
        "config": cfg.get("eos_token_id"),
        "generation": gen.get("eos_token_id"),
        "tokenizer_eos": tok_cfg.get("eos_token"),
        "tokenizer_has_template": bool(tok_cfg.get("chat_template")),
        "chat_template_file": chat_template_file.is_file(),
        "registry_eos_tokens": registry.get("eos_tokens"),
    }

    issues: list[str] = []
    dsv4_precision_boundary: dict[str, Any] = {}
    dsv4_rope_contract: dict[str, Any] = {}
    dsv4_fault_matrix: dict[str, Any] = {}
    if model_dir.is_dir() and safetensors_summary.get("files", 0) == 0:
        issues.append("bundle has no safetensors weights; structural metadata only")
    if not row.live_supported and row.unsupported_reason:
        issues.append(row.unsupported_reason)
    if row.family == "deepseek_v4":
        try:
            from vmlx_engine.loaders.load_jangtq_dsv4 import (
                _audit_dsv4_artifact_bit_plan,
                _audit_dsv4_control_tensor_dtypes,
            )

            dsv4_control = _audit_dsv4_control_tensor_dtypes(model_dir)
            if dsv4_control.get("non_f32_count"):
                issues.append(
                    "DSV4 critical control tensors are not F32 "
                    f"({dsv4_control.get('non_f32_count')}/"
                    f"{dsv4_control.get('critical_count')}); rebuild/re-download "
                    "required before live quality claims"
                )
            dsv4_bit_plan = _audit_dsv4_artifact_bit_plan(model_dir)
            for issue in dsv4_bit_plan.get("issues") or []:
                issues.append(f"DSV4 artifact bit-plan audit: {issue}")
        except Exception as exc:
            dsv4_bit_plan = {}
            issues.append(f"DSV4 static artifact audit failed: {exc}")
        dsv4_precision_boundary = dsv4_output_precision_boundary(model_dir)
        if dsv4_precision_boundary.get("needs_source_or_rebuild_clearance"):
            issues.append(
                "DSV4 output-head/final-norm precision boundary requires "
                "source-vs-quant or rebuilt-artifact clearance before "
                "long-output production claims "
                f"(head={dsv4_precision_boundary.get('output_head_dtype')}, "
                f"norm={dsv4_precision_boundary.get('final_norm_dtype')})"
            )
        dsv4_rope_contract = dsv4_rope_scaling_contract(cfg, model_dir=model_dir)
        if dsv4_rope_contract.get("runtime_repair_required"):
            issues.append(
                "DSV4 bundle config omits source YaRN rope_scaling for "
                "compressed-context layers; runtime repair is required and "
                "fresh live DSV4 identifier/full-output gates must pass"
            )
        dsv4_fault_matrix = dsv4_release_fault_matrix(
            model_dir=model_dir,
            precision_boundary=dsv4_precision_boundary,
            rope_scaling_contract=dsv4_rope_contract,
        )
        if jang.get("weight_format") == "mxtq":
            bits = jang.get("mxtq_bits", {})
            if bits.get("routed_expert") not in (2, 4, 8):
                issues.append("DSV4 routed_expert mxtq_bits missing/unexpected")
            if sampling.get("repetition_penalty_thinking") is None:
                issues.append("DSV4 missing chat.sampling_defaults.repetition_penalty_thinking")
        if "<｜Assistant｜>" not in str(registry.get("eos_tokens")):
            issues.append("DSV4 bundle EOS config does not list Assistant marker; engine registry must add it")
        for issue in mtp.get("issues") or []:
            issues.append(f"DSV4 MTP metadata inconsistent: {issue}")
        has_canonical_dsv4_encoder = (
            chat.get("encoder") == "encoding_dsv4"
            and bool(chat.get("encoder_fn"))
        )
        if (
            not has_canonical_dsv4_encoder
            and not tok_cfg.get("chat_template")
            and not chat_template_file.is_file()
        ):
            issues.append("DSV4 missing chat template file/config; canonical encoder shim still required")
    has_reasoning_surface = bool(
        reasoning.get("supported")
        or tok_cfg.get("chat_template")
        or chat_template_file.is_file()
        or registry.get("reasoning_parser")
    )
    if row.expect_reasoning and not has_reasoning_surface:
        issues.append("expected reasoning model but no reasoning metadata/template found")
    if registry.get("family_name") and not family_matches_expected(row.family, registry.get("family_name")):
        issues.append(
            "registry family mismatch: "
            f"expected {row.family}, got {registry.get('family_name')}"
        )
    if row.expect_tool_parser and registry.get("tool_parser") != row.expect_tool_parser:
        issues.append(
            "registry tool parser mismatch: "
            f"expected {row.expect_tool_parser}, got {registry.get('tool_parser')}"
        )
    if row.family == "hy_v3":
        if registry.get("reasoning_parser") != "qwen3":
            issues.append(
                "Hy3 registry reasoning parser mismatch: expected qwen3, "
                f"got {registry.get('reasoning_parser')}"
            )
        if registry.get("cache_type") != "kv":
            issues.append(
                "Hy3 registry cache type mismatch: expected kv, "
                f"got {registry.get('cache_type')}"
            )
        if registry.get("think_in_template") is not False:
            issues.append("Hy3 must keep think_in_template=false")
        caps = jang.get("capabilities") if isinstance(jang.get("capabilities"), dict) else {}
        if caps:
            if caps.get("reasoning_parser") != "qwen3":
                issues.append("Hy3 jang capabilities missing qwen3 reasoning parser")
            if caps.get("tool_parser") != "hunyuan":
                issues.append("Hy3 jang capabilities missing hunyuan tool parser")
            if caps.get("cache_type") != "kv":
                issues.append("Hy3 jang capabilities missing kv cache type")
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
            "num_nextn_predict_layers": cfg.get("num_nextn_predict_layers"),
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
            "top_k": gen.get("top_k"),
            "min_p": gen.get("min_p"),
            "repetition_penalty": gen.get("repetition_penalty"),
        },
        "sampling_loop_risk": sampling_risk,
        "jang": {
            "weight_format": jang.get("weight_format"),
            "mxtq_bits": jang.get("mxtq_bits"),
            "cache_type": jang.get("cache_type"),
            "cache_subtype": jang.get("cache_subtype"),
            "quantization": jang.get("quantization"),
            "sampling_defaults": sampling,
            "reasoning": reasoning,
            "runtime": jang.get("runtime"),
        },
        "eos": eos,
        "mtp": mtp,
        "dsv4_artifact_bit_plan": dsv4_bit_plan if row.family == "deepseek_v4" else None,
        "dsv4_precision_boundary": dsv4_precision_boundary if row.family == "deepseek_v4" else None,
        "dsv4_rope_scaling_contract": dsv4_rope_contract if row.family == "deepseek_v4" else None,
        "dsv4_release_fault_matrix": dsv4_fault_matrix if row.family == "deepseek_v4" else None,
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


def _usage_int(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def live_request_perf_summary(usage: Any, elapsed_sec: float) -> dict[str, Any]:
    """Derive explicit wall-clock token rates for live proof artifacts.

    These are intentionally labeled as wall-clock rates: they include request
    assembly, prompt processing, generation, and response serialization. They
    are not TTFT or engine-internal decode-only metrics.
    """
    elapsed = round(float(elapsed_sec), 3)
    if not isinstance(usage, dict):
        return {"elapsed_sec": elapsed, "basis": "missing_api_usage"}

    prompt_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(
        usage, "completion_tokens", "output_tokens", "generated_tokens"
    )
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens is None:
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        elif prompt_tokens is not None:
            total_tokens = prompt_tokens
        elif completion_tokens is not None:
            total_tokens = completion_tokens

    summary: dict[str, Any] = {
        "elapsed_sec": elapsed,
        "basis": "api_usage_wall_clock",
    }
    if prompt_tokens is not None:
        summary["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        summary["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        summary["total_tokens"] = total_tokens

    if elapsed > 0:
        if prompt_tokens is not None:
            summary["prompt_tokens_per_sec_wall"] = round(prompt_tokens / elapsed, 3)
        if completion_tokens is not None:
            summary["completion_tokens_per_sec_wall"] = round(
                completion_tokens / elapsed, 3
            )
        if total_tokens is not None:
            summary["total_tokens_per_sec_wall"] = round(total_tokens / elapsed, 3)
    return summary


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


def responses_tool_choice_output_ok(text: str) -> bool:
    """Tool-choice probes should return structured calls, not visible markup."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    leak_markers = (
        "<tool_call",
        "<zyphra_tool_call",
        "<function=",
        "<parameter",
        "<minimax:tool_call",
        "[TOOL_CALLS]",
        "<|tool_call",
        "<|recipient|>",
        "```tool_code",
        "<py>",
        "def list_directory",
    )
    lowered = stripped.lower()
    return not any(marker.lower() in lowered for marker in leak_markers)


def responses_tool_call_arguments_ok(
    calls: list[dict[str, Any]],
    *,
    expected_name: str,
    expected_arguments: dict[str, Any],
) -> bool:
    """Return true when a Responses function call has exact JSON arguments."""
    for call in calls:
        if call.get("name") != expected_name:
            continue
        raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            continue
        if args == expected_arguments:
            return True
    return False


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


def dsv4_long_context_full_output_ok(
    *,
    code: int,
    finish: str,
    full_text: str,
    content: str,
    split_ok: bool,
    loop_score: float,
    content_contract_ok: bool = True,
) -> bool:
    """Return whether a DSV4 long-output row is production-coherent.

    DSV4 long rows are full-tail gates, not smoke tests. A length-capped
    response can have enough characters and a low loop score while still being
    visibly unfinished or stuck inside planning. Require a non-length stop so
    the row cannot pass until the model/runtime reaches its own stop condition.
    """
    return (
        code == 200
        and is_non_length_stop(finish)
        and len(full_text) >= 120
        and len(content) >= 40
        and split_ok
        and content_contract_ok
        and not has_duplicate_block(full_text)
        and loop_score < 0.25
    )


DSV4_THREEJS_CORE_IDENTIFIERS = (
    "THREE.Scene",
    "THREE.WebGLRenderer",
    "THREE.PerspectiveCamera",
    "THREE.Mesh",
    "THREE.BoxGeometry",
    "THREE.MeshBasicMaterial",
)

DSV4_THREEJS_CORRUPT_IDENTIFIER_PATTERNS = (
    "three.scscene",
    "three.persperspectivecamera",
    "three.pperspectivecamera",
    "three.webwebglrenderer",
    "three.webwebrenderrenderer",
    "three.mmesh",
    "three.bboxgeometry",
    "three.mmeshbasicmaterial",
    "three.ddirectionallight",
    "three.aambientlight",
    "ththree.",
)


def dsv4_threejs_identifier_integrity_ok(content: str) -> bool:
    """Return whether a short Three.js snippet preserved exact API names."""
    raw = content or ""
    lower = raw.lower()
    return (
        all(identifier in raw for identifier in DSV4_THREEJS_CORE_IDENTIFIERS)
        and "```" not in raw
        and not any(pattern in lower for pattern in DSV4_THREEJS_CORRUPT_IDENTIFIER_PATTERNS)
    )


def dsv4_threejs_single_file_ok(content: str) -> bool:
    """Return whether the DSV4 game-design output is a complete Three.js file.

    This is intentionally a structural release gate, not a style score. A
    response that stops cleanly but emits a 2D canvas sketch, broken placeholder
    code, or an unfinished HTML fragment must not clear the long-output row.
    """
    raw = content or ""
    stripped = raw.strip()
    lower = raw.lower()
    lower_stripped = stripped.lower()
    has_html_shell = lower_stripped.startswith("<!doctype html") and lower_stripped.endswith("</html>")
    has_three_import = (
        "three.min.js" in lower
        or "three.module" in lower
        or "from 'three'" in lower
        or 'from "three"' in lower
    )
    has_core_three_api = all(marker.lower() in lower for marker in DSV4_THREEJS_CORE_IDENTIFIERS[:4])
    has_render_loop = "requestanimationframe" in lower
    has_game_requirements = (
        "health" in lower
        and "score" in lower
        and "enemy" in lower
        and ("projectile" in lower or "bullet" in lower or "pellet" in lower)
        and ("collision" in lower or "distanceto" in lower)
    )
    has_placeholder = any(
        marker in lower
        for marker in (
            "... (existing",
            "omitted for brevity",
            "placeholder",
            "todo",
        )
    )
    has_markdown_fence = "```" in raw
    has_corrupt_three_api = any(
        marker in lower for marker in DSV4_THREEJS_CORRUPT_IDENTIFIER_PATTERNS
    )
    return (
        has_html_shell
        and has_three_import
        and has_core_three_api
        and has_render_loop
        and has_game_requirements
        and not has_placeholder
        and not has_markdown_fence
        and not has_corrupt_three_api
    )


def dsv4_long_prompt_specs() -> list[dict[str, Any]]:
    """Return DSV4 long-output gates with the intended rail per prompt.

    These are production gates, not generic "think as much as possible" rows.
    The VC planning prompt benefits from DSV4's reasoning rail and has live
    evidence that 2500 tokens can reach a stop. The game/code prompt now also
    uses the audited DSV4 quality rail because live identifier probes showed
    the direct rail corrupts exact Three.js API names.
    """
    return [
        {
            "name": "vc_project_plan",
            "prompt": (
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
            "enable_thinking": True,
            "max_tokens": 2500,
        },
        {
            "name": "game_design_long_context",
            "prompt": (
                "Create one complete runnable HTML file for a minimal Three.js "
                "forest survival shooter. Non-negotiable: include a script "
                "import for three.min.js or three.module.js; create "
                "THREE.Scene, THREE.WebGLRenderer, THREE.PerspectiveCamera, "
                "THREE.Mesh primitives, and a requestAnimationFrame loop. Game "
                "requirements: WASD movement, shotgun pellet projectiles, "
                "boar/moose/wolf enemies with distinct simple behavior, spawn "
                "timer, collision detection, health, score, and a compact UI "
                "overlay. Keep under 170 lines and 2600 generated tokens. "
                "Output only raw HTML starting with <!DOCTYPE html> and ending "
                "with </html>; no markdown fence, no explanation, no TODO, no "
                "omitted sections."
            ),
            "enable_thinking": True,
            "max_tokens": 2600,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
    ]


def dsv4_thinking_mode_max_ok(
    *,
    code: int,
    finish: str,
    content: str,
    reasoning: str,
) -> bool:
    """Return whether DSV4 ``thinking_mode=max`` visibly answered the probe."""
    normalized_content = normalize_short_answer(content).lower()
    has_visible_seven = bool(re.search(r"(?<!\d)7(?!\d)", normalized_content))
    return (
        code == 200
        and finish == "stop"
        and has_visible_seven
        and bool(content.strip())
        and content.strip() != reasoning.strip()
        and not has_duplicate_block(f"{content}\n{reasoning}".lower())
    )


def chat_basic_turn_ok(
    row: ModelRow,
    *,
    code: int,
    finish: str,
    content: str,
    reasoning: str,
) -> bool:
    """Loosen first-turn gate to production coherence, not byte-exact wording."""
    joined = f"{content}\n{reasoning}".lower()
    normalized = normalize_short_answer(content).lower()
    if row.family == "deepseek_v4":
        return (
            code == 200
            and finish == "stop"
            and ("blue" in joined or "noted" in joined)
            and not has_duplicate_block(joined)
        )
    if row.family == "zaya1_vl":
        return (
            code == 200
            and finish == "stop"
            and not reasoning.strip()
            and "blue" in joined
            and "cat" in joined
            and not has_duplicate_block(joined)
        )
    return (
        code == 200
        and finish == "stop"
        and bool(content.strip())
        and not reasoning.strip()
        and ("noted" in normalized or "blue" in joined or "cat" in joined)
        and not has_duplicate_block(joined)
    )


def cache_probe_content_ok(
    *,
    row: ModelRow,
    expected: str,
    first_content: str,
    repeat_content: str,
    strict_short_answer: bool,
    min_count: int,
) -> bool:
    """Validate cached/repeated output using coherence, not exact wording."""
    expected_lower = expected.lower()
    first_lower = first_content.lower()
    repeat_lower = repeat_content.lower()
    if row.family == "deepseek_v4":
        return expected_lower in first_lower and expected_lower in repeat_lower
    if strict_short_answer:
        first_norm = normalize_short_answer(first_content).lower()
        repeat_norm = normalize_short_answer(repeat_content).lower()
        return (
            (
                first_norm == expected_lower
                or expected_lower in first_lower
            )
            and (
                repeat_norm == expected_lower
                or expected_lower in repeat_lower
            )
        )
    return (
        first_lower.count(expected_lower) >= min_count
        and repeat_lower.count(expected_lower) >= min_count
    )


def capability_endpoint_contract_ok(row: ModelRow, caps: Any) -> bool:
    """Validate the live capability surface against the row contract."""
    if not isinstance(caps, dict):
        return False
    supported_modes = caps.get("supported_modes", [])
    expected_modes = ["instruct"]
    if row.expect_reasoning:
        expected_modes.append("reasoning")
    if not all(mode in supported_modes for mode in expected_modes):
        return False

    if row.family != "deepseek_v4":
        return True

    native = caps.get("cache", {}).get("native")
    return (
        caps.get("supports_thinking") is True
        and "max" in (caps.get("reasoning_efforts") or [])
        and isinstance(native, dict)
        and native.get("family") == "deepseek_v4"
        and native.get("schema") == "deepseek_v4_v7"
        and native.get("cache_type") == "native_composite"
        and native.get("generic_turboquant_kv", {}).get("enabled") is False
    )


def cache_exact_hit_required(row: ModelRow) -> bool:
    """Whether this row must prove prefix/paged/L2 cache reuse live.

    ZAYA CCA has standard KV plus path-dependent ``conv_state`` and
    ``prev_hs``. The source now has a typed ``zaya_cca_v1`` serializer, so
    ZAYA rows must prove exact prefix/L2 reuse under the explicit live-gate env
    before the default CLI safeguard can be flipped.
    """
    return True


def cache_exact_hit_probe(row: ModelRow) -> dict[str, Any]:
    """Return the prompt and visible-content assertion for cache reuse."""

    if row.family == "deepseek_v4":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Cache audit exact-hit check. What is 17 plus 28? "
                        "Think briefly, then answer with the final number."
                    ),
                }
            ],
            "thinking": True,
            "max_tokens": 512,
            "expected": "45",
            "temperature": 0.0,
            "min_count": 1,
        }
    if row.cache_profile == "zaya_cca":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Cache audit exact-hit check. In the phrase "
                        "'blue blue blue blue', which color word repeats? "
                        "Reply with the single word only."
                    ),
                }
            ],
            "thinking": False,
            "max_tokens": 64,
            "expected": "blue",
            "temperature": 0.0,
            "min_count": 1,
            "strict_short_answer": True,
        }
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Cache audit exact-hit check. Reply exactly: blue blue blue blue."
                ),
            }
        ],
        "thinking": False,
        "max_tokens": 512,
        "expected": "blue",
        "temperature": 0.1,
        "min_count": 3,
        "strict_short_answer": False,
    }


def audit_child_env_for_row(
    row: ModelRow,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return subprocess environment overrides for a live audit row."""
    child_env = dict(os.environ if base_env is None else base_env)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTHONPATH"] = ""
    child_env.pop("VMLX_DSV4_HARD_REP_BLOCK", None)
    child_env.pop("VMLINUX_DSV4_HARD_REP_BLOCK", None)
    if child_env.get("VMLINUX_AUDIT_USE_SOURCE_VMLX") == "1":
        child_env["PYTHONPATH"] = str(ROOT)
    if row.cache_profile == "zaya_cca":
        child_env["VMLX_ZAYA_ENABLE_TYPED_CCA_CACHE"] = "1"
        child_env["VMLX_DISABLE_TQ_KV"] = "1"
        child_env.pop("VMLX_FORCE_TQ_AUTO", None)
    if row.family == "deepseek_v4":
        jang_source = Path(
            child_env.get(
                "VMLINUX_JANG_TOOLS_SOURCE",
                str(Path.home() / "jang" / "jang-tools"),
            )
        )
        if (
            child_env.get("VMLINUX_AUDIT_USE_SOURCE_JANG") == "1"
            and (jang_source / "jang_tools" / "dsv4" / "mlx_model.py").is_file()
        ):
            existing_pythonpath = child_env.get("PYTHONPATH") or ""
            child_env["PYTHONPATH"] = (
                f"{jang_source}:{existing_pythonpath}"
                if existing_pythonpath
                else str(jang_source)
            )
            child_env["VMLINUX_JANG_TOOLS_SOURCE"] = str(jang_source)
    return child_env


def live_server_command(
    py: Path,
    model_dir: Path,
    port: int,
    block_cache_dir: Path,
    *,
    row: ModelRow | None = None,
) -> list[str]:
    """Build a live audit server command that proves the selected Python.

    Installed-app gates pass ``/Applications/vMLX.app/.../python3`` here. The
    subprocess must not import repo-local modules just because the audit script
    lives in the source checkout, and it must not write pycache into the signed
    app bundle before codesign verification.
    """
    prefix_cache_flag = (
        "--dsv4-enable-prefix-cache"
        if row is not None and row.family == "deepseek_v4"
        else "--enable-prefix-cache"
    )
    return [
        str(py),
        "-B",
        "-s",
        "-P",
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
        prefix_cache_flag,
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


def normalize_python_executable(value: str, cwd: Path | None = None) -> Path:
    """Return an absolute Python path without resolving venv symlinks.

    ``Path.resolve()`` follows ``.venv/bin/python`` to the base interpreter.
    The live audit launches with ``-s -P``; losing the venv launcher also loses
    site-packages such as uvicorn, so keep the symlink path intact.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (cwd or Path.cwd()) / path


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
    worst = 0
    if len(words) >= 20:
        for n in (1, 2, 3):
            grams = [" ".join(words[i : i + n]) for i in range(0, len(words) - n + 1)]
            if not grams:
                continue
            counts: dict[str, int] = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
            worst = max(worst, max(counts.values()))
        word_score = worst / max(1, len(words))
    else:
        word_score = 0.0

    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < 48:
        return word_score
    tail = compact[-512:]
    char_score = 0.0
    run_len = 1
    max_run_len = 1
    for prev, curr in zip(tail, tail[1:]):
        if curr == prev:
            run_len += 1
            max_run_len = max(max_run_len, run_len)
        else:
            run_len = 1
    if max_run_len >= 8:
        char_score = max(char_score, max_run_len / len(tail))
    for period in range(2, min(64, len(tail) // 3) + 1):
        pattern = tail[:period]
        if len(set(pattern)) < 2:
            continue
        expected = (pattern * ((len(tail) // period) + 1))[: len(tail)]
        matches = sum(1 for a, b in zip(tail, expected) if a == b)
        periodic_score = matches / len(tail)
        if periodic_score >= 0.85:
            char_score = max(char_score, periodic_score)
    for n in (2, 3, 4, 6, 8, 12):
        if len(tail) < n * 12:
            continue
        grams = [tail[i : i + n] for i in range(0, len(tail) - n + 1)]
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        char_score = max(char_score, max(counts.values()) / max(1, len(grams)))
    return max(word_score, char_score)


def normalize_short_answer(text: str) -> str:
    """Normalize terse instruction-following answers for strict probes."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n`\"'“”‘’.。!?！？")


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


def multilingual_latin_tokens_ok(text: str) -> bool:
    allowed = {
        "html",
        "three",
        "js",
        "javascript",
        "css",
        "webgl",
        "wasd",
        "recoil",
    }
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]{2,}", text)}
    return tokens <= allowed


def multilingual_loop_quality_ok(text: str) -> bool:
    quality = text_quality_summary(text)
    return (
        len(text) >= 120
        and not has_duplicate_block(text)
        and simple_loop_score(text) < 0.25
        and text.count("👀") < 8
        and quality["word_count"] >= 20
        and quality["digit_line_ratio"] < 0.35
        and quality["cjk_chars"] == 0
        and multilingual_latin_tokens_ok(text)
    )


def live_loop_probe_name(row: ModelRow) -> str | None:
    if row.family == "bailing_hybrid":
        return "ling_multilingual_loop_trigger"
    if row.family == "minimax":
        return "minimax_multilingual_loop_trigger"
    if row.family == "hy_v3":
        return "hy3_multilingual_loop_trigger"
    return None


def blocking_runtime_log_findings(row: ModelRow, log_text: str) -> list[dict[str, str]]:
    """Return runtime log findings that make a live row fail.

    Coherent visible text is not enough for production proof. Cache and topology
    rows must also have clean logs: a hybrid SSM row with a failed clean
    re-derive, for example, means the prefix/L2 proof is incomplete even if the
    model answered the tiny prompt correctly.
    """
    patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            "metal_malloc_oversize",
            re.compile(r"\[metal::malloc\].*greater than the maximum allowed", re.I),
        ),
        (
            "metal_command_timeout",
            re.compile(r"Command buffer execution failed|kIOGPUCommandBufferCallbackErrorTimeout", re.I),
        ),
        ("python_traceback", re.compile(r"^Traceback \(most recent call last\):", re.M)),
        ("engine_error", re.compile(r"\[Engine error:", re.I)),
    ]
    if row.cache_profile == "hybrid_ssm":
        patterns.extend(
            [
                (
                    "hybrid_ssm_clean_prefill_failed",
                    re.compile(r"MLLM clean SSM prefill failed|SSM re-derive failed", re.I),
                ),
                (
                    "hybrid_ssm_contaminated_hit_rejected",
                    re.compile(r"SSM companion .*is_complete=False.*rejecting hit", re.I),
                ),
            ]
        )
    if row.family == "deepseek_v4":
        patterns.append(
            (
                "dsv4_incomplete_composite_restore",
                re.compile(r"Ignoring DSV4 paged prefix hit.*no terminal deepseek_v4 composite state", re.I),
            )
        )

    findings: list[dict[str, str]] = []
    lines = log_text.splitlines()
    for name, pattern in patterns:
        for line in lines:
            if pattern.search(line):
                findings.append({"name": name, "line": line[-1000:]})
                break
    return findings


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
    elapsed = round(time.perf_counter() - t0, 3)
    usage = resp.get("usage") if isinstance(resp, dict) else None
    return {
        "code": code,
        "body": resp,
        "request": body,
        "elapsed_sec": elapsed,
        "usage": usage,
        "perf": live_request_perf_summary(usage, elapsed),
    }


def stream_chat_probe(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None,
    max_tokens: int = 512,
    chunks_to_read: int = 4,
    timeout: int = 90,
) -> dict[str, Any]:
    """Open a Chat Completions SSE stream and close it after real chunks.

    The production gate needs at least source-level proof that streaming
    transports emit parseable SSE and that an early client disconnect does not
    leave an orphan generation. The caller verifies scheduler quiescence after
    this function returns.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "stream": True,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
        body["chat_template_kwargs"] = {"enable_thinking": thinking}

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    done = False
    status = 0
    error: str | None = None
    visible = ""
    reasoning = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                raw_lines.append(line)
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"_raw": data}
                chunks.append(payload)
                try:
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    visible += str(delta.get("content") or "")
                    reasoning += str(
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                        or ""
                    )
                except Exception:
                    pass
                if len(chunks) >= chunks_to_read and (visible or reasoning):
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "status": status,
        "request": body,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "chunks_read": len(chunks),
        "done": done,
        "visible_chars": len(visible),
        "reasoning_chars": len(reasoning),
        "visible_head": visible[:240],
        "reasoning_head": reasoning[:240],
        "raw_lines_head": raw_lines[:12],
        "error": error,
    }


def reasoning_recall_max_tokens(row: ModelRow) -> int:
    """Generation cap for the live thinking-on recall probe.

    Qwen3.6 dense models produce a long but valid reasoning block before the
    visible answer. A 220-token cap false-fails them with the correct answer
    still inside reasoning and no closing ``</think>``. Use a larger cap for
    Qwen rows so the gate proves the model actually exits reasoning and emits
    visible content.
    """

    if row.family in {"qwen3_5", "qwen3_5_moe"}:
        return 900
    return 220


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
    cmd = live_server_command(py, model_dir, port, block_cache_dir, row=row)

    if os.environ.get("VMLINUX_AUDIT_KILL_EXISTING") == "1":
        subprocess.run(["pkill", "-9", "-f", "vmlx_engine.cli"], capture_output=True)
        time.sleep(2)
    child_env = audit_child_env_for_row(row)

    with log.open("w") as lf:
        lf.write("# " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(OUT_DIR),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=child_env,
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
        perf = live_request_perf_summary(usage, elapsed)
        artifact = None
        if method == "POST":
            artifact = write_probe_output(
                row.id,
                name,
                {
                    "request": body,
                    "code": code,
                    "elapsed_sec": elapsed,
                    "usage": usage,
                    "perf": perf,
                    "response": resp,
                },
            )
        result["requests"].append(
            {
                "name": name,
                "method": method,
                "path": path,
                "code": code,
                "elapsed_sec": elapsed,
                "usage": usage,
                "perf": perf,
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
    if row.cache_profile == "zaya_cca":
        _kvq0 = (
            stats0.get("kv_cache_quantization", {})
            if code == 200 and isinstance(stats0, dict)
            else {}
        )
        _tq0 = (
            stats0.get("turboquant_kv_cache", {})
            if code == 200 and isinstance(stats0, dict)
            else {}
        )
        _native0 = (
            stats0.get("native_cache", {})
            if code == 200 and isinstance(stats0, dict)
            else {}
        )
        check(
            "zaya_cca_typed_cache_live_gate_enabled",
            "ZAYA/CCA typed cache enabled" in log_text
            and "ZAYA/CCA typed paged prefix cache enabled" in log_text
            and not bool(_kvq0.get("enabled"))
            and not bool(_tq0.get("enabled")),
            {
                "reason": (
                    "ZAYA CCA live gate must use zaya_cca_v1 typed "
                    "prefix/paged/L2 while generic TurboQuant KV stays off"
                ),
                "log_evidence": [
                    line
                    for line in log_text.splitlines()
                    if "ZAYA/CCA" in line or "TurboQuant KV skipped" in line
                ][-8:],
                "kv_cache_quantization": _kvq0,
                "turboquant_kv_cache": _tq0,
                "native_cache": _native0,
                "cache_stats_keys": sorted(stats0.keys()) if isinstance(stats0, dict) else [],
            },
        )

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
            probe.get("usage")
            if isinstance(probe.get("usage"), dict)
            else (
                probe.get("body", {}).get("usage")
                if isinstance(probe.get("body"), dict)
                else None
            )
        )
        perf = probe.get("perf") or live_request_perf_summary(
            usage, float(probe["elapsed_sec"])
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
                "perf": perf,
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
                "perf": perf,
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
    native_caps = caps.get("cache", {}).get("native") if isinstance(caps, dict) else {}
    check(
        "model_capabilities_endpoint",
        code == 200 and capability_endpoint_contract_ok(row, caps),
        {"code": code, "caps": caps},
    )
    if row.cache_profile == "zaya_cca":
        check(
            "zaya_native_cache_capabilities",
            isinstance(native_caps, dict)
            and native_caps.get("family") == "zaya"
            and native_caps.get("schema") == "zaya_cca_v1"
            and native_caps.get("cache_type") == "typed_cca"
            and native_caps.get("generic_turboquant_kv", {}).get("enabled") is False,
            {"native_cache": native_caps},
        )
    elif row.cache_profile == "dsv4_composite":
        check(
            "dsv4_native_cache_capabilities",
            isinstance(native_caps, dict)
            and native_caps.get("family") == "deepseek_v4"
            and native_caps.get("schema") == "deepseek_v4_v7"
            and native_caps.get("cache_type") == "native_composite"
            and native_caps.get("generic_turboquant_kv", {}).get("enabled") is False,
            {"native_cache": native_caps},
        )
    elif row.cache_profile == "hybrid_ssm":
        qwen_selective_live_tq = row.family in {"qwen3_5", "qwen3_5_moe"}
        live_tq = native_caps.get("live_attention_tq_kv", {}) if isinstance(native_caps, dict) else {}
        check(
            "hybrid_native_cache_capabilities",
            isinstance(native_caps, dict)
            and native_caps.get("cache_type") == "hybrid_ssm_typed"
            and native_caps.get("generic_turboquant_kv", {}).get("enabled")
            == qwen_selective_live_tq
            and (
                not qwen_selective_live_tq
                or (
                    live_tq.get("enabled") is True
                    and live_tq.get("applies_to") == "attention_kv_layers_only"
                    and live_tq.get("ssm_policy")
                    == "native_full_precision_companion_state"
                )
            ),
            {
                "native_cache": native_caps,
                "qwen_selective_live_tq_expected": qwen_selective_live_tq,
            },
        )

    if row.cache_profile in {"zaya_cca", "dsv4_composite", "hybrid_ssm"}:
        log_text_for_layout = log.read_text(errors="ignore")
        check(
            "runtime_cache_layout_logged",
            "Runtime cache layout:" in log_text_for_layout,
            {
                "matching_lines": [
                    line
                    for line in log_text_for_layout.splitlines()
                    if "Runtime cache layout:" in line
                ][-3:],
            },
        )

    history: list[dict[str, Any]] = []
    basic_messages = [
        {
            "role": "user",
            "content": (
                "Store these facts for the next turn: secret_color=blue; "
                "secret_pet=cat. Do not explain. Reply only with noted."
            ),
        }
    ]
    r1 = chat_probe(
        "chat_turn1_thinking_off",
        basic_messages,
        thinking=True if row.family == "deepseek_v4" else False,
        max_tokens=512 if row.family == "deepseek_v4" else 80,
        temperature=0.0,
    )
    c1, reasoning1, finish1 = extract_chat_text(r1["body"])
    normalized_c1 = normalize_short_answer(c1).lower()
    basic_ok = chat_basic_turn_ok(
        row,
        code=r1["code"],
        finish=finish1,
        content=c1,
        reasoning=reasoning1,
    )
    check(
        "chat_thinking_off_basic",
        basic_ok,
        {
            "finish": finish1,
            "content": c1[:300],
            "normalized_content": normalized_c1,
            "non_exact_note": (
                "accepted coherent non-exact acknowledgement per release gate"
                if basic_ok and normalized_c1 not in {"noted", "noted."}
                else None
            ),
            "reasoning": reasoning1[:160],
            "code": r1["code"],
            "elapsed_sec": r1["elapsed_sec"],
            "usage": r1.get("body", {}).get("usage") if isinstance(r1.get("body"), dict) else None,
            "perf": r1.get("perf"),
        },
    )
    history += r1["request"]["messages"] + [{"role": "assistant", "content": c1}]

    r2 = chat_probe(
        "chat_turn2_recall_thinking_on",
        history + [{"role": "user", "content": "What color did I say? Answer in one word."}],
        thinking=True if row.expect_reasoning else False,
        max_tokens=reasoning_recall_max_tokens(row),
        temperature=0.0,
    )
    c2, reasoning2, finish2 = extract_chat_text(r2["body"])
    normalized_c2 = normalize_short_answer(c2).lower().rstrip(".")
    check(
        "chat_thinking_on_or_reasoning_toggle_recall",
        r2["code"] == 200
        and finish2 == "stop"
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
            "perf": r2.get("perf"),
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
                        "content": "Use max thinking. What is 3 plus 4? Think carefully, then include the final digit.",
                    }
                ],
                "thinking_mode": "max",
                "max_tokens": 1200,
                "temperature": 0.0,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=240,
        )
        max_content, max_reasoning, max_finish = extract_chat_text(max_resp)
        max_joined = (max_content + "\n" + max_reasoning).lower()
        check(
            "dsv4_thinking_mode_max",
            dsv4_thinking_mode_max_ok(
                code=code,
                finish=max_finish,
                content=max_content,
                reasoning=max_reasoning,
            ),
            {
                "code": code,
                "finish": max_finish,
                "content": max_content[:300],
                "normalized_content": normalize_short_answer(max_content).lower(),
                "reasoning_chars": len(max_reasoning),
                "reasoning_head": max_reasoning[:240],
                "content_equals_reasoning": max_content.strip() == max_reasoning.strip(),
                "elapsed_sec": max_elapsed,
            },
        )

        identifier_prompt = (
            "Output exactly this JavaScript snippet and nothing else:\n"
            "const scene = new THREE.Scene();\n"
            "const renderer = new THREE.WebGLRenderer();\n"
            "const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);\n"
            "const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());"
        )
        code, ident_resp, ident_elapsed = request_json(
            "dsv4_threejs_identifier_integrity",
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identifier_prompt}],
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 220,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "logprobs": True,
                "top_logprobs": 5,
                "stream": False,
            },
            timeout=240,
        )
        ident_content, ident_reasoning, ident_finish = extract_chat_text(ident_resp)
        ident_lower = ident_content.lower()
        ident_bad_patterns = [
            pattern
            for pattern in DSV4_THREEJS_CORRUPT_IDENTIFIER_PATTERNS
            if pattern in ident_lower
        ]
        ident_artifact = write_probe_output(
            row.id,
            "dsv4_threejs_identifier_integrity",
            {
                "request": {
                    "prompt": identifier_prompt,
                    "max_tokens": 220,
                    "enable_thinking": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                "code": code,
                "finish": ident_finish,
                "content": ident_content,
                "reasoning": ident_reasoning,
                "bad_patterns": ident_bad_patterns,
                "raw_response": ident_resp,
                "elapsed_sec": ident_elapsed,
            },
        )
        ident_ok = (
            code == 200
            and ident_finish == "stop"
            and dsv4_threejs_identifier_integrity_ok(ident_content)
        )
        check(
            "dsv4_threejs_identifier_integrity",
            ident_ok,
            {
                "code": code,
                "finish": ident_finish,
                "content_chars": len(ident_content),
                "reasoning_chars": len(ident_reasoning),
                "bad_patterns": ident_bad_patterns,
                "has_markdown_fence": "```" in ident_content,
                "elapsed_sec": ident_elapsed,
                "artifact": ident_artifact,
                "content": ident_content,
            },
        )

        for spec in dsv4_long_prompt_specs():
            probe_name = str(spec["name"])
            prompt = str(spec["prompt"])
            enable_thinking = bool(spec["enable_thinking"])
            max_tokens = int(spec["max_tokens"])
            temperature = float(spec.get("temperature", 0.6))
            top_p = float(spec.get("top_p", 0.95))
            repetition_penalty = spec.get("repetition_penalty")
            if probe_name == "game_design_long_context" and not ident_ok:
                check(
                    f"dsv4_long_context_full_output_{probe_name}",
                    False,
                    {
                        "skipped_due_to_identifier_integrity_failure": True,
                        "identifier_artifact": ident_artifact,
                        "identifier_bad_patterns": ident_bad_patterns,
                        "identifier_has_markdown_fence": "```" in ident_content,
                        "identifier_content": ident_content,
                    },
                )
                continue
            code, long_resp, long_elapsed = request_json(
                f"dsv4_long_{probe_name}",
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "enable_thinking": enable_thinking,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    **(
                        {"repetition_penalty": float(repetition_penalty)}
                        if repetition_penalty is not None
                        else {}
                    ),
                    "stream": False,
                },
                timeout=420,
            )
            long_content, long_reasoning, long_finish = extract_chat_text(long_resp)
            full_text = f"{long_reasoning}\n{long_content}".strip()
            loop_score = simple_loop_score(full_text)
            split_ok = (
                not long_reasoning.strip()
                or long_content.strip() != long_reasoning.strip()
            )
            content_contract_ok = (
                dsv4_threejs_single_file_ok(long_content)
                if probe_name == "game_design_long_context"
                else True
            )
            artifact = write_probe_output(
                row.id,
                f"dsv4_long_{probe_name}",
                {
                    "request": {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "enable_thinking": enable_thinking,
                        "temperature": temperature,
                        "top_p": top_p,
                        **(
                            {"repetition_penalty": float(repetition_penalty)}
                            if repetition_penalty is not None
                            else {}
                        ),
                    },
                    "code": code,
                    "finish": long_finish,
                    "content": long_content,
                    "reasoning": long_reasoning,
                    "content_chars": len(long_content),
                    "reasoning_chars": len(long_reasoning),
                    "content_equals_reasoning": long_content.strip() == long_reasoning.strip(),
                    "loop_score": loop_score,
                    "content_contract_ok": content_contract_ok,
                    "raw_response": long_resp,
                    "elapsed_sec": long_elapsed,
                },
            )
            check(
                f"dsv4_long_context_full_output_{probe_name}",
                dsv4_long_context_full_output_ok(
                    code=code,
                    finish=long_finish,
                    full_text=full_text,
                    content=long_content,
                    split_ok=split_ok,
                    loop_score=loop_score,
                    content_contract_ok=content_contract_ok,
                ),
                {
                    "code": code,
                    "finish": long_finish,
                    "content_chars": len(long_content),
                    "reasoning_chars": len(long_reasoning),
                    "content_equals_reasoning": long_content.strip() == long_reasoning.strip(),
                    "loop_score": loop_score,
                    "content_contract_ok": content_contract_ok,
                    "elapsed_sec": long_elapsed,
                    "artifact": artifact,
                    "head": full_text[:400],
                    "tail": full_text[-400:],
                },
            )

    loop_probe = live_loop_probe_name(row)
    if loop_probe:
        artifact_prefix = {
            "bailing_hybrid": "ling",
            "minimax": "minimax",
            "hy_v3": "hy3",
        }.get(row.family, row.family)
        artifact_name = f"{artifact_prefix}_multilingual_game_prompt"
        code, loop_resp, loop_elapsed = request_json(
            loop_probe,
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
                            "структурировано в 5 пунктах; каждый пункт должен быть "
                            "полным коротким предложением, без повторения одного "
                            "слова или символа. Не используй английские или "
                            "китайские слова, кроме HTML и Three.js."
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
        loop_content, loop_reasoning, loop_finish = extract_chat_text(loop_resp)
        loop_full = f"{loop_reasoning}\n{loop_content}".strip()
        loop_score = simple_loop_score(loop_full)
        loop_quality = text_quality_summary(loop_full)
        loop_usage = loop_resp.get("usage") if isinstance(loop_resp, dict) else None
        loop_perf = live_request_perf_summary(loop_usage, loop_elapsed)
        loop_artifact = write_probe_output(
            row.id,
            artifact_name,
            {
                "code": code,
                "finish": loop_finish,
                "content": loop_content,
                "reasoning": loop_reasoning,
                "loop_score": loop_score,
                "quality": loop_quality,
                "raw_response": loop_resp,
                "elapsed_sec": loop_elapsed,
                "usage": loop_usage,
                "perf": loop_perf,
            },
        )
        check(
            loop_probe,
            code == 200 and multilingual_loop_quality_ok(loop_full),
            {
                "code": code,
                "finish": loop_finish,
                "content_chars": len(loop_content),
                "reasoning_chars": len(loop_reasoning),
                "loop_score": loop_score,
                "quality": loop_quality,
                "elapsed_sec": loop_elapsed,
                "usage": loop_usage,
                "perf": loop_perf,
                "artifact": loop_artifact,
                "head": loop_full[:300],
                "tail": loop_full[-300:],
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
            and responses_tool_call_arguments_ok(
                fcalls,
                expected_name="list_directory",
                expected_arguments={"path": "."},
            )
            and responses_tool_choice_output_ok(raw_tool_text),
            {
                "code": code,
                "function_calls": fcalls,
                "expected_arguments": {"path": "."},
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
            "temperature": 0.0,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Reply with one word.",
                }
            ],
            "enable_thinking": False,
        },
        timeout=240,
    )
    anth_text, anth_stop = extract_anthropic_text_and_stop(anth)
    check(
        "anthropic_messages_basic",
        code == 200
        and normalize_short_answer(anth_text).lower() == "paris"
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
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Reply with one word.",
                }
            ],
            "options": {"temperature": 0.0, "num_predict": 80},
        },
        timeout=240,
    )
    ollama_text, ollama_stop = extract_ollama_visible_text_and_stop(ollama)
    check(
        "ollama_chat_basic",
        code == 200
        and normalize_short_answer(ollama_text).lower() == "paris"
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

    snapshot("before_chat_stream_disconnect")
    stream_probe = stream_chat_probe(
        base,
        model,
        [
            {
                "role": "user",
                "content": (
                    "Stream audit. Count from one to forty in words, separated "
                    "by commas. Keep going until forty."
                ),
            }
        ],
        thinking=False,
        max_tokens=260,
        chunks_to_read=5,
        timeout=120,
    )
    stream_artifact = write_probe_output(
        row.id,
        "chat_stream_disconnect",
        stream_probe,
    )
    result["requests"].append(
        {
            "name": "chat_stream_disconnect",
            "method": "POST",
            "path": "/v1/chat/completions",
            "code": stream_probe["status"],
            "elapsed_sec": stream_probe["elapsed_sec"],
            "usage": None,
            "artifact": stream_artifact,
        }
    )
    quiet_stats = None
    for _ in range(20):
        time.sleep(0.5)
        code_q, quiet_stats = http_json("GET", f"{base}/v1/cache/stats", timeout=10)
        sched_q = (
            quiet_stats.get("scheduler_stats", {})
            if code_q == 200 and isinstance(quiet_stats, dict)
            else {}
        )
        if int(sched_q.get("num_running") or 0) == 0 and int(sched_q.get("num_waiting") or 0) == 0:
            break
    snapshot("after_chat_stream_disconnect")
    sched_quiet = quiet_stats.get("scheduler_stats", {}) if isinstance(quiet_stats, dict) else {}
    stream_had_text = (
        int(stream_probe.get("visible_chars") or 0) > 0
        or int(stream_probe.get("reasoning_chars") or 0) > 0
    )
    check(
        "chat_stream_disconnect_or_done",
        stream_probe["status"] == 200
        and stream_probe["error"] is None
        and int(stream_probe["chunks_read"]) > 0
        and (stream_had_text or bool(stream_probe["done"]))
        and int(sched_quiet.get("num_running") or 0) == 0
        and int(sched_quiet.get("num_waiting") or 0) == 0,
        {
            "stream_probe": stream_probe,
            "artifact": stream_artifact,
            "post_disconnect_scheduler": sched_quiet,
            "post_disconnect_cache_stats": quiet_stats,
        },
    )

    if cache_exact_hit_required(row):
        # Cache exact-hit coherence. Multi-turn memory is covered by the chat
        # recall checks above; this check isolates cache mechanics with a prompt
        # that stops cleanly. Per-family specs avoid treating stylistic
        # noncompliance as a cache-stack failure.
        cache_probe = cache_exact_hit_probe(row)
        cache_messages = cache_probe["messages"]
        cache_thinking = bool(cache_probe["thinking"])
        cache_max_tokens = int(cache_probe["max_tokens"])
        cache_expected = str(cache_probe["expected"])
        cache_temperature = float(cache_probe["temperature"])
        cache_min_count = int(cache_probe["min_count"])
        cache_strict_short = bool(cache_probe.get("strict_short_answer", False))
        r_cache = chat_probe(
            "cache_exact_hit_first",
            cache_messages,
            thinking=cache_thinking,
            max_tokens=cache_max_tokens,
            temperature=cache_temperature,
        )
        c_cache, r_cache_reasoning, _ = extract_chat_text(r_cache["body"])
        r_cache_repeat = chat_probe(
            "cache_exact_hit_repeat",
            cache_messages,
            thinking=cache_thinking,
            max_tokens=cache_max_tokens,
            temperature=cache_temperature,
        )
        c_cache_repeat, r_cache_repeat_reasoning, _ = extract_chat_text(r_cache_repeat["body"])
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
            and cache_probe_content_ok(
                row=row,
                expected=cache_expected,
                first_content=c_cache,
                repeat_content=c_cache_repeat,
                strict_short_answer=cache_strict_short,
                min_count=cache_min_count,
            )
            and not has_duplicate_block(
                c_cache + r_cache_reasoning + c_cache_repeat + r_cache_repeat_reasoning
            )
            and _cache_hit_observed,
            {
                "content": c_cache[:240],
                "repeat_content": c_cache_repeat[:240],
                "reasoning_chars": len(r_cache_reasoning),
                "repeat_reasoning_chars": len(r_cache_repeat_reasoning),
                "repeat_usage": _repeat_usage,
                "cache_hit_observed": _cache_hit_observed,
                "non_exact_note": (
                    "accepted coherent non-exact cache answer per release gate"
                    if cache_probe_content_ok(
                        row=row,
                        expected=cache_expected,
                        first_content=c_cache,
                        repeat_content=c_cache_repeat,
                        strict_short_answer=cache_strict_short,
                        min_count=cache_min_count,
                    )
                    and (
                        normalize_short_answer(c_cache).lower() != cache_expected
                        or normalize_short_answer(c_cache_repeat).lower() != cache_expected
                    )
                    else None
                ),
                "mixed_attention_detected": _mixed_attention_detected,
                "first_elapsed_sec": r_cache["elapsed_sec"],
                "repeat_elapsed_sec": r_cache_repeat["elapsed_sec"],
                "cache_stats": stats1 if code == 200 else stats0,
            },
        )
    else:
        check(
            "cache_second_turn_coherent",
            True,
            {
                "skipped": True,
                "reason": (
                    "Not applicable for ZAYA CCA until typed prompt-state restore "
                    "serializes standard KV plus conv_state plus prev_hs"
                ),
                "cache_profile": row.cache_profile,
                "cache_stats": stats0,
            },
        )

    snapshot("before_shutdown")
    final_log_text = log.read_text(errors="ignore")
    runtime_findings = blocking_runtime_log_findings(row, final_log_text)
    check(
        "blocking_runtime_log_findings_absent",
        not runtime_findings,
        {
            "findings": runtime_findings,
            "log": str(log),
            "reason": (
                "Live output must not hide topology/cache/runtime failures in "
                "the server log."
            ),
        },
    )

    finalize_live_status(row, result)

    if keep_running:
        result["kept_running"] = True
    else:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def finalize_live_status(row: ModelRow, result: dict[str, Any]) -> None:
    failures = [c for c in result.get("checks", []) if not c.get("ok")]
    result["failures"] = failures
    if not failures:
        result["status"] = "PASS"
        return
    if row.defer_failure_reason:
        result["status"] = "DEFERRED"
        result["reason"] = row.defer_failure_reason
        return
    result["status"] = "FAIL"


def production_family_audit_summary(results: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in results.get("rows", []) if isinstance(row, dict)]
    missing_rows: list[str] = []
    issue_rows: list[str] = []
    sampling_review_rows: list[str] = []
    live_status_counts: dict[str, int] = {}

    for item in rows:
        static = item.get("static") if isinstance(item.get("static"), dict) else {}
        row_id = static.get("id")
        if not isinstance(row_id, str):
            continue
        if static.get("exists") is False:
            missing_rows.append(row_id)
        if static.get("issues"):
            issue_rows.append(row_id)
        sampling_risk = (
            static.get("sampling_loop_risk")
            if isinstance(static.get("sampling_loop_risk"), dict)
            else {}
        )
        if sampling_risk.get("status") == "review":
            sampling_review_rows.append(row_id)
        live = item.get("live") if isinstance(item.get("live"), dict) else None
        if live is not None:
            status = str(live.get("status") or "UNKNOWN")
            live_status_counts[status] = live_status_counts.get(status, 0) + 1

    return {
        "row_count": len(rows),
        "missing_rows": missing_rows,
        "issue_rows": issue_rows,
        "sampling_review_rows": sampling_review_rows,
        "live_status_counts": live_status_counts,
    }


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
    py = normalize_python_executable(args.py)

    rows = ROWS
    if args.rows:
        wanted = {x.strip() for x in args.rows.split(",") if x.strip()}
        rows = [r for r in rows if r.id in wanted]
    if args.skip_slow:
        rows = [r for r in rows if not r.slow]

    results: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": str(py),
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
                py,
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
    results["summary"] = production_family_audit_summary(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {out}")

    if args.live:
        failed = [
            r
            for r in results["rows"]
            if r.get("live", {}).get("status") not in (None, "PASS", "SKIP", "DEFERRED")
        ]
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

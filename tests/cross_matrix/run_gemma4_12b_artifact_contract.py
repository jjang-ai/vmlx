#!/usr/bin/env python3
"""Validate local Gemma 4 12B artifact sidecars without loading weights.

This gate pins the runtime-sensitive metadata for the shipped and candidate local
Gemma 4 12B bundles:

- OsaurusAI/gemma-4-12B-it-MXFP4
- OsaurusAI/gemma-4-12B-it-MXFP8
- OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE
- OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE
- JANGQ-AI/gemma-4-12B-it-JANG_4M

It intentionally does not import MLX or load tensors. The goal is to catch
sidecar/template/config drift before any live-server proof is interpreted as a
runtime issue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-gemma4-12b-artifact-contract-local.json")

DEFAULT_BUNDLES: dict[str, Path] = {
    "mxfp4": Path("/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP4"),
    "mxfp8": Path("/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8"),
    "mxfp8_attnfp16": Path("/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE"),
    "mxfp8_attnfp16_l024": Path("/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE"),
    "jang4m": Path("/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M"),
}

REQUIRED_SIDECARS = (
    "config.json",
    "jang_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)

EXPECTED_FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47]
EXPECTED_LAYER_TYPES = [
    "full_attention" if idx in EXPECTED_FULL_ATTENTION_LAYERS else "sliding_attention"
    for idx in range(48)
]

EXPECTED_QUANTIZATION: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "weight_format": "mxfp4",
        "method": "mxfp4",
        "mode": "mxfp4",
        "bits": 4,
        "group_size": 32,
    },
    "mxfp8": {
        "weight_format": "mxfp8",
        "method": "mxfp8",
        "mode": "mxfp8",
        "bits": 8,
        "group_size": 32,
    },
    "mxfp8_attnfp16": {
        "weight_format": "mxfp8",
        "method": "mxfp8",
        "mode": "mxfp8",
        "bits": 8,
        "group_size": 32,
        "selective_passthrough": {
            "preserve_attention_fp16": True,
            "preserve_full_attention_fp16": False,
            "preserve_first_layers": 0,
        },
    },
    "mxfp8_attnfp16_l024": {
        "weight_format": "mxfp8",
        "method": "mxfp8",
        "mode": "mxfp8",
        "bits": 8,
        "group_size": 32,
        "selective_passthrough": {
            "preserve_attention_fp16": True,
            "preserve_full_attention_fp16": False,
            "preserve_first_layers": 24,
        },
    },
    "jang4m": {
        "weight_format": "jang_affine",
        "method": "jang_affine",
        "mode": "affine",
        "group_size": 32,
        "tier_bits": {"attention": 8, "mlp": 4, "embed": 16},
        "per_module_override_count": 144,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_equal(
    failures: list[str],
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def _check_true(failures: list[str], label: str, value: Any) -> None:
    if value is not True:
        failures.append(f"{label}: expected true, got {value!r}")


def _check_template(template: str, failures: list[str]) -> None:
    required_fragments = (
        "<|turn>model",
        "<|channel>thought",
        "<|tool_call>call:",
        "<|tool_response>",
        "<|\"|>",
        "tool_choice",
        "required",
        "strip_thinking",
    )
    for fragment in required_fragments:
        if fragment not in template:
            failures.append(f"chat_template.jinja missing {fragment!r}")
    if "<|turn>model\n<|channel>thought" in template:
        failures.append(
            "chat_template.jinja pre-seeds an empty thought channel for normal no-thinking generation"
        )


def _check_config(bundle_name: str, config: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    text = config.get("text_config") or {}
    vision = config.get("vision_config") or {}
    audio = config.get("audio_config") or {}
    rope = text.get("rope_parameters") or {}
    full_rope = rope.get("full_attention") or {}
    sliding_rope = rope.get("sliding_attention") or {}

    _check_equal(failures, "config.model_type", config.get("model_type"), "gemma4_unified")
    if "Gemma4UnifiedForConditionalGeneration" not in (config.get("architectures") or []):
        failures.append("config.architectures missing Gemma4UnifiedForConditionalGeneration")
    _check_equal(
        failures,
        "config.text_config.model_type",
        text.get("model_type"),
        "gemma4_unified_text",
    )
    _check_equal(
        failures,
        "config.vision_config.model_type",
        vision.get("model_type"),
        "gemma4_unified_vision",
    )
    _check_equal(
        failures,
        "config.audio_config.model_type",
        audio.get("model_type"),
        "gemma4_unified_audio",
    )
    _check_equal(failures, "text_config.num_hidden_layers", text.get("num_hidden_layers"), 48)
    _check_equal(failures, "text_config.hidden_size", text.get("hidden_size"), 3840)
    _check_equal(failures, "text_config.intermediate_size", text.get("intermediate_size"), 15360)
    _check_equal(failures, "text_config.num_attention_heads", text.get("num_attention_heads"), 16)
    _check_equal(failures, "text_config.num_key_value_heads", text.get("num_key_value_heads"), 8)
    _check_equal(
        failures,
        "text_config.num_global_key_value_heads",
        text.get("num_global_key_value_heads"),
        1,
    )
    _check_equal(failures, "text_config.head_dim", text.get("head_dim"), 256)
    _check_equal(failures, "text_config.global_head_dim", text.get("global_head_dim"), 512)
    _check_equal(
        failures,
        "text_config.max_position_embeddings",
        text.get("max_position_embeddings"),
        131072,
    )
    _check_equal(failures, "text_config.sliding_window", text.get("sliding_window"), 1024)
    _check_equal(failures, "text_config.layer_types", text.get("layer_types"), EXPECTED_LAYER_TYPES)
    _check_true(failures, "text_config.attention_k_eq_v", text.get("attention_k_eq_v"))
    _check_true(failures, "text_config.tie_word_embeddings", text.get("tie_word_embeddings"))
    _check_equal(failures, "text_config.enable_moe_block", text.get("enable_moe_block"), False)
    _check_equal(failures, "text_config.num_experts", text.get("num_experts"), None)
    _check_equal(failures, "text_config.num_kv_shared_layers", text.get("num_kv_shared_layers"), 0)
    _check_equal(
        failures,
        "text_config.hidden_size_per_layer_input",
        text.get("hidden_size_per_layer_input"),
        0,
    )
    _check_equal(failures, "text_config.use_double_wide_mlp", text.get("use_double_wide_mlp"), False)
    _check_equal(
        failures,
        "text_config.use_bidirectional_attention",
        text.get("use_bidirectional_attention"),
        "vision",
    )
    _check_equal(failures, "full_attention.rope_type", full_rope.get("rope_type"), "proportional")
    _check_equal(
        failures,
        "full_attention.partial_rotary_factor",
        full_rope.get("partial_rotary_factor"),
        0.25,
    )
    _check_equal(failures, "full_attention.rope_theta", full_rope.get("rope_theta"), 1000000.0)
    _check_equal(failures, "sliding_attention.rope_type", sliding_rope.get("rope_type"), "default")
    _check_equal(failures, "sliding_attention.rope_theta", sliding_rope.get("rope_theta"), 10000.0)

    expected = EXPECTED_QUANTIZATION[bundle_name]
    quant = config.get("quantization") or {}
    _check_equal(failures, "config.weight_format", config.get("weight_format"), expected["weight_format"])
    _check_equal(failures, "config.quantization.mode", quant.get("mode"), expected["mode"])
    if expected["method"] in {"mxfp4", "mxfp8"}:
        _check_equal(failures, "config.quantization.bits", quant.get("bits"), expected["bits"])
    _check_equal(
        failures,
        "config.quantization.group_size",
        quant.get("group_size"),
        expected["group_size"],
    )
    if bundle_name in {"mxfp4", "mxfp8"}:
        _check_equal(
            failures,
            "config.quantization.norm_convention",
            quant.get("norm_convention"),
            "gemma4_scale_shift_zero",
        )
        _check_equal(
            failures,
            "config.quantization.multimodal",
            quant.get("multimodal"),
            "fp16_passthrough_embedders_early_fusion",
        )
        _check_equal(failures, "config.quantization.mtp", quant.get("mtp"), "none")
    if "selective_passthrough" in expected:
        selective = quant.get("selective_passthrough") or {}
        for key, expected_value in expected["selective_passthrough"].items():
            _check_equal(
                failures,
                f"config.quantization.selective_passthrough.{key}",
                selective.get(key),
                expected_value,
            )
    if bundle_name == "jang4m":
        override_count = sum(
            1
            for value in quant.values()
            if isinstance(value, dict)
            and value.get("bits") == 4
            and value.get("group_size") == 32
            and value.get("mode") == "affine"
        )
        _check_equal(
            failures,
            "config.quantization.4bit_mlp_override_count",
            override_count,
            expected["per_module_override_count"],
        )

    return {
        "model_type": config.get("model_type"),
        "text_model_type": text.get("model_type"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "full_attention_layers": EXPECTED_FULL_ATTENTION_LAYERS,
        "sliding_window": text.get("sliding_window"),
        "attention_k_eq_v": text.get("attention_k_eq_v"),
        "weight_format": config.get("weight_format"),
        "quantization": {
            key: quant.get(key)
            for key in (
                "method",
                "mode",
                "bits",
                "group_size",
                "tier_bits",
                "norm_convention",
                "multimodal",
                "mtp",
                "selective_passthrough",
            )
            if key in quant
        },
    }


def _check_jang_config(
    bundle_name: str,
    jang_config: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    expected = EXPECTED_QUANTIZATION[bundle_name]
    quant = jang_config.get("quantization") or {}
    runtime = jang_config.get("runtime") or {}
    caps = jang_config.get("capabilities") or {}

    _check_equal(failures, "jang_config.weight_format", jang_config.get("weight_format"), expected["weight_format"])
    _check_equal(failures, "jang_config.quantization.method", quant.get("method"), expected["method"])
    _check_equal(failures, "jang_config.quantization.mode", quant.get("mode"), expected["mode"])
    if "bits" in expected:
        _check_equal(failures, "jang_config.quantization.bits", quant.get("bits"), expected["bits"])
    _check_equal(
        failures,
        "jang_config.quantization.group_size",
        quant.get("group_size"),
        expected["group_size"],
    )
    _check_equal(
        failures,
        "jang_config.quantization.norm_convention",
        quant.get("norm_convention"),
        "gemma4_scale_shift_zero",
    )
    _check_equal(
        failures,
        "jang_config.quantization.multimodal",
        quant.get("multimodal"),
        "fp16_passthrough_embedders_early_fusion",
    )
    if "tier_bits" in expected:
        _check_equal(failures, "jang_config.quantization.tier_bits", quant.get("tier_bits"), expected["tier_bits"])
        _check_equal(
            failures,
            "jang_config.quantization.per_module_override_count",
            quant.get("per_module_override_count"),
            expected["per_module_override_count"],
        )
    if "selective_passthrough" in expected:
        selective = quant.get("selective_passthrough") or {}
        for key, expected_value in expected["selective_passthrough"].items():
            _check_equal(
                failures,
                f"jang_config.quantization.selective_passthrough.{key}",
                selective.get(key),
                expected_value,
            )
    _check_equal(failures, "jang_config.quantization.mtp_policy", quant.get("mtp_policy"), "none")
    _check_true(failures, "jang_config.has_vision", jang_config.get("has_vision"))
    _check_true(failures, "jang_config.has_audio", jang_config.get("has_audio"))
    _check_equal(failures, "jang_config.runtime.attention", runtime.get("attention"), "hybrid_swa_full_5to1")
    _check_equal(failures, "jang_config.runtime.sliding_window", runtime.get("sliding_window"), 1024)
    _check_equal(
        failures,
        "jang_config.runtime.full_attention_layers",
        runtime.get("full_attention_layers"),
        EXPECTED_FULL_ATTENTION_LAYERS,
    )
    _check_true(
        failures,
        "jang_config.runtime.attention_k_eq_v_on_full_layers",
        runtime.get("attention_k_eq_v_on_full_layers"),
    )
    _check_equal(failures, "jang_config.capabilities.family", caps.get("family"), "gemma4")
    _check_equal(failures, "jang_config.capabilities.reasoning_parser", caps.get("reasoning_parser"), "gemma4")
    _check_equal(failures, "jang_config.capabilities.tool_parser", caps.get("tool_parser"), "gemma4")
    _check_equal(failures, "jang_config.capabilities.cache_type", caps.get("cache_type"), "kv")
    _check_equal(failures, "jang_config.capabilities.modality", caps.get("modality"), "vision")

    return {
        "weight_format": jang_config.get("weight_format"),
        "mtp_policy": quant.get("mtp_policy"),
        "has_vision": jang_config.get("has_vision"),
        "has_audio": jang_config.get("has_audio"),
        "runtime": {
            "attention": runtime.get("attention"),
            "sliding_window": runtime.get("sliding_window"),
            "full_attention_layers": runtime.get("full_attention_layers"),
            "attention_k_eq_v_on_full_layers": runtime.get("attention_k_eq_v_on_full_layers"),
        },
        "capabilities": {
            key: caps.get(key)
            for key in (
                "family",
                "reasoning_parser",
                "tool_parser",
                "cache_type",
                "modality",
                "think_in_template",
            )
        },
    }


def _check_generation_config(generation: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    _check_equal(failures, "generation_config.do_sample", generation.get("do_sample"), True)
    _check_equal(failures, "generation_config.temperature", generation.get("temperature"), 1.0)
    _check_equal(failures, "generation_config.top_p", generation.get("top_p"), 0.95)
    _check_equal(failures, "generation_config.top_k", generation.get("top_k"), 64)
    _check_equal(failures, "generation_config.eos_token_id", generation.get("eos_token_id"), [1, 106, 50])
    _check_equal(failures, "generation_config.suppress_tokens", generation.get("suppress_tokens"), [258883, 258882])
    _check_equal(failures, "generation_config.pad_token_id", generation.get("pad_token_id"), 0)
    _check_equal(failures, "generation_config.bos_token_id", generation.get("bos_token_id"), 2)
    return {
        "do_sample": generation.get("do_sample"),
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "eos_token_id": generation.get("eos_token_id"),
        "suppress_tokens": generation.get("suppress_tokens"),
    }


def _check_processor_config(processor: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    _check_equal(failures, "processor_config.processor_class", processor.get("processor_class"), "Gemma4UnifiedProcessor")
    image_processor = processor.get("image_processor") or {}
    video_processor = processor.get("video_processor") or {}
    audio_extractor = processor.get("audio_feature_extractor") or processor.get("feature_extractor") or {}
    _check_equal(failures, "image_processor.patch_size", image_processor.get("patch_size"), 16)
    _check_equal(failures, "image_processor.pooling_kernel_size", image_processor.get("pooling_kernel_size"), 3)
    _check_equal(failures, "image_processor.do_normalize", image_processor.get("do_normalize"), False)
    _check_equal(failures, "image_processor.do_rescale", image_processor.get("do_rescale"), True)
    _check_equal(failures, "video_processor.do_normalize", video_processor.get("do_normalize"), True)
    _check_equal(failures, "audio_feature_extractor.feature_size", audio_extractor.get("feature_size"), 640)
    _check_equal(
        failures,
        "audio_feature_extractor.audio_samples_per_token",
        audio_extractor.get("audio_samples_per_token"),
        640,
    )
    return {
        "processor_class": processor.get("processor_class"),
        "image": {
            "patch_size": image_processor.get("patch_size"),
            "pooling_kernel_size": image_processor.get("pooling_kernel_size"),
            "do_normalize": image_processor.get("do_normalize"),
            "do_rescale": image_processor.get("do_rescale"),
        },
        "audio": {
            "feature_size": audio_extractor.get("feature_size"),
            "audio_samples_per_token": audio_extractor.get("audio_samples_per_token"),
        },
    }


def validate_bundle(bundle_name: str, path: Path) -> dict[str, Any]:
    failures: list[str] = []
    sidecars: dict[str, dict[str, Any]] = {}
    sidecar_hashes: dict[str, str] = {}
    if not path.is_dir():
        return {
            "name": bundle_name,
            "path": str(path),
            "status": "missing",
            "failures": [f"bundle path does not exist: {path}"],
        }

    for rel in REQUIRED_SIDECARS:
        sidecar = path / rel
        if not sidecar.is_file():
            failures.append(f"missing sidecar: {rel}")
            continue
        sidecar_hashes[rel] = _sha256(sidecar)

    if failures:
        return {
            "name": bundle_name,
            "path": str(path),
            "status": "fail",
            "failures": failures,
            "sidecar_hashes": sidecar_hashes,
        }

    config = _load_json(path / "config.json")
    jang_config = _load_json(path / "jang_config.json")
    generation = _load_json(path / "generation_config.json")
    tokenizer_config = _load_json(path / "tokenizer_config.json")
    processor = _load_json(path / "processor_config.json")
    index = _load_json(path / "model.safetensors.index.json")
    template = (path / "chat_template.jinja").read_text(encoding="utf-8")

    config_summary = _check_config(bundle_name, config, failures)
    jang_summary = _check_jang_config(bundle_name, jang_config, failures)
    generation_summary = _check_generation_config(generation, failures)
    processor_summary = _check_processor_config(processor, failures)
    _check_template(template, failures)

    if tokenizer_config.get("chat_template") != template:
        failures.append("tokenizer_config.chat_template does not match chat_template.jinja")
    if not isinstance(index.get("weight_map"), dict) or not index.get("weight_map"):
        failures.append("model.safetensors.index.json has no weight_map")

    return {
        "name": bundle_name,
        "path": str(path),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "sidecar_hashes": sidecar_hashes,
        "config": config_summary,
        "jang_config": jang_summary,
        "generation_config": generation_summary,
        "processor_config": processor_summary,
        "template": {
            "sha256": sidecar_hashes.get("chat_template.jinja"),
            "has_required_tool_choice": "tool_choice" in template and "required" in template,
            "has_no_thinking_prefill_guard": "<|turn>model\n<|channel>thought" not in template,
        },
        "index": {
            "weight_count": len(index.get("weight_map") or {}),
            "metadata": index.get("metadata") or {},
        },
    }


def build_artifact(bundles: dict[str, Path]) -> dict[str, Any]:
    bundle_results = {
        name: validate_bundle(name, path)
        for name, path in bundles.items()
    }
    checks = {
        "all_expected_bundles_present": all(
            result.get("status") != "missing" for result in bundle_results.values()
        )
        and set(bundle_results) == set(DEFAULT_BUNDLES),
        "sidecars_present": all(
            all(rel in (result.get("sidecar_hashes") or {}) for rel in REQUIRED_SIDECARS)
            for result in bundle_results.values()
        ),
        "architecture_pinned": all(result.get("status") == "pass" for result in bundle_results.values()),
        "native_mtp_absent": all(
            ((result.get("jang_config") or {}).get("mtp_policy") == "none")
            for result in bundle_results.values()
        ),
        "gemma4_parsers_declared": all(
            ((result.get("jang_config") or {}).get("capabilities") or {}).get("tool_parser") == "gemma4"
            and ((result.get("jang_config") or {}).get("capabilities") or {}).get("reasoning_parser") == "gemma4"
            for result in bundle_results.values()
        ),
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "bundles": bundle_results,
        "runtime_boundary": {
            "validated_here": "sidecars/config/chat_template/processor/index only",
            "not_validated_here": [
                "live vmlx-engine serve",
                "prefix cache reuse",
                "paged cache reuse",
                "L2 disk cache reuse",
                "TurboQuant KV encoding",
                "continuous batching",
                "native MTP",
                "quantized KV cache",
                "multimodal image/audio/video output quality",
            ],
            "conservative_flags_until_live_rows_pass": [
                "--no-continuous-batching",
                "--disable-prefix-cache",
                "--kv-cache-quantization none",
                "--disable-native-mtp",
            ],
        },
    }


def _parse_bundle_arg(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return DEFAULT_BUNDLES
    bundles: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--bundle must be name=/path, got {value!r}")
        name, raw_path = value.split("=", 1)
        bundles[name] = Path(raw_path)
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        help="Override bundle path as name=/path. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(_parse_bundle_arg(args.bundle))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    for name, result in artifact["bundles"].items():
        print(f"{name}: status={result['status']} failures={len(result.get('failures') or [])}")
        for failure in result.get("failures") or []:
            print(f"  - {failure}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

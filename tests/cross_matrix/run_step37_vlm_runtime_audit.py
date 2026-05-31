#!/usr/bin/env python3
"""Refresh the no-heavy Step-3.7 VLM runtime boundary artifact."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(
    "/Users/example/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
)
DEFAULT_OUT = Path("build/current-step37-vlm-runtime-audit-20260530-source-surface.json")
JANG_RUNTIME_NOTE = Path(
    "/Users/example/jang/docs/runtime/2026-05-29-step37-lfm25-source-and-quant-status.md"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - audit reports malformed files as absent facts
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _has_mlx_vlm_step3p7() -> bool:
    try:
        return importlib.util.find_spec("mlx_vlm.models.step3p7") is not None
    except ModuleNotFoundError:
        return False


def _inspect_mlx_vlm_step3p7_runtime() -> dict[str, Any]:
    """Inspect whether mlx-vlm exposes a real Step3p7 VLM runtime surface."""
    runtime: dict[str, Any] = {
        "importable": False,
        "origin": None,
        "has_model": False,
        "has_vision_model": False,
        "has_language_model": False,
        "has_model_config": False,
        "has_processor": False,
        "available": False,
    }
    try:
        spec = importlib.util.find_spec("mlx_vlm.models.step3p7")
    except ModuleNotFoundError:
        return runtime
    if spec is None:
        return runtime
    runtime["importable"] = True
    runtime["origin"] = spec.origin
    try:
        module = importlib.import_module("mlx_vlm.models.step3p7")
    except Exception as exc:  # noqa: BLE001 - audit records import failure
        runtime["import_error"] = f"{type(exc).__name__}: {exc}"
        return runtime

    runtime["has_model"] = hasattr(module, "Model")
    runtime["has_vision_model"] = hasattr(module, "VisionModel")
    runtime["has_language_model"] = hasattr(module, "LanguageModel")
    runtime["has_model_config"] = hasattr(module, "ModelConfig")
    for processor_module in (
        "mlx_vlm.models.step3p7.processing_step3p7",
        "mlx_vlm.models.step3p7.processing_step3",
    ):
        try:
            if importlib.util.find_spec(processor_module) is not None:
                runtime["has_processor"] = True
                runtime["processor_module"] = processor_module
                break
        except ModuleNotFoundError:
            continue
    runtime["available"] = all(
        bool(runtime[key])
        for key in (
            "has_model",
            "has_vision_model",
            "has_language_model",
            "has_model_config",
            "has_processor",
        )
    )
    return runtime


def _bridge_kind(text: str) -> str:
    lowered = text.lower()
    if (
        "text model but no step3p7 vlm wrapper" in lowered
        and "drops" in lowered
        and "vision tensors" in lowered
        and "vision runtime remains a separate follow-up" in lowered
    ):
        return "text_only"
    if text.strip():
        return "unknown_custom_bridge"
    return "missing"


def _inspect_local_reference_vlm_runtime(model_path: Path) -> dict[str, Any]:
    """Record whether the bundle carries only a PyTorch/HF VLM reference."""
    modeling_path = model_path / "modeling_step3p7.py"
    processing_path = model_path / "processing_step3.py"
    vision_path = model_path / "vision_encoder.py"
    config_path = model_path / "configuration_step3p7.py"
    modeling_text = _read_text(modeling_path)
    processing_text = _read_text(processing_path)
    vision_text = _read_text(vision_path)
    config_text = _read_text(config_path)

    has_torch_dependency = (
        "import torch" in modeling_text
        or "import torch" in processing_text
        or "import torch" in vision_text
        or "torch." in modeling_text
        or "torch." in processing_text
        or "torch." in vision_text
    )
    has_mlx_dependency = (
        "import mlx" in modeling_text
        or "import mlx" in processing_text
        or "import mlx" in vision_text
        or "mlx." in modeling_text
        or "mlx." in processing_text
        or "mlx." in vision_text
    )
    runtime_kind = (
        "torch_reference_not_mlx"
        if has_torch_dependency and not has_mlx_dependency
        else "mlx_reference"
        if has_mlx_dependency
        else "unknown"
    )

    return {
        "modeling_step3p7_path": str(modeling_path),
        "processing_step3_path": str(processing_path),
        "vision_encoder_path": str(vision_path),
        "configuration_step3p7_path": str(config_path),
        "has_modeling_step3p7": modeling_path.exists(),
        "has_processing_step3": processing_path.exists(),
        "has_vision_encoder": vision_path.exists(),
        "has_configuration_step3p7": config_path.exists(),
        "has_torch_dependency": has_torch_dependency,
        "has_mlx_dependency": has_mlx_dependency,
        "has_step3p7_model": "class Step3p7Model" in modeling_text,
        "has_conditional_generation": (
            "class Step3p7ForConditionalGeneration" in modeling_text
        ),
        "has_vision_model": "StepRoboticsVisionEncoder" in modeling_text
        or "class StepRoboticsVisionEncoder" in vision_text,
        "has_multimodal_merge": "merge_multimodal_embeddings" in modeling_text,
        "has_image_placeholder_merge": "image_token_id" in modeling_text
        or "image_placeholder_token_id" in modeling_text,
        "has_step3_vl_processor": "class Step3VLProcessor" in processing_text,
        "has_vision_processor": "class Step3VisionProcessor" in processing_text,
        "has_patch_image_inputs": "patch_pixel_values" in processing_text
        or "patch_pixel_values" in modeling_text,
        "has_2d_vision_rope": "EncoderRope2D" in vision_text,
        "has_full_and_sliding_attention_masks": (
            "create_sliding_window_causal_mask" in modeling_text
            and "sliding_window" in modeling_text
        ),
        "has_qk_norms": (
            "self.q_norm" in modeling_text and "self.k_norm" in modeling_text
        ),
        "has_head_wise_attention_gate": (
            "use_head_wise_attn_gate" in modeling_text
            and "self.g_proj" in modeling_text
        ),
        "has_step3p7_config": "class Step3p7Config" in config_text,
        "runtime_kind": runtime_kind,
    }


def _inspect_step_jangtq_status(jang_config: dict[str, Any]) -> dict[str, Any]:
    quantization = jang_config.get("quantization")
    if not isinstance(quantization, dict):
        quantization = {}
    artifact_format = str(
        jang_config.get("format")
        or jang_config.get("weight_format")
        or quantization.get("format")
        or quantization.get("weight_format")
        or ""
    )
    quantization_method = str(quantization.get("method") or "")
    quantization_profile = str(quantization.get("profile") or "")
    quantization_backend = str(quantization.get("quantization_backend") or "")
    markers = " ".join(
        value.lower()
        for value in (
            artifact_format,
            quantization_method,
            quantization_profile,
            quantization_backend,
        )
        if value
    )
    step_jangtq_available = "jangtq" in markers or "mxtq" in markers
    return {
        "artifact_format": artifact_format or None,
        "quantization_method": quantization_method or None,
        "quantization_profile": quantization_profile or None,
        "quantization_backend": quantization_backend or None,
        "step_jangtq_available": step_jangtq_available,
        "step_jangtq_reason": (
            "step_jangtq_detected"
            if step_jangtq_available
            else "step_jangtq_not_made_current_artifact_is_jang_2l"
        ),
    }


def _step3p7_mlx_vlm_implementation_contract(
    *,
    config: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        vision_config = {}

    capabilities: list[str] = []
    if reference.get("has_patch_image_inputs"):
        capabilities.append("image_patch_processing")
    if reference.get("has_2d_vision_rope") or vision_config.get("use_rope2d") is True:
        capabilities.append("2d_vision_rope")
    if (
        reference.get("has_full_and_sliding_attention_masks")
        or "sliding_attention" in {str(item) for item in text_config.get("layer_types", [])}
    ):
        capabilities.append("full_and_sliding_attention_cache")
    if (
        reference.get("has_head_wise_attention_gate")
        or text_config.get("use_head_wise_attn_gate") is True
    ):
        capabilities.append("head_wise_attention_gate")
    if reference.get("has_qk_norms"):
        capabilities.append("qk_norms")
    if reference.get("has_multimodal_merge") or reference.get("has_image_placeholder_merge"):
        capabilities.append("multimodal_embedding_merge")
    if reference.get("has_vision_model"):
        capabilities.append("vision_encoder")

    return {
        "closest_reference_packages": [
            "mlx_vlm.models.qwen3_vl_moe",
            "mlx_vlm.models.lfm2_vl",
            "mlx_vlm.models.gemma4",
        ],
        "required_module_files": [
            "__init__.py",
            "config.py",
            "language.py",
            "step3p7.py",
            "vision.py",
            "processing_step3p7.py",
        ],
        "must_expose_runtime_symbols": [
            "Model",
            "VisionModel",
            "LanguageModel",
            "ModelConfig",
            "TextConfig",
            "VisionConfig",
        ],
        "required_capabilities": capabilities,
        "explicitly_rejected_clearance_paths": [
            "text_only_bridge",
            "importable_stub",
            "processor_without_patch_pixel_values",
            "runtime_that_drops_vision_tensors",
            "runtime_without_full_and_sliding_attention_cache",
        ],
        "minimum_noheavy_proofs": [
            "mlx_vlm.models.step3p7 import exposes complete runtime symbols",
            "Step3p7 config maps nested text_config and vision_config without dropping vision",
            "sanitize maps language, vision, and projector weights without swallowing vision tensors",
            "processor produces patch_pixel_values and image placeholders required by the model",
        ],
        "minimum_live_proofs_after_noheavy": [
            "source text-only Step3.7 still matches existing text bridge quality",
            "real Electron UI image request loads Step3.7 VLM path",
            "image placeholder count equals projected visual token count",
            "cache telemetry records the correct full/sliding attention family",
            "parser and reasoning text are stripped from visible output",
        ],
    }


def _inspect_source_owned_runtime_progress() -> dict[str, Any]:
    progress: dict[str, Any] = {
        "module": "vmlx_engine.models.step3p7_mlx_vlm",
        "config_layer_available": False,
        "model_layer_available": False,
        "vision_layer_available": False,
        "projector_layer_available": False,
        "processor_layer_available": False,
        "image_embeds_merge_available": False,
        "pixel_values_processor_available": False,
        "vision_patch_embed_available": False,
        "vision_abs_posemb_resize_available": False,
        "vision_transformer_blocks_available": False,
        "vision_downsamplers_available": False,
        "pixel_values_to_projector_available": False,
        "placeholder_projected_token_count_available": False,
        "release_clearance": "still_blocked_partial_runtime_only",
    }
    try:
        module = importlib.import_module("vmlx_engine.models.step3p7_mlx_vlm")
    except Exception as exc:  # noqa: BLE001 - audit records progress, not failure
        progress["import_error"] = f"{type(exc).__name__}: {exc}"
        return progress

    progress["config_layer_available"] = all(
        hasattr(module, name)
        for name in ("ModelConfig", "TextConfig", "VisionConfig")
    )
    progress["model_layer_available"] = hasattr(module, "Model") and hasattr(
        module, "LanguageModel"
    )
    progress["vision_layer_available"] = hasattr(module, "VisionModel")
    progress["projector_layer_available"] = hasattr(module, "Step3p7Projector")
    progress["processor_layer_available"] = hasattr(module, "Step3VLProcessor")
    progress["pixel_values_processor_available"] = hasattr(module, "Step3VisionProcessor")
    progress["vision_downsamplers_available"] = False
    model_config = getattr(module, "ModelConfig", None)
    model = getattr(module, "Model", None)
    if model_config is not None and model is not None:
        try:
            probe_config = model_config.from_dict(
                {
                    "text_config": {
                        "model_type": "step3p5",
                        "hidden_size": 8,
                        "num_hidden_layers": 1,
                        "vocab_size": 16,
                        "num_attention_heads": 2,
                        "num_attention_groups": 1,
                        "head_dim": 4,
                        "intermediate_size": 16,
                    },
                    "vision_config": {
                        "hidden_size": 2,
                        "num_channels": 3,
                        "patch_size": 2,
                        "image_size": 4,
                        "layers": 1,
                        "heads": 1,
                        "use_abs_posemb": False,
                        "use_rope2d": False,
                    },
                    "projector_config": {"hidden_size": 8, "text_hidden_size": 8},
                }
            )
            probe_model = model(probe_config)
            progress["vision_patch_embed_available"] = all(
                hasattr(probe_model.vision_model, name)
                for name in ("conv1", "patch_embed")
            )
            progress["vision_abs_posemb_resize_available"] = hasattr(
                probe_model.vision_model,
                "sample_abs_posemb",
            )
            progress["vision_transformer_blocks_available"] = (
                hasattr(probe_model.vision_model, "transformer")
                and hasattr(probe_model.vision_model.transformer, "resblocks")
                and len(probe_model.vision_model.transformer.resblocks) == 1
            )
            progress["vision_downsamplers_available"] = all(
                hasattr(probe_model.vision_model, name)
                for name in ("vit_downsampler1", "vit_downsampler2")
            )
            import mlx.core as mx

            probe_embeds = probe_model.get_multimodal_embeddings(
                pixel_values=mx.zeros((1, 3, 4, 4)),
                num_patches=[0],
            )
            progress["pixel_values_to_projector_available"] = (
                isinstance(probe_embeds, list)
                and len(probe_embeds) == 1
                and tuple(probe_embeds[0].shape) == (1, 8)
            )

            class _ProbeTokenizer:
                vocab = {
                    "<im_patch>": 101,
                    "<im_start>": 102,
                    "<im_end>": 103,
                    "<patch_start>": 104,
                    "<patch_end>": 105,
                    "<patch_newline>": 106,
                }

                def get_vocab(self) -> dict[str, int]:
                    return self.vocab

                def convert_tokens_to_ids(self, token: str) -> int:
                    return self.vocab[token]

                def __call__(self, texts: list[str]) -> dict[str, list[str]]:
                    return {"input_text": texts}

            from PIL import Image
            import numpy as np

            processor = module.Step3VLProcessor(tokenizer=_ProbeTokenizer())
            image = Image.fromarray(
                np.full((80, 400, 3), 64, dtype=np.uint8),
                mode="RGB",
            )
            batch = processor("wide <im_patch>", images=image, return_tensors="mlx")
            placeholder_count = batch["input_text"][0].count("<im_patch>")
            real_shape_config = model_config.from_dict(
                {
                    "text_config": {
                        "model_type": "step3p5",
                        "hidden_size": 8,
                        "num_hidden_layers": 1,
                        "vocab_size": 256,
                        "num_attention_heads": 2,
                        "num_attention_groups": 1,
                        "head_dim": 4,
                        "intermediate_size": 16,
                    },
                    "vision_config": {
                        "width": 4,
                        "num_channels": 3,
                        "patch_size": 14,
                        "image_size": 728,
                        "layers": 0,
                        "heads": 1,
                        "use_abs_posemb": False,
                        "use_rope2d": False,
                    },
                    "projector_config": {"hidden_size": 16, "text_hidden_size": 8},
                }
            )
            real_shape_model = model(real_shape_config)
            real_shape_embeds = real_shape_model.get_multimodal_embeddings(
                pixel_values=batch["pixel_values"],
                patch_pixel_values=batch["patch_pixel_values"],
                num_patches=batch["num_patches"],
            )
            progress["placeholder_projected_token_count_available"] = (
                isinstance(real_shape_embeds, list)
                and sum(int(embeds.shape[0]) for embeds in real_shape_embeds)
                == placeholder_count
            )
        except Exception as exc:  # noqa: BLE001 - audit records progress only
            progress["vision_downsampler_error"] = f"{type(exc).__name__}: {exc}"
    model = getattr(module, "Model", None)
    progress["image_embeds_merge_available"] = all(
        hasattr(model, name)
        for name in (
            "get_multimodal_embeddings",
            "get_input_embeddings",
            "merge_multimodal_embeddings",
        )
    )
    if all(
        progress[key]
        for key in (
            "config_layer_available",
            "model_layer_available",
            "vision_layer_available",
            "projector_layer_available",
            "processor_layer_available",
            "image_embeds_merge_available",
            "pixel_values_processor_available",
            "vision_patch_embed_available",
            "vision_abs_posemb_resize_available",
            "vision_transformer_blocks_available",
            "vision_downsamplers_available",
            "pixel_values_to_projector_available",
            "placeholder_projected_token_count_available",
        )
    ):
        progress["release_clearance"] = "source_runtime_surface_present_needs_live_proof"
    return progress


def build_audit(
    *,
    model_path: Path,
    mlx_vlm_step3p7_importable: bool | None = None,
    mlx_vlm_step3p7_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _read_json(model_path / "config.json")
    jang_config = _read_json(model_path / "jang_config.json")
    bridge_path = model_path / "step3p7_mlx.py"
    bridge_text = _read_text(bridge_path)
    runtime_note_text = _read_text(JANG_RUNTIME_NOTE)
    local_reference_vlm_runtime = _inspect_local_reference_vlm_runtime(model_path)
    step_jangtq_status = _inspect_step_jangtq_status(jang_config)
    mlx_vlm_implementation_contract = _step3p7_mlx_vlm_implementation_contract(
        config=config,
        reference=local_reference_vlm_runtime,
    )
    source_owned_runtime_progress = _inspect_source_owned_runtime_progress()
    try:
        from vmlx_engine.models import mllm

        mllm._register_local_mlx_vlm_runtime_if_needed(model_path)
    except Exception as exc:  # noqa: BLE001 - audit records runtime state below
        source_owned_runtime_progress["registration_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    if mlx_vlm_step3p7_runtime is None:
        if mlx_vlm_step3p7_importable is None:
            mlx_vlm_step3p7_runtime = _inspect_mlx_vlm_step3p7_runtime()
        else:
            mlx_vlm_step3p7_runtime = {
                "importable": mlx_vlm_step3p7_importable,
                "available": mlx_vlm_step3p7_importable,
            }
    mlx_vlm_step3p7_importable = bool(
        mlx_vlm_step3p7_runtime.get("importable")
    )
    mlx_vlm_step3p7_runtime_available = bool(
        mlx_vlm_step3p7_runtime.get("available")
    )

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    vision_config = config.get("vision_config")
    architecture = jang_config.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}

    model_type = str(config.get("model_type") or "")
    text_model_type = str(text_config.get("model_type") or "")
    has_vision_config = isinstance(vision_config, dict) and bool(vision_config)
    jang_has_vision = architecture.get("has_vision") is True
    local_bridge_kind = _bridge_kind(bridge_text)
    runtime_note_confirms_open_vlm = (
        "requires a Step3p7/Step3p5 runtime path" in runtime_note_text
        and "image patch handling" in runtime_note_text
        and "text-only" in runtime_note_text
    )

    runtime_missing = (
        model_type == "step3p7"
        and text_model_type == "step3p5"
        and has_vision_config
        and jang_has_vision
        and local_bridge_kind == "text_only"
        and not mlx_vlm_step3p7_runtime_available
    )

    if runtime_missing:
        status = "open"
        root_cause = "missing_mlx_vlm_step3p7_vlm_runtime"
        if mlx_vlm_step3p7_importable:
            failure_summary = (
                "mlx_vlm.models.step3p7 is importable but only exposes a stub or "
                "incomplete runtime surface; local step3p7_mlx.py bridge is "
                "text-only and drops vision tensors."
            )
        else:
            failure_summary = (
                "No module named 'mlx_vlm.models.step3p7'; local step3p7_mlx.py "
                "bridge is text-only and drops vision tensors."
            )
        release_clearance = "blocked_until_real_step3p7_vlm_runtime"
    else:
        status = "pass"
        root_cause = "step37_vlm_runtime_present_or_not_applicable"
        failure_summary = ""
        release_clearance = "audit_does_not_block_release"

    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "model_type": model_type,
        "architectures": config.get("architectures") or [],
        "text_model_type": text_model_type,
        "has_vision_config": has_vision_config,
        "image_token_id": config.get("image_token_id"),
        "jang_has_vision": jang_has_vision,
        "mlx_vlm_step3p7_importable": mlx_vlm_step3p7_importable,
        "mlx_vlm_step3p7_runtime_available": mlx_vlm_step3p7_runtime_available,
        "mlx_vlm_step3p7_runtime": mlx_vlm_step3p7_runtime,
        "local_bridge_path": str(bridge_path),
        "local_bridge_exists": bridge_path.exists(),
        "local_bridge_kind": local_bridge_kind,
        "local_reference_vlm_runtime": local_reference_vlm_runtime,
        "step_jangtq_status": step_jangtq_status,
        "mlx_vlm_implementation_contract": mlx_vlm_implementation_contract,
        "source_owned_runtime_progress": source_owned_runtime_progress,
        "runtime_note": str(JANG_RUNTIME_NOTE),
        "runtime_note_confirms_open_vlm": runtime_note_confirms_open_vlm,
        "root_cause": root_cause,
        "failure_summary": failure_summary,
        "release_clearance": release_clearance,
        "next_proof": (
            "Implement and prove a real Step3p7 VLM runtime path in vMLX/MLX-VLM "
            "covering image patch handling, full+sliding KV cache, q/k norms, "
            "K/V scale sidecars, parser stripping, and real Electron UI image proof."
        ),
        "commands": {
            "audit": (
                ".venv/bin/python tests/cross_matrix/"
                f"run_step37_vlm_runtime_audit.py --out {DEFAULT_OUT}"
            )
        },
    }


def write_audit(*, model_path: Path, out_path: Path) -> dict[str, Any]:
    payload = build_audit(model_path=model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = write_audit(model_path=args.model_path, out_path=args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": payload["status"],
                "root_cause": payload["root_cause"],
                "release_clearance": payload["release_clearance"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

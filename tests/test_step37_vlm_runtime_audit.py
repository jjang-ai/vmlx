# SPDX-License-Identifier: Apache-2.0
"""Contracts for the local Step-3.7 VLM runtime audit."""

import json
from pathlib import Path

from tests.cross_matrix import run_step37_vlm_runtime_audit as audit


def test_step37_vlm_runtime_audit_default_out_tracks_current_release_artifact():
    from tests.cross_matrix import release_regression_manifest

    assert audit.DEFAULT_OUT == Path(
        "build/current-step37-vlm-runtime-audit-after-gemma4-vl-refresh-20260606.json"
    )
    assert audit.DEFAULT_OUT == Path(
        release_regression_manifest.CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    )


def test_step37_vlm_runtime_audit_records_missing_mlx_vlm_runtime(tmp_path):
    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "architectures": ["Step3p7ForConditionalGeneration"],
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"model_type": "perception_encoder"},
                "image_token_id": 128001,
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "quantization": {
                    "method": "jang-importance",
                    "profile": "JANG_2L",
                    "quantization_backend": "mx.quantize",
                },
                "architecture": {"has_vision": True},
            }
        ),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "The current MLX runtime has a Step3p5 text model but no Step3p7 VLM wrapper.\n"
        "loads the nested text_config through mlx_lm.models.step3p5 and drops\n"
        "vision tensors.\n"
        "Vision runtime remains a separate follow-up.\n",
        encoding="utf-8",
    )

    payload = audit.build_audit(
        model_path=model,
        mlx_vlm_step3p7_importable=False,
    )

    assert payload["status"] == "open"
    assert payload["root_cause"] == "missing_mlx_vlm_step3p7_vlm_runtime"
    assert payload["model_type"] == "step3p7"
    assert payload["text_model_type"] == "step3p5"
    assert payload["has_vision_config"] is True
    assert payload["jang_has_vision"] is True
    assert payload["local_bridge_kind"] == "text_only"
    assert payload["step_jangtq_status"] == {
        "artifact_format": "jang",
        "quantization_method": "jang-importance",
        "quantization_profile": "JANG_2L",
        "quantization_backend": "mx.quantize",
        "step_jangtq_available": False,
        "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
    }
    assert payload["mlx_vlm_step3p7_importable"] is False
    assert "No module named 'mlx_vlm.models.step3p7'" in payload["failure_summary"]
    assert payload["release_clearance"] == "blocked_until_real_step3p7_vlm_runtime"
    progress = payload["source_owned_runtime_progress"]
    assert progress["config_layer_available"] is True
    assert progress["model_layer_available"] is True
    assert progress["vision_layer_available"] is True
    assert progress["projector_layer_available"] is True
    assert progress["processor_layer_available"] is True
    assert progress["image_embeds_merge_available"] is True
    assert progress["pixel_values_processor_available"] is True
    assert progress["vision_patch_embed_available"] is True
    assert progress["vision_abs_posemb_resize_available"] is True
    assert progress["vision_transformer_blocks_available"] is True
    assert progress["vision_downsamplers_available"] is True
    assert progress["pixel_values_to_projector_available"] is True
    assert progress["placeholder_projected_token_count_available"] is True
    assert progress["release_clearance"] == "source_runtime_surface_present_needs_live_proof"


def test_step37_vlm_runtime_audit_uses_source_registered_mlx_vlm_runtime(tmp_path):
    import sys

    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"model_type": "perception_encoder"},
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps({"architecture": {"has_vision": True}}),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "text model but no Step3p7 VLM wrapper. drops vision tensors. "
        "Vision runtime remains a separate follow-up.",
        encoding="utf-8",
    )

    for name in (
        "mlx_vlm.models.step3p7.processing_step3",
        "mlx_vlm.models.step3p7",
    ):
        sys.modules.pop(name, None)

    payload = audit.build_audit(model_path=model)

    assert payload["mlx_vlm_step3p7_importable"] is True
    assert payload["mlx_vlm_step3p7_runtime_available"] is True
    assert payload["status"] == "pass"
    assert payload["release_clearance"] == "audit_does_not_block_release"

    sys.modules.pop("mlx_vlm.models.step3p7.processing_step3", None)
    sys.modules.pop("mlx_vlm.models.step3p7", None)


def test_step37_vlm_runtime_audit_rejects_importable_stub_runtime(tmp_path):
    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "architectures": ["Step3p7ForConditionalGeneration"],
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"model_type": "perception_encoder"},
                "image_token_id": 128001,
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps({"architecture": {"has_vision": True}}),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "The current MLX runtime has a Step3p5 text model but no Step3p7 VLM wrapper.\n"
        "loads the nested text_config through mlx_lm.models.step3p5 and drops\n"
        "vision tensors.\n"
        "Vision runtime remains a separate follow-up.\n",
        encoding="utf-8",
    )

    payload = audit.build_audit(
        model_path=model,
        mlx_vlm_step3p7_runtime={
            "importable": True,
            "has_model": False,
            "has_vision_model": False,
            "has_model_config": False,
            "has_processor": False,
        },
    )

    assert payload["status"] == "open"
    assert payload["mlx_vlm_step3p7_importable"] is True
    assert payload["mlx_vlm_step3p7_runtime_available"] is False
    assert payload["root_cause"] == "missing_mlx_vlm_step3p7_vlm_runtime"
    assert "stub" in payload["failure_summary"]


def test_step37_vlm_runtime_audit_records_local_reference_vlm_surface(tmp_path):
    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "architectures": ["Step3p7ForConditionalGeneration"],
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"model_type": "perception_encoder"},
                "image_token_id": 128001,
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps({"architecture": {"has_vision": True}}),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "The current MLX runtime has a Step3p5 text model but no Step3p7 VLM wrapper.\n"
        "loads the nested text_config through mlx_lm.models.step3p5 and drops\n"
        "vision tensors.\n"
        "Vision runtime remains a separate follow-up.\n",
        encoding="utf-8",
    )
    (model / "modeling_step3p7.py").write_text(
        "import torch\n"
        "from transformers.masking_utils import create_sliding_window_causal_mask\n"
        "from .vision_encoder import StepRoboticsVisionEncoder\n"
        "def merge_multimodal_embeddings(): pass\n"
        "class Step3p7Model: pass\n"
        "class Step3p7ForConditionalGeneration: pass\n"
        "self.q_norm = None\n"
        "self.k_norm = None\n"
        "use_head_wise_attn_gate = True\n"
        "self.g_proj = None\n"
        "sliding_window = 512\n"
        "image_token_id = 128001\n",
        encoding="utf-8",
    )
    (model / "processing_step3.py").write_text(
        "from transformers.processing_utils import ProcessorMixin\n"
        "class Step3VisionProcessor: pass\n"
        "class Step3VLProcessor(ProcessorMixin): pass\n"
        "image_token = '<im_patch>'\n"
        "patch_pixel_values = None\n",
        encoding="utf-8",
    )
    (model / "vision_encoder.py").write_text(
        "class EncoderRope2D: pass\n"
        "class StepRoboticsVisionEncoder: pass\n",
        encoding="utf-8",
    )
    (model / "configuration_step3p7.py").write_text(
        "class Step3p7Config: pass\n",
        encoding="utf-8",
    )

    payload = audit.build_audit(
        model_path=model,
        mlx_vlm_step3p7_importable=False,
    )

    reference = payload["local_reference_vlm_runtime"]
    assert reference["has_modeling_step3p7"] is True
    assert reference["has_processing_step3"] is True
    assert reference["has_vision_encoder"] is True
    assert reference["has_configuration_step3p7"] is True
    assert reference["has_torch_dependency"] is True
    assert reference["has_step3p7_model"] is True
    assert reference["has_conditional_generation"] is True
    assert reference["has_step3_vl_processor"] is True
    assert reference["has_multimodal_merge"] is True
    assert reference["has_patch_image_inputs"] is True
    assert reference["has_2d_vision_rope"] is True
    assert reference["has_full_and_sliding_attention_masks"] is True
    assert reference["has_qk_norms"] is True
    assert reference["has_head_wise_attention_gate"] is True
    assert reference["has_step3p7_config"] is True
    assert reference["runtime_kind"] == "torch_reference_not_mlx"


def test_step37_vlm_runtime_audit_records_real_mlx_vlm_port_contract(tmp_path):
    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "architectures": ["Step3p7ForConditionalGeneration"],
                "text_config": {
                    "model_type": "step3p5",
                    "layer_types": ["full_attention", "sliding_attention"],
                    "use_head_wise_attn_gate": True,
                },
                "vision_config": {
                    "model_type": "perception_encoder",
                    "use_rope2d": True,
                },
                "image_token_id": 128001,
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps({"format": "jang", "architecture": {"has_vision": True}}),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "The current MLX runtime has a Step3p5 text model but no Step3p7 VLM wrapper.\n"
        "loads the nested text_config through mlx_lm.models.step3p5 and drops\n"
        "vision tensors.\n"
        "Vision runtime remains a separate follow-up.\n",
        encoding="utf-8",
    )
    (model / "modeling_step3p7.py").write_text(
        "import torch\n"
        "from transformers.masking_utils import create_sliding_window_causal_mask\n"
        "from .vision_encoder import StepRoboticsVisionEncoder\n"
        "def merge_multimodal_embeddings(): pass\n"
        "class Step3p7Model: pass\n"
        "class Step3p7ForConditionalGeneration: pass\n"
        "self.q_norm = None\n"
        "self.k_norm = None\n"
        "use_head_wise_attn_gate = True\n"
        "self.g_proj = None\n"
        "sliding_window = 512\n"
        "image_token_id = 128001\n",
        encoding="utf-8",
    )
    (model / "processing_step3.py").write_text(
        "class Step3VLProcessor: pass\npatch_pixel_values = None\n",
        encoding="utf-8",
    )
    (model / "vision_encoder.py").write_text(
        "class EncoderRope2D: pass\n"
        "class StepRoboticsVisionEncoder: pass\n",
        encoding="utf-8",
    )
    (model / "configuration_step3p7.py").write_text(
        "class Step3p7Config: pass\n",
        encoding="utf-8",
    )

    payload = audit.build_audit(
        model_path=model,
        mlx_vlm_step3p7_importable=False,
    )

    contract = payload["mlx_vlm_implementation_contract"]
    assert contract["closest_reference_packages"] == [
        "mlx_vlm.models.qwen3_vl_moe",
        "mlx_vlm.models.lfm2_vl",
        "mlx_vlm.models.gemma4",
    ]
    assert contract["required_module_files"] == [
        "__init__.py",
        "config.py",
        "language.py",
        "step3p7.py",
        "vision.py",
        "processing_step3p7.py",
    ]
    assert contract["must_expose_runtime_symbols"] == [
        "Model",
        "VisionModel",
        "LanguageModel",
        "ModelConfig",
        "TextConfig",
        "VisionConfig",
    ]
    assert "image_patch_processing" in contract["required_capabilities"]
    assert "2d_vision_rope" in contract["required_capabilities"]
    assert "full_and_sliding_attention_cache" in contract["required_capabilities"]
    assert "head_wise_attention_gate" in contract["required_capabilities"]
    assert "multimodal_embedding_merge" in contract["required_capabilities"]
    assert "text_only_bridge" in contract["explicitly_rejected_clearance_paths"]
    assert "importable_stub" in contract["explicitly_rejected_clearance_paths"]
    assert contract["minimum_noheavy_proofs"] == [
        "mlx_vlm.models.step3p7 import exposes complete runtime symbols",
        "Step3p7 config maps nested text_config and vision_config without dropping vision",
        "sanitize maps language, vision, and projector weights without swallowing vision tensors",
        "processor produces patch_pixel_values and image placeholders required by the model",
    ]


def test_step37_vlm_runtime_audit_writes_json(tmp_path):
    model = tmp_path / "Step-3.7-Flash-JANG_2L"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"model_type": "perception_encoder"},
            }
        ),
        encoding="utf-8",
    )
    (model / "jang_config.json").write_text(
        json.dumps({"architecture": {"has_vision": True}}),
        encoding="utf-8",
    )
    (model / "step3p7_mlx.py").write_text(
        "text model but no Step3p7 VLM wrapper. drops vision tensors. "
        "Vision runtime remains a separate follow-up.",
        encoding="utf-8",
    )
    out = tmp_path / "audit.json"

    payload = audit.write_audit(model_path=model, out_path=out)

    assert payload["status"] == "pass"
    assert payload["mlx_vlm_step3p7_runtime_available"] is True
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"root_cause": "step37_vlm_runtime_present_or_not_applicable"' in text

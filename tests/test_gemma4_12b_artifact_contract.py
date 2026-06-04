import copy
import json
from pathlib import Path

from tests.cross_matrix import run_gemma4_12b_artifact_contract as gate


def _base_config() -> dict:
    layer_types = [
        "full_attention" if idx in gate.EXPECTED_FULL_ATTENTION_LAYERS else "sliding_attention"
        for idx in range(48)
    ]
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "vision_config": {"model_type": "gemma4_unified_vision"},
        "audio_config": {"model_type": "gemma4_unified_audio"},
        "text_config": {
            "model_type": "gemma4_unified_text",
            "num_hidden_layers": 48,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "max_position_embeddings": 131072,
            "sliding_window": 1024,
            "layer_types": layer_types,
            "attention_k_eq_v": True,
            "tie_word_embeddings": True,
            "enable_moe_block": False,
            "num_experts": None,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": False,
            "use_bidirectional_attention": "vision",
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
            },
        },
        "weight_format": "mxfp4",
        "quantization": {
            "method": "mxfp4",
            "mode": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "norm_convention": "gemma4_scale_shift_zero",
            "multimodal": "fp16_passthrough_embedders_early_fusion",
            "mtp": "none",
        },
    }


def _base_jang_config() -> dict:
    return {
        "weight_format": "mxfp4",
        "mtp_policy": "none",
        "has_vision": True,
        "has_audio": True,
        "runtime": {
            "attention": "hybrid_swa_full_5to1",
            "sliding_window": 1024,
            "full_attention_layers": gate.EXPECTED_FULL_ATTENTION_LAYERS,
            "attention_k_eq_v_on_full_layers": True,
        },
        "capabilities": {
            "family": "gemma4",
            "reasoning_parser": "gemma4",
            "tool_parser": "gemma4",
            "cache_type": "kv",
            "modality": "vision",
            "think_in_template": False,
        },
        "quantization": {
            "method": "mxfp4",
            "mode": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "norm_convention": "gemma4_scale_shift_zero",
            "multimodal": "fp16_passthrough_embedders_early_fusion",
            "mtp_policy": "none",
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_bundle(root: Path, name: str = "mxfp4") -> Path:
    path = root / name
    path.mkdir()
    config = _base_config()
    jang_config = _base_jang_config()
    if name in {"mxfp8", "mxfp8_attnfp16", "mxfp8_attnfp16_l024"}:
        config["weight_format"] = "mxfp8"
        config["quantization"].update({"method": "mxfp8", "mode": "mxfp8", "bits": 8})
        jang_config["weight_format"] = "mxfp8"
        jang_config["quantization"].update({"method": "mxfp8", "mode": "mxfp8", "bits": 8})
        if name in {"mxfp8_attnfp16", "mxfp8_attnfp16_l024"}:
            selective_passthrough = {
                "preserve_attention_fp16": True,
                "preserve_full_attention_fp16": False,
                "preserve_first_layers": 24 if name == "mxfp8_attnfp16_l024" else 0,
            }
            config["quantization"]["selective_passthrough"] = selective_passthrough
            jang_config["quantization"]["selective_passthrough"] = selective_passthrough
    elif name == "jang4m":
        tier_bits = {"attention": 8, "mlp": 4, "embed": 16}
        config["weight_format"] = "jang_affine"
        config["quantization"].pop("bits")
        config["quantization"].update(
            {
                "method": "jang_affine",
                "mode": "affine",
                "tier_bits": tier_bits,
                "per_module_override_count": 144,
            }
        )
        for idx in range(48):
            for proj in ("down_proj", "gate_proj", "up_proj"):
                config["quantization"][f"language_model.model.layers.{idx}.mlp.{proj}"] = {
                    "group_size": 32,
                    "bits": 4,
                    "mode": "affine",
                }
        jang_config["weight_format"] = "jang_affine"
        jang_config["quantization"].pop("bits")
        jang_config["quantization"].update(
            {
                "method": "jang_affine",
                "mode": "affine",
                "tier_bits": tier_bits,
                "per_module_override_count": 144,
            }
        )

    template = (
        "{% macro strip_thinking(x) %}{{ x }}{% endmacro %}"
        "{% if tool_choice == 'required' %}required{% endif %}"
        "<|turn>model\n"
        "{{ '<|channel>thought' }}\n"
        "<|tool_call>call:name{arg:<|\"|>v<|\"|>}<tool_call|>"
        "<|tool_response>response:name{}<tool_response|>"
    )

    _write_json(path / "config.json", config)
    _write_json(path / "jang_config.json", jang_config)
    _write_json(
        path / "generation_config.json",
        {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "eos_token_id": [1, 106, 50],
            "suppress_tokens": [258883, 258882],
            "pad_token_id": 0,
            "bos_token_id": 2,
        },
    )
    _write_json(
        path / "processor_config.json",
        {
            "processor_class": "Gemma4UnifiedProcessor",
            "image_processor": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "do_normalize": False,
                "do_rescale": True,
            },
            "video_processor": {"do_normalize": True},
            "feature_extractor": {
                "feature_size": 640,
                "audio_samples_per_token": 640,
            },
        },
    )
    _write_json(path / "tokenizer_config.json", {"chat_template": template})
    _write_json(path / "model.safetensors.index.json", {"weight_map": {"a": "model-00001.safetensors"}})
    (path / "chat_template.jinja").write_text(template, encoding="utf-8")
    return path


def test_valid_five_bundle_contract_passes(tmp_path):
    bundles = {
        "mxfp4": _make_bundle(tmp_path, "mxfp4"),
        "mxfp8": _make_bundle(tmp_path, "mxfp8"),
        "mxfp8_attnfp16": _make_bundle(tmp_path, "mxfp8_attnfp16"),
        "mxfp8_attnfp16_l024": _make_bundle(tmp_path, "mxfp8_attnfp16_l024"),
        "jang4m": _make_bundle(tmp_path, "jang4m"),
    }

    artifact = gate.build_artifact(bundles)

    assert artifact["status"] == "pass"
    assert artifact["checks"]["all_expected_bundles_present"] is True
    assert artifact["checks"]["native_mtp_absent"] is True


def test_contract_fails_on_wrong_architecture(tmp_path):
    path = _make_bundle(tmp_path, "mxfp4")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "gemma4"
    _write_json(config_path, config)

    result = gate.validate_bundle("mxfp4", path)

    assert result["status"] == "fail"
    assert any("config.model_type" in failure for failure in result["failures"])


def test_contract_fails_on_missing_sidecar(tmp_path):
    path = _make_bundle(tmp_path, "mxfp4")
    (path / "processor_config.json").unlink()

    result = gate.validate_bundle("mxfp4", path)

    assert result["status"] == "fail"
    assert "missing sidecar: processor_config.json" in result["failures"]


def test_contract_fails_if_jang4m_enables_native_mtp(tmp_path):
    path = _make_bundle(tmp_path, "jang4m")
    jang_path = path / "jang_config.json"
    jang_config = json.loads(jang_path.read_text())
    jang_config["quantization"]["mtp_policy"] = "native"
    _write_json(jang_path, jang_config)

    result = gate.validate_bundle("jang4m", path)

    assert result["status"] == "fail"
    assert any("mtp_policy" in failure for failure in result["failures"])


def test_contract_fails_if_jang4m_tier_bits_drift(tmp_path):
    path = _make_bundle(tmp_path, "jang4m")
    jang_path = path / "jang_config.json"
    jang_config = json.loads(jang_path.read_text())
    jang_config["quantization"] = copy.deepcopy(jang_config["quantization"])
    jang_config["quantization"]["tier_bits"]["mlp"] = 8
    _write_json(jang_path, jang_config)

    result = gate.validate_bundle("jang4m", path)

    assert result["status"] == "fail"
    assert any("tier_bits" in failure for failure in result["failures"])


def test_contract_fails_if_mxfp_bits_drift(tmp_path):
    path = _make_bundle(tmp_path, "mxfp8")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"]["bits"] = 4
    _write_json(config_path, config)

    result = gate.validate_bundle("mxfp8", path)

    assert result["status"] == "fail"
    assert any("config.quantization.bits" in failure for failure in result["failures"])

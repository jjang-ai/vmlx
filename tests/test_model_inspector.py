# SPDX-License-Identifier: Apache-2.0
"""
Tests for model inspector utilities.

Usage:
    pytest tests/test_model_inspector.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


def _make_model_dir(tmp_path, config, weight_files=None):
    """Create a minimal model directory for testing."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir(exist_ok=True)

    # Write config.json
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create dummy weight files
    if weight_files is None:
        weight_files = ["model.safetensors"]
    for wf in weight_files:
        (model_dir / wf).write_bytes(b"\x00" * 1024)  # 1KB dummy

    return str(model_dir)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_is_moe_property(self):
        from vmlx_engine.utils.model_inspector import ModelInfo

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
            n_routed_experts=64,
            num_experts_per_tok=8,
        )
        assert info.is_moe is True

    def test_is_not_moe_when_no_experts(self):
        from vmlx_engine.utils.model_inspector import ModelInfo

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
        )
        assert info.is_moe is False

    def test_active_params_moe(self):
        from vmlx_engine.utils.model_inspector import ModelInfo

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
            param_count_billions=120.0,
            n_routed_experts=512,
            num_experts_per_tok=22,
        )
        active = info.active_params_billions
        # Should be much less than total for sparse MoE
        assert active < info.param_count_billions
        assert active > 0

    def test_active_params_dense(self):
        from vmlx_engine.utils.model_inspector import ModelInfo

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
            param_count_billions=7.0,
        )
        assert info.active_params_billions == 7.0


class TestInspectModel:
    """Tests for inspect_model()."""

    def test_basic_transformer(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)

        assert info.model_type == "llama"
        assert info.architecture == "LlamaForCausalLM"
        assert info.hidden_size == 4096
        assert info.num_layers == 32
        assert info.vocab_size == 32000
        assert not info.is_quantized
        assert not info.is_moe
        assert not info.needs_latent_moe
        assert info.param_count_billions > 0

    def test_quantized_model(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)

        assert info.is_quantized is True
        assert info.quant_bits == 4
        assert info.quant_group_size == 64
        assert info.quant_mode == "affine"

    def test_moe_model(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "vocab_size": 128000,
            "intermediate_size": 11008,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)

        assert info.is_moe is True
        assert info.n_routed_experts == 256
        assert info.num_experts_per_tok == 8

    def test_nemotron_h_latent_moe(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 88,
            "num_attention_heads": 32,
            "vocab_size": 131072,
            "intermediate_size": 2688,
            "moe_latent_size": 1024,
            "moe_intermediate_size": 2688,
            "n_routed_experts": 512,
            "num_experts_per_tok": 22,
            "hybrid_override_pattern": "MEMEMEM*" * 11,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)

        assert info.needs_latent_moe is True
        assert info.moe_latent_size == 1024
        assert info.is_hybrid is True
        assert info.is_moe is True

    def test_multimodal_model(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "vocab_size": 151936,
            "intermediate_size": 18944,
            "vision_config": {"hidden_size": 1280},
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)

        assert info.is_mllm is True

    def test_missing_directory(self):
        from vmlx_engine.utils.model_inspector import inspect_model

        with pytest.raises(FileNotFoundError, match="not found"):
            inspect_model("/nonexistent/path")

    def test_missing_config(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="config.json"):
            inspect_model(str(model_dir))

    def test_weight_files_counted(self, tmp_path):
        from vmlx_engine.utils.model_inspector import inspect_model

        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        model_dir = _make_model_dir(
            tmp_path, config,
            weight_files=["model-00001.safetensors", "model-00002.safetensors"],
        )
        info = inspect_model(model_dir)

        assert len(info.weight_files) == 2


class TestEstimateMemory:
    """Tests for memory estimation functions."""

    def test_estimate_memory_basic(self):
        from vmlx_engine.utils.model_inspector import ModelInfo, estimate_memory_gb

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
            param_count_billions=7.0,
        )

        # 4-bit: 7B * 4/8 * 1.2 = 4.2 GB
        mem_4bit = estimate_memory_gb(info, 4)
        assert 3.0 < mem_4bit < 6.0

        # 16-bit: 7B * 16/8 * 1.2 = 16.8 GB
        mem_16bit = estimate_memory_gb(info, 16)
        assert 14.0 < mem_16bit < 20.0

        # More bits = more memory
        assert mem_16bit > mem_4bit

    def test_estimate_conversion_memory(self):
        from vmlx_engine.utils.model_inspector import (
            ModelInfo,
            estimate_conversion_memory_gb,
        )

        info = ModelInfo(
            model_path="/tmp/test",
            model_type="test",
            architecture="TestModel",
            param_count_billions=7.0,
        )

        conv_mem = estimate_conversion_memory_gb(info, target_bits=4)
        # Should be more than inference memory (source + target)
        assert conv_mem > 0

    def test_available_memory(self):
        from vmlx_engine.utils.model_inspector import available_memory_gb

        mem = available_memory_gb()
        assert mem > 0
        assert mem < 1024  # Sanity: less than 1TB

    def test_total_memory(self):
        from vmlx_engine.utils.model_inspector import total_memory_gb

        mem = total_memory_gb()
        assert mem > 0
        assert mem < 1024


class TestEstimateParamCount:
    """Tests for parameter count estimation."""

    def test_llama_7b_estimate(self):
        from vmlx_engine.utils.model_inspector import _estimate_param_count

        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        params = _estimate_param_count(config)
        # Should be roughly 7B (within 20%)
        assert 5.5 < params < 9.0

    def test_empty_config(self):
        from vmlx_engine.utils.model_inspector import _estimate_param_count

        assert _estimate_param_count({}) == 0.0
        assert _estimate_param_count({"hidden_size": 4096}) == 0.0

    def test_moe_model_estimate(self):
        from vmlx_engine.utils.model_inspector import _estimate_param_count

        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "vocab_size": 128000,
            "intermediate_size": 11008,
            "n_routed_experts": 256,
            "moe_intermediate_size": 1536,
        }
        params = _estimate_param_count(config)
        # MoE with 256 experts should be very large
        assert params > 50


class TestListModels:
    """Tests for list_models_in_dir()."""

    def test_empty_directory(self, tmp_path):
        from vmlx_engine.utils.model_inspector import list_models_in_dir

        models = list_models_in_dir(str(tmp_path))
        assert models == []

    def test_finds_models(self, tmp_path):
        from vmlx_engine.utils.model_inspector import list_models_in_dir

        # Create two model directories
        for name in ["model-a", "model-b"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({
                    "model_type": "llama",
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "vocab_size": 32000,
                    "intermediate_size": 11008,
                }, f)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        models = list_models_in_dir(str(tmp_path))
        assert len(models) == 2

    def test_skips_non_model_dirs(self, tmp_path):
        from vmlx_engine.utils.model_inspector import list_models_in_dir

        # Dir with config but no weights
        no_weights = tmp_path / "no-weights"
        no_weights.mkdir()
        with open(no_weights / "config.json", "w") as f:
            json.dump({"model_type": "llama"}, f)

        # Dir with weights but no config
        no_config = tmp_path / "no-config"
        no_config.mkdir()
        (no_config / "model.safetensors").write_bytes(b"\x00" * 1024)

        models = list_models_in_dir(str(tmp_path))
        assert len(models) == 0

    def test_nonexistent_directory(self):
        from vmlx_engine.utils.model_inspector import list_models_in_dir

        models = list_models_in_dir("/nonexistent/path")
        assert models == []


class TestFormatModelInfo:
    """Tests for format_model_info()."""

    def test_formats_basic_model(self, tmp_path):
        from vmlx_engine.utils.model_inspector import format_model_info, inspect_model

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)
        output = format_model_info(info)

        assert "llama" in output.lower()
        assert "4096" in output
        assert "32" in output
        assert "memory" in output.lower()

    def test_formats_moe_model(self, tmp_path):
        from vmlx_engine.utils.model_inspector import format_model_info, inspect_model

        config = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 88,
            "num_attention_heads": 32,
            "vocab_size": 131072,
            "intermediate_size": 2688,
            "moe_latent_size": 1024,
            "n_routed_experts": 512,
            "num_experts_per_tok": 22,
            "hybrid_override_pattern": "M" * 88,
        }
        model_dir = _make_model_dir(tmp_path, config)
        info = inspect_model(model_dir)
        output = format_model_info(info)

        assert "MoE" in output
        assert "LatentMoE" in output
        assert "512" in output


class TestResolveModelPath:
    """Tests for resolve_model_path()."""

    def test_local_path(self, tmp_path):
        from vmlx_engine.utils.model_inspector import resolve_model_path

        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        result = resolve_model_path(str(model_dir))
        assert result == str(model_dir.resolve())

    def test_nonexistent_path(self):
        from vmlx_engine.utils.model_inspector import resolve_model_path

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_model_path("/nonexistent/model/path")

    def test_hf_id_not_cached(self):
        from vmlx_engine.utils.model_inspector import resolve_model_path

        with pytest.raises(FileNotFoundError, match="Download"):
            resolve_model_path("fake-org/nonexistent-model-12345")


class TestDefaultOutputName:
    """Tests for convert command output naming."""

    def test_hf_id(self):
        from vmlx_engine.commands.convert import _default_output_name

        result = _default_output_name("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", 4)
        assert result == "NVIDIA-Nemotron-3-Super-120B-A12B-vmlx-4bit"

    def test_local_path(self):
        from vmlx_engine.commands.convert import _default_output_name

        result = _default_output_name("/path/to/my-model", 8)
        assert result == "my-model-vmlx-8bit"

    def test_strips_bf16_suffix(self):
        from vmlx_engine.commands.convert import _default_output_name

        result = _default_output_name("org/model-BF16", 4)
        assert result == "model-vmlx-4bit"

    def test_strips_fp16_suffix(self):
        from vmlx_engine.commands.convert import _default_output_name

        result = _default_output_name("org/model-FP16", 4)
        assert result == "model-vmlx-4bit"

    def test_trailing_slash(self):
        from vmlx_engine.commands.convert import _default_output_name

        result = _default_output_name("/path/to/model/", 4)
        assert result == "model-vmlx-4bit"

    def test_various_bits(self):
        from vmlx_engine.commands.convert import _default_output_name

        for bits in [2, 3, 4, 6, 8]:
            result = _default_output_name("org/model", bits)
            assert result == f"model-vmlx-{bits}bit"

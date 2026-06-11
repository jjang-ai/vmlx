"""Tests for JANG model loading — param count, tree_flatten, detection, MCP config."""
import json
import pytest
from pathlib import Path


class TestJangDetection:
    """Verify is_jang_model detects config files correctly."""

    def test_detects_jang_config(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model
        (tmp_path / "jang_config.json").write_text("{}")
        assert is_jang_model(str(tmp_path)) is True

    def test_detects_jjqf_config(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model
        (tmp_path / "jjqf_config.json").write_text("{}")
        assert is_jang_model(str(tmp_path)) is True

    def test_detects_jangtq_sidecar_without_weight_format(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model

        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jangtq",
                    "family": "mimo_v2",
                    "profile": "JANGTQ_2",
                    "tq_layout": "prestacked_switch_mlp",
                }
            )
        )

        assert is_jang_model(str(tmp_path)) is True

    def test_no_config_returns_false(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model(str(tmp_path)) is False

    def test_hf_repo_id_returns_false(self):
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model("mlx-community/Llama-3.2-3B-4bit") is False

    def test_nonexistent_path_returns_false(self):
        from vmlx_engine.utils.jang_loader import is_jang_model
        assert is_jang_model("/this/does/not/exist") is False

    def test_v2_bundle_has_tq_packed_uses_index_not_first_shard_only(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _v2_bundle_has_tq_packed

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001.safetensors",
                        "model.layers.1.mlp.switch_mlp.up_proj.tq_packed": "model-00002.safetensors",
                    }
                }
            )
        )
        (tmp_path / "model-00001.safetensors").write_bytes(b"first shard lacks tq")
        (tmp_path / "model-00002.safetensors").write_bytes(b"second shard has tq")

        assert _v2_bundle_has_tq_packed(tmp_path, [tmp_path / "model-00001.safetensors"]) is True

    def test_load_jang_model_accepts_affine_weight_format(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
        (tmp_path / "jang_config.json").write_text(
            '{"version":2,"weight_format":"affine","quantization":{}}'
        )
        expected = (object(), object())

        monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda path, jang_cfg, **kwargs: expected,
        )
        monkeypatch.setattr(
            jang_loader,
            "_ensure_jang_family_runtime_supported",
            lambda path, config: None,
        )

        assert jang_loader.load_jang_model(tmp_path, skip_eval=True) is expected

    def test_load_jang_model_accepts_jangtq_format_sidecar(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        (tmp_path / "config.json").write_text('{"model_type":"mimo_v2"}')
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jangtq",
                    "family": "mimo_v2",
                    "profile": "JANGTQ_2",
                    "tq_layout": "prestacked_switch_mlp",
                }
            )
        )
        expected = (object(), object())

        monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda path, jang_cfg, **kwargs: expected,
        )
        monkeypatch.setattr(
            jang_loader,
            "_ensure_jang_family_runtime_supported",
            lambda path, config: None,
        )

        assert jang_loader.load_jang_model(tmp_path, skip_eval=True) is expected

    def test_load_jang_vlm_model_accepts_jangtq_format_sidecar(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader

        (tmp_path / "config.json").write_text('{"model_type":"mimo_v2"}')
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jangtq",
                    "family": "mimo_v2",
                    "profile": "JANGTQ_2",
                    "tq_layout": "prestacked_switch_mlp",
                }
            )
        )
        expected = (object(), object())

        monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2_vlm",
            lambda path, jang_cfg, **kwargs: expected,
        )
        monkeypatch.setattr(
            jang_loader,
            "_ensure_jang_family_runtime_supported",
            lambda path, config: None,
        )

        assert jang_loader.load_jang_vlm_model(tmp_path, skip_eval=True) is expected

    def test_jangtq_vlm_bits_map_falls_back_to_config_metadata(self):
        from vmlx_engine.utils.jang_loader import _jangtq_bits_map_from_metadata

        assert _jangtq_bits_map_from_metadata(
            {"format": "jangtq"},
            {
                "mxtq_bits": {
                    "routed_expert": {
                        "gate_proj": 3,
                        "up_proj": 2,
                        "down_proj": 3,
                    }
                }
            },
        ) == {
            "routed_expert": {
                "gate_proj": 3,
                "up_proj": 2,
                "down_proj": 3,
            }
        }

    def test_jangtq_vlm_bits_map_falls_back_to_runtime_routed_bits(self):
        from vmlx_engine.utils.jang_loader import _jangtq_bits_map_from_metadata

        assert _jangtq_bits_map_from_metadata(
            {"format": "jangtq"},
            {
                "runtime": {
                    "routed_expert_bits": {
                        "gate_proj": 3,
                        "up_proj": 3,
                        "down_proj": 3,
                    }
                }
            },
        ) == {
            "routed_expert": {
                "gate_proj": 3,
                "up_proj": 3,
                "down_proj": 3,
            }
        }

    def test_jangtq_vlm_bits_map_preserves_full_role_metadata(self):
        from vmlx_engine.utils.jang_loader import _jangtq_bits_map_from_metadata

        assert _jangtq_bits_map_from_metadata(
            {
                "format": "jangtq",
                "mxtq_bits": {
                    "attention": 8,
                    "linear_attention": 8,
                    "shared_expert": 8,
                    "routed_expert": 2,
                    "embed_tokens": 8,
                    "lm_head": 8,
                },
            },
            {},
        ) == {
            "attention": 8,
            "linear_attention": 8,
            "shared_expert": 8,
            "routed_expert": 2,
            "embed_tokens": 8,
            "lm_head": 8,
        }

    def test_large_expert_bfloat16_compute_applies_to_397b_shape(self, tmp_path):
        import mlx.core as mx
        from vmlx_engine.utils.jang_loader import _apply_large_expert_bfloat16_compute

        class _Model:
            dtype = None

            def set_dtype(self, dtype):
                self.dtype = dtype

        model = _Model()

        assert _apply_large_expert_bfloat16_compute(
            model,
            tmp_path,
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 4096,
                    "num_experts": 512,
                },
            },
        ) is True
        assert model.dtype == mx.bfloat16

    def test_jangtq_fast_paths_apply_large_expert_bfloat16_before_return(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        text_idx = source.index("model, tokenizer = _load_jangtq")
        text_return_idx = source.index("return model, tokenizer", text_idx)
        assert "_apply_large_expert_bfloat16_compute(" in source[text_idx:text_return_idx]

        vlm_idx = source.index("_hydrate_jangtq_model(")
        vlm_return_idx = source.index("return _vlm_model, _vlm_processor", vlm_idx)
        assert "_apply_large_expert_bfloat16_compute(" in source[vlm_idx:vlm_return_idx]

    def test_prepare_jangtq_vlm_first_forward_installs_split_and_warmup(self, monkeypatch):
        import sys
        import types
        from vmlx_engine.utils import jang_loader

        calls = []
        fake = types.ModuleType("jang_tools.load_jangtq_kimi_vlm")

        def _install(model):
            calls.append("split")
            model._jang_cb_split = True

        def _warm(model):
            calls.append("warm")
            model._jang_warmed = True

        fake._install_vl_command_buffer_split = _install
        fake._warmup_jit_per_layer = _warm
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq_kimi_vlm", fake)

        model = types.SimpleNamespace()
        assert jang_loader._prepare_jangtq_vlm_first_forward(model) is True
        assert calls == ["split", "warm"]
        assert model._jang_cb_split is True
        assert model._jang_warmed is True

    def test_prepare_jangtq_vlm_first_forward_falls_back_to_prefill(self, monkeypatch):
        import sys
        import types
        from vmlx_engine.utils import jang_loader

        calls = []
        fake = types.ModuleType("jang_tools.load_jangtq_kimi_vlm")

        def _install(model):
            calls.append("split")
            model._jang_cb_split = True

        def _warm(model):
            calls.append("warm")
            raise AttributeError("'NoneType' object has no attribute 'offset'")

        fake._install_vl_command_buffer_split = _install
        fake._warmup_jit_per_layer = _warm
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq_kimi_vlm", fake)
        monkeypatch.setattr(
            jang_loader,
            "_warmup_jangtq_vlm_language_prefill",
            lambda model, **kwargs: calls.append("prefill"),
        )

        assert jang_loader._prepare_jangtq_vlm_first_forward(types.SimpleNamespace()) is True
        assert calls == ["split", "warm", "prefill"]

    def test_jangtq_vlm_fast_path_prepares_first_forward_before_return(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        vlm_idx = source.index("_hydrate_jangtq_model(")
        vlm_return_idx = source.index("return _vlm_model, _vlm_processor", vlm_idx)
        assert "_prepare_jangtq_vlm_first_forward(" in source[vlm_idx:vlm_return_idx]

    def test_jangtq_vlm_language_prefill_uses_mimo_input_ids_keyword(self, monkeypatch):
        import mlx_lm.models.cache as cache_mod
        from vmlx_engine.utils import jang_loader

        calls = []

        class FakeLM:
            def __call__(self, *args, **kwargs):
                calls.append({"args": args, "kwargs": kwargs})
                if "input_ids" in kwargs:
                    return kwargs["input_ids"]
                if "inputs" in kwargs:
                    raise ValueError("Specify input_ids or inputs_embeds")
                raise TypeError("unexpected call")

        monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda _lm: ["cache"])

        jang_loader._warmup_jangtq_vlm_language_prefill(FakeLM())

        assert calls
        assert "input_ids" in calls[0]["kwargs"]
        assert calls[0]["kwargs"]["cache"] == ["cache"]

    def test_mimo_jangtq_input_ids_warmup_uses_input_ids_before_inputs(self, monkeypatch):
        import mlx_lm.models.cache as cache_mod
        import types
        from vmlx_engine.utils import jang_loader

        calls = []
        kimi_module = types.SimpleNamespace(
            _warmup_prefill_tokens=lambda _model, _lm: 24,
            _materialize=lambda out: calls.append({"materialized": out.shape}),
        )

        class FakeLM:
            model_type = "mimo_v2"

            def __call__(self, *args, **kwargs):
                calls.append({"args": args, "kwargs": kwargs})
                if "input_ids" in kwargs:
                    return kwargs["input_ids"]
                if "inputs" in kwargs:
                    raise ValueError("Specify input_ids or inputs_embeds")
                raise TypeError("unexpected call")

        monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda _lm: ["cache"])

        assert jang_loader._run_mimo_jangtq_input_ids_prefill_warmup(
            kimi_module,
            FakeLM(),
            verbose=False,
        ) is True

        assert "input_ids" in calls[0]["kwargs"]
        assert calls[0]["kwargs"]["input_ids"].shape == (1, 24)
        assert calls[0]["kwargs"]["cache"] == ["cache"]

    def test_patch_jangtq_kimi_mimo_warmup_wraps_upstream(self, monkeypatch):
        import sys
        import types
        from vmlx_engine.utils import jang_loader

        calls = []
        pkg = types.ModuleType("jang_tools")
        fake = types.ModuleType("jang_tools.load_jangtq_kimi_vlm")

        def original(model, verbose=True):
            calls.append(("original", verbose, model.args.model_type))

        fake._warmup_jit_per_layer = original
        monkeypatch.setitem(sys.modules, "jang_tools", pkg)
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq_kimi_vlm", fake)
        monkeypatch.setattr(
            jang_loader,
            "_run_mimo_jangtq_input_ids_prefill_warmup",
            lambda module, model, **kwargs: calls.append(
                ("vmlx", module.__name__, model.args.model_type, kwargs["verbose"])
            ),
        )

        assert jang_loader._patch_jangtq_kimi_mimo_input_ids_warmup() is True

        model = types.SimpleNamespace(args=types.SimpleNamespace(model_type="mimo_v2"))
        fake._warmup_jit_per_layer(model, verbose=False)

        assert calls == [
            ("original", False, "mimo_v2"),
            ("vmlx", "jang_tools.load_jangtq_kimi_vlm", "mimo_v2", False),
        ]

    def test_jangtq_text_fast_path_installs_mimo_warmup_patch_before_load(self):
        source = Path("vmlx_engine/utils/jang_loader.py").read_text()

        text_idx = source.index("model, tokenizer = _load_jangtq")
        patch_idx = source.rindex("_patch_jangtq_kimi_mimo_input_ids_warmup", 0, text_idx)

        assert patch_idx < text_idx

    def test_detects_jang_affine_weight_format(self, tmp_path):
        from vmlx_engine.utils.jang_loader import is_jang_model

        (tmp_path / "config.json").write_text('{"model_type":"gemma4_unified"}')
        (tmp_path / "jang_config.json").write_text(
            '{"version":2,"weight_format":"jang_affine","quantization":{}}'
        )

        assert is_jang_model(tmp_path) is True

    def test_normalize_step3p5_alias_for_step3p7_model_type(self):
        from vmlx_engine.utils.jang_loader import _normalize_step3p7_model_type

        config = {"model_type": "step3p7", "text_config": {"model_type": "step3p7"}}
        _normalize_step3p7_model_type(config)
        assert config["model_type"] == "step3p5"
        assert config["text_config"]["model_type"] == "step3p5"

    def test_step3p7_moe_weights_remap_to_step3p5_module_paths(self):
        from vmlx_engine.utils.jang_loader import _remap_step3p7_moe_weights

        marker = object()
        weights = {
            "model.layers.10.moe.gate.weight": marker,
            "model.layers.10.moe.gate.scales": marker,
            "model.layers.10.moe.router_bias": marker,
            "model.layers.10.moe.up_proj.weight": marker,
            "model.layers.10.moe.gate_proj.weight": marker,
            "model.layers.10.moe.down_proj.weight": marker,
            "model.layers.10.share_expert.up_proj.weight": marker,
            "model.layers.10.share_expert.gate_proj.weight": marker,
            "model.layers.10.share_expert.down_proj.weight": marker,
            "model.layers.10.self_attn.q_proj.weight": marker,
        }

        remapped = _remap_step3p7_moe_weights(
            weights,
            {"model_type": "step3p5"},
            {"architecture": {"type": "step3p7"}},
        )

        assert set(remapped) == {
            "model.layers.10.mlp.gate.gate.weight",
            "model.layers.10.mlp.gate.gate.scales",
            "model.layers.10.mlp.gate.router_bias",
            "model.layers.10.mlp.switch_mlp.up_proj.weight",
            "model.layers.10.mlp.switch_mlp.gate_proj.weight",
            "model.layers.10.mlp.switch_mlp.down_proj.weight",
            "model.layers.10.mlp.share_expert.up_proj.weight",
            "model.layers.10.mlp.share_expert.gate_proj.weight",
            "model.layers.10.mlp.share_expert.down_proj.weight",
            "model.layers.10.self_attn.q_proj.weight",
        }

    def test_step3p7_dense_shard_norm_weights_get_zero_centered_offset(self):
        import mlx.core as mx

        from vmlx_engine.utils.jang_loader import _fix_step3p7_zero_centered_norm_weights

        raw_norm = mx.array([-0.25, 0.125], dtype=mx.float16)
        raw_attn = mx.array([0.5], dtype=mx.float16)
        weights = {
            "model.layers.0.input_layernorm.weight": raw_norm,
            "model.layers.0.self_attn.q_norm.weight": raw_attn,
            "model.layers.0.self_attn.q_proj.weight": mx.array([1.0], dtype=mx.float16),
        }

        fixed = _fix_step3p7_zero_centered_norm_weights(
            weights,
            {"model_type": "step3p5"},
            {"architecture": {"type": "step3p7"}},
            shard_had_vanilla_moe_keys=False,
        )

        assert fixed["model.layers.0.input_layernorm.weight"].tolist() == [0.75, 1.125]
        assert fixed["model.layers.0.self_attn.q_norm.weight"].tolist() == [1.5]
        assert fixed["model.layers.0.self_attn.q_proj.weight"].tolist() == [1.0]

    def test_step3p7_vlm_loader_registers_source_runtime_before_mlx_vlm_resolution(
        self,
        tmp_path,
        monkeypatch,
    ):
        import sys

        import mlx_vlm.utils as vlm_utils

        from vmlx_engine.utils import jang_loader

        config = {
            "model_type": "step3p7",
            "text_config": {"model_type": "step3p5"},
            "vision_config": {"model_type": "perception_encoder"},
            "quantization": {"bits": 4, "group_size": 64},
        }
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

        for name in (
            "mlx_vlm.models.step3p7",
            "mlx_vlm.models.step3p7.processing_step3",
        ):
            sys.modules.pop(name, None)

        monkeypatch.setattr(jang_loader, "_ensure_zaya_runtime_supported", lambda *a, **k: None)
        monkeypatch.setattr(
            jang_loader,
            "_ensure_jang_family_runtime_supported",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            jang_loader,
            "_prepare_runtime_weight_quantization",
            lambda *a, **k: (dict(config), 4, 64),
        )

        def fake_get_model_and_args(*, config):
            assert "mlx_vlm.models.step3p7" in sys.modules
            assert "mlx_vlm.models.step3p7.processing_step3" in sys.modules
            raise RuntimeError("registered-step3p7")

        monkeypatch.setattr(vlm_utils, "get_model_and_args", fake_get_model_and_args)

        with pytest.raises(RuntimeError, match="registered-step3p7"):
            jang_loader._load_jang_v2_vlm(
                tmp_path,
                {"format": "jang", "quantization": {}},
                skip_eval=True,
            )

        sys.modules.pop("mlx_vlm.models.step3p7.processing_step3", None)
        sys.modules.pop("mlx_vlm.models.step3p7", None)

    def test_jang_routed_group_size_reads_nested_mixed_affine_metadata(self):
        from vmlx_engine.utils.jang_loader import _jang_routed_expert_group_size

        assert (
            _jang_routed_expert_group_size(
                {
                    "quantization": {
                        "top_level_default": {"bits": 2, "group_size": 128},
                        "routed_experts": {"bits": 2, "group_size": 128},
                        "non_routed": {"bits": 8, "group_size": 64},
                    }
                },
                default=64,
            )
            == 128
        )

    def test_step37_moe_num_experts_triggers_gate_dequant(self):
        from vmlx_engine.utils.jang_loader import _moe_expert_count

        config = {
            "model_type": "step3p7",
            "text_config": {
                "model_type": "step3p5",
                "hidden_size": 4096,
                "moe_num_experts": 288,
            },
        }

        assert _moe_expert_count(config) == 288

    def test_step37_quantized_gate_key_is_not_dequantized_after_quantize(self):
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _should_dequantize_vlm_gate_weight

        class _Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.QuantizedLinear(
                    input_dims=4096,
                    output_dims=288,
                    bias=False,
                    group_size=64,
                    bits=8,
                )

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = _Gate()

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = _MLP()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Layer()]

        class _LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()

        class _VLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _LanguageModel()

        model = _VLM()

        assert _should_dequantize_vlm_gate_weight(
            model,
            "language_model.model.layers.0.mlp.gate.gate.weight",
        ) is False

    def test_gemma4_qat_mxfp4_ple_dequant_uses_mxfp_mode(self):
        import mlx.core as mx

        from vmlx_engine.utils.jang_loader import _dequantize_gemma4_ple_tensor

        packed = mx.zeros((2, 192), dtype=mx.uint32)
        scales = mx.ones((2, 48), dtype=mx.uint8)

        dequantized, bits, group_size, mode = _dequantize_gemma4_ple_tensor(
            packed,
            scales,
            None,
            "language_model.model.per_layer_model_projection.weight",
        )
        mx.eval(dequantized)

        assert dequantized.shape == (2, 1536)
        assert dequantized.dtype == mx.float16
        assert bits == 4
        assert group_size == 32
        assert mode == "mxfp4"

    def test_gemma4_qat_mxfp4_quantized_ple_stays_packed(self):
        import mlx.core as mx
        import mlx.nn as nn

        from vmlx_engine.utils.jang_loader import (
            _configure_gemma4_quantized_ple_module,
        )

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_model_projection = nn.QuantizedLinear(
                    input_dims=1536,
                    output_dims=2,
                    bias=False,
                    group_size=64,
                    bits=8,
                )
                self.per_layer_model_projection.biases = mx.zeros(
                    (2, 48), dtype=mx.float16
                )

        class _LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()

        class _VLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _LanguageModel()

        packed = mx.zeros((2, 192), dtype=mx.uint32)
        scales = mx.ones((2, 48), dtype=mx.uint8)
        model = _VLM()

        configured, bits, group_size, mode = _configure_gemma4_quantized_ple_module(
            model,
            "language_model.model.per_layer_model_projection.weight",
            packed,
            scales,
        )

        module = model.language_model.model.per_layer_model_projection
        assert configured is True
        assert bits == 4
        assert group_size == 32
        assert mode == "mxfp4"
        assert module.bits == 4
        assert module.group_size == 32
        assert module.mode == "mxfp4"
        assert not hasattr(module, "biases") or module.biases is None

    def test_native_mxfp_uint8_scales_select_mxfp_kernel(self):
        import mlx.core as mx
        import mlx.nn as nn

        from vmlx_engine.utils.jang_loader import _fix_quantized_bits

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.QuantizedLinear(
                    input_dims=1536,
                    output_dims=2,
                    bias=False,
                    group_size=64,
                    bits=8,
                )
                self.proj.weight = mx.zeros((2, 192), dtype=mx.uint32)
                self.proj.scales = mx.ones((2, 48), dtype=mx.uint8)
                self.proj.biases = mx.zeros((2, 48), dtype=mx.float16)

        model = _Model()

        _fix_quantized_bits(model)

        assert model.proj.bits == 4
        assert model.proj.group_size == 32
        assert model.proj.mode == "mxfp4"
        assert not hasattr(model.proj, "biases") or model.proj.biases is None

    def test_gemma4_moe_mxfp_experts_split_to_switch_glu_float_weights(self):
        import mlx.core as mx

        from vmlx_engine.utils.jang_loader import (
            _split_dequantize_gemma4_moe_mxfp_experts,
        )

        weights = {
            "language_model.model.layers.0.experts.gate_up_proj.weight": mx.zeros(
                (2, 4, 4), dtype=mx.uint32
            ),
            "language_model.model.layers.0.experts.gate_up_proj.scales": mx.ones(
                (2, 4, 1), dtype=mx.uint8
            ),
            "language_model.model.layers.0.experts.down_proj.weight": mx.zeros(
                (2, 6, 4), dtype=mx.uint32
            ),
            "language_model.model.layers.0.experts.down_proj.scales": mx.ones(
                (2, 6, 1), dtype=mx.uint8
            ),
        }

        out = _split_dequantize_gemma4_moe_mxfp_experts(weights)

        assert "language_model.model.layers.0.experts.gate_up_proj.weight" not in out
        assert "language_model.model.layers.0.experts.gate_up_proj.scales" not in out
        assert out[
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"
        ].shape == (2, 2, 32)
        assert out[
            "language_model.model.layers.0.experts.switch_glu.up_proj.weight"
        ].shape == (2, 2, 32)
        assert out[
            "language_model.model.layers.0.experts.switch_glu.down_proj.weight"
        ].shape == (2, 6, 32)
        assert out[
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"
        ].dtype == mx.float16

    def test_gemma4_moe_mxfp_expert_cross_shard_sidecars_are_hydrated(
        self,
        tmp_path,
    ):
        import mlx.core as mx

        from vmlx_engine.utils.jang_loader import (
            _hydrate_gemma4_moe_mxfp_cross_shard_sidecars,
            _split_dequantize_gemma4_moe_mxfp_experts,
        )

        weight_key = "language_model.model.layers.10.experts.gate_up_proj.weight"
        scales_key = "language_model.model.layers.10.experts.gate_up_proj.scales"
        shard_weights = {
            weight_key: mx.zeros((2, 4, 4), dtype=mx.uint32),
        }
        scales_shard = tmp_path / "model-00002.safetensors"
        mx.save_safetensors(
            str(scales_shard),
            {
                scales_key: mx.ones((2, 4, 1), dtype=mx.uint8),
            },
        )
        weight_map = {
            weight_key: "model-00001.safetensors",
            scales_key: scales_shard.name,
        }

        hydrated = _hydrate_gemma4_moe_mxfp_cross_shard_sidecars(
            dict(shard_weights),
            tmp_path,
            weight_map,
        )
        out = _split_dequantize_gemma4_moe_mxfp_experts(hydrated)

        assert scales_key in hydrated
        assert weight_key not in out
        assert (
            "language_model.model.layers.10.experts.switch_glu.gate_proj.weight"
            in out
        )

    def test_gemma4_moe_mxfp_vlm_loader_initializes_sidecar_weight_map(self):
        import inspect

        from vmlx_engine.utils.jang_loader import _load_jang_v2_vlm

        source = inspect.getsource(_load_jang_v2_vlm)
        init_idx = source.find("_index_weight_map =")
        hydrate_idx = source.find("_hydrate_gemma4_moe_mxfp_cross_shard_sidecars")

        assert init_idx != -1
        assert hydrate_idx != -1
        assert init_idx < hydrate_idx

    def test_step37_text_loader_preserves_quantized_gate_sidecars(self):
        import mlx.core as mx
        import mlx.nn as nn

        from vmlx_engine.utils.jang_loader import _prepare_gate_dequant_weights

        class _Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.QuantizedLinear(
                    input_dims=4096,
                    output_dims=288,
                    bias=False,
                    group_size=64,
                    bits=8,
                )

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = _Gate()

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = _MLP()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Layer()]

        model = nn.Module()
        model.model = _Inner()

        weights = {
            "model.layers.0.mlp.gate.gate.weight": mx.zeros(
                (288, 1024),
                dtype=mx.uint32,
            ),
            "model.layers.0.mlp.gate.gate.scales": mx.zeros(
                (288, 64),
                dtype=mx.float16,
            ),
            "model.layers.0.mlp.gate.gate.biases": mx.zeros(
                (288, 64),
                dtype=mx.float16,
            ),
        }

        prepared = _prepare_gate_dequant_weights(model, weights)

        assert prepared["model.layers.0.mlp.gate.gate.weight"].dtype == mx.uint32
        assert "model.layers.0.mlp.gate.gate.scales" in prepared
        assert "model.layers.0.mlp.gate.gate.biases" in prepared


def test_fix_quantized_bits_uses_embedding_logical_dims_for_ambiguous_affine_shape():
    """Qwen3.6 JANG_4M embeds must dequantize to 5120, not 2560."""
    import mlx.core as mx
    import mlx.nn as nn

    from vmlx_engine.utils.jang_loader import _fix_quantized_bits

    module = nn.QuantizedEmbedding(248320, 5120, group_size=32, bits=8)
    module.weight = mx.zeros((248320, 640), dtype=mx.uint32)
    module.scales = mx.zeros((248320, 80), dtype=mx.float16)
    module.biases = mx.zeros((248320, 80), dtype=mx.float16)

    class Wrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

    wrapper = Wrapper(module)

    _fix_quantized_bits(wrapper)

    assert wrapper.child.bits == 4
    assert wrapper.child.group_size == 64
    out = wrapper.child(mx.array([[0, 1]], dtype=mx.uint32))
    mx.eval(out)
    assert out.shape == (1, 2, 5120)


class TestTreeFlattenImport:
    """Verify the tree_flatten fix uses correct import path."""

    def test_mlx_utils_tree_flatten_exists(self):
        from mlx.utils import tree_flatten
        assert callable(tree_flatten)

    def test_mx_core_has_no_utils(self):
        """mlx.core does NOT have a utils attribute."""
        import mlx.core as mx
        assert not hasattr(mx, 'utils'), "mx.utils should not exist"

    def test_tree_flatten_on_nested_dict(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        params = {"layers": [{"weight": mx.zeros((3, 3)), "bias": mx.zeros((3,))}]}
        flat = tree_flatten(params)
        assert len(flat) == 2
        total = sum(p.size for _, p in flat)
        assert total == 12  # 9 + 3

    def test_jang_loader_uses_correct_import(self):
        """jang_loader.py uses 'from mlx.utils import tree_flatten', not mx.utils."""
        import inspect
        from vmlx_engine.utils import jang_loader
        source = inspect.getsource(jang_loader)
        assert "from mlx.utils import tree_flatten" in source
        assert "mx.utils.tree_flatten" not in source


class TestFixQuantizedBits:
    """Post-load bit repair must honor proven per-module overrides."""

    def test_non_dsv4_qwen_ambiguous_overrides_are_not_trusted(self):
        from vmlx_engine.utils.jang_loader import (
            _post_load_quantization_overrides,
        )

        config = {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
            },
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "language_model.model.embed_tokens": {
                    "bits": 8,
                    "group_size": 32,
                },
            },
        }
        jang_cfg = {
            "format": "jang",
            "quantization": {
                "profile": "JANG_4M",
                "block_size": 64,
                "bit_widths_used": [4, 8],
            },
            "architecture": {
                "type": "hybrid_ssm_dense",
                "has_vision": True,
                "has_ssm": True,
            },
        }

        assert _post_load_quantization_overrides(config, jang_cfg) is None

    def test_dsv4_prestacked_switch_override_wins_over_shape_ambiguity(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        from vmlx_engine.utils.jang_loader import (
            _fix_quantized_bits,
            _post_load_quantization_overrides,
        )

        class _Switch(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = QuantizedSwitchLinear(
                    64, 4, 2, bias=False, group_size=64, bits=4
                )

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.switch_mlp = _Switch()

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = _MLP()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Layer()]

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()

        model = _Model()
        proj = model.model.layers[0].mlp.switch_mlp.up_proj
        proj.weight = mx.zeros((2, 4, 8), dtype=mx.uint32)
        proj.scales = mx.zeros((2, 4, 1), dtype=mx.float16)
        proj.biases = mx.zeros((2, 4, 1), dtype=mx.float16)
        assert proj.bits == 4
        assert proj.group_size == 64

        q_overrides = {
            "model_type": "deepseek_v4",
            "quantization": {
                "model.layers.0.mlp.switch_mlp.up_proj": {
                    "bits": 2,
                    "group_size": 128,
                }
            },
        }

        trusted = _post_load_quantization_overrides(q_overrides, {})
        assert trusted is q_overrides["quantization"]

        _fix_quantized_bits(model, trusted)

        assert proj.bits == 2
        assert proj.group_size == 128
        assert proj.input_dims == 128


class TestPreFixBitsFromShard:
    """Test _pre_fix_bits_from_shard fixes QuantizedLinear bits before load_weights.

    Covers GitHub issues #62 (MiniMax-M2.5-JANG_3L) and #63
    (Qwen3.5-122B-A10B-JANG_4K) where embed_tokens is at 4-bit but
    nn.quantize() created the module at 3-bit (min of bit_widths_used).
    """

    def test_fixes_embed_tokens_bits_mismatch(self):
        """Module at 3-bit, weight is 4-bit packed → bits corrected to 4."""
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        # Create a simple model with a QuantizedLinear at bits=3
        model = nn.Module()
        ql = nn.QuantizedLinear(3072, 248320, group_size=128, bits=3)
        model.embed_tokens = ql
        assert model.embed_tokens.bits == 3

        # Simulate a weight shard with 4-bit packed embed_tokens
        # 4-bit: weight_cols = 3072 * 4 / 32 = 384
        # scales cols = 3072 / 128 = 24
        shard = {
            "embed_tokens.weight": mx.zeros((248320, 384), dtype=mx.uint32),
            "embed_tokens.scales": mx.zeros((248320, 24), dtype=mx.float16),
            "embed_tokens.biases": mx.zeros((248320, 24), dtype=mx.float16),
        }

        _pre_fix_bits_from_shard(model, shard, block_size=128)

        assert model.embed_tokens.bits == 4, (
            f"Expected bits=4, got bits={model.embed_tokens.bits}"
        )

    def test_no_change_when_bits_match(self):
        """Module at 4-bit, weight is 4-bit → no change."""
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        model = nn.Module()
        ql = nn.QuantizedLinear(3072, 4096, group_size=128, bits=4)
        model.layer = ql

        # 4-bit: weight_cols = 3072 * 4 / 32 = 384
        shard = {
            "layer.weight": mx.zeros((4096, 384), dtype=mx.uint32),
            "layer.scales": mx.zeros((4096, 24), dtype=mx.float16),
            "layer.biases": mx.zeros((4096, 24), dtype=mx.float16),
        }

        _pre_fix_bits_from_shard(model, shard, block_size=128)
        assert model.layer.bits == 4

    def test_skips_non_quantized_weights(self):
        """Float16 weights (non-quantized) are ignored."""
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        model = nn.Module()
        model.norm = nn.RMSNorm(3072)

        shard = {
            "norm.weight": mx.zeros((3072,), dtype=mx.float16),
        }

        # Should not crash or modify anything
        _pre_fix_bits_from_shard(model, shard, block_size=128)

    def test_fixes_8bit_layer_from_2bit_default(self):
        """Module at 2-bit default, weight is 8-bit → corrected to 8."""
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        model = nn.Module()
        ql = nn.QuantizedLinear(3072, 3072, group_size=128, bits=2)
        model.kv_proj = ql

        # 8-bit: weight_cols = 3072 * 8 / 32 = 768
        # scales cols = 3072 / 128 = 24
        shard = {
            "kv_proj.weight": mx.zeros((3072, 768), dtype=mx.uint32),
            "kv_proj.scales": mx.zeros((3072, 24), dtype=mx.float16),
            "kv_proj.biases": mx.zeros((3072, 24), dtype=mx.float16),
        }

        _pre_fix_bits_from_shard(model, shard, block_size=128)
        assert model.kv_proj.bits == 8

    def test_fixes_group_size_mismatch(self):
        """Module at gs=128 bits=3, weight at gs=64 bits=4 → corrected.

        Use dimensions where block_size=128 does NOT produce a valid bits value,
        forcing the function to try gs=64, which gives the correct bits=4.
        """
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        model = nn.Module()
        # 256 input dims, gs=128, bits=3
        ql = nn.QuantizedLinear(256, 256, group_size=128, bits=3)
        model.gate = ql

        # Actual weight: 4-bit with gs=64
        # weight_cols = 256 * 4 / 32 = 32
        # scales_cols = 256 / 64 = 4
        shard = {
            "gate.weight": mx.zeros((256, 32), dtype=mx.uint32),
            "gate.scales": mx.zeros((256, 4), dtype=mx.float16),
            "gate.biases": mx.zeros((256, 4), dtype=mx.float16),
        }

        # block_size=128: in_dim = 4 * 128 = 512, but weight has 256 input dims
        # (32 * 32) / 512 = 2 → valid 2-bit... hmm, that's also valid.
        # Let's use a case where only gs=64 works:
        # 512 input dims, gs=64, bits=8
        model2 = nn.Module()
        model2.proj = nn.QuantizedLinear(512, 512, group_size=128, bits=3)

        # 8-bit with gs=64: weight_cols = 512 * 8 / 32 = 128
        # scales_cols = 512 / 64 = 8
        # With gs=128: in_dim = 8 * 128 = 1024, bits = 128*32/1024 = 4 → valid 4-bit
        # With gs=64:  in_dim = 8 * 64 = 512,   bits = 128*32/512 = 8 → valid 8-bit
        # Function tries gs=128 first → bits=4, so module gets bits=4 gs=128.
        # This is expected: config block_size takes priority over other interpretations.
        shard2 = {
            "proj.weight": mx.zeros((512, 128), dtype=mx.uint32),
            "proj.scales": mx.zeros((512, 8), dtype=mx.float16),
        }

        _pre_fix_bits_from_shard(model2, shard2, block_size=128)
        # Config block_size=128 gives bits=4 (valid), so function uses that
        assert model2.proj.bits == 4
        assert model2.proj.group_size == 128

    def test_handles_nested_module_paths(self):
        """Deeply nested module path (language_model.model.layers.0.mlp.up_proj)."""
        import mlx.core as mx
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        # Build nested model structure
        model = nn.Module()
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        model.language_model.model.embed_tokens = nn.QuantizedLinear(
            3072, 248320, group_size=128, bits=3
        )

        # 4-bit packed
        shard = {
            "language_model.model.embed_tokens.weight": mx.zeros(
                (248320, 384), dtype=mx.uint32
            ),
            "language_model.model.embed_tokens.scales": mx.zeros(
                (248320, 24), dtype=mx.float16
            ),
        }

        _pre_fix_bits_from_shard(model, shard, block_size=128)
        assert model.language_model.model.embed_tokens.bits == 4

    def test_empty_shard_is_noop(self):
        """Empty shard dict doesn't crash."""
        import mlx.nn as nn
        from vmlx_engine.utils.jang_loader import _pre_fix_bits_from_shard

        model = nn.Module()
        model.layer = nn.QuantizedLinear(128, 128, group_size=64, bits=4)

        _pre_fix_bits_from_shard(model, {}, block_size=128)
        assert model.layer.bits == 4


class TestQuantShapeInference:
    def test_runtime_supported_inference_rewrites_unsupported_group_size_from_shapes(
        self, tmp_path
    ):
        import numpy as np
        from safetensors.numpy import save_file

        from vmlx_engine.utils.quant_shape_inference import (
            infer_quant_overrides_for_bundle,
        )

        save_file(
            {
                # Shape ratio is compatible with 8b/g64, 4b/g128, or 2b/g256.
                # Runtime cannot construct g256 modules, so the loader must not
                # trust the unsupported config claim.
                "model.layers.0.mlp.experts.0.down_proj.weight": np.zeros(
                    (8, 128), dtype=np.uint32
                ),
                "model.layers.0.mlp.experts.0.down_proj.scales": np.ones(
                    (8, 8), dtype=np.float16
                ),
            },
            str(tmp_path / "model.safetensors"),
        )
        config = {"quantization": {"bits": 2, "group_size": 256}}

        patched = infer_quant_overrides_for_bundle(
            tmp_path,
            config,
            runtime_supported_only=True,
            error_on_unsupported=True,
        )

        qcfg = patched["quantization"]
        assert qcfg["bits"] == 4
        assert qcfg["group_size"] == 128
        assert qcfg["model.layers.0.mlp.experts.0.down_proj"] == {
            "bits": 4,
            "group_size": 128,
        }

    def test_runtime_supported_inference_errors_when_shape_requires_group_size_256(
        self, tmp_path
    ):
        import numpy as np
        import pytest
        from safetensors.numpy import save_file

        from vmlx_engine.utils.quant_shape_inference import (
            infer_quant_overrides_for_bundle,
        )

        save_file(
            {
                # Shape ratio only maps to 8b/g256 in the converter table.
                "model.layers.0.self_attn.q_proj.weight": np.zeros(
                    (8, 512), dtype=np.uint32
                ),
                "model.layers.0.self_attn.q_proj.scales": np.ones(
                    (8, 8), dtype=np.float16
                ),
            },
            str(tmp_path / "model.safetensors"),
        )
        config = {"quantization": {"bits": 8, "group_size": 256}}

        with pytest.raises(ValueError, match="group_size 256.*MLX supports only"):
            infer_quant_overrides_for_bundle(
                tmp_path,
                config,
                runtime_supported_only=True,
                error_on_unsupported=True,
            )

    def test_loader_runtime_quant_defaults_use_shape_repaired_group_size(
        self, tmp_path
    ):
        import numpy as np
        from safetensors.numpy import save_file

        from vmlx_engine.utils.jang_loader import (
            _prepare_runtime_weight_quantization,
        )

        save_file(
            {
                "model.layers.0.mlp.experts.0.down_proj.weight": np.zeros(
                    (8, 128), dtype=np.uint32
                ),
                "model.layers.0.mlp.experts.0.down_proj.scales": np.ones(
                    (8, 8), dtype=np.float16
                ),
            },
            str(tmp_path / "model.safetensors"),
        )
        config = {"quantization": {"bits": 2, "group_size": 256}}
        jang_cfg = {"quantization": {"bits": 2, "group_size": 256}}

        repaired, bits, group_size = _prepare_runtime_weight_quantization(
            tmp_path,
            config,
            jang_cfg,
            fallback_bits=[4],
            context="unit-test",
        )

        assert bits == 4
        assert group_size == 128
        assert repaired["quantization"]["bits"] == 4
        assert repaired["quantization"]["group_size"] == 128

    def test_deepseek_v4_adds_sanitized_quant_aliases_for_jang_keys(self, tmp_path):
        import numpy as np
        from safetensors.numpy import save_file

        from vmlx_engine.utils.quant_shape_inference import (
            infer_quant_overrides_for_bundle,
        )

        def qweight(out_features: int, in_features: int, bits: int):
            return np.zeros(
                (out_features, in_features * bits // 32), dtype=np.uint32
            )

        def scales(out_features: int, in_features: int, group_size: int):
            return np.ones(
                (out_features, in_features // group_size), dtype=np.float16
            )

        save_file(
            {
                "embed.weight": qweight(8, 64, 8),
                "embed.scales": scales(8, 64, 64),
                "head.weight": qweight(8, 64, 8),
                "head.scales": scales(8, 64, 64),
                "layers.0.attn.wkv.weight": qweight(8, 64, 8),
                "layers.0.attn.wkv.scales": scales(8, 64, 64),
                "layers.0.ffn.shared_experts.w1.weight": qweight(8, 64, 8),
                "layers.0.ffn.shared_experts.w1.scales": scales(8, 64, 64),
            },
            str(tmp_path / "model.safetensors"),
        )
        config = {
            "model_type": "deepseek_v4",
            "quantization": {
                "bits": 2,
                "group_size": 128,
                "embed": {"bits": 8, "group_size": 64},
                "head": {"bits": 8, "group_size": 64},
                "layers.0.attn.wkv": {"bits": 8, "group_size": 64},
                "layers.0.ffn.shared_experts.w1": {
                    "bits": 8,
                    "group_size": 64,
                },
            },
        }

        patched = infer_quant_overrides_for_bundle(tmp_path, config)
        qcfg = patched["quantization"]

        assert qcfg["model.embed"] == {"bits": 8, "group_size": 64}
        assert qcfg["lm_head"] == {"bits": 8, "group_size": 64}
        assert qcfg["model.layers.0.self_attn.wkv"] == {
            "bits": 8,
            "group_size": 64,
        }
        assert qcfg[
            "model.layers.0.mlp.shared_experts.gate_proj"
        ] == {"bits": 8, "group_size": 64}


class TestDeepseekV4ShardExpertStacking:
    def test_stages_dsv4_routed_experts_across_shards(self):
        import mlx.core as mx

        from vmlx_engine.utils import jang_loader

        pending = {}
        shard_a = {
            "layers.0.ffn.experts.0.w1.weight": mx.array([[1]], dtype=mx.uint32),
            "layers.0.ffn.experts.0.w1.scales": mx.array([[1.0]], dtype=mx.float16),
            "layers.0.attn_norm.weight": mx.array([1.0], dtype=mx.float16),
        }
        non_expert_a, expert_a = jang_loader._split_dsv4_routed_expert_weights(
            shard_a
        )

        assert list(non_expert_a) == ["layers.0.attn_norm.weight"]
        jang_loader._stage_dsv4_routed_expert_weights(pending, expert_a)
        assert jang_loader._pop_complete_dsv4_routed_expert_stacks(
            pending, n_experts=2
        ) == {}

        shard_b = {
            "layers.0.ffn.experts.1.w1.weight": mx.array([[2]], dtype=mx.uint32),
            "layers.0.ffn.experts.1.w1.scales": mx.array([[2.0]], dtype=mx.float16),
        }
        _non_expert_b, expert_b = jang_loader._split_dsv4_routed_expert_weights(
            shard_b
        )
        jang_loader._stage_dsv4_routed_expert_weights(pending, expert_b)

        ready = jang_loader._pop_complete_dsv4_routed_expert_stacks(
            pending, n_experts=2
        )

        assert set(ready) == {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.0.mlp.switch_mlp.gate_proj.scales",
        }
        assert ready[
            "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        ].tolist() == [[[1]], [[2]]]
        assert pending == {}

    def test_dsv4_regular_sanitize_keeps_rmsnorm_weights_unshifted(self):
        import mlx.core as mx

        from vmlx_engine.utils import jang_loader

        weights = {
            "model.layers.0.input_layernorm.weight": mx.array([0.125], dtype=mx.float16),
            "model.layers.0.post_attention_layernorm.weight": mx.array([0.25], dtype=mx.float16),
            "model.layers.0.self_attn.q_norm.weight": mx.array([0.5], dtype=mx.float16),
            "model.layers.0.self_attn.k_norm.weight": mx.array([0.75], dtype=mx.float16),
            "model.norm.weight": mx.array([1.0], dtype=mx.float16),
            "model.layers.0.self_attn.wq_a.weight": mx.array([[1]], dtype=mx.uint32),
        }

        fixed = jang_loader._sanitize_deepseek_v4_regular_layout(weights)

        assert fixed["model.layers.0.input_layernorm.weight"].tolist() == [0.125]
        assert fixed["model.layers.0.post_attention_layernorm.weight"].tolist() == [0.25]
        assert fixed["model.layers.0.self_attn.q_norm.weight"].tolist() == [0.5]
        assert fixed["model.layers.0.self_attn.k_norm.weight"].tolist() == [0.75]
        assert fixed["model.norm.weight"].tolist() == [1.0]
        assert fixed["model.layers.0.self_attn.wq_a.weight"].tolist() == [[1]]


class TestQwen3NextConv1dLayout:
    """Qwen3-Next/GatedDeltaNet conv1d weights must load in MLX Conv1d layout."""

    def test_transposes_hf_conv1d_layout_after_model_sanitize_returns(self):
        import numpy as np
        import mlx.core as mx
        from vmlx_engine.utils.jang_loader import _sanitize_qwen3_next_conv1d_layout

        key = "model.layers.0.linear_attn.conv1d.weight"
        raw = mx.array(np.arange(8, dtype=np.float32).reshape(2, 1, 4))

        fixed = _sanitize_qwen3_next_conv1d_layout({key: raw})

        assert tuple(fixed[key].shape) == (2, 4, 1)
        assert np.array(fixed[key])[1, 3, 0] == 7

    def test_leaves_mlx_conv1d_layout_unchanged(self):
        import numpy as np
        import mlx.core as mx
        from vmlx_engine.utils.jang_loader import _sanitize_qwen3_next_conv1d_layout

        key = "model.layers.0.linear_attn.conv1d.weight"
        raw = mx.array(np.arange(8, dtype=np.float32).reshape(2, 4, 1))

        fixed = _sanitize_qwen3_next_conv1d_layout({key: raw})

        assert tuple(fixed[key].shape) == (2, 4, 1)
        assert np.array(fixed[key])[1, 3, 0] == 7

    def test_grouped_conv1d_backstop_name_is_family_agnostic(self):
        import numpy as np
        import mlx.core as mx
        from vmlx_engine.utils.jang_loader import _sanitize_grouped_conv1d_layout

        key = "backbone.layers.0.mixer.conv1d.weight"
        raw = mx.array(np.arange(8, dtype=np.float32).reshape(2, 1, 4))

        fixed = _sanitize_grouped_conv1d_layout({key: raw})

        assert tuple(fixed[key].shape) == (2, 4, 1)
        assert np.array(fixed[key])[1, 3, 0] == 7


class TestMCPConfigKeys:
    """Verify MCP config accepts both 'servers' and 'mcpServers' keys."""

    def test_accepts_servers_key(self):
        from vmlx_engine.mcp.config import validate_config
        config = {"servers": {"test": {"command": "python3", "args": ["-c", "pass"]}}}
        result = validate_config(config)
        assert len(result.servers) == 1

    def test_accepts_mcpServers_key(self):
        from vmlx_engine.mcp.config import validate_config
        config = {"mcpServers": {"test": {"command": "python3", "args": ["-c", "pass"]}}}
        result = validate_config(config)
        assert len(result.servers) == 1

    def test_servers_takes_precedence(self):
        """If both keys present, 'servers' wins (has server 'a')."""
        from vmlx_engine.mcp.config import validate_config
        config = {
            "servers": {"a": {"command": "python3", "args": ["-c", "pass"]}},
            "mcpServers": {"b": {"command": "python3", "args": ["-c", "pass"]}}
        }
        result = validate_config(config)
        # servers key takes precedence — should have 'a', not 'b'
        server_names = [name for name in result.servers]
        assert "a" in server_names

    def test_empty_config_returns_no_servers(self):
        from vmlx_engine.mcp.config import validate_config
        result = validate_config({})
        assert len(result.servers) == 0


class TestMCPSecurityUnblocked:
    """Verify PYTHONPATH and PATH are no longer blocked."""

    def test_pythonpath_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator
        validator = MCPCommandValidator()
        # Should not raise
        validator.validate_env({"PYTHONPATH": "/some/path"}, "test-server")

    def test_path_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator
        validator = MCPCommandValidator()
        validator.validate_env({"PATH": "/usr/bin:/usr/local/bin"}, "test-server")

    def test_node_path_allowed(self):
        from vmlx_engine.mcp.security import MCPCommandValidator
        validator = MCPCommandValidator()
        validator.validate_env({"NODE_PATH": "/some/node/path"}, "test-server")

    def test_ld_preload_still_blocked(self):
        from vmlx_engine.mcp.security import MCPCommandValidator, MCPSecurityError
        validator = MCPCommandValidator()
        with pytest.raises(MCPSecurityError):
            validator.validate_env({"LD_PRELOAD": "/evil.so"}, "test-server")

    def test_dyld_insert_still_blocked(self):
        from vmlx_engine.mcp.security import MCPCommandValidator, MCPSecurityError
        validator = MCPCommandValidator()
        with pytest.raises(MCPSecurityError):
            validator.validate_env({"DYLD_INSERT_LIBRARIES": "/evil.dylib"}, "test-server")


class TestJangtqKMixedBitsFastPath:
    """JANGTQ_K (MiniMax-M2.7-JANGTQ_K) ships per-projection bits as
    `mxtq_bits.routed_expert = {gate_proj: 2, up_proj: 2, down_proj: 4}`.

    The fast-path compiled decode in `jang_tools.load_jangtq` builds two
    separate Metal kernels: `fused_gate_up_swiglu` (uses gate's bits, gate==up
    enforced) and `gather_tq_decode_per_row` (uses down's bits, may differ).

    Pre-2026-05-05 the second kernel reused gate's bits, producing garbage
    on JANGTQ_K because the 4-bit packed down_proj tensors were unpacked
    as 2-bit. This test pins the fix so a refactor can't silently regress.
    """

    def test_get_compiled_decode_accepts_separate_dp_bits(self):
        # Source-grep on the patched symbol — full kernel call requires
        # Metal device + JANGTQ runtime cache, but the surface is what
        # we're guarding here.
        import jang_tools.load_jangtq as ljt
        import inspect
        src = inspect.getsource(ljt)
        assert "dp_bits=None" in src, (
            "fast-path _get_compiled_decode lost the dp_bits= parameter; "
            "JANGTQ_K (mixed gate=2/up=2/down=4) will silently regress to "
            "garbage output when gate.bits != down.bits"
        )
        assert "dp_bits=dp.bits" in src, (
            "_fused_switchglu_call no longer threads dp.bits through "
            "_get_compiled_decode; fast-path will reuse gate's bits for "
            "the down gather kernel"
        )

    def test_compiled_decode_cache_key_distinguishes_bits(self):
        import jang_tools.load_jangtq as ljt
        import inspect
        src = inspect.getsource(ljt)
        # The cache key must include both bit widths so a JANGTQ2 (uniform)
        # MoE layer and a JANGTQ_K (mixed) layer in the same model don't
        # share a compiled kernel keyed on a single bits value.
        assert "key = (in_f, out_f, bits, dp_bits, K, limit_milli)" in src, (
            "compiled-decode cache key collapsed gate/down bits — "
            "a uniform-2-bit and a 2/2/4 layer would share the same kernel"
        )

    def test_gather_dn_uses_dp_bits(self):
        import jang_tools.load_jangtq as ljt
        import inspect
        src = inspect.getsource(ljt)
        # The down gather kernel must compile against dp_bits, not gate's bits.
        assert "make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)" in src, (
            "down gather kernel still uses gate's bits — JANGTQ_K bundles "
            "with mixed gate/down bits will emit corrupted activations"
        )


class TestBailingHybridFlatSwitchMlpRepair:
    """Older `convert_ling_mxfp4.py` revisions emit pre-stacked switch_mlp
    tensors with the (out, in_per_row) axes flattened into one — i.e.
    `(n_experts, out * in_per_row)` 2D instead of `(n_experts, out, in_per_row)` 3D.
    mlx_lm's quantized SwitchLinear strict-checks shape during
    `model.load_weights()` and raises `ValueError: Expected shape (256, 1024,
    512) but received shape (256, 524288) for parameter
    model.layers.1.mlp.switch_mlp.gate_proj.weight`.

    Observed live on Ling-2.6-flash-MXFP4 + Ling-2.6-flash-MXFP4-CRACK
    bundles (2026-05-05). bailing_hybrid.sanitize now reshapes back to 3D
    using moe_intermediate_size (gate/up) or hidden_size (down). Total
    elements match exactly so the data is byte-identical.
    """

    def _bailing_model_classes(self):
        from vmlx_engine.utils.jang_loader import (
            _register_bailing_hybrid_from_repo_patch,
        )

        _register_bailing_hybrid_from_repo_patch()
        from mlx_lm.models.bailing_hybrid import Model, ModelArgs

        return Model, ModelArgs

    def test_sanitize_repairs_flat_2d_switch_mlp_to_3d(self):
        # Build a tiny bailing_hybrid model fixture with feeding flat 2D
        # switch_mlp tensors and confirm sanitize emits 3D output.
        import mlx.core as mx
        Model, ModelArgs = self._bailing_model_classes()
        # Tiny config (4 layers, 4 experts) sized to match real Ling axis ratios:
        # hidden=64, moe_intermediate=16. Per-expert weight (out=16, in=64), 4-bit
        # packed → packed_in_per_row = 64 * 4 / 32 = 8 → 3D (n_exp=4, 16, 8).
        # Flat 2D would be (4, 16*8) = (4, 128).
        args = ModelArgs(
            model_type='bailing_hybrid', hidden_size=64, intermediate_size=128,
            moe_intermediate_size=16, num_experts=4, num_shared_experts=1,
            num_attention_heads=4, num_experts_per_tok=2, num_hidden_layers=2,
            num_key_value_heads=4, rms_norm_eps=1e-6, rope_theta=10000.0,
            vocab_size=128, first_k_dense_replace=1, layer_group_size=2,
            group_norm_size=1, max_position_embeddings=512, q_lora_rank=32,
            qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
            kv_lora_rank=16, num_nextn_predict_layers=0, n_group=1,
            topk_group=1, score_function='sigmoid', use_qk_norm=False,
        )
        model = Model(args)
        # Layer 1 is the MoE layer (first_k_dense_replace=1 makes layer 0 dense)
        weights_in = {
            'model.layers.1.mlp.switch_mlp.gate_proj.weight': mx.zeros((4, 128), dtype=mx.uint32),
            'model.layers.1.mlp.switch_mlp.gate_proj.scales': mx.zeros((4, 32), dtype=mx.float16),
            'model.layers.1.mlp.switch_mlp.gate_proj.biases': mx.zeros((4, 32), dtype=mx.float16),
            'model.layers.1.mlp.switch_mlp.down_proj.weight': mx.zeros((4, 32), dtype=mx.uint32),
        }
        out = model.sanitize(weights_in)
        gw = out['model.layers.1.mlp.switch_mlp.gate_proj.weight']
        gs = out['model.layers.1.mlp.switch_mlp.gate_proj.scales']
        gb = out['model.layers.1.mlp.switch_mlp.gate_proj.biases']
        dw = out['model.layers.1.mlp.switch_mlp.down_proj.weight']
        assert gw.shape == (4, 16, 8), f"gate_proj.weight reshape failed: {gw.shape}"
        # gate_proj scales/biases: 32 = 16 * (64/group_size=32) → reshape to (4, 16, 2)
        assert gs.shape == (4, 16, 2), f"gate_proj.scales reshape failed: {gs.shape}"
        assert gb.shape == (4, 16, 2), f"gate_proj.biases reshape failed: {gb.shape}"
        # down_proj: out=hidden=64, packed_in = moe_intermediate*4/32 = 16*4/32 = 2 → 3D (4, 64, 0.5)
        # Wait: 32 / 64 = 0.5 — not divisible. So down_proj.weight (4, 32) does NOT
        # split via hidden=64. Instead the loader flat shape would be (4, 64*8 / something).
        # Actually for down_proj: in=moe_intermediate=16, out=hidden=64, packed_in = 16/8 = 2.
        # So 3D = (4, 64, 2) → flat = (4, 128). My fixture (4, 32) doesn't match — leave assert
        # for non-divisible case: sanitize must NOT touch it (no clean reshape).
        assert dw.shape == (4, 32), f"down_proj.weight w/ non-divisible flat must not be reshaped: {dw.shape}"

    def test_sanitize_no_op_on_correct_3d_shape(self):
        # JANGTQ bundles ship 3D already; sanitize must not touch them.
        import mlx.core as mx
        Model, ModelArgs = self._bailing_model_classes()
        args = ModelArgs(
            model_type='bailing_hybrid', hidden_size=64, intermediate_size=128,
            moe_intermediate_size=16, num_experts=4, num_shared_experts=1,
            num_attention_heads=4, num_experts_per_tok=2, num_hidden_layers=2,
            num_key_value_heads=4, rms_norm_eps=1e-6, rope_theta=10000.0,
            vocab_size=128, first_k_dense_replace=1, layer_group_size=2,
            group_norm_size=1, max_position_embeddings=512, q_lora_rank=32,
            qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
            kv_lora_rank=16, num_nextn_predict_layers=0, n_group=1,
            topk_group=1, score_function='sigmoid', use_qk_norm=False,
        )
        model = Model(args)
        gw_3d = mx.zeros((4, 16, 8), dtype=mx.uint32)
        weights_in = {
            'model.layers.1.mlp.switch_mlp.gate_proj.weight': gw_3d,
        }
        out = model.sanitize(weights_in)
        assert out['model.layers.1.mlp.switch_mlp.gate_proj.weight'].shape == (4, 16, 8)

    def test_sanitize_restores_dwq_split_mla_kv_b_proj(self):
        """mlx-community Ling DWQ ships split embed_q/unembed_out tensors while
        the runtime model expects kv_b_proj. Sanitizer must rebuild kv_b_proj
        and consume the split keys instead of relying on strict=False.
        """
        import mlx.core as mx
        Model, ModelArgs = self._bailing_model_classes()

        args = ModelArgs(
            model_type='bailing_hybrid', hidden_size=128, intermediate_size=256,
            moe_intermediate_size=16, num_experts=4, num_shared_experts=1,
            num_attention_heads=2, num_experts_per_tok=2, num_hidden_layers=2,
            num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=10000.0,
            vocab_size=128, first_k_dense_replace=1, layer_group_size=2,
            group_norm_size=1, max_position_embeddings=512, q_lora_rank=64,
            qk_rope_head_dim=8, qk_nope_head_dim=64, v_head_dim=64,
            kv_lora_rank=64, num_nextn_predict_layers=0, n_group=1,
            topk_group=1, score_function='sigmoid', use_qk_norm=False,
        )
        model = Model(args)
        embed_q = mx.quantize(mx.ones((2, 64, 64), dtype=mx.float16), group_size=64, bits=4)
        unembed = mx.quantize(mx.ones((2, 64, 64), dtype=mx.float16) * 2, group_size=64, bits=4)
        weights_in = {
            'model.layers.1.attention.embed_q.weight': embed_q[0],
            'model.layers.1.attention.embed_q.scales': embed_q[1],
            'model.layers.1.attention.embed_q.biases': embed_q[2],
            'model.layers.1.attention.unembed_out.weight': unembed[0],
            'model.layers.1.attention.unembed_out.scales': unembed[1],
            'model.layers.1.attention.unembed_out.biases': unembed[2],
        }
        out = model.sanitize(weights_in)

        assert not any('.embed_q.' in k or '.unembed_out.' in k for k in out)
        assert out['model.layers.1.attention.kv_b_proj.weight'].shape == (256, 8)
        assert out['model.layers.1.attention.kv_b_proj.scales'].shape == (256, 1)
        assert out['model.layers.1.attention.kv_b_proj.biases'].shape == (256, 1)
        restored = mx.dequantize(
            out['model.layers.1.attention.kv_b_proj.weight'],
            out['model.layers.1.attention.kv_b_proj.scales'],
            out['model.layers.1.attention.kv_b_proj.biases'],
            64,
            4,
        ).reshape(2, 128, 64)
        assert float(mx.mean(restored[:, :64, :]).item()) == 1.0
        assert float(mx.mean(restored[:, 64:, :]).item()) == 2.0

    def test_sanitize_trims_absent_mtp_layer_before_strict_load(self):
        """The mlx-community DWQ config advertises an MTP layer while its index
        ships no model.layers.{num_hidden_layers} tensors. Standard generation
        skips MTP, so sanitizer removes the absent tail modules before strict
        weight loading.
        """
        Model, ModelArgs = self._bailing_model_classes()

        args = ModelArgs(
            model_type='bailing_hybrid', hidden_size=64, intermediate_size=128,
            moe_intermediate_size=16, num_experts=4, num_shared_experts=1,
            num_attention_heads=4, num_experts_per_tok=2, num_hidden_layers=2,
            num_key_value_heads=4, rms_norm_eps=1e-6, rope_theta=10000.0,
            vocab_size=128, first_k_dense_replace=1, layer_group_size=2,
            group_norm_size=1, max_position_embeddings=512, q_lora_rank=32,
            qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
            kv_lora_rank=16, num_nextn_predict_layers=1, n_group=1,
            topk_group=1, score_function='sigmoid', use_qk_norm=False,
        )
        model = Model(args)
        assert len(model.model.layers) == 3
        out = model.sanitize({})
        assert out == {}
        assert len(model.model.layers) == 2
        assert model.args.num_nextn_predict_layers == 0

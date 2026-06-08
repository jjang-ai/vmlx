# SPDX-License-Identifier: Apache-2.0
"""Native MTP autodetect and runtime activation regression pins."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vmlx_engine.request import SamplingParams


def _write_qwen36_jang_mtp_bundle(path):
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5_vl"},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 64,
                    "mtp_num_hidden_layers": 1,
                },
            }
        )
    )
    (path / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "runtime": {
                    "bundle_has_mtp": True,
                    "mtp_layers": 1,
                    "mtp_mode": "preserved_enabled",
                },
                "mtp": {
                    "kept": True,
                    "enabled": True,
                    "num_layers": 1,
                    "tensor_count": 15,
                },
                "capabilities": {
                    "family": "qwen3_5",
                    "modality": "vision",
                    "cache_type": "hybrid",
                },
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.visual.patch_embed.proj.weight": "model-00001-of-00001.safetensors",
                    "mtp.fc.weight": "model-00001-of-00001.safetensors",
                    "mtp.fc.scales": "model-00001-of-00001.safetensors",
                    "mtp.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                    "mtp.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
                    "mtp.norm.weight": "model-00001-of-00001.safetensors",
                }
            }
        )
    )


def _write_qwen36_mxfp4_mtp_bundle(path):
    _write_qwen36_jang_mtp_bundle(path)

    config = json.loads((path / "config.json").read_text())
    config["weight_format"] = "mxfp4"
    config["quantization"] = {"mode": "mxfp4", "bits": 4, "group_size": 32}
    (path / "config.json").write_text(json.dumps(config))

    jang_config = json.loads((path / "jang_config.json").read_text())
    jang_config["weight_format"] = "mxfp4"
    jang_config["mtp"] = {
        "kept": True,
        "enabled": True,
        "num_layers": 1,
        "source_tensor_count": 15,
        "runtime_tensor_count": 23,
    }
    jang_config["quantization"] = {
        "method": "mxfp4",
        "mode": "mxfp4",
        "bits": 4,
        "group_size": 32,
        "mtp_policy": "native_mxfp4_linears_fp16_norms",
    }
    (path / "jang_config.json").write_text(json.dumps(jang_config))

    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                    "language_model.model.embed_tokens.scales": "model-00001-of-00004.safetensors",
                    "vision_tower.patch_embed.proj.weight": "model-00001-of-00004.safetensors",
                    "mtp.fc.weight": "model-00004-of-00004.safetensors",
                    "mtp.fc.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.input_layernorm.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.q_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.k_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.k_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.v_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.v_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.o_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.o_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.q_norm.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.self_attn.k_norm.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.gate_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.gate_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.up_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.up_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.mlp.down_proj.scales": "model-00004-of-00004.safetensors",
                    "mtp.layers.0.post_attention_layernorm.weight": "model-00004-of-00004.safetensors",
                    "mtp.norm.weight": "model-00004-of-00004.safetensors",
                    "mtp.pre_fc_norm_embedding.weight": "model-00004-of-00004.safetensors",
                    "mtp.pre_fc_norm_hidden.weight": "model-00004-of-00004.safetensors",
                }
            }
        )
    )


def _write_qwen36_mxfp8_mtp_bundle(path):
    _write_qwen36_mxfp4_mtp_bundle(path)

    config = json.loads((path / "config.json").read_text())
    config["weight_format"] = "mxfp8"
    config["quantization"] = {"mode": "mxfp8", "bits": 8, "group_size": 32}
    (path / "config.json").write_text(json.dumps(config))

    jang_config = json.loads((path / "jang_config.json").read_text())
    jang_config["weight_format"] = "mxfp8"
    jang_config["quantization"] = {
        "method": "mxfp8",
        "mode": "mxfp8",
        "bits": 8,
        "group_size": 32,
        "mtp_policy": "native_mxfp8_linears_fp16_norms",
    }
    (path / "jang_config.json").write_text(json.dumps(jang_config))


def _write_qwen36_jang2k_mtp_bundle(path):
    _write_qwen36_jang_mtp_bundle(path)

    config = json.loads((path / "config.json").read_text())
    config["model_type"] = "qwen3_5_moe"
    config["text_config"]["model_type"] = "qwen3_5_moe_text"
    config["quantization"] = {"bits": 2, "group_size": 128}
    (path / "config.json").write_text(json.dumps(config))

    jang_config = json.loads((path / "jang_config.json").read_text())
    jang_config["capabilities"]["family"] = "qwen3_5_moe"
    jang_config["quantization"] = {
        "method": "jang-importance",
        "profile": "JANG_2K",
        "target_bits": 2.0,
        "actual_bits": 2.11,
        "bit_widths_used": [2, 3, 6, 8],
    }
    (path / "jang_config.json").write_text(json.dumps(jang_config))


class TestNativeMtpAutodetect:
    def test_cli_exposes_native_mtp_runtime_flags(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "vmlx_engine.cli", "serve", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "--native-mtp-depth" in result.stdout
        assert "--native-mtp-sampling-policy" in result.stdout
        assert "--disable-native-mtp" in result.stdout

    def test_qwen36_nested_config_and_layered_tensors_are_native_ready(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        _write_qwen36_jang_mtp_bundle(tmp_path)

        status = _model_mtp_status(str(tmp_path))

        assert status["config_num_nextn_predict_layers"] == 1
        assert status["config_mtp_layer_source"] == "config.text_config.mtp_num_hidden_layers"
        assert status["jang_mtp_layers"] == 1
        assert status["index_has_mtp_tensors"] is True
        assert status["index_mtp_layer_count"] == 1
        assert status["mtp_tensor_count"] == 5
        assert status["artifact_available"] is True
        assert status["family"] == "qwen3_5"
        assert status["has_vision_config"] is True
        assert status["cache_type"] == "hybrid"
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is True
        assert status["runtime_scope"] == "text+vl"
        assert status["vl_runtime_available"] is True
        assert status["status"] == "native_runtime_ready"
        assert status["issues"] == []

    def test_qwen36_mxfp4_mtp_bundle_is_text_native_ready(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        status = _model_mtp_status(str(tmp_path))

        assert status["config_mtp_layer_source"] == "config.text_config.mtp_num_hidden_layers"
        assert status["config_num_nextn_predict_layers"] == 1
        assert status["index_has_mtp_tensors"] is True
        assert status["index_mtp_layer_count"] == 1
        assert status["mtp_tensor_count"] == 23
        assert status["family"] == "qwen3_5"
        assert status["has_vision_config"] is True
        assert status["cache_type"] == "hybrid"
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is True
        assert status["runtime_scope"] == "text+vl"
        assert status["vl_runtime_available"] is True
        assert status["has_vision_weights"] is True
        assert status["vision_tensor_count"] == 1
        assert status["status"] == "native_runtime_ready"
        assert status["issues"] == []

    def test_native_mtp_detection_uses_weights_not_path_name(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        model_dir = tmp_path / "Qwen3.6-27B-MXFP4-MTP"
        model_dir.mkdir()
        _write_qwen36_mxfp4_mtp_bundle(model_dir)

        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            key: value
            for key, value in index["weight_map"].items()
            if ".mtp." not in key and not key.startswith("mtp.")
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        status = _model_mtp_status(str(model_dir))

        assert status["artifact_available"] is False
        assert status["runtime_available"] is False
        assert status["vl_runtime_available"] is False
        assert status["status"] == "metadata_inconsistent"
        assert any("index has no mtp.* tensors" in issue for issue in status["issues"])

    def test_config_only_mtp_bundle_does_not_activate_native_runtime(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        _write_qwen36_jang_mtp_bundle(tmp_path)
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            key: value
            for key, value in index["weight_map"].items()
            if not key.startswith("mtp.")
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        status = _model_mtp_status(str(tmp_path))

        assert status["config_num_nextn_predict_layers"] == 1
        assert status["index_has_mtp_tensors"] is False
        assert status["mtp_tensor_count"] == 0
        assert status["artifact_available"] is False
        assert status["runtime_available"] is False
        assert status["runtime_active"] is False
        assert status["effective_depth"] is None
        assert status["status"] == "metadata_inconsistent"
        assert status["runtime_reason"] == "metadata_inconsistent"
        assert status["issues"] == [
            "config expects MTP next-token prediction layers, but the bundle "
            "index has no mtp.* tensors"
        ]

    def test_runtime_metadata_can_explicitly_drop_configured_mtp(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "hy_v3",
                    "num_nextn_predict_layers": 1,
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "weight_format": "affine",
                    "profile": "JANG_2K",
                    "runtime": {
                        "bundle_has_mtp": False,
                        "mtp_layers": 1,
                        "mtp_mode": "dropped_for_smallest_affine",
                    },
                    "capabilities": {
                        "family": "hy_v3",
                        "cache_type": "kv",
                    },
                }
            )
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    }
                }
            )
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["config_num_nextn_predict_layers"] == 1
        assert status["jang_mtp_layers"] == 1
        assert status["jang_drop_mtp"] is True
        assert status["index_has_mtp_tensors"] is False
        assert status["artifact_available"] is False
        assert status["runtime_available"] is False
        assert status["runtime_active"] is False
        assert status["status"] == "dropped"
        assert status["runtime_bundle_has_mtp"] is False
        assert status["runtime_mtp_mode"] == "dropped_for_smallest_affine"
        assert status["runtime_reason"] == "jang_config.runtime.bundle_has_mtp=false"
        assert status["issues"] == []

    def test_native_mtp_detection_scans_safetensors_headers_without_index(self, tmp_path):
        import numpy as np
        from safetensors.numpy import save_file

        from vmlx_engine.server import _model_mtp_status

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "vision_config": {"model_type": "qwen3_5_vl"},
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "mtp_num_hidden_layers": 1,
                    },
                }
            )
        )
        (tmp_path / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "runtime": {"mtp_layers": 1},
                    "capabilities": {"family": "qwen3_5", "cache_type": "hybrid"},
                }
            )
        )
        save_file(
            {
                "vision_tower.patch_embed.proj.weight": np.zeros((1, 1), dtype=np.float16),
                "mtp.fc.weight": np.zeros((1, 1), dtype=np.float16),
                "mtp.layers.0.self_attn.q_proj.weight": np.zeros((1, 1), dtype=np.float16),
                "mtp.norm.weight": np.zeros((1,), dtype=np.float16),
            },
            str(tmp_path / "model.safetensors"),
        )

        status = _model_mtp_status(str(tmp_path))

        assert status["index_has_mtp_tensors"] is True
        assert status["index_mtp_layer_count"] == 1
        assert status["runtime_scope"] == "text+vl"
        assert status["vl_runtime_available"] is True
        assert status["issues"] == []

    def test_vl_runtime_requires_real_vision_tensors(self, tmp_path):
        from vmlx_engine.server import _model_mtp_status

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            key: value
            for key, value in index["weight_map"].items()
            if "vision_tower" not in key and "model.visual" not in key
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        status = _model_mtp_status(str(tmp_path))

        assert status["artifact_available"] is True
        assert status["has_vision_config"] is True
        assert status["has_vision_weights"] is False
        assert status["runtime_scope"] == "text"
        assert status["vl_runtime_available"] is False

    def test_native_mtp_vl_artifact_overrides_affine_qwen_text_only_guard(
        self, tmp_path
    ):
        from vmlx_engine.api import utils

        _write_qwen36_jang_mtp_bundle(tmp_path)
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(tmp_path)) is True
        assert utils.is_mllm_model(str(tmp_path), force_mllm=True) is True

    def test_native_mtp_vl_artifact_keeps_vl_route_when_runtime_disabled(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.api import utils

        _write_qwen36_jang_mtp_bundle(tmp_path)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP", "0")
        utils._IS_MLLM_CACHE.clear()

        assert utils.is_mllm_model(str(tmp_path)) is True

    def test_jang2k_profile_blocks_native_mtp_runtime_but_keeps_vl_artifact_route(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.api import utils
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_ALLOW_JANG2K", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_ALLOW_JANG2K", raising=False)
        _write_qwen36_jang2k_mtp_bundle(tmp_path)
        utils._IS_MLLM_CACHE.clear()

        status = inspect_native_mtp_bundle(tmp_path)

        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is False
        assert status["runtime_validation_blocked"] is True
        assert status["status"] == "runtime_validation_blocked"
        assert "JANG_2K native MTP acceleration is blocked" in status["runtime_reason"]
        assert status["runtime_scope"] == "text+vl"
        assert utils.is_mllm_model(str(tmp_path)) is True

    def test_native_mtp_inspection_reuses_weight_key_scan_for_startup(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine import native_mtp

        _write_qwen36_jang_mtp_bundle(tmp_path)
        calls = []

        def fake_weight_keys(bundle_path):
            calls.append(str(bundle_path))
            return (
                [
                    "model.embed_tokens.weight",
                    "model.visual.patch_embed.proj.weight",
                    "mtp.fc.weight",
                    "mtp.layers.0.self_attn.q_proj.weight",
                    "mtp.norm.weight",
                ],
                "fake-index",
                None,
            )

        monkeypatch.setattr(native_mtp, "_bundle_weight_keys", fake_weight_keys)

        status = native_mtp.inspect_native_mtp_bundle(tmp_path)

        assert calls == [str(tmp_path)]
        assert status["artifact_available"] is True
        assert status["mtp_tensor_count"] == 3
        assert status["vision_tensor_count"] == 1
        assert status["index_mtp_layer_count"] == 1

    def test_jang2k_native_mtp_validation_block_has_explicit_override(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ALLOW_JANG2K", "1")
        _write_qwen36_jang2k_mtp_bundle(tmp_path)

        status = inspect_native_mtp_bundle(tmp_path)

        assert status["runtime_available"] is True
        assert status["runtime_validation_blocked"] is False
        assert status["status"] == "native_runtime_ready"

    def test_jang2k_native_mtp_patch_preserves_validation_block_reason(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine import native_mtp

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_ALLOW_JANG2K", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_ALLOW_JANG2K", raising=False)
        _write_qwen36_jang2k_mtp_bundle(tmp_path)
        calls = []
        monkeypatch.setattr(native_mtp, "_apply_mlx_lm_mtp_patch", lambda: True)
        monkeypatch.setattr(
            native_mtp,
            "_set_mtp_active",
            lambda active: calls.append(("active", active)),
        )

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert calls == [("active", False)]
        assert status["runtime_active"] is False
        assert status["runtime_available"] is False
        assert status["status"] == "runtime_validation_blocked"
        assert "JANG_2K native MTP acceleration is blocked" in status["runtime_reason"]

    def test_native_mtp_vl_loader_reaches_real_vlm_path_for_affine_qwen(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        class NativeVlmReached(RuntimeError):
            pass

        _write_qwen36_jang_mtp_bundle(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        jang_cfg = json.loads((tmp_path / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: (_ for _ in ()).throw(NativeVlmReached()),
        )
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: pytest.fail(
                "native MTP VL Qwen must not use text-only fallback"
            ),
        )

        with pytest.raises(NativeVlmReached):
            jang_loader._load_jang_v2_vlm(tmp_path, jang_cfg)
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is False

    def test_native_mtp_vl_loader_stays_vlm_when_runtime_disabled(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.utils import jang_loader
        import mlx_vlm.utils as vlm_utils

        class NativeVlmReached(RuntimeError):
            pass

        _write_qwen36_jang_mtp_bundle(tmp_path)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP", "0")
        config = json.loads((tmp_path / "config.json").read_text())
        jang_cfg = json.loads((tmp_path / "jang_config.json").read_text())

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            lambda *, config: (_ for _ in ()).throw(NativeVlmReached()),
        )
        monkeypatch.setattr(
            jang_loader,
            "_load_jang_v2",
            lambda *args, **kwargs: pytest.fail(
                "native MTP VL Qwen must keep the VLM path when runtime is disabled"
            ),
        )

        with pytest.raises(NativeVlmReached):
            jang_loader._load_jang_v2_vlm(tmp_path, jang_cfg)
        assert jang_loader._LAST_LOAD_VLM_FALLBACK is False

    def test_jang_vlm_processor_eos_resolver_uses_generation_config_and_text_config(
        self, tmp_path
    ):
        from vmlx_engine.utils.jang_loader import _resolve_vlm_processor_eos_token_ids

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {"eos_token_id": 248044},
                }
            )
        )
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [248046, 248044]})
        )
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=None))

        assert _resolve_vlm_processor_eos_token_ids(tmp_path, model) == [
            248046,
            248044,
        ]

    def test_jang_vlm_processor_eos_resolver_dedupes_model_and_bundle_ids(
        self, tmp_path
    ):
        from vmlx_engine.utils.jang_loader import _resolve_vlm_processor_eos_token_ids

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "eos_token_id": [248046, 248044],
                    "text_config": {"eos_token_id": 248044},
                }
            )
        )
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [248046, 248044]})
        )
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[248046]))

        assert _resolve_vlm_processor_eos_token_ids(tmp_path, model) == [
            248046,
            248044,
        ]

    @pytest.mark.parametrize(
        ("bundle_writer", "bits", "mode"),
        [
            (_write_qwen36_mxfp4_mtp_bundle, 4, "mxfp4"),
            (_write_qwen36_mxfp8_mtp_bundle, 8, "mxfp8"),
        ],
    )
    def test_mxfp_vlm_loader_quantizes_with_declared_mode(
        self, tmp_path, monkeypatch, bundle_writer, bits, mode
    ):
        import numpy as np
        from safetensors.numpy import save_file

        import mlx.nn as nn
        import mlx_vlm.utils as vlm_utils
        from vmlx_engine.utils import jang_loader

        class ReachedQuantize(RuntimeError):
            pass

        class FakeModelConfig:
            @classmethod
            def from_dict(cls, _config):
                return SimpleNamespace(
                    vision_config=SimpleNamespace(),
                    text_config=SimpleNamespace(),
                )

        class FakeModel:
            def __init__(self, _config):
                self.language_model = SimpleNamespace()
                fake_models.append(self)

        class FakeModelClass:
            ModelConfig = FakeModelConfig
            Model = FakeModel

        bundle_writer(tmp_path)
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            key: "model-00001-of-00001.safetensors"
            for key in index["weight_map"].keys()
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        save_file(
            {
                "language_model.model.embed_tokens.weight": np.zeros(
                    (2, 1), dtype=np.uint32
                ),
                "language_model.model.embed_tokens.scales": np.ones(
                    (2, 1), dtype=np.uint8
                ),
                "vision_tower.patch_embed.proj.weight": np.zeros(
                    (1, 1), dtype=np.float16
                ),
            },
            str(tmp_path / "model-00001-of-00001.safetensors"),
        )

        config = json.loads((tmp_path / "config.json").read_text())
        jang_cfg = json.loads((tmp_path / "jang_config.json").read_text())
        calls = []
        fake_models = []

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils, "get_model_and_args", lambda *, config: (FakeModelClass, {})
        )
        monkeypatch.setattr(
            vlm_utils,
            "update_module_configs",
            lambda model_config, *_args, **_kwargs: model_config,
        )

        def fake_quantize(_model, *, group_size, bits, mode="affine", **_kwargs):
            calls.append({"group_size": group_size, "bits": bits, "mode": mode})
            raise ReachedQuantize

        monkeypatch.setattr(nn, "quantize", fake_quantize)

        with pytest.raises(ReachedQuantize):
            jang_loader._load_jang_v2_vlm(tmp_path, jang_cfg, skip_eval=True)

        assert calls == [{"group_size": 32, "bits": bits, "mode": mode}]
        assert fake_models[0].language_model._vmlx_force_text_rope_1d is True

    def test_jang4m_vlm_loader_defaults_to_affine_quant_mode(self, tmp_path):
        from vmlx_engine.utils.jang_loader import _jang_quant_mode

        assert (
            _jang_quant_mode(
                {
                    "format": "jang",
                    "quantization": {
                        "method": "jang-importance",
                        "block_size": 64,
                        "bits": 4,
                    },
                },
                {"quantization": {"group_size": 64, "bits": 4}},
            )
            == "affine"
        )

    def test_jang_quant_mode_supports_mxfp8_metadata(self):
        from vmlx_engine.utils.jang_loader import _jang_quant_mode

        assert (
            _jang_quant_mode(
                {"weight_format": "mxfp8", "quantization": {"bits": 8}},
                {"quantization": {"group_size": 32, "bits": 8}},
            )
            == "mxfp8"
        )

    def test_uint32_post_load_upgrade_preserves_mxfp8_mode(self):
        import mlx.core as mx
        import mlx.nn as nn

        from vmlx_engine.utils.jang_loader import _upgrade_modules_with_uint32_weights

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 4, bias=False)
                self.proj.weight = mx.zeros((4, 8), dtype=mx.uint32)
                self.proj.scales = mx.ones((4, 1), dtype=mx.uint8)

        model = Tiny()

        upgraded = _upgrade_modules_with_uint32_weights(model, 8, 32, "mxfp8")

        assert upgraded == 1
        assert isinstance(model.proj, nn.QuantizedLinear)
        assert model.proj.mode == "mxfp8"
        assert model.proj.bits == 8
        assert model.proj.group_size == 32
        assert model.proj.biases is None

    def test_jang4m_native_mtp_vlm_uses_text_rope_for_text_rows(
        self, tmp_path, monkeypatch
    ):
        import numpy as np
        from safetensors.numpy import save_file

        import mlx.nn as nn
        import mlx_vlm.utils as vlm_utils
        from vmlx_engine.utils import jang_loader

        class ReachedQuantize(RuntimeError):
            pass

        class FakeModelConfig:
            @classmethod
            def from_dict(cls, _config):
                return SimpleNamespace(
                    vision_config=SimpleNamespace(),
                    text_config=SimpleNamespace(),
                )

        class FakeModel:
            def __init__(self, _config):
                self.language_model = SimpleNamespace()
                fake_models.append(self)

        class FakeModelClass:
            ModelConfig = FakeModelConfig
            Model = FakeModel

        _write_qwen36_jang_mtp_bundle(tmp_path)
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            key: "model-00001-of-00001.safetensors"
            for key in index["weight_map"].keys()
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        save_file(
            {
                "language_model.model.embed_tokens.weight": np.zeros(
                    (2, 1), dtype=np.uint32
                ),
                "language_model.model.embed_tokens.scales": np.ones(
                    (2, 1), dtype=np.float16
                ),
                "vision_tower.patch_embed.proj.weight": np.zeros(
                    (1, 1), dtype=np.float16
                ),
            },
            str(tmp_path / "model-00001-of-00001.safetensors"),
        )

        config = json.loads((tmp_path / "config.json").read_text())
        jang_cfg = json.loads((tmp_path / "jang_config.json").read_text())
        fake_models = []

        monkeypatch.setattr(vlm_utils, "load_config", lambda path: dict(config))
        monkeypatch.setattr(
            vlm_utils, "get_model_and_args", lambda *, config: (FakeModelClass, {})
        )
        monkeypatch.setattr(
            vlm_utils,
            "update_module_configs",
            lambda model_config, *_args, **_kwargs: model_config,
        )
        monkeypatch.setattr(
            nn,
            "quantize",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedQuantize()),
        )

        with pytest.raises(ReachedQuantize):
            jang_loader._load_jang_v2_vlm(tmp_path, jang_cfg, skip_eval=True)

        assert fake_models[0].language_model._vmlx_force_text_rope_1d is True

    def test_vlm_quant_candidates_map_sanitized_mtp_paths(self):
        from vmlx_engine.utils.jang_loader import _vlm_quant_module_path_candidates

        candidates = _vlm_quant_module_path_candidates("language_model.mtp.fc")

        assert "language_model.mtp.fc" in candidates
        assert "mtp.fc" in candidates
        assert "model.language_model.mtp.fc" in candidates

    def test_pre_load_activation_patches_mlx_vlm_qwen36_language_runtime(
        self, tmp_path
    ):
        import inspect

        import mlx.core as mx
        from mlx_vlm.models.qwen3_5 import qwen3_5 as qwen_vl
        from mlx_vlm.models.qwen3_5 import language as vl_language
        from mlx_vlm.models.qwen3_5_moe import language as moe_vl_language
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as qwen_moe_vl

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert status["runtime_active"] is True
        assert hasattr(vl_language.LanguageModel, "mtp_forward")
        assert hasattr(vl_language.LanguageModel, "make_mtp_cache")
        assert hasattr(qwen_vl.Model, "mtp_forward")
        assert hasattr(qwen_vl.Model, "make_mtp_cache")
        assert "n_confirmed" in inspect.signature(
            vl_language.Qwen3_5GatedDeltaNet.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            vl_language.Qwen3_5GatedDeltaNet.__call__
        ).parameters
        assert "n_confirmed" in inspect.signature(
            vl_language.Qwen3_5DecoderLayer.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            vl_language.Qwen3_5DecoderLayer.__call__
        ).parameters
        assert "n_confirmed" in inspect.signature(
            vl_language.Qwen3_5Model.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            vl_language.Qwen3_5Model.__call__
        ).parameters
        assert hasattr(moe_vl_language.LanguageModel, "mtp_forward")
        assert hasattr(moe_vl_language.LanguageModel, "make_mtp_cache")
        assert hasattr(qwen_moe_vl.Model, "mtp_forward")
        assert hasattr(qwen_moe_vl.Model, "make_mtp_cache")
        assert "n_confirmed" in inspect.signature(
            moe_vl_language.Qwen3_5MoeGatedDeltaNet.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            moe_vl_language.Qwen3_5MoeGatedDeltaNet.__call__
        ).parameters
        assert "n_confirmed" in inspect.signature(
            moe_vl_language.Qwen3_5MoeDecoderLayer.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            moe_vl_language.Qwen3_5MoeDecoderLayer.__call__
        ).parameters
        assert "n_confirmed" in inspect.signature(
            moe_vl_language.Qwen3_5MoeModel.__call__
        ).parameters
        assert "gdn_sink" in inspect.signature(
            moe_vl_language.Qwen3_5MoeModel.__call__
        ).parameters
        patched_gdn = inspect.getsource(vl_language.Qwen3_5GatedDeltaNet.__call__)
        patched_decoder = inspect.getsource(vl_language.Qwen3_5DecoderLayer.__call__)
        patched_model = inspect.getsource(vl_language.Qwen3_5Model.__call__)
        assert "gdn_sink.append" in patched_gdn
        assert "conv_input" in patched_gdn
        assert "gdn_sink=gdn_sink" in patched_decoder
        assert "gdn_sink=gdn_sink" in patched_model
        fake_attention = vl_language.Qwen3_5Attention.__new__(
            vl_language.Qwen3_5Attention
        )
        fake_attention.rotary_emb = SimpleNamespace(dim=4, base=10000.0)
        text_rope = vl_language.Qwen3_5Attention._vmlx_text_rope(fake_attention)
        rope_out = text_rope(mx.zeros((1, 1, 1, 4), dtype=mx.float16), offset=0)
        assert rope_out.shape == (1, 1, 1, 4)

        fake_model = qwen_vl.Model.__new__(qwen_vl.Model)
        fake_model.config = SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=True)
        )
        fake_model.language_model = SimpleNamespace(sanitize=lambda weights: weights)

        sanitized = qwen_vl.Model.sanitize(
            fake_model,
            {
                "mtp.fc.weight": mx.ones((2, 2), dtype=mx.float16),
                "mtp.fc.scales": mx.ones((2,), dtype=mx.uint8),
                "model.visual.patch_embed.proj.weight": mx.ones(
                    (1152, 3, 2, 16, 16), dtype=mx.float16
                ),
                "lm_head.weight": mx.ones((2, 2), dtype=mx.float16),
            },
        )

        assert "language_model.mtp.fc.weight" in sanitized
        assert "language_model.mtp.fc.scales" in sanitized
        assert "vision_tower.patch_embed.proj.weight" in sanitized
        assert sanitized["vision_tower.patch_embed.proj.weight"].shape == (
            1152,
            2,
            16,
            16,
            3,
        )
        assert "language_model.lm_head.weight" not in sanitized

        fake_moe_model = qwen_moe_vl.Model.__new__(qwen_moe_vl.Model)
        fake_moe_model.config = SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=True, num_hidden_layers=1)
        )
        fake_moe_model.language_model = SimpleNamespace(
            sanitize=lambda weights: weights
        )

        sanitized_moe = qwen_moe_vl.Model.sanitize(
            fake_moe_model,
            {
                "mtp.fc.weight": mx.ones((2, 2), dtype=mx.float16),
                "mtp.layers.0.mlp.switch_mlp.down_proj.weight": mx.ones(
                    (1, 1, 1), dtype=mx.float16
                ),
                "model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight": mx.ones(
                    (1, 1, 1), dtype=mx.float16
                ),
                "model.language_model.layers.0.mlp.switch_mlp.up_proj.weight": mx.ones(
                    (1, 1, 1), dtype=mx.float16
                ),
                "model.language_model.layers.0.mlp.switch_mlp.down_proj.weight": mx.ones(
                    (1, 1, 1), dtype=mx.float16
                ),
                "model.visual.patch_embed.proj.weight": mx.ones(
                    (1, 1), dtype=mx.float16
                ),
                "lm_head.weight": mx.ones((2, 2), dtype=mx.float16),
            },
        )

        assert "language_model.mtp.fc.weight" in sanitized_moe
        assert (
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"
            in sanitized_moe
        )
        assert (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
            in sanitized_moe
        )
        assert (
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"
            in sanitized_moe
        )
        assert (
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"
            in sanitized_moe
        )
        assert "vision_tower.patch_embed.proj.weight" in sanitized_moe
        assert "language_model.lm_head.weight" not in sanitized_moe

    def test_qwen36_vlm_mtp_gdn_sink_does_not_leak_to_upstream_originals(self):
        from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

        class DenseDecoder:
            def __call__(self, x, mask=None, cache=None, position_ids=None):
                return x + 10

        class MoeDecoder:
            def __call__(self, x, mask=None, cache=None, position_ids=None):
                return x + 20

        class QwenModel:
            def __call__(
                self,
                inputs,
                inputs_embeds=None,
                mask=None,
                cache=None,
                position_ids=None,
            ):
                return inputs + 30

        dense_lang = SimpleNamespace(Qwen3_5DecoderLayer=DenseDecoder)
        moe_lang = SimpleNamespace(Qwen3_5MoeDecoderLayer=MoeDecoder)

        def create_mask(_hidden, _cache):
            return "mask"

        model_lang = SimpleNamespace(
            Qwen3_5Model=QwenModel,
            create_attention_mask=create_mask,
            create_ssm_mask=create_mask,
        )

        qwen35_vl._patch_decoder_layer(dense_lang)
        qwen35_vl._patch_moe_decoder_layer(moe_lang)
        qwen35_vl._patch_qwen_model(model_lang)

        for layer_cls, original_value in ((DenseDecoder, 11), (MoeDecoder, 21)):
            layer = layer_cls()
            layer.is_linear = True
            layer.input_layernorm = lambda x: x
            layer.linear_attn = (
                lambda x, mask=None, cache=None, gdn_sink=None, n_confirmed=0: x + 1
            )
            layer.post_attention_layernorm = lambda x: x
            layer.mlp = lambda x: x + 2

            assert layer(1, gdn_sink=None, n_confirmed=0) == original_value
            assert layer(1, gdn_sink=[], n_confirmed=0) == 8

        model = QwenModel()
        model.embed_tokens = lambda inputs: inputs
        class ModelLayer:
            is_linear = True

            def __call__(self, h, *args, **kwargs):
                return h + 1

        model.layers = [ModelLayer()]
        model.fa_idx = 0
        model.ssm_idx = 0

        assert model(1, gdn_sink=None, n_confirmed=0) == 31
        assert model(1, cache=[None], gdn_sink=[], n_confirmed=0) == 2

    def test_qwen36_vlm_text_rope_fresh_prefill_uses_scalar_cache_offset(
        self, tmp_path
    ):
        import inspect

        from mlx_vlm.models.qwen3_5 import language as vl_language

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert status["runtime_active"] is True
        patched = inspect.getsource(vl_language.Qwen3_5Attention.__call__)
        assert "position_ids = qlang.mx.arange(cache.offset" not in patched
        assert "offset = cache.offset if cache is not None else 0" in patched

    def test_qwen36_vlm_gated_delta_normal_path_advances_cache(self, tmp_path):
        import inspect

        from mlx_vlm.models.qwen3_5 import language as vl_language

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert status["runtime_active"] is True
        patched = inspect.getsource(vl_language.Qwen3_5GatedDeltaNet.__call__)
        assert "if n_confirmed == 0:" not in patched
        assert "advance = getattr(cache, \"advance\", None)" in patched
        assert "advance(seq_len)" in patched

    def test_qwen36_vlm_gated_delta_kernel_has_diagnostic_disable_env(self, tmp_path):
        import inspect

        from mlx_vlm.models.qwen3_5 import language as vl_language

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert status["runtime_active"] is True
        from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

        helper = inspect.getsource(qwen35_vl._qwen_vlm_gated_delta_kernel_enabled)
        patched = inspect.getsource(vl_language.Qwen3_5GatedDeltaNet._process_chunk)
        assert "VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL" in helper
        assert "VMLX_QWEN_VLM_GATED_DELTA_KERNEL" in helper
        assert "use_kernel=_qwen_vlm_gated_delta_kernel_enabled(self)" in patched

    def test_qwen36_vlm_gated_delta_kernel_defaults_to_inference_fast_path(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        from vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl import (
            _qwen_vlm_gated_delta_kernel_enabled,
        )

        monkeypatch.delenv("VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL", raising=False)
        monkeypatch.delenv("VMLX_QWEN_VLM_GATED_DELTA_KERNEL", raising=False)

        assert _qwen_vlm_gated_delta_kernel_enabled(SimpleNamespace(training=False)) is True
        assert _qwen_vlm_gated_delta_kernel_enabled(SimpleNamespace(training=True)) is False

        monkeypatch.setenv("VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL", "1")
        assert _qwen_vlm_gated_delta_kernel_enabled(SimpleNamespace(training=True)) is True

        monkeypatch.setenv("VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL", "0")
        assert _qwen_vlm_gated_delta_kernel_enabled(SimpleNamespace(training=False)) is False

    def test_mxfp4_vlm_sanitize_shifts_mtp_norms_only(self, tmp_path):
        import mlx.core as mx

        from mlx_vlm.models.qwen3_5 import qwen3_5 as qwen_vl

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        fake_model = qwen_vl.Model.__new__(qwen_vl.Model)
        fake_model._vmlx_norms_are_mlx_ready = True
        fake_model.config = SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=False)
        )

        sanitized = qwen_vl.Model.sanitize(
            fake_model,
            {
                "model.language_model.layers.0.input_layernorm.weight": mx.zeros(
                    (2,), dtype=mx.float16
                ),
                "mtp.layers.0.input_layernorm.weight": mx.zeros(
                    (2,), dtype=mx.float16
                ),
                "mtp.layers.0.self_attn.q_norm.weight": mx.zeros(
                    (2,), dtype=mx.float16
                ),
                "mtp.pre_fc_norm_hidden.weight": mx.zeros(
                    (2,), dtype=mx.float16
                ),
                "mtp.norm.weight": mx.zeros((2,), dtype=mx.float16),
            },
        )

        assert sanitized[
            "language_model.model.layers.0.input_layernorm.weight"
        ].tolist() == [0.0, 0.0]
        assert sanitized["language_model.mtp.layers.0.input_layernorm.weight"].tolist() == [
            1.0,
            1.0,
        ]
        assert sanitized["language_model.mtp.layers.0.self_attn.q_norm.weight"].tolist() == [
            1.0,
            1.0,
        ]
        assert sanitized["language_model.mtp.pre_fc_norm_hidden.weight"].tolist() == [
            1.0,
            1.0,
        ]
        assert sanitized["language_model.mtp.norm.weight"].tolist() == [1.0, 1.0]

    def test_qwen36_vlm_sanitize_shifts_unshifted_jang_transformer_norms(
        self, tmp_path
    ):
        import mlx.core as mx

        from mlx_vlm.models.qwen3_5 import qwen3_5 as qwen_vl
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as qwen_moe_vl

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        for model_cls in (qwen_vl.Model, qwen_moe_vl.Model):
            fake_model = model_cls.__new__(model_cls)
            fake_model.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    tie_word_embeddings=False,
                    num_hidden_layers=1,
                )
            )

            sanitized = model_cls.sanitize(
                fake_model,
                {
                    "model.language_model.layers.0.input_layernorm.weight": mx.zeros(
                        (2,), dtype=mx.float16
                    ),
                    "model.language_model.layers.0.post_attention_layernorm.weight": mx.zeros(
                        (2,), dtype=mx.float16
                    ),
                },
            )

            assert sanitized[
                "language_model.model.layers.0.input_layernorm.weight"
            ].tolist() == [1.0, 1.0]
            assert sanitized[
                "language_model.model.layers.0.post_attention_layernorm.weight"
            ].tolist() == [1.0, 1.0]

    def test_qwen36_vlm_sanitize_does_not_shift_already_shifted_norm_shard(
        self, tmp_path
    ):
        import mlx.core as mx

        from mlx_vlm.models.qwen3_5 import qwen3_5 as qwen_vl

        from vmlx_engine import native_mtp

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        fake_model = qwen_vl.Model.__new__(qwen_vl.Model)
        fake_model.config = SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=1,
            )
        )

        sanitized = qwen_vl.Model.sanitize(
            fake_model,
            {
                "model.language_model.layers.0.input_layernorm.weight": mx.array(
                    [0.9375, 0.9375], dtype=mx.float16
                ),
                "model.language_model.norm.weight": mx.array(
                    [1.9, 1.9], dtype=mx.float16
                ),
            },
        )

        assert sanitized[
            "language_model.model.layers.0.input_layernorm.weight"
        ].tolist() == [0.9375, 0.9375]
        assert sanitized["language_model.model.norm.weight"].tolist() == [
            1.900390625,
            1.900390625,
        ]

    def test_bundle_index_layer_count_accepts_qwen_mtp_layers_layout(self, tmp_path):
        from vmlx_engine.server import _bundle_index_mtp_layer_count

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "mtp.fc.weight": "a",
                        "mtp.layers.0.self_attn.q_proj.weight": "a",
                        "mtp.layers.0.self_attn.k_proj.weight": "a",
                        "mtp.layers.0.mlp.down_proj.weight": "a",
                    }
                }
            )
        )

        assert _bundle_index_mtp_layer_count(str(tmp_path)) == 1

    def test_pre_load_activation_applies_patch_and_marks_runtime_active(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine import native_mtp

        _write_qwen36_jang_mtp_bundle(tmp_path)
        calls = []

        monkeypatch.setattr(native_mtp, "_apply_mlx_lm_mtp_patch", lambda: calls.append("apply") or True)
        monkeypatch.setattr(native_mtp, "_set_mtp_active", lambda active: calls.append(("active", active)))

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert calls == ["apply", ("active", True)]
        assert status["runtime_available"] is True
        assert status["runtime_active"] is True

    def test_pre_load_activation_can_preserve_sanitize_patch_without_runtime(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine import native_mtp

        _write_qwen36_jang_mtp_bundle(tmp_path)
        calls = []

        monkeypatch.setattr(native_mtp, "_apply_mlx_lm_mtp_patch", lambda: calls.append("apply") or True)
        monkeypatch.setattr(native_mtp, "_set_mtp_active", lambda active: calls.append(("active", active)))

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=False)

        assert calls == ["apply", ("active", False)]
        assert status["artifact_available"] is True
        assert status["runtime_available"] is False
        assert status["runtime_active"] is False
        assert status["status"] == "runtime_disabled"

    def test_pre_load_activation_failure_deactivates_stale_runtime_flag(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine import native_mtp

        _write_qwen36_jang_mtp_bundle(tmp_path)
        calls = []

        monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
        monkeypatch.setattr(native_mtp, "_apply_mlx_lm_mtp_patch", lambda: False)
        monkeypatch.setattr(
            native_mtp,
            "_set_mtp_active",
            lambda active: calls.append(("active", active)),
        )

        status = native_mtp.maybe_apply_native_mtp(str(tmp_path), allow_runtime=True)

        assert calls == [("active", False)]
        assert native_mtp._ACTIVE_NATIVE_MTP_MODEL_PATH is None
        assert status["runtime_available"] is False
        assert status["runtime_active"] is False
        assert status["status"] == "runtime_patch_failed"

    def test_runtime_active_model_detector_unwraps_engine_wrappers(self):
        from vmlx_engine.native_mtp import model_has_native_mtp_runtime

        class _Inner:
            def __init__(self):
                self.mtp = object()

            def mtp_forward(self, *_):
                pass

            def make_mtp_cache(self):
                return []

        class _Outer:
            def __init__(self):
                self.language_model = _Inner()

            def mtp_forward(self, *_):
                pass

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        class _Wrapper:
            def __init__(self):
                self._model = _Outer()

            def __getattr__(self, name):
                return getattr(self._model, name)

        assert model_has_native_mtp_runtime(_Wrapper()) is True

    def test_runtime_active_model_detector_rejects_fake_mtp_shape(self):
        from vmlx_engine.native_mtp import model_has_native_mtp_runtime

        class _NoCacheBuilder:
            def __init__(self):
                self.mtp = object()

            def mtp_forward(self, *_):
                pass

        class _NonCallableForward:
            def __init__(self):
                self.mtp = object()
                self.mtp_forward = object()

            def make_mtp_cache(self):
                return []

        assert model_has_native_mtp_runtime(_NoCacheBuilder()) is False
        assert model_has_native_mtp_runtime(_NonCallableForward()) is False

    def test_server_mtp_status_marks_loaded_runtime_active(self, monkeypatch, tmp_path):
        from vmlx_engine import server

        _write_qwen36_mxfp4_mtp_bundle(tmp_path)

        class _Language:
            mtp = object()

            def mtp_forward(self, *_args, **_kwargs):
                raise AssertionError("not called")

            def make_mtp_cache(self):
                return []

        class _Engine:
            _model = type("_Wrapper", (), {"language_model": _Language()})()

        monkeypatch.setattr(server, "_engine", _Engine())

        status = server._model_mtp_status_with_loaded_runtime(str(tmp_path))

        assert status["runtime_active"] is True
        assert status["status"] == "native_runtime_active"
        assert status["runtime_reason"] == "native MTP runtime is active for text+vl"
        assert status["effective_depth"] == 3
        assert status["effective_depth_source"] == "default"

    def test_native_mtp_depth_defaults_to_three(self, monkeypatch):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)

        depth, source = native_mtp_effective_depth()

        assert depth == 3
        assert source == "default"

    def test_native_mtp_depth_uses_validated_model_tuning_sidecar_by_default(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_USE_TUNING", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_USE_TUNING", raising=False)
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps(
                {
                    "native_mtp": {
                        "best_depth": 2,
                        "validated": True,
                        "output_equivalent": True,
                    }
                }
            )
        )

        depth, source = native_mtp_effective_depth(tmp_path)

        assert depth == 2
        assert source == "vmlx_mtp_tuning.json:native_mtp.best_depth"

    def test_native_mtp_depth_can_disable_model_tuning_sidecar(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_USE_TUNING", "0")
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps(
                {
                    "native_mtp": {
                        "best_depth": 2,
                        "validated": True,
                        "output_equivalent": True,
                    }
                }
            )
        )

        depth, source = native_mtp_effective_depth(tmp_path)

        assert depth == 3
        assert source == "default"

    def test_native_mtp_depth_env_overrides_model_tuning(self, monkeypatch, tmp_path):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps({"native_mtp": {"best_depth": 2}})
        )

        depth, source = native_mtp_effective_depth(tmp_path)

        assert depth == 3
        assert source == "VMLINUX_NATIVE_MTP_DEPTH"

    def test_native_mtp_depth_ignores_unvalidated_or_blocked_sidecar(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_USE_TUNING", "1")
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps(
                {
                    "native_mtp": {
                        "best_depth": 2,
                        "validated": False,
                        "output_equivalent": False,
                        "blocked": True,
                    }
                }
            )
        )

        depth, source = native_mtp_effective_depth(tmp_path)

        assert depth == 3
        assert source == "default"

    def test_native_mtp_inspection_reports_tuned_effective_depth(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_USE_TUNING", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_USE_TUNING", raising=False)
        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps(
                {
                    "native_mtp": {
                        "best_depth": 2,
                        "validated": True,
                        "output_equivalent": True,
                    }
                }
            )
        )

        status = inspect_native_mtp_bundle(tmp_path)

        assert status["effective_depth"] == 2
        assert status["effective_depth_source"] == "vmlx_mtp_tuning.json:native_mtp.best_depth"

    def test_maybe_apply_native_mtp_sets_active_model_path_for_tuned_depth(
        self, monkeypatch, tmp_path
    ):
        from vmlx_engine import native_mtp

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_USE_TUNING", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_USE_TUNING", raising=False)
        monkeypatch.setattr(native_mtp, "_apply_mlx_lm_mtp_patch", lambda: True)
        monkeypatch.setattr(native_mtp, "_set_mtp_active", lambda _active: None)
        monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", None, raising=False)
        _write_qwen36_mxfp4_mtp_bundle(tmp_path)
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps({"native_mtp": {"best_depth": 2}})
        )

        native_mtp.maybe_apply_native_mtp(tmp_path, allow_runtime=True)
        depth, source = native_mtp.native_mtp_effective_depth()

        assert depth == 2
        assert source == "vmlx_mtp_tuning.json:native_mtp.best_depth"

    def test_scheduler_uses_mlx_batch_generator_when_native_mtp_head_attached(self):
        import mlx.core as mx

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig

        class _Cache:
            pass

        class _MtpInner:
            def __init__(self):
                self.mtp = object()

            def mtp_forward(self, *_):
                pass

            def make_mtp_cache(self):
                return []

        class _MtpModel:
            def __init__(self):
                self.language_model = _MtpInner()

            def mtp_forward(self, *_):
                pass

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(self, input_ids, cache, **_kwargs):
                batch, seq_len = input_ids.shape
                return mx.zeros((batch, seq_len, 8), dtype=mx.float32)

        class _Tokenizer:
            clean_up_tokenization_spaces = False

            def decode(self, tokens):
                return "".join(str(int(t)) for t in tokens)

        scheduler = Scheduler(
            _MtpModel(),
            tokenizer=_Tokenizer(),
            config=SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams(max_tokens=1))

        assert generator.__class__.__module__ == "mlx_lm.generate"
        assert generator.__class__.__name__ == "BatchGenerator"

    def test_native_mtp_multibatch_keeps_batched_generator_and_cache_stack(
        self, tmp_path
    ):
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, KVCache

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig

        class _MtpHybridInner:
            def __init__(self):
                self.mtp = object()

            def mtp_forward(self, *_):
                pass

            def make_mtp_cache(self):
                return [KVCache()]

        class _MtpHybridModel:
            config = {"model_type": "qwen3_5"}

            def __init__(self):
                self.language_model = _MtpHybridInner()

            def make_cache(self):
                return [ArraysCache(size=2), KVCache(), ArraysCache(size=2), KVCache()]

            def mtp_forward(self, *_):
                pass

            def __call__(self, input_ids, cache, **_kwargs):
                batch, seq_len = input_ids.shape
                return mx.zeros((batch, seq_len, 8), dtype=mx.float32)

        class _Tokenizer:
            clean_up_tokenization_spaces = False

            def decode(self, tokens):
                return "".join(str(int(t)) for t in tokens)

        scheduler = Scheduler(
            _MtpHybridModel(),
            tokenizer=_Tokenizer(),
            config=SchedulerConfig(
                max_num_seqs=2,
                enable_prefix_cache=True,
                use_paged_cache=True,
                enable_block_disk_cache=True,
                block_disk_cache_dir=str(tmp_path / "blocks"),
                kv_cache_quantization="q4",
            ),
        )

        generator = scheduler._create_batch_generator(SamplingParams(max_tokens=1))
        stats = scheduler.get_stats()

        assert generator.__class__.__name__ == "BatchGenerator"
        assert scheduler.config.max_num_seqs == 2
        assert scheduler.config.kv_cache_quantization == "q4"
        assert scheduler._kv_cache_bits == 4
        assert scheduler.block_aware_cache is not None
        assert scheduler.paged_cache_manager is not None
        assert scheduler._ssm_state_cache is not None
        assert stats["batch_generator"]["max_num_seqs"] == 2

    def test_qwen_patch_preserves_nested_text_config_mtp_layers(self):
        from vmlx_engine.patches.mlx_lm_mtp import qwen35_model

        if not qwen35_model.apply():
            raise AssertionError("qwen35_model patch did not apply")

        from mlx_lm.models.qwen3_5 import TextModelArgs

        args = TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
                "text_config": {"mtp_num_hidden_layers": 1},
            }
        )

        assert args.mtp_num_hidden_layers == 1

    def test_qwen_text_sanitize_shifts_unshifted_mtp_artifact_norms(self):
        import mlx.core as mx

        from vmlx_engine.patches.mlx_lm_mtp import qwen35_model

        if not qwen35_model.apply():
            raise AssertionError("qwen35_model patch did not apply")

        from mlx_lm.models.qwen3_5 import TextModel

        fake_model = TextModel.__new__(TextModel)
        fake_model.args = SimpleNamespace(tie_word_embeddings=False)
        fake_model.mtp = object()

        sanitized = TextModel.sanitize(
            fake_model,
            {
                "model.layers.0.input_layernorm.weight": mx.array(
                    [-0.0625, -0.0625], dtype=mx.float16
                ),
                "model.norm.weight": mx.array([0.91, 0.91], dtype=mx.float16),
            },
        )

        assert sanitized["model.layers.0.input_layernorm.weight"].tolist() == [
            0.9375,
            0.9375,
        ]
        assert sanitized["model.norm.weight"].tolist() == [
            1.91015625,
            1.91015625,
        ]

    def test_mllm_generator_runs_vmlx_owned_native_mtp_decode_loop(self, monkeypatch):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "1")
        vocab_size = 8

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _Cache:
            def __init__(self):
                self.tokens = []

            def is_trimmable(self):
                return True

            def trim(self, count):
                if count:
                    self.tokens = self.tokens[:-count]

        class _NativeMtpLanguageModel:
            def __init__(self):
                self.mtp = object()
                self.calls = []
                self.mtp_calls = []

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                self.calls.append((tokens, int(n_confirmed)))
                if cache:
                    cache[0].tokens.extend(tokens)
                if tokens == [2]:
                    targets = [3]
                elif tokens == [3, 4]:
                    targets = [4, 5]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size]
                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                self.mtp_calls.append(next_id)
                return logits_for([(next_id + 1) % vocab_size])

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp",
            prompt="",
            max_tokens=4,
            temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]

        language_model = _NativeMtpLanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator.model = type("_VLM", (), {"language_model": language_model})()
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator.stop_tokens = set()
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator._make_request_sampler = lambda _req: (
            lambda logits: mx.argmax(logits, axis=-1).astype(mx.uint32)
        )

        first_token = mx.array([2], dtype=mx.uint32)
        first_logprobs = [logits_for([2]).squeeze(0).squeeze(0)]
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["vl-mtp"],
            y=first_token,
            logprobs=first_logprobs,
            max_tokens=[4],
            num_tokens=[0],
            cache=[_Cache()],
            requests=[req],
        )

        generator._seed_native_mtp_from_prefill(
            req,
            generator.active_batch.cache,
            first_token,
            first_logprobs,
        )

        tokens = [generator._next()[0].token for _ in range(4)]

        assert tokens == [2, 3, 4, 5]
        assert language_model.calls == [([2], 0), ([3, 4], 0)]
        assert language_model.mtp_calls == [3, 5]
        assert req.output_tokens == [2, 3, 4, 5]

    def test_mllm_generator_runs_depth3_native_mtp_verify_cycle(self, monkeypatch):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        vocab_size = 16

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _Cache:
            def is_trimmable(self):
                return True

            def trim(self, _count):
                pass

        class _Depth3LanguageModel:
            def __init__(self):
                self.mtp = object()
                self.calls = []
                self.mtp_calls = []

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                self.calls.append((tokens, int(n_confirmed)))
                if tokens == [2]:
                    targets = [3]
                elif tokens == [3, 4, 5, 6]:
                    targets = [4, 5, 6, 7]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size]
                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=False,
            ):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                self.mtp_calls.append(next_id)
                logits = logits_for([(next_id + 1) % vocab_size])
                hidden = mx.ones((1, 1, 4), dtype=mx.float32)
                if return_hidden:
                    return logits, hidden
                return logits

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp-d3",
            prompt="",
            max_tokens=6,
            temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]

        language_model = _Depth3LanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator.model = type("_VLM", (), {"language_model": language_model})()
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator.stop_tokens = set()
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator._make_request_sampler = lambda _req: (
            lambda logits: mx.argmax(logits, axis=-1).astype(mx.uint32)
        )

        first_token = mx.array([2], dtype=mx.uint32)
        first_logprobs = [logits_for([2]).squeeze(0).squeeze(0)]
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["vl-mtp-d3"],
            y=first_token,
            logprobs=first_logprobs,
            max_tokens=[6],
            num_tokens=[0],
            cache=[_Cache()],
            requests=[req],
        )

        generator._seed_native_mtp_from_prefill(
            req,
            generator.active_batch.cache,
            first_token,
            first_logprobs,
        )

        tokens = [generator._next()[0].token for _ in range(6)]

        assert tokens == [2, 3, 4, 5, 6, 7]
        assert language_model.calls[:2] == [([2], 0), ([3, 4, 5, 6], 0)]
        assert language_model.mtp_calls[:3] == [3, 4, 5]
        assert req.output_tokens == [2, 3, 4, 5, 6, 7]

    def test_depth3_reject_does_not_commit_correction_before_next_verify(
        self, monkeypatch
    ):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        vocab_size = 16

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _Cache:
            def __init__(self):
                self.tokens = []

            def is_trimmable(self):
                return True

            def trim(self, count):
                if count:
                    self.tokens = self.tokens[:-count]

        class _RejectLanguageModel:
            def __init__(self):
                self.mtp = object()
                self.calls = []

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                self.calls.append((tokens, int(n_confirmed)))
                if cache:
                    cache[0].tokens.extend(tokens)
                if tokens == [2]:
                    targets = [3]
                elif tokens == [3, 9, 10, 11]:
                    targets = [4, 5, 6, 7]
                elif tokens == [4, 5, 6, 7]:
                    targets = [5, 6, 7, 8]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size]
                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=False,
            ):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                target = {3: 9, 9: 10, 10: 11}.get(next_id, (next_id + 1) % vocab_size)
                logits = logits_for([target])
                hidden = mx.ones((1, 1, 4), dtype=mx.float32)
                if return_hidden:
                    return logits, hidden
                return logits

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp-reject",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]

        language_model = _RejectLanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator.model = type("_VLM", (), {"language_model": language_model})()
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator.stop_tokens = set()
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator._make_request_sampler = lambda _req: (
            lambda logits: mx.argmax(logits, axis=-1).astype(mx.uint32)
        )

        first_token = mx.array([2], dtype=mx.uint32)
        first_logprobs = [logits_for([2]).squeeze(0).squeeze(0)]
        cache = _Cache()
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["vl-mtp-reject"],
            y=first_token,
            logprobs=first_logprobs,
            max_tokens=[8],
            num_tokens=[0],
            cache=[cache],
            requests=[req],
        )

        generator._seed_native_mtp_from_prefill(
            req,
            generator.active_batch.cache,
            first_token,
            first_logprobs,
        )

        tokens = [generator._next()[0].token for _ in range(6)]

        assert tokens[:3] == [2, 3, 4]
        assert ([4], 0) not in language_model.calls
        assert language_model.calls[:4] == [
            ([2], 0),
            ([3, 9, 10, 11], 0),
            ([3], 0),
            ([4, 5, 6, 7], 0),
        ]
        assert cache.tokens[:6] == [2, 3, 4, 5, 6, 7]

    def test_depth3_partial_reject_restores_hybrid_cache_before_replay(
        self, monkeypatch
    ):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        vocab_size = 32

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _HybridCache:
            def __init__(self):
                self.tokens = []
                self.position = 0
                self.conv_state = 0
                self.ssm_state = 0
                self.rollback_state = None

            def append(self, tokens, n_confirmed=0):
                for index, token in enumerate(tokens, start=1):
                    self.tokens.append(int(token))
                    self.position += 1
                    self.conv_state = self.position
                    self.ssm_state = self.position
                    if n_confirmed and index == n_confirmed:
                        self.rollback_state = (self.conv_state, self.ssm_state)

        class _PartialRejectLanguageModel:
            def __init__(self):
                self.mtp = object()

            def make_mtp_cache(self):
                return [_HybridCache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                if cache:
                    cache[0].append(tokens, int(n_confirmed))

                if tokens == [2]:
                    targets = [3]
                elif tokens == [3, 4, 5, 9]:
                    targets = [4, 5, 6, 7]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size for _ in tokens]

                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=False,
            ):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                target = {3: 4, 4: 5, 5: 9}.get(next_id, (next_id + 1) % vocab_size)
                logits = logits_for([target])
                hidden = mx.ones((1, 1, 4), dtype=mx.float32)
                if return_hidden:
                    return logits, hidden
                return logits

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp-partial-reject",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]

        language_model = _PartialRejectLanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator.model = type("_VLM", (), {"language_model": language_model})()
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator.stop_tokens = set()
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator._make_request_sampler = lambda _req: (
            lambda logits: mx.argmax(logits, axis=-1).astype(mx.uint32)
        )

        first_token = mx.array([2], dtype=mx.uint32)
        first_logprobs = [logits_for([2]).squeeze(0).squeeze(0)]
        cache = _HybridCache()
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["vl-mtp-partial-reject"],
            y=first_token,
            logprobs=first_logprobs,
            max_tokens=[8],
            num_tokens=[0],
            cache=[cache],
            requests=[req],
        )

        generator._seed_native_mtp_from_prefill(
            req,
            generator.active_batch.cache,
            first_token,
            first_logprobs,
        )
        state = req._native_mtp_state
        state.queue.clear()

        generator._run_native_mtp_verify_cycle(req, generator.active_batch.cache, state)

        assert [token for token, _lp, source in state.queue if source != "init"] == [
            4,
            5,
            6,
        ]
        assert cache.tokens == [2, 3, 4, 5]
        assert cache.position == 4
        assert cache.conv_state == 4
        assert cache.ssm_state == 4

    def test_native_mtp_greedy_sampler_uses_logits_fast_path(self, monkeypatch):
        import mlx.core as mx

        import vmlx_engine.mllm_batch_generator as mbg
        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "1")

        def fail_logprobs(_logits):
            raise AssertionError("greedy native MTP should not materialize logprobs")

        monkeypatch.setattr(mbg, "_native_mtp_logprobs", fail_logprobs)
        vocab_size = 8

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _Cache:
            def is_trimmable(self):
                return True

            def trim(self, _count):
                pass

        class _GreedyLanguageModel:
            def __init__(self):
                self.mtp = object()

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                if tokens == [2]:
                    targets = [3]
                elif tokens == [3, 4]:
                    targets = [4, 5]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size]
                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=False,
            ):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                logits = logits_for([(next_id + 1) % vocab_size])
                hidden = mx.ones((1, 1, 4), dtype=mx.float32)
                if return_hidden:
                    return logits, hidden
                return logits

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp-greedy-fast",
            prompt="",
            max_tokens=4,
            temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]

        language_model = _GreedyLanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator.model = type("_VLM", (), {"language_model": language_model})()
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator.stop_tokens = set()
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()

        def greedy_sampler(logits):
            return mx.argmax(logits, axis=-1).astype(mx.uint32)

        greedy_sampler._vmlx_accepts_logits = True
        generator._make_request_sampler = lambda _req: greedy_sampler

        first_token = mx.array([2], dtype=mx.uint32)
        first_logprobs = [None]
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["vl-mtp-greedy-fast"],
            y=first_token,
            logprobs=first_logprobs,
            max_tokens=[4],
            num_tokens=[0],
            cache=[_Cache()],
            requests=[req],
        )

        assert generator._seed_native_mtp_from_prefill(
            req,
            generator.active_batch.cache,
            first_token,
            first_logprobs,
        )

        tokens = [generator._next()[0].token for _ in range(4)]

        assert tokens == [2, 3, 4, 5]

    def test_mllm_prefill_sampler_skips_logsumexp_for_greedy_logits_sampler(
        self, monkeypatch
    ):
        import mlx.core as mx

        import vmlx_engine.mllm_batch_generator as mbg

        def fail_logsumexp(*_args, **_kwargs):
            raise AssertionError("greedy MLLM prefill should not materialize logprobs")

        def greedy_sampler(logits):
            return mx.argmax(logits, axis=-1)

        greedy_sampler._vmlx_accepts_logits = True
        monkeypatch.setattr(mbg.mx, "logsumexp", fail_logsumexp)

        sampled, logprobs = mbg._sample_mllm_prefill_logits(
            mx.array([[0.0, 3.0, 1.0]], dtype=mx.float32),
            greedy_sampler,
        )
        mx.eval(sampled)

        assert int(sampled.item()) == 1
        assert logprobs is None

    def test_native_mtp_sample_rows_returns_materialized_ids_for_accept_loop(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import _native_mtp_sample_rows

        logits = mx.array(
            [
                [0.0, 4.0, 1.0],
                [9.0, 1.0, 0.0],
                [0.0, 2.0, 7.0],
            ],
            dtype=mx.float32,
        )

        def greedy(rows):
            return mx.argmax(rows, axis=-1).astype(mx.uint32)

        greedy._vmlx_accepts_logits = True

        tokens, logprobs, token_ids = _native_mtp_sample_rows(logits, greedy)

        assert [int(token.tolist()[0]) for token in tokens] == [1, 0, 2]
        assert logprobs == [None, None, None]
        assert token_ids == [1, 0, 2]

    @pytest.mark.parametrize(
        ("temperature", "repetition_penalty"),
        [
            (0.6, 1.0),
            (0.0, 1.1),
        ],
    )
    def test_mllm_native_mtp_only_enables_for_proven_deterministic_sampling(
        self,
        temperature,
        repetition_penalty,
    ):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

        class _LanguageModel:
            mtp = object()

            def make_mtp_cache(self):
                return []

            def mtp_forward(self, *_args, **_kwargs):
                raise AssertionError("runtime activation check should not draft")

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = _LanguageModel()
        req = MLLMBatchRequest(
            uid=0,
            request_id="unsafe-sampling-mtp",
            prompt="",
            max_tokens=8,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        assert generator._native_mtp_enabled_for_request(req) is False

    def test_native_mtp_stats_log_includes_phase_timings(self, caplog):
        import logging

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPStats,
            _native_mtp_log_stats,
        )

        stats = MLLMNativeMTPStats(
            cycles=2,
            drafted_tokens=6,
            accepted_tokens=5,
            verify_ms=12.0,
            sample_ms=3.0,
            draft_ms=8.0,
            snapshot_ms=1.0,
            restore_ms=0.5,
            replay_ms=2.5,
            materialize_ms=0.25,
        )

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            _native_mtp_log_stats("trace-row", stats, "length")

        assert "timings_ms" in caplog.text
        assert "verify=12.00" in caplog.text
        assert "draft=8.00" in caplog.text
        assert "avg_cycle=13.62" in caplog.text

    def test_native_mtp_stats_log_includes_depth_acceptance_and_forward_counts(
        self, caplog
    ):
        import logging

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPStats,
            _native_mtp_log_stats,
        )

        stats = MLLMNativeMTPStats(
            cycles=3,
            drafted_tokens=9,
            accepted_tokens=7,
            accepted_by_depth=[3, 2, 2],
            drafted_by_depth=[3, 3, 3],
            seed_main_forwards=1,
            verify_main_forwards=3,
            replay_main_forwards=1,
            mtp_forwards=12,
        )

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            _native_mtp_log_stats("depth-row", stats, "length")

        assert "accept_by_depth[d1=3/3,d2=2/3,d3=2/3]" in caplog.text
        assert (
            "forwards[seed_main=1,verify_main=3,replay_main=1,mtp=12]"
            in caplog.text
        )

    def test_native_mtp_adaptive_depth_lowers_d3_after_poor_third_position(
        self, monkeypatch, caplog
    ):
        import logging

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES", "8")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_D3_MIN_ACCEPT", "0.80")
        state = MLLMNativeMTPState(depth=3)
        state.stats.cycles = 8
        state.stats.drafted_by_depth = [8, 8, 8]
        state.stats.accepted_by_depth = [8, 8, 3]

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            _native_mtp_maybe_adapt_depth("adaptive-row", state)

        assert state.depth == 2
        assert "adaptive depth D3 -> D2" in caplog.text

    def test_native_mtp_adaptive_depth_can_be_disabled(self, monkeypatch):
        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        state = MLLMNativeMTPState(depth=3)
        state.stats.cycles = 8
        state.stats.drafted_by_depth = [8, 8, 8]
        state.stats.accepted_by_depth = [8, 8, 0]

        _native_mtp_maybe_adapt_depth("adaptive-row", state)

        assert state.depth == 3

    def test_native_mtp_adaptive_policy_marks_low_d1_for_ar_fallback(
        self, monkeypatch, caplog
    ):
        import logging

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_AR_FALLBACK", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES", "8")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT", "0.70")
        state = MLLMNativeMTPState(depth=1)
        state.stats.cycles = 12
        state.stats.drafted_by_depth = [12, 0, 0]
        state.stats.accepted_by_depth = [7, 0, 0]

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            _native_mtp_maybe_adapt_depth("adaptive-row", state)

        assert state.depth == 1
        assert state.ar_fallback_pending is True
        assert "adaptive depth D1 -> AR" in caplog.text

    def test_native_mtp_raw_acceptance_ar_fallback_is_opt_in(
        self, monkeypatch
    ):
        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_AR_FALLBACK", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_AR_FALLBACK", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES", "8")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_D1_MIN_ACCEPT", "0.70")
        state = MLLMNativeMTPState(depth=1)
        state.stats.cycles = 12
        state.stats.drafted_by_depth = [12, 0, 0]
        state.stats.accepted_by_depth = [7, 0, 0]

        _native_mtp_maybe_adapt_depth("adaptive-row", state)

        assert state.ar_fallback_pending is False

    def test_native_mtp_cost_policy_marks_ar_fallback_when_mtp_is_slower(
        self, monkeypatch, caplog
    ):
        import logging

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_FALLBACK", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_AR_STEP_MS", "30")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD", "1.0")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES", "4")
        state = MLLMNativeMTPState(depth=3)
        state.stats.cycles = 4
        state.stats.accepted_tokens = 2
        state.stats.drafted_tokens = 12
        state.stats.drafted_by_depth = [4, 4, 4]
        state.stats.accepted_by_depth = [2, 0, 0]
        state.stats.verify_ms = 180.0
        state.stats.draft_ms = 60.0

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            _native_mtp_maybe_adapt_depth("cost-row", state)

        # MTP produced cycles + accepted = 6 confirmed tokens at 240ms total,
        # which is 40ms/token versus a calibrated 30ms AR step.
        assert state.ar_fallback_pending is True
        assert state.depth == 1
        assert "adaptive depth D3 -> AR" in caplog.text
        assert "cost_ratio=1.333" in state.ar_fallback_reason

    def test_native_mtp_cost_policy_requires_calibration_and_timings(
        self, monkeypatch
    ):
        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_maybe_adapt_depth,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_FALLBACK", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES", "4")
        state = MLLMNativeMTPState(depth=3)
        state.stats.cycles = 4
        state.stats.accepted_tokens = 0
        state.stats.drafted_tokens = 12
        state.stats.drafted_by_depth = [4, 4, 4]
        state.stats.accepted_by_depth = [0, 0, 0]
        state.stats.verify_ms = 180.0
        state.stats.draft_ms = 60.0

        _native_mtp_maybe_adapt_depth("cost-row", state)
        assert state.ar_fallback_pending is False

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_AR_STEP_MS", "30")
        state.stats.verify_ms = 0.0
        state.stats.draft_ms = 0.0
        _native_mtp_maybe_adapt_depth("cost-row", state)
        assert state.ar_fallback_pending is False

    def test_native_mtp_ar_fallback_requires_clean_cycle_boundary(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            _native_mtp_ar_fallback_ready,
        )

        class _Layer:
            rollback_state = None

        state = MLLMNativeMTPState(
            next_main=mx.array([7], dtype=mx.uint32),
            ar_fallback_pending=True,
        )

        ok, reason = _native_mtp_ar_fallback_ready([_Layer()], state, 7)
        assert ok is True
        assert reason == "ready"

        stale = _Layer()
        stale.rollback_state = ("conv", "ssm")
        ok, reason = _native_mtp_ar_fallback_ready([stale], state, 7)
        assert ok is False
        assert reason == "pending_rollback_state"

        ok, reason = _native_mtp_ar_fallback_ready([_Layer()], state, 8)
        assert ok is False
        assert reason == "next_main_mismatch"

    def test_native_mtp_ar_fallback_drains_last_token_before_standard_decode(
        self,
        caplog,
    ):
        import logging

        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
            MLLMNativeMTPState,
        )

        req = MLLMBatchRequest(
            uid=0,
            request_id="fallback-mtp",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        req._native_mtp_state = MLLMNativeMTPState(
            next_main=mx.array([7], dtype=mx.uint32),
            ar_fallback_pending=True,
        )
        req._native_mtp_state.queue.append((7, None, "verify"))

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["fallback-mtp"],
            y=mx.array([6], dtype=mx.uint32),
            logprobs=[None],
            max_tokens=[8],
            num_tokens=[0],
            cache=[],
            requests=[req],
        )
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        stepped_inputs = []

        def step_after_fallback(input_tokens, cache):
            stepped_inputs.append(input_tokens.tolist())
            assert cache == []
            return mx.array([8], dtype=mx.uint32), [None]

        generator._step = step_after_fallback

        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            responses = generator._next()

        assert [response.token for response in responses] == [7]
        assert stepped_inputs == [[[7]]]
        assert generator.active_batch is not None
        assert generator.active_batch.y.tolist() == [8]
        assert not hasattr(req, "_native_mtp_state")
        assert "finish=fallback_to_ar" in caplog.text

    def test_depth3_verify_uses_lazy_draft_ids_until_after_target_sample(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMNativeMTPState,
        )

        vocab_size = 16

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _Cache:
            def is_trimmable(self):
                return True

            def trim(self, _count):
                pass

        state = MLLMNativeMTPState(
            next_main=mx.array([3], dtype=mx.uint32),
            drafts=[
                mx.array([4], dtype=mx.uint32),
                mx.array([5], dtype=mx.uint32),
                mx.array([6], dtype=mx.uint32),
            ],
            draft_lps=[None, None, None],
            draft_ids=[],
            depth=3,
        )
        draft_ids_seen_at_verify = []

        class _LanguageModel:
            def __init__(self):
                self.mtp = object()

            def make_mtp_cache(self):
                return [_Cache()]

            def __call__(
                self,
                input_ids,
                cache=None,
                return_hidden=False,
                n_confirmed=0,
                **_kwargs,
            ):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                if tokens == [3, 4, 5, 6]:
                    draft_ids_seen_at_verify.append(list(state.draft_ids))
                    targets = [4, 5, 6, 7]
                else:
                    targets = [(tokens[-1] + 1) % vocab_size for _ in tokens]
                logits = logits_for(targets)
                if return_hidden:
                    hidden = mx.ones((1, len(targets), 4), dtype=mx.float32)
                    return logits, hidden
                return logits

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=False,
            ):
                next_id = int(next_token_ids.reshape(-1).tolist()[0])
                logits = logits_for([(next_id + 1) % vocab_size])
                hidden = mx.ones((1, 1, 4), dtype=mx.float32)
                if return_hidden:
                    return logits, hidden
                return logits

        req = MLLMBatchRequest(
            uid=0,
            request_id="vl-mtp-lazy-ids",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        language_model = _LanguageModel()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = language_model
        generator._make_request_sampler = lambda _req: (
            lambda logits: mx.argmax(logits, axis=-1).astype(mx.uint32)
        )

        generator._run_native_mtp_verify_cycle(req, [_Cache()], state)

        assert draft_ids_seen_at_verify == [[]]
        assert [token for token, _lp, _source in state.queue] == [4, 5, 6, 7]

    def test_native_mtp_depth_accepts_vmlx_env_alias(self, monkeypatch):
        from vmlx_engine.mllm_batch_generator import _native_mtp_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")

        assert _native_mtp_depth() == 3

    def test_native_mtp_tool_request_caps_depth_to_one(self, monkeypatch, caplog):
        from vmlx_engine.mllm_batch_generator import _native_mtp_depth_for_request

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")

        req = SimpleNamespace(
            request_id="tool-depth-cap",
            extra_kwargs={
                "tools": [{"type": "function", "function": {"name": "record_fact"}}],
                "tool_choice": "required",
            },
        )

        with caplog.at_level("INFO"):
            assert _native_mtp_depth_for_request(req) == 1

        assert "MLLM native MTP depth capped to D1 for request=tool-depth-cap" in caplog.text

    def test_native_mtp_tool_present_metadata_caps_depth_to_one(self, monkeypatch):
        from vmlx_engine.mllm_batch_generator import _native_mtp_depth_for_request

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")

        req = SimpleNamespace(
            request_id="tool-present-metadata",
            extra_kwargs={"_vmlx_tools_present": True},
        )

        assert _native_mtp_depth_for_request(req) == 1

    def test_native_mtp_rendered_tool_prompt_caps_depth_to_one(self, monkeypatch):
        from vmlx_engine.mllm_batch_generator import _native_mtp_depth_for_request

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")

        req = SimpleNamespace(
            request_id="rendered-tool-prompt",
            prompt='<tools>[{"type": "function", "function": {"name": "record_fact"}}]</tools>',
            extra_kwargs={},
        )

        assert _native_mtp_depth_for_request(req) == 1

    def test_native_mtp_non_tool_request_keeps_configured_depth(self, monkeypatch):
        from vmlx_engine.mllm_batch_generator import _native_mtp_depth_for_request

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")

        req = SimpleNamespace(request_id="plain-depth", extra_kwargs={})

        assert _native_mtp_depth_for_request(req) == 3

    def test_mllm_native_mtp_active_row_does_not_extend_standard_batch(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        active = MLLMBatchRequest(
            uid=0,
            request_id="active-mtp",
            prompt="",
            max_tokens=4,
            temperature=0.0,
        )
        active._native_mtp_state = object()
        queued = MLLMBatchRequest(
            uid=1,
            request_id="queued-standard",
            prompt="",
            max_tokens=4,
            temperature=0.0,
        )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["active-mtp"],
            y=mx.array([2], dtype=mx.uint32),
            logprobs=[mx.zeros((8,), dtype=mx.float32)],
            max_tokens=[4],
            num_tokens=[0],
            cache=[],
            requests=[active],
        )
        generator.unprocessed_requests = [queued]
        generator.completion_batch_size = 2
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator._next_native_mtp_token = lambda *_args: (
            2,
            mx.zeros((8,), dtype=mx.float32),
        )

        def fail_process_prompts(*_args, **_kwargs):
            raise AssertionError(
                "standard batch extension must wait while a native-MTP row is active"
            )

        generator._process_prompts = fail_process_prompts

        responses = generator._next()

        assert [response.token for response in responses] == [2]
        assert generator.unprocessed_requests == [queued]
        assert generator.active_batch is not None
        assert generator.active_batch.requests == [active]

    def test_mllm_native_mtp_next_burst_drains_verified_queue_when_safe(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
            MLLMNativeMTPState,
        )

        req = MLLMBatchRequest(
            uid=0,
            request_id="burst-mtp",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        req._native_mtp_state = MLLMNativeMTPState()
        req._native_mtp_state.queue.extend(
            [
                (2, None, "draft"),
                (3, None, "draft"),
                (4, None, "bonus"),
            ]
        )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["burst-mtp"],
            y=mx.array([2], dtype=mx.uint32),
            logprobs=[None],
            max_tokens=[8],
            num_tokens=[0],
            cache=[],
            requests=[req],
        )
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        old_stream = MLLMBatchGenerator._stream
        MLLMBatchGenerator._stream = mx.default_stream(mx.default_device())

        try:
            responses = generator.next_burst()
        finally:
            MLLMBatchGenerator._stream = old_stream

        assert [response.token for response in responses] == [2, 3, 4]
        assert generator.active_batch is not None
        assert generator.active_batch.num_tokens == [3]
        assert req.output_tokens == [2, 3, 4]

    def test_mllm_native_mtp_next_burst_respects_string_stop_boundary(self):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
            MLLMNativeMTPState,
        )

        req = MLLMBatchRequest(
            uid=0,
            request_id="burst-stop-mtp",
            prompt="",
            max_tokens=8,
            temperature=0.0,
        )
        req._stop_strings = ["stop"]
        req._native_mtp_state = MLLMNativeMTPState()
        req._native_mtp_state.queue.extend(
            [
                (2, None, "draft"),
                (3, None, "draft"),
            ]
        )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = MLLMBatch(
            uids=[0],
            request_ids=["burst-stop-mtp"],
            y=mx.array([2], dtype=mx.uint32),
            logprobs=[None],
            max_tokens=[8],
            num_tokens=[0],
            cache=[],
            requests=[req],
        )
        generator.unprocessed_requests = []
        generator.completion_batch_size = 1
        generator._prefill_errors = []
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        old_stream = MLLMBatchGenerator._stream
        MLLMBatchGenerator._stream = mx.default_stream(mx.default_device())

        try:
            responses = generator.next_burst()
        finally:
            MLLMBatchGenerator._stream = old_stream

        assert [response.token for response in responses] == [2]
        assert list(req._native_mtp_state.queue) == [(3, None, "draft")]

    def test_mllm_scheduler_coalesces_safe_native_mtp_burst_outputs(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
        from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vmlx_engine.request import SamplingParams

        class CountingTokenizer:
            clean_up_tokenization_spaces = False

            def __init__(self):
                self.decode_calls = 0

            def decode(self, tokens):
                self.decode_calls += 1
                return "".join(chr(65 + int(token)) for token in tokens)

        tokenizer = CountingTokenizer()
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.processor = tokenizer
        scheduler.uid_to_request_id = {0: "burst-coalesce"}
        scheduler.running = {
            "burst-coalesce": MLLMRequest(
                request_id="burst-coalesce",
                prompt="",
                sampling_params=SamplingParams(max_tokens=8),
            )
        }
        scheduler._detokenizer_pool = {}
        scheduler.total_prompt_tokens = 0
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0

        responses = [
            MLLMBatchResponse(
                uid=0,
                request_id="burst-coalesce",
                token=0,
                logprobs=None,
                prompt_token_ids=[10, 11],
            ),
            MLLMBatchResponse(
                uid=0,
                request_id="burst-coalesce",
                token=1,
                logprobs=None,
                prompt_token_ids=[10, 11],
            ),
            MLLMBatchResponse(
                uid=0,
                request_id="burst-coalesce",
                token=2,
                logprobs=None,
                prompt_token_ids=[10, 11],
            ),
        ]

        outputs, finished_ids = scheduler._process_batch_responses(responses)

        assert finished_ids == set()
        assert len(outputs) == 1
        assert outputs[0].new_token_ids == [0, 1, 2]
        assert outputs[0].new_text == "ABC"
        assert outputs[0].output_token_ids == [0, 1, 2]
        assert outputs[0].completion_tokens == 3
        # One constructor sanity decode plus one burst decode.
        assert tokenizer.decode_calls == 2

    def test_mllm_scheduler_coalesced_burst_preserves_stop_token_accounting(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
        from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vmlx_engine.request import SamplingParams

        class CountingTokenizer:
            clean_up_tokenization_spaces = False

            def decode(self, tokens):
                return "".join(chr(65 + int(token)) for token in tokens)

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.processor = CountingTokenizer()
        scheduler.uid_to_request_id = {0: "burst-stop"}
        scheduler.running = {
            "burst-stop": MLLMRequest(
                request_id="burst-stop",
                prompt="",
                sampling_params=SamplingParams(max_tokens=8),
            )
        }
        scheduler._detokenizer_pool = {}
        scheduler.total_prompt_tokens = 0
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0
        scheduler._record_cache_hit = lambda *_args, **_kwargs: None

        outputs, finished_ids = scheduler._process_batch_responses(
            [
                MLLMBatchResponse(
                    uid=0,
                    request_id="burst-stop",
                    token=0,
                    logprobs=None,
                    prompt_token_ids=[7, 8],
                ),
                MLLMBatchResponse(
                    uid=0,
                    request_id="burst-stop",
                    token=99,
                    logprobs=None,
                    finish_reason="stop",
                    prompt_token_ids=[7, 8],
                ),
            ]
        )

        assert finished_ids == {"burst-stop"}
        assert len(outputs) == 1
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "stop"
        assert outputs[0].new_token_ids == [0, 99]
        assert outputs[0].output_token_ids == [0, 99]
        assert outputs[0].new_text == "A"
        assert outputs[0].output_text == "A"
        assert outputs[0].completion_tokens == 2
        assert scheduler.total_completion_tokens == 2

    def test_mllm_scheduler_coalescing_starts_after_gen_prefix_divergence(self):
        from vmlx_engine.mllm_batch_generator import MLLMBatchResponse
        from vmlx_engine.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vmlx_engine.request import SamplingParams

        class CountingTokenizer:
            clean_up_tokenization_spaces = False

            def decode(self, tokens):
                return "".join(chr(65 + int(token)) for token in tokens)

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.processor = CountingTokenizer()
        scheduler.uid_to_request_id = {0: "burst-prefix"}
        scheduler.running = {
            "burst-prefix": MLLMRequest(
                request_id="burst-prefix",
                prompt="",
                sampling_params=SamplingParams(max_tokens=8),
            )
        }
        scheduler._detokenizer_pool = {}
        scheduler.total_prompt_tokens = 0
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0

        responses = [
            MLLMBatchResponse(
                uid=0,
                request_id="burst-prefix",
                token=0,
                logprobs=None,
                gen_prefix_tokens=[99],
            ),
            MLLMBatchResponse(
                uid=0,
                request_id="burst-prefix",
                token=1,
                logprobs=None,
                gen_prefix_tokens=[99],
            ),
            MLLMBatchResponse(
                uid=0,
                request_id="burst-prefix",
                token=2,
                logprobs=None,
                gen_prefix_tokens=[99],
            ),
        ]

        outputs, finished_ids = scheduler._process_batch_responses(responses)

        assert finished_ids == set()
        assert [output.new_token_ids for output in outputs] == [[0], [1, 2]]
        assert [output.new_text for output in outputs] == ["A", "BC"]
        assert outputs[-1].output_token_ids == [0, 1, 2]
        assert scheduler.running["burst-prefix"]._gen_prefix_tokens == []

import json
import logging
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from vmlx_engine.cli import (
    ALLOW_UNSAFE_KV_CACHE_QUANTIZATION_ENV,
    FamilyKVCacheStoragePolicyError,
    _apply_family_kv_storage_policy,
)
from vmlx_engine.commands.convert import (
    _conversion_source_with_model_config_compat,
    _run_jang_conversion,
)
from vmlx_engine.model_config_registry import (
    ModelConfig,
    ModelConfigRegistry,
    _minicpm_model_config_compat_override,
)
from vmlx_engine.model_configs import register_all
from vmlx_engine.utils.chat_template_kwargs import ensure_minicpm_chat_bos
from vmlx_engine.utils.tokenizer import _minicpm_jang_tokenizer_config


def test_legacy_minicpm_missing_official_defaults_gets_narrow_override():
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "hidden_size": 1024,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }

    assert _minicpm_model_config_compat_override(config) == {
        "model_type": "minicpm",
        "rope_theta": 10_000.0,
    }


def test_legacy_minicpm_override_refuses_explicit_or_partial_configs():
    explicit = {
        "model_type": "custom_minicpm",
        "architectures": ["MiniCPMForCausalLM"],
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }
    partial = {
        "architectures": ["MiniCPMForCausalLM"],
        "scale_emb": 12,
        "scale_depth": 1.4,
    }
    unrelated = {
        "architectures": ["LlamaForCausalLM"],
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }

    assert _minicpm_model_config_compat_override(explicit) is None
    assert _minicpm_model_config_compat_override(partial) is None
    assert _minicpm_model_config_compat_override(unrelated) is None


@pytest.mark.parametrize(
    "media_signal",
    [
        {"vision_config": {}},
        {"audio_config": {}},
        {"video_config": {}},
        {"image_token_id": 73441},
    ],
)
def test_legacy_minicpm_override_rejects_conflicting_media_config(media_signal):
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
        **media_signal,
    }

    assert _minicpm_model_config_compat_override(config) is None


def test_exact_minicpm_jang_tokenizer_enables_local_custom_code(tmp_path):
    config = {
        "model_type": "minicpm",
        "architectures": ["MiniCPMForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "vocab_size": 73448,
        "intermediate_size": 4096,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert _minicpm_jang_tokenizer_config(
        tmp_path,
        {"eos_token": "<eos>"},
    ) == {
        "eos_token": "<eos>",
        "trust_remote_code": True,
    }


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama"},
        {"model_type": "minicpmv"},
        {
            "model_type": "minicpm",
            "architectures": ["MiniCPMVForCausalLM"],
            "vision_config": {},
        },
    ],
)
def test_unrelated_jang_tokenizers_do_not_receive_implicit_trust(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _minicpm_jang_tokenizer_config(tmp_path, {"existing": True}) is None


def test_registry_detects_legacy_minicpm_as_plain_kv_family(tmp_path):
    model_dir = tmp_path / "MiniCPM4-0.5B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniCPMForCausalLM"],
                "hidden_size": 1024,
                "scale_emb": 12,
                "scale_depth": 1.4,
                "dim_model_base": 256,
            }
        ),
        encoding="utf-8",
    )

    registry = ModelConfigRegistry()
    register_all(registry)
    config = registry.lookup(str(model_dir))

    assert config.family_name == "minicpm"
    assert config.cache_type == "kv"
    assert config.is_mllm is False
    assert config.reasoning_parser is None
    assert config.tool_parser is None
    assert config.architecture_hints == {
        "auto_kv_cache_storage_quantization": "none",
        "blocked_kv_cache_storage_quantizations": ["q4"],
        "warn_kv_cache_storage_quantizations": ["q8"],
    }


def test_registry_does_not_promote_legacy_minicpm_with_media_config(tmp_path):
    model_dir = tmp_path / "MiniCPM4-conflicting-media"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniCPMForCausalLM"],
                "scale_emb": 12,
                "scale_depth": 1.4,
                "dim_model_base": 256,
                "vision_config": {},
            }
        ),
        encoding="utf-8",
    )

    registry = ModelConfigRegistry()
    register_all(registry)
    config = registry.lookup(str(model_dir))

    assert config.family_name == "unknown"


def test_jang_stamped_minicpm_inherits_family_kv_storage_policy(tmp_path):
    model_dir = tmp_path / "minicpm4-JANG_6M"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "minicpm"}),
        encoding="utf-8",
    )
    (model_dir / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "profile": "JANG_6M",
                "quantization": {"target_bits": 6, "actual_bits": 6.52},
                "capabilities": {
                    "family": "minicpm",
                    "cache_type": "kv",
                    "modality": "text",
                    "supports_tools": False,
                    "supports_thinking": False,
                },
            }
        ),
        encoding="utf-8",
    )

    registry = ModelConfigRegistry()
    register_all(registry)
    config = registry.lookup(str(model_dir))

    assert config.family_name == "minicpm"
    assert config.architecture_hints == {
        "auto_kv_cache_storage_quantization": "none",
        "blocked_kv_cache_storage_quantizations": ["q4"],
        "warn_kv_cache_storage_quantizations": ["q8"],
    }
    assert json.loads((model_dir / "jang_config.json").read_text())["profile"] == (
        "JANG_6M"
    )


def test_minicpm_batched_prompt_gets_exactly_one_bos():
    tokenizer = types.SimpleNamespace(bos_token="<s>")
    rendered = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    corrected = ensure_minicpm_chat_bos(
        rendered,
        family_name="minicpm",
        tokenizer=tokenizer,
    )

    assert corrected == "<s>" + rendered
    assert (
        ensure_minicpm_chat_bos(
            corrected,
            family_name="minicpm",
            tokenizer=tokenizer,
        )
        == corrected
    )


def test_minicpm_batched_prompt_resolves_wrapped_tokenizer_bos():
    tokenizer = types.SimpleNamespace(
        bos_token=None,
        _tokenizer=types.SimpleNamespace(bos_token="<s>"),
    )

    assert ensure_minicpm_chat_bos(
        "<|im_start|>user\nHello",
        family_name="minicpm",
        tokenizer=tokenizer,
    ).startswith("<s><|im_start|>")


@pytest.mark.parametrize(
    "family_name",
    [None, "unknown", "llama", "qwen2", "minicpm_v", "minicpmv"],
)
def test_batched_prompt_bos_correction_is_exact_minicpm_only(family_name):
    tokenizer = types.SimpleNamespace(bos_token="<s>")
    rendered = "<|im_start|>user\nHello"

    assert (
        ensure_minicpm_chat_bos(
            rendered,
            family_name=family_name,
            tokenizer=tokenizer,
        )
        == rendered
    )


def test_minicpm_batched_prompt_without_usable_bos_is_unchanged():
    rendered = "<|im_start|>user\nHello"

    assert (
        ensure_minicpm_chat_bos(
            rendered,
            family_name="minicpm",
            tokenizer=types.SimpleNamespace(bos_token=None),
        )
        == rendered
    )


def test_minicpm_batched_prompt_tolerates_inaccessible_wrapper_properties():
    class ProcessorWithoutAccessibleTokenizer:
        bos_token = None

        @property
        def _tokenizer(self):
            raise RuntimeError("optional tokenizer is not initialized")

        @property
        def tokenizer(self):
            raise RuntimeError("optional processor tokenizer is not initialized")

    rendered = "<|im_start|>user\nHello"

    assert (
        ensure_minicpm_chat_bos(
            rendered,
            family_name="minicpm",
            tokenizer=ProcessorWithoutAccessibleTokenizer(),
        )
        == rendered
    )
    assert (
        ensure_minicpm_chat_bos(
            rendered,
            family_name="minicpm",
            tokenizer=types.SimpleNamespace(bos_token=""),
        )
        == rendered
    )


def test_registry_keeps_minicpm_v_outside_text_bos_family(tmp_path):
    model_dir = tmp_path / "MiniCPM-V"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "minicpmv"}),
        encoding="utf-8",
    )

    registry = ModelConfigRegistry()
    register_all(registry)

    assert registry.lookup(str(model_dir)).family_name == "minicpm_v"


def _policy_args(codec, *, explicit):
    return Namespace(
        kv_cache_quantization=codec,
        kv_cache_quantization_explicit=explicit,
    )


def _minicpm_policy_config():
    return ModelConfig(
        family_name="minicpm",
        model_types=["minicpm"],
        architecture_hints={
            "auto_kv_cache_storage_quantization": "none",
            "blocked_kv_cache_storage_quantizations": ["q4"],
            "warn_kv_cache_storage_quantizations": ["q8"],
        },
    )


def test_minicpm_auto_storage_becomes_raw_without_disabling_live_tq(
    monkeypatch,
):
    args = _policy_args("q4", explicit=False)
    monkeypatch.setenv("VMLX_FORCE_TQ_AUTO", "1")
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)

    result = _apply_family_kv_storage_policy(
        args,
        _minicpm_policy_config(),
        logging.getLogger("test_minicpm_auto_storage"),
    )

    assert result == "auto_override"
    assert args.kv_cache_quantization == "none"
    assert args.kv_cache_quantization_explicit is False
    assert os.environ["VMLX_FORCE_TQ_AUTO"] == "1"
    assert "VMLX_DISABLE_TQ_KV" not in os.environ


def test_minicpm_explicit_q4_requires_deliberate_unsafe_override(monkeypatch):
    args = _policy_args("q4", explicit=True)
    config = _minicpm_policy_config()
    logger = logging.getLogger("test_minicpm_q4_policy")
    monkeypatch.delenv(ALLOW_UNSAFE_KV_CACHE_QUANTIZATION_ENV, raising=False)

    with pytest.raises(FamilyKVCacheStoragePolicyError, match="cold/warm"):
        _apply_family_kv_storage_policy(args, config, logger)

    monkeypatch.setenv(ALLOW_UNSAFE_KV_CACHE_QUANTIZATION_ENV, "1")
    assert _apply_family_kv_storage_policy(args, config, logger) == ("unsafe_override")
    assert args.kv_cache_quantization == "q4"


def test_minicpm_explicit_q8_warns_and_explicit_none_remains_exact(caplog):
    config = _minicpm_policy_config()
    logger = logging.getLogger("test_minicpm_q8_policy")
    caplog.set_level(logging.WARNING, logger=logger.name)

    q8_args = _policy_args("q8", explicit=True)
    assert _apply_family_kv_storage_policy(q8_args, config, logger) == "warned"
    assert q8_args.kv_cache_quantization == "q8"
    assert "prompt-dependent cold/warm divergence" in caplog.text

    none_args = _policy_args("none", explicit=True)
    assert _apply_family_kv_storage_policy(none_args, config, logger) == "unchanged"
    assert none_args.kv_cache_quantization == "none"


def test_family_kv_storage_policy_leaves_unrelated_family_unchanged():
    args = _policy_args("q4", explicit=False)
    unrelated = ModelConfig(family_name="llama", model_types=["llama"])

    assert (
        _apply_family_kv_storage_policy(
            args,
            unrelated,
            logging.getLogger("test_unrelated_kv_policy"),
        )
        == "unchanged"
    )
    assert args.kv_cache_quantization == "q4"


def test_conversion_uses_temporary_minicpm_config_overlay(tmp_path):
    model_dir = tmp_path / "MiniCPM4-0.5B"
    model_dir.mkdir()
    original_config = {
        "architectures": ["MiniCPMForCausalLM"],
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }
    (model_dir / "config.json").write_text(
        json.dumps(original_config),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")

    with _conversion_source_with_model_config_compat(model_dir) as source:
        overlay_dir = Path(source)
        assert overlay_dir != model_dir
        overlay_config = json.loads((overlay_dir / "config.json").read_text())
        assert overlay_config["model_type"] == "minicpm"
        assert overlay_config["rope_theta"] == 10_000.0
        assert (overlay_dir / "model.safetensors").is_symlink()
        assert (overlay_dir / "model.safetensors").resolve() == (
            model_dir / "model.safetensors"
        )

    assert not overlay_dir.exists()
    assert json.loads((model_dir / "config.json").read_text()) == original_config


def test_conversion_leaves_unrelated_model_source_unchanged(tmp_path):
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )

    with _conversion_source_with_model_config_compat(model_dir) as source:
        assert Path(source) == model_dir


def test_jang_conversion_uses_temporary_minicpm_config_overlay(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "MiniCPM4-0.5B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniCPMForCausalLM"],
                "scale_emb": 12,
                "scale_depth": 1.4,
                "dim_model_base": 256,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")

    observed = {}

    def fake_convert_model(**kwargs):
        source = Path(kwargs["model_path"])
        observed["source"] = source
        observed["source_name"] = source.name
        observed["config"] = json.loads((source / "config.json").read_text())
        observed["weight_is_symlink"] = (source / "model.safetensors").is_symlink()
        return {"actual_bits": 4.5, "total_weight_gb": 0.25}

    fake_package = types.ModuleType("jang_tools")
    fake_convert = types.ModuleType("jang_tools.convert")
    fake_convert.convert_model = fake_convert_model
    monkeypatch.setitem(sys.modules, "jang_tools", fake_package)
    monkeypatch.setitem(sys.modules, "jang_tools.convert", fake_convert)

    result = _run_jang_conversion(
        model_path=model_dir,
        output_path=str(tmp_path / "output"),
        target_bits=4,
        profile="JANG_4M",
        quantization_method="mse",
        calibration_method="weights",
        imatrix_path=None,
        use_awq=False,
        awq_alpha=0.25,
    )

    assert result["actual_bits"] == 4.5
    assert observed["source_name"] == model_dir.name
    assert observed["config"]["model_type"] == "minicpm"
    assert observed["config"]["rope_theta"] == 10_000.0
    assert observed["weight_is_symlink"] is True
    assert not observed["source"].exists()
    assert "model_type" not in json.loads((model_dir / "config.json").read_text())
    assert "rope_theta" not in json.loads((model_dir / "config.json").read_text())

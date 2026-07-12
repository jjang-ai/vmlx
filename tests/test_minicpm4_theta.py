import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmlx_engine.commands.convert import _conversion_source_with_model_config_compat
from vmlx_engine.model_config_registry import _minicpm_model_config_compat_override
from vmlx_engine.prefix_cache import compute_model_cache_key
from vmlx_engine.utils.jang_loader import _load_config_with_minicpm_compat


def _exact_minicpm_config(**updates):
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "hidden_size": 1024,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }
    config.update(updates)
    return config


def test_typed_minicpm_only_receives_absent_theta_default():
    config = _exact_minicpm_config(model_type="minicpm")

    assert _minicpm_model_config_compat_override(config) == {"rope_theta": 10_000.0}
    assert "rope_theta" not in config


@pytest.mark.parametrize("explicit_theta", [None, 0, -1, "custom", 12_345.5])
def test_typed_minicpm_preserves_every_explicit_theta_value(explicit_theta):
    config = _exact_minicpm_config(
        model_type="minicpm",
        rope_theta=explicit_theta,
    )

    assert _minicpm_model_config_compat_override(config) is None
    assert config["rope_theta"] == explicit_theta


@pytest.mark.parametrize("explicit_theta", [None, 0, -1, "custom", 12_345.5])
def test_legacy_minicpm_adds_only_identity_when_theta_is_explicit(explicit_theta):
    config = _exact_minicpm_config(rope_theta=explicit_theta)

    assert _minicpm_model_config_compat_override(config) == {"model_type": "minicpm"}
    assert config["rope_theta"] == explicit_theta


@pytest.mark.parametrize(
    "config",
    [
        _exact_minicpm_config(model_type="custom_minicpm"),
        _exact_minicpm_config(model_type="MiniCPM"),
        _exact_minicpm_config(model_type="minicpmv"),
        {
            "model_type": "minicpm",
            "architectures": ["MiniCPMVForCausalLM"],
            "scale_emb": 12,
            "scale_depth": 1.4,
            "dim_model_base": 256,
        },
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "scale_emb": 12,
            "scale_depth": 1.4,
            "dim_model_base": 256,
        },
        {
            "architectures": ["MiniCPMForCausalLM"],
            "scale_emb": 12,
            "scale_depth": 1.4,
        },
    ],
)
def test_minicpm_normalizer_refuses_lookalikes_and_other_families(config):
    before = json.loads(json.dumps(config))

    assert _minicpm_model_config_compat_override(config) is None
    assert config == before


def test_typed_conversion_overlay_adds_theta_without_rewriting_source(tmp_path):
    model_dir = tmp_path / "minicpm-typed"
    model_dir.mkdir()
    original = _exact_minicpm_config(model_type="minicpm")
    (model_dir / "config.json").write_text(json.dumps(original), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    with _conversion_source_with_model_config_compat(model_dir) as source:
        overlay = Path(source)
        assert overlay != model_dir
        effective = json.loads((overlay / "config.json").read_text())
        assert effective["model_type"] == "minicpm"
        assert effective["rope_theta"] == 10_000.0

    assert json.loads((model_dir / "config.json").read_text()) == original


def test_jang_config_wrapper_changes_only_exact_minicpm():
    minicpm = _exact_minicpm_config(model_type="minicpm")
    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}

    normalized = _load_config_with_minicpm_compat(lambda _path: minicpm, Path("m"))
    untouched = _load_config_with_minicpm_compat(lambda _path: llama, Path("l"))

    assert normalized == {**minicpm, "rope_theta": 10_000.0}
    assert "rope_theta" not in minicpm
    assert untouched is llama


def _runtime_model(model_type, theta):
    return SimpleNamespace(
        args=SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_size=1024,
            vocab_size=73_448,
            rope_theta=theta,
        )
    )


def test_cache_key_separates_only_minicpm_theta(monkeypatch):
    monkeypatch.setattr(
        "vmlx_engine.prefix_cache.runtime_cache_fingerprint",
        lambda: "runtime=test",
    )

    minicpm_10k = compute_model_cache_key(
        _runtime_model("minicpm", 10_000), model_path="/models/minicpm"
    )
    minicpm_10k_float = compute_model_cache_key(
        _runtime_model("minicpm", 10_000.0), model_path="/models/minicpm"
    )
    minicpm_1m = compute_model_cache_key(
        _runtime_model("minicpm", 1_000_000.0), model_path="/models/minicpm"
    )
    llama_10k = compute_model_cache_key(
        _runtime_model("llama", 10_000.0), model_path="/models/llama"
    )
    llama_1m = compute_model_cache_key(
        _runtime_model("llama", 1_000_000.0), model_path="/models/llama"
    )

    assert minicpm_10k == minicpm_10k_float
    assert minicpm_10k != minicpm_1m
    assert llama_10k == llama_1m

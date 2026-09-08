# SPDX-License-Identifier: Apache-2.0
"""Source/unit contracts for Nanbeige's looped-transformer integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _write_stub_shard(root, name="model.safetensors"):
    """A minimal valid safetensors file so bundle integrity (which now requires a
    weight shard) accepts a test bundle that never loads weights."""
    import json as _json, struct as _struct
    header = _json.dumps({"__metadata__": {"format": "pt"}}).encode()
    (root / name).write_bytes(_struct.pack("<Q", len(header)) + header)


def _write_bundle(tmp_path, *, runtime_slots: int = 44):
    config = {
        "model_type": "nanbeige",
        "num_hidden_layers": 22,
        "num_loops": 2,
        "jang_runtime": {
            "cache_layout": "looped_kv_v1",
            "num_hidden_layers": 22,
            "num_loops": 2,
            "cache_slots": runtime_slots,
        },
    }
    jang = {
        "version": 2,
        "weight_format": "affine",
        "runtime": {
            "num_loops": 2,
            "cache_slots": 44,
        },
        "capabilities": {
            "family": "nanbeige",
            "cache_type": "kv",
            "reasoning_parser": "qwen3",
            "tool_parser": "xml_function",
            "think_in_template": True,
            "supports_thinking": True,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "jang_config.json").write_text(json.dumps(jang))
    _write_stub_shard(tmp_path)
    return config, jang


class _FakeNanbeige:
    def __init__(self, *, made_slots: int = 44, reported_slots: int = 44):
        self.args = SimpleNamespace(
            num_hidden_layers=22,
            total_loops=2,
            num_loops=2,
        )
        self.layers = [object() for _ in range(22)]
        self.cache_slots = reported_slots
        self._made_slots = made_slots

    def make_cache(self):
        return [object() for _ in range(self._made_slots)]


def test_nanbeige_registration_installs_both_runtime_modules(tmp_path, monkeypatch):
    from vmlx_engine.utils import nanbeige_runtime

    _write_bundle(tmp_path)
    imported = []

    def _import(name):
        imported.append(name)
        return object()

    monkeypatch.setattr(nanbeige_runtime.importlib, "import_module", _import)

    assert nanbeige_runtime.ensure_nanbeige_runtime_registered(tmp_path) is True
    assert imported == [
        "jang_tools.nanbeige.mlx_register",
        "mlx_lm.models.nanbeige",
    ]


def test_nanbeige_registration_fails_closed_only_for_known_bundle(
    tmp_path, monkeypatch
):
    from vmlx_engine.utils import nanbeige_runtime

    _write_bundle(tmp_path)
    attempts = []

    def _missing(name):
        attempts.append(name)
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(nanbeige_runtime.importlib, "import_module", _missing)

    with pytest.raises(RuntimeError, match="loop-aware"):
        nanbeige_runtime.ensure_nanbeige_runtime_registered(tmp_path)

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert (
        nanbeige_runtime.ensure_nanbeige_runtime_registered(unrelated) is False
    )
    assert attempts == ["jang_tools.nanbeige.mlx_register"]


def test_nanbeige_loop_cache_contract_joins_all_authoritative_sources(tmp_path):
    from vmlx_engine.utils.nanbeige_runtime import (
        validate_nanbeige_loop_cache_contract,
    )

    _write_bundle(tmp_path)
    model = _FakeNanbeige()

    contract = validate_nanbeige_loop_cache_contract(model, tmp_path)

    assert contract == {
        "cache_layout": "looped_kv_v1",
        "num_hidden_layers": 22,
        "num_loops": 2,
        "cache_slots": 44,
    }
    assert model._vmlx_looped_cache_contract == contract


def test_nanbeige_cache_identity_includes_validated_loop_layout_without_paths():
    import inspect

    from vmlx_engine.prefix_cache import _looped_cache_identity_parts
    from vmlx_engine.scheduler import Scheduler

    model = _FakeNanbeige()
    model._vmlx_looped_cache_contract = {
        "cache_layout": "looped_kv_v1",
        "num_hidden_layers": 22,
        "num_loops": 2,
        "cache_slots": 44,
    }

    assert _looped_cache_identity_parts(model) == [
        "cache_layout=looped_kv_v1",
        "num_loops=2",
        "cache_slots=44",
        "looped_cache_shape=22x2=44",
    ]
    scheduler_source = inspect.getsource(Scheduler.__init__)
    # The looped-model identity is now folded into the shared namespace builder
    # rather than appended by each scheduler, so BOTH the text and the VLM path
    # get it (the VLM path previously did not).
    from vmlx_engine.prefix_cache import build_block_cache_namespace
    import inspect as _inspect

    assert "build_block_cache_namespace(" in scheduler_source
    assert "looped_cache_identity_scope(model)" in _inspect.getsource(
        build_block_cache_namespace
    )
    assert "scope_key = _append_looped_cache_identity_scope(" in scheduler_source


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("cache_layout", "looped_kv_v2"),
        ("num_loops", 3),
        ("cache_slots", 22),
    ),
)
def test_nanbeige_cache_identity_and_text_l2_namespace_invalidate_layout_drift(
    monkeypatch,
    field,
    changed_value,
):
    from vmlx_engine import prefix_cache
    from vmlx_engine.scheduler import _append_looped_cache_identity_scope

    monkeypatch.setattr(
        prefix_cache,
        "runtime_cache_fingerprint",
        lambda: "runtime_cache=test",
    )
    contract = {
        "cache_layout": "looped_kv_v1",
        "num_hidden_layers": 22,
        "num_loops": 2,
        "cache_slots": 44,
    }
    baseline = _FakeNanbeige()
    baseline._vmlx_looped_cache_contract = dict(contract)
    changed = _FakeNanbeige()
    changed._vmlx_looped_cache_contract = dict(contract)
    changed._vmlx_looped_cache_contract[field] = changed_value

    assert prefix_cache.compute_model_cache_key(
        baseline
    ) != prefix_cache.compute_model_cache_key(changed)
    base_scope = "/same/model:quant=q4:runtime_cache=test"
    assert _append_looped_cache_identity_scope(
        base_scope, baseline
    ) != _append_looped_cache_identity_scope(base_scope, changed)


def test_ordinary_model_cache_identity_ignores_non_looped_runtime_metadata(
    monkeypatch,
):
    from vmlx_engine import prefix_cache
    from vmlx_engine.scheduler import _append_looped_cache_identity_scope

    monkeypatch.setattr(
        prefix_cache,
        "runtime_cache_fingerprint",
        lambda: "runtime_cache=test",
    )
    base_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "vocab_size": 32000,
    }
    plain = SimpleNamespace(config=SimpleNamespace(**base_config))
    with_irrelevant_fields = SimpleNamespace(
        config=SimpleNamespace(
            **base_config,
            jang_runtime={"num_loops": 2, "cache_slots": 44},
        )
    )

    assert prefix_cache.compute_model_cache_key(
        plain
    ) == prefix_cache.compute_model_cache_key(with_irrelevant_fields)
    base_scope = "/same/model:quant=q4:runtime_cache=test"
    assert _append_looped_cache_identity_scope(base_scope, plain) == base_scope
    assert (
        _append_looped_cache_identity_scope(base_scope, with_irrelevant_fields)
        == base_scope
    )


def test_nanbeige_rejects_silent_22_slot_stock_cache(tmp_path):
    from vmlx_engine.utils.nanbeige_runtime import (
        validate_nanbeige_loop_cache_contract,
    )

    _write_bundle(tmp_path)

    with pytest.raises(RuntimeError, match="slot mismatch"):
        validate_nanbeige_loop_cache_contract(
            _FakeNanbeige(made_slots=22),
            tmp_path,
        )


def test_nanbeige_rejects_mismatched_bundle_cache_stamp(tmp_path):
    from vmlx_engine.utils.nanbeige_runtime import (
        validate_nanbeige_loop_cache_contract,
    )

    _write_bundle(tmp_path, runtime_slots=22)

    with pytest.raises(RuntimeError, match="slot mismatch"):
        validate_nanbeige_loop_cache_contract(_FakeNanbeige(), tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    (
        ("num_hidden_layers", 21, "layer mismatch"),
        ("num_loops", 3, "loop-count mismatch"),
    ),
)
def test_nanbeige_rejects_mismatched_config_runtime_shape_stamps(
    tmp_path,
    field,
    bad_value,
    error,
):
    from vmlx_engine.utils.nanbeige_runtime import (
        validate_nanbeige_loop_cache_contract,
    )

    config, _ = _write_bundle(tmp_path)
    config["jang_runtime"][field] = bad_value
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match=error):
        validate_nanbeige_loop_cache_contract(_FakeNanbeige(), tmp_path)


def test_public_jang_loader_applies_nanbeige_post_load_contract(
    tmp_path, monkeypatch
):
    import vmlx_engine.utils.jang_loader as loader
    from vmlx_engine.utils import nanbeige_runtime

    _write_bundle(tmp_path)
    model = _FakeNanbeige()
    tokenizer = object()

    monkeypatch.setattr(
        nanbeige_runtime,
        "ensure_nanbeige_runtime_registered",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(loader, "_ensure_zaya_runtime_supported", lambda *_: None)
    monkeypatch.setattr(loader, "_is_codebook_vq_model", lambda *_: False)
    monkeypatch.setattr(loader, "_is_v2_model", lambda *_: True)
    monkeypatch.setattr(loader, "_load_jang_v2", lambda *args, **kwargs: (model, tokenizer))

    loaded, loaded_tokenizer = loader.load_jang_model(tmp_path)

    assert loaded is model
    assert loaded_tokenizer is tokenizer
    assert loaded._vmlx_looped_cache_contract["cache_slots"] == 44


def test_generic_loader_uses_shared_nanbeige_registration_and_validator(
    tmp_path, monkeypatch
):
    import mlx_lm

    import vmlx_engine.utils.nanbeige_runtime as nanbeige_runtime
    import vmlx_engine.utils.tokenizer as tokenizer_utils

    config = {
        "model_type": "nanbeige",
        "num_hidden_layers": 22,
        "num_loops": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_stub_shard(tmp_path)
    model = _FakeNanbeige()
    tokenizer = object()
    calls = []

    monkeypatch.setattr(
        nanbeige_runtime,
        "ensure_nanbeige_runtime_registered",
        lambda path, **kwargs: calls.append(("register", str(path))) or True,
    )
    monkeypatch.setattr(
        nanbeige_runtime,
        "validate_nanbeige_loop_cache_contract",
        lambda loaded, path, **kwargs: calls.append(
            ("validate", loaded, str(path))
        ),
    )
    monkeypatch.setattr(
        mlx_lm,
        "load",
        lambda *args, **kwargs: (model, tokenizer),
    )
    monkeypatch.setattr(
        tokenizer_utils,
        "_inject_chat_template_if_missing",
        lambda *args, **kwargs: None,
    )

    loaded, loaded_tokenizer = tokenizer_utils.load_model_with_fallback(
        str(tmp_path),
        skip_turboquant=True,
    )

    assert loaded is model
    assert loaded_tokenizer is tokenizer
    assert calls == [
        ("register", str(tmp_path)),
        ("validate", model, str(tmp_path)),
    ]


def test_python_registry_declares_nanbeige_protocol_and_dual_eos():
    from vmlx_engine.model_config_registry import ModelConfigRegistry
    from vmlx_engine.model_configs import register_all

    registry = ModelConfigRegistry()
    register_all(registry)
    config = next(
        row for row in registry._configs if row.family_name == "nanbeige"
    )

    assert config.model_types == ["nanbeige"]
    assert config.cache_type == "kv"
    assert config.eos_tokens == ["<|im_end|>", "<|endoftext|>"]
    assert config.reasoning_parser == "qwen3"
    assert config.tool_parser == "xml_function"
    assert config.supports_thinking is True
    assert config.supports_native_tools is True
    assert config.think_in_template is True
    assert config.is_mllm is False
    assert config.architecture_hints == {
        "default_enable_thinking": True,
        "cache_schema": "looped_kv_v1",
        "num_loops": 2,
        "cache_slots": 44,
        "eos_token_ids": [166101, 166102],
    }


def test_external_speculative_decode_is_disabled_for_nanbeige(tmp_path, monkeypatch):
    import vmlx_engine.speculative as speculative

    _write_bundle(tmp_path)
    monkeypatch.setattr(
        speculative,
        "_spec_config",
        speculative.SpeculativeConfig(model="draft"),
    )
    monkeypatch.setattr(speculative, "_draft_model", object())

    reason = speculative.external_speculative_incompatibility_reason(str(tmp_path))

    assert reason is not None
    assert "44 slots" in reason
    assert (
        speculative.should_use_speculative(
            is_batched=False,
            is_mllm=False,
            target_model_name=str(tmp_path),
        )
        is False
    )


def test_cli_external_speculative_decode_is_disabled_for_nanbeige(tmp_path):
    from vmlx_engine.cli import _speculative_incompatibility_reason

    _write_bundle(tmp_path)
    args = SimpleNamespace(
        speculative_model="draft",
        continuous_batching=False,
        model=str(tmp_path),
        is_mllm=False,
        force_text_only=False,
    )

    reason = _speculative_incompatibility_reason(args)

    assert reason is not None
    assert "44 slots" in reason


def test_nanbeige_is_explicitly_excluded_from_native_mtp_runtime(
    tmp_path, monkeypatch
):
    import vmlx_engine.native_mtp as native_mtp

    config, jang = _write_bundle(tmp_path)
    config["num_nextn_predict_layers"] = 1
    jang["runtime"].update({"mtp_layers": 1, "bundle_has_mtp": True})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "jang_config.json").write_text(json.dumps(jang))
    monkeypatch.setattr(
        native_mtp,
        "_bundle_weight_keys",
        lambda _path: (["mtp.0.proj.weight"], "test", None),
    )

    status = native_mtp.inspect_native_mtp_bundle(tmp_path)

    assert status["family"] == "nanbeige"
    assert status["artifact_available"] is True
    assert status["runtime_supported"] is False
    assert status["runtime_available"] is False
    assert status["runtime_active"] is False
    assert status["status"] == "weights_present_runtime_unwired"

# SPDX-License-Identifier: Apache-2.0
"""DSV4 post-9c688af5 / 3e256e3f contract-hardening regression pins.

These tests do NOT require a live model load. They guard against silent
regressions of the DSV4 contract that landed across:

- ``9c688af5`` Harden DSV4 thinking and runtime contracts
- ``3e256e3f`` Update DSV4 production gate contracts

Specifically:

1. The removed env-var force-flips (``VMLX_DSV4_ALLOW_CHAT`` /
   ``VMLX_DSV4_ALLOW_THINKING`` / ``VMLX_DSV4_FORCE_DIRECT_RAIL``) must not
   reappear anywhere under ``vmlx_engine/``.
2. The capabilities payload for ``family == "deepseek_v4"`` must report empty
   ``experimental_modes`` (no leftover ``raw-thinking`` shape).
3. ``_native_cache_status`` for the DSV4 branch must report the
   ``deepseek_v4_v7`` schema, ``cache_type == "native_composite"``, and
   ``generic_turboquant_kv.enabled is False`` (per
   ``~/wiki/research/topics/path-dependent-cache-restore.md`` the composite
   cache IS the cache-size strategy; layering generic TQ-KV on top would
   double-quantize the compressed CSA/HCA latents).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT / "vmlx_engine"


def _engine_python_files() -> list[Path]:
    return [p for p in ENGINE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_removed_dsv4_force_flip_env_vars_absent_from_vmlx_engine():
    """Removed DSV4 rail force env vars must stay gone.

    Both env vars used to force-flip the DSV4 thinking rail. They were removed
    in favour of ``_resolve_dsv4_thinking_policy``. If they reappear, the new
    rail-resolution contract has been broken or shadowed.
    """
    forbidden = (
        "VMLX_DSV4_ALLOW_CHAT",
        "VMLX_DSV4_ALLOW_THINKING",
        "VMLX_DSV4_FORCE_DIRECT_RAIL",
    )
    offenders: list[tuple[Path, str]] = []
    for path in _engine_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in forbidden:
            if needle in text:
                offenders.append((path, needle))

    assert not offenders, (
        "Removed DSV4 force-flip env var(s) reappeared under vmlx_engine/: "
        + ", ".join(f"{p.relative_to(REPO_ROOT)} -> {n}" for p, n in offenders)
    )


def test_dsv4_capabilities_endpoint_emits_current_contract(monkeypatch):
    """The actual capabilities endpoint must emit the current DSV4 contract.

    This pins the endpoint behavior directly instead of grepping source text.
    A regression that re-introduces the stale ``raw-thinking`` capability shape
    will fail here without needing a live model load.
    """
    from vmlx_engine import model_config_registry, server

    cfg = SimpleNamespace(
        family_name="deepseek_v4",
        reasoning_parser="deepseek_r1",
        tool_parser="deepseek",
        think_in_template=False,
        is_mllm=False,
        supports_thinking=None,
    )

    class FakeRegistry:
        def lookup(self, _model_key):
            return cfg

    fake_scheduler = SimpleNamespace(
        _model_type_for_runtime="deepseek_v4",
        _uses_dsv4_cache=True,
        config=SimpleNamespace(enable_prefix_cache=True),
        block_aware_cache=object(),
        paged_cache_manager=SimpleNamespace(_disk_store=object()),
        memory_aware_cache=None,
        prefix_cache=None,
    )

    monkeypatch.setattr(model_config_registry, "get_model_config_registry", lambda: FakeRegistry())
    monkeypatch.setattr(server, "_get_scheduler", lambda: fake_scheduler)
    monkeypatch.setattr(server, "_loaded_omni_modalities", lambda: None)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_model_path", "/tmp/DeepSeek-V4-Flash-JANGTQ-V3-F32-MIXED")
    monkeypatch.setattr(server, "_model_name", None)
    monkeypatch.setattr(server, "_engine", None)

    payload = asyncio.run(server.model_capabilities("DeepSeek-V4-Flash"))

    assert payload["supports_thinking"] is True
    assert payload["supported_modes"] == ["instruct", "reasoning"]
    assert payload["experimental_modes"] == []
    assert payload["reasoning_efforts"] == ["high", "max"]
    assert payload["cache"]["native"]["family"] == "deepseek_v4"
    assert payload["cache"]["native"]["schema"] == "deepseek_v4_v7"
    assert payload["cache"]["native"]["cache_type"] == "native_composite"
    assert payload["cache"]["native"]["generic_turboquant_kv"]["enabled"] is False


def test_dsv4_native_cache_status_reports_native_composite_v7_schema():
    """The DSV4 branch of ``_native_cache_status`` must keep its current shape.

    The shape is contract-checked live by
    ``run_production_family_audit.capability_endpoint_contract_ok``; this test
    pins it against a synthetic scheduler so a unit run catches drift even
    when no DSV4 bundle is loaded.
    """
    from vmlx_engine import server

    fake_scheduler = SimpleNamespace(
        _model_type_for_runtime="deepseek_v4",
        _uses_dsv4_cache=True,
        block_aware_cache=None,
        paged_cache_manager=None,
    )
    status = server._native_cache_status(fake_scheduler, family="deepseek_v4", cfg=None)

    assert status["family"] == "deepseek_v4"
    assert status["schema"] == "deepseek_v4_v7"
    assert status["cache_type"] == "native_composite"
    assert status["generic_turboquant_kv"]["enabled"] is False
    assert status["generic_turboquant_kv"]["reason"] == "native_dsv4_composite"
    expected_components = {
        "swa_local",
        "csa_compressed_pool",
        "hca_compressed_pool",
        "incomplete_tail_state",
    }
    assert expected_components.issubset(set(status["components"]))


def test_dsv4_native_cache_status_reports_ratio_and_window_contract():
    from vmlx_engine import server

    class _Local:
        max_size = 128

    class _DSV4Cache:
        local = _Local()

        def __init__(self, ratio):
            self.compress_ratio = ratio

    fake_scheduler = SimpleNamespace(
        _model_type_for_runtime="deepseek_v4",
        _uses_dsv4_cache=True,
        block_aware_cache=None,
        paged_cache_manager=None,
        model=SimpleNamespace(
            make_cache=lambda: [
                _DSV4Cache(0),
                _DSV4Cache(0),
                _DSV4Cache(4),
                _DSV4Cache(128),
                _DSV4Cache(4),
                _DSV4Cache(0),
            ]
        ),
    )

    status = server._native_cache_status(fake_scheduler, family="deepseek_v4", cfg=None)

    assert status["sliding_window"] == 128
    assert status["layers"] == 6
    assert status["compress_ratio_counts"] == {"0": 3, "4": 2, "128": 1}
    assert status["layer_cache_roles"] == {
        "ratio_0": "swa_local_only",
        "ratio_4": "csa_overlap_compressed_pool_plus_indexer",
        "ratio_128": "hca_compressed_pool",
    }
    assert status["cache_store_policy"] == {
        "prompt_boundary_snapshot": "preferred",
        "post_generation_trim": "disabled_unless_explicit_unsafe_override",
        "generic_kv_quantization": "forced_off",
    }


def test_dsv4_capability_runner_check_accepts_current_contract_only():
    """Mirror of the runner shape check; guards against the stale shape passing.

    ``capability_endpoint_contract_ok`` already lives in the cross-matrix
    runner; pinning a DSV4-row case here keeps the runner's shape contract
    visible from the unit-test set.
    """
    from tests.cross_matrix.run_production_family_audit import (
        ROWS,
        capability_endpoint_contract_ok,
    )

    dsv4 = next(row for row in ROWS if row.family == "deepseek_v4")

    current_caps = {
        "supports_thinking": True,
        "supported_modes": ["instruct", "reasoning"],
        "experimental_modes": [],
        "reasoning_efforts": ["high", "max"],
        "cache": {
            "native": {
                "family": "deepseek_v4",
                "schema": "deepseek_v4_v7",
                "cache_type": "native_composite",
                "generic_turboquant_kv": {"enabled": False},
            }
        },
    }
    assert capability_endpoint_contract_ok(dsv4, current_caps)

    stale_caps = {
        "supports_thinking": False,
        "supported_modes": ["instruct"],
        "experimental_modes": ["raw-thinking"],
        "reasoning_efforts": [],
        "cache": {"dsv4_composite_state": True},
    }
    assert not capability_endpoint_contract_ok(dsv4, stale_caps)


def test_dsv4_attention_contract_verifier_does_not_patch_current_jang():
    """Current JANG already carries the native DSV4 attention contract.

    vMLX must not replace ``DeepseekV4Attention.__call__`` in that case; a
    hidden class-level patch changes the runtime being debugged and can mask
    compressed-pool attention regressions.
    """
    from jang_tools.dsv4.mlx_model import DeepseekV4Attention
    from vmlx_engine.loaders import load_jangtq_dsv4

    original_call = DeepseekV4Attention.__call__
    original_flag = load_jangtq_dsv4._PREFILL_PATCH_INSTALLED
    try:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = False
        load_jangtq_dsv4._verify_dsv4_attention_contract()
        assert DeepseekV4Attention.__call__ is original_call
        assert load_jangtq_dsv4._PREFILL_PATCH_INSTALLED is True
    finally:
        DeepseekV4Attention.__call__ = original_call
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = original_flag


def test_dsv4_loader_has_no_attention_monkeypatch_fallback():
    """DSV4 production must fail loudly on stale JANG, not monkeypatch it.

    The previous compatibility fallback copied a stale attention forward and
    lost JANG's per-query compressed-pool mask semantics. Keep this source pin
    so the fallback cannot quietly return.
    """
    source = (ENGINE_ROOT / "loaders" / "load_jangtq_dsv4.py").read_text(
        encoding="utf-8"
    )

    assert "DeepseekV4Attention.__call__ = _patched_call" not in source
    assert "def _patched_call(" not in source
    assert "DSV4 installed jang_tools is too old" in source
    assert "_install_dsv4_prefill_patch" not in source
    assert "_verify_dsv4_attention_contract" in source


_DSV4_ATTENTION_MARKERS = {
    "symmetric_mask_trim": "if attn_mask.shape[-1] > full_kv.shape[2]",
    "per_query_compressed_pool_mask": "comp_mask = comp_mask & selected",
    "indexer_topk_threshold": "pooled.shape[1] > self.indexer.index_topk",
}


@pytest.mark.parametrize("missing_name", sorted(_DSV4_ATTENTION_MARKERS))
def test_dsv4_attention_contract_verifier_rejects_each_missing_marker(
    monkeypatch, missing_name
):
    """A stale JANG attention implementation must fail before inference."""
    import inspect

    from vmlx_engine.loaders import load_jangtq_dsv4

    source = "\n".join(
        needle
        for name, needle in _DSV4_ATTENTION_MARKERS.items()
        if name != missing_name
    )
    original_flag = load_jangtq_dsv4._PREFILL_PATCH_INSTALLED
    try:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = False
        monkeypatch.setattr(inspect, "getsource", lambda _obj: source)
        with pytest.raises(RuntimeError) as exc:
            load_jangtq_dsv4._verify_dsv4_attention_contract()
    finally:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = original_flag

    assert missing_name in str(exc.value)
    assert "monkeypatches this class" in str(exc.value)


def test_dsv4_attention_contract_verifier_rejects_uninspectable_source(monkeypatch):
    """Packaged JANG must ship inspectable source or the loader fails loudly."""
    import inspect

    from vmlx_engine.loaders import load_jangtq_dsv4

    original_flag = load_jangtq_dsv4._PREFILL_PATCH_INSTALLED
    try:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = False

        def _raise(_obj):
            raise OSError("no source")

        monkeypatch.setattr(inspect, "getsource", _raise)
        with pytest.raises(RuntimeError, match="source inspection failed"):
            load_jangtq_dsv4._verify_dsv4_attention_contract()
    finally:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = original_flag


def test_dsv4_attention_contract_verifier_is_idempotent(monkeypatch):
    """After one successful verification, repeat calls must be no-ops."""
    import inspect

    from vmlx_engine.loaders import load_jangtq_dsv4

    original_flag = load_jangtq_dsv4._PREFILL_PATCH_INSTALLED
    try:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = False
        load_jangtq_dsv4._verify_dsv4_attention_contract()

        def _raise(_obj):
            raise AssertionError("getsource should not run on second call")

        monkeypatch.setattr(inspect, "getsource", _raise)
        load_jangtq_dsv4._verify_dsv4_attention_contract()
    finally:
        load_jangtq_dsv4._PREFILL_PATCH_INSTALLED = original_flag


def test_bundled_jang_dsv4_attention_contract_markers_present():
    """The bundled Python copy must not ship stale DSV4 attention code."""
    bundled = (
        REPO_ROOT
        / "panel"
        / "bundled-python"
        / "python"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "jang_tools"
        / "dsv4"
        / "mlx_model.py"
    )
    if not bundled.exists():
        pytest.skip("bundled-python JANG copy not present in this checkout")

    source = bundled.read_text(encoding="utf-8", errors="replace")
    missing = [
        name for name, needle in _DSV4_ATTENTION_MARKERS.items()
        if needle not in source
    ]
    assert not missing


def test_dsv4_runtime_config_reinjects_source_yarn_rope_scaling():
    """Converted DSV4 Flash bundles must not run compressed layers with null YaRN."""
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _DSV4_FLASH_REQUIRED_ROPE_SCALING,
        _normalize_dsv4_runtime_config,
    )

    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 0, 4, 128, 0],
        "rope_scaling": None,
    }

    repaired, changed = _normalize_dsv4_runtime_config(
        config,
        model_path="/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
    )

    assert changed is True
    assert repaired is not config
    assert repaired["rope_scaling"] == _DSV4_FLASH_REQUIRED_ROPE_SCALING
    assert config["rope_scaling"] is None


def test_dsv4_runtime_config_preserves_existing_rope_scaling():
    from vmlx_engine.loaders.load_jangtq_dsv4 import _normalize_dsv4_runtime_config

    existing = {
        "type": "yarn",
        "factor": 8,
        "original_max_position_embeddings": 32768,
        "beta_fast": 16,
        "beta_slow": 2,
    }
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 4, 0],
        "rope_scaling": existing,
    }

    repaired, changed = _normalize_dsv4_runtime_config(config)

    assert changed is False
    assert repaired is config
    assert repaired["rope_scaling"] == existing


def test_dsv4_normalized_load_config_is_scoped_and_restored(tmp_path):
    import mlx_lm.utils as mlx_lm_utils

    from vmlx_engine.loaders.load_jangtq_dsv4 import _dsv4_normalized_load_config

    model_dir = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    other_dir = tmp_path / "Other"
    model_dir.mkdir()
    other_dir.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 0, 4, 128, 0],
        "rope_scaling": None,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (other_dir / "config.json").write_text(json.dumps(config))

    original = mlx_lm_utils.load_config
    with _dsv4_normalized_load_config(model_dir):
        assert mlx_lm_utils.load_config(model_dir)["rope_scaling"]["factor"] == 16
        assert mlx_lm_utils.load_config(other_dir)["rope_scaling"] is None

    assert mlx_lm_utils.load_config is original


def test_dsv4_safe_auto_tokenizer_falls_back_to_tokenizer_json(tmp_path, monkeypatch):
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from vmlx_engine.loaders.load_jangtq_dsv4 import _dsv4_safe_auto_tokenizer

    model_dir = tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"
    other_dir = tmp_path / "Other"
    model_dir.mkdir()
    other_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "max_position_embeddings": 1048576,
                "compress_rope_theta": 160000,
                "compress_ratios": [0, 0, 4, 128, 0],
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                },
            }
        )
    )
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<｜User｜>": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))

    def failing_from_pretrained(*args, **kwargs):
        raise ValueError("deepseek_v4 is unknown and rope_scaling rejected")

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        failing_from_pretrained,
    )

    with _dsv4_safe_auto_tokenizer(model_dir):
        loaded = transformers.AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        assert loaded.convert_tokens_to_ids("<｜User｜>") == 1
        try:
            transformers.AutoTokenizer.from_pretrained(str(other_dir))
        except ValueError as exc:
            assert "deepseek_v4" in str(exc)
        else:
            raise AssertionError("non-target tokenizer path should not be hidden")

    assert transformers.AutoTokenizer.from_pretrained is failing_from_pretrained

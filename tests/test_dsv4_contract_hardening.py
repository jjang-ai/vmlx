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
    assert payload["reasoning_efforts"] == ["low", "medium", "high", "max"]
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
        "reasoning_efforts": ["low", "medium", "high", "max"],
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

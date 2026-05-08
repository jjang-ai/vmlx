# SPDX-License-Identifier: Apache-2.0
"""DSV4 post-9c688af5 / 3e256e3f contract-hardening regression pins.

These tests do NOT require a live model load. They guard against silent
regressions of the DSV4 contract that landed across:

- ``9c688af5`` Harden DSV4 thinking and runtime contracts
- ``3e256e3f`` Update DSV4 production gate contracts

Specifically:

1. The removed env-var force-flips (``VMLX_DSV4_ALLOW_CHAT`` /
   ``VMLX_DSV4_ALLOW_THINKING``) must not reappear anywhere under
   ``vmlx_engine/``. The only DSV4 rail debug switch that is allowed today is
   ``VMLX_DSV4_FORCE_DIRECT_RAIL``.
2. The capabilities payload for ``family == "deepseek_v4"`` must report empty
   ``experimental_modes`` (no leftover ``raw-thinking`` shape).
3. ``_native_cache_status`` for the DSV4 branch must report the
   ``deepseek_v4_v7`` schema, ``cache_type == "native_composite"``, and
   ``generic_turboquant_kv.enabled is False`` (per
   ``~/wiki/research/topics/path-dependent-cache-restore.md`` the composite
   cache IS the cache-size strategy; layering generic TQ-KV on top would
   double-quantize the compressed CSA/HSA latents).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT / "vmlx_engine"


def _engine_python_files() -> list[Path]:
    return [p for p in ENGINE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_removed_dsv4_force_flip_env_vars_absent_from_vmlx_engine():
    """``VMLX_DSV4_ALLOW_CHAT`` / ``VMLX_DSV4_ALLOW_THINKING`` must stay gone.

    Both env vars used to force-flip the DSV4 thinking rail. They were removed
    in favour of ``_resolve_dsv4_thinking_policy``. If they reappear, the new
    rail-resolution contract has been broken or shadowed.
    """
    forbidden = ("VMLX_DSV4_ALLOW_CHAT", "VMLX_DSV4_ALLOW_THINKING")
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

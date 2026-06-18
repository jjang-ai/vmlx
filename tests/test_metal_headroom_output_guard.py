# SPDX-License-Identifier: Apache-2.0

"""Contracts for projected Metal headroom output-token guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def test_explicit_max_tokens_over_safe_headroom_rejects(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)

    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(8192, "tight-headroom-model")

    assert exc.value.status_code == 413
    assert "8192" in str(exc.value.detail)
    assert "1024" in str(exc.value.detail)
    assert "Metal" in str(exc.value.detail)


def test_explicit_max_tokens_rejects_when_projected_safe_cap_is_zero(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 0)

    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(1, "no-headroom-model")

    assert exc.value.status_code == 413
    assert "requested=1" in str(exc.value.detail)
    assert "safe_cap=0" in str(exc.value.detail)


def test_implicit_default_over_safe_headroom_clamps(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)

    assert server._resolve_max_tokens(None, "tight-headroom-model") == 1024


def test_implicit_default_with_zero_safe_cap_clamps_to_one(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 0)

    assert server._resolve_max_tokens(None, "no-headroom-model") == 1


def test_explicit_default_over_safe_headroom_rejects(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", True)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)

    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(None, "tight-headroom-model")

    assert exc.value.status_code == 413
    assert "4096" in str(exc.value.detail)
    assert "1024" in str(exc.value.detail)


def test_projection_uses_loaded_model_config(monkeypatch):
    from vmlx_engine import server

    class FakeMX:
        @staticmethod
        def get_active_memory():
            return int(105.41 * 1024**3)

        @staticmethod
        def device_info():
            return {"max_recommended_working_set_size": int(107.52 * 1024**3)}

    cfg = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=4,
        head_dim=8,
        torch_dtype="bfloat16",
    )
    engine = SimpleNamespace(model=SimpleNamespace(config=cfg))

    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(
        server,
        "_metal_projection_stats",
        lambda: (
            int(105.41 * 1024**3),
            int(107.52 * 1024**3),
        ),
    )
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_TRANSIENT_MULTIPLIER", "4")
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_BUDGET_FRACTION", "0.5")

    cap = server._metal_projected_output_token_cap("tight-headroom-model")

    assert cap is not None
    assert cap > 0

# SPDX-License-Identifier: Apache-2.0

"""Focused tests for shared Metal working-set guard helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.utils.memory_limits import (
    estimate_kv_bytes_per_token_from_config,
    get_effective_metal_working_set_bytes,
    get_metal_ws_guard_threshold,
    projected_output_token_cap,
    resolve_working_set_override,
    is_metal_ws_guard_enabled,
    _parse_float_env,
    _parse_working_set_bytes,
)


def test_resolve_working_set_override_clamps_to_base():
    base = 64 * (1024**3)
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_GB": "96"},
        clear=True,
    ):
        assert resolve_working_set_override(base) == base


def test_resolve_working_set_override_bytes_override():
    base = 128 * (1024**3)
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_BYTES": str(32 * 1024**3)},
        clear=True,
    ):
        assert resolve_working_set_override(base) == 32 * (1024**3)


def test_parse_working_set_bytes_rejects_invalid():
    assert _parse_working_set_bytes("abc") is None
    assert _parse_working_set_bytes("") is None
    assert _parse_working_set_bytes("12x") is None


def test_get_effective_metal_working_set_bytes_apply_override():
    mx = SimpleNamespace(
        get_active_memory=lambda: 2 * 1024**3,
        device_info=lambda: {"max_recommended_working_set_size": 64 * 1024**3},
    )
    with patch.dict(
        "os.environ",
        {"VMLX_METAL_WS_MAX_GB": "48"},
        clear=True,
    ):
        active, max_ws = get_effective_metal_working_set_bytes(mx)
        assert active == 2 * 1024**3
        assert max_ws == 48 * 1024**3


def test_guard_threshold_parse_default_and_override():
    assert get_metal_ws_guard_threshold() == 98.0
    assert get_metal_ws_guard_threshold(85.0) == 85.0
    with patch.dict("os.environ", {"VMLX_METAL_WS_REJECT_PCT": "30"}, clear=True):
        assert get_metal_ws_guard_threshold(85.0) == 30.0
    with patch.dict("os.environ", {"VMLX_METAL_WS_REJECT_PCT": "oops"}, clear=True):
        assert get_metal_ws_guard_threshold(85.0) == 85.0


def test_guard_is_enabled_default_and_disable():
    with patch.dict("os.environ", {}, clear=True):
        assert is_metal_ws_guard_enabled() is True
    with patch.dict("os.environ", {"VMLX_METAL_WS_GUARD": "0"}, clear=True):
        assert is_metal_ws_guard_enabled() is False
    with patch.dict("os.environ", {"VMLX_METAL_WS_GUARD": "1"}, clear=True):
        assert is_metal_ws_guard_enabled() is True


def test_parse_float_env_negative_reverts_to_default():
    assert _parse_float_env("MISSING", 42.0) == 42.0


def test_estimate_kv_bytes_per_token_from_text_config_dict():
    cfg = {
        "text_config": {
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "torch_dtype": "bfloat16",
        }
    }

    assert estimate_kv_bytes_per_token_from_config(cfg) == 2 * 2 * 4 * 8 * 2


def test_projected_output_token_cap_accounts_for_transient_multiplier():
    gib = 1024**3
    cap = projected_output_token_cap(
        active_bytes=int(105.41 * gib),
        max_working_set_bytes=int(107.52 * gib),
        bytes_per_token=256 * 1024,
        budget_fraction=0.50,
        transient_multiplier=4.0,
    )

    assert 1000 <= cap <= 1100


def test_scheduler_waiting_uses_shared_memory_helper():
    import inspect

    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.scheduler import Scheduler

    assert "get_effective_metal_working_set_bytes" in inspect.getsource(Scheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold" in inspect.getsource(Scheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold(85.0)" not in inspect.getsource(
        Scheduler._schedule_waiting
    )
    assert (
        "get_effective_metal_working_set_bytes"
        in inspect.getsource(MLLMScheduler._schedule_waiting)
    )
    assert "get_metal_ws_guard_threshold" in inspect.getsource(MLLMScheduler._schedule_waiting)
    assert "get_metal_ws_guard_threshold(85.0)" not in inspect.getsource(
        MLLMScheduler._schedule_waiting
    )

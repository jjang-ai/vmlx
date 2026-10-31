"""MiniMax-M3: a requested paged RAM tier (RAM + SSD) is honoured under the aligned prefill geometry."""

import pytest

from vmlx_engine.cli import _m3_forces_paged_off


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VMLX_M3_ALIGNED_PREFIX", raising=False)
    monkeypatch.delenv("VMLX_M3_ALLOW_PAGED", raising=False)


def test_paged_tier_kept_with_the_default_aligned_geometry():
    assert _m3_forces_paged_off(True) is False


def test_no_paged_request_is_never_changed():
    assert _m3_forces_paged_off(False) is False


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_relative_chunking_still_turns_the_paged_tier_off(monkeypatch, value):
    monkeypatch.setenv("VMLX_M3_ALIGNED_PREFIX", value)
    assert _m3_forces_paged_off(True) is True


def test_experiment_switch_keeps_paged_under_relative_chunking(monkeypatch):
    monkeypatch.setenv("VMLX_M3_ALIGNED_PREFIX", "0")
    monkeypatch.setenv("VMLX_M3_ALLOW_PAGED", "1")
    assert _m3_forces_paged_off(True) is False

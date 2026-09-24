"""The CLI must not re-enable a native media path disabled by its operator."""
import pytest


@pytest.mark.parametrize("inherited", [None, "1", "true", "0"])
def test_force_text_only_clears_inherited_m3_media_gate(monkeypatch, inherited):
    from vmlx_engine.models.minimax_m3.m3_vl_preprocess import (
        configure_m3_vl_environment, m3_vl_enabled,
    )
    if inherited is None:
        monkeypatch.delenv("VMLX_M3_VL", raising=False)
    else:
        monkeypatch.setenv("VMLX_M3_VL", inherited)
    assert configure_m3_vl_environment(has_vision=True, force_text_only=True) is False
    assert m3_vl_enabled() is False


def test_m3_auto_keeps_native_media_enabled(monkeypatch):
    from vmlx_engine.models.minimax_m3.m3_vl_preprocess import configure_m3_vl_environment
    monkeypatch.delenv("VMLX_M3_VL", raising=False)
    assert configure_m3_vl_environment(has_vision=True, force_text_only=False) is True

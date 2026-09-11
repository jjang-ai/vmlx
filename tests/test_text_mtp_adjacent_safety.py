"""Fixed depth is a ceiling; losses descend one measured rung at a time."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("depth", [3, 2, 1])
def test_text_safety_descends_adjacent_rung_and_updates_telemetry(monkeypatch, depth):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane

    state = lane._MtpState()
    state.depth = state.stats.depth = depth
    state.depth_ceiling = 3
    state.ar_step_ms = 10.0
    state.stats.cycles = 40
    trip = SimpleNamespace(
        mtp_ms_per_tok=20.0, ar_baseline=10.0,
        reason=lambda d: f"windowed_ar_safety d{d}",
        log_text=lambda d: f"losing d{d}",
    )
    monkeypatch.setattr(lane, "ar_safety_step", lambda *args, **kwargs: trip)
    fallback = lane._text_mtp_maybe_ar_safety_fallback("adjacent", state)
    assert fallback is (depth == 1)
    assert state.ar_fallback_pending is (depth == 1)
    if depth > 1:
        assert state.depth == depth - 1
        assert state.stats.depth == state.depth
        assert state.ar_safety.cycle_base == 40
    assert state.depth_ceiling == 3

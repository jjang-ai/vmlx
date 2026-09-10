"""A fresh, mean-and-median D1 loss needs no second losing window."""
import pytest


@pytest.mark.parametrize("ceiling", [1, 2, 3])
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("ratio", [1.07, 2.0])
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("median_loses", [False, True])
def test_d1_confirmation_respects_loss_severity(
    monkeypatch, ceiling, measured, ratio, fresh, median_loses
):
    from vmlx_engine import mllm_batch_generator as m
    from vmlx_engine.native_mtp_ar_safety import ArSafetyTrip

    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[],
        depth=1,
    )
    state.depth_ceiling = state.ladder_depth = ceiling
    state.ar_step_ms = 20.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    state.last_ar_measure_emitted = 0 if fresh else -512
    if measured:
        state.ar_tier = m.NativeMTPArTier(depth=ceiling)
        state.ar_tier.step_walls_ms = [20.0] * 16
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    trip = ArSafetyTrip(
        cycles=40, mtp_ms_per_tok=20.0 * ratio, ar_baseline=20.0,
        seed_ar_ms=20.0, margin=1.0, window=8,
        cycle_median_ms_per_tok=20.0 * ratio if median_loses else 20.0,
        cycle_max_ms_per_tok=20.0 * ratio, anchor_cycle_ms=20.0,
        cur_cycle_ms=20.0 * ratio, anchor_context_tokens=100,
        context_now=140,
    )
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: trip)
    expected_fast = measured and fresh and median_loses and ratio > 1.0
    assert m._native_mtp_maybe_ar_safety_fallback("fast-loss", state) is expected_fast
    assert state.ar_fallback_pending is expected_fast
    assert state.depth <= ceiling
    assert state.ar_trip_pending_cycle == (0 if expected_fast else 40)

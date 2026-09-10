"""Recovery stays available after failed probes without preempting AR safety."""
import pytest


@pytest.mark.parametrize("ceiling", [1, 2, 3])
@pytest.mark.parametrize("attempts", [0, 3, 8])
@pytest.mark.parametrize("winning", [False, True])
@pytest.mark.parametrize("due", [False, True])
def test_scheduled_promotion_requires_winning_d1_not_attempt_budget(
    monkeypatch, ceiling, attempts, winning, due
):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    monkeypatch.setattr(m.time, "perf_counter", lambda: 100.0)
    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[], depth=1,
    )
    state.depth_ceiling = state.ladder_depth = ceiling
    state.stats.cycles = 100
    state.promote_at_cycle = 100 if due else 120
    state.promotions = attempts
    state.ar_tier = m.NativeMTPArTier(depth=ceiling)
    state.ar_tier.step_walls_ms = [30.0] * 16
    wall = .020 if winning else .040
    state.ar_safety.ring = [(c, c, 100.0 - (100-c)*wall) for c in range(91,100)]
    state.ar_safety.anchor_cycle_ms = wall * 1000

    m._native_mtp_maybe_ar_safety_fallback("scheduled-recovery", state)
    expected = due and winning and ceiling > 1
    assert state.promote_probe is expected
    assert state.depth == (ceiling if expected else 1)
    assert state.promotions == attempts + int(expected)
    if not winning:
        assert state.ar_fallback_pending

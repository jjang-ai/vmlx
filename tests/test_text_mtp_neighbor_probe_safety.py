"""Neighbor experiments must use probe safety, not settled-phase warmup."""
import pytest


@pytest.mark.parametrize("origin,target", [(1, 2), (2, 3), (2, 1)])
def test_losing_neighbor_probe_is_judged_before_eight_sample_completion(monkeypatch, origin, target):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    from vmlx_engine.native_mtp_recovery import NativeMTPRecovery

    state = lane._MtpState(depth=target, depth_ceiling=3, ar_step_ms=20)
    state.recovery = NativeMTPRecovery("neighbor", 3, False)
    state.recovery.samples.extend([20.0] * 8)
    state.adaptive_value.active_probe_origin = origin
    state.adaptive_value.active_probe_target = target
    state.ar_safety.reset(48)
    clock = [1.0]
    monkeypatch.setattr(lane.time, "perf_counter", lambda: clock[0])
    for offset in range(1, 9):
        clock[0] += 0.120
        state.stats.cycles = 48 + offset
        state.stats.draft_tokens_accepted = 48 + offset
        lane._text_mtp_maybe_ar_safety_fallback("neighbor", state)
    assert state.ar_fallback_pending if target == 1 else state.depth == target - 1
    assert state.depth_ceiling == 3
    assert state.recovery_probe_started == 0.0


def test_unrelated_neighbor_marker_does_not_shorten_settled_warmup(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    state = lane._MtpState(depth=2, depth_ceiling=3, ar_step_ms=20)
    state.adaptive_value.active_probe_origin = 1
    state.adaptive_value.active_probe_target = 3
    calls = []
    monkeypatch.setattr(lane, "ar_safety_step", lambda *a, **k: calls.append(k))
    lane._text_mtp_maybe_ar_safety_fallback("stale", state)
    assert calls[0]["probe"] is False

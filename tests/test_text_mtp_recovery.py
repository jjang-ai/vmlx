"""Recovery scheduling is per request and never substitutes acceptance for cost."""
from types import SimpleNamespace

import pytest

from vmlx_engine.native_mtp_recovery import NativeMTPRecovery


def test_recovery_uses_recent_productive_ar_and_bounded_backoff():
    r = NativeMTPRecovery("a", 3, False)
    r.observe_standard(9000)  # Pipeline transition is excluded from baseline.
    for _ in range(126):
        r.observe_standard(10)
    assert not r.ready
    r.observe_standard(10)
    assert r.ready and r.ar_ms == 10
    assert r.standard_tokens == 128
    assert r.standard_wall_ms == 10270
    for expected in (256, 512, 1024, 2048, 4096, 4096):
        r.park(failed_probe=True)
        assert r.cooldown == r.remaining == expected
        assert not r.ready and not r.samples
    r.park(failed_probe=False)
    assert r.remaining == 128


def _ready_batch():
    r = NativeMTPRecovery("a", 2, False)
    for _ in range(128):
        r.observe_standard(10)
    b = SimpleNamespace(
        uids=["a"], _vmlx_mtp_recovery=r, _next_tokens=object(),
        model=SimpleNamespace(mtp_forward=lambda: None, mtp=[object()]),
        prompt_cache=[], max_tokens=[256], _num_tokens=[128],
    )
    return b, r


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_calibration_preserves_failure_backoff_and_measures_productive_ar(depth):
    r = NativeMTPRecovery("a", depth, False)
    r.park(failed_probe=True)
    r.attempts = 2
    r.park_for_calibration(depth=depth)
    assert (r.cooldown, r.failed_probes, r.attempts) == (256, 1, 2)
    assert r.calibrating and r.resume_depth == depth and r.calibrations == 1
    r.observe_standard(9000)
    for _ in range(7):
        r.observe_standard(10)
    assert not r.ready
    r.observe_standard(10)
    assert r.ready and r.ar_ms == 10
    assert r.standard_wall_ms == 9080  # Transition cost remains accounted.
    assert r.snapshot()["calibrating"] is True
    r.park(failed_probe=True)
    assert not r.calibrating and r.resume_depth == 1
    assert r.cooldown == 512 and r.failed_probes == 2


@pytest.mark.parametrize("depth", [0, 3, True, 1.5])
def test_calibration_rejects_invalid_or_above_ceiling_depth_without_mutation(depth):
    r = NativeMTPRecovery("a", 2, False)
    before = r.snapshot()
    with pytest.raises(ValueError):
        r.park_for_calibration(depth=depth)
    assert r.snapshot() == before


def test_calibration_requires_eight_valid_samples_not_only_elapsed_steps():
    r = NativeMTPRecovery("a", 3, False)
    r.park_for_calibration(depth=3)
    for value in [1, float("nan"), float("inf"), 0, -1, 10, 10, 10, 10]:
        r.observe_standard(value)
    assert r.remaining == 0 and not r.ready
    for _ in range(4):
        r.observe_standard(10)
    assert r.ready and r.ar_ms == 10


@pytest.mark.parametrize("condition", ["multirow", "rollback", "terminal", "no_head", "pending_mtp"])
def test_reentry_refuses_unsafe_or_unprofitable_boundaries(monkeypatch, condition):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    b, r = _ready_batch()
    if condition == "multirow":
        b.uids.append("b")
    elif condition == "rollback":
        b.prompt_cache = [SimpleNamespace(rollback_state=object())]
    elif condition == "terminal":
        b._num_tokens = [254]
    elif condition == "no_head":
        b.model.mtp = None
    else:
        b._omlx_mtp_state = object()
    monkeypatch.setattr(lane, "_post_init_mtp", lambda *a, **k: pytest.fail("unsafe re-entry"))
    assert not lane._text_mtp_maybe_resume(b, r)


def test_recovery_cannot_leak_to_replacement_uid():
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    b, r = _ready_batch()
    b.uids = ["replacement"]
    assert lane._text_recovery_for_batch(b) is None
    assert not hasattr(b, "_vmlx_mtp_recovery")


def test_fixed_ceiling_can_probe_higher_after_reentry_qualification(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    s = lane._MtpState(depth=1, depth_ceiling=2, adaptive_enabled=False)
    s.stats.cycles = 64
    seen = []
    def choose(*args, **kwargs):
        seen.append(kwargs["depth_ceiling"])
        return SimpleNamespace(target_depth=2, event="probe", reason="measured")
    monkeypatch.setattr(lane, "choose_depth_by_value", choose)
    lane._adaptive_finish_cycle("a", s, completed_depth=1, accepted=1, now=1)
    assert seen == [2] and s.depth == s.stats.depth == 2
    assert not s.adaptive_enabled
    assert s.ar_safety.cycle_base == 64


def test_recovery_probe_pays_seed_and_priming_cost(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    s = lane._MtpState(depth=1, depth_ceiling=3, ar_step_ms=10)
    s.recovery = NativeMTPRecovery("a", 3, False)
    s.recovery_probe_started = 1.0
    s.stats.cycles = 12
    s.stats.draft_tokens_accepted = 12
    monkeypatch.setattr(lane, "ar_safety_step", lambda *a, **k: None)
    monkeypatch.setattr(lane.time, "perf_counter", lambda: 2.0)
    assert lane._text_mtp_maybe_ar_safety_fallback("a", s)
    assert s.ar_fallback_reason == "reentry_total_cost"
    assert s.stats.fallback_mtp_ms_per_token == pytest.approx(1000 / 26)


def test_no_promotion_during_recovery_probe(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    s = lane._MtpState(depth=1, depth_ceiling=3, recovery_probe_started=1)
    monkeypatch.setattr(lane, "choose_depth_by_value", lambda *a, **k: pytest.fail("unqualified promotion"))
    lane._adaptive_finish_cycle("a", s, completed_depth=1, accepted=1, now=2)
    assert s.depth == 1


def test_parked_payload_reports_ar_not_previous_mtp_depth():
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane
    stats = lane._MtpStats(depth=0, starting_depth=3, depth_ceiling=3)
    payload = lane._native_mtp_payload("a", stats, "ar")
    assert payload["final_depth"] == 0
    assert payload["depth_ceiling"] == payload["starting_depth"] == 3

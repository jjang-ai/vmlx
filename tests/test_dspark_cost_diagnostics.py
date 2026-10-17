"""Terminal diagnostics must not modify the measured speculation policy."""
from copy import deepcopy

from vmlx_engine.dspark.generate import ARFallback


def test_cost_snapshot_is_read_only_and_detached():
    policy = ARFallback()
    policy.record_draft(2, 0.03, rows=4)
    policy.record_ar(0.01)
    before = deepcopy(policy.__dict__)
    snapshot = policy.cost_snapshot()
    assert snapshot == {
        "ar_step_seconds": 0.01,
        "draft_tokens_per_second": 2 / 0.03,
        "round_seconds_by_rows": {4: 0.03},
    }
    snapshot["round_seconds_by_rows"][4] = 100
    assert policy.__dict__ == before
    control = ARFallback()
    control.__dict__.update(deepcopy(before))
    assert policy.next_is_ar() == control.next_is_ar()
    assert policy.__dict__ == control.__dict__


def test_unmeasured_costs_remain_missing():
    assert ARFallback().cost_snapshot() == {
        "ar_step_seconds": None,
        "draft_tokens_per_second": None,
        "round_seconds_by_rows": {},
    }


def test_ar_fallback_reprobes_a_stale_ar_cost_and_skips_a_stalled_burst(monkeypatch):
    """Cross-lane port of the AdaptiveSpec fix (2026-10-17): once drafting wins, the AR step was never re-measured for
    the rest of the request; one stalled burst moved the estimate by 0.3 x its excess."""
    from vmlx_engine.dspark.generate import ARFallback

    monkeypatch.setenv("VMLX_DFLASH_AR_REPROBE_ROUNDS", "10")
    f = ARFallback()
    f.record_draft(3, 0.06)                                # one round, then the one-step AR seed
    assert f.next_is_ar()
    f.record_ar(0.030)
    f.record_ar_burst(8, 8 * 0.30)                          # stalled: skipped
    assert abs(f.ar_seconds - 0.030) < 1e-9 and f.ar_outliers_skipped == 1
    f.ar_seconds = 0.200                                    # stale estimate: drafting always wins ...
    for _ in range(10):
        assert not f.next_is_ar()
        f.record_draft(3, 0.06)
    assert f.next_is_ar() and f.burst_size() >= 1           # ... until the re-probe
    f.record_ar_burst(8, 8 * 0.025)
    assert abs(f.ar_seconds - (0.3 * 0.200 + 0.7 * 0.025)) < 1e-9

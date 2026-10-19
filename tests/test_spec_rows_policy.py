"""Per-round verify-row policy (vmlx_engine/spec_rows_policy.py): pure state, no model."""
from vmlx_engine.spec_rows_policy import (MIMO_V26_LADDER, Calibration, Drafting, PlainCost, VerifyCosts,
                                          choose_rows, prior_verify_costs)


def test_prior_costs_scale_with_the_plain_step_and_never_drop():
    c = prior_verify_costs(20.9, MIMO_V26_LADDER)
    assert abs(c[1] - 20.9) < 1e-6 and abs(c[2] - 28.2) < 1e-6 and abs(c[8] - 58.2) < 1e-6
    assert all(c[r] >= c[r - 1] for r in range(2, 17))
    half = prior_verify_costs(10.45, MIMO_V26_LADDER)
    assert abs(half[8] - c[8] / 2) < 1e-6


def test_choose_rows_plain_when_nothing_pays_whole_chain_when_everything_does():
    c = prior_verify_costs(20.9, MIMO_V26_LADDER)
    assert choose_rows([0.3, 0.2, 0.1, 0.1], c, 8)[0] == 1
    assert choose_rows([0.96] * 7, c, 8)[0] == 8
    # a poor first draft is not verified for the confident ones behind it
    assert choose_rows([0.05, 0.99, 0.99], c, 8)[0] == 1
    assert choose_rows([float("nan"), 0.9], c, 8)[0] >= 1
    assert choose_rows([0.96] * 7, c, 4)[0] <= 4


def test_second_row_pays_only_past_its_break_even():
    c = prior_verify_costs(20.9, MIMO_V26_LADDER)
    # 2 rows cost 28.2 vs 20.9: worth it once (1 + p) / 28.2 >= 0.98 / 20.9, i.e. p >~ 0.32
    assert choose_rows([0.2], c, 8)[0] == 1
    assert choose_rows([0.5], c, 8)[0] == 2


def test_calibration_prior_then_data_bin_by_bin_and_old_regime_fades():
    cal = Calibration()
    p0 = cal.probability(0.9)
    for _ in range(40):
        cal.observe(0.9, False)
    assert cal.probability(0.9) < 0.3 < p0
    assert abs(cal.probability(0.2) - Calibration().probability(0.2)) < 1e-9
    for _ in range(80):
        cal.observe(0.7, True)
    assert cal.probability(0.7) > 0.9
    for _ in range(80):
        cal.observe(0.7, False)
    assert cal.probability(0.7) < 0.2


def test_drafting_backoff_doubles_to_a_cap_and_a_winning_probe_resets():
    d = Drafting()
    d.record(7, 10)                       # loses: one plain step owed
    assert d.take_owed(100) == 1 and d.take_owed(100) == 0
    d.record(7, 10)                       # loses again: two owed
    assert d.take_owed(100) == 2
    for _ in range(12):
        d.record(5, 10)
        assert d.take_owed(100) <= 32
    d.record(5, 10)
    assert d.take_owed(100) == 32
    d.record(60, 10)                      # a probe that pays clears the debt
    assert d.take_owed(100) == 0
    assert d.take_owed(3) == 0


def test_drafting_keeps_going_on_a_thin_win_with_scattered_misses():
    d = Drafting()
    for g, w in [(70, 40), (18, 40), (70, 40), (70, 40), (18, 40), (120, 45), (18, 40), (120, 45), (70, 40), (18, 40)]:
        assert d.take_owed(100) == 0
        d.record(g, w)


def test_plain_cost_bounded_and_verify_costs_monotone():
    pc = PlainCost(ms=20.9)
    pc.sample(400)
    assert 20.9 < pc.ms < 21.5
    for _ in range(80):
        pc.sample(18.0)
    assert 18.0 <= pc.ms < 18.5
    v = VerifyCosts(ladder=dict(MIMO_V26_LADDER))
    v.reset_prior(20.9)
    v.sample(3, 10.0)                     # measured below the 2-row prior: the table stays monotone
    assert v.costs[3] >= v.costs[2]

"""A trained drafter block seeds exploration; measured wins govern growth."""
import pytest

pytest.importorskip("mlx.core")
from vmlx_engine.glm_dflash2 import AdaptiveSpec


def policy(**kw):
    p = AdaptiveSpec(4, grow_horizon=True, **kw)
    p.t_ar = 0.03
    return p


def test_profitable_boundary_explores_and_promotes_beyond_trained_block():
    p = policy()
    p.update_spec(4, 3, 0.06)
    assert p.choose() == ("spec", 5)
    p.update_spec(5, 4, 0.065)
    assert p.bmax == 5
    assert p.horizon_promotions == 1
    p.update_spec(5, 4, 0.065)
    assert p.choose() == ("spec", 6)


def test_slower_deeper_probe_retreats_and_does_not_immediately_repeat():
    p = policy()
    p.update_spec(4, 3, 0.06)
    assert p.choose() == ("spec", 5)
    p.update_spec(5, 4, 0.12)
    assert p.bmax == 4
    assert p.horizon_promotions == 0
    p.update_spec(4, 3, 0.06)
    assert p.choose()[1] <= 4


def test_boundary_that_loses_to_ar_does_not_expand():
    p = policy()
    p.update_spec(4, 3, 0.2)
    assert p.bmax == 4
    assert p.horizon_probes == 0


def test_compile_sample_does_not_promote_or_reject_probe():
    p = policy(costs={"ar": 0.03, "warm_rounds": {4}})
    p.update_spec(4, 3, 0.06)
    assert p.choose() == ("spec", 5)
    p.update_spec(5, 4, 2.0)
    assert p.horizon_promotions == 0
    assert p.choose() == ("spec", 5)
    p.update_spec(5, 4, 0.065)
    assert p.horizon_promotions == 1


def test_after_ar_context_ingest_is_not_a_horizon_measurement():
    p = policy()
    p.update_spec(4, 3, 0.06, after_ar=True)
    assert p.bmax == 4
    assert p.horizon_probes == 0


def test_existing_fixed_range_policy_stays_bounded():
    p = AdaptiveSpec(4)
    p.t_ar = 0.03
    for _ in range(20):
        p.update_spec(4, 3, 0.06)
        assert p.choose()[1] <= 4


def test_output_budget_cut_does_not_promote_unmeasured_width():
    p = policy()
    p.update_spec(4, 3, 0.06)
    assert p.choose() == ("spec", 5)
    p.update_spec(3, 2, 0.01)
    assert p.bmax == 4
    assert p.horizon_promotions == 0


def test_deeper_probe_must_beat_previous_width_even_if_it_beats_ar():
    p = policy()
    p.update_spec(4, 3, 0.06)
    assert p.choose() == ("spec", 5)
    p.update_spec(5, 4, 0.08)  # 62.5 > AR33.3, but < boundary66.7
    assert p.bmax == 4
    assert p.horizon_promotions == 0

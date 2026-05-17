from __future__ import annotations

import numpy as np


def test_acceptance_probability_uses_target_over_draft_ratio():
    from vmlx_engine.native_mtp_research import acceptance_probability

    target = np.array([0.50, 0.30, 0.20])
    draft = np.array([0.25, 0.50, 0.25])

    assert acceptance_probability(target, draft, 0) == 1.0
    assert acceptance_probability(target, draft, 1) == 0.60
    assert acceptance_probability(target, np.array([0.25, 0.0, 0.75]), 1) == 0.0


def test_residual_correction_makes_speculative_distribution_equal_target():
    from vmlx_engine.native_mtp_research import (
        residual_distribution,
        speculative_output_distribution,
    )

    target = np.array([0.50, 0.20, 0.30])
    draft = np.array([0.20, 0.40, 0.40])

    residual = residual_distribution(target, draft)
    observed = speculative_output_distribution(target, draft)

    np.testing.assert_allclose(residual, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(observed, target, atol=1e-12)


def test_residual_correction_falls_back_to_target_when_distributions_match():
    from vmlx_engine.native_mtp_research import residual_distribution

    target = np.array([0.10, 0.70, 0.20])

    np.testing.assert_allclose(residual_distribution(target, target), target)


def test_mlx_lm_filter_order_is_top_p_then_min_p_then_top_k():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        filter_distribution,
    )

    base = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
    settings = ResearchSamplerSettings(
        temperature=1.0,
        top_p=0.70,
        min_p=0.30,
        top_k=1,
    )

    filtered = filter_distribution(base, settings)

    np.testing.assert_allclose(filtered, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_min_p_keeps_requested_minimum_tokens_before_top_k():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        filter_distribution,
    )

    base = np.array([0.80, 0.08, 0.06, 0.04, 0.02])
    settings = ResearchSamplerSettings(
        temperature=1.0,
        min_p=0.50,
        min_tokens_to_keep=3,
        top_k=2,
    )

    filtered = filter_distribution(base, settings)

    assert np.count_nonzero(filtered) == 2
    np.testing.assert_allclose(filtered[0] + filtered[1], 1.0)
    np.testing.assert_allclose(filtered[2:], np.zeros(3))


def test_temperature_is_applied_after_filters():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        filter_distribution,
    )

    base = np.array([0.70, 0.20, 0.10])
    settings = ResearchSamplerSettings(temperature=0.5, top_k=2)

    filtered = filter_distribution(base, settings)
    expected = np.array([0.70**2, 0.20**2, 0.0])
    expected = expected / expected.sum()

    np.testing.assert_allclose(filtered, expected)


def test_filter_distribution_support_matches_installed_mlx_lm_order():
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        filter_distribution,
    )

    base = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
    settings = ResearchSamplerSettings(
        temperature=1.0,
        top_p=0.70,
        min_p=0.30,
        top_k=2,
    )
    logprobs = mx.array(np.log(base), dtype=mx.float32)
    filtered = apply_top_p(logprobs, settings.top_p)
    filtered = apply_min_p(filtered, settings.min_p, settings.min_tokens_to_keep)
    filtered = apply_top_k(filtered, settings.top_k)
    mx.eval(filtered)
    mlx_support = np.isfinite(np.array(filtered.tolist()))

    research_support = filter_distribution(base, settings) > 0

    np.testing.assert_array_equal(research_support, mlx_support)


def test_policy_metrics_report_acceptance_and_distribution_delta():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        speculative_policy_metrics,
    )

    target = np.log(np.array([0.50, 0.20, 0.30]))
    draft = np.log(np.array([0.20, 0.40, 0.40]))

    metrics = speculative_policy_metrics(
        target,
        draft,
        ResearchSamplerSettings(temperature=1.0),
    )

    assert metrics["expected_acceptance_rate"] == 0.70
    assert metrics["reject_rate"] == 0.30
    assert metrics["delta"]["max_abs"] < 1e-12
    np.testing.assert_allclose(metrics["target_distribution"], [0.50, 0.20, 0.30])
    np.testing.assert_allclose(metrics["observed_distribution"], [0.50, 0.20, 0.30])


def test_policy_metrics_can_use_separate_draft_temperature():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        speculative_policy_metrics,
    )

    target = np.log(np.array([0.70, 0.20, 0.10]))
    draft = np.log(np.array([0.70, 0.20, 0.10]))

    metrics = speculative_policy_metrics(
        target,
        draft,
        target_settings=ResearchSamplerSettings(temperature=0.6),
        draft_settings=ResearchSamplerSettings(temperature=0.8),
    )

    assert metrics["target_settings"]["temperature"] == 0.6
    assert metrics["draft_settings"]["temperature"] == 0.8
    assert metrics["expected_acceptance_rate"] < 1.0


def test_sweep_policy_grid_ranks_by_acceptance_then_distribution_delta():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        sweep_policy_grid,
    )

    target = np.log(np.array([0.55, 0.25, 0.15, 0.05]))
    draft = np.log(np.array([0.50, 0.20, 0.20, 0.10]))
    settings = [
        ResearchSamplerSettings(temperature=1.0),
        ResearchSamplerSettings(temperature=1.0, top_k=1),
        ResearchSamplerSettings(temperature=1.0, min_p=0.40),
    ]

    ranked = sweep_policy_grid(target, draft, settings)

    assert len(ranked) == 3
    assert ranked[0]["expected_acceptance_rate"] >= ranked[1]["expected_acceptance_rate"]
    assert ranked[0]["delta"]["max_abs"] < 1e-12
    assert ranked[0]["rank"] == 1


def test_cached_pair_sweep_reuses_filtered_distributions(monkeypatch):
    import vmlx_engine.native_mtp_research as research

    settings_a = research.ResearchSamplerSettings(temperature=1.0, top_p=0.95)
    settings_b = research.ResearchSamplerSettings(temperature=0.8, top_k=2)
    target_rows = [
        np.log(np.array([0.55, 0.25, 0.15, 0.05])),
        np.log(np.array([0.45, 0.30, 0.15, 0.10])),
    ]
    draft_rows = [
        np.log(np.array([0.50, 0.20, 0.20, 0.10])),
        np.log(np.array([0.30, 0.40, 0.20, 0.10])),
    ]
    settings_pairs = [
        (settings_a, settings_a),
        (settings_a, settings_b),
        (settings_a, settings_a),
    ]
    calls = []
    original = research.distribution_from_logits

    def counted_distribution_from_logits(logits, settings):
        calls.append((tuple(np.asarray(logits).round(8)), settings))
        return original(logits, settings)

    monkeypatch.setattr(
        research,
        "distribution_from_logits",
        counted_distribution_from_logits,
    )

    ranked = research.sweep_policy_pairs_cached(
        target_rows,
        draft_rows,
        settings_pairs,
        top_n=3,
    )

    assert len(ranked) == 3
    assert len(calls) == 6
    assert ranked[0]["rank"] == 1
    assert ranked[0]["mean_acceptance"] >= ranked[-1]["mean_acceptance"]
    assert all(row["delta"]["mode"] == "residual_exact_by_formula" for row in ranked)
    assert ranked[0]["mean_target_support"] >= 1.0
    assert ranked[0]["mean_draft_support"] >= 1.0
    assert ranked[0]["min_target_support"] >= 1
    assert ranked[0]["min_draft_support"] >= 1


def test_cached_pair_sweep_sorts_tied_rows_without_dict_comparison():
    from vmlx_engine.native_mtp_research import (
        ResearchSamplerSettings,
        sweep_policy_pairs_cached,
    )

    logits = [np.log(np.array([0.70, 0.20, 0.10]))]
    settings_a = ResearchSamplerSettings(temperature=1.0, top_k=0)
    settings_b = ResearchSamplerSettings(temperature=1.0, top_k=3)

    ranked = sweep_policy_pairs_cached(
        logits,
        logits,
        [(settings_a, settings_a), (settings_b, settings_b)],
    )

    assert len(ranked) == 2
    assert ranked[0]["mean_acceptance"] == 1.0
    assert ranked[1]["mean_acceptance"] == 1.0

"""Draft acceptance for native MTP verify cycles.

Exact-match acceptance is only correct under greedy decode.  These tests pin
both arms: the default exact-match behaviour must stay byte-identical to the
loop it replaced, and the gated stochastic arm must implement the standard
min(1, p_target/p_draft) rule so drafts survive at temperature > 0.
"""

import math

import mlx.core as mx
import pytest

from vmlx_engine import mllm_batch_generator as gen


def _lp(pairs, vocab=8):
    """Build a log-prob row from {token_id: probability}."""
    probs = [1e-9] * vocab
    for token_id, prob in pairs.items():
        probs[token_id] = prob
    total = sum(probs)
    return mx.array([math.log(p / total) for p in probs])


def _accepted(draft_ids, target_ids, draft_lps=None, target_lps=None):
    return gen._native_mtp_accepted_count(
        draft_ids,
        target_ids,
        draft_lps if draft_lps is not None else [None] * len(draft_ids),
        target_lps if target_lps is not None else [None] * len(draft_ids),
    )


class TestExactMatchArm:
    """Default arm: identical to the original inline loop."""

    def test_all_matching_drafts_accepted(self):
        assert _accepted([3, 4, 5], [3, 4, 5, 9]) == 3

    def test_stops_at_first_mismatch(self):
        assert _accepted([3, 4, 5], [3, 7, 5, 9]) == 1

    def test_leading_mismatch_accepts_nothing(self):
        assert _accepted([3, 4], [7, 4, 9]) == 0

    def test_no_drafts_accepts_nothing(self):
        assert _accepted([], [9]) == 0

    def test_mismatch_is_not_rescued_when_gate_is_off(self, monkeypatch):
        """Even with distributions available, the default arm must break."""
        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", False)
        # p_target(draft) far exceeds p_draft(draft): the stochastic arm would
        # accept this outright, so it proves the gate is what decides.
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.05, 2: 0.95})],
                target_lps=[_lp({1: 0.95, 2: 0.05})],
            )
            == 0
        )


class TestStochasticArm:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", True)

    def test_matching_draft_still_accepted(self):
        assert _accepted([3], [3, 9], draft_lps=[None], target_lps=[None]) == 1

    def test_matching_sampled_token_still_requires_probability_check(self):
        from vmlx_engine.native_mtp_acceptance import accepted_count

        class Sampler:
            draws = 0

            def _vmlx_random_uniform(self):
                self.draws += 1
                return 0.75

        sampler = Sampler()
        accepted = accepted_count(
            [0], [0],
            [mx.log(mx.array([0.8, 0.2]))],
            [mx.log(mx.array([0.4, 0.6]))],
            stochastic=True, sampler=sampler, filtered=True,
        )
        assert accepted == 0
        assert sampler.draws == 1

    def test_independent_target_draw_does_not_bias_sampled_marginal(self):
        from vmlx_engine.native_mtp_acceptance import accepted_count, residual_sample

        class Sampler:
            def __init__(self, draw):
                self.draw = draw

            def _vmlx_random_uniform(self):
                return self.draw

        target, proposal = [0.4, 0.6], [0.8, 0.2]
        p, q = mx.log(mx.array(target)), mx.log(mx.array(proposal))
        # The positive residual has support only at token 1. Exercise the
        # production correction once; enumerate acceptance without RNG noise.
        correction, _ = residual_sample(p, q)
        assert correction == 1
        marginal = [0.0, 0.0]
        for draft in range(2):
            for independently_sampled_target in range(2):
                for step in range(20):
                    accepted = accepted_count(
                        [draft], [independently_sampled_target], [q], [p],
                        stochastic=True, filtered=True,
                        sampler=Sampler((step + 0.5) / 20),
                    )
                    emitted = draft if accepted else correction
                    marginal[emitted] += proposal[draft] * target[independently_sampled_target] / 20
        assert marginal == pytest.approx(target, abs=1e-12)

    def test_mismatch_accepted_when_target_prefers_the_draft(self):
        """p_target/p_draft >= 1 accepts deterministically, no draw needed."""
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.05, 2: 0.95})],
                target_lps=[_lp({1: 0.95, 2: 0.05})],
            )
            == 1
        )

    def test_mismatch_rejected_when_target_finds_draft_near_impossible(self):
        """p_target/p_draft ~ 0 must reject for any draw in (0, 1]."""
        assert (
            _accepted(
                [1],
                [2],
                draft_lps=[_lp({1: 0.999, 2: 0.001})],
                target_lps=[_lp({1: 1e-9, 2: 0.999})],
            )
            == 0
        )

    def test_falls_back_to_exact_match_without_distributions(self):
        """Greedy/compact samplers expose no rows; exact match is correct."""
        assert _accepted([1, 2], [9, 2], draft_lps=[None, None], target_lps=[None, None]) == 0

    def test_acceptance_is_never_negative_or_past_the_draft_count(self):
        accepted = _accepted(
            [1, 2],
            [1, 2, 3],
            draft_lps=[_lp({1: 0.9}), _lp({2: 0.9})],
            target_lps=[_lp({1: 0.9}), _lp({2: 0.9})],
        )
        assert 0 <= accepted <= 2

    def test_ratio_is_read_at_the_draft_token_not_the_target_token(self):
        """Regression: using p at the target's own token inverts the test.

        Target overwhelmingly prefers token 2 (what it sampled) and considers
        the draft token 1 nearly impossible, so the draft must be rejected.
        Reading the ratio at token 2 instead would accept it every time.
        """
        rejected = 0
        for _ in range(12):
            rejected += (
                _accepted(
                    [1],
                    [2],
                    draft_lps=[_lp({1: 0.98, 2: 0.02})],
                    target_lps=[_lp({1: 1e-9, 2: 0.999})],
                )
                == 0
            )
        assert rejected == 12

    def test_malformed_distribution_does_not_raise(self):
        """A short/ragged row must end the cycle, never propagate an error."""
        assert _accepted([5], [6], draft_lps=[mx.array([0.0])], target_lps=[mx.array([0.0])]) == 0

    def test_impossible_draft_is_rejected_not_accepted(self):
        """p_draft == 0 (-inf) means the draft could never have been drawn.

        The ratio is meaningless there, so it must reject rather than produce a
        non-finite comparison that slips through as an accept.
        """
        draft = mx.array([0.0, float("-inf")])
        target = mx.array([0.0, 0.0])
        assert _accepted([1], [0], draft_lps=[draft], target_lps=[target]) == 0

    def test_request_local_uniform_owns_the_acceptance_draw(self):
        class _Sampler:
            temp = 1.0
            top_p = 1.0
            min_p = 0.0
            top_k = 0

            def __init__(self):
                self.draws = 0

            def _vmlx_random_uniform(self):
                self.draws += 1
                return 0.99

        sampler = _Sampler()
        from vmlx_engine.native_mtp_acceptance import accepted_count

        accepted = accepted_count(
            [1],
            [2],
            [_lp({1: 0.8, 2: 0.2})],
            [_lp({1: 0.2, 2: 0.8})],
            stochastic=True,
            sampler=sampler,
        )
        assert accepted == 0
        assert sampler.draws == 1

    def test_residual_correction_excludes_proposal_only_mass(self):
        from vmlx_engine.native_mtp_acceptance import residual_sample

        # q puts all useful mass on token 0 while p puts its surplus on token
        # 1.  The positive residual therefore has exactly one possible draw.
        target = _lp({0: 0.1, 1: 0.9})
        draft = _lp({0: 0.9, 1: 0.1})
        token, returned = residual_sample(target, draft)
        assert token == 1
        assert bool(mx.array_equal(returned, target))

    def test_mllm_rejection_uses_residual_not_independent_target_draw(self):
        from vmlx_engine.mllm_batch_generator import (
            _native_mtp_rejection_correction,
        )

        class _Sampler:
            temp = 1.0
            top_p = 1.0
            min_p = 0.0
            top_k = 0

        original = mx.array([0], dtype=mx.uint32)
        correction, correction_id = _native_mtp_rejection_correction(
            original,
            0,
            _lp({0: 0.1, 1: 0.9}),
            _lp({0: 0.9, 1: 0.1}),
            _Sampler(),
        )
        assert correction_id == 1
        assert int(correction.item()) == 1


class TestBothSchedulersShareOneRule:
    """The MLLM path and the text path must not drift apart.

    They already did once: /health advertised
    'stochastic=rejection-sampling-acceptance' while the MLLM generator was
    exact-match only, which is why --is-mllm bundles had to pin temperature 0.
    """

    def test_text_path_delegates_to_the_shared_helper(self):
        from vmlx_engine import native_mtp_acceptance
        from vmlx_engine.patches.mlx_lm_mtp import batch_generator as text_gen

        sentinel = object()
        captured = {}

        def _fake(sampler, lp):
            captured["called"] = True
            return sentinel

        original = native_mtp_acceptance.accept_lp_for
        native_mtp_acceptance.accept_lp_for = _fake
        try:
            assert text_gen._accept_lp_for(None, mx.array([0.0])) is sentinel
        finally:
            native_mtp_acceptance.accept_lp_for = original
        assert captured.get("called"), "text path must route through the shared rule"

    def test_mllm_path_delegates_to_the_shared_helper(self):
        assert gen._shared_accepted_count is not None
        from vmlx_engine.native_mtp_acceptance import accepted_count

        assert gen._shared_accepted_count is accepted_count

    def test_greedy_sampler_leaves_the_distribution_untouched(self):
        """temp 0 means raw lp already is the acceptance distribution."""
        from vmlx_engine.native_mtp_acceptance import accept_lp_for

        row = mx.array([-1.0, -2.0, -3.0])
        assert accept_lp_for(None, row) is row

    def test_health_gate_reports_imported_mllm_runtime_flag(self, monkeypatch):
        from vmlx_engine import mllm_batch_generator as gen
        from vmlx_engine import server

        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", True)
        enabled, gate = server._mllm_native_mtp_request_gate_status()
        assert enabled is True
        assert gate.endswith("stochastic=rejection-sampling-acceptance")

        monkeypatch.setattr(gen, "_NATIVE_MTP_STOCHASTIC_ACCEPT", False)
        enabled, gate = server._mllm_native_mtp_request_gate_status()
        assert enabled is False
        assert gate.endswith("stochastic=skipped-by-policy")

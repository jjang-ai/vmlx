"""Legacy acceptance-only adaptive depth fallback contracts.

The controller could only ever LOWER draft depth, so a bundle whose tuning
sidecar says depth 1 stayed at depth 1 no matter how well its head performed.
That caps throughput at (1 + acceptance) tokens per cycle. MTPLX runs the same
model family at depth 3 with 0.95/0.88/0.80 acceptance = 3.46 tokens per cycle,
which is the whole difference between a 1.5x and a 2.5x speedup.

The rolling wall-value controller has its own focused suite.  These tests run
with that controller disabled so the older cumulative acceptance gates remain
available as a controlled A/B fallback.
"""

import math

import mlx.core as mx
import pytest

from vmlx_engine import mllm_batch_generator as gen


class _Stats:
    def __init__(self, drafted_by_depth, accepted_by_depth, cycles=None):
        self.drafted_by_depth = list(drafted_by_depth)
        self.accepted_by_depth = list(accepted_by_depth)
        self.cycles = cycles if cycles is not None else max(drafted_by_depth)
        self.mtp_forwards = self.cycles
        self.verify_qmm_calls = 0


class _State:
    def __init__(self, depth, drafted_by_depth, accepted_by_depth, ceiling=3):
        self.depth = depth
        self.depth_ceiling = ceiling
        self.stats = _Stats(drafted_by_depth, accepted_by_depth)
        self.ar_fallback_pending = False
        self.ar_fallback_reason = None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "VMLINUX_NATIVE_MTP_ADAPTIVE_RAISE",
        "VMLX_NATIVE_MTP_ADAPTIVE_RAISE",
        "VMLINUX_NATIVE_MTP_RAISE_MIN_ACCEPT",
        "VMLX_NATIVE_MTP_RAISE_MIN_ACCEPT",
        "VMLINUX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        "VMLX_NATIVE_MTP_RAISE_MIN_SAMPLE",
        "VMLINUX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLX_NATIVE_MTP_ADAPTIVE_WARMUP_CYCLES",
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
        "VMLINUX_NATIVE_MTP_ADAPTIVE_VALUE",
        "VMLX_NATIVE_MTP_ADAPTIVE_VALUE",
        "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
        "VMLX_NATIVE_MTP_ADAPTIVE_DEPTH",
        "VMLINUX_NATIVE_MTP_DEPTH",
        "VMLX_NATIVE_MTP_DEPTH",
        "VMLINUX_NATIVE_MTP_DRAFT_MARGIN",
        "VMLX_NATIVE_MTP_DRAFT_MARGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_VALUE", "0")


class TestFixedDepthConfidenceGate:
    def test_fixed_d3_uses_measured_margin_default(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        assert gen._native_mtp_draft_margin_threshold("qwen4_exp") == pytest.approx(
            1.0
        )

    def test_adaptive_and_fixed_d2_keep_margin_disabled(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        assert gen._native_mtp_draft_margin_threshold("qwen4_exp") == pytest.approx(
            0.0
        )

        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "2")
        assert gen._native_mtp_draft_margin_threshold("qwen4_exp") == pytest.approx(
            0.0
        )

    def test_unmeasured_families_keep_margin_disabled(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        assert gen._native_mtp_draft_margin_threshold("glm5_next") == pytest.approx(
            0.0
        )
        assert gen._native_mtp_draft_margin_threshold("qwen3_5") == pytest.approx(
            0.0
        )

    def test_explicit_margin_still_overrides_fixed_d3_default(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DRAFT_MARGIN", "0.25")
        assert gen._native_mtp_draft_margin_threshold("glm5_next") == pytest.approx(
            0.25
        )


class TestDraftMarginReduction:
    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    @pytest.mark.parametrize("size", [2, 129, 248320])
    @pytest.mark.parametrize("two_rows", [False, True])
    @pytest.mark.parametrize("case", [
        "distinct", "duplicate_max", "equal", "signed_zero", "positive_inf",
        "two_positive_inf", "negative_inf", "nan_first", "nan_last",
    ])
    def test_matches_topk_including_ties_and_nonfinite(self, dtype, size, two_rows, case):
        row = (mx.arange(size) % 31).astype(dtype) / 32
        if case == "distinct":
            row = mx.concatenate([row[:-1], mx.array([2.0], dtype=dtype)])
        if case == "duplicate_max":
            row = mx.concatenate([mx.array([5.0], dtype=dtype), row[1:-1],
                                  mx.array([5.0], dtype=dtype)])
        elif case == "equal":
            row = mx.full((size,), 7, dtype=dtype)
        elif case == "signed_zero":
            row = mx.concatenate([mx.array([-0.0], dtype=dtype),
                                  mx.zeros((size - 1,), dtype=dtype)])
        elif case == "positive_inf":
            row = mx.concatenate([row[:-1], mx.array([float("inf")], dtype=dtype)])
        elif case == "two_positive_inf":
            row = mx.concatenate([mx.array([float("inf")], dtype=dtype), row[1:-1],
                                  mx.array([float("inf")], dtype=dtype)])
        elif case == "negative_inf":
            row = mx.full((size,), -float("inf"), dtype=dtype)
        elif case == "nan_first":
            row = mx.concatenate([mx.array([float("nan")], dtype=dtype), row[1:]])
        elif case == "nan_last":
            row = mx.concatenate([row[:-1], mx.array([float("nan")], dtype=dtype)])
        logits = mx.stack([mx.zeros_like(row), row]) if two_rows else row
        original = mx.topk(row, 2)
        expected = mx.abs(original[0] - original[1])
        actual = gen._native_mtp_top2_margin(logits)
        mx.eval(expected, actual)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        a, b = actual.item(), expected.item()
        assert a == b or (math.isnan(a) and math.isnan(b))

    def test_float_logits_do_not_use_topk(self, monkeypatch):
        def fail_topk(*args, **kwargs):
            raise AssertionError("confidence margin must not sort float logits")

        monkeypatch.setattr(mx, "topk", fail_topk)
        assert gen._native_mtp_top2_margin(mx.array([2.0, 5.0, 5.0])).item() == 0
        assert gen._native_mtp_top2_margin(mx.array([2.0, 5.0, 3.0])).item() == 2

    def test_too_small_input_keeps_prior_rejection(self):
        with pytest.raises(ValueError):
            gen._native_mtp_top2_margin(mx.array([1.0]))


class TestRaises:
    def test_excellent_d1_climbs_to_depth_two(self):
        """0.95 at depth 1 — the MTPLX-grade head we are building toward."""
        state = _State(1, [200, 0, 0], [190, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2

    def test_climbs_one_step_at_a_time(self):
        """Never jump 1 -> 3 on a single observation."""
        state = _State(1, [200, 0, 0], [199, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2

    def test_depth_two_can_climb_to_three(self):
        state = _State(2, [200, 200, 0], [195, 190, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_never_exceeds_three(self):
        state = _State(3, [200, 200, 200], [199, 199, 199])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3


class TestDoesNotRaise:
    def test_current_bundle_acceptance_never_triggers_a_raise(self):
        """53-65% is what the 6-bit-head bundle actually measures live.

        This is the safety property: shipping the raise path must not change
        behaviour for the bundles we serve today.
        """
        for accepted in (34, 40, 42):  # 53.1%, 62.5%, 65.6% of 64
            state = _State(1, [64, 0, 0], [accepted, 0, 0])
            gen._native_mtp_maybe_adapt_depth("req", state)
            assert state.depth == 1

    def test_small_sample_does_not_trigger(self):
        """A lucky opening streak must not promote the whole request."""
        state = _State(1, [12, 0, 0], [12, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_RAISE", "0")
        state = _State(1, [200, 0, 0], [195, 0, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_threshold_is_tunable(self, monkeypatch):
        monkeypatch.setenv("VMLX_NATIVE_MTP_RAISE_MIN_ACCEPT", "0.60")
        state = _State(1, [200, 0, 0], [130, 0, 0])  # 65%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2


class TestHysteresis:
    def test_depth_three_gate_compares_joint_rate_to_joint_floor(self):
        """A healthy conditional d3 rate must not be compared to 0.85 raw."""
        # Joint d2=85%, joint d3=75%, so conditional d3=88.2% and passes.
        state = _State(3, [200, 200, 200], [190, 170, 150])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_depth_three_gate_still_demotes_bad_conditional_rate(self):
        # Joint d2=85%, joint d3=65%, so conditional d3=76.5% and fails.
        state = _State(3, [200, 200, 200], [190, 170, 130])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2
        assert state.depth_ceiling == 3

    def test_accelerated_depth_three_waits_for_a_real_sample(self):
        # The exact live cold window was 22/48 joint d3. The accelerated lane
        # waits for 128 drafts because the completed run recovered to 582/767.
        state = _State(3, [48, 48, 48], [45, 30, 22])
        state.stats.verify_qmm_calls = 1
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_accelerated_depth_three_uses_profitable_floor(self):
        # Representative 128-cycle window: joint d2=64.8%, joint d3=47.7%,
        # therefore conditional d3=73.5%. This remains profitable with the
        # four-row verifier while independently satisfying the D2 gate.
        state = _State(3, [128, 128, 128], [118, 83, 61])
        state.stats.verify_qmm_calls = 1
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_installed_verifier_without_request_calls_keeps_stock_gate(
        self, monkeypatch
    ):
        from vmlx_engine.metal import native_mtp_verify_qmm

        monkeypatch.setattr(
            native_mtp_verify_qmm,
            "native_mtp_verify_qmm_active",
            lambda: True,
        )
        # A q6 or otherwise ineligible artifact sees the installed dispatcher
        # but records zero request-local accelerated calls. Its 47.7%
        # conditional d3 rate must therefore use the conservative stock gate.
        state = _State(3, [128, 128, 128], [118, 83, 61])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2

    def test_ceiling_blocks_returning_to_a_failed_depth(self):
        """A depth demoted for poor acceptance is never retried."""
        state = _State(1, [200, 0, 0], [195, 0, 0], ceiling=1)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1

    def test_demotion_preserves_the_capability_ceiling(self):
        """A bad phase must not permanently destroy a supported depth."""
        state = _State(2, [200, 200, 0], [195, 40, 0])  # d2 = 20%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 3

    def test_acceptance_fallback_can_recover_after_a_demotion(self):
        """A later predictable phase can climb back under the same ceiling."""
        state = _State(2, [200, 200, 0], [195, 40, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        # Now d1 looks superb; the preserved capability ceiling permits D2.
        state.stats = _Stats([400, 200, 0], [395, 40, 0])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 2


class TestRestoredPrefixGates:
    """Restored-prefix requests start with a COLD head cache (backbone
    hiddens are not stored), so early gate windows measure a context-starved
    head, not the bundle. Live A/B: run 3 of a warm conversation demoted
    D3->D1 at cycle 129 on d2=0.574 that recovers to ~0.85 warm, and the
    lowered ceiling made 17.4 t/s permanent (cold run 1 = 40.2 t/s)."""

    def _state(self, depth, drafted, accepted, restored=True, ceiling=3):
        state = _State(depth, drafted, accepted, ceiling=ceiling)
        state.restored_prefix = restored
        return state

    def test_cold_window_sample_does_not_demote_restored_request(self):
        # 129 drafted at joint d2=0.574 — the exact live demotion window.
        # Fresh requests demote here; restored ones must wait for 4x sample.
        state = self._state(3, [129, 129, 129], [110, 74, 50])
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 3

    def test_fresh_request_still_demotes_on_the_same_window(self):
        state = self._state(3, [129, 129, 129], [110, 74, 50], restored=False)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth < 3

    def test_restored_demote_at_full_sample_keeps_ceiling(self):
        # Even when a restored request eventually demotes (sustained bad d2
        # over the stretched sample), the ceiling stays put so the raise
        # path can climb back once the head cache is warm.
        state = self._state(2, [800, 800, 0], [780, 160, 0])  # d2 20%
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 3

    def test_fresh_demote_also_preserves_capability_ceiling(self):
        state = self._state(2, [800, 800, 0], [780, 160, 0], restored=False)
        gen._native_mtp_maybe_adapt_depth("req", state)
        assert state.depth == 1
        assert state.depth_ceiling == 3


class TestRollingValueIntegration:
    def test_tool_request_ceiling_blocks_wall_value_promotion(self, monkeypatch):
        from vmlx_engine.native_mtp_adaptive import add_depth_cycle_sample

        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_VALUE", "1")
        state = gen.MLLMNativeMTPState(depth=1, depth_ceiling=1)
        state.stats.cycles = 12
        state.stats.drafted_by_depth = [12, 0, 0]
        state.stats.accepted_by_depth = [12, 0, 0]
        for cycle in range(5, 13):
            add_depth_cycle_sample(
                state.adaptive_value,
                depth=1,
                accepted_drafts=1,
                elapsed_ms=6.0,
                cycle=cycle,
                window=16,
            )

        gen._native_mtp_maybe_adapt_depth("tool-value-row", state)

        assert state.depth == 1
        assert state.stats.drafted_by_depth[1:] == [0, 0]
        assert state.adaptive_value.active_probe_origin == 0
        assert state.adaptive_value.active_probe_target == 0

    def test_generator_policy_uses_wall_value_and_publishes_telemetry(
        self, monkeypatch
    ):
        from vmlx_engine.native_mtp_adaptive import add_depth_cycle_sample

        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_VALUE", "1")
        monkeypatch.setenv("VMLX_NATIVE_MTP_VALUE_INITIAL_PROBE_CYCLES", "8")
        state = gen.MLLMNativeMTPState(depth=2)
        state.stats.cycles = 12
        state.stats.drafted_by_depth = [12, 12, 0]
        state.stats.accepted_by_depth = [12, 12, 0]
        for cycle in range(5, 13):
            add_depth_cycle_sample(
                state.adaptive_value,
                depth=2,
                accepted_drafts=2,
                elapsed_ms=6.0,
                cycle=cycle,
                window=16,
            )

        gen._native_mtp_maybe_adapt_depth("value-row", state)

        assert state.depth == 3
        assert state.stats.adaptive_depth_value["basis"] == (
            "rolling_wall_confirmed_tokens_per_second"
        )
        assert state.stats.adaptive_depth_value["active_probe"] == {
            "origin": 2,
            "target": 3,
        }

    def test_generator_interval_helpers_discard_a_depth_transition(
        self, monkeypatch
    ):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_VALUE", "1")
        state = gen.MLLMNativeMTPState(depth=2)
        state.stats.cycles = 1
        gen._native_mtp_arm_value_cycle(state, now=10.0)
        gen._native_mtp_finish_value_cycle(
            state,
            depth=3,
            accepted=3,
            now=10.01,
        )

        assert state.adaptive_value.samples_by_depth == [[], [], []]

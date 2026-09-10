"""Session-scoped validated adaptive-MTP profile contract.

Audited contract (2026-08-30): unknown profiles stay AR unless the owning
runtime supplies a live-measured family cold-start depth.  Immediate seeding
otherwise requires a validated tuning record or an in-process entry that beat
its own AR baseline; cancelled/errored/sample-starved requests teach nothing.
"""

from vmlx_engine.native_mtp_profile import (
    MIN_WALL_SAMPLES,
    NativeMTPProfileStore,
    profile_key,
)


def _observe_profitable(store, key, depth=2, value=45.0, ar=30.0, n=None):
    label = f"d{depth}"
    store.observe(
        key,
        final_depth=depth,
        fallback_to_ar=False,
        fallback_reason=None,
        finish_reason="stop",
        values_tok_s={label: value},
        sample_counts={label: MIN_WALL_SAMPLES if n is None else n},
        ar_baseline_tps=ar,
    )


class TestProfileKey:
    def test_sampler_class_split(self):
        greedy = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=100)
        sampled = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=100)
        assert greedy[0] == "greedy"
        assert sampled[0] == "sampled"
        assert greedy != sampled

    def test_restored_and_tools_split(self):
        base = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=100)
        warm = profile_key(temperature=0.0, restored_prefix=True, prompt_tokens=100)
        tools = profile_key(temperature=0.0, restored_prefix=False,
                            prompt_tokens=100, has_tools=True)
        assert len({base, warm, tools}) == 3

    def test_context_buckets(self):
        assert profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)[2] == "short"
        assert profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=8000)[2] == "medium"
        assert profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=40000)[2] == "long"


class TestStartDepth:
    def test_unknown_profile_stays_ar(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 0, (
            "An unknown workload shape must stay AR — the only honest "
            "first-turn guarantee. MTP requires measured evidence."
        )
        assert source == "unseen_ar"

    def test_validated_tuning_record_seeds_immediately(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        depth, source = store.start_depth(
            key, configured_depth=2, tuning_validated=True
        )
        assert depth == 2
        assert source == "tuning_validated_d2"

    def test_measured_family_cold_start_is_explicit_and_clamped(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        assert store.start_depth(
            key,
            configured_depth=3,
            capability_ceiling=2,
            unseen_start_depth=3,
            unseen_start_source="qwen4_exp_measured_cold_start",
        ) == (2, "qwen4_exp_measured_cold_start_d2")

    def test_learned_ar_verdict_overrides_family_cold_start(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(
            key,
            final_depth=1,
            fallback_to_ar=True,
            fallback_reason="cost",
            finish_reason="fallback_to_ar",
        )
        assert store.start_depth(
            key,
            configured_depth=3,
            unseen_start_depth=3,
            unseen_start_source="qwen4_exp_measured_cold_start",
        ) == (0, "profile_ar")

    def test_profitable_learned_depth_is_reused(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        _observe_profitable(store, key, depth=2, value=45.0, ar=30.0)
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 2
        assert source == "profile_validated_d2"

    def test_learned_depth_can_improve_beyond_configured_start(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        _observe_profitable(store, key, depth=3, value=50.0, ar=30.0)
        depth, _ = store.start_depth(key, configured_depth=2)
        assert depth == 3

    def test_learned_depth_clamped_to_runtime_capability_ceiling(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        _observe_profitable(store, key, depth=3, value=50.0, ar=30.0)
        depth, _ = store.start_depth(
            key,
            configured_depth=2,
            capability_ceiling=2,
        )
        assert depth == 2

    def test_ar_verdict_holds_within_ttl_no_per_request_reprobe(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        results = [store.start_depth(key, configured_depth=3, now=100.0 + i)
                   for i in range(50)]
        assert all(d == 0 for d, _ in results), (
            "inside the TTL no user request may be sacrificed to a re-probe"
        )

    def test_ar_verdict_expires_and_reprobes_exactly_once(self):
        from vmlx_engine.native_mtp_profile import AR_VERDICT_TTL_S

        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        base = store._profiles[key].ar_verdict_at
        later = base + AR_VERDICT_TTL_S + 1
        first = store.start_depth(key, configured_depth=3, now=later)
        assert first == (1, "ar_reprobe_d1"), (
            "the AR verdict is adaptive — after the TTL one bounded D1 "
            "re-probe re-validates it"
        )
        # Immediately after, the TTL is re-armed: no probe storm.
        second = store.start_depth(key, configured_depth=3, now=later + 1)
        assert second == (0, "profile_ar")

    def test_keys_do_not_poison_each_other(self):
        store = NativeMTPProfileStore()
        sampled = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        greedy = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(sampled, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        depth, source = store.start_depth(greedy, configured_depth=3)
        assert (depth, source) == (0, "unseen_ar")


class TestObserve:
    def test_cancelled_and_error_teach_nothing(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        for reason in ("cancelled", "error"):
            store.observe(key, final_depth=3, fallback_to_ar=False,
                          fallback_reason=None, finish_reason=reason,
                          values_tok_s={"d3": 99.0},
                          sample_counts={"d3": 10}, ar_baseline_tps=10.0)
        assert store.start_depth(key, configured_depth=3) == (0, "unseen_ar")

    def test_sample_starved_request_teaches_nothing(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        _observe_profitable(store, key, depth=2, value=99.0, ar=10.0,
                            n=MIN_WALL_SAMPLES - 1)
        assert store.start_depth(key, configured_depth=3) == (0, "unseen_ar")

    def test_value_below_ar_margin_learns_ar(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        _observe_profitable(store, key, depth=2, value=30.5, ar=30.0)
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 0
        assert source == "profile_ar"

    def test_transient_final_probe_does_not_override_best_measured_depth(self):
        store = NativeMTPProfileStore()
        key = profile_key(
            temperature=0.0, restored_prefix=False, prompt_tokens=64
        )
        store.observe(
            key,
            # The request ended during a D3 -> D2 neighbor probe.
            final_depth=2,
            fallback_to_ar=False,
            fallback_reason=None,
            finish_reason="stop",
            values_tok_s={"d1": 58.0, "d2": 66.0, "d3": 75.0},
            sample_counts={"d1": 8, "d2": 8, "d3": 16},
            ar_baseline_tps=40.0,
        )

        assert store.start_depth(key, configured_depth=3) == (
            3,
            "profile_validated_d3",
        )

    def test_missing_ar_baseline_teaches_nothing(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=2, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="stop",
                      values_tok_s={"d2": 99.0},
                      sample_counts={"d2": 10}, ar_baseline_tps=None)
        assert store.start_depth(key, configured_depth=3) == (0, "unseen_ar")

    def test_fallback_records_ar_wins(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost_ratio=1.2",
                      finish_reason="fallback_to_ar")
        assert store.start_depth(key, configured_depth=3) == (0, "profile_ar")

    def test_snapshot_shape(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=True,
                          prompt_tokens=64, has_tools=True)
        _observe_profitable(store, key, depth=2, value=41.5, ar=30.0)
        snap = store.snapshot()
        assert "greedy|restored|short|tools" in snap
        entry = snap["greedy|restored|short|tools"]
        assert entry["learned_depth"] == 2
        assert entry["last_ar_baseline_tps"] == 30.0


class TestSeedPathIntegration:
    def _build_generator(self, monkeypatch):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_USE_TUNING", "0")
        vocab_size = 16

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _LM:
            def make_mtp_cache(self):
                return []

            def __call__(self, input_ids, cache=None, return_hidden=False, **_kw):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                logits = logits_for([(tokens[-1] + 1) % vocab_size])
                if return_hidden:
                    return logits, mx.ones((1, 1, 4), dtype=mx.float32)
                return logits

            def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                            return_hidden=False):
                next_id = int(next_token_ids.reshape(-1).tolist()[-1])
                logits = logits_for([(next_id + 1) % vocab_size])
                if return_hidden:
                    return logits, mx.ones((1, 1, 4), dtype=mx.float32)
                return logits

        lm = _LM()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = lm
        generator.model = type("_VLM", (), {"language_model": lm})()
        generator.stop_tokens = set()
        generator._stats = MLLMBatchStats()
        generator._native_mtp_enabled_for_request = lambda _req: True

        def greedy_sampler(logits):
            return mx.argmax(logits, axis=-1).astype(mx.uint32)

        greedy_sampler._vmlx_accepts_logits = True
        generator._make_request_sampler = lambda _req: greedy_sampler

        req = MLLMBatchRequest(
            uid=0, request_id="profile-seed-test", prompt="",
            max_tokens=8, temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]
        first_token = mx.array([2], dtype=mx.uint32)
        return generator, req, first_token

    def test_unknown_profile_stays_ar_and_validated_profile_activates(
        self, monkeypatch
    ):
        from vmlx_engine.native_mtp_profile import NativeMTPProfileStore

        generator, req, first_token = self._build_generator(monkeypatch)

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is False, "unknown workload shape must stay AR"
        assert not hasattr(req, "_native_mtp_state")

        # A validated in-process profile entry activates MTP at its depth.
        store = generator._native_mtp_profiles
        assert isinstance(store, NativeMTPProfileStore)
        from vmlx_engine.native_mtp_profile import profile_key as pk
        key = pk(temperature=0.0, restored_prefix=False, prompt_tokens=2)
        _observe_profitable(store, key, depth=2, value=45.0, ar=30.0)
        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 2
        assert state.stats.profile_seed == "profile_validated_d2"
        assert state.stats.profile_key_label == "greedy|False|short|False"

    def test_explicit_env_depth_override_bypasses_profile(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 3
        assert state.stats.profile_seed == "configured"

    def test_scheduled_reentry_is_not_vetoed_by_startup_ar_profile(self, monkeypatch, caplog):
        import logging
        caplog.set_level(logging.INFO)
        generator, req, first_token = self._build_generator(monkeypatch)
        generator._model_type = "qwen4_exp"
        store = generator._native_mtp_profiles = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=2)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        # Fresh startup still honors the advisory profile. A scheduled probe
        # must instead reach the request-local measured value controller.
        assert not generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None])
        for intended in (1, 2, 3):
            assert generator._seed_native_mtp_from_prefill(
                req, [object()], first_token, [None], start_depth_override=intended)
            state = req._native_mtp_state
            assert state.depth == intended
            assert state.profile_key == key
            assert state.stats.profile_seed == "request_local_reentry"
            assert f"depth={intended} ceiling=3 seed=request_local_reentry" in caplog.text

    def test_scheduled_reentry_keeps_request_eligibility_gate(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        generator._native_mtp_enabled_for_request = lambda _req: False
        assert not generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None], start_depth_override=3)

    def test_learned_ar_start_arms_measured_recovery_without_seed(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        store = generator._native_mtp_profiles = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=2)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        assert not generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None])
        assert not hasattr(req, "_native_mtp_state")
        tier = req._native_mtp_ar_tier
        assert tier.depth == 3
        assert tier.fallbacks == 0
        assert not tier.probe_due()
        for n in range(tier.next_probe_tokens):
            tier.record_step(1.0 + n * 0.03)
        assert tier.probe_due()
        assert tier.probe_depth(tier.depth) == 1

    def test_unknown_start_does_not_invent_recovery_eligibility(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        assert not generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None])
        assert getattr(req, "_native_mtp_ar_tier", None) is None

    def test_learned_ar_start_respects_reentry_disabled(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_AR_REENTRY", "0")
        store = generator._native_mtp_profiles = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=2)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        assert not generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None])
        assert getattr(req, "_native_mtp_ar_tier", None) is None

    def test_cached_prompt_uses_full_context_for_profile_and_governor(self, monkeypatch):
        import mlx.core as mx

        generator, req, first_token = self._build_generator(monkeypatch)
        generator._model_type = "qwen4_exp"
        req._original_token_ids = [101] * 10000
        req._gen_prefix_tokens = [102, 103]
        req._cached_tokens = 9999
        req.input_ids = mx.array([[101, 102, 103]])
        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.stats.profile_key_label == "greedy|True|medium|False"
        assert state.ar_safety.prompt_tokens == 10002

    def test_fixed_cached_prompt_keeps_full_context_across_reseed(self, monkeypatch):
        import mlx.core as mx

        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        req._original_token_ids = [101] * 10000
        req._cached_tokens = 9998
        req.input_ids = mx.array([[101, 102]])
        for override in (None, 1):
            assert generator._seed_native_mtp_from_prefill(
                req, [object()], first_token, [None], start_depth_override=override
            ) is True
            assert req._native_mtp_state.ar_safety.prompt_tokens == 10000

    def test_reentry_context_includes_tokens_generated_before_this_phase(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        req._original_token_ids = [101] * 10000
        req._gen_prefix_tokens = [102, 103]
        for generated, depth in ((0, None), (1024, 1), (2048, 2)):
            req.num_tokens = generated
            assert generator._seed_native_mtp_from_prefill(
                req, [object()], first_token, [None], start_depth_override=depth
            )
            state = req._native_mtp_state
            assert state.ar_safety.prompt_tokens == 10002 + generated
            # A rung-window reset must not forget the request-wide offset.
            state.ar_safety.reset(12)
            assert state.ar_safety.prompt_tokens == 10002 + generated

    def test_seed_without_canonical_ids_includes_restored_prefix(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        del req._original_token_ids
        req._cached_tokens = 8000
        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        assert req._native_mtp_state.ar_safety.prompt_tokens == 8002

    def test_qwen4_exp_unseen_profile_starts_at_measured_d3(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        generator._model_type = "qwen4_exp"

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 3
        assert state.stats.profile_seed == "qwen4_exp_measured_cold_start_d3"

    def test_qwen4_sampled_profile_does_not_trust_coarse_tuning(self, monkeypatch):
        from vmlx_engine import native_mtp

        generator, req, first_token = self._build_generator(monkeypatch)
        generator._model_type = "qwen4_exp"
        req.temperature = 1.0
        monkeypatch.setattr(
            native_mtp,
            "native_mtp_effective_depth",
            lambda: (2, "vmlx_mtp_tuning.json:best_depth"),
        )

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is False
        assert not hasattr(req, "_native_mtp_state")

    def test_batch_request_tool_metadata_separates_adaptive_profile(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        req.extra_kwargs = {
            "_vmlx_tools_present": True,
            "_vmlx_template_tools": [
                {"type": "function", "function": {"name": "run_command"}}
            ],
            "tool_choice": "auto",
        }

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is False
        store = generator._native_mtp_profiles
        from vmlx_engine.native_mtp_profile import profile_key as pk
        tool_key = pk(
            temperature=0.0,
            restored_prefix=False,
            prompt_tokens=2,
            has_tools=True,
        )
        _observe_profitable(store, tool_key, depth=2, value=45.0, ar=30.0)

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 2
        assert state.stats.profile_seed == "profile_validated_d2"
        assert state.stats.profile_key_label == "greedy|False|short|True"


class TestStateIntegration:
    def test_stats_report_profile_fields(self):
        from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats

        stats = MLLMNativeMTPStats()
        stats.profile_seed = "profile_validated_d2"
        stats.profile_key_label = "greedy|False|short|False"
        payload = stats.to_dict(
            request_id="r1", finish_reason="stop", final_depth=2
        )
        assert payload["profile_seed"] == "profile_validated_d2"
        assert payload["profile_key"] == "greedy|False|short|False"

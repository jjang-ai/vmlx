# SPDX-License-Identifier: Apache-2.0
"""Depth-N MTP draft/verify cycle — greedy identity + rollback correctness.

Drives a real ``mlx_lm.generate.GenerationBatch`` (the patched one) with a
real Hy3 model instance and a real KV cache, so the test exercises the actual
production seam: post-init -> draft chain -> verify forward -> longest-prefix
acceptance -> KV trim rollback -> emit queue.

The load-bearing invariant: **greedy output must be byte-identical with MTP
off, depth 1, depth 2 and depth 3.** Speculative decoding is only sound if the
draft never changes what the base model would have produced. A depth-N bug
(wrong rollback count, off-by-one hidden index, stale draft cache) shows up
here as a token divergence, not as a crash.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


def _tiny_hy3_args(nextn: int = 1):
    from jang_tools.hy3.model import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "hy_v3",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "route_norm": True,
            "router_scaling_factor": 2.826,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
            "num_nextn_predict_layers": nextn,
            "enable_lm_head_fp32": True,
        }
    )


def _build_model(attach_mtp: bool):
    """Deterministic tiny Hy3. Same seed => same weights => same greedy tokens."""
    from jang_tools.hy3.model import Model

    mx.random.seed(1234)
    model = Model(_tiny_hy3_args())
    if attach_mtp:
        model.attach_mtp()
    model.eval()
    mx.eval(model.parameters())
    return model


def _greedy_sampler(lp):
    return mx.argmax(lp, axis=-1).astype(mx.uint32)


def _make_batch(model, prompt, max_tokens: int, cache=None):
    """Build a patched GenerationBatch the way BatchGenerator does: the cache
    already holds ``prompt[:-1]``; ``inputs`` is the last prompt token (B,)."""
    import sys

    gm = sys.modules["mlx_lm.generate"]

    # BatchKVCache is what BatchGenerator hands GenerationBatch; plain KVCache
    # lacks extract()/filter() and blows up on the finish path.
    if cache is None:
        cache = [gm.BatchKVCache(left_padding=[0]) for _ in model.layers]
    if len(prompt) > 1:
        mx.eval(model(mx.array(prompt[:-1], dtype=mx.uint32)[None, :], cache=cache))

    return gm.GenerationBatch(
        model=model,
        uids=[0],
        inputs=mx.array([prompt[-1]], dtype=mx.uint32),
        prompt_cache=cache,
        tokens=[list(prompt)],
        samplers=[None],
        fallback_sampler=_greedy_sampler,
        logits_processors=[None],
        state_machines=[gm.SequenceStateMachine()],
        max_tokens=[max_tokens],
    )


def _run_generation(model, prompt, max_tokens: int):
    """Drive GenerationBatch (patched) to completion, greedy, no stop tokens."""
    batch = _make_batch(model, prompt, max_tokens)

    out = []
    while len(out) < max_tokens:
        responses = batch.next()
        if not responses:
            break
        for r in responses:
            out.append(int(r.token))
            if r.finish_reason is not None:
                return out
    return out


def _tokens_at_depth(monkeypatch, depth, attach_mtp: bool, max_tokens=24):
    from vmlx_engine.patches.mlx_lm_mtp import (
        apply_mlx_lm_mtp_patch,
        set_mtp_active,
    )

    assert apply_mlx_lm_mtp_patch() is True
    if depth is None:
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
    else:
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", str(depth))
    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")

    prev = None
    try:
        from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active

        prev = is_mtp_active()
        set_mtp_active(attach_mtp)
        model = _build_model(attach_mtp)
        return _run_generation(model, [3, 5, 7, 11, 13], max_tokens)
    finally:
        if prev is not None:
            set_mtp_active(prev)


class TestMtpDepthGreedyIdentity:
    """MTP must be a pure throughput optimization: identical greedy tokens."""

    def test_baseline_no_mtp_matches_depth_1_2_3(self, monkeypatch):
        baseline = _tokens_at_depth(monkeypatch, None, attach_mtp=False)
        assert len(baseline) == 24

        for depth in (1, 2, 3):
            got = _tokens_at_depth(monkeypatch, depth, attach_mtp=True)
            assert got == baseline, (
                f"depth={depth} diverged from non-MTP greedy baseline\n"
                f"  baseline={baseline}\n  got     ={got}"
            )

    def test_depth_resolves_and_is_recorded_in_stats(self, monkeypatch):
        import sys



        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        gm = sys.modules["mlx_lm.generate"]

        prev = is_mtp_active()
        try:
            set_mtp_active(True)
            model = _build_model(attach_mtp=True)
            batch = _make_batch(model, [3, 5, 7], 16)
            state = batch._omlx_mtp_state
            assert state.depth == 3
            assert state.stats.depth == 3
            # post-init drafts a full chain of `depth` tokens. The chain is
            # lazy: draft_ids stay empty until the verify cycle's single eval
            # materializes them (no per-draft host sync).
            assert len(state.draft_toks) == 3
            assert state.draft_ids == []
            assert len(state.draft_lps) == 3

            # Drain the 2 init tokens, then one verify cycle.
            batch.next()
            batch.next()
            batch.next()
            assert state.stats.cycles == 1
            assert state.stats.draft_tokens_proposed == 3
            assert 0 <= state.stats.draft_tokens_accepted <= 3
            # A fresh chain is always drafted for the next cycle.
            assert len(state.draft_toks) == 3
        finally:
            set_mtp_active(prev)

    def test_adaptive_starts_shallow_and_retains_configured_ceiling(self, monkeypatch):
        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
        prev = is_mtp_active()
        try:
            set_mtp_active(True)
            model = _build_model(attach_mtp=True)
            batch = _make_batch(model, [3, 5, 7], 16)
            state = batch._omlx_mtp_state
            assert state.depth == 1
            assert state.depth_ceiling == 3
            assert state.stats.starting_depth == 1
            assert state.stats.depth_ceiling == 3
            assert state.stats.depth_policy == "adaptive"
            assert len(state.draft_toks) == 1
        finally:
            set_mtp_active(prev)

    def test_adaptive_promotes_only_after_measured_shallow_cycles(self, monkeypatch):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _adaptive_arm_cycle,
            _adaptive_finish_cycle,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_VALUE_MIN_SAMPLES", "2")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_VALUE_COOLDOWN_CYCLES", "2")
        state = _MtpState(depth=1, depth_ceiling=3, adaptive_enabled=True)
        state.stats.depth = 1
        _adaptive_arm_cycle(state, now=1.0)

        # Production holds the configured/profile seed for 48 cycles so a
        # cold MTP-head cache cannot trigger a false early depth decision.
        # Exercise the final two measured D1 cycles at that boundary.
        state.stats.cycles = 47
        _adaptive_finish_cycle(
            "adaptive-row",
            state,
            completed_depth=1,
            accepted=1,
            now=2.0,
        )
        assert state.depth == 1
        _adaptive_arm_cycle(state, now=2.0)

        state.stats.cycles = 48
        _adaptive_finish_cycle(
            "adaptive-row",
            state,
            completed_depth=1,
            accepted=1,
            now=3.0,
        )
        assert state.depth == 2
        assert state.stats.depth == 2
        assert state.stats.adaptive_depth_value["active_probe"] == {
            "origin": 1,
            "target": 2,
        }

    def test_text_cost_fallback_is_measured_and_explicit_for_fixed_depth(
        self, monkeypatch
    ):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _text_mtp_maybe_cost_fallback,
        )

        state = _MtpState(depth=3, depth_ceiling=3, adaptive_enabled=False)
        state.stats.cycles = 8
        state.stats.draft_tokens_accepted = 8
        state.stats.backbone_ms = 320.0
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_COST_FALLBACK", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_COST_FALLBACK", raising=False)

        # Fixed D3 is an exact UI/user selection. The default adaptive runtime
        # gate must never override it.
        assert not _text_mtp_maybe_cost_fallback("fixed", state, now=10.0)
        assert state.ar_fallback_pending is False

        # The benchmark/research switch supplies its matched AR calibration
        # explicitly and is therefore allowed to terminate a fixed arm.
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_FALLBACK", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_AR_STEP_MS", "10")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD", "1.0")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_MIN_CYCLES", "8")
        assert _text_mtp_maybe_cost_fallback("fixed", state, now=10.0)
        assert state.ar_fallback_pending is True
        assert state.stats.fallback_cost_ratio == pytest.approx(2.0)
        assert "calibrated_cost" in (state.ar_fallback_reason or "")

    def test_text_runtime_cost_gate_only_owns_active_adaptive_requests(
        self, monkeypatch
    ):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _text_mtp_maybe_cost_fallback,
        )

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_RUNTIME_COST_MIN_CYCLES", "8")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_RUNTIME_COST_MARGIN", "1.25")
        state = _MtpState(
            depth=1,
            depth_ceiling=3,
            adaptive_enabled=True,
            ar_step_ms=10.0,
            cycle_span_start=1.0,
        )
        state.stats.cycles = 8
        state.stats.draft_tokens_accepted = 8
        assert _text_mtp_maybe_cost_fallback("adaptive", state, now=1.3)
        assert state.stats.fallback_cost_ratio == pytest.approx(1.875)
        assert "runtime_cost" in (state.ar_fallback_reason or "")

    def test_text_cost_fallback_drains_verified_queue_then_matches_ar(
        self, monkeypatch
    ):
        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            native_mtp_stats_snapshot,
        )

        assert apply_mlx_lm_mtp_patch() is True
        prompt = [3, 5, 7, 11, 13]
        max_tokens = 24
        previous = is_mtp_active()
        try:
            set_mtp_active(False)
            baseline = _run_generation(
                _build_model(attach_mtp=False), prompt, max_tokens
            )

            monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_FALLBACK", "1")
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_AR_STEP_MS", "0.0001")
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD", "1")
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_COST_MIN_CYCLES", "1")
            set_mtp_active(True)
            batch = _make_batch(
                _build_model(attach_mtp=True), prompt, max_tokens
            )
            got = []
            while len(got) < max_tokens:
                responses = batch.next()
                if not responses:
                    break
                got.extend(int(response.token) for response in responses)
                if responses[-1].finish_reason is not None:
                    break

            assert got == baseline
            assert getattr(batch, "_omlx_mtp_state", None) is None
            receipt = native_mtp_stats_snapshot()["last_native_mtp"]
            assert receipt["finish_reason"] == "length"
            assert receipt["final_depth"] == 0
            assert "calibrated_cost" in receipt["fallback_reason"]
            assert receipt["recovery"]["productive_ar_tokens"] > 0
            assert receipt["recovery"]["attempts"] == 0
        finally:
            set_mtp_active(previous)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    @pytest.mark.parametrize("wrong_from_step", [None, 1, 2])
    def test_productive_ar_reentry_preserves_pending_token_and_history(
        self, monkeypatch, depth, wrong_from_step
    ):
        """Pin the resume seam before wiring an automatic recovery policy.

        A performance handoff consumes the last visible MTP token. After
        productive stock steps, post-init must consume the *pending* sample,
        not the last visible token, and emit it exactly once.
        """
        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch, is_mtp_active, set_mtp_active,
        )
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import _post_init_mtp

        assert apply_mlx_lm_mtp_patch() is True
        prompt, limit = [3, 5, 7, 11, 13], 48
        previous = is_mtp_active()
        try:
            set_mtp_active(False)
            expected = _run_generation(_FakeCacheModel(), prompt, limit)
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", str(depth))
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
            set_mtp_active(True)
            batch = _make_batch(_FakeCacheModel(wrong_from_step), prompt, limit)
            # The shared fixture passes the complete prompt to stock init,
            # whose initial _step appends the final prompt token. Isolate
            # handoff/re-entry mutations from that existing fixture convention.
            initial_history = list(batch.tokens[0])
            state = batch._omlx_mtp_state
            state.ar_fallback_pending = True
            state.ar_fallback_reason = "test controlled performance handoff"
            got = []
            while getattr(batch, "_omlx_mtp_state", None) is not None:
                got.extend(int(r.token) for r in batch.next())
                assert len(got) < limit
            for _ in range(5):
                got.extend(int(r.token) for r in batch.next())
            assert batch.tokens[0] == initial_history + got
            for cache in batch.prompt_cache:
                assert cache.offset == len(prompt) + len(got)
            pending = int(batch._next_tokens.item())
            _post_init_mtp(batch)
            # Re-priming computes ahead, but cannot publish or append history.
            assert batch.tokens[0] == initial_history + got
            assert batch._omlx_mtp_state.queue[0][0] == pending
            first = batch.next()
            assert int(first[0].token) == pending
            got.extend(int(r.token) for r in first)
            while len(got) < limit:
                responses = batch.next()
                assert responses
                got.extend(int(r.token) for r in responses)
                if responses[-1].finish_reason is not None:
                    break
            assert got == expected
        finally:
            set_mtp_active(previous)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    @pytest.mark.parametrize("wrong_from_step", [None, 1, 2])
    def test_automatic_ar_reentry_keeps_ceiling_and_exact_output(
        self, monkeypatch, depth, wrong_from_step
    ):
        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch, is_mtp_active, set_mtp_active,
        )
        from vmlx_engine.native_mtp_recovery import NativeMTPRecovery
        assert apply_mlx_lm_mtp_patch()
        previous = is_mtp_active()
        prompt, limit = [3, 5, 7, 11], 180
        # Deterministic timing input tests policy/identity, not performance.
        observe = NativeMTPRecovery.observe_standard
        monkeypatch.setattr(NativeMTPRecovery, "observe_standard", lambda self, ms: observe(self, 100.0))
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", str(depth))
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        try:
            set_mtp_active(False)
            expected = _run_generation(_FakeCacheModel(), prompt, limit)
            set_mtp_active(True)
            batch = _make_batch(_FakeCacheModel(wrong_from_step), prompt, limit)
            batch._omlx_mtp_state.ar_fallback_pending = True
            batch._omlx_mtp_state.ar_fallback_reason = "controlled performance park"
            got, entered = [], False
            while len(got) < limit:
                before = getattr(batch, "_omlx_mtp_state", None)
                responses = batch.next()
                assert responses
                got.extend(int(r.token) for r in responses)
                state = getattr(batch, "_omlx_mtp_state", None)
                if before is None and state is not None:
                    entered = True
                    assert state.depth == 1
                    assert state.depth_ceiling == depth
                    assert state.recovery.standard_tokens == 128
                    assert state.recovery.attempts == 1
                    assert state.ar_step_ms == 100.0
                    assert state.stats.request_counted
                if state is not None:
                    assert 1 <= state.depth <= depth
                if responses[-1].finish_reason is not None:
                    break
            assert entered and got == expected
            assert not hasattr(batch, "_vmlx_mtp_recovery")
        finally:
            set_mtp_active(previous)

    def test_cache_length_tracks_emitted_tokens_exactly(self, monkeypatch):
        """Rollback must leave the KV cache exactly at the confirmed prefix.

        Off-by-one trims are invisible in short greedy runs (the model
        re-reads a stale key) but corrupt long generations. Pin the cache
        offset against prompt_len + emitted tokens.
        """
        import sys



        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        gm = sys.modules["mlx_lm.generate"]

        prev = is_mtp_active()
        try:
            set_mtp_active(True)
            model = _build_model(attach_mtp=True)
            prompt = [3, 5, 7, 11]
            import sys as _s; cache = [_s.modules["mlx_lm.generate"].BatchKVCache(left_padding=[0]) for _ in model.layers]
            batch = _make_batch(model, prompt, 20, cache=cache)

            emitted = 0
            for _ in range(20):
                responses = batch.next()
                if not responses:
                    break
                emitted += len(responses)
                if responses[-1].finish_reason is not None:
                    break
                # The cache holds the prompt plus every token the backbone
                # has consumed as *confirmed* input. Speculative positions
                # beyond the emit frontier are always rolled back, so the
                # offset can never exceed prompt + emitted.
                for c in cache:
                    assert c.offset <= len(prompt) + emitted, (
                        f"cache offset {c.offset} exceeds confirmed frontier "
                        f"{len(prompt) + emitted} — rollback under-trimmed"
                    )
        finally:
            set_mtp_active(prev)


class _FakeCacheModel:
    """Deterministic toy model with an exactly-controllable MTP head.

    A randomly-initialized real model almost never accepts a draft (measured:
    1/111 draft tokens, 0 full-accept cycles), so a greedy-identity test built
    on one exercises the reject path only. This fake pins the successor rule

        next(t) = (t * 7 + 1) % vocab

    in both the backbone and the MTP head, so every draft is correct and the
    ACCEPT path (k == n, bonus emit, no rollback) runs every cycle.

    ``wrong_from_step`` poisons the head from the given 1-based chain step so
    partial acceptance (0 < k < n) and an ``n - k`` rollback are exercised.
    """

    vocab = 64
    hidden = 8

    def __init__(self, wrong_from_step: int | None = None):
        self.wrong_from_step = wrong_from_step
        self.layers = [object()]
        self.mtp = [object()]  # presence gates _is_mtp_eligible
        self.n_backbone_calls = 0

    # --- helpers ---
    @staticmethod
    def _succ(tok: int) -> int:
        return (tok * 7 + 1) % _FakeCacheModel.vocab

    def _onehot(self, ids: list[int]):
        """(1, L, vocab) logits peaked at ``_succ(id)`` for each position."""
        targets = mx.array([[self._succ(t) for t in ids]])  # (1, L)
        return mx.where(
            mx.arange(self.vocab)[None, None, :] == targets[:, :, None],
            100.0,
            0.0,
        )

    def _advance_cache(self, cache, length: int):
        for c in cache:
            k = mx.zeros((1, 1, length, 4))
            c.update_and_fetch(k, k)

    # --- runtime contract ---
    def __call__(self, inputs, cache=None, return_hidden=False,
                 return_logits=True, n_confirmed=0):
        self.n_backbone_calls += 1
        ids = [int(t) for t in inputs[0].tolist()]
        if cache:
            self._advance_cache(cache, len(ids))
        logits = self._onehot(ids)
        if not return_logits:
            return mx.zeros((1, len(ids), self.hidden))
        if return_hidden:
            # hidden[..., 0] = token id, hidden[..., 1] = 1.0 backbone marker.
            # The marker lets mtp_forward tell a chain's first step (fed the
            # BACKBONE hidden) from later steps (fed the HEAD's own hidden) —
            # so this fixture also pins that _draft_chain recurses correctly.
            h = mx.zeros((1, len(ids), self.hidden))
            tok_ch = mx.array([[float(t) for t in ids]])[:, :, None] * mx.array(
                [1.0] + [0.0] * (self.hidden - 1)
            )
            marker = mx.array([0.0, 1.0] + [0.0] * (self.hidden - 2))
            return logits, h + tok_ch + marker
        return logits

    def make_mtp_cache(self):
        return [{"step": 0}]

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                    return_hidden=False):
        # Chain step 1 iff we were handed a backbone hidden (marker set).
        is_chain_start = float(hidden_states[0, 0, 1].item()) > 0.5
        if mtp_cache:
            mtp_cache[0]["step"] = 1 if is_chain_start else mtp_cache[0]["step"] + 1
            step = mtp_cache[0]["step"]
        else:
            step = 1 if is_chain_start else 2

        tok = int(next_token_ids[0, 0].item())
        draft = self._succ(tok)
        if self.wrong_from_step is not None and step >= self.wrong_from_step:
            draft = (draft + 1) % self.vocab  # deliberately wrong
        logits = mx.where(
            mx.arange(self.vocab)[None, None, :] == draft, 100.0, 0.0
        )
        if return_hidden:
            # No backbone marker: this is the head's own hidden.
            h = mx.zeros((1, 1, self.hidden)) + mx.array(
                [[[float(draft)] + [0.0] * (self.hidden - 1)]]
            )
            return logits, h
        return logits


class _TrackingMtpCache:
    def __init__(self):
        self.history = []

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = max(0, min(int(count), len(self.history)))
        if count:
            del self.history[-count:]
        return count


class _AlignedGlmFakeCacheModel(_FakeCacheModel):
    """GLM-shaped head that records cache history at aligned commit calls."""

    model_type = "glm5_next"

    def __init__(self, wrong_from_step: int | None = None):
        super().__init__(wrong_from_step=wrong_from_step)
        self.config = {"model_type": "glm5_next"}
        self.aligned_commit_prefixes = []

    def make_mtp_cache(self):
        return [_TrackingMtpCache()]

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache, return_hidden=False
    ):
        ids = [int(tok) for tok in next_token_ids[0].tolist()]
        cache = mtp_cache[0]
        if len(ids) > 1:
            self.aligned_commit_prefixes.append(list(cache.history))
        cache.history.extend(ids)

        rows = []
        hidden_rows = []
        for index, tok in enumerate(ids):
            draft = self._succ(tok)
            backbone_hidden = float(hidden_states[0, index, 1].item()) > 0.5
            if (
                self.wrong_from_step is not None
                and not backbone_hidden
                and self.wrong_from_step <= 2
            ):
                draft = (draft + 1) % self.vocab
            rows.append(
                mx.where(mx.arange(self.vocab) == draft, 100.0, 0.0)
            )
            hidden_rows.append([float(draft)] + [0.0] * (self.hidden - 1))
        logits = mx.stack(rows)[None, :, :]
        if return_hidden:
            return logits, mx.array([hidden_rows])
        return logits


def _run_fake(monkeypatch, depth: int, wrong_from_step=None, max_tokens=18,
              attach: bool = True, model_cls=_FakeCacheModel):
    import sys

    from vmlx_engine.patches.mlx_lm_mtp import (
        apply_mlx_lm_mtp_patch,
        is_mtp_active,
        set_mtp_active,
    )

    assert apply_mlx_lm_mtp_patch() is True
    monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", str(depth))
    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    gm = sys.modules["mlx_lm.generate"]

    prev = is_mtp_active()
    try:
        set_mtp_active(attach)
        model = model_cls(wrong_from_step=wrong_from_step)
        if not attach:
            model.mtp = None
        cache = [gm.BatchKVCache(left_padding=[0])]
        batch = gm.GenerationBatch(
            model=model,
            uids=[0],
            inputs=mx.array([3], dtype=mx.uint32),
            prompt_cache=cache,
            tokens=[[3]],
            samplers=[None],
            fallback_sampler=_greedy_sampler,
            logits_processors=[None],
            state_machines=[gm.SequenceStateMachine()],
            max_tokens=[max_tokens],
        )
        # Grab the stats object now: the finish path deletes
        # ``_omlx_mtp_state`` from the batch before returning the last token.
        state = getattr(batch, "_omlx_mtp_state", None)
        stats = state.stats if state is not None else None

        out = []
        while len(out) < max_tokens:
            responses = batch.next()
            if not responses:
                break
            for r in responses:
                out.append(int(r.token))
                if r.finish_reason is not None:
                    return out, stats, cache
        return out, stats, cache
    finally:
        set_mtp_active(prev)


class TestMtpAcceptPathWithOracleDrafts:
    """Force 100% acceptance so the accept branch is actually covered."""

    def test_cache_snapshot_runs_once_at_terminal_not_per_cycle(self, monkeypatch):
        import vmlx_engine.patches.mlx_lm_mtp.batch_generator as mtp_batch

        original_snapshot = mtp_batch.native_mtp_cache_snapshot
        snapshot_calls = []

        def snapshot_spy(cache):
            snapshot_calls.append(cache)
            return original_snapshot(cache)

        monkeypatch.setattr(
            mtp_batch,
            "native_mtp_cache_snapshot",
            snapshot_spy,
        )

        _out, stats, _cache = _run_fake(
            monkeypatch,
            depth=3,
            max_tokens=18,
        )

        assert stats is not None
        assert stats.cycles > 1
        assert stats.mtp_forwards > 1
        assert len(snapshot_calls) == 1

    def test_failed_rollback_publishes_fallback_without_false_retention(
        self, monkeypatch
    ):
        import vmlx_engine.patches.mlx_lm_mtp.batch_generator as mtp_batch

        original_snapshot = mtp_batch.native_mtp_cache_snapshot
        snapshot_calls = []

        def snapshot_spy(cache):
            snapshot_calls.append(cache)
            return original_snapshot(cache)

        monkeypatch.setattr(
            mtp_batch,
            "native_mtp_cache_snapshot",
            snapshot_spy,
        )
        monkeypatch.setattr(
            mtp_batch,
            "_restore_or_trim_caches",
            lambda *_args, **_kwargs: False,
        )

        # A refused rollback leaves the rejected speculative advance in the
        # cache — no continuation is sound, so the cycle now fails the
        # request loudly (uniform with the MLLM path) after publishing
        # terminal telemetry. No false retention count either way.
        retained_before = mtp_batch.native_mtp_stats_snapshot()["native_mtp_totals"][
            "mtp_cache_retained_on_rejects"
        ]
        with pytest.raises(RuntimeError, match="rejected rollback"):
            _run_fake(
                monkeypatch,
                depth=1,
                wrong_from_step=1,
                max_tokens=8,
            )
        snapshot = mtp_batch.native_mtp_stats_snapshot()
        published = snapshot["last_native_mtp"]

        assert published["finish_reason"] == "rollback_refused"
        assert published["rejects"] == 1
        assert (
            snapshot["native_mtp_totals"]["mtp_cache_retained_on_rejects"]
            == retained_before
        )
        assert len(snapshot_calls) == 1

    def test_oracle_draft_accepts_every_cycle_and_matches_baseline(self, monkeypatch):
        # Ground truth: the successor rule, applied from the prompt token.
        expected, t = [], 3
        for _ in range(18):
            t = _FakeCacheModel._succ(t)
            expected.append(t)

        baseline, _, _ = _run_fake(monkeypatch, depth=1, attach=False)
        assert baseline == expected

        for depth in (1, 2, 3):
            got, stats, _ = _run_fake(monkeypatch, depth=depth)
            assert got == expected, f"depth={depth} diverged: {got} != {expected}"
            assert stats is not None and stats.cycles > 0
            # every cycle fully accepted its whole chain
            assert stats.rejects == 0, f"depth={depth} had rejects"
            assert stats.accepts == stats.cycles
            assert stats.draft_tokens_accepted == stats.draft_tokens_proposed
            assert stats.draft_tokens_proposed == stats.cycles * depth

    def test_deeper_chains_need_fewer_backbone_calls(self, monkeypatch):
        """The whole point of depth: fewer verify forwards per token."""
        calls = {}
        for depth in (1, 2, 3):
            import sys

            from vmlx_engine.patches.mlx_lm_mtp import (
                apply_mlx_lm_mtp_patch,
                is_mtp_active,
                set_mtp_active,
            )

            assert apply_mlx_lm_mtp_patch() is True
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", str(depth))
            monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
            gm = sys.modules["mlx_lm.generate"]
            prev = is_mtp_active()
            try:
                set_mtp_active(True)
                model = _FakeCacheModel()
                batch = gm.GenerationBatch(
                    model=model,
                    uids=[0],
                    inputs=mx.array([3], dtype=mx.uint32),
                    prompt_cache=[gm.BatchKVCache(left_padding=[0])],
                    tokens=[[3]],
                    samplers=[None],
                    fallback_sampler=_greedy_sampler,
                    logits_processors=[None],
                    state_machines=[gm.SequenceStateMachine()],
                    max_tokens=[24],
                )
                n = 0
                while n < 24:
                    r = batch.next()
                    if not r:
                        break
                    n += len(r)
                    if r[-1].finish_reason is not None:
                        break
                calls[depth] = model.n_backbone_calls
            finally:
                set_mtp_active(prev)

        assert calls[2] < calls[1], f"depth 2 not cheaper: {calls}"
        assert calls[3] < calls[2], f"depth 3 not cheaper: {calls}"

    def test_partial_acceptance_rolls_back_exactly_n_minus_k(self, monkeypatch):
        """Head correct for d1, wrong from d2 onward => k == 1 at depth 3."""
        expected, t = [], 3
        for _ in range(18):
            t = _FakeCacheModel._succ(t)
            expected.append(t)

        got, stats, cache = _run_fake(
            monkeypatch, depth=3, wrong_from_step=2, max_tokens=18
        )
        # Correctness is preserved despite bad drafts.
        assert got == expected
        assert stats.cycles > 0
        assert stats.accepts == 0  # never a full-chain accept
        assert stats.rejects == stats.cycles
        # Exactly one draft token accepted per cycle (d1 always right).
        assert stats.draft_tokens_accepted == stats.cycles
        assert stats.draft_tokens_proposed == stats.cycles * 3
        assert stats.mtp_cache_recreated_on_rejects == 0
        assert stats.mtp_cache_retained_on_rejects == stats.rejects
        # Cache must sit at the confirmed frontier: 1 prompt token + emits.
        for c in cache:
            assert int(c.offset.tolist()[0]) == 1 + len(got)


class TestGlmAlignedHeadCache:
    def test_gate_is_exact_family_and_default_off(self, monkeypatch):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _glm_aligned_head_cache_enabled,
        )

        monkeypatch.delenv(
            "VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", raising=False
        )
        monkeypatch.delenv(
            "VMLINUX_GLM5_ALIGNED_MTP_HEAD_CACHE", raising=False
        )

        class _Batch:
            model = type("Model", (), {"model_type": "glm5_next"})()

        assert _glm_aligned_head_cache_enabled(_Batch()) is False
        monkeypatch.setenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "1")
        assert _glm_aligned_head_cache_enabled(_Batch()) is True
        _Batch.model.model_type = "qwen4_exp"
        assert _glm_aligned_head_cache_enabled(_Batch()) is False

    def test_prompt_priming_gate_is_exact_family_and_default_off(
        self, monkeypatch
    ):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _glm_prompt_priming_enabled,
        )

        monkeypatch.delenv("VMLX_GLM5_MTP_PROMPT_PRIMING", raising=False)
        monkeypatch.delenv("VMLINUX_GLM5_MTP_PROMPT_PRIMING", raising=False)
        glm = type("Model", (), {"model_type": "glm5_next"})()
        qwen = type("Model", (), {"model_type": "qwen4_exp"})()
        assert _glm_prompt_priming_enabled(glm) is False
        monkeypatch.setenv("VMLX_GLM5_MTP_PROMPT_PRIMING", "1")
        assert _glm_prompt_priming_enabled(glm) is True
        assert _glm_prompt_priming_enabled(qwen) is False

    def test_trim_removes_only_unverified_chain_pairs(self):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _MtpState,
            _trim_glm_head_chain,
        )

        cache = _TrackingMtpCache()
        cache.history = [10, 11, 90, 91]
        state = _MtpState(mtp_cache=[cache], head_chain_pairs=2)
        assert _trim_glm_head_chain(state) is True
        assert cache.history == [10, 11]
        assert state.head_chain_pairs == 0

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_aligned_glm_keeps_target_tokens_and_reports_policy(
        self, monkeypatch, depth
    ):
        monkeypatch.setenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "1")
        expected, token = [], 3
        for _ in range(18):
            token = _FakeCacheModel._succ(token)
            expected.append(token)
        got, stats, _cache = _run_fake(
            monkeypatch,
            depth=depth,
            wrong_from_step=2,
            max_tokens=18,
            model_cls=_AlignedGlmFakeCacheModel,
        )
        assert got == expected
        assert stats is not None
        assert stats.mtp_head_cache_policy == "glm_aligned"

    def test_rejected_recursive_pair_is_trimmed_before_aligned_commit(
        self, monkeypatch
    ):
        import sys

        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "1")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        gm = sys.modules["mlx_lm.generate"]
        previous = is_mtp_active()
        try:
            set_mtp_active(True)
            model = _AlignedGlmFakeCacheModel(wrong_from_step=2)
            batch = gm.GenerationBatch(
                model=model,
                uids=[0],
                inputs=mx.array([3], dtype=mx.uint32),
                prompt_cache=[gm.BatchKVCache(left_padding=[0])],
                tokens=[[3]],
                samplers=[None],
                fallback_sampler=_greedy_sampler,
                logits_processors=[None],
                state_machines=[gm.SequenceStateMachine()],
                max_tokens=[12],
            )
            # Drain the two init tokens, then force one partial-reject cycle.
            batch.next()
            batch.next()
            batch.next()
            assert model.aligned_commit_prefixes
            # Initial history contains one confirmed pair plus two recursive
            # proposals. The aligned commit must see only the confirmed pair.
            assert model.aligned_commit_prefixes[0] == [
                model.aligned_commit_prefixes[0][0]
            ]
        finally:
            set_mtp_active(previous)


class TestDepthGating:
    def test_non_trimmable_cache_forces_depth_1(self, monkeypatch):
        """Depth > 1 needs partial rollback, which only trimmable KV supports.

        An SSM/hybrid layer exposes rollback_state (restores to the confirmed
        prefix wholesale) and must clamp to depth 1 so its proven behavior is
        untouched.
        """
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import _effective_depth

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "3")

        class _Trimmable:
            rollback_state = None

            def is_trimmable(self):
                return True

        class _SsmLike:
            rollback_state = (object(), object())

        class _Untrimmable:
            rollback_state = None

            def is_trimmable(self):
                return False

        class _Batch:
            def __init__(self, cache):
                self.prompt_cache = cache

        assert _effective_depth(_Batch([_Trimmable(), _Trimmable()])) == 3
        assert _effective_depth(_Batch([_Trimmable(), _SsmLike()])) == 1
        assert _effective_depth(_Batch([_Untrimmable()])) == 1

    def test_env_depth_clamped_to_1_3(self, monkeypatch):
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import _effective_depth

        class _Trimmable:
            rollback_state = None

            def is_trimmable(self):
                return True

        class _Batch:
            prompt_cache = [_Trimmable()]

        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "9")
        assert _effective_depth(_Batch()) == 3
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "0")
        assert _effective_depth(_Batch()) == 1

    def test_adaptive_unmeasured_text_workload_stays_ar(self, monkeypatch):
        from vmlx_engine import native_mtp
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _adaptive_mtp_activation_decision,
        )

        class _Cache:
            rollback_state = None

            def is_trimmable(self):
                return True

        class _Batch:
            prompt_cache = [_Cache()]

        monkeypatch.setattr(
            native_mtp,
            "native_mtp_effective_depth",
            lambda _path=None: (3, "default"),
        )
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")

        assert _adaptive_mtp_activation_decision(_Batch()) == (
            False,
            3,
            "adaptive_unseen_ar",
        )

    def test_adaptive_unmeasured_generation_batch_skips_seed(self, monkeypatch):
        from vmlx_engine import native_mtp
        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            native_mtp_stats_snapshot,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setattr(
            native_mtp,
            "native_mtp_effective_depth",
            lambda _path=None: (3, "default"),
        )
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")

        previous = is_mtp_active()
        try:
            set_mtp_active(True)
            batch = _make_batch(_build_model(attach_mtp=True), [3, 5, 7], 16)
            assert getattr(batch, "_omlx_mtp_state", None) is None
            skip = native_mtp_stats_snapshot()["last_native_mtp_skip"]
            assert skip == {
                "uid": "0",
                "reason": "adaptive_unseen_ar",
                "configured_depth": 3,
            }
        finally:
            set_mtp_active(previous)

    def test_adaptive_validated_tuning_activates_text_mtp(self, monkeypatch):
        from vmlx_engine import native_mtp
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _adaptive_mtp_activation_decision,
        )

        class _Cache:
            rollback_state = None

            def is_trimmable(self):
                return True

        class _Batch:
            prompt_cache = [_Cache()]

        source = "vmlx_mtp_tuning.json:native_mtp.best_depth"
        monkeypatch.setattr(
            native_mtp,
            "native_mtp_effective_depth",
            lambda _path=None: (2, source),
        )
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")

        assert _adaptive_mtp_activation_decision(_Batch()) == (True, 2, source)

    def test_fixed_depth_policy_still_activates_exact_depth(self, monkeypatch):
        from vmlx_engine import native_mtp
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import (
            _adaptive_mtp_activation_decision,
        )

        class _Cache:
            rollback_state = None

            def is_trimmable(self):
                return True

        class _Batch:
            prompt_cache = [_Cache()]

        monkeypatch.setattr(
            native_mtp,
            "native_mtp_effective_depth",
            lambda _path=None: (2, "VMLINUX_NATIVE_MTP_DEPTH"),
        )
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")

        assert _adaptive_mtp_activation_decision(_Batch()) == (
            True,
            2,
            "fixed_policy",
        )


class TestMeasurementHooksDefaultOff:
    """The profiling/bypass hooks added while diagnosing MTP throughput must be
    inert unless their env var is set — they add barriers / skip the MTP path,
    which would silently change production behavior if left on."""

    def test_profile_and_bypass_default_off(self):
        from vmlx_engine.patches.mlx_lm_mtp import batch_generator as bg

        assert bg._MTP_PROFILE is False
        assert bg._MTP_BYPASS is False

    def test_pbar_is_noop_when_profiling_off(self, monkeypatch):
        """_pbar must not force an eval (or import mlx) when profiling is off."""
        from vmlx_engine.patches.mlx_lm_mtp import batch_generator as bg

        monkeypatch.setattr(bg, "_MTP_PROFILE", False)
        # Passing a bare object would raise inside mx.eval; a no-op ignores it.
        bg._pbar(object(), object())  # must not raise


class TestBundleDepthSidecar:
    """Hy3-JANG_2L pins best_depth=1 so a forced MTP run does not inherit the
    global depth-3 default (which measured ~-43%). Depth 2/3 are strictly worse
    on this MoE: the verify grows ~11.5ms/extra token while chained-draft
    acceptance collapses."""

    def test_blocked_sidecar_leaks_no_legacy_depth(self, tmp_path, monkeypatch):
        import json

        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps({"native_mtp": {"blocked": True, "best_depth": 1},
                        "best_depth": 1})
        )
        depth, source = native_mtp_effective_depth(str(tmp_path))
        assert depth == 3
        assert source == "default"

    def test_no_sidecar_still_defaults_to_3(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import native_mtp_effective_depth

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        depth, source = native_mtp_effective_depth(str(tmp_path))
        assert depth == 3
        assert source == "default"


class TestTqLiveEncodeCrossingGuard:
    def test_verify_cycle_falls_back_before_tq_compress_crossing(self, monkeypatch):
        """A verify advance that would cross a TQ layer's one-time compress()
        must fall back to the standard step instead of running the cycle.

        trim() rewinds offset only, so a partial rejection after compress()
        fires inside the advance would leave draft KV baked into the
        compressed buffers — the text-path twin of the MLLM
        ``_native_mtp_should_snapshot_layer`` crossing guard.
        """
        import sys

        from vmlx_engine.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            is_mtp_active,
            set_mtp_active,
        )

        assert apply_mlx_lm_mtp_patch() is True
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_DEPTH", "1")

        prev = is_mtp_active()
        try:
            set_mtp_active(True)
            model = _build_model(attach_mtp=True)
            prompt = [3, 5, 7, 11]
            gm = sys.modules["mlx_lm.generate"]
            cache = [gm.BatchKVCache(left_padding=[0]) for _ in model.layers]
            batch = _make_batch(model, prompt, 16, cache=cache)
            assert getattr(batch, "_omlx_mtp_state", None) is not None

            # Arm a TQ-style crossing on one layer: post-init leaves the
            # cache at prompt+1 positions, so the first verify advance
            # (depth 1 -> 2 tokens) crosses this threshold.
            cache[0].compress_after = cache[0].offset + 1
            cache[0]._compressed_tokens = 0

            emitted = 0
            for _ in range(8):
                responses = batch.next()
                if not responses:
                    break
                emitted += len(responses)
                if responses[-1].finish_reason is not None:
                    break

            # The two queued init tokens drain, then the verify cycle hits
            # the crossing and drops MTP state; the standard path continues
            # emitting — no exception, no dead generation.
            assert emitted >= 3
            assert getattr(batch, "_omlx_mtp_state", None) is None
        finally:
            set_mtp_active(prev)

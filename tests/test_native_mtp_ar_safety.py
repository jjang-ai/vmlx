"""Windowed AR-safety governor — shared decision math + both lanes.

The governor drops a request to AR when a WINDOWED wall ms/token exceeds the
seed AR step scaled by the growth of the request's OWN cycle span (context
and thermal, never concurrency), for BOTH fixed and adaptive depth.
"""

import time
import pytest

from vmlx_engine.native_mtp_ar_safety import (
    ArSafetyState,
    ar_safety_step,
    windowed_ar_verdict as _native_mtp_windowed_ar_verdict,
)

BASE = dict(
    ar_step_ms=10.0,
    anchor_cycle_ms=30.0,
    cur_cycle_ms=30.0,   # flat -> scale 1.0
    context_ratio=1.0,
    delta_emitted=32,
    delta_wall_ms=160.0,  # 5 ms/tok
    margin=1.25,
)


def verdict(**over):
    return _native_mtp_windowed_ar_verdict(**{**BASE, **over})


def test_fast_mtp_holds():
    assert verdict() is None


def test_slow_mtp_trips():
    v = verdict(delta_emitted=16, delta_wall_ms=320.0)  # 20 ms/tok vs 10
    assert v is not None
    mtp_ms_per_tok, ar_baseline = v
    assert abs(mtp_ms_per_tok - 20.0) < 1e-6
    assert abs(ar_baseline - 10.0) < 1e-6


def test_acceptance_collapse_trips_without_cycle_wall_change():
    # Cycle wall unchanged (30ms), tokens/cycle fell from 3.2 to 1.0 ->
    # 30 ms/tok vs baseline 10 x 1.25 -> trips. Never masked by the scale.
    v = verdict(delta_emitted=16, delta_wall_ms=480.0, cur_cycle_ms=30.0)
    assert v is not None


def test_context_growth_scales_baseline_up_and_seed_is_a_floor():
    # Cycle wall 30 -> 45 (context/thermal): baseline 15 -> threshold 18.75;
    # MTP at 17 ms/tok holds.
    assert verdict(cur_cycle_ms=45.0, delta_emitted=16, delta_wall_ms=272.0) is None
    # Cycle wall BELOW the anchor (slow early cycles after a big prefill
    # inflated the anchor): the scale clamps at 1, baseline stays at the
    # seed (10 -> threshold 12.5); MTP at 12 ms/tok must NOT trip. Measured
    # 2026-09-04: a scale below 1 falsely demoted healthy MTP 3x in 22 turns.
    v = verdict(cur_cycle_ms=24.0, delta_emitted=16, delta_wall_ms=192.0)
    assert v is None
    v2 = verdict(cur_cycle_ms=24.0, delta_emitted=16, delta_wall_ms=272.0)
    assert v2 is not None and abs(v2[1] - 10.0) < 1e-6


def test_single_stall_cycle_does_not_trip():
    per_cycle = [5.0] * 15 + [400.0]
    v = verdict(delta_emitted=16, delta_wall_ms=sum(per_cycle),
                per_cycle_ms_per_tok=per_cycle)
    assert v is None
    sustained = [sum(per_cycle) / 16] * 16
    v2 = verdict(delta_emitted=16, delta_wall_ms=sum(sustained),
                 per_cycle_ms_per_tok=sustained)
    assert v2 is not None


def test_guards_return_none():
    assert verdict(ar_step_ms=0.0, delta_emitted=16, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=0, delta_wall_ms=320.0) is None
    assert verdict(delta_emitted=16, delta_wall_ms=0.0) is None


def _drive(st, *, cycle_ms, tok_per_cycle, seed_ar_ms=10.0,
           primed=True, start_cycle=1, end_cycle=60, t0=0.0, emitted0=0):
    """Feed the governor a synthetic run; returns (trip_cycle, trip, t, emitted).
    ``cycle_ms`` is the wall between cycles."""
    t = t0
    emitted = emitted0
    for cyc in range(start_cycle, end_cycle):
        t += cycle_ms(cyc) / 1000.0
        emitted += tok_per_cycle(cyc)
        trip = ar_safety_step(st, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=seed_ar_ms, primed=primed)
        if trip is not None:
            return cyc, trip, t, emitted
    return None, None, t, emitted


def test_first_judgment_cycle_primed_vs_unprimed(monkeypatch):
    for name in ("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", "VMLX_NATIVE_MTP_AR_SAFETY_WINDOW"):
        monkeypatch.delenv(name, raising=False)
    # 20 ms/cycle, 1 tok/cycle = 20 ms/tok vs AR 10 -> losing from the start.
    c, *_ = _drive(ArSafetyState(prompt_tokens=50), cycle_ms=lambda c: 20.0,
                   tok_per_cycle=lambda c: 1, primed=True)
    assert c == 16  # warmup 8 + window 8
    c2, *_ = _drive(ArSafetyState(prompt_tokens=50), cycle_ms=lambda c: 20.0,
                    tok_per_cycle=lambda c: 1, primed=False)
    assert c2 == 24  # warmup 16 + window 8


def test_anchor_is_warmup_median_of_own_span(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    # Solo healthy: 30ms cycles, 4 tok/cycle (7.5 ms/tok vs AR 10) -> holds,
    # anchor = 30 (warmup own-span median; cycle 1 is 200ms cold, ignored by
    # the median).
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 200.0 if c == 1 else 30.0,
                         tok_per_cycle=lambda c: 4, end_cycle=80)
    assert abs(st.anchor_cycle_ms - 30.0) < 1e-6
    assert c is None, (c, trip and trip.log_text(3))


def test_acceptance_collapse_trips_solo(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WARMUP", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    c, trip, *_ = _drive(st, cycle_ms=lambda c: 30.0,
                         tok_per_cycle=lambda c: 3 if c < 40 else 1, end_cycle=80)
    assert c is not None and 40 < c <= 50


def _vlm_state(m, depth=3):
    state = m.MLLMNativeMTPState(
        mtp_cache=[], next_main=None, drafts=[], draft_lps=[], draft_ids=[],
        depth=depth,
    )
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    return state


def test_safety_runs_and_trips_under_fixed_policy(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")  # fixed
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth_ceiling = 3
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    # Full ring, 1 tok/cycle at 20ms/cycle = 20 ms/tok, flat cycle wall.
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    # D3 loses -> D2 with a fresh window (not AR yet).
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is False
    assert state.depth == 2 and state.ar_fallback_pending is False
    assert state.ar_safety.ring == [] and state.ar_safety.cycle_base == 40
    assert state.promote_at_cycle > 40
    state.stats.cycles = 60
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(51 + i, 51 + i, base_t + i * .020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is False
    assert state.depth == 1 and not state.ar_fallback_pending
    # Rung 2: D1 loses a window -> pending confirmation, NOT yet AR.
    state.stats.cycles = 80
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(71 + i, 71 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is False
    assert state.ar_fallback_pending is False and state.ar_trip_pending_cycle == 80
    assert state.ar_safety.ring == []  # judged again from a fresh window
    # A second losing window within three windows confirms -> AR.
    state.stats.cycles = 90
    state.ar_safety.ring = [(81 + i, 81 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req-fixed", state) is True
    assert state.ar_fallback_pending is True and state.ar_trip_pending_cycle == 0
    assert "windowed_ar_safety d1" in (state.ar_fallback_reason or "")


def test_disabled_by_kill_switch(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_SAFETY", "0")
    state = _vlm_state(m)
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False


def test_text_lane_trips_under_fixed_policy(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as tl

    monkeypatch.setenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = tl._MtpState()
    state.adaptive_enabled = False
    state.depth = 3
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.anchor_context_tokens = 120
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.draft_tokens_accepted = 0
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]
    assert tl._text_mtp_maybe_cost_fallback("req", state, now=time.perf_counter()) is False
    # Rung 1: D3 -> D1.
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.ar_fallback_pending is False
    # Rung 2: D1 -> AR.
    state.stats.cycles = 80
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(71 + i, 71 + i, base_t + i * 0.020) for i in range(9)]
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is True
    assert state.ar_fallback_pending is True
    assert "windowed_ar_safety d1" in (state.ar_fallback_reason or "")
    assert state.stats.fallback_reason == state.ar_fallback_reason


def test_text_lane_fast_window_holds(monkeypatch):
    from vmlx_engine.patches.mlx_lm_mtp import batch_generator as tl

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = tl._MtpState()
    state.depth = 3
    state.ar_step_ms = 10.0
    state.ar_safety.prompt_tokens = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.draft_tokens_accepted = 40
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.010) for i in range(9)]
    assert tl._text_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_fallback_pending is False


def test_probe_margin_requires_beating_measured_ar():
    # A re-entry probe is judged with margin 1/1.10: MTP at 9.5 ms/tok vs
    # measured AR 10 does NOT beat it by the hysteresis -> trip (probe lost);
    # at 8.5 it does -> hold (probe kept).
    st = ArSafetyState(prompt_tokens=50)
    c, *_ = _drive(st, cycle_ms=lambda c: 19.0, tok_per_cycle=lambda c: 2,
                   seed_ar_ms=10.0, primed=False)
    assert c is None  # 9.5 <= 12.5 at the default margin: a settled request holds
    st2 = ArSafetyState(prompt_tokens=50)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(1, 60):
        t += 0.019; emitted += 2
        trip = ar_safety_step(st2, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10)
        if trip is not None:
            tripped = cyc; break
    assert tripped == 24  # unprimed warmup 16 + window 8
    st3 = ArSafetyState(prompt_tokens=50)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(1, 60):
        t += 0.017; emitted += 2
        trip = ar_safety_step(st3, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10)
        if trip is not None:
            tripped = cyc; break
    assert tripped is None


def test_ar_tier_backoff_and_measurement():
    from vmlx_engine.mllm_batch_generator import NativeMTPArTier

    tier = NativeMTPArTier(depth=3)
    t = 100.0
    for _ in range(15):
        tier.record_step(t); t += 0.025
    assert tier.probe_due() is False  # 15 < 16 tokens
    tier.record_step(t); t += 0.025
    assert tier.probe_due() is True
    assert abs(tier.measured_ar_ms_per_tok() - 25.0) < 1e-6
    tier.probe_failed()
    assert tier.next_probe_tokens == 32 and tier.backoff == 1 and tier.fallbacks == 2
    assert tier.probe_due() is False
    tier.probe_failed()
    assert tier.next_probe_tokens == 64


def test_probe_kept_switches_baseline_to_measured_ar(monkeypatch):
    # A probe that beats measured AR is kept: probe flag clears, reentries++.
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    tier = m.NativeMTPArTier(depth=3)
    t0 = time.perf_counter() - 2.0
    for i in range(16):
        tier.record_step(t0 + i * 0.025)  # measured AR 25 ms/tok
    state.ar_tier = tier
    state.probe = True
    state.depth = 1
    state.stats.prompt_prime_source = "unprimed"
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 80  # 3 tok/cycle
    state.ar_safety.anchor_cycle_ms = 40.0
    # 40ms cycles, 3 tok/cycle = 13.3 ms/tok < 25/1.10 -> kept
    state.ar_safety.ring = [(31 + i, 3 * (31 + i), base_t + i * 0.040) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.probe is False
    assert tier.reentries == 1
    assert state.promote_at_cycle == 40 + m._NATIVE_MTP_PROMOTE_FIRST_CYCLES
    # A probe that LOSES: 1 tok/cycle at 40ms = 40 ms/tok -> falls back, backoff.
    state2 = _vlm_state(m)
    tier2 = m.NativeMTPArTier(depth=3)
    for i in range(16):
        tier2.record_step(t0 + i * 0.025)
    state2.ar_tier = tier2
    state2.probe = True
    state2.stats.prompt_prime_source = "unprimed"
    state2.stats.cycles = 40
    state2.stats.accepted_tokens = 0
    state2.ar_safety.anchor_cycle_ms = 40.0
    state2.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.040) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req2", state2) is True
    assert state2.ar_fallback_pending is True
    assert tier2.backoff == 1 and tier2.next_probe_tokens == 32


def test_settled_reentry_that_trips_again_backs_off(monkeypatch):
    # After a kept probe, a later trip must count as a fallback, restart the
    # AR window, and back off further (no immediate re-probe).
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    tier = m.NativeMTPArTier(depth=3)
    t0 = time.perf_counter() - 2.0
    for i in range(20):
        tier.record_step(t0 + i * 0.030)
    tier.reentries = 1
    state.ar_tier = tier
    state.probe = False  # settled at D1 after a kept probe
    state.depth = 1
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 0
    state.ar_safety.anchor_cycle_ms = 40.0
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.040) for i in range(9)]
    # Fresh measured AR: the first complete losing window is sufficient.
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
    assert tier.fallbacks == 2
    assert tier.tokens_since_fallback == 0
    assert tier.next_probe_tokens == 32 and tier.backoff == 1  # rapid re-trip: back off
    assert tier.probe_due() is False
    # A re-entry that ran 300 tokens before tripping again was a real win: relax.
    state2 = _vlm_state(m); state2.ar_tier = tier; state2.depth = 1
    state2.stats.cycles = 300; state2.stats.accepted_tokens = 0
    state2.ar_safety.anchor_cycle_ms = 40.0
    state2.ar_safety.ring = [(291 + i, 291 + i, base_t + i * 0.040) for i in range(9)]
    # The tier retains last_measured_ms_per_tok across its empty AR window;
    # this 300-token-old baseline is still fresh, so fallback is immediate.
    assert m._native_mtp_maybe_ar_safety_fallback("req", state2) is True
    assert tier.backoff == 0 and tier.next_probe_tokens == 16 and tier.fallbacks == 3




@pytest.mark.parametrize("adaptive", ["0", "1"])
@pytest.mark.parametrize("ceiling", [2, 3])
def test_pending_d1_confirmation_cannot_be_preempted_by_promotion(monkeypatch, adaptive, ceiling):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", adaptive)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m, depth=1)
    state.depth_ceiling = state.ladder_depth = ceiling
    state.ar_tier = m.NativeMTPArTier(depth=ceiling)
    _ar_steps(state.ar_tier, 8, ms=10.0)
    state.stats.cycles = 115
    state.ar_trip_pending_cycle = 108
    state.promote_at_cycle = 115
    state.ar_safety.ring = [(c, c, 100.0 - (115 - c) * .020) for c in range(109, 115)]
    monkeypatch.setattr(m.time, "perf_counter", lambda: 100.0)

    assert m._native_mtp_maybe_ar_safety_fallback("pending-confirmation", state) is False
    assert state.depth == 1 and not state.promote_probe
    assert state.promotions == 0 and state.promote_at_cycle == 115
    assert state.ar_trip_pending_cycle == 108
    assert len(state.ar_safety.ring) == 7  # keep observing; no reset by promotion


@pytest.mark.parametrize("adaptive", ["0", "1"])
@pytest.mark.parametrize("ceiling", [2, 3])
@pytest.mark.parametrize("recovers", [False, True])
def test_pending_d1_confirmation_resolves_before_overdue_promotion(monkeypatch, adaptive, ceiling, recovers):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", adaptive)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m, depth=1)
    state.depth_ceiling = state.ladder_depth = ceiling
    state.ar_tier = m.NativeMTPArTier(depth=ceiling)
    _ar_steps(state.ar_tier, 8, ms=10.0)
    state.stats.cycles = 117
    tokens_per_cycle = 3 if recovers else 1
    state.stats.accepted_tokens = 117 * (tokens_per_cycle - 1)
    state.ar_trip_pending_cycle = 108
    state.promote_at_cycle = 115
    state.ar_safety.ring = [
        (c, c * tokens_per_cycle, 100.0 - (117 - c) * .020) for c in range(109, 117)
    ]
    monkeypatch.setattr(m.time, "perf_counter", lambda: 100.0)

    assert m._native_mtp_maybe_ar_safety_fallback("resolve-confirmation", state) is (not recovers)
    assert state.ar_trip_pending_cycle == 0
    assert state.depth == 1 and not state.promote_probe and state.promotions == 0
    assert state.ar_fallback_pending is (not recovers)
    if recovers:
        # No permanent promotion suppression: after a winning confirmation,
        # the next decision may spend the still-pending scheduled probe.
        state.stats.cycles = 118
        state.stats.accepted_tokens = 118 * (tokens_per_cycle - 1)
        monkeypatch.setattr(m.time, "perf_counter", lambda: 100.020)
        assert m._native_mtp_maybe_ar_safety_fallback("after-recovery", state) is False
        assert state.promote_probe and state.depth == 2 and state.promotions == 1


def test_promotion_probe_wins_and_loses(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    # D1 is faster than measured AR before probing the configured depth.
    state.ar_tier = m.NativeMTPArTier(depth=3)
    _ar_steps(state.ar_tier, 8, ms=12.0)
    state.depth = 1
    state.ladder_depth = 3
    state.promote_at_cycle = 40
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40
    state.stats.accepted_tokens = 40
    # D1 has been running at 2 tok/cycle, 20ms/cycle = 10 ms/tok.
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.promote_probe is True and state.depth == 2
    assert abs(state.d1_ms_per_tok - 10.0) < 1e-6
    # D2 at 2.5 tok/cycle, 20ms/cycle = 8 ms/tok < 10/1.1.
    state.stats.cycles = 60
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(51 + i, int(2.5 * (51 + i)), base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.promote_probe is False and state.depth == 2
    assert state.promote_at_cycle > 60  # D3 is a separate later experiment.
    # Later D2 loses to AR -> adjacent D1 with backoff.
    state.stats.cycles = 100
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(91 + i, 91 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 1 and state.ar_fallback_pending is False
    assert state.promote_at_cycle > 100


@pytest.mark.parametrize("ceiling", [1, 2, 3])
def test_recovery_climbs_adjacent_rungs_and_stops_at_selected_ceiling(monkeypatch, ceiling):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    monkeypatch.setattr(m, "_native_mtp_depth_probe_enabled", lambda: False)
    # Decision math is covered separately; here pin scheduling and rung ownership.
    monkeypatch.setattr(m, "ar_safety_step", lambda *a, **k: None)
    monkeypatch.setattr(m, "_native_mtp_recent_ms_per_tok", lambda st: 8.0)
    state = _vlm_state(m, depth=1)
    state.depth_ceiling = state.ladder_depth = ceiling
    state.ar_step_ms = 12.0
    state.promote_at_cycle = 32
    seen = [1]
    for origin in range(1, ceiling):
        state.stats.cycles = state.promote_at_cycle
        assert not m._native_mtp_maybe_ar_safety_fallback("climb", state)
        assert state.promote_probe and state.promote_from_depth == origin
        assert state.depth == origin + 1
        seen.append(state.depth)
        state.stats.cycles += 16
        state.ar_safety.ring = [(i, i * 2, i * .016) for i in range(9)]
        assert not m._native_mtp_maybe_ar_safety_fallback("keep", state)
        assert not state.promote_probe
    state.stats.cycles += 100
    assert not m._native_mtp_maybe_ar_safety_fallback("ceiling", state)
    assert state.depth == ceiling and not state.promote_probe
    assert seen == list(range(1, ceiling + 1))


def test_failed_d2_to_d3_promotion_returns_to_d2(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    state = _vlm_state(m, depth=3)
    state.ladder_depth = 3
    state.promote_probe = True
    state.promote_from_depth = 2
    state.d1_ms_per_tok = 10.0  # historical field name: measured origin cost
    state.stats.cycles = 40
    now = time.perf_counter() - 1
    state.ar_safety.ring = [(31+i, 31+i, now+i*.020) for i in range(9)]
    assert not m._native_mtp_maybe_ar_safety_fallback("reject-promotion", state)
    assert state.depth == 2 and not state.promote_probe
    assert "windowed AR safety D3 -> D2" in caplog.text
    assert "min(D2, AR)" in caplog.text
    assert "D3 -> AR" not in caplog.text


@pytest.mark.parametrize("mode", ["transition", "probe-completed", "pending", "new-window"])
def test_second_controller_cannot_override_ladder_decision(monkeypatch, mode):
    from vmlx_engine import mllm_batch_generator as m

    state = _vlm_state(m, depth=3)
    state.stats.cycles = 40
    state.promote_probe = mode == "probe-completed"
    def safety(_id, st):
        if mode == "transition":
            st.depth = 2
        elif mode == "probe-completed":
            st.promote_probe = False
        elif mode == "pending":
            st.ar_trip_pending_cycle = 39
        else:
            st.ar_safety.reset(39)
        return False
    monkeypatch.setattr(m, "_native_mtp_maybe_ar_safety_fallback", safety)
    def unexpected(*args):
        pytest.fail("another controller overwrote the ladder's measurement phase")
    monkeypatch.setattr(m, "_native_mtp_maybe_adapt_depth", unexpected)
    m._native_mtp_step_depth_controllers("one-owner", state)


@pytest.mark.parametrize("depth", [2, 3])
def test_collapsed_acceptance_drops_only_one_rung(monkeypatch, depth):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "1")
    monkeypatch.setenv("VMLX_NATIVE_MTP_RUNTIME_COST_GATE", "0")
    monkeypatch.setenv("VMLX_NATIVE_MTP_COST_FALLBACK", "0")
    state = _vlm_state(m, depth=depth)
    state.stats.cycles = 200
    state.stats.drafted_by_depth = [200, 200, 200]
    state.stats.accepted_by_depth = [1, 0, 0]
    m._native_mtp_maybe_adapt_depth("collapse", state)
    assert state.depth == depth - 1 and not state.ar_fallback_pending


def test_probe_early_abort_on_clear_loser(monkeypatch):
    from vmlx_engine.native_mtp_ar_safety import ArSafetyState, ar_safety_step

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    st = ArSafetyState(prompt_tokens=50)
    st.reset(100)
    t = 0.0; emitted = 0; tripped = None
    for cyc in range(101, 140):
        t += 0.040; emitted += 1  # 40 ms/tok vs AR 10: a clear loser
        trip = ar_safety_step(st, cycles=cyc, emitted=emitted, now=t,
                              seed_ar_ms=10.0, primed=False, margin=1 / 1.10, probe=True)
        if trip is not None:
            tripped = cyc; break
    # warmup 4 + 5 ring samples -> aborted by cycle ~110, not after 24 cycles
    assert tripped is not None and tripped <= 112


def test_reentry_spacing_is_geometric_and_uncapped_in_count():
    from vmlx_engine import mllm_batch_generator as m

    tier = m.NativeMTPArTier(depth=3)
    t = 100.0
    for _ in range(20):
        tier.record_step(t); t += 0.025
    assert tier.probe_due() is True
    tier.probes = 40  # many probes later a probe is still due once the interval passed
    assert tier.probe_due() is True
    for _ in range(12):
        tier.probe_failed()
    assert tier.backoff == 12 and tier.next_probe_tokens == 16 << m._NATIVE_MTP_REENTRY_MAX_BACKOFF


# ---- seed-uncertainty margin + sticky-start evidence (2026-09-05) ----------

def _drive_uniform(st, *, ms_per_tok, cycles=40, seed=10.0, **kw):
    """Feed uniform cycles (1 token each) at ms_per_tok; return the first trip."""
    t = 100.0
    for c in range(1, cycles + 1):
        t += ms_per_tok / 1000.0
        trip = ar_safety_step(st, cycles=c, emitted=c, now=t, seed_ar_ms=seed, **kw)
        if trip is not None:
            return trip
    return None


def test_seed_baseline_uses_seed_margin_measured_uses_exact(monkeypatch):
    monkeypatch.delenv("VMLX_NATIVE_MTP_SEED_COST_MARGIN", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN", raising=False)
    # 5% slower than the one-step seed: inside the seed's error band -> hold.
    assert _drive_uniform(ArSafetyState(), ms_per_tok=10.5) is None
    # Same cost against a MEASURED baseline: a real loss -> trip.
    trip = _drive_uniform(ArSafetyState(), ms_per_tok=10.5, baseline_measured=True)
    assert trip is not None and abs(trip.margin - 1.0) < 1e-9
    # 15% slower than the seed is outside the band -> trip with the seed margin.
    trip = _drive_uniform(ArSafetyState(), ms_per_tok=11.5)
    assert trip is not None and abs(trip.margin - 1.10) < 1e-9


def test_seed_margin_never_below_runtime_margin(monkeypatch):
    from vmlx_engine.native_mtp_ar_safety import ar_safety_seed_margin

    monkeypatch.setenv("VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN", "1.25")
    monkeypatch.setenv("VMLX_NATIVE_MTP_SEED_COST_MARGIN", "1.05")
    assert abs(ar_safety_seed_margin() - 1.25) < 1e-9


def test_sticky_start_evidence_rules():
    from vmlx_engine.mllm_batch_generator import (
        _native_mtp_first_promotion_cycle,
        _native_mtp_should_record_last_tier,
        _NATIVE_MTP_PROMOTE_FIRST_CYCLES,
    )

    # first request after load never seeds the next start rung
    assert not _native_mtp_should_record_last_tier(0, 200, 8)
    # too few judged cycles never seeds it
    assert not _native_mtp_should_record_last_tier(3, 15, 8)
    assert _native_mtp_should_record_last_tier(1, 16, 8)
    # a sticky D1 start is re-examined after one window, own demotion waits
    assert _native_mtp_first_promotion_cycle(True) == 8
    assert _native_mtp_first_promotion_cycle(False) == _NATIVE_MTP_PROMOTE_FIRST_CYCLES


def test_depth_probe_keeps_adjacent_rung_when_it_beats_configured_depth(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth = 3
    state.ladder_depth = 3
    state.depth_probe_at_cycle = 16
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20
    state.stats.accepted_tokens = 40
    # D3 window: 3 tok/cycle at 30 ms/cycle = 10 ms/tok (AR seed 12 -> healthy)
    state.ar_step_ms = 12.0
    state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is True and state.depth == 2 and state.depth_probes == 1
    assert abs(state.dcfg_ms_per_tok - 10.0) < 1e-6
    # D1 window: 1.9 tok/cycle at 15 ms/cycle = 7.9 ms/tok < 10/1.1 -> keep D1
    state.stats.cycles = 40
    state.ar_safety.anchor_cycle_ms = 15.0
    state.ar_safety.ring = [(31 + i, int(1.9 * (31 + i)), base_t + i * 0.015) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is False and state.depth == 2
    assert state.promote_at_cycle > 40 and state.d1_ms_per_tok > 0


def test_depth_probe_returns_to_origin_when_adjacent_rung_loses(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    state = _vlm_state(m)
    state.depth = 3
    state.ladder_depth = 3
    state.depth_probe_at_cycle = 16
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20
    state.stats.accepted_tokens = 40
    state.ar_step_ms = 12.0
    state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 2 and state.depth_probe is True
    # D1 window at 9.5 ms/tok: not 10% better than D3's 10 -> probe lost
    state.stats.cycles = 40
    state.ar_safety.anchor_cycle_ms = 19.0
    state.ar_safety.ring = [(31 + i, 2 * (31 + i), base_t + i * 0.019) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probe is False and state.depth == 3
    assert state.depth_probe_at_cycle > 40 and state.depth_probe_backoff == 1
    # budget: a second probe is allowed, a third is not
    state.stats.cycles = state.depth_probe_at_cycle + 1
    state.ar_safety.ring = [(state.stats.cycles - 9 + i, 3 * (state.stats.cycles - 9 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth_probes == 2 and state.depth == 2
    state.depth_probe = False; state.depth = 3; state.depth_probe_at_cycle = state.stats.cycles
    state.ar_safety.ring = [(state.stats.cycles - 9 + i, 3 * (state.stats.cycles - 9 + i), base_t + i * 0.030) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 3 and state.depth_probes == 2


def test_depth_probe_follows_policy_and_explicit_switch(monkeypatch):
    """Contract: a FIXED policy request that beats AR at its configured depth
    never tries D1 (only the AR-safety valve may step it down); an ADAPTIVE
    request tries D1 once; VMLX_NATIVE_MTP_DEPTH_PROBE overrides either way."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.delenv("VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH", raising=False)
    for policy_env, probe_env, expect_depth in (
        ("0", None, 3),   # fixed: no probe
        ("1", None, 2),   # adaptive: adjacent probe
        ("0", "1", 2),    # fixed + explicit probe on
        ("1", "0", 3),    # adaptive + explicit probe off
    ):
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", policy_env)
        if probe_env is None:
            monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH_PROBE", raising=False)
        else:
            monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH_PROBE", probe_env)
        state = _vlm_state(m)
        state.depth = 3
        state.ladder_depth = 3
        state.depth_probe_at_cycle = 16
        state.ar_step_ms = 12.0
        base_t = time.perf_counter() - 1.0
        state.stats.cycles = 20
        state.stats.accepted_tokens = 40
        state.ar_safety.anchor_cycle_ms = 30.0  # steady: ring cycles match the anchor (no drift)
        state.ar_safety.ring = [(11 + i, 3 * (11 + i), base_t + i * 0.030) for i in range(9)]
        assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
        assert state.depth == expect_depth, f"policy={policy_env} probe_env={probe_env}"
    monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH_PROBE", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", raising=False)


def test_fixed_policy_still_steps_down_only_through_ar_safety(monkeypatch):
    """Under fixed, a window slower than plain decoding still trips the valve
    (D3 -> D1 with the logged reason); a healthy window never moves."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    state = _vlm_state(m)
    state.depth = 3; state.ladder_depth = 3; state.depth_probe_at_cycle = 16
    state.ar_step_ms = 10.0
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 20; state.stats.accepted_tokens = 20
    # 1 tok/cycle at 20 ms/cycle = 20 ms/tok vs AR 10 -> valve trips D3 -> D1
    state.ar_safety.ring = [(11 + i, 11 + i, base_t + i * 0.020) for i in range(9)]
    state.ar_safety.anchor_cycle_ms = 20.0
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 2 and state.depth_probe is False
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_finish_line_fields_policy_configured_and_throughput(caplog):
    import logging
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats, _native_mtp_log_stats

    stats = MLLMNativeMTPStats()
    stats.cycles = 10; stats.accepted_tokens = 12; stats.drafted_tokens = 30
    stats.init_emits = 2; stats.draft_emits = 12; stats.bonus_emits = 4; stats.verify_emits = 6
    stats.cycles_by_depth[3] = 7; stats.cycles_by_depth[1] = 3
    stats.configured_depth = 3; stats.depth_policy = "fixed"; stats.span_seconds = 0.5
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        _native_mtp_log_stats("req", stats, "length", None)
    line = next(r.getMessage() for r in caplog.records if "finish=length" in r.getMessage())
    assert "cycles_by_depth[d1=3,d3=7]" in line
    assert "policy=fixed configured=D3" in line
    assert "confirmed_tok_s=48.0 span_s=0.50" in line


def test_finish_line_reports_cycles_by_depth():
    from vmlx_engine.mllm_batch_generator import MLLMNativeMTPStats

    stats = MLLMNativeMTPStats()
    for depth in (3, 3, 1, 3, 1, 1, 1):
        stats.cycles_by_depth[depth] += 1
    assert stats.cycles_by_depth[3] == 3 and stats.cycles_by_depth[1] == 4
    occupancy = ",".join(f"d{d}={n}" for d, n in enumerate(stats.cycles_by_depth) if d > 0 and n > 0)
    assert occupancy == "d1=4,d3=3"


# ---- bounded AR calibration (context-matched baseline refresh) -----------------

def _running_state(m, *, depth=3, anchor_ms=30.0, cycle_ms=30.0, cycles=40, emitted_per_cycle=3):
    state = _vlm_state(m)
    state.depth = depth; state.ladder_depth = depth
    state.ar_step_ms = 12.0
    state.stats.cycles = cycles
    state.stats.accepted_tokens = cycles * (emitted_per_cycle - 1)
    base_t = time.perf_counter() - 1.0
    state.ar_safety.anchor_cycle_ms = anchor_ms
    state.ar_safety.ring = [(cycles - 8 + i, emitted_per_cycle * (cycles - 8 + i), base_t + i * cycle_ms / 1000.0) for i in range(9)]
    # Same-depth calibration ledger: this depth's anchor and its recent walls.
    state.cal_anchor_ms = {depth: anchor_ms}
    state.cal_recent_ms = {depth: [cycle_ms] * 8}
    return state


def test_calibration_triggers_on_cycle_wall_drift_and_reenters_at_same_depth(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    # cycle wall 2x the anchor (drift +100%), MTP still cheap per token (10 ms/tok), >512 tokens since the seed
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6, cycles=100)
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
    assert state.ar_fallback_pending and "ar_calibration" in state.ar_fallback_reason
    assert "drift=+1.00" in state.ar_fallback_reason
    tier = state.ar_tier
    assert tier is not None and tier.calibration and tier.reenter_depth == 3
    assert tier.next_probe_tokens == m._NATIVE_MTP_CALIBRATION_TOKENS
    assert tier.backoff == 0 and state.calibrations == 1
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_unmeasured_seed_gets_early_calibration_at_each_depth(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    for depth in (1, 2, 3):
        state = _running_state(m, depth=depth, cycles=64, emitted_per_cycle=2)
        # Observed6S cold seed83.8ms versus measured AR31ms. A cheap-looking
        # MTP window cannot validate that seed; measure AR instead of guessing.
        state.ar_step_ms = 83.8
        assert m._native_mtp_maybe_ar_safety_fallback("seed-refresh", state)
        assert "seed_refresh=True" in state.ar_fallback_reason
        assert state.ar_tier.calibration
        assert state.ar_tier.reenter_depth == depth
        assert state.ar_tier.next_probe_tokens == m._NATIVE_MTP_CALIBRATION_TOKENS


def test_seed_refresh_does_not_override_measured_baseline_or_short_turn(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    short = _running_state(m, cycles=63, emitted_per_cycle=2)
    short.ar_step_ms = 83.8
    assert not m._native_mtp_maybe_ar_safety_fallback("short", short)
    measured = _running_state(m, cycles=64, emitted_per_cycle=2)
    measured.ar_step_ms = 83.8
    measured.ar_tier = m.NativeMTPArTier(depth=3)
    _ar_steps(measured.ar_tier, 8, ms=31.0)
    assert not m._native_mtp_maybe_ar_safety_fallback("measured", measured)
    assert not measured.ar_fallback_pending
    off = _running_state(m, cycles=64, emitted_per_cycle=2)
    off.ar_step_ms = 83.8
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_CALIBRATION", "0")
    assert not m._native_mtp_maybe_ar_safety_fallback("disabled", off)


def test_calibration_triggers_on_stale_interval_not_on_steady_short_runs(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    steady = _running_state(m, anchor_ms=30.0, cycle_ms=30.0, cycles=40)
    assert m._native_mtp_maybe_ar_safety_fallback("req", steady) is False
    assert not steady.ar_fallback_pending and steady.calibrations == 0
    stale = _running_state(m, anchor_ms=30.0, cycle_ms=30.0, cycles=400, emitted_per_cycle=3)  # ~1,200 emitted
    assert m._native_mtp_maybe_ar_safety_fallback("req", stale) is True
    assert "stale=True" in stale.ar_fallback_reason and stale.calibrations == 1
    monkeypatch.setenv("VMLX_NATIVE_MTP_AR_CALIBRATION", "0")
    off = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6)
    assert m._native_mtp_maybe_ar_safety_fallback("req", off) is False
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_CALIBRATION")
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_is_rate_limited_by_spacing_not_capped_by_count(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    # drift alone within 512 tokens of the last AR measurement: refused
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6, cycles=40)
    state.calibrations = 9; state.last_ar_measure_emitted = 240 - 100
    m._native_mtp_maybe_ar_safety_fallback("req", state)
    assert "ar_calibration" not in (state.ar_fallback_reason or "")
    # same drift after 512 tokens: allowed no matter how many calibrations ran before
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6, cycles=200)
    state.calibrations = 9; state.last_ar_measure_emitted = 1200 - 600
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
    assert "ar_calibration" in state.ar_fallback_reason and state.calibrations == 10
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_jump_shortens_the_next_recheck(monkeypatch):
    """A measurement 1.5x above the previous one (a stall-time reading) is
    re-checked after 256 tokens instead of 768; a normal one keeps 768."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    gen = _FakeGen(m)
    tier = m.NativeMTPArTier(depth=3, calibration=True, reenter_depth=3, calibrations=1)
    _ar_steps(tier, 8, ms=30.0)
    ok, st = _reseed(m, gen, tier)
    assert ok and st.calibration_interval_tokens == m._NATIVE_MTP_CALIBRATION_INTERVAL_TOKENS
    assert abs(tier.prev_measured_ar_ms - 30.0) < 1e-6
    tier.calibration = True; tier.step_walls_ms = []; tier.last_step_t = 0.0
    _ar_steps(tier, 8, ms=90.0)  # 3x jump
    ok, st2 = _reseed(m, gen, tier)
    assert ok and st2.calibration_interval_tokens == m._NATIVE_MTP_CALIBRATION_RECHECK_TOKENS
    tier.calibration = True; tier.step_walls_ms = []; tier.last_step_t = 0.0
    _ar_steps(tier, 8, ms=36.0)  # back to normal: not a jump above 90 -> full interval again
    ok, st3 = _reseed(m, gen, tier)
    assert ok and st3.calibration_interval_tokens == m._NATIVE_MTP_CALIBRATION_INTERVAL_TOKENS
    # stale trigger honours the shortened interval
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    state = _running_state(m, anchor_ms=30.0, cycle_ms=30.0, emitted_per_cycle=3, cycles=200)  # 600 emitted, no drift
    state.calibration_interval_tokens = m._NATIVE_MTP_CALIBRATION_RECHECK_TOKENS
    state.last_ar_measure_emitted = 600 - 300
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True and "stale=True" in state.ar_fallback_reason
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_tier_records_ar_wall_and_calibration_counters():
    from vmlx_engine.mllm_batch_generator import NativeMTPArTier

    t = NativeMTPArTier(depth=3)
    now = 100.0
    for _ in range(6):
        t.record_step(now); now += 0.025
    assert t.total_ar_tokens == 6 and abs(t.total_ar_ms - 125.0) < 1e-6
    assert abs(t.measured_ar_ms_per_tok() - 25.0) < 1e-6


# ---- calibration lifecycle (model-free): budgets, return, next-request policy ----------
from types import SimpleNamespace


class _FakeGen:
    """Stand-in for the generator in ``_reseed_native_mtp_probe``: seeds a fresh
    state at the requested depth and records the depth the seed was asked for."""

    def __init__(self, m):
        self.m = m
        self.seeds = []

    def _seed_native_mtp_from_prefill(self, req, cache, y, logprobs, start_depth_override=None):
        st = _vlm_state(self.m, depth=int(start_depth_override or 3))
        st.depth_ceiling = 3
        self.seeds.append(start_depth_override)
        req._native_mtp_state = st
        return True


def _reseed(m, gen, tier):
    req = SimpleNamespace(request_id="r", num_tokens=100, _native_mtp_ar_tier=tier)
    batch = SimpleNamespace(requests=[req], cache=[], y=None, logprobs=None)
    ok = m.MLLMBatchGenerator._reseed_native_mtp_probe(gen, batch, tier)
    return ok, getattr(req, "_native_mtp_state", None)


def _ar_steps(tier, n, ms=25.0):
    now = 1000.0
    for _ in range(n):
        tier.record_step(now); now += ms / 1000.0


def test_loss_probe_depth_always_reenters_adjacent_d1():
    from vmlx_engine.mllm_batch_generator import NativeMTPArTier

    t = NativeMTPArTier(depth=3)
    seen = []
    for k in range(4):
        t.probes = k
        seen.append(t.probe_depth(3))
    assert seen == [1, 1, 1, 1]
    assert NativeMTPArTier(depth=1).probe_depth(1) == 1


def test_deep_backoff_spaces_probes_but_never_delays_a_calibration_return(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    loss = m.NativeMTPArTier(depth=3)
    loss.backoff = 9; loss._restart_ar_window()  # deep backoff: 4,096 AR tokens between probes
    _ar_steps(loss, 40)
    assert loss.probe_due() is False and loss.next_probe_tokens == 4096
    cal = m.NativeMTPArTier(depth=3, calibration=True, reenter_depth=3)
    cal.probes = 9
    cal.next_probe_tokens = m._NATIVE_MTP_CALIBRATION_TOKENS
    _ar_steps(cal, m._NATIVE_MTP_CALIBRATION_TOKENS)
    assert cal.probe_due() is True  # a calibration always returns


def test_calibration_return_resumes_at_left_depth_without_spending_the_budget(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    gen = _FakeGen(m)
    tier = m.NativeMTPArTier(depth=3, calibration=True, reenter_depth=3, calibrations=1)
    tier.probes = 4; tier.backoff = 3
    _ar_steps(tier, 8)
    ok, state = _reseed(m, gen, tier)
    assert ok and state is not None
    assert gen.seeds == [3]  # the FIRST draft chain is already the return width
    assert state.probe is False and state.depth == 3  # resumed run, not a probe
    assert tier.probes == 4 and tier.backoff == 3 and tier.calibration is False  # nothing spent
    assert state.ar_tier is tier and state.epoch == 6  # 4 probes + 1 calibration + 1
    assert state.calibrations == 1


def test_loss_reentry_repeats_d1_and_counts_each_probe(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    gen = _FakeGen(m)
    tier = m.NativeMTPArTier(depth=3)
    _ar_steps(tier, 16)
    ok, state = _reseed(m, gen, tier)
    assert ok and state.probe is True and state.depth == 1 and tier.probes == 1
    tier2 = m.NativeMTPArTier(depth=3); tier2.probes = 1
    _ar_steps(tier2, 32)
    ok, state2 = _reseed(m, gen, tier2)
    assert ok and state2.probe is True and state2.depth == 1 and tier2.probes == 2
    assert gen.seeds == [1, 1]


def test_calibration_preserves_ladder_schedule_and_attempts(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: True)
    tier = m.NativeMTPArTier(depth=3)
    _ar_steps(tier, 8)
    state = _calibrated_running_state(m, tier, tok_per_cycle=1, cycles=900)
    state.depth = 1
    state.promote_backoff = 4
    state.promotions = 2
    state.promote_at_cycle = 1300
    state.depth_probe_backoff = 3
    state.depth_probes = 1
    state.depth_probe_at_cycle = 1500
    state.last_ar_measure_emitted = 0
    assert m._native_mtp_maybe_ar_safety_fallback("cal-schedule", state)
    assert tier.calibration
    _ar_steps(tier, 8)
    ok, resumed = _reseed(m, _FakeGen(m), tier)
    assert ok and resumed.depth == 1 and not resumed.probe
    assert resumed.promote_backoff == 4
    assert resumed.promotions == 2
    assert resumed.promote_at_cycle - resumed.stats.cycles == 400
    assert resumed.depth_probe_backoff == 3
    assert resumed.depth_probes == 1
    assert resumed.depth_probe_at_cycle - resumed.stats.cycles == 600
    assert tier.calibration_schedule is None
    # Scheduling survives, but cost samples belong to the new measured phase.
    assert resumed.ar_safety.ring == []
    assert resumed.d1_ms_per_tok == 0


def test_calibration_schedule_preserves_exhaustion_and_inactive_deadlines():
    from vmlx_engine import mllm_batch_generator as m

    prior = _vlm_state(m, depth=1)
    prior.stats.cycles = 100
    prior.promotions = 12  # diagnostic count, no longer an exhausted budget
    prior.promote_backoff = 5
    prior.promote_at_cycle = 0
    prior.depth_probes = m._NATIVE_MTP_MAX_DEPTH_PROBES
    prior.depth_probe_at_cycle = 99
    schedule = m.NativeMTPCalibrationSchedule.capture(prior)
    resumed = _vlm_state(m, depth=1)
    resumed.stats.cycles = 7
    schedule.restore(resumed)
    assert resumed.promotions == 12
    assert resumed.promote_backoff == 5
    assert resumed.promote_at_cycle == 0
    assert resumed.depth_probes == m._NATIVE_MTP_MAX_DEPTH_PROBES
    # A due deadline remains due on the next verify cycle, not zero/disabled.
    assert resumed.depth_probe_at_cycle == 8


def _calibrated_running_state(m, tier, *, tok_per_cycle, cycle_ms=45.0, cycles=40):
    """A resumed (calibration-returned) D3 run with a full judged window."""
    state = _vlm_state(m, depth=3)
    state.depth_ceiling = 3; state.ladder_depth = 3
    state.ar_tier = tier
    state.stats.cycles = cycles
    state.stats.accepted_tokens = cycles * (tok_per_cycle - 1)
    base_t = time.perf_counter() - 1.0
    state.ar_safety.anchor_cycle_ms = cycle_ms
    state.ar_safety.ring = [(cycles - 8 + i, tok_per_cycle * (cycles - 8 + i), base_t + i * cycle_ms / 1000.0) for i in range(9)]
    state.cal_anchor_ms = {3: cycle_ms}; state.cal_recent_ms = {3: [cycle_ms] * 8}
    state.last_ar_measure_emitted = cycles * tok_per_cycle - 20
    return state


def test_calibration_return_that_wins_keeps_depth_and_counters(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    tier = m.NativeMTPArTier(depth=3, calibrations=1)
    _ar_steps(tier, 8, ms=25.0)  # measured AR 25 ms/tok
    state = _calibrated_running_state(m, tier, tok_per_cycle=3, cycle_ms=45.0)  # 15 ms/tok
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 3 and not state.ar_fallback_pending
    assert tier.fallbacks == 1 and tier.probes == 0 and tier.backoff == 0
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_return_that_loses_steps_down_the_ladder_not_to_ar(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    tier = m.NativeMTPArTier(depth=3, calibrations=1)
    _ar_steps(tier, 8, ms=25.0)
    state = _calibrated_running_state(m, tier, tok_per_cycle=1, cycle_ms=45.0)  # 45 ms/tok, losing
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.depth == 2 and not state.ar_fallback_pending  # adjacent rung, margin 1.0
    assert tier.fallbacks == 1 and tier.backoff == 0
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_calibration_lifecycle_three_and_four_returns_then_budget_closes(monkeypatch):
    """Trigger -> 8 AR steps -> return at the same depth, four times; the
    fifth trigger is refused and the ordinary valve judges the window."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_REENTRY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    gen = _FakeGen(m)
    tier = None
    for n in range(1, 5):
        state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=6, cycles=100 * n)
        state.ar_tier = tier; state.calibrations = n - 1
        assert m._native_mtp_maybe_ar_safety_fallback("req", state) is True
        assert "ar_calibration" in state.ar_fallback_reason
        tier = state.ar_tier
        assert tier.calibration and tier.calibrations == n and tier.reenter_depth == 3
        _ar_steps(tier, m._NATIVE_MTP_CALIBRATION_TOKENS)
        assert tier.probe_due()
        ok, resumed = _reseed(m, gen, tier)
        assert ok and resumed.probe is False and resumed.depth == 3
        assert resumed.calibrations == n and tier.probes == 0 and tier.fallbacks == 1
    assert gen.seeds == [3, 3, 3, 3]
    # fifth, within 512 tokens of the last AR measurement: a losing window is judged by the valve (D3 -> D1), not calibrated
    state = _running_state(m, anchor_ms=30.0, cycle_ms=60.0, emitted_per_cycle=1, cycles=400)
    state.ar_tier = tier; state.calibrations = 4; state.last_ar_measure_emitted = 400 - 100
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert "ar_calibration" not in (state.ar_fallback_reason or "") and state.depth == 2
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")


def test_finish_during_calibration_is_not_an_ar_ending():
    from vmlx_engine import mllm_batch_generator as m

    cal = m.NativeMTPArTier(depth=3, calibration=True, reenter_depth=3)
    loss = m.NativeMTPArTier(depth=3)
    assert m._native_mtp_finish_tier_label(None, cal) == "D3(calibrating)"
    assert m._native_mtp_finish_tier_label(None, loss) == "AR"
    st = _vlm_state(m, depth=2)
    assert m._native_mtp_finish_tier_label(st, cal) == "D2"
    # the sticky start rung reads the label before "(": D3 never starts the next request at D1
    assert m._native_mtp_finish_tier_label(None, cal).split("(")[0] not in ("AR", "D1")
    st.ar_tier = cal
    assert m._native_mtp_handoff_is_calibration(st) is True
    st.ar_tier = loss
    assert m._native_mtp_handoff_is_calibration(st) is False


def test_profile_ignores_a_calibration_handoff():
    from vmlx_engine.native_mtp_profile import NativeMTPProfileStore

    store = NativeMTPProfileStore()
    key = ("qwen4_exp", False, "text", False)
    store.observe(key, final_depth=3, fallback_to_ar=False, fallback_reason="x",
                  finish_reason="length", values_tok_s={"d3": 50.0}, sample_counts={"d3": 100},
                  ar_baseline_tps=35.0)
    before = store.snapshot()
    learned = next(iter(before.values()))["learned_depth"]
    assert learned == 3
    store.observe(key, final_depth=3, fallback_to_ar=False, fallback_reason="ar_calibration drift=+0.30",
                  finish_reason="ar_calibration", values_tok_s={}, sample_counts={}, ar_baseline_tps=35.0)
    after = next(iter(store.snapshot().values()))
    assert after["learned_depth"] == 3 and after["last_finish_reason"] == "length"


def test_calibration_drift_is_same_depth_and_consecutive_only(monkeypatch):
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY_WINDOW", raising=False)
    st = _vlm_state(m, depth=3)
    st.ar_safety.anchor_cycle_ms = 40.0  # valve warmup done
    t = 100.0
    # first full window at 45 ms (post-prefill), second at 30 ms -> anchor = second window
    for c in range(1, 10):
        d = m._native_mtp_calibration_drift(st, 3, t, c); t += 0.045
    assert 3 not in st.cal_anchor_ms
    for c in range(10, 19):
        d = m._native_mtp_calibration_drift(st, 3, t, c); t += 0.030
    assert abs(st.cal_anchor_ms.get(3, 0.0) - 30.0) < 1e-6 and abs(d) < 1e-6
    # interleaved D1 cycles (confidence gate / ladder) are NOT depth-3 evidence
    for c in range(19, 39):
        depth = 1 if c % 2 else 3
        d = m._native_mtp_calibration_drift(st, depth, t, c); t += (0.030 if depth == 3 else 0.020)
    assert abs(d) < 0.01  # no depth-3 wall is recorded across a depth change
    assert abs(m._native_mtp_calibration_drift(st, 3, t, 39)) < 0.01
    # a gap in cycles (a probe ran in between) never counts as one wall
    t += 5.0
    d = m._native_mtp_calibration_drift(st, 3, t, 50); t += 0.030
    assert abs(d) < 0.01 and max(st.cal_recent_ms[3]) < 100.0
    # 9 consecutive D3 cycles at 60 ms -> drift +1.0 for depth 3
    for c in range(51, 60):
        d = m._native_mtp_calibration_drift(st, 3, t, c); t += 0.060
    assert abs(d - 1.0) < 0.05


def test_d1_stale_baseline_single_losing_window_recovers_without_fallback(monkeypatch):
    """One losing D1 window sets a pending mark; a winning window clears it;
    a losing window long after the mark starts a new confirmation instead
    of falling back."""
    from vmlx_engine import mllm_batch_generator as m

    monkeypatch.delenv("VMLX_NATIVE_MTP_AR_SAFETY", raising=False)
    monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
    state = _vlm_state(m, depth=1); state.depth_ceiling = 3; state.ladder_depth = 3
    # A stale measurement still requires confirmation; fresh AR is covered
    # by test_native_mtp_fast_loss and the settled re-entry test above.
    monkeypatch.setattr(m, "_native_mtp_calibration_enabled", lambda: False)
    state.last_ar_measure_emitted = -512
    state.ar_tier = m.NativeMTPArTier(depth=3)
    _ar_steps(state.ar_tier, 8, ms=18.0)
    base_t = time.perf_counter() - 1.0
    state.stats.cycles = 40; state.stats.accepted_tokens = 0
    state.ar_safety.ring = [(31 + i, 31 + i, base_t + i * 0.020) for i in range(9)]  # marginal: 20 ms/tok vs AR 18
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_trip_pending_cycle == 40 and not state.ar_fallback_pending
    # winning window (3 tok/cycle at 20 ms = 6.7 ms/tok): pending clears
    state.stats.cycles = 60; state.stats.accepted_tokens = 120
    state.ar_safety.anchor_cycle_ms = 20.0
    state.ar_safety.ring = [(51 + i, 3 * (51 + i), base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_trip_pending_cycle == 0
    # a losing window much later starts a NEW confirmation (no fallback)
    state.stats.cycles = 200; state.stats.accepted_tokens = 120
    state.ar_safety.ring = [(191 + i, 120 + 191 + i, base_t + i * 0.020) for i in range(9)]
    assert m._native_mtp_maybe_ar_safety_fallback("req", state) is False
    assert state.ar_trip_pending_cycle == 200 and not state.ar_fallback_pending
    monkeypatch.delenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH")

"""DFlash2 verify width chooser: expected tokens (from per-position acceptance) per measured second."""
import random

from vmlx_engine.dflash2_runtime import _BlockChooser, _dflash2_block_plan


def _run(p, cost5, cost8, cycles=200, seed=0):
    rng, c, seq = random.Random(seed), _BlockChooser(5, 8), []
    for _ in range(cycles):
        w = c.width
        seq.append(w)
        accepted = 0
        while accepted < w - 1 and rng.random() < p:
            accepted += 1
        c.observe(w, accepted + 1, cost5 if w == 5 else cost8)
    return seq


def test_expensive_wide_verify_on_prose_stays_at_five():   # JANG_4D: 8 rows cost 1.49x
    seq = _run(0.55, 0.053, 0.079)
    assert seq.count(5) > 180 and 8 in seq                 # still re-times 8


def test_cheap_wide_verify_on_easy_text_goes_to_eight():   # JANGH2: 8 rows cost 1.13x
    seq = _run(0.97, 0.078, 0.088)
    assert seq.count(8) > 180 and 5 in seq


def test_width_five_cycles_inform_width_eight():
    c = _BlockChooser(5, 8)
    for _ in range(20):
        c.observe(5, 5, 0.05)                              # every block fully accepted
    assert c.expected_tokens(8) > 6.5


def test_clipped_final_block_is_ignored():
    c = _BlockChooser(5, 8)
    c.observe(3, 3, 0.04)
    assert c.cycles == 0 and c.cost == {}


def test_fixed_width_env(monkeypatch):
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "8")
    assert _dflash2_block_plan(8) == (8,)
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "32")
    assert _dflash2_block_plan(8) == (16,)                 # capped at 2x trained
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "auto")
    assert _dflash2_block_plan(8) == (5, 8, 16)
    assert _dflash2_block_plan(4) == (4, 8)
    assert _dflash2_block_plan(8, lane_flat=True) == (8, 16)     # flat verify cost: 5 is dominated


def test_first_cycle_at_a_width_is_warmup_and_costs_persist_across_requests():
    shared: dict = {}
    c = _BlockChooser(5, 8, shared)
    widths = []
    for _ in range(4):                                     # warm 5, time 5, warm 8, time 8
        w = c.width
        widths.append(w)
        c.observe(w, w, 2.0 if w not in shared.get("warm", set()) else (0.05 if w == 5 else 0.06))
    assert widths == [5, 5, 8, 8]
    assert shared[5] == 0.05 and shared[8] == 0.06        # the 2.0 s warm-up cycles were discarded
    c2 = _BlockChooser(5, 8, shared)                       # the next request starts with measured costs
    c2.observe(5, 5, 0.05)
    assert c2.width == 8                                   # 5/5 accepted -> 8 predicted to pay at 1.2x cost


def test_three_widths_pick_sixteen_when_cheap_and_accepted():
    rng, c = random.Random(3), _BlockChooser((5, 8, 16))
    seq = []
    for _ in range(300):
        w = c.width
        seq.append(w)
        accepted = 0
        while accepted < w - 1 and rng.random() < 0.97:
            accepted += 1
        c.observe(w, accepted + 1, {5: 0.060, 8: 0.062, 16: 0.068}[w])
    assert seq.count(16) > 250 and 5 in seq and 8 in seq     # still re-times the others


# --- AR fallback (next_action) -------------------------------------------------------------------------------------
# Measured 2026-10-11, Qwen3.8-27B JANGH2, sampled defaults: AR 29.9 tok/s (33 ms/token) vs DFlash2 25.5-28.5 tok/s;
# a 5/8/16-row cycle ~0.09-0.10 s for ~2.5 tokens. Greedy: ~4.8 tokens per 0.10 s cycle (48 tok/s).

def _seeded(shared, p, cost=None):
    """A chooser whose widths are timed (warm-up + one timed cycle each) at acceptance p."""
    cost = cost or {5: 0.092, 8: 0.098, 16: 0.104}
    c = _BlockChooser((5, 8, 16), cost=shared)
    rng = random.Random(7)
    for _ in range(12):
        w = c.width
        accepted = 0
        while accepted < w - 1 and rng.random() < p:
            accepted += 1
        c.observe(w, accepted + 1, cost[w])
    return c


def _ar_burst(c, per_token=0.033, n=8):
    c.observe_ar(per_token * n, n)


def test_first_action_of_every_request_is_a_verify_cycle():
    shared = {}
    c = _seeded(shared, 0.25)
    _ar_burst(c); _ar_burst(c)
    c2 = _BlockChooser((5, 8, 16), cost=shared)            # next request: everything measured, AR faster
    assert c2.next_action()[0] == "spec"                  # exit checkpoints need the drafter to have run


def test_low_acceptance_sampled_text_falls_back_to_ar_and_keeps_probing():
    shared = {}
    c = _seeded(shared, 0.25)
    assert c.next_action() == ("ar", 8)                   # AR not measured yet: one burst to measure it
    _ar_burst(c)                                           # warm-up burst (1-row kernels compile): not timed
    assert "ar" not in shared
    _ar_burst(c)
    assert abs(shared["ar"] - 0.033) < 1e-9
    actions = []
    for _ in range(12):
        mode, n = c.next_action()
        actions.append(mode)
        if mode == "ar":
            _ar_burst(c, n=n)
        else:
            c.observe(c.width, 2, 0.095, after_ar=True)
    assert actions.count("ar") >= 8 and "spec" in actions  # mostly AR, a verify probe every 24 AR tokens


def test_high_acceptance_greedy_text_stays_speculative():
    shared = {}
    c = _seeded(shared, 0.92)
    _ar_burst(c); _ar_burst(c)
    assert all(c.next_action()[0] == "spec" for _ in range(20))


def test_near_tie_stays_speculative_by_the_margin():
    shared = {}
    c = _seeded(shared, 0.25)
    _ar_burst(c)
    best = max(c.widths, key=lambda w: c.expected_tokens(w) / c.cost[w])
    spec_rate = c.expected_tokens(best) / c.cost[best]
    _ar_burst(c, per_token=1.0 / (spec_rate * 1.03))        # AR only 3 % faster: inside the 5 % hysteresis
    assert c.next_action()[0] == "spec"


def test_cycle_after_an_ar_burst_counts_acceptance_not_time():
    shared = {}
    c = _seeded(shared, 0.5)
    before = dict(shared)
    trials = list(c.trials)
    c.observe(c.width, 3, 5.0, after_ar=True)               # slow: the drafter also ingested the burst's rows
    assert all(shared[w] == before[w] for w in (5, 8, 16))
    assert c.trials != trials


def test_ar_fallback_is_off_for_a_pinned_width_and_by_env(monkeypatch):
    assert _BlockChooser((8,)).allow_ar is False
    monkeypatch.setenv("VMLX_DFLASH2_AR_FALLBACK", "0")
    assert _BlockChooser((5, 8, 16)).allow_ar is False


def test_exit_checkpoint_rolls_back_a_step_launched_past_the_last_token():
    from vmlx_engine.dflash2_runtime import _checkpoint_correction
    assert _checkpoint_correction(1, 0) == (-1, 1)         # one 1-row forward past confirmed[:-1]
    assert _checkpoint_correction(0, 0) is None             # a burst that ended on time is already exact


def test_gdn_rollback_of_a_whole_one_row_forward_restores_its_start_state():
    import mlx.core as mx

    from vmlx_engine.dflash2_runtime import _VLMGDNStateCapture

    class Attn:
        def __init__(self):
            self.trimmed = 0
        def is_trimmable(self):
            return True
        def trim(self, n):
            self.trimmed += n

    class GDN(list):
        def is_trimmable(self):
            return False

    initial = mx.ones((1, 2, 3, 3))
    conv_input = mx.arange(4 * 5).reshape(1, 4, 5).astype(mx.float32)   # kernel 4: 3 old rows + 1 new
    adapter = type("A", (), {"_target": object(), "gdn_states": [
        (None, None, None, None, None, None, None, initial, None, conv_input, 4)]})()
    attn, gdn = Attn(), GDN([mx.zeros((1, 3, 5)), mx.zeros((1, 2, 3, 3))])
    _VLMGDNStateCapture(adapter).rollback([attn, gdn], -1, 1)
    assert attn.trimmed == 1
    assert bool(mx.array_equal(gdn[1], initial))
    assert bool(mx.array_equal(gdn[0], conv_input[:, :3]))


def test_stale_ar_cost_is_reprobed_and_a_stalled_burst_is_skipped(monkeypatch):
    """Ported from glm_dflash2.AdaptiveSpec (MiMo lane 2026-10-16): one stalled burst froze the shared AR cost and
    spec won for the life of the process; cut bursts measured launch overhead."""
    monkeypatch.setenv("VMLX_DFLASH_AR_REPROBE_ROUNDS", "20")
    shared = {}
    c = _seeded(shared, 0.92)
    _ar_burst(c); _ar_burst(c)                             # warm-up, then 33 ms
    _ar_burst(c, per_token=0.30)                           # a stalled burst: skipped
    assert abs(shared["ar"] - 0.033) < 1e-9 and c.ar_outliers_skipped == 1
    c.observe_ar(0.5, 1)                                   # EOS after one token: not an AR cost sample
    assert abs(shared["ar"] - 0.033) < 1e-9
    shared["ar"] = 0.200                                   # a stale estimate: spec always wins ...
    actions = []
    for _ in range(25):
        mode, n = c.next_action()
        actions.append(mode)
        if mode == "ar":
            _ar_burst(c, n=n)                              # ... until the re-probe (always taken)
        else:
            c.observe(c.width, c.width, 0.098)
    assert "ar" in actions and shared["ar"] < 0.1


def test_burst_cut_before_its_planned_length_is_not_an_ar_sample(monkeypatch):
    shared = {}
    c = _seeded(shared, 0.92)
    _ar_burst(c)                                           # warm-up
    c.observe_ar(0.0285 * 5, 5, requested=8)               # EOS after 5 of 8 tokens: not a sample
    assert "ar" not in shared
    c.observe_ar(0.020 * 8, 8, requested=8)
    assert abs(shared["ar"] - 0.020) < 1e-9

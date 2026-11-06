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
    assert seq.count(5) > 180                              # 8 runs only when predicted to win


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
    assert _dflash2_block_plan(8) == (5, 8, 16)               # lane targets with the head on MLX (JANG_4D)
    assert _dflash2_block_plan(4) == (4, 8)
    assert _dflash2_block_plan(8, codebook=True) == (3, 5, 8, 16)   # a dense codebook MLP: narrow rounds are cheap now
    assert _dflash2_block_plan(4, codebook=True) == (3, 4, 8)
    assert _dflash2_block_plan(8, lane_flat=True) == (8, 16)     # flat verify cost: 5 is dominated


def test_first_cycle_at_a_width_is_warmup_and_costs_persist_across_requests():
    shared: dict = {}
    c = _BlockChooser(5, 8, shared)
    widths = []
    for _ in range(4):                                     # warm 5, time 5, warm 8, time 8
        w = c.width
        widths.append(w)
        c.observe(w, w, 2.0 if w not in shared.get("warm", set()) else (0.05 if w == 5 else 0.06))
    assert widths[:2] == [5, 5] and shared[5] == 0.05 and 2.0 not in shared.values()   # 2.0 s warm-up discarded
    cb_shared, cb_widths = {}, []
    cb = _BlockChooser(5, 8, cb_shared, first_widest=True)  # codebook targets: widest first, warm, then timed
    for _ in range(2):
        w = cb.width
        cb_widths.append(w)
        cb.observe(w, w, 2.0 if w not in cb_shared.get("warm", set()) else (0.05 if w == 5 else 0.06))
    assert cb_widths == [8, 8] and cb_shared[8] == 0.06
    shared[8] = 0.06
    c2 = _BlockChooser(5, 8, shared)                       # the next request starts with measured costs
    c2.observe(5, 5, 0.05)
    assert c2.width == 8                                   # 5/5 accepted -> 8 predicted to pay at 1.2x cost


def test_first_widest_cycle_uses_measured_costs():
    """Codebook targets start wide to observe every draft position, but not at a width that costs many cycles:
    27B JANGH2-like costs (widest ~1.9x the narrowest) keep the widest start; Ripple JANGH2's row-exact verify (32 rows
    ~10x 3 rows) starts at the widest width within 2x, instead of ~1 s before every request's second token."""
    qwen = {3: 0.059, 5: 0.077, 8: 0.088, 16: 0.110}
    assert _BlockChooser((3, 5, 8, 16), cost=dict(qwen), first_widest=True).width == 16
    ripple = {3: 0.115, 5: 0.170, 16: 0.580, 32: 1.130}
    assert _BlockChooser((3, 5, 16, 32), cost=dict(ripple), first_widest=True).width == 5
    assert _BlockChooser((3, 5, 16, 32), cost={}, first_widest=True).width == 16      # nothing measured yet: below the widest
    assert _BlockChooser((3, 5, 16, 32), cost=dict(ripple)).width == 3               # non-codebook: narrow start


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


def test_ar_wins_as_soon_as_it_is_predicted_faster(monkeypatch):
    """Turning the drafter on must not cost speed: verify keeps no margin over a faster AR."""
    shared = {}
    c = _seeded(shared, 0.25)
    _ar_burst(c)
    spec_rate = c.expected_tokens(c._best_width()) / c._cost_of(c._best_width())
    _ar_burst(c, per_token=1.0 / (spec_rate * 1.03))        # AR only 3 % faster
    c.ar_since_probe = 0                                     # a probe this close costs almost nothing: not what is tested
    assert c.next_action()[0] == "ar"
    monkeypatch.setenv("VMLX_DFLASH2_CHOOSER_LEGACY", "1")   # the old 5 % hysteresis is the A/B control
    legacy = _BlockChooser((5, 8, 16), cost={5: 0.092, 8: 0.098, 16: 0.104, "ar": 1.0 / (spec_rate * 1.03), "warm": {5, 8, 16, "ar"}})
    legacy._request_ar_pending = False
    legacy.hits, legacy.trials, legacy.cycles = list(c.hits), list(c.trials), 1
    assert legacy.next_action()[0] == "spec"


def test_a_width_without_a_cost_runs_only_when_predicted_to_win():
    """Seeding ran every width twice on the first reply after a load: ~4 s of row-exact verify on hard text."""
    cost = {3: 0.110, 5: 0.150, 16: 0.550, 32: 1.100}
    shared, rng = {}, random.Random(5)
    c = _BlockChooser((3, 5, 16, 32), cost=shared)
    seen = set()
    for _ in range(100):
        w = c.width
        seen.add(w)
        accepted = 0
        while accepted < w - 1 and rng.random() < 0.3:
            accepted += 1
        c.observe(w, accepted + 1, cost[w])
    assert 16 not in seen and 32 not in seen
    assert c._cost_of(16) is not None and 3 in shared        # still priced, so easy text can reach them
    easy = _BlockChooser((3, 5, 16, 32), cost=shared)
    for _ in range(12):
        w = easy.width
        easy.observe(w, w, cost[w] / 5)                      # every draft accepted and now cheap
    assert easy.width >= 16


def test_retiming_stays_inside_the_time_budget():
    """One row-exact 32-row verify costs ~15 AR steps; every 32 cycles was up to a third of the decode time."""
    cost = {3: 0.110, 5: 0.150, 16: 0.550, 32: 1.100}
    shared = {w: v for w, v in cost.items()} | {"warm": {3, 5, 16, 32}}
    c = _BlockChooser((3, 5, 16, 32), cost=shared)
    spent = {w: 0.0 for w in cost}
    rng = random.Random(9)
    for _ in range(1000):
        w = c.width
        spent[w] += cost[w]
        accepted = 0
        while accepted < w - 1 and rng.random() < 0.5:
            accepted += 1
        c.observe(w, accepted + 1, cost[w])
    total = sum(spent.values())
    assert 0 < spent[32] + spent[16] and (spent[32] + spent[16]) / total < 0.02


def _ar_until_probe(c, per_token):
    """AR tokens run before the chooser probes verify again."""
    since = 0
    while True:
        mode, n = c.next_action()
        if mode != "ar":
            return since
        c.observe_ar(n * per_token, n)
        since += n


def test_probes_stay_inside_their_time_budget_on_text_verify_cannot_win():
    shared = {5: 0.170, 16: 0.230, 32: 0.300, "ar": 0.100, "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    c.observe(5, 1, 0.170)
    c._request_ar_pending = False
    ar_s = lost_s = 0.0
    for _ in range(200):
        since = _ar_until_probe(c, 0.100)
        ar_s += since * 0.100
        lost_s += 0.170 - 1.0 * 0.100                        # the probe accepts nothing
        assert c.width == 5
        c.observe(c.width, 1, 0.170, after_ar=True)
    assert lost_s / ar_s <= c.ar_probe_budget * 1.25          # the first probes' estimate still holds the prior


def test_a_bad_start_does_not_lock_a_reply_into_ar():
    """Three failed cycles put AR ahead; the same text then turns draftable, and verify is back within a few bursts."""
    shared = {5: 0.170, 16: 0.230, 32: 0.300, "ar": 0.120, "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    for _ in range(3):
        c.observe(5, 1, 0.170)
    c._request_ar_pending = False
    assert c.next_action()[0] == "ar"
    tokens = 0
    for _ in range(40):
        mode, n = c.next_action()
        if mode == "spec":
            c.observe(c.width, 3, 0.170, after_ar=True)      # two drafts accepted
            tokens += 3
        else:
            c.observe_ar(n * 0.120, n)
            tokens += n
        if c.expected_tokens(c.width) / c.cost[c.width] >= 1.0 / c.cost["ar"]:
            break
    assert tokens <= 32


def test_a_probe_runs_the_best_width_not_a_pending_retime():
    shared = {5: 0.170, 16: 0.230, 32: 0.300, "ar": 0.100, "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    c.observe(5, 1, 0.170)
    c._request_ar_pending = False
    c.width = 32
    _ar_until_probe(c, 0.100)
    assert c.width == 5


def test_legacy_keeps_the_fixed_probe_gap(monkeypatch):
    monkeypatch.setenv("VMLX_DFLASH2_CHOOSER_LEGACY", "1")
    shared = {5: 0.170, 16: 0.230, 32: 0.300, "ar": 0.100, "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    c.observe(5, 1, 0.170)
    c._request_ar_pending = False
    assert _ar_until_probe(c, 0.100) >= c.ar_probe_every


def test_easy_text_skips_the_requests_ar_measurement():
    shared = {5: 0.150, 16: 0.240, 32: 0.270, "ar": 0.070, "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    c.observe(5, 5, 0.150)
    assert c.next_action()[0] == "spec"


def test_cycle_after_an_ar_burst_counts_acceptance_not_time():
    shared = {}
    c = _seeded(shared, 0.5)
    before = dict(shared)
    trials = list(c.trials)
    w = c.width
    assert w in shared
    c.observe(w, 3, 5.0, after_ar=True)                     # slow: the drafter also ingested the burst's rows
    assert shared[w] == before[w] and c.catchup_ms
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


def test_new_request_remeasures_ar_after_transient_scheduling_slowdown():
    # H2 live recovery: normal priorities returned, but the previous request's
    # 32.3 ms AR estimate made ~48 ms / 2-token verification look faster.
    shared = {5: 0.0484, 16: 0.1107, 32: 0.1936, "ar": 0.0323,
              "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    assert c.next_action() == ("spec", 5)  # preserve the initial draft/checkpoint
    for _ in range(4):
        c.observe(5, 2, 0.0484)                              # soon no longer clearly ahead of the old AR rate: measure it
    assert c.next_action() == ("ar", 8)
    c.observe_ar(8 * 0.0185, 8)
    assert abs(shared["ar"] - 0.0185) < 1e-9
    assert c.next_action() == ("ar", 8)


def test_new_request_ar_measurement_handles_slowdown_and_ignores_cut_burst():
    shared = {5: 0.0484, 16: 0.1107, 32: 0.1936, "ar": 0.005,
              "warm": {5, 16, 32, "ar"}}
    c = _BlockChooser((5, 16, 32), cost=shared)
    c.observe(5, 2, 0.0484)
    assert c.next_action() == ("ar", 8)
    c.observe_ar(0.5, 1)  # EOS/max-token cut is not a valid new baseline
    assert shared["ar"] == 0.005
    assert c.next_action() == ("ar", 8)
    c.observe_ar(8 * 0.0323, 8)
    assert abs(shared["ar"] - 0.0323) < 1e-9
    assert c.ar_outliers_skipped == 0  # a new request can have genuinely changed cost


def test_block_plan_env_candidate_list_is_clipped_sorted_and_optional(monkeypatch):
    from vmlx_engine.dflash2_runtime import _dflash2_block_plan

    monkeypatch.delenv("VMLX_DFLASH2_BLOCK", raising=False)
    monkeypatch.setenv("VMLX_DFLASH2_WIDTHS", "3,5,8,16")
    assert _dflash2_block_plan(8) == (3, 5, 8, 16)
    assert _dflash2_block_plan(8, lane_flat=True) == (3, 5, 8, 16)
    monkeypatch.setenv("VMLX_DFLASH2_WIDTHS", "40,1,8,8")
    assert _dflash2_block_plan(8) == (2, 8, 16)              # clipped to 2..2*trained, deduplicated, sorted
    monkeypatch.setenv("VMLX_DFLASH2_WIDTHS", "x")
    assert _dflash2_block_plan(8) == (5, 8, 16)              # unparsable: the default plan
    assert _dflash2_block_plan(8, codebook=True) == (3, 5, 8, 16)
    monkeypatch.setenv("VMLX_DFLASH2_WIDTHS", "")
    assert _dflash2_block_plan(8) == (5, 8, 16)
    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "4")             # a pinned width still wins
    monkeypatch.setenv("VMLX_DFLASH2_WIDTHS", "3,5")
    assert _dflash2_block_plan(8) == (4,)


def test_codebook_mlp_detects_dense_codebook_projections():
    # Width 3 and the widest first cycle apply only to targets with dense codebook projections (JANGH2); a lane target
    # with its head on MLX (JANG_4D) keeps the base plan and start (it measured worse with both, 2026-10-19).
    import types

    import mlx.nn as nn

    from vmlx_engine.dflash2_runtime import _codebook_mlp

    class TQLinear(nn.Module):
        pass

    def adapter(**mods):
        t = nn.Module()
        for k, v in mods.items():
            setattr(t, k, v)
        return types.SimpleNamespace(_target=t)

    assert _codebook_mlp(adapter(mlp=TQLinear(), head=nn.Linear(4, 4))) is True
    assert _codebook_mlp(adapter(q=nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4))) is False

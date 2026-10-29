# SPDX-License-Identifier: Apache-2.0
"""DSpark planner exploration (``VMLX_DSPARK_EXPLORE``): positions past the drafted width are never observed, so a
position whose estimate fell stays low and the planned width never grows again; with the lever a round drafts one
deeper when the next position has not been observed for ``explore_every`` planned rounds and the chain likely reaches
it. Widths only: greedy DSpark still equals AR (the planner never touches tokens)."""

from __future__ import annotations

import pytest

from vmlx_engine.dspark.generate import ARFallback


def _drop_position_one(policy: ARFallback) -> None:
    for _ in range(6):
        policy.record_acceptance(4, 1)                   # position 0 accepted, position 1 rejected


def test_without_the_lever_a_dropped_position_locks_the_width(monkeypatch):
    monkeypatch.delenv("VMLX_DSPARK_EXPLORE", raising=False)
    p = ARFallback()
    _drop_position_one(p)
    assert p.planned_width(8, 0.12) == 1
    for _ in range(300):                                 # the text turned easy: every drafted token accepted
        k = p.planned_width(8, 0.12)
        p.record_acceptance(k, k)
    assert p.planned_width(8, 0.12) == 1                 # never re-measured past position 0


def test_with_the_lever_the_planner_retries_wider_rounds(monkeypatch):
    monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
    p = ARFallback()
    _drop_position_one(p)
    assert p.planned_width(8, 0.12) == 1
    seen = []
    for _ in range(300):
        k = p.planned_width(8, 0.12)
        seen.append(k)
        p.record_acceptance(k, k)
    assert max(seen) >= 6 and seen[-1] >= 6              # re-explored and climbed once the drafts kept landing


def test_with_the_lever_a_really_bad_position_is_pulled_back_down(monkeypatch):
    monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
    p = ARFallback()
    _drop_position_one(p)
    widths = []
    for _ in range(400):
        k = p.planned_width(8, 0.12)
        widths.append(k)
        p.record_acceptance(k, min(k, 1))                # position 1 always rejected
    assert max(widths) >= 2                              # it is retried ...
    assert sum(w > 1 for w in widths[-320:]) <= 320 // ARFallback.explore_every + 1   # ... once per explore_every


def test_no_probe_where_the_chain_does_not_reach(monkeypatch):
    """All drafts wrong: position 1 can never be observed (position 0 always rejects first), so it is never probed;
    the plan stays at one draft (no probe every round once the position is stale)."""
    monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
    p = ARFallback()
    for _ in range(5):
        p.record_acceptance(3, 0)
    widths = []
    for _ in range(320):
        k = p.planned_width(8, 0.12)
        widths.append(k)
        p.record_acceptance(k, 0)
    assert widths == [1] * 320


def test_the_lever_does_not_widen_while_positions_are_fresh(monkeypatch):
    """A steady stream whose positions are all observed every round plans exactly what the lever-off planner plans."""
    monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
    on, off = ARFallback(), ARFallback()
    for i in range(100):
        k_on = on.planned_width(8, 0.12)
        monkeypatch.delenv("VMLX_DSPARK_EXPLORE")
        k_off = off.planned_width(8, 0.12)
        monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
        if i < 32:
            assert k_on == k_off
        acc = 3 if i % 3 else 1
        on.record_acceptance(k_on, min(acc, k_on))
        off.record_acceptance(k_off, min(acc, k_off))


@pytest.mark.parametrize("prompt_len", [773, 509])
def test_dspark_greedy_equals_ar_with_exploration(monkeypatch, prompt_len, tmp_path):
    """Greedy DSpark with the lever (pipelined planner, every round's width from the history + probes) == greedy AR,
    on the tiny M3 target with the released drafter geometry and drafts wrong at every fifth position."""
    mx = pytest.importorskip("mlx.core")
    import tests.test_minimax_m3_dspark as TD
    from vmlx_engine.dspark import DSparkStats, generate_ar, generate_dspark, load_drafter
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    monkeypatch.setenv("VMLX_DSPARK_EXPLORE", "1")
    target = MiniMaxM3DSparkTarget(TD._model())
    drafter_dir = TD._save_release_drafter(tmp_path / "dspark")
    prompt = TD._ids(prompt_len, 11)
    reference = generate_ar(target, prompt, 60, set(), prefill_step=256)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    stats = DSparkStats()
    policy = TD._WidthOnly()
    policy.explore_every = 4                              # probe often in a short run
    out = generate_dspark(target, drafter, prompt, 60, set(), prefill_step=256, stats=stats,
                          max_draft=2 * TD.DRAFT_BLOCK, ar_fallback=policy)
    assert out == reference
    assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted
    assert getattr(policy, "planned_rounds", 0) == stats.rounds          # the lever's planner ran every round

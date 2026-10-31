# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3's runtime sampler (make_minimax_m3_sampler, the M3 / M3.1 serving and DSpark-lane sampler): temperature,
top_k, top_p and min_p each restrict / reshape the distribution as documented, alone and combined.

Each case draws 4,000 samples from one row of known probabilities [0.5, 0.3, 0.15, 0.05] (rows replicated, so one
categorical call) and checks the support exactly and the frequencies loosely. min_p used to be ignored by this
sampler; its cases fail on that version (the excluded tokens appear)."""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.sampling import make_minimax_m3_sampler  # noqa: E402

PROBS = [0.5, 0.3, 0.15, 0.05]
N = 4000


def _draw(**kw):
    sampler = make_minimax_m3_sampler(seed=1234, **kw)
    row = mx.array([math.log(p) for p in PROBS], dtype=mx.float32)
    logits = mx.broadcast_to(row, (N, len(PROBS)))
    out = sampler(logits)
    mx.eval(out)
    counts = [0] * len(PROBS)
    for t in out.tolist():
        counts[int(t)] += 1
    return counts


def test_greedy_is_argmax():
    assert _draw(temp=0.0) == [N, 0, 0, 0]


def test_temperature_one_matches_the_distribution():
    counts = _draw(temp=1.0)
    for c, p in zip(counts, PROBS):
        assert abs(c / N - p) < 0.03, counts


def test_low_temperature_sharpens():
    counts = _draw(temp=0.25)
    # p^(1/0.25) renormalized: [0.877, 0.114, 0.0071, 0.00009]
    assert counts[0] / N > 0.83 and counts[3] <= 5, counts


@pytest.mark.parametrize("k, support", [(1, {0}), (2, {0, 1}), (3, {0, 1, 2})])
def test_top_k(k, support):
    counts = _draw(temp=1.0, top_k=k)
    assert {i for i, c in enumerate(counts) if c} == support, counts


@pytest.mark.parametrize("top_p, support", [(0.4, {0}), (0.6, {0, 1}), (0.85, {0, 1, 2})])
def test_top_p(top_p, support):
    counts = _draw(temp=1.0, top_p=top_p)
    assert {i for i, c in enumerate(counts) if c} == support, counts


@pytest.mark.parametrize("min_p, support", [(0.7, {0}), (0.5, {0, 1}), (0.25, {0, 1, 2}), (0.05, {0, 1, 2, 3})])
def test_min_p(min_p, support):
    """Keep tokens with p >= min_p * p_max (p_max 0.5): 0.7 -> {0}; 0.5 -> {0, 1}; 0.25 -> {0, 1, 2}; 0.05 -> all."""
    counts = _draw(temp=1.0, min_p=min_p)
    assert {i for i, c in enumerate(counts) if c} == support, counts


def test_min_p_renormalizes_over_the_kept_tokens():
    counts = _draw(temp=1.0, min_p=0.5)
    # kept [0.5, 0.3] -> [0.625, 0.375]
    assert abs(counts[0] / N - 0.625) < 0.03 and abs(counts[1] / N - 0.375) < 0.03, counts


def test_combined_top_k_min_p_top_p():
    # top_k 3 -> {0,1,2}; min_p 0.25 keeps all three; top_p 0.6 -> {0,1}
    assert {i for i, c in enumerate(_draw(temp=1.0, top_k=3, min_p=0.25, top_p=0.6)) if c} == {0, 1}
    # min_p applies after temperature: at temp 2 the probabilities flatten to ~[0.38, 0.29, 0.21, 0.12] and
    # min_p 0.5 keeps {0, 1, 2}
    assert {i for i, c in enumerate(_draw(temp=2.0, min_p=0.5)) if c} == {0, 1, 2}


def test_seeded_sampler_is_reproducible():
    assert _draw(temp=1.0, top_p=0.95) == _draw(temp=1.0, top_p=0.95)

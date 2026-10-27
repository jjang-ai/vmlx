"""Sampled DSpark with a fixed seed is reproducible whatever the adaptive governor chooses.

DSparkRunner.decode_stream draws every committed token as argmax(f(logits) / T + g_pos) with g_pos keyed by the request
seed and the ABSOLUTE position (SpecSampler). The governor's plain AR steps pick the next token from the same noise
row as a speculative step would, so the token sequence must not depend on which steps ran as AR and which speculated.
Synthetic backbone + draft (no weights): logits are a fixed function of (token, position).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import mlx.core as mx

dspark = pytest.importorskip("jang_tools.dsv41.dspark")
if not hasattr(dspark, "SpecSampler"):
    pytest.skip("jang_tools without sampled DSpark", allow_module_level=True)

V = 48
K = 97
_TABLE = mx.array(np.random.default_rng(3).standard_normal((K, V)).astype(np.float32) * 2.0)


def _row(tok: int, pos: int):
    return _TABLE[(int(tok) * 31 + int(pos)) % K]


class _Sess:
    def __init__(self):
        self.toks: list[int] = []

    @property
    def pos(self):
        return len(self.toks)

    def checkpoint(self):
        return list(self.toks)

    def restore(self, ck):
        self.toks = list(ck)


class _Backbone:
    """forward(ids, sess) appends ids; row j's logits predict position (start + j + 1)."""

    def forward(self, ids, sess, all_logits=False, return_hidden=False, **_):
        start = sess.pos
        sess.toks.extend(int(t) for t in ids)
        rows = mx.stack([_row(t, start + j) for j, t in enumerate(ids)])
        out = rows if all_logits else rows[-1]
        if return_hidden:
            return out, mx.zeros((len(ids), 4))
        return out


class _Draft:
    """Proposes the target's own sample for the first two positions, then a wrong token (partial accepts)."""

    bs = 4

    def __init__(self, sampler):
        self.sampler = sampler

    def seed_windows(self, *_a, **_k):
        pass

    def draft(self, seed_tok, start_pos, noise=None, temperature=1.0):
        out, tok = [int(seed_tok)], int(seed_tok)
        for i in range(self.bs):
            pos = start_pos + 1 + i  # position of `tok`; its logits predict pos + 1
            if i < 2:
                tok = int(self.sampler.pick(_row(tok, pos).reshape(1, -1), noise[i : i + 1])[0])
            else:
                tok = (tok + 5) % V
            out.append(tok)
        return out, mx.ones((self.bs,))


def _run(seed, *, adaptive, choose=None, monkeypatch=None, n=40):
    sampler = dspark.SpecSampler(0.8, top_p=0.95, seed=seed)
    runner = dspark.DSparkRunner(_Backbone(), _Draft(sampler), adaptive=adaptive)
    if choose is not None:
        it = itertools.cycle(choose)
        monkeypatch.setattr(dspark._Governor, "choose", lambda self: next(it))
    sess = _Sess()
    prompt = [3, 9, 27, 4]
    pending = runner.m.forward(prompt, sess)
    stats = {}
    toks = list(runner.decode_stream(sess, pending, len(prompt) - 1, n, stats=stats, sampler=sampler))
    return toks, stats


def test_same_seed_same_tokens_regardless_of_governor(monkeypatch):
    spec_only, st_spec = _run(11, adaptive=False)
    all_ar, st_ar = _run(11, adaptive=True, choose=["ar"], monkeypatch=monkeypatch)
    mixed, st_mix = _run(11, adaptive=True, choose=["spec", "ar", "ar", "spec", "ar"], monkeypatch=monkeypatch)
    assert st_spec["ar_steps"] == 0 and st_spec["spec_steps"] > 0
    assert st_ar["spec_steps"] == 0 and st_ar["ar_steps"] == len(all_ar)
    assert st_mix["ar_steps"] > 0 and st_mix["spec_steps"] > 0
    assert spec_only == all_ar == mixed
    assert len(spec_only) == 40


def test_seed_changes_the_draw():
    a, _ = _run(11, adaptive=False)
    b, _ = _run(12, adaptive=False)
    assert a != b

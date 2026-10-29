# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 cold prefill through SingleBatchGenerator: the speed changes keep every bit (2026-10-29).

Tiny synthetic M3 (tests/test_minimax_m3_dspark.py: MSA on layers 1-3 with the released geometry, blocks of 128 keys,
top-16, so selection turns sparse past 2,048 keys; MoE routed + shared experts; MXFP8 / affine trunk). Prompt lengths
and prefill steps do not divide by 64 / 128 / 2048.

Reference = the previous generator behaviour written out: the same chunks through ``model()``, every chunk's FULL
logits evaluated (the plan-less path did that), the first token sampled from the final chunk's last row. Exactness is
``mx.array_equal`` on the logits the sampler sees and on every cache lane (K / V / index keys).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_minimax_m3_dspark import VOCAB, _model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return _model(block=128, topk=16)


def _ids(n: int, seed: int) -> list:
    return np.random.default_rng(seed).integers(0, VOCAB - 1, n).tolist()


def _lanes(cache) -> list:
    out = []
    for layer in cache:
        for a in layer.state if isinstance(layer.state, (list, tuple)) else (layer.state,):
            if isinstance(a, mx.array):
                out.append(a)
        idx = getattr(layer, "idx_keys", None)
        if isinstance(idx, mx.array):
            out.append(idx[:, :, : layer.offset])
    return out


def _reference(model, prompt, step):
    """The previous plan-less prefill: chunks of ``step`` (the last chunk keeps >= 2 tokens), full logits per chunk."""
    cache = model.make_cache()
    pos = 0
    while len(prompt) - pos > step:
        n = min(step, len(prompt) - pos)
        if len(prompt) - (pos + n) == 1 and n > 1:
            n -= 1
        mx.eval(model(mx.array([prompt[pos:pos + n]]), cache=cache))
        pos += n
    logits = model(mx.array([prompt[pos:]]), cache=cache)
    mx.eval(logits)
    return logits[:, -1, :], cache, len(prompt) - pos


def _served(model, prompt, step, plan=None):
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    gen = SingleBatchGenerator(model, max_tokens=1, sampler=lambda x: mx.argmax(x, axis=-1), prefill_step_size=step)
    seen, evals = {}, []
    o_sample, o_eval = gen._sample_from_logits, gen._eval_on_stream

    def sample(logits, *a, **k):
        seen.setdefault("logits", logits)
        return o_sample(logits, *a, **k)

    def ev(*a, **k):
        evals.append(a[0].shape if a else None)
        return o_eval(*a, **k)

    gen._sample_from_logits, gen._eval_on_stream = sample, ev
    cache = model.make_cache()
    gen.insert([prompt], caches=[cache], **({"m3_prefill_plans": [plan]} if plan else {}))
    gen.next()
    return seen["logits"], cache, evals


@pytest.mark.parametrize("n,step", [(2701, 320), (4333, 2048), (641, 320)])
def test_planless_prefill_equals_the_full_logits_reference(model, n, step):
    prompt = _ids(n, n)
    ref_logits, ref_cache, final = _reference(model, prompt, step)
    logits, cache, evals = _served(model, prompt, step)
    assert mx.array_equal(logits, ref_logits).item()
    a, b = _lanes(cache), _lanes(ref_cache)
    assert len(a) == len(b) > 0
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a, b))
    # only the final chunk's logits are materialized (the vocabulary projection of earlier chunks is never read)
    assert [s for s in evals if s is not None and len(s) == 3] == [(1, final, VOCAB)]


def test_control_comparison_sees_a_one_token_change(model):
    """Fail closed: the equality above is not vacuous; one early prompt token changed moves the logits and lanes."""
    prompt = _ids(2701, 5)
    other = list(prompt)
    other[100] = (other[100] + 1) % (VOCAB - 1)
    a_logits, a_cache, _ = _reference(model, prompt, 320)
    b_logits, b_cache, _ = _reference(model, other, 320)
    assert not mx.array_equal(a_logits, b_logits).item()
    assert any(not mx.array_equal(x, y).item() for x, y in zip(_lanes(a_cache), _lanes(b_cache)))


# ------------------------------------------------------------------------------------------- capacity reservation
def _generate(model, prompt, step, plan, n_tokens, monkeypatch, reserve: str, cache=None, context=None):
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    monkeypatch.setenv("VMLX_M3_PREFILL_RESERVE", reserve)
    grows = []
    real = MiniMaxM3SparseCache._grow_rows

    def spy(self, key, prev, n, step_):
        rows = real(self, key, prev, n, step_)
        grows.append((key, prev, n, rows))
        return rows

    monkeypatch.setattr(MiniMaxM3SparseCache, "_grow_rows", spy)
    gen = SingleBatchGenerator(model, max_tokens=n_tokens, sampler=lambda x: mx.argmax(x, axis=-1),
                               prefill_step_size=step)
    seen = []
    o_sample = gen._sample_from_logits

    def sample(logits, *a, **k):
        seen.append(logits)
        return o_sample(logits, *a, **k)

    gen._sample_from_logits = sample
    cache = cache if cache is not None else model.make_cache()
    kw = {"m3_prefill_plans": [plan]} if plan else {}
    if context is not None:
        kw.update(all_tokens=[context])
    gen.insert([prompt], caches=[cache], max_tokens=[n_tokens], **kw)
    toks, fin = [], None
    while fin is None:
        out = gen.next()
        resp = (out[0] or out[1])[0]
        toks.append(resp.token)
        fin = resp.finish_reason
    monkeypatch.setattr(MiniMaxM3SparseCache, "_grow_rows", real)
    return toks, seen, cache, grows


@pytest.mark.parametrize("n,step,planned", [(2701, 320, True), (2701, 320, False), (4333, 2048, True), (300, 320, True)])
def test_reserved_prefill_is_bit_identical_and_grows_once(model, monkeypatch, n, step, planned):
    prompt = _ids(n, n + 1)
    plan = {"boundary": n - 3, "atomic": False} if planned else None
    a_toks, a_seen, a_cache, a_grows = _generate(model, prompt, step, plan, 9, monkeypatch, "0")
    b_toks, b_seen, b_cache, b_grows = _generate(model, prompt, step, plan, 9, monkeypatch, "1")
    assert a_toks == b_toks and len(a_seen) == len(b_seen) == 9
    assert all(mx.array_equal(x, y).item() for x, y in zip(a_seen, b_seen))   # first-token AND decode logits
    la, lb = _lanes(a_cache), _lanes(b_cache)
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(la, lb))
    sparse = sum(1 for c in b_cache if hasattr(c, "reserve"))
    pieces = -(-n // step) + 1
    # with the reservation every lane grows once during the prefill (K / V and the index lane), without it per chunk
    pre_b = [g for g in b_grows if g[1] < n]
    assert len(pre_b) == 2 * sparse
    assert all(g[3] >= n for g in pre_b)
    if n > step:
        assert len([g for g in a_grows if g[1] < n]) > len(pre_b)
    # the reservation is consumed: decode growth is the stock 256-row step again
    assert all(int(c.__dict__.get("_reserve_until") or 0) == 0 for c in b_cache if hasattr(c, "reserve"))
    del pieces


def test_reserved_continuation_after_an_unaligned_restore(model, monkeypatch):
    """A restored prefix whose length is not a multiple of the 256-row step (trimmed back, capacity past it), then a reserved
    continuation: same tokens, logits and lanes as the stock growth."""
    import copy

    prompt = _ids(3001, 77)
    _, _, base, _ = _generate(model, prompt[:1337], 320, None, 1, monkeypatch, "0")
    for c in base:
        if hasattr(c, "trim"):
            c.trim(int(c.offset) - 1337)
    rest = prompt[1337:]
    a = _generate(model, rest, 320, None, 5, monkeypatch, "0", cache=copy.deepcopy(base), context=prompt[:1337])
    b = _generate(model, rest, 320, None, 5, monkeypatch, "1", cache=copy.deepcopy(base), context=prompt[:1337])
    assert a[0] == b[0]
    assert all(mx.array_equal(x, y).item() for x, y in zip(a[1], b[1]))
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(_lanes(a[2]), _lanes(b[2])))

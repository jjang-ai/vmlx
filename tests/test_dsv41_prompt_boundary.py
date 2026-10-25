"""DeepSeek-V4.1: prompt-boundary snapshot contract (CPU only, fake runtime).

The scheduler stores a generator's prompt snapshot under ``prompt_tokens[:-1]`` and re-feeds the
last prompt token on a hit, so both the AR SingleBatchGenerator path and the DSpark generator must
snapshot at exactly N-1 tokens. They used to snapshot after the final prompt token (N under an N-1
key), so a hit re-fed that token at position N.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")
dspark = pytest.importorskip("jang_tools.dsv41.dspark")

from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator  # noqa: E402
from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator  # noqa: E402

V = 37


def _next(tok: int, pos: int) -> int:
    return (tok * 7 + pos * 3 + 1) % V


class FakeRuntime:
    """Duck-types jang_tools.dsv41.model.DSV41.forward over the engine's DeepseekV41Cache list.
    Each layer's window holds every fed token as a row (so the snapshot boundary is observable)."""

    def __init__(self, n_layers=3):
        self.args = SimpleNamespace(num_hidden_layers=n_layers, compress_ratios=[0, 2, 1][:n_layers])
        self.forwards = []

    def new_session(self):
        return mlx_model._SeqState()

    def forward(self, ids, sess, all_logits=False, return_hidden=False, images=None, token_types=None):
        ids = [int(t) for t in ids]
        self.forwards.append((sess.pos, list(ids)))
        rows = mx.array(ids, dtype=mx.float32).reshape(1, -1, 1)
        for c in sess.caches:
            c.window = rows if c.window is None else mx.concatenate([c.window, rows], axis=1)
        nxt = [_next(t, sess.pos + i) for i, t in enumerate(ids)]
        sess.pos += len(ids)
        logits = mx.stack([mx.eye(V)[n] * 10.0 for n in nxt])  # [L, V]
        out = logits if all_logits else logits[-1]
        if return_hidden:
            return out, mx.zeros((len(ids), 4))
        return out


class FakeDraft:
    """Minimal dSpark draft: perfect (or always-wrong) proposals; records seeded positions."""

    def __init__(self, win=4, bs=3, correct=True, rt=None):
        self.bs = bs
        self.correct = correct
        self.seeded = []
        self.stages = [SimpleNamespace(attn=SimpleNamespace(win=win, window=dspark.BoundedWindow(win)))]

    def reset_windows(self):
        self.seeded = []

    def seed_windows(self, main_hidden, positions):
        positions = list(positions)
        assert main_hidden.shape[1] == len(positions)
        self.seeded.extend(positions)

    def draft(self, seed_tok, start_pos):
        out, t, p = [int(seed_tok)], int(seed_tok), start_pos + 1
        for _ in range(self.bs):
            n = _next(t, p) if self.correct else (_next(t, p) + 1) % V
            out.append(n)
            t, p = n, p + 1
        return out, mx.ones((self.bs,))


def _model(rt):
    return mlx_model.Model(mlx_model.ModelArgs()).attach(rt)


def _greedy_sampler():
    s = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
    s._vmlx_greedy = True
    s._vmlx_accepts_logits = True
    return s


def _run(gen, prompt, max_tokens=6, caches=None, all_tokens=None):
    gen.insert([prompt], max_tokens=[max_tokens], caches=caches, all_tokens=all_tokens)
    out, first = [], None
    while True:
        prompt_resp, gen_resp = gen.next()
        resp = (prompt_resp or gen_resp or [None])[0]
        if resp is None:
            break
        first = first or resp
        out.append(resp.token)
        if resp.finish_reason is not None:
            break
    return out, first


def _window_tokens(cache):
    return [int(x) for x in cache[0].window.reshape(-1).tolist()]


def _expected(prompt, n):
    toks, pos, out = list(prompt), len(prompt), []
    t = _next(toks[-1], pos - 1)
    for _ in range(n):
        out.append(t)
        t = _next(t, pos)
        pos += 1
    return out


@pytest.mark.parametrize("gen_cls", [SingleBatchGenerator, DSV41DSparkGenerator])
def test_cold_snapshot_is_exact_n_minus_one(gen_cls):
    rt = FakeRuntime()
    model = _model(rt)
    model._dspark = FakeDraft()
    gen = gen_cls(model, sampler=_greedy_sampler(), max_tokens=6, prefill_step_size=2)
    prompt = [5, 6, 7, 8, 9]
    out, first = _run(gen, prompt)
    assert out == _expected(prompt, 6)
    snap = first.prompt_cache_snapshot
    assert snap is not None
    assert all(int(c.offset) == len(prompt) - 1 for c in snap)
    assert _window_tokens(snap) == prompt[:-1]
    assert first.prefill_usage and first.prefill_usage["tokens"] == len(prompt)
    # all_tokens carries the whole prompt plus the emitted token
    assert first.all_tokens[: len(prompt)] == prompt


@pytest.mark.parametrize("gen_cls", [SingleBatchGenerator, DSV41DSparkGenerator])
def test_prefix_hit_snapshot_is_exact_n_minus_one(gen_cls):
    rt = FakeRuntime()
    model = _model(rt)
    model._dspark = FakeDraft()
    prompt = [5, 6, 7, 8, 9, 10]
    # a restored 3-token prefix (what reconstruct_cache returns for a 3-token block table)
    cache = model.make_cache()
    model(mx.array([prompt[:3]]), cache=cache)
    gen = gen_cls(model, sampler=_greedy_sampler(), max_tokens=4, prefill_step_size=2)
    out, first = _run(gen, prompt[3:], max_tokens=4, caches=[cache], all_tokens=[prompt[:3]])
    assert out == _expected(prompt, 4)
    snap = first.prompt_cache_snapshot
    assert all(int(c.offset) == len(prompt) - 1 for c in snap)
    assert _window_tokens(snap) == prompt[:-1]
    assert first.all_tokens[: len(prompt)] == prompt


def test_dspark_output_equals_ar_and_seeds_only_last_window():
    prompt = list(range(1, 12))
    rt_ar = FakeRuntime()
    ar_out, _ = _run(SingleBatchGenerator(_model(rt_ar), sampler=_greedy_sampler(), max_tokens=9), prompt, 9)
    for correct in (True, False):
        rt = FakeRuntime()
        model = _model(rt)
        draft = FakeDraft(win=4, correct=correct)
        model._dspark = draft
        gen = DSV41DSparkGenerator(model, sampler=_greedy_sampler(), max_tokens=9, prefill_step_size=3)
        gen.insert([prompt], max_tokens=[9])
        prompt_resp, _ = gen.next()
        # prefill seeded only the last `win` prompt positions
        assert draft.seeded[:4] == list(range(len(prompt) - 4, len(prompt)))
        out = [prompt_resp[0].token]
        while prompt_resp[0].finish_reason is None:
            _, g = gen.next()
            out.append(g[0].token)
            if g[0].finish_reason is not None:
                break
        assert out == ar_out
        # chunked prefill: prompt[:-1] in chunks of 3, then the last token alone
        prefill = [f for f in rt.forwards if f[0] < len(prompt)]
        assert [len(f[1]) for f in prefill] == [3, 3, 3, 1, 1]


@pytest.mark.parametrize(
    "sampler_attrs,processors",
    [({}, None), ({"_vmlx_greedy": True}, [lambda toks, logits: logits])],
)
def test_dspark_only_for_greedy_without_processors(sampler_attrs, processors):
    rt = FakeRuntime()
    model = _model(rt)
    model._dspark = FakeDraft()
    sampler = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
    for k, v in sampler_attrs.items():
        setattr(sampler, k, v)
    gen = DSV41DSparkGenerator(model, sampler=sampler, max_tokens=3)
    gen.insert([[1, 2, 3]], max_tokens=[3], logits_processors=[processors] if processors else None)
    gen.next()
    req = gen._request
    assert req is None or not getattr(req, "_dspark_on", False)
    assert not model._dspark.seeded  # the draft never ran

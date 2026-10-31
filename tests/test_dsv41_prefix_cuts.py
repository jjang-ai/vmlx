"""DeepSeek-V4.1 prefix cuts: extra exact snapshots inside a prefill (CPU only, fake runtime).

A DeepseekV41Cache cannot be trimmed, so a stored prefix is reusable only at the exact position it was taken at.
The scheduler asks the generator for snapshots at two cuts (after the first and after the last ``<｜User｜>``);
the generator ends a prefill segment at each cut (its own prompt_end) and returns (position, snapshot) pairs. A
later request restored at a cut continues with exactly the segments a cold prefill of it runs after that cut.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

mlx_model = pytest.importorskip("jang_tools.dsv4_1.mlx_model")
dspark = pytest.importorskip("jang_tools.dsv41.dspark")

from vmlx_engine.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig  # noqa: E402
from vmlx_engine.scheduler import Scheduler  # noqa: E402
from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator, _prefill_takes_cuts  # noqa: E402
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator  # noqa: E402

from tests.test_dsv41_prompt_boundary import (  # noqa: E402
    FakeDraft,
    FakeRuntime,
    _expected,
    _greedy_sampler,
    _model,
    _window_tokens,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _run(gen, prompt, max_tokens=4, caches=None, all_tokens=None, cuts=None):
    gen.insert([prompt], max_tokens=[max_tokens], caches=caches, all_tokens=all_tokens,
               prefix_cuts=[cuts] if cuts is not None else None)
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


def _gens():
    out = [SingleBatchGenerator]
    rt = FakeRuntime()
    m = _model(rt)
    m._dspark = FakeDraft(rt=rt)
    if _prefill_takes_cuts(dspark.DSparkRunner(m.runtime, m._dspark)):
        out.append(DSV41DSparkGenerator)
    return out


def _make(gen_cls, step=3):
    rt = FakeRuntime()
    model = _model(rt)
    model._dspark = FakeDraft(win=4)
    return rt, model, gen_cls(model, sampler=_greedy_sampler(), max_tokens=4, prefill_step_size=step)


PROMPT = [11, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]


@pytest.mark.parametrize("gen_cls", _gens())
def test_cuts_return_exact_snapshots_and_keep_output(gen_cls):
    rt, model, gen = _make(gen_cls)
    out, first = _run(gen, PROMPT, cuts=[4, 9])
    assert out == _expected(PROMPT, 4)
    cps = first.boundary_checkpoints
    assert [p for p, _ in cps] == [4, 9]
    for pos, snap in cps:
        assert {int(c.offset) for c in snap} == {pos}         # tokens held == key length
        assert _window_tokens(snap) == PROMPT[:pos]
    # the N-1 snapshot is unchanged by the cuts
    assert {int(c.offset) for c in first.prompt_cache_snapshot} == {len(PROMPT) - 1}
    assert _window_tokens(first.prompt_cache_snapshot) == PROMPT[:-1]
    # every segment ends its own prefill: prompt_end = the cut, then N-1
    body_ends = [e for e in rt.prompt_ends if e is not None]
    assert set(body_ends) == {4, 9, len(PROMPT) - 1}
    assert body_ends == sorted(body_ends)


@pytest.mark.parametrize("gen_cls", _gens())
def test_no_cuts_is_the_previous_prefill(gen_cls):
    rt_a, _, gen_a = _make(gen_cls)
    out_a, first_a = _run(gen_a, PROMPT)
    rt_b, _, gen_b = _make(gen_cls)
    out_b, first_b = _run(gen_b, PROMPT, cuts=[])
    assert out_a == out_b
    assert rt_a.forwards == rt_b.forwards and rt_a.prompt_ends == rt_b.prompt_ends
    assert first_a.boundary_checkpoints is None and first_b.boundary_checkpoints is None


@pytest.mark.parametrize("gen_cls", _gens())
def test_restore_at_cut_continues_like_the_cold_split(gen_cls):
    """Restore-then-prefill == a cold prefill cut at the same position: same forwards after the cut (positions,
    token chunks, prompt_end), same N-1 snapshot, same tokens."""
    pos = 9
    rt_cold, _, gen_cold = _make(gen_cls)
    out_cold, first_cold = _run(gen_cold, PROMPT, cuts=[pos])
    snap = dict(first_cold.boundary_checkpoints)[pos]
    # another request with the same first `pos` tokens restores the cut and prefills only the rest
    rt_hit, model_hit, gen_hit = _make(gen_cls)
    restored = mlx_model.clone_cache_list(snap)
    out_hit, first_hit = _run(gen_hit, PROMPT[pos:], caches=[restored], all_tokens=[PROMPT[:pos]])
    assert out_hit == out_cold
    after = lambda rt: [(p, ids, e) for (p, ids), e in zip(rt.forwards, rt.prompt_ends) if p >= pos]  # noqa: E731
    assert after(rt_hit) == after(rt_cold)
    assert _window_tokens(first_hit.prompt_cache_snapshot) == _window_tokens(first_cold.prompt_cache_snapshot)


@pytest.mark.parametrize("gen_cls", _gens())
def test_cuts_inside_a_restored_prefix_are_ignored(gen_cls):
    rt, model, gen = _make(gen_cls)
    cache = model.make_cache()
    model(mx.array([PROMPT[:5]]), cache=cache)
    out, first = _run(gen, PROMPT[5:], caches=[cache], all_tokens=[PROMPT[:5]], cuts=[4, 9, len(PROMPT) - 1])
    assert out == _expected(PROMPT, 4)
    assert [p for p, _ in first.boundary_checkpoints] == [9]


# ----------------------------------------------------------------------------------------------- scheduler side
USER = 900


class _Tok:
    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return USER if t == "<｜User｜>" else None


def _sched(memory=None, block=None):
    s = Scheduler.__new__(Scheduler)
    s.tokenizer = _Tok()
    s._actual_tokenizer = None
    s.config = SimpleNamespace(enable_prefix_cache=True)
    s.memory_aware_cache = memory
    s.block_aware_cache = block
    return s


def test_cut_positions_first_and_last_user_header():
    s = _sched()
    prompt = [0, 1, 2, USER, 5, 6, 7, 1, USER, 8, 9, 1, USER, 10, 11, 12, 13]
    req = SimpleNamespace(prompt_token_ids=prompt, cached_tokens=0)
    assert s._dsv41_prefix_cuts(req) == [4, 13]
    req.cached_tokens = 9                      # the first cut is inside the restored prefix
    assert s._dsv41_prefix_cuts(req) == [13]
    req.cached_tokens = 13
    assert s._dsv41_prefix_cuts(req) == []
    short = SimpleNamespace(prompt_token_ids=[0, USER, 3], cached_tokens=0)
    assert s._dsv41_prefix_cuts(short) == []   # a cut at the last token is the N-1 entry's job


def test_cuts_only_near_the_end_of_the_prompt(monkeypatch):
    """A cut far from the end costs the late layers' pruned tail again (measured -5.5 % cold prefill at 32k with a
    3k system prompt): only cuts within VMLX_DSV41_PREFIX_CUT_MAX_TAIL tokens of the end are taken."""
    s = _sched()
    prompt = [0, 1, USER] + [5] * 700 + [1, USER] + [6] * 40 + [7, 8]
    req = SimpleNamespace(prompt_token_ids=prompt, cached_tokens=0)
    assert s._dsv41_prefix_cuts(req) == [705]                  # the first cut is 744 tokens before the end
    monkeypatch.setenv("VMLX_DSV41_PREFIX_CUT_MAX_TAIL", "1000")
    assert s._dsv41_prefix_cuts(req) == [3, 705]


def test_cut_positions_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_PREFIX_CUTS", "0")
    req = SimpleNamespace(prompt_token_ids=[0, 1, USER, 4, 5, 6], cached_tokens=0)
    assert _sched()._dsv41_prefix_cuts(req) == []


def _snap_at(n):
    rt = FakeRuntime()
    model = _model(rt)
    cache = model.make_cache()
    model(mx.array([list(range(1, n + 1))]), cache=cache)
    return cache


def test_memory_aware_store_and_partial_fetch():
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    s = _sched(memory=mem)
    prompt = list(range(1, 14))
    req = SimpleNamespace(prompt_token_ids=prompt, _cache_extra_keys=None, _bypass_prefix_cache=False,
                          _dsv41_cut_checkpoints=[(4, _snap_at(4)), (9, _snap_at(9)), (7, _snap_at(5))])
    assert s._store_dsv41_cut_checkpoints(req, "r1") == 2          # the offset-5 snapshot keyed at 7 is refused
    # a request diverging after position 9 restores the 9-token cut, not less
    other = prompt[:9] + [77, 78, 79]
    cache, remaining = mem.fetch(other)
    assert cache is not None and {int(c.offset) for c in cache} == {9}
    assert remaining == [77, 78, 79]
    # diverging after 4 (new conversation, same system prompt): the 4-token cut
    cache, remaining = mem.fetch(prompt[:4] + [50, 51])
    assert {int(c.offset) for c in cache} == {4} and remaining == [50, 51]


def test_bypass_request_stores_no_cut():
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    s = _sched(memory=mem)
    req = SimpleNamespace(prompt_token_ids=list(range(1, 14)), _cache_extra_keys=None, _bypass_prefix_cache=True,
                          _dsv41_cut_checkpoints=[(4, _snap_at(4))])
    assert s._store_dsv41_cut_checkpoints(req, "r2") == 0
    assert len(mem) == 0


def test_dspark_segments_chunk_like_a_restored_prefill(monkeypatch):
    """The short-prompt one-chunk rule applies per segment: the cold segment after a cut and the restored request's
    uncached tail get the same chunk."""
    if DSV41DSparkGenerator not in _gens():
        pytest.skip("jang_tools DSparkRunner.prefill has no cuts")
    pos, prompt = 9, list(range(20, 40))
    rt_cold, _, gen_cold = _make(DSV41DSparkGenerator, step=4)
    out_cold, first_cold = _run(gen_cold, prompt, cuts=[pos])
    rt_hit, _, gen_hit = _make(DSV41DSparkGenerator, step=4)
    snap = dict(first_cold.boundary_checkpoints)[pos]
    out_hit, _ = _run(gen_hit, prompt[pos:], caches=[mlx_model.clone_cache_list(snap)], all_tokens=[prompt[:pos]])
    after = lambda rt: [(p, ids, e) for (p, ids), e in zip(rt.forwards, rt.prompt_ends) if p >= pos]  # noqa: E731
    assert after(rt_hit) == after(rt_cold) and out_hit == out_cold


def test_generation_end_entry_and_tool_turn_hit():
    """The live cache a finished request leaves (prompt + generated tokens it consumed) is stored under exactly those
    tokens; the next tool-loop request that renders the turn back identically restores it and prefills only what
    follows."""
    mem = MemoryAwarePrefixCache(model=None, config=MemoryCacheConfig(max_memory_mb=64))
    s = _sched(memory=mem)
    prompt, output = list(range(1, 11)), [40, 41, 42, 43]
    live = _snap_at(len(prompt) + len(output) - 1)          # decode fed every output token but the last
    req = SimpleNamespace(prompt_token_ids=prompt, output_token_ids=output, _cache_extra_keys=None,
                          _bypass_prefix_cache=False)
    resp = SimpleNamespace(prompt_cache=live)
    key, cache = s._dsv41_generation_end_entry(req, resp)
    assert key == (prompt + output)[:13] and cache is live
    req._dsv41_generation_end = (key, cache)
    assert s._store_dsv41_cut_checkpoints(req, "r3") == 1
    nxt = prompt + output + [900, 70, 71, 72]               # EOS, then the tool result turn
    hit, remaining = mem.fetch(nxt)
    assert {int(c.offset) for c in hit} == {13} and remaining == [43, 900, 70, 71, 72]
    # a cache that holds the prompt only, or more than prompt + output, is not a generation-end entry
    assert s._dsv41_generation_end_entry(req, SimpleNamespace(prompt_cache=_snap_at(len(prompt)))) is None
    assert s._dsv41_generation_end_entry(req, SimpleNamespace(prompt_cache=_snap_at(15))) is None


def test_generation_end_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_GENERATION_END_CACHE", "0")
    req = SimpleNamespace(prompt_token_ids=list(range(1, 11)), output_token_ids=[40, 41])
    assert _sched()._dsv41_generation_end_entry(req, SimpleNamespace(prompt_cache=_snap_at(11))) is None


@pytest.mark.parametrize("eos_at", [1, 2, 3])
def test_dspark_stream_ending_mid_block_leaves_exactly_the_emitted_tokens(eos_at):
    """A DSpark step verifies several drafted tokens at once; when the EOS is one of them (not the last) the
    session used to keep the tokens after it, so the finished cache held more than prompt + output and could not be
    stored as the generation-end entry. The session now holds exactly the emitted tokens."""
    if DSV41DSparkGenerator not in _gens():
        pytest.skip("jang_tools DSparkRunner without the end-of-stream commit")
    prompt = [11, 3, 4, 5, 6, 7]
    eos = _expected(prompt, 6)[eos_at]
    rt = FakeRuntime()
    model = _model(rt)
    model._dspark = FakeDraft(win=4, bs=5)
    gen = DSV41DSparkGenerator(model, sampler=_greedy_sampler(), max_tokens=16, prefill_step_size=4,
                               stop_tokens=[eos])
    gen.insert([prompt], max_tokens=[16])
    out, last = [], None
    while True:
        prompt_resp, gen_resp = gen.next()
        resp = (prompt_resp or gen_resp or [None])[0]
        if resp is None:
            break
        out.append(resp.token)
        last = resp
        if resp.finish_reason is not None:
            break
    assert out[-1] == eos and len(out) == eos_at + 1
    pos = {int(c.offset) for c in last.prompt_cache}
    assert pos <= {len(prompt) + len(out)} or pos <= {len(prompt) + len(out) - 1}, pos
    assert max(pos) <= len(prompt) + len(out)

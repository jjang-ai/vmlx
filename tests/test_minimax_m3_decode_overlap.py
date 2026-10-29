# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 / M3.1 one-token decode overlap in the serving path, and the fused attention's in-place appends there.

Tiny synthetic M3 with M3.1's attention formats (tests/test_minimax_m3_fused_attn.py ``_model``: layers 1-3 sparse and
fused-capable, layer 0 dense attention), blocks of 32 keys / top-6 (sparse selection from 193 keys on) unless noted.

* Qualification of ``SingleBatchGenerator._can_overlap_decode`` for the M3 cache layout, as GLM-5.3 was
  (tests/test_glm5_decode_overlap.py): from a primed cache with sparse selection active, the overlapped run equals the
  synchronous one (tokens, finish reasons, logprobs, final K / V / index lanes bit-identical) through length, stop and
  cancel, with and without VMLX_M3_FUSED_ATTN; thousands of generated tokens past the sparse switch (and past 2,048
  keys at the released 128 / top-16 geometry) token for token.
* Lane ownership in serving: through the real Scheduler (prefix cache, zero-copy prompt marker, terminal store) on the
  plain lane and the DSpark text lane, steady decode appends in place (one copy-on-write append per layer when decode
  starts on lanes the prefill exposed). Control: a per-step ``cache.state`` read (what the old MSA sync did) turns
  every append copy-on-write.
* DSpark AR bursts carried across burst boundaries (dspark/generate.py): greedy DSpark == AR with carries taken and
  carries rolled back.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

from tests import test_minimax_m3_dspark as TD  # noqa: E402
from tests.test_minimax_m3_fused_attn import _model as fused_model  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache, clone_minimax_m3_sparse  # noqa: E402
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator  # noqa: E402


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("VMLX_M3_DECODE_OVERLAP", "VMLX_M3_DECODE_OVERLAP_DEFER", "VMLX_M3_AFFINE2_SWITCH", "VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK",
                 "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH", "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE",
                 "VMLX_DSPARK_AR_FALLBACK", "VMLX_DSPARK_AR_CARRY", "VMLX_MINIMAX_MSA", "VMLX_MINIMAX_MSA_SELECT",
                 "VMLX_M3_ALIGNED_PREFIX", "VMLX_DISABLE_GEN_PROMPT_STRIP", F.ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def model():
    return fused_model(seed=3)


def _greedy():
    from vmlx_engine.sampling import make_minimax_m3_sampler

    sampler = make_minimax_m3_sampler(temp=0.0)
    sampler._vmlx_greedy = True
    return sampler


def _ids(n: int, seed: int) -> list:
    return np.random.default_rng(seed).integers(0, TD.VOCAB - 8, n).tolist()


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _prime(model, n: int, seed: int = 1):
    cache = model.make_cache()
    ids = _ids(n, seed)
    for a in range(0, n, 256):
        out = model(mx.array([ids[a:a + 256]]), cache=cache)
        mx.eval(out, [c.state for c in cache])
    return cache


def _clone(cache):
    from mlx_lm.models.cache import KVCache

    out = []
    for c in cache:
        if isinstance(c, MiniMaxM3SparseCache):
            out.append(clone_minimax_m3_sparse(c, copy_fn=lambda a: a * 1))
        else:
            k, v = c.state
            fresh = KVCache()
            fresh.state = (k * 1, v * 1)
            out.append(fresh)
    mx.eval([c.state for c in out])
    return out


def _lanes(cache):
    """Every lane's logical rows, copied (end-of-run comparison)."""
    out = []
    for c in cache:
        for a in c.state:
            if a is not None:
                out.append(a * 1)
    mx.eval(out)
    return out


# ----------------------------------------------------------------------------------------------- admission
def test_m3_overlap_requires_the_native_layout(model, monkeypatch):
    generator = SingleBatchGenerator(model)
    cache = model.make_cache()
    request = SimpleNamespace(cache=cache, output_tokens=[])
    assert not generator._can_overlap_decode(request)                  # the first step stays synchronous
    request.output_tokens = [1]
    assert generator._can_overlap_decode(request)
    request.cache = cache[:-1] + [object()]                             # an unknown layer kind
    assert not generator._can_overlap_decode(request)
    request.cache = cache
    monkeypatch.setenv("VMLX_M3_DECODE_OVERLAP", "0")
    assert not generator._can_overlap_decode(request)
    monkeypatch.delenv("VMLX_M3_DECODE_OVERLAP")
    monkeypatch.setenv("VMLX_M3_AFFINE2_SWITCH", "1")                  # the affine-2 fast path keeps its sync
    assert not generator._can_overlap_decode(request)
    monkeypatch.delenv("VMLX_M3_AFFINE2_SWITCH")
    # the admission check and the lane list never expose (un-own) a lane
    primed = _prime(model, 300)
    for c in primed:
        if isinstance(c, MiniMaxM3SparseCache):
            c.__dict__["_lane_excl"] = [True, True, True]
    request.cache = primed
    assert generator._can_overlap_decode(request)
    assert generator._m3_lane_arrays(primed)
    assert all(c.__dict__["_lane_excl"] == [True] * 3 for c in primed if isinstance(c, MiniMaxM3SparseCache))


# ----------------------------------------------------------------------------------------------- overlap == sync
def _run(model, primed, monkeypatch, *, overlap: bool, stop=None, mode="length", max_tokens=14, prompt=(19, 20),
         logprobs=True):
    generator = SingleBatchGenerator(model, sampler=_greedy(), max_tokens=max_tokens, prefill_step_size=256,
                                     stop_tokens=[] if stop is None else [stop])
    if logprobs:
        monkeypatch.setattr(generator, "_logprobs_required", lambda request: True)
    if not overlap:
        monkeypatch.setattr(generator, "_can_overlap_decode", lambda request: False)
    submits = []
    submit = generator._submit_on_stream

    def record(*values, **kw):
        submits.append(True)
        return submit(*values, **kw)

    monkeypatch.setattr(generator, "_submit_on_stream", record)
    rows = []
    with ThreadPoolExecutor(max_workers=1) as worker:
        def start():
            cache = _clone(primed) if primed is not None else None
            uid = generator.insert([list(prompt)], caches=[cache] if cache is not None else None)[0]
            return cache, uid
        cache, uid = worker.submit(start).result()
        cancelled = False
        while True:
            first, rest = worker.submit(generator.next).result()
            if not first and not rest:
                break
            rows.extend(first + rest)
            if mode == "cancel" and len(rows) >= 3 and not cancelled:
                worker.submit(generator.remove, [uid]).result()
                assert worker.submit(generator.next).result() == ([], [])
                worker.submit(generator.insert, [[21, 22]], max_tokens=[5]).result()
                cancelled = True
        lanes = worker.submit(_lanes, cache).result() if cache is not None else []
    return rows, lanes, submits


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("mode", ["length", "stop", "cancel"])
def test_m3_overlap_matches_sync(model, monkeypatch, mode, fused, defer):
    """From 700 primed keys (sparse selection on every step): tokens, finish reasons, logprobs and the final K / V /
    index lanes of the overlapped run equal the synchronous run's; the overlapped run really submits asynchronously.
    ``defer``: VMLX_M3_DECODE_OVERLAP_DEFER (the next step submitted once the previous one is released)."""
    if fused:
        monkeypatch.setenv(F.ENV, "1")
    if defer:
        monkeypatch.setenv("VMLX_M3_DECODE_OVERLAP_DEFER", "1")
    primed = _prime(model, 700)
    stop = _run(model, primed, monkeypatch, overlap=False)[0][2].token if mode == "stop" else None
    F.STATS.update(calls=0)
    expected, expected_lanes, _ = _run(model, primed, monkeypatch, overlap=False, stop=stop, mode=mode)
    assert (F.STATS["calls"] > 0) == fused
    actual, actual_lanes, submits = _run(model, primed, monkeypatch, overlap=True, stop=stop, mode=mode)
    assert [(r.token, r.finish_reason) for r in actual] == [(r.token, r.finish_reason) for r in expected]
    assert len({r.token for r in expected}) > 2
    for left, right in zip(actual, expected):
        assert _eq(left.logprobs, right.logprobs)
    assert len(actual_lanes) == len(expected_lanes) > 0
    for a, b in zip(actual_lanes, expected_lanes):
        assert a.shape == b.shape and _eq(a, b)
    if mode != "stop":
        assert submits, "the overlapped path must submit asynchronously"


@pytest.mark.parametrize("fused,defer", [(False, False), (True, False), (False, True)])
def test_m3_long_generation_overlap_equals_sync_past_the_sparse_switch(model, monkeypatch, fused, defer):
    """2,000 generated tokens from a 150-token prompt (dense up to 192 keys, sparse after): overlapped == synchronous
    token for token, with the lanes crossing several 256-row growths."""
    if fused:
        monkeypatch.setenv(F.ENV, "1")
    if defer:
        monkeypatch.setenv("VMLX_M3_DECODE_OVERLAP_DEFER", "1")
    prompt = _ids(150, 77)
    sync, sync_lanes, _ = _run(model, None, monkeypatch, overlap=False, max_tokens=2000, prompt=prompt,
                               logprobs=False)
    over, over_lanes, submits = _run(model, None, monkeypatch, overlap=True, max_tokens=2000, prompt=prompt,
                                     logprobs=False)
    assert [r.token for r in over] == [r.token for r in sync] and len(sync) == 2000
    assert len(set(r.token for r in sync[-500:])) > 5                  # not collapsed into a short loop
    assert len(submits) > 1900


@pytest.fixture(scope="module")
def released_model():
    return fused_model(seed=4, block=128, topk=16)


@pytest.mark.parametrize("fused", [False, True])
def test_m3_overlap_equals_sync_across_2048_keys_at_the_released_geometry(released_model, monkeypatch, fused):
    """Blocks of 128 keys, top-16 (M3.1): a 1,990-token prompt generating 300 tokens crosses 2,048 keys (the sparse
    switch) at step 58; overlapped == synchronous."""
    if fused:
        monkeypatch.setenv(F.ENV, "1")
    prompt = _ids(1990, 5)
    sync, _, _ = _run(released_model, None, monkeypatch, overlap=False, max_tokens=300, prompt=prompt,
                      logprobs=False)
    over, _, submits = _run(released_model, None, monkeypatch, overlap=True, max_tokens=300, prompt=prompt,
                            logprobs=False)
    assert [r.token for r in over] == [r.token for r in sync] and len(sync) == 300 and submits


# ----------------------------------------------------------------------------------------------- overlap lane copies
class OverlapLaneCopies(Warning):
    """Reported, not failed: lanes the one-token decode overlap copied in ``test_overlap_lane_copies_are_tracked``."""


@pytest.fixture(scope="module")
def wide_model():
    """16 heads over 8 KV groups of 256 (as tests/test_minimax_m3_streamed_store.py): a 4,000-token cache is 134 MB of
    lanes, one K or V lane 16 MB, far above a decode step's few MB of transients."""
    saved = (TD.HEADS, TD.KVH, TD.HD)
    TD.HEADS, TD.KVH, TD.HD = 16, 8, 256
    try:
        yield TD._model(seed=5, block=128, topk=16)
    finally:
        TD.HEADS, TD.KVH, TD.HD = saved


def _decode_peaks(model, prompt, steps=40):
    """One request, nothing else holding the lanes: (tokens, per-step peak above the memory at the first decode token,
    memory held after the decode, the largest K lane's bytes, every lane's bytes). A step's peak covers its call: the next step built and submitted
    (a lane copy is allocated then, while the old lane is still retained) and the current one read back."""
    gen = SingleBatchGenerator(model, sampler=_greedy(), max_tokens=steps + 1, prefill_step_size=2048, stop_tokens=[])
    gen._clone_admissible_prompt_cache_snapshot = lambda cache: None
    gen.insert([prompt], max_tokens=[steps + 1])
    first, = gen.next()[0]
    cache = gen._request.cache
    lane = max(int(c.nbytes) for c in cache) // 2
    total = sum(int(c.nbytes) + int(getattr(c.__dict__.get("_idx_keys"), "nbytes", 0) or 0) for c in cache)
    del cache
    gen._sync()
    mx.synchronize()
    a0 = mx.get_active_memory()
    tokens, peaks = [first.token], []
    for _ in range(steps):
        start = mx.get_active_memory()
        mx.reset_peak_memory()                      # (MLX's peak restarts from 0, not from the active memory)
        resp, = gen.next()[1]
        peaks.append(max(mx.get_peak_memory(), start) - a0)
        tokens.append(resp.token)
        if resp.finish_reason is not None:
            break
    gen._sync()
    mx.synchronize()
    held = mx.get_active_memory() - a0
    del gen, first, resp
    return tokens, peaks, held, lane, total


def test_overlap_lane_copies_are_tracked(wide_model, monkeypatch):
    """The one-token decode overlap submits step k+1 while step k is in flight; a command buffer retains its inputs
    until its completion handler runs, so step k+1's stock slice-update appends can find a lane step k still reads
    and copy it whole (MLX ``is_donatable``). On this model (one step = a couple of command buffers) that happens at
    random, 0 .. ~2 lane sets per step; on the real M3.1 bundle the host's ~60 layers of graph building keep step
    k+1's appends behind step k's buffers (measured in the M3.1 campaign). This test keeps it visible: the overlap's
    copies are measured and reported (warning ``OverlapLaneCopies`` + VMLX_M3_OVERLAP_COPY_REPORT), asserted bounded
    and transient; the deferred submit (VMLX_M3_DECODE_OVERLAP_DEFER) copies nothing, and every arm decodes the same
    tokens."""
    import json
    import os
    import warnings

    prompt = _ids(4000, 7)                          # capacity 4,096 rows: 40 decode steps append without growth
    arms = {"overlap": {}, "deferred": {"VMLX_M3_DECODE_OVERLAP_DEFER": "1"}, "sync": {"VMLX_M3_DECODE_OVERLAP": "0"}}
    runs = {name: [] for name in arms}
    for _ in range(3):
        for name, env in arms.items():
            for k in ("VMLX_M3_DECODE_OVERLAP", "VMLX_M3_DECODE_OVERLAP_DEFER"):
                monkeypatch.delenv(k, raising=False)
            for k, v in env.items():
                monkeypatch.setenv(k, v)
            runs[name].append(_decode_peaks(wide_model, prompt))
    lane, lanes_total = runs["sync"][0][3], runs["sync"][0][4]
    assert lane > 8 * 1024 * 1024                   # big enough to show
    floor = min(min(r[1]) for r in runs["deferred"])

    def copies(peaks):                              # lane-equivalents above the step's own transients
        return sum(max(0, round((p - floor) / lane)) for p in peaks)

    summary = {name: {"lane_copies": [copies(r[1]) for r in rs], "max_step_peak_mb": round(max(max(r[1]) for r in rs)
               / 2 ** 20, 1), "held_mb": [round(r[2] / 2 ** 20, 1) for r in rs]} for name, rs in runs.items()}
    out = os.environ.get("VMLX_M3_OVERLAP_COPY_REPORT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps({"test": "overlap_lane_copies", "lane_mb": round(lane / 2 ** 20, 1),
                                "lanes_mb": round(lanes_total / 2 ** 20, 1), "summary": summary}) + "\n")
    if any(summary["overlap"]["lane_copies"]):
        warnings.warn(OverlapLaneCopies(f"decode overlap copied lanes: {summary}"))
    # every arm decodes the same tokens
    reference = runs["sync"][0][0]
    assert all(r[0] == reference for rs in runs.values() for r in rs)
    # the deferred submit never copies: every step's peak within a quarter lane of the arm's lowest step
    assert summary["deferred"]["lane_copies"] == [0, 0, 0], summary
    assert max(max(r[1]) for r in runs["deferred"]) < floor + lane // 4, summary
    # the overlap's copies are transient and bounded: nothing held after the decode; at most three versions of a lane
    # at once (step k+1's copy, the one step k reads, the one step k-1 read until its handler runs)
    held = [r[2] for rs in runs.values() for r in rs]
    assert max(held) - min(held) < 1024 * 1024, summary
    assert max(max(r[1]) for r in runs["overlap"]) <= floor + 3 * lanes_total, summary


# ----------------------------------------------------------------------------------------------- serving: lane ownership
class _Tok:
    clean_up_tokenization_spaces = False
    eos_token_id = None

    def decode(self, tokens, **_kw):
        return "".join(chr(65 + int(t) % 26) for t in tokens)


def _scheduler(model, tmp_path):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    return Scheduler(model, _Tok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=256, enable_prefix_cache=True, use_paged_cache=False,
        enable_block_disk_cache=True, block_disk_cache_dir=str(tmp_path), paged_cache_block_size=64,
        max_cache_blocks=4096,
    ))


_RID = [0]


def _serve(sched, prompt, max_tokens):
    from vmlx_engine.request import Request, SamplingParams

    _RID[0] += 1
    req = Request(request_id=f"own{_RID[0]}", prompt=list(prompt),
                  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0))
    req.prompt_token_ids = list(prompt)
    req.num_prompt_tokens = len(prompt)
    req._gen_prompt_len = 0
    sched.add_request(req)
    for _ in range(max_tokens + 16):
        if any(o.finished for o in sched.step().outputs):
            return list(req.output_token_ids), int(req.cached_tokens or 0)
    pytest.fail("request did not finish")


@pytest.fixture
def pinned_disk_hits(monkeypatch):
    from vmlx_engine.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_should_clean_prefill_over_disk_only",
                        lambda self, **kw: (False, {"cost_reason": "pinned_by_test"}))


def _assert_in_place(fused_layers: int, requests: int, calls_min: int):
    copies, inplace = F.STATS["copy_appends"], F.STATS["inplace"]
    # one copy-on-write append per fused layer when decode starts on lanes the stock prefill exposed
    assert copies <= fused_layers * requests, F.STATS
    assert inplace >= calls_min and inplace > 10 * copies, F.STATS


def test_serving_plain_lane_appends_in_place(model, tmp_path, monkeypatch, pinned_disk_hits):
    """Two turns through the Scheduler (block-disk prefix cache, zero-copy prompt marker, terminal store; the second
    turn restores the first's prefix): every decode step after the first of each request writes its lanes in place."""
    monkeypatch.setenv(F.ENV, "1")
    sched = _scheduler(model, tmp_path / "plain")
    F.STATS.update(calls=0, inplace=0, copy_appends=0)
    first = _ids(700, 41)
    out1, _ = _serve(sched, first, 40)
    out2, cached = _serve(sched, first + out1 + _ids(90, 42), 40)
    assert cached > 0                                                   # the second turn hit the prefix cache
    _assert_in_place(3, 2, calls_min=3 * 2 * 30)


def test_control_a_per_step_state_read_makes_every_append_copy_on_write(model, tmp_path, monkeypatch,
                                                                       pinned_disk_hits):
    """Control: reading ``cache.state`` after every step (what the removed MSA sync did) un-owns the lanes each step,
    so the assertion above fails."""
    monkeypatch.setenv(F.ENV, "1")
    real = SingleBatchGenerator._compute_next_from_input_array

    def with_state_read(self, req, input_tokens):
        real(self, req, input_tokens)
        [c.state for c in req.cache]

    monkeypatch.setattr(SingleBatchGenerator, "_compute_next_from_input_array", with_state_read)
    sched = _scheduler(model, tmp_path / "control")
    F.STATS.update(calls=0, inplace=0, copy_appends=0)
    _serve(sched, _ids(700, 41), 40)
    with pytest.raises(AssertionError):
        _assert_in_place(3, 1, calls_min=3 * 30)
    assert F.STATS["copy_appends"] >= 3 * 30


def _dspark_bundle(tmp_path):
    import json

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(TD._config()))
    TD._save_release_drafter(root / "dspark")
    return root


def test_serving_dspark_lane_appends_in_place(tmp_path, monkeypatch, pinned_disk_hits):
    """The DSpark text lane through the Scheduler: verify rounds, rollbacks and AR bursts (carried across burst
    boundaries) all append in place after the first fused call of each layer."""
    from vmlx_engine.dspark.runtime import load_dspark_runtime

    monkeypatch.setenv(F.ENV, "1")
    m = fused_model(seed=6)
    rt = load_dspark_runtime(m, _dspark_bundle(tmp_path))
    object.__setattr__(m, "_vmlx_dspark_runtime", rt)
    sched = _scheduler(m, tmp_path / "dspark")
    F.STATS.update(calls=0, inplace=0, copy_appends=0)
    prompt = _ids(700, 51)
    out, _ = _serve(sched, prompt, 60)
    gen = sched.batch_generator
    assert type(gen).__name__ == "DSparkTextGenerator"
    stats = gen.last_dspark_stats
    assert stats is not None and stats["rounds"] > 0
    assert F.STATS["calls"] > 3 * 10
    _assert_in_place(3, 1, calls_min=F.STATS["calls"] - 3)
    # the same tokens as the plain lane
    plain = _scheduler(fused_model(seed=6), tmp_path / "plain_ref")
    ref, _ = _serve(plain, prompt, 60)
    assert out == ref


# ----------------------------------------------------------------------------------------------- DSpark AR carry
@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return TD._save_release_drafter(tmp_path_factory.mktemp("m3_overlap") / "dspark")


def _ARHeavy():
    """ARFallback whose recorded costs are pinned (AR step 10 ms, draft rounds 1 token/s) whatever the machine measures:
    the policy keeps choosing AR bursts and probes a draft round every 24 AR tokens, like Adaptive on low-acceptance
    text."""
    from vmlx_engine.dspark.generate import ARFallback

    class Pinned(ARFallback):
        def record_draft(self, tokens, seconds, rows=None):
            super().record_draft(1, 1.0, rows)

        def record_ar_burst(self, steps, seconds):
            super().record_ar_burst(steps, 0.01 * steps)

    p = Pinned()
    p.record_draft(1, 1.0, rows=2)
    p.record_ar(0.01)
    return p


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("guess", ["policy", "always", "never"])
def test_dspark_ar_bursts_carried_across_boundaries_equal_ar(drafter_dir, monkeypatch, fused, guess):
    """Greedy DSpark with AR bursts == greedy AR. ``policy``: the carry follows ARFallback.continues_after (carried
    into following bursts); ``always``: a step is launched after every burst and
    rolled back whenever a draft probe follows (wrong guesses); ``never``: no carry (VMLX_DSPARK_AR_CARRY=0)."""
    from vmlx_engine.dspark import DSparkStats, generate_ar, iter_dspark, load_drafter
    from vmlx_engine.dspark.generate import ARFallback
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    if fused:
        monkeypatch.setenv(F.ENV, "1")
    if guess == "never":
        monkeypatch.setenv("VMLX_DSPARK_AR_CARRY", "0")
    if guess == "always":
        monkeypatch.setattr(ARFallback, "continues_after", lambda self, steps: True)
    target = MiniMaxM3DSparkTarget(fused_model(seed=8))
    prompt = _ids(773, 11)
    reference = generate_ar(target, prompt, 90, set(), prefill_step=256)
    drafter = TD._Scripted(load_drafter(drafter_dir), prompt + reference, TD._every_fifth)
    session, stats = {}, DSparkStats()
    out = []
    for chunk in iter_dspark(target, drafter, prompt, 90, set(), prefill_step=256, stats=stats,
                             ar_fallback=_ARHeavy(), max_draft=2 * TD.DRAFT_BLOCK, session=session):
        out.extend(chunk)
    assert out == reference
    assert stats.ar_tokens > 40 and stats.rounds >= 2
    cache = session["target_cache"]
    assert int(cache[0].offset) == len(session["forwarded"])           # no carried row left behind
    if guess == "never":
        assert session["ar_carried"] == session["ar_carry_rolled_back"] == 0
    elif guess == "policy":
        assert session["ar_carried"] > 0
    else:
        assert session["ar_carried"] > 0 and session["ar_carry_rolled_back"] > 0


def test_dspark_ar_carry_is_rolled_back_when_the_consumer_stops(drafter_dir, monkeypatch):
    """A consumer closing the stream right after a burst that launched a carry: the cache keeps exactly the forwarded
    tokens."""
    from vmlx_engine.dspark import load_drafter
    from vmlx_engine.dspark.generate import ARFallback, iter_dspark
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    monkeypatch.setattr(ARFallback, "continues_after", lambda self, steps: True)
    target = MiniMaxM3DSparkTarget(fused_model(seed=8))
    prompt = _ids(300, 12)
    drafter = load_drafter(drafter_dir)
    session = {}
    it = iter_dspark(target, drafter, prompt, 200, set(), prefill_step=256, ar_fallback=_ARHeavy(),
                     max_draft=2 * TD.DRAFT_BLOCK, session=session)
    produced = 0
    for chunk in it:
        produced += len(chunk)
        if produced > 30 and session.get("ar_carried", 0) >= 1:
            break
    it.close()
    cache = session["target_cache"]
    assert int(cache[0].offset) == len(session["forwarded"]) and session["ar_carry_rolled_back"] >= 1

# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 aligned-chunk prefix cache, end to end through the real Scheduler (block-disk-only: the M3 serving tier).

Tiny synthetic M3 (tests/test_minimax_m3_dspark.py: 4 layers, MSA on layers 1-3 with the RELEASED geometry -- blocks
of 128 keys, top-16, so selection turns sparse past 2,048 keys --, 16 query heads over 2 KV groups so the MSA prefill
kernel, the decode kernel and the selection kernels all run (counted), MoE routed + shared experts on layers 2-3, MXFP8
/ affine trunk), driven through ``Scheduler.add_request`` / ``step`` exactly like the engine drives it: the SSD
block-disk-only tier M3 serves with (plus a restart onto the same SSD store), the whole-prompt memory and prompt-SSD
tiers, and the DSpark text lane. The cold reference of every request is the same prompt with
``_bypass_prefix_cache`` (no lookup, no store) -- the live harness's control arm -- and checked against a direct model
prefill.

Exactness is bit-equality (``mx.array_equal``) of the logits the sampler sees at every generated step, plus the greedy
tokens. Prefill steps (320 = 5 x 64 but not a multiple of 128 / 256 / 2048; 2048) and every prompt / tool-output length
are chosen so that nothing divides evenly. Invariants are asserted on every restore: the restored cache's offset (K/V
and index lanes) equals the token count the generator is given, and the reused length is a chunk boundary or the
request's own store boundary.

Controls that must FAIL (fail closed: each asserts it actually reused a non-aligned prefix first): the previous
token-only key without aligned chunking, aligned chunking with a misaligned continuation start, and the aligned
cache without the cold prefill's cut at the store boundary (exact repeats).
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_minimax_m3_dspark import (  # noqa: E402
    VOCAB,
    _config,
    _drafter_config,
    _model,
    _save_release_drafter,
)
from vmlx_engine.models.minimax_m3 import prefill_geometry  # noqa: E402

GEN = [VOCAB - 5, VOCAB - 4, VOCAB - 3]          # generation-prompt suffix (stripped from cache keys)
END = [VOCAB - 6]                                # end-of-turn marker between messages
TOOL_LENS = (517, 333, 1291, 77, 2113, 845)      # tool outputs: no multiple of 64 / 128 / 2048
MAX_TOKENS = 7


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("VMLX_M3_ALIGNED_PREFIX", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
                 "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_AR_FALLBACK", "VMLX_DISABLE_GEN_PROMPT_STRIP"):
        monkeypatch.delenv(name, raising=False)
    # The SSD-only tier may pick a clean prefill over a disk hit from measured rates; on this tiny model prefill is
    # nearly free, so under a busy disk that cost choice flips run to run. It is not under test: always take the hit.
    from vmlx_engine.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_should_clean_prefill_over_disk_only",
                        lambda self, **kw: (False, {"cost_reason": "pinned_by_test"}))


def _m3_config() -> dict:
    """The DSpark tests' tiny M3 with the released MSA geometry (blocks of 128, top-16) and 16 query heads over 2 KV
    groups, so 8 heads per group: the shape the MSA prefill kernel takes (R7), next to the decode kernel and the
    selection kernels (R8)."""
    cfg = _config(128, 16)
    cfg["text_config"]["num_attention_heads"] = 16
    return cfg


@pytest.fixture(scope="module")
def model():
    import tests.test_minimax_m3_dspark as T

    original = T._config
    T._config = lambda block=128, topk=16: _m3_config()
    try:
        return _model(seed=5, block=128, topk=16)
    finally:
        T._config = original


@pytest.fixture
def msa_paths(monkeypatch):
    """Counts of the MSA attention / selection paths the model actually ran."""
    from vmlx_engine.models.minimax_m3 import msa

    counts = {"kernel": 0, "decode": 0, "gather": 0, "dense": 0, "select_kernel": 0, "select_reference": 0}
    real_choose, real_kernel_scores, real_ref_scores = msa.choose_path, msa.kernel_block_scores, \
        msa.reference_block_scores

    def choose(*a, **k):
        path = real_choose(*a, **k)
        counts[path] += 1
        return path

    def kernel_scores(*a, **k):
        counts["select_kernel"] += 1
        return real_kernel_scores(*a, **k)

    def ref_scores(*a, **k):
        counts["select_reference"] += 1
        return real_ref_scores(*a, **k)

    monkeypatch.setattr(msa, "choose_path", choose)
    monkeypatch.setattr(msa, "kernel_block_scores", kernel_scores)
    monkeypatch.setattr(msa, "reference_block_scores", ref_scores)
    return counts


def _ids(n: int, seed: int) -> list:
    return np.random.default_rng(seed).integers(0, VOCAB - 8, n).tolist()


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


class _Tok:
    clean_up_tokenization_spaces = False
    eos_token_id = None

    def decode(self, tokens, **_kw):
        return "".join(chr(65 + int(t) % 26) for t in tokens)


class _Trace:
    """Records what the sampler saw (logits per call) and what every generator insert was handed."""

    def __init__(self, monkeypatch):
        import vmlx_engine.scheduler as sched_mod
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        self.logits: list = []
        self.inserts: list = []
        real_sampler = sched_mod.make_minimax_m3_sampler
        trace = self

        def make_sampler(**kw):
            inner = real_sampler(**kw)

            def sampler(x):
                trace.logits.append(mx.array(x).astype(mx.float32))
                return inner(x)

            for k, v in vars(inner).items():
                setattr(sampler, k, v)
            return sampler

        monkeypatch.setattr(sched_mod, "make_minimax_m3_sampler", make_sampler)
        real_insert = SingleBatchGenerator.insert

        @functools.wraps(real_insert)        # keeps insert()'s signature: the scheduler probes its kwargs
        def insert(gen, prompts, *args, **kwargs):
            caches = kwargs.get("caches") or [None]
            context = (kwargs.get("all_tokens") or [[]])[0]
            cache = caches[0]
            rec = {"context": len(context or []), "prompt": len(prompts[0]),
                   "plan": (kwargs.get("m3_prefill_plans") or [None])[0], "offsets": None}
            if cache is not None:
                rec["offsets"] = sorted({int(c.offset) for c in cache})
                rec["idx_offsets"] = sorted({int(c._idx_offset) for c in cache if hasattr(c, "idx_keys")})
                rec["state_rows"] = sorted({int(a.shape[2]) for c in cache for a in c.state if a is not None})
            trace.inserts.append(rec)
            return real_insert(gen, prompts, *args, **kwargs)

        monkeypatch.setattr(SingleBatchGenerator, "insert", insert)


def _scheduler(model, tmp_path, step: int):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    return Scheduler(model, _Tok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=step, enable_prefix_cache=True, use_paged_cache=False,
        enable_block_disk_cache=True, block_disk_cache_dir=str(tmp_path), paged_cache_block_size=64,
        max_cache_blocks=4096,
    ))


_COUNTER = [0]


def _run(sched, trace, prompt, *, bypass=False, max_tokens=MAX_TOKENS) -> dict:
    """One request through add_request / step; returns tokens, sampler logits, cached tokens, insert record."""
    from vmlx_engine.request import Request, SamplingParams

    _COUNTER[0] += 1
    rid = f"r{_COUNTER[0]}{'-cold' if bypass else ''}"
    req = Request(request_id=rid, prompt=list(prompt), sampling_params=SamplingParams(max_tokens=max_tokens,
                                                                                        temperature=0.0))
    req.prompt_token_ids = list(prompt)
    req.num_prompt_tokens = len(prompt)
    req._gen_prompt_len = len(GEN)
    if bypass:
        req._bypass_prefix_cache = True
    trace.logits = []
    n_inserts = len(trace.inserts)
    sched.add_request(req)
    for _ in range(max_tokens + 8):
        out = sched.step()
        if any(o.finished for o in out.outputs):
            break
    else:
        pytest.fail(f"request {rid} did not finish")
    assert len(trace.inserts) == n_inserts + 1, "exactly one generator insert per request"
    rec = trace.inserts[-1]
    return {"tokens": list(req.output_token_ids), "logits": list(trace.logits), "cached": int(req.cached_tokens or 0),
            "insert": rec, "prompt": len(prompt), "detail": str(getattr(req, "_cache_detail", None) or "")}


def _boundary(prompt) -> int:
    return len(prompt) - len(GEN) - 1


def _assert_restore_invariants(res, prompt, step):
    """The restored cache holds exactly the cached rows (K/V and index lanes) and the reuse is a chunk boundary or
    this request's own store boundary; the generator received the aligned plan."""
    rec, cached = res["insert"], res["cached"]
    assert rec["plan"] == {"boundary": _boundary(prompt), "atomic": False}
    assert rec["context"] == cached
    if cached:
        assert rec["offsets"] == [cached] and rec["idx_offsets"] == [cached] and rec["state_rows"] == [cached], rec
        assert cached % step == 0 or cached == _boundary(prompt), (cached, step, _boundary(prompt))
        assert rec["prompt"] == len(prompt) - cached
    else:
        assert rec["offsets"] is None and rec["prompt"] == len(prompt)


def _same(a, b) -> bool:
    return a["tokens"] == b["tokens"] and len(a["logits"]) == len(b["logits"]) and all(
        _eq(x, y) for x, y in zip(a["logits"], b["logits"]))


def _first_logit_gap(a, b) -> float:
    return float(mx.max(mx.abs(a["logits"][0] - b["logits"][0])).item())


def _loop_prompts(n_turns: int = 6, base: int = 1100):
    """Prompt k+1 = prompt k (incl. its generation suffix) + assistant output (filled in by the caller) + END + tool
    output + GEN. Returns the system+first-user prefix and the tool outputs."""
    first = _ids(base, 1) + _ids(57, 2) + GEN
    tools = [_ids(n, 100 + i) for i, n in enumerate(TOOL_LENS[: n_turns - 1])]
    return first, tools


def _expected_reuse(prev_prompt, step: int, block: int = 64) -> int:
    """What the next turn may reuse of the previous turn's stored prompt: the last chunk boundary at or below the
    previous store boundary, rounded to the block that still lies wholly before the trailing (boundary-keyed) chunk."""
    trailing = prefill_geometry.trailing_chunk_start(_boundary(prev_prompt), step)
    whole_blocks = (trailing // block) * block
    return (whole_blocks // step) * step


# ----------------------------------------------------------------------------------------------- the agentic loop
@pytest.mark.parametrize("step", [320, 2048])
def test_agentic_tool_loop_reuses_aligned_prefix_bit_identically(model, tmp_path, monkeypatch, step, msa_paths):
    """(a) multi-turn append, (b) a 6-turn tool loop, (c) a partial hit diverging mid-chunk, (d) a continuation that
    crosses the 2,048-key MSA dense->sparse switch, exact repeats: every cached request equals its cold prefill bit for
    bit, and turn k reuses everything up to the last aligned boundary of turn k-1's prompt."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step)
    replays = []
    monkeypatch.setattr(sched, "_prefill_for_prompt_only_cache",
                        lambda *a, **k: replays.append(a) or pytest.fail("aligned M3 must not re-prefill to store"))
    assert sched._m3_aligned_prefix and sched.block_aware_cache._m3_aligned_step == step
    report = []
    try:
        prompt, tools = _loop_prompts()
        report.append({"msa_paths_before": dict(msa_paths)})
        prev = None
        crossed_sparse = False
        for turn in range(6):
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            assert cold["cached"] == 0 and len(cold["tokens"]) == MAX_TOKENS     # a real, non-empty baseline
            assert cold["insert"]["plan"] == warm["insert"]["plan"] and cold["insert"]["context"] == 0
            if turn in (0, 5):
                # The cold arm is a from-scratch prefill: the model's own forward over the same absolute pieces.
                ref = model.make_cache()
                for a, b in prefill_geometry.prefill_pieces(0, len(prompt), step, _boundary(prompt)):
                    last = model(mx.array([prompt[a:b]]), cache=ref)
                assert _eq(last[:, -1, :].astype(mx.float32), cold["logits"][0])
            _assert_restore_invariants(warm, prompt, step)
            assert _same(warm, cold), f"turn {turn}: cached != cold (step {step})"
            expected = 0 if prev is None else _expected_reuse(prev, step)
            assert warm["cached"] == expected, (turn, warm["cached"], expected)
            if turn:
                assert warm["cached"] > 0 or _boundary(prev) < step
            if warm["cached"] < 2048 < len(prompt):
                crossed_sparse = True
            report.append({"turn": turn + 1, "prompt": len(prompt), "cached": warm["cached"],
                           "prefill": len(prompt) - warm["cached"],
                           "prev_boundary": None if prev is None else _boundary(prev)})
            prev = prompt
            if turn < 5:
                prompt = prompt + warm["tokens"] + END + tools[turn] + GEN
        assert crossed_sparse, "no continuation crossed the 2,048-key dense->sparse switch"
        assert replays == []
        # The sparse kernels really ran: the prefill chunk kernel past 5,120 keys (matrix units), the decode kernel
        # (decode steps and the boundary piece), and the selection kernel.
        from vmlx_engine.models.minimax_m3 import msa

        assert msa_paths["decode"] > 0 and msa_paths["select_kernel"] > 0, msa_paths
        if msa.nax_available():
            assert msa_paths["kernel"] > 0, msa_paths

        # Exact repeat: restores this request's own store boundary (the cold prefill cut there), bit-identical.
        rep = _run(sched, trace, prev)
        _assert_restore_invariants(rep, prev, step)
        assert rep["cached"] == _boundary(prev)
        assert _same(rep, _run(sched, trace, prev, bypass=True))

        # (c) Partial hit: the last tool result replaced from a mid-chunk position on.
        cut = len(prev) - len(GEN) - TOOL_LENS[4] + 213
        assert cut % step and cut % 64
        variant = prev[:cut] + _ids(len(prev) - len(GEN) - cut, 777) + GEN
        part = _run(sched, trace, variant)
        _assert_restore_invariants(part, variant, step)
        assert part["cached"] == (cut // step) * step > 0
        assert _same(part, _run(sched, trace, variant, bypass=True))
        report.append({"partial_hit": True, "prompt": len(variant), "divergence": cut, "cached": part["cached"]})

        # The block chain stored by a CACHED turn equals a direct cold prefill of that prompt, row for row.
        table, _ = sched.block_aware_cache.fetch_cache("probe", prev[: len(prev) - len(GEN)])
        assert table is not None and table.num_tokens == _boundary(prev)
        restored = sched.block_aware_cache.reconstruct_cache(table)
        ref = model.make_cache()
        for a, b in prefill_geometry.prefill_pieces(0, len(prev), step, _boundary(prev)):
            mx.eval(model(mx.array([prev[a:b]]), cache=ref))
        for got, want in zip(restored, ref):
            assert int(got.offset) == _boundary(prev)
            for x, y in zip(got.state, want.state):
                if y is not None:
                    assert _eq(x, y[..., : _boundary(prev), :])
        sched.block_aware_cache.release_cache("probe")
        report.append({"msa_paths": dict(msa_paths)})
    finally:
        sched.shutdown()
    out = os.environ.get("VMLX_M3_ALIGNED_REPORT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps({"step": step, "turns": report}) + "\n")


def test_short_prompt_and_disk_restore_after_restart(model, tmp_path, monkeypatch):
    """(d) a prompt shorter than the prefill step (exact repeat reuses its boundary; an extension has no full chunk to
    reuse), and (e) a fresh scheduler on the same SSD block store (no RAM tier state) restores from disk bit-exactly."""
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step)
    try:
        short = _ids(211, 7) + GEN
        cold = _run(sched, trace, short)
        assert cold["cached"] == 0
        again = _run(sched, trace, short)
        _assert_restore_invariants(again, short, step)
        assert again["cached"] == _boundary(short) and _same(again, cold)
        longer = short + cold["tokens"] + END + _ids(97, 8) + GEN
        ext = _run(sched, trace, longer)
        _assert_restore_invariants(ext, longer, step)
        assert ext["cached"] == 0 and _same(ext, _run(sched, trace, longer, bypass=True))

        long_prompt = _ids(2390, 9) + GEN
        first = _run(sched, trace, long_prompt)
        nxt = long_prompt + first["tokens"] + END + _ids(611, 10) + GEN
        warm_ram = _run(sched, trace, nxt)
        assert warm_ram["cached"] == _expected_reuse(long_prompt, step) > 2048
    finally:
        sched.shutdown()
    # Restart: a new scheduler (empty RAM index) on the same SSD block store.
    sched = _scheduler(model, tmp_path, step)
    try:
        from_disk = _run(sched, trace, nxt)
        _assert_restore_invariants(from_disk, nxt, step)
        assert from_disk["cached"] == _boundary(nxt)          # nxt's own boundary was stored by the run above
        assert "disk" in from_disk["detail"], from_disk["detail"]
        cold = _run(sched, trace, nxt, bypass=True)
        assert _same(from_disk, cold) and _same(warm_ram, cold)
        third = nxt + cold["tokens"] + END + _ids(150, 11) + GEN
        cont = _run(sched, trace, third)
        assert cont["cached"] == _expected_reuse(nxt, step)
        assert _same(cont, _run(sched, trace, third, bypass=True))
    finally:
        sched.shutdown()


def test_prefill_step_not_a_block_multiple_still_exact(model, tmp_path, monkeypatch):
    """A prefill step that is not a multiple of the 64-token block (1000): reuse is coarser (a chunk boundary inside
    a block is cut on restore), every hit still bit-identical."""
    step = 1000
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step)
    try:
        prompt, tools = _loop_prompts(4, base=1430)
        prev = None
        reused = []
        for turn in range(4):
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            _assert_restore_invariants(warm, prompt, step)
            assert _same(warm, cold), turn
            if prev is not None:
                # The block that straddles the previous trailing-chunk start carries its boundary key, so reuse is
                # the chunk boundary below the last whole block before it (one chunk less when that start is not
                # block-aligned).
                assert warm["cached"] == _expected_reuse(prev, step)
                reused.append(warm["cached"])
            prev = prompt
            if turn < 3:
                prompt = prompt + warm["tokens"] + END + tools[turn] + GEN
        assert sum(1 for c in reused if c > 0) >= 2 and all(c % step == 0 for c in reused), reused
    finally:
        sched.shutdown()


@pytest.mark.parametrize("tier", ["memory", "prompt_disk"])
def test_whole_entry_tiers_follow_the_same_rule(model, tmp_path, monkeypatch, tier):
    """The whole-prompt tiers (memory-aware RAM; the prompt-level SSD cache after a restart) restore an entry and
    the scheduler cuts it to the last chunk boundary, or keeps it whole for an exact-geometry repeat: bit-identical."""
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    step = 320

    def make():
        return Scheduler(model, _Tok(), SchedulerConfig(
            max_num_seqs=1, prefill_step_size=step, enable_prefix_cache=True, use_paged_cache=False,
            enable_block_disk_cache=False, use_memory_aware_cache=True,
            enable_disk_cache=(tier == "prompt_disk"), disk_cache_dir=str(tmp_path / "prompt"),
        ))

    trace = _Trace(monkeypatch)
    sched = make()
    assert sched.block_aware_cache is None and sched.memory_aware_cache is not None
    try:
        prompt, tools = _loop_prompts(4)
        prev = None
        for turn in range(4):
            if tier == "prompt_disk" and prev is not None:
                sched.shutdown()
                sched = make()                       # empty RAM tier: the hit can only come from the SSD
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            _assert_restore_invariants(warm, prompt, step)
            assert _same(warm, cold), turn
            if prev is not None:
                assert warm["cached"] == (_boundary(prev) // step) * step > 0, (turn, warm["cached"])
                assert warm["detail"].split("+")[0] == ("disk" if tier == "prompt_disk" else "memory"), warm["detail"]
            prev = prompt
            if turn < 3:
                prompt = prompt + warm["tokens"] + END + tools[turn] + GEN
        if tier == "prompt_disk":
            sched.shutdown()
            sched = make()
        rep = _run(sched, trace, prev)
        _assert_restore_invariants(rep, prev, step)
        assert rep["cached"] == _boundary(prev) and _same(rep, cold)
        assert rep["detail"].split("+")[0] == ("disk" if tier == "prompt_disk" else "memory"), rep["detail"]
    finally:
        sched.shutdown()


# ----------------------------------------------------------------------------------------------- DSpark serving lane
def test_dspark_lane_cached_equals_cold(model, tmp_path, monkeypatch):
    """The M3.1 default lane (DSparkTextGenerator: the bundled drafter attached) runs the same aligned prefill: the
    first-token logits, every greedy token and the stored prompt state of a cached request equal its cold run."""
    from vmlx_engine.dspark.runtime import load_dspark_runtime

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_m3_config()))
    _save_release_drafter(root / "dspark", _drafter_config())
    rt = load_dspark_runtime(model, root)
    object.__setattr__(model, "_vmlx_dspark_runtime", rt)
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", step)
    try:
        prompt, tools = _loop_prompts(4)
        prev = None
        for turn in range(4):
            warm = _run(sched, trace, prompt, max_tokens=12)
            assert type(sched.batch_generator).__name__ == "DSparkTextGenerator"
            assert sched.batch_generator.last_dspark_stats["rounds"] > 0
            cold = _run(sched, trace, prompt, bypass=True, max_tokens=12)
            _assert_restore_invariants(warm, prompt, step)
            assert warm["tokens"] == cold["tokens"] and _eq(warm["logits"][0], cold["logits"][0]), turn
            if prev is not None:
                assert warm["cached"] == _expected_reuse(prev, step) > 0
            prev = prompt
            if turn < 3:
                prompt = prompt + warm["tokens"] + END + tools[turn] + GEN
        rep = _run(sched, trace, prev, max_tokens=12)
        assert rep["cached"] == _boundary(prev)
        assert rep["tokens"] == cold["tokens"] and _eq(rep["logits"][0], cold["logits"][0])
    finally:
        sched.shutdown()
        object.__setattr__(model, "_vmlx_dspark_runtime", None)
    # The plain lane (SingleBatchGenerator) cuts the same pieces: same first logits, same greedy tokens.
    sched = _scheduler(model, tmp_path / "ssd-plain", step)
    try:
        plain = _run(sched, trace, prev, bypass=True, max_tokens=12)
        assert type(sched.batch_generator).__name__ == "SingleBatchGenerator"
        assert plain["tokens"] == cold["tokens"] and _eq(plain["logits"][0], cold["logits"][0])
    finally:
        sched.shutdown()


# ----------------------------------------------------------------------------------------------- controls
def _control_loop(sched, trace, n_turns=6):
    prompt, tools = _loop_prompts(n_turns)
    rows = []
    for turn in range(n_turns):
        warm = _run(sched, trace, prompt)
        cold = _run(sched, trace, prompt, bypass=True)
        rows.append((len(prompt), warm["cached"], _same(warm, cold), _first_logit_gap(warm, cold),
                     warm["tokens"] == cold["tokens"]))
        if turn < n_turns - 1:
            prompt = prompt + warm["tokens"] + END + tools[turn] + GEN
    return rows


def test_control_token_only_key_without_aligned_chunking_drifts(model, tmp_path, monkeypatch):
    """The pre-07-30 behaviour (token-only block key, prefill chunked from the restored offset): turns reuse the
    previous turn's whole N-1 prefix -- and the cached output is NOT the cold output."""
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    monkeypatch.setenv("VMLX_M3_ALIGNED_PREFIX", "0")
    monkeypatch.setattr(BlockAwarePrefixCache, "_shape_scoped_cache_extra_keys",
                        lambda self, tokens, keys, **kw: keys)
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, 320)
    try:
        assert not sched._m3_aligned_prefix
        rows = _control_loop(sched, trace)
    finally:
        sched.shutdown()
    hits = [r for r in rows[1:] if r[1] > 0 and r[1] % 320]
    assert len(hits) == len(rows) - 1, rows                  # it really reused unaligned prefixes
    assert not all(r[2] for r in rows[1:]), rows             # ... and that is not bit-identical
    os.environ.get("VMLX_M3_ALIGNED_REPORT") and open(os.environ["VMLX_M3_ALIGNED_REPORT"], "a").write(
        json.dumps({"control": "token_only_unaligned", "rows": rows}) + "\n")


def test_control_misaligned_continuation_start_drifts(model, tmp_path, monkeypatch):
    """Aligned chunking, but the reuse allowed to stop anywhere (no chunk-boundary cap, no trailing-chunk key): the
    continuation starts mid-chunk and the cached output is NOT the cold output."""
    monkeypatch.setattr(prefill_geometry, "reuse_limit",
                        lambda matched, boundary, step, **kw: min(int(matched), int(boundary)))
    real_keys = prefill_geometry.cache_extra_keys
    monkeypatch.setattr(prefill_geometry, "cache_extra_keys",
                        lambda keys, *, boundary, step, **kw: real_keys(keys, boundary=step * 10 ** 6, step=step,
                                                                         **kw))
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, 320)
    try:
        assert sched._m3_aligned_prefix
        rows = _control_loop(sched, trace)
    finally:
        sched.shutdown()
    hits = [r for r in rows[1:] if r[1] > 0 and r[1] % 320]
    assert len(hits) == len(rows) - 1, rows
    assert not all(r[2] for r in rows[1:]), rows
    os.environ.get("VMLX_M3_ALIGNED_REPORT") and open(os.environ["VMLX_M3_ALIGNED_REPORT"], "a").write(
        json.dumps({"control": "aligned_misaligned_start", "rows": rows}) + "\n")


def test_control_exact_repeat_without_boundary_cut_drifts(model, tmp_path, monkeypatch):
    """The aligned cache, but the cold prefill NOT cut at the store boundary: an exact repeat restores the boundary
    and re-feeds the tail as its own forward, which the cold prefill ran inside its last chunk -- not identical."""
    from vmlx_engine.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_m3_prefill_plan", lambda self, request: {"boundary": None, "atomic": False})
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, 320)
    rows = []
    try:
        for seed, n in ((3, 1500), (4, 2711), (5, 4466)):
            prompt = _ids(n, seed) + GEN
            first = _run(sched, trace, prompt)
            rep = _run(sched, trace, prompt)
            assert first["cached"] == 0 and rep["cached"] == _boundary(prompt)
            rows.append((n, rep["cached"], _same(rep, first), _first_logit_gap(rep, first)))
    finally:
        sched.shutdown()
    assert not all(r[2] for r in rows), rows
    os.environ.get("VMLX_M3_ALIGNED_REPORT") and open(os.environ["VMLX_M3_ALIGNED_REPORT"], "a").write(
        json.dumps({"control": "exact_repeat_no_boundary_cut", "rows": rows}) + "\n")


# ----------------------------------------------------------------------------------------------- generator unit
class _FakeM3:
    """Records every forward (absolute start, tokens, media kwargs) on a real MiniMaxM3SparseCache."""

    def __init__(self):
        self.calls = []
        from types import SimpleNamespace

        self.args = SimpleNamespace(image_token_index=900, video_token_index=901)

    def make_cache(self):
        from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

        return [MiniMaxM3SparseCache()]

    def __call__(self, tokens, cache=None, **kw):
        n = int(tokens.shape[1])
        c = cache[0]
        self.calls.append((int(c.offset), tokens.tolist()[0], kw.get("pixel_values") is not None))
        k = mx.zeros((1, 1, n, 8), dtype=mx.bfloat16)
        c.update_and_fetch(k, k)
        c.update_index(k)
        return mx.zeros((1, n, 16), dtype=mx.float32)


def test_generator_cuts_at_absolute_chunks_and_the_boundary():
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    model = _FakeM3()
    gen = SingleBatchGenerator(model, max_tokens=1, sampler=lambda x: mx.array([3]), prefill_step_size=320, stream=None)
    prompt = list(range(1013))
    gen.insert([prompt], m3_prefill_plans=[{"boundary": 1009, "atomic": False}])
    gen.next()
    assert [(off, len(t)) for off, t, _ in model.calls] == [(0, 320), (320, 320), (640, 320), (960, 49), (1009, 4)]

    # A continuation after restoring 640 rows runs exactly the cold prefill's remaining pieces.
    restored = model.make_cache()
    k = mx.zeros((1, 1, 640, 8), dtype=mx.bfloat16)
    restored[0].update_and_fetch(k, k)
    restored[0].update_index(k)
    model.calls = []
    gen.insert([prompt[640:]], caches=[restored], all_tokens=[prompt[:640]],
               m3_prefill_plans=[{"boundary": 1009, "atomic": False}])
    gen.next()
    assert [(off, len(t)) for off, t, _ in model.calls] == [(640, 320), (960, 49), (1009, 4)]

    # A restored cache that does not hold the context it is given is refused as a cache mismatch.
    model.calls = []
    gen.insert([prompt[600:]], caches=[model.make_cache()], all_tokens=[prompt[:600]],
               m3_prefill_plans=[{"boundary": 1009, "atomic": False}])
    with pytest.raises(ValueError, match="cache mismatch"):
        gen.next()


def test_generator_media_prefill_is_atomic_up_to_the_boundary():
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    model = _FakeM3()
    gen = SingleBatchGenerator(model, max_tokens=1, sampler=lambda x: mx.array([3]), prefill_step_size=320, stream=None)
    prompt = list(range(400)) + [900] * 64 + list(range(500, 1049))
    gen.insert([prompt], pixel_values=[mx.zeros((64, 4))], image_grid_thw=[mx.array([[1, 8, 8]])],
               m3_prefill_plans=[{"boundary": 1009, "atomic": True}])
    gen.next()
    assert [(off, len(t), media) for off, t, media in model.calls] == [(0, 1009, True), (1009, 4, False)]
    # An exact-geometry replay of the same media request feeds only [boundary, N), without pixels.
    restored = model.make_cache()
    k = mx.zeros((1, 1, 1009, 8), dtype=mx.bfloat16)
    restored[0].update_and_fetch(k, k)
    restored[0].update_index(k)
    model.calls = []
    gen.insert([prompt[1009:]], caches=[restored], all_tokens=[prompt[:1009]],
               m3_prefill_plans=[{"boundary": 1009, "atomic": True}])
    gen.next()
    assert [(off, len(t), media) for off, t, media in model.calls] == [(1009, 4, False)]


# ----------------------------------------------------------------------------------------------- geometry unit
def test_prefill_pieces_and_reuse_limit():
    P = prefill_geometry.prefill_pieces
    assert P(0, 1013, 320, 1009) == [(0, 320), (320, 640), (640, 960), (960, 1009), (1009, 1013)]
    assert P(640, 1013, 320, 1009) == [(640, 960), (960, 1009), (1009, 1013)]
    assert P(1009, 1013, 320, 1009) == [(1009, 1013)]
    assert P(0, 700, 320, 640) == [(0, 320), (320, 640), (640, 700)]
    assert P(0, 1013, 320, 1009, atomic=True) == [(0, 1009), (1009, 1013)]
    R = prefill_geometry.reuse_limit
    assert R(1009, 1009, 320, exact_geometry=True) == 1009
    assert R(1009, 1009, 320, exact_geometry=False) == 960
    assert R(1000, 1500, 320, exact_geometry=True) == 960
    assert R(1009, 1009, 320, exact_geometry=True, atomic=True) == 1009
    assert R(960, 1009, 320, exact_geometry=True, atomic=True) == 0
    assert R(1009, 1009, 1000, exact_geometry=False, granule=64) == 0          # lcm(1000, 64) = 8000
    keys = prefill_geometry.cache_extra_keys(None, boundary=1009, step=320)
    from vmlx_engine.cache_key import cache_extra_keys_for_token_range

    assert prefill_geometry.TRAILING_KEY not in cache_extra_keys_for_token_range(keys, 896, 960)
    assert prefill_geometry.TRAILING_KEY in cache_extra_keys_for_token_range(keys, 960, 1009)
    assert prefill_geometry.TRAILING_KEY not in prefill_geometry.cache_extra_keys(None, boundary=960, step=320)
    media = prefill_geometry.cache_extra_keys({"mllm_media": "x"}, boundary=1009, step=320)
    assert media[prefill_geometry.GEOMETRY_KEY]["atomic_boundary"] == 1009

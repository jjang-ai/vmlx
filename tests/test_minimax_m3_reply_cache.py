# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3: the next turn must not re-prefill the reply the previous turn generated.

The live failure: turn 1 prompt 201 tokens, about 10,600 generated (reasoning), turn 2 prompt 10,814 with 197 cached:
8 s TTFT with the SSD cache on. The store kept only the prompt boundary, so every generated token was prefilled again
on the next turn, and a stopped turn stored nothing.

Same tiny synthetic M3 and real-Scheduler harness as tests/test_minimax_m3_structural_prefix_cache.py. Replies here
are rendered back exactly as generated, unlike that file's loop, whose re-rendered head differs on purpose.

Reused reply rows come from decode, so a turn after one is not bit-identical to a cold prefill. These tests pin what
must hold instead: every lane holds exactly the reused rows, the prompt that continues is the rest of the conversation,
cold prefill geometry is unchanged, and a reply that does not render back falls back to the previous boundary.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_minimax_m3_aligned_prefix_cache import (  # noqa: E402,F401  (fixtures are used by name)
    MAX_TOKENS,
    _defaults,
    _ids,
    _run,
    _scheduler,
    _Trace,
    model,
    msa_paths,
)
from tests.test_minimax_m3_structural_prefix_cache import (  # noqa: E402
    AI,
    BOS,
    EOS,
    MARKERS,
    NL,
    SYS,
    TOOL,
    USER,
    _boundary,
    _first_prompt,
    _gen,
    _msg,
    _pieces,
    _TurnTok,
)
from vmlx_engine.models.minimax_m3 import prefill_geometry  # noqa: E402

STEP = 320


@pytest.fixture(autouse=True)
def _reply_cache_on(_defaults, monkeypatch):
    """The imported fixture pins the bit-identical rule for the older tests; these test the default."""
    monkeypatch.delenv("VMLX_M3_GENERATION_END_CACHE", raising=False)


def _exact_next(prompt: list, out_tokens: list, body: list, role: int = USER) -> list:
    """The next prompt with the reply rendered back token for token: the previous prompt (which ends with the
    generation prompt), the generated tokens, the assistant turn's end, then the new message."""
    return list(prompt) + list(out_tokens) + [EOS, NL] + _msg(role, body) + _gen("adaptive")


def _rows_held(res: dict) -> int:
    assert res["insert"]["offsets"] == [res["cached"]], res["insert"]
    assert res["insert"]["idx_offsets"] == [res["cached"]], res["insert"]
    assert res["insert"]["state_rows"] == [res["cached"]], res["insert"]
    return res["cached"]


def _reply_floor(prompt: list, out_tokens: list) -> int:
    """Rows a finished reply leaves in the cache: the prompt and every generated token but the last."""
    return len(prompt) + len(out_tokens) - 1


@pytest.mark.parametrize("max_tokens", [MAX_TOKENS, 64, 65, 200])
def test_next_turn_restores_the_reply(model, tmp_path, monkeypatch, max_tokens):
    """Reply lengths on and off a block boundary. Only the new message is prefilled."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=max_tokens)
        assert one["cached"] == 0 and len(one["tokens"]) == max_tokens
        nxt = _exact_next(first, one["tokens"], _ids(41, 7))
        two = _run(sched, trace, nxt, max_tokens=8)
        floor = _reply_floor(first, one["tokens"])
        assert _rows_held(two) == floor, (two["cached"], floor, _boundary(first))
        assert two["insert"]["prompt"] == len(nxt) - floor
        assert len(two["tokens"]) == 8
    finally:
        sched.shutdown()


def test_third_turn_restores_the_second_reply(model, tmp_path, monkeypatch):
    """Consecutive turns stay short: the second reply is reused too, not only the first."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
        second = _exact_next(first, one["tokens"], _ids(41, 7))
        two = _run(sched, trace, second, max_tokens=90)
        third = _exact_next(second, two["tokens"], _ids(23, 8))
        three = _run(sched, trace, third, max_tokens=8)
        assert _rows_held(two) == _reply_floor(first, one["tokens"])
        assert _rows_held(three) == _reply_floor(second, two["tokens"])
    finally:
        sched.shutdown()


def test_a_reply_that_renders_differently_falls_back_to_the_previous_boundary(model, tmp_path, monkeypatch):
    """A template that does not render the reply back token for token must not restore rows it does not hold. The
    block chain stops at the first difference and reuse falls back to the previous prompt, with every lane consistent."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
        changed = list(one["tokens"])
        changed[2] = 3 if changed[2] != 3 else 4
        nxt = _exact_next(first, changed, _ids(41, 7))
        two = _run(sched, trace, nxt, max_tokens=8)
        assert _rows_held(two) == _boundary(first), (two["cached"], _boundary(first))
    finally:
        sched.shutdown()


def test_edited_last_message_keeps_the_reply(model, tmp_path, monkeypatch):
    """A partial hit: two different next messages after the same reply both reuse it."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=70)
        floor = _reply_floor(first, one["tokens"])
        a = _run(sched, trace, _exact_next(first, one["tokens"], _ids(41, 7)), max_tokens=8)
        b = _run(sched, trace, _exact_next(first, one["tokens"], _ids(77, 9)), max_tokens=8)
        assert _rows_held(a) == floor and _rows_held(b) == floor
    finally:
        sched.shutdown()


def test_restart_restores_the_reply_from_ssd(model, tmp_path, monkeypatch):
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
    finally:
        sched.shutdown()
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        two = _run(sched, trace, _exact_next(first, one["tokens"], _ids(41, 7)), max_tokens=8)
        assert _rows_held(two) == _reply_floor(first, one["tokens"])
        assert "disk" in two["detail"], two["detail"]
    finally:
        sched.shutdown()


@pytest.mark.parametrize("tier", ["paged_ram", "memory_ram"])
def test_ram_tiers_restore_the_reply(model, tmp_path, monkeypatch, tier):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    trace = _Trace(monkeypatch)
    sched = Scheduler(model, _TurnTok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=STEP, enable_prefix_cache=True, use_paged_cache=tier == "paged_ram",
        use_memory_aware_cache=tier == "memory_ram", enable_block_disk_cache=False, paged_cache_block_size=64,
        max_cache_blocks=4096,
    ))
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
        two = _run(sched, trace, _exact_next(first, one["tokens"], _ids(41, 7)), max_tokens=8)
        assert _rows_held(two) == _reply_floor(first, one["tokens"]), two["detail"]
        assert "disk" not in two["detail"], two["detail"]
    finally:
        sched.shutdown()


def test_ram_and_ssd_tier_restores_the_reply(model, tmp_path, monkeypatch):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    trace = _Trace(monkeypatch)
    sched = Scheduler(model, _TurnTok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=STEP, enable_prefix_cache=True, use_paged_cache=True,
        enable_block_disk_cache=True, block_disk_cache_dir=str(tmp_path), paged_cache_block_size=64,
        max_cache_blocks=4096,
    ))
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
        two = _run(sched, trace, _exact_next(first, one["tokens"], _ids(41, 7)), max_tokens=8)
        assert _rows_held(two) == _reply_floor(first, one["tokens"])
    finally:
        sched.shutdown()


def test_stopped_turn_keeps_its_prompt(model, tmp_path, monkeypatch):
    """The live failure: a stopped turn stored nothing, so the following turn prefilled the whole conversation again."""
    from vmlx_engine.request import Request, SamplingParams

    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        req = Request(request_id="stopped", prompt=list(first), sampling_params=SamplingParams(max_tokens=50, temperature=0.0))
        req.prompt_token_ids = list(first)
        req.num_prompt_tokens = len(first)
        req._gen_prompt_len = len(_gen("adaptive"))
        sched.add_request(req)
        for _ in range(6):
            sched.step()
        assert req.output_token_ids, "the turn must have started generating before it is stopped"
        sched.abort_request("stopped")
        sched.step()
        # The panel replaces a stopped reply with its own text, so only the prompt can match.
        nxt = list(first) + [5, 6, EOS, NL] + _msg(USER, _ids(41, 7)) + _gen("adaptive")
        two = _run(sched, trace, nxt, max_tokens=8)
        assert _rows_held(two) == _boundary(first), (two["cached"], _boundary(first))
    finally:
        sched.shutdown()


def test_stopped_turn_keeps_its_partial_reply(model, tmp_path, monkeypatch):
    """Found with the real Stop button: the panel keeps a stopped reply's text in history, so the next turn must not
    prefill those rows again. The tail the panel does not render back stops the chain at the first difference."""
    from vmlx_engine.request import Request, SamplingParams

    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        req = Request(request_id="stopped", prompt=list(first), sampling_params=SamplingParams(max_tokens=200, temperature=0.0))
        req.prompt_token_ids = list(first)
        req.num_prompt_tokens = len(first)
        req._gen_prompt_len = len(_gen("adaptive"))
        sched.add_request(req)
        for _ in range(80):
            sched.step()
        partial = list(req.output_token_ids)
        assert len(partial) > 40, len(partial)
        sched.abort_request("stopped")
        sched.step()
        nxt = _exact_next(first, partial, _ids(41, 7))
        two = _run(sched, trace, nxt, max_tokens=8)
        assert _rows_held(two) >= len(first) + len(partial) - 2, (two["cached"], len(first), len(partial))
        assert two["cached"] < len(nxt)
    finally:
        sched.shutdown()


def test_stopped_dspark_turn_keeps_its_partial_reply(model, tmp_path, monkeypatch):
    """The default lane. DSpark commits tokens in bursts, so at a stop the cache can hold rows past the tokens emitted.
    The plain-lane test above cannot see this: it passed while the real Stop button reused only the prompt."""
    import json

    from tests.test_minimax_m3_aligned_prefix_cache import _drafter_config, _m3_config, _save_release_drafter
    from vmlx_engine.dspark.runtime import load_dspark_runtime
    from vmlx_engine.request import Request, SamplingParams

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_m3_config()))
    _save_release_drafter(root / "dspark", _drafter_config())
    object.__setattr__(model, "_vmlx_dspark_runtime", load_dspark_runtime(model, root))
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        req = Request(request_id="stopped", prompt=list(first), sampling_params=SamplingParams(max_tokens=200, temperature=0.0))
        req.prompt_token_ids = list(first)
        req.num_prompt_tokens = len(first)
        req._gen_prompt_len = len(_gen("adaptive"))
        sched.add_request(req)
        for _ in range(60):
            sched.step()
        assert type(sched.batch_generator).__name__ == "DSparkTextGenerator"
        partial = list(req.output_token_ids)
        assert len(partial) > 30, len(partial)
        live = sched.batch_generator._find(next(iter(sched.uid_to_request_id)))
        ahead = int(live.cache[0].offset) - (len(first) + len(partial))
        sched.abort_request("stopped")
        sched.step()
        nxt = _exact_next(first, partial, _ids(41, 7))
        two = _run(sched, trace, nxt, max_tokens=8)
        assert _rows_held(two) >= len(first) + len(partial) - 2, (two["cached"], len(first), len(partial), ahead)
    finally:
        sched.shutdown()
        object.__setattr__(model, "_vmlx_dspark_runtime", None)


def test_cold_prefill_cuts_do_not_change(model):
    """An assistant turn's end stays out of the cold prefill plan: only what a hit may reuse changes."""
    first = _first_prompt("adaptive")
    nxt = _exact_next(first, _ids(90, 5), _ids(41, 7))
    k = len(nxt) - len(_gen("adaptive")) - 1
    plain = prefill_geometry.turn_cuts(nxt, MARKERS, k)
    end_of_reply = len(first) + 90 + 1
    assert end_of_reply not in plain
    assert _pieces(nxt, STEP) == prefill_geometry.prefill_pieces(0, len(nxt), STEP, k, cuts=plain)


def test_reply_rule():
    """Only rows inside an assistant turn, or right after its end, may stop a hit there."""
    gen = [BOS, AI, NL]
    toks = _msg(SYS, [1, 2]) + _msg(USER, [3]) + gen + [4, 5, 6, EOS, NL] + _msg(TOOL, [7]) + gen
    start = toks.index(AI) + 2
    for matched in range(len(toks)):
        allowed = prefill_geometry.inside_assistant_turn(toks, MARKERS, matched)
        want = start < matched <= start + 4
        assert allowed == want, (matched, allowed, want)


def test_lever_restores_the_previous_store(model, tmp_path, monkeypatch):
    """The control: with the lever off the live failure comes back (reuse stops at the previous prompt)."""
    monkeypatch.setenv("VMLX_M3_GENERATION_END_CACHE", "0")
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, STEP, tok=_TurnTok())
    try:
        first = _first_prompt("adaptive")
        one = _run(sched, trace, first, max_tokens=90)
        two = _run(sched, trace, _exact_next(first, one["tokens"], _ids(41, 7)), max_tokens=8)
        assert _rows_held(two) == _boundary(first)
    finally:
        sched.shutdown()

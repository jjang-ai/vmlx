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


def test_ram_tier_evicts_the_oldest_chat_first(model, tmp_path, monkeypatch):
    """Eric: the RAM cache grows per turn and evicts the oldest. Real prompts are stored as "user" and the tier evicts
    every "assistant" entry before any "user" one, so a reply stored as "assistant" went before an older chat's prompt."""
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setattr(Scheduler, "_pick_cache_type_for_request", staticmethod(lambda _request: "user"))
    trace = _Trace(monkeypatch)
    sched = Scheduler(model, _TurnTok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=STEP, enable_prefix_cache=True, use_paged_cache=False,
        use_memory_aware_cache=True, enable_block_disk_cache=False,
    ))
    tier = sched.memory_aware_cache
    try:
        chats = [_msg(SYS, _ids(173, 40 + i)) + _msg(USER, _ids(41, 50 + i)) + _gen("adaptive") for i in range(3)]
        first = _run(sched, trace, chats[0], max_tokens=40)
        assert len(tier._entries) >= 2, "a prompt entry and a reply entry"
        one_entry = max(e.memory_bytes for e in tier._entries.values())
        second = _run(sched, trace, chats[1], max_tokens=40)
        # Room for four of the six entries: the third chat fits only if two go, and they should be the oldest chat's.
        tier._max_memory = int(one_entry * 4.5)
        _run(sched, trace, chats[2], max_tokens=40)
        middle = _run(sched, trace, _exact_next(chats[1], second["tokens"], _ids(41, 7)), max_tokens=8)
        assert _rows_held(middle) == _reply_floor(chats[1], second["tokens"]), "the middle chat must keep its reply"
        oldest = _run(sched, trace, _exact_next(chats[0], first["tokens"], _ids(41, 7)), max_tokens=8)
        assert oldest["cached"] < _boundary(chats[0]), ("the oldest chat must be gone", oldest["cached"])
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


def _media_run(sched, trace, prompt, digest, *, max_tokens):
    """One image request. The vision splice itself has its own tests; these are about which rows may be reused."""
    from vmlx_engine.request import Request, SamplingParams

    req = Request(request_id=f"img-{len(trace.inserts)}", prompt=list(prompt),
                  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0))
    req.prompt_token_ids = list(prompt)
    req.num_prompt_tokens = len(prompt)
    req._gen_prompt_len = len(_gen("adaptive"))
    req._cache_extra_keys = {"mllm_media": digest}
    sched.add_request(req)
    for _ in range(max_tokens + 8):
        if any(o.finished for o in sched.step().outputs):
            break
    else:
        pytest.fail("image request did not finish")
    return {"tokens": list(req.output_token_ids), "cached": int(req.cached_tokens or 0), "insert": trace.inserts[-1]}


def _image_chat(monkeypatch):
    img = 77
    monkeypatch.setattr(prefill_geometry, "MEDIA_TOKEN_IDS", (img,))
    return img, _msg(SYS, _ids(173, 3)) + _msg(USER, _ids(61, 5) + [img] * 16 + _ids(23, 6)) + _gen("adaptive")


@pytest.mark.parametrize("step", [STEP, 64])
def test_turn_after_an_image_reuses_the_rows_that_hold_it(model, tmp_path, monkeypatch, step):
    """Same image still in history: only the new text is prefilled. 678 tokens, 0 cached on the real model before."""
    _, first = _image_chat(monkeypatch)
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", step, tok=_TurnTok())
    try:
        one = _media_run(sched, trace, first, "image-a", max_tokens=40)
        second = _exact_next(first, one["tokens"], _ids(41, 7))
        two = _media_run(sched, trace, second, "image-a", max_tokens=8)
        assert two["cached"] == _reply_floor(first, one["tokens"]), two["cached"]
        third = _exact_next(second, two["tokens"], _ids(19, 8))
        three = _media_run(sched, trace, third, "image-a", max_tokens=8)
        assert three["cached"] == _reply_floor(second, two["tokens"]), three["cached"]
    finally:
        sched.shutdown()


def test_a_different_image_with_the_same_tokens_reuses_nothing(model, tmp_path, monkeypatch):
    """The dangerous shape: same size, so same placeholder tokens. Rows from the first image must never answer for it."""
    _, first = _image_chat(monkeypatch)
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", STEP, tok=_TurnTok())
    try:
        one = _media_run(sched, trace, first, "image-a", max_tokens=40)
        second = _exact_next(first, one["tokens"], _ids(41, 7))
        # The next turn, with another image's digest: every row it could find belongs to image-a.
        assert _media_run(sched, trace, second, "image-b", max_tokens=4)["cached"] == 0
        # The same prompt again, with yet another image.
        assert _media_run(sched, trace, first, "image-c", max_tokens=4)["cached"] == 0
        # A second image in a later turn changes the digest: the whole conversation is prefilled, as before.
        assert _media_run(sched, trace, second, "image-a+image-d", max_tokens=4)["cached"] == 0
        # Rows of image-a are still there for image-a, so the zeros above are not an empty cache.
        assert _media_run(sched, trace, second, "image-a", max_tokens=4)["cached"] == _reply_floor(first, one["tokens"])
    finally:
        sched.shutdown()


def test_a_hit_that_ends_before_the_image_is_not_used(model, tmp_path, monkeypatch):
    """Those rows do not hold the image, and the vision tensor has to meet every placeholder in one forward."""
    img, first = _image_chat(monkeypatch)
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", STEP, tok=_TurnTok())
    try:
        _media_run(sched, trace, first, "image-a", max_tokens=8)
        at = first.index(img)
        changed = first[:at - 1] + [first[at - 1] + 1] + first[at:]
        assert _media_run(sched, trace, changed, "image-a", max_tokens=4)["cached"] == 0
    finally:
        sched.shutdown()


def _per_image_run(sched, trace, prompt, values, *, max_tokens):
    """One request carrying per-image keys, the way the engine builds them (pixels themselves stay off the toy model)."""
    import mlx.core as mx

    from vmlx_engine.request import Request, SamplingParams

    pv = mx.concatenate([mx.full((64, 4), float(v), dtype=mx.float32) for v in values], axis=0)
    grid = mx.array([[1, 8, 8]] * len(values), dtype=mx.int32)
    keys = prefill_geometry.per_image_cache_keys(prompt, pv, grid)
    assert keys is not None
    req = Request(request_id=f"pi-{len(trace.inserts)}", prompt=list(prompt),
                  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0))
    req.prompt_token_ids = list(prompt)
    req.num_prompt_tokens = len(prompt)
    req._gen_prompt_len = len(_gen("adaptive"))
    req._cache_extra_keys = keys
    sched.add_request(req)
    for _ in range(max_tokens + 8):
        if any(o.finished for o in sched.step().outputs):
            break
    else:
        pytest.fail("request did not finish")
    return {"tokens": list(req.output_token_ids), "cached": int(req.cached_tokens or 0)}


@pytest.mark.parametrize("tier", ["ssd", "ram"])
def test_a_new_image_only_prefills_the_new_part_of_the_chat(model, tmp_path, monkeypatch, tier):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    img, first = _image_chat(monkeypatch)
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", STEP, tok=_TurnTok()) if tier == "ssd" else Scheduler(model, _TurnTok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=STEP, enable_prefix_cache=True, use_paged_cache=False,
        use_memory_aware_cache=True, enable_block_disk_cache=False,
    ))
    try:
        one = _per_image_run(sched, trace, first, [1], max_tokens=40)
        floor = _reply_floor(first, one["tokens"])
        body = _ids(41, 5) + [img] * 16 + _ids(23, 6)
        second = _exact_next(first, one["tokens"], body)
        two = _per_image_run(sched, trace, second, [1, 2], max_tokens=8)
        assert floor - 64 < two["cached"] <= floor, (two["cached"], floor)
        # The same place with another picture: rows up to that image are fine, rows past it are not.
        other = _per_image_run(sched, trace, second, [1, 3], max_tokens=4)
        assert other["cached"] <= second.index(img, len(first)), (other["cached"], second.index(img, len(first)))
        # Changing the FIRST picture leaves nothing past that picture.
        assert _per_image_run(sched, trace, second, [9, 2], max_tokens=4)["cached"] <= second.index(img)
    finally:
        sched.shutdown()


def test_media_rows_are_covered_only_past_the_last_placeholder(monkeypatch):
    monkeypatch.setattr(prefill_geometry, "MEDIA_TOKEN_IDS", (9,))
    tokens = [1, 9, 9, 2, 9, 3, 4]
    assert prefill_geometry.media_rows_covered(tokens, 5) is True
    assert prefill_geometry.media_rows_covered(tokens, 4) is False
    assert prefill_geometry.media_rows_covered(tokens, 2) is False
    assert prefill_geometry.media_rows_covered([1, 2, 3], 3) is False, "no placeholder found proves nothing"
    keys = lambda boundary: prefill_geometry.cache_extra_keys({"mllm_media": "x"}, boundary=boundary, step=320)
    assert keys(1009) == keys(1500), "the next turn has to find these rows"


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

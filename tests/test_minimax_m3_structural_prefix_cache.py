# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 turn cuts: an agentic tool loop reuses the WHOLE previous prompt, bit-identically.

Same tiny synthetic M3 and real-Scheduler harness as tests/test_minimax_m3_aligned_prefix_cache.py (4 layers, MSA with
the released 128-key blocks / top-16 so selection turns sparse past 2,048 keys, MoE, block-disk-only tier), but the
prompts are rendered in the M3 chat structure with synthetic marker ids, and the scheduler's tokenizer encodes the M3
markers, so the scheduler resolves them and cuts every prefill after each non-assistant message end
(models/minimax_m3/prefill_geometry.py).

Every message is ``]~b] <role> \\n <body> [e~[ \\n``; the generation prompt is ``]~b] ai \\n`` (+ ``<mm:think>`` /
``</mm:think>`` for thinking_mode enabled / disabled: gpl 4). The previous request's store boundary K_prev (the ``\\n``
before its generation prompt) is then the end of the tool message that precedes the next prompt's previous assistant
turn: a turn cut of the next prompt. Turn k must reuse EXACTLY K_{k-1} rows (asserted as a number) and produce the
cold prefill's logits at every step, greedy tokens and stored rows bit for bit.

Controls: the chunk-only rule on the same loop (its reuse numbers: the gain), reuse at K_prev with the cold prefill NOT
cut there (must differ), and the lookahead form of the cut (``]~b]ai\\n`` after it) on a branch that diverges right
after a cut (must differ; the lookbehind rule on the same branch is identical).
"""

from __future__ import annotations

import json
import os

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_minimax_m3_aligned_prefix_cache import (  # noqa: E402,F401  (fixtures are used by name)
    END,
    GEN,
    MAX_TOKENS,
    VOCAB,
    _defaults,
    _eq,
    _first_logit_gap,
    _ids,
    _m3_config,
    _run,
    _same,
    _scheduler,
    _Tok,
    _Trace,
    _drafter_config,
    _save_release_drafter,
    model,
    msa_paths,
)
from vmlx_engine.models.minimax_m3 import prefill_geometry  # noqa: E402

BOS, AI, NL = GEN                       # ]~b]  ai  \n   (the generation prompt of thinking_mode adaptive)
EOS = END[0]                            # [e~[
THINK, THINK_END = VOCAB - 2, VOCAB - 1  # <mm:think>  </mm:think>
SYS, DEV, USER, TOOL = 11, 12, 13, 14   # role tokens (content ids may repeat them: only the token after ]~b] counts)
MARKERS = prefill_geometry.TurnMarkers(bos=BOS, eos=EOS, assistant=(AI, NL))
TOOL_LENS = (517, 333, 1291, 77, 2113)  # tool outputs (with the 5-token frame no message length is a multiple of 64)
REPORT_ENV = "VMLX_M3_STRUCTURAL_REPORT"


class _TurnTok(_Tok):
    """The test tokenizer, plus the M3 marker strings encoded to the synthetic marker ids (what the scheduler probes
    to resolve its turn markers)."""

    _TEXT = (("]~b]", BOS), ("[e~[", EOS), ("ai", AI), ("\n", NL))

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for piece, tid in self._TEXT:
                if text.startswith(piece, i):
                    out.append(tid)
                    i += len(piece)
                    break
            else:
                raise ValueError(f"unencodable test text {text!r}")
        return out


@pytest.fixture(autouse=True)
def _structural_default(monkeypatch):
    monkeypatch.delenv(prefill_geometry.STRUCTURAL_ENV, raising=False)


def _report(row: dict) -> None:
    out = os.environ.get(REPORT_ENV)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------------------------- prompt rendering
def _msg(role: int, body: list) -> list:
    return [BOS, role, NL] + list(body) + [EOS, NL]


def _gen(mode: str) -> list:
    return [BOS, AI, NL] + {"enabled": [THINK], "disabled": [THINK_END]}.get(mode, [])


def _assistant(out_tokens: list, mode: str) -> list:
    """The previous generation rendered as a history assistant turn (it starts with that request's generation prompt,
    as the M3 template renders it): reasoning + tool call when thinking is enabled, the ``</mm:think>`` prefix
    otherwise. Marker ids the tiny model happened to emit are dropped (the real template never renders them)."""
    content = [t for t in out_tokens if t < VOCAB - 8] or [7]
    head = [THINK] + content[:2] + [THINK_END] if mode == "enabled" else [THINK_END]
    return [BOS, AI, NL] + head + content + [EOS, NL]


def _first_prompt(mode: str, sys_len: int = 301, dev_len: int = 707, user_len: int = 57) -> list:
    return _msg(SYS, _ids(sys_len, 1)) + _msg(DEV, _ids(dev_len, 2)) + _msg(USER, _ids(user_len, 3)) + _gen(mode)


def _gpl(mode: str) -> int:
    return len(_gen(mode))


def _next_prompt(prev: list, out_tokens: list, tool: list, mode: str) -> list:
    return prev[: len(prev) - _gpl(mode)] + _assistant(out_tokens, mode) + _msg(TOOL, tool) + _gen(mode)


def _boundary(prompt: list, mode: str = "adaptive") -> int:
    return len(prompt) - _gpl(mode) - 1


def _turn_cuts(prompt: list, mode: str = "adaptive") -> list:
    return prefill_geometry.turn_cuts(prompt, MARKERS, _boundary(prompt, mode))


def _cut_set(prompt: list, step: int, mode: str = "adaptive") -> set:
    k = _boundary(prompt, mode)
    return set(range(step, k + 1, step)) | set(_turn_cuts(prompt, mode)) | {k}


def _pieces(prompt: list, step: int, mode: str = "adaptive", start: int = 0, turn_cuts: bool = True) -> list:
    return prefill_geometry.prefill_pieces(start, len(prompt), step, _boundary(prompt, mode),
                                           cuts=_turn_cuts(prompt, mode) if turn_cuts else None)


def _chunk_floor(boundary: int, step: int, block: int = 64) -> int:
    """The chunk-only (v1) rule's reuse of a previous boundary on the block tier (its trailing-chunk blocks carry the
    boundary): the last chunk boundary below the trailing chunk's first block."""
    trailing = prefill_geometry.trailing_chunk_start(boundary, step)
    return (((trailing // block) * block) // step) * step


def _check_restore(res: dict, prompt: list, step: int, mode: str, *, turn_cut_rule: bool = True) -> None:
    """The generator got this request's plan (boundary + every turn cut), the restored cache holds exactly the reused
    rows on every lane, and the reuse is one of this request's cuts."""
    rec, cached = res["insert"], res["cached"]
    want_plan = {"boundary": _boundary(prompt, mode), "atomic": False}
    if turn_cut_rule:
        want_plan["cuts"] = _turn_cuts(prompt, mode)
    assert rec["plan"] == want_plan, rec["plan"]
    assert rec["context"] == cached
    if cached:
        assert rec["offsets"] == [cached] and rec["idx_offsets"] == [cached] and rec["state_rows"] == [cached], rec
        assert cached in _cut_set(prompt, step, mode), (cached, step)
        assert rec["prompt"] == len(prompt) - cached
    else:
        assert rec["offsets"] is None and rec["prompt"] == len(prompt)


def _direct_prefill(model, prompt: list, step: int, mode: str = "adaptive"):
    """A from-scratch prefill over the structural pieces: (cache, last-row logits)."""
    ref = model.make_cache()
    last = None
    for a, b in _pieces(prompt, step, mode):
        last = model(mx.array([prompt[a:b]]), cache=ref)
        mx.eval([c.state for c in ref])
    return ref, last[:, -1, :].astype(mx.float32)


def _assert_stored_chain_is_cold(sched, model, prompt: list, step: int, mode: str = "adaptive") -> None:
    """The blocks the scheduler stored for ``prompt`` (a cache-HIT turn's own store) equal a direct cold prefill of
    that prompt, row for row, on every lane."""
    k = _boundary(prompt, mode)
    table, _ = sched.block_aware_cache.fetch_cache("probe", prompt[: len(prompt) - _gpl(mode)])
    try:
        assert table is not None and table.num_tokens == k, (None if table is None else table.num_tokens, k)
        restored = sched.block_aware_cache.reconstruct_cache(table)
        ref, _ = _direct_prefill(model, prompt, step, mode)
        for got, want in zip(restored, ref):
            assert int(got.offset) == k
            for x, y in zip(got.state, want.state):
                if y is not None:
                    assert _eq(x, y[..., :k, :])
    finally:
        sched.block_aware_cache.release_cache("probe")


def _loop(sched, trace, model, step: int, mode: str = "adaptive", n_turns: int = 6, first=None, tools=TOOL_LENS,
          check_direct=True, max_tokens=MAX_TOKENS):
    """Runs the tool loop (each turn: cached request, then its cold bypass control). Returns (rows, last prompt, last
    cold result)."""
    prompt = list(first if first is not None else _first_prompt(mode))
    tool_bodies = [_ids(n, 100 + i) for i, n in enumerate(tools)]
    rows, prev, cold = [], None, None
    for turn in range(n_turns):
        warm = _run(sched, trace, prompt, gpl=_gpl(mode), max_tokens=max_tokens)
        cold = _run(sched, trace, prompt, bypass=True, gpl=_gpl(mode), max_tokens=max_tokens)
        assert cold["cached"] == 0 and len(cold["tokens"]) == max_tokens        # a real, non-empty baseline
        assert cold["insert"]["plan"] == warm["insert"]["plan"] and cold["insert"]["context"] == 0
        if check_direct and turn in (0, n_turns - 1):
            _, logits = _direct_prefill(model, prompt, step, mode)
            assert _eq(logits, cold["logits"][0]), "the cold arm is not the model's own structural prefill"
        _check_restore(warm, prompt, step, mode)
        assert _same(warm, cold), f"turn {turn + 1}: cached != cold (step {step}, {mode})"
        k_prev = None if prev is None else _boundary(prev, mode)
        rows.append({
            "turn": turn + 1, "prompt": len(prompt), "boundary": _boundary(prompt, mode), "prev_boundary": k_prev,
            "cached": warm["cached"], "prefill": len(prompt) - warm["cached"],
            "chunk_only_cached": 0 if prev is None else _chunk_floor(k_prev, step),
            "forwards": len(_pieces(prompt, step, mode, start=warm["cached"])),
            "chunk_only_forwards": len(prefill_geometry.prefill_pieces(
                0 if prev is None else _chunk_floor(k_prev, step), len(prompt), step, _boundary(prompt, mode))),
            "cold_forwards": len(_pieces(prompt, step, mode)),
            "chunk_only_cold_forwards": len(_pieces(prompt, step, mode, turn_cuts=False)),
            "first_logit_gap": _first_logit_gap(warm, cold),
        })
        if prev is not None:
            assert k_prev in _turn_cuts(prompt, mode), "the previous boundary must be a turn cut of the next prompt"
            assert warm["cached"] == k_prev > 0, (turn + 1, warm["cached"], k_prev)
        else:
            assert warm["cached"] == 0
        prev = prompt
        if turn < n_turns - 1:
            prompt = _next_prompt(prompt, warm["tokens"], tool_bodies[turn], mode)
    return rows, prev, cold


# ----------------------------------------------------------------------------------------------- the agentic loop
@pytest.mark.parametrize("step", [320, 2048])
def test_tool_loop_reuses_the_whole_previous_prompt(model, tmp_path, monkeypatch, step, msa_paths):
    """6-turn tool loop (short and long tool outputs; prompts below and above the step; a continuation that crosses
    the 2,048-key dense->sparse switch): turn k reuses exactly K_{k-1}, bit-identical to its cold prefill. Then exact
    repeats, partial hits diverging inside the last tool output and inside the last assistant turn, and the stored
    chain of a cache-hit turn equals a direct cold prefill row for row."""
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    monkeypatch.setattr(sched, "_prefill_for_prompt_only_cache",
                        lambda *a, **k: pytest.fail("aligned M3 must not re-prefill to store"))
    assert sched._m3_aligned_prefix and sched._m3_turn_markers == MARKERS
    assert sched.block_aware_cache._m3_turn_markers == MARKERS
    try:
        rows, last, last_cold = _loop(sched, trace, model, step)
        assert any(r["prev_boundary"] is not None and r["prev_boundary"] < 2048 < r["prompt"] for r in rows), rows
        if step == 2048:
            assert rows[1]["chunk_only_cached"] == 0 < rows[1]["cached"], "below one step the old rule reused nothing"
        # The turn cuts bought reuse on every turn the chunk-only rule could not reach.
        assert all(r["cached"] > r["chunk_only_cached"] for r in rows[1:]), rows
        from vmlx_engine.models.minimax_m3 import msa

        assert msa_paths["decode"] > 0 and msa_paths["select_kernel"] > 0 and msa_paths["dense"] > 0, msa_paths
        if msa.nax_available():
            assert msa_paths["kernel"] > 0, msa_paths

        # Exact repeat: its own boundary (a turn cut), identical.
        rep = _run(sched, trace, last)
        _check_restore(rep, last, step, "adaptive")
        assert rep["cached"] == _boundary(last) and _same(rep, last_cold)

        # Partial hits: same prefix, different suffix from inside the last tool output / the last assistant turn.
        k_prev = rows[-1]["prev_boundary"]
        tool_body_start = len(last) - len(GEN) - 2 - TOOL_LENS[4]
        partial = []
        for name, div in (("tool_output", tool_body_start + 213), ("assistant_turn", k_prev + 1 + 5)):
            assert div % 64 and div % step
            variant = last[:div] + _ids(len(last) - len(GEN) - 2 - div, 777 + div) + [EOS, NL] + GEN
            part = _run(sched, trace, variant)
            _check_restore(part, variant, step, "adaptive")
            want = max(c for c in _cut_set(variant, step) if c <= div)
            assert part["cached"] == want > 0, (name, part["cached"], want)
            assert _same(part, _run(sched, trace, variant, bypass=True))
            partial.append({"partial_hit": name, "divergence": div, "cached": part["cached"],
                            "turn_cut": part["cached"] % step != 0, "chunk_floor": (div // step) * step})
        # With the 2,048 step the divergence inside the last assistant turn falls back to the previous turn's cut.
        if step == 2048:
            assert partial[1]["turn_cut"] and partial[1]["cached"] == k_prev, partial

        _assert_stored_chain_is_cold(sched, model, last, step)
    finally:
        sched.shutdown()
    _report({"test": "tool_loop", "step": step, "turns": rows, "partial": partial, "msa_paths": dict(msa_paths)})


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_thinking_mode_generation_prefix(model, tmp_path, monkeypatch, mode):
    """thinking_mode enabled / disabled: the generation prompt ends with <mm:think> / </mm:think> (gpl 4). K = N - 5
    is still the end of the previous tool message, so the next turn reuses exactly K_prev, bit-identically."""
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        rows, last, last_cold = _loop(sched, trace, model, step, mode=mode, n_turns=4)
        rep = _run(sched, trace, last, gpl=_gpl(mode))
        _check_restore(rep, last, step, mode)
        assert rep["cached"] == _boundary(last, mode) and _same(rep, last_cold)
        _assert_stored_chain_is_cold(sched, model, last, step, mode)
    finally:
        sched.shutdown()
    _report({"test": "thinking_mode", "mode": mode, "step": step, "turns": rows})


def test_short_conversation_below_one_step(model, tmp_path, monkeypatch):
    """The live miss: a conversation shorter than the 2,048 step (live turn 2 had a 508-token previous prompt and
    reused 0). Every turn now reuses the whole previous prompt."""
    step = 2048
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        first = _msg(SYS, _ids(173, 21)) + _msg(USER, _ids(41, 22)) + _gen("adaptive")
        rows, _, _ = _loop(sched, trace, model, step, n_turns=5, first=first, tools=(91, 203, 37, 150))
        assert all(r["chunk_only_cached"] == 0 for r in rows) and all(r["cached"] > 0 for r in rows[1:]), rows
    finally:
        sched.shutdown()
    _report({"test": "short_conversation", "step": step, "turns": rows})


def test_restart_onto_the_same_block_store(model, tmp_path, monkeypatch):
    """A fresh scheduler (empty RAM index) on the same SSD block store: the next tool turn restores the previous
    boundary from disk (its terminal partial block is found through the persisted partial sizes), bit-identically."""
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        _, prev, prev_cold = _loop(sched, trace, model, step, n_turns=2, check_direct=False)
    finally:
        sched.shutdown()
    nxt = _next_prompt(prev, prev_cold["tokens"], _ids(419, 31), "adaptive")
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        warm = _run(sched, trace, nxt)
        _check_restore(warm, nxt, step, "adaptive")
        assert warm["cached"] == _boundary(prev) and _boundary(prev) % 64, warm["cached"]
        assert "disk" in warm["detail"], warm["detail"]
        assert _same(warm, _run(sched, trace, nxt, bypass=True))
    finally:
        sched.shutdown()


def test_chunk_only_store_is_not_reused_by_the_turn_cut_rule(model, tmp_path, monkeypatch):
    """An SSD block store written under the chunk-only rule (v1 keys) and read under the turn-cut rule: its rows were
    computed without the turn cuts, so none is reused (schema v2 keys), and the request equals its cold prefill."""
    step = 320
    trace = _Trace(monkeypatch)
    prompt = _first_prompt("adaptive")
    monkeypatch.setenv(prefill_geometry.STRUCTURAL_ENV, "0")
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        assert sched._m3_turn_markers is None
        _run(sched, trace, prompt)
        assert _run(sched, trace, prompt)["cached"] == _boundary(prompt)          # the v1 entry is there
    finally:
        sched.shutdown()
    monkeypatch.delenv(prefill_geometry.STRUCTURAL_ENV)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        assert sched._m3_turn_markers == MARKERS
        warm = _run(sched, trace, prompt)
        assert warm["cached"] == 0, warm["cached"]
        assert _same(warm, _run(sched, trace, prompt, bypass=True))
        assert _run(sched, trace, prompt)["cached"] == _boundary(prompt)          # its own v2 entry
    finally:
        sched.shutdown()


def test_paged_ram_tier(model, tmp_path, monkeypatch):
    """The paged RAM block tier (no SSD store): same keys, same reuse, bit-identical."""
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    step = 320
    trace = _Trace(monkeypatch)
    sched = Scheduler(model, _TurnTok(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=step, enable_prefix_cache=True, use_paged_cache=True,
        enable_block_disk_cache=False, paged_cache_block_size=64, max_cache_blocks=4096,
    ))
    try:
        assert sched.block_aware_cache is not None and sched.block_aware_cache._m3_turn_markers == MARKERS
        rows, last, last_cold = _loop(sched, trace, model, step, n_turns=4, check_direct=False)
        rep = _run(sched, trace, last)
        assert rep["cached"] == _boundary(last) and _same(rep, last_cold)
        assert "disk" not in rep["detail"], rep["detail"]
    finally:
        sched.shutdown()
    _report({"test": "paged_ram", "step": step, "turns": rows})


@pytest.mark.parametrize("aligned,structural", [("0", "1"), ("1", "0"), ("1", "1")])
def test_levers(model, tmp_path, monkeypatch, aligned, structural):
    """VMLX_M3_ALIGNED_PREFIX=0: neither aligned prefill nor turn cuts (previous behaviour); VMLX_M3_STRUCTURAL_CUTS=0:
    aligned chunks without turn cuts (v1 keys and plan); default: turn cuts in the plan, the block keys and the
    prompt-SSD scope."""
    monkeypatch.setenv(prefill_geometry.ENV, aligned)
    monkeypatch.setenv(prefill_geometry.STRUCTURAL_ENV, structural)
    sched = _scheduler(model, tmp_path, 320, tok=_TurnTok())
    try:
        assert sched._m3_aligned_prefix == (aligned == "1")
        want = MARKERS if aligned == structural == "1" else None
        assert sched._m3_turn_markers == want and sched.block_aware_cache._m3_turn_markers == want
        assert sched.block_aware_cache._m3_aligned_step == (320 if aligned == "1" else None)
        if aligned == "1":
            from vmlx_engine.request import Request, SamplingParams

            prompt = _first_prompt("adaptive")
            req = Request(request_id="lever", prompt=prompt, sampling_params=SamplingParams(max_tokens=1))
            req.prompt_token_ids = prompt
            req._gen_prompt_len = len(GEN)
            plan = sched._m3_prefill_plan(req)
            assert plan.get("cuts") == (_turn_cuts(prompt) if want else None), plan
            req._cache_extra_keys = {"mllm_media": "x"}
            media_plan = sched._m3_prefill_plan(req)
            assert media_plan["atomic"] and media_plan.get("cuts") == ([] if want else None), media_plan
    finally:
        sched.shutdown()


@pytest.mark.parametrize("tier", ["memory", "prompt_disk"])
def test_whole_entry_tiers_reuse_the_previous_boundary(model, tmp_path, monkeypatch, tier):
    """The whole-prompt tiers (memory-aware RAM; the prompt-level SSD cache, restarted every turn so a hit can only
    come from the SSD): the entry the previous turn stored at K_prev is used whole by the next turn."""
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    step = 320

    def make():
        return Scheduler(model, _TurnTok(), SchedulerConfig(
            max_num_seqs=1, prefill_step_size=step, enable_prefix_cache=True, use_paged_cache=False,
            enable_block_disk_cache=False, use_memory_aware_cache=True,
            enable_disk_cache=(tier == "prompt_disk"), disk_cache_dir=str(tmp_path / "prompt"),
        ))

    trace = _Trace(monkeypatch)
    sched = make()
    assert sched.block_aware_cache is None and sched.memory_aware_cache is not None
    assert sched._m3_turn_markers == MARKERS
    rows = []
    try:
        prompt = _first_prompt("adaptive")
        tools = [_ids(n, 100 + i) for i, n in enumerate(TOOL_LENS[:3])]
        prev = None
        for turn in range(4):
            if tier == "prompt_disk" and prev is not None:
                sched.shutdown()
                sched = make()
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            _check_restore(warm, prompt, step, "adaptive")
            assert _same(warm, cold), turn
            if prev is not None:
                assert warm["cached"] == _boundary(prev) > 0, (turn, warm["cached"], _boundary(prev))
                assert warm["detail"].split("+")[0] == ("disk" if tier == "prompt_disk" else "memory"), warm["detail"]
            rows.append({"turn": turn + 1, "prompt": len(prompt), "cached": warm["cached"],
                         "chunk_only_cached": 0 if prev is None else (_boundary(prev) // step) * step})
            prev = prompt
            if turn < 3:
                prompt = _next_prompt(prompt, warm["tokens"], tools[turn], "adaptive")
        if tier == "prompt_disk":
            sched.shutdown()
            sched = make()
        rep = _run(sched, trace, prev)
        _check_restore(rep, prev, step, "adaptive")
        assert rep["cached"] == _boundary(prev) and _same(rep, cold)
    finally:
        sched.shutdown()
    _report({"test": "whole_entry", "tier": tier, "step": step, "turns": rows})


def test_dspark_lane_reuses_the_previous_boundary(model, tmp_path, monkeypatch):
    """The M3.1 default lane (DSparkTextGenerator, bundled drafter attached) cuts the same turn pieces: every cached
    turn reuses K_prev and equals its cold run (first-token logits, every greedy token)."""
    from vmlx_engine.dspark.runtime import load_dspark_runtime

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_m3_config()))
    _save_release_drafter(root / "dspark", _drafter_config())
    rt = load_dspark_runtime(model, root)
    object.__setattr__(model, "_vmlx_dspark_runtime", rt)
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path / "ssd", step, tok=_TurnTok())
    rows = []
    try:
        prompt = _first_prompt("adaptive")
        tools = [_ids(n, 100 + i) for i, n in enumerate(TOOL_LENS[:3])]
        prev = None
        for turn in range(4):
            warm = _run(sched, trace, prompt, max_tokens=12)
            assert type(sched.batch_generator).__name__ == "DSparkTextGenerator"
            assert sched.batch_generator.last_dspark_stats["rounds"] > 0
            cold = _run(sched, trace, prompt, bypass=True, max_tokens=12)
            _check_restore(warm, prompt, step, "adaptive")
            assert warm["tokens"] == cold["tokens"] and _eq(warm["logits"][0], cold["logits"][0]), turn
            if prev is not None:
                assert warm["cached"] == _boundary(prev) > 0, (turn, warm["cached"])
            rows.append({"turn": turn + 1, "prompt": len(prompt), "cached": warm["cached"]})
            prev = prompt
            if turn < 3:
                prompt = _next_prompt(prompt, warm["tokens"], tools[turn], "adaptive")
        rep = _run(sched, trace, prev, max_tokens=12)
        assert rep["cached"] == _boundary(prev)
        assert rep["tokens"] == cold["tokens"] and _eq(rep["logits"][0], cold["logits"][0])
    finally:
        sched.shutdown()
        object.__setattr__(model, "_vmlx_dspark_runtime", None)
    # The plain lane (SingleBatchGenerator) cuts the same pieces: same first logits, same greedy tokens.
    sched = _scheduler(model, tmp_path / "ssd-plain", step, tok=_TurnTok())
    try:
        plain = _run(sched, trace, prev, bypass=True, max_tokens=12)
        assert type(sched.batch_generator).__name__ == "SingleBatchGenerator"
        assert plain["tokens"] == cold["tokens"] and _eq(plain["logits"][0], cold["logits"][0])
    finally:
        sched.shutdown()
    _report({"test": "dspark_lane", "step": step, "turns": rows})


# ----------------------------------------------------------------------------------------------- controls
@pytest.mark.parametrize("step", [320, 2048])
def test_control_chunk_only_rule_on_the_same_loop(model, tmp_path, monkeypatch, step):
    """VMLX_M3_STRUCTURAL_CUTS=0 (the v1 chunk-only rule, kept for A/B): the same loop is bit-identical but reuses only
    the last chunk boundary below each previous boundary -- the numbers the turn cuts improve on."""
    monkeypatch.setenv(prefill_geometry.STRUCTURAL_ENV, "0")
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    rows = []
    try:
        assert sched._m3_aligned_prefix and sched._m3_turn_markers is None
        prompt = _first_prompt("adaptive")
        tools = [_ids(n, 100 + i) for i, n in enumerate(TOOL_LENS)]
        prev = None
        for turn in range(6):
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            _check_restore(warm, prompt, step, "adaptive", turn_cut_rule=False)
            assert _same(warm, cold), turn
            want = 0 if prev is None else _chunk_floor(_boundary(prev), step)
            assert warm["cached"] == want, (turn, warm["cached"], want)
            rows.append({"turn": turn + 1, "prompt": len(prompt), "cached": warm["cached"],
                         "prev_boundary": None if prev is None else _boundary(prev)})
            prev = prompt
            if turn < 5:
                prompt = _next_prompt(prompt, warm["tokens"], tools[turn], "adaptive")
    finally:
        sched.shutdown()
    assert sum(r["cached"] for r in rows) < sum(r["prev_boundary"] or 0 for r in rows)
    _report({"control": "chunk_only_rule", "step": step, "turns": rows})


def test_control_reuse_at_prev_boundary_without_the_cold_cut_drifts(model, tmp_path, monkeypatch):
    """Turn-cut reuse (the cache still restores K_prev), but the prefill plans without the turn cuts: a cold prefill
    then runs K_prev inside a chunk the cached turns ended at it, and the cached output is NOT the cold output."""
    from vmlx_engine.scheduler import Scheduler

    real_plan = Scheduler._m3_prefill_plan

    def plan_without_turn_cuts(self, request):
        plan = dict(real_plan(self, request))
        plan.pop("cuts", None)
        return plan

    monkeypatch.setattr(Scheduler, "_m3_prefill_plan", plan_without_turn_cuts)
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    rows = []
    try:
        prompt = _first_prompt("adaptive")
        tools = [_ids(n, 100 + i) for i, n in enumerate(TOOL_LENS)]
        for turn in range(6):
            warm = _run(sched, trace, prompt)
            cold = _run(sched, trace, prompt, bypass=True)
            rows.append((len(prompt), warm["cached"], _same(warm, cold), _first_logit_gap(warm, cold),
                         warm["tokens"] == cold["tokens"]))
            if turn < 5:
                prompt = _next_prompt(prompt, warm["tokens"], tools[turn], "adaptive")
    finally:
        sched.shutdown()
    hits = [r for r in rows[1:] if r[1] > 0 and r[1] % step]
    assert len(hits) == len(rows) - 1, rows                  # it really reused every previous boundary (mid-chunk)
    assert not all(r[2] for r in rows[1:]), rows             # ... and without the cold cut that is not bit-identical
    _report({"control": "reuse_without_cold_turn_cut", "step": step, "rows": rows})


def _lookahead_turn_cuts(tokens, markers, end=None):
    """The rejected form: a cut at p - 1 for every ``]~b] ai \\n`` starting at p (decided by the tokens AFTER it)."""
    if markers is None:
        return []
    n = len(tokens) if end is None else min(int(end), len(tokens))
    pat = [markers.bos, *markers.assistant]
    return [p - 1 for p in range(1, len(tokens) - len(pat) + 1)
            if 0 < p - 1 <= n and list(tokens[p:p + len(pat)]) == pat]


@pytest.mark.parametrize("rule", ["lookahead", "lookbehind"])
def test_control_lookahead_cut_breaks_a_branch_that_diverges_after_it(model, tmp_path, monkeypatch, rule):
    """A conversation P (user turn ending exactly one token before a 64-token block / 320-step boundary, then an
    assistant turn and a tool result) is cached; a branch Q keeps P's first 640 tokens but continues with a user
    message where P had the assistant turn. With the lookahead cut, P was cut at 639 (an assistant turn starts at 640)
    and Q is not: Q restores 640 rows (a step multiple), which P computed in a different chunk shape -- NOT
    identical. The lookbehind rule cuts both at 639 (the token before is the user message's end): identical."""
    if rule == "lookahead":
        monkeypatch.setattr(prefill_geometry, "turn_cuts", _lookahead_turn_cuts)
    step = 320
    trace = _Trace(monkeypatch)
    sched = _scheduler(model, tmp_path, step, tok=_TurnTok())
    try:
        head = _msg(SYS, _ids(301, 41)) + _msg(USER, _ids(640 - 2 - 301 - 5 - 3, 42))
        assert len(head) == 640 and head[638] == EOS
        p = head + [BOS, AI, NL, THINK_END] + _ids(61, 43) + [EOS, NL] + _msg(TOOL, _ids(287, 44)) + GEN
        q = head + _msg(USER, _ids(353, 45)) + GEN
        first = _run(sched, trace, p)
        assert first["cached"] == 0
        warm = _run(sched, trace, q)
        cold = _run(sched, trace, q, bypass=True)
        assert warm["cached"] == 640, warm["cached"]            # it really reused across the cut
        identical = _same(warm, cold)
        gap = _first_logit_gap(warm, cold)
    finally:
        sched.shutdown()
    if rule == "lookahead":
        assert not identical, gap
    else:
        assert identical
    _report({"control": "lookahead_cut", "rule": rule, "identical": identical, "first_logit_gap": gap})


# ----------------------------------------------------------------------------------------------- units
def test_turn_cuts_rule():
    T = prefill_geometry.turn_cuts
    sys_, user, asst, tool = _msg(SYS, [1, 2]), _msg(USER, [3]), [BOS, AI, NL, 5, 6, EOS, NL], _msg(TOOL, [8, 9])
    prompt = sys_ + user + asst + tool + GEN
    ends = [len(sys_), len(sys_) + len(user), len(sys_) + len(user) + len(asst), len(prompt) - len(GEN)]
    # After system, user and tool message ends; NOT after the assistant turn's end.
    assert T(prompt, MARKERS) == [ends[0] - 1, ends[1] - 1, ends[3] - 1]
    assert T(prompt, MARKERS, _boundary(prompt)) == [ends[0] - 1, ends[1] - 1, ends[3] - 1]
    assert _boundary(prompt) == ends[3] - 1                  # K is the tool message's turn cut
    assert T(prompt, None) == []
    # An end-of-message with no open message counts as a non-assistant end; a ]~b] with a truncated role is not ai.
    assert T([4, EOS, 7], MARKERS) == [2]
    assert T([BOS, AI, 4, EOS, 7], MARKERS) == [4]
    # Prefix-determined: the cuts at or below r are those of the first r tokens, for every r.
    import random

    rng = random.Random(7)
    alphabet = [BOS, EOS, AI, NL, SYS, USER, 1, 2, 3]
    for _ in range(200):
        x = [rng.choice(alphabet) for _ in range(rng.randint(0, 40))]
        full = T(x, MARKERS)
        for r in range(len(x) + 1):
            assert T(x[:r], MARKERS) == [c for c in full if c <= r] == T(x, MARKERS, r)


def test_reuse_limit_trailing_key_and_markers():
    R = prefill_geometry.reuse_limit
    assert R(1009, 1500, 320, exact_geometry=False, cuts=[700, 1003]) == 1003
    assert R(1009, 1500, 320, exact_geometry=False, cuts=[700]) == 960
    assert R(1000, 1003, 320, exact_geometry=False, cuts=[1003]) == 960         # a cut above the restored rows
    assert R(1003, 1003, 320, exact_geometry=False, cuts=[1003]) == 1003        # own boundary = a turn cut
    assert R(1009, 1009, 320, exact_geometry=True, atomic=True, cuts=[700]) == 1009
    assert R(960, 1009, 320, exact_geometry=True, atomic=True, cuts=[700]) == 0
    assert R(1009, 1500, 1000, exact_geometry=False, granule=64, cuts=[704, 1003]) == 704
    assert prefill_geometry.trailing_chunk_start(1009, 320, [700, 1003]) == 1003
    assert prefill_geometry.trailing_chunk_start(1009, 320, [1100]) == 960
    from vmlx_engine.cache_key import cache_extra_keys_for_token_range

    prompt = _first_prompt("adaptive")
    k = _boundary(prompt)
    keys = prefill_geometry.cache_extra_keys(None, boundary=k, step=320, markers=MARKERS, tokens=prompt)
    assert keys[prefill_geometry.GEOMETRY_KEY] == {"schema": prefill_geometry.SCHEMA_TURNS, "step": 320,
                                                   "turns": MARKERS.signature()}
    assert prefill_geometry.TRAILING_KEY not in keys                          # K is a turn cut: shareable
    # K not a turn cut (the message before the generation prompt is an assistant turn): exact-geometry key from the
    # last cut below it.
    odd = prompt[: len(prompt) - len(GEN)] + [BOS, AI, NL, 5, 6, EOS, NL] + GEN
    keys = prefill_geometry.cache_extra_keys(None, boundary=_boundary(odd), step=320, markers=MARKERS, tokens=odd)
    assert keys[prefill_geometry.TRAILING_KEY] == {"boundary": _boundary(odd)}
    assert cache_extra_keys_for_token_range(keys, k - 1, k).get(prefill_geometry.TRAILING_KEY) is None
    assert prefill_geometry.TRAILING_KEY in cache_extra_keys_for_token_range(keys, k, k + 1)
    # The v1 key without markers is unchanged.
    assert prefill_geometry.cache_extra_keys(None, boundary=k, step=320)[prefill_geometry.GEOMETRY_KEY] == {
        "schema": prefill_geometry.SCHEMA, "step": 320}
    # Media requests: atomic, the turn markers change nothing.
    media = prefill_geometry.cache_extra_keys({"mllm_media": "x"}, boundary=k, step=320, markers=MARKERS,
                                              tokens=prompt)
    assert media[prefill_geometry.GEOMETRY_KEY] == {"schema": prefill_geometry.SCHEMA, "atomic_boundary": k}
    P = prefill_geometry.plan_pieces
    assert P(0, 1013, 320, {"boundary": 1009, "atomic": False, "cuts": [100, 640, 1009]}) == [
        (0, 100), (100, 320), (320, 640), (640, 960), (960, 1009), (1009, 1013)]
    assert P(0, 1013, 320, {"boundary": 1009, "atomic": True, "cuts": [100]}) == [(0, 1009), (1009, 1013)]


def test_turn_markers_from_tokenizer():
    assert prefill_geometry.turn_markers_from_tokenizer(_TurnTok()) == MARKERS
    assert prefill_geometry.turn_markers_from_tokenizer(_Tok()) is None

    class _Wrapper:                                   # mlx_lm TokenizerWrapper / processor shapes
        def __init__(self, inner):
            self._tokenizer = inner

    assert prefill_geometry.turn_markers_from_tokenizer(_Wrapper(_TurnTok())) == MARKERS

    class _Merging(_TurnTok):                         # a tokenizer that merges across the joint: not M3's encoding
        def encode(self, text, add_special_tokens=False):
            return [1] if text == "[e~[\n]~b]ai\n" else super().encode(text, add_special_tokens)

    assert prefill_geometry.turn_markers_from_tokenizer(_Merging()) is None


def test_generator_cuts_at_turn_cuts_and_media_stays_atomic():
    from tests.test_minimax_m3_aligned_prefix_cache import _FakeM3
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    fake = _FakeM3()
    gen = SingleBatchGenerator(fake, max_tokens=1, sampler=lambda x: mx.array([3]), prefill_step_size=320, stream=None)
    prompt = list(range(1013))
    plan = {"boundary": 1009, "atomic": False, "cuts": [171, 403, 1009]}
    gen.insert([prompt], m3_prefill_plans=[plan])
    gen.next()
    assert [(off, len(t)) for off, t, _ in fake.calls] == [
        (0, 171), (171, 149), (320, 83), (403, 237), (640, 320), (960, 49), (1009, 4)]
    # A continuation from the 403 turn cut runs exactly the cold prefill's remaining pieces.
    restored = fake.make_cache()
    k = mx.zeros((1, 1, 403, 8), dtype=mx.bfloat16)
    restored[0].update_and_fetch(k, k)
    restored[0].update_index(k)
    fake.calls = []
    gen.insert([prompt[403:]], caches=[restored], all_tokens=[prompt[:403]], m3_prefill_plans=[plan])
    gen.next()
    assert [(off, len(t)) for off, t, _ in fake.calls] == [(403, 237), (640, 320), (960, 49), (1009, 4)]
    # Media: [0, boundary) atomic whatever the cuts.
    fake = _FakeM3()
    gen = SingleBatchGenerator(fake, max_tokens=1, sampler=lambda x: mx.array([3]), prefill_step_size=320, stream=None)
    prompt = list(range(400)) + [900] * 64 + list(range(500, 1049))
    gen.insert([prompt], pixel_values=[mx.zeros((64, 4))], image_grid_thw=[mx.array([[1, 8, 8]])],
               m3_prefill_plans=[{"boundary": 1009, "atomic": True, "cuts": [171, 403]}])
    gen.next()
    assert [(off, len(t), media) for off, t, media in fake.calls] == [(0, 1009, True), (1009, 4, False)]

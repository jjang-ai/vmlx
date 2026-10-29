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
    # with the reservation every lane grows at most twice during the prefill (K / V and the index lane): the first
    # piece's stock growth creates the lanes, the reservation sizes them for the rest; without it, once per piece
    pre_b = [g for g in b_grows if g[1] < n]
    assert sparse and len(pre_b) <= 4 * sparse
    for key in ("_reserve_until", "_idx_reserve_until"):
        per_lane = [g for g in pre_b if g[0] == key]
        assert len(per_lane) <= 2 * sparse
        assert max(g[1] + g[3] for g in per_lane) >= n          # the last growth covers the whole prompt
    if n > 2 * step:
        assert len([g for g in a_grows if g[1] < n]) > len(pre_b)
    # the reservation is consumed: decode growth is the stock 256-row step again
    assert all(int(c.__dict__.get("_reserve_until") or 0) == 0 for c in b_cache if hasattr(c, "reserve"))


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


# ------------------------------------------------------------------------------------------- allocator cache release
@pytest.mark.parametrize("planned", [True, False])
def test_buffer_cache_released_once_after_the_prefill(model, monkeypatch, planned):
    """Default: MLX's buffer cache is kept through the M3 prefill (every chunk re-allocates the same transient sizes)
    and released once at its end; VMLX_M3_PREFILL_CLEAR_EACH_CHUNK=1 releases it after every non-final chunk (the
    previous behaviour); VMLX_PREFILL_KEEP_ALLOC=1 never. Outputs are bit-identical across the three."""
    import vmlx_engine.utils.single_batch_generator as SBG

    n, step = 2701, 320
    prompt = _ids(n, 99)
    plan = {"boundary": n - 3, "atomic": False} if planned else None
    calls, outs = {}, {}
    real = SBG.mx.clear_cache
    for mode, env in (("end", {}), ("each", {"VMLX_M3_PREFILL_CLEAR_EACH_CHUNK": "1"}),
                      ("never", {"VMLX_PREFILL_KEEP_ALLOC": "1"})):
        for k in ("VMLX_M3_PREFILL_CLEAR_EACH_CHUNK", "VMLX_PREFILL_KEEP_ALLOC", "VMLINUX_PREFILL_KEEP_ALLOC"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        count = []
        monkeypatch.setattr(SBG.mx, "clear_cache", lambda: (count.append(1), real())[1])
        logits, cache, _ = _served(model, prompt, step, plan)
        monkeypatch.setattr(SBG.mx, "clear_cache", real)
        calls[mode], outs[mode] = len(count), (logits, _lanes(cache))
    chunks = -(-n // step)
    assert calls["end"] == 1
    assert calls["never"] == 0
    assert calls["each"] >= chunks - 1
    for mode in ("each", "never"):
        assert mx.array_equal(outs[mode][0], outs["end"][0]).item()
        assert all(mx.array_equal(x, y).item() for x, y in zip(outs[mode][1], outs["end"][1]))


def test_dspark_lane_prefill_reserved_is_bit_identical(model, monkeypatch):
    """The DSpark text lane (MiniMaxM3DSparkTarget.prefill: the live app's M3.1 prefill) reserves the lanes too: last-row
    logits, tap features and every lane equal the chunk-by-chunk growth, with and without an aligned plan."""
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    target = MiniMaxM3DSparkTarget(model)
    n = 2701
    ids = mx.array([_ids(n, 4242)])
    for plan in (None, {"boundary": n - 3, "atomic": False}):
        outs = {}
        for reserve in ("0", "1"):
            monkeypatch.setenv("VMLX_M3_PREFILL_RESERVE", reserve)
            cache = model.make_cache()
            logits, feats = target.prefill(ids, cache, (1, 3), 320, keep_last=700, plan=plan)
            mx.eval(logits, feats)
            outs[reserve] = (logits, feats, _lanes(cache), [int(c.keys.shape[2]) for c in cache if hasattr(c, "reserve")])
        a, b = outs["0"], outs["1"]
        assert mx.array_equal(a[0], b[0]).item() and mx.array_equal(a[1], b[1]).item()
        assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a[2], b[2]))
        assert all(cap >= n for cap in b[3])


@pytest.mark.parametrize("keep_last", [700, 2500, 5000])
def test_dspark_lane_skips_taps_before_the_drafter_window(model, monkeypatch, keep_last):
    """Pieces that end before the drafter's window run without taps; the kept features, the last-row logits and every
    lane equal tapping every piece (VMLX_DSPARK_PREFILL_ALL_TAPS=1), and the tapped forwards are only the window's."""
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    target = MiniMaxM3DSparkTarget(model)
    n = 2701
    ids = mx.array([_ids(n, 515)])
    plan = {"boundary": n - 3, "atomic": False}
    monkeypatch.setenv("VMLX_M3_PREFILL_GROUP", "0")    # piece by piece (the spy counts _stock calls); grouped: below
    outs = {}
    for all_taps in ("1", "0"):
        monkeypatch.setenv("VMLX_DSPARK_PREFILL_ALL_TAPS", all_taps)
        tapped = []
        real = target._stock

        def spy(tokens, cache, taps, logits):
            tapped.append((int(tokens.shape[1]), bool(taps)))
            return real(tokens, cache, taps, logits)

        monkeypatch.setattr(target, "_stock", spy)
        cache = model.make_cache()
        logits, feats = target.prefill(ids, cache, (1, 3), 320, keep_last=keep_last, plan=plan)
        mx.eval(logits, feats)
        monkeypatch.setattr(target, "_stock", real)
        outs[all_taps] = (logits, feats, _lanes(cache), tapped)
    a, b = outs["1"], outs["0"]
    assert mx.array_equal(a[0], b[0]).item()
    assert a[1].shape == b[1].shape == (1, min(n, keep_last), 2 * model.args.hidden_size)
    assert mx.array_equal(a[1], b[1]).item()
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a[2], b[2]))
    assert all(t for _, t in a[3])
    rows_tapped = sum(r for r, t in b[3] if t)
    assert keep_last <= rows_tapped < keep_last + 320 + 3 or (keep_last >= n and rows_tapped == n)
    if keep_last < n - 320:
        assert any(not t for _, t in b[3])



def test_reservation_waits_for_the_lanes_and_declines_without_allocator_room(model, monkeypatch):
    """reserve_prefill: None on a cold cache (no lane shapes yet), True once the lanes exist and the allocator has room,
    False when disabled or when the reserved rows would leave MLX's allocator less than the margin below its GC
    threshold (measured on a 128 GB M5 Max at 100k: reserving pushed every chunk into the allocator's release-on-allocate
    regime). Declined, the prefill grows chunk by chunk exactly like VMLX_M3_PREFILL_RESERVE=0, same outputs."""
    from vmlx_engine.models.minimax_m3 import cache as C

    monkeypatch.delenv("VMLX_M3_PREFILL_RESERVE", raising=False)
    cold = model.make_cache()
    assert C.reserve_prefill(cold, 1000) is None
    mx.eval(model(mx.array([_ids(300, 1)]), cache=cold))
    assert all(c.row_bytes() > 0 for c in cold if hasattr(c, "reserve"))
    assert C.reserve_prefill(cold, 1000) is True
    monkeypatch.setattr(C, "allocator_headroom_bytes", lambda: 0)
    warm = model.make_cache()
    mx.eval(model(mx.array([_ids(300, 1)]), cache=warm))
    assert C.reserve_prefill(warm, 1000) is False
    monkeypatch.setenv("VMLX_M3_PREFILL_RESERVE", "0")
    assert C.reserve_prefill(warm, 1000) is False

    # a declined reservation is the stock growth: same growth events and outputs as the reservation turned off
    monkeypatch.delenv("VMLX_M3_PREFILL_RESERVE", raising=False)
    prompt = _ids(2701, 31)
    plan = {"boundary": 2698, "atomic": False}
    declined = _generate(model, prompt, 320, plan, 3, monkeypatch, "1")
    monkeypatch.undo()
    off = _generate(model, prompt, 320, plan, 3, monkeypatch, "0")
    monkeypatch.undo()
    assert declined[0] == off[0]
    assert all(mx.array_equal(x, y).item() for x, y in zip(declined[1], off[1]))
    assert [g[1:] for g in declined[3]] == [g[1:] for g in off[3]]


# ------------------------------------------------------------------------------------- layer-major piece groups
from tests.test_minimax_m3_janght import bundle as janght_bundle  # noqa: E402,F401  (module fixture: tiny JANGHT M3)

CUT_SETS = ([5, 17, 18, 40, 100, 333, 700, 961, 1000, 1290, 1937, 2400], [64, 1100, 1700], [])


def _plan_run(model, prompt, step, plan, n_tokens, monkeypatch, group: str, routed_calls=None):
    monkeypatch.setenv("VMLX_M3_PREFILL_GROUP", group)
    toks, seen, cache, _ = _generate(model, prompt, step, plan, n_tokens, monkeypatch, "1")
    return toks, seen, cache


@pytest.mark.parametrize("cuts", CUT_SETS)
def test_grouped_pieces_equal_piece_by_piece_plain_lane(model, monkeypatch, cuts):
    """The aligned plan's pieces of one chunk as one layer-major forward == piece by piece: first-token and decode
    logits, every lane, bit for bit (affine-switch tiny M3: no routed batching, only the layer-major order)."""
    n = 2701
    prompt = _ids(n, 404)
    plan = {"boundary": n - 3, "atomic": False, "cuts": cuts}
    a = _plan_run(model, prompt, 320, plan, 6, monkeypatch, "0")
    b = _plan_run(model, prompt, 320, plan, 6, monkeypatch, "1")
    assert a[0] == b[0]
    assert all(mx.array_equal(x, y).item() for x, y in zip(a[1], b[1]))
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(_lanes(a[2]), _lanes(b[2])))


def test_grouped_pieces_batch_the_jangt_routed_experts_bit_identically(janght_bundle, monkeypatch):
    """Tiny JANGHT M3 (JANGT routed experts, the sorted NAX prefill on M5): pieces of >= 18 rows of one chunk run their
    routed experts in ONE call, pieces below that (the decode kernels) alone; logits and lanes equal piece by piece,
    and the routed calls drop. Control: the same pieces without grouping make one routed call per piece."""
    from vmlx_engine.jangt import switch as SW

    if not SW.nax_prefill():
        pytest.skip("sorted JANGT prefill kernel needs NAX (M5-class GPU)")
    _, _, mem, _, _ = janght_bundle
    n = 1301
    prompt = np.random.default_rng(9).integers(0, 511, n).tolist()
    plan = {"boundary": n - 3, "atomic": False, "cuts": [7, 30, 47, 200, 333, 512 + 40, 700, 1100, 1290]}
    counts = {}
    real = SW.JTMixedSwitchGLU.routed

    def spy(self, x, indices, scores):
        counts[key] = counts.get(key, 0) + 1
        return real(self, x, indices, scores)

    monkeypatch.setattr(SW.JTMixedSwitchGLU, "routed", spy)
    out = {}
    for key in ("0", "1"):
        out[key] = _plan_run(mem, prompt, 512, plan, 5, monkeypatch, key)
    assert out["0"][0] == out["1"][0]
    assert all(mx.array_equal(x, y).item() for x, y in zip(out["0"][1], out["1"][1]))
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(_lanes(out["0"][2]), _lanes(out["1"][2])))
    assert counts["1"] < counts["0"]


@pytest.mark.parametrize("keep_last", [300, 5000])
def test_grouped_pieces_dspark_lane(model, monkeypatch, keep_last):
    """DSpark text lane: grouped pieces (taps on the window's pieces) == piece by piece: logits, kept features, lanes."""
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    target = MiniMaxM3DSparkTarget(model)
    n = 2701
    ids = mx.array([_ids(n, 77)])
    plan = {"boundary": n - 3, "atomic": False, "cuts": CUT_SETS[0]}
    res = {}
    for group in ("0", "1"):
        monkeypatch.setenv("VMLX_M3_PREFILL_GROUP", group)
        cache = model.make_cache()
        logits, feats = target.prefill(ids, cache, (1, 3), 320, keep_last=keep_last, plan=plan)
        mx.eval(logits, feats)
        res[group] = (logits, feats, _lanes(cache))
    a, b = res["0"], res["1"]
    assert mx.array_equal(a[0], b[0]).item()
    assert a[1].shape == b[1].shape and mx.array_equal(a[1], b[1]).item()
    assert all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a[2], b[2]))

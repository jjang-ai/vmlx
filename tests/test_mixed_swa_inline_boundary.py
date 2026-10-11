"""LLM-lane mixed-SWA: exact N-1 boundary without the clean re-prefill.

The generator snapshots each RotatingKVCache at the N-1 boundary (right after
the N-1 prefill, before the final prompt token). The scheduler assembles the
store cache from that snapshot plus the append-only full-attention KV sliced
to the boundary. These tests pin that the result equals a clean re-prefill of
the same N-1 tokens, and that every unprovable case declines.
"""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache  # noqa: E402

from vmlx_engine.utils.mixed_swa_boundary import (  # noqa: E402
    _window_rows,
    capture_rotating_boundary,
    compare_boundary_caches,
)


def _kv(seed, s, heads=2, dim=8):
    mx.random.seed(seed)
    return (
        mx.random.normal((1, heads, s, dim)).astype(mx.bfloat16),
        mx.random.normal((1, heads, s, dim)).astype(mx.bfloat16),
    )


def _feed(cache, sizes, seed0=0):
    for i, s in enumerate(sizes):
        k, v = _kv(seed0 + i, s)
        cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)


@pytest.mark.parametrize(
    "max_size,keep,chunks",
    [
        (8, 0, [5, 5, 3]),        # wrapped window, non-aligned total (13)
        (8, 0, [3, 2]),           # window not yet full (5 < 8)
        (8, 2, [6, 6, 1 + 3]),    # attention-sink keep rows, wrapped
        (16, 0, [7, 7, 7, 4]),    # boundary 25, not a multiple of 16 or the block
        (4, 0, [9]),              # single large chunk, overhang >> window
    ],
)
def test_capture_continues_identically_to_the_uninterrupted_ring(max_size, keep, chunks):
    live = RotatingKVCache(max_size=max_size, keep=keep)
    _feed(live, chunks)
    capture, reason = capture_rotating_boundary([live], block_size=2)
    assert reason == "captured"
    boundary, snaps = capture
    assert boundary == sum(chunks)
    snap = snaps[0]
    assert snap.offset == boundary
    assert snap.keys.shape[2] <= min(boundary, max_size) + 2 * 2
    assert snap.keys is not live.keys  # detached, window-sized copy

    # The capture must behave exactly like the ring it was taken from:
    # continue both with the final prompt token, then decode, then a chunk.
    resumed = copy.deepcopy(snap)
    reference = live
    for step, s in enumerate([1, 1, 1, 3, 1]):
        k, v = _kv(1000 + step, s)
        a = resumed.update_and_fetch(k, v)
        b = reference.update_and_fetch(k, v)
        mx.eval(*a, *b)
        ak, av = _window_rows(resumed)
        bk, bv = _window_rows(reference)
        assert resumed.offset == reference.offset
        assert bool(mx.array_equal(ak, bk)) and bool(mx.array_equal(av, bv))


def test_capture_is_immune_to_later_decode_on_the_live_ring():
    live = RotatingKVCache(max_size=8)
    _feed(live, [5, 6])
    (boundary, snaps), _ = capture_rotating_boundary([live], block_size=4)
    before = [snaps[0].keys * 1, snaps[0].values * 1]
    mx.eval(*before)
    for step in range(12):  # rolls the live ring well past the window
        k, v = _kv(500 + step, 1)
        live.update_and_fetch(k, v)
    mx.eval(live.keys, live.values)
    assert bool(mx.array_equal(snaps[0].keys, before[0]))
    assert bool(mx.array_equal(snaps[0].values, before[1]))
    assert snaps[0].offset == boundary == 11


@pytest.mark.parametrize("case", ["rolled", "disagree", "no_rotating", "empty"])
def test_capture_declines_when_the_boundary_cannot_be_proven(case):
    a = RotatingKVCache(max_size=4)
    b = RotatingKVCache(max_size=4)
    if case == "rolled":
        _feed(a, [6, 1, 1])   # single-token updates rolled the ring in place
        cache = [a]
    elif case == "disagree":
        _feed(a, [5]); _feed(b, [6])
        cache = [a, b]
    elif case == "no_rotating":
        kv = KVCache(); _feed(kv, [5])
        cache = [kv]
    else:
        cache = [a]
    capture, reason = capture_rotating_boundary(cache)
    assert capture is None and reason != "captured"


# --- real tiny mixed-SWA model through the actual generator -----------------

def _tiny_gemma3():
    from mlx_lm.models import gemma3_text as g

    mx.random.seed(7)
    model = g.Model(g.ModelArgs(
        model_type="gemma3_text", hidden_size=32, num_hidden_layers=4,
        intermediate_size=64, num_attention_heads=2, head_dim=16,
        rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=1,
        sliding_window=4, sliding_window_pattern=2,
    ))
    mx.eval(model.parameters())
    return model


def _clean_reprefill(model, tokens, step):
    cache = model.make_cache()
    for start in range(0, len(tokens), step):
        model(mx.array([tokens[start:start + step]]), cache=cache)
    mx.eval(*[x for c in cache for x in (c.keys, c.values)])
    return cache


@pytest.mark.parametrize("n_tokens,step", [(11, 4), (14, 4), (9, 3), (6, 8), (23, 5), (30, 7)])
def test_generator_capture_plus_sliced_kv_equals_clean_reprefill(n_tokens, step):
    from vmlx_engine.mllm_scheduler import _assemble_mixed_swa_boundary_cache
    from vmlx_engine.utils import single_batch_generator as sbg

    model = _tiny_gemma3()
    prompt = [int(t) % 60 + 1 for t in range(3, 3 + n_tokens)]
    generator = sbg.SingleBatchGenerator(
        model, max_tokens=6, prefill_step_size=step,
        mixed_swa_boundary_capture=True, mixed_swa_capture_block_size=2,
    )
    generator.insert([prompt], max_tokens=[6])
    first, _ = generator.next()
    capture = first[0].mixed_swa_boundary
    if (n_tokens - 1) % step == 1:
        # The final N-1 prefill chunk is a single token: the ring was rolled in
        # place, so the capture declines (same rule as the MLLM lane) and this
        # request keeps the clean re-prefill.
        assert capture is None
        assert generator.mixed_swa_boundary_last_reason == "not_temporal_window"
        assert generator.mixed_swa_boundary_declines == 1
        return
    assert capture is not None and capture[0] == n_tokens - 1
    while generator._request is not None:  # decode past the boundary (rings roll)
        _, finished = generator.next()
    live = first[0].prompt_cache
    assert all(c.offset > n_tokens - 1 for c in live)

    holder = SimpleNamespace(request_id="r", _mixed_swa_boundary=capture)
    assembled = _assemble_mixed_swa_boundary_cache(None, holder, live, n_tokens - 1)
    reference = _clean_reprefill(model, prompt[:-1], step)
    verdict = compare_boundary_caches(assembled, reference)
    assert verdict["exact"], verdict
    assert generator.mixed_swa_boundary_captures == 1


def test_capture_is_off_unless_the_scheduler_enables_it():
    from vmlx_engine.utils import single_batch_generator as sbg

    model = _tiny_gemma3()
    generator = sbg.SingleBatchGenerator(model, max_tokens=2, prefill_step_size=4)
    generator.insert([list(range(1, 10))], max_tokens=[2])
    first, _ = generator.next()
    assert first[0].mixed_swa_boundary is None


# --- scheduler assembly + materializer ---------------------------------------

def _scheduler():
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._mixed_attention_cache_model = True
    scheduler._kv_cache_bits = 0
    scheduler.disk_cache = None
    scheduler._is_hybrid = False
    scheduler.batch_generator = SimpleNamespace(mixed_swa_boundary_last_reason="captured")
    return scheduler


def _live_and_capture(boundary=11, extra=5):
    full = KVCache()
    _feed(full, [boundary])
    ring = RotatingKVCache(max_size=4)
    _feed(ring, [6, boundary - 6], seed0=50)
    capture, _ = capture_rotating_boundary([full, ring], block_size=2)
    for step in range(extra):  # decode: full grows, ring rolls
        k, v = _kv(900 + step, 1)
        full.update_and_fetch(k, v)
        ring.update_and_fetch(*_kv(950 + step, 1))
    mx.eval(full.keys, ring.keys)
    return [full, ring], capture


def test_scheduler_assembles_sliced_full_kv_and_captured_ring():
    scheduler = _scheduler()
    live, capture = _live_and_capture()
    full_before = live[0].keys[..., :11, :] * 1
    mx.eval(full_before)
    deferred = {"boundary_capture": capture, "live_cache": live}
    out = scheduler._assemble_inline_mixed_swa_boundary("r", deferred, list(range(11)), "paged")
    assert out is not None and [c.offset for c in out] == [11, 11]
    assert bool(mx.array_equal(out[0].keys, full_before))
    assert out[1] is capture[1][1]
    assert "live_cache" not in deferred and "boundary_capture" not in deferred


@pytest.mark.parametrize("case", ["mismatch", "quantized", "disabled", "missing"])
def test_scheduler_assembly_declines_to_the_clean_reprefill(case, monkeypatch):
    scheduler = _scheduler()
    live, capture = _live_and_capture()
    key = list(range(11))
    if case == "mismatch":
        key = list(range(12))
    elif case == "quantized":
        live[0] = QuantizedKVCache()
    elif case == "disabled":
        monkeypatch.setenv("VMLX_MIXED_SWA_INLINE_BOUNDARY", "0")
    elif case == "missing":
        capture = None
    deferred = {"boundary_capture": capture, "live_cache": live}
    assert scheduler._assemble_inline_mixed_swa_boundary("r", deferred, key, "paged") is None


def test_materializer_skips_the_reprefill_when_inline_engages():
    scheduler = _scheduler()
    live, capture = _live_and_capture()
    scheduler._prefill_for_prompt_only_cache = Mock(side_effect=AssertionError("re-prefill ran"))
    scheduler._mixed_swa_deferred_seed = Mock(side_effect=AssertionError("seed ran"))
    request = SimpleNamespace(request_id="r", _deferred_prompt_cache={
        "family": "Mixed-SWA", "mode": "object", "key_tokens": list(range(11)),
        "boundary_capture": capture, "live_cache": live,
    })
    scheduler._materialize_deferred_prompt_cache("r", request)
    assert [c.offset for c in request._extracted_cache] == [11, 11]
    assert request._extracted_cache_key_tokens == list(range(11))
    assert request._extracted_cache_from_prompt_snapshot is True
    assert request._deferred_prompt_cache is None


def test_materializer_still_reprefills_without_a_capture():
    scheduler = _scheduler()
    live, _ = _live_and_capture()
    clean = [SimpleNamespace(offset=11)]
    scheduler._prefill_for_prompt_only_cache = Mock(return_value=clean)
    scheduler._mixed_swa_deferred_seed = Mock(return_value=(None, 0))
    request = SimpleNamespace(request_id="r", _deferred_prompt_cache={
        "family": "Mixed-SWA", "mode": "object", "key_tokens": list(range(11)),
        "boundary_capture": None, "live_cache": live,
    })
    scheduler._materialize_deferred_prompt_cache("r", request)
    scheduler._prefill_for_prompt_only_cache.assert_called_once_with(list(range(11)))
    assert request._extracted_cache is clean


def test_verify_mode_compares_against_a_real_reprefill(monkeypatch, caplog):
    scheduler = _scheduler()
    live, capture = _live_and_capture()
    monkeypatch.setenv("VMLX_MIXED_SWA_INLINE_VERIFY", "1")
    deferred = {"boundary_capture": capture, "live_cache": live}
    # A reference that differs in one full-attention row must be reported.
    reference = [copy.deepcopy(live[0]), copy.deepcopy(capture[1][1])]
    reference[0].offset = 11
    reference[0].keys = reference[0].keys[..., :11, :] + 1
    reference[0].values = reference[0].values[..., :11, :]
    scheduler._prefill_for_prompt_only_cache = Mock(return_value=reference)
    scheduler._mixed_swa_deferred_seed = Mock(return_value=(None, 0))
    with caplog.at_level("INFO"):
        out = scheduler._assemble_inline_mixed_swa_boundary("r", deferred, list(range(11)), "paged")
    assert out is not None
    assert "MIXED_SWA_INLINE_VERIFY request=r boundary=11 reference=full DIFF" in caplog.text


def test_exact_n1_hit_logs_restored_boundary_not_a_stale_reason(caplog):
    scheduler = _scheduler()
    live, _ = _live_and_capture()
    request = SimpleNamespace(request_id="r", cached_tokens=11)
    deferred = {"boundary_capture": None, "live_cache": live}
    with caplog.at_level("INFO"):
        out = scheduler._assemble_inline_mixed_swa_boundary("r", deferred, list(range(11)), "paged", request=request)
    assert out is None
    assert "already restored from the prefix cache (cached=11 of 11 key tokens)" in caplog.text
    assert "(captured)" not in caplog.text

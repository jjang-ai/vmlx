# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 long prompts reach the SSD block store without a whole-prompt copy (models/minimax_m3/prompt_boundary.py).

The generator hands the scheduler a boundary marker (the live cache objects + the prompt row count) instead of cloning
the prompt cache; the terminal store takes views of the live rows and streams them to the SSD writer under a bounded
in-flight budget; an SSD-only restore fills preallocated lanes one payload at a time.

Proof on the tiny synthetic M3 of tests/test_minimax_m3_dspark.py (and a wide variant: 16 heads over 8 KV groups of
256, so a few thousand tokens are hundreds of MB), real Scheduler, block-disk-only like the app:

1. the streamed store holds exactly what the clone path stores (restored from SSD after a restart, bit for bit, every
   lane), and continuations / exact repeats on it equal their cold prefill; plain lane and DSpark lane (rounds with
   rejected drafts, i.e. rollbacks, before the store);
2. peak memory added by the store stays under the in-flight cap at every prompt length, and an SSD restore peaks at
   the restored cache plus one payload;
3. controls: the clone path's memory scales with the prompt; the generic restore peaks at twice the restored cache;
4. decode is unaffected by the marker (no extra memory, same step time), while holding array VIEWS of the live cache
   (the hazard the marker avoids) costs a full copy of every lane.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import tests.test_minimax_m3_dspark as T  # noqa: E402
from tests.test_minimax_m3_aligned_prefix_cache import (  # noqa: E402,F401  (fixtures are used by name)
    GEN,
    _defaults,
    _eq,
    _m3_config,
    _run,
    _same,
    _scheduler,
    _Tok,
    _Trace,
    _drafter_config,
    _save_release_drafter,
    model,
)
from vmlx_engine.models.minimax_m3 import prefill_geometry, prompt_boundary  # noqa: E402

REPORT_ENV = "VMLX_M3_STREAM_REPORT"
MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _levers(monkeypatch):
    for name in (prompt_boundary.ENV, prompt_boundary.INFLIGHT_ENV, prompt_boundary.RESTORE_ENV,
                 prefill_geometry.STRUCTURAL_ENV):
        monkeypatch.delenv(name, raising=False)


def _report(row: dict) -> None:
    out = os.environ.get(REPORT_ENV)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def _ids(n: int, seed: int) -> list:
    return np.random.default_rng(seed).integers(0, T.VOCAB - 8, n).tolist()


def _key(prompt: list) -> list:
    return prompt[: len(prompt) - len(GEN)]


def _boundary(prompt: list) -> int:
    return len(prompt) - len(GEN) - 1


def _restore_from_ssd(model, ssd, step: int, prompt: list):
    """A fresh scheduler on the SSD store (empty RAM index): the prompt's stored rows, read back from disk."""
    sched = _scheduler(model, ssd, step)
    try:
        table, _ = sched.block_aware_cache.fetch_cache("probe", _key(prompt))
        assert table is not None and table.num_tokens == _boundary(prompt), (None if table is None else table.num_tokens)
        restored = sched.block_aware_cache.reconstruct_cache(table)
        assert restored is not None
        mx.eval([c.state for c in restored])
        disk_blocks = sched.block_aware_cache._last_reconstruct_disk_blocks
        sched.block_aware_cache.release_cache("probe")
        return restored, disk_blocks
    finally:
        sched.shutdown()


def _assert_same_state(a, b, rows: int) -> None:
    assert len(a) == len(b) and len(a) > 0
    for la, lb in zip(a, b):
        assert type(la) is type(lb) and int(la.offset) == int(lb.offset) == rows
        sa, sb = la.state, lb.state
        assert len(sa) == len(sb)
        for x, y in zip(sa, sb):
            assert x is not None and y is not None and x.size > 0                  # fail closed: real content
            assert x.shape == y.shape and _eq(x, y)


def _direct_prefill(model, prompt: list, step: int):
    ref = model.make_cache()
    for a, b in prefill_geometry.prefill_pieces(0, len(prompt), step, _boundary(prompt)):
        model(mx.array([prompt[a:b]]), cache=ref)
        mx.eval([c.state for c in ref])
    return ref


# ----------------------------------------------------------------------------------------------- 1. bit-exact
@pytest.mark.parametrize("lane", ["plain", "dspark"])
def test_streamed_store_equals_clone_store(model, tmp_path, monkeypatch, lane):
    """The streamed store (marker -> live views -> bounded SSD writes) stores exactly the clone path's rows, and a
    restore from it continues exactly like a cold prefill (continuation turn, exact repeat with the K tail)."""
    step = 320
    trace = _Trace(monkeypatch)
    max_tokens = 12 if lane == "dspark" else 7
    if lane == "dspark":
        from vmlx_engine.dspark.runtime import load_dspark_runtime

        root = tmp_path / "bundle"
        root.mkdir()
        (root / "config.json").write_text(json.dumps(_m3_config()))
        _save_release_drafter(root / "dspark", _drafter_config())
        object.__setattr__(model, "_vmlx_dspark_runtime", load_dspark_runtime(model, root))
    prompt = _ids(2371, 3) + GEN                       # 2374 tokens, K = 2370: no multiple of 64 or the step
    stores = {}
    try:
        for mode in ("clone", "stream"):
            monkeypatch.setenv(prompt_boundary.ENV, "0" if mode == "clone" else "1")
            # A small cap so the stream actually waits on the writer.
            monkeypatch.setenv(prompt_boundary.INFLIGHT_ENV, "0.25")
            ssd = tmp_path / f"ssd-{mode}"
            sched = _scheduler(model, ssd, step)
            try:
                first = _run(sched, trace, prompt, max_tokens=max_tokens)
                assert first["cached"] == 0
                if lane == "dspark":
                    assert type(sched.batch_generator).__name__ == "DSparkTextGenerator"
                    stats = sched.batch_generator.last_dspark_stats
                    assert stats["rounds"] > 0 and stats["accepted"] < stats["drafted"], stats   # rollbacks ran
                reason = sched.batch_generator.prompt_snapshot_last_reason
                assert reason == ("boundary_marker" if mode == "stream" else "captured"), reason
                stream = getattr(sched.block_aware_cache, "_last_m3_stream_store", None)
                if mode == "stream":
                    assert stream and stream["blocks"] > 0 and stream["waits"] > 0 and stream["timeouts"] == 0, stream
                stores[mode] = {"first": first, "stream": dict(stream or {})}
            finally:
                sched.shutdown()
            restored, disk_blocks = _restore_from_ssd(model, ssd, step, prompt)
            assert disk_blocks > 0
            stores[mode]["restored"] = restored
        k = _boundary(prompt)
        _assert_same_state(stores["stream"]["restored"], stores["clone"]["restored"], k)
        # ... and both are the cold prefill's rows.
        ref = _direct_prefill(model, prompt, step)
        for got, want in zip(stores["stream"]["restored"], ref):
            for x, y in zip(got.state, want.state):
                assert _eq(x, y[..., :k, :])
        assert stores["stream"]["first"]["tokens"] == stores["clone"]["first"]["tokens"]

        # Continue on the streamed store after a restart: exact repeat (restores K, re-feeds the tail) and the next
        # tool turn (restores K too: a turn-cut-free prompt reuses its chunk floor / own K) equal their cold runs.
        sched = _scheduler(model, tmp_path / "ssd-stream", step)
        try:
            rep = _run(sched, trace, prompt, max_tokens=max_tokens)
            cold = _run(sched, trace, prompt, bypass=True, max_tokens=max_tokens)
            assert rep["cached"] == k and "disk" in rep["detail"], (rep["cached"], rep["detail"])
            if lane == "dspark":
                assert rep["tokens"] == cold["tokens"] and _eq(rep["logits"][0], cold["logits"][0])
            else:
                assert _same(rep, cold)
            nxt = prompt + cold["tokens"] + [T.VOCAB - 6] + _ids(613, 4) + GEN
            warm = _run(sched, trace, nxt, max_tokens=max_tokens)
            cold2 = _run(sched, trace, nxt, bypass=True, max_tokens=max_tokens)
            assert warm["cached"] == (k // step) * step > 0
            if lane == "dspark":
                assert warm["tokens"] == cold2["tokens"] and _eq(warm["logits"][0], cold2["logits"][0])
            else:
                assert _same(warm, cold2)
        finally:
            sched.shutdown()
    finally:
        if lane == "dspark":
            object.__setattr__(model, "_vmlx_dspark_runtime", None)
    _report({"test": "streamed_equals_clone", "lane": lane, "boundary": _boundary(prompt),
             "stream": stores["stream"]["stream"]})


# ----------------------------------------------------------------------------------------------- 2-3. memory
@pytest.fixture(scope="module")
def wide_model():
    saved = (T.HEADS, T.KVH, T.HD)
    T.HEADS, T.KVH, T.HD = 16, 8, 256
    try:
        yield T._model(seed=5, block=128, topk=16)
    finally:
        T.HEADS, T.KVH, T.HD = saved


def _measure_request(monkeypatch, sched, trace, prompt) -> dict:
    """Run one request; record the memory the prompt snapshot added, the peak the store added, and the restore's
    peak on an exact repeat (fresh scheduler, from SSD)."""
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    rec = {"store": [], "snapshot": []}
    real_store = BlockAwarePrefixCache.store_cache
    real_snap = SingleBatchGenerator._clone_admissible_prompt_cache_snapshot

    def store(self, *a, **k):
        mx.synchronize()
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        out = real_store(self, *a, **k)
        mx.synchronize()
        rec["store"].append(mx.get_peak_memory() - before)
        return out

    def snap(self, cache):
        mx.synchronize()
        before = mx.get_active_memory()
        out = real_snap(self, cache)
        mx.synchronize()
        rec["snapshot"].append(mx.get_active_memory() - before)
        return out

    monkeypatch.setattr(BlockAwarePrefixCache, "store_cache", store)
    monkeypatch.setattr(SingleBatchGenerator, "_clone_admissible_prompt_cache_snapshot", snap)
    res = _run(sched, trace, prompt)
    monkeypatch.setattr(BlockAwarePrefixCache, "store_cache", real_store)
    monkeypatch.setattr(SingleBatchGenerator, "_clone_admissible_prompt_cache_snapshot", real_snap)
    assert rec["store"] and rec["snapshot"], rec
    return {"res": res, "store_peak": max(rec["store"]), "snapshot": rec["snapshot"][0]}


def _measure_restore(sched, prompt):
    """Peak memory from the lookup through the restore of ``prompt``'s stored rows (the lookup itself may load
    payloads: on a cold index every matched block is read and promoted)."""
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    table, _ = sched.block_aware_cache.fetch_cache("probe", _key(prompt))
    assert table is not None and table.num_tokens == _boundary(prompt)
    restored = sched.block_aware_cache.reconstruct_cache(table)
    assert restored is not None
    mx.eval([c.state for c in restored])
    mx.synchronize()
    peak = mx.get_peak_memory() - before
    sched.block_aware_cache.release_cache("probe")
    size = sum(a.nbytes for c in restored for a in (c.keys, c.values, getattr(c, "idx_keys", None)) if a is not None)
    del restored
    return {"peak": peak, "restored_bytes": size}


def test_store_and_restore_memory_is_bounded(wide_model, tmp_path, monkeypatch):
    """Peak memory added by the store stays under the in-flight cap (plus one block) at every prompt length, while
    the clone path's grows with the prompt; an SSD restore peaks at the restored cache plus one payload, while the
    generic restore peaks at twice the restored cache."""
    step = 2048
    cap_mb = 8
    monkeypatch.setenv(prompt_boundary.INFLIGHT_ENV, str(cap_mb))
    trace = _Trace(monkeypatch)
    rows = []
    for n in (2109, 4157, 8253):                       # + GEN: K = n - 1, not a multiple of 64 or the step
        prompt = _ids(n, 50 + n) + GEN
        row = {"prompt": len(prompt)}
        for mode in ("clone", "stream"):
            monkeypatch.setenv(prompt_boundary.ENV, "0" if mode == "clone" else "1")
            ssd = tmp_path / f"ssd-{mode}-{n}"
            sched = _scheduler(wide_model, ssd, step)
            try:
                m = _measure_request(monkeypatch, sched, trace, prompt)
                stream = dict(getattr(sched.block_aware_cache, "_last_m3_stream_store", None) or {})
                if mode == "stream":
                    # Same session (the live app's second request): the index holds metadata only.
                    for restore_mode in ("streamed", "generic"):
                        monkeypatch.setenv(prompt_boundary.RESTORE_ENV, "1" if restore_mode == "streamed" else "0")
                        r = _measure_restore(sched, prompt)
                        row[f"{restore_mode}_restore_peak_mb"] = r["peak"] / MB
                        row["restored_mb"] = r["restored_bytes"] / MB
            finally:
                sched.shutdown()
            row[f"{mode}_snapshot_mb"] = m["snapshot"] / MB
            row[f"{mode}_store_peak_mb"] = m["store_peak"] / MB
            if mode == "stream":
                row["stream"] = stream
                # After a restart: a cold index (the lookup reads and promotes every block first).
                for restore_mode in ("streamed", "generic"):
                    monkeypatch.setenv(prompt_boundary.RESTORE_ENV, "1" if restore_mode == "streamed" else "0")
                    sched = _scheduler(wide_model, ssd, step)
                    try:
                        r = _measure_restore(sched, prompt)
                    finally:
                        sched.shutdown()
                    row[f"{restore_mode}_restart_restore_peak_mb"] = r["peak"] / MB
                monkeypatch.delenv(prompt_boundary.RESTORE_ENV)
        rows.append(row)
    _report({"test": "memory", "cap_mb": cap_mb, "rows": rows})
    block_mb = max(r["restored_mb"] for r in rows) / (max(r["prompt"] for r in rows) / 64)   # one 64-token block
    for r in rows:
        assert r["restored_mb"] > 4 * cap_mb                          # big enough to show it
        assert r["stream_snapshot_mb"] < 0.5                           # the marker allocates nothing
        assert r["stream_store_peak_mb"] <= cap_mb + 2 * block_mb + 1, r
        assert r["stream"]["blocks"] > 0 and r["stream"]["max_pending"] <= cap_mb * MB and r["stream"]["timeouts"] == 0
        assert r["clone_snapshot_mb"] >= 0.9 * r["restored_mb"], r      # control: the clone is the whole prompt
        # Restore: the streamed restore holds one payload (a group file: <= 16 blocks) on top of the cache, in the
        # same session and after a restart; the generic restore holds every payload and the concatenation.
        for when in ("", "restart_"):
            assert r[f"streamed_{when}restore_peak_mb"] <= r["restored_mb"] + 17 * block_mb + 1, r
            assert r[f"generic_{when}restore_peak_mb"] >= 1.8 * r["restored_mb"], r
    # The clone path's memory scales with the prompt; the stream's store peak does not.
    small, big = rows[0], rows[-1]
    assert big["clone_snapshot_mb"] >= 3 * small["clone_snapshot_mb"]
    assert big["stream_store_peak_mb"] <= small["stream_store_peak_mb"] + 2 * block_mb + 1


# ----------------------------------------------------------------------------------------------- 4. decode
def _decode_probe(model, prompt, holder, steps=40):
    """One request on SingleBatchGenerator (prefill + ``steps`` decode tokens); ``holder(cache)`` is what the
    generator keeps as its prompt snapshot (the thing under test). Returns (decode-window peak: from the snapshot to
    the end, above the memory at the snapshot; memory held at the end with the snapshot alive, above the start;
    decode step times)."""
    from vmlx_engine.sampling import make_minimax_m3_sampler
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    sampler = make_minimax_m3_sampler(temp=0.0)
    gen = SingleBatchGenerator(model, sampler=sampler, max_tokens=steps + 1, prefill_step_size=2048, stop_tokens=[])
    at_snapshot = {}

    def take(cache):
        out = holder(cache)
        gen._sync()                      # the generator's own stream (mx.synchronize() is the default stream's)
        mx.synchronize()
        at_snapshot["active"] = mx.get_active_memory()
        mx.reset_peak_memory()           # the decode window starts here: after prefill, before the first decode
        return out

    gen._clone_admissible_prompt_cache_snapshot = take
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    gen.insert([prompt], max_tokens=[steps + 1])
    first, = gen.next()[0]
    snapshot = first.prompt_cache_snapshot
    times = []
    for _ in range(steps):
        gen._sync()                      # every command buffer of the generator's stream complete AND released
        t = time.perf_counter()
        resp, = gen.next()[1]
        times.append(time.perf_counter() - t)
        if resp.finish_reason is not None:
            break
    mx.synchronize()
    peak, held = mx.get_peak_memory() - at_snapshot["active"], mx.get_active_memory() - base
    del snapshot, first, resp, gen
    return peak, held, times


def test_decode_is_unaffected_by_the_marker(wide_model, monkeypatch):
    """The marker holds cache objects, not arrays: decode memory and step time equal a run with no snapshot at all.
    Holding array views of the live cache instead (the hazard) blocks MLX buffer donation in the cache's slice
    update: the first decode step copies every lane and the copy stays while the views are held. The old clone holds
    a full copy for the whole decode.

    The holder is the only variable, so the decode runs synchronously from a synchronized device every step. MLX's
    slice update copies a lane whenever a command buffer still retains the lane's data (``is_donatable``: a commit's
    completion handler holds every input until it runs). Under the one-token decode overlap (VMLX_M3_DECODE_OVERLAP,
    default on) the previous step is in flight by design, and this tiny model copied a random number of lanes in every
    arm, the marker's included (0 .. 240 MB against 134 MB of lanes). ``mx.eval`` returns on the step's completion
    event, before the handler releases the inputs, so even a synchronous loop copied a lane set now and then;
    synchronizing the generator's stream waits for the release."""
    monkeypatch.setenv("VMLX_M3_DECODE_OVERLAP", "0")
    prompt = _ids(4000, 7)                              # capacity 4096 rows: 40 decode steps append without growth

    def none(cache):
        return None

    def marker(cache):
        return prompt_boundary.capture(list(cache))

    def views(cache):
        held = [c.state for c in cache]
        mx.eval(held)
        return held

    def clone(cache):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        return SingleBatchGenerator._clone_prompt_cache_snapshot(cache)

    results = {name: [] for name in ("none", "marker", "views", "clone")}
    for _ in range(3):                                   # interleaved probes: memory median, step time fastest
        for name, fn in (("none", none), ("marker", marker), ("views", views), ("clone", clone)):
            results[name].append(_decode_probe(wide_model, prompt, fn))
    summary = {}
    for name, runs in results.items():
        summary[name] = {
            "decode_peak_mb": statistics.median(r[0] for r in runs) / MB,
            "held_mb": statistics.median(r[1] for r in runs) / MB,
            "median_step_ms": min(statistics.median(r[2][1:]) for r in runs) * 1e3,   # load only adds time
            "first_step_ms": statistics.median(r[2][0] for r in runs) * 1e3,
        }
    _report({"test": "decode", "summary": summary})
    lanes_mb = summary["clone"]["held_mb"] - summary["none"]["held_mb"]   # one copy of every lane
    assert lanes_mb > 50, summary                                        # big enough to show it
    # The marker costs nothing during decode: same decode peak, same memory held, same step time.
    assert abs(summary["marker"]["held_mb"] - summary["none"]["held_mb"]) < 1.0, summary
    assert summary["marker"]["decode_peak_mb"] <= summary["none"]["decode_peak_mb"] + 1.0, summary
    assert summary["marker"]["median_step_ms"] <= 1.5 * summary["none"]["median_step_ms"] + 0.5, summary
    # Controls: views pin the old buffers (the first decode step copies every lane, and the copy stays); the clone
    # is a copy held through the decode.
    assert summary["views"]["decode_peak_mb"] >= summary["none"]["decode_peak_mb"] + 0.9 * lanes_mb, summary
    assert summary["views"]["held_mb"] >= summary["none"]["held_mb"] + 0.9 * lanes_mb, summary
    assert summary["clone"]["held_mb"] >= summary["none"]["held_mb"] + 0.9 * lanes_mb, summary


# ----------------------------------------------------------------------------------------------- units
def test_marker_contract():
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    dense, sparse = KVCache(), MiniMaxM3SparseCache()
    k = mx.random.normal((1, 2, 70, 8)).astype(mx.bfloat16)
    dense.update_and_fetch(k, k)
    sparse.update_and_fetch(k, k)
    sparse.update_index(k[:, :1])
    marker = prompt_boundary.capture([dense, sparse])
    assert marker is not None and marker.rows == 70 and bool(marker)
    assert marker.nbytes() == 70 * (2 * 2 * 8 * 2 * 2 + 1 * 8 * 2)
    # Decode appends rows; the views still return the first 70, as views of the live buffers.
    t = mx.random.normal((1, 2, 3, 8)).astype(mx.bfloat16)
    dense.update_and_fetch(t, t)
    sparse.update_and_fetch(t, t)
    sparse.update_index(t[:, :1])
    views = marker.views()
    assert [int(c.offset) for c in views] == [70, 70]
    assert _eq(views[1].state[0], k) and _eq(views[1].state[2], k[:, :1]) and _eq(views[0].state[1], k)
    # A cache that lost rows (trimmed below the boundary) gives no views; non-M3 caches give no marker.
    sparse.trim(10)
    assert marker.views() is None
    assert prompt_boundary.capture([dense]) is None and prompt_boundary.capture([]) is None
    assert prompt_boundary.store_inflight_bytes() == 256 * MB

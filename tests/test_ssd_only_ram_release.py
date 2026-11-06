"""SSD caching keeps RAM at the model's size: DFlash2 RAM snapshots go once their SSD copy is durable, and an idle
SimpleEngine returns MLX's allocator free list (2026-10-12).

Measured before: GLM-5.3 JANGHT2.4 (93.75 GiB with its drafter) idled at 100.6-101.4 GiB after 17k/67k chats -- its
lane's RAM prefix snapshots (<= 3 GiB) plus the 4 GB allocator free list.
"""

import asyncio
from concurrent.futures import Future

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine import dflash2_runtime as D  # noqa: E402
from vmlx_engine import glm_dflash2 as G  # noqa: E402
from vmlx_engine import mlx_memory  # noqa: E402


def _receipt(durable=None):
    f = Future()
    if durable is not None:
        f.set_result({"outcome": "stored" if durable else "refused", "durable": bool(durable)})
    return f


class _Disk:
    def __init__(self, snap=None):
        self.snap, self.lookups = snap, 0

    def lookup(self, ids, min_len=0):
        self.lookups += 1
        return None if self.snap is None else (3, self.snap)


def _snap(receipt):
    s = G._PrefixSnapshot([], None, None)
    s.receipt = receipt
    return s


def test_glm_prefix_store_drops_ram_copy_once_durable(monkeypatch):
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    st = G.PrefixStore(4, 3 * 2 ** 30, disk=_Disk())
    pending, failed = _receipt(), _receipt(False)
    st.entries[(1, 2, 3)] = _snap(pending)
    st.entries[(1, 2)] = _snap(failed)
    assert st.release_durable_ram() == (0, 1)                 # write in flight: RAM copy still needed
    pending.set_result({"outcome": "stored", "durable": True})
    assert st.release_durable_ram() == (1, 0)
    assert list(st.entries) == [(1, 2)]                       # a refused SSD write keeps its RAM copy
    assert st.ram_released == 1


def test_glm_disk_hit_is_not_kept_in_ram(monkeypatch):
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    disk_snap = _snap(None)
    st = G.PrefixStore(4, 3 * 2 ** 30, disk=_Disk(disk_snap))
    n, snap = st.lookup([1, 2, 3, 4])
    assert (n, snap) == (3, disk_snap) and st.last_source == "disk"
    assert not st.entries


def test_glm_release_switch_off_keeps_ram_tier(monkeypatch):
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "0")
    disk_snap = _snap(None)
    st = G.PrefixStore(4, 3 * 2 ** 30, disk=_Disk(disk_snap))
    st.entries[(9,)] = _snap(_receipt(True))
    assert st.release_durable_ram() == (0, 0) and (9,) in st.entries
    st.lookup([1, 2, 3, 4])
    assert (1, 2, 3) in st.entries                            # old behaviour: disk hits cached in RAM


def test_glm_put_under_ssd_only_keeps_no_ram_copy_and_chains_deltas(monkeypatch):
    """SSD-only + SSD tier: put() writes the snapshot to SSD only (no RAM entry) and a later snapshot of the same
    request is a delta whose parent is the previous one."""
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    monkeypatch.setattr(G, "_clone_caches", lambda cache, detach_mla=False: cache)

    class Disk(_Disk):
        def __init__(self):
            super().__init__()
            self.puts = []

        def put(self, key, snap, parent_receipt=None):
            r = _receipt(True)
            self.puts.append((key, snap, parent_receipt))
            return r

    disk = Disk()
    st = G.PrefixStore(4, 3 * 2 ** 30, disk=disk)
    st.put([5, 6], [], None, None)
    st.put([5, 6, 7, 8], [], None, None)
    assert not st.entries
    (k1, s1, _), (k2, s2, pr2) = disk.puts
    assert s1.parent is None and s2.parent == {"name": G.PrefixDiskTier.digest((5, 6)), "n": 2, "depth": 0}
    assert s2.depth == 1 and pr2 is s1.receipt and st.delta_writes == 1


def test_generic_session_store_releases_durable_entries(monkeypatch):
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    store = D._DFlash2SessionStore(4)
    a = {"model_key": "m", "tokens": [1, 2], "cache_len": 2}
    b = {"model_key": "m", "tokens": [1, 2, 3], "cache_len": 3}
    c = {"model_key": "m", "tokens": [4], "cache_len": 1}
    for e in (a, b, c):
        store.put(e)
    ra, rb = _receipt(), _receipt()
    store.attach_receipt(a, ra)
    store.attach_receipt(b, rb)                               # c: media entry, RAM only (no receipt)
    assert store.release_durable() == (0, 2)
    rb.set_result({"outcome": "stored", "durable": True})
    ra.set_result({"outcome": "failed", "durable": False})
    assert store.release_durable() == (1, 0)
    assert [e["tokens"] for e in store._entries] == [[1, 2], [4]]
    assert store.take_matching("m", [1, 2, 3, 4]) is a        # the durable [1,2,3] entry now restores from SSD
    assert store.stats()["ram_released_after_ssd"] == 1


def test_release_idle_memory_consults_loaded_lanes_and_clears(monkeypatch):
    calls = []

    class FakeMx:
        cache = 5 << 30

        def get_cache_memory(self):
            return self.cache

        def clear_cache(self):
            calls.append("clear")
            self.cache = 0

    monkeypatch.setattr(G, "release_durable_prefix_ram", lambda: (calls.append("glm") or (2, 1)))
    monkeypatch.setattr(D, "release_durable_session_ram", lambda: (calls.append("generic") or (1, 0)))
    out = mlx_memory.release_idle_memory(mx=FakeMx())
    assert calls == ["glm", "generic", "clear"]
    assert out == {"released_snapshots": 3, "pending_snapshots": 1, "freed_bytes": 5 << 30}


def test_simple_engine_idle_lock_releases_after_grace_only_when_idle(monkeypatch):
    from vmlx_engine.engine import simple as S

    monkeypatch.setenv("VMLX_IDLE_ALLOCATOR_RELEASE_S", "0.05")
    fired = []
    pending = {"n": 1}

    def fake_release():
        fired.append(time_now())
        p = pending["n"]
        pending["n"] = 0
        return {"released_snapshots": 0, "pending_snapshots": p, "freed_bytes": 1 << 30}

    import time
    time_now = time.monotonic
    monkeypatch.setattr(S, "release_idle_memory", fake_release)

    async def scenario():
        from concurrent.futures import ThreadPoolExecutor

        eng = S.SimpleEngine.__new__(S.SimpleEngine)
        eng._loaded = True
        eng._model_executor = ThreadPoolExecutor(max_workers=1)   # not the process-wide one other tests replace
        lock = S._IdleReleaseLock(eng)
        eng._generation_lock = lock
        async with lock:
            pass
        await asyncio.sleep(0.02)
        async with lock:                                      # re-taken inside the grace: no release yet
            pass
        assert fired == []
        await asyncio.sleep(0.12)
        assert len(fired) == 1                                # idle for the grace: one release
        await asyncio.sleep(1.2)
        assert len(fired) == 2                                # a pending SSD write re-arms it (>= 1 s backoff)
        await asyncio.sleep(1.2)
        assert len(fired) == 2
        eng._model_executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_batched_idle_releaser_fires_once_per_idle_period(monkeypatch):
    monkeypatch.setenv("VMLX_IDLE_ALLOCATOR_RELEASE_S", "1.0")
    calls = []
    monkeypatch.setattr(mlx_memory, "release_idle_memory",
                        lambda log=None: calls.append(1) or {"released_snapshots": 0, "pending_snapshots": 0,
                                                             "freed_bytes": 2 << 30})
    now = [100.0]
    r = mlx_memory.IdleMemoryReleaser()
    r._clock = lambda: now[0]
    assert not r.due()                                        # nothing served yet
    r.note_work()
    now[0] += 0.5
    assert not r.due()
    now[0] += 0.6
    assert r.due()
    r.release()
    assert calls == [1] and not r.due()                       # once per idle period
    now[0] += 10
    assert not r.due()
    r.note_work()
    now[0] += 1.1
    assert r.due()
    monkeypatch.setenv("VMLX_IDLE_ALLOCATOR_RELEASE_S", "-1")
    assert not r.due()                                        # negative grace: never


def test_batched_loops_wire_the_idle_releaser():
    import inspect

    from vmlx_engine import engine_core, mllm_scheduler

    for src in (inspect.getsource(engine_core.EngineCore._engine_loop),
                inspect.getsource(mllm_scheduler.MLLMScheduler._process_loop)):
        assert "IdleMemoryReleaser()" in src and "_idle_release.note_work()" in src
        assert "_idle_release.due()" in src and "_idle_release.release" in src


def test_glm_disk_writer_does_not_pin_the_last_snapshot(tmp_path):
    """The writer thread must not keep the last written snapshot alive while it waits for the next one."""
    import gc
    import weakref

    disk = G.PrefixDiskTier(tmp_path, 2 ** 30)
    try:
        class Snap:                                           # weakref-able stand-in; _write is stubbed below
            pass

        disk._write = lambda name, key, snap: {"outcome": "stored", "durable": True, "retained_tokens": len(key)}
        snap = Snap()
        ref = weakref.ref(snap)
        receipt = disk.put((1, 2, 3), snap)
        disk.flush()
        assert receipt.result()["durable"]
        del snap
        gc.collect()
        assert ref() is None, "writer thread still references the last snapshot"
    finally:
        disk.close()


class _SSD:
    """Stands in for DFlash2SessionSSD.put: a durable receipt per write."""

    def __init__(self):
        self.puts = []

    def put(self, entry):
        self.puts.append(entry["kind"])
        r = _receipt(True)
        r.vmlx_ssd_ref = {"sig": entry["kind"], "len": entry["cache_len"], "depth": 0}
        return r


def test_finished_turn_stays_hot_for_an_immediate_continuation_then_releases(monkeypatch):
    """SSD-only: the finished turn's cache (the live cache, no copy to hold) is served from RAM to a tool continuation
    sent at once; prefill snapshots stay SSD-only; the idle release drops the hot entry once durable."""
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    store = D._DFlash2SessionStore(4)
    monkeypatch.setattr(D, "_SESSION_STORE", store)
    monkeypatch.setattr(D, "_SESSION_SSD", _SSD())
    boundary = {"model_key": "m", "kind": "boundary", "tokens": [1, 2], "cache_len": 2}
    turn = {"model_key": "m", "kind": "turn", "tokens": [1, 2, 3, 4], "cache_len": 3}
    rb = D._store_session(boundary)
    rt = D._store_session(turn)
    assert [e["kind"] for e in store._entries] == ["turn"]          # the boundary view is SSD-only
    # the next admission still waits for the turn's write (no buffer shared with the continuation's appends)
    assert rt.vmlx_lookup_served_from_ram is False and rb.vmlx_lookup_served_from_ram is False
    assert store.take_matching("m", [1, 2, 3, 4, 5, 6]) is turn       # durable, but taken before the release
    store.put(dict(turn))
    store.attach_receipt(store._entries[-1], _receipt(True))
    assert D.release_durable_session_ram() == (1, 0)                 # idle: back to the model's size
    assert store._entries == []

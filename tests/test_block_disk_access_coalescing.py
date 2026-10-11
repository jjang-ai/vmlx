"""LRU touches are coalesced: no queue item per touch, same rows/timestamps/counts applied.

A long-prefix restore touches every block it reads. Each touch used to be its own queue item and its own
exclusive lock + UPDATE + COMMIT on the writer (~1 ms of writer time per block, competing with the
restoring thread). Touches now accumulate in a table the writer applies in one transaction (in each
batch, on its idle wakeup, and before it exits) -- never delaying real work queued behind them.
"""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.block_disk_store import BlockDiskStore


def _block():
    k = mx.zeros((1, 2, 8, 16), dtype=mx.float16)
    v = mx.ones((1, 2, 8, 16), dtype=mx.float16)
    mx.eval(k, v)
    return [("kv", k, v)]


def _rows(store, hashes):
    out = {}
    for h in hashes:
        row = store._read_conn.execute(
            "SELECT last_accessed, access_count FROM blocks WHERE block_hash = ?", (h.hex(),)
        ).fetchone()
        out[h] = (float(row[0]), int(row[1]))
    return out


def _settle(store, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lock = getattr(store, "_access_lock", None)
        if lock is None:
            idle = True
        else:
            with lock:
                idle = not store._access_pending
        if idle and store.get_stats()["write_pipeline"]["pending_items"] == 0:
            time.sleep(0.1)
            return
        time.sleep(0.02)
    raise AssertionError("access touches did not settle")


def test_burst_of_touches_enqueues_nothing_and_applies_counts(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        hashes = [bytes([i]) * 32 for i in range(1, 9)]
        for h in hashes:
            assert store.write_block_async(h, _block(), token_count=8)
        assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
        _settle(store)
        before = _rows(store, hashes)

        enqueued = []
        original = store._try_enqueue_write_item

        def spy(item, **kwargs):
            enqueued.append(item[0])
            return original(item, **kwargs)

        monkeypatch.setattr(store, "_try_enqueue_write_item", spy)
        # Hold the writer's view still: record a burst faster than the flush delay.
        t_start = time.time()
        for _ in range(3):
            for h in hashes:
                store._queue_access_update(h.hex())
        assert enqueued.count("__access__") == 0, enqueued
        _settle(store)
        after = _rows(store, hashes)
        for h in hashes:
            assert after[h][1] == before[h][1] + 3, (before[h], after[h])
            assert after[h][0] >= t_start
    finally:
        store.shutdown()


def test_reads_still_refresh_lru_rows(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        hashes = [bytes([i]) * 32 for i in range(30, 34)]
        for h in hashes:
            assert store.write_block_async(h, _block(), token_count=8)
        assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
        _settle(store)
        before = _rows(store, hashes)
        t_read = time.time()
        for h in hashes:
            assert store.read_block(h) is not None
        _settle(store)
        after = _rows(store, hashes)
        for h in hashes:
            assert after[h][1] >= before[h][1] + 1
            assert after[h][0] >= t_read
    finally:
        store.shutdown()

"""DFlash2 session SSD writer: a new request waits only for QUEUED snapshots, not for a write already in progress."""
import threading
import time

import pytest

pytest.importorskip("mlx.core")
from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD  # noqa: E402


def _ssd(tmp_path, monkeypatch, gate):
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=0, model_key="m")

    def slow_save(sig, writer):
        assert gate.wait(5)
        raise OSError("not writing in this test")

    monkeypatch.setattr(ssd.store, "save", slow_save)
    return ssd


def _item():
    from concurrent.futures import Future
    f = Future()
    f.set_running_or_notify_cancel()
    return ("sig", {}, {"kind": "turn", "cache_len": 1}, f)


def test_backlog_only_does_not_wait_for_the_write_in_progress(tmp_path, monkeypatch):
    gate = threading.Event()
    ssd = _ssd(tmp_path, monkeypatch, gate)
    try:
        ssd._q.put_nowait(_item())
        for _ in range(100):                               # the writer picked it up
            if ssd._q.qsize() == 0:
                break
            time.sleep(0.01)
        t0 = time.perf_counter()
        ssd.wait_pending_writes(backlog_only=True)
        assert time.perf_counter() - t0 < 0.05
        assert ssd._q.unfinished_tasks == 1                # still writing
    finally:
        gate.set()
        ssd.flush(5)


def test_backlog_only_waits_while_snapshots_are_queued(tmp_path, monkeypatch):
    gate = threading.Event()
    ssd = _ssd(tmp_path, monkeypatch, gate)
    try:
        ssd._q.put_nowait(_item())
        ssd._q.put_nowait(_item())                         # one in progress, one queued behind it
        done = threading.Event()
        threading.Thread(target=lambda: (ssd.wait_pending_writes(backlog_only=True), done.set()), daemon=True).start()
        assert not done.wait(0.1)
        gate.set()
        assert done.wait(5)
        assert ssd._q.unfinished_tasks == 0
    finally:
        gate.set()
        ssd.flush(5)

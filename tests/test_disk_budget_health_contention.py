"""Health observations must not join SSD publication/maintenance lock waits."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget, _LOCK_NAME


@contextmanager
def held_by_thread(budget):
    held = threading.Event()
    release = threading.Event()

    def holder():
        with budget._exclusive_guard():
            held.set()
            release.wait(3)

    worker = threading.Thread(target=holder)
    worker.start()
    try:
        assert held.wait(2)
        yield
    finally:
        release.set()
        worker.join(3)
        assert not worker.is_alive()


@pytest.mark.parametrize("initialized", [False, True])
def test_health_and_janitor_do_not_wait_on_root_thread_lock(tmp_path, initialized):
    budget = GlobalDiskCacheBudget(tmp_path, 1_000_000)
    try:
        if initialized:
            budget.enforce(force=True)
        previous = budget.last_result
        with held_by_thread(budget):
            started = time.perf_counter()
            result = budget.refresh_health()
            janitor = budget.janitor_status()
            assert time.perf_counter() - started < 0.5
            assert result.telemetry_stale is True
            assert janitor["telemetry_stale"] is True
            assert janitor["idle_reconcile_due"] is None
            # A stale observation must not alter maintenance's state.
            assert budget.last_result is previous
            if previous is not None:
                assert result.bytes_after == previous.bytes_after
                assert result.accounting_generation == previous.accounting_generation
            else:
                assert result.accounted is False
                assert result.compliant is False
        assert budget.refresh_health().telemetry_stale is False
        assert budget.janitor_status()["telemetry_stale"] is False
    finally:
        budget.close()


def test_health_does_not_wait_on_other_process_flock(tmp_path):
    budget = GlobalDiskCacheBudget(tmp_path, 1_000_000)
    budget.enforce(force=True)
    process = subprocess.Popen(
        [sys.executable, "-c", (
            "import fcntl,sys; f=open(sys.argv[1],'r+'); "
            "fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); "
            "sys.stdin.read(1)"
        ), str(tmp_path / _LOCK_NAME)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "held"
        started = time.perf_counter()
        result = budget.refresh_health()
        assert time.perf_counter() - started < 0.5
        assert result.telemetry_stale is True
        assert budget.last_result.telemetry_stale is False
        process.communicate("x", timeout=3)
        assert budget.refresh_health().telemetry_stale is False
    finally:
        if process.poll() is None:
            process.communicate("x", timeout=3)
        budget.close()


def test_mutation_guard_still_waits_for_writer_lock(tmp_path):
    budget = GlobalDiskCacheBudget(tmp_path, 1_000_000)
    entered = threading.Event()

    def mutation():
        with budget.mutation_guard() as locked:
            assert locked
            entered.set()

    try:
        with held_by_thread(budget):
            worker = threading.Thread(target=mutation)
            worker.start()
            assert not entered.wait(0.05)
            assert budget.refresh_health().telemetry_stale is True
        worker.join(3)
        assert entered.is_set()
        assert not worker.is_alive()
    finally:
        budget.close()

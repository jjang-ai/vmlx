"""Request-correlated post-eviction telemetry for block-disk writes."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _cache_data(value: float) -> list[tuple]:
    keys = mx.full((1, 1, 8, 16), value, dtype=mx.float16)
    values = mx.full((1, 1, 8, 16), value + 1, dtype=mx.float16)
    mx.eval(keys, values)  # noqa: S307 - MLX tensor materialization
    return [("kv", keys, values)]


def _large_cache_data(value: float) -> list[tuple]:
    keys = mx.full((1, 1, 8, 4096), value, dtype=mx.float16)
    values = mx.full((1, 1, 8, 4096), value + 1, dtype=mx.float16)
    mx.eval(keys, values)  # noqa: S307 - MLX tensor materialization
    return [("kv", keys, values)]


def _wait_for_fence(
    store: BlockDiskStore,
    fence_id: str,
    *,
    timeout: float = 5.0,
) -> tuple[dict, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = store.get_stats()
        pipeline = stats["write_pipeline"]
        fence = next(
            (
                item
                for item in pipeline["recent_fences"]
                if item["fence_id"] == fence_id
            ),
            None,
        )
        if (
            fence is not None
            and fence["post_eviction_complete"]
            and pipeline["queue_depth"] == 0
            and pipeline["inflight"] == 0
        ):
            return stats, fence
        time.sleep(0.01)
    raise AssertionError(f"write fence {fence_id} did not settle")


def _write_request(
    store: BlockDiskStore,
    *,
    request_id: str,
    block_hash: bytes,
    value: float,
    cache_data: list[tuple] | None = None,
) -> tuple[dict, dict]:
    fence_id = store.begin_write_fence(request_id)
    assert store.write_block_async(
        block_hash,
        cache_data if cache_data is not None else _cache_data(value),
        8,
        request_id=request_id,
        fence_id=fence_id,
    )
    assert store.seal_write_fence(fence_id)
    return _wait_for_fence(store, fence_id)


def test_stats_forward_capacity_evictions_separately_from_cleanup(tmp_path, monkeypatch):
    from dataclasses import replace

    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        health = store.global_budget.refresh_health()
        monkeypatch.setattr(store.global_budget, "refresh_health", lambda: replace(
            health, evicted_entries_total=269, capacity_evicted_entries_total=3,
        ))
        stats = store.get_stats()["global_budget"]
        assert stats["evicted_entries_total"] == 269
        assert stats["capacity_evicted_entries_total"] == 3
    finally:
        store.shutdown()


def test_write_fence_correlates_request_without_exposing_hashes(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        stats, fence = _write_request(
            store,
            request_id="resp-request-a",
            block_hash=b"a" * 32,
            value=1,
        )
    finally:
        store.shutdown()

    assert fence == {
        "fence_id": fence["fence_id"],
        "request_id": "resp-request-a",
        "expected": 1,
        "queued": 1,
        "completed": 1,
        "failed": 0,
        "dropped": 0,
        "retained": 1,
        "sealed": True,
        "seal_enqueued": True,
        "seal_failed": False,
        "producer_aborted": False,
        "post_eviction_complete": True,
        "completion_generation": 1,
        "global_accounting_generation": fence["global_accounting_generation"],
        "global_reconciliation_generation": fence[
            "global_reconciliation_generation"
        ],
        "global_bytes_after": fence["global_bytes_after"],
        "global_max_size_bytes": 0,
    }
    assert stats["write_pipeline"]["writer_alive"] is True
    assert not any("hash" in key for key in fence)


def test_read_only_chain_inspection_is_path_free_and_does_not_touch_lru(
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    block_hash = b"i" * 32
    try:
        _write_request(
            store,
            request_id="resp-inspect",
            block_hash=block_hash,
            value=9,
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            before = connection.execute(
                "SELECT last_accessed, access_count FROM blocks "
                "WHERE block_hash = ?",
                (block_hash.hex(),),
            ).fetchone()

        observed = store.inspect_block_chain([block_hash])

        with sqlite3.connect(str(store._db_path)) as connection:
            after = connection.execute(
                "SELECT last_accessed, access_count FROM blocks "
                "WHERE block_hash = ?",
                (block_hash.hex(),),
            ).fetchone()
    finally:
        store.shutdown()

    assert before == after
    assert observed["schema"] == "vmlx-block-disk-chain-inspection-v1"
    assert observed["access_metadata_mutated"] is False
    assert observed["expected_blocks"] == 1
    assert observed["blocks"][0]["indexed"] is True
    assert observed["blocks"][0]["readable"] is True
    serialized = json.dumps(observed, sort_keys=True)
    assert block_hash.hex() not in serialized
    assert str(tmp_path) not in serialized
    assert "file_name" not in serialized


def test_read_only_chain_inspection_distinguishes_stale_and_missing_entries(
    tmp_path,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    stale_hash = b"s" * 32
    missing_hash = b"m" * 32
    try:
        now = time.time()
        with sqlite3.connect(str(store._db_path)) as connection:
            connection.execute(
                "INSERT INTO blocks "
                "(block_hash, file_name, num_tokens, num_layers, dtype, "
                "file_size, created_at, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stale_hash.hex(),
                    "blocks/no-longer-present.safetensors",
                    16,
                    1,
                    "float16",
                    128,
                    now,
                    now,
                    3,
                ),
            )
            connection.commit()
        observed = store.inspect_block_chain([stale_hash, missing_hash])
    finally:
        store.shutdown()

    assert observed["blocks"][0]["indexed"] is True
    assert observed["blocks"][0]["readable"] is False
    assert observed["blocks"][0]["access_count"] == 3
    assert observed["blocks"][1]["indexed"] is False
    assert observed["blocks"][1]["readable"] is False


def test_read_only_chain_inspection_rejects_indexed_size_mismatch(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    block_hash = b"z" * 32
    try:
        _write_request(
            store,
            request_id="resp-size-mismatch",
            block_hash=block_hash,
            value=11,
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            indexed_size = connection.execute(
                "SELECT file_size FROM blocks WHERE block_hash = ?",
                (block_hash.hex(),),
            ).fetchone()[0]
            connection.execute(
                "UPDATE blocks SET file_size = ? WHERE block_hash = ?",
                (indexed_size + 1, block_hash.hex()),
            )
            connection.commit()
        observed = store.inspect_block_chain([block_hash])
    finally:
        store.shutdown()

    assert observed["blocks"][0]["indexed"] is True
    assert observed["blocks"][0]["readable"] is False


def test_write_fence_settles_after_full_capacity_replacement(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    try:
        first_stats, first_fence = _write_request(
            store,
            request_id="resp-old",
            block_hash=b"a" * 32,
            value=1,
            cache_data=_large_cache_data(1),
        )
        first_size = first_stats["disk_size_bytes"]
        assert first_size > 0
        first_global_bytes = first_stats["global_budget"]["bytes_after"]
        assert first_global_bytes >= first_size
        store.max_size_bytes = int(first_global_bytes + first_size * 0.5)
        store.global_budget._requested_max_size_bytes = store.max_size_bytes
        store.global_budget._publish_budget(store.max_size_bytes)

        second_stats, second_fence = _write_request(
            store,
            request_id="resp-new",
            block_hash=b"b" * 32,
            value=2,
            cache_data=_large_cache_data(2),
        )
    finally:
        store.shutdown()

    assert first_fence["retained"] == 1
    assert second_fence["expected"] == 1
    assert second_fence["completed"] == 1
    assert second_fence["failed"] == 0
    assert second_fence["dropped"] == 0
    assert second_fence["retained"] == 1
    assert second_fence["post_eviction_complete"] is True
    assert second_stats["disk_writes"] == 2
    assert second_stats["disk_evictions"] >= 1
    assert second_stats["blocks_on_disk"] == first_stats["blocks_on_disk"] == 1


def test_open_fence_pins_reused_parent_across_split_writer_batches(tmp_path):
    root = tmp_path / "root"
    store = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    parent_hash = b"p" * 32
    unrelated_hash = b"u" * 32
    trim_budget = None
    try:
        _write_request(
            store,
            request_id="seed-parent",
            block_hash=parent_hash,
            value=1,
            cache_data=_large_cache_data(1),
        )
        _write_request(
            store,
            request_id="seed-unrelated",
            block_hash=unrelated_hash,
            value=2,
            cache_data=_large_cache_data(2),
        )

        with sqlite3.connect(str(store._db_path)) as connection:
            rows = connection.execute(
                "SELECT block_hash, file_name FROM blocks"
            ).fetchall()
            paths = {
                str(block_hash): store.cache_dir / str(file_name)
                for block_hash, file_name in rows
            }
            old = time.time() - 600
            connection.execute(
                "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                (old, parent_hash.hex()),
            )
            connection.execute(
                "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                (old + 300, unrelated_hash.hex()),
            )
            connection.commit()
        os.utime(paths[parent_hash.hex()], (old, old))
        os.utime(paths[unrelated_hash.hex()], (old + 300, old + 300))

        fence_id = store.begin_write_fence("split-reused-parent")
        assert store.write_block_async(
            parent_hash,
            _large_cache_data(1),
            8,
            request_id="split-reused-parent",
            fence_id=fence_id,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pipeline = store.get_stats()["write_pipeline"]
            with sqlite3.connect(str(store._db_path)) as connection:
                pin_count = connection.execute(
                    "SELECT COUNT(*) FROM block_write_pins "
                    "WHERE owner_lease_id = ? AND fence_id = ?",
                    (store.global_budget.lease_id, fence_id),
                ).fetchone()[0]
            if (
                pin_count == 1
                and pipeline["queue_depth"] == 0
                and pipeline["inflight"] == 0
            ):
                break
            time.sleep(0.01)
        assert pin_count == 1

        total = store.global_budget.enforce(force=True).bytes_after
        unrelated_size = paths[unrelated_hash.hex()].stat().st_size
        after_one = total - unrelated_size
        cap = ((after_one * 10 + 8) // 9) + 1024
        assert after_one < cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.compliant is True
        assert store.has_block(parent_hash) is True
        assert store.has_block(unrelated_hash) is False

        trim_budget.close()
        trim_budget = None
        child_hash = b"c" * 32
        assert store.write_block_async(
            child_hash,
            _large_cache_data(3),
            8,
            parent_hash=parent_hash,
            request_id="split-reused-parent",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        _stats, fence = _wait_for_fence(store, fence_id)
        with sqlite3.connect(str(store._db_path)) as connection:
            pin_count = connection.execute(
                "SELECT COUNT(*) FROM block_write_pins WHERE fence_id = ?",
                (fence_id,),
            ).fetchone()[0]

        assert fence["expected"] == 2
        assert fence["completed"] == 2
        assert fence["failed"] == 0
        assert fence["retained"] == 2
        assert store.has_block(parent_hash) is True
        assert store.has_block(child_hash) is True
        assert pin_count == 0
    finally:
        if trim_budget is not None:
            trim_budget.close()
        store.shutdown()


def test_cross_process_global_trim_honors_live_fence_pin(tmp_path):
    root = tmp_path / "root"
    store = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    parent_hash = b"x" * 32
    unrelated_hash = b"y" * 32
    fence_id = ""
    try:
        _write_request(
            store,
            request_id="cross-process-parent",
            block_hash=parent_hash,
            value=4,
            cache_data=_large_cache_data(4),
        )
        _write_request(
            store,
            request_id="cross-process-unrelated",
            block_hash=unrelated_hash,
            value=5,
            cache_data=_large_cache_data(5),
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            rows = connection.execute(
                "SELECT block_hash, file_name FROM blocks"
            ).fetchall()
            paths = {
                str(block_hash): store.cache_dir / str(file_name)
                for block_hash, file_name in rows
            }
            old = time.time() - 600
            connection.execute(
                "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                (old, parent_hash.hex()),
            )
            connection.execute(
                "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                (old + 300, unrelated_hash.hex()),
            )
            connection.commit()
        os.utime(paths[parent_hash.hex()], (old, old))
        os.utime(paths[unrelated_hash.hex()], (old + 300, old + 300))

        fence_id = store.begin_write_fence("cross-process-pin")
        assert store.write_block_async(
            parent_hash,
            _large_cache_data(4),
            8,
            request_id="cross-process-pin",
            fence_id=fence_id,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with sqlite3.connect(str(store._db_path)) as connection:
                pin_count = connection.execute(
                    "SELECT COUNT(*) FROM block_write_pins WHERE fence_id = ?",
                    (fence_id,),
                ).fetchone()[0]
            if pin_count == 1:
                break
            time.sleep(0.01)
        assert pin_count == 1

        total = store.global_budget.enforce(force=True).bytes_after
        after_one = total - paths[unrelated_hash.hex()].stat().st_size
        cap = ((after_one * 10 + 8) // 9) + 1024
        child_code = "\n".join(
            (
                "import json, sys",
                "from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget",
                "budget = GlobalDiskCacheBudget(sys.argv[1], int(sys.argv[2]), orphan_grace_seconds=0)",
                "result = budget.enforce(force=True)",
                "print(json.dumps({'compliant': result.compliant, 'after': result.bytes_after}), flush=True)",
                "budget.close()",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", child_code, str(root), str(cap)],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        observed = json.loads(completed.stdout.strip())

        assert observed["compliant"] is True
        assert store.has_block(parent_hash) is True
        assert store.has_block(unrelated_hash) is False
    finally:
        if fence_id:
            store._fail_write_fence(fence_id, "test cleanup")
        store.shutdown()


def test_global_trim_reaps_stale_owner_pin_before_eviction(tmp_path):
    root = tmp_path / "root"
    store = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    block_hash = b"s" * 32
    trim_budget = None
    try:
        _write_request(
            store,
            request_id="stale-pin-seed",
            block_hash=block_hash,
            value=6,
            cache_data=_large_cache_data(6),
        )
        with sqlite3.connect(str(store._db_path)) as connection:
            row = connection.execute(
                "SELECT file_name FROM blocks WHERE block_hash = ?",
                (block_hash.hex(),),
            ).fetchone()
            assert row is not None
            payload = store.cache_dir / str(row[0])
            connection.execute(
                "INSERT INTO block_write_pins "
                "(block_hash, owner_lease_id, fence_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (block_hash.hex(), "999999-dead", "stale-fence", time.time()),
            )
            connection.commit()

        total = store.global_budget.enforce(force=True).bytes_after
        after_block = total - payload.stat().st_size
        # Allow one SQLite WAL page for stale-pin deletion while still forcing
        # the much larger payload block through the global trim.
        cap = ((after_block * 10 + 8) // 9) + 16 * 1024
        assert after_block < cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.compliant is True
        assert store.has_block(block_hash) is False
        with sqlite3.connect(str(store._db_path)) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM block_write_pins"
            ).fetchone()[0] == 0
    finally:
        if trim_budget is not None:
            trim_budget.close()
        store.shutdown()


def test_store_cache_exception_terminates_begun_write_fence(
    tmp_path,
    monkeypatch,
):
    import vmlx_engine.prefix_cache as prefix_cache_module
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    manager = PagedCacheManager(
        block_size=4,
        max_blocks=16,
        disk_store=store,
    )
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    cache_data = [
        {
            "state": (
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
            ),
            "meta_state": (4,),
            "class_name": "KVCache",
        }
    ]

    def fail_after_fence_begin():
        raise RuntimeError("forced post-begin store failure")

    monkeypatch.setattr(prefix_cache_module.mx, "synchronize", fail_after_fence_begin)
    try:
        with pytest.raises(RuntimeError, match="forced post-begin store failure"):
            cache.store_cache("resp-aborted", [1, 2, 3, 4], cache_data)

        fence_id = store.get_stats()["write_pipeline"]["recent_fences"][-1]["fence_id"]
        stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["request_id"] == "resp-aborted"
    assert fence["sealed"] is True
    assert fence["seal_enqueued"] is True
    assert fence["producer_aborted"] is True
    assert fence["post_eviction_complete"] is True
    assert fence["expected"] == 0
    assert fence["queued"] == 0
    assert all(item["sealed"] for item in stats["write_pipeline"]["recent_fences"])


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, False), ("1", True)],
)
def test_prefix_cache_passes_explicit_strict_fence_mode_from_startup_env(
    tmp_path,
    monkeypatch,
    env_value,
    expected,
):
    import vmlx_engine.prefix_cache as prefix_cache_module
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    if env_value is None:
        monkeypatch.delenv("VMLX_STRICT_BLOCK_DISK_WRITE_FENCE", raising=False)
    else:
        monkeypatch.setenv("VMLX_STRICT_BLOCK_DISK_WRITE_FENCE", env_value)
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    manager = PagedCacheManager(block_size=4, max_blocks=16, disk_store=store)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    observed: list[bool] = []
    original_begin = store.begin_write_fence

    def record_begin(request_id, *, strict_reconcile=False):
        observed.append(bool(strict_reconcile))
        return original_begin(
            request_id,
            strict_reconcile=strict_reconcile,
        )

    def stop_after_fence_begin():
        raise RuntimeError("stop after fence begin")

    monkeypatch.setattr(store, "begin_write_fence", record_begin)
    monkeypatch.setattr(
        prefix_cache_module.mx,
        "synchronize",
        stop_after_fence_begin,
    )
    cache_data = [
        {
            "state": (
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
                mx.ones((1, 1, 4, 8), dtype=mx.float16),
            ),
            "meta_state": (4,),
            "class_name": "KVCache",
        }
    ]
    try:
        with pytest.raises(RuntimeError, match="stop after fence begin"):
            cache.store_cache("strict-env", [1, 2, 3, 4], cache_data)
    finally:
        store.shutdown()

    assert observed == [expected]
    assert cache.get_stats()["strict_block_disk_write_fence"] is expected


def test_prefix_cache_stats_default_strict_fence_for_legacy_fixture():
    from vmlx_engine.prefix_cache import (
        _CACHE_TYPE_PRIORITY,
        BlockAwarePrefixCache,
    )

    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = SimpleNamespace(get_memory_usage=lambda: {})
    cache._hits = 0
    cache._misses = 0
    cache._tokens_saved = 0
    cache._request_tables = {}
    cache._entries_by_type = {key: [] for key in _CACHE_TYPE_PRIORITY}

    assert cache.get_stats()["strict_block_disk_write_fence"] is False


def test_constructor_failure_after_budget_publish_releases_owner_lease(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget

    def fail_startup_reconcile(*_args, **_kwargs):
        raise RuntimeError("forced startup reconcile failure")

    monkeypatch.setattr(
        GlobalDiskCacheBudget,
        "enforce",
        fail_startup_reconcile,
    )
    with pytest.raises(RuntimeError, match="forced startup reconcile failure"):
        BlockDiskStore(str(tmp_path), max_size_gb=1)

    lease_dir = tmp_path / ".vmlx-global-cache-budget-leases"
    assert list(lease_dir.glob("*.json")) == []


def test_write_fence_eviction_failure_is_terminal_and_prunable(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    def fail_eviction(*_args, **_kwargs):
        raise RuntimeError("forced eviction failure")

    original_account = store.global_budget.account_finalized_write_locked
    monkeypatch.setattr(
        store.global_budget,
        "account_finalized_write_locked",
        fail_eviction,
    )
    try:
        fence_id = store.begin_write_fence("resp-evict")
        assert store.write_block_async(
            b"e" * 32,
            _cache_data(1),
            8,
            request_id="resp-evict",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        stats, fence = _wait_for_fence(store, fence_id)
        assert fence["post_eviction_complete"] is True
        assert "forced eviction failure" in fence["post_eviction_error"]
        assert stats["blocks_on_disk"] == 0

        # Once aggregate accounting cannot certify a write, future writes fail
        # closed and cannot grow the physical store until a new owner/session
        # performs a healthy startup reconciliation.
        monkeypatch.setattr(
            store.global_budget,
            "account_finalized_write_locked",
            original_account,
        )
        assert store.write_block_async(
            b"r" * 32,
            _cache_data(2),
            8,
        ) is False
        assert list(store.blocks_dir.rglob("*.safetensors")) == []
    finally:
        store.shutdown()


def test_write_fence_finalization_error_does_not_kill_writer(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    def fail_finalize(_conn, _fence_id):
        raise RuntimeError("forced finalization failure")

    monkeypatch.setattr(store, "_finalize_write_fence", fail_finalize)
    try:
        fence_id = store.begin_write_fence("resp-finalize")
        assert store.write_block_async(
            b"f" * 32,
            _cache_data(9),
            8,
            request_id="resp-finalize",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["post_eviction_complete"] is True
    assert "forced finalization failure" in fence["post_eviction_error"]
    assert stats["write_pipeline"]["writer_alive"] is True


def test_write_fence_waits_for_active_producer_before_finalizing(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence_id = store.begin_write_fence("resp-race")
        assert store._write_fence_expected(fence_id)
        assert store.seal_write_fence(fence_id)

        fence = store.get_stats()["write_pipeline"]["recent_fences"][-1]
        assert fence["sealed"] is True
        assert fence["seal_enqueued"] is False
        assert fence["post_eviction_complete"] is False

        store._write_fence_queue_result(fence_id, block_hash=b"r" * 32)
        # The synthetic producer does not queue a real disk item, so mirror the
        # writer's publication callback explicitly before waiting.
        store._write_fence_completion(fence_id, failed=False)
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()


def test_request_mismatch_balances_active_producer_and_terminalizes(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence_id = store.begin_write_fence("expected-request")
        assert store.write_block_async(
            b"m" * 32,
            _cache_data(1),
            8,
            request_id="wrong-request",
            fence_id=fence_id,
        ) is False
        assert store.seal_write_fence(fence_id)
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["expected"] == 1
    assert fence["failed"] == 1
    assert fence["post_eviction_complete"] is True


def test_normal_fence_uses_accounting_without_full_root_scan(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    calls = 0
    original = store.global_budget._scan_locked

    def counted_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store.global_budget, "_scan_locked", counted_scan)
    try:
        _stats, fence = _write_request(
            store,
            request_id="resp-normal-accounted",
            block_hash=b"n" * 32,
            value=1,
        )
        assert fence["post_eviction_complete"] is True
        assert calls == 0
    finally:
        store.shutdown()


def test_explicit_strict_fence_forces_physical_reconciliation(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    calls = 0
    original = store.global_budget._scan_locked

    def counted_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store.global_budget, "_scan_locked", counted_scan)
    try:
        startup_generation = (
            store.global_budget.last_result.reconciliation_generation
        )
        fence_id = store.begin_write_fence(
            "resp-strict",
            strict_reconcile=True,
        )
        assert store.write_block_async(
            b"s" * 32,
            _cache_data(1),
            8,
            request_id="resp-strict",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        _stats, fence = _wait_for_fence(store, fence_id)
        assert fence["post_eviction_complete"] is True
        # one physical scan: nothing was evicted, so the post-trim rescan is skipped
        assert calls >= 1
        assert (
            fence["global_reconciliation_generation"]
            > startup_generation
        )
    finally:
        store.shutdown()


def test_fence_block_wait_rejects_commit_visible_before_budget_eviction(
    tmp_path,
    monkeypatch,
):
    """Index visibility before accounting must not release a native fallback."""
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    account_entered = threading.Event()
    allow_account = threading.Event()
    original_account = store.global_budget.account_finalized_write_locked

    def delayed_account(
        net_bytes_delta,
        *,
        require_reconciled=False,
        protected_blocks=None,
    ):
        account_entered.set()
        if not allow_account.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to exercise budget eviction")
        return original_account(
            net_bytes_delta,
            require_reconciled=require_reconciled,
            protected_blocks=protected_blocks,
        )

    monkeypatch.setattr(
        store.global_budget,
        "account_finalized_write_locked",
        delayed_account,
    )
    # Force the just-committed payload through the eviction/error path. The
    # SQLite index itself is larger than this ceiling, so the publication cannot
    # be certified retained even though its row is briefly readable.
    store.max_size_bytes = 1
    store.global_budget._requested_max_size_bytes = 1
    store.global_budget._publish_budget(1)
    block_hash = b"v" * 32
    fence_id = store.begin_write_fence("dsv4-pre-eviction-visibility")
    result: dict[str, set[bytes]] = {}

    def wait_for_retained_block():
        result["hashes"] = store.wait_for_write_fence_blocks(
            fence_id,
            [block_hash],
            timeout=5.0,
        )

    waiter = threading.Thread(target=wait_for_retained_block, daemon=True)
    waiter_started = False
    try:
        assert store.write_block_async(
            block_hash,
            _cache_data(7),
            8,
            request_id="dsv4-pre-eviction-visibility",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        assert account_entered.wait(timeout=5.0)

        # This is the old, insufficient boundary: _write_block() has committed
        # the row/file, but aggregate accounting and eviction are still paused.
        assert store.wait_for_blocks([block_hash], timeout=0.5) == {block_hash}

        waiter.start()
        waiter_started = True
        time.sleep(0.05)
        assert waiter.is_alive()

        allow_account.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive()
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        allow_account.set()
        if waiter_started:
            waiter.join(timeout=5.0)
        store.shutdown()

    assert result["hashes"] == set()
    assert fence["post_eviction_complete"] is True
    assert fence.get("post_eviction_error") or fence["retained"] == 0
    with sqlite3.connect(str(store._db_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM block_write_pins WHERE fence_id = ?",
            (fence_id,),
        ).fetchone()[0] == 0


def test_fence_block_wait_returns_exact_hash_after_terminal_retention(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    block_hash = b"r" * 32
    fence_id = store.begin_write_fence("dsv4-retained")
    try:
        assert store.write_block_async(
            block_hash,
            _cache_data(8),
            8,
            request_id="dsv4-retained",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        assert store.wait_for_write_fence_blocks(
            fence_id,
            [block_hash, b"m" * 32],
            timeout=5.0,
        ) == {block_hash}
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["expected"] == fence["completed"] == fence["retained"] == 1
    assert fence["failed"] == fence["dropped"] == 0


def test_fence_block_wait_fails_closed_on_terminal_error(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    fence_id = store.begin_write_fence("dsv4-terminal-error")
    try:
        store._fail_write_fence(fence_id, "forced terminal error")
        assert store.wait_for_write_fence_blocks(
            fence_id,
            [b"e" * 32],
            timeout=0.1,
        ) == set()
        stats = store.get_stats()
    finally:
        store.shutdown()

    fence = next(
        item
        for item in stats["write_pipeline"]["recent_fences"]
        if item["fence_id"] == fence_id
    )
    assert fence["post_eviction_complete"] is True
    assert fence["post_eviction_error"] == "forced terminal error"


def test_full_write_queue_skips_metal_serialization(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    called = False

    def unexpected_save(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("save must not run when queue is already full")

    monkeypatch.setattr(store._write_queue, "full", lambda: True)
    monkeypatch.setattr(mx, "save_safetensors", unexpected_save)
    try:
        assert store.write_block_async(
            b"q" * 32,
            _cache_data(1),
            8,
        ) is False
        assert called is False
    finally:
        store.shutdown()


def test_noncompliant_startup_budget_rejects_unfenced_growth(
    tmp_path,
    monkeypatch,
):
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    protected = blocks / "recent.tmp.safetensors"
    protected.write_bytes(b"p" * 64_000)
    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=1 / 1024**3,
    )
    called = False

    def unexpected_save(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("disabled aggregate writer must not serialize")

    monkeypatch.setattr(mx, "save_safetensors", unexpected_save)
    try:
        assert store._global_budget_write_enabled is False
        assert store.write_block_async(
            b"g" * 32,
            _cache_data(1),
            8,
        ) is False
        assert called is False
        assert protected.exists()
    finally:
        store.shutdown()


def test_noncompliant_startup_budget_terminalizes_fenced_failure(tmp_path):
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    (blocks / "recent.tmp.safetensors").write_bytes(b"p" * 64_000)
    store = BlockDiskStore(
        str(tmp_path),
        max_size_gb=1 / 1024**3,
    )
    try:
        assert store._global_budget_write_enabled is False
        fence_id = store.begin_write_fence("over-budget")
        assert not store.write_block_async(
            b"o" * 32,
            _cache_data(1),
            8,
            request_id="over-budget",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()

    assert fence["expected"] == 1
    assert fence["failed"] == 1
    assert fence["post_eviction_complete"] is True


def test_unlimited_owner_cannot_bypass_unhealthy_finite_root_budget(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget

    root = tmp_path / "root"
    finite_owner = GlobalDiskCacheBudget(root, 1)

    def unhealthy_startup(_self, **_kwargs):
        return SimpleNamespace(
            accounted=False,
            compliant=False,
            bytes_after=2,
            max_size_bytes=1,
            error="forced aggregate reconciliation failure",
        )

    monkeypatch.setattr(GlobalDiskCacheBudget, "enforce", unhealthy_startup)
    store = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    try:
        assert store.max_size_bytes == 0
        assert store._global_budget_write_enabled is False
        fence_id = store.begin_write_fence("unlimited-owner")
        assert not store.write_block_async(
            b"u" * 32,
            _cache_data(1),
            8,
            request_id="unlimited-owner",
            fence_id=fence_id,
        )
        assert store.seal_write_fence(fence_id)
        _stats, fence = _wait_for_fence(store, fence_id)
    finally:
        store.shutdown()
        finite_owner.close()

    assert fence["expected"] == 1
    assert fence["failed"] == 1
    assert fence["post_eviction_complete"] is True


def test_unlimited_owner_clear_does_not_reenable_unhealthy_finite_root(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget

    root = tmp_path / "root"
    finite_owner = GlobalDiskCacheBudget(root, 10_000_000)
    store = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    unhealthy = SimpleNamespace(accounted=False, compliant=False)
    monkeypatch.setattr(
        store.global_budget,
        "account_finalized_write_locked",
        lambda *_args, **_kwargs: unhealthy,
    )
    try:
        assert store._global_budget_write_enabled is True
        store.clear()
        assert store._global_budget_write_enabled is False
    finally:
        store.shutdown()
        finite_owner.close()


def test_stopped_writer_terminalizes_unqueued_fence_sentinel(
    tmp_path,
    monkeypatch,
):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)

    try:
        fence_id = store.begin_write_fence("resp-sentinel-full")
        monkeypatch.setattr(
            store,
            "_try_enqueue_write_item",
            lambda *_args, **_kwargs: "writer_stopped",
        )
        assert store.seal_write_fence(fence_id) is False
        fence = next(
            item
            for item in store.get_stats()["write_pipeline"]["recent_fences"]
            if item["fence_id"] == fence_id
        )
    finally:
        store.shutdown()

    assert fence["seal_failed"] is True
    assert fence["post_eviction_complete"] is True
    assert "writer quiescing" in fence["post_eviction_error"]


def test_clear_terminalizes_unsettled_write_fences(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        fence_id = store.begin_write_fence("resp-clear")
        assert store._write_fence_expected(fence_id)
        assert store.seal_write_fence(fence_id)
        store.clear()
        fence = next(
            item
            for item in store.get_stats()["write_pipeline"]["recent_fences"]
            if item["fence_id"] == fence_id
        )
    finally:
        store.shutdown()

    assert fence["sealed"] is True
    assert fence["post_eviction_complete"] is True
    assert "disk cache cleared" in fence["post_eviction_error"]


def test_native_queue_backpressure_admits_4096_fake_blocks_without_loss():
    """Exercise 4096-block pressure without allocating model/cache tensors."""

    store = BlockDiskStore.__new__(BlockDiskStore)
    store._write_lifecycle_lock = threading.Lock()
    store._write_lifecycle = threading.Condition(store._write_lifecycle_lock)
    store._accepting_writes = True
    store._shutdown_started = False
    store._pending_write_items = 0
    store._write_queue = queue.Queue(maxsize=4)
    store._writer_thread = SimpleNamespace(is_alive=lambda: True)
    consumed = []

    def consume():
        for _ in range(4096):
            consumed.append(store._write_queue.get(timeout=2.0))
            store._complete_write_items(1)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    for index in range(4096):
        assert store._try_enqueue_write_item(
            ("fake", index),
            timeout=2.0,
        ) == "queued"
    consumer.join(timeout=5.0)

    assert not consumer.is_alive()
    assert len(consumed) == 4096
    assert store._pending_write_items == 0
    assert store._write_queue.empty()


def test_native_pending_byte_backpressure_retries_after_release():
    store = BlockDiskStore.__new__(BlockDiskStore)
    store._stats_lock = threading.Lock()
    store._pending_write_condition = threading.Condition(store._stats_lock)
    store._max_pending_write_bytes = 16
    store._pending_write_bytes = 16
    store._pending_write_byte_drops = 0
    result = []

    waiter = threading.Thread(
        target=lambda: result.append(
            store._reserve_pending_write_bytes(16, timeout=1.0)
        ),
        daemon=True,
    )
    waiter.start()
    time.sleep(0.02)
    assert waiter.is_alive()
    store._release_pending_write_bytes(16)
    waiter.join(timeout=2.0)

    assert result == [True]
    assert store._pending_write_bytes == 16
    assert store._pending_write_byte_drops == 0
    store._release_pending_write_bytes(16)


def test_native_chain_abort_accounts_all_4096_expected_blocks():
    store = BlockDiskStore.__new__(BlockDiskStore)
    store._stats_lock = threading.Lock()
    fence_id = "block-write-4096-accounting"
    store._write_fences = {
        fence_id: {
            "expected": 1,
            "queued": 0,
            "failed": 0,
            "dropped": 1,
            "sealed": False,
            "seal_enqueued": False,
            "seal_failed": False,
            "post_eviction_complete": False,
            "_active": 0,
        }
    }

    store.record_write_fence_unadmitted(fence_id, 4095)

    state = store._write_fences[fence_id]
    assert state["expected"] == 4096
    assert state["dropped"] == 4096


def test_partial_terminal_fence_releases_survivors_only_when_requested():
    retained_hash = b"r" * 32
    evicted_hash = b"e" * 32
    fence_id = "block-write-partial-cap"
    store = BlockDiskStore.__new__(BlockDiskStore)
    store._stats_lock = threading.Lock()
    store._write_fences = {
        fence_id: {
            "sealed": True,
            "seal_enqueued": True,
            "seal_failed": False,
            "producer_aborted": False,
            "post_eviction_complete": True,
            "expected": 2,
            "queued": 2,
            "completed": 2,
            "retained": 1,
            "failed": 0,
            "dropped": 0,
        }
    }
    store._writer_thread = SimpleNamespace(is_alive=lambda: True)
    store.global_budget = SimpleNamespace(
        mutation_guard=lambda: nullcontext(True)
    )
    store._has_block_guarded = lambda value: value == retained_hash

    # Legacy callers remain all-or-nothing.
    assert store.wait_for_write_fence_blocks(
        fence_id,
        [retained_hash, evicted_hash],
        timeout=0.0,
    ) == set()
    # Native settlement releases only the exact hash that survived cap eviction.
    assert store.wait_for_write_fence_blocks(
        fence_id,
        [retained_hash, evicted_hash],
        timeout=0.0,
        allow_partial=True,
    ) == {retained_hash}


def test_full_data_queue_defers_fence_control_until_payload_is_processed():
    """A full data FIFO must not make an admitted native fence unsealable."""

    fence_id = "block-write-deferred-control"
    store = BlockDiskStore.__new__(BlockDiskStore)
    store._stats_lock = threading.Lock()
    store._write_lifecycle_lock = threading.Lock()
    store._write_lifecycle = threading.Condition(store._write_lifecycle_lock)
    store._accepting_writes = True
    store._shutdown_started = False
    store._pending_write_items = 1
    store._writer_thread = SimpleNamespace(is_alive=lambda: True)
    store._write_queue = queue.Queue(maxsize=1)
    store._write_queue.put_nowait((b"p" * 32, "payload"))
    store._write_fences = {
        fence_id: {
            "expected": 1,
            "queued": 1,
            "completed": 0,
            "_processed": 0,
            "failed": 0,
            "dropped": 0,
            "sealed": True,
            "seal_enqueued": False,
            "seal_failed": False,
            "post_eviction_complete": False,
            "_active": 0,
            "_admission_timeout": 0.0,
        }
    }

    assert store._enqueue_write_fence_sentinel(fence_id) is True
    state = store._write_fences[fence_id]
    assert state["seal_enqueued"] is True
    assert state["seal_failed"] is False
    assert state["_sentinel_deferred"] is True

    store._write_queue.get_nowait()
    store._write_fence_completion(fence_id, failed=False)
    assert store._write_fence_publication_ready_locked(state) is True


def test_clear_waits_for_cross_process_shared_cache_operation(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    code = "\n".join(
        (
            "import sys",
            "from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget",
            "budget = GlobalDiskCacheBudget(sys.argv[1], 0)",
            "with budget.mutation_guard() as locked:",
            "    assert locked",
            "    print('LOCKED', flush=True)",
            "    input()",
            "budget.close()",
        )
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    complete = threading.Event()
    errors: list[BaseException] = []

    def run_clear() -> None:
        try:
            store.clear()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            complete.set()

    thread = threading.Thread(target=run_clear)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        thread.start()
        assert complete.wait(0.2) is False
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait(timeout=10)
        thread.join(timeout=10)
        assert child.returncode == 0, child.stderr.read() if child.stderr else ""
        assert complete.is_set()
        assert errors == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        thread.join(timeout=5)
        store.shutdown()


def test_shutdown_releases_only_its_aggregate_budget_lease(tmp_path):
    root = tmp_path / "root"
    low = BlockDiskStore(
        str(root / "low"),
        max_size_gb=1,
        global_cache_root=str(root),
    )
    high = BlockDiskStore(
        str(root / "high"),
        max_size_gb=2,
        global_cache_root=str(root),
    )
    lease_dir = root / ".vmlx-global-cache-budget-leases"
    try:
        assert len(list(lease_dir.glob("*.json"))) == 2
        assert high.global_budget.enforce(force=True).max_size_bytes == 1024**3

        low.shutdown()

        assert len(list(lease_dir.glob("*.json"))) == 1
        assert high.global_budget.enforce(force=True).max_size_bytes == 2 * 1024**3
    finally:
        low.shutdown()
        high.shutdown()

    assert list(lease_dir.glob("*.json")) == []


def test_shutdown_timeout_defers_lease_release_until_writer_stops(
    tmp_path,
    monkeypatch,
):
    release_writer = threading.Event()
    original_writer = BlockDiskStore._background_writer

    def delayed_writer(_self):
        release_writer.wait(timeout=5.0)

    monkeypatch.setattr(BlockDiskStore, "_background_writer", delayed_writer)
    first = BlockDiskStore(str(tmp_path), max_size_gb=1)
    monkeypatch.setattr(BlockDiskStore, "_background_writer", original_writer)
    first._writer_shutdown_timeout_seconds = 0.01
    first_lease = first.global_budget._lease_path()
    second = None
    try:
        first.shutdown()
        assert first_lease.exists()
        assert first._delayed_shutdown_thread is not None

        second = BlockDiskStore(str(tmp_path), max_size_gb=1)
        second_lease = second.global_budget._lease_path()
        assert second_lease.exists()

        release_writer.set()
        deadline = time.monotonic() + 5.0
        while first_lease.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert first._shutdown_finalized is True
        assert not first_lease.exists()
        assert second_lease.exists()
    finally:
        release_writer.set()
        if first._delayed_shutdown_thread is not None:
            first._delayed_shutdown_thread.join(timeout=5.0)
        if second is not None:
            second.shutdown()


def test_transient_budget_failure_recovers_at_bounded_retry(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    healthy = SimpleNamespace(accounted=True, compliant=True)
    monkeypatch.setattr(
        store.global_budget,
        "enforce",
        lambda **_kwargs: healthy,
    )
    try:
        store._disable_global_budget_writes()
        store._budget_recovery_interval_ns = 0
        assert store.write_block_async(b"h" * 32, _cache_data(1), 8)
        assert store.wait_for_blocks([b"h" * 32], timeout=5.0) == {b"h" * 32}
        assert store._global_budget_write_enabled is True
    finally:
        store.shutdown()


def test_block_disk_namespace_must_be_inside_aggregate_budget_root(tmp_path):
    budget_root = tmp_path / "managed-root"
    outside_namespace = tmp_path / "outside-root" / "namespace"

    with pytest.raises(
        ValueError,
        match="namespace must be contained by its aggregate budget root",
    ):
        BlockDiskStore(
            str(outside_namespace),
            max_size_gb=1,
            global_cache_root=str(budget_root),
        )

    # Reject before claiming or creating the out-of-budget namespace.  Leaving
    # payloads there would make the user-visible aggregate max unenforceable.
    assert not outside_namespace.exists()


def test_successful_block_read_refreshes_cross_namespace_global_lru(tmp_path):
    root = tmp_path / "root"
    first = BlockDiskStore(
        str(root / "aaaaaaaaaaaa"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    second = BlockDiskStore(
        str(root / "bbbbbbbbbbbb"),
        max_size_gb=0,
        global_cache_root=str(root),
    )
    first_hash = b"a" * 32
    second_hash = b"b" * 32
    trim_budget = None
    try:
        _write_request(
            first,
            request_id="touch-first",
            block_hash=first_hash,
            value=1,
            cache_data=_large_cache_data(1),
        )
        _write_request(
            second,
            request_id="touch-second",
            block_hash=second_hash,
            value=2,
            cache_data=_large_cache_data(2),
        )

        def indexed_path(store, block_hash):
            with sqlite3.connect(str(store._db_path)) as connection:
                row = connection.execute(
                    "SELECT file_name FROM blocks WHERE block_hash = ?",
                    (block_hash.hex(),),
                ).fetchone()
            assert row is not None
            return store.cache_dir / row[0]

        first_path = indexed_path(first, first_hash)
        second_path = indexed_path(second, second_hash)
        now = time.time()
        for store, block_hash, path, accessed in (
            (first, first_hash, first_path, now - 200),
            (second, second_hash, second_path, now - 100),
        ):
            os.utime(path, (accessed, accessed))
            with sqlite3.connect(str(store._db_path)) as connection:
                connection.execute(
                    "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                    (accessed, block_hash.hex()),
                )
                connection.commit()

        # A successful real deserialize updates the physical mtime immediately,
        # so even a saturated async metadata queue cannot evict the fresh hit.
        assert first.read_block(first_hash) is not None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pipeline = first.get_stats()["write_pipeline"]
            if (
                pipeline["queue_depth"] == 0
                and pipeline["inflight"] == 0
                and pipeline["pending_items"] == 0
            ):
                break
            time.sleep(0.01)
        assert pipeline["pending_items"] == 0

        total = first.global_budget.enforce(force=True).bytes_after
        second_size = second_path.stat().st_size
        remaining_after_oldest = total - second_size
        cap = ((remaining_after_oldest * 10 + 8) // 9) + 1024
        assert cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.compliant is True
        assert result.bytes_after <= result.max_size_bytes
        assert first_path.exists()
        assert not second_path.exists()
        with sqlite3.connect(str(second._db_path)) as connection:
            assert connection.execute(
                "SELECT 1 FROM blocks WHERE block_hash = ?",
                (second_hash.hex(),),
            ).fetchone() is None
    finally:
        if trim_budget is not None:
            trim_budget.close()
        first.shutdown()
        second.shutdown()


def test_cross_process_writers_share_one_cap_and_trim_oldest_namespace(tmp_path):
    root = tmp_path / "root"
    child_code = "\n".join(
        (
            "import sys, time",
            "import mlx.core as mx",
            "from vmlx_engine.block_disk_store import BlockDiskStore",
            "root, namespace, byte_text = sys.argv[1:4]",
            "store = BlockDiskStore(namespace, max_size_gb=0, global_cache_root=root)",
            "print('READY', flush=True)",
            "assert input().strip() == 'GO'",
            "value = float(int(byte_text))",
            "keys = mx.full((1, 1, 8, 4096), value, dtype=mx.float16)",
            "values = mx.full((1, 1, 8, 4096), value + 1, dtype=mx.float16)",
            "mx.eval(keys, values)",
            "block_hash = bytes([int(byte_text)]) * 32",
            "assert store.write_block_async(block_hash, [('kv', keys, values)], 8)",
            "assert store.wait_for_blocks([block_hash], timeout=15.0) == {block_hash}",
            "print('DONE', flush=True)",
            "assert input().strip() == 'STOP'",
            "store.shutdown()",
        )
    )
    children = []
    trim_budget = None
    try:
        for name, byte_value in (("aaaaaaaaaaaa", 97), ("bbbbbbbbbbbb", 98)):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(root),
                    str(root / name),
                    str(byte_value),
                ],
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(child)
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "READY"
        for child in children:
            assert child.stdin is not None
            child.stdin.write("GO\n")
            child.stdin.flush()
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "DONE"

        indexed = []
        now = time.time()
        for index, (name, byte_value) in enumerate(
            (("aaaaaaaaaaaa", 97), ("bbbbbbbbbbbb", 98))
        ):
            namespace = root / name
            block_hash = (bytes([byte_value]) * 32).hex()
            with sqlite3.connect(str(namespace / "block_index.db")) as connection:
                row = connection.execute(
                    "SELECT file_name FROM blocks WHERE block_hash = ?",
                    (block_hash,),
                ).fetchone()
                assert row is not None
                accessed = now - 200 + (index * 100)
                connection.execute(
                    "UPDATE blocks SET last_accessed = ? WHERE block_hash = ?",
                    (accessed, block_hash),
                )
                connection.commit()
            path = namespace / row[0]
            os.utime(path, (accessed, accessed))
            indexed.append((namespace, block_hash, path))

        probe = GlobalDiskCacheBudget(root, 0, orphan_grace_seconds=0)
        try:
            total = probe.enforce(force=True).bytes_after
        finally:
            probe.close()
        oldest_size = indexed[0][2].stat().st_size
        remaining_after_oldest = total - oldest_size
        cap = ((remaining_after_oldest * 10 + 8) // 9) + 1024
        assert cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.accounted and result.compliant
        assert result.bytes_after <= result.max_size_bytes
        assert not indexed[0][2].exists()
        assert indexed[1][2].exists()
        with sqlite3.connect(str(indexed[0][0] / "block_index.db")) as connection:
            assert connection.execute(
                "SELECT 1 FROM blocks WHERE block_hash = ?",
                (indexed[0][1],),
            ).fetchone() is None
        with sqlite3.connect(str(indexed[1][0] / "block_index.db")) as connection:
            assert connection.execute(
                "SELECT 1 FROM blocks WHERE block_hash = ?",
                (indexed[1][1],),
            ).fetchone() == (1,)
        assert not [
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                ".tmp." in path.name
                or path.name.endswith((".tmp", ".tmp.safetensors", ".tmp.json"))
            )
        ]
    finally:
        if trim_budget is not None:
            trim_budget.close()
        for child in children:
            if child.poll() is None and child.stdin is not None:
                try:
                    child.stdin.write("STOP\n")
                    child.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            assert child.returncode == 0, (
                child.stderr.read() if child.stderr is not None else ""
            )

    lease_dir = root / ".vmlx-global-cache-budget-leases"
    assert list(lease_dir.glob("*.json")) == []

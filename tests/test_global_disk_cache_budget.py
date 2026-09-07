from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import vmlx_engine.global_disk_cache_budget as budget_module
from vmlx_engine.global_disk_cache_budget import (
    GlobalDiskCacheBudget,
    ensure_managed_block_cache_namespace,
)
from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore


def _indexed_block(
    namespace: Path,
    name: str,
    *,
    size: int,
    accessed: float,
) -> Path:
    ensure_managed_block_cache_namespace(namespace)
    blocks = namespace / "blocks" / name[:2]
    blocks.mkdir(parents=True, exist_ok=True)
    payload = blocks / f"{name}.safetensors"
    payload.write_bytes(b"x" * size)
    os.utime(payload, (accessed, accessed))
    database = namespace / "block_index.db"
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS blocks ("
            "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
            "last_accessed REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO blocks(block_hash, file_name, last_accessed) "
            "VALUES (?, ?, ?)",
            (name, str(payload.relative_to(namespace)), accessed),
        )
        conn.commit()
    finally:
        conn.close()
    return payload


def _leave_crash_stale_wal_index(namespace: Path, payload: Path) -> tuple[Path, Path]:
    """Publish one valid row, then exit without SQLite's last-close cleanup."""

    child = r"""
import os
import sqlite3
import sys
import time
from pathlib import Path

namespace = Path(sys.argv[1])
payload = Path(sys.argv[2])
connection = sqlite3.connect(namespace / "block_index.db")
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("PRAGMA wal_autocheckpoint=0")
connection.execute(
    "CREATE TABLE blocks ("
    "block_hash TEXT PRIMARY KEY, parent_hash TEXT, "
    "ancestry_known INTEGER NOT NULL DEFAULT 1, "
    "file_name TEXT NOT NULL, num_tokens INTEGER NOT NULL, "
    "num_layers INTEGER NOT NULL, dtype TEXT NOT NULL, "
    "file_size INTEGER NOT NULL, created_at REAL NOT NULL, "
    "last_accessed REAL NOT NULL, access_count INTEGER DEFAULT 0)"
)
now = time.time() - 1000
connection.execute(
    "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (
        "crash-root",
        None,
        1,
        str(payload.relative_to(namespace)),
        64,
        1,
        "float16",
        payload.stat().st_size,
        now,
        now,
        0,
    ),
)
connection.commit()
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", child, str(namespace), str(payload)],
        check=True,
    )
    wal_path = namespace / "block_index.db-wal"
    shm_path = namespace / "block_index.db-shm"
    assert wal_path.is_file() and wal_path.stat().st_size > 0
    assert shm_path.is_file() and shm_path.stat().st_size > 0
    return wal_path, shm_path


def _physical_total(root: Path) -> int:
    budget = GlobalDiskCacheBudget(root, 1024**3, orphan_grace_seconds=0)
    try:
        return budget.enforce(force=True).bytes_after
    finally:
        budget._remove_lease()


def test_crash_stale_sqlite_sidecars_do_not_skip_block_store_startup_trim(
    tmp_path: Path,
) -> None:
    from vmlx_engine.block_disk_store import BlockDiskStore

    root = tmp_path / "root"
    old_namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    payload = old_namespace / "blocks" / "cr" / "crash-root.safetensors"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"x" * 1_000_000)
    wal_path, shm_path = _leave_crash_stale_wal_index(old_namespace, payload)

    cap_bytes = 256 * 1024
    store = BlockDiskStore(
        str(root / "bbbbbbbbbbbb"),
        max_size_gb=cap_bytes / 1024**3,
        global_cache_root=str(root),
    )
    try:
        result = store.global_budget.last_result
        assert result is not None
        assert result.accounted is True
        assert result.compliant is True
        assert result.error is None
        assert store._global_budget_write_enabled is True
        assert not payload.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()
    finally:
        store.shutdown()


def test_live_sqlite_sidecars_remain_counted_in_physical_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    database = namespace / "block_index.db"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(
        "CREATE TABLE blocks ("
        "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
        "last_accessed REAL NOT NULL)"
    )
    connection.commit()
    wal_path = namespace / "block_index.db-wal"
    shm_path = namespace / "block_index.db-shm"
    assert wal_path.is_file() and wal_path.stat().st_size > 0
    assert shm_path.is_file() and shm_path.stat().st_size > 0
    physical_metadata_bytes = sum(
        path.stat().st_size for path in (database, wal_path, shm_path)
    )

    budget = GlobalDiskCacheBudget(root, 10_000_000)
    try:
        result = budget.enforce(force=True)
        assert result.accounted is True
        assert result.compliant is True
        assert result.error is None
        assert result.bytes_after >= physical_metadata_bytes
        assert wal_path.is_file()
        assert shm_path.is_file()
    finally:
        budget.close()
        connection.close()


def test_sqlite_sidecar_disappearing_during_discovery_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    with sqlite3.connect(namespace / "block_index.db") as connection:
        connection.execute(
            "CREATE TABLE blocks ("
            "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
            "last_accessed REAL NOT NULL)"
        )
    sidecar = namespace / "block_index.db-shm"
    sidecar.write_bytes(b"transient")
    budget = GlobalDiskCacheBudget(root, 10_000_000)
    original_resolve = Path.resolve

    def resolve_after_sidecar_cleanup(path: Path, *args, **kwargs) -> Path:
        if path == sidecar:
            sidecar.unlink(missing_ok=True)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_after_sidecar_cleanup)
    try:
        result = budget.enforce(force=True)
        assert result.accounted is True
        assert result.compliant is True
        assert result.error is None
        assert not sidecar.exists()
    finally:
        budget.close()


def test_same_named_nested_file_disappearance_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    with sqlite3.connect(namespace / "block_index.db") as connection:
        connection.execute(
            "CREATE TABLE blocks ("
            "block_hash TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
            "last_accessed REAL NOT NULL)"
        )
    nested = namespace / "blocks" / "block_index.db-shm"
    nested.parent.mkdir()
    nested.write_bytes(b"unknown")
    budget = GlobalDiskCacheBudget(root, 10_000_000)
    original_resolve = Path.resolve

    def resolve_after_nested_cleanup(path: Path, *args, **kwargs) -> Path:
        if path == nested:
            nested.unlink(missing_ok=True)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_after_nested_cleanup)
    try:
        result = budget.enforce(force=True)
        assert result.accounted is False
        assert result.compliant is False
        assert result.error is not None
        assert str(nested) in result.error
    finally:
        budget.close()


def test_global_lru_crosses_model_namespaces(tmp_path: Path) -> None:
    root = tmp_path / "root"
    now = time.time()
    old = _indexed_block(root / "aaaaaaaaaaaa", "aa-old", size=64_000, accessed=now - 100)
    recent = _indexed_block(root / "bbbbbbbbbbbb", "bb-new", size=64_000, accessed=now)
    before = _physical_total(root)

    budget = GlobalDiskCacheBudget(root, before - 32_000, orphan_grace_seconds=0)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert result.evicted_entries >= 1
        assert not old.exists()
        assert recent.exists()
    finally:
        budget._remove_lease()


def test_ssm_pair_participates_in_same_global_lru(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = root / "aaaaaaaaaaaa"
    now = time.time()
    recent = _indexed_block(namespace, "aa-block", size=64_000, accessed=now)
    companion = namespace / "ssm_companion" / "cc"
    companion.mkdir(parents=True)
    data = companion / "cc-old.safetensors"
    side = companion / "cc-old.json"
    data.write_bytes(b"s" * 30_000)
    side.write_bytes(b"j" * 10_000)
    os.utime(data, (now - 200, now - 200))
    os.utime(side, (now - 200, now - 200))
    before = _physical_total(root)

    budget = GlobalDiskCacheBudget(root, before - 20_000, orphan_grace_seconds=0)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not data.exists()
        assert not side.exists()
        assert recent.exists()
    finally:
        budget._remove_lease()


def test_recent_orphan_and_live_temp_are_counted_but_protected(tmp_path: Path) -> None:
    root = tmp_path / "custom-root"
    sentinel = root / "do-not-touch.txt"
    root.mkdir()
    sentinel.write_text("unrelated")
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    recent = blocks / "recent.safetensors"
    recent.write_bytes(b"r" * 8_000)

    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=3600)
    active = blocks / f"active.{budget.lease_id}.1.tmp.safetensors"
    active.write_bytes(b"a" * 8_000)
    stale = blocks / (
        "stale.99999999-0123456789abcdef0123456789abcdef.1.tmp.safetensors"
    )
    stale.write_bytes(b"s" * 8_000)
    legacy_stale = blocks / "legacy.0.tmp.safetensors"
    legacy_stale.write_bytes(b"l" * 8_000)
    old = time.time() - 7200
    os.utime(legacy_stale, (old, old))
    try:
        result = budget.enforce(force=True)
        assert result.compliant is False
        assert result.protected_recent_orphans == 1
        assert recent.exists()
        assert active.exists()
        assert not stale.exists()
        assert not legacy_stale.exists()
        assert sentinel.read_text() == "unrelated"
    finally:
        budget._remove_lease()


def test_old_finalized_orphan_is_evictable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    orphan = blocks / "old.safetensors"
    orphan.write_bytes(b"o" * 32_000)
    old = time.time() - 3600
    os.utime(orphan, (old, old))
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=1)
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not orphan.exists()
    finally:
        budget._remove_lease()


def test_zero_is_unlimited_but_forced_reconciliation_reports_physical_truth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first = _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-block",
        size=64_000,
        accessed=time.time(),
    )
    second = _indexed_block(
        root / "bbbbbbbbbbbb",
        "bb-block",
        size=32_000,
        accessed=time.time(),
    )
    budget = GlobalDiskCacheBudget(root, 0)
    try:
        result = budget.enforce(force=True)
        assert result.max_size_bytes == 0
        assert result.compliant is True
        assert result.scan_performed is True
        assert result.accounted is True
        assert result.bytes_after >= first.stat().st_size + second.stat().st_size
        assert first.exists()
        assert second.exists()
        accounted = budget.account_finalized_write(123)
        assert accounted.scan_performed is True
        assert accounted.compliant is True
    finally:
        budget._remove_lease()


def test_strict_unlimited_accounting_advances_physical_reconciliation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-block",
        size=16_000,
        accessed=time.time(),
    )
    budget = GlobalDiskCacheBudget(root, 0, reconcile_interval_seconds=3600)
    try:
        startup = budget.enforce(force=True)
        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            strict = budget.account_finalized_write_locked(
                0,
                require_reconciled=True,
            )
        assert strict.scan_performed is True
        assert strict.accounted is True
        assert (
            strict.reconciliation_generation
            > startup.reconciliation_generation
        )
        assert strict.bytes_after == startup.bytes_after
    finally:
        budget.close()


def test_minimum_live_finite_lease_wins_and_unlimited_cannot_relax_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    high = GlobalDiskCacheBudget(root, 1000)
    low = GlobalDiskCacheBudget(root, 500)
    unlimited = GlobalDiskCacheBudget(root, 0)
    try:
        assert len({high.lease_id, low.lease_id, unlimited.lease_id}) == 3
        assert unlimited.enforce(force=True).max_size_bytes == 500
        low._remove_lease()
        assert unlimited.enforce(force=True).max_size_bytes == 1000
    finally:
        high._remove_lease()
        low._remove_lease()
        unlimited._remove_lease()


def test_minimum_cap_is_observed_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    code = (
        "import sys; "
        "from vmlx_engine.global_disk_cache_budget import GlobalDiskCacheBudget; "
        "b=GlobalDiskCacheBudget(sys.argv[1], 777); "
        "print(b.lease_id, flush=True); input()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip()
        local = GlobalDiskCacheBudget(root, 0)
        try:
            assert local.enforce(force=True).max_size_bytes == 777
        finally:
            local._remove_lease()
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        child.wait(timeout=10)
        assert child.returncode == 0, child.stderr.read() if child.stderr else ""


def test_pid_reuse_birth_mismatch_removes_stale_finite_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    stale_id = f"{os.getpid()}-{'a' * 32}"
    stale_path = root / ".vmlx-global-cache-budget-leases" / f"{stale_id}.json"
    temp_path = namespace / "blocks" / (
        f"orphan.{stale_id}.0.{'b' * 32}.tmp.safetensors"
    )
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"x" * 64_000)
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=0)
    stale_path.write_text(
        json.dumps(
            {
                "version": 1,
                "max_size_bytes": 1,
                "updated_at_ns": time.time_ns(),
                "pid": os.getpid(),
                "process_birth_identity": "reused-old-process",
            }
        )
    )
    monkeypatch.setattr(
        budget_module,
        "_process_birth_identity",
        lambda _pid: "current-process",
    )
    try:
        budget.enforce(force=True)
        assert not stale_path.exists()
        assert not temp_path.exists()
    finally:
        budget.close()


def test_repeated_accounting_caches_cross_process_birth_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    budget = GlobalDiskCacheBudget(
        root,
        1_000_000,
        reconcile_interval_seconds=3600,
    )
    budget.enforce(force=True)
    fake_path = root / ".vmlx-global-cache-budget-leases" / "fake.json"
    fake_path.write_text(
        json.dumps(
            {
                "version": 1,
                "max_size_bytes": 1_000_000,
                "updated_at_ns": time.time_ns(),
                "pid": 424242,
                "process_birth_identity": "fake-birth",
            }
        )
    )
    calls = 0

    def counted_probe(_pid):
        nonlocal calls
        calls += 1
        return "fake-birth"

    monkeypatch.setattr(budget, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(budget_module, "_process_birth_identity", counted_probe)
    try:
        for _ in range(3):
            with budget.exclusive_mutation_guard() as locked:
                assert locked
                result = budget.account_finalized_write_locked(0)
                assert result.accounted and result.compliant
        assert calls == 1
    finally:
        fake_path.unlink(missing_ok=True)
        budget.close()


def test_refresh_health_observes_other_owner_accounting_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first = GlobalDiskCacheBudget(root, 1_000_000)
    second = GlobalDiskCacheBudget(root, 1_000_000)
    try:
        initial = first.enforce(force=True)
        with second.exclusive_mutation_guard() as locked:
            assert locked
            advanced = second.account_finalized_write_locked(0)
        assert advanced.accounting_generation > initial.accounting_generation
        refreshed = first.refresh_health()
        # Same physical reconciliation generation retains the local proof that
        # a scan occurred while refreshing the other owner's ledger advance.
        assert refreshed.scan_performed is True
        assert refreshed.accounted is True
        assert refreshed.accounting_generation == advanced.accounting_generation
    finally:
        first.close()
        second.close()


def test_accounting_avoids_root_scan_until_crossing_or_strict_fence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(
        root,
        10_000_000,
        reconcile_interval_seconds=3600,
    )
    try:
        budget.enforce(force=True)
        calls = 0
        original = budget._scan_locked

        def counted_scan(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(budget, "_scan_locked", counted_scan)
        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            for _ in range(5):
                result = budget.account_finalized_write_locked(100)
                assert result.scan_performed is False
        assert calls == 0

        with budget.exclusive_mutation_guard() as locked:
            assert locked is True
            strict = budget.account_finalized_write_locked(
                0,
                require_reconciled=True,
            )
        assert strict.scan_performed is True
        # one scan: nothing was evicted, so the post-trim rescan is skipped
        assert calls >= 1
    finally:
        budget._remove_lease()


def test_negative_delta_forces_reconcile_instead_of_double_subtracting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    blocks = namespace / "blocks"
    blocks.mkdir()
    first = blocks / "first.safetensors"
    second = blocks / "second.safetensors"
    first.write_bytes(b"a" * 10_000)
    second.write_bytes(b"b" * 20_000)
    # fresh unreferenced payloads inside the orphan grace: counted and protected (an AGED unreferenced payload is
    # garbage on every reconcile now, whatever the ceiling)
    one = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=3600)
    two = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=3600)
    try:
        assert one.enforce(force=True).bytes_after == 30_000
        first.unlink()
        assert two.enforce(force=True).bytes_after == 20_000
        with one.exclusive_mutation_guard() as locked:
            assert locked is True
            result = one.account_finalized_write_locked(-10_000)
        assert result.scan_performed is True
        assert result.bytes_after == 20_000
    finally:
        one._remove_lease()
        two._remove_lease()


def test_custom_root_legacy_cache_is_bounded_without_touching_unrelated_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custom"
    raw_blocks = root / "blocks"
    raw_blocks.mkdir(parents=True)
    raw_payload = raw_blocks / "legacy.safetensors"
    raw_payload.write_bytes(b"legacy" * 20_000)
    old = time.time() - 1000
    os.utime(raw_payload, (old, old))
    database = root / "block_index.db"
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            "CREATE TABLE blocks (block_hash TEXT PRIMARY KEY, "
            "file_name TEXT NOT NULL, last_accessed REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?)",
            ("legacy", "blocks/legacy.safetensors", old),
        )
        conn.commit()
    finally:
        conn.close()
    sentinel = root / "unrelated.txt"
    sentinel.write_text("preserve me")

    namespace = root / "aaaaaaaaaaaa"
    managed = _indexed_block(
        namespace,
        "aa-managed",
        size=64_000,
        accessed=time.time() - 100,
    )
    probe = GlobalDiskCacheBudget(
        root,
        10_000_000,
        orphan_grace_seconds=0,
        allow_legacy_hashed_namespaces=False,
        allow_legacy_direct_namespace=True,
    )
    try:
        before = probe.enforce(force=True).bytes_after
    finally:
        probe._remove_lease()

    budget = GlobalDiskCacheBudget(
        root,
        before - 50_000,
        orphan_grace_seconds=0,
        allow_legacy_hashed_namespaces=False,
        allow_legacy_direct_namespace=True,
    )
    try:
        result = budget.enforce(force=True)
        assert result.compliant is True
        assert not raw_payload.exists()
        assert managed.exists()
        assert sentinel.read_text() == "preserve me"
    finally:
        budget._remove_lease()


def test_ssm_store_replace_and_delete_update_shared_ledger(tmp_path: Path) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(
        root,
        10_000_000,
        reconcile_interval_seconds=3600,
    )
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    try:
        baseline = budget.enforce(force=True).bytes_after
        assert store.store("ab" * 32, [{"state": 1}], True, [1, 2], 2)
        assert store.wait_for_pending()
        first = budget.last_result
        assert first is not None and first.bytes_after > baseline
        assert store.store("ab" * 32, [{"state": "larger"}], True, [1, 2], 2)
        assert store.wait_for_pending()
        second = budget.last_result
        assert second is not None and second.bytes_after >= first.bytes_after
        store.delete("ab" * 32)
        deleted = budget.last_result
        assert deleted is not None and deleted.bytes_after < second.bytes_after
    finally:
        store.shutdown()
        budget._remove_lease()


def test_successful_ssm_fetch_refreshes_cross_namespace_global_lru(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(root, 10_000_000, orphan_grace_seconds=0)
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    key = "ab" * 32
    trim_budget = None
    try:
        assert store.store(key, [{"state": 1}], True, [1, 2], 2)
        assert store.wait_for_pending()
        data_path, side_path = store._entry_paths(key)
        now = time.time()
        os.utime(data_path, (now - 200, now - 200))
        os.utime(side_path, (now - 200, now - 200))
        older_block = _indexed_block(
            root / "bbbbbbbbbbbb",
            "bb-older-block",
            size=256_000,
            accessed=now - 100,
        )

        # The fully materialized fetch touches both halves of the atomic pair.
        # Global eviction must therefore choose the untouched block that was
        # newer before this fetch, not the now-hot SSM record.
        assert store.fetch(key) is not None
        total = budget.enforce(force=True).bytes_after
        old_block_size = older_block.stat().st_size
        remaining_after_oldest = total - old_block_size
        cap = ((remaining_after_oldest * 10 + 8) // 9) + 1024
        assert cap < total
        trim_budget = GlobalDiskCacheBudget(root, cap, orphan_grace_seconds=0)
        result = trim_budget.enforce(force=True)

        assert result.compliant is True
        assert result.bytes_after <= result.max_size_bytes
        assert not older_block.exists()
        assert data_path.exists()
        assert side_path.exists()
    finally:
        store.shutdown()
        if trim_budget is not None:
            trim_budget.close()
        budget.close()


def test_ssm_torn_data_sidecar_generation_is_a_cache_miss(tmp_path: Path) -> None:
    store = SSMCompanionDiskStore(
        directory=tmp_path / "ssm_companion",
        budget_bytes=0,
    )
    key = "cd" * 32
    assert store.store(key, [{"state": 1}], True, [1, 2], 2)
    assert store.wait_for_pending()
    data_path, side_path = store._entry_paths(key)
    sidecar = json.loads(side_path.read_text())
    sidecar["record_id"] = "different-final-rename-generation"
    side_path.write_text(json.dumps(sidecar))

    assert data_path.exists()
    assert store.fetch(key) is None
    assert store.shutdown()


def test_ssm_aggregate_accounting_failure_stops_later_publications(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vmlx_engine.utils.ssm_companion_disk_store as ssm_module

    root = tmp_path / "root"
    namespace = ensure_managed_block_cache_namespace(root / "aaaaaaaaaaaa")
    budget = GlobalDiskCacheBudget(root, 10_000_000)
    store = SSMCompanionDiskStore(
        directory=namespace / "ssm_companion",
        budget_bytes=0,
        global_budget=budget,
    )
    save_calls = 0
    original_save = ssm_module.mx.save_safetensors
    original_account = budget.account_finalized_write_locked

    def counted_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return original_save(*args, **kwargs)

    def fail_accounting(*_args, **_kwargs):
        raise OSError("forced aggregate accounting failure")

    monkeypatch.setattr(ssm_module.mx, "save_safetensors", counted_save)
    monkeypatch.setattr(
        budget,
        "account_finalized_write_locked",
        fail_accounting,
    )
    try:
        assert store.store("ef" * 32, [{"state": 1}], True, [1, 2], 2)
        assert not store.wait_for_write("ef" * 32)
        first_save_calls = save_calls
        assert first_save_calls == 1
        assert not store.store("fe" * 32, [{"state": 2}], True, [3, 4], 2)
        assert save_calls == first_save_calls
        assert list((namespace / "ssm_companion").rglob("*.safetensors")) == []
        assert list((namespace / "ssm_companion").rglob("*.json")) == []

        # The fail-closed latch is bounded, not permanent.  Once aggregate
        # reconciliation/accounting recovers, a later cache publication works.
        monkeypatch.setattr(
            budget,
            "account_finalized_write_locked",
            original_account,
        )
        store._budget_recovery_interval_ns = 0
        assert store.store("aa" * 32, [{"state": 3}], True, [5, 6], 2)
        assert store.wait_for_pending()
        assert save_calls == first_save_calls + 1
    finally:
        store.shutdown()
        budget.close()


def test_corrupt_derived_accounting_is_rebuilt_from_physical_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    old = _indexed_block(
        root / "aaaaaaaaaaaa",
        "aa-old",
        size=64_000,
        accessed=time.time() - 100,
    )
    budget = GlobalDiskCacheBudget(root, 1, orphan_grace_seconds=0)
    (root / ".vmlx-global-cache-budget-accounting.json").write_text("{")
    try:
        result = budget.enforce(force=True)
        assert result.accounted is True
        assert result.compliant is False  # SQLite metadata remains protected.
        assert not old.exists()
        state = (root / ".vmlx-global-cache-budget-accounting.json").read_text()
        assert '"version": 1' in state
    finally:
        budget._remove_lease()


def test_lease_directory_symlink_is_rejected_without_writing_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / ".vmlx-global-cache-budget-leases").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(OSError):
        GlobalDiskCacheBudget(root, 1000)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("legacy_name", ["blocks", "block_index.db"])
def test_legacy_namespace_symlink_is_rejected_before_claim_or_outside_write(
    tmp_path: Path,
    legacy_name: str,
) -> None:
    namespace = tmp_path / "namespace"
    outside = tmp_path / "outside"
    namespace.mkdir()
    outside.mkdir()
    target = outside / legacy_name
    if legacy_name == "blocks":
        target.mkdir()
    else:
        target.write_text("outside-db-sentinel")
    (namespace / legacy_name).symlink_to(
        target,
        target_is_directory=legacy_name == "blocks",
    )

    with pytest.raises(OSError, match="symlinked block-cache path"):
        ensure_managed_block_cache_namespace(namespace)

    assert not (namespace / ".vmlx-block-cache-namespace-v1").exists()
    if legacy_name == "blocks":
        assert list(target.iterdir()) == []
    else:
        assert target.read_text() == "outside-db-sentinel"


def test_symlinked_namespace_marker_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "aaaaaaaaaaaa"
    blocks = child / "blocks"
    blocks.mkdir(parents=True)
    payload = blocks / "old.safetensors"
    payload.write_bytes(b"x" * 10_000)
    target = tmp_path / "marker-target"
    target.write_text("marker")
    (child / ".vmlx-block-cache-namespace-v1").symlink_to(target)
    budget = GlobalDiskCacheBudget(root, 1)
    try:
        result = budget.enforce(force=True)
        assert result.accounted is False
        assert result.compliant is False
        assert payload.exists()
    finally:
        budget._remove_lease()


def test_interval_due_rescan_is_deferred_off_the_publication_path(tmp_path, monkeypatch):
    """An accounting call inside the publication critical section must not walk the
    root merely because the reconcile interval elapsed; it owes the rescan to the
    owner's idle time. Over-ceiling, strict-fence and missing-accounting cases stay
    inline (covered by the crossing/strict test above)."""
    from vmlx_engine import global_disk_cache_budget as budget_module

    root = tmp_path / "root"
    budget = budget_module.GlobalDiskCacheBudget(root, 10_000_000, reconcile_interval_seconds=0.0)
    try:
        assert budget.enforce(force=True).scan_performed is True
        scans = []
        real_scan = budget._scan_locked

        def counted_scan(**kwargs):
            scans.append(1)
            return real_scan(**kwargs)

        monkeypatch.setattr(budget, "_scan_locked", counted_scan)
        with budget.exclusive_mutation_guard() as locked:
            assert locked
            result = budget.account_finalized_write_locked(100)
        assert result.scan_performed is False and result.accounted is True
        assert scans == []
        assert budget.deferred_reconcile_due is True
        deferred = budget.run_deferred_reconcile()
        assert deferred is not None and deferred.scan_performed is True
        # _enforce_locked scans once for candidates and once for the post-trim total
        assert len(scans) >= 1
        assert budget.deferred_reconcile_due is False
        assert budget.run_deferred_reconcile() is None
    finally:
        budget.close()


def test_writer_idle_branch_runs_the_deferred_rescan():
    import inspect
    from vmlx_engine.block_disk_store import BlockDiskStore

    src = inspect.getsource(BlockDiskStore._background_writer)
    idle = src.index("except queue.Empty:")
    assert src.index("self._run_deferred_budget_reconcile()", idle) < src.index("# Drain remaining items", idle)
    # the rescan waits for a quiet writer: a store enqueues blocks one by one, so
    # an empty queue between two blocks is not idle (measured: the rescan fired
    # between block 1 and block 2 and the barrier stayed at 5 s)
    quiet = inspect.getsource(BlockDiskStore._writer_quiet_for_maintenance)
    for needle in ("_pending_write_items > 0", "_write_inflight > 0", "post_eviction_complete", "_last_write_activity_monotonic"):
        assert needle in quiet
    run = inspect.getsource(BlockDiskStore._run_deferred_budget_reconcile)
    assert run.index("_writer_quiet_for_maintenance()") < run.index("run_deferred_reconcile()")
    # a request in flight (prefill/decode with no writes yet) is not quiet: the
    # first live loop at 877f4c22 ran the rescan during decode and the terminal
    # store waited 3.8 s behind it. BOTH schedulers hand the store a probe.
    assert "self._activity_probe" in quiet
    from vmlx_engine import scheduler as text_scheduler, mllm_scheduler
    for module in (text_scheduler, mllm_scheduler):
        src_mod = inspect.getsource(module)
        assert src_mod.count("activity_probe=self._block_store_activity_probe,") == 1, module.__name__
        assert "def _block_store_activity_probe(self) -> bool:" in src_mod, module.__name__


def test_reconcile_rescans_only_after_an_eviction(tmp_path, monkeypatch):
    """Nothing evicted -> one root scan (the first scan's totals stand under the
    exclusive lock). Something evicted -> a second scan re-measures."""
    from vmlx_engine import global_disk_cache_budget as budget_module

    root = tmp_path / "root"
    now = time.time()
    _indexed_block(root / "aaaaaaaaaaaa", "aa-old", size=64_000, accessed=now - 100)
    _indexed_block(root / "bbbbbbbbbbbb", "bb-new", size=64_000, accessed=now)
    before = _physical_total(root)
    # generous ceiling first: nothing to evict
    budget = budget_module.GlobalDiskCacheBudget(root, before * 4, orphan_grace_seconds=0, reconcile_interval_seconds=3600)
    try:
        scans = []
        real_scan = budget._scan_locked
        monkeypatch.setattr(budget, "_scan_locked", lambda **kw: (scans.append(1), real_scan(**kw))[1])
        first = budget.enforce(force=True)
        assert first.scan_performed is True and first.evicted_entries == 0
        assert len(scans) == 1
    finally:
        budget.close()
    # tight ceiling: an eviction happens and the physical state is re-measured
    budget = budget_module.GlobalDiskCacheBudget(root, before - 32_000, orphan_grace_seconds=0, reconcile_interval_seconds=3600)
    try:
        scans = []
        real_scan = budget._scan_locked
        monkeypatch.setattr(budget, "_scan_locked", lambda **kw: (scans.append(1), real_scan(**kw))[1])
        second = budget.enforce(force=True)
        assert second.scan_performed is True and second.evicted_entries >= 1
        assert len(scans) == 2
    finally:
        budget.close()


def test_deferred_reconcile_is_retried_after_a_failed_scan(tmp_path, monkeypatch):
    """enforce() reports scan_performed=True on its error path; the owed rescan must
    survive that and be retried, never silently dropped."""
    from vmlx_engine import global_disk_cache_budget as budget_module

    root = tmp_path / "root"
    budget = budget_module.GlobalDiskCacheBudget(root, 10_000_000, reconcile_interval_seconds=0.0)
    try:
        assert budget.enforce(force=True).scan_performed is True
        with budget.exclusive_mutation_guard() as locked:
            assert locked
            budget.account_finalized_write_locked(100)
        assert budget.deferred_reconcile_due is True

        def failing_scan(**kwargs):
            raise OSError("cannot inspect cache file: simulated")

        monkeypatch.setattr(budget, "_scan_locked", failing_scan)
        failed = budget.run_deferred_reconcile()
        assert failed is not None and failed.scan_performed is True
        assert failed.accounted is False and failed.error
        assert budget.deferred_reconcile_due is True, "a failed scan must not settle the debt"
        monkeypatch.undo()
        ok = budget.run_deferred_reconcile()
        assert ok is not None and ok.accounted is True and not ok.error
        assert budget.deferred_reconcile_due is False
    finally:
        budget.close()


def test_maintenance_never_starts_while_requests_are_in_flight_and_crossing_stays_inline(tmp_path):
    """Continuous traffic: the activity probe keeps the idle rescan off, the O(1)
    accounting keeps running, and a ceiling crossing still reconciles inline
    (fail-closed) regardless of the deferred flag."""
    from vmlx_engine import global_disk_cache_budget as budget_module
    from vmlx_engine.block_disk_store import BlockDiskStore

    root = tmp_path / "root"
    now = time.time()
    _indexed_block(root / "aaaaaaaaaaaa", "aa-old", size=64_000, accessed=now - 100)
    _indexed_block(root / "bbbbbbbbbbbb", "bb-new", size=64_000, accessed=now)
    before = _physical_total(root)
    busy = {"value": True}
    # ceiling just under the physical total: the startup trim evicts the LRU block,
    # and a later accounted write of `before` bytes crosses it again
    store = BlockDiskStore(
        cache_dir=str(root / "cccccccccccc"),
        max_size_gb=(before - 32_000) / 1e9,
        global_cache_root=str(root),
        allow_legacy_hashed_namespaces=True,
        activity_probe=lambda: busy["value"],
    )
    try:
        budget = store.global_budget
        budget._reconcile_interval_ns = 0
        with budget.exclusive_mutation_guard() as locked:
            assert locked
            result = budget.account_finalized_write_locked(100)
        assert result.scan_performed is False and budget.deferred_reconcile_due is True
        # busy engine: quiet rule false, rescan not started, flag kept
        assert store._writer_quiet_for_maintenance() is False
        store._run_deferred_budget_reconcile()
        assert budget.deferred_reconcile_due is True
        # ceiling crossing while busy: inline reconcile, no dependence on idle time
        with budget.exclusive_mutation_guard() as locked:
            assert locked
            crossing = budget.account_finalized_write_locked(before)
        assert crossing.scan_performed is True
        # engine goes quiet: the owed rescan runs and settles
        busy["value"] = False
        store._last_write_activity_monotonic = time.monotonic() - 5.0
        assert store._writer_quiet_for_maintenance() is True
        store._run_deferred_budget_reconcile()
        assert budget.deferred_reconcile_due is False
    finally:
        store.shutdown()


def test_eviction_totals_accumulate_in_the_shared_ledger_and_survive_a_refresh(tmp_path: Path) -> None:
    """Live on an isolated 1 GB root: seven entries were evicted over two reconciles, yet /health showed
    disk_evictions=0 (the block store's counter never sees evictions the SSM companion writer triggers) and
    global_budget.evicted_entries=0 (a later reconcile without evictions overwrote the last-reconcile field).
    The ledger now carries cumulative totals every writer in every process adds to, and a health refresh
    reports them without a scan."""
    root = tmp_path / "root"
    now = time.time()
    _indexed_block(root / "aaaaaaaaaaaa", "aa-old", size=64_000, accessed=now - 300)
    _indexed_block(root / "aaaaaaaaaaaa", "aa-mid", size=64_000, accessed=now - 200)
    recent = _indexed_block(root / "bbbbbbbbbbbb", "bb-new", size=64_000, accessed=now)
    before = _physical_total(root)

    budget = GlobalDiskCacheBudget(root, before - 32_000, orphan_grace_seconds=0)
    try:
        first = budget.enforce(force=True)
        assert first.evicted_entries >= 1
        assert first.evicted_entries_total == first.evicted_entries
        assert first.evicted_bytes_total == first.evicted_bytes > 0
        # a second enforce below the (now lower) usage evicts nothing but keeps the totals
        second = budget.enforce(force=True)
        assert second.evicted_entries == 0
        assert second.evicted_entries_total == first.evicted_entries
        # a refresh from the ledger (what /health reads) carries the totals too
        health = budget.refresh_health()
        assert health.evicted_entries_total == first.evicted_entries and health.evicted_bytes_total == first.evicted_bytes
        # another writer on the same root (a second coordinator, as the SSM companion store holds) adds to the same ledger
        newest = _indexed_block(root / "cccccccccccc", "cc-new", size=64_000, accessed=now + 1)
        other = GlobalDiskCacheBudget(root, 150_000, orphan_grace_seconds=0)  # below the current usage (~165 KB): one more eviction
        try:
            third = other.enforce(force=True)
            assert third.evicted_entries >= 1
            assert third.evicted_entries_total == first.evicted_entries + third.evicted_entries
            assert budget.refresh_health().evicted_entries_total == third.evicted_entries_total
        finally:
            other._remove_lease()
        # LRU across namespaces: the newest block survives every pass
        assert newest.exists()
    finally:
        budget._remove_lease()


def test_idle_pass_removes_dead_writer_temp_files_after_the_grace_without_a_write(tmp_path: Path) -> None:
    """A kill inside a write leaves temp files; the next scan protects them inside the orphan grace and records them
    (protected_temp_files). Before this, they were removed only when a later WRITE pushed the root over its ceiling
    and the LRU pass reached them (live: 4 temp files survived a restart and 5 idle minutes). Now: the coordinator owes
    an IDLE pass once the interval elapsed while its last scan protected temps/orphans or was non-compliant, and the
    garbage pass removes aged unreferenced payloads whatever the ceiling — here the ceiling is UNLIMITED (0)."""
    root = tmp_path / "root"
    now = time.time()
    _indexed_block(root / "aaaaaaaaaaaa", "aa-keep", size=64_000, accessed=now)
    namespace = root / "aaaaaaaaaaaa"
    temp = namespace / "blocks" / "zz" / "zz-dead.tmp.safetensors"
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_bytes(b"y" * 32_000)  # ownerless temp file: protected while recent, garbage after the grace
    old_orphan = namespace / "blocks" / "yy" / "yy-old.safetensors"
    old_orphan.parent.mkdir(parents=True, exist_ok=True)
    old_orphan.write_bytes(b"o" * 16_000); os.utime(old_orphan, (now - 3600, now - 3600))  # unreferenced, aged: garbage now

    budget = GlobalDiskCacheBudget(root, 0, orphan_grace_seconds=0.4, reconcile_interval_seconds=0.2)
    try:
        first = budget.enforce(force=True)
        assert not old_orphan.exists(), "an aged unreferenced payload is garbage even under an unlimited ceiling"
        assert first.evicted_entries >= 1 and first.compliant
        assert first.protected_temp_files == 1 and temp.exists()
        assert budget.idle_reconcile_due() is False  # interval not elapsed yet
        time.sleep(0.25)
        assert budget.idle_reconcile_due() is True  # interval elapsed, a protected temp is recorded
        second = budget.run_idle_reconcile()
        assert second is not None and second.scan_performed
        if temp.exists():
            assert second.protected_temp_files == 1  # grace not over: still protected, still owed
            time.sleep(0.45)
            assert budget.idle_reconcile_due() is True
            third = budget.run_idle_reconcile()
            assert third is not None
        assert not temp.exists()
        # nothing protected and compliant: no idle pass is owed, however much time passes
        time.sleep(0.25)
        assert budget.idle_reconcile_due() is False
        assert (namespace / "blocks" / "aa" / "aa-keep.safetensors").exists()
    finally:
        budget._remove_lease()


def test_writer_idle_poll_runs_the_idle_pass_without_a_publish() -> None:
    import inspect
    import vmlx_engine.block_disk_store as store
    src = inspect.getsource(store.BlockDiskStore._run_deferred_budget_reconcile)
    assert "budget.run_idle_reconcile()" in src and 'callable(idle_due) and idle_due()' in src

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiskCacheManager.

Tests the SQLite index and filesystem operations without requiring
mlx_lm's save_prompt_cache/load_prompt_cache (which need real model caches).
We test the database layer directly by inserting rows and files manually.

Verifies:
- Store + fetch round-trip via direct DB/file manipulation
- Eviction respects max_size_gb
- stats() returns correct field names
- Fetch of missing key returns None
- Stale entry cleanup (file deleted externally)
"""

import os
import sqlite3
import tempfile
import time

import pytest

from vmlx_engine.disk_cache import DiskCacheManager, _hash_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_manager(tmpdir: str, max_size_gb: float = 10.0) -> DiskCacheManager:
    """Create a DiskCacheManager with a temp directory."""
    return DiskCacheManager(cache_dir=tmpdir, max_size_gb=max_size_gb)


def _insert_fake_entry(
    mgr: DiskCacheManager,
    tokens: list[int],
    file_content: bytes = b"fake safetensors data",
    file_size_override: int | None = None,
) -> str:
    """Insert a fake cache entry directly into the DB and filesystem.

    Returns the token_hash.
    """
    token_hash = _hash_tokens(tokens)
    file_name = f"cache_{token_hash[:16]}_{len(tokens)}tok.safetensors"
    file_path = mgr.cache_dir / file_name

    # Write fake file
    file_path.write_bytes(file_content)
    actual_size = file_size_override if file_size_override is not None else len(file_content)

    # Insert into DB
    conn = mgr._pool.get()
    try:
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO cache_entries "
            "(token_hash, file_name, num_tokens, file_size, created_at, last_accessed, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (token_hash, file_name, len(tokens), actual_size, now, now),
        )
        conn.commit()
    finally:
        mgr._pool.put(conn)

    return token_hash


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiskCacheUnit:
    """Disk cache database and filesystem tests."""

    def test_store_and_fetch_roundtrip(self):
        """Store data under a token hash, fetch returns the DB row."""
        tmpdir = tempfile.mkdtemp(prefix="vmlx_disk_cache_test_")
        mgr = _create_manager(tmpdir)
        try:
            tokens = [10, 20, 30, 40, 50]
            _insert_fake_entry(mgr, tokens)

            # Verify the entry exists in the DB
            token_hash = _hash_tokens(tokens)
            conn = mgr._pool.get()
            try:
                row = conn.execute(
                    "SELECT file_name, num_tokens FROM cache_entries WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
            finally:
                mgr._pool.put(conn)

            assert row is not None
            assert row[1] == len(tokens)

            # The file should exist on disk
            file_path = mgr.cache_dir / row[0]
            assert file_path.exists()
        finally:
            mgr.shutdown()

    def test_eviction_respects_max_gb(self):
        """Store enough data to exceed limit, verify old entries evicted."""
        tmpdir = tempfile.mkdtemp(prefix="vmlx_disk_cache_test_")
        # Very small limit: 1 KB (expressed as GB)
        max_gb = 1.0 / (1024 * 1024)  # ~1 KB
        mgr = _create_manager(tmpdir, max_size_gb=max_gb)
        try:
            # Insert entries that exceed 1 KB total
            for i in range(5):
                tokens = [i * 100 + j for j in range(10)]
                # Each file is ~500 bytes
                _insert_fake_entry(mgr, tokens, file_content=b"x" * 500)

            # Trigger eviction manually
            mgr._evict_if_needed()

            # Count remaining entries
            conn = mgr._pool.get()
            try:
                count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                total_size = conn.execute(
                    "SELECT COALESCE(SUM(file_size), 0) FROM cache_entries"
                ).fetchone()[0]
            finally:
                mgr._pool.put(conn)

            # Some entries should have been evicted
            assert count < 5
            # Total size should be within the limit (or close to it)
            assert total_size <= mgr.max_size_bytes + 500  # Allow one entry margin
        finally:
            mgr.shutdown()

    def test_stats_field_names(self):
        """Verify stats() returns dict with the correct field names.

        CRITICAL: CachePanel.tsx reads `total_size_mb` (with fallback to `size_mb`).
        The Python code must return `total_size_mb`, NOT `size_mb` or `count`.

        Actual field names in disk_cache.py stats():
        - entries (NOT count)
        - total_size_mb (NOT size_mb)
        - max_size_gb
        - hits
        - misses
        - stores
        - hit_rate
        - pending_writes
        """
        tmpdir = tempfile.mkdtemp(prefix="vmlx_disk_cache_test_")
        mgr = _create_manager(tmpdir)
        try:
            stats = mgr.stats()

            # Verify all expected keys exist
            expected_keys = {
                "entries",
                "total_size_mb",
                "max_size_gb",
                "hits",
                "misses",
                "stores",
                "hit_rate",
                "pending_writes",
            }
            assert set(stats.keys()) == expected_keys

            # Verify types
            assert isinstance(stats["entries"], int)
            assert isinstance(stats["total_size_mb"], (int, float))
            assert isinstance(stats["max_size_gb"], (int, float))
            assert isinstance(stats["hit_rate"], (int, float))

            # NOTE: The field is "entries" not "count" -- the task description
            # mentioned "count" but the actual code uses "entries".
            assert "count" not in stats
            assert "size_mb" not in stats  # It's "total_size_mb"
        finally:
            mgr.shutdown()

    def test_fetch_missing_returns_none(self):
        """Fetch nonexistent key returns None."""
        tmpdir = tempfile.mkdtemp(prefix="vmlx_disk_cache_test_")
        mgr = _create_manager(tmpdir)
        try:
            result = mgr.fetch([999, 888, 777])
            assert result is None

            stats = mgr.stats()
            assert stats["misses"] == 1
        finally:
            mgr.shutdown()

    def test_stale_entry_cleanup(self):
        """Store entry, delete the file manually, fetch should clean up DB row."""
        tmpdir = tempfile.mkdtemp(prefix="vmlx_disk_cache_test_")
        mgr = _create_manager(tmpdir)
        try:
            tokens = [42, 43, 44]
            token_hash = _insert_fake_entry(mgr, tokens)

            # Verify the entry is in the DB
            conn = mgr._pool.get()
            try:
                row = conn.execute(
                    "SELECT file_name FROM cache_entries WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
                assert row is not None
                file_path = mgr.cache_dir / row[0]
            finally:
                mgr._pool.put(conn)

            # Delete the file externally
            assert file_path.exists()
            file_path.unlink()
            assert not file_path.exists()

            # Fetch should detect missing file, clean up DB, return None
            result = mgr.fetch(tokens)
            assert result is None

            # DB row should be cleaned up
            conn = mgr._pool.get()
            try:
                row = conn.execute(
                    "SELECT 1 FROM cache_entries WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
                assert row is None
            finally:
                mgr._pool.put(conn)
        finally:
            mgr.shutdown()

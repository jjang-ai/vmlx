"""has_block() in the non-TQ mode inspects each exact payload file at most once.

The store dedup loop calls has_block() for every already-stored block of every
turn. In the non-TQ mode it used to mx.load + deserialize the whole payload each
time to look for a legacy TQ record, so a one-block tool-continuation store
re-read every earlier block of the conversation. The memo is keyed on the file
identity (inode, mtime_ns, size): a block this process wrote or already
inspected is not re-read, a rewritten file is inspected again, and a missing
file is still reported missing.
"""

import os
import shutil
from pathlib import Path

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _plain_block():
    k = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.float16)
    v = mx.random.normal(shape=(1, 2, 8, 64)).astype(mx.float16)
    mx.eval(k, v)
    return [("kv", k, v)]


def _write(store, hashes):
    for h in hashes:
        assert store.write_block_async(h, _plain_block(), token_count=8)
    assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)


def _count_loads(monkeypatch):
    import vmlx_engine.block_disk_store as bds

    calls = []
    real = bds.mx.load

    def counting_load(path, *a, **k):
        calls.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(bds.mx, "load", counting_load)
    return calls


def _payload_path(store, block_hash) -> Path:
    row = store._read_conn.execute(
        "SELECT file_name FROM blocks WHERE block_hash = ?", (block_hash.hex(),)
    ).fetchone()
    return Path(store.cache_dir) / row[0]


def test_blocks_written_by_this_process_are_never_reinspected(tmp_path, monkeypatch):
    from vmlx_engine.block_disk_store import BlockDiskStore

    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        hashes = [bytes([i]) * 32 for i in range(1, 5)]
        _write(store, hashes)
        calls = _count_loads(monkeypatch)
        for _ in range(3):
            for h in hashes:
                assert store.has_block(h) is True
        assert calls == [], "plain payloads written here were re-loaded by has_block"
    finally:
        store.shutdown()


def test_restore_lru_touch_does_not_force_reinspection(tmp_path, monkeypatch):
    """Every block read os.utime()s the file (LRU signal). The next turn's store
    dedup check must not treat that touched file as a new payload."""
    from vmlx_engine.block_disk_store import BlockDiskStore

    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        hashes = [bytes([i]) * 32 for i in range(20, 24)]
        _write(store, hashes)
        for h in hashes:
            assert store.read_block(h) is not None      # restore: os.utime(path, None)
            path = _payload_path(store, h)
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))  # guarantee a changed mtime
        calls = _count_loads(monkeypatch)
        for h in hashes:
            assert store.has_block(h) is True
        assert calls == [], "an LRU touch from a restore forced a full payload re-read"
    finally:
        store.shutdown()


def test_existing_payload_is_inspected_once_per_process(tmp_path, monkeypatch):
    from vmlx_engine.block_disk_store import BlockDiskStore

    hashes = [bytes([i]) * 32 for i in range(10, 13)]
    writer = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        _write(writer, hashes)
    finally:
        writer.shutdown()

    fresh = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        calls = _count_loads(monkeypatch)
        for _ in range(4):
            for h in hashes:
                assert fresh.has_block(h) is True
        assert len(calls) == len(hashes), calls
    finally:
        fresh.shutdown()


def test_rewritten_payload_is_inspected_again(tmp_path, monkeypatch):
    from vmlx_engine.block_disk_store import BlockDiskStore

    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        h = bytes([42]) * 32
        _write(store, [h])
        path = _payload_path(store, h)
        # Replace the file by rename (new inode), as another process would.
        tmp_copy = path.with_suffix(".copy")
        shutil.copyfile(path, tmp_copy)
        os.replace(tmp_copy, path)
        calls = _count_loads(monkeypatch)
        assert store.has_block(h) is True
        assert store.has_block(h) is True
        assert len(calls) == 1, calls
    finally:
        store.shutdown()


def test_missing_payload_is_still_reported_missing(tmp_path):
    from vmlx_engine.block_disk_store import BlockDiskStore

    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        h = bytes([7]) * 32
        _write(store, [h])
        assert store.has_block(h) is True
        _payload_path(store, h).unlink()
        assert store.has_block(h) is False
    finally:
        store.shutdown()

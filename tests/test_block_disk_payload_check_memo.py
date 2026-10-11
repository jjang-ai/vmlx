"""Read-time payload checks run once per exact file.

A tool loop re-restores the same prefix blocks every turn. The safetensors header validation and the mixed-SWA
rotating-window probe depend only on the immutable payload, so they are memoized per file identity (device,
inode, size, birth). A rewritten file (new inode) is checked again; the post-load validator runs on every read.
"""

import os
import shutil

import pytest

mx = pytest.importorskip("mlx.core")

import vmlx_engine.block_disk_store as bds
import vmlx_engine.cache_record_validator as crv
from vmlx_engine.block_disk_store import BlockDiskStore


def _plain():
    k = mx.zeros((1, 2, 8, 16), dtype=mx.float16)
    v = mx.ones((1, 2, 8, 16), dtype=mx.float16)
    mx.eval(k, v)
    return [("kv", k, v)]


def _rotating(offset):
    k = mx.full((1, 2, 8, 16), 2.0, dtype=mx.float16)
    v = mx.full((1, 2, 8, 16), 3.0, dtype=mx.float16)
    mx.eval(k, v)
    return [("kv", k, v), ("rotating_kv", k, v, 8, 0, offset, 0)]


def _counters(monkeypatch):
    calls = {"header": 0, "probe": 0}
    real_header = crv.reject_safetensors_or_warn
    real_probe = bds._load_reconstruction_payload_without_stale_rotating

    def header(*a, **k):
        calls["header"] += 1
        return real_header(*a, **k)

    def probe(*a, **k):
        calls["probe"] += 1
        return real_probe(*a, **k)

    monkeypatch.setattr(crv, "reject_safetensors_or_warn", header)
    monkeypatch.setattr(bds, "_load_reconstruction_payload_without_stale_rotating", probe)
    return calls


def _path(store, h):
    row = store._read_conn.execute("SELECT file_name FROM blocks WHERE block_hash = ?", (h.hex(),)).fetchone()
    return store.cache_dir / row[0]


def test_header_validated_once_per_file_and_again_after_rewrite(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        h = bytes([5]) * 32
        assert store.write_block_async(h, _plain(), token_count=8)
        assert store.wait_for_blocks([h], timeout=5.0) == {h}
        calls = _counters(monkeypatch)
        for _ in range(3):
            assert store.read_block(h) is not None
        assert calls["header"] == 1
        path = _path(store, h)
        copy = path.with_suffix(".copy")
        shutil.copyfile(path, copy)
        os.replace(copy, path)  # new inode: must be validated again
        assert store.read_block(h) is not None
        assert calls["header"] == 2
    finally:
        store.shutdown()


def test_rotating_probe_skipped_when_known_offsets_match_target(tmp_path, monkeypatch):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        h = bytes([6]) * 32
        assert store.write_block_async(h, _rotating(offset=64), token_count=8)
        assert store.wait_for_blocks([h], timeout=5.0) == {h}
        calls = _counters(monkeypatch)
        first = store.read_block_for_reconstruction(h, rotating_target_offset=64)
        assert first is not None and first[1][0] == "rotating_kv"
        assert calls["probe"] == 1
        again = store.read_block_for_reconstruction(h, rotating_target_offset=64)
        assert again is not None and again[1][0] == "rotating_kv"
        assert calls["probe"] == 1, "known matching offsets must not re-open the file"
        # A different target makes the stored window stale: probe again and omit it.
        stale = store.read_block_for_reconstruction(h, rotating_target_offset=128)
        assert stale is not None and stale[1][0] == "rotating_kv_pending"
        assert calls["probe"] == 2
    finally:
        store.shutdown()

"""Cold-cache read-ahead for SSD-only reconstruction.

reconstruct_cache reads block payload files one at a time; on a cold page cache that left the SSD at queue
depth 1. It now hints the OS to read upcoming payload files (sliding window) via
BlockDiskStore.advise_block_reads. Hints never load data, take no lock, skip missing files, and the restored
KV is unchanged.
"""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import PagedCacheManager
import vmlx_engine.prefix_cache as prefix_cache


def _block():
    k = mx.zeros((1, 2, 8, 16), dtype=mx.float16)
    v = mx.ones((1, 2, 8, 16), dtype=mx.float16)
    mx.eval(k, v)
    return [("kv", k, v)]


def test_advise_counts_existing_files_and_skips_missing(tmp_path):
    store = BlockDiskStore(str(tmp_path), max_size_gb=0.1, allow_tq_native=False)
    try:
        hashes = [bytes([i]) * 32 for i in range(1, 6)]
        for h in hashes:
            assert store.write_block_async(h, _block(), token_count=8)
        assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
        assert store.advise_block_reads(hashes) == len(hashes)
        assert store.advise_block_reads([b"\xff" * 32]) == 0
        # A file removed behind the index is skipped, not an error.
        row = store._read_conn.execute(
            "SELECT file_name FROM blocks WHERE block_hash = ?", (hashes[0].hex(),)
        ).fetchone()
        (tmp_path / row[0]).unlink()
        assert store.advise_block_reads(hashes) == len(hashes) - 1
        assert store.advise_block_reads([]) == 0
    finally:
        store.shutdown()


def test_reconstruct_advises_every_disk_block_in_order(tmp_path, monkeypatch):
    mx.random.seed(3)
    cache = KVCache()
    n_tokens = 64 * 300  # 300 blocks > one read-ahead window
    for start in range(0, n_tokens, 2048):
        n = min(2048, n_tokens - start)
        cache.update_and_fetch(
            mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16),
            mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16),
        )
    mx.eval(cache.state)
    data = [{"state": cache.state, "meta_state": cache.meta_state, "class_name": "KVCache"}]
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    manager = PagedCacheManager(block_size=64, max_blocks=400, disk_store=store, disk_only=True)
    pc = prefix_cache.BlockAwarePrefixCache(
        model=None, paged_cache_manager=manager,
        uses_dsv4_cache=False, uses_zaya_cache=False, mixed_attention_cache_model=False,
    )
    tokens = list(range(n_tokens))
    assert pc.store_cache("seed", tokens, data) is not None
    deadline = time.time() + 30
    while time.time() < deadline and store.get_stats()["write_pipeline"]["pending_items"]:
        time.sleep(0.02)
    pc.release_cache("seed")

    advised: list = []
    original = store.advise_block_reads

    def spy(hashes):
        advised.extend(hashes)
        return original(hashes)

    monkeypatch.setattr(store, "advise_block_reads", spy)
    try:
        table, remaining = pc.fetch_cache("hit", tokens + [7])
        assert table is not None and len(remaining) == 1
        restored = pc.reconstruct_cache(table)
        assert restored is not None
        expected = [
            manager.allocated_blocks[block_id].block_hash for block_id in table.block_ids
        ]
        assert advised == expected, (len(advised), len(expected))
        rk, rv = restored[0].state
        k, v = cache.state
        assert mx.array_equal(rk.view(mx.uint16), k.view(mx.uint16)).item()
        assert mx.array_equal(rv.view(mx.uint16), v.view(mx.uint16)).item()
    finally:
        pc.release_cache("hit")
        store.shutdown()

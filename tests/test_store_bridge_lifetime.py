"""Full-store NumPy mirrors: no fp32 bridge for BF16, nothing outlives cleanup, bytes exact.

BF16 positional sources used to be widened to an fp32 "bridge" copy of the whole
cache on every store so NumPy could hold them. They are now mirrored as their raw
uint16 bits (a zero-copy view), so no bridge exists at all; the stored payload
must still round-trip bit-exactly."""

import weakref

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import PagedCacheManager
import vmlx_engine.prefix_cache as prefix_cache


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_full_store_releases_conversion_bridges_before_cleanup(tmp_path, monkeypatch, dtype):
    shape = (1, 4, 256, 64)
    pairs = [(mx.full(shape, 1.25 + i, dtype=dtype),
              mx.full(shape, -2.5 - i, dtype=dtype)) for i in range(2)]
    mx.eval(*[a for pair in pairs for a in pair])
    data = [{"state": pair, "meta_state": (256,), "class_name": "KVCache"}
            for pair in pairs]
    bridges = []
    bits_views = []
    observations = []
    original_view = prefix_cache._readonly_numpy_buffer_view
    original_finish = prefix_cache._cleanup_phase_finish

    def view(array):
        if dtype == mx.bfloat16 and array.dtype == mx.float32:
            bridges.append(weakref.ref(array))
        if dtype == mx.bfloat16 and array.dtype == mx.uint16:
            bits_views.append(array.shape)
        return original_view(array)

    def finish(start, site, request_id, succeeded, collected=None):
        if site == "store_views_gc":
            observations.append(sum(ref() is not None for ref in bridges))
        return original_finish(start, site, request_id, succeeded, collected)

    monkeypatch.setattr(prefix_cache, "_readonly_numpy_buffer_view", view)
    monkeypatch.setattr(prefix_cache, "_cleanup_phase_finish", finish)
    store = BlockDiskStore(str(tmp_path), max_size_gb=1)
    manager = PagedCacheManager(block_size=256, max_blocks=4,
                                disk_store=store, disk_only=True)
    cache = prefix_cache.BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    hashes = []
    original_write = store.write_block_async

    def write(block_hash, *args, **kwargs):
        hashes.append(block_hash)
        return original_write(block_hash, *args, **kwargs)

    monkeypatch.setattr(store, "write_block_async", write)
    try:
        assert cache.store_cache("bridge-lifetime", list(range(256)), data) is not None
        assert observations == [0], "Conversion bridges outlived the full-cache views"
        assert len(bridges) == 0, "BF16 store built an fp32 bridge copy of the cache"
        assert len(bits_views) == (4 if dtype == mx.bfloat16 else 0)
        assert len(hashes) == 1
        restored = store.read_block(hashes[0])
        assert restored is not None and len(restored) == 2
        for entry, pair in zip(restored, pairs):
            for actual, expected in zip(entry[1:3], pair):
                assert actual.dtype == dtype
                assert actual.shape == expected.shape
                assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()
        assert store.get_stats()["write_pipeline"]["pending_items"] == 0
    finally:
        store.shutdown()

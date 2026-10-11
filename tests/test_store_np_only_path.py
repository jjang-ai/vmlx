"""SSD-only frugal stores of plain/rotating KV build one payload per block (no discarded MLX copy).

In disk-only mode the per-block L1 payload is never kept and the disk payload is the
NumPy-mirror slice, so _extract_block_tensor_slice must not run for caches made only of
KVCache/RotatingKVCache layers. Any other layer type keeps the typed extractor. Stored
blocks must restore bit-exactly either way.
"""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache, RotatingKVCache

from vmlx_engine.block_disk_store import BlockDiskStore
from vmlx_engine.paged_cache import PagedCacheManager
import vmlx_engine.prefix_cache as prefix_cache


def _caches(tokens):
    mx.random.seed(5)
    caches = [KVCache(), RotatingKVCache(max_size=64, keep=0), KVCache()]
    dtypes = [mx.bfloat16, mx.bfloat16, mx.float16]
    fed = 0
    while fed < tokens:
        n = min(48, tokens - fed)
        for c, dt in zip(caches, dtypes):
            c.update_and_fetch(
                (mx.random.normal((1, 2, n, 32)) * 3).astype(dt),
                (mx.random.normal((1, 2, n, 32)) * 3).astype(dt),
            )
        fed += n
    mx.eval([c.state for c in caches])
    return caches


def _extracted(caches):
    return [{"state": c.state, "meta_state": c.meta_state, "class_name": type(c).__name__} for c in caches]


def _store(tmp_path, data, tokens):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    manager = PagedCacheManager(block_size=64, max_blocks=32, disk_store=store, disk_only=True)
    cache = prefix_cache.BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    assert cache.store_cache("np-only", list(range(tokens)), data) is not None
    deadline = time.time() + 10
    while time.time() < deadline and store.get_stats()["write_pipeline"]["pending_items"]:
        time.sleep(0.02)
    return store, manager


def test_plain_and_rotating_kv_skip_the_typed_extractor(tmp_path, monkeypatch):
    calls = []
    original = prefix_cache.BlockAwarePrefixCache._extract_block_tensor_slice

    def counting(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(prefix_cache.BlockAwarePrefixCache, "_extract_block_tensor_slice", counting)
    caches = _caches(200)
    store, manager = _store(tmp_path, _extracted(caches), 200)
    try:
        assert calls == [], "disk-only plain/rotating store built a discarded MLX block copy"
        hashes = [h for h in manager.cached_block_hash_to_block._cache]
        assert len(hashes) == 4
        full_k, full_v = caches[0].state
        restored_full = []
        for h in hashes:
            entries = store.read_block(h)
            assert entries is not None and len(entries) == 3
            restored_full.append(entries[0])
        # Positional bf16 layer 0 round-trips bit-exactly across the four blocks (64,64,64,8).
        got_k = mx.concatenate([e[1] for e in restored_full], axis=2)
        got_v = mx.concatenate([e[2] for e in restored_full], axis=2)
        assert got_k.dtype == mx.bfloat16 and got_k.shape == full_k.shape
        assert mx.array_equal(got_k.view(mx.uint16), full_k.view(mx.uint16)).item()
        assert mx.array_equal(got_v.view(mx.uint16), full_v.view(mx.uint16)).item()
    finally:
        store.shutdown()


def test_other_layer_types_keep_the_typed_extractor(tmp_path, monkeypatch):
    calls = []
    original = prefix_cache.BlockAwarePrefixCache._extract_block_tensor_slice

    def counting(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(prefix_cache.BlockAwarePrefixCache, "_extract_block_tensor_slice", counting)
    caches = _caches(128)
    data = _extracted(caches)
    # A cumulative (SSM-style) layer is not a plain/rotating KV layer.
    conv = mx.zeros((1, 3, 16), dtype=mx.bfloat16)
    ssm = mx.zeros((1, 2, 8, 8), dtype=mx.float32)
    mx.eval(conv, ssm)
    data.append({"state": [conv, ssm], "meta_state": "", "class_name": "ArraysCache"})
    store, _manager = _store(tmp_path, data, 128)
    try:
        assert calls, "a non-KV layer must keep the typed extractor path"
    finally:
        store.shutdown()

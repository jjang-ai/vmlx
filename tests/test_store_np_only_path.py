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


def _store(tmp_path, data, tokens, *, store_cumulative_state=True):
    store = BlockDiskStore(str(tmp_path), max_size_gb=1, allow_tq_native=False)
    manager = PagedCacheManager(block_size=64, max_blocks=32, disk_store=store, disk_only=True)
    cache = prefix_cache.BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
    assert cache.store_cache(
        "np-only", list(range(tokens)), data,
        store_cumulative_state=store_cumulative_state,
    ) is not None
    deadline = time.time() + 10
    while time.time() < deadline and store.get_stats()["write_pipeline"]["pending_items"]:
        time.sleep(0.02)
    return store, manager


@pytest.mark.parametrize("external_companion", [False, True])
def test_plain_and_rotating_kv_skip_the_typed_extractor(tmp_path, monkeypatch, external_companion):
    calls = []
    original = prefix_cache.BlockAwarePrefixCache._extract_block_tensor_slice

    def counting(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(prefix_cache.BlockAwarePrefixCache, "_extract_block_tensor_slice", counting)
    caches = _caches(200)
    data = _extracted(caches)
    if external_companion:
        # These native state tensors belong to the separately keyed SSM store;
        # every KV block must carry only their positional skip marker.
        data.append({
            "state": [mx.ones((1, 3, 16), dtype=mx.bfloat16), mx.ones((1, 2, 8, 8))],
            "meta_state": "", "class_name": "ArraysCache",
        })
    store, manager = _store(
        tmp_path, data, 200, store_cumulative_state=not external_companion,
    )
    try:
        assert calls == [], "disk-only plain/rotating store built a discarded MLX block copy"
        hashes = [h for h in manager.cached_block_hash_to_block._cache]
        assert len(hashes) == 4
        full_k, full_v = caches[0].state
        restored_full = []
        for h in hashes:
            entries = store.read_block(h)
            assert entries is not None and len(entries) == 3 + int(external_companion)
            if external_companion:
                assert entries[-1] == ("skip",)
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


def test_external_companion_pending_payload_matches_typed_payload():
    """A pending write-behind hit must see identical native state and metadata."""
    caches = _caches(200)
    data = _extracted(caches)
    data.insert(1, {"state": [mx.ones((1, 3, 16)), mx.ones((1, 2, 8, 8))],
                    "meta_state": "", "class_name": "ArraysCache"})
    sources = {}
    for i, layer in enumerate(data):
        if layer["class_name"] == "ArraysCache":
            continue
        k, v = layer["state"]
        encoding = prefix_cache._BF16_BITS if k.dtype == mx.bfloat16 else None
        a, b = (k.view(mx.uint16), v.view(mx.uint16)) if encoding else (k, v)
        mx.eval(a, b)
        sources[i] = (prefix_cache._readonly_numpy_buffer_view(a),
                      prefix_cache._readonly_numpy_buffer_view(b), k.dtype, encoding)
    manager = PagedCacheManager(block_size=64, max_blocks=32)
    cache = prefix_cache.BlockAwarePrefixCache(model=None, paged_cache_manager=manager)

    def same(a, b):
        if isinstance(a, mx.array):
            assert isinstance(b, mx.array) and a.dtype == b.dtype and a.shape == b.shape
            assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()
        elif isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        else:
            assert a == b

    for start in range(0, 200, 64):
        end = min(start + 64, 200)
        typed = cache._extract_block_tensor_slice(
            data, start, end, is_last_block=end == 200,
            np_sources=sources, store_cumulative_state=False,
        )
        single = prefix_cache._numpy_block_slice(
            data, sources, start, end, end == 200,
            store_cumulative_state=False, rotating_resume_block_size=64,
        )
        same(typed, single)

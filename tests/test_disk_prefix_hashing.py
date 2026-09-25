"""Preserve persistent cache identities while hashing shared prefixes once."""

import random

import pytest

import vmlx_engine.disk_cache as disk
from vmlx_engine.cache_key import canonical_cache_extra_marker


@pytest.mark.parametrize("marker", [None, "", "media:α", "lone-surrogate:\ud800"])
def test_incremental_hashes_match_existing_disk_keys(marker):
    rng = random.Random(83)
    tokens = [rng.randrange(-100, 2**32) for _ in range(1025)]
    lengths = [1025, 0, 1, 512, 513, 64, 512, 1024]
    actual = disk._hash_token_prefixes_with_marker(tokens, lengths, marker)
    assert actual == {
        n: disk._hash_tokens_with_marker(tokens[:n], marker) for n in lengths
    }


def test_shared_tokens_are_serialized_once(monkeypatch):
    tokens = list(range(8192))
    lengths = list(range(64, len(tokens) + 1, 64))
    original = disk.json.dumps
    serialized = []

    def counting(value, **kwargs):
        serialized.append(len(value))
        return original(value, **kwargs)

    monkeypatch.setattr(disk.json, "dumps", counting)
    actual = disk._hash_token_prefixes_with_marker(tokens, lengths, None)
    assert len(actual) == len(lengths)
    assert sum(serialized) == len(tokens)


def add_entry(manager, tokens, extra=None):
    token_hash = disk._hash_tokens(tokens, extra)
    conn = manager._pool.get()
    try:
        conn.execute(
            "INSERT INTO cache_entries "
            "(token_hash, file_name, num_tokens, file_size, created_at, "
            "last_accessed, access_count, metadata, cache_type, "
            "payload_prefix_hash, cache_extra_marker) "
            "VALUES (?, ?, ?, 1, 1, 1, 1, '{}', 'user', ?, ?)",
            (token_hash, token_hash + ".safetensors", len(tokens),
             disk._hash_tokens(tokens[:-1], extra), canonical_cache_extra_marker(extra)),
        )
        conn.commit()
    finally:
        manager._pool.put(conn)
    return token_hash


@pytest.mark.parametrize("extra", [None, {"media": "image-a", "reasoning": False}])
@pytest.mark.parametrize("change_sentinel", [False, True])
def test_many_candidates_keep_longest_prefix_and_payload_fallback(
    tmp_path, monkeypatch, extra, change_sentinel
):
    manager = disk.DiskCacheManager(str(tmp_path), max_size_gb=0)
    try:
        current = list(range(100))
        for n in [95, 85, 75, 65]:
            add_entry(manager, [999] * n, extra)
        stored = current[:55]
        if change_sentinel:
            stored[-1] = 777
        wanted = add_entry(manager, stored, extra)
        sentinel = [object()]
        loads = []

        def load(token_hash, tokens):
            loads.append((token_hash, list(tokens)))
            return sentinel if token_hash == wanted else None

        monkeypatch.setattr(manager, "_fetch_indexed_hash", load)
        monkeypatch.setattr(manager, "fetch", lambda tokens, cache_extra_keys=None:
                            load(disk._hash_tokens(tokens, cache_extra_keys), tokens))
        cache, matched = manager.fetch_longest_prefix(current, cache_extra_keys=extra)
        assert cache is sentinel
        assert matched == current[:55]
        assert loads == [(wanted, current[:55])]
        if extra is not None:
            assert manager.fetch_longest_prefix(current, cache_extra_keys={"media": "image-b"}) == (None, [])
    finally:
        manager.shutdown()


def test_longest_exact_hit_does_not_expand_all_hashes(tmp_path, monkeypatch):
    manager = disk.DiskCacheManager(str(tmp_path), max_size_gb=0)
    try:
        tokens = list(range(100))
        for n in [100, 90, 80, 70]:
            add_entry(manager, tokens[:n])
        def forbidden(*args):
            raise AssertionError("exact hit must keep the existing fast path")
        monkeypatch.setattr(disk, "_hash_token_prefixes_with_marker", forbidden)
        sentinel = [object()]
        monkeypatch.setattr(manager, "fetch", lambda _: sentinel)
        assert manager.fetch_longest_prefix(tokens) == (sentinel, tokens)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("layout", ["kv", "rotating", "hybrid"])
def test_incremental_lookup_restores_native_payload_after_restart(tmp_path, layout):
    mx = pytest.importorskip("mlx.core")
    tree_flatten = pytest.importorskip("mlx.utils").tree_flatten
    caches = pytest.importorskip("mlx_lm.models.cache")
    ArraysCache, KVCache, RotatingKVCache = (
        caches.ArraysCache, caches.KVCache, caches.RotatingKVCache
    )

    tokens = list(range(20))
    layer = KVCache()
    values = mx.arange(19 * 8).reshape(1, 1, 19, 8).astype(mx.bfloat16)
    layer.update_and_fetch(values, values)
    cache = [layer]
    if layout == "rotating":
        rotating = RotatingKVCache(max_size=8)
        rotating.update_and_fetch(values, values)
        cache.append(rotating)
    elif layout == "hybrid":
        recurrent = ArraysCache(size=2)
        recurrent[0] = mx.ones((1, 4, 8), dtype=mx.bfloat16)
        recurrent[1] = mx.ones((1, 4, 8, 8), dtype=mx.float32)
        cache.append(recurrent)
    extra = {"media": "unchanged-payload", "reasoning": "native"}
    writer = disk.DiskCacheManager(str(tmp_path), max_size_gb=0.1)
    try:
        assert writer.store(tokens, cache, cache_extra_keys=extra)
        writer._write_queue.join()
        for n in [30, 40, 50, 60]:
            add_entry(writer, [999] * n, extra)
    finally:
        writer.shutdown()
    reader = disk.DiskCacheManager(str(tmp_path), max_size_gb=0.1)
    try:
        current = tokens[:-1] + [777] + list(range(20, 80))
        restored, matched = reader.fetch_longest_prefix(current, cache_extra_keys=extra)
        assert restored is not None
        assert matched == current[:20]
        assert [type(c) for c in restored] == [type(c) for c in cache]
        for before, after in zip(cache, restored):
            original = tree_flatten(before.state)
            recovered = tree_flatten(after.state)
            assert len(original) == len(recovered)
            for (key, a), (other_key, b) in zip(original, recovered):
                assert key == other_key
                assert a.dtype == b.dtype
                assert mx.array_equal(a, b).item()
    finally:
        reader.shutdown()

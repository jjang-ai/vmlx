# SPDX-License-Identifier: Apache-2.0
"""CPU-only key parity and lookup routing; no native payload/GPU proof."""
import json

import pytest

from vmlx_engine.cache_key import scope_cache_extra_key
from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache


class Disk:
    def __init__(self, lengths, records=None):
        self.lengths = lengths
        self.records = records or {}
        self.lookups = []

    def candidate_lengths(self, max_len):
        return self.lengths

    def fetch(self, key):
        self.lookups.append(key)
        value = self.records.get(key)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("extra", [None, {}, {"setting": False}, {"media": "red"},
    scope_cache_extra_key({"media": "red"}, "media", 3),
    scope_cache_extra_key({"media": "red", "other": 1}, "media", 3),
])
def test_batch_hashes_match_both_existing_namespaces(extra):
    cache = SSMCompanionCache(model_key="模型/alpha", disk_store=False)
    tokens = [0, 1, -3, 151669, 987654321, 42]
    lengths = [9, 6, 5, 4, 3, 1, 0, -1, 3]
    actual = cache._lookup_prefix_hashes(tokens, lengths, extra)
    for length in set(lengths):
        assert actual[length] == (cache._prefix_hash(tokens, length, extra), cache._key(tokens, length, extra))


def test_salt_callback_resolved_for_each_boundary(monkeypatch):
    cache = SSMCompanionCache(disk_store=False)
    calls = []
    def salt(extra, n):
        calls.append(n)
        return b"" if n <= 3 else b"\x00extra=" + json.dumps([extra, n]).encode()
    monkeypatch.setattr(cache, "_extra_key_bytes", salt)
    actual = cache._lookup_prefix_hashes(list(range(10)), [6, 3, 4], "payload")
    assert calls == [3, 4, 6]
    for n in [3, 4, 6]:
        assert actual[n] == (cache._prefix_hash(list(range(10)), n, "payload"), cache._key(list(range(10)), n, "payload"))


@pytest.mark.parametrize("failure", [None, ValueError("corrupt row")])
def test_longest_disk_fallback_keeps_validation_and_key_order(failure):
    tokens = list(range(12))
    disk = Disk([4, 8, 10])
    cache = SSMCompanionCache(max_entries=0, max_bytes=0, model_key="qwen", disk_store=disk)
    keys = {n: cache._key(tokens, n) for n in [12, 10, 8, 4]}
    disk.records = {keys[10]: failure, keys[8]: ([{"state": "eight"}], True), keys[4]: ([{"state": "four"}], True)}
    assert cache.fetch_longest_prefix(tokens, 12) == (8, [{"state": "eight"}], True)
    assert disk.lookups == [keys[12], keys[10], keys[8]]
    assert cache.size == 0


def test_scoped_media_only_shares_pre_media_boundary():
    tokens = list(range(12))
    red = scope_cache_extra_key({"media": "red"}, "media", 4)
    blue = scope_cache_extra_key({"media": "blue"}, "media", 4)
    disk = Disk([4, 8])
    cache = SSMCompanionCache(max_entries=0, disk_store=disk)
    disk.records = {cache._key(tokens, n, red): ([{"n": n}], True) for n in [4, 8]}
    assert cache.fetch_longest_prefix(tokens, 12, blue)[0] == 4
    assert cache.fetch_longest_prefix(tokens, 12, red)[0] == 8


def test_public_fetch_and_l1_clone_contract():
    tokens = [1, 2, 3, 4]
    cache = SSMCompanionCache(max_entries=2, disk_store=False)
    key, family = cache._key(tokens, 3), cache._prefix_hash(tokens, 3)
    cache._store[key] = ([{"nested": [1]}], True)
    cache._length_index[3] = {family: key}
    exact = cache.fetch(tokens, 3)
    exact[0][0]["nested"].append(2)
    assert cache.fetch_longest_prefix(tokens, 4) == (3, [{"nested": [1]}], True)
    assert cache.fetch(tokens, 0) is None


def test_subclass_fetch_hook_retained():
    class Hook(SSMCompanionCache):
        def fetch(self, token_ids, num_tokens, cache_extra_keys=None):
            calls.append(num_tokens)
            return None
    calls = []
    cache = Hook(disk_store=Disk([3, 2]))
    assert cache.fetch_longest_prefix([1, 2, 3, 4], 4) is None
    assert calls == [4, 3, 2]


def test_exact_hit_keeps_fast_path_and_incomplete_flag():
    tokens = [1, 2, 3, 4]
    disk = Disk([2, 3])
    cache = SSMCompanionCache(max_entries=0, disk_store=disk)
    key = cache._key(tokens, 4)
    disk.records[key] = ([{"state": "exact"}], False)
    assert cache.fetch_longest_prefix(tokens, 4) == (4, [{"state": "exact"}], False)
    assert disk.lookups == [key]
    assert cache.last_prefix_lookup["source"] == "exact_boundary_l1_or_l2"


def test_injected_fetch_hook_retained(monkeypatch):
    cache = SSMCompanionCache(disk_store=Disk([3, 2]))
    calls = []
    def fetch(tokens, n, cache_extra_keys=None):
        calls.append(n)
        return None
    monkeypatch.setattr(cache, "fetch", fetch)
    assert cache.fetch_longest_prefix([1, 2, 3, 4], 4, exact_boundary_already_missed=True) is None
    assert calls == [3, 2]


@pytest.mark.parametrize("override", ["_key", "_prefix_hash"])
def test_custom_hash_namespace_retained(monkeypatch, override):
    tokens = [1, 2, 3, 4]
    cache = SSMCompanionCache(max_entries=2, disk_store=False)
    original = getattr(cache, override)
    def custom(*args, **kwargs):
        return original(*args, **kwargs) + "-custom"
    monkeypatch.setattr(cache, override, custom)
    key, family = cache._key(tokens, 3), cache._prefix_hash(tokens, 3)
    cache._store[key] = ([{"state": "custom"}], True)
    cache._length_index[3] = {family: key}
    assert cache.fetch_longest_prefix(tokens, 4) == (3, [{"state": "custom"}], True)


def test_lookup_serializes_each_token_once(monkeypatch):
    import vmlx_engine.utils.ssm_companion_cache as module
    original = module.json.dumps
    count = 0
    def dumps(value, *args, **kwargs):
        nonlocal count
        if isinstance(value, list):
            count += len(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(module.json, "dumps", dumps)
    cache = SSMCompanionCache(disk_store=Disk(list(range(1, 64))))
    assert cache.fetch_longest_prefix(list(range(64)), 64, exact_boundary_already_missed=True) is None
    assert count == 63

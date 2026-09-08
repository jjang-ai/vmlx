"""Store-side cache identities in the log (S5 audit, 2026-09-07).

A restore could only be bound to its publication by token count: the paged
store/hit lines carried no prefix identity and the SSM companion logged a hash
on fetch but none on store. Length is eligibility, not identity — the grader
had to call every restore INVALID. Both stores now log the same identity the
fetch side logs: ``prefix_key`` (terminal block hash) on the paged store and
hit lines, ``hash`` on the companion's store line (same units as ``SSM disk
HIT``)."""

import logging
import re
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.prefix_cache import BlockAwarePrefixCache
from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache


class _Layer:
    def __init__(self, marker: float) -> None:
        self.cache = [mx.array([marker] * 4)]


class _FakeDisk:
    """Captures the key the companion stores under and serves it back on fetch."""

    def __init__(self) -> None:
        self.entries = {}

    def store(self, key, states, is_complete, token_ids, num_tokens):
        self.entries[key] = (states, is_complete)

    def fetch(self, key):
        return self.entries.get(key)


def test_ssm_companion_store_logs_the_same_hash_the_fetch_logs(caplog):
    disk = _FakeDisk()
    cache = SSMCompanionCache(max_entries=0, max_bytes=0, model_key="m", disk_store=disk)
    assert cache.ram_enabled is False
    tokens = list(range(1, 41))
    with caplog.at_level(logging.INFO, logger="vmlx_engine.utils.ssm_companion_cache"):
        cache.store(tokens, 40, [_Layer(1.0)], is_complete=True)
        fetched = cache.fetch(tokens, 40)
    assert fetched is not None
    stored = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SSM stored:")]
    hits = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SSM disk HIT:")]
    assert len(stored) == 1 and len(hits) == 1, (stored, hits)
    h_store = re.search(r"N=(\d+) hash=([0-9a-f]{12})", stored[0])
    h_hit = re.search(r"N=(\d+) hash=([0-9a-f]{12})", hits[0])
    assert h_store and h_hit
    assert h_store.groups() == h_hit.groups() == ("40", h_hit.group(2))
    assert "disk=True" in stored[0] and "complete=True" in stored[0]
    # the identity partitions on the token content, not the length
    with caplog.at_level(logging.INFO, logger="vmlx_engine.utils.ssm_companion_cache"):
        cache.store(list(range(100, 140)), 40, [_Layer(2.0)], is_complete=True)
    stored2 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SSM stored:")]
    assert len(stored2) == 2 and re.search(r"hash=([0-9a-f]{12})", stored2[1]).group(1) != h_store.group(2)


def test_paged_prefix_key_is_the_terminal_block_hash_on_both_sides():
    blocks = [SimpleNamespace(block_hash=bytes([i]) * 32) for i in range(1, 4)]
    key = BlockAwarePrefixCache.prefix_key_for_blocks(blocks)
    assert key == (bytes([3]) * 32).hex()[:12]
    assert BlockAwarePrefixCache.prefix_key_for_blocks([]) is None
    assert BlockAwarePrefixCache.prefix_key_for_blocks([SimpleNamespace(block_hash=None)]) is None
    # the store side reads the same identity from the stored block table's physical ids
    cache = object.__new__(BlockAwarePrefixCache)
    cache.paged_cache = SimpleNamespace(blocks=[SimpleNamespace(block_hash=None)] + blocks)
    assert cache.prefix_key_for_block_ids([1, 2, 3]) == key
    assert cache.prefix_key_for_block_ids([]) is None
    assert cache.prefix_key_for_block_ids([99]) is None
    assert cache.prefix_key_for_block_ids(None) is None


def test_store_and_hit_log_lines_carry_prefix_key():
    import inspect

    from vmlx_engine import mllm_scheduler, prefix_cache

    hit = inspect.getsource(prefix_cache)
    assert 'checkpoint_tokens=%d%s prefix_key=%s' in hit
    store = open(mllm_scheduler.__file__).read()
    assert '"requested_cache_key_tokens=%d%s prefix_key=%s%s"' in store
    assert 'prefix_key_for_block_ids' in store


def test_store_line_lists_the_block_chain_keys_for_bounded_chains():
    """A later request sharing only a prefix hits an interior block; the store
    line's block_keys let that partial restore be bound by identity."""
    blocks = [SimpleNamespace(block_hash=bytes([i]) * 32) for i in range(1, 6)]
    cache = object.__new__(BlockAwarePrefixCache)
    cache.paged_cache = SimpleNamespace(blocks=[SimpleNamespace(block_hash=None)] + blocks)
    keys = cache.block_keys_for_block_ids([1, 2, 3, 4, 5])
    assert keys == ",".join((bytes([i]) * 32).hex()[:12] for i in range(1, 6))
    # the terminal key of a 3-block partial hit is inside the 5-block chain
    assert BlockAwarePrefixCache.prefix_key_for_blocks(blocks[:3]) in keys.split(",")
    assert cache.block_keys_for_block_ids([]) is None and cache.block_keys_for_block_ids([99]) is None
    cache.paged_cache = SimpleNamespace(blocks=[SimpleNamespace(block_hash=bytes([7]) * 32)] * 100)
    assert cache.block_keys_for_block_ids(list(range(65))) is None  # bounded: long chains log the terminal key only
    assert cache.block_keys_for_block_ids(list(range(64))) is not None
    from vmlx_engine import mllm_scheduler
    assert 'block_keys=' in open(mllm_scheduler.__file__).read()

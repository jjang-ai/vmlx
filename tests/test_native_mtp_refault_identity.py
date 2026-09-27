from collections import OrderedDict
from types import SimpleNamespace
import threading
import logging

import pytest

from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def _cache(chained=False):
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 2
    cache._chained_prefix_index_hash = chained
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()
    cache._prefix_index = {}
    tokens = [11, 22, 33, 44]
    first = compute_block_hash(None, tokens[:2], extra_keys=None)
    last = compute_block_hash(first, tokens[2:3], extra_keys=None)
    blocks = {
        20: SimpleNamespace(block_hash=first, token_count=2, ref_count=1),
        21: SimpleNamespace(block_hash=last, token_count=1, ref_count=1),
    }
    table = SimpleNamespace(block_ids=[20, 21], num_tokens=3)
    tables = {"refault": table}
    cache.paged_cache = SimpleNamespace(
        _lock=threading.RLock(),
        allocated_blocks=blocks,
        compute_block_hash=PagedCacheManager.compute_block_hash,
        get_block_table=tables.get,
    )
    cache._request_tables = {"refault": SimpleNamespace(block_table=table)}
    # Old slots are gone; the current request owns an exact refaulted chain.
    cache._prefix_index[cache._prefix_index_key(tokens[:3])] = (
        tokens[:3], [10, 11], None
    )
    marker = object()
    assert cache.store_mtp_prefix_snapshot(tokens, 3, marker)
    return cache, tokens, marker, tables


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("pruned", [False, True])
def test_refaulted_partial_snapshot_requires_exact_pinned_request(chained, pruned, caplog):
    cache, tokens, marker, _ = _cache(chained)
    if pruned:
        cache._prefix_index.clear()
    assert cache.restore_mtp_prefix_snapshot(tokens, 3) is None
    with caplog.at_level(logging.INFO):
        assert cache.restore_mtp_prefix_snapshot(tokens, 3, request_id="refault") is marker
    expected = "partial_index_missing" if pruned else "partial_chain_stale"
    assert f"sidecar refault restored: boundary=3 index_reason={expected}" in caplog.text


@pytest.mark.parametrize(
    "broken", ["table_boundary", "block_count", "unpinned", "hash", "canonical", "owner"]
)
def test_refault_snapshot_rejects_broken_ownership(broken):
    cache, tokens, _, tables = _cache()
    cache._prefix_index.clear()
    table = tables["refault"]
    blocks = cache.paged_cache.allocated_blocks
    if broken == "table_boundary":
        table.num_tokens = 2
    elif broken == "block_count":
        blocks[21].token_count = 2
    elif broken == "unpinned":
        blocks[20].ref_count = 0
    elif broken == "hash":
        blocks[21].block_hash = b"wrong"
    elif broken == "canonical":
        tables["refault"] = SimpleNamespace(block_ids=[20, 21], num_tokens=3)
    else:
        cache._request_tables.clear()
    assert cache.restore_mtp_prefix_snapshot(tokens, 3, request_id="refault") is None


def test_partial_snapshot_identity_excludes_uncached_suffix():
    cache, tokens, marker, _ = _cache()
    assert cache._mtp_prefix_snapshot_key(tokens, 3) == cache._mtp_prefix_snapshot_key(
        tokens[:3] + [900, 901], 3
    )
    assert cache.restore_mtp_prefix_snapshot(
        tokens[:3] + [900, 901], 3, request_id="refault"
    ) is marker


def test_partial_snapshot_identity_is_independent_of_legacy_index_collision():
    cache, tokens, marker, tables = _cache()
    cache.paged_cache.compute_block_hash = lambda _tokens: "forced-legacy-collision"
    changed = [11, 22, 77, 44]
    first = compute_block_hash(None, changed[:2], extra_keys=None)
    cache.paged_cache.allocated_blocks[21].block_hash = compute_block_hash(
        first, changed[2:3], extra_keys=None
    )
    cache._prefix_index.clear()
    assert cache._mtp_request_owns_boundary("refault", changed[:3])
    assert cache._mtp_prefix_snapshot_key(tokens, 3) != cache._mtp_prefix_snapshot_key(changed, 3)
    assert cache.restore_mtp_prefix_snapshot(changed, 3, request_id="refault") is None
    assert marker in [entry[1] for entry in cache._mtp_prefix_snapshots.values()]


def test_partial_snapshot_identity_separates_media_scope():
    cache, tokens, _, _ = _cache()
    assert cache._mtp_prefix_snapshot_key(tokens, 3, extra_keys=("image-a",)) != (
        cache._mtp_prefix_snapshot_key(tokens, 3, extra_keys=("image-b",))
    )
    assert cache.restore_mtp_prefix_snapshot(
        tokens, 3, request_id="refault", extra_keys=("image-a",)
    ) is None

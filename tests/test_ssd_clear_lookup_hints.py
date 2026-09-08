"""Explicit SSD clear must not let stale longer hints mask durable prefixes."""
from types import SimpleNamespace
from unittest.mock import Mock

from vmlx_engine.paged_cache import CacheBlock, PagedCacheManager
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def fixture(blocks, readable=()):
    manager = PagedCacheManager(block_size=64, max_blocks=16)
    manager._disk_store = SimpleNamespace(
        has_block_record=Mock(side_effect=lambda key: key in readable)
    )
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.paged_cache = manager
    cache._prefix_index = {}
    for block in blocks:
        manager.allocated_blocks[block.block_id] = block
        manager.cached_block_hash_to_block.insert(block.block_hash, block)
        manager.hash_to_block[block.hash_value] = block.block_id
        cache._prefix_index[str(block.block_id)] = ([block.block_id], [block.block_id])
    return cache, manager


def block(number, **kwargs):
    return CacheBlock(number, block_hash=bytes([number]) * 32,
                      hash_value=str(number), token_count=64, **kwargs)


def test_missing_longer_hint_retired_shorter_readable_prefix_kept():
    short, tail = block(1), block(2)
    cache, manager = fixture([short, tail], readable=[short.block_hash])
    cache._prefix_index['long'] = (list(range(128)), [1, 2])
    old_hash = tail.block_hash
    assert cache.retire_missing_disk_hints() == 1
    assert set(cache._prefix_index) == {'1'}
    assert manager.cached_block_hash_to_block.get_block(old_hash) is None
    assert manager.cached_block_hash_to_block.get_block(short.block_hash) is short
    assert tail.block_hash is None
    assert '2' not in manager.hash_to_block
    assert manager.allocated_blocks[2] is tail  # no allocator/ref mutation
    assert cache.retire_missing_disk_hints() == 0


def test_resident_referenced_pending_and_native_pinned_are_preserved():
    payload = object()
    blocks = [block(1, cache_data=payload), block(2, ref_count=1),
              block(3, durability_write_pending=True), block(4, keep_resident=True)]
    cache, manager = fixture(blocks)
    assert cache.retire_missing_disk_hints() == 0
    manager._disk_store.has_block_record.assert_not_called()
    assert blocks[0].cache_data is payload
    assert blocks[1].ref_count == 1
    assert len(cache._prefix_index) == 4


def test_duplicate_hash_representations_are_all_retired_with_one_lookup():
    first, second = block(1), block(2)
    second.block_hash = first.block_hash
    cache, manager = fixture([first, second])
    assert cache.retire_missing_disk_hints() == 2
    assert manager._disk_store.has_block_record.call_count == 1
    assert not cache._prefix_index


def test_revalidation_preserves_payload_installed_during_disk_check():
    candidate = block(1)
    cache, manager = fixture([candidate])
    payload = object()
    def changed(_key):
        candidate.cache_data = payload
        return False
    manager._disk_store.has_block_record.side_effect = changed
    assert cache.retire_missing_disk_hints() == 0
    assert candidate.cache_data is payload
    assert candidate.block_hash is not None


def test_no_disk_store_is_noop():
    cache, manager = fixture([block(1)])
    manager._disk_store = None
    assert cache.retire_missing_disk_hints() == 0
    assert cache._prefix_index

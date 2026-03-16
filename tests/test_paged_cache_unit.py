# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PagedCacheManager block operations.

Verifies:
- Block hash determinism (compute_block_hash)
- Different parent hashes produce different block hashes
- Allocate and free blocks round-trip
- Max blocks limit triggers eviction/failure
- Block computation for known token sequences
- First block (no parent) produces valid hash
"""

import pytest

from vmlx_engine.paged_cache import (
    BlockHash,
    CacheBlock,
    FreeKVCacheBlockQueue,
    PagedCacheManager,
    compute_block_hash,
)


# ---------------------------------------------------------------------------
# Tests for compute_block_hash (module-level function)
# ---------------------------------------------------------------------------


class TestBlockHash:
    """Tests for the compute_block_hash function."""

    def test_block_hash_deterministic(self):
        """Same (parent_hash, tokens) must produce the same block hash."""
        parent = BlockHash(b"parent_hash_bytes_here_32_bytes_")
        tokens = [10, 20, 30, 40]

        h1 = compute_block_hash(parent, tokens)
        h2 = compute_block_hash(parent, tokens)

        assert h1 == h2
        assert isinstance(h1, bytes)
        assert len(h1) == 32  # SHA-256 digest

    def test_different_parents_different_hashes(self):
        """Different parent hashes must produce different block hashes."""
        tokens = [10, 20, 30, 40]

        parent_a = BlockHash(b"a" * 32)
        parent_b = BlockHash(b"b" * 32)

        h_a = compute_block_hash(parent_a, tokens)
        h_b = compute_block_hash(parent_b, tokens)

        assert h_a != h_b

    def test_block_hash_with_empty_parent(self):
        """First block (no parent) produces valid hash using root seed."""
        tokens = [1, 2, 3]
        h = compute_block_hash(None, tokens)

        assert isinstance(h, bytes)
        assert len(h) == 32  # SHA-256 digest

        # Same tokens with None parent should be deterministic
        h2 = compute_block_hash(None, tokens)
        assert h == h2

        # Different tokens with None parent should differ
        h3 = compute_block_hash(None, [4, 5, 6])
        assert h != h3


# ---------------------------------------------------------------------------
# Tests for PagedCacheManager
# ---------------------------------------------------------------------------


class TestPagedCacheManager:
    """Tests for PagedCacheManager block operations."""

    def test_allocate_and_free_blocks(self):
        """Allocate blocks, free them, verify reusable."""
        mgr = PagedCacheManager(block_size=16, max_blocks=10, enable_caching=False)

        # Initial state: 9 free (10 total - 1 null block)
        assert mgr.free_blocks == 9

        # Allocate 3 blocks
        blocks = mgr.get_new_blocks(3)
        assert len(blocks) == 3
        assert mgr.free_blocks == 6

        # Free them
        mgr.free_blocks_method = mgr.free_block
        for b in blocks:
            freed = mgr.free_block(b.block_id)
            assert freed is True

        # All 9 should be free again
        assert mgr.free_blocks == 9

        # Re-allocate to confirm they are reusable
        blocks2 = mgr.get_new_blocks(3)
        assert len(blocks2) == 3
        assert mgr.free_blocks == 6

    def test_max_blocks_limit(self):
        """Allocate past max_blocks, verify we get an error (no space)."""
        mgr = PagedCacheManager(block_size=16, max_blocks=5, enable_caching=False)

        # 4 free blocks (5 total - 1 null block)
        assert mgr.free_blocks == 4

        # Allocate all 4
        blocks = mgr.get_new_blocks(4)
        assert len(blocks) == 4
        assert mgr.free_blocks == 0

        # Trying to allocate more should raise ValueError
        with pytest.raises(ValueError, match="Cannot allocate"):
            mgr.get_new_blocks(1)

        # allocate_block should return None when out of blocks
        result = mgr.allocate_block()
        assert result is None

    def test_compute_blocks_for_tokens(self):
        """Verify correct block computation for known token sequence via cache_full_blocks."""
        mgr = PagedCacheManager(block_size=4, max_blocks=20, enable_caching=True)

        # Allocate 3 blocks for 10 tokens (block_size=4 => 3 blocks, last partial)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        num_full_blocks = len(tokens) // mgr.block_size  # 2 full blocks

        blocks = mgr.get_new_blocks(3)  # 3 blocks to hold 10 tokens

        # Cache the 2 full blocks
        mgr.cache_full_blocks(blocks, tokens, num_cached_blocks=0, num_full_blocks=num_full_blocks)

        # Verify block hashes were set on the 2 full blocks
        assert blocks[0].block_hash is not None
        assert blocks[1].block_hash is not None

        # Verify the hashes are chain-based (block 1's hash depends on block 0)
        expected_h0 = compute_block_hash(None, [1, 2, 3, 4])
        expected_h1 = compute_block_hash(expected_h0, [5, 6, 7, 8])
        assert blocks[0].block_hash == expected_h0
        assert blocks[1].block_hash == expected_h1

        # The partial block (block 2, tokens [9, 10]) should NOT have been cached
        # because it's not a full block
        # (cache_full_blocks only caches up to num_full_blocks)


class TestFreeKVCacheBlockQueue:
    """Tests for the FreeKVCacheBlockQueue doubly-linked list."""

    def test_popleft_and_append_round_trip(self):
        """Pop from front, append to back, verify ordering."""
        blocks = [CacheBlock(block_id=i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)

        assert q.num_free_blocks == 5

        # Pop first
        b = q.popleft()
        assert b.block_id == 0
        assert q.num_free_blocks == 4

        # Append back
        q.append(b)
        assert q.num_free_blocks == 5

        # Pop all -- should come out in order 1,2,3,4,0 (0 was appended to end)
        ids = [q.popleft().block_id for _ in range(5)]
        assert ids == [1, 2, 3, 4, 0]

    def test_empty_queue_raises(self):
        """Popping from empty queue should raise ValueError."""
        q = FreeKVCacheBlockQueue([])
        assert q.num_free_blocks == 0

        with pytest.raises(ValueError, match="No free blocks"):
            q.popleft()

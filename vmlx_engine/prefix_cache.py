# SPDX-License-Identifier: Apache-2.0
"""
Prefix Cache Manager for vmlx-engine.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.

This module provides two implementations:
- PrefixCacheManager: Original trie-based LRU cache (for backward compatibility)
- BlockAwarePrefixCache: Block-based cache with PagedCacheManager integration
"""

import copy
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .paged_cache import BlockTable, PagedCacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: List[Any]  # The cached KV state
    count: int  # Reference count for sharing


@dataclass
class PrefixCacheStats:
    """Statistics for prefix cache performance."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


class PrefixCacheManager:
    """
    Manages prefix caching for vmlx-engine using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for vmlx-engine's batching architecture.

    The cache stores KV states keyed by token sequences, allowing:
    - Exact match: Full prompt found in cache
    - Shorter match: Partial prefix found, process remaining tokens
    - Longer match: Cached prefix longer than request, trim excess

    Example:
        cache_manager = PrefixCacheManager(model, max_entries=100)

        # Check for cached prefix
        cache, remaining_tokens = cache_manager.fetch_cache(tokens)
        if cache:
            # Use cached KV, only process remaining_tokens
            pass

        # After generation, store cache for reuse
        cache_manager.store_cache(full_tokens, prompt_cache)
    """

    def __init__(self, model: Any, max_entries: int = 100):
        """
        Initialize the prefix cache manager.

        Args:
            model: The MLX model (used for cache key identification)
            max_entries: Maximum number of cached entries before LRU eviction
        """
        self.model = model
        self.model_key = id(model)
        self.max_size = max_entries

        # Trie-based cache: nested dicts with token keys
        # Structure: {model_key: {token1: {token2: {..., "cache": CacheEntry}}}}
        self._cache: Dict[Any, Dict] = {}

        # LRU tracking: OrderedDict keyed by (model_key, tuple(tokens))
        # O(1) move_to_end, O(1) popitem, no duplicate entries
        self._lru: OrderedDict = OrderedDict()

        # Statistics
        self.stats = PrefixCacheStats()

    def _search(
        self, tokens: List[int]
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], int]:
        """
        Search for cached prefix matching tokens.

        Returns:
            Tuple of (exact, shorter, longer, common_prefix_len)
            - exact: Tokens if exact match found
            - shorter: Tokens of shorter cached prefix
            - longer: Tokens of longer cached prefix
            - common_prefix_len: Length of common prefix with longer match
        """
        if self.model_key not in self._cache:
            return None, None, None, 0

        current = self._cache[self.model_key]
        path = []

        # Traverse trie following token sequence
        for i, tok in enumerate(tokens):
            if tok not in current:
                # No match for this token
                # Check if we have a shorter prefix with cache
                if "cache" in current:
                    return None, list(path), None, 0
                return None, None, None, 0

            path.append(tok)
            current = current[tok]

        # Reached end of tokens
        if "cache" in current:
            # Exact match
            return list(tokens), None, None, 0

        # Check for longer cached prefix
        # BFS to find shortest extension with cache
        from collections import deque
        queue = deque([(current, list(path))])
        while queue:
            node, node_path = queue.popleft()
            if "cache" in node:
                return None, None, node_path, len(tokens)
            for tok, child in node.items():
                if tok != "cache":
                    queue.append((child, node_path + [tok]))

        return None, None, None, 0

    def fetch_cache(self, tokens: List[int]) -> Tuple[Optional[List[Any]], List[int]]:
        """
        Find cached prefix for the given tokens.

        Args:
            tokens: Input token sequence

        Returns:
            Tuple of (cache, remaining_tokens)
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self.stats.total_queries += 1
        tokens_tuple = tuple(tokens)

        exact, shorter, longer, common_len = self._search(tokens)

        if exact:
            # Exact match - return full cache
            cache_entry = self._get_cache_entry(exact)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                self._touch_lru(tokens_tuple)
                # Return reference directly — MLX arrays are immutable,
                # so sharing the cache is safe (no mutation possible).
                return cache_entry.prompt_cache, []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter))
                remaining = tokens[len(shorter) :]
                # Return reference directly — MLX arrays are immutable,
                # so sharing the cache is safe (no mutation possible).
                return cache_entry.prompt_cache, remaining

        if longer:
            # Longer prefix cached - trim to match and return
            cache_entry = self._get_cache_entry(longer)
            if cache_entry:
                # Check if cache supports trimming
                prompt_cache = cache_entry.prompt_cache
                if self._can_trim_cache(prompt_cache):
                    trim_amount = len(longer) - len(tokens)
                    # Deep copy IS needed here: _trim_cache mutates the cache
                    # objects' offset/keys/values refs via .trim(), which would
                    # corrupt the stored entry. (Unlike the read-only fetch paths
                    # above where sharing is safe.)
                    trimmed_cache = self._trim_cache(
                        copy.deepcopy(prompt_cache), trim_amount
                    )
                    self.stats.hits += 1
                    self.stats.tokens_saved += len(tokens)
                    return trimmed_cache, []

        # No cache hit
        self.stats.misses += 1
        return None, tokens

    def store_cache(self, tokens: List[int], prompt_cache: List[Any]) -> None:
        """
        Store computed cache for future reuse.

        Args:
            tokens: Token sequence that was processed
            prompt_cache: The computed KV cache to store
        """
        if not tokens:
            return

        tokens_tuple = tuple(tokens)

        # Build trie path
        if self.model_key not in self._cache:
            self._cache[self.model_key] = {}

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Store or update cache entry
        key = (self.model_key, tokens_tuple)
        if "cache" in current:
            current["cache"].count += 1
        else:
            current["cache"] = CacheEntry(prompt_cache, 1)

        # Move to end (most recently used); OrderedDict handles dedup
        self._lru[key] = True
        self._lru.move_to_end(key)

        # Evict if over capacity
        while len(self._lru) > self.max_size:
            self._evict_lru()

    def _get_cache_entry(self, tokens: List[int]) -> Optional[CacheEntry]:
        """Get cache entry for given tokens."""
        if self.model_key not in self._cache:
            return None

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                return None
            current = current[tok]

        return current.get("cache")

    def _touch_lru(self, tokens_tuple: tuple) -> None:
        """Move entry to end of LRU queue (most recently used). O(1)."""
        key = (self.model_key, tokens_tuple)
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = True

    def _evict_lru(self) -> None:
        """Evict least recently used entry. O(1)."""
        if not self._lru:
            return

        (model_key, tokens_tuple), _ = self._lru.popitem(last=False)
        self._delete_cache(model_key, list(tokens_tuple))
        self.stats.evictions += 1

    def _delete_cache(self, model_key: Any, tokens: List[int]) -> None:
        """Delete cache entry and clean up empty trie branches."""
        if model_key not in self._cache:
            return

        # Navigate to entry
        path = [(self._cache[model_key], None)]
        current = self._cache[model_key]

        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]

        # Delete cache entry
        if "cache" in current:
            del current["cache"]

        # Clean up empty branches (bottom-up)
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:  # Empty dict
                del parent[tok]

    def _can_trim_cache(self, prompt_cache: List[Any]) -> bool:
        """Check if cache can be trimmed."""
        if not prompt_cache:
            return False
        # Check if first cache layer has is_trimmable method
        first_cache = prompt_cache[0]
        if hasattr(first_cache, "is_trimmable"):
            return first_cache.is_trimmable()
        return hasattr(first_cache, "trim")

    def _trim_cache(self, prompt_cache: List[Any], num_tokens: int) -> List[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._lru.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._lru)


# =============================================================================
# Block-Aware Prefix Cache (uses PagedCacheManager)
# =============================================================================


def _block_needs_cumulative_update(cache_data) -> bool:
    """Check if a block's cache_data is missing cumulative SSM state.

    Returns True if the block has "skip" entries (SSM layers stored as
    non-last) but no "cumulative" entries. This indicates the block was
    originally stored at a non-last position and needs cumulative state
    before it can be used as the last block in a sequence.

    Checks both top-level entries and sub-slices inside CacheList entries
    (for MoE hybrid models where SSM sub-caches are nested).

    Used by store_cache() to detect when block reuse would lose SSM state
    for hybrid models (KVCache + MambaCache/ArraysCache).
    """
    if not cache_data:
        return False
    has_skip = False
    has_cumulative = False
    for entry in cache_data:
        if not isinstance(entry, (tuple, list)) or len(entry) == 0:
            continue
        tag = entry[0]
        if tag == "skip":
            has_skip = True
        elif tag == "cumulative":
            has_cumulative = True
        elif tag == "cache_list" and len(entry) > 1:
            # Recurse into CacheList sub-slices for MoE hybrid models
            for sub in entry[1]:
                if isinstance(sub, (tuple, list)) and len(sub) > 0:
                    if sub[0] == "skip":
                        has_skip = True
                    elif sub[0] == "cumulative":
                        has_cumulative = True
    return has_skip and not has_cumulative


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: List[Any]  # Actual KV cache data per block
    last_access: float


class BlockAwarePrefixCache:
    """
    Prefix cache that uses PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (64 tokens per block)
    - Copy-on-Write for efficient forking
    - Hash-based deduplication across requests
    - Reference counting for memory efficiency

    This is the recommended cache for production use when memory
    efficiency for concurrent requests is important.

    Example:
        paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)

        # After generation, store cache
        cache.store_cache(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
    ):
        """
        Initialize block-aware prefix cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache_manager: The PagedCacheManager instance for block management
        """
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size

        # Hash table for quick prefix lookup
        # Maps hash(tokens[:block_size*n]) -> (tokens, block_ids)
        self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

        # Lazy-cached expected KV head count for validation
        self._n_kv_heads: Optional[int] = None

    def _get_n_kv_heads(self) -> int:
        """Get expected KV head count from model config (cached).

        Only returns num_key_value_heads or num_kv_heads — never falls back
        to num_attention_heads, which is wrong for GQA models (e.g. 32 attn
        heads but 8 KV heads). Returns 0 if unknown (skips validation).
        """
        if self._n_kv_heads is not None:
            return self._n_kv_heads
        n_kv = 0
        try:
            for attr in ('args', 'config'):
                cfg = getattr(self.model, attr, None)
                if cfg is None:
                    continue
                n_kv = (
                    getattr(cfg, 'num_key_value_heads', 0)
                    or getattr(cfg, 'num_kv_heads', 0)
                )
                if n_kv:
                    break
        except Exception:
            pass
        if not isinstance(n_kv, int):
            n_kv = 0
        self._n_kv_heads = n_kv
        return n_kv

    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # Try to find shared prefix blocks
        shared_block_ids, remaining = self.paged_cache.find_shared_prefix(tokens)

        if shared_block_ids:
            # Create block table for this request with shared blocks
            block_table = self.paged_cache.create_block_table(request_id)

            for block_id in shared_block_ids:
                # Increment ref count for sharing
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            num_prefix_tokens = len(tokens) - len(remaining)
            self._hits += 1
            self._tokens_saved += num_prefix_tokens

            logger.debug(
                f"Cache hit for {request_id}: "
                f"{len(shared_block_ids)} blocks, {num_prefix_tokens} tokens"
            )

            return block_table, remaining

        # Try prefix index for longer matches
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            matched_tokens, matched_block_ids = best_match

            # Fork the matched blocks
            block_table = self.paged_cache.create_block_table(request_id)
            for block_id in matched_block_ids:
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            remaining = tokens[len(matched_tokens) :]
            self._hits += 1
            self._tokens_saved += len(matched_tokens)

            logger.debug(
                f"Prefix index hit for {request_id}: "
                f"{len(matched_tokens)} tokens matched"
            )

            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        This method stores actual tensor data (not references) when cache_data
        contains extracted states from mlx-lm's KVCache.state property.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy, stores references)
                - List of dicts with 'state': (keys, values) tensors (new, stores slices)

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            # All tokens already cached
            return block_table

        # Allocate blocks for new tokens
        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size

        # For disk write-through, compute chain hashes over the full token sequence.
        # Reconstruct parent_hash from existing blocks (if any).
        from .paged_cache import compute_block_hash as _compute_chain_hash
        parent_hash = None
        if existing_tokens > 0:
            # Recompute chain hash up to where the existing blocks end
            num_existing_full = existing_tokens // self.block_size
            for eb_idx in range(num_existing_full):
                eb_start = eb_idx * self.block_size
                eb_end = eb_start + self.block_size
                parent_hash = _compute_chain_hash(parent_hash, tokens[eb_start:eb_end])

        disk_store = self.paged_cache._disk_store  # May be None

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx

            # Compute chain hash for this block
            block_chain_hash = _compute_chain_hash(parent_hash, block_tokens)

            # Check if this block already exists via chain hash (deduplication)
            # IMPORTANT: lookup + ref bump must be atomic under _lock to prevent
            # the block from being freed between get_block and increment_ref.
            is_last = (i == num_new_blocks - 1)
            reused = False
            with self.paged_cache._lock:
                existing_block = self.paged_cache.cached_block_hash_to_block.get_block(
                    block_chain_hash
                )
                if existing_block:
                    # If this is the last block in the new sequence and the
                    # existing block was stored as non-last (SSM layers tagged
                    # "skip" without cumulative state), skip reuse so we can
                    # allocate a new block with proper cumulative SSM state.
                    # Without this, hybrid SSM models get "Reconstructed 10
                    # layers but expected 40" because cumulative entries are
                    # never stored.
                    if is_last and is_tensor_data and _block_needs_cumulative_update(
                        existing_block.cache_data
                    ):
                        pass  # Fall through to new block allocation
                    else:
                        existing_block.ref_count += 1
                        existing_block.touch()
                        if existing_block.ref_count == 2:
                            self.paged_cache.stats.shared_blocks += 1
                        reused = True
            if reused:
                block_table.block_ids.append(existing_block.block_id)
                block_table.num_tokens += len(block_tokens)
                self.paged_cache.stats.total_tokens_cached += len(block_tokens)
                parent_hash = block_chain_hash
                continue

            # Also check legacy hash for blocks stored before chain hashing.
            # find_cached_block already holds _lock internally, but we need to
            # do the ref bump under the same lock to avoid the same race.
            with self.paged_cache._lock:
                hash_value = self.paged_cache.compute_block_hash(block_tokens)
                existing_block = None
                if hash_value in self.paged_cache.hash_to_block:
                    bid = self.paged_cache.hash_to_block[hash_value]
                    if bid in self.paged_cache.allocated_blocks:
                        existing_block = self.paged_cache.allocated_blocks[bid]
                        # Same cumulative state check as chain hash path:
                        # skip reuse if last block needs cumulative SSM state
                        if is_last and is_tensor_data and _block_needs_cumulative_update(
                            existing_block.cache_data
                        ):
                            existing_block = None  # Fall through to allocation
                        else:
                            existing_block.ref_count += 1
                            existing_block.touch()
                            if existing_block.ref_count == 2:
                                self.paged_cache.stats.shared_blocks += 1
                            self.paged_cache.stats.cache_hits += 1
                            # Set chain hash on the existing block if it doesn't have one
                            if existing_block.block_hash is None:
                                existing_block.block_hash = block_chain_hash
                                self.paged_cache.cached_block_hash_to_block.insert(
                                    block_chain_hash, existing_block
                                )
                if existing_block is None:
                    self.paged_cache.stats.cache_misses += 1
            if existing_block:
                block_table.block_ids.append(existing_block.block_id)
                block_table.num_tokens += len(block_tokens)
                self.paged_cache.stats.total_tokens_cached += len(block_tokens)
                parent_hash = block_chain_hash
                continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                # Handle memory pressure
                if not self.paged_cache.handle_memory_pressure(1):
                    logger.warning(f"Cannot allocate block for {request_id}")
                    break
                block = self.paged_cache.allocate_block()
                if not block:
                    break

            # Store block data
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)
            self.paged_cache.stats.total_tokens_cached += len(block_tokens)

            # Set chain hash on the block (for L1 dedup and L2 disk addressing)
            block.block_hash = block_chain_hash

            # Extract and store actual tensor slices for this block
            if is_tensor_data and HAS_MLX:
                # is_last already computed above (for cumulative state checks)
                block_kv_data = self._extract_block_tensor_slice(
                    cache_data, global_start, global_end, is_last_block=is_last,
                )
                if block_kv_data:
                    block.cache_data = block_kv_data
                    logger.debug(
                        f"Stored tensor slice for block {block.block_id}: "
                        f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                        f"{' (includes cumulative states)' if is_last else ''}"
                    )

                    # Write-through to disk L2 (async, non-blocking)
                    if disk_store is not None:
                        disk_store.write_block_async(
                            block_chain_hash, block_kv_data, len(block_tokens)
                        )

            # Register in hash caches under lock (both chain hash and legacy)
            with self.paged_cache._lock:
                self.paged_cache.cached_block_hash_to_block.insert(
                    block_chain_hash, block
                )
            self.paged_cache.register_block_hash(block, block_tokens)

            parent_hash = block_chain_hash

        # Update prefix index
        self._update_prefix_index(tokens, block_table.block_ids)

        # Store entry for request (for legacy compatibility)
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data,
            last_access=time.time(),
        )

        blocks_with_data = sum(
            1
            for bid in block_table.block_ids
            if self.paged_cache.allocated_blocks.get(bid)
            and self.paged_cache.allocated_blocks[bid].cache_data is not None
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_with_data} with tensor data), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    @staticmethod
    def _is_positional_cache(state_tuple, class_name: str = "") -> bool:
        """
        Determine if a cache layer's state is position-indexed or cumulative.

        Positional (sliceable by token position):
            KVCache, RotatingKVCache, QuantizedKVCache
        Cumulative (represents all processed tokens):
            MambaCache, ArraysCache

        Args:
            state_tuple: Cache state (keys, values) or arrays list
            class_name: Cache class name for disambiguation
        """
        if not state_tuple:
            return False

        # Class name is most reliable
        if class_name:
            positional = {"KVCache", "BatchKVCache", "RotatingKVCache",
                          "BatchRotatingKVCache", "QuantizedKVCache"}
            cumulative = {"MambaCache", "BatchMambaCache", "ArraysCache"}
            if any(cls in class_name for cls in positional):
                return True
            if any(cls in class_name for cls in cumulative):
                return False

        # Structure-based fallback
        if isinstance(state_tuple, (tuple, list)) and len(state_tuple) == 2:
            first = state_tuple[0]
            if hasattr(first, "shape") and len(first.shape) in (3, 4):
                return True
            # QuantizedKVCache: state is ((data, scales, zeros), (data, scales, zeros))
            # first element is a tuple of arrays, not an array itself
            if isinstance(first, tuple) and len(first) >= 2:
                if hasattr(first[0], "shape") and len(first[0].shape) in (3, 4):
                    return True
        return False

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_last_block: bool = False,
    ) -> Optional[List[Tuple]]:
        """
        Extract tensor slices for a single block from cache data.

        Handles both positional caches (KVCache - attention layers) and
        cumulative caches (MambaCache - SSM/hybrid layers).

        For KVCache layers: slices the KV tensors by token position.
        For MambaCache layers: stores the full cumulative state (only in
        the last block, since it represents all processed tokens).

        Args:
            cache_data: List of layer states from _extract_cache_states
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence
            is_last_block: Whether this is the last block in the sequence

        Returns:
            List of tuples per layer. Each tuple is either:
            - ("kv", keys_slice, values_slice) for positional layers
            - ("cumulative", state_list) for cumulative/SSM layers
            - ("skip",) for cumulative layers in non-last blocks
        """
        if not HAS_MLX or not cache_data:
            return None

        block_slices = []
        for layer_idx, layer_state in enumerate(cache_data):
            if "state" not in layer_state:
                continue

            class_name = layer_state.get("class_name", "")

            # CacheList (MoE models): extract slices from each sub-cache
            if class_name == "CacheList" and "sub_caches" in layer_state:
                sub_slices = []
                for sub in layer_state["sub_caches"]:
                    sub_state = sub.get("state")
                    sub_cls = sub.get("class_name", "")
                    if sub_state is None:
                        sub_slices.append(("skip",))
                        continue
                    if self._is_positional_cache(sub_state, sub_cls):
                        try:
                            keys, values = sub_state
                            if isinstance(keys, (tuple, list)):
                                first_k = keys[0]
                                seq_len = first_k.shape[-2]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                keys_slice = tuple(t[..., start_idx:actual_end, :] for t in keys)
                                values_slice = tuple(t[..., start_idx:actual_end, :] for t in values)
                                meta = sub.get("meta_state", ())
                                sub_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                            else:
                                ndim = len(keys.shape)
                                seq_dim = 2 if ndim == 4 else (1 if ndim == 3 else -1)
                                if seq_dim < 0:
                                    sub_slices.append(("skip",))
                                    continue
                                seq_len = keys.shape[seq_dim]
                                actual_end = min(end_idx, seq_len)
                                if start_idx >= actual_end:
                                    sub_slices.append(("skip",))
                                    continue
                                if ndim == 4:
                                    ks = keys[:, :, start_idx:actual_end, :]
                                    vs = values[:, :, start_idx:actual_end, :]
                                else:
                                    ks = keys[:, start_idx:actual_end, :]
                                    vs = values[:, start_idx:actual_end, :]
                                sub_slices.append(("kv", ks, vs))
                        except Exception:
                            sub_slices.append(("skip",))
                    else:
                        if is_last_block:
                            meta = sub.get("meta_state", "")
                            sub_slices.append(("cumulative", sub_state, meta, sub_cls))
                        else:
                            sub_slices.append(("skip",))
                block_slices.append(("cache_list", sub_slices))
                continue

            state = layer_state["state"]

            # Detect if this is a positional (KVCache) or cumulative (MambaCache) layer
            if self._is_positional_cache(state, class_name):
                try:
                    keys, values = state

                    # QuantizedKVCache: keys/values are tuples or lists of (data, scales, zeros)
                    if isinstance(keys, (tuple, list)):
                        # Use first component to detect shape
                        first_k = keys[0]
                        seq_len = first_k.shape[-2]  # seq axis is always -2
                        actual_end = min(end_idx, seq_len)
                        if start_idx >= actual_end:
                            block_slices.append(("skip",))
                            continue

                        keys_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in keys
                        )
                        values_slice = tuple(
                            t[..., start_idx:actual_end, :] for t in values
                        )
                        meta = layer_state.get("meta_state", ())
                        block_slices.append(("quantized_kv", keys_slice, values_slice, meta))
                        continue

                    ndim = len(keys.shape)

                    # Handle both 3D (n_kv_heads, seq, dim) and
                    # 4D (batch, n_kv_heads, seq, dim) tensors
                    if ndim == 4:
                        seq_dim = 2
                    elif ndim == 3:
                        seq_dim = 1
                    else:
                        block_slices.append(("skip",))
                        continue

                    seq_len = keys.shape[seq_dim]
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        block_slices.append(("skip",))
                        continue

                    if ndim == 4:
                        keys_slice = keys[:, :, start_idx:actual_end, :]
                        values_slice = values[:, :, start_idx:actual_end, :]
                    else:  # ndim == 3
                        keys_slice = keys[:, start_idx:actual_end, :]
                        values_slice = values[:, start_idx:actual_end, :]

                    # Use rotating_kv tag for RotatingKVCache to preserve params
                    if "Rotating" in class_name:
                        # Extract max_size/keep from meta_state tuple
                        # RotatingKVCache.meta_state returns (keep, max_size, ...)
                        meta = layer_state.get("meta_state", ())
                        if meta and len(meta) >= 2:
                            try:
                                keep = int(meta[0])
                                max_size = int(meta[1])
                            except (ValueError, TypeError):
                                max_size = seq_len
                                keep = 0
                        else:
                            max_size = layer_state.get("max_size", seq_len)
                            keep = layer_state.get("keep", 0)
                        block_slices.append(("rotating_kv", keys_slice, values_slice, max_size, keep))
                    else:
                        block_slices.append(("kv", keys_slice, values_slice))
                except Exception as e:
                    logger.warning(
                        f"Layer {layer_idx} ({class_name}): "
                        f"failed to slice positional cache: {e}"
                    )
                    block_slices.append(("skip",))
            else:
                # Cumulative cache (MambaCache, ArraysCache, etc.)
                # State is not position-indexed — it represents ALL tokens processed
                # Only store in the last block (it encompasses all prior tokens)
                if is_last_block:
                    meta = layer_state.get("meta_state", "")
                    block_slices.append(("cumulative", state, meta, class_name))
                else:
                    block_slices.append(("skip",))

        if block_slices:
            tag_counts: dict = {}
            for bs in block_slices:
                t = bs[0] if isinstance(bs, (tuple, list)) else "?"
                tag_counts[t] = tag_counts.get(t, 0) + 1
            tag_str = ", ".join(f"{k}={v}" for k, v in tag_counts.items())
            logger.info(
                f"Block tensor slice: {len(block_slices)}/{len(cache_data)} layers "
                f"({tag_str}), tokens [{start_idx}:{end_idx}], is_last={is_last_block}"
            )
        return block_slices if block_slices else None

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, applying COW if needed.

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_copied)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        blocks, was_copied = self.paged_cache.get_blocks_for_generation(
            entry.block_table
        )

        if was_copied:
            # Deep copy cache data for modified blocks
            cache_data = copy.deepcopy(entry.cache_data)
        else:
            cache_data = entry.cache_data

        entry.last_access = time.time()
        return cache_data, was_copied

    def release_cache(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Args:
            request_id: Request identifier
        """
        entry = self._request_tables.pop(request_id, None)
        if entry:
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry with reference to same cache data
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,  # Shared reference
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from stored block data.

        Handles both positional caches (KVCache - attention layers) and
        cumulative caches (MambaCache - SSM/hybrid layers):
        - KVCache: concatenates tensor slices from all blocks along seq axis
        - MambaCache: restores full cumulative state from the last block

        Uses mlx_lm's cache classes and from_state() for proper reconstruction.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed cache objects (one per layer),
            or None if reconstruction fails
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(f"Block {block_id} not found in allocated blocks")
                    return None

                if block.cache_data is None:
                    logger.debug(f"Block {block_id} has no tensor data stored")
                    return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # Import cache classes
            try:
                from mlx_lm.models.cache import KVCache, MambaCache, RotatingKVCache
                has_mamba = True
                has_rotating = True
            except ImportError:
                try:
                    from mlx_lm.models.cache import KVCache, MambaCache
                    has_mamba = True
                    has_rotating = False
                except ImportError:
                    from mlx_lm.models.cache import KVCache
                    has_mamba = False
                    has_rotating = False

            # Reconstruct each layer
            reconstructed_caches = []
            kv_count = 0
            cumulative_count = 0

            for layer_idx in range(num_layers):
                # Collect this layer's data from all blocks
                layer_entries = []
                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        layer_entries.append(block_data[layer_idx])

                if not layer_entries:
                    continue

                # Check the type tag from _extract_block_tensor_slice
                # Collect entries by type, find best cumulative entry
                best_cumulative = None
                kv_slices_keys = []
                kv_slices_values = []
                rotating_kv_slices_keys = []
                rotating_kv_slices_values = []
                rotating_params = None  # (max_size, keep)
                quantized_kv_slices_keys = []  # list of tuples of (data, scales, zeros)
                quantized_kv_slices_values = []
                quantized_meta = None

                cache_list_entries = []  # sub-slice lists from CacheList blocks

                for entry in layer_entries:
                    if not isinstance(entry, (tuple, list)):
                        continue
                    tag = entry[0]
                    if tag == "kv":
                        kv_slices_keys.append(entry[1])
                        kv_slices_values.append(entry[2])
                    elif tag == "quantized_kv":
                        quantized_kv_slices_keys.append(entry[1])  # tuple of 3
                        quantized_kv_slices_values.append(entry[2])  # tuple of 3
                        if len(entry) > 3 and quantized_meta is None:
                            quantized_meta = entry[3]
                    elif tag == "rotating_kv":
                        rotating_kv_slices_keys.append(entry[1])
                        rotating_kv_slices_values.append(entry[2])
                        if len(entry) > 3 and rotating_params is None:
                            rotating_params = (entry[3], entry[4] if len(entry) > 4 else 0)
                    elif tag == "cumulative":
                        best_cumulative = entry  # Last cumulative entry wins
                    elif tag == "cache_list":
                        cache_list_entries.append(entry[1])  # list of sub-slices
                    # "skip" entries are ignored

                if quantized_kv_slices_keys:
                    # QuantizedKVCache: concatenate each component of the tuple
                    # Each entry is a tuple of 3 arrays (data, scales, zeros)
                    num_components = len(quantized_kv_slices_keys[0])
                    concat_keys = tuple(
                        mx.concatenate([s[i] for s in quantized_kv_slices_keys], axis=-2)
                        for i in range(num_components)
                    )
                    concat_values = tuple(
                        mx.concatenate([s[i] for s in quantized_kv_slices_values], axis=-2)
                        for i in range(num_components)
                    )
                    # Materialize concatenated quantized tensors
                    mx.eval(*concat_keys, *concat_values)

                    # Validate head count for quantized cache too
                    first_k = concat_keys[0]
                    if len(first_k.shape) == 4:
                        n_kv = self._get_n_kv_heads()
                        if n_kv > 0 and first_k.shape[1] != n_kv:
                            logger.warning(
                                f"Quantized head count mismatch in layer {layer_idx}: "
                                f"got {first_k.shape[1]}, expected {n_kv} — "
                                f"forcing cache miss"
                            )
                            return None

                    try:
                        from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                        # Parse meta_state for group_size and bits
                        g_size, q_bits = 64, 8
                        if quantized_meta and len(quantized_meta) >= 3:
                            try:
                                _, g_size, q_bits = map(int, quantized_meta[:3])
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Layer {layer_idx}: failed to parse quantized meta "
                                    f"{quantized_meta!r}, using defaults "
                                    f"g_size={g_size} bits={q_bits} — "
                                    f"dequantize may produce wrong values"
                                )
                        elif quantized_kv_slices_keys:
                            # Have quantized data but no valid metadata —
                            # this is a corruption risk (wrong dequantize params)
                            logger.warning(
                                f"Layer {layer_idx}: quantized cache block has no "
                                f"metadata (meta={quantized_meta!r}), "
                                f"using defaults g_size={g_size} bits={q_bits} — "
                                f"possible stale disk cache"
                            )
                        cache = QKVCache(group_size=g_size, bits=q_bits)
                        cache.keys = concat_keys
                        cache.values = concat_values
                        cache.offset = concat_keys[0].shape[-2]
                        reconstructed_caches.append(cache)
                        kv_count += 1
                    except ImportError:
                        logger.warning("Cannot reconstruct QuantizedKVCache: import failed")
                        return None

                elif kv_slices_keys:
                    # Standard KVCache: concatenate slices
                    # Detect dimensionality: 3D (heads, seq, dim) vs 4D (batch, heads, seq, dim)
                    ndim = len(kv_slices_keys[0].shape)
                    seq_axis = 1 if ndim == 3 else 2
                    concat_keys = mx.concatenate(kv_slices_keys, axis=seq_axis)
                    concat_values = mx.concatenate(kv_slices_values, axis=seq_axis)
                    # Materialize lazy concatenation to avoid accumulating a massive
                    # Metal command buffer that can trigger GPU timeout (SIGTERM)
                    mx.eval(concat_keys, concat_values)

                    # Validate head count against model config.
                    # Catches stale blocks with inflated H from BatchKVCache.merge().
                    n_kv = self._get_n_kv_heads()
                    if n_kv > 0:
                        # 4D: (batch, heads, seq, dim), 3D: (heads, seq, dim)
                        head_axis = 1 if ndim == 4 else 0
                        if concat_keys.shape[head_axis] != n_kv:
                            logger.warning(
                                f"Head count mismatch in layer {layer_idx}: "
                                f"got {concat_keys.shape[head_axis]}, expected {n_kv} — "
                                f"forcing cache miss"
                            )
                            return None

                    cache = KVCache()
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = concat_keys.shape[seq_axis]
                    reconstructed_caches.append(cache)
                    kv_count += 1

                elif rotating_kv_slices_keys:
                    # RotatingKVCache: concatenate + restore window params
                    ndim = len(rotating_kv_slices_keys[0].shape)
                    seq_axis = 1 if ndim == 3 else 2
                    concat_keys = mx.concatenate(rotating_kv_slices_keys, axis=seq_axis)
                    concat_values = mx.concatenate(rotating_kv_slices_values, axis=seq_axis)
                    mx.eval(concat_keys, concat_values)

                    if has_rotating and rotating_params:
                        max_size, keep = rotating_params
                        cache = RotatingKVCache(max_size=max_size, keep=keep)
                    else:
                        # Fallback to standard KVCache
                        cache = KVCache()
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = concat_keys.shape[seq_axis]
                    reconstructed_caches.append(cache)
                    kv_count += 1

                elif best_cumulative is not None:
                    # Cumulative cache (MambaCache/ArraysCache): restore full state
                    _, state, meta, class_name = best_cumulative

                    # Try class-specific restoration
                    try:
                        import mlx_lm.models.cache as cache_mod
                        cache_cls = getattr(cache_mod, class_name, None)
                        if cache_cls and hasattr(cache_cls, "from_state"):
                            cache = cache_cls.from_state(state, meta)
                        elif has_mamba and "Mamba" in class_name:
                            cache = MambaCache.from_state(state, meta)
                        elif has_mamba:
                            cache = MambaCache.from_state(state, meta)
                        else:
                            cache = KVCache.from_state(state, meta)
                    except Exception:
                        if has_mamba:
                            cache = MambaCache.from_state(state, meta)
                        else:
                            logger.warning(
                                f"Cannot reconstruct layer {layer_idx} "
                                f"({class_name}): no suitable cache class"
                            )
                            return None

                    reconstructed_caches.append(cache)
                    cumulative_count += 1

                elif cache_list_entries:
                    # CacheList (MoE models): reconstruct each sub-cache
                    # then wrap in CacheList
                    try:
                        from mlx_lm.models.cache import CacheList as CLCache
                    except ImportError:
                        logger.warning("Cannot reconstruct CacheList: import failed")
                        return None

                    # Determine number of sub-caches from first entry
                    num_subs = len(cache_list_entries[0])
                    sub_caches_rebuilt = []
                    for sub_idx in range(num_subs):
                        # Collect this sub-cache's slices across all blocks
                        sub_kv_keys = []
                        sub_kv_vals = []
                        sub_qkv_keys = []  # quantized KV
                        sub_qkv_vals = []
                        sub_qkv_meta = None
                        sub_cumulative = None
                        for block_entry in cache_list_entries:
                            if sub_idx >= len(block_entry):
                                continue
                            sub = block_entry[sub_idx]
                            if not isinstance(sub, (tuple, list)):
                                continue
                            st = sub[0]
                            if st == "kv":
                                sub_kv_keys.append(sub[1])
                                sub_kv_vals.append(sub[2])
                            elif st == "quantized_kv":
                                sub_qkv_keys.append(sub[1])
                                sub_qkv_vals.append(sub[2])
                                if len(sub) > 3 and sub_qkv_meta is None:
                                    sub_qkv_meta = sub[3]
                            elif st == "cumulative":
                                sub_cumulative = sub

                        if sub_qkv_keys:
                            # Quantized sub-cache: concatenate each component
                            n_comp = len(sub_qkv_keys[0])
                            ck = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_keys], axis=-2)
                                for c in range(n_comp)
                            )
                            cv = tuple(
                                mx.concatenate([s[c] for s in sub_qkv_vals], axis=-2)
                                for c in range(n_comp)
                            )
                            mx.eval(*ck, *cv)
                            try:
                                from mlx_lm.models.cache import QuantizedKVCache as QKVCache
                                g_size, q_bits = 64, 8
                                if sub_qkv_meta and len(sub_qkv_meta) >= 3:
                                    try:
                                        _, g_size, q_bits = map(int, sub_qkv_meta[:3])
                                    except (ValueError, TypeError):
                                        pass
                                sc = QKVCache(group_size=g_size, bits=q_bits)
                                sc.keys = ck
                                sc.values = cv
                                sc.offset = ck[0].shape[-2]
                                sub_caches_rebuilt.append(sc)
                            except ImportError:
                                logger.warning(f"CacheList sub {sub_idx}: QuantizedKVCache import failed")
                                return None
                        elif sub_kv_keys:
                            ndim = len(sub_kv_keys[0].shape)
                            seq_axis = 1 if ndim == 3 else 2
                            ck = mx.concatenate(sub_kv_keys, axis=seq_axis)
                            cv = mx.concatenate(sub_kv_vals, axis=seq_axis)
                            mx.eval(ck, cv)

                            # Validate head count in sub-cache
                            if ndim == 4:
                                n_kv = self._get_n_kv_heads()
                                if n_kv > 0 and ck.shape[1] != n_kv:
                                    logger.warning(
                                        f"CacheList sub {sub_idx} head mismatch: "
                                        f"got {ck.shape[1]}, expected {n_kv}"
                                    )
                                    return None

                            sc = KVCache()
                            sc.keys = ck
                            sc.values = cv
                            sc.offset = ck.shape[seq_axis]
                            sub_caches_rebuilt.append(sc)
                        elif sub_cumulative is not None:
                            _, sstate, smeta, scls = sub_cumulative
                            try:
                                import mlx_lm.models.cache as cache_mod
                                scls_obj = getattr(cache_mod, scls, None)
                                if scls_obj and hasattr(scls_obj, "from_state"):
                                    sc = scls_obj.from_state(sstate, smeta)
                                elif has_mamba:
                                    sc = MambaCache.from_state(sstate, smeta)
                                else:
                                    sc = KVCache.from_state(sstate, smeta)
                            except Exception:
                                logger.warning(f"CacheList sub {sub_idx}: reconstruction failed")
                                return None
                            sub_caches_rebuilt.append(sc)
                        else:
                            # Skip-only sub-cache — can't reconstruct
                            return None

                    cl = CLCache.__new__(CLCache)
                    cl.caches = tuple(sub_caches_rebuilt)
                    reconstructed_caches.append(cl)
                    kv_count += 1

            if not reconstructed_caches:
                return None

            if len(reconstructed_caches) != num_layers:
                # Count how many layers were missed and why
                missed_skip_only = 0
                missed_other = 0
                for li in range(num_layers):
                    layer_entries = []
                    for bd in all_block_data:
                        if li < len(bd):
                            layer_entries.append(bd[li])
                    tags = [e[0] if isinstance(e, (tuple, list)) else "?" for e in layer_entries]
                    if all(t == "skip" for t in tags):
                        missed_skip_only += 1
                    elif li >= len(reconstructed_caches) or not any(
                        t in ("kv", "quantized_kv", "rotating_kv", "cumulative", "cache_list")
                        for t in tags
                    ):
                        missed_other += 1

                if missed_other == 0 and missed_skip_only > 0 and reconstructed_caches:
                    # All missing layers are cumulative/SSM with only "skip" entries
                    # (non-last blocks don't store cumulative state).
                    # Return partial reconstruction — the caller's _fix_hybrid_cache
                    # will expand it by inserting fresh SSM caches at missing positions.
                    logger.info(
                        f"Partial hybrid reconstruction: {len(reconstructed_caches)}/{num_layers} layers "
                        f"({missed_skip_only} cumulative layers deferred to _fix_hybrid_cache)"
                    )
                else:
                    logger.warning(
                        f"Reconstructed {len(reconstructed_caches)} layers "
                        f"but expected {num_layers} "
                        f"(skip_only={missed_skip_only}, other={missed_other})"
                    )
                    return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers "
                f"({kv_count} KV + {cumulative_count} cumulative), "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _find_best_prefix_match(
        self,
        tokens: List[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Find best matching prefix in the index."""
        best_match = None
        best_len = 0

        # Try progressively longer prefixes
        for num_blocks in range(1, len(tokens) // self.block_size + 1):
            prefix_len = num_blocks * self.block_size
            if prefix_len > len(tokens):
                break

            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)

            if prefix_hash in self._prefix_index:
                cached_tokens, block_ids = self._prefix_index[prefix_hash]
                if cached_tokens == prefix_tokens and len(cached_tokens) > best_len:
                    # Validate that all referenced blocks still exist
                    valid = all(
                        bid in self.paged_cache.allocated_blocks
                        for bid in block_ids
                    )
                    if valid:
                        best_match = (cached_tokens, block_ids)
                        best_len = len(cached_tokens)
                    else:
                        # Stale entry — remove it
                        del self._prefix_index[prefix_hash]

        return best_match

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Update prefix index with new token sequence."""
        # Index block-aligned prefixes
        for i in range(1, len(block_ids) + 1):
            prefix_len = min(i * self.block_size, len(tokens))
            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)
            self._prefix_index[prefix_hash] = (prefix_tokens, block_ids[:i])

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        paged_stats = self.paged_cache.get_memory_usage()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self.paged_cache.reset_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self._prefix_index.clear()
        self.paged_cache.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)

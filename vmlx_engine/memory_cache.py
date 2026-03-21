# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for vmlx-engine.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- No unnecessary deep copies (MLX arrays are immutable)

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
import gc
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.30  # 30% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails
_PRESSURE_CHECK_INTERVAL = 60.0  # Seconds between memory pressure checks
_PRESSURE_THRESHOLD = 0.20  # Reduce budget when <20% of total RAM available


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


def _estimate_state_memory(state: Any) -> int:
    """Estimate memory of a cache state (tuple/list of tensors)."""
    total = 0
    if isinstance(state, (tuple, list)):
        for item in state:
            if item is not None and hasattr(item, "nbytes"):
                total += item.nbytes
    return total


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of any cache type in bytes.

    Supports KVCache, RotatingKVCache, QuantizedKVCache, MambaCache,
    ArraysCache, and CacheList.

    Args:
        cache: List of layer cache objects.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Extracted state dict (from _extract_cache_states)
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            total_bytes += _estimate_state_memory(layer_cache["state"])

        # KVCache/RotatingKVCache/QuantizedKVCache with keys/values attributes
        # IMPORTANT: Check keys/values BEFORE .state because QuantizedKVCache has
        # both, but .state returns nested tuples that _estimate_state_memory misses.
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            if keys_attr is None:
                pass  # Empty cache
            elif isinstance(keys_attr, tuple):
                # QuantizedKVCache: keys/values are tuples of (data, scales, zeros)
                for arr in keys_attr:
                    if arr is not None and hasattr(arr, "nbytes"):
                        total_bytes += arr.nbytes
                for arr in values_attr:
                    if arr is not None and hasattr(arr, "nbytes"):
                        total_bytes += arr.nbytes
            else:
                if not callable(keys_attr) and hasattr(keys_attr, "nbytes"):
                    total_bytes += keys_attr.nbytes
                if not callable(values_attr) and hasattr(values_attr, "nbytes"):
                    total_bytes += values_attr.nbytes

        # MambaCache/ArraysCache: .cache attribute is a list of arrays
        elif hasattr(layer_cache, "cache") and isinstance(
            getattr(layer_cache, "cache", None), list
        ):
            for arr in layer_cache.cache:
                if arr is not None and hasattr(arr, "nbytes"):
                    total_bytes += arr.nbytes

        # CacheList: recursively estimate sub-caches
        elif hasattr(layer_cache, "caches") and isinstance(
            getattr(layer_cache, "caches", None), (list, tuple)
        ):
            total_bytes += estimate_kv_cache_memory(list(layer_cache.caches))

        # Fallback: cache object with .state property
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            try:
                total_bytes += _estimate_state_memory(layer_cache.state)
            except (TypeError, ValueError):
                pass

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        ttl_minutes: Time-to-live for cache entries in minutes. 0 = no TTL.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    ttl_minutes: float = 0  # 0 = no TTL (unlimited lifetime)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.ttl_minutes < 0:
            raise ValueError(f"ttl_minutes must be >= 0, got {self.ttl_minutes}")

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        # 32GB hard cap — Metal GPU doesn't get 100% of system RAM
        max_cache_bytes = 32 * 1024 * 1024 * 1024

        if self.max_memory_mb is not None:
            # Explicit MB setting capped at 32GB for safety, but no minimum floor
            # (user/test may intentionally set a very small cache)
            limit = self.max_memory_mb * _BYTES_PER_MB
            return min(limit, max_cache_bytes)

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            limit = min(limit, max_cache_bytes)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 4GB available (typical on 8GB Mac), use configured percent
        logger.warning(
            "Could not detect available memory (psutil returned 0). "
            "Assuming 4GB available for cache sizing. Use --cache-memory-mb to set explicitly."
        )
        fallback_available = 4 * 1024 * _BYTES_PER_MB
        return max(min(int(fallback_available * self.max_memory_percent), max_cache_bytes), _MIN_MEMORY_BYTES)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking and timestamp."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int
    last_accessed_at: float = field(default_factory=time.monotonic)

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
            last_accessed_at=time.monotonic(),
        )

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed_at = time.monotonic()


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        All public mutating methods (store, fetch) are protected by a
        threading.Lock. The lock is non-reentrant since these methods
        do not call each other recursively.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()

        # Thread safety: non-reentrant lock (store/fetch/_evict_lru don't
        # recurse into each other, so RLock is unnecessary overhead).
        self._lock = threading.Lock()

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Memory tracking
        self._base_memory_limit = self._config.compute_memory_limit()
        self._max_memory = self._base_memory_limit
        self._current_memory = 0

        # Periodic memory pressure tracking (Issue #53)
        self._last_pressure_check: float = time.monotonic()

        # TTL configuration (seconds, 0 = disabled)
        self._ttl_seconds = self._config.ttl_minutes * 60.0 if self._config.ttl_minutes > 0 else 0.0

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        ttl_str = f"{self._config.ttl_minutes}min" if self._config.ttl_minutes > 0 else "disabled"
        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}, "
            f"ttl={ttl_str}"
        )

    @staticmethod
    def _truncate_cache(
        cache: list[Any], target_len: int
    ) -> list[Any] | None:
        """
        Create a truncated view of a cache for reverse prefix matching.

        When a cached entry covers MORE tokens than requested (e.g., cached
        prompt+output but only prompt requested), we create lightweight
        KVCache wrappers that reference the same underlying tensors but
        with a reduced offset.

        IMPORTANT: This only works for purely positional caches (KVCache).
        Hybrid models that mix KVCache and MambaCache layers CANNOT be
        truncated because MambaCache state is cumulative — it includes
        ALL processed tokens' contributions and cannot be "un-done" to a
        shorter token count. Passing truncated KVCache + full MambaCache
        creates a state mismatch that crashes the model (SIGKILL).

        Args:
            cache: Original cache (list of layer cache objects).
            target_len: Number of token positions to keep.

        Returns:
            New list of cache objects with truncated offsets, or None if
            any layer cannot be truncated (e.g., MambaCache present).
        """
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None

        truncated = []
        for layer_cache in cache:
            if _CacheList is not None and isinstance(layer_cache, _CacheList):
                # MoE CacheList: recurse into sub-caches (each is KVCache)
                truncated_subs = []
                for sc in layer_cache.caches:
                    if (hasattr(sc, "keys") and sc.keys is not None
                            and hasattr(sc, "values") and sc.values is not None):
                        try:
                            from mlx_lm.models.cache import KVCache
                        except ImportError:
                            return None
                        safe_target = min(target_len, sc.keys.shape[-2])
                        new_sc = KVCache()
                        new_sc.keys = sc.keys[..., :safe_target, :]
                        new_sc.values = sc.values[..., :safe_target, :]
                        new_sc.offset = safe_target
                        truncated_subs.append(new_sc)
                    else:
                        return None  # Sub-cache can't be truncated
                truncated.append(_CacheList(*truncated_subs))
            elif hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                k = layer_cache.keys
                if isinstance(k, tuple):
                    # QuantizedKVCache: keys/values are tuples of 3 arrays
                    # (data_uint32, scales, zeros) with seq axis at dim -2
                    try:
                        from mlx_lm.models.cache import QuantizedKVCache
                        v = layer_cache.values
                        safe_target = min(target_len, k[0].shape[-2])
                        new_cache = QuantizedKVCache(
                            group_size=layer_cache.group_size,
                            bits=layer_cache.bits,
                        )
                        new_cache.keys = tuple(t[..., :safe_target, :] for t in k)
                        new_cache.values = tuple(t[..., :safe_target, :] for t in v)
                        new_cache.offset = safe_target
                        truncated.append(new_cache)
                    except ImportError:
                        return None
                else:
                    # Positional cache (KVCache): slice tensors to target_len.
                    # Must slice (not just adjust offset) because data beyond
                    # target_len is from a longer sequence and would corrupt
                    # attention if the model attends to those positions.
                    try:
                        from mlx_lm.models.cache import KVCache
                    except ImportError:
                        return None
                    safe_target = min(target_len, k.shape[-2])
                    new_cache = KVCache()
                    new_cache.keys = k[..., :safe_target, :]
                    new_cache.values = layer_cache.values[..., :safe_target, :]
                    new_cache.offset = safe_target
                    truncated.append(new_cache)
            elif hasattr(layer_cache, "cache") and isinstance(
                getattr(layer_cache, "cache", None), list
            ):
                # Cumulative cache (MambaCache/ArraysCache): CANNOT truncate.
                # State includes all tokens and can't be reduced without
                # re-running the model. Return None to force cache miss.
                return None
            else:
                # Unknown cache type — can't truncate
                return None

        return truncated

    def _check_memory_pressure(self) -> None:
        """
        Periodically re-check system memory and reduce the effective budget
        if available RAM has dropped below the pressure threshold.

        Called from store() at most once every _PRESSURE_CHECK_INTERVAL seconds.
        When available memory is less than _PRESSURE_THRESHOLD (20%) of total RAM,
        the effective budget is temporarily reduced to half of available memory,
        preventing the cache from causing swap pressure. The budget is restored
        to the original limit once memory pressure subsides.
        """
        now = time.monotonic()
        if now - self._last_pressure_check < _PRESSURE_CHECK_INTERVAL:
            return
        self._last_pressure_check = now

        try:
            import psutil
            vm = psutil.virtual_memory()
        except (ImportError, Exception):
            return

        available_fraction = vm.available / vm.total if vm.total > 0 else 1.0

        if available_fraction < _PRESSURE_THRESHOLD:
            # Under pressure: shrink budget to half of currently available RAM,
            # but never below the minimum floor.
            reduced = max(vm.available // 2, _MIN_MEMORY_BYTES)
            if reduced < self._max_memory:
                logger.warning(
                    f"Memory pressure detected: {available_fraction:.0%} of RAM available. "
                    f"Reducing cache budget from {self._max_memory / _BYTES_PER_MB:.0f}MB "
                    f"to {reduced / _BYTES_PER_MB:.0f}MB"
                )
                self._max_memory = reduced
                self._stats.max_memory_bytes = self._max_memory
        else:
            # Pressure relieved: restore original budget
            if self._max_memory < self._base_memory_limit:
                logger.info(
                    f"Memory pressure relieved ({available_fraction:.0%} available). "
                    f"Restoring cache budget to {self._base_memory_limit / _BYTES_PER_MB:.0f}MB"
                )
                self._max_memory = self._base_memory_limit
                self._stats.max_memory_bytes = self._max_memory

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for:
        1. Exact matches
        2. Forward prefix matches (cached key is shorter, is a prefix of request)
        3. Reverse prefix matches (cached key is longer, request is a prefix of it)

        For reverse prefix matches, the cache is truncated to match the
        requested token count. This handles the common case where the cache
        stores prompt+output tokens but the next request sends the same prompt.

        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            return None, tokens

        with self._lock:
            # Evict expired entries before lookup
            if self._ttl_seconds > 0:
                self._evict_expired()

            tokens_key = tuple(tokens)

            # Check for exact match
            if tokens_key in self._entries:
                entry = self._entries[tokens_key]
                # Move to end (most recently used) and update access time
                self._entries.move_to_end(tokens_key)
                entry.touch()
                self._stats.hits += 1
                self._stats.tokens_saved += len(tokens)
                # Return reference directly - MLX arrays are immutable
                return entry.cache, []

            # Prefix scan: O(n) over all entries (Issue #62).
            #
            # This is an intentional design trade-off. A trie would give O(k)
            # lookup (k = token length), but adds significant complexity for
            # insertion, eviction, and memory tracking. The flat OrderedDict
            # approach is simpler, easier to reason about for correctness, and
            # performs well in practice because:
            #   - max_entries defaults to 1000 (config.max_entries), bounding n
            #   - Prefix comparison short-circuits on first mismatch (tuple ==)
            #   - The exact-match dict lookup above handles the common case in O(1)
            # If profiling shows this scan as a bottleneck with large entry counts,
            # consider replacing OrderedDict with a trie + LRU linked list.
            best_forward_match: _CacheEntry | None = None
            best_forward_length = 0
            best_reverse_match: _CacheEntry | None = None
            best_reverse_length = 0

            for cached_key, entry in self._entries.items():
                cached_len = len(cached_key)

                if cached_len < len(tokens):
                    # Forward: cached sequence is a prefix of requested tokens
                    if (
                        cached_len > best_forward_length
                        and tokens_key[:cached_len] == cached_key
                    ):
                        best_forward_match = entry
                        best_forward_length = cached_len

                elif cached_len > len(tokens):
                    # Reverse: requested tokens are a prefix of cached sequence
                    # This handles repeated prompts where cache stores prompt+output
                    if (
                        len(tokens) > best_reverse_length
                        and cached_key[: len(tokens)] == tokens_key
                    ):
                        best_reverse_match = entry
                        best_reverse_length = len(tokens)

            # Prefer forward match (exact prefix reuse, no truncation needed)
            if best_forward_match is not None:
                self._entries.move_to_end(best_forward_match.tokens)
                best_forward_match.touch()
                self._stats.hits += 1
                self._stats.tokens_saved += best_forward_length
                remaining = tokens[best_forward_length:]
                return best_forward_match.cache, remaining

            # Fall back to reverse match with cache truncation
            if best_reverse_match is not None:
                truncated = self._truncate_cache(best_reverse_match.cache, len(tokens))
                if truncated is not None:
                    self._entries.move_to_end(best_reverse_match.tokens)
                    best_reverse_match.touch()
                    self._stats.hits += 1
                    self._stats.tokens_saved += len(tokens)
                    return truncated, []

            self._stats.misses += 1
            return None, tokens

    def store(self, tokens: list[int], cache: list[Any]) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Every 60 seconds, re-checks system memory pressure and temporarily
        reduces the effective budget if available RAM drops below 20% of total.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        with self._lock:
            # Periodic memory pressure adaptation (Issue #53)
            self._check_memory_pressure()

            tokens_key = tuple(tokens)

            # If already cached, just update LRU order
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

            # Evict expired entries first to free space before LRU eviction
            if self._ttl_seconds > 0:
                self._evict_expired()

            # Create entry and estimate memory
            entry = _CacheEntry.create(tokens, cache)

            # Check if single entry exceeds limit (95% of total cache budget)
            _max_entry_bytes = int(self._max_memory * 0.95)
            if entry.memory_bytes > _max_entry_bytes:
                logger.info(
                    f"Cache entry too large: "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                    f"(limit {_max_entry_bytes / _BYTES_PER_MB:.0f}MB) - skipping"
                )
                return False

            # Evict until we have room
            evicted_any = False
            while (
                self._current_memory + entry.memory_bytes > self._max_memory
                or len(self._entries) >= self._config.max_entries
            ) and self._entries:
                self._evict_lru()
                evicted_any = True

            # After batch eviction, run GC to release Python references.
            # We do NOT call mx.clear_memory_cache() here because it can
            # interfere with in-flight GPU operations during active prefills.
            # Metal will reclaim memory naturally when tensors are deallocated.
            if evicted_any:
                gc.collect()

            # Store entry
            self._entries[tokens_key] = entry
            self._current_memory += entry.memory_bytes
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

            logger.debug(
                f"Stored cache: {len(tokens)} tokens, "
                f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
            )

            return True

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        Caller MUST hold self._lock. This is a private method only called
        from store() which acquires the lock before entering the eviction loop.
        """
        if not self._entries:
            return

        # popitem(last=False) removes oldest entry (FIFO order = LRU)
        tokens_key, entry = self._entries.popitem(last=False)
        freed_bytes = entry.memory_bytes

        # Delete the entry's cache reference to allow Metal to free GPU memory
        if hasattr(entry, 'cache'):
            del entry.cache
        del entry

        self._current_memory = max(0, self._current_memory - freed_bytes)
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Evicted cache: {len(tokens_key)} tokens, "
            f"freed {freed_bytes / _BYTES_PER_MB:.2f}MB"
        )

    def _evict_expired(self) -> int:
        """Evict entries that have exceeded their TTL.

        Returns:
            Number of entries evicted.
        """
        if self._ttl_seconds <= 0:
            return 0

        now = time.monotonic()
        expired_keys = []

        for tokens_key, entry in self._entries.items():
            if now - entry.last_accessed_at > self._ttl_seconds:
                expired_keys.append(tokens_key)

        for key in expired_keys:
            entry = self._entries.pop(key)
            freed_bytes = entry.memory_bytes
            if hasattr(entry, 'cache'):
                del entry.cache
            del entry
            self._current_memory = max(0, self._current_memory - freed_bytes)
            self._stats.evictions += 1

        if expired_keys:
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
            gc.collect()
            logger.info(
                f"TTL eviction: removed {len(expired_keys)} expired entries, "
                f"{len(self._entries)} entries remaining"
            )

        return len(expired_keys)

    def remove(self, tokens: list[int]) -> bool:
        """Remove a specific cache entry by token key.

        Returns True if removed, False if not found.
        """
        with self._lock:
            tokens_key = tuple(tokens)
            if tokens_key not in self._entries:
                return False
            entry = self._entries.pop(tokens_key)
            self._current_memory = max(0, self._current_memory - entry.memory_bytes)
            self._stats.evictions += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
            return True

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters while preserving cached entries."""
        self._stats = CacheStats(
            max_memory_bytes=self._max_memory,
            current_memory_bytes=self._current_memory,
            entry_count=len(self._entries),
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._current_memory = 0
        self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

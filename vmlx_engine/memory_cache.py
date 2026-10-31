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

from .utils.cache_extent import logical_truncate_target

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB

# Last-resort ceiling, used only when the OS cannot tell us the real one.
_FALLBACK_CACHE_CEILING_BYTES = 32 * 1024 * 1024 * 1024

# Floor for the residency-adjusted ceiling. Subtracting live Metal residency
# from the working-set limit can legitimately approach zero on a machine whose
# model nearly fills the GPU budget, and a zero ceiling would silently turn
# caching off rather than merely shrink it.
_MIN_CACHE_CEILING_BYTES = 512 * _BYTES_PER_MB


def resolve_paged_resident_policy(config, disk_only: bool):
    """The ONE rule for how a cache-memory setting becomes a paged-RAM policy.

    Returns ``(max_resident_bytes, frugal)``.

    Why this is a function and not two inline copies: ``PagedCacheManager``
    documents ``max_resident_bytes == 0`` as the legacy UNBOUNDED sentinel,
    while ``MemoryAwarePrefixCache`` reads the same 0 as "store nothing". So an
    explicit ``--cache-memory-mb 0`` -- a user asking for the smallest possible
    resident cache -- resolved to a 0 budget and bought the LARGEST one: no
    byte ceiling, and every gated ``enforce_byte_budget`` call site skipping.

    Routing that request to the frugal policy instead was first written only in
    the text scheduler. MEASURED on the box afterwards: Gemma 4 (a VLM, so the
    MLLM scheduler) still reported ``paged_frugal=False,
    ram_mirror_policy=resident`` on /health with ``--cache-memory-mb 0``. The
    unit tests all passed; the fix was simply inert on half the engine. Hence
    one function, called by both.

    Only an EXPLICIT zero counts. ``None`` means "auto-detect from percent",
    and disk-only already has its own zero-budget meaning.
    """
    if disk_only:
        return 0, False
    budget = MemoryCacheConfig(
        max_memory_mb=config.cache_memory_mb,
        max_memory_percent=config.cache_memory_percent,
    ).compute_memory_limit()
    explicit_zero = (
        config.cache_memory_mb is not None and int(config.cache_memory_mb) == 0
    )
    return budget, explicit_zero


def _resolve_cache_memory_ceiling_bytes() -> int:
    """The largest cache this machine may hold, from the OS rather than a guess.

    This was a flat 32 GB with the comment "Metal GPU doesn't get 100% of system
    RAM" — true in general, but the number was invented and it silently reduced
    an explicit ``--cache-memory-mb``. On a 128 GB machine whose owner has raised
    ``iogpu.wired_limit_mb`` it also contradicts a standing rule this repo
    already records elsewhere (cli.py `_install_jangtq_wired_limit_from_sysctl`):
    the user's sysctl is authoritative, and a hardcoded default that clips below
    it is a bug, not a safety feature.

    MLX reports the same bound the loader already trusts. Use it; fall back to
    the old constant only when it cannot be read (non-Darwin, MLX missing).

    SUBTRACT WHAT METAL ALREADY HOLDS. The MLX number is the working-set limit
    for EVERYTHING on the GPU — model weights included — not a cache allowance.
    Returning it whole let an explicit ``--cache-memory-mb 65536`` authorize a
    64 GB resident cache on top of an 80 GB resident model against a ~115 GB
    ceiling, and this value becomes the paged cache's ``max_resident_bytes``
    (scheduler.py, ``_paged_resident_budget``), which is the default tier. The
    pressure check that would claw it back only runs every
    ``_PRESSURE_CHECK_INTERVAL`` seconds, so the over-commit has a real window
    in which to OOM.

    ``active`` is read once, when the cache is constructed. If the model is
    already resident by then this is the true headroom; if it is not, ``active``
    is near zero and the result degrades to the previous behaviour rather than
    to something smaller — so this never clips harder than before.
    """
    try:
        import mlx.core as mx

        from .utils.memory_limits import get_effective_metal_working_set_bytes

        active_bytes, max_ws_bytes = get_effective_metal_working_set_bytes(mx)
        if max_ws_bytes and max_ws_bytes > 0:
            headroom = int(max_ws_bytes) - max(0, int(active_bytes or 0))
            # Never return a ceiling so small it cannot hold anything; a bad or
            # transiently huge `active` reading must not disable caching.
            return max(headroom, _MIN_CACHE_CEILING_BYTES)
    except Exception:
        pass
    return _FALLBACK_CACHE_CEILING_BYTES
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails
_PRESSURE_CHECK_INTERVAL = 60.0  # Seconds between memory pressure checks
_PRESSURE_THRESHOLD = 0.20  # Reduce budget when <20% of total RAM available

# Cache type priority for LRU eviction (F1 backport from PrefixCacheManager).
# Lower-priority types are evicted first: assistant → user → system. System
# entries stay pinned and evicted last so cross-session shared system prompts
# survive across users. Mirrors mlx-lm 0.31.2 LRUPromptCache.CacheOrder.
_CACHE_TYPE_PRIORITY: tuple[str, ...] = ("assistant", "user", "system")


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


def _estimate_state_memory(state: Any, _seen: set[int] | None = None) -> int:
    """Estimate memory of a cache state, recursing through nested containers.

    This was a FLAT one-level scan, so anything whose elements are themselves
    containers counted as ZERO bytes:

      * QuantizedKVCache extracted state — tuples of (packed, scales, biases)
      * CacheList sub-caches — a list of per-layer state dicts
      * DSV4 transport dicts

    Every --kv-cache-quantization entry and every MoE-hybrid entry was therefore
    invisible to the byte budgets that call this, which is how a cache bounded
    "by bytes" could still grow without limit.

    Arrays are deduplicated by id() so aliased views (common after a slice or a
    COW copy) are not counted twice — the sibling estimator in paged_cache.py
    lacks that and over-reports.
    """
    if _seen is None:
        _seen = set()
    if state is None:
        return 0
    if hasattr(state, "nbytes"):
        marker = id(state)
        if marker in _seen:
            return 0
        _seen.add(marker)
        try:
            return int(state.nbytes)
        except Exception:  # noqa: BLE001
            return 0
    if isinstance(state, dict):
        return sum(_estimate_state_memory(v, _seen) for v in state.values())
    if isinstance(state, (tuple, list)):
        return sum(_estimate_state_memory(v, _seen) for v in state)
    return 0


def _common_prefix_len(a: tuple, b: tuple) -> int:
    """Length of the common prefix of two token tuples (binary search over C-speed slice comparisons)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def clone_dsv41_cache_at(cache: list[Any], target_len: int) -> list[Any] | None:
    """Independent copy of a DeepSeek-V4.1 layer-cache list that sits exactly at ``target_len`` tokens.

    DeepseekV41Cache layers share one sequence state (position + Engram history) and cannot be trimmed,
    so a stored prefix is valid only at the exact boundary it was captured at. Returns None (no store /
    clean miss) for a mixed list, a list at another position, or a copy that fails."""
    if not cache or not all(type(layer).__name__ == "DeepseekV41Cache" for layer in cache):
        return None
    offsets = {int(getattr(layer, "offset", -1)) for layer in cache}
    if offsets != {int(target_len)}:
        logger.info(
            "DSV4.1 cache at offset(s) %s, not the %d-token boundary; not reusing it (cannot be trimmed)",
            sorted(offsets), int(target_len),
        )
        return None
    try:
        from jang_tools.dsv4_1.mlx_model import clone_cache_list

        return clone_cache_list(list(cache))
    except Exception as exc:
        logger.warning("DSV4.1 cache copy failed (%s); not reusing it", exc)
        return None


def estimate_kv_cache_memory(cache: list[Any], *, resident: bool = True) -> int:
    """
    Estimate retained cache bytes, or logical snapshot bytes with resident=False.

    Supports KVCache, RotatingKVCache, QuantizedKVCache, MambaCache,
    ArraysCache, and CacheList.

    Args:
        cache: List of layer cache objects.
        resident: Honor explicit native backing-allocation accounting. False
            describes independent logical copies/serialization, not live views.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Explicit opt-in contract, not arbitrary .nbytes from an unknown
        # family. Native capacity/slack must not inflate SSD snapshot admission.
        resident_bytes = getattr(layer_cache, "resident_nbytes", None) if resident else None
        if isinstance(resident_bytes, int) and resident_bytes >= 0:
            total_bytes += resident_bytes
            continue
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
                # Empty KVCache layers have keys=None and no resident buffers.
                # Compressed TurboQuantKVCache layers also clear keys/values
                # after packing, but expose their real resident footprint via
                # nbytes. Count that or cache budgets under-report TQ hits as 0.
                nbytes = getattr(layer_cache, "nbytes", None)
                if isinstance(nbytes, (int, float)) and nbytes > 0:
                    total_bytes += int(nbytes)
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

            # MiniMaxM3SparseCache carries an extra append-only idx_keys buffer
            # ([B, 1, S, index_dim]) beyond the inherited KV. Count it or M3
            # cache budgets under-report the resident MSA footprint.
            idx_keys_attr = getattr(layer_cache, "idx_keys", None)
            if idx_keys_attr is not None and hasattr(idx_keys_attr, "nbytes"):
                total_bytes += idx_keys_attr.nbytes
            # Consumer-owned derived state (retained pooled QSA keys).
            derived_nbytes = getattr(layer_cache, "derived_nbytes", 0)
            if isinstance(derived_nbytes, int):
                total_bytes += derived_nbytes

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
            total_bytes += estimate_kv_cache_memory(
                list(layer_cache.caches), resident=resident
            )

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
        max_cache_bytes = _resolve_cache_memory_ceiling_bytes()

        if self.max_memory_mb is not None:
            # No minimum floor — a user or test may intentionally set a very
            # small cache. But an explicit value must never be reduced in
            # silence: the user asked for a number and the engine would have
            # gone on reporting a different one.
            limit = self.max_memory_mb * _BYTES_PER_MB
            if limit > max_cache_bytes:
                logger.warning(
                    "Requested cache memory %.1f GB exceeds this machine's "
                    "Metal working-set ceiling of %.1f GB; using the ceiling. "
                    "Raise it with `sudo sysctl iogpu.wired_limit_mb=<MB>`.",
                    limit / 1e9,
                    max_cache_bytes / 1e9,
                )
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
    cache_type: str = "assistant"  # "system" | "user" | "assistant" — F1 backport

    @classmethod
    def create(
        cls,
        tokens: list[int],
        cache: list[Any],
        cache_type: str = "assistant",
    ) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
            last_accessed_at=time.monotonic(),
            cache_type=cache_type if cache_type in _CACHE_TYPE_PRIORITY else "assistant",
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
        model_path: str | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
            model_path: Optional model directory path. Used to derive a stable
                content-hash key (replaces id(model) — F6) and feeds the loader
                fingerprint so divergent loader configs never share entries.
        """
        # Lazy import to avoid circular dependency with prefix_cache.
        try:
            from .prefix_cache import compute_model_cache_key as _ck
            self._model_id: str | int = _ck(model, model_path=model_path)
        except Exception:
            self._model_id = id(model)
        self.last_fetch_kind: str | None = None  # exact | forward | partial | reverse (the last hit)
        self._config = config or MemoryCacheConfig()

        # Thread safety: non-reentrant lock (store/fetch/_evict_lru don't
        # recurse into each other, so RLock is unnecessary overhead).
        self._lock = threading.Lock()
        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        # Unsalted entries retain the historical tuple(tokens) key shape.
        # Salted/discriminated entries use ("__extra__", marker, tuple(tokens))
        # so token-prefix matching remains separate from the non-token key.
        self._entries: OrderedDict[tuple[Any, ...], _CacheEntry] = OrderedDict()

        # Per-cache-type LRU buckets for priority eviction (F1 backport).
        # Each bucket holds the same keys as `_entries` but partitioned by
        # cache_type. Eviction pops from the lowest-priority non-empty bucket
        # first so system entries survive longest. Mirrors PrefixCacheManager.
        self._lru_by_type: dict[str, OrderedDict[tuple[Any, ...], bool]] = {
            t: OrderedDict() for t in _CACHE_TYPE_PRIORITY
        }
        # Per-type byte counters
        self._bytes_by_type: dict[str, int] = {t: 0 for t in _CACHE_TYPE_PRIORITY}

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

    def _clone_cache_for_fetch(self, cache: list, length: int) -> list | None:
        """Return an isolated clone of a cache hit, or None to force a clean miss.

        Issue #198 (1C): _truncate_cache rebuilds each layer as a fresh object with
        copied arrays -> independent offset and (on next update) independent buffer,
        so generation cannot contaminate the stored entry. Handled: KVCache /
        QuantizedKVCache (incl. CacheList of those), RotatingKVCache,
        MiniMaxM3SparseCache, TurboQuantKVCache, GLM typed KDA/MLA state, and
        cumulative SSM/conv state (MambaCache / ArraysCache) — the last two only
        because this method is by contract called at the entry's FULL cached
        length, where the state needs copying rather than reducing.

        This NEVER returns the stored reference. Decode writes through every layer
        it is handed, so sharing the cached object corrupts it (F11's sibling F20:
        a hybrid KV+SSM entry's offsets walked 18 -> 19 after one decode step, and
        later hits replayed the polluted prefix). If a layer cannot be isolated we
        return None and the caller recomputes.
        """
        try:
            from mlx_lm.models.cache import KVCache, QuantizedKVCache
        except Exception:
            return None
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except Exception:
            _CacheList = ()

        def _safe(layer) -> bool:
            if type(layer).__name__ in {"Glm5KDACache", "Glm5MLACache"}:
                return True
            # DeepSeek-V4.1: _truncate_cache copies the whole layer list as one unit (shared sequence
            # state) at the stored boundary, so the hit never aliases the stored entry.
            if type(layer).__name__ == "DeepseekV41Cache":
                return True
            # openPangu exact-boundary composite: safe only because
            # _truncate_cache has a dedicated full-length clone that preserves
            # inner KV, DSA indexer, rotating metadata, and all conv states.
            if type(layer).__name__ == "OpenPanguV2LayerCache":
                return True
            # MiniMax-M3 MSA cache (KVCache subclass + idx_keys lane): _truncate_cache
            # now rebuilds it AS a fresh isolated MiniMaxM3SparseCache with
            # keys/values/idx_keys all sliced, so cloning fully isolates the stored
            # entry. WITHOUT this, the hit returned the stored object by reference and
            # decode appended into the cached prefix -> polluted reuse -> repetition
            # loop on the next hit (the real multi-turn / repeated-prompt bug).
            if type(layer).__name__ == "MiniMaxM3SparseCache":
                return True
            # RotatingKVCache (gemma-4 / mixed-SWA): _clone_cache_for_fetch is
            # only ever called at FULL cached length (exact / forward match — the
            # reverse/truncating path calls _truncate_cache directly), so we can
            # faithfully isolate-clone it without any lossy reduction. WITHOUT
            # this it fell through to the stored-reference return below and decode
            # rotated/appended into the shared buffer → polluted reuse → gemma-4
            # divergence + dropped tool calls after ~3 identical requests.
            if type(layer).__name__ == "RotatingKVCache":
                return True
            # TurboQuantKVCache (M2.7 all-layer, Qwen3.6 every-4th, gemma-4 mixed-SWA
            # full-attention slots). It is NOT a KVCache subclass, so it used to fall
            # through to the stored-reference return below — and because TQ is
            # monotonic-growth (update_and_fetch does `self.offset += num_new` and
            # writes into self.keys), decode then appended straight into the CACHED
            # entry. Proven live on gemma-4 + VMLX_SWA_TQ: the stored 18-token entry's
            # offset walked 18 -> 34 -> 50 -> 66 across repeated hits, so a later
            # "capital of France?" hit replayed a polluted prefix and answered
            # "Berlin." _truncate_cache now rebuilds it as a fresh isolated cache.
            if type(layer).__name__ == "TurboQuantKVCache":
                return True
            if isinstance(layer, (KVCache, QuantizedKVCache)):
                return type(layer).__name__ in ("KVCache", "QuantizedKVCache")
            if _CacheList and isinstance(layer, _CacheList):
                subs = getattr(layer, "caches", None) or []
                return bool(subs) and all(
                    type(sc).__name__ in ("KVCache", "QuantizedKVCache")
                    for sc in subs
                )
            # Cumulative SSM/conv state (MambaCache / ArraysCache: a `.cache` list of
            # arrays). It cannot be REDUCED, but we are at the full cached length, so
            # _truncate_cache copies its arrays instead. This is what makes a hybrid
            # KVCache+SSM layer list cloneable at all; previously one such layer made
            # _safe() reject the WHOLE list and the caller shared the stored entry.
            if isinstance(getattr(layer, "cache", None), list):
                return True
            return False

        if not cache:
            return None
        if not all(_safe(layer) for layer in cache):
            # No cache layer is safe to share: decode writes through every one of
            # them. If we cannot isolate a layer we must take a clean miss and
            # recompute rather than hand back the stored object (F20).
            logger.warning(
                "prefix-cache hit not isolatable (layers=%s); forcing a clean miss",
                [type(layer).__name__ for layer in cache],
            )
            return None
        return self._clone_on_stream_owner(cache, length)

    def set_clone_executor(self, executor: Any, worker_name_prefix: str | None = None) -> None:
        """Register the single-thread worker executor that owns the MLX stream.

        Optional: without it the clone runs inline, which is what every caller
        already on the worker wants. The name prefix is read off the executor
        (ThreadPoolExecutor names its threads "<prefix>_N") so this works for
        both the LLM scheduler's "llm-worker" and the MLLM scheduler's
        "mllm-worker" without either having to declare it.
        """
        self._clone_executor = executor
        if worker_name_prefix is None:
            worker_name_prefix = getattr(executor, "_thread_name_prefix", "") or "llm-worker"
        self._clone_worker_prefix = worker_name_prefix

    def _clone_on_stream_owner(
        self, cache: list[Any], length: int, allow_cumulative_clone: bool = True
    ) -> list[Any] | None:
        """Build the isolate-clone on the thread that owns the MLX stream.

        _truncate_cache slices the stored KV arrays, and MLX binds each new array
        to a completion event on the stream of the *creating* thread. fetch() is
        reached from add_request() on the API/event-loop thread, so the clone came
        out tied to MainThread's Stream(gpu, 0). The llm-worker then evaluated
        logits computed from it and could not resolve that stream:

            RuntimeError: There is no Stream(gpu, 0) in current thread.

        mx.eval() on the API thread does not sever this — the event stays with the
        producing stream. The engine already defers stored-cache q4/q8 dequant to
        the worker for exactly this reason (see _schedule_waiting); the clone
        itself was still born on the API thread.

        Single-sequence families surface it first: SingleBatchGenerator samples
        straight from the restored cache, so the throw lands in _start_request and
        the request is aborted into an empty 200 — Laguna returned 0 tokens on
        every prefix-cache hit. The batched generators re-home cache tensors onto
        their own stream before use, which is why gemma-4 never hit it.

        Today fetch() is only reached from add_request() on the API thread, but
        clone inline whenever we are already on the worker: submitting to a
        single-thread executor from inside that same thread would deadlock.
        """
        executor = getattr(self, "_clone_executor", None)
        prefix = getattr(self, "_clone_worker_prefix", "llm-worker")
        if executor is None or threading.current_thread().name.startswith(prefix):
            cloned = self._truncate_cache(cache, length, allow_cumulative_clone)
        else:
            try:
                cloned = executor.submit(
                    self._truncate_cache, cache, length, allow_cumulative_clone
                ).result()
            except Exception as exc:
                # A clone we cannot build on the owning worker is a cache miss, not
                # a reason to improvise. Cloning inline here would bind the arrays to
                # the API thread's stream and resurrect F11 (empty 200 on every hit);
                # returning the stored entry would resurrect F20 (polluted prefix).
                # Recomputing is the only correct answer.
                logger.warning(
                    "worker-side prefix-cache clone failed (%s); forcing a clean miss",
                    exc,
                )
                return None
        if cloned is not None:
            self._materialize_cloned_cache(cloned)
        return cloned

    @staticmethod
    def _materialize_cloned_cache(cloned: list[Any]) -> None:
        """Evaluate the clone's arrays on the thread that just created them.

        Leaves concrete data with no pending graph, so the slice ops cannot be
        re-scheduled onto another thread's stream later. Only sound when called
        from _clone_on_stream_owner, i.e. on the stream owner.
        """
        try:
            import mlx.core as mx
        except Exception:
            return

        arrays: list[Any] = []

        def _collect(obj) -> None:
            if type(obj).__name__ == "OpenPanguV2LayerCache":
                _collect(getattr(obj, "kv", None))
                indexer = getattr(obj, "indexer_kv", None)
                if indexer is not None:
                    _collect(indexer)
                for state in getattr(obj, "conv_states", ()):
                    if isinstance(state, mx.array):
                        arrays.append(state)
                return
            subs = getattr(obj, "caches", None)
            if subs:
                for sub in subs:
                    _collect(sub)
                return
            # "cache" carries the cumulative SSM/conv state arrays (ArraysCache /
            # MambaCache); the rest are the positional KV lanes.
            for attr in ("keys", "values", "idx_keys", "biases", "scales", "cache"):
                val = getattr(obj, attr, None)
                if isinstance(val, mx.array):
                    arrays.append(val)
                elif isinstance(val, (tuple, list)):
                    arrays.extend(v for v in val if isinstance(v, mx.array))

        for layer in cloned:
            if layer is not None:
                _collect(layer)
        if not arrays:
            return
        try:
            mx.eval(*arrays)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cache clone materialization skipped: %s", exc)

    @staticmethod
    def _truncate_cache(
        cache: list[Any], target_len: int, allow_cumulative_clone: bool = False
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
            allow_cumulative_clone: Set only by _clone_cache_for_fetch, which by
                contract is called at the FULL cached length. Cumulative SSM/conv
                state cannot be REDUCED, but at full length it needs no reduction —
                a faithful array copy isolates it. Without this the hybrid layer
                list fell through to the stored-reference return and decode mutated
                the cached entry in place (F20).

        Returns:
            New list of cache objects with truncated offsets, or None if
            any layer cannot be truncated (e.g., MambaCache present and this
            is a true reduction).
        """
        try:
            from mlx_lm.models.cache import CacheList as _CacheList
        except ImportError:
            _CacheList = None

        def _copy_positional_slice(value):
            """Materialize a KV slice so cache hits are worker-stream safe."""
            try:
                import mlx.core as mx
                import numpy as np

                if isinstance(value, np.ndarray):
                    return value.copy()

                orig_dtype = getattr(value, "dtype", None)
                if orig_dtype == mx.bfloat16:
                    copied = mx.array(np.array(value.astype(mx.float32))).astype(
                        mx.bfloat16
                    )
                else:
                    copied = mx.array(np.array(value))
                return copied
            except Exception:
                return value

        def _clone_cumulative_layer(src):
            """Copy a cumulative SSM/conv layer (MambaCache / ArraysCache).

            The state arrays are the whole cache — there is no offset to slice — so
            a faithful copy is only valid at the full cached length, which is the
            only place the caller enables this. Shallow-copy preserves the concrete
            class and its scalar attributes; replacing `.cache` with copied arrays
            severs the one piece of mutable state decode writes through.
            """
            import copy as _copy

            arrays = getattr(src, "cache", None)
            if not isinstance(arrays, list):
                return None
            try:
                clone = _copy.copy(src)
                clone.cache = [
                    None if a is None else _copy_positional_slice(a) for a in arrays
                ]
            except Exception as exc:
                logger.warning("cumulative cache clone failed (%s); forcing miss", exc)
                return None
            return clone

        def _clone_turboquant_layer(src, length):
            """Rebuild a TurboQuantKVCache as a fresh, independent cache.

            _get_full_cache() is pure: it returns the decoded float view the source
            itself attends over (joined buffer if it has compressed, else the float
            buffer), so slicing it is faithful even when part of the source is packed.
            The clone starts in the pre-compress state with empty packed buffers and
            will re-compress on its own once it passes compress_after; keeping the
            source's encoders or packed arrays would re-share mutable state, which is
            the whole thing we are here to prevent.

            Reset to the clean single-sequence contract (_idx=None, no padding) — the
            live-cache validator whitelists _idx=None for genuine TQ classes (F10).
            """
            try:
                from jang_tools.turboquant.cache import TurboQuantKVCache
            except Exception:
                return None
            full_k, full_v = src._get_full_cache()
            if full_k is None or full_v is None or length > full_k.shape[2]:
                return None
            clone = TurboQuantKVCache(
                key_dim=src.key_dim,
                value_dim=src.value_dim,
                key_bits=src.key_bits,
                value_bits=src.value_bits,
                seed=src._seed,
                compress_after=src.compress_after,
                sink_tokens=src.sink_tokens,
            )
            clone.keys = _copy_positional_slice(full_k[..., :length, :])
            clone.values = _copy_positional_slice(full_v[..., :length, :])
            clone.offset = int(length)
            return clone

        if any(type(layer).__name__ == "DeepseekV41Cache" for layer in cache):
            # DeepSeek-V4.1: one sequence state (position, Engram history) shared by every layer, and not
            # trimmable (sliding window + compressor groups are path state). Only a whole-list copy at the
            # stored boundary is valid; any reduction is a clean miss.
            return clone_dsv41_cache_at(cache, target_len) if allow_cumulative_clone else None

        truncated = []
        for layer_cache in cache:
            if type(layer_cache).__name__ in {
                "Glm5KDACache",
                "Glm5MLACache",
            }:
                # KDA recurrent/conv state and MLA packed DSA history form one
                # typed, path-dependent boundary. They can be isolated at the
                # stored length but never truncated to a shorter prefix.
                if not allow_cumulative_clone:
                    return None
                if (
                    type(layer_cache).__name__ == "Glm5MLACache"
                    and int(getattr(layer_cache, "offset", 0) or 0) != target_len
                ):
                    return None
                try:
                    from .models.glm5_next.glm5_next import (
                        clone_glm5_next_layer_cache,
                    )

                    cloned = clone_glm5_next_layer_cache(
                        layer_cache,
                        # GLM typed wrappers may share an exact stored boundary.
                        # Compact-v2 marks those physical lanes shared and
                        # detaches each one on its first later write; legacy-v1
                        # still replaces arrays. Avoiding an eager second cache
                        # image is required beside a roughly 95GB model.
                        copy_fn=lambda value: value,
                    )
                except Exception as exc:
                    logger.warning(
                        "GLM typed cache clone failed (%s); forcing miss",
                        exc,
                    )
                    return None
                truncated.append(cloned)
                continue
            if type(layer_cache).__name__ == "OpenPanguV2LayerCache":
                # This class is cumulative/path-dependent.  Only a faithful copy
                # at the stored logical boundary is valid; reverse-prefix
                # truncation must become a clean miss.
                current_len = int(getattr(layer_cache, "offset", 0) or 0)
                if not allow_cumulative_clone or target_len != current_len:
                    return None
                try:
                    from .models.openpangu_v2.cache import (
                        clone_openpangu_layer_cache,
                    )

                    cloned = clone_openpangu_layer_cache(
                        layer_cache,
                        copy_fn=_copy_positional_slice,
                    )
                except Exception as exc:
                    logger.warning(
                        "openPangu composite cache clone failed (%s); forcing miss",
                        exc,
                    )
                    return None
                truncated.append(cloned)
                continue
            if type(layer_cache).__name__ == "TurboQuantKVCache":
                tq_clone = _clone_turboquant_layer(layer_cache, target_len)
                if tq_clone is None:
                    return None
                truncated.append(tq_clone)
                continue
            # MiniMax-M3 MSA cache: a KVCache subclass with an EXTRA append-only
            # idx_keys lane (Lightning-Indexer). The generic KVCache branch below
            # would downcast it to a plain KVCache and DROP idx_keys — then the
            # indexer's `isinstance(cache, MiniMaxM3SparseCache)` guard fails, it
            # skips its idx history, block selection degenerates, and reuse loops.
            # Rebuild AS a MiniMaxM3SparseCache with keys/values/idx_keys all
            # sliced to target_len so SSD/prefix reuse stays correct.
            if type(layer_cache).__name__ == "MiniMaxM3SparseCache":
                try:
                    from .models.minimax_m3.cache import clone_minimax_m3_sparse
                except Exception:
                    return None

                def _copy_m3_slice(value):
                    try:
                        import mlx.core as mx
                        import numpy as np

                        orig_dtype = getattr(value, "dtype", None)
                        if orig_dtype == mx.bfloat16:
                            copied = mx.array(np.array(value.astype(mx.float32))).astype(
                                mx.bfloat16
                            )
                        else:
                            copied = mx.array(np.array(value))
                        return copied
                    except Exception:
                        return value

                new_cache = clone_minimax_m3_sparse(
                    layer_cache,
                    target_len,
                    copy_fn=_copy_m3_slice,
                    require_idx_keys=True,
                )
                if new_cache is None:
                    return None
                truncated.append(new_cache)
                continue
            # RotatingKVCache (gemma-4 sliding + full mixed-SWA): faithfully
            # ISOLATE-clone into a fresh RotatingKVCache so a cache hit cannot be
            # mutated in place by decode (issue: gemma-4 prefix-cache pollution —
            # identical greedy requests diverged after ~3 reuses because the hit
            # returned the stored object by reference). We only reach here at full
            # cached length (see _safe / _clone_cache_for_fetch), so no lossy
            # rotating reduction is needed. If a TRUE reduction is ever requested
            # (target < logical offset — only via the direct reverse-match
            # _truncate_cache call), rotating truncation is lossy → return None to
            # force a clean cache miss instead of downcasting to a plain KVCache.
            if type(layer_cache).__name__ == "RotatingKVCache":
                try:
                    from mlx_lm.models.cache import RotatingKVCache
                except ImportError:
                    return None
                keys = getattr(layer_cache, "keys", None)
                values = getattr(layer_cache, "values", None)
                if keys is None or values is None:
                    return None
                offset = int(getattr(layer_cache, "offset", 0) or 0)
                if target_len < offset:
                    return None  # lossy rotating truncation unsupported
                new_cache = RotatingKVCache(
                    max_size=layer_cache.max_size,
                    keep=layer_cache.keep,
                )
                # full-buffer isolated copy (breaks aliasing with the stored entry)
                new_cache.keys = _copy_positional_slice(keys)
                new_cache.values = _copy_positional_slice(values)
                new_cache.offset = offset
                new_cache._idx = int(getattr(layer_cache, "_idx", offset) or 0)
                if hasattr(layer_cache, "step"):
                    new_cache.step = layer_cache.step
                truncated.append(new_cache)
                continue
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
                        safe_target = logical_truncate_target(sc, target_len, sc.keys.shape[-2])
                        new_sc = KVCache()
                        new_sc.keys = _copy_positional_slice(
                            sc.keys[..., :safe_target, :]
                        )
                        new_sc.values = _copy_positional_slice(
                            sc.values[..., :safe_target, :]
                        )
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
                        safe_target = logical_truncate_target(layer_cache, target_len, k[0].shape[-2])
                        new_cache = QuantizedKVCache(
                            group_size=layer_cache.group_size,
                            bits=layer_cache.bits,
                        )
                        # Tuple slicing alone leaves lazy views tied to the
                        # stream that produced the stored q4/q8 entry. The next
                        # request may then fail on its first sample with
                        # ``no Stream(gpu, 0)``. Materialize independent arrays
                        # just like the float KV branch below.
                        new_cache.keys = tuple(
                            _copy_positional_slice(t[..., :safe_target, :]) for t in k
                        )
                        new_cache.values = tuple(
                            _copy_positional_slice(t[..., :safe_target, :]) for t in v
                        )
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
                    safe_target = logical_truncate_target(layer_cache, target_len, k.shape[-2])
                    new_cache = KVCache()
                    new_cache.keys = _copy_positional_slice(
                        k[..., :safe_target, :]
                    )
                    new_cache.values = _copy_positional_slice(
                        layer_cache.values[..., :safe_target, :]
                    )
                    new_cache.offset = safe_target
                    truncated.append(new_cache)
            elif hasattr(layer_cache, "cache") and isinstance(
                getattr(layer_cache, "cache", None), list
            ):
                # Cumulative cache (MambaCache/ArraysCache): CANNOT truncate.
                # State includes all tokens and can't be reduced without
                # re-running the model. Return None to force cache miss.
                #
                # At FULL cached length no reduction is being asked for, so a
                # faithful copy of the state arrays is both correct and necessary:
                # without it the whole hybrid layer list fell through to the
                # stored-reference return in _clone_cache_for_fetch and decode
                # advanced the CACHED entry's offsets in place (F20 — proven on a
                # KV+ArraysCache list: stored offsets walked 18 -> 19 after one
                # decode step, so later hits replayed a polluted prefix).
                if not allow_cumulative_clone:
                    return None
                cumulative_clone = _clone_cumulative_layer(layer_cache)
                if cumulative_clone is None:
                    return None
                truncated.append(cumulative_clone)
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

    @staticmethod
    def _entry_key(
        tokens: list[int] | tuple[int, ...],
        cache_extra_keys: Any = None,
    ) -> tuple[Any, ...]:
        from .cache_key import canonical_cache_extra_marker

        tokens_key = tuple(tokens)
        marker = canonical_cache_extra_marker(cache_extra_keys)
        if marker is None:
            return tokens_key
        return ("__extra__", marker, tokens_key)

    @staticmethod
    def _entry_key_parts(key: tuple[Any, ...]) -> tuple[str | None, tuple[int, ...]]:
        if len(key) == 3 and key[0] == "__extra__" and isinstance(key[1], str):
            return key[1], tuple(key[2])
        return None, tuple(key)

    def fetch(
        self,
        tokens: list[int],
        cache_extra_keys: Any = None,
    ) -> tuple[list[Any] | None, list[int]]:
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

        # issue #233: the match phase runs under self._lock, but the CLONE runs
        # with the lock RELEASED. _clone_cache_for_fetch hops to the llm-worker
        # executor (F11: the clone must be born on the MLX stream owner) and
        # blocks on .result(); the same worker's step() -> _cleanup_finished()
        # -> store() blocks on self._lock — holding the lock across the hop is
        # a lock inversion that deadlocks in the DEFAULT config (and likewise
        # against remove()). Releasing it is safe because the single-thread
        # executor — not the mutex — is what serializes the clone against
        # decode, and _truncate_cache is read-only on the source arrays. We
        # snapshot entry.cache (the LIST) under the lock: evictors do
        # `del entry.cache`, so touching the entry attribute after release is
        # an intermittent AttributeError that reads as a flaky miss.
        _hit_entry = None  # the _CacheEntry OBJECT — identity token for revalidation
        _hit_key: tuple[Any, ...] | None = None
        _hit_cache: list | None = None  # snapshot of entry.cache
        _hit_kind = ""  # "exact" | "forward"
        _hit_len = 0
        _rev_entry = None  # reverse-match fallback (original fallthrough semantics)
        _rev_cache: list | None = None

        with self._lock:
            # Evict expired entries before lookup
            if self._ttl_seconds > 0:
                self._evict_expired()

            from .cache_key import canonical_cache_extra_marker

            tokens_key = tuple(tokens)
            extra_marker = canonical_cache_extra_marker(cache_extra_keys)
            entry_key = self._entry_key(tokens_key, cache_extra_keys)

            # Check for exact match
            if entry_key in self._entries:
                entry = self._entries[entry_key]
                # issue #198 (1C): return an isolated clone so generation
                # advancing offset/appending KV in place cannot contaminate
                # the stored entry for future hits on the same prefix. A clone we
                # cannot build is a miss — only count the hit once it succeeds.
                _hit_entry, _hit_key, _hit_cache = entry, entry_key, entry.cache
                _hit_kind, _hit_len = "exact", len(tokens)

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
            # Partial match: the stored key and the request share a prefix and then differ (same system prompt and
            # tools, same document with another question, an edited last message). The entry is reusable up to the
            # common prefix when its cache can be cut there (_truncate_cache; families that cannot be trimmed, e.g.
            # DeepSeek-V4.1, return None and fall back to the other matches).
            best_partial_match: _CacheEntry | None = None
            best_partial_length = 0

            _scan_entries = () if _hit_entry is not None else self._entries.items()
            for stored_key, entry in _scan_entries:
                cached_marker, cached_key = self._entry_key_parts(stored_key)
                if cached_marker != extra_marker:
                    continue
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
                        continue
                if (
                    cached_len
                    and tokens_key[0] == cached_key[0]
                    and min(cached_len, len(tokens)) > max(best_partial_length, best_forward_length)
                ):
                    common = _common_prefix_len(tokens_key, cached_key)
                    if common < cached_len and common < len(tokens) and common > max(
                        best_partial_length, best_forward_length
                    ):
                        best_partial_match = entry
                        best_partial_length = common

            # Prefer forward match (exact prefix reuse, no truncation needed)
            if _hit_entry is None and best_forward_match is not None:
                # issue #198 (1C): isolated clone (see exact-match path). This is the
                # entry's FULL cached length (best_forward_length == cached_len), which
                # is what lets cumulative SSM layers be copied rather than reduced.
                _hit_entry = best_forward_match
                _hit_key = self._entry_key(
                    best_forward_match.tokens,
                    cache_extra_keys,
                )
                _hit_cache = best_forward_match.cache
                _hit_kind, _hit_len = "forward", best_forward_length

            # Reverse match with cache truncation — snapshot as the fallback so a
            # failed forward clone still falls through to it (original semantics).
            if best_reverse_match is not None:
                _rev_entry = best_reverse_match
                _rev_cache = best_reverse_match.cache
            _part_entry = None
            _part_cache: list | None = None
            if (
                best_partial_match is not None
                and best_partial_length > (_hit_len if _hit_entry is not None else 0)
                and _rev_entry is None
            ):
                _part_entry = best_partial_match
                _part_cache = best_partial_match.cache

            if _hit_entry is None and _rev_entry is None and _part_entry is None:
                self._stats.misses += 1
                return None, tokens

        # ── clone phase: LOCK RELEASED (issue #233, see header comment) ──
        cloned: list | None = None
        remaining: list[int] = []
        self.last_fetch_kind = None
        if _part_entry is not None:
            # Cut on the stream owner (F11, see _clone_on_stream_owner) and never copy cumulative state: SSM / conv
            # / typed path-dependent layers cannot be reduced to a shorter prefix, so such an entry is no partial hit.
            cloned = self._clone_on_stream_owner(_part_cache, best_partial_length, False)
            if cloned is not None:
                _hit_entry, _hit_key = (
                    _part_entry,
                    self._entry_key(_part_entry.tokens, cache_extra_keys),
                )
                _hit_len = best_partial_length
                remaining = tokens[best_partial_length:]
                self.last_fetch_kind = "partial"
        if cloned is None and _hit_entry is not None and self.last_fetch_kind is None:
            cloned = self._clone_cache_for_fetch(_hit_cache, _hit_len)
            if cloned is not None and _hit_kind == "forward":
                remaining = tokens[_hit_len:]
            if cloned is not None:
                self.last_fetch_kind = _hit_kind
        if cloned is None and _rev_entry is not None:
            # Fall back to reverse match with cache truncation
            cloned = self._truncate_cache(_rev_cache, len(tokens))
            if cloned is not None:
                _hit_entry, _hit_key = (
                    _rev_entry,
                    self._entry_key(_rev_entry.tokens, cache_extra_keys),
                )
                _hit_len = len(tokens)
                remaining = []
                self.last_fetch_kind = "reverse"

        with self._lock:
            if cloned is None:
                self._stats.misses += 1
                return None, tokens
            # Revalidate on the ENTRY OBJECT'S identity, not just the key: a
            # concurrent evict+re-store swaps the object behind the same key,
            # and _touch_type_lru INSERTS absent keys (which would desync
            # _lru_by_type for an entry that no longer exists). On identity
            # failure the clone is still real data built from a then-valid
            # snapshot — return it and count the hit; just skip LRU touching.
            if _hit_key is not None and self._entries.get(_hit_key) is _hit_entry:
                self._entries.move_to_end(_hit_key)
                self._touch_type_lru(_hit_key, _hit_entry.cache_type)
                _hit_entry.touch()
            self._stats.hits += 1
            self._stats.tokens_saved += _hit_len
            return cloned, remaining

    def store(
        self,
        tokens: list[int],
        cache: list[Any],
        cache_type: str = "assistant",
        cache_extra_keys: Any = None,
    ) -> bool:
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
            cache_type: One of "system" | "user" | "assistant". Controls LRU
                eviction priority — system entries are pinned and evicted last
                so shared system prompts persist across users/sessions
                (F1 backport from PrefixCacheManager).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        with self._lock:
            # Periodic memory pressure adaptation (Issue #53)
            self._check_memory_pressure()

            tokens_key = self._entry_key(tokens, cache_extra_keys)
            norm_type = cache_type if cache_type in _CACHE_TYPE_PRIORITY else "assistant"

            # If already cached, refresh LRU order and possibly upgrade type.
            # Type upgrades go assistant → user → system; never downgrade.
            if tokens_key in self._entries:
                old = self._entries[tokens_key]
                self._entries.move_to_end(tokens_key)
                old_priority = _CACHE_TYPE_PRIORITY.index(old.cache_type)
                new_priority = _CACHE_TYPE_PRIORITY.index(norm_type)
                if new_priority > old_priority:
                    # Upgrade: move bytes between buckets
                    self._bytes_by_type[old.cache_type] = max(
                        0, self._bytes_by_type[old.cache_type] - old.memory_bytes
                    )
                    self._bytes_by_type[norm_type] += old.memory_bytes
                    if tokens_key in self._lru_by_type[old.cache_type]:
                        del self._lru_by_type[old.cache_type][tokens_key]
                    old.cache_type = norm_type
                self._lru_by_type[old.cache_type][tokens_key] = True
                self._lru_by_type[old.cache_type].move_to_end(tokens_key)
                return True

            # Evict expired entries first to free space before LRU eviction
            if self._ttl_seconds > 0:
                self._evict_expired()

            # Create entry and estimate memory
            entry = _CacheEntry.create(tokens, cache, cache_type=norm_type)

            # Check if single entry exceeds limit (95% of total cache budget)
            _max_entry_bytes = int(self._max_memory * 0.95)
            if entry.memory_bytes > _max_entry_bytes:
                logger.info(
                    f"Cache entry too large: "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                    f"(limit {_max_entry_bytes / _BYTES_PER_MB:.0f}MB) - skipping"
                )
                return False

            # Evict until we have room — pop lowest-priority bucket first.
            evicted_any = False
            while (
                self._current_memory + entry.memory_bytes > self._max_memory
                or len(self._entries) >= self._config.max_entries
            ) and self._entries:
                if not self._evict_lru():
                    break
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
            self._bytes_by_type[entry.cache_type] += entry.memory_bytes
            self._lru_by_type[entry.cache_type][tokens_key] = True
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

            logger.debug(
                f"Stored cache ({entry.cache_type}): {len(tokens)} tokens, "
                f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
            )

            return True

    def _evict_lru(self) -> bool:
        """Evict one entry, preferring the lowest-priority non-empty bucket
        (assistant → user → system). System entries are pinned and only
        evicted when no lower-priority entries remain (F1 backport).

        Caller MUST hold self._lock. Returns True if an entry was evicted,
        False if all buckets were empty.
        """
        if not self._entries:
            return False

        target_key: tuple[Any, ...] | None = None
        target_type: str | None = None
        for t in _CACHE_TYPE_PRIORITY:
            d = self._lru_by_type[t]
            if d:
                target_key, _ = d.popitem(last=False)
                target_type = t
                break

        if target_key is None:
            # Buckets out of sync with _entries — fall back to flat order
            target_key, _ = next(iter(self._entries.items()))
            for t in _CACHE_TYPE_PRIORITY:
                if target_key in self._lru_by_type[t]:
                    del self._lru_by_type[t][target_key]
                    target_type = t
                    break

        entry = self._entries.pop(target_key, None)
        if entry is None:
            return False
        freed_bytes = entry.memory_bytes
        ent_type = entry.cache_type
        token_count = len(entry.tokens)

        # Delete the entry's cache reference to allow Metal to free GPU memory
        if hasattr(entry, 'cache'):
            del entry.cache
        del entry

        self._current_memory = max(0, self._current_memory - freed_bytes)
        self._bytes_by_type[ent_type] = max(
            0, self._bytes_by_type[ent_type] - freed_bytes
        )
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Evicted cache ({ent_type}): {token_count} tokens, "
            f"freed {freed_bytes / _BYTES_PER_MB:.2f}MB"
        )
        return True

    def _touch_type_lru(self, tokens_key: tuple[Any, ...], cache_type: str) -> None:
        """Mark an entry as most-recently-used in its per-type bucket. O(1)."""
        d = self._lru_by_type.get(cache_type)
        if d is None:
            d = self._lru_by_type["assistant"]
        if tokens_key in d:
            d.move_to_end(tokens_key)
        else:
            d[tokens_key] = True

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
            ent_type = entry.cache_type
            if key in self._lru_by_type.get(ent_type, {}):
                del self._lru_by_type[ent_type][key]
            if hasattr(entry, 'cache'):
                del entry.cache
            del entry
            self._current_memory = max(0, self._current_memory - freed_bytes)
            self._bytes_by_type[ent_type] = max(
                0, self._bytes_by_type[ent_type] - freed_bytes
            )
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

    def remove(self, tokens: list[int], cache_extra_keys: Any = None) -> bool:
        """Remove a specific cache entry by token key.

        Returns True if removed, False if not found.
        """
        with self._lock:
            tokens_key = self._entry_key(tokens, cache_extra_keys)
            if tokens_key not in self._entries:
                return False
            entry = self._entries.pop(tokens_key)
            ent_type = entry.cache_type
            if tokens_key in self._lru_by_type.get(ent_type, {}):
                del self._lru_by_type[ent_type][tokens_key]
            self._current_memory = max(0, self._current_memory - entry.memory_bytes)
            self._bytes_by_type[ent_type] = max(
                0, self._bytes_by_type[ent_type] - entry.memory_bytes
            )
            self._stats.evictions += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
            return True

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters while preserving cached entries.

        Locked: called worker-side from _ensure_batch_generator while fetch()
        may be iterating/mutating state under the same structures (issue #233
        review — the unlocked form raced fetch()'s stats writes).
        """
        with self._lock:
            self._stats = CacheStats(
                max_memory_bytes=self._max_memory,
                current_memory_bytes=self._current_memory,
                entry_count=len(self._entries),
            )

    def clear(self) -> None:
        """Clear all cached entries.

        Locked: called worker-side from _ensure_batch_generator while fetch()
        iterates self._entries.items() under the lock — the unlocked clear()
        mutated the OrderedDict mid-iteration (live RuntimeError; issue #233
        review). Now that fetch() releases the lock across its clone, the
        interleavings that reach this got more frequent, not less.
        """
        with self._lock:
            self._entries.clear()
            for d in self._lru_by_type.values():
                d.clear()
            for t in self._bytes_by_type:
                self._bytes_by_type[t] = 0
            self._current_memory = 0
            self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics with per-type byte / entry breakdown."""
        d = self._stats.to_dict()
        d["nbytes"] = self._current_memory
        d["nbytes_by_type"] = {
            t: self._bytes_by_type[t] for t in _CACHE_TYPE_PRIORITY
        }
        d["entries_by_type"] = {
            t: len(self._lru_by_type[t]) for t in _CACHE_TYPE_PRIORITY
        }
        d["max_bytes"] = self._max_memory
        return d

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

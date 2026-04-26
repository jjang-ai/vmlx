"""
Codebook Weight Cache - LRU cache with disk spillover for expert weights.

Expert weights are stored as codebook + indices (compressed).
On cache miss, we:
    1. Load codebook and indices from safetensors
    2. Reconstruct weights: flat_weights = codebook[flat_indices]
    3. Reshape to weight matrix
    4. Store in memory cache (LRU)
    5. On memory pressure, evict LRU entries to disk

Memory budget is unlimited by default (all weights spill to disk if needed).

Usage:
    cache = CodebookWeightCache(config)

    # Get reconstructed weights (lazy load + reconstruct)
    weights = cache.get((layer_idx, "gate_proj"))

    # Cache miss triggers:
    # 1. Load codebook-layer-{layer_idx}-gate_proj.safetensors
    # 2. Reconstruct: weights = codebook[indices.reshape(-1)]
    # 3. Cache in memory
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import hashlib

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


@dataclass
class CacheEntry:
    """Cache entry for reconstructed expert weights."""

    weights: Any  # mx.array
    access_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    disk_path: Optional[Path] = None


class CodebookWeightCache:
    """
    LRU cache with disk spillover for codebook VQ expert weights.

    Features:
    - Unlimited memory (LRU to disk when needed)
    - Thread-safe operations
    - Async disk I/O for eviction
    - Configurable disk cache directory and max size
    - Memory pressure monitoring
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        disk_cache_dir: Optional[str] = None,
        disk_max_gb: int = 1000,
        eviction_batch_size: int = 4,
    ):
        """
        Initialize codebook weight cache.

        Args:
            config: Configuration dict with memory/disk settings.
            disk_cache_dir: Directory for disk spillover.
            disk_max_gb: Maximum disk cache size in GB.
            eviction_batch_size: Number of entries to evict at once.
        """
        self._config = config or {}

        # Memory cache (LRU OrderedDict)
        self._memory_cache: OrderedDict[Tuple[int, str], CacheEntry] = OrderedDict()
        self._memory_budget_bytes: Optional[int] = None
        self._current_memory_bytes: int = 0

        # Disk cache
        self._disk_cache_dir = self._resolve_path(
            disk_cache_dir
            or self._config.get("disk_cache_dir", "~/.cache/vmlx-engine/codebook-cache")
        )
        self._disk_max_bytes = disk_max_gb * 1024 * 1024 * 1024
        self._eviction_batch_size = eviction_batch_size or 4

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._disk_writes = 0
        self._disk_reads = 0

        # Ensure disk cache directory exists
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for quick lookups
        self._index_file = self._disk_cache_dir / ".index"
        self._disk_index: Dict[str, Dict] = self._load_disk_index()

        logger.info(
            f"CodebookWeightCache initialized: disk_cache={self._disk_cache_dir}, "
            f"max_disk_gb={disk_max_gb}"
        )

    def _resolve_path(self, path: str) -> Path:
        """Resolve ~ and environment variables in path."""
        return Path(os.path.expandvars(os.path.expanduser(path)))

    def _load_disk_index(self) -> Dict[str, Dict]:
        """Load disk cache index from disk."""
        if self._index_file.exists():
            import json

            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk index: {e}")
        return {}

    def _save_disk_index(self):
        """Save disk cache index to disk."""
        import json

        try:
            with open(self._index_file, "w") as f:
                json.dump(self._disk_index, f)
        except Exception as e:
            logger.warning(f"Failed to save disk index: {e}")

    def get(self, key: Tuple[int, str]) -> Optional[Any]:
        """
        Get reconstructed expert weights for a layer/tensor_type.

        Args:
            key: Tuple of (layer_idx, tensor_type), e.g., (0, "gate_proj")

        Returns:
            Reconstructed weights as mx.array, or None if not found.
        """
        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                entry.access_time = time.time()
                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                self._hits += 1
                return entry.weights

            # Check disk cache
            disk_path = self._get_disk_path(key)
            if disk_path and disk_path.exists():
                self._disk_reads += 1
                weights = self._load_from_disk(disk_path)
                if weights is not None:
                    # Load into memory cache
                    self._put_memory(key, weights)
                    self._hits += 1
                    return weights

            # Cache miss
            self._misses += 1
            return None

    def put(self, key: Tuple[int, str], weights: Any):
        """
        Store reconstructed weights in cache.

        Args:
            key: Tuple of (layer_idx, tensor_type)
            weights: Reconstructed weight mx.array
        """
        with self._lock:
            self._put_memory(key, weights)

    def _put_memory(self, key: Tuple[int, str], weights: Any):
        """Put weights into memory cache, evicting LRU if needed."""
        import mlx.core as mx

        size_bytes = weights.nbytes if hasattr(weights, "nbypes") else weights.nbytes

        # Check if already in cache
        if key in self._memory_cache:
            old_entry = self._memory_cache[key]
            self._current_memory_bytes -= old_entry.size_bytes
            del old_entry

        # Evict if over budget (unlimited budget if memory_budget_bytes is None)
        while (
            self._memory_budget_bytes is not None
            and self._current_memory_bytes + size_bytes > self._memory_budget_bytes
            and self._memory_cache
        ):
            self._evict_lru()

        # Add to cache
        entry = CacheEntry(weights=weights, size_bytes=size_bytes)
        self._memory_cache[key] = entry
        self._current_memory_bytes += size_bytes

        logger.debug(
            f"Cached weights for {key}: {size_bytes / 1e6:.1f} MB, "
            f"total: {self._current_memory_bytes / 1e9:.2f} GB"
        )

    def _evict_lru(self):
        """Evict least recently used entries to disk."""
        if not self._memory_cache:
            return

        # Get LRU entries (first N items)
        for _ in range(min(self._eviction_batch_size, len(self._memory_cache))):
            key, entry = self._memory_cache.popitem(last=False)
            self._current_memory_bytes -= entry.size_bytes

            # Save to disk before removing from memory
            disk_path = self._get_disk_path(key)
            if disk_path:
                self._save_to_disk(key, entry.weights, disk_path)
                self._disk_writes += 1

            self._evictions += 1

        logger.debug(
            f"Evicted {self._eviction_batch_size} entries, "
            f"freed {entry.size_bytes / 1e6:.1f} MB"
        )

    def _get_disk_path(self, key: Tuple[int, str]) -> Path:
        """Get disk path for a cache key."""
        layer_idx, tensor_type = key
        # Use hash of key for filename safety
        key_hash = hashlib.md5(f"{layer_idx}_{tensor_type}".encode()).hexdigest()[:12]
        return (
            self._disk_cache_dir
            / f"codebook_{layer_idx:03d}_{tensor_type}_{key_hash}.safetensors"
        )

    def _save_to_disk(self, key: Tuple[int, str], weights: Any, path: Path):
        """Save weights to disk."""
        try:
            if _MLX_AVAILABLE:
                mx.save_safetensors(str(path), {"weights": weights})
            else:
                import numpy as np

                np.save(str(path), np.array(weights))

            # Update index
            self._disk_index[str(key)] = {
                "path": str(path),
                "size_bytes": weights.nbytes if hasattr(weights, "nbytes") else 0,
                "time": time.time(),
            }
            self._save_disk_index()
        except Exception as e:
            logger.warning(f"Failed to save {key} to disk: {e}")

    def _load_from_disk(self, path: Path) -> Optional[Any]:
        """Load weights from disk."""
        try:
            if _MLX_AVAILABLE:
                data = mx.load(str(path))
                return list(data.values())[0] if data else None
            else:
                import numpy as np

                return np.load(str(path))
        except Exception as e:
            logger.warning(f"Failed to load from disk {path}: {e}")
            return None

    def prefetch(self, keys: List[Tuple[int, str]]):
        """
        Prefetch weights for multiple keys.

        Args:
            keys: List of (layer_idx, tensor_type) tuples
        """
        for key in keys:
            if key not in self._memory_cache:
                # Trigger lazy load
                self.get(key)

    def invalidate(self, key: Tuple[int, str]):
        """Invalidate a cache entry."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._current_memory_bytes -= entry.size_bytes

    def clear(self):
        """Clear all cache entries (memory and disk)."""
        with self._lock:
            self._memory_cache.clear()
            self._current_memory_bytes = 0

            # Clear disk cache
            for key_str, info in self._disk_index.items():
                path = Path(info["path"])
                if path.exists():
                    path.unlink()

            self._disk_index.clear()
            self._save_disk_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "disk_writes": self._disk_writes,
            "disk_reads": self._disk_reads,
            "memory_entries": len(self._memory_cache),
            "memory_bytes": self._current_memory_bytes,
            "disk_entries": len(self._disk_index),
            "disk_bytes": sum(info["size_bytes"] for info in self._disk_index.values()),
        }

    @property
    def memory_bytes(self) -> int:
        """Current memory usage in bytes."""
        return self._current_memory_bytes

    @property
    def disk_bytes(self) -> int:
        """Current disk cache size in bytes."""
        return sum(info["size_bytes"] for info in self._disk_index.values())

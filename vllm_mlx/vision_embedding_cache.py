# SPDX-License-Identifier: Apache-2.0
"""
Vision Embedding Cache for MLLM continuous batching.

Caches processed image tensors (prepare_inputs output) to avoid redundant
computation when the same images are processed multiple times.

Performance Impact:
- Without cache: ~0.5-1s per image for preprocessing
- With cache hit: ~0.01s (50-100x speedup for repeated images)
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class VisionCacheStats:
    """Statistics for vision cache performance."""

    pixel_cache_hits: int = 0
    pixel_cache_misses: int = 0
    total_time_saved: float = 0.0
    total_images_processed: int = 0

    @property
    def pixel_hit_rate(self) -> float:
        total = self.pixel_cache_hits + self.pixel_cache_misses
        return self.pixel_cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_cache_hits": self.pixel_cache_hits,
            "pixel_cache_misses": self.pixel_cache_misses,
            "pixel_hit_rate": self.pixel_hit_rate,
            "total_time_saved": self.total_time_saved,
            "total_images_processed": self.total_images_processed,
        }


@dataclass
class PixelCacheEntry:
    """Cached pixel values from prepare_inputs."""

    pixel_values: mx.array
    input_ids: mx.array
    attention_mask: Optional[mx.array]
    image_grid_thw: Optional[mx.array]
    extra_kwargs: Dict[str, Any]
    processing_time: float = 0.0


def compute_image_hash(image_path: str) -> str:
    """
    Compute hash of image content.

    For files: hash the actual content
    For URLs/base64: hash the string
    """
    try:
        path = Path(image_path)
        if path.exists() and path.is_file():
            # Hash full file content for correctness
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        else:
            # Hash the string (URL or base64)
            return hashlib.sha256(image_path.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]


def compute_images_hash(images: List[str]) -> str:
    """Compute combined hash for multiple images."""
    if not images:
        return "no_images"
    hashes = [compute_image_hash(img) for img in images]
    return hashlib.sha256("_".join(hashes).encode()).hexdigest()[:16]


class VisionEmbeddingCache:
    """
    Pixel values cache for vision processing in MLLM.

    Caches output of prepare_inputs() (pixel_values, input_ids, etc.)
    keyed by hash(images) + hash(prompt). Saves image loading, resizing,
    and normalization time (~0.5-1s per image).

    Example:
        >>> cache = VisionEmbeddingCache(max_pixel_entries=50)
        >>>
        >>> # First request - cache miss
        >>> pixel_entry = cache.get_pixel_cache(images, prompt)
        >>> if pixel_entry is None:
        ...     # Process images...
        ...     cache.set_pixel_cache(images, prompt, pixel_values, ...)
        >>>
        >>> # Second request with same image - cache hit!
        >>> pixel_entry = cache.get_pixel_cache(images, prompt)  # Returns cached data
    """

    def __init__(
        self,
        max_pixel_entries: int = 100,
        enabled: bool = True,
        **kwargs,
    ):
        """
        Initialize the vision embedding cache.

        Args:
            max_pixel_entries: Max entries in pixel cache (LRU eviction)
            enabled: Whether caching is enabled
        """
        self.max_pixel_entries = max_pixel_entries
        self.enabled = enabled

        # LRU cache using OrderedDict
        self._pixel_cache: OrderedDict[str, PixelCacheEntry] = OrderedDict()

        self.stats = VisionCacheStats()

    def _make_key(self, images: List[str], prompt: str) -> str:
        """Create cache key from images and prompt."""
        img_hash = compute_images_hash(images)
        # Use shorter prompt hash for cache key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return f"{img_hash}_{prompt_hash}"

    def get_pixel_cache(
        self,
        images: List[str],
        prompt: str,
    ) -> Optional[PixelCacheEntry]:
        """
        Get cached pixel values for images+prompt.

        Returns:
            PixelCacheEntry if found, None otherwise
        """
        if not self.enabled or not images:
            return None

        key = self._make_key(images, prompt)

        if key in self._pixel_cache:
            # Move to end (most recently used)
            entry = self._pixel_cache.pop(key)
            self._pixel_cache[key] = entry

            self.stats.pixel_cache_hits += 1
            self.stats.total_time_saved += entry.processing_time
            logger.debug(
                f"Pixel cache hit: {key[:20]}... (saved {entry.processing_time:.2f}s)"
            )
            return entry

        self.stats.pixel_cache_misses += 1
        return None

    def set_pixel_cache(
        self,
        images: List[str],
        prompt: str,
        pixel_values: mx.array,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0,
    ) -> None:
        """Store pixel values in cache."""
        if not self.enabled or not images:
            return

        key = self._make_key(images, prompt)

        # Evict oldest if at capacity
        while len(self._pixel_cache) >= self.max_pixel_entries:
            oldest_key = next(iter(self._pixel_cache))
            del self._pixel_cache[oldest_key]
            logger.debug(f"Pixel cache evicted: {oldest_key[:20]}...")

        entry = PixelCacheEntry(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            extra_kwargs=extra_kwargs or {},
            processing_time=processing_time,
        )
        self._pixel_cache[key] = entry
        self.stats.total_images_processed += len(images)
        logger.debug(f"Pixel cache stored: {key[:20]}...")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["pixel_cache_size"] = len(self._pixel_cache)
        return stats

    def clear(self) -> None:
        """Clear cache and reset stats."""
        self._pixel_cache.clear()
        self.stats = VisionCacheStats()

    def __repr__(self) -> str:
        return (
            f"<VisionEmbeddingCache "
            f"pixel={len(self._pixel_cache)}/{self.max_pixel_entries}>"
        )

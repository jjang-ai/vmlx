# SPDX-License-Identifier: Apache-2.0
"""
Disk-based prompt cache persistence for vllm-mlx.

Saves and loads pre-computed KV/Mamba caches to disk using mlx-lm's
safetensors-based save_prompt_cache/load_prompt_cache. A SQLite index
maps token hash → cache file for fast lookup.

This acts as an L2 cache: the in-memory prefix cache is L1 (fast, limited),
and the disk cache is L2 (slower I/O, much larger capacity). On cache miss
in L1, the scheduler checks L2 before doing a full prefill.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _hash_tokens(tokens: List[int]) -> str:
    """Create a stable hash of a token sequence for cache indexing."""
    # Use SHA-256 of the token list serialized as compact JSON
    data = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class DiskCacheManager:
    """
    Persistent disk-based cache for KV/Mamba states.

    Stores prompt caches as .safetensors files indexed by a SQLite database.
    Compatible with all mlx-lm cache types (KVCache, QuantizedKVCache,
    RotatingKVCache, ArraysCache/MambaCache, CacheList).

    Args:
        cache_dir: Directory to store cache files. Created if it doesn't exist.
        max_size_gb: Maximum total cache size in GB. Oldest entries are evicted
            when this limit is exceeded. 0 = unlimited.
    """

    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024) if max_size_gb > 0 else 0

        # SQLite index for fast token hash → file lookup
        self._db_path = self.cache_dir / "cache_index.db"
        self._init_db()

        # Stats
        self.hits = 0
        self.misses = 0
        self.stores = 0

        logger.info(
            f"Disk cache initialized: dir={self.cache_dir}, "
            f"max_size={'unlimited' if not self.max_size_bytes else f'{max_size_gb:.1f}GB'}, "
            f"entries={self._count_entries()}"
        )

    def _init_db(self) -> None:
        """Create the SQLite index if it doesn't exist."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                token_hash TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                num_tokens INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON cache_entries(last_accessed)
        """)
        conn.commit()
        conn.close()

    def _count_entries(self) -> int:
        conn = sqlite3.connect(str(self._db_path))
        count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        conn.close()
        return count

    def _total_size(self) -> int:
        """Get total size of all cached files in bytes."""
        conn = sqlite3.connect(str(self._db_path))
        result = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM cache_entries").fetchone()[0]
        conn.close()
        return result

    def fetch(self, tokens: List[int]) -> Optional[List[Any]]:
        """
        Look up a cached KV state for the given token sequence.

        Returns the cache object list if found, None on miss.
        The returned cache is ready to be used as prompt_cache in BatchGenerator.
        """
        token_hash = _hash_tokens(tokens)

        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT file_name FROM cache_entries WHERE token_hash = ?",
            (token_hash,)
        ).fetchone()

        if row is None:
            conn.close()
            self.misses += 1
            return None

        file_name = row[0]
        file_path = self.cache_dir / file_name

        if not file_path.exists():
            # File was deleted externally — clean up the index
            conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
            conn.commit()
            conn.close()
            self.misses += 1
            logger.warning(f"Disk cache file missing: {file_path}, removed index entry")
            return None

        try:
            from mlx_lm.models.cache import load_prompt_cache
            cache = load_prompt_cache(str(file_path))

            # Update access time and count
            now = time.time()
            conn.execute(
                "UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 "
                "WHERE token_hash = ?",
                (now, token_hash)
            )
            conn.commit()
            conn.close()

            self.hits += 1
            logger.info(
                f"Disk cache hit: {len(tokens)} tokens, "
                f"file={file_name} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)"
            )
            return cache

        except Exception as e:
            conn.close()
            self.misses += 1
            logger.warning(f"Failed to load disk cache {file_path}: {e}")
            # Remove corrupt entry
            try:
                self._remove_entry(token_hash)
            except Exception:
                pass
            return None

    def store(self, tokens: List[int], cache: List[Any], metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Store a KV cache to disk for the given token sequence.

        Args:
            tokens: The prompt token IDs this cache corresponds to.
            cache: The cache object list (from BatchGenerator/prefix cache).
            metadata: Optional string metadata to store alongside the cache.

        Returns:
            True if stored successfully, False otherwise.
        """
        token_hash = _hash_tokens(tokens)

        # Check if already cached
        conn = sqlite3.connect(str(self._db_path))
        existing = conn.execute(
            "SELECT file_name FROM cache_entries WHERE token_hash = ?",
            (token_hash,)
        ).fetchone()
        if existing:
            conn.close()
            return True  # Already cached

        conn.close()

        # Generate filename
        file_name = f"cache_{token_hash[:16]}_{len(tokens)}tok.safetensors"
        file_path = self.cache_dir / file_name

        try:
            from mlx_lm.models.cache import save_prompt_cache

            # Ensure cache objects have the required .state/.meta_state protocol
            for i, c in enumerate(cache):
                if not hasattr(c, 'state') or not hasattr(c, 'meta_state'):
                    logger.warning(
                        f"Cache layer {i} ({type(c).__name__}) missing state/meta_state protocol, "
                        "cannot save to disk"
                    )
                    return False

            save_metadata = metadata or {}
            save_metadata["num_tokens"] = str(len(tokens))
            save_metadata["created_at"] = str(time.time())

            save_prompt_cache(str(file_path), cache, save_metadata)

            file_size = file_path.stat().st_size
            now = time.time()

            # Insert into index
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(token_hash, file_name, num_tokens, file_size, created_at, last_accessed, access_count, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (token_hash, file_name, len(tokens), file_size, now, now,
                     json.dumps(save_metadata) if save_metadata else None)
                )
                conn.commit()

            self.stores += 1
            logger.info(
                f"Disk cache stored: {len(tokens)} tokens, "
                f"{file_size / 1024 / 1024:.1f}MB → {file_name}"
            )

            # Evict if over size limit
            self._evict_if_needed()

            return True

        except Exception as e:
            logger.warning(f"Failed to store disk cache: {e}")
            # Clean up partial file
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            return False

    def _remove_entry(self, token_hash: str) -> None:
        """Remove a cache entry (both file and index)."""
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT file_name FROM cache_entries WHERE token_hash = ?",
            (token_hash,)
        ).fetchone()
        if row:
            file_path = self.cache_dir / row[0]
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
            conn.commit()
        conn.close()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if total size exceeds the limit."""
        if not self.max_size_bytes:
            return

        total = self._total_size()
        if total <= self.max_size_bytes:
            return

        conn = sqlite3.connect(str(self._db_path))
        # Get entries ordered by LRU (least recently accessed first)
        rows = conn.execute(
            "SELECT token_hash, file_name, file_size FROM cache_entries "
            "ORDER BY last_accessed ASC"
        ).fetchall()

        evicted = 0
        for token_hash, file_name, file_size in rows:
            if total <= self.max_size_bytes:
                break
            file_path = self.cache_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
            total -= file_size
            evicted += 1

        if evicted:
            conn.commit()
            logger.info(f"Disk cache evicted {evicted} entries to stay within size limit")
        conn.close()

    def clear(self) -> None:
        """Remove all cached files and reset the index."""
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute("SELECT file_name FROM cache_entries").fetchall()
        for (file_name,) in rows:
            file_path = self.cache_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
        conn.execute("DELETE FROM cache_entries")
        conn.commit()
        conn.close()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        logger.info("Disk cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_size = self._total_size()
        count = self._count_entries()
        return {
            "entries": count,
            "total_size_mb": total_size / 1024 / 1024,
            "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024 if self.max_size_bytes else 0,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
        }

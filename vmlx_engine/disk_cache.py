# SPDX-License-Identifier: Apache-2.0
"""
Disk-based prompt cache persistence for vmlx-engine.

Saves and loads pre-computed KV/Mamba caches to disk using mlx-lm's
safetensors-based save_prompt_cache/load_prompt_cache. A SQLite index
maps token hash → cache file for fast lookup.

This acts as an L2 cache: the in-memory prefix cache is L1 (fast, limited),
and the disk cache is L2 (slower I/O, much larger capacity). On cache miss
in L1, the scheduler checks L2 before doing a full prefill.

Architecture:
- Background writer thread: store() enqueues writes so they don't block inference
- SQLite connection pool: reuses connections instead of opening per-operation
- Graceful shutdown: flushes pending writes before exit
"""

import errno
import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def _hash_tokens(tokens: List[int]) -> str:
    """Create a stable hash of a token sequence for cache indexing."""
    # Use SHA-256 of the token list serialized as compact JSON
    data = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class _ConnectionPool:
    """Simple SQLite connection pool (thread-safe).

    Reuses connections instead of opening a new one per operation.
    SQLite in WAL mode supports concurrent readers with a single writer.
    """

    def __init__(self, db_path: str, max_size: int = 4):
        self._db_path = db_path
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._max_size = max_size

    def get(self) -> sqlite3.Connection:
        """Get a connection from the pool (or create a new one)."""
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            conn = sqlite3.connect(self._db_path, timeout=5.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            return conn

    def put(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()

    def close_all(self) -> None:
        """Close all pooled connections."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


class DiskCacheManager:
    """
    Persistent disk-based cache for KV/Mamba states.

    Stores prompt caches as .safetensors files indexed by a SQLite database.
    Compatible with all mlx-lm cache types (KVCache, QuantizedKVCache,
    RotatingKVCache, ArraysCache/MambaCache, CacheList).

    Features:
    - Background writer thread: store() is non-blocking
    - SQLite connection pool: avoids per-operation connection overhead
    - Graceful shutdown: flushes pending writes

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
        self._db_path = str(self.cache_dir / "cache_index.db")
        self._init_db()

        # Connection pool
        self._pool = _ConnectionPool(self._db_path, max_size=4)

        # Stats (thread-safe via lock)
        self._stats_lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stores = 0

        # Background writer thread
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._background_writer, daemon=True, name="disk-cache-writer"
        )
        self._writer_thread.start()

        # Clean up orphaned .tmp files from crashed writes
        self._cleanup_orphaned_tmp()

        logger.info(
            f"Disk cache initialized: dir={self.cache_dir}, "
            f"max_size={'unlimited' if not self.max_size_bytes else f'{max_size_gb:.1f}GB'}, "
            f"entries={self._count_entries()}"
        )

    def _init_db(self) -> None:
        """Create the SQLite index if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
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

    def _cleanup_orphaned_tmp(self) -> None:
        """Remove orphaned .tmp files left from crashed or buggy writes.
        Matches both *.tmp and *.tmp.safetensors (from mx.save_safetensors
        appending .safetensors to the path)."""
        try:
            count = 0
            for pattern in ("*.tmp", "*.tmp.safetensors"):
                for tmp in self.cache_dir.glob(pattern):
                    try:
                        tmp.unlink()
                        count += 1
                    except OSError:
                        pass
            if count:
                logger.info(f"Disk cache: cleaned up {count} orphaned tmp file(s)")
        except Exception:
            pass

    def _count_entries(self) -> int:
        conn = self._pool.get()
        try:
            count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            return count
        finally:
            self._pool.put(conn)

    def _total_size(self) -> int:
        """Get total size of all cached files in bytes."""
        conn = self._pool.get()
        try:
            result = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM cache_entries").fetchone()[0]
            return result
        finally:
            self._pool.put(conn)

    def fetch(self, tokens: List[int]) -> Optional[List[Any]]:
        """
        Look up a cached KV state for the given token sequence.

        Returns the cache object list if found, None on miss.
        The returned cache is ready to be used as prompt_cache in BatchGenerator.
        """
        token_hash = _hash_tokens(tokens)

        conn = self._pool.get()
        try:
            row = conn.execute(
                "SELECT file_name FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()

            if row is None:
                with self._stats_lock:
                    self.misses += 1
                return None

            file_name = row[0]
            file_path = self.cache_dir / file_name

            if not file_path.exists():
                # File was deleted externally — clean up the index
                conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
                conn.commit()
                with self._stats_lock:
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

                with self._stats_lock:
                    self.hits += 1
                try:
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    logger.info(f"Disk cache hit: {len(tokens)} tokens, file={file_name} ({size_mb:.1f}MB)")
                except OSError:
                    logger.info(f"Disk cache hit: {len(tokens)} tokens, file={file_name}")
                return cache

            except Exception as e:
                with self._stats_lock:
                    self.misses += 1
                logger.warning(f"Failed to load disk cache {file_path}: {e}")
                # Remove corrupt entry
                try:
                    conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
                    conn.commit()
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass
                return None
        finally:
            self._pool.put(conn)

    def store(self, tokens: List[int], cache: List[Any], metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Enqueue a KV cache for background storage to disk.

        The actual I/O happens on the background writer thread so this call
        is non-blocking. Returns True if the write was enqueued (or already
        cached), False if the queue is full or the cache is not serializable.

        IMPORTANT: All MLX operations (serialize + materialize) happen on the
        calling thread to prevent concurrent GPU access from the background
        writer. The background thread only does file I/O — no MLX ops.
        This mirrors the pattern in BlockDiskStore.write_block_async().

        Args:
            tokens: The prompt token IDs this cache corresponds to.
            cache: The cache object list (from BatchGenerator/prefix cache).
            metadata: Optional string metadata to store alongside the cache.

        Returns:
            True if enqueued or already cached, False otherwise.
        """
        token_hash = _hash_tokens(tokens)

        # Quick check if already cached (read-only, no write lock needed)
        conn = self._pool.get()
        try:
            existing = conn.execute(
                "SELECT 1 FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
        finally:
            self._pool.put(conn)

        if existing:
            return True  # Already cached

        # Verify cache objects have the required .state/.meta_state protocol
        for i, c in enumerate(cache):
            if not hasattr(c, 'state') or not hasattr(c, 'meta_state'):
                logger.warning(
                    f"Cache layer {i} ({type(c).__name__}) missing state/meta_state protocol, "
                    "cannot save to disk"
                )
                return False

        if not _HAS_MLX:
            return False

        # Pre-serialize on the calling (main) thread to prevent MLX GPU ops
        # on the background writer thread, which causes Metal assertion failures
        # from concurrent command buffer access (SIGSEGV / failed assertion).
        try:
            # Extract tensor data + metadata the same way save_prompt_cache does
            cache_data = [c.state for c in cache]
            cache_info = [c.meta_state for c in cache]
            cache_data_flat = dict(tree_flatten(cache_data))
            cache_classes = [type(c).__name__ for c in cache]

            save_metadata = metadata or {}
            save_metadata["num_tokens"] = str(len(tokens))
            save_metadata["created_at"] = str(time.time())

            cache_metadata = [cache_info, save_metadata, cache_classes]
            cache_metadata_flat = dict(tree_flatten(cache_metadata))

            # Materialize all lazy MLX arrays on the calling thread.
            # mx.eval forces GPU computation NOW so the background thread
            # won't need to trigger any Metal operations — the arrays are
            # fully concrete values in memory with no pending Metal work.
            # We pass pre-evaluated MLX arrays (not numpy) to preserve
            # bfloat16 and other dtypes that numpy may not support.
            arrays_to_materialize = [v for v in cache_data_flat.values()
                                     if isinstance(v, mx.array)]
            if arrays_to_materialize:
                mx.eval(*arrays_to_materialize)  # noqa: S307 — MLX GPU eval, not Python eval

        except Exception as e:
            logger.warning(f"Failed to pre-serialize cache for disk: {e}")
            return False

        # Enqueue for background write (pre-evaluated arrays — no lazy Metal ops)
        try:
            self._write_queue.put_nowait(
                (token_hash, tokens, cache_data_flat, cache_metadata_flat)
            )
            return True
        except queue.Full:
            logger.warning("Disk cache write queue full, dropping store request")
            return False

    def _background_writer(self) -> None:
        """Background thread: drain write queue and persist caches.

        All tensor data arrives as pre-evaluated MLX arrays (mx.eval called
        on the main thread). mx.save_safetensors reads the raw bytes without
        triggering new Metal computation, preserving bfloat16 and all dtypes.
        """
        while not self._stop_event.is_set():
            try:
                item = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                token_hash, tokens, cache_data_flat, cache_metadata_flat = item
                self._write_cache(token_hash, tokens, cache_data_flat, cache_metadata_flat)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logger.warning(
                        "Disk cache: filesystem full (ENOSPC), skipping write. "
                        "Free disk space or reduce max_size_gb."
                    )
                else:
                    logger.warning(f"Background disk cache write failed: {e}")
            except Exception as e:
                logger.warning(f"Background disk cache write failed: {e}")

    def _write_cache(
        self,
        token_hash: str,
        tokens: List[int],
        cache_data_flat: Dict[str, Any],
        cache_metadata_flat: Dict[str, str],
    ) -> None:
        """Write a pre-serialized cache to disk (called from background thread).

        All tensor data arrives as pre-evaluated MLX arrays (mx.eval already
        called on the main thread). No lazy Metal operations will be triggered
        — the arrays are fully concrete values in memory. mx.save_safetensors
        only reads the raw bytes, preserving bfloat16 and all other dtypes.
        """
        # Double-check not already cached (race with concurrent stores)
        conn = self._pool.get()
        try:
            existing = conn.execute(
                "SELECT 1 FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
            if existing:
                return
        finally:
            self._pool.put(conn)

        # Generate filename
        file_name = f"cache_{token_hash[:16]}_{len(tokens)}tok.safetensors"
        file_path = self.cache_dir / file_name
        tmp_path = self.cache_dir / f"cache_{token_hash[:16]}_{len(tokens)}tok.tmp.safetensors"

        try:
            # Ensure the cache directory exists before writing.
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp file then rename.
            # os.rename is atomic on macOS (same filesystem), so readers
            # never see a partially-written cache file.
            # Arrays are pre-evaluated (mx.eval on main thread) — no lazy
            # Metal ops will be triggered. mx.save_safetensors preserves
            # bfloat16 and all dtypes (numpy conversion would lose bfloat16).
            if not _HAS_MLX:
                return
            mx.save_safetensors(str(tmp_path), cache_data_flat, cache_metadata_flat)

            os.rename(str(tmp_path), str(file_path))

            file_size = file_path.stat().st_size
            now = time.time()

            # Insert into index
            db_meta = {"num_tokens": str(len(tokens)), "created_at": str(now)}
            conn = self._pool.get()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(token_hash, file_name, num_tokens, file_size, created_at, last_accessed, access_count, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (token_hash, file_name, len(tokens), file_size, now, now,
                     json.dumps(db_meta))
                )
                conn.commit()
            finally:
                self._pool.put(conn)

            with self._stats_lock:
                self.stores += 1
            logger.info(
                f"Disk cache stored: {len(tokens)} tokens, "
                f"{file_size / 1024 / 1024:.1f}MB → {file_name}"
            )

            # Evict if over size limit
            self._evict_if_needed()

        except Exception as e:
            logger.warning(f"Failed to store disk cache: {e}")
            # Clean up temp file and any partial final file
            for p in (tmp_path, file_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if total size exceeds the limit."""
        if not self.max_size_bytes:
            return

        total = self._total_size()
        if total <= self.max_size_bytes:
            return

        conn = self._pool.get()
        try:
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
                        # File deletion failed — skip this entry to avoid
                        # orphaning the file (DB row gone but file remains)
                        continue
                conn.execute("DELETE FROM cache_entries WHERE token_hash = ?", (token_hash,))
                total -= file_size
                evicted += 1

            if evicted:
                conn.commit()
                logger.info(f"Disk cache evicted {evicted} entries to stay within size limit")
        finally:
            self._pool.put(conn)

    def clear(self) -> None:
        """Remove all cached files and reset the index."""
        conn = self._pool.get()
        try:
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
        finally:
            self._pool.put(conn)
        with self._stats_lock:
            self.hits = 0
            self.misses = 0
            self.stores = 0
        logger.info("Disk cache cleared")

    def shutdown(self) -> None:
        """Stop background writer, flush pending writes, and close connections."""
        self._stop_event.set()
        self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            logger.warning("Disk cache writer thread did not stop in time")

        # Flush remaining items from write queue
        while not self._write_queue.empty():
            try:
                item = self._write_queue.get_nowait()
                token_hash, tokens, cache_data_flat, cache_metadata_flat = item
                self._write_cache(token_hash, tokens, cache_data_flat, cache_metadata_flat)
            except queue.Empty:
                break
            except Exception as e:
                logger.warning(f"Failed to flush disk cache write: {e}")

        # Close connection pool
        self._pool.close_all()
        logger.info("Disk cache shut down")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_size = self._total_size()
        count = self._count_entries()
        with self._stats_lock:
            return {
                "entries": count,
                "total_size_mb": total_size / 1024 / 1024,
                "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024 if self.max_size_bytes else 0,
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "hit_rate": self.hits / max(self.hits + self.misses, 1),
                "pending_writes": self._write_queue.qsize(),
            }

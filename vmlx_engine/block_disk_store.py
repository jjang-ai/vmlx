# SPDX-License-Identifier: Apache-2.0
"""
Block-level disk persistence for paged KV cache.

Provides an L2 disk tier behind the L1 in-memory PagedCacheManager.
Blocks are stored as safetensors files indexed by their chain hash.

Architecture:
- Each block (e.g. 64 tokens of KV data) is stored as a separate safetensors file
- Content-addressable: file path derived from the block's chain hash
- SQLite WAL index maps hash → file for fast lookup
- Background writer thread prevents disk I/O from blocking inference
- LRU eviction when total disk usage exceeds configured max

Integration points:
- On L1 eviction: write block to disk before freeing RAM
- On L1 lookup miss: check disk before recomputing
- On generation complete: write-through new blocks to disk

Supported cache_data tuple types (from prefix_cache.py):
- ("kv", keys_slice, values_slice) — standard KVCache
- ("quantized_kv", keys_tuple, values_tuple, meta) — QuantizedKVCache
- ("rotating_kv", keys_slice, values_slice, max_size, keep) — RotatingKVCache
- ("cumulative", state_list, meta, class_name) — MambaCache/ArraysCache
- ("skip",) — placeholder for cumulative layers in non-last blocks
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False



class BlockDiskStore:
    """
    Content-addressable block storage on disk for paged KV cache.

    Each block is serialized as a safetensors file containing per-layer
    KV tensors. A SQLite index maps chain hashes to file paths for O(1) lookup.

    Args:
        cache_dir: Directory to store cache files.
        max_size_gb: Maximum total cache size in GB. 0 = unlimited.
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 10.0,
        **kwargs,
    ):
        self.cache_dir = Path(cache_dir)
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max(0.0, max_size_gb) * 1024**3)

        # SQLite index
        self._db_path = self.cache_dir / "block_index.db"
        self._init_db()

        # Stats (protected by _stats_lock for cross-thread accuracy)
        self._stats_lock = threading.Lock()
        self.disk_hits = 0
        self.disk_misses = 0
        self.disk_writes = 0
        self.disk_evictions = 0

        # Persistent read connection (main thread only — not shared with writer)
        self._read_conn = sqlite3.connect(str(self._db_path), timeout=1.0)
        self._read_conn.execute("PRAGMA journal_mode=WAL")

        # Background writer thread
        # Queue items: (block_hash, tmp_path_str, dtype_str, num_layers, token_count)
        # or special commands: ("__access__", ...) or ("__cleanup__", ...)
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._tmp_seq = 0  # Monotonic counter for unique temp file names
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._background_writer, daemon=True, name="block-disk-writer"
        )
        self._writer_thread.start()

        # Clean up orphaned .tmp files from crashed writes
        self._cleanup_orphaned_tmp()

        entry_count = self._count_entries()
        total_size = self._total_size()
        logger.info(
            f"BlockDiskStore initialized: dir={self.cache_dir}, "
            f"max_size={max_size_gb:.1f}GB, entries={entry_count}, "
            f"size={total_size / 1024**3:.2f}GB"
        )

    def _init_db(self) -> None:
        """Create SQLite index with WAL mode."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_hash    TEXT PRIMARY KEY,
                    file_name     TEXT NOT NULL,
                    num_tokens    INTEGER NOT NULL,
                    num_layers    INTEGER NOT NULL,
                    dtype         TEXT NOT NULL,
                    file_size     INTEGER NOT NULL,
                    created_at    REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count  INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_lru ON blocks(last_accessed ASC)"
            )
            conn.commit()
        finally:
            conn.close()

    def _cleanup_orphaned_tmp(self) -> None:
        """Remove orphaned .tmp files left from crashed writes."""
        try:
            count = 0
            for tmp in self.blocks_dir.rglob("*.tmp.safetensors"):
                try:
                    tmp.unlink()
                    count += 1
                except OSError:
                    pass
            if count:
                logger.info(f"BlockDiskStore: cleaned up {count} orphaned .tmp file(s)")
        except Exception:
            pass

    def _count_entries(self) -> int:
        return self._read_conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

    def _total_size(self) -> int:
        return self._read_conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM blocks"
        ).fetchone()[0]

    def _hash_to_path(self, hash_hex: str) -> Path:
        """Shard by first 2 chars for filesystem efficiency."""
        shard = hash_hex[:2]
        shard_dir = self.blocks_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{hash_hex}.safetensors"

    # =========================================================================
    # Read
    # =========================================================================

    def read_block(self, block_hash: bytes) -> Optional[List[Tuple]]:
        """
        Read a block from disk by its chain hash.

        This method is read-only on the main thread — access metadata updates
        are deferred to the background writer to avoid blocking inference.

        Args:
            block_hash: The BlockHash (SHA-256 chain hash bytes)

        Returns:
            cache_data in the same format as CacheBlock.cache_data,
            or None if not found on disk.
        """
        if not HAS_MLX:
            return None

        hash_hex = block_hash.hex()

        try:
            row = self._read_conn.execute(
                "SELECT file_name, dtype FROM blocks WHERE block_hash = ?",
                (hash_hex,)
            ).fetchone()
        except sqlite3.OperationalError:
            # Connection might be stale after writer vacuum — reconnect
            self._read_conn = sqlite3.connect(str(self._db_path), timeout=1.0)
            row = self._read_conn.execute(
                "SELECT file_name, dtype FROM blocks WHERE block_hash = ?",
                (hash_hex,)
            ).fetchone()

        if row is None:
            with self._stats_lock:
                self.disk_misses += 1
            return None

        file_name, dtype = row

        file_path = self.cache_dir / file_name

        if not file_path.exists():
            # Stale index entry — queue cleanup to background writer
            self._queue_index_cleanup(hash_hex)
            with self._stats_lock:
                self.disk_misses += 1
            return None

        try:
            data = mx.load(str(file_path))
            cache_data = _deserialize_block(data, dtype)
            with self._stats_lock:
                self.disk_hits += 1
            # Queue access metadata update to background (non-blocking)
            self._queue_access_update(hash_hex)
            logger.debug(f"Disk cache hit: {hash_hex[:12]} ({dtype}, {len(cache_data)} layers)")
            return cache_data
        except Exception as e:
            logger.warning(f"Failed to load block {hash_hex[:12]}: {e}")
            # Corrupt file — queue removal
            self._queue_index_cleanup(hash_hex)
            with self._stats_lock:
                self.disk_misses += 1
            return None

    def _queue_access_update(self, hash_hex: str) -> None:
        """Queue an access time update for the background writer."""
        try:
            self._write_queue.put_nowait(("__access__", hash_hex, time.time()))
        except queue.Full:
            pass  # Non-critical metadata update — safe to drop

    def _queue_index_cleanup(self, hash_hex: str) -> None:
        """Queue a stale index entry cleanup for the background writer."""
        try:
            self._write_queue.put_nowait(("__cleanup__", hash_hex, 0))
        except queue.Full:
            pass  # Will be cleaned up on next access attempt

    # =========================================================================
    # Write (async)
    # =========================================================================

    def write_block_async(
        self,
        block_hash: bytes,
        cache_data: List[Tuple],
        token_count: int,
    ) -> None:
        """
        Queue a block for background writing to disk. Non-blocking.

        ALL MLX operations happen on the calling (main) thread:
        - Serialize cache_data to flat tensor dict
        - Materialize lazy arrays with mx.eval()
        - Write safetensors file with mx.save_safetensors()

        The background thread ONLY does: atomic rename + SQLite index update.
        This prevents Metal command buffer crashes from concurrent GPU access
        (mx.save_safetensors accesses Metal buffer memory internally).

        Args:
            block_hash: Chain hash (BlockHash bytes)
            cache_data: CacheBlock.cache_data — list of typed tuples per layer
            token_count: Number of tokens in this block
        """
        if not HAS_MLX:
            return

        hash_hex = block_hash.hex()

        # Pre-serialize on the calling (main) thread.
        try:
            tensors, dtype, num_layers = _serialize_block(cache_data)
            if num_layers == 0:
                return

            # Normalize all tensors to MLX arrays that mx.save_safetensors
            # can handle.  Two issues to fix:
            # 1. numpy ndarrays (from numpy-sliced block data) — convert to mx
            # 2. bfloat16 dtype (unsupported by safetensors) — cast to float16
            import numpy as np
            needs_eval = []
            for k, v in tensors.items():
                if isinstance(v, np.ndarray):
                    tensors[k] = mx.array(v)
                    needs_eval.append(tensors[k])
                elif isinstance(v, mx.array) and v.dtype == mx.bfloat16:
                    tensors[k] = v.astype(mx.float16)
                    needs_eval.append(tensors[k])

            # Materialize all lazy MLX arrays on the calling thread.
            arrays_to_eval = [v for v in tensors.values() if isinstance(v, mx.array)]
            if arrays_to_eval:
                mx.eval(*arrays_to_eval)  # noqa: S307 — mlx tensor materialization

            # Write safetensors file on the main thread.
            # mx.save_safetensors accesses Metal buffer memory internally,
            # so it MUST run on the same thread as inference to avoid
            # concurrent Metal command buffer assertions.
            file_path = self._hash_to_path(hash_hex)
            seq = self._tmp_seq
            self._tmp_seq += 1
            tmp_path = file_path.with_name(f"{file_path.stem}.{seq}.tmp.safetensors")
            mx.save_safetensors(str(tmp_path), tensors)
        except Exception as e:
            logger.debug(f"Pre-serialize/write failed for block {hash_hex[:12]}: {e}")
            return

        # Queue only the rename + DB update for the background thread.
        # No MLX operations happen after this point.
        try:
            self._write_queue.put_nowait(
                (block_hash, str(tmp_path), dtype, num_layers, token_count)
            )
        except queue.Full:
            # Clean up the temp file since background won't process it
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            logger.warning("BlockDiskStore write queue full (1000), dropping block write")

    def _background_writer(self) -> None:
        """Background thread: drain write queue and persist blocks.

        Uses a persistent write connection for the lifetime of this thread,
        avoiding the overhead of opening/closing a SQLite connection per
        operation. The connection is created once at thread start.
        """
        write_conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        write_conn.execute("PRAGMA journal_mode=WAL")

        try:
            while not self._stop_event.is_set():
                # Collect a batch: block on the first item (with timeout so we
                # can check the stop event), then drain any remaining items.
                batch = []
                try:
                    item = self._write_queue.get(timeout=0.2)
                    batch.append(item)
                except queue.Empty:
                    continue

                # Drain remaining items without blocking
                while True:
                    try:
                        batch.append(self._write_queue.get_nowait())
                    except queue.Empty:
                        break

                for item in batch:
                    try:
                        if item[0] == "__access__":
                            _, hash_hex, ts = item
                            self._update_access(write_conn, hash_hex, ts)
                        elif item[0] == "__cleanup__":
                            _, hash_hex, _ = item
                            self._cleanup_entry(write_conn, hash_hex)
                        else:
                            block_hash, tmp_path_str, dtype, num_layers, token_count = item
                            self._write_block(write_conn, block_hash, tmp_path_str, dtype, num_layers, token_count)
                    except Exception as e:
                        h = item[0] if isinstance(item[0], str) else (
                            item[0].hex()[:12] if isinstance(item[0], bytes) else "?"
                        )
                        logger.warning(f"Background writer error ({h}): {e}")

                # Evict if over budget
                self._maybe_evict(write_conn)
        finally:
            write_conn.close()

    def _update_access(self, conn: sqlite3.Connection, hash_hex: str, ts: float) -> None:
        """Update last_accessed time in the index (background thread only)."""
        conn.execute(
            "UPDATE blocks SET last_accessed = ?, access_count = access_count + 1 "
            "WHERE block_hash = ?",
            (ts, hash_hex)
        )
        conn.commit()

    def _cleanup_entry(self, conn: sqlite3.Connection, hash_hex: str) -> None:
        """Remove a stale index entry and its file (background thread only)."""
        row = conn.execute(
            "SELECT file_name FROM blocks WHERE block_hash = ?", (hash_hex,)
        ).fetchone()
        if row:
            file_path = self.cache_dir / row[0]
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
            conn.execute("DELETE FROM blocks WHERE block_hash = ?", (hash_hex,))
            conn.commit()

    def _write_block(
        self,
        conn: sqlite3.Connection,
        block_hash: bytes,
        tmp_path_str: str,
        dtype: str,
        num_layers: int,
        token_count: int,
    ) -> None:
        """Finalize a pre-written block file (called from background thread).

        The safetensors file was already written by the main thread.
        This method ONLY does: atomic rename + SQLite index update.
        No MLX operations — prevents Metal command buffer crashes.
        """
        hash_hex = block_hash.hex()
        tmp_path = Path(tmp_path_str)

        # Skip if already on disk
        exists = conn.execute(
            "SELECT 1 FROM blocks WHERE block_hash = ?", (hash_hex,)
        ).fetchone()
        if exists:
            # Clean up the temp file — block already persisted
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Atomic rename from temp to final path
        file_path = self._hash_to_path(hash_hex)
        rel_path = file_path.relative_to(self.cache_dir)

        try:
            os.rename(str(tmp_path), str(file_path))
        except Exception:
            # Clean up partial file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        file_size = file_path.stat().st_size
        now = time.time()

        conn.execute(
            """INSERT OR IGNORE INTO blocks
               (block_hash, file_name, num_tokens, num_layers, dtype,
                file_size, created_at, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (hash_hex, str(rel_path), token_count, num_layers, dtype,
             file_size, now, now)
        )
        conn.commit()

        with self._stats_lock:
            self.disk_writes += 1
        logger.debug(
            f"Disk cache write: {hash_hex[:12]} ({dtype}, {num_layers} layers, "
            f"{file_size / 1024:.1f}KB, {token_count} tokens)"
        )

    # =========================================================================
    # Eviction
    # =========================================================================

    def _maybe_evict(self, conn: sqlite3.Connection) -> None:
        """Evict LRU blocks if total disk usage exceeds max."""
        if self.max_size_bytes <= 0:
            return

        total = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM blocks"
        ).fetchone()[0]

        if total <= self.max_size_bytes:
            return

        target = int(self.max_size_bytes * 0.8)  # Free down to 80%
        rows = conn.execute(
            "SELECT block_hash, file_name, file_size FROM blocks "
            "ORDER BY last_accessed ASC"
        ).fetchall()

        evicted = 0
        for hash_hex, file_name, file_size in rows:
            if total <= target:
                break
            file_path = self.cache_dir / file_name
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
            conn.execute("DELETE FROM blocks WHERE block_hash = ?", (hash_hex,))
            total -= file_size
            evicted += 1

        conn.commit()

        if evicted:
            with self._stats_lock:
                self.disk_evictions += evicted
            logger.info(
                f"Disk cache eviction: removed {evicted} blocks "
                f"(now {total / 1024**3:.2f}GB)"
            )

    # =========================================================================
    # Management
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        conn = sqlite3.connect(str(self._db_path), timeout=1.0)
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0), "
                "COALESCE(SUM(access_count), 0) FROM blocks"
            ).fetchone()
        finally:
            conn.close()
        with self._stats_lock:
            return {
                "blocks_on_disk": row[0],
                "disk_size_bytes": row[1],
                "disk_size_gb": round(row[1] / 1024**3, 3),
                "total_accesses": row[2],
                "disk_hits": self.disk_hits,
                "disk_misses": self.disk_misses,
                "disk_writes": self.disk_writes,
                "disk_evictions": self.disk_evictions,
            }

    def clear(self) -> None:
        """Clear all cached blocks from disk."""
        import shutil
        # Drain the write queue so the background writer doesn't write
        # blocks we're about to delete
        while not self._write_queue.empty():
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                break
        if self.blocks_dir.exists():
            shutil.rmtree(self.blocks_dir)
            self.blocks_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        try:
            conn.execute("DELETE FROM blocks")
            conn.commit()
        finally:
            conn.close()
        logger.info("Disk cache cleared")

    def shutdown(self) -> None:
        """Stop background writer and flush pending writes."""
        self._stop_event.set()
        self._writer_thread.join(timeout=5.0)
        if self._writer_thread.is_alive():
            logger.warning("BlockDiskStore writer thread did not stop in time, skipping flush")
            return
        # Flush remaining (safe because writer thread has stopped)
        remaining = []
        while not self._write_queue.empty():
            try:
                remaining.append(self._write_queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            flush_conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            flush_conn.execute("PRAGMA journal_mode=WAL")
            try:
                for item in remaining:
                    try:
                        if item[0] == "__access__":
                            self._update_access(flush_conn, item[1], item[2])
                        elif item[0] == "__cleanup__":
                            self._cleanup_entry(flush_conn, item[1])
                        elif isinstance(item[0], bytes):
                            self._write_block(flush_conn, item[0], item[1], item[2], item[3], item[4])
                    except Exception:
                        pass
            finally:
                flush_conn.close()
        # Close persistent read connection
        try:
            self._read_conn.close()
        except Exception:
            pass


# =============================================================================
# Serialization / Deserialization (module-level functions)
# =============================================================================

def _serialize_block(
    cache_data: List[Tuple],
) -> Tuple[Dict[str, Any], str, int]:
    """
    Convert CacheBlock.cache_data to a flat dict of named tensors for safetensors.

    Per-layer type tags are stored in metadata so deserialization can handle
    mixed-type blocks (e.g. hybrid Mamba-Transformer models).

    The naming convention encodes layer index and tensor role:
      layer_{i}_keys, layer_{i}_values          — standard kv
      layer_{i}_keys_data, _scales, _zeros       — quantized kv
      layer_{i}_max_size, layer_{i}_keep          — rotating kv params
      layer_{i}_cumulative_{j}                    — cumulative state arrays

    Returns:
        (tensor_dict, dtype_string, num_layers_with_data)
    """
    if not HAS_MLX:
        return {}, "unknown", 0

    tensors: Dict[str, Any] = {}
    num_layers = 0
    # Per-layer metadata including type tags for mixed-type block support
    meta: Dict[str, Any] = {"__layer_types__": {}}

    for i, layer_data in enumerate(cache_data):
        tag = layer_data[0]

        if tag == "skip":
            continue

        num_layers += 1
        meta["__layer_types__"][str(i)] = tag

        if tag == "kv":
            _, keys, values = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values

        elif tag == "quantized_kv":
            _, keys_tuple, values_tuple, layer_meta = layer_data
            tensors[f"layer_{i}_keys_data"] = keys_tuple[0]
            tensors[f"layer_{i}_keys_scales"] = keys_tuple[1]
            tensors[f"layer_{i}_keys_zeros"] = keys_tuple[2]
            tensors[f"layer_{i}_values_data"] = values_tuple[0]
            tensors[f"layer_{i}_values_scales"] = values_tuple[1]
            tensors[f"layer_{i}_values_zeros"] = values_tuple[2]
            if layer_meta:
                meta[str(i)] = {"quant_meta": layer_meta}

        elif tag == "rotating_kv":
            _, keys, values, max_size, keep = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            tensors[f"layer_{i}_max_size"] = mx.array([max_size], dtype=mx.int32)
            tensors[f"layer_{i}_keep"] = mx.array([keep], dtype=mx.int32)

        elif tag == "cumulative":
            _, state_list, layer_meta, class_name = layer_data
            if isinstance(state_list, (list, tuple)):
                for j, arr in enumerate(state_list):
                    if hasattr(arr, "shape"):
                        tensors[f"layer_{i}_cumulative_{j}"] = arr
            meta[str(i)] = {"class_name": class_name, "meta": layer_meta}

    # Determine dominant dtype for the DB index (informational only)
    type_set = set(meta["__layer_types__"].values())
    if len(type_set) == 1:
        dtype = type_set.pop()
    elif type_set:
        dtype = "mixed"
    else:
        dtype = "kv"

    # Store metadata as a serialized JSON tensor
    if meta:
        meta_bytes = json.dumps(meta).encode("utf-8")
        tensors["__metadata__"] = mx.array(
            list(meta_bytes), dtype=mx.uint8
        )

    return tensors, dtype, num_layers


def _deserialize_block(
    data: Dict[str, Any],
    dtype: str,
) -> List[Tuple]:
    """
    Reconstruct CacheBlock.cache_data from loaded safetensors dict.

    Uses per-layer type tags from metadata for mixed-type block support.
    Falls back to the dtype field for backward compatibility with blocks
    serialized before per-layer tags were added.
    """
    # Extract metadata if present
    meta: Dict[str, Any] = {}
    meta_arr = data.get("__metadata__")
    if meta_arr is not None:
        try:
            meta_bytes = bytes(meta_arr.tolist())
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
            pass
    # Remove from data dict so it's not picked up as a layer
    data.pop("__metadata__", None)

    # Per-layer type map (new format with __layer_types__)
    layer_types = meta.get("__layer_types__", {})

    # Find all layer indices
    layer_indices: Dict[int, str] = {}
    for key in data:
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                idx = int(parts[1])
                if idx not in layer_indices:
                    layer_indices[idx] = key
            except ValueError:
                continue

    if not layer_indices:
        return []

    max_layer = max(layer_indices.keys())
    cache_data: List[Tuple] = []

    for i in range(max_layer + 1):
        if i not in layer_indices:
            cache_data.append(("skip",))
            continue

        # Determine this layer's type: prefer per-layer tag, fallback to global dtype
        layer_type = layer_types.get(str(i), _infer_layer_type(data, i, dtype))

        if layer_type == "kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            if keys is not None and values is not None:
                # Restore bfloat16 from float16 (serialization casts
                # bfloat16→float16 because safetensors doesn't support it)
                if HAS_MLX and keys.dtype == mx.float16:
                    keys = keys.astype(mx.bfloat16)
                    values = values.astype(mx.bfloat16)
                cache_data.append(("kv", keys, values))
            else:
                cache_data.append(("skip",))

        elif layer_type == "quantized_kv":
            try:
                keys_tuple = (
                    data[f"layer_{i}_keys_data"],
                    data[f"layer_{i}_keys_scales"],
                    data[f"layer_{i}_keys_zeros"],
                )
                values_tuple = (
                    data[f"layer_{i}_values_data"],
                    data[f"layer_{i}_values_scales"],
                    data[f"layer_{i}_values_zeros"],
                )
                layer_meta_dict = meta.get(str(i), {})
                layer_meta = layer_meta_dict.get("quant_meta", layer_meta_dict.get("meta", {}))
                cache_data.append(("quantized_kv", keys_tuple, values_tuple, layer_meta))
            except KeyError:
                cache_data.append(("skip",))

        elif layer_type == "rotating_kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            max_size_arr = data.get(f"layer_{i}_max_size")
            keep_arr = data.get(f"layer_{i}_keep")
            if keys is not None and values is not None:
                if HAS_MLX and keys.dtype == mx.float16:
                    keys = keys.astype(mx.bfloat16)
                    values = values.astype(mx.bfloat16)
                max_size = int(max_size_arr.item()) if max_size_arr is not None else 0
                keep = int(keep_arr.item()) if keep_arr is not None else 0
                cache_data.append(("rotating_kv", keys, values, max_size, keep))
            else:
                cache_data.append(("skip",))

        elif layer_type == "cumulative":
            layer_meta_dict = meta.get(str(i), {})
            class_name = layer_meta_dict.get("class_name", "")
            layer_meta_val = layer_meta_dict.get("meta", "")
            state_arrays = []
            j = 0
            while f"layer_{i}_cumulative_{j}" in data:
                state_arrays.append(data[f"layer_{i}_cumulative_{j}"])
                j += 1
            if state_arrays:
                cache_data.append(("cumulative", state_arrays, layer_meta_val, class_name))
            else:
                cache_data.append(("skip",))
        else:
            cache_data.append(("skip",))

    return cache_data


def _infer_layer_type(data: Dict[str, Any], layer_idx: int, fallback_dtype: str) -> str:
    """Infer a layer's type from its tensor keys (backward compat for old blocks)."""
    prefix = f"layer_{layer_idx}_"
    has_keys_data = f"{prefix}keys_data" in data
    has_cumulative = f"{prefix}cumulative_0" in data
    has_max_size = f"{prefix}max_size" in data
    has_keys = f"{prefix}keys" in data

    if has_keys_data:
        return "quantized_kv"
    if has_cumulative:
        return "cumulative"
    if has_max_size and has_keys:
        return "rotating_kv"
    if has_keys:
        return "kv"
    return fallback_dtype

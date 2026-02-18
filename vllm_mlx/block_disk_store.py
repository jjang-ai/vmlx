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

import hashlib
import json
import logging
import os
import sqlite3
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
        max_size_gb: Maximum total cache size in GB. LRU eviction when exceeded.
        write_through: If True, write blocks on store (not just on eviction).
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 10.0,
        write_through: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.write_through = write_through

        # SQLite index
        self._db_path = self.cache_dir / "block_index.db"
        self._init_db()

        # Stats
        self.disk_hits = 0
        self.disk_misses = 0
        self.disk_writes = 0
        self.disk_evictions = 0

        # Background writer thread
        self._write_queue: List[Tuple[bytes, List[Tuple], int]] = []
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._background_writer, daemon=True, name="block-disk-writer"
        )
        self._writer_thread.start()

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
        conn.close()

    def _count_entries(self) -> int:
        conn = sqlite3.connect(str(self._db_path))
        count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        conn.close()
        return count

    def _total_size(self) -> int:
        conn = sqlite3.connect(str(self._db_path))
        size = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM blocks"
        ).fetchone()[0]
        conn.close()
        return size

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

        Args:
            block_hash: The BlockHash (SHA-256 chain hash bytes)

        Returns:
            cache_data in the same format as CacheBlock.cache_data,
            or None if not found on disk.
        """
        if not HAS_MLX:
            return None

        hash_hex = block_hash.hex()

        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT file_name, dtype FROM blocks WHERE block_hash = ?",
            (hash_hex,)
        ).fetchone()

        if row is None:
            conn.close()
            self.disk_misses += 1
            return None

        file_name, dtype = row
        file_path = self.cache_dir / file_name

        if not file_path.exists():
            # Stale index entry
            conn.execute("DELETE FROM blocks WHERE block_hash = ?", (hash_hex,))
            conn.commit()
            conn.close()
            self.disk_misses += 1
            return None

        # Update access metadata
        conn.execute(
            "UPDATE blocks SET last_accessed = ?, access_count = access_count + 1 "
            "WHERE block_hash = ?",
            (time.time(), hash_hex)
        )
        conn.commit()
        conn.close()

        try:
            data = mx.load(str(file_path))
            cache_data = _deserialize_block(data, dtype)
            self.disk_hits += 1
            logger.debug(f"Disk cache hit: {hash_hex[:12]} ({dtype}, {len(cache_data)} layers)")
            return cache_data
        except Exception as e:
            logger.warning(f"Failed to load block {hash_hex[:12]}: {e}")
            self.disk_misses += 1
            return None

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

        Args:
            block_hash: Chain hash (BlockHash bytes)
            cache_data: CacheBlock.cache_data — list of typed tuples per layer
            token_count: Number of tokens in this block
        """
        with self._write_lock:
            self._write_queue.append((block_hash, cache_data, token_count))

    def _background_writer(self) -> None:
        """Background thread: drain write queue and persist blocks."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.2)  # 200ms poll

            with self._write_lock:
                if not self._write_queue:
                    continue
                batch = self._write_queue[:]
                self._write_queue.clear()

            for block_hash, cache_data, token_count in batch:
                try:
                    self._write_block(block_hash, cache_data, token_count)
                except Exception as e:
                    h = block_hash.hex()[:12] if isinstance(block_hash, bytes) else "?"
                    logger.warning(f"Failed to write block {h}: {e}")

            # Evict if over budget
            self._maybe_evict()

    def _write_block(
        self,
        block_hash: bytes,
        cache_data: List[Tuple],
        token_count: int,
    ) -> None:
        """Write a single block to disk (called from background thread)."""
        if not HAS_MLX:
            return

        hash_hex = block_hash.hex()

        # Skip if already on disk
        conn = sqlite3.connect(str(self._db_path))
        exists = conn.execute(
            "SELECT 1 FROM blocks WHERE block_hash = ?", (hash_hex,)
        ).fetchone()
        if exists:
            conn.close()
            return

        # Serialize
        tensors, dtype, num_layers = _serialize_block(cache_data)
        if not tensors:
            conn.close()
            return

        # Save as safetensors
        file_path = self._hash_to_path(hash_hex)
        rel_path = file_path.relative_to(self.cache_dir)

        mx.save_safetensors(str(file_path), tensors)
        mx.metal.clear_cache()  # Don't let serialization buffers linger

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
        conn.close()

        self.disk_writes += 1
        logger.debug(
            f"Disk cache write: {hash_hex[:12]} ({dtype}, {num_layers} layers, "
            f"{file_size / 1024:.1f}KB, {token_count} tokens)"
        )

    # =========================================================================
    # Eviction
    # =========================================================================

    def _maybe_evict(self) -> None:
        """Evict LRU blocks if total disk usage exceeds max."""
        if self.max_size_bytes <= 0:
            return

        conn = sqlite3.connect(str(self._db_path))
        total = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM blocks"
        ).fetchone()[0]

        if total <= self.max_size_bytes:
            conn.close()
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
        conn.close()

        if evicted:
            self.disk_evictions += evicted
            logger.info(
                f"Disk cache eviction: removed {evicted} blocks "
                f"(now {total / 1024**3:.2f}GB)"
            )

    # =========================================================================
    # Management
    # =========================================================================

    def has_block(self, block_hash: bytes) -> bool:
        """Check if a block exists on disk without loading it."""
        hash_hex = block_hash.hex()
        conn = sqlite3.connect(str(self._db_path))
        exists = conn.execute(
            "SELECT 1 FROM blocks WHERE block_hash = ?", (hash_hex,)
        ).fetchone()
        conn.close()
        return exists is not None

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(file_size), 0), "
            "COALESCE(SUM(access_count), 0) FROM blocks"
        ).fetchone()
        conn.close()
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
        if self.blocks_dir.exists():
            shutil.rmtree(self.blocks_dir)
            self.blocks_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("DELETE FROM blocks")
        conn.commit()
        conn.close()
        logger.info("Disk cache cleared")

    def shutdown(self) -> None:
        """Stop background writer and flush pending writes."""
        self._stop_event.set()
        self._writer_thread.join(timeout=5.0)
        # Flush remaining
        with self._write_lock:
            remaining = self._write_queue[:]
            self._write_queue.clear()
        for block_hash, cache_data, token_count in remaining:
            try:
                self._write_block(block_hash, cache_data, token_count)
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
    dtype = "kv"
    num_layers = 0
    # Metadata stored as a JSON string in a special key
    meta: Dict[str, Any] = {}

    for i, layer_data in enumerate(cache_data):
        tag = layer_data[0]

        if tag == "skip":
            continue

        num_layers += 1

        if tag == "kv":
            _, keys, values = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values

        elif tag == "quantized_kv":
            _, keys_tuple, values_tuple, layer_meta = layer_data
            dtype = "quantized_kv"
            tensors[f"layer_{i}_keys_data"] = keys_tuple[0]
            tensors[f"layer_{i}_keys_scales"] = keys_tuple[1]
            tensors[f"layer_{i}_keys_zeros"] = keys_tuple[2]
            tensors[f"layer_{i}_values_data"] = values_tuple[0]
            tensors[f"layer_{i}_values_scales"] = values_tuple[1]
            tensors[f"layer_{i}_values_zeros"] = values_tuple[2]
            if layer_meta:
                meta[str(i)] = layer_meta

        elif tag == "rotating_kv":
            _, keys, values, max_size, keep = layer_data
            dtype = "rotating_kv"
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            # Store scalar params as 1-element arrays
            tensors[f"layer_{i}_max_size"] = mx.array([max_size], dtype=mx.int32)
            tensors[f"layer_{i}_keep"] = mx.array([keep], dtype=mx.int32)

        elif tag == "cumulative":
            _, state_list, layer_meta, class_name = layer_data
            dtype = "cumulative"
            # state_list can be various structures — store each array
            if isinstance(state_list, (list, tuple)):
                for j, arr in enumerate(state_list):
                    if hasattr(arr, "shape"):
                        tensors[f"layer_{i}_cumulative_{j}"] = arr
            meta[str(i)] = {"class_name": class_name, "meta": layer_meta}

    # Store metadata as a serialized JSON tensor if non-empty
    if meta:
        meta_bytes = json.dumps(meta).encode("utf-8")
        # Store as uint8 array so it survives safetensors round-trip
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

    Handles all tuple types: kv, quantized_kv, rotating_kv, cumulative, skip.
    """
    # Extract metadata if present
    meta: Dict[str, Any] = {}
    if "__metadata__" in data:
        try:
            meta_arr = data.pop("__metadata__")
            meta_bytes = bytes(meta_arr.tolist())
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
            pass

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

    # Determine max layer index to reconstruct in order
    max_layer = max(layer_indices.keys())
    cache_data: List[Tuple] = []

    for i in range(max_layer + 1):
        if i not in layer_indices:
            cache_data.append(("skip",))
            continue

        if dtype == "kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            if keys is not None and values is not None:
                cache_data.append(("kv", keys, values))
            else:
                cache_data.append(("skip",))

        elif dtype == "quantized_kv":
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
                layer_meta = meta.get(str(i), {})
                cache_data.append(("quantized_kv", keys_tuple, values_tuple, layer_meta))
            except KeyError:
                cache_data.append(("skip",))

        elif dtype == "rotating_kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            max_size_arr = data.get(f"layer_{i}_max_size")
            keep_arr = data.get(f"layer_{i}_keep")
            if keys is not None and values is not None:
                max_size = int(max_size_arr.item()) if max_size_arr is not None else 0
                keep = int(keep_arr.item()) if keep_arr is not None else 0
                cache_data.append(("rotating_kv", keys, values, max_size, keep))
            else:
                cache_data.append(("skip",))

        elif dtype == "cumulative":
            layer_meta_dict = meta.get(str(i), {})
            class_name = layer_meta_dict.get("class_name", "")
            layer_meta_val = layer_meta_dict.get("meta", "")
            # Reconstruct state list from indexed arrays
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

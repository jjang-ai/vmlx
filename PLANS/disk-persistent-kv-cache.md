# Disk-Persistent KV Cache — Implementation Plan

## Overview

Add block-level disk persistence to the paged KV cache in vllm-mlx. When blocks are evicted from RAM, they're saved to SSD. On cache miss, blocks are loaded from SSD before recomputing. This creates a tiered L1 (RAM) + L2 (SSD) cache system optimized for Apple Silicon's fast NVMe.

## Motivation

- **Cold start elimination**: After server restart, the first request for a previously-seen prompt can load cached KV blocks from disk instead of recomputing
- **Larger effective cache**: SSD acts as overflow when RAM fills up — blocks that don't fit in the paged cache's `max_blocks` slots get persisted instead of lost
- **Prefix reuse across sessions**: Common system prompts, instruction prefixes, and few-shot examples persist across server restarts
- **M4 Max SSD advantage**: 7.4 GB/s sequential read makes disk-to-RAM transfer fast enough to be competitive with recomputation for long prompts

## Current Architecture

### Three L1 Cache Backends (in-memory)

| Backend | Config Flag | Class | Storage |
|---------|-------------|-------|---------|
| **Paged** (block-level) | `use_paged_cache=True` | `BlockAwarePrefixCache` + `PagedCacheManager` | Per-block tensor slices in `CacheBlock.cache_data` |
| **Memory-aware** | `use_memory_aware_cache=True` (default) | `MemoryAwarePrefixCache` | Full `KVCache` objects in `OrderedDict`, evicted by byte budget |
| **Legacy** | both False | `PrefixCacheManager` | Full `KVCache` objects in trie dict, evicted by entry count |

### Existing L2 Disk Cache (whole-prompt)

- **File**: `vllm_mlx/disk_cache.py`
- **Format**: mlx-lm's `save_prompt_cache` / `load_prompt_cache` using safetensors files
- **Index**: SQLite WAL database (`token_hash TEXT PK, file_name, num_tokens, file_size, created_at, last_accessed, access_count, metadata`)
- **Key**: `SHA-256(json.dumps(tokens))` — full-precision, no chain hash
- **Filename**: `cache_{hash16}_{N}tok.safetensors`
- **Eviction**: LRU by `last_accessed`, triggered when `sum(file_size) > max_size_bytes`
- **Cache dir**: `~/.cache/vllm-mlx/prompt-cache/{model_slug}_{model_hash12}/`
- **Status**: Disabled by default (`enable_disk_cache: bool = False` in `SchedulerConfig`)

### Paged Cache Block Details

**`CacheBlock` dataclass** (`paged_cache.py:84`):
```python
@dataclass
class CacheBlock:
    block_id: int            # Physical index into the flat block array
    ref_count: int = 0       # Reference count; 0 = evictable
    block_hash: Optional[BlockHash] = None   # SHA-256 chain hash for prefix matching
    prev_free_block: Optional["CacheBlock"] = None   # Doubly-linked free list
    next_free_block: Optional["CacheBlock"] = None
    is_null: bool = False    # Block 0 is reserved null/placeholder
    cache_data: Optional[List[Tuple[Any, Any]]] = None  # Actual KV tensors
    token_count: int = 0
    hash_value: Optional[str] = None   # Legacy string hash (16-char hex)
    last_access: float       # Unix timestamp
```

**Block tensor storage formats** (typed tuples per layer):
- `("kv", keys_slice, values_slice)` — standard KVCache
  - 4D: `(1, n_kv_heads, block_tokens, head_dim)`
  - 3D: `(n_kv_heads, block_tokens, head_dim)`
- `("quantized_kv", keys_tuple, values_tuple, meta)` — QuantizedKVCache
  - keys/values each are `(data_uint32, scales, zeros)` tuples
- `("rotating_kv", keys_slice, values_slice, max_size, keep)` — RotatingKVCache
- `("cumulative", state_list, meta, class_name)` — MambaCache/ArraysCache (last block only)
- `("skip",)` — cumulative layers in non-last blocks

**Content-addressable hashing** (Merkle chain):
- Block hash = `SHA-256(parent_hash || str(tuple(token_ids)) || str(extra_keys))`
- Root seed: `b"vllm-mlx-root"` for the first block
- Each block's hash depends on ALL preceding blocks → same tokens at different positions produce different hashes

**Free list**: `FreeKVCacheBlockQueue` — doubly-linked list, LRU at front, MRU at back.
- `popleft()` → allocate (LRU eviction)
- `append()` → return block (MRU position)
- `remove()` → pull from middle on cache hit

---

## Proposed Design: Block-Level Disk Persistence

### Architecture: "Swap-In, Swap-Out"

```
┌─────────────────────────────────────────┐
│           Inference Request              │
│                                         │
│  1. Tokenize prompt                     │
│  2. Compute chain hashes per block      │
│  3. For each block:                     │
│     a. Check L1 (PagedCacheManager)     │
│     b. On miss → Check L2 (DiskBlock)   │
│     c. On miss → Must recompute         │
└─────────────────────────────────────────┘

┌─────────────┐     evict      ┌──────────────┐
│   L1: RAM   │ ──────────────→│  L2: Disk    │
│ PagedCache  │                │ BlockStore   │
│ (max_blocks)│ ←──────────────│ (max_gb)     │
│             │   load on miss │              │
└─────────────┘                └──────────────┘
```

### Addressing System (Content-Addressable)

Reuse the existing chain hash from `PagedCacheManager`:
- Hash = `SHA-256(parent_hash || token_ids || extra_keys)` (already computed)
- This gives stable, position-aware hashes that survive restarts
- No tokenizer drift risk — same tokens always produce same hash

**Granularity**: Per-block (default 16 tokens per block). This allows partial prefix reuse — if a 1000-token prompt changes the last 50 tokens, the first 950 tokens' blocks still hit cache.

### Storage Format

**Per-block safetensors files** organized by hash prefix:

```
~/.cache/vllm-mlx/block-cache/{model_slug}_{hash12}/
  index.db              # SQLite WAL: block_hash → filename, size, dtype, access_count
  blocks/
    ab/                 # First 2 chars of hash for directory sharding
      ab3f7c...safetensors   # One file per block
    cd/
      cd91a2...safetensors
```

**File contents** (per block, all layers):
- Standard KV: keys and values tensors per layer, stored as `layer_{i}_keys`, `layer_{i}_values`
- Quantized KV: `layer_{i}_keys_data`, `layer_{i}_keys_scales`, `layer_{i}_keys_zeros` + same for values
- Metadata sidecar: dtype info, block_size, token_count, layer types

**On-disk quantization** (optional optimization):
- Store in 4-bit quantized format on disk even if L1 uses full precision
- Reduces I/O by 4x — at 7.4 GB/s SSD, this means loading a block in ~microseconds
- Dequantize on load (CPU work overlaps with GPU inference)
- Config flag: `disk_cache_quantize: bool = True`

### SQLite Index Schema

```sql
CREATE TABLE blocks (
    block_hash   TEXT PRIMARY KEY,  -- Full SHA-256 hex of chain hash
    file_name    TEXT NOT NULL,     -- Relative path under blocks/
    num_tokens   INTEGER NOT NULL,  -- Tokens in this block
    num_layers   INTEGER NOT NULL,  -- Model layer count
    dtype        TEXT NOT NULL,     -- 'kv', 'quantized_kv', etc.
    file_size    INTEGER NOT NULL,  -- Bytes on disk
    created_at   REAL NOT NULL,     -- Unix timestamp
    last_accessed REAL NOT NULL,    -- For LRU eviction
    access_count INTEGER DEFAULT 0  -- Hit frequency
);

CREATE INDEX idx_blocks_lru ON blocks(last_accessed ASC);
```

---

## Integration Points

### 1. Block Eviction Hook — `paged_cache.py:609`

`_maybe_evict_cached_block()` is called when a block needs to be reused. Add disk write here:

```python
def _maybe_evict_cached_block(self, block: CacheBlock) -> bool:
    if block.block_hash is None:
        return False
    evicted = self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)
    if evicted:
        # NEW: Persist to disk before clearing
        if self._disk_block_store and block.cache_data is not None:
            self._disk_block_store.write_block_async(
                block.block_hash, block.cache_data, block.token_count
            )
        block.reset_hash()
        block.cache_data = None  # Frees tensor references
        ...
```

### 2. Prefix Hash Lookup Miss — `paged_cache.py:867`

Inside `get_computed_blocks()`, when `cached_block_hash_to_block.get_block(block_hash)` returns None:

```python
cached_block = self.cached_block_hash_to_block.get_block(block_hash)
if cached_block is None:
    # NEW: Check disk L2 before declaring cache miss
    if self._disk_block_store:
        disk_data = self._disk_block_store.read_block(block_hash)
        if disk_data is not None:
            # Allocate a RAM block and populate from disk
            new_block = self.allocate_block()
            new_block.cache_data = disk_data
            new_block.block_hash = block_hash
            new_block.token_count = len(disk_data[0][1]) if disk_data else 0  # infer from tensor
            self.cached_block_hash_to_block.insert(block_hash, new_block)
            cached_block = new_block
            self.stats.disk_hits += 1
            # Continue matching next blocks...
    if cached_block is None:
        self.stats.cache_misses += 1
        break
```

### 3. Block Store After Generation — `prefix_cache.py:601`

In `store_cache()`, after `block.cache_data = block_kv_data`:

```python
if block_kv_data:
    block.cache_data = block_kv_data
    # NEW: Write-through to disk for persistence
    if self._disk_block_store and block.block_hash:
        self._disk_block_store.write_block_async(
            block.block_hash, block_kv_data, block.token_count
        )
```

### 4. Config Flags — `scheduler.py:67`

Add to `SchedulerConfig`:

```python
@dataclass
class SchedulerConfig:
    # ... existing fields ...

    # Block-level disk cache (L2)
    enable_block_disk_cache: bool = False
    block_disk_cache_dir: Optional[str] = None  # Default: ~/.cache/vllm-mlx/block-cache/
    block_disk_cache_max_gb: float = 10.0       # Max disk usage per model
    block_disk_cache_quantize: bool = True       # Quantize to 4-bit on disk for faster I/O
```

### 5. CLI Flags — `cli.py`

```
--enable-block-disk-cache    Enable block-level disk KV cache persistence
--block-disk-cache-dir DIR   Directory for block cache files
--block-disk-cache-max-gb N  Maximum disk usage in GB (default: 10)
--no-block-disk-quantize     Store full precision on disk (slower I/O, exact values)
```

### 6. Panel UI — Session Config Form

Add toggle in session settings:
- "Disk KV Cache" checkbox (maps to `--enable-block-disk-cache`)
- Size slider: 1-50 GB (maps to `--block-disk-cache-max-gb`)

---

## New File: `block_disk_store.py`

The core new module implementing the disk block store:

```python
"""
Block-level disk persistence for paged KV cache.

Provides an L2 disk tier behind the L1 in-memory PagedCacheManager.
Blocks are stored as safetensors files indexed by their chain hash.
Supports optional on-disk quantization for faster I/O.
"""

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx


class BlockDiskStore:
    """Content-addressable block storage on disk."""

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int = 10 * 1024**3,  # 10 GB
        quantize_on_disk: bool = True,
        quantize_bits: int = 4,
        quantize_group_size: int = 64,
    ):
        self.cache_dir = Path(cache_dir)
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.quantize_on_disk = quantize_on_disk
        self.quantize_bits = quantize_bits
        self.quantize_group_size = quantize_group_size

        # SQLite index
        self.db_path = self.cache_dir / "index.db"
        self._init_db()

        # Background write thread
        self._write_queue = []
        self._write_lock = threading.Lock()
        self._writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._writer_thread.start()

    def _init_db(self):
        """Initialize SQLite index with WAL mode."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_hash   TEXT PRIMARY KEY,
                file_name    TEXT NOT NULL,
                num_tokens   INTEGER NOT NULL,
                num_layers   INTEGER NOT NULL,
                dtype        TEXT NOT NULL,
                file_size    INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_lru ON blocks(last_accessed ASC)")
        conn.commit()
        conn.close()

    def _hash_to_path(self, block_hash: str) -> Path:
        """Shard by first 2 chars for filesystem efficiency."""
        prefix = block_hash[:2]
        shard_dir = self.blocks_dir / prefix
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{block_hash}.safetensors"

    def read_block(self, block_hash: bytes) -> Optional[List[Tuple]]:
        """
        Read a block from disk by its chain hash.
        Returns cache_data in the same format as CacheBlock.cache_data,
        or None if not found.
        """
        hash_hex = block_hash.hex()

        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT file_name, dtype FROM blocks WHERE block_hash = ?",
            (hash_hex,)
        ).fetchone()

        if row is None:
            conn.close()
            return None

        file_name, dtype = row
        file_path = self.cache_dir / file_name

        if not file_path.exists():
            # Stale index entry — clean up
            conn.execute("DELETE FROM blocks WHERE block_hash = ?", (hash_hex,))
            conn.commit()
            conn.close()
            return None

        # Update access metadata
        conn.execute(
            "UPDATE blocks SET last_accessed = ?, access_count = access_count + 1 WHERE block_hash = ?",
            (time.time(), hash_hex)
        )
        conn.commit()
        conn.close()

        # Load tensors
        try:
            data = mx.load(str(file_path))
            return self._deserialize_block(data, dtype)
        except Exception as e:
            print(f"[DISK-CACHE] Failed to load block {hash_hex[:12]}: {e}")
            return None

    def write_block_async(self, block_hash: bytes, cache_data: List[Tuple], token_count: int):
        """Queue a block for background writing. Non-blocking."""
        with self._write_lock:
            self._write_queue.append((block_hash, cache_data, token_count))

    def _background_writer(self):
        """Background thread that writes queued blocks to disk."""
        while True:
            time.sleep(0.1)  # 100ms poll interval

            with self._write_lock:
                if not self._write_queue:
                    continue
                batch = self._write_queue[:]
                self._write_queue.clear()

            for block_hash, cache_data, token_count in batch:
                try:
                    self._write_block(block_hash, cache_data, token_count)
                except Exception as e:
                    hash_hex = block_hash.hex()[:12] if isinstance(block_hash, bytes) else str(block_hash)[:12]
                    print(f"[DISK-CACHE] Failed to write block {hash_hex}: {e}")

            # Check eviction after writes
            self._maybe_evict()

    def _write_block(self, block_hash: bytes, cache_data: List[Tuple], token_count: int):
        """Write a single block to disk."""
        hash_hex = block_hash.hex()
        file_path = self._hash_to_path(hash_hex)
        rel_path = file_path.relative_to(self.cache_dir)

        # Skip if already exists
        conn = sqlite3.connect(str(self.db_path))
        exists = conn.execute(
            "SELECT 1 FROM blocks WHERE block_hash = ?", (hash_hex,)
        ).fetchone()
        if exists:
            conn.close()
            return

        # Serialize and optionally quantize
        tensors, dtype = self._serialize_block(cache_data)

        # Save as safetensors
        mx.save_safetensors(str(file_path), tensors)
        file_size = file_path.stat().st_size
        num_layers = sum(1 for t in cache_data if t[0] != "skip")

        # Index
        conn.execute(
            """INSERT OR REPLACE INTO blocks
               (block_hash, file_name, num_tokens, num_layers, dtype, file_size, created_at, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (hash_hex, str(rel_path), token_count, num_layers, dtype,
             file_size, time.time(), time.time())
        )
        conn.commit()
        conn.close()

    def _serialize_block(self, cache_data: List[Tuple]) -> Tuple[Dict[str, mx.array], str]:
        """
        Convert CacheBlock.cache_data to a flat dict of named tensors for safetensors.
        Returns (tensor_dict, dtype_string).
        """
        tensors = {}
        dtype = "kv"  # default

        for i, layer_data in enumerate(cache_data):
            if layer_data[0] == "skip":
                continue
            elif layer_data[0] == "kv":
                _, keys, values = layer_data
                tensors[f"layer_{i}_keys"] = keys
                tensors[f"layer_{i}_values"] = values
            elif layer_data[0] == "quantized_kv":
                _, keys_tuple, values_tuple, meta = layer_data
                dtype = "quantized_kv"
                # keys_tuple = (data, scales, zeros)
                tensors[f"layer_{i}_keys_data"] = keys_tuple[0]
                tensors[f"layer_{i}_keys_scales"] = keys_tuple[1]
                tensors[f"layer_{i}_keys_zeros"] = keys_tuple[2]
                tensors[f"layer_{i}_values_data"] = values_tuple[0]
                tensors[f"layer_{i}_values_scales"] = values_tuple[1]
                tensors[f"layer_{i}_values_zeros"] = values_tuple[2]
            elif layer_data[0] == "rotating_kv":
                _, keys, values, max_size, keep = layer_data
                dtype = "rotating_kv"
                tensors[f"layer_{i}_keys"] = keys
                tensors[f"layer_{i}_values"] = values
                # max_size and keep stored as scalar tensors
                tensors[f"layer_{i}_max_size"] = mx.array([max_size])
                tensors[f"layer_{i}_keep"] = mx.array([keep])

        return tensors, dtype

    def _deserialize_block(self, data: Dict[str, mx.array], dtype: str) -> List[Tuple]:
        """Reconstruct CacheBlock.cache_data from loaded safetensors dict."""
        # Determine layer count from keys
        layer_indices = set()
        for key in data:
            parts = key.split("_")
            if len(parts) >= 2 and parts[0] == "layer":
                layer_indices.add(int(parts[1]))

        cache_data = []
        for i in sorted(layer_indices):
            if dtype == "kv":
                keys = data.get(f"layer_{i}_keys")
                values = data.get(f"layer_{i}_values")
                if keys is not None and values is not None:
                    cache_data.append(("kv", keys, values))
            elif dtype == "quantized_kv":
                cache_data.append(("quantized_kv",
                    (data[f"layer_{i}_keys_data"], data[f"layer_{i}_keys_scales"], data[f"layer_{i}_keys_zeros"]),
                    (data[f"layer_{i}_values_data"], data[f"layer_{i}_values_scales"], data[f"layer_{i}_values_zeros"]),
                    {}  # meta reconstructed from context
                ))
            elif dtype == "rotating_kv":
                cache_data.append(("rotating_kv",
                    data[f"layer_{i}_keys"],
                    data[f"layer_{i}_values"],
                    int(data[f"layer_{i}_max_size"].item()),
                    int(data[f"layer_{i}_keep"].item())
                ))

        return cache_data

    def _maybe_evict(self):
        """Evict LRU blocks if total disk usage exceeds max."""
        conn = sqlite3.connect(str(self.db_path))
        total = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM blocks").fetchone()[0]

        if total <= self.max_size_bytes:
            conn.close()
            return

        # Evict LRU blocks until under budget
        target = int(self.max_size_bytes * 0.8)  # Free 20% headroom
        rows = conn.execute(
            "SELECT block_hash, file_name, file_size FROM blocks ORDER BY last_accessed ASC"
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
            print(f"[DISK-CACHE] Evicted {evicted} blocks (freed to {total / 1024**3:.1f} GB)")

    def stats(self) -> Dict:
        """Return cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(file_size), 0), COALESCE(SUM(access_count), 0) FROM blocks"
        ).fetchone()
        conn.close()
        return {
            "blocks": row[0],
            "size_bytes": row[1],
            "size_gb": row[1] / 1024**3,
            "total_hits": row[2],
        }

    def clear(self):
        """Clear all cached blocks."""
        import shutil
        if self.blocks_dir.exists():
            shutil.rmtree(self.blocks_dir)
            self.blocks_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM blocks")
        conn.commit()
        conn.close()
```

---

## Implementation Sequence

### Phase 1: BlockDiskStore module (2-3 hours)
1. Create `vllm_mlx/block_disk_store.py` with read/write/evict/stats
2. Serialize/deserialize all cache_data tuple types (kv, quantized_kv, rotating_kv, skip)
3. Background writer thread with async queue
4. SQLite index with WAL mode
5. LRU eviction by total disk size

### Phase 2: Integration with PagedCacheManager (2-3 hours)
1. Add `_disk_block_store: Optional[BlockDiskStore]` to `PagedCacheManager.__init__`
2. Hook into `_maybe_evict_cached_block()` — write to disk before eviction
3. Hook into `get_computed_blocks()` — check disk on L1 miss
4. Hook into `store_cache()` — write-through to disk after generation
5. Add `disk_hits`, `disk_writes`, `disk_evictions` to `CacheStats`

### Phase 3: Config and CLI (1 hour)
1. Add flags to `SchedulerConfig`
2. Add CLI args to `cli.py`
3. Wire through `BatchedEngine` → `Scheduler` → `PagedCacheManager`

### Phase 4: Panel UI (1 hour)
1. Add "Disk KV Cache" toggle to SessionConfigForm
2. Add cache size slider (1-50 GB)
3. Pass `--enable-block-disk-cache` in `buildArgs`
4. Show disk cache stats in session health info

### Phase 5: Testing and optimization
1. Verify round-trip: generate → evict → reload → generate matches
2. Benchmark: disk load time vs recomputation time for various prompt lengths
3. Test with quantized KV cache (most common path)
4. Test with hybrid models (Mamba + KV)
5. Memory pressure testing — verify eviction doesn't cause OOM
6. Startup time with warm cache vs cold

---

## Performance Expectations (M4 Max)

| Metric | Expected |
|--------|----------|
| SSD read speed | 7.4 GB/s sequential |
| Block size (16 tokens, 32 layers, FP16) | ~2-8 MB depending on model |
| Block load time | ~0.3-1.0 ms per block |
| 1000-token prefix load (62 blocks) | ~20-60 ms from disk |
| vs recomputation (1000 tokens) | ~200-500 ms (model dependent) |
| Speedup for cached prompts | **5-10x faster prefill** |
| Disk usage per 1000 tokens cached | ~125-500 MB (FP16), ~30-125 MB (4-bit quantized) |

---

## Key Design Decisions

1. **Block-level, not prompt-level**: Allows partial prefix reuse. Changing the last sentence of a 10k-token prompt still reuses 95%+ of cached blocks.

2. **Content-addressable chain hashes**: Reuses existing `BlockHash` from `PagedCacheManager`. Same tokens at the same position always produce the same hash. Stable across restarts.

3. **Async writes**: Background thread prevents disk I/O from blocking inference. The write queue is drained every 100ms.

4. **Optional on-disk quantization**: 4-bit quantized blocks are 4x smaller, cutting I/O time proportionally. The CPU handles dequantization while GPU does inference.

5. **SQLite index**: WAL mode for concurrent read/write. LRU eviction by last_accessed timestamp. Access counting for analytics.

6. **Directory sharding**: First 2 chars of hash as subdirectory prevents filesystem slowdown with millions of small files.

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `vllm_mlx/block_disk_store.py` | **CREATE** | Core disk store module |
| `vllm_mlx/paged_cache.py` | MODIFY | Add disk store hooks at eviction, lookup, and store |
| `vllm_mlx/prefix_cache.py` | MODIFY | Pass disk store to `BlockAwarePrefixCache` |
| `vllm_mlx/scheduler.py` | MODIFY | Instantiate `BlockDiskStore`, add config flags |
| `vllm_mlx/cli.py` | MODIFY | Add CLI arguments |
| `vllm_mlx/engine/batched.py` | MODIFY | Wire config through |
| `panel/src/main/sessions.ts` | MODIFY | Pass new flags in `buildArgs` |
| `panel/src/renderer/.../SessionConfigForm.tsx` | MODIFY | Add UI controls |

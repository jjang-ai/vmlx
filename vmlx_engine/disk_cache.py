# SPDX-License-Identifier: Apache-2.0
"""
Disk-based prompt cache persistence for vmlx-engine.

L2 cache tier: when the in-memory L1 prefix/paged cache misses, the
scheduler checks this disk cache before doing a full prefill.

Two storage formats:
- **TQ-native** (for JANG/TurboQuant models): tq_disk_store.serialize_tq_cache()
  extracts 3-bit compressed data directly — 26x smaller files. Written via
  mx.save_safetensors on the main thread. Detected by __tq_native__ metadata.
- **Standard** (for non-TQ models): safetensors.numpy.save_file with
  pre-serialized numpy arrays from .state property.

Architecture:
- Background writer thread: store() pre-serializes on main thread (Metal-safe),
  enqueues atomic rename + SQLite update for background
- SQLite connection pool: WAL mode, reuses connections
- TQ-native deserialization requires TurboQuantEncoder for codebook decode —
  creates temporary TQ cache to access .key_encoder/.value_encoder
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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def _hash_tokens_with_marker(
    tokens: List[int],
    cache_extra_marker: Optional[str],
) -> str:
    # Use SHA-256 of the token list serialized as compact JSON
    data = json.dumps(tokens, separators=(",", ":")).encode()
    if cache_extra_marker is not None:
        data += b"\0vmlx-cache-extra-v1\0" + cache_extra_marker.encode(
            "utf-8",
            "surrogatepass",
        )
    return hashlib.sha256(data).hexdigest()


def _hash_tokens(tokens: List[int], cache_extra_keys: Any = None) -> str:
    """Create a stable hash of tokens plus non-token cache discriminators."""
    from .cache_key import canonical_cache_extra_marker

    return _hash_tokens_with_marker(
        tokens,
        canonical_cache_extra_marker(cache_extra_keys),
    )


def _hash_token_prefixes_with_marker(
    tokens: List[int], lengths: Iterable[int], cache_extra_marker: Optional[str]
) -> Dict[int, str]:
    """Hash overlapping prefixes without serializing the shared tokens again.

    Each digest is byte-identical to the existing compact-JSON cache key,
    including its closing bracket and optional discriminator. Only the
    unclosed JSON prefix is shared; copying the hash state preserves every
    existing on-disk key and N-1 payload identity.
    """
    suffix = b"]"
    if cache_extra_marker is not None:
        suffix += b"\0vmlx-cache-extra-v1\0" + cache_extra_marker.encode(
            "utf-8", "surrogatepass"
        )
    state = hashlib.sha256(b"[")
    previous = 0
    result = {}
    for length in sorted(set(lengths)):
        if not 0 <= length <= len(tokens):
            raise ValueError("cache prefix length is outside the token sequence")
        if length > previous:
            if previous:
                state.update(b",")
            state.update(
                json.dumps(tokens[previous:length], separators=(",", ":")).encode()[1:-1]
            )
        digest = state.copy()
        digest.update(suffix)
        result[length] = digest.hexdigest()
        previous = length
    return result


def _runtime_cache_fingerprint() -> str:
    try:
        from .prefix_cache import runtime_cache_fingerprint

        return runtime_cache_fingerprint()
    except Exception:
        return "unknown"


def _metadata_runtime_fingerprint(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    return (
        metadata.get("__runtime_cache_fingerprint__")
        or metadata.get("runtime_cache_fingerprint")
        or metadata.get("1.runtime_cache_fingerprint")
    )


def _metadata_cache_classes(metadata: Optional[Dict[str, Any]]) -> List[str]:
    """Return cache class names from flattened mlx-lm prompt-cache metadata."""
    if not isinstance(metadata, dict):
        return []
    classes: List[Tuple[int, str]] = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.startswith("2."):
            continue
        try:
            idx = int(key.split(".", 1)[1])
        except (IndexError, ValueError):
            continue
        if isinstance(value, str) and value:
            classes.append((idx, value))
    return [name for _, name in sorted(classes)]


# save_metadata key under which store() records which flattened arrays were
# widened for serialization (bf16 -> f32; numpy has no bf16). In the file's
# flat metadata namespace it appears as "1.widened_dtypes" because
# save_metadata is element 1 of [cache_info, save_metadata, cache_classes].
_WIDENED_DTYPES_META_KEY = "widened_dtypes"
_BITCAST_DTYPES_META_KEY = "bitcast_dtypes"


def _metadata_widened_dtypes(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return the {flattened-array-key: original-dtype-name} widening record.

    Files written before the record existed have no entry and return {} —
    they keep loading exactly as before (widened-but-exact f32), never an
    error.
    """
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get(f"1.{_WIDENED_DTYPES_META_KEY}") or metadata.get(
        _WIDENED_DTYPES_META_KEY
    )
    if not raw or not isinstance(raw, str):
        return {}
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Disk cache widened-dtype record is malformed; "
            "leaving restored arrays widened"
        )
        return {}
    if not isinstance(record, dict):
        return {}
    return {
        k: v
        for k, v in record.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _restore_widened_dtypes(
    raw_arrays: Dict[str, Any], file_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Cast arrays widened at store time back to their recorded original dtype.

    Reverses ONLY the keys named by the store-side record, so TQ float16
    decompressions and genuinely-f32 states are untouched. A restored cache
    must re-enter generation in its compute dtype: mlx-lm caches extend via
    mx.concatenate, which silently promotes a bf16 model's fresh KV into the
    restored f32 — the cache then stays f32 for its whole life (2x live KV
    bytes, attention numerics off the fresh-compute bf16 path). Measured
    2026-08-12: restored state 2.00x bytes, extended buffer 2.12x, attention
    output diverges from fresh compute at bf16 rounding scale.
    """
    record = _metadata_widened_dtypes(file_metadata)
    if not record or not _HAS_MLX:
        return raw_arrays
    restored = dict(raw_arrays)
    for key, dtype_name in record.items():
        arr = restored.get(key)
        target = getattr(mx, dtype_name, None)
        if (
            arr is None
            or not isinstance(arr, mx.array)
            or not isinstance(target, mx.Dtype)
        ):
            continue
        if arr.dtype != target:
            restored[key] = arr.astype(target)
    return restored


def _metadata_bitcast_dtypes(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return arrays persisted as raw integer bits instead of widened floats."""

    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get(f"1.{_BITCAST_DTYPES_META_KEY}") or metadata.get(
        _BITCAST_DTYPES_META_KEY
    )
    if not raw or not isinstance(raw, str):
        return {}
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Disk cache bitcast-dtype record is malformed")
        return {}
    if not isinstance(record, dict):
        return {}
    return {
        key: dtype
        for key, dtype in record.items()
        if isinstance(key, str) and isinstance(dtype, str)
    }


def _restore_bitcast_dtypes(
    raw_arrays: Dict[str, Any], file_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Reinterpret raw uint16 payloads as their exact original BF16 bits."""

    record = _metadata_bitcast_dtypes(file_metadata)
    if not record or not _HAS_MLX:
        return raw_arrays
    restored = dict(raw_arrays)
    for key, dtype_name in record.items():
        arr = restored.get(key)
        target = getattr(mx, dtype_name, None)
        if arr is None or not isinstance(arr, mx.array) or target is None:
            continue
        if arr.dtype == mx.uint16 and target == mx.bfloat16:
            restored[key] = arr.view(mx.bfloat16)
        elif arr.dtype != target:
            restored[key] = arr.astype(target)
    return restored


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
    RotatingKVCache, ArraysCache/MambaCache, CacheList, TurboQuantKVCache).

    TQ-native storage:
    When TurboQuantKVCache layers are detected with compressed data, the store
    uses TQ-native serialization (tq_disk_store.py) which saves the 3-bit packed
    compressed form directly — 26x smaller than the decompressed float16 state.
    On fetch, compressed data is decoded and wrapped in KVCache objects, then
    the caller's _recompress_to_tq() converts back to TurboQuantKVCache.

    Features:
    - Background writer thread: store() is non-blocking
    - SQLite connection pool: avoids per-operation connection overhead
    - TQ-native compression: 26x disk savings for JANG models
    - Graceful shutdown: flushes pending writes

    Args:
        cache_dir: Directory to store cache files. Created if it doesn't exist.
        max_size_gb: Maximum total cache size in GB. Oldest entries are evicted
            when this limit is exceeded. 0 = unlimited.
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 10.0,
        expected_num_layers: Optional[int] = None,
        required_cache_class: Optional[str] = None,
        required_cache_classes: Optional[Iterable[str]] = None,
        allow_tq_native: Optional[bool] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024) if max_size_gb > 0 else 0

        # Expected layer count from model config. Used by the safetensors
        # header validator to reject wrong-model L2 files.
        self._expected_num_layers: Optional[int] = expected_num_layers
        # Families with first-class cache subclasses (MiniMax-M3 MSA, DSV4, etc.)
        # must not accept stale legacy KV-only files from the same prompt hash.
        self._required_cache_class: Optional[str] = required_cache_class
        required = {str(name) for name in (required_cache_classes or ()) if name}
        if required_cache_class:
            required.add(str(required_cache_class))
        self._required_cache_classes: frozenset[str] = frozenset(required)
        if allow_tq_native is None:
            allow_tq_native = os.environ.get("VMLX_DISABLE_TQ_KV", "").lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }
        # None/q4/q8 explicitly disable the native TurboQuant route.  That
        # choice must apply to existing L2 records too, not only newly-created
        # live cache objects.  The token hash has one index slot, so an
        # incompatible TQ record is evicted to let the Off run write a standard
        # replacement after its clean prefill.
        self._allow_tq_native = bool(allow_tq_native)

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
        # TQ-native stats: track how many stores/hits used TQ compressed format
        self.tq_native_stores = 0
        self.tq_native_hits = 0
        # Flag set by fetch() to indicate last fetch was TQ-native.
        # Checked by scheduler to annotate cache_detail as "disk+tq".
        self._last_fetch_tq_native = False
        self._load_executor = None
        self._load_worker_prefix = "llm-worker"

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
                metadata TEXT,
                cache_type TEXT DEFAULT 'assistant'
            )
        """)
        # F3: cache_type column for L2 disk cache. Migrate older DBs that
        # were created before the column existed (defaults to 'assistant'
        # so existing entries don't lose system-pinning eligibility — they
        # just need to be re-stored to be properly tagged).
        try:
            conn.execute(
                "ALTER TABLE cache_entries ADD COLUMN cache_type TEXT DEFAULT 'assistant'"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Prompt-cache payloads are deliberately N-1: the SQLite key describes
        # the full rendered prompt while the serialized cache covers every
        # token except its final generation sentinel.  Multi-turn templates can
        # legitimately replace that last sentinel (for example <think> with
        # </think>) while preserving the entire reusable payload prefix.  Keep
        # a second hash for the state the file actually owns so SSD/L2 lookup
        # can find that reusable boundary without storing raw prompt tokens.
        try:
            conn.execute(
                "ALTER TABLE cache_entries ADD COLUMN payload_prefix_hash TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute(
                "ALTER TABLE cache_entries ADD COLUMN cache_extra_marker TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON cache_entries(last_accessed)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_type
            ON cache_entries(cache_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_payload_prefix_hash
            ON cache_entries(payload_prefix_hash, num_tokens)
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

    def _total_tokens(self) -> int:
        """Get total prompt tokens represented by persistent L2 entries."""
        conn = self._pool.get()
        try:
            result = conn.execute(
                "SELECT COALESCE(SUM(num_tokens), 0) FROM cache_entries"
            ).fetchone()[0]
            return int(result)
        finally:
            self._pool.put(conn)

    def set_load_executor(
        self,
        executor: Any,
        worker_name_prefix: Optional[str] = None,
    ) -> None:
        """Run MLX-backed disk loads on the model-owning worker thread.

        Safetensors load, TQ decode, and cache-class reconstruction create MLX
        arrays. Those arrays retain the creating thread's stream identity, so a
        cache loaded on the API/event-loop thread cannot safely be consumed by
        the dedicated LLM/MLLM worker. The executor is single-threaded; execute
        inline when fetch already runs on that worker to avoid self-deadlock.
        """
        self._load_executor = executor
        if worker_name_prefix is None:
            worker_name_prefix = (
                getattr(executor, "_thread_name_prefix", "") or "llm-worker"
            )
        self._load_worker_prefix = worker_name_prefix

    def fetch(
        self,
        tokens: List[int],
        cache_extra_keys: Any = None,
    ) -> Optional[List[Any]]:
        """Load a disk cache with the same discriminator used at store time."""
        if cache_extra_keys is None:
            return self._fetch_on_owner(tokens)
        return self._fetch_on_owner(
            tokens,
            cache_extra_keys=cache_extra_keys,
        )

    def _fetch_on_owner(
        self,
        tokens: List[int],
        *,
        token_hash_override: Optional[str] = None,
        cache_extra_keys: Any = None,
    ) -> Optional[List[Any]]:
        """Load one indexed record, optionally by its stored full-key hash.

        The override is used only after ``fetch_longest_prefix`` has matched a
        record's N-1 payload hash.  ``tokens`` remains the current prompt prefix
        for accurate diagnostics; all file/class/runtime validation still runs
        through the ordinary fetch implementation.
        """
        executor = getattr(self, "_load_executor", None)
        prefix = getattr(self, "_load_worker_prefix", "llm-worker")
        if executor is None or threading.current_thread().name.startswith(prefix):
            if token_hash_override is None:
                if cache_extra_keys is None:
                    return self._fetch_impl(tokens)
                return self._fetch_impl(tokens, cache_extra_keys=cache_extra_keys)
            return self._fetch_impl(tokens, token_hash_override)
        try:
            if token_hash_override is None:
                if cache_extra_keys is None:
                    return executor.submit(self._fetch_impl, tokens).result()
                return executor.submit(
                    self._fetch_impl,
                    tokens,
                    None,
                    cache_extra_keys,
                ).result()
            return executor.submit(
                self._fetch_impl,
                tokens,
                token_hash_override,
            ).result()
        except Exception as exc:
            logger.warning(
                "Disk cache worker-owned load failed; treating as miss: %s",
                exc,
            )
            with self._stats_lock:
                self.misses += 1
            return None

    def _fetch_indexed_hash(
        self,
        token_hash: str,
        current_prefix_tokens: List[int],
    ) -> Optional[List[Any]]:
        """Load a record selected by its N-1 payload-prefix index."""
        return self._fetch_on_owner(
            current_prefix_tokens,
            token_hash_override=token_hash,
        )

    def _fetch_impl(
        self,
        tokens: List[int],
        token_hash_override: Optional[str] = None,
        cache_extra_keys: Any = None,
    ) -> Optional[List[Any]]:
        """
        Look up a cached KV state for the given token sequence.

        Returns the cache object list if found, None on miss.
        The returned cache is ready to be used as prompt_cache in BatchGenerator.
        """
        token_hash = token_hash_override or _hash_tokens(tokens, cache_extra_keys)

        conn = self._pool.get()
        try:
            row = conn.execute(
                "SELECT file_name, cache_type FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()

            if row is None:
                with self._stats_lock:
                    self.misses += 1
                return None

            file_name = row[0]
            # F3: surface the stored cache_type so the L1 backfill in scheduler
            # can re-tag the entry with the same role (preserves system pinning
            # across restart).
            self._last_fetch_cache_type: str = row[1] if len(row) > 1 and row[1] else "assistant"
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
                # Validate safetensors HEADER metadata BEFORE mx.load. Skips
                # files whose declared shapes describe multi-hundred-GB tensors
                # (corrupt entry from older schema or interrupted write would
                # otherwise blow up Metal here).
                try:
                    from .cache_record_validator import (
                        reject_safetensors_or_warn,
                    )
                except Exception:
                    reject_safetensors_or_warn = None
                if reject_safetensors_or_warn is not None:
                    if not reject_safetensors_or_warn(
                        str(file_path),
                        expected_num_layers=getattr(self, "_expected_num_layers", None),
                        source=f"L2-prompt-disk-header:{token_hash[:12]}",
                        delete_on_reject=True,
                    ):
                        # File deleted on reject; clear index too.
                        conn.execute(
                            "DELETE FROM cache_entries WHERE token_hash = ?",
                            (token_hash,),
                        )
                        conn.commit()
                        with self._stats_lock:
                            self.misses += 1
                        return None

                # ─── Step 1: Load raw safetensors + metadata header ───
                # Always load raw first so we can check for TQ-native format
                # before falling back to mlx-lm's load_prompt_cache().
                raw_arrays, file_metadata = mx.load(
                    str(file_path), return_metadata=True
                )
                cache_classes = _metadata_cache_classes(file_metadata)
                required_classes = set(
                    getattr(self, "_required_cache_classes", ()) or ()
                )
                required_class = getattr(self, "_required_cache_class", None)
                if required_class:
                    required_classes.add(required_class)
                missing_classes = sorted(required_classes - set(cache_classes))
                if missing_classes:
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    conn.execute(
                        "DELETE FROM cache_entries WHERE token_hash = ?",
                        (token_hash,),
                    )
                    conn.commit()
                    with self._stats_lock:
                        self.misses += 1
                    logger.warning(
                        "Disk prompt cache class mismatch; treating as miss "
                        "(missing=%s, stored=%s, file=%s)",
                        ",".join(missing_classes),
                        ",".join(cache_classes[:8]) or "missing",
                        file_name,
                    )
                    return None
                stored_runtime = _metadata_runtime_fingerprint(file_metadata)
                current_runtime = _runtime_cache_fingerprint()
                if stored_runtime != current_runtime:
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    conn.execute(
                        "DELETE FROM cache_entries WHERE token_hash = ?",
                        (token_hash,),
                    )
                    conn.commit()
                    with self._stats_lock:
                        self.misses += 1
                    logger.info(
                        "Disk prompt cache runtime fingerprint mismatch; "
                        "treating as miss (stored=%s current=%s)",
                        stored_runtime or "missing",
                        current_runtime,
                    )
                    return None

                # ─── Step 2: Check for TQ-native format ───
                # TQ-native files have "__tq_native__" = "true" in metadata.
                # These store 3-bit compressed TQ data directly (26x smaller).
                is_tq_native = (
                    isinstance(file_metadata, dict)
                    and file_metadata.get("__tq_native__") == "true"
                )

                if (
                    not self._allow_tq_native
                    and (is_tq_native or "TurboQuantKVCache" in cache_classes)
                ):
                    self._last_fetch_tq_native = False
                    try:
                        conn.execute(
                            "DELETE FROM cache_entries WHERE token_hash = ?",
                            (token_hash,),
                        )
                        conn.commit()
                        file_path.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.warning(
                            "Failed to evict incompatible TQ-native disk cache %s: %s",
                            file_name,
                            exc,
                        )
                    with self._stats_lock:
                        self.misses += 1
                    logger.info(
                        "Evicted TQ-native disk cache %s because persisted TQ reads are disabled",
                        file_name,
                    )
                    return None

                if is_tq_native:
                    # TQ-native: decode compressed data → KVCache objects.
                    # The caller's _recompress_to_tq() will convert back to
                    # TurboQuantKVCache using the model's make_cache() template.
                    from .tq_disk_store import deserialize_tq_cache
                    cache = deserialize_tq_cache(raw_arrays, file_metadata)
                    is_tq_hit = True
                    self._last_fetch_tq_native = True
                    logger.info(
                        f"TQ-native disk cache loaded: {len(cache)} layers "
                        f"from {file_name}"
                    )
                else:
                    # ─── Step 3: Standard format (mlx-lm or legacy TQ remap) ───
                    is_tq_hit = False
                    self._last_fetch_tq_native = False
                    # Reverse the serialization widening BEFORE cache objects
                    # are built: mlx-lm's from_state has no dtype hook, and a
                    # cache restored as f32 stays f32 for its whole life
                    # (update_and_fetch extends through mx.concatenate, which
                    # promotes bf16+f32 to f32). Old files carry no record and
                    # pass through unchanged.
                    raw_arrays = _restore_bitcast_dtypes(
                        raw_arrays, file_metadata
                    )
                    raw_arrays = _restore_widened_dtypes(
                        raw_arrays, file_metadata
                    )
                    try:
                        # Same construction as mlx-lm's load_prompt_cache, but
                        # from the arrays this function already loaded (and
                        # dtype-restored) instead of a second mx.load of the
                        # same file. Exception parity with load_prompt_cache:
                        # it resolves classes via globals()[c] (KeyError);
                        # default-less getattr raises AttributeError — both
                        # feed the legacy remap below, as before.
                        import mlx_lm.models.cache as _cache_mod
                        from mlx_lm.utils import tree_unflatten
                        _arrays_unflat = tree_unflatten(
                            list(raw_arrays.items())
                        )
                        _meta_unflat = tree_unflatten(
                            list(file_metadata.items())
                        )
                        _info, _save_meta, _classes = _meta_unflat
                        cache = [
                            getattr(_cache_mod, _cls).from_state(
                                _state, _meta_state
                            )
                            for _cls, _state, _meta_state in zip(
                                _classes, _arrays_unflat, _info
                            )
                        ]
                    except (KeyError, AttributeError):
                        # TurboQuantKVCache not in mlx-lm globals — remap to KVCache.
                        # This handles old-format disk caches written before TQ-native.
                        from mlx_lm.models.cache import KVCache as _KVC
                        from mlx_lm.utils import tree_unflatten
                        arrays = tree_unflatten(list(raw_arrays.items()))
                        cache_metadata_unflat = tree_unflatten(
                            list(file_metadata.items())
                        )
                        info, _meta, classes = cache_metadata_unflat
                        cache = []
                        for c, state, meta_state in zip(classes, arrays, info):
                            if c == 'TurboQuantKVCache':
                                kv = _KVC()
                                if isinstance(state, (tuple, list)) and len(state) == 2:
                                    kv.keys, kv.values = state[0], state[1]
                                    kv.offset = int(meta_state[0]) if meta_state else 0
                                cache.append(kv)
                            elif c == "MiniMaxM3SparseCache":
                                if not isinstance(state, (tuple, list)) or len(state) != 3:
                                    logger.warning(
                                        "Disk cache MiniMax-M3 entry missing "
                                        "keys/values/idx_keys; treating as miss"
                                    )
                                    with self._stats_lock:
                                        self.misses += 1
                                    return None
                                if state[2] is None:
                                    logger.warning(
                                        "Disk cache MiniMax-M3 entry missing "
                                        "idx_keys; treating as miss"
                                    )
                                    with self._stats_lock:
                                        self.misses += 1
                                    return None
                                from .models.minimax_m3.cache import (
                                    restore_minimax_m3_sparse,
                                )

                                cache.append(
                                    restore_minimax_m3_sparse(
                                        state[0],
                                        state[1],
                                        state[2],
                                    )
                                )
                            else:
                                import mlx_lm.models.cache as _cache_mod
                                cls = getattr(_cache_mod, c, _KVC)
                                try:
                                    cache.append(cls.from_state(state, meta_state))
                                except Exception:
                                    kv = _KVC()
                                    cache.append(kv)
                        restored_classes = [type(c).__name__ for c in cache]
                        if "MiniMaxM3SparseCache" in restored_classes:
                            logger.info(
                                "Disk cache loaded with MiniMax-M3 sparse "
                                "restore: %d layers (%d MSA sparse)",
                                len(cache),
                                restored_classes.count("MiniMaxM3SparseCache"),
                            )
                        elif "TurboQuantKVCache" in classes:
                            logger.info(
                                "Disk cache loaded with TQ→KVCache remap: "
                                "%d layers",
                                len(cache),
                            )
                        else:
                            logger.info(
                                "Disk cache loaded with custom cache restore: "
                                "%d layers",
                                len(cache),
                            )

                # ─── Step 4: Update access metadata + stats ───
                now = time.time()
                conn.execute(
                    "UPDATE cache_entries SET last_accessed = ?, "
                    "access_count = access_count + 1 "
                    "WHERE token_hash = ?",
                    (now, token_hash)
                )
                conn.commit()

                with self._stats_lock:
                    self.hits += 1
                    if is_tq_hit:
                        self.tq_native_hits += 1

                try:
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    restored_classes = {type(c).__name__ for c in cache}
                    if is_tq_hit:
                        fmt = "TQ-native"
                    elif "MiniMaxM3SparseCache" in restored_classes:
                        fmt = "m3-sparse"
                    else:
                        fmt = "standard"
                    logger.info(
                        f"Disk cache hit ({fmt}): {len(tokens)} tokens, "
                        f"file={file_name} ({size_mb:.1f}MB)"
                    )
                except OSError:
                    logger.info(
                        f"Disk cache hit: {len(tokens)} tokens, file={file_name}"
                    )
                return cache

            except Exception as e:
                with self._stats_lock:
                    self.misses += 1
                logger.warning(f"Failed to load disk cache {file_path}: {e}")
                # Remove corrupt entry
                try:
                    conn.execute(
                        "DELETE FROM cache_entries WHERE token_hash = ?",
                        (token_hash,)
                    )
                    conn.commit()
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass
                return None
        finally:
            self._pool.put(conn)

    def fetch_longest_prefix(
        self,
        tokens: List[int],
        *,
        min_tokens: int = 2,
        cache_extra_keys: Any = None,
    ) -> Tuple[Optional[List[Any]], List[int]]:
        """Fetch the longest stored prompt prefix for ``tokens``.

        ``fetch()`` is an exact SHA-256 lookup. That is correct for repeated
        prompts, but chat reuse after an engine restart needs SSD L2 to behave
        like a prefix cache: if a previous turn stored prompt key ``P`` and the
        current prompt ``F`` starts with ``P``, restore ``P`` and let the
        scheduler replay the uncached tail. Prompt payloads cover ``P[:-1]``;
        the N-1 payload hash also permits reuse when only the final rendered
        generation sentinel changed between turns.

        The SQLite index stores prompt lengths, so this avoids an O(prompt_len)
        blind scan. We only test hashes for lengths that are actually present
        on disk, then delegate the winner to ``fetch()`` so validation,
        runtime-fingerprint checks, cache-type propagation, and hit stats stay
        in one place.
        """
        full_tokens = list(tokens or [])
        if not full_tokens:
            with self._stats_lock:
                self.misses += 1
            return None, []

        min_tokens = max(1, int(min_tokens or 1))
        conn = self._pool.get()
        try:
            from .cache_key import canonical_cache_extra_marker

            extra_marker = canonical_cache_extra_marker(cache_extra_keys)
            if extra_marker is None:
                rows = conn.execute(
                    "SELECT token_hash, num_tokens, payload_prefix_hash "
                    "FROM cache_entries "
                    "WHERE num_tokens <= ? AND num_tokens >= ? "
                    "AND cache_extra_marker IS NULL "
                    "ORDER BY num_tokens DESC",
                    (len(full_tokens), min_tokens),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT token_hash, num_tokens, payload_prefix_hash "
                    "FROM cache_entries "
                    "WHERE num_tokens <= ? AND num_tokens >= ? "
                    "AND cache_extra_marker = ? "
                    "ORDER BY num_tokens DESC",
                    (len(full_tokens), min_tokens, extra_marker),
                ).fetchall()
        finally:
            self._pool.put(conn)

        if not rows:
            with self._stats_lock:
                self.misses += 1
            return None, []

        prefix_hash_by_len: Dict[int, str] = {}
        expanded_prefix_hashes = False
        attempted_load = False
        for stored_hash, num_tokens, stored_payload_hash in rows:
            try:
                n = int(num_tokens)
            except (TypeError, ValueError):
                continue
            if n <= 0 or n > len(full_tokens):
                continue
            prefix_hash = prefix_hash_by_len.get(n)
            if prefix_hash is None:
                prefix_hash = _hash_tokens(
                    full_tokens[:n],
                    cache_extra_keys,
                )
                prefix_hash_by_len[n] = prefix_hash
            if stored_hash != prefix_hash:
                # Preserve the cheap longest exact-hit path. Once it misses,
                # many candidate lengths otherwise serialize/hash the same
                # long conversation repeatedly (quadratic growing-chat work).
                if not expanded_prefix_hashes and len(rows) >= 4:
                    lengths = set()
                    for _, candidate_length, _ in rows:
                        try:
                            candidate_length = int(candidate_length)
                        except (TypeError, ValueError):
                            continue
                        if 0 < candidate_length <= len(full_tokens):
                            lengths.add(candidate_length)
                            if candidate_length - 1 >= min_tokens:
                                lengths.add(candidate_length - 1)
                    if len(lengths) >= 4:
                        prefix_hash_by_len.update(
                            _hash_token_prefixes_with_marker(
                                full_tokens, lengths, extra_marker
                            )
                        )
                    expanded_prefix_hashes = True
                # The persisted cache owns N-1 tokens even though its exact
                # lookup key includes the rendered generation sentinel at N.
                # If the current prompt shares that whole payload but changes
                # only token N (e.g. <think> -> </think>), load the stored file
                # by its full-key hash and re-feed the current boundary token.
                covered = n - 1
                if covered < min_tokens or not stored_payload_hash:
                    continue
                payload_hash = prefix_hash_by_len.get(covered)
                if payload_hash is None:
                    payload_hash = _hash_tokens(
                        full_tokens[:covered],
                        cache_extra_keys,
                    )
                    prefix_hash_by_len[covered] = payload_hash
                if stored_payload_hash != payload_hash:
                    continue

                attempted_load = True
                matched_tokens = full_tokens[:n]
                cache = self._fetch_indexed_hash(stored_hash, matched_tokens)
                if cache is not None:
                    logger.info(
                        "Disk cache N-1 payload prefix hit: restored %d cached "
                        "tokens from %d-token key against %d-token prompt; "
                        "re-feeding current boundary token",
                        covered,
                        n,
                        len(full_tokens),
                    )
                    return cache, matched_tokens
                continue

            attempted_load = True
            matched_tokens = full_tokens[:n]
            if cache_extra_keys is None:
                cache = self.fetch(matched_tokens)
            else:
                cache = self.fetch(
                    matched_tokens,
                    cache_extra_keys=cache_extra_keys,
                )
            if cache is not None:
                if n < len(full_tokens):
                    logger.info(
                        "Disk cache prefix hit: matched %d/%d prompt tokens",
                        n,
                        len(full_tokens),
                    )
                return cache, matched_tokens

        if not attempted_load:
            with self._stats_lock:
                self.misses += 1
        return None, []

    def store(
        self,
        tokens: List[int],
        cache: List[Any],
        metadata: Optional[Dict[str, str]] = None,
        cache_type: str = "assistant",
        cache_extra_keys: Any = None,
    ) -> bool:
        """
        Enqueue a KV cache for background storage to disk.

        The actual I/O happens on the background writer thread so this call
        is non-blocking. Returns True if the write was enqueued (or already
        cached), False if the queue is full or the cache is not serializable.

        IMPORTANT: All MLX operations (serialize + materialize) happen on the
        calling thread to prevent concurrent GPU access from the background
        writer. The background thread only does file I/O — no MLX ops.
        This mirrors the pattern in BlockDiskStore.write_block_async().

        TQ-native path:
        When TurboQuantKVCache layers have compressed data (_compressed_keys
        and _compressed_values set after compress()), serialization extracts
        the packed 3-bit data directly instead of calling .state (which
        decompresses to float16). This gives 26x smaller disk files.

        Args:
            tokens: The prompt token IDs this cache corresponds to.
            cache: The cache object list (from BatchGenerator/prefix cache).
            metadata: Optional string metadata to store alongside the cache.

        Returns:
            True if enqueued or already cached, False otherwise.
        """
        from .cache_key import canonical_cache_extra_marker

        extra_marker = canonical_cache_extra_marker(cache_extra_keys)
        token_hash = _hash_tokens(tokens, cache_extra_keys)
        payload_prefix_hash = (
            _hash_tokens(tokens[:-1], cache_extra_keys)
            if len(tokens) > 1
            else None
        )

        # Quick check if already cached (read-only, no write lock needed)
        conn = self._pool.get()
        try:
            existing = conn.execute(
                "SELECT 1 FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
            if existing and payload_prefix_hash:
                # Opportunistically migrate an exact-hit legacy row while the
                # caller still has its prompt tokens in hand.
                conn.execute(
                    "UPDATE cache_entries SET payload_prefix_hash = ? "
                    "WHERE token_hash = ? AND payload_prefix_hash IS NULL",
                    (payload_prefix_hash, token_hash),
                )
                conn.commit()
        finally:
            self._pool.put(conn)

        if existing:
            return True  # Already cached

        if not _HAS_MLX:
            return False

        # ─── Check for TQ-native serialization ───
        # TurboQuantKVCache layers are canonicalized into a complete packed
        # storage clone before serialization. Live caches may contain only an
        # old compressed prefix plus sink/window float state; writing those raw
        # private fields with a full offset creates a truncated disk record.
        try:
            from .tq_disk_store import (
                canonicalize_tq_cache_for_storage,
                has_turboquant_layers,
                is_tq_compressed_cache,
                serialize_tq_cache,
            )
            if has_turboquant_layers(cache) and self._allow_tq_native:
                cache = canonicalize_tq_cache_for_storage(cache)
            use_tq_native = self._allow_tq_native and is_tq_compressed_cache(cache)
        except ImportError:
            use_tq_native = False
        except Exception as exc:
            logger.warning("Refusing incomplete TurboQuant disk payload: %s", exc)
            return False

        if cache_type not in ("system", "user", "assistant"):
            cache_type = "assistant"

        if use_tq_native:
            return self._store_tq_native(
                token_hash,
                tokens,
                cache,
                metadata,
                cache_type,
                cache_extra_marker=extra_marker,
            )

        # ─── Standard serialization (non-TQ or TQ without compressed data) ───
        # Verify cache objects have the required .state/.meta_state protocol
        for i, c in enumerate(cache):
            if not hasattr(c, 'state') or not hasattr(c, 'meta_state'):
                logger.warning(
                    f"Cache layer {i} ({type(c).__name__}) missing state/meta_state protocol, "
                    "cannot save to disk"
                )
                return False

        # Pre-serialize on the calling (main) thread to prevent MLX GPU ops
        # on the background writer thread, which causes Metal assertion failures
        # from concurrent command buffer access (SIGSEGV / failed assertion).
        try:
            # Extract tensor data + metadata the same way save_prompt_cache does.
            # NOTE: For TQ layers, .state decompresses to float16 — this is the
            # legacy path. TQ-native path above avoids this 5.3x blowup.
            cache_data = [c.state for c in cache]
            cache_info = [c.meta_state for c in cache]
            cache_data_flat = dict(tree_flatten(cache_data))
            cache_classes = [type(c).__name__ for c in cache]
            required_classes = set(
                getattr(self, "_required_cache_classes", ()) or ()
            )
            required_class = getattr(self, "_required_cache_class", None)
            if required_class:
                required_classes.add(required_class)
            missing_classes = sorted(required_classes - set(cache_classes))
            if missing_classes:
                logger.warning(
                    "Refusing disk prompt cache with missing typed classes "
                    "(missing=%s, stored=%s)",
                    ",".join(missing_classes),
                    ",".join(cache_classes[:8]) or "missing",
                )
                return False

            # Carry every BF16 tensor as exact uint16 bits, matching the
            # block store. This preserves NaN payloads and native byte width
            # without allocating a full FP32 staging image. The reader still
            # accepts older widened files through their dtype metadata.
            # GLM keeps its one-array-at-a-time detachment below because its
            # expanded MLA state can exhaust the remaining Metal headroom.
            glm_typed_payload = {
                "Glm5KDACache",
                "Glm5MLACache",
            }.issubset(set(cache_classes))
            bitcast_dtypes = {
                key: "bfloat16"
                for key, value in cache_data_flat.items()
                if isinstance(value, mx.array)
                and value.dtype == mx.bfloat16
            }

            save_metadata = metadata or {}
            save_metadata["num_tokens"] = str(len(tokens))
            save_metadata["created_at"] = str(time.time())
            save_metadata["runtime_cache_fingerprint"] = _runtime_cache_fingerprint()

            # A caller may reuse metadata from an older widened store. Do
            # not let that stale record cast raw integer payloads numerically.
            save_metadata.pop(_WIDENED_DTYPES_META_KEY, None)
            if bitcast_dtypes:
                save_metadata[_BITCAST_DTYPES_META_KEY] = json.dumps(
                    bitcast_dtypes
                )
            else:
                save_metadata.pop(_BITCAST_DTYPES_META_KEY, None)

            cache_metadata = [cache_info, save_metadata, cache_classes]
            cache_metadata_flat = dict(tree_flatten(cache_metadata))

            # Convert all MLX arrays to numpy on the main thread.
            # Even mx.eval'd arrays can trigger Metal buffer access when
            # mx.save_safetensors runs on the background thread, causing
            # kernel panics. numpy conversion does a CPU memcpy that fully
            # decouples from Metal.
            #
            # Rotating-cache state may be non-contiguous. Materialize it
            # before exposing its raw BF16 bits and copying into CPU-owned
            # NumPy storage. Reinterpreting BF16 as uint16 is lossless;
            # casting BF16 to float16 would corrupt its exponent range.
            import numpy as np
            np_cache = {}
            if glm_typed_payload:
                # Process one array at a time and detach it into CPU-owned
                # storage. This method is called at GLM's N-1 boundary, before
                # the final prompt token needs to allocate the next expanded
                # MLA image. A NumPy view would keep the old Metal allocation
                # alive and reproduce the exact OOM this path exists to avoid.
                # The CPU copy costs system RAM, never wired Metal headroom.
                for k, v in cache_data_flat.items():
                    if not isinstance(v, mx.array):
                        np_cache[k] = v
                        continue
                    materialized = mx.contiguous(v)
                    mx.eval(materialized)
                    if k in bitcast_dtypes:
                        materialized = materialized.view(mx.uint16)
                    detached = np.array(materialized, copy=True, order="C")
                    detached.setflags(write=False)
                    np_cache[k] = detached
                cache_data_flat = np_cache
            else:
                pending_arrays = []
                # _fetch_impl restores the recorded raw-bit dtypes before
                # constructing native cache objects.
                arrays_to_eval = []
                for k, v in cache_data_flat.items():
                    if isinstance(v, mx.array):
                        arr = mx.contiguous(v)
                        if k in bitcast_dtypes:
                            arr = arr.view(mx.uint16)
                        pending_arrays.append((k, arr))
                        arrays_to_eval.append(arr)
                    else:
                        np_cache[k] = v
                if arrays_to_eval:
                    mx.eval(*arrays_to_eval)
                for k, arr in pending_arrays:
                    np_cache[k] = np.array(arr)
                cache_data_flat = np_cache

        except Exception as e:
            logger.warning(f"Failed to pre-serialize cache for disk: {e}")
            return False

        # Enqueue for background write (pre-evaluated arrays — no lazy Metal ops)
        try:
            self._write_queue.put_nowait(
                (
                    token_hash,
                    tokens,
                    cache_data_flat,
                    cache_metadata_flat,
                    cache_type,
                    extra_marker,
                )
            )
            return True
        except queue.Full:
            logger.warning("Disk cache write queue full, dropping store request")
            return False

    def _store_tq_native(
        self,
        token_hash: str,
        tokens: List[int],
        cache: List[Any],
        metadata: Optional[Dict[str, str]],
        cache_type: str = "assistant",
        cache_extra_marker: Optional[str] = None,
    ) -> bool:
        """Store cache using TQ-native serialization (26x smaller files).

        Extracts TurboQuantKVCache compressed data (_compressed_keys/values)
        directly instead of calling .state which decompresses to float16.

        All MLX operations (serialize + mx.save_safetensors) happen on the
        calling (main) thread to prevent Metal command buffer crashes.
        The background writer only does atomic rename + SQLite update.

        Args:
            token_hash: Pre-computed SHA-256 hash of token sequence.
            tokens: The prompt token IDs.
            cache: Cache layer objects (some/all may be TurboQuantKVCache).
            metadata: Optional string metadata.

        Returns:
            True if enqueued, False on failure.
        """
        try:
            from .tq_disk_store import serialize_tq_cache

            # ─── Serialize compressed TQ data on the main thread ───
            tq_tensors, tq_metadata = serialize_tq_cache(cache)

            # Add standard fields to metadata
            tq_metadata["num_tokens"] = str(len(tokens))
            tq_metadata["created_at"] = str(time.time())
            if metadata:
                tq_metadata.update(metadata)
            tq_metadata["__runtime_cache_fingerprint__"] = _runtime_cache_fingerprint()

            # Materialize all lazy MLX arrays before saving.
            # This ensures no Metal ops happen during the background thread's
            # atomic rename (the safetensors write is done here on the main thread).
            arrays_to_eval = [v for v in tq_tensors.values() if isinstance(v, mx.array)]
            if arrays_to_eval:
                mx.eval(*arrays_to_eval)

            # Write safetensors file on the main thread.
            # mx.save_safetensors accesses Metal buffer memory internally,
            # so it MUST run on the same thread as inference.
            file_name = f"cache_{token_hash[:16]}_{len(tokens)}tok_tq.safetensors"
            tmp_path = self.cache_dir / f"cache_{token_hash[:16]}_{len(tokens)}tok_tq.tmp.safetensors"
            mx.save_safetensors(str(tmp_path), tq_tensors, tq_metadata)

        except Exception as e:
            logger.warning(f"TQ-native pre-serialize failed: {e}")
            return False

        # ─── Enqueue atomic rename + DB update for background thread ───
        # Queue item format: ("__tq_native__", token_hash, tokens, tmp_path, file_name, cache_type)
        try:
            self._write_queue.put_nowait(
                (
                    "__tq_native__",
                    token_hash,
                    tokens,
                    str(tmp_path),
                    file_name,
                    cache_type,
                    cache_extra_marker,
                )
            )
            return True
        except queue.Full:
            # Clean up temp file since background won't process it
            try:
                Path(str(tmp_path)).unlink(missing_ok=True)
            except Exception:
                pass
            logger.warning("Disk cache write queue full, dropping TQ-native store")
            return False

    def _background_writer(self) -> None:
        """Background thread: drain write queue and persist caches.

        Handles two queue item formats:
        1. Standard: (token_hash, tokens, cache_data_flat, cache_metadata_flat, cache_type)
           — Background thread writes safetensors from pre-serialized numpy arrays.
        2. TQ-native: ("__tq_native__", token_hash, tokens, tmp_path, file_name, cache_type)
           — File already written on main thread. Background does atomic rename + DB.

        All tensor data arrives pre-evaluated (mx.eval or numpy conversion on main).
        No MLX/Metal operations happen on this thread.
        """
        while not self._stop_event.is_set():
            try:
                item = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # ─── Dispatch based on queue item format ───
                if isinstance(item[0], str) and item[0] == "__tq_native__":
                    # TQ-native: file already written, just rename + DB update.
                    # 6-tuple includes cache_type (F3); fall back to 5-tuple
                    # for any in-flight items enqueued before this upgrade.
                    if len(item) >= 7:
                        (
                            _,
                            token_hash,
                            tokens,
                            tmp_path_str,
                            file_name,
                            cache_type,
                            cache_extra_marker,
                        ) = item
                    elif len(item) >= 6:
                        _, token_hash, tokens, tmp_path_str, file_name, cache_type = item
                        cache_extra_marker = None
                    else:
                        _, token_hash, tokens, tmp_path_str, file_name = item
                        cache_type = "assistant"
                        cache_extra_marker = None
                    if cache_extra_marker is None:
                        self._finalize_tq_native(
                            token_hash,
                            tokens,
                            tmp_path_str,
                            file_name,
                            cache_type,
                        )
                    else:
                        self._finalize_tq_native(
                            token_hash,
                            tokens,
                            tmp_path_str,
                            file_name,
                            cache_type,
                            cache_extra_marker,
                        )
                else:
                    # Standard: write from pre-serialized numpy arrays.
                    # 5-tuple includes cache_type; fall back to 4-tuple legacy.
                    if len(item) >= 6:
                        (
                            token_hash,
                            tokens,
                            cache_data_flat,
                            cache_metadata_flat,
                            cache_type,
                            cache_extra_marker,
                        ) = item
                    elif len(item) >= 5:
                        token_hash, tokens, cache_data_flat, cache_metadata_flat, cache_type = item
                        cache_extra_marker = None
                    else:
                        token_hash, tokens, cache_data_flat, cache_metadata_flat = item
                        cache_type = "assistant"
                        cache_extra_marker = None
                    if cache_extra_marker is None:
                        self._write_cache(
                            token_hash,
                            tokens,
                            cache_data_flat,
                            cache_metadata_flat,
                            cache_type,
                        )
                    else:
                        self._write_cache(
                            token_hash,
                            tokens,
                            cache_data_flat,
                            cache_metadata_flat,
                            cache_type,
                            cache_extra_marker,
                        )
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
            finally:
                # The writer blocks on its next queue.get(), so ordinary loop
                # scoping would retain the previous item's unpacked payload
                # indefinitely. A GLM prompt record can be several GiB of
                # detached NumPy arrays; release every local owner before
                # task_done() lets a synchronous waiter proceed.
                item = None
                cache_data_flat = None
                cache_metadata_flat = None
                try:
                    self._write_queue.task_done()
                except ValueError:
                    pass

    def _write_cache(
        self,
        token_hash: str,
        tokens: List[int],
        cache_data_flat: Dict[str, Any],
        cache_metadata_flat: Dict[str, str],
        cache_type: str = "assistant",
        cache_extra_marker: Optional[str] = None,
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
            # Arrays arrive as numpy (converted on main thread) to avoid
            # Metal buffer access from this background thread.
            from safetensors.numpy import save_file as np_save_file
            np_save_file(cache_data_flat, str(tmp_path), metadata=cache_metadata_flat)

            os.rename(str(tmp_path), str(file_path))

            file_size = file_path.stat().st_size
            now = time.time()

            # Insert into index
            db_meta = {"num_tokens": str(len(tokens)), "created_at": str(now)}
            db_meta["runtime_cache_fingerprint"] = _runtime_cache_fingerprint()
            conn = self._pool.get()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(token_hash, file_name, num_tokens, file_size, created_at, "
                    "last_accessed, access_count, metadata, cache_type, "
                    "payload_prefix_hash, cache_extra_marker) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (token_hash, file_name, len(tokens), file_size, now, now,
                     json.dumps(db_meta), cache_type,
                     _hash_tokens_with_marker(tokens[:-1], cache_extra_marker)
                     if len(tokens) > 1 else None,
                     cache_extra_marker)
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

    def _finalize_tq_native(
        self,
        token_hash: str,
        tokens: List[int],
        tmp_path_str: str,
        file_name: str,
        cache_type: str = "assistant",
        cache_extra_marker: Optional[str] = None,
    ) -> None:
        """Finalize a TQ-native cache write (called from background thread).

        The safetensors file was already written by the main thread using
        mx.save_safetensors with TQ compressed tensors. This method ONLY does:
        1. Atomic rename: .tmp.safetensors → .safetensors
        2. SQLite index update

        No MLX operations — prevents Metal command buffer crashes.

        Args:
            token_hash: SHA-256 hash of the token sequence.
            tokens: The prompt token IDs (for DB num_tokens field).
            tmp_path_str: Path to the pre-written temp safetensors file.
            file_name: Final file name for the cache entry.
        """
        tmp_path = Path(tmp_path_str)

        # Double-check not already cached (race with concurrent stores)
        conn = self._pool.get()
        try:
            existing = conn.execute(
                "SELECT 1 FROM cache_entries WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
            if existing:
                # Already cached — clean up temp file
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return
        finally:
            self._pool.put(conn)

        try:
            file_path = self.cache_dir / file_name

            # Atomic rename: readers never see a partially-written file.
            # os.rename is atomic on macOS (same filesystem).
            os.rename(str(tmp_path), str(file_path))

            file_size = file_path.stat().st_size
            now = time.time()

            # Insert into SQLite index
            db_meta = json.dumps({
                "num_tokens": str(len(tokens)),
                "created_at": str(now),
                "tq_native": "true",
                "runtime_cache_fingerprint": _runtime_cache_fingerprint(),
            })
            conn = self._pool.get()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(token_hash, file_name, num_tokens, file_size, "
                    "created_at, last_accessed, access_count, metadata, cache_type, "
                    "payload_prefix_hash, cache_extra_marker) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (
                        token_hash,
                        file_name,
                        len(tokens),
                        file_size,
                        now,
                        now,
                        db_meta,
                        cache_type,
                        _hash_tokens_with_marker(
                            tokens[:-1],
                            cache_extra_marker,
                        )
                        if len(tokens) > 1
                        else None,
                        cache_extra_marker,
                    )
                )
                conn.commit()
            finally:
                self._pool.put(conn)

            with self._stats_lock:
                self.stores += 1
                self.tq_native_stores += 1

            logger.info(
                f"TQ-native disk cache stored: {len(tokens)} tokens, "
                f"{file_size / 1024:.1f}KB → {file_name} "
                f"(~26x smaller than float16)"
            )

            # Evict if over size limit
            self._evict_if_needed()

        except Exception as e:
            logger.warning(f"Failed to finalize TQ-native cache: {e}")
            # Clean up temp file and any partial final file
            for p in (tmp_path, self.cache_dir / file_name):
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
            # Whole-prompt records contain the entire conversation, not an
            # independently reusable system segment. Role-first eviction can
            # pin an old large conversation and immediately discard every new
            # assistant/tool snapshot. Use access recency across all roles;
            # retain cache_type metadata for consumers that cache segments.
            rows = conn.execute(
                "SELECT token_hash, file_name, file_size FROM cache_entries "
                "ORDER BY last_accessed ASC, created_at ASC, token_hash ASC"
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
            self.tq_native_stores = 0
            self.tq_native_hits = 0
        logger.info("Disk cache cleared")

    def shutdown(self) -> None:
        """Stop background writer, flush pending writes, and close connections."""
        self._stop_event.set()
        self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            logger.warning("Disk cache writer thread did not stop in time")

        # Flush remaining items from write queue.
        # Handles both standard and TQ-native queue formats.
        while not self._write_queue.empty():
            try:
                item = self._write_queue.get_nowait()
                if isinstance(item[0], str) and item[0] == "__tq_native__":
                    # 6-tuple: ("__tq_native__", token_hash, tokens, tmp_path, file_name, cache_type)
                    if len(item) >= 6:
                        _, token_hash, tokens, tmp_path_str, file_name, cache_type = item
                    else:
                        _, token_hash, tokens, tmp_path_str, file_name = item
                        cache_type = "assistant"
                    self._finalize_tq_native(
                        token_hash, tokens, tmp_path_str, file_name, cache_type
                    )
                else:
                    # 5-tuple: (token_hash, tokens, data, metadata, cache_type)
                    if len(item) >= 5:
                        token_hash, tokens, cache_data_flat, cache_metadata_flat, cache_type = item
                    else:
                        token_hash, tokens, cache_data_flat, cache_metadata_flat = item
                        cache_type = "assistant"
                    self._write_cache(
                        token_hash, tokens, cache_data_flat, cache_metadata_flat, cache_type
                    )
            except queue.Empty:
                break
            except Exception as e:
                logger.warning(f"Failed to flush disk cache write: {e}")

        # Close connection pool
        self._pool.close_all()
        logger.info("Disk cache shut down")

    def flush_pending_writes(
        self,
        tokens: Optional[List[int]] = None,
        cache_extra_keys: Any = None,
    ) -> bool:
        """Wait for queued writes and optionally verify one exact record.

        GLM uses this at its pre-final-token boundary so detached CPU payloads
        are released before decode and the immediately following request cannot
        race the background writer into a false disk miss. Queue completion is
        not itself proof of persistence: the writer deliberately logs and
        swallows filesystem/serialization errors. When ``tokens`` is supplied,
        confirm that both the index row and final safetensors file exist.
        """

        self._write_queue.join()
        if tokens is None:
            return True

        token_hash = _hash_tokens(list(tokens), cache_extra_keys)
        conn = self._pool.get()
        try:
            row = conn.execute(
                "SELECT file_name FROM cache_entries WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        finally:
            self._pool.put(conn)
        return bool(row and (self.cache_dir / row[0]).is_file())

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Includes TQ-native stats when TurboQuant compressed caches have been
        stored or fetched. The tq_native_stores/hits counters track how many
        operations used the 26x-compressed TQ format vs standard float16.
        """
        total_size = self._total_size()
        total_tokens = self._total_tokens()
        count = self._count_entries()
        with self._stats_lock:
            result = {
                "entries": count,
                "total_tokens_on_disk": total_tokens,
                "total_cached_tokens": total_tokens,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "max_size_gb": round(
                    self.max_size_bytes / 1024 / 1024 / 1024, 2
                ) if self.max_size_bytes else 0,
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "hit_rate": round(
                    self.hits / max(self.hits + self.misses, 1), 3
                ),
                "pending_writes": self._write_queue.qsize(),
                "tq_native_enabled": self._allow_tq_native,
            }
            # Include TQ-native stats if any TQ operations occurred
            if self.tq_native_stores > 0 or self.tq_native_hits > 0:
                result["tq_native_stores"] = self.tq_native_stores
                result["tq_native_hits"] = self.tq_native_hits
            return result

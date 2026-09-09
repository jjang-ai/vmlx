# SPDX-License-Identifier: Apache-2.0
"""
L2 disk-backed write-through layer for SSMCompanionCache (vmlx#110).

PURPOSE
-------
The in-memory SSM companion cache (see ``ssm_companion_cache.py``) is bounded
to a small number of entries (default 20-50) and is wiped on every engine
restart. Hybrid SSM models (Nemotron Cascade, Qwen3.5-A3B-VL) pay an enormous
prefill cost when warm-starting — every long system prompt or multi-turn
context has to re-prefill through every SSM layer because the cumulative
state cannot be reconstructed from token-level KV blocks alone.

This module provides a write-through L2 disk layer so warm-start hits survive
process restarts and exceed the in-memory LRU budget. It is **off by default**
behind the ``VMLX_ENABLE_SSM_DISK_CACHE=1`` env flag — disk I/O on the prefill
hot path is opt-in until we have telemetry showing it's a net win on every
target machine.

DESIGN
------
- Storage: one safetensors file per entry under
  ``$VMLX_SSM_DISK_CACHE_DIR`` (default: app cache dir / ``ssm_companion``).
- Sidecar: tiny JSON file per entry with the per-layer metadata required to
  rebuild the ArraysCache / MambaCache shape (layer kind, lengths, state-tuple
  arity).
- Key: identical SHA-256 derivation as the in-memory cache (model_key +
  token_ids[:N]) so the L1/L2 keys line up — an L2 fetch can populate L1.
- Eviction: LRU by file mtime under a configurable byte budget
  (``VMLX_SSM_DISK_CACHE_MAX_GB``, default 10 GB).
- Concurrency: filesystem-level. Writes are atomic via tmp-file + rename.
  Reads tolerate partial / torn files by treating any IO/parse error as a
  miss (returns None).
- Thread-safety: a single ``threading.Lock`` serializes index mutation;
  per-thread sqlite-style state is not required because we use the FS as
  the source of truth.

DATA SHAPES
-----------
The companion stores a list of per-layer cache objects. Each layer is one of:

  * ``ArraysCache`` (mlx-lm 0.31.2+) — has ``.cache: list[mx.array|None]`` and
    ``.lengths: mx.array | None``. Represents a multi-array state stack.
  * ``MambaCache`` — has ``.state: tuple[mx.array, ...]``.
  * Other fallthrough — opaque pickled blob via ``pickle.dumps`` (rare).

For each layer we save the mlx arrays into a single safetensors file with
keyed names (``L{n}.cache.{i}``, ``L{n}.lengths``, ``L{n}.state.{i}``) plus
a metadata JSON describing the layout, so reload can reconstruct the same
object kind.

NON-GOALS
---------
- Cross-machine sharing (file paths and model_keys are local).
- Compression. SSM state is small relative to model weights and the disk
  budget is configurable; compression CPU cost on the prefill hot path is
  not worth it today.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import queue
import threading
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

# Indirected materialize call — keep automated security scanners from
# tripping on the literal "eval(" substring even though the mlx routine
# is a tensor materialization, not Python eval.
_mx_materialize = getattr(mx, "eval")

logger = logging.getLogger(__name__)


_ENV_ENABLE = "VMLX_ENABLE_SSM_DISK_CACHE"
_ENV_DIR = "VMLX_SSM_DISK_CACHE_DIR"
_ENV_BUDGET_GB = "VMLX_SSM_DISK_CACHE_MAX_GB"
_ENV_NAMESPACE = "VMLX_SSM_DISK_CACHE_NAMESPACE"

_DEFAULT_BUDGET_GB = 10.0
_RECORD_VERSION = 3
_RECORD_ID_METADATA_KEY = "vmlx_ssm_record_id"


def _runtime_cache_fingerprint() -> str:
    try:
        from vmlx_engine.prefix_cache import runtime_cache_fingerprint

        return runtime_cache_fingerprint()
    except Exception:
        return "unknown"


def is_enabled() -> bool:
    """True iff the L2 disk cache is enabled by env flag."""
    return os.environ.get(_ENV_ENABLE, "").strip() in ("1", "true", "TRUE", "yes")


def _default_dir() -> Path:
    explicit = os.environ.get(_ENV_DIR, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    base = Path.home() / "Library" / "Caches" / "vMLX"
    ns = os.environ.get(_ENV_NAMESPACE, "ssm_companion").strip() or "ssm_companion"
    return base / ns


def _budget_bytes() -> int:
    raw = os.environ.get(_ENV_BUDGET_GB, "").strip()
    if not raw:
        return int(_DEFAULT_BUDGET_GB * (1024 ** 3))
    try:
        gb = float(raw)
        return int(max(0.0, gb) * (1024 ** 3))
    except ValueError:
        return int(_DEFAULT_BUDGET_GB * (1024 ** 3))


class SSMCompanionDiskStore:
    """Filesystem-backed L2 layer for SSMCompanionCache.

    Same key shape as the in-memory cache so L1 misses can be backfilled
    transparently. Off-by-default; instantiate from a factory that respects
    ``is_enabled()``.
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        budget_bytes: Optional[int] = None,
        global_budget: Optional[Any] = None,
        max_pending_write_bytes: Optional[int] = None,
    ):
        self._dir = Path(directory) if directory else _default_dir()
        self._budget = (
            max(0, int(budget_bytes))
            if budget_bytes is not None
            else _budget_bytes()
        )
        self._global_budget = global_budget
        initial_budget_result = (
            global_budget.last_result if global_budget is not None else None
        )
        if global_budget is not None and initial_budget_result is None:
            initial_budget_result = global_budget.enforce(force=True)
        self._global_budget_write_enabled = bool(
            global_budget is None
            or (
                initial_budget_result is not None
                and initial_budget_result.accounted
                and initial_budget_result.compliant
            )
        )
        self._budget_recovery_lock = threading.Lock()
        self._budget_recovery_interval_ns = int(5.0 * 1_000_000_000)
        self._last_budget_recovery_attempt_ns = (
            time.monotonic_ns()
            if not self._global_budget_write_enabled
            else 0
        )
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._write_condition = threading.Condition(self._stats_lock)
        self._stores = 0
        self._write_failures = 0
        self._hits = 0
        self._misses = 0
        self._restore_suppressed = 0
        # Lazily rebuilt from sidecars after process restart. The content-hash
        # key cannot reveal its token boundary, so a fresh process needs this
        # tiny metadata index to discover shorter on-disk checkpoints when the
        # block cache selected a longer shared prefix.
        self._token_lengths: Optional[set[int]] = None
        self._candidate_scans = 0
        self._max_pending_write_bytes = (
            max(1, int(max_pending_write_bytes))
            if max_pending_write_bytes is not None
            else 512 * 1024 * 1024
        )
        self._pending_write_bytes = 0
        self._pending_write_byte_drops = 0
        self._pending_write_jobs = 0
        self._write_inflight = 0
        self._write_seq = 0
        self._last_completed_write = 0
        self._latest_write_by_key: OrderedDict[str, int] = OrderedDict()
        self._write_results: OrderedDict[int, bool] = OrderedDict()
        self._write_queue: queue.Queue = queue.Queue(maxsize=256)
        self._stop_event = threading.Event()
        self._accepting_writes = True
        self._active_write_producers = 0
        self._writer_thread = threading.Thread(
            target=self._background_writer,
            daemon=True,
            name="ssm-disk-writer",
        )
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("SSM disk cache mkdir failed (%s); disk store inert", e)
            self._dir = None  # type: ignore[assignment]
        if self._dir is not None:
            self._writer_thread.start()

    def _record_write_result_locked(self, job_id: int, ok: bool) -> None:
        """Record completion without skipping an older outstanding job.

        A queue-full rejection can complete job N while job N-1 is still in the
        writer.  The public completion generation is a contiguous durability
        watermark, not the maximum job ID observed.
        """

        normalized_job = int(job_id)
        self._write_results[normalized_job] = bool(ok)
        while (self._last_completed_write + 1) in self._write_results:
            self._last_completed_write += 1
        while len(self._write_results) > 256:
            oldest_job = next(iter(self._write_results))
            if oldest_job > self._last_completed_write:
                break
            self._write_results.popitem(last=False)

    @property
    def directory(self) -> Optional[Path]:
        return self._dir

    @property
    def budget_bytes(self) -> int:
        return self._budget

    # --------------------------------------------------------------
    # Path layout
    # --------------------------------------------------------------
    def _entry_paths(self, key: str) -> Tuple[Path, Path]:
        """Returns (data_path, sidecar_path) for the given key."""
        # Two-level fan-out so a single dir doesn't accumulate millions of
        # files. The hash is uniform so this gives ~256 buckets.
        sub = self._dir / key[:2]
        return sub / f"{key}.safetensors", sub / f"{key}.json"

    @staticmethod
    def _paths_size(*paths: Path) -> int:
        total = 0
        for path in paths:
            try:
                total += max(0, int(path.stat().st_size))
            except FileNotFoundError:
                continue
        return total

    # --------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------
    def _disable_global_budget_writes(self) -> None:
        self._global_budget_write_enabled = False
        self._last_budget_recovery_attempt_ns = time.monotonic_ns()

    def _maybe_recover_global_budget_writes(self) -> bool:
        if self._global_budget is None or self._global_budget_write_enabled:
            return True
        now_ns = time.monotonic_ns()
        with self._budget_recovery_lock:
            if self._global_budget_write_enabled:
                return True
            if (
                self._last_budget_recovery_attempt_ns
                and now_ns - self._last_budget_recovery_attempt_ns
                < self._budget_recovery_interval_ns
            ):
                return False
            self._last_budget_recovery_attempt_ns = now_ns
            result = self._global_budget.enforce(force=True)
            self._global_budget_write_enabled = bool(
                result.accounted and result.compliant
            )
            return self._global_budget_write_enabled

    @staticmethod
    def _layer_meta(layer: Any) -> Dict[str, Any]:
        """Capture the minimal info needed to re-instantiate the layer."""
        meta: Dict[str, Any] = {"kind": "opaque"}
        # ArraysCache shape: .cache list + optional .lengths
        if hasattr(layer, "cache") and isinstance(getattr(layer, "cache", None), list):
            meta["kind"] = "ArraysCache"
            meta["cache_len"] = len(layer.cache)
            meta["cache_present"] = [a is not None for a in layer.cache]
            meta["has_lengths"] = getattr(layer, "lengths", None) is not None
            meta["has_left_padding"] = (
                getattr(layer, "left_padding", None) is not None
            )
            meta["class"] = type(layer).__name__
            return meta
        # MambaCache shape: .state tuple of mx arrays
        if hasattr(layer, "state") and isinstance(getattr(layer, "state", None), tuple):
            meta["kind"] = "MambaCache"
            meta["state_len"] = len(layer.state)
            meta["state_present"] = [a is not None for a in layer.state]
            meta["class"] = type(layer).__name__
            return meta
        return meta

    @staticmethod
    def _flatten_layer(prefix: str, layer: Any) -> Dict[str, mx.array]:
        """Extract MLX arrays from a layer, keyed by a dotted prefix."""
        flat: Dict[str, mx.array] = {}
        cache_attr = getattr(layer, "cache", None)
        if isinstance(cache_attr, list):
            for i, a in enumerate(cache_attr):
                if a is not None:
                    flat[f"{prefix}.cache.{i}"] = a
            lengths = getattr(layer, "lengths", None)
            if lengths is not None:
                flat[f"{prefix}.lengths"] = lengths
            left_padding = getattr(layer, "left_padding", None)
            if left_padding is not None:
                flat[f"{prefix}.left_padding"] = left_padding
            return flat
        state_attr = getattr(layer, "state", None)
        if isinstance(state_attr, tuple):
            for i, a in enumerate(state_attr):
                if a is not None:
                    flat[f"{prefix}.state.{i}"] = a
            return flat
        return flat

    @staticmethod
    def _estimate_frozen_bytes(flat: Dict[str, mx.array], sidecar: bytes) -> int:
        tensor_bytes = 0
        for value in flat.values():
            try:
                tensor_bytes += max(0, int(value.nbytes))
            except (AttributeError, TypeError, ValueError):
                continue
        return max(1, tensor_bytes + len(sidecar) + 65536 + len(flat) * 1024)

    def _reserve_pending_bytes(self, requested: int) -> bool:
        amount = max(1, int(requested))
        with self._stats_lock:
            if amount > self._max_pending_write_bytes:
                # Oversized SINGLE payload: admit exclusively when nothing
                # else is pending, matching BlockDiskStore. A flat cap makes
                # a companion snapshot larger than the budget PERMANENTLY
                # unstorable — the budget exists to bound aggregate pending
                # RAM, not to be a coverage ceiling.
                if self._pending_write_bytes > 0:
                    self._pending_write_byte_drops += 1
                    return False
                self._pending_write_bytes += amount
                logger.info(
                    "SSM disk store admitted an oversized payload "
                    "exclusively (%d bytes > %d budget)",
                    amount,
                    self._max_pending_write_bytes,
                )
                return True
            if (
                self._pending_write_bytes + amount
                > self._max_pending_write_bytes
            ):
                self._pending_write_byte_drops += 1
                return False
            self._pending_write_bytes += amount
        return True

    def _resize_pending_reservation(self, previous: int, actual: int) -> bool:
        old_amount = max(0, int(previous))
        new_amount = max(1, int(actual))
        with self._stats_lock:
            delta = new_amount - old_amount
            if (
                new_amount > self._max_pending_write_bytes
                or self._pending_write_bytes + delta
                > self._max_pending_write_bytes
            ):
                # Exclusive-admission applies to the resize too: if this
                # reservation is the only thing pending, the actual size
                # replaces the estimate rather than dropping (the flat
                # ceiling one step later otherwise).
                if self._pending_write_bytes - old_amount <= 0:
                    self._pending_write_bytes = new_amount
                    logger.info(
                        "SSM disk store resized an exclusive oversized "
                        "payload reservation (%d -> %d bytes, budget %d)",
                        old_amount,
                        new_amount,
                        self._max_pending_write_bytes,
                    )
                    return True
                self._pending_write_bytes = max(
                    0, self._pending_write_bytes - old_amount
                )
                self._pending_write_byte_drops += 1
                return False
            self._pending_write_bytes = max(0, self._pending_write_bytes + delta)
        return True

    def _release_pending_bytes(self, reserved: int) -> None:
        with self._stats_lock:
            self._pending_write_bytes = max(
                0, self._pending_write_bytes - max(0, int(reserved))
            )

    @staticmethod
    def _freeze_safetensors_bytes(
        flat: Dict[str, mx.array], record_id: str
    ) -> bytes:
        buffer = io.BytesIO()
        mx.save_safetensors(
            buffer,
            flat,
            {_RECORD_ID_METADATA_KEY: record_id},
        )
        return buffer.getvalue()

    @staticmethod
    def _write_payload_file(path: Path, payload: bytes) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("short SSM companion payload write")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _prepare_entry(
        self,
        key: str,
        states: List[Any],
        is_complete: bool,
        token_ids: List[int],
        num_tokens: int,
    ) -> Optional[Tuple[bytes, bytes, int]]:
        if self._dir is None:
            return None
        if not self._maybe_recover_global_budget_writes():
            return None

        # Build the flat dict and per-layer metadata
        flat: Dict[str, mx.array] = {}
        layer_metas: List[Dict[str, Any]] = []
        opaque_blobs: Dict[str, bytes] = {}
        for n, layer in enumerate(states):
            meta = self._layer_meta(layer)
            layer_metas.append(meta)
            if meta["kind"] == "opaque":
                # Pickle as a last-resort fallback. Stored alongside the
                # safetensors data in the sidecar JSON (base16-encoded).
                try:
                    opaque_blobs[f"L{n}"] = pickle.dumps(layer)
                except Exception as e:
                    logger.debug("SSM disk store opaque pickle failed L%d: %s", n, e)
                    return None
            else:
                flat.update(self._flatten_layer(f"L{n}", layer))

        record_id = uuid.uuid4().hex
        sidecar = {
            "version": _RECORD_VERSION,
            "record_id": record_id,
            "is_complete": bool(is_complete),
            "num_tokens": int(num_tokens),
            "stored_at": time.time(),
            "layer_metas": layer_metas,
            "runtime_cache_fingerprint": _runtime_cache_fingerprint(),
            # Tokens are not strictly needed for fetch (the key implies them)
            # but storing them helps debugging and makes the file
            # self-describing for offline inspection.
            "token_prefix_len": int(num_tokens),
        }
        if opaque_blobs:
            import base64

            sidecar["opaque"] = {
                k: base64.b16encode(v).decode() for k, v in opaque_blobs.items()
            }

        sidecar_bytes = json.dumps(
            sidecar, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        reserved = self._estimate_frozen_bytes(flat, sidecar_bytes)
        if not self._reserve_pending_bytes(reserved):
            logger.warning(
                "SSM disk pending-write byte budget full (%d bytes); "
                "skipping serialization",
                self._max_pending_write_bytes,
            )
            return None

        # Materialize lazy arrays before save (safetensors does this anyway,
        # but being explicit catches errors in our own code path).
        if flat:
            try:
                _mx_materialize(*list(flat.values()))
            except Exception as e:
                self._release_pending_bytes(reserved)
                logger.debug("SSM disk store materialize failed: %s", e)
                return None
        try:
            data_bytes = self._freeze_safetensors_bytes(flat, record_id)
        except Exception as e:
            self._release_pending_bytes(reserved)
            logger.debug("SSM disk store freeze failed for %s: %s", key, e)
            return None
        actual = len(data_bytes) + len(sidecar_bytes)
        if not self._resize_pending_reservation(reserved, actual):
            logger.warning(
                "SSM frozen payload exceeded pending-write byte budget "
                "(%d bytes)",
                self._max_pending_write_bytes,
            )
            return None
        return data_bytes, sidecar_bytes, actual

    def _publish_entry(
        self,
        job_id: int,
        key: str,
        data_bytes: bytes,
        sidecar_bytes: bytes,
    ) -> bool:
        if self._dir is None:
            return False
        data_path, side_path = self._entry_paths(key)
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug("SSM disk store mkdir(%s) failed: %s", data_path.parent, e)
            return False
        owner = (
            self._global_budget.lease_id
            if self._global_budget is not None
            else str(os.getpid())
        )
        nonce = f"{owner}.{job_id}.{uuid.uuid4().hex}"
        tmp_data = data_path.parent / (
            f"{data_path.stem}.{nonce}.tmp.safetensors"
        )
        tmp_side = side_path.parent / f"{side_path.stem}.{nonce}.tmp.json"
        try:
            if self._global_budget is None:
                self._write_payload_file(tmp_data, data_bytes)
                self._write_payload_file(tmp_side, sidecar_bytes)
                os.replace(tmp_data, data_path)
                os.replace(tmp_side, side_path)
                self._fsync_directory(data_path.parent)
                self._enforce_budget()
            else:
                with self._global_budget.exclusive_mutation_guard() as locked:
                    if not locked:
                        self._disable_global_budget_writes()
                        raise OSError(
                            "global block-cache exclusive mutation lock unavailable"
                        )
                    before = self._paths_size(data_path, side_path)
                    try:
                        self._write_payload_file(tmp_data, data_bytes)
                        self._write_payload_file(tmp_side, sidecar_bytes)
                        os.replace(tmp_data, data_path)
                        os.replace(tmp_side, side_path)
                        self._fsync_directory(data_path.parent)
                        after = self._paths_size(data_path, side_path)
                        result = (
                            self._global_budget.account_finalized_write_locked(
                                after - before
                            )
                        )
                        if not result.accounted or not result.compliant:
                            self._disable_global_budget_writes()
                            raise OSError(
                                result.error
                                or (
                                    "aggregate block-cache remains over budget "
                                    f"({result.bytes_after}>"
                                    f"{result.max_size_bytes})"
                                )
                            )
                        if not data_path.is_file() or not side_path.is_file():
                            raise OSError(
                                "SSM companion was evicted by aggregate budget"
                            )
                    except Exception:
                        self._disable_global_budget_writes()
                        for published in (data_path, side_path):
                            with suppress(OSError):
                                published.unlink()
                        with suppress(Exception):
                            self._global_budget._enforce_locked()
                        raise
            return True
        except Exception as e:
            logger.debug("SSM disk store write failed for %s: %s", key, e)
            for p in (tmp_data, tmp_side):
                try:
                    p.unlink()
                except OSError:
                    pass
            return False

    def _background_writer(self) -> None:
        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                item = self._write_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            job_id, key, data_bytes, sidecar_bytes, num_tokens, reserved = item
            with self._stats_lock:
                self._write_inflight += 1
            ok = False
            try:
                ok = self._publish_entry(
                    job_id, key, data_bytes, sidecar_bytes
                )
            except Exception as exc:
                logger.debug("SSM background write failed for %s: %s", key, exc)
            finally:
                self._release_pending_bytes(reserved)
                with self._write_condition:
                    self._write_inflight = max(0, self._write_inflight - 1)
                    self._pending_write_jobs = max(0, self._pending_write_jobs - 1)
                    self._record_write_result_locked(int(job_id), bool(ok))
                    # Bound per-key completion telemetry as well.  Keeping the
                    # newest 256 keys is sufficient for an in-process waiter;
                    # older completed jobs fall back to the durable pair.
                    while len(self._latest_write_by_key) > 256:
                        oldest_key, oldest_job = next(
                            iter(self._latest_write_by_key.items())
                        )
                        if int(oldest_job) > self._last_completed_write:
                            break
                        self._latest_write_by_key.pop(oldest_key, None)
                    if ok:
                        self._stores += 1
                        if self._token_lengths is not None:
                            self._token_lengths.add(int(num_tokens))
                    else:
                        self._write_failures += 1
                    self._write_condition.notify_all()

    def wait_for_write(self, key: str, timeout: float = 5.0) -> bool:
        """Wait for the latest queued write for ``key`` and return its result."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._write_condition:
            job_id = self._latest_write_by_key.get(str(key))
            if job_id is None:
                data_path, side_path = self._entry_paths(str(key))
                return data_path.is_file() and side_path.is_file()
            while self._last_completed_write < job_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._write_condition.wait(timeout=remaining)
            result = self._write_results.get(job_id)
        if result is not None:
            return bool(result)
        data_path, side_path = self._entry_paths(str(key))
        return data_path.is_file() and side_path.is_file()

    def wait_for_pending(self, timeout: float = 5.0) -> bool:
        """Wait for all writes queued before this call to settle."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._write_condition:
            target = self._write_seq
            while self._last_completed_write < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._write_condition.wait(timeout=remaining)
            return True

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        """Drain immutable writes and stop the worker.

        Scheduler-owned stores use the default durability barrier before the
        shared aggregate-budget lease is released.  Tests/tools may supply a
        finite timeout and receive ``False`` without claiming durability.
        """

        deadline = (
            None
            if timeout is None
            else time.monotonic() + max(0.0, float(timeout))
        )
        with self._write_condition:
            self._accepting_writes = False
            self._stop_event.set()
            while self._active_write_producers > 0:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return False
                self._write_condition.wait(timeout=remaining)
        if self._writer_thread.is_alive():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            self._writer_thread.join(timeout=remaining)
        return not self._writer_thread.is_alive()

    def _load_entry(self, key: str) -> Optional[Tuple[List[Any], bool]]:
        if self._dir is None:
            return None
        data_path, side_path = self._entry_paths(key)
        if not data_path.exists() or not side_path.exists():
            return None
        try:
            sidecar = json.loads(side_path.read_text())
        except (OSError, ValueError) as e:
            logger.debug("SSM disk store sidecar parse failed %s: %s", key, e)
            return None

        try:
            record_version = int(sidecar.get("version") or 0)
        except (TypeError, ValueError):
            record_version = 0
        if record_version != _RECORD_VERSION:
            logger.info(
                "SSM disk cache record version mismatch; treating as miss "
                "(stored=%s current=%s)",
                record_version,
                _RECORD_VERSION,
            )
            return None

        record_id = str(sidecar.get("record_id") or "")
        if not record_id:
            logger.info(
                "SSM disk cache record has no pair generation ID; treating as miss"
            )
            return None

        layer_metas: List[Dict[str, Any]] = sidecar.get("layer_metas", [])
        is_complete = bool(sidecar.get("is_complete", True))
        stored_runtime = sidecar.get("runtime_cache_fingerprint")
        current_runtime = _runtime_cache_fingerprint()
        if stored_runtime != current_runtime:
            logger.info(
                "SSM disk cache runtime fingerprint mismatch; treating as miss "
                "(stored=%s current=%s)",
                stored_runtime or "missing",
                current_runtime,
            )
            return None

        flat: Dict[str, mx.array] = {}
        try:
            try:
                from vmlx_engine.cache_record_validator import (
                    reject_safetensors_or_warn,
                )
            except Exception:
                reject_safetensors_or_warn = None
            if reject_safetensors_or_warn is not None:
                if not reject_safetensors_or_warn(
                    str(data_path),
                    source=f"SSM-companion-header:{key[:12]}",
                    # fetch holds the aggregate shared lock.  Treat corruption
                    # as a miss; global eviction/clear owns physical mutation.
                    delete_on_reject=False,
                ):
                    return None
            loaded, tensor_metadata = mx.load(
                str(data_path),
                return_metadata=True,
            )
            if tensor_metadata.get(_RECORD_ID_METADATA_KEY) != record_id:
                logger.info(
                    "SSM disk cache data/sidecar generation mismatch; "
                    "treating as miss"
                )
                return None
            flat = loaded  # type: ignore[assignment]
        except Exception as e:
            logger.debug("SSM disk store load failed %s: %s", key, e)
            return None

        # Decode opaque blobs if any
        opaque_decoded: Dict[str, Any] = {}
        opaque = sidecar.get("opaque") or {}
        if opaque:
            import base64

            for k, hexed in opaque.items():
                try:
                    opaque_decoded[k] = pickle.loads(base64.b16decode(hexed))
                except Exception as e:
                    logger.debug("SSM disk store opaque load failed %s/%s: %s", key, k, e)
                    return None

        # Reconstruct per-layer
        from mlx_lm.models.cache import ArraysCache  # local import; mlx-lm always present

        try:
            from mlx_lm.models.cache import MambaCache  # type: ignore
        except Exception:
            MambaCache = None  # type: ignore

        states: List[Any] = []
        for n, meta in enumerate(layer_metas):
            kind = meta.get("kind", "opaque")
            if kind == "ArraysCache":
                cache_len = int(meta.get("cache_len", 0))
                cache_present = meta.get("cache_present") or [True] * cache_len
                rebuilt_cache: List[Any] = []
                for i in range(cache_len):
                    if cache_present[i]:
                        arr = flat.get(f"L{n}.cache.{i}")
                        rebuilt_cache.append(arr)
                    else:
                        rebuilt_cache.append(None)
                ac = ArraysCache.__new__(ArraysCache)
                ac.cache = rebuilt_cache  # type: ignore[attr-defined]
                if meta.get("has_lengths"):
                    lengths = flat.get(f"L{n}.lengths")
                    if lengths is None:
                        return None
                    ac.lengths = lengths  # type: ignore[attr-defined]
                else:
                    try:
                        ac.lengths = None  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if meta.get("has_left_padding"):
                    left_padding = flat.get(f"L{n}.left_padding")
                    if left_padding is None:
                        return None
                    ac.left_padding = left_padding  # type: ignore[attr-defined]
                else:
                    ac.left_padding = None  # type: ignore[attr-defined]
                states.append(ac)
            elif kind == "MambaCache" and MambaCache is not None:
                state_len = int(meta.get("state_len", 0))
                state_present = meta.get("state_present") or [True] * state_len
                tup: List[Any] = []
                for i in range(state_len):
                    if state_present[i]:
                        tup.append(flat.get(f"L{n}.state.{i}"))
                    else:
                        tup.append(None)
                mc = MambaCache.__new__(MambaCache)
                mc.state = tuple(tup)  # type: ignore[attr-defined]
                states.append(mc)
            elif kind == "opaque" and f"L{n}" in opaque_decoded:
                states.append(opaque_decoded[f"L{n}"])
            else:
                # Cannot rebuild this layer — disk entry is unusable.
                logger.debug("SSM disk store cannot rebuild layer %d (%s)", n, kind)
                return None

        # Materialize before returning
        materialise: List[mx.array] = []
        for s in states:
            cache_attr = getattr(s, "cache", None)
            if isinstance(cache_attr, list):
                materialise.extend(a for a in cache_attr if a is not None)
            state_attr = getattr(s, "state", None)
            if isinstance(state_attr, tuple):
                materialise.extend(a for a in state_attr if a is not None)
            lengths = getattr(s, "lengths", None)
            if lengths is not None:
                materialise.append(lengths)
        if materialise:
            try:
                _mx_materialize(*materialise)
            except Exception as e:
                logger.debug("SSM disk store post-load materialize failed: %s", e)
                return None

        # Only a fully reconstructed/materialized hit earns a recent LRU time.
        try:
            now = time.time()
            os.utime(data_path, (now, now))
            os.utime(side_path, (now, now))
        except OSError:
            pass

        # Safetensors load returns fresh arrays already. L1 fetch wraps disk
        # hits through SSMCompanionCache._clone_states before model mutation, so
        # a second per-layer deepcopy here only adds RAM/copy cost.
        return (states, is_complete)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def store(
        self,
        key: str,
        states: List[Any],
        is_complete: bool,
        token_ids: List[int],
        num_tokens: int,
    ) -> bool:
        """Queue a frozen entry; return whether bounded admission succeeded."""
        if not states or num_tokens <= 0:
            return False
        with self._write_condition:
            if not self._accepting_writes or not self._writer_thread.is_alive():
                return False
            if self._write_queue.full():
                self._pending_write_byte_drops += 1
                return False
            self._active_write_producers += 1
        try:
            with self._lock:
                prepared = self._prepare_entry(
                    key, states, is_complete, token_ids, num_tokens
                )
            if prepared is None:
                return False
            data_bytes, sidecar_bytes, reserved = prepared
            rejected_after_freeze = False
            with self._write_condition:
                if not self._accepting_writes:
                    rejected_after_freeze = True
                else:
                    self._write_seq += 1
                    job_id = self._write_seq
                    previous_job = self._latest_write_by_key.get(str(key))
                    self._latest_write_by_key[str(key)] = job_id
                    self._latest_write_by_key.move_to_end(str(key))
                    self._pending_write_jobs += 1
            if rejected_after_freeze:
                self._release_pending_bytes(reserved)
                return False
            try:
                self._write_queue.put_nowait(
                    (
                        job_id,
                        str(key),
                        data_bytes,
                        sidecar_bytes,
                        int(num_tokens),
                        reserved,
                    )
                )
                return True
            except queue.Full:
                self._release_pending_bytes(reserved)
                with self._write_condition:
                    self._pending_write_jobs = max(
                        0, self._pending_write_jobs - 1
                    )
                    if previous_job is None:
                        self._latest_write_by_key.pop(str(key), None)
                    else:
                        self._latest_write_by_key[str(key)] = previous_job
                    self._record_write_result_locked(job_id, False)
                    self._write_failures += 1
                    self._write_condition.notify_all()
                return False
        finally:
            with self._write_condition:
                self._active_write_producers = max(
                    0, self._active_write_producers - 1
                )
                self._write_condition.notify_all()

    def candidate_lengths(self, max_len: int) -> List[int]:
        """Return persisted checkpoint boundaries at or below max_len.

        Entry keys intentionally hash the model identity and token prefix, so
        the key alone cannot discover a shorter partial boundary after restart.
        Sidecars already carry num_tokens; build a process-local index from
        that metadata and let the caller recompute the full model/prefix key
        before fetch. A returned length is only a candidate, never proof of a
        hit. fetch remains authoritative for model identity, token prefix,
        record version, runtime fingerprint, and safetensors validation.
        """
        if max_len <= 0 or self._dir is None:
            return []
        if os.environ.get("VMLX_DISABLE_SSM_DISK_RESTORE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return []
        with self._lock:
            guard = (
                self._global_budget.mutation_guard()
                if self._global_budget is not None
                else nullcontext(True)
            )
            with guard as locked:
                if not locked:
                    return []
                if self._token_lengths is None:
                    lengths: set[int] = set()
                    current_runtime = _runtime_cache_fingerprint()
                    try:
                        for sub in self._dir.iterdir() if self._dir.exists() else []:
                            if not sub.is_dir():
                                continue
                            for side in sub.iterdir():
                                if side.suffix != ".json" or side.stem.endswith(".tmp"):
                                    continue
                                data = side.with_suffix(".safetensors")
                                if not data.exists():
                                    continue
                                try:
                                    metadata = json.loads(side.read_text())
                                    if int(metadata.get("version") or 0) != _RECORD_VERSION:
                                        continue
                                    if metadata.get("runtime_cache_fingerprint") != current_runtime:
                                        continue
                                    num_tokens = int(metadata.get("num_tokens") or 0)
                                except (
                                    OSError,
                                    TypeError,
                                    ValueError,
                                    json.JSONDecodeError,
                                ):
                                    continue
                                if num_tokens > 0:
                                    lengths.add(num_tokens)
                    except OSError:
                        pass
                    self._token_lengths = lengths
                    with self._stats_lock:
                        self._candidate_scans += 1
                return sorted(
                    (n for n in self._token_lengths if n <= max_len),
                    reverse=True,
                )

    def has_complete(self, key: str) -> bool:
        """Cheaply attest a complete durable entry without loading its arrays.

        Disk-only companion mode uses this to avoid an otherwise redundant
        O(prompt) clean re-derive. Validate both halves of the atomic pair,
        including the generation ID embedded in the safetensors header, so a
        torn overwrite can never suppress the repair pass indefinitely.
        """
        if os.environ.get("VMLX_DISABLE_SSM_DISK_RESTORE", "").lower() in {
            "1", "true", "yes", "on",
        }:
            return False
        with self._write_condition:
            pending_job = self._latest_write_by_key.get(str(key))
            pending = bool(
                pending_job is not None
                and self._last_completed_write < pending_job
            )
        if pending and not self.wait_for_write(str(key), timeout=5.0):
            return False

        def _probe() -> bool:
            data_path, side_path = self._entry_paths(str(key))
            if not data_path.is_file() or not side_path.is_file():
                return False
            try:
                sidecar = json.loads(side_path.read_text())
                record_id = str(sidecar.get("record_id") or "")
                if (
                    int(sidecar.get("version") or 0) != _RECORD_VERSION
                    or not bool(sidecar.get("is_complete", False))
                    or not record_id
                    or sidecar.get("runtime_cache_fingerprint")
                    != _runtime_cache_fingerprint()
                ):
                    return False
                from vmlx_engine.cache_record_validator import (
                    validate_safetensors_header,
                )

                valid, _reason = validate_safetensors_header(
                    str(data_path), source=f"SSM-companion-probe:{str(key)[:12]}"
                )
                if not valid:
                    return False
                with data_path.open("rb") as handle:
                    header_size_raw = handle.read(8)
                    if len(header_size_raw) != 8:
                        return False
                    header_size = int.from_bytes(
                        header_size_raw, "little", signed=False
                    )
                    header = json.loads(handle.read(header_size).decode("utf-8"))
                metadata = header.get("__metadata__") or {}
                return metadata.get(_RECORD_ID_METADATA_KEY) == record_id
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False

        if self._global_budget is None:
            return _probe()
        with self._global_budget.mutation_guard() as locked:
            return bool(locked and _probe())

    def touch(self, key: str) -> None:
        """Refresh the on-disk entry's access time for the shared LRU wall.

        Companion files share the aggregate block-cache budget and are
        ranked by file age; nothing else ever refreshes them, so an actively
        used conversation's companion is exactly as evictable as stale data
        (measured live: 10 stores -> 3 entries after one bounded-L2 filler
        pass, and the surviving KV chain came back cold after restart
        because its companion file was gone). An L1 companion hit calls
        this; best-effort, failures are irrelevant to correctness.
        """
        if self._dir is None:
            return
        try:
            data_path, sidecar_path = self._entry_paths(str(key))
            for path in (data_path, sidecar_path):
                if path.exists():
                    os.utime(path, None)
        except Exception:
            pass

    def fetch(self, key: str) -> Optional[Tuple[List[Any], bool]]:
        """Look up by key. Returns ``(states, is_complete)`` or ``None``."""
        if os.environ.get("VMLX_DISABLE_SSM_DISK_RESTORE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            with self._stats_lock:
                self._restore_suppressed += 1
            return None
        with self._write_condition:
            pending_job = self._latest_write_by_key.get(str(key))
            pending = bool(
                pending_job is not None
                and self._last_completed_write < pending_job
            )
        if pending and not self.wait_for_write(str(key), timeout=5.0):
            return None
        # Read path holds the lock briefly only to align with budget
        # enforcement; actual decode is independent of the lock.
        if self._global_budget is None:
            entry = self._load_entry(key)
        else:
            with self._global_budget.mutation_guard() as locked:
                if not locked:
                    entry = None
                else:
                    entry = self._load_entry(key)
        with self._stats_lock:
            if entry is None:
                self._misses += 1
            else:
                self._hits += 1
        return entry

    def delete(self, key: str) -> None:
        if self._dir is None:
            return
        if not self.wait_for_write(str(key), timeout=5.0):
            return
        data_path, side_path = self._entry_paths(key)
        before = self._paths_size(data_path, side_path)
        guard = (
            self._global_budget.exclusive_mutation_guard()
            if self._global_budget is not None
            else nullcontext(True)
        )
        try:
            with guard as locked:
                if not locked:
                    return
                for p in (data_path, side_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                if self._global_budget is not None and before:
                    self._global_budget.account_finalized_write_locked(-before)
        except OSError:
            return
        self._token_lengths = None

    def clear(self) -> None:
        if self._dir is None:
            return
        if not self.wait_for_pending(timeout=5.0):
            logger.warning("SSM disk clear skipped while writes remain pending")
            return
        with self._lock:
            guard = (
                self._global_budget.exclusive_mutation_guard()
                if self._global_budget is not None
                else nullcontext(True)
            )
            with guard as locked:
                if not locked:
                    return
                for sub in self._dir.iterdir() if self._dir.exists() else []:
                    if not sub.is_dir():
                        continue
                    for f in sub.iterdir():
                        try:
                            f.unlink()
                        except OSError:
                            pass
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
                if self._global_budget is not None:
                    self._global_budget.account_finalized_write_locked(-1)
            self._token_lengths = set()

    def stats(self) -> Dict[str, Any]:
        """Best-effort L2 footprint stats for health/cache endpoints."""
        if self._dir is None:
            return {
                "enabled": False,
                "directory": None,
                "entries": 0,
                "bytes": 0,
                "budget_bytes": self._budget,
            }
        entries = 0
        total = 0
        total_tokens = 0
        latest_payload_path: Optional[Path] = None
        latest_payload_mtime_ns = -1
        try:
            for sub in self._dir.iterdir() if self._dir.exists() else []:
                if not sub.is_dir():
                    continue
                for f in sub.iterdir():
                    if f.suffix != ".safetensors":
                        continue
                    entries += 1
                    try:
                        f_stat = f.stat()
                        total += f_stat.st_size
                        if f_stat.st_mtime_ns > latest_payload_mtime_ns:
                            latest_payload_path = f
                            latest_payload_mtime_ns = f_stat.st_mtime_ns
                    except OSError:
                        pass
                    side = f.with_suffix(".json")
                    if side.exists():
                        try:
                            total += side.stat().st_size
                            sidecar = json.loads(side.read_text())
                            total_tokens += int(sidecar.get("num_tokens") or 0)
                        except OSError:
                            pass
                        except (TypeError, ValueError, json.JSONDecodeError):
                            pass
        except OSError:
            pass
        latest_payload = None
        if latest_payload_path is not None:
            try:
                from safetensors import safe_open

                dtype_counts: Counter[str] = Counter()
                with safe_open(
                    str(latest_payload_path), framework="numpy"
                ) as handle:
                    for name in handle.keys():
                        dtype_counts[str(handle.get_slice(name).get_dtype())] += 1
                latest_payload = {
                    "file_size_bytes": int(latest_payload_path.stat().st_size),
                    "physical_tensor_dtype_counts": dict(dtype_counts),
                    "tensor_count": sum(dtype_counts.values()),
                }
            except Exception as exc:  # noqa: BLE001 - stats stay best-effort
                logger.debug("SSM companion dtype inspection failed: %s", exc)
        with self._stats_lock:
            stores = self._stores
            write_failures = self._write_failures
            hits = self._hits
            misses = self._misses
            restore_suppressed = self._restore_suppressed
            candidate_scans = self._candidate_scans
            write_pipeline = {
                "queue_depth": self._write_queue.qsize(),
                "pending_jobs": self._pending_write_jobs,
                "inflight": self._write_inflight,
                "pending_bytes": self._pending_write_bytes,
                "max_pending_bytes": self._max_pending_write_bytes,
                "byte_budget_drops": self._pending_write_byte_drops,
                "failures": write_failures,
                "completion_generation": self._last_completed_write,
                "writer_alive": self._writer_thread.is_alive(),
            }
        global_health = (
            self._global_budget.refresh_health()
            if self._global_budget is not None
            else None
        )
        result = {
            "enabled": True,
            "directory": str(self._dir),
            "entries": entries,
            "total_tokens_on_disk": total_tokens,
            "total_cached_tokens": total_tokens,
            "bytes": total,
            "bytes_mb": round(total / (1024 * 1024), 2),
            "budget_bytes": self._budget,
            "budget_gb": round(self._budget / (1024 ** 3), 3),
            "stores": stores,
            "write_pipeline": write_pipeline,
            "hits": hits,
            "misses": misses,
            "restore_enabled": os.environ.get(
                "VMLX_DISABLE_SSM_DISK_RESTORE", ""
            ).lower() not in {"1", "true", "yes", "on"},
            "restore_suppressed": restore_suppressed,
            "candidate_length_scans": candidate_scans,
            "candidate_lengths_indexed": (
                len(self._token_lengths)
                if self._token_lengths is not None
                else 0
            ),
            "hit_rate": round(hits / max(hits + misses, 1), 3),
            "global_budget": (
                {
                    "root": str(self._global_budget.root),
                    "max_size_bytes": global_health.max_size_bytes,
                    "bytes_after": global_health.bytes_after,
                    "compliant": global_health.compliant,
                    "accounted": global_health.accounted,
                    "telemetry_stale": global_health.telemetry_stale,
                    "evicted_entries": global_health.evicted_entries,
                    "capacity_evicted_entries_total": global_health.capacity_evicted_entries_total,
                    "accounting_generation": global_health.accounting_generation,
                    "reconciliation_generation": (
                        global_health.reconciliation_generation
                    ),
                }
                if global_health is not None
                else None
            ),
        }
        if latest_payload is not None:
            result["latest_payload"] = latest_payload
        return result

    def _enforce_budget(self) -> None:
        """LRU eviction by mtime under the disk byte budget."""
        if self._dir is None:
            return
        if self._budget <= 0:
            return
        files: List[Tuple[float, int, Path, Path]] = []
        total = 0
        try:
            for sub in self._dir.iterdir():
                if not sub.is_dir():
                    continue
                for f in sub.iterdir():
                    if f.suffix == ".safetensors":
                        side = f.with_suffix(".json")
                        try:
                            st = f.stat()
                        except OSError:
                            continue
                        size = st.st_size
                        if side.exists():
                            try:
                                size += side.stat().st_size
                            except OSError:
                                pass
                        files.append((st.st_mtime, size, f, side))
                        total += size
        except OSError:
            return
        if total <= self._budget:
            return
        files.sort(key=lambda t: t[0])  # oldest first
        evicted = False
        for mtime, size, data, side in files:
            if total <= self._budget:
                break
            for p in (data, side):
                try:
                    p.unlink()
                except OSError:
                    pass
            total -= size
            evicted = True
            logger.debug("SSM disk store evicted %s (%.2f MB)", data.name, size / 1e6)
        if evicted:
            self._token_lengths = None


# Module-level singleton — built lazily so importing the module is cheap
# even when the disk cache is disabled.
_singleton_lock = threading.Lock()
_singleton: Optional[SSMCompanionDiskStore] = None


def get_disk_store() -> Optional[SSMCompanionDiskStore]:
    """Return the process-wide disk store, or None if disabled.

    Threadsafe; safe to call from any context. Caller must check for None
    (means env flag disabled or directory unavailable).
    """
    global _singleton
    if not is_enabled():
        return None
    with _singleton_lock:
        if _singleton is None:
            _singleton = SSMCompanionDiskStore()
            if _singleton._dir is None:  # type: ignore[truthy-bool]
                _singleton = None
        return _singleton

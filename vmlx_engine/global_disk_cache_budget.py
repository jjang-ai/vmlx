# SPDX-License-Identifier: Apache-2.0
"""Process-safe aggregate byte budget for block-cache SSD payloads.

``BlockDiskStore`` indexes one model/config namespace at a time.  Applying the
user's ``--block-disk-cache-max-gb`` value independently to every namespace
allows the physical cache root to grow by ``N * max_gb`` and hybrid models can
add a second, equally large SSM-companion budget.  This module owns the one
aggregate LRU trim for the whole configured block-cache root.

Only finalized cache payloads are managed:

* SQLite-indexed block ``.safetensors`` files, using index ``last_accessed``;
* paired SSM companion ``.safetensors`` + ``.json`` records, using mtime; and
* old finalized orphan payloads inside known cache layout directories.

Temporary/in-flight files are counted as physical usage but never deleted while
their writer lease is live.  Cooperating writers hold a shared ``flock`` while
publishing final files; eviction holds the exclusive lock.  Older/uncoordinated
writers are protected by an orphan grace period, and any ambiguous
database/path/lock condition aborts the trim without affecting inference.
"""

from __future__ import annotations

import atexit
import fcntl
import heapq
import json
import logging
import os
import re
import sqlite3
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_NAME = ".vmlx-global-cache-budget.lock"
_LEASE_DIR_NAME = ".vmlx-global-cache-budget-leases"
_ACCOUNTING_NAME = ".vmlx-global-cache-budget-accounting.json"
_ROOT_MARKER_NAME = ".vmlx-block-cache-root-v1"
_NAMESPACE_MARKER_NAME = ".vmlx-block-cache-namespace-v1"
_STATE_VERSION = 1
_DEFAULT_ORPHAN_GRACE_SECONDS = 5 * 60
_DEFAULT_RECONCILE_INTERVAL_SECONDS = 30.0
_DEFAULT_BIRTH_VALIDATION_INTERVAL_SECONDS = 30.0
_SQLITE_TRANSIENT_SIDECAR_NAMES = frozenset(
    {
        "block_index.db-wal",
        "block_index.db-shm",
        "block_index.db-journal",
    }
)

_ROOT_THREAD_LOCKS_GUARD = threading.Lock()
_ROOT_THREAD_LOCKS: dict[Path, threading.RLock] = {}


def _root_thread_lock(root: Path) -> threading.RLock:
    with _ROOT_THREAD_LOCKS_GUARD:
        return _ROOT_THREAD_LOCKS.setdefault(root, threading.RLock())


def _is_sqlite_transient_sidecar(path: Path, namespace: Path) -> bool:
    """Return whether this is a real namespace-root SQLite index sidecar."""

    database = namespace / "block_index.db"
    return bool(
        path.parent == namespace
        and path.name in _SQLITE_TRANSIENT_SIDECAR_NAMES
        and not database.is_symlink()
        and database.is_file()
    )


def _process_birth_identity(pid: int) -> str | None:
    """Return a stable process-start identity, not merely a reusable PID."""

    if pid <= 0:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        raw = proc_stat.read_text()
        # Field 2 (comm) may contain spaces/parentheses.  Fields after its final
        # ')' start at field 3; starttime is field 22, hence index 19 here.
        remainder = raw[raw.rfind(")") + 2 :].split()
        if len(remainder) > 19:
            return f"proc-start-ticks:{remainder[19]}"
    except (OSError, ValueError):
        pass
    try:
        started = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(int(pid))],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return f"ps-lstart:{started}" if started else None


def ensure_managed_block_cache_namespace(
    namespace: str | os.PathLike[str],
) -> Path:
    """Claim only an empty or unambiguously legacy block-cache namespace."""

    path = Path(namespace).expanduser()
    if path.is_symlink():
        raise OSError(f"refusing symlinked block-cache namespace: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"invalid block-cache namespace: {path}")
    resolved_namespace = path.resolve(strict=True)

    recognized_types = {
        "blocks": "directory",
        "ssm_companion": "directory",
        "block_index.db": "file",
        "block_index.db-wal": "file",
        "block_index.db-shm": "file",
        "block_index.db-journal": "file",
    }
    for name, expected_type in recognized_types.items():
        entry = path / name
        # ``Path.exists`` is false for a broken symlink, so test the link first.
        if entry.is_symlink():
            raise OSError(f"refusing symlinked block-cache path: {entry}")
        if not entry.exists():
            continue
        if expected_type == "directory" and not entry.is_dir():
            raise OSError(f"expected block-cache directory: {entry}")
        if expected_type == "file" and not entry.is_file():
            if _is_sqlite_transient_sidecar(entry, path) and not entry.exists():
                continue
            raise OSError(f"expected block-cache file: {entry}")
        try:
            resolved_entry = entry.resolve(strict=True)
        except FileNotFoundError:
            if _is_sqlite_transient_sidecar(entry, path):
                continue
            raise
        if not resolved_entry.is_relative_to(resolved_namespace):
            raise OSError(f"block-cache path escaped its namespace: {entry}")

    marker = path / _NAMESPACE_MARKER_NAME
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise OSError(f"invalid block-cache namespace marker: {marker}")
        return resolved_namespace

    existing = {entry.name for entry in path.iterdir()}
    legacy_names = {
        "blocks",
        "block_index.db",
        "block_index.db-wal",
        "block_index.db-shm",
        "block_index.db-journal",
        "ssm_companion",
    }
    if existing and not existing.issubset(legacy_names):
        raise OSError(
            "refusing to claim non-empty unrecognized block-cache namespace "
            f"{path}"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(marker, flags, 0o600)
    except FileExistsError as exc:
        if marker.is_symlink() or not marker.is_file():
            raise OSError(
                f"invalid block-cache namespace marker: {marker}"
            ) from exc
    else:
        os.close(fd)
    return resolved_namespace


@dataclass(frozen=True)
class _BudgetCandidate:
    kind: str
    size_bytes: int
    last_accessed_ns: int
    paths: tuple[Path, ...]
    database: Path | None = None
    block_hash: str | None = None
    indexed_file_name: str | None = None
    parent_hash: str | None = None
    ancestry_known: bool = False
    publication_pinned: bool = False


@dataclass(frozen=True)
class GlobalBudgetResult:
    """Best-effort result from one aggregate trim."""

    max_size_bytes: int
    bytes_before: int
    bytes_after: int
    evicted_entries: int
    evicted_bytes: int
    protected_recent_orphans: int
    compliant: bool
    scan_performed: bool
    reconciled_at_ns: int
    accounted: bool
    accounting_generation: int
    reconciliation_generation: int
    error: str | None = None
    # Cumulative totals from the shared ledger: every eviction any writer in
    # any process performed on this root since the ledger was created. The
    # per-reconcile fields above only describe the LAST reconcile, and a
    # store's own counter never sees evictions triggered by another writer
    # (live: the SSM companion writer evicted 7 entries while the block
    # store's counter and the last-reconcile fields both read 0).
    evicted_entries_total: int = 0
    evicted_bytes_total: int = 0
    # temp files still protected by a live lease or the orphan grace at the last
    # scan (a kill inside a write leaves such files; the idle pass owes them a
    # later look)
    protected_temp_files: int = 0


class GlobalDiskCacheBudget:
    """Aggregate LRU budget shared by all namespaces below one cache root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        max_size_bytes: int,
        *,
        orphan_grace_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS,
        reconcile_interval_seconds: float = _DEFAULT_RECONCILE_INTERVAL_SECONDS,
        allow_legacy_hashed_namespaces: bool = False,
        allow_legacy_direct_namespace: bool = False,
    ) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()
        self._requested_max_size_bytes = max(0, int(max_size_bytes))
        self._orphan_grace_ns = max(
            0,
            int(float(orphan_grace_seconds) * 1_000_000_000),
        )
        self._reconcile_interval_ns = max(
            0,
            int(float(reconcile_interval_seconds) * 1_000_000_000),
        )
        self._allow_legacy_hashed_namespaces = bool(
            allow_legacy_hashed_namespaces
        )
        self._allow_legacy_direct_namespace = bool(
            allow_legacy_direct_namespace
        )
        self._thread_lock = _root_thread_lock(self.root)
        self._last_result: GlobalBudgetResult | None = None
        self._last_reconcile_monotonic_ns = 0
        # Set when a periodic (interval-due) physical rescan was owed by an
        # accounting call that ran inside a publication critical section.
        # The rescan walks every managed namespace and opens every block
        # index; on a root with hundreds of namespaces that is seconds, and
        # a request's terminal durability barrier must never wait for it.
        # The owning store runs it from its writer thread while idle.
        self._deferred_reconcile_due = False
        self._lease_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._lease_owner_pid = os.getpid()
        self._lease_owner_birth_identity = _process_birth_identity(
            self._lease_owner_pid
        )
        self._birth_validation_interval_ns = int(
            _DEFAULT_BIRTH_VALIDATION_INTERVAL_SECONDS * 1_000_000_000
        )
        self._birth_validation_cache: dict[tuple[int, str], tuple[int, bool]] = {}
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._ensure_root_marker()
        self._publish_budget(self._requested_max_size_bytes)
        self._atexit_callback = self.close
        atexit.register(self._atexit_callback)

    @property
    def requested_max_size_bytes(self) -> int:
        return self._requested_max_size_bytes

    @property
    def last_result(self) -> GlobalBudgetResult | None:
        return self._last_result

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def _open_lock(self) -> int:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.root / _LOCK_NAME, flags, 0o600)
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            os.close(fd)
            raise OSError("global cache budget lock is not a regular file")
        return fd

    def _ensure_root_marker(self) -> None:
        marker = self.root / _ROOT_MARKER_NAME
        try:
            flags = os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(marker, flags, 0o600)
            mode = os.fstat(fd).st_mode
            os.close(fd)
            if not stat.S_ISREG(mode):
                raise OSError("managed root marker is not a regular file")
        except OSError as exc:
            raise OSError(f"cannot establish managed block-cache root: {exc}") from exc

    @contextmanager
    def mutation_guard(self) -> Iterator[bool]:
        """Hold the process-shared publication lock for a finalized write.

        The boolean is false only when the lock cannot be acquired.  Callers
        must fail the cache operation closed in that case; publishing without
        the lock would make an eviction race possible.
        """

        with self._thread_lock:
            if self._closed:
                yield False
                return
            fd: int | None = None
            try:
                fd = self._open_lock()
                fcntl.flock(fd, fcntl.LOCK_SH)
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                logger.warning(
                    "Global block-cache publication lock unavailable; "
                    "cache operation must fail closed: %s",
                    exc,
                )
                yield False
                return
            try:
                yield True
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    @contextmanager
    def _exclusive_guard(self) -> Iterator[None]:
        with self._thread_lock:
            fd = self._open_lock()
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    @contextmanager
    def exclusive_mutation_guard(self) -> Iterator[bool]:
        """Hold the root-exclusive lock across publish/delete and accounting."""

        with self._thread_lock:
            if self._closed:
                yield False
                return
            fd: int | None = None
            try:
                fd = self._open_lock()
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                logger.warning(
                    "Global block-cache exclusive mutation lock unavailable; "
                    "cache operation must fail closed: %s",
                    exc,
                )
                yield False
                return
            try:
                yield True
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _lease_dir(self) -> Path:
        return self.root / _LEASE_DIR_NAME

    def _lease_path(self) -> Path:
        return self._lease_dir() / f"{self._lease_id}.json"

    def _publish_budget(self, max_size_bytes: int) -> None:
        """Publish this live process's requested cap as a root lease."""

        try:
            with self._exclusive_guard():
                lease_dir = self._lease_dir()
                with suppress(FileExistsError):
                    lease_dir.mkdir(parents=True, exist_ok=False)
                if lease_dir.is_symlink() or not lease_dir.is_dir():
                    raise OSError(
                        f"invalid global cache budget lease directory: {lease_dir}"
                    )
                state = {
                    "version": _STATE_VERSION,
                    "max_size_bytes": max(0, int(max_size_bytes)),
                    "updated_at_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "process_birth_identity": self._lease_owner_birth_identity,
                }
                temp = self._lease_dir() / (
                    f"{self._lease_id}.{threading.get_ident()}."
                    f"{uuid.uuid4().hex}.tmp"
                )
                try:
                    fd = os.open(
                        temp,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    try:
                        payload = json.dumps(state, sort_keys=True).encode("utf-8")
                        os.write(fd, payload)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    os.replace(temp, self._lease_path())
                finally:
                    with suppress(OSError):
                        temp.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not publish global block-cache budget: %s", exc)
            # The caller can disable Block Disk L2 and continue inference, but
            # a cache store must not run without its cap/lease coordination.
            raise

    def _remove_lease(self) -> bool:
        """Compatibility alias for tests and older direct callers."""

        return self.close()

    def close(self) -> bool:
        """Release this owner's live cap after every dependent writer stops.

        ``BlockDiskStore`` owns the coordinator; an SSM companion merely shares
        it and must not close it independently.  A failed removal leaves the
        atexit callback registered so process exit can retry instead of making
        a still-live lower ceiling disappear silently.
        """

        with self._lifecycle_lock:
            if self._closed:
                return True
            if os.getpid() != self._lease_owner_pid:
                # A forked child never owns the parent's lease.
                self._closed = True
                with suppress(Exception):
                    atexit.unregister(self._atexit_callback)
                return True
            try:
                # Keep the local root lock through the closed-state flip so a
                # new shared mutation cannot slip between lease removal and
                # observing that this coordinator is no longer an owner.
                with self._thread_lock, self._exclusive_guard():
                    self._lease_path().unlink(missing_ok=True)
                    self._closed = True
            except OSError as exc:
                logger.warning(
                    "Could not remove global block-cache budget lease %s: %s",
                    self._lease_id,
                    exc,
                )
                return False
            with suppress(Exception):
                atexit.unregister(self._atexit_callback)
            return True

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _lease_birth_matches(self, pid: int, stored_birth: str) -> bool:
        """Validate PID birth at a bounded cadence while holding root lock.

        A missing birth marker is a legacy/unavailable-probe lease.  Keep it
        live while its PID exists (fail closed: a possibly stale low cap may
        remain, but the user ceiling is never silently relaxed).  New leases on
        macOS/Linux normally have a birth marker.
        """

        if not stored_birth:
            return True
        if (
            pid == self._lease_owner_pid
            and stored_birth == (self._lease_owner_birth_identity or "")
        ):
            return True
        key = (pid, stored_birth)
        now_ns = time.monotonic_ns()
        cached = self._birth_validation_cache.get(key)
        if (
            cached is not None
            and now_ns - cached[0] < self._birth_validation_interval_ns
        ):
            return cached[1]
        current_birth = _process_birth_identity(pid)
        # Failure to probe must not relax a finite cap or expose an in-flight
        # temp owned by a process that may still be alive.
        matches = current_birth is None or current_birth == stored_birth
        self._birth_validation_cache[key] = (now_ns, matches)
        return matches

    def _effective_max_size_bytes_locked(self) -> int:
        lease_dir = self._lease_dir()
        if not lease_dir.exists():
            return self._requested_max_size_bytes
        active_caps: list[int] = []
        try:
            paths = list(lease_dir.iterdir())
        except OSError as exc:
            raise OSError(f"cannot inspect global cache budget leases: {exc}") from exc
        for path in paths:
            if path.name.endswith(".tmp"):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise OSError(f"invalid global cache budget lease path: {path}")
            try:
                state = json.loads(path.read_text())
                if int(state.get("version") or 0) != _STATE_VERSION:
                    raise ValueError("unsupported lease version")
                pid = int(state["pid"])
                cap = max(0, int(state["max_size_bytes"]))
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Removing invalid atomically-published cache budget lease %s: %s",
                    path,
                    exc,
                )
                with suppress(OSError):
                    path.unlink()
                continue
            if not self._pid_is_alive(pid):
                with suppress(OSError):
                    path.unlink()
                continue
            stored_birth = str(state.get("process_birth_identity") or "")
            if not self._lease_birth_matches(pid, stored_birth):
                logger.info(
                    "Removing stale cache-budget lease whose PID was reused: %s",
                    path,
                )
                with suppress(OSError):
                    path.unlink()
                continue
            active_caps.append(cap)
        if not active_caps:
            return self._requested_max_size_bytes
        finite = [cap for cap in active_caps if cap > 0]
        # Unlimited may never relax another live process's finite ceiling.
        return min(finite) if finite else 0

    def _accounting_path(self) -> Path:
        return self.root / _ACCOUNTING_NAME

    def _read_accounting_locked(self) -> dict[str, int] | None:
        accounting_path = self._accounting_path()
        if accounting_path.is_symlink():
            raise OSError("global cache accounting path is a symlink")
        try:
            state = json.loads(accounting_path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise OSError(f"invalid global cache accounting state: {exc}") from exc
        try:
            if int(state.get("version") or 0) != _STATE_VERSION:
                raise ValueError("unsupported accounting state version")
            return {
                "bytes_estimate": max(0, int(state["bytes_estimate"])),
                "accounting_generation": max(
                    0,
                    int(state.get("accounting_generation") or 0),
                ),
                "reconciliation_generation": max(
                    0,
                    int(state.get("reconciliation_generation") or 0),
                ),
                "reconciled_at_ns": max(
                    0,
                    int(state.get("reconciled_at_ns") or 0),
                ),
                "evicted_entries_total": max(
                    0, int(state.get("evicted_entries_total") or 0)
                ),
                "evicted_bytes_total": max(
                    0, int(state.get("evicted_bytes_total") or 0)
                ),
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise OSError(f"invalid global cache accounting values: {exc}") from exc

    def _read_or_reset_accounting_locked(self) -> dict[str, int] | None:
        """Drop corrupt derived accounting so physical scan can rebuild it."""

        try:
            return self._read_accounting_locked()
        except OSError as exc:
            logger.warning(
                "Resetting invalid derived global cache accounting state: %s",
                exc,
            )
            with suppress(OSError):
                self._accounting_path().unlink()
            return None

    def _write_accounting_locked(self, state: dict[str, int]) -> None:
        temp = self.root / (
            f"{_ACCOUNTING_NAME}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid.uuid4().hex}.tmp"
        )
        payload = {
            "version": _STATE_VERSION,
            **{key: max(0, int(value)) for key, value in state.items()},
        }
        try:
            fd = os.open(
                temp,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
            finally:
                os.close(fd)
            os.replace(temp, self._accounting_path())
        finally:
            with suppress(OSError):
                temp.unlink(missing_ok=True)

    def account_finalized_write(
        self,
        net_bytes_delta: int,
        *,
        require_reconciled: bool = False,
    ) -> GlobalBudgetResult:
        """Safely reconcile a mutation not held in our exclusive transaction.

        A caller that already released its physical-mutation lock cannot apply
        a delta without a scan: another process may have reconciled those bytes
        in between. Fast O(1) writers use ``exclusive_mutation_guard`` plus
        ``account_finalized_write_locked`` instead.
        """

        try:
            with self._exclusive_guard():
                return self._enforce_locked()
        except (OSError, sqlite3.Error) as exc:
            return GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=False,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )

    def account_finalized_write_locked(
        self,
        net_bytes_delta: int,
        *,
        require_reconciled: bool = False,
        protected_blocks: set[tuple[Path, str]] | None = None,
    ) -> GlobalBudgetResult:
        """Account a mutation while ``exclusive_mutation_guard`` is held.

        Keeping final publication/deletion and ledger mutation in one root
        critical section prevents both double-add and double-subtract races
        with another process's physical reconciliation.
        """

        delta = int(net_bytes_delta)
        if delta < 0:
            return self._enforce_locked(protected_blocks=protected_blocks)
        maximum = self._effective_max_size_bytes_locked()
        state = self._read_or_reset_accounting_locked()
        force_reconcile = bool(require_reconciled or state is None)
        if state is None:
            state = {
                "bytes_estimate": 0,
                "accounting_generation": 0,
                "reconciliation_generation": 0,
                "reconciled_at_ns": 0,
            }
        estimate = max(0, state["bytes_estimate"] + delta)
        generation = state["accounting_generation"] + 1
        self._write_accounting_locked(
            {
                **state,
                "bytes_estimate": estimate,
                "accounting_generation": generation,
            }
        )
        interval_due = (
            state["reconciled_at_ns"] <= 0
            or time.time_ns() - state["reconciled_at_ns"]
            >= self._reconcile_interval_ns
        )
        # Inline (inside the publication critical section) only when the
        # write cannot be published without it: a strict fence, missing
        # accounting, or an estimate above the ceiling. A merely interval-due
        # rescan is deferred to the owner's idle time; publication then costs
        # one O(1) ledger update instead of a full root scan.
        force_reconcile = bool(
            force_reconcile
            or (maximum > 0 and estimate > maximum)
        )
        if force_reconcile:
            return self._enforce_locked(protected_blocks=protected_blocks)
        if interval_due:
            self._deferred_reconcile_due = True
        result = GlobalBudgetResult(
            max_size_bytes=maximum,
            bytes_before=estimate,
            bytes_after=estimate,
            evicted_entries=0,
            evicted_bytes=0,
            protected_recent_orphans=0,
            compliant=maximum <= 0 or estimate <= maximum,
            scan_performed=False,
            reconciled_at_ns=state["reconciled_at_ns"],
            accounted=True,
            accounting_generation=generation,
            reconciliation_generation=state["reconciliation_generation"],
        )
        self._last_result = result
        return result

    @property
    def deferred_reconcile_due(self) -> bool:
        """True when an interval-due physical rescan is owed off the publish path."""

        return bool(self._deferred_reconcile_due)

    def idle_reconcile_due(self) -> bool:
        """True when an IDLE pass is owed: no publish has raised the deferred
        flag, the interval has elapsed since this process last reconciled, and
        the last scan left something only time can settle — the root was above
        its ceiling, or dead-lease temp files were protected by the orphan grace.
        A compliant root without protected orphans never triggers idle scanning.
        Live: a kill -9 inside a companion write left 4 temp files that the
        restart's trim protected; they were removed only when the next WRITE
        happened after the grace. This pass removes them on idle instead."""

        with self._thread_lock:
            last = self._last_result
            if last is None or self._deferred_reconcile_due:
                return False
            if (
                time.monotonic_ns() - self._last_reconcile_monotonic_ns
                < self._reconcile_interval_ns
            ):
                return False
            return (
                (not last.compliant)
                or int(last.protected_recent_orphans or 0) > 0
                or int(last.protected_temp_files or 0) > 0
            )

    def run_idle_reconcile(self) -> GlobalBudgetResult | None:
        """Perform an owed idle pass (see ``idle_reconcile_due``)."""

        if not self.idle_reconcile_due():
            return None
        return self.enforce(force=True)

    def run_deferred_reconcile(self) -> GlobalBudgetResult | None:
        """Perform the owed periodic rescan under the root-exclusive lock.

        Returns ``None`` when nothing is owed. The flag is cleared only after a
        scan actually ran, so a skipped (lock-unavailable / errored) attempt is
        retried on the next idle opportunity.
        """

        if not self._deferred_reconcile_due:
            return None
        result = self.enforce(force=True)
        # enforce() reports scan_performed=True on its error path too (with
        # accounted=False and an error); only a completed, accounted scan
        # settles the debt. A failed or lock-unavailable attempt is retried on
        # the next quiet opportunity.
        if result.scan_performed and result.accounted and not result.error:
            self._deferred_reconcile_due = False
        return result

    @staticmethod
    def _is_temp_path(path: Path) -> bool:
        name = path.name
        return (
            ".tmp." in name
            or name.endswith(".tmp")
            or name.endswith(".tmp.safetensors")
            or name.endswith(".tmp.json")
        )

    def _active_lease_ids_locked(self) -> set[str]:
        active: set[str] = set()
        lease_dir = self._lease_dir()
        if not lease_dir.exists():
            return active
        for path in lease_dir.iterdir():
            if path.suffix != ".json" or path.is_symlink() or not path.is_file():
                continue
            try:
                state = json.loads(path.read_text())
                pid = int(state["pid"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if not self._pid_is_alive(pid):
                continue
            stored_birth = str(state.get("process_birth_identity") or "")
            if not self._lease_birth_matches(pid, stored_birth):
                continue
            active.add(path.stem)
        return active

    @staticmethod
    def _temp_owner_lease(path: Path) -> str | None:
        match = re.search(r"\.(\d+-[0-9a-f]{32})\.", path.name)
        return match.group(1) if match else None

    def _safe_resolved_file(self, path: Path) -> Path | None:
        if self._is_temp_path(path) or path.is_symlink():
            return None
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.root):
                return None
            if not resolved.is_file() or resolved.is_symlink():
                return None
            return resolved
        except OSError:
            return None

    def _managed_namespace_dirs_locked(self) -> list[Path]:
        namespaces: list[Path] = []
        root_marker = self.root / _NAMESPACE_MARKER_NAME
        if root_marker.is_symlink():
            raise OSError(f"refusing symlinked namespace marker: {root_marker}")
        if root_marker.is_file():
            namespaces.append(self.root)
        elif (
            self._allow_legacy_direct_namespace
            and not (self.root / "block_index.db").is_symlink()
            and (self.root / "block_index.db").is_file()
            and not (self.root / "blocks").is_symlink()
            and (self.root / "blocks").is_dir()
        ):
            # A pre-namespace custom root is owned cache data but unscoped, so
            # it is never reused for a new model. It still participates in the
            # physical cap and can be evicted without touching unrelated files
            # beside the recognized layout.
            namespaces.append(self.root)
        try:
            children = list(self.root.iterdir())
        except OSError as exc:
            raise OSError(f"cannot inspect managed cache root: {exc}") from exc
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name == _LEASE_DIR_NAME:
                continue
            child_marker = child / _NAMESPACE_MARKER_NAME
            if child_marker.is_symlink():
                raise OSError(
                    f"refusing symlinked namespace marker: {child_marker}"
                )
            if child_marker.is_file():
                namespaces.append(child.resolve())
                continue
            if (
                self._allow_legacy_hashed_namespaces
                and re.fullmatch(r"(?:[0-9a-f]{12}|default)", child.name)
                and (child / "block_index.db").is_file()
                and (child / "blocks").is_dir()
            ):
                # The built-in default root predates namespace markers. Its
                # hash/default layout is unambiguous and may be budgeted, but
                # arbitrary custom-root children never receive this exception.
                namespaces.append(child.resolve())
        return sorted(set(namespaces))

    def _database_paths_locked(self) -> list[Path]:
        databases: list[Path] = []
        for namespace in self._managed_namespace_dirs_locked():
            database = namespace / "block_index.db"
            if not database.exists():
                continue
            if database.is_symlink():
                raise OSError(f"refusing symlinked block index: {database}")
            databases.append(database)
        return databases

    def _scan_locked(
        self,
        *,
        now_ns: int,
    ) -> tuple[list[_BudgetCandidate], int, int]:
        candidates: list[_BudgetCandidate] = []
        referenced: set[Path] = set()
        protected_recent_orphans = 0
        active_lease_ids = self._active_lease_ids_locked()

        all_files: set[Path] = set()
        sqlite_transient_sidecars: set[Path] = set()
        for namespace in self._managed_namespace_dirs_locked():
            legacy_direct = bool(
                namespace == self.root
                and not (self.root / _NAMESPACE_MARKER_NAME).is_file()
            )
            walk_roots = (
                [
                    path
                    for path in (
                        self.root / "blocks",
                        self.root / "ssm_companion",
                    )
                    if path.is_dir() and not path.is_symlink()
                ]
                if legacy_direct
                else [namespace]
            )
            if legacy_direct:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    path = Path(f"{self.root / 'block_index.db'}{suffix}")
                    if path.is_file() and not path.is_symlink():
                        resolved = path.resolve()
                        all_files.add(resolved)
                        if _is_sqlite_transient_sidecar(path, self.root):
                            sqlite_transient_sidecars.add(resolved)
            for walk_root in walk_roots:
                for current, directories, files in os.walk(
                    walk_root,
                    followlinks=False,
                ):
                    current_path = Path(current)
                    directories[:] = [
                        name
                        for name in directories
                        if not (current_path / name).is_symlink()
                        and not (
                            namespace == self.root
                            and current_path == self.root
                            and name == _LEASE_DIR_NAME
                        )
                    ]
                    for name in files:
                        path = current_path / name
                        if name == _NAMESPACE_MARKER_NAME or (
                            namespace == self.root
                            and current_path == self.root
                            and name
                            in {
                                _LOCK_NAME,
                                _ACCOUNTING_NAME,
                                _ROOT_MARKER_NAME,
                            }
                        ):
                            continue
                        if path.is_symlink():
                            raise OSError(f"refusing symlinked cache file: {path}")
                        try:
                            resolved = path.resolve(strict=True)
                        except FileNotFoundError as exc:
                            if _is_sqlite_transient_sidecar(path, namespace):
                                continue
                            raise OSError(
                                f"cannot inspect cache file {path}: {exc}"
                            ) from exc
                        except OSError as exc:
                            raise OSError(
                                f"cannot inspect cache file {path}: {exc}"
                            ) from exc
                        if not resolved.is_relative_to(namespace):
                            raise OSError(
                                f"cache file escaped managed namespace: {path}"
                            )
                        try:
                            resolved_stat = resolved.stat()
                        except FileNotFoundError as exc:
                            if _is_sqlite_transient_sidecar(path, namespace):
                                continue
                            raise OSError(
                                f"cannot inspect cache file {path}: {exc}"
                            ) from exc
                        except OSError as exc:
                            raise OSError(
                                f"cannot inspect cache file {path}: {exc}"
                            ) from exc
                        if not stat.S_ISREG(resolved_stat.st_mode):
                            raise OSError(
                                f"cache file escaped managed namespace: {path}"
                            )
                        all_files.add(resolved)
                        if _is_sqlite_transient_sidecar(path, namespace):
                            sqlite_transient_sidecars.add(resolved)

        for database in self._database_paths_locked():
            namespace = database.parent.resolve()
            conn = sqlite3.connect(str(database), timeout=1.0)
            try:
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(blocks)").fetchall()
                }
                has_ancestry = {
                    "parent_hash",
                    "ancestry_known",
                }.issubset(columns)
                if has_ancestry:
                    rows = conn.execute(
                        "SELECT block_hash, file_name, last_accessed, "
                        "parent_hash, ancestry_known FROM blocks"
                    ).fetchall()
                else:
                    # A legacy NULL cannot be assumed to mean chain root.
                    rows = [
                        (block_hash, file_name, last_accessed, None, 0)
                        for block_hash, file_name, last_accessed in conn.execute(
                            "SELECT block_hash, file_name, last_accessed "
                            "FROM blocks"
                        ).fetchall()
                    ]

                pin_table_exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'block_write_pins'"
                ).fetchone() is not None
                pin_rows = (
                    conn.execute(
                        "SELECT block_hash, owner_lease_id "
                        "FROM block_write_pins"
                    ).fetchall()
                    if pin_table_exists
                    else []
                )
            except sqlite3.Error as exc:
                raise OSError(f"cannot inspect block index {database}: {exc}") from exc
            finally:
                conn.close()

            indexed_rows: dict[
                str,
                tuple[Path, Path, int, int, str | None, bool],
            ] = {}
            for (
                block_hash,
                file_name,
                last_accessed,
                parent_hash,
                ancestry_known,
            ) in rows:
                relative = Path(str(file_name))
                if relative.is_absolute() or ".." in relative.parts:
                    raise OSError(f"unsafe block index path in {database}")
                resolved = self._safe_resolved_file(namespace / relative)
                if resolved is None:
                    continue
                referenced.add(resolved)
                try:
                    block_stat = resolved.stat()
                    size = block_stat.st_size
                except OSError as exc:
                    raise OSError(f"cannot stat indexed block {resolved}: {exc}") from exc
                indexed_access_ns = max(
                    0,
                    int(float(last_accessed or 0.0) * 1_000_000_000),
                )
                indexed_rows[str(block_hash)] = (
                    resolved,
                    relative,
                    max(0, int(size)),
                    max(indexed_access_ns, block_stat.st_mtime_ns),
                    str(parent_hash) if parent_hash is not None else None,
                    bool(ancestry_known),
                )

            # Pins are useful only while both their process lease and their
            # readable finalized row remain live.  Clean dead owners and
            # partial/crashed publications under the same root-exclusive scan.
            active_pinned_hashes = {
                str(block_hash)
                for block_hash, owner_lease_id in pin_rows
                if (
                    str(owner_lease_id) in active_lease_ids
                    and str(block_hash) in indexed_rows
                )
            }
            stale_pins = [
                (str(block_hash), str(owner_lease_id))
                for block_hash, owner_lease_id in pin_rows
                if (
                    str(owner_lease_id) not in active_lease_ids
                    or str(block_hash) not in indexed_rows
                )
            ]
            if stale_pins:
                cleanup_conn = sqlite3.connect(str(database), timeout=1.0)
                try:
                    cleanup_conn.executemany(
                        "DELETE FROM block_write_pins "
                        "WHERE block_hash = ? AND owner_lease_id = ?",
                        stale_pins,
                    )
                    cleanup_conn.commit()
                finally:
                    cleanup_conn.close()

            # Validate that every "known" row reaches an explicit known root in
            # this same namespace. Missing parents and cycles are fail-closed
            # legacy/invalid sets; they are never considered independent roots.
            # Walk iteratively: a 512K context at 64-token blocks is 8192 nodes,
            # far beyond Python's recursion limit.
            ancestry_valid: dict[str, bool] = {}

            def _valid_ancestry(block_hash: str) -> bool:
                cached = ancestry_valid.get(block_hash)
                if cached is not None:
                    return cached
                path: list[str] = []
                visiting: set[str] = set()
                current = block_hash
                valid = False
                while True:
                    cached = ancestry_valid.get(current)
                    if cached is not None:
                        valid = cached
                        break
                    if current in visiting:
                        valid = False
                        break
                    row = indexed_rows.get(current)
                    if row is None or not row[5]:
                        valid = False
                        break
                    visiting.add(current)
                    path.append(current)
                    parent_hash = row[4]
                    if parent_hash is None:
                        valid = True
                        break
                    current = parent_hash
                for visited_hash in path:
                    ancestry_valid[visited_hash] = valid
                return valid

            for block_hash, (
                resolved,
                relative,
                size_bytes,
                last_accessed_ns,
                parent_hash,
                _declared_known,
            ) in indexed_rows.items():
                valid_ancestry = _valid_ancestry(block_hash)
                candidates.append(
                    _BudgetCandidate(
                        kind="block",
                        size_bytes=size_bytes,
                        # A successful read touches the finalized payload under
                        # a shared root lock before returning.  This durable FS
                        # signal keeps global LRU correct even if the optional
                        # async SQLite access update queue is saturated.
                        last_accessed_ns=last_accessed_ns,
                        paths=(resolved,),
                        database=database,
                        block_hash=str(block_hash),
                        indexed_file_name=str(relative),
                        parent_hash=parent_hash,
                        ancestry_known=valid_ancestry,
                        publication_pinned=(
                            str(block_hash) in active_pinned_hashes
                        ),
                    )
                )

        finalized_files: list[Path] = []
        for path in all_files:
            if self._is_temp_path(path):
                continue
            if path.suffix not in {".safetensors", ".json"}:
                continue
            relative_parts = path.relative_to(self.root).parts
            if "blocks" not in relative_parts and "ssm_companion" not in relative_parts:
                continue
            finalized_files.append(path)

        finalized_set = set(finalized_files)
        for data_path in sorted(finalized_set):
            if data_path in referenced or data_path.suffix != ".safetensors":
                continue
            relative_parts = data_path.relative_to(self.root).parts
            if "ssm_companion" not in relative_parts:
                continue
            side_path = data_path.with_suffix(".json")
            if side_path not in finalized_set:
                continue
            try:
                data_stat = data_path.stat()
                side_stat = side_path.stat()
            except OSError as exc:
                raise OSError(f"cannot stat SSM companion pair: {exc}") from exc
            referenced.add(data_path)
            referenced.add(side_path)
            candidates.append(
                _BudgetCandidate(
                    kind="ssm",
                    size_bytes=max(0, data_stat.st_size) + max(0, side_stat.st_size),
                    last_accessed_ns=max(data_stat.st_mtime_ns, side_stat.st_mtime_ns),
                    paths=(data_path, side_path),
                )
            )

        for path in finalized_files:
            if path in referenced:
                continue
            try:
                file_stat = path.stat()
            except OSError as exc:
                raise OSError(f"cannot stat orphan cache payload {path}: {exc}") from exc
            age_ns = max(0, now_ns - file_stat.st_mtime_ns)
            evictable = age_ns >= self._orphan_grace_ns
            if not evictable:
                protected_recent_orphans += 1
            referenced.add(path)
            candidates.append(
                _BudgetCandidate(
                    kind="orphan" if evictable else "recent_orphan",
                    size_bytes=max(0, file_stat.st_size),
                    last_accessed_ns=file_stat.st_mtime_ns,
                    paths=(path,),
                )
            )

        # Count every other physical cache-root file—including SQLite index,
        # WAL/SHM files and active temp files—without ever selecting it for
        # deletion.  This keeps the configured max honest while preserving
        # in-flight writers and typed metadata.  A temporary overage is
        # reported non-compliant and settles on the next post-write trim.
        for path in all_files - referenced:
            try:
                file_stat = path.stat()
            except FileNotFoundError as exc:
                # Opening and closing the last SQLite connection above can
                # checkpoint and remove an already-enumerated WAL/SHM/journal.
                # Its absence is the final physical truth, not an ambiguous
                # cache mutation. Every other vanished path remains fail-closed.
                if path in sqlite_transient_sidecars:
                    continue
                raise OSError(f"cannot stat cache metadata {path}: {exc}") from exc
            except OSError as exc:
                raise OSError(f"cannot stat cache metadata {path}: {exc}") from exc
            is_temp = self._is_temp_path(path)
            temp_owner = self._temp_owner_lease(path) if is_temp else None
            age_ns = max(0, now_ns - file_stat.st_mtime_ns)
            temp_is_stale = bool(
                is_temp
                and (
                    (
                        temp_owner is not None
                        and temp_owner not in active_lease_ids
                    )
                    or (
                        temp_owner is None
                        and age_ns >= self._orphan_grace_ns
                    )
                )
            )
            candidates.append(
                _BudgetCandidate(
                    kind=(
                        "stale_temp"
                        if temp_is_stale
                        else "protected_temp"
                        if is_temp
                        else "protected_metadata"
                    ),
                    size_bytes=max(0, file_stat.st_size),
                    last_accessed_ns=file_stat.st_mtime_ns,
                    paths=(path,),
                )
            )

        total = sum(candidate.size_bytes for candidate in candidates)
        return candidates, total, protected_recent_orphans

    def _evict_block_locked(self, candidate: _BudgetCandidate) -> int:
        if (
            candidate.database is None
            or candidate.block_hash is None
            or not candidate.ancestry_known
        ):
            return 0
        conn = sqlite3.connect(str(candidate.database), timeout=1.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT file_name, last_accessed, parent_hash, ancestry_known "
                "FROM blocks WHERE block_hash = ?",
                (candidate.block_hash,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return 0
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'block_write_pins'"
            ).fetchone() is not None and conn.execute(
                "SELECT 1 FROM block_write_pins WHERE block_hash = ? LIMIT 1",
                (candidate.block_hash,),
            ).fetchone() is not None:
                conn.rollback()
                return 0
            file_name, last_accessed, parent_hash, ancestry_known = row
            current_access_ns = max(
                0,
                int(float(last_accessed or 0.0) * 1_000_000_000),
            )
            if str(file_name) != candidate.indexed_file_name:
                conn.rollback()
                return 0
            if (
                int(ancestry_known or 0) != 1
                or (
                    str(parent_hash) if parent_hash is not None else None
                )
                != candidate.parent_hash
            ):
                conn.rollback()
                return 0
            # A parent with any live child is never independently evictable.
            # The caller makes leaves eligible dynamically, and this transaction
            # re-check closes the race with a stale scan.
            if conn.execute(
                "SELECT 1 FROM blocks WHERE ancestry_known = 1 "
                "AND parent_hash = ? LIMIT 1",
                (candidate.block_hash,),
            ).fetchone() is not None:
                conn.rollback()
                return 0
            try:
                current_mtime_ns = candidate.paths[0].stat().st_mtime_ns
            except OSError:
                current_mtime_ns = 0
            if max(current_access_ns, current_mtime_ns) > candidate.last_accessed_ns:
                conn.rollback()
                return 0
            path = candidate.paths[0]
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'block_write_pins'"
            ).fetchone() is not None:
                conn.execute(
                    "DELETE FROM block_write_pins WHERE block_hash = ?",
                    (candidate.block_hash,),
                )
            conn.execute(
                "DELETE FROM blocks WHERE block_hash = ?",
                (candidate.block_hash,),
            )
            conn.commit()
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                size = candidate.size_bytes
            except OSError:
                # The committed index deletion makes the remaining payload a
                # non-retrievable orphan. It remains counted and is eligible
                # for a later safe-orphan trim; no stale retrievable row is left.
                return 0
            return max(0, int(size))
        except sqlite3.Error:
            with suppress(sqlite3.Error):
                conn.rollback()
            return 0
        finally:
            conn.close()

    def _evict_invalid_block_set_locked(
        self,
        database: Path,
        block_hashes: set[str],
    ) -> tuple[int, int]:
        """Invalidate one complete legacy/broken ancestry set atomically.

        Pre-migration NULL ancestry cannot distinguish roots from unknown
        parents.  When cap pressure requires reclaiming such rows, deleting a
        subset could recreate the head-loss defect, so the caller passes the
        complete invalid set for this namespace.
        """
        if not block_hashes:
            return 0, 0
        conn = sqlite3.connect(str(database), timeout=1.0)
        try:
            current_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(blocks)").fetchall()
            }
            rows: list[tuple[str, str]] = []
            hashes = sorted(block_hashes)
            for start in range(0, len(hashes), 500):
                chunk = hashes[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    (str(block_hash), str(file_name))
                    for block_hash, file_name in conn.execute(
                        f"SELECT block_hash, file_name FROM blocks "
                        f"WHERE block_hash IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
            if not rows:
                return 0, 0

            # If ancestry columns appeared/changed after the scan, abandon the
            # invalid-set deletion and let the next reconciled scan decide.
            has_ancestry = {
                "parent_hash",
                "ancestry_known",
            }.issubset(current_columns)
            if has_ancestry:
                current_invalid = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT block_hash FROM blocks "
                        "WHERE ancestry_known = 0"
                    ).fetchall()
                }
                # Broken declared-known descendants are included by the scan,
                # so only require every still-legacy row to be covered.
                if not current_invalid.issubset(block_hashes):
                    return 0, 0

            conn.execute("BEGIN IMMEDIATE")
            for start in range(0, len(hashes), 500):
                chunk = hashes[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'block_write_pins'"
                ).fetchone() is not None:
                    conn.execute(
                        f"DELETE FROM block_write_pins "
                        f"WHERE block_hash IN ({placeholders})",
                        chunk,
                    )
                conn.execute(
                    f"DELETE FROM blocks WHERE block_hash IN ({placeholders})",
                    chunk,
                )
            conn.commit()
        except sqlite3.Error:
            with suppress(sqlite3.Error):
                conn.rollback()
            return 0, 0
        finally:
            conn.close()

        namespace = database.parent.resolve()
        evicted_entries = 0
        freed = 0
        for _block_hash, file_name in rows:
            relative = Path(file_name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            path = self._safe_resolved_file(namespace / relative)
            if path is None:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                freed += max(0, int(size))
                evicted_entries += 1
            except FileNotFoundError:
                evicted_entries += 1
            except OSError:
                # The committed row deletion leaves a counted orphan for the
                # next reconciled trim rather than a retrievable broken chain.
                continue
        return evicted_entries, freed

    @staticmethod
    def _evict_paths_locked(candidate: _BudgetCandidate) -> int:
        freed = 0
        for path in candidate.paths:
            try:
                size = path.stat().st_size
                path.unlink()
                freed += max(0, int(size))
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return freed

    def refresh_health(self) -> GlobalBudgetResult:
        """Refresh O(1) aggregate telemetry from the shared root ledger.

        Each process owns a coordinator instance, but all writers publish into
        one accounting file.  Reading only ``last_result`` can therefore stay
        stale forever in an idle process while another process advances or
        repairs the root.  This refresh reads the shared cap leases and ledger
        under the short root lock; it never claims a physical scan occurred.
        """

        previous = self._last_result
        try:
            with self._exclusive_guard():
                maximum = self._effective_max_size_bytes_locked()
                state = self._read_accounting_locked()
                if state is None:
                    raise OSError("global cache accounting has not been reconciled")
                estimate = max(0, int(state["bytes_estimate"]))
                reconciliation_generation = max(
                    0, int(state["reconciliation_generation"])
                )
                same_local_reconciliation = bool(
                    previous is not None
                    and previous.reconciliation_generation
                    == reconciliation_generation
                )
                result = GlobalBudgetResult(
                    max_size_bytes=maximum,
                    bytes_before=estimate,
                    bytes_after=estimate,
                    evicted_entries=(
                        previous.evicted_entries
                        if same_local_reconciliation and previous is not None
                        else 0
                    ),
                    evicted_bytes=(
                        previous.evicted_bytes
                        if same_local_reconciliation and previous is not None
                        else 0
                    ),
                    protected_recent_orphans=(
                        previous.protected_recent_orphans
                        if previous is not None
                        else 0
                    ),
                    compliant=maximum <= 0 or estimate <= maximum,
                    scan_performed=bool(
                        same_local_reconciliation
                        and previous is not None
                        and previous.scan_performed
                    ),
                    reconciled_at_ns=max(0, int(state["reconciled_at_ns"])),
                    accounted=True,
                    accounting_generation=max(
                        0, int(state["accounting_generation"])
                    ),
                    reconciliation_generation=reconciliation_generation,
                    evicted_entries_total=int(state.get("evicted_entries_total") or 0),
                    evicted_bytes_total=int(state.get("evicted_bytes_total") or 0),
                    protected_temp_files=(
                        previous.protected_temp_files if previous is not None else 0
                    ),
                )
                self._last_result = result
                return result
        except (OSError, sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            result = GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=False,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )
            self._last_result = result
            return result

    def enforce(self, *, force: bool = False) -> GlobalBudgetResult:
        """Trim aggregate finalized payload bytes to the configured ceiling.

        ``0`` means unlimited.  Any uncertainty fails closed: no candidate is
        removed unless the root lock, every block index scan, and path safety
        checks succeed.
        """

        monotonic_now = time.monotonic_ns()
        with self._thread_lock:
            if (
                not force
                and self._last_result is not None
                and monotonic_now - self._last_reconcile_monotonic_ns
                < self._reconcile_interval_ns
            ):
                previous = self._last_result
                return GlobalBudgetResult(
                    max_size_bytes=previous.max_size_bytes,
                    bytes_before=previous.bytes_after,
                    bytes_after=previous.bytes_after,
                    evicted_entries=0,
                    evicted_bytes=0,
                    protected_recent_orphans=previous.protected_recent_orphans,
                    compliant=previous.compliant,
                    scan_performed=False,
                    reconciled_at_ns=previous.reconciled_at_ns,
                    accounted=previous.accounted,
                    accounting_generation=previous.accounting_generation,
                    reconciliation_generation=previous.reconciliation_generation,
                    error=previous.error,
                )

        try:
            lock_started = time.perf_counter()
            with self._exclusive_guard():
                lock_wait_s = time.perf_counter() - lock_started
                return self._enforce_locked(lock_wait_s=lock_wait_s)
        except (OSError, sqlite3.Error) as exc:
            result = GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=True,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )
            self._last_result = result
            logger.warning(
                "Global block-cache budget enforcement skipped without deletion: %s",
                exc,
            )
            return result

    def _enforce_locked(
        self,
        *,
        protected_blocks: set[tuple[Path, str]] | None = None,
        lock_wait_s: float = 0.0,
    ) -> GlobalBudgetResult:
        """Reconcile and trim while the root-exclusive lock is held.

        Phase timings (lock wait, scan, eviction, rescan, accounting write) are
        logged for every performed reconcile so a slow one is attributable.
        """

        phase_started = time.perf_counter()
        protected_keys = {
            (Path(database).resolve(), str(block_hash))
            for database, block_hash in (protected_blocks or set())
        }

        max_size_bytes = self._effective_max_size_bytes_locked()
        stored_accounting = self._read_or_reset_accounting_locked()
        accounting = stored_accounting or {
            "bytes_estimate": 0,
            "accounting_generation": 0,
            "reconciliation_generation": 0,
            "reconciled_at_ns": 0,
            "evicted_entries_total": 0,
            "evicted_bytes_total": 0,
        }
        now_ns = time.time_ns()
        candidates, total, _protected = self._scan_locked(now_ns=now_ns)
        scan_s = time.perf_counter() - phase_started
        evict_started = time.perf_counter()
        before = total
        evicted_entries = 0
        evicted_bytes = 0
        protected_temp_files = sum(
            1 for candidate in candidates if candidate.kind == "protected_temp"
        )
        # Garbage first, whatever the ceiling (including unlimited): a finalized
        # payload no index references once the orphan grace has passed, and a
        # temp file whose writer lease is dead (or ownerless and aged), can never
        # be restored by anyone. Before this they lingered until the root went
        # OVER its ceiling and the LRU pass happened to reach them (live: 4 temp
        # files from a kill -9 survived a restart and 5 idle minutes).
        garbage = [
            candidate
            for candidate in candidates
            if candidate.kind in {"orphan", "stale_temp"}
            and not candidate.publication_pinned
        ]
        garbage_entries = 0
        garbage_bytes = 0
        for candidate in garbage:
            freed = self._evict_paths_locked(candidate)
            if freed <= 0:
                continue
            total = max(0, total - freed)
            garbage_entries += 1
            garbage_bytes += freed
        if garbage_entries:
            evicted_entries += garbage_entries
            evicted_bytes += garbage_bytes
            candidates = [candidate for candidate in candidates if candidate not in garbage]
            logger.info(
                "Global block-cache garbage: removed %d unreferenced payload(s) "
                "(%.3fGB: aged orphans and dead-writer temp files)",
                garbage_entries,
                garbage_bytes / 1024**3,
            )
        if max_size_bytes > 0:
            trim_target = (
                max(0, int(max_size_bytes * 0.9))
                if total > max_size_bytes
                else max_size_bytes
            )

            # Legacy rows and declared-known rows whose ancestry does not reach
            # a readable root are an indivisible invalid set per namespace.
            # They enter the same LRU heap below as one conservative candidate
            # timestamped by the newest member; partial deletion would preserve
            # arbitrary suffixes.
            invalid_by_database: dict[Path, list[_BudgetCandidate]] = {}
            for candidate in candidates:
                if (
                    candidate.kind == "block"
                    and candidate.database is not None
                    and not candidate.ancestry_known
                ):
                    invalid_by_database.setdefault(
                        candidate.database,
                        [],
                    ).append(candidate)
            # Known blocks are a forest. Only leaves enter the LRU heap; after a
            # leaf is removed its parent becomes eligible. This gives true
            # suffix-first trimming while preserving shared prefixes and
            # branches. Non-block cache records retain ordinary global LRU.
            known_blocks: dict[
                tuple[Path, str],
                _BudgetCandidate,
            ] = {}
            child_counts: dict[tuple[Path, str], int] = {}
            for candidate in candidates:
                if (
                    candidate.kind != "block"
                    or candidate.database is None
                    or candidate.block_hash is None
                    or not candidate.ancestry_known
                ):
                    continue
                key = (candidate.database, candidate.block_hash)
                known_blocks[key] = candidate
                child_counts.setdefault(key, 0)
            for key, candidate in known_blocks.items():
                if candidate.parent_hash is None:
                    continue
                parent_key = (key[0], candidate.parent_hash)
                if parent_key in known_blocks:
                    child_counts[parent_key] = child_counts.get(parent_key, 0) + 1

            eligible: list[
                tuple[int, str, str, int, _BudgetCandidate]
            ] = []
            sequence = 0

            def _push(candidate: _BudgetCandidate) -> None:
                nonlocal sequence
                if candidate.publication_pinned:
                    return
                if (
                    candidate.kind == "block"
                    and candidate.database is not None
                    and candidate.block_hash is not None
                    and (
                        candidate.database.resolve(),
                        candidate.block_hash,
                    )
                    in protected_keys
                ):
                    return
                heapq.heappush(
                    eligible,
                    (
                        candidate.last_accessed_ns,
                        candidate.kind,
                        str(candidate.paths[0]),
                        sequence,
                        candidate,
                    ),
                )
                sequence += 1

            for key, candidate in known_blocks.items():
                if child_counts.get(key, 0) == 0:
                    _push(candidate)
            for candidate in candidates:
                if candidate.kind in {
                    "block",
                    "recent_orphan",
                    "protected_temp",
                    "protected_metadata",
                }:
                    continue
                _push(candidate)
            invalid_hashes: dict[Path, set[str]] = {}
            for database, invalid_candidates in invalid_by_database.items():
                if any(
                    candidate.publication_pinned
                    or (
                        candidate.database is not None
                        and candidate.block_hash is not None
                        and (
                            candidate.database.resolve(),
                            candidate.block_hash,
                        )
                        in protected_keys
                    )
                    for candidate in invalid_candidates
                ):
                    continue
                hashes = {
                    str(candidate.block_hash)
                    for candidate in invalid_candidates
                    if candidate.block_hash is not None
                }
                if not hashes:
                    continue
                invalid_hashes[database] = hashes
                _push(
                    _BudgetCandidate(
                        kind="invalid_block_set",
                        size_bytes=sum(
                            candidate.size_bytes
                            for candidate in invalid_candidates
                        ),
                        last_accessed_ns=max(
                            candidate.last_accessed_ns
                            for candidate in invalid_candidates
                        ),
                        paths=(database,),
                        database=database,
                    )
                )

            remaining_blocks = set(known_blocks)
            while total > trim_target and eligible:
                _, _, _, _, candidate = heapq.heappop(eligible)
                if candidate.kind == "block":
                    assert candidate.database is not None
                    assert candidate.block_hash is not None
                    key = (candidate.database, candidate.block_hash)
                    if (
                        key not in remaining_blocks
                        or child_counts.get(key, 0) != 0
                    ):
                        continue
                    freed = self._evict_block_locked(candidate)
                    if freed <= 0:
                        continue
                    remaining_blocks.remove(key)
                    if candidate.parent_hash is not None:
                        parent_key = (
                            candidate.database,
                            candidate.parent_hash,
                        )
                        if parent_key in remaining_blocks:
                            child_counts[parent_key] = max(
                                0,
                                child_counts.get(parent_key, 0) - 1,
                            )
                            if child_counts[parent_key] == 0:
                                _push(known_blocks[parent_key])
                elif candidate.kind == "invalid_block_set":
                    assert candidate.database is not None
                    removed_count, freed = self._evict_invalid_block_set_locked(
                        candidate.database,
                        invalid_hashes.get(candidate.database, set()),
                    )
                    if removed_count <= 0:
                        continue
                    total = max(0, total - freed)
                    evicted_entries += removed_count
                    evicted_bytes += freed
                    continue
                else:
                    freed = self._evict_paths_locked(candidate)
                    if freed <= 0:
                        continue
                total = max(0, total - freed)
                evicted_entries += 1
                evicted_bytes += freed

        evict_s = time.perf_counter() - evict_started
        rescan_started = time.perf_counter()
        if evicted_entries:
            # Physical state changed under the exclusive lock: re-measure.
            _, after, protected_after = self._scan_locked(now_ns=time.time_ns())
        else:
            # Nothing was removed and the root-exclusive lock is still held,
            # so the first scan's totals are the current physical state; a
            # second full walk of every namespace would only repeat it.
            after, protected_after = total, _protected
        rescan_s = time.perf_counter() - rescan_started
        account_started = time.perf_counter()
        reconciled_at_ns = time.time_ns()
        reconciliation_generation = accounting["reconciliation_generation"] + 1
        evicted_entries_total = (
            int(accounting.get("evicted_entries_total") or 0) + evicted_entries
        )
        evicted_bytes_total = (
            int(accounting.get("evicted_bytes_total") or 0) + evicted_bytes
        )
        self._write_accounting_locked(
            {
                "bytes_estimate": after,
                "accounting_generation": accounting["accounting_generation"],
                "reconciliation_generation": reconciliation_generation,
                "reconciled_at_ns": reconciled_at_ns,
                "evicted_entries_total": evicted_entries_total,
                "evicted_bytes_total": evicted_bytes_total,
            }
        )
        result = GlobalBudgetResult(
            max_size_bytes=max_size_bytes,
            bytes_before=before,
            bytes_after=after,
            evicted_entries=evicted_entries,
            evicted_bytes=evicted_bytes,
            protected_recent_orphans=protected_after,
            compliant=max_size_bytes <= 0 or after <= max_size_bytes,
            scan_performed=True,
            reconciled_at_ns=reconciled_at_ns,
            accounted=True,
            accounting_generation=accounting["accounting_generation"],
            reconciliation_generation=reconciliation_generation,
            evicted_entries_total=evicted_entries_total,
            evicted_bytes_total=evicted_bytes_total,
            protected_temp_files=protected_temp_files,
        )
        self._last_result = result
        self._last_reconcile_monotonic_ns = time.monotonic_ns()
        account_s = time.perf_counter() - account_started
        logger.info(
            "Global block-cache budget reconcile: total=%.3fs lock_wait=%.3fs "
            "scan=%.3fs evict=%.3fs rescan=%.3fs account=%.3fs evicted=%d "
            "usage=%.3fGB / %.3fGB",
            time.perf_counter() - phase_started + lock_wait_s,
            lock_wait_s,
            scan_s,
            evict_s,
            rescan_s,
            account_s,
            evicted_entries,
            after / 1e9,
            max_size_bytes / 1e9,
        )
        if evicted_entries:
            logger.info(
                "Global block-cache eviction: removed %d entries (%.3fGB); "
                "aggregate physical usage now %.3fGB / %.3fGB",
                evicted_entries,
                evicted_bytes / 1024**3,
                after / 1024**3,
                max_size_bytes / 1024**3,
            )
        if not result.compliant:
            logger.warning(
                "Global block-cache remains above its ceiling because only "
                "recent/in-flight or non-removable payloads remain: "
                "%.3fGB / %.3fGB",
                after / 1024**3,
                max_size_bytes / 1024**3,
            )
        return result


def get_global_disk_cache_budget(
    root: str | os.PathLike[str],
    max_size_bytes: int,
    *,
    orphan_grace_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS,
    reconcile_interval_seconds: float = _DEFAULT_RECONCILE_INTERVAL_SECONDS,
    allow_legacy_hashed_namespaces: bool = False,
    allow_legacy_direct_namespace: bool = False,
) -> GlobalDiskCacheBudget:
    """Create one independently leased coordinator for a cache-store owner.

    Multiple stores can coexist in one process and can request different caps.
    Reusing a singleton would overwrite a lease and let the newest store relax
    an older store's finite cap, so every owner deliberately publishes its own
    lease while sharing only the in-process root lock.
    """

    return GlobalDiskCacheBudget(
        Path(root).expanduser().resolve(),
        max_size_bytes,
        orphan_grace_seconds=orphan_grace_seconds,
        reconcile_interval_seconds=reconcile_interval_seconds,
        allow_legacy_hashed_namespaces=allow_legacy_hashed_namespaces,
        allow_legacy_direct_namespace=allow_legacy_direct_namespace,
    )

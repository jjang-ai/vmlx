"""Native Omni snapshots in the configured, aggregate SSD cache pool.

Serialization belongs to the model owner. This store manages only complete
native snapshots; it never rewinds recurrent state or changes tensor precision.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from ..global_disk_cache_budget import (
    ensure_managed_block_cache_namespace,
    get_global_disk_cache_budget,
)

SCHEMA = "omni_native_snapshot_v2"


class OmniSessionDiskStore:
    def __init__(self, *, root, model_key, max_size_bytes, ttl_minutes=0):
        self.root = Path(root).expanduser().resolve()
        namespace = hashlib.sha256(f"{model_key}:{SCHEMA}".encode()).hexdigest()[:16]
        directory = ensure_managed_block_cache_namespace(self.root / namespace)
        self.directory = directory / "native_sessions"
        self.directory.mkdir(exist_ok=True)
        if self.directory.is_symlink():
            raise OSError("native session cache must not be a symlink")
        self.ttl_seconds = max(0.0, float(ttl_minutes)) * 60
        self.max_size_bytes = int(max_size_bytes)
        self.budget = get_global_disk_cache_budget(self.root, self.max_size_bytes)
        self.last_path = None

    def paths(self, signature):
        name = hashlib.sha256(str(signature).encode()).hexdigest()
        return self.directory / (name + ".safetensors"), self.directory / (name + ".json")

    @staticmethod
    def _fsync_directory(directory):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _expired(self, path):
        return self.ttl_seconds > 0 and time.time() - path.stat().st_mtime > self.ttl_seconds

    def _prune_expired_locked(self):
        if not self.ttl_seconds:
            return
        removed = 0
        for side in self.directory.glob("*.json"):
            if ".tmp." in side.name or side.is_symlink() or not self._expired(side):
                continue
            for path in (side.with_suffix(".safetensors"), side):
                if path.is_file() and not path.is_symlink():
                    removed += path.stat().st_size
                    path.unlink()
        if removed:
            self.budget.account_finalized_write_locked(-removed)

    def save(self, signature, writer):
        """Return only after native data, sidecar and directory are durable."""
        data, side = self.paths(signature)
        nonce = f"{self.budget.lease_id}.{uuid.uuid4().hex}"
        temp = data.with_name(f"{data.stem}.{nonce}.tmp.safetensors")
        temp_side = side.with_name(f"{side.stem}.{nonce}.tmp.json")
        with self.budget.exclusive_mutation_guard() as locked:
            if not locked:
                raise OSError("native SSD publication lock unavailable")
            for path in (data, side):
                if path.is_symlink():
                    raise OSError("native SSD record must not be a symlink")
            self._prune_expired_locked()
            before = sum(p.stat().st_size for p in (data, side) if p.is_file())
            published = False
            try:
                writer(temp)
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                with temp_side.open("x") as handle:
                    json.dump({"schema": SCHEMA, "signature": signature}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                size = temp.stat().st_size + temp_side.stat().st_size
                if self.max_size_bytes > 0 and size > self.max_size_bytes:
                    raise OSError(f"native snapshot {size} bytes exceeds configured SSD cap {self.max_size_bytes}")
                os.replace(temp, data)
                published = True
                os.replace(temp_side, side)
                self._fsync_directory(self.directory)
                result = self.budget.account_finalized_write_locked(size - before)
                if not result.accounted or not result.compliant or not data.is_file() or not side.is_file():
                    raise OSError(result.error or "native snapshot could not fit the aggregate SSD budget")
                self.last_path = data
                return data
            except Exception:
                if published:
                    data.unlink(missing_ok=True)
                    side.unlink(missing_ok=True)
                    self.budget.account_finalized_write_locked(-1)
                raise
            finally:
                temp.unlink(missing_ok=True)
                temp_side.unlink(missing_ok=True)

    def load(self, signature, reader):
        """Read an exact causal snapshot under the pool eviction lock."""
        data, side = self.paths(signature)
        with self.budget.mutation_guard() as locked:
            if not locked or not data.is_file() or not side.is_file():
                return None
            if data.is_symlink() or side.is_symlink() or self._expired(side):
                return None
            metadata = json.loads(side.read_text())
            if metadata.get("schema") != SCHEMA or metadata.get("signature") != signature:
                return None
            value = reader(data)
            os.utime(data, None)
            os.utime(side, None)
            self.last_path = data
            return value

    def clear(self):
        """Remove this model's finalized snapshots, preserving other namespaces."""
        with self.budget.exclusive_mutation_guard() as locked:
            if not locked:
                raise OSError("native SSD clear lock unavailable")
            removed = 0
            for path in self.directory.iterdir():
                if (path.suffix not in {".json", ".safetensors"}
                        or len(path.stem) != 64
                        or any(c not in "0123456789abcdef" for c in path.stem)):
                    continue
                if path.is_symlink():
                    raise OSError("native SSD clear refused a symlink")
                removed += path.stat().st_size
                path.unlink()
            self._fsync_directory(self.directory)
            result = self.budget.account_finalized_write_locked(-removed)
            if not result.accounted:
                raise OSError(result.error or "native SSD clear accounting failed")
            self.last_path = None
            return removed

    def close(self):
        self.budget.close()

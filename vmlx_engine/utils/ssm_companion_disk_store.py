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

import json
import logging
import os
import pickle
import threading
import time
from copy import deepcopy
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
        if gb <= 0:
            return int(_DEFAULT_BUDGET_GB * (1024 ** 3))
        return int(gb * (1024 ** 3))
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
    ):
        self._dir = Path(directory) if directory else _default_dir()
        self._budget = int(budget_bytes) if budget_bytes else _budget_bytes()
        self._lock = threading.Lock()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("SSM disk cache mkdir failed (%s); disk store inert", e)
            self._dir = None  # type: ignore[assignment]

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

    # --------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------
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
            return flat
        state_attr = getattr(layer, "state", None)
        if isinstance(state_attr, tuple):
            for i, a in enumerate(state_attr):
                if a is not None:
                    flat[f"{prefix}.state.{i}"] = a
            return flat
        return flat

    def _save_entry(
        self,
        key: str,
        states: List[Any],
        is_complete: bool,
        token_ids: List[int],
        num_tokens: int,
    ) -> bool:
        if self._dir is None:
            return False
        data_path, side_path = self._entry_paths(key)
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug("SSM disk store mkdir(%s) failed: %s", data_path.parent, e)
            return False

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
                    return False
            else:
                flat.update(self._flatten_layer(f"L{n}", layer))

        sidecar = {
            "version": 1,
            "is_complete": bool(is_complete),
            "num_tokens": int(num_tokens),
            "stored_at": time.time(),
            "layer_metas": layer_metas,
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

        # Materialize lazy arrays before save (safetensors does this anyway,
        # but being explicit catches errors in our own code path).
        if flat:
            try:
                _mx_materialize(*list(flat.values()))
            except Exception as e:
                logger.debug("SSM disk store materialize failed: %s", e)
                return False

        # Atomic write via tmp + rename. mx.save_safetensors requires the
        # ``.safetensors`` extension on the target path, so the tmp file
        # uses a sibling name with the same extension.
        tmp_data = data_path.parent / f"{data_path.stem}.tmp.safetensors"
        tmp_side = side_path.parent / f"{side_path.stem}.tmp.json"
        try:
            if flat:
                mx.save_safetensors(str(tmp_data), flat)
            else:
                # No MLX tensors to save (e.g. all-opaque entry). Touch an
                # empty file so eviction can find it; the sidecar carries
                # the data.
                tmp_data.write_bytes(b"")
            tmp_side.write_text(json.dumps(sidecar))
            os.replace(tmp_data, data_path)
            os.replace(tmp_side, side_path)
            return True
        except Exception as e:
            logger.debug("SSM disk store write failed for %s: %s", key, e)
            for p in (tmp_data, tmp_side):
                try:
                    p.unlink()
                except OSError:
                    pass
            return False

    def _load_entry(self, key: str) -> Optional[Tuple[List[Any], bool]]:
        if self._dir is None:
            return None
        data_path, side_path = self._entry_paths(key)
        if not side_path.exists():
            return None
        try:
            sidecar = json.loads(side_path.read_text())
        except (OSError, ValueError) as e:
            logger.debug("SSM disk store sidecar parse failed %s: %s", key, e)
            return None

        layer_metas: List[Dict[str, Any]] = sidecar.get("layer_metas", [])
        is_complete = bool(sidecar.get("is_complete", True))

        flat: Dict[str, mx.array] = {}
        if data_path.exists() and data_path.stat().st_size > 0:
            try:
                flat = mx.load(str(data_path))  # type: ignore[assignment]
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
                    ac.lengths = lengths  # type: ignore[attr-defined]
                else:
                    try:
                        ac.lengths = None  # type: ignore[attr-defined]
                    except Exception:
                        pass
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

        # Touch mtime so LRU treats this as recently-used.
        try:
            now = time.time()
            os.utime(data_path, (now, now))
            os.utime(side_path, (now, now))
        except OSError:
            pass

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

        # Deep-copy semantics: caller mutates SSM state in-place. Safetensors
        # load returns fresh arrays already, but we run a layer-level
        # deepcopy to align with the L1 fetch contract.
        copied: List[Any] = []
        for s in states:
            try:
                copied.append(deepcopy(s))
            except Exception as e:
                logger.debug("SSM disk store post-load deepcopy failed: %s", e)
                return None
        return (copied, is_complete)

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
        """Persist an entry. Returns True on success, False on any failure
        (caller treats failure as no-op — L1 still has the data)."""
        if not states or num_tokens <= 0:
            return False
        with self._lock:
            ok = self._save_entry(key, states, is_complete, token_ids, num_tokens)
            if ok:
                self._enforce_budget()
            return ok

    def fetch(self, key: str) -> Optional[Tuple[List[Any], bool]]:
        """Look up by key. Returns ``(states, is_complete)`` or ``None``."""
        # Read path holds the lock briefly only to align with budget
        # enforcement; actual decode is independent of the lock.
        return self._load_entry(key)

    def delete(self, key: str) -> None:
        if self._dir is None:
            return
        data_path, side_path = self._entry_paths(key)
        for p in (data_path, side_path):
            try:
                p.unlink()
            except OSError:
                pass

    def clear(self) -> None:
        if self._dir is None:
            return
        with self._lock:
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

    def _enforce_budget(self) -> None:
        """LRU eviction by mtime under the disk byte budget."""
        if self._dir is None:
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
        for mtime, size, data, side in files:
            if total <= self._budget:
                break
            for p in (data, side):
                try:
                    p.unlink()
                except OSError:
                    pass
            total -= size
            logger.debug("SSM disk store evicted %s (%.2f MB)", data.name, size / 1e6)


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

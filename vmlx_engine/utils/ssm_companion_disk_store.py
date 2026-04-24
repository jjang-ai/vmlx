# SPDX-License-Identifier: Apache-2.0
"""Optional L2 disk store for the SSM companion cache — see vmlx#110.

The in-memory ``SSMCompanionCache`` is lost on every engine restart. Paged
KV blocks persist via ``block_disk_store.py`` already, so a cold-start
workload with a stable system prompt still pays full prefill on the SSM
side even when the KV side hits from disk. This module adds an optional,
strictly-additive L2 that mirrors the L1 semantics: per-prompt entries
keyed by the same SHA-256 the L1 uses, serialized via ``mx.save_safetensors``
+ a JSON sidecar, LRU-evicted by mtime, bounded by a byte cap.

Behavior summary
----------------
- Writes are batched to a single background thread so the scheduler hot
  path does not block on fsync. Main-thread cost is the ``mx.eval`` +
  dict flatten only.
- Reads happen inline (``mx.load`` is mmap-backed and cheap).
- Atomic rename on write; corrupt/incomplete reads are skipped with a
  DEBUG log and treated as a miss.
- Disabled unless explicitly constructed by the caller — see the wiring
  in ``scheduler.py`` / ``mllm_batch_generator.py`` which reads the
  ``VMLX_SSM_COMPANION_DISK_DIR`` env var.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

_DISK_STORE_VERSION = 1


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def _serialize_ssm_layers(
    ssm_layers: List[Any],
    num_tokens: int,
    is_complete: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Flatten a list of SSM layer cache objects into (arrays_dict, meta_dict).

    Each layer contributes per-slot mx.arrays under keys ``L{li}.S{si}``,
    plus metadata (class name, ``meta_state``, None-mask) in the sidecar.
    """
    arrays: Dict[str, Any] = {}
    layers_meta: List[Dict[str, Any]] = []
    materialise_targets: List[Any] = []
    for li, layer in enumerate(ssm_layers):
        cls_name = type(layer).__name__
        try:
            state = layer.state
        except Exception:
            state = getattr(layer, "cache", None)
        try:
            meta_state = layer.meta_state
        except Exception:
            meta_state = ""
        if state is None:
            layers_meta.append(
                {
                    "class_name": cls_name,
                    "meta_state": _json_safe(meta_state),
                    "state_len": 0,
                    "state_nones": [],
                }
            )
            continue
        try:
            state_list = list(state)
        except TypeError:
            state_list = [state]
        nones: List[int] = []
        for si, arr in enumerate(state_list):
            if arr is None:
                nones.append(si)
                continue
            arrays[f"L{li:04d}.S{si:04d}"] = arr
            materialise_targets.append(arr)
        layers_meta.append(
            {
                "class_name": cls_name,
                "meta_state": _json_safe(meta_state),
                "state_len": len(state_list),
                "state_nones": nones,
            }
        )
    if materialise_targets:
        mx.eval(*materialise_targets)
    meta = {
        "version": _DISK_STORE_VERSION,
        "num_tokens": int(num_tokens),
        "is_complete": bool(is_complete),
        "layers": layers_meta,
    }
    return arrays, meta


def _deserialize_ssm_layers(
    arrays: Dict[str, Any],
    meta: Dict[str, Any],
) -> Optional[List[Any]]:
    if int(meta.get("version", 0)) != _DISK_STORE_VERSION:
        return None
    try:
        import mlx_lm.models.cache as _cm
    except ImportError:
        return None
    layers: List[Any] = []
    for li, lm in enumerate(meta.get("layers", [])):
        cls = getattr(_cm, lm.get("class_name", ""), None)
        if cls is None or not hasattr(cls, "from_state"):
            return None
        n = int(lm.get("state_len", 0))
        nones = set(lm.get("state_nones", []))
        state_list: List[Any] = []
        for si in range(n):
            if si in nones:
                state_list.append(None)
            else:
                k = f"L{li:04d}.S{si:04d}"
                if k not in arrays:
                    return None
                state_list.append(arrays[k])
        meta_state = lm.get("meta_state", "")
        try:
            layers.append(cls.from_state(state_list, meta_state))
        except Exception:
            try:
                layers.append(cls.from_state(tuple(state_list), meta_state))
            except Exception:
                return None
    return layers


class SSMCompanionDiskStore:
    """File-backed L2 tier for ``SSMCompanionCache`` (vmlx#110).

    Layout under ``cache_dir``:
        ``<sha256-key>.safetensors``   — per-layer state arrays
        ``<sha256-key>.meta.json``     — class names, meta_state, None mask

    Writes go through a single-thread background executor so the scheduler
    hot path does not block on fsync. Reads happen inline. Eviction is
    LRU-by-mtime under a byte cap.

    Thread safety
    -------------
    ``store()`` and ``fetch()`` are safe to call from the scheduler thread.
    Internal ``_write_async`` runs on a dedicated executor. Disk-level
    operations (eviction, orphan cleanup) are serialised on ``self._lock``.

    Failure model
    -------------
    All disk I/O errors are caught and logged at DEBUG or WARNING; the
    L1 (in-memory) path is never impacted. A torn write (process crash
    between the two ``os.replace`` calls) is detected on read as a
    sidecar/binary mismatch and treated as a miss.
    """

    def __init__(self, cache_dir: str, max_size_gb: float = 10.0) -> None:
        self._dir = pathlib.Path(os.path.expanduser(cache_dir))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_size_gb * 1_000_000_000) if max_size_gb > 0 else 0
        self._lock = threading.Lock()
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ssm-disk-writer"
        )
        orphans = 0
        for pattern in (
            "*.tmp.safetensors",
            "*.meta.tmp.json",
            "*.safetensors.tmp.safetensors",
            "*.meta.json.tmp",
        ):
            try:
                for p in list(self._dir.glob(pattern)):
                    try:
                        p.unlink()
                        orphans += 1
                    except OSError:
                        pass
            except Exception:
                pass
        try:
            entries = sum(
                1
                for p in self._dir.glob("*.safetensors")
                if ".tmp." not in p.name
            )
        except Exception:
            entries = 0
        if orphans:
            logger.info(
                "SSMCompanionDiskStore: cleaned %d orphaned tmp files", orphans
            )
        logger.info(
            "SSMCompanionDiskStore: dir=%s, max_size=%s, entries=%d",
            self._dir,
            "unlimited" if not self._max_bytes else f"{max_size_gb:.1f}GB",
            entries,
        )

    def _paths(self, key: str) -> Tuple[pathlib.Path, pathlib.Path]:
        return (
            self._dir / f"{key}.safetensors",
            self._dir / f"{key}.meta.json",
        )

    def store(
        self,
        key: str,
        ssm_layers: List[Any],
        is_complete: bool,
        num_tokens: int,
    ) -> None:
        """Write-through from ``SSMCompanionCache.store``. Non-blocking."""
        try:
            arrays, meta = _serialize_ssm_layers(
                ssm_layers, num_tokens, is_complete
            )
        except Exception as e:
            logger.debug("SSMCompanionDiskStore.store serialize failed: %s", e)
            return
        if not arrays:
            return
        self._writer.submit(self._write_async, key, arrays, meta)

    def _write_async(
        self,
        key: str,
        arrays: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        bin_path, meta_path = self._paths(key)
        bin_tmp = bin_path.with_name(bin_path.stem + ".tmp" + bin_path.suffix)
        meta_tmp = meta_path.with_name(meta_path.stem + ".tmp" + meta_path.suffix)
        try:
            mx.save_safetensors(str(bin_tmp), arrays)
            meta_tmp.write_text(json.dumps(meta, separators=(",", ":")))
            os.replace(bin_tmp, bin_path)
            os.replace(meta_tmp, meta_path)
            with self._lock:
                self._evict_if_over_cap()
        except Exception as e:
            logger.warning("SSMCompanionDiskStore write failed: %s", e)
            for p in (bin_tmp, meta_tmp):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def fetch(self, key: str) -> Optional[Tuple[List[Any], bool]]:
        """Read-through fallback from ``SSMCompanionCache.fetch``.

        Returns (ssm_layers, is_complete) on hit, None on miss. The caller
        is responsible for the deep-copy + materialisation discipline
        (already handled by ``SSMCompanionCache.fetch`` which promotes the
        returned value through the L1 LRU path).
        """
        bin_path, meta_path = self._paths(key)
        if not (bin_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            arrays = mx.load(str(bin_path))
            layers = _deserialize_ssm_layers(arrays, meta)
            if layers is None:
                return None
            now = time.time()
            try:
                os.utime(bin_path, (now, now))
                os.utime(meta_path, (now, now))
            except OSError:
                pass
            return (layers, bool(meta.get("is_complete", True)))
        except Exception as e:
            logger.debug("SSMCompanionDiskStore.fetch failed: %s", e)
            return None

    def clear_disk(self) -> None:
        with self._lock:
            for p in list(self._dir.glob("*.safetensors")):
                try:
                    p.unlink()
                except Exception:
                    pass
            for p in list(self._dir.glob("*.meta.json")):
                try:
                    p.unlink()
                except Exception:
                    pass

    def _evict_if_over_cap(self) -> None:
        if not self._max_bytes:
            return
        try:
            entries: List[Tuple[float, int, pathlib.Path]] = []
            total = 0
            for bin_path in self._dir.glob("*.safetensors"):
                if ".tmp." in bin_path.name:
                    continue
                try:
                    st = bin_path.stat()
                    entries.append((st.st_mtime, st.st_size, bin_path))
                    total += st.st_size
                except OSError:
                    continue
            if total <= self._max_bytes:
                return
            entries.sort()
            for _mt, sz, bin_path in entries:
                if total <= self._max_bytes:
                    break
                meta_path = bin_path.parent / (bin_path.stem + ".meta.json")
                try:
                    bin_path.unlink()
                    meta_path.unlink(missing_ok=True)
                    total -= sz
                except OSError:
                    pass
        except Exception as e:
            logger.debug("SSMCompanionDiskStore eviction failed: %s", e)


def build_from_env() -> Optional["SSMCompanionDiskStore"]:
    """Construct a disk store from env vars, or return None if disabled.

    Env vars:
        VMLX_SSM_COMPANION_DISK_DIR — root path. Unset/empty ⇒ disabled.
        VMLX_SSM_COMPANION_DISK_MAX_GB — byte cap (default 10.0).

    Never raises — init failures are logged and return None so callers
    can stay on the pure-memory path.
    """
    cache_dir = os.environ.get("VMLX_SSM_COMPANION_DISK_DIR")
    if not cache_dir:
        return None
    try:
        max_gb = float(os.environ.get("VMLX_SSM_COMPANION_DISK_MAX_GB", "10.0"))
        return SSMCompanionDiskStore(cache_dir, max_size_gb=max_gb)
    except Exception as e:
        logger.warning("SSMCompanionDiskStore init failed: %s", e)
        return None

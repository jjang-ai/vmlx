# SPDX-License-Identifier: Apache-2.0
"""Bounded, exact RoPE table sharing for validated DeepSeek-V4 instances."""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - non-Apple package inspection
    mx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_DEFAULT_MAX_ENTRIES = 64
_DEFAULT_MAX_TOKENS = 4096


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, raw)
        return default
    return value


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_ROPE_CACHE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _TableGroup:
    signature: tuple[float, ...]
    tables: OrderedDict[tuple[int, int, str], tuple[Any, Any]] = field(
        default_factory=OrderedDict
    )
    registrations: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class _WeakIdentityMap:
    def __init__(self) -> None:
        self._items: dict[int, tuple[weakref.ReferenceType[Any], _TableGroup]] = {}

    def __setitem__(self, owner: Any, group: _TableGroup) -> None:
        identity = id(owner)
        items = self._items
        current = items.get(identity)
        if current is not None and current[0]() is owner:
            if current[1] is group:
                return
            _release_group(current[1])

        def _remove(reference: weakref.ReferenceType[Any]) -> None:
            current = items.get(identity)
            if current is not None and current[0] is reference:
                items.pop(identity, None)
                _release_group(current[1])

        self._items[identity] = (weakref.ref(owner, _remove), group)
        with _GROUP_LOCK:
            group.registrations += 1

    def get(self, owner: Any) -> _TableGroup | None:
        entry = self._items.get(id(owner))
        if entry is None or entry[0]() is not owner:
            return None
        return entry[1]


_CONFIGS = _WeakIdentityMap()
_GROUPS: dict[tuple[float, ...], _TableGroup] = {}
_GROUP_LOCK = threading.RLock()
_CLASS_PATCHES: dict[type, tuple[Any, Any]] = {}


def _release_group(group: _TableGroup) -> None:
    with _GROUP_LOCK:
        group.registrations = max(0, group.registrations - 1)
        if group.registrations == 0 and _GROUPS.get(group.signature) is group:
            _GROUPS.pop(group.signature, None)
            group.tables.clear()


def _table_group(signature: tuple[float, ...]) -> _TableGroup:
    with _GROUP_LOCK:
        group = _GROUPS.get(signature)
        if group is None:
            group = _TableGroup(signature=signature)
            _GROUPS[signature] = group
        return group


def _frequency_signature(rope: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in rope.inv_freq.tolist())


def _rotate_with_tables(x: Any, cos: Any, sin: Any, inverse: bool) -> Any:
    broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
    cos = cos.reshape(broadcast_shape)
    sin = sin.reshape(broadcast_shape)
    if inverse:
        sin = -sin
    reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = reshaped[..., 0], reshaped[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


def _install_class_wrapper(rope_class: type) -> bool:
    existing = _CLASS_PATCHES.get(rope_class)
    if existing is not None:
        return rope_class.__call__ is existing[1]

    original_call = rope_class.__call__

    def _cached_call(
        self: Any,
        x: Any,
        offset: int = 0,
        inverse: bool = False,
        positions: Any = None,
    ) -> Any:
        group = _CONFIGS.get(self)
        length = int(x.shape[-2]) if getattr(x, "ndim", 0) >= 2 else 0
        width = int(x.shape[-1]) if getattr(x, "ndim", 0) >= 1 else 0
        expected_width = int(getattr(self.inv_freq, "shape", (0,))[0]) * 2
        if (
            group is None
            or not _enabled()
            or positions is not None
            or length <= 0
            or width != expected_width
            or length
            > _positive_env("VMLX_DSV4_ROPE_CACHE_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
        ):
            return original_call(
                self,
                x,
                offset=offset,
                inverse=inverse,
                positions=positions,
            )

        key = (int(offset), length, str(x.dtype))
        try:
            with group.lock:
                entry = group.tables.get(key)
                if entry is None:
                    pos = mx.arange(offset, offset + length, dtype=mx.float32)
                    freqs = pos[:, None] * self.inv_freq[None, :]
                    cos = mx.cos(freqs).astype(x.dtype)
                    sin = mx.sin(freqs).astype(x.dtype)
                    mx.eval(cos, sin)
                    entry = (cos, sin)
                    maximum = _positive_env(
                        "VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES",
                        _DEFAULT_MAX_ENTRIES,
                    )
                    while len(group.tables) >= maximum:
                        group.tables.popitem(last=False)
                    group.tables[key] = entry
                else:
                    group.tables.move_to_end(key)
            return _rotate_with_tables(x, entry[0], entry[1], inverse)
        except Exception as exc:
            logger.warning("DSV4 RoPE table cache miss failed; using reference path: %s", exc)
            return original_call(
                self,
                x,
                offset=offset,
                inverse=inverse,
                positions=positions,
            )

    rope_class.__call__ = _cached_call
    rope_class._vmlx_dsv4_rope_original_call = original_call
    rope_class._vmlx_dsv4_rope_cache = True
    _CLASS_PATCHES[rope_class] = (original_call, _cached_call)
    return True


def _install_for_class(model: Any, rope_class: type) -> int:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return 0
    ropes = [module for _name, module in list(named_modules()) if isinstance(module, rope_class)]
    if not ropes or not _install_class_wrapper(rope_class):
        return 0
    registered = 0
    for rope in ropes:
        try:
            signature = _frequency_signature(rope)
        except Exception as exc:
            logger.info("DSV4 RoPE instance not cached: %s", exc)
            continue
        group = _table_group(signature)
        _CONFIGS[rope] = group
        registered += 1
    return registered


def install_dsv4_rope_cache(model: Any) -> int:
    """Register exact DSV4 RoPE instances for bounded table sharing."""

    if mx is None or not _enabled():
        return 0
    from jang_tools.dsv4.mlx_model import DeepseekV4RoPE

    registered = _install_for_class(model, DeepseekV4RoPE)
    if registered:
        logger.info("DSV4 exact RoPE table sharing enabled for %d instances", registered)
    return registered


def dsv4_rope_cache_status(model: Any) -> dict[str, int]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return {"registered_instances": 0, "table_entries": 0}
    groups: dict[int, _TableGroup] = {}
    registered = 0
    for _name, module in list(named_modules()):
        group = _CONFIGS.get(module)
        if group is None:
            continue
        registered += 1
        groups[id(group)] = group
    return {
        "registered_instances": registered,
        "table_entries": sum(len(group.tables) for group in groups.values()),
    }


__all__ = ["dsv4_rope_cache_status", "install_dsv4_rope_cache"]

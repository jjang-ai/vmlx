# SPDX-License-Identifier: Apache-2.0
"""Safe exact dequantized vocabulary-head cache for DeepSeek-V4.

The JANG DSV4 reference path dequantizes the vocabulary matrix on every
forward call before applying an FP32 matrix multiplication. This module
materializes the dequantized FP32 matrix once, subject to projected Metal
headroom and allocator-cache budget gates, then reuses it for the same FP32
matmul as the reference. The arithmetic is unchanged and fallback never runs
the transformer or mutates its cache twice.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - non-Apple package inspection
    mx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "cache",
        "exact-cache",
        "fp32",
    }


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, raw)
        return default
    return value


def _model_type(model: Any) -> str:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        value = getattr(owner, "model_type", None)
        if value:
            return str(value)
        args = getattr(owner, "args", None)
        value = getattr(args, "model_type", None)
        if value:
            return str(value)
        config = getattr(owner, "config", None)
        if isinstance(config, dict) and config.get("model_type"):
            return str(config["model_type"])
    return ""


def _head_signature(head: Any) -> tuple[Any, ...]:
    return (
        id(getattr(head, "weight", None)),
        id(getattr(head, "scales", None)),
        id(getattr(head, "biases", None)),
        int(getattr(head, "bits", -1)),
        int(getattr(head, "group_size", -1)),
        str(getattr(head, "mode", "affine")),
    )


def _validate_head(head: Any) -> tuple[Any, ...]:
    if mx is None or not hasattr(mx, "dequantize"):
        raise ValueError("MLX dequantize is unavailable")
    weight = getattr(head, "weight", None)
    scales = getattr(head, "scales", None)
    biases = getattr(head, "biases", None)
    bits = int(getattr(head, "bits", -1))
    group_size = int(getattr(head, "group_size", -1))
    mode = str(getattr(head, "mode", "affine"))
    if weight is None or scales is None:
        raise ValueError("lm_head is not quantized")
    if getattr(weight, "ndim", 0) != 2 or getattr(scales, "ndim", 0) != 2:
        raise ValueError("lm_head tensors do not use the expected rank-2 layout")
    if biases is not None and getattr(biases, "shape", None) != scales.shape:
        raise ValueError("lm_head affine bias geometry differs from scales")
    if bits <= 0 or group_size <= 0:
        raise ValueError("lm_head quantization metadata is invalid")
    if mode != "affine":
        raise ValueError(f"lm_head mode {mode!r} is not affine")
    try:
        if "bias" in head:
            raise ValueError("post-matmul lm_head bias is unsupported")
    except TypeError:
        pass
    return _head_signature(head)


@dataclass
class _HeadConfig:
    signature: tuple[Any, ...]
    weight: Any
    reservation_bytes: int = 0
    validated: bool = False
    disabled_reason: str | None = None


class _WeakIdentityMap:
    """Weak instance map that accepts unhashable MLX modules."""

    def __init__(self) -> None:
        self._items: dict[int, tuple[weakref.ReferenceType[Any], _HeadConfig]] = {}

    def __setitem__(self, owner: Any, config: _HeadConfig) -> None:
        identity = id(owner)
        items = self._items

        def _remove(reference: weakref.ReferenceType[Any]) -> None:
            current = items.get(identity)
            if current is not None and current[0] is reference:
                items.pop(identity, None)

        self._items[identity] = (weakref.ref(owner, _remove), config)

    def get(self, owner: Any) -> _HeadConfig | None:
        entry = self._items.get(id(owner))
        if entry is None or entry[0]() is not owner:
            return None
        return entry[1]


_CONFIGS = _WeakIdentityMap()
_CLASS_PATCHES: dict[type, tuple[Any, Any]] = {}
_RESERVATION_LOCK = threading.RLock()
_RESERVATIONS: dict[int, tuple[weakref.ReferenceType[Any], int]] = {}
_CACHE_LIMIT_BASE_BYTES: int | None = None
_CACHE_LIMIT_EFFECTIVE_BYTES: int | None = None


def _reference_logits(hidden: Any, head: Any) -> Any:
    """Apply the JANG reference head to an already-computed hidden state."""

    weight = mx.dequantize(
        head.weight,
        head.scales,
        getattr(head, "biases", None),
        group_size=head.group_size,
        bits=head.bits,
        mode=getattr(head, "mode", "affine"),
    ).astype(mx.float32)
    return hidden.astype(mx.float32) @ weight.T


def _estimated_cache_bytes(head: Any) -> int:
    scales = head.scales
    output_dims = int(head.weight.shape[0])
    input_dims = int(scales.shape[1]) * int(head.group_size)
    return output_dims * input_dims * 4


def _cache_limit_minimum_bytes() -> int:
    return int(_positive_float_env("VMLX_DSV4_MLX_CACHE_MIN_GB", 4.0) * 1024**3)


def _reservation_bytes_locked() -> int:
    return sum(item[1] for item in _RESERVATIONS.values())


def _allocator_floor_bytes_locked() -> int:
    if _CACHE_LIMIT_BASE_BYTES is None:
        return _cache_limit_minimum_bytes()
    return min(_CACHE_LIMIT_BASE_BYTES, _cache_limit_minimum_bytes())


def _apply_allocator_limit_locked() -> int | None:
    global _CACHE_LIMIT_EFFECTIVE_BYTES

    if mx is None or _CACHE_LIMIT_BASE_BYTES is None:
        _CACHE_LIMIT_EFFECTIVE_BYTES = None
        return None
    effective = max(
        _allocator_floor_bytes_locked(),
        _CACHE_LIMIT_BASE_BYTES - _reservation_bytes_locked(),
    )
    mx.set_cache_limit(effective)
    _CACHE_LIMIT_EFFECTIVE_BYTES = effective
    return effective


def configure_dsv4_lm_head_cache_budget(base_limit_bytes: int) -> int:
    """Apply the DSV4 allocator ceiling after persistent head reservations."""

    global _CACHE_LIMIT_BASE_BYTES

    if base_limit_bytes <= 0:
        raise ValueError("base_limit_bytes must be positive")
    with _RESERVATION_LOCK:
        _CACHE_LIMIT_BASE_BYTES = int(base_limit_bytes)
        floor = _allocator_floor_bytes_locked()
        if _reservation_bytes_locked() + floor > _CACHE_LIMIT_BASE_BYTES:
            logger.info(
                "Releasing DSV4 lm_head caches to honor a reduced MLX cache ceiling"
            )
            for reference, _reserved in tuple(_RESERVATIONS.values()):
                model = reference()
                if model is None:
                    continue
                config = _CONFIGS.get(model)
                if config is not None:
                    config.disabled_reason = "allocator cache ceiling was reduced"
                    config.weight = None
                    config.reservation_bytes = 0
            _RESERVATIONS.clear()
        effective = _apply_allocator_limit_locked()
    if effective is None:
        raise RuntimeError("MLX allocator cache control is unavailable")
    return effective


def _release_cache_reservation(
    identity: int,
    reference: weakref.ReferenceType[Any] | None = None,
) -> None:
    with _RESERVATION_LOCK:
        current = _RESERVATIONS.get(identity)
        if current is None or (reference is not None and current[0] is not reference):
            return
        _RESERVATIONS.pop(identity, None)
        try:
            _apply_allocator_limit_locked()
        except Exception as exc:
            logger.warning("Could not restore the DSV4 MLX cache budget: %s", exc)


def _reserve_allocator_cache(model: Any, estimated: int) -> bool:
    if estimated <= 0:
        return False
    identity = id(model)
    with _RESERVATION_LOCK:
        current = _RESERVATIONS.get(identity)
        if current is not None and current[0]() is model:
            return True
        if _CACHE_LIMIT_BASE_BYTES is None:
            logger.info(
                "DSV4 exact lm_head cache skipped: allocator budget was not configured"
            )
            return False
        projected = _reservation_bytes_locked() + estimated
        floor = _allocator_floor_bytes_locked()
        if _CACHE_LIMIT_BASE_BYTES - projected < floor:
            logger.info(
                "DSV4 exact lm_head cache skipped: %.2f GB reservation would "
                "reduce the MLX cache below its %.2f GB safety floor",
                estimated / 1024**3,
                floor / 1024**3,
            )
            return False

        def _release(reference: weakref.ReferenceType[Any]) -> None:
            _release_cache_reservation(identity, reference)

        reference = weakref.ref(model, _release)
        _RESERVATIONS[identity] = (reference, estimated)
        try:
            _apply_allocator_limit_locked()
        except Exception as exc:
            _RESERVATIONS.pop(identity, None)
            logger.warning("Could not reserve the DSV4 MLX cache budget: %s", exc)
            return False
    return True


def _release_model_cache(model: Any, config: _HeadConfig, reason: str) -> None:
    config.disabled_reason = reason
    config.weight = None
    config.reservation_bytes = 0
    _release_cache_reservation(id(model))


def _parameter_bytes(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters) or mx is None:
        return 0
    try:
        from mlx.utils import tree_flatten
    except ImportError:
        return 0
    total = 0
    seen: set[int] = set()
    for item in tree_flatten(parameters()):
        value = item[-1] if isinstance(item, tuple) else item
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(getattr(value, "nbytes", 0) or 0)
    return total


def _cache_admitted(model: Any, estimated: int) -> bool:
    maximum_gb = _positive_float_env("VMLX_DSV4_LM_HEAD_CACHE_MAX_GB", 4.0)
    reserve_gb = _positive_float_env("VMLX_DSV4_LM_HEAD_CACHE_RESERVE_GB", 4.0)
    if estimated <= 0 or estimated > maximum_gb * 1024**3:
        return False
    try:
        from ..utils.memory_limits import get_effective_metal_working_set_bytes

        active, maximum = get_effective_metal_working_set_bytes(mx)
    except Exception as exc:
        logger.info("DSV4 exact lm_head cache skipped: headroom lookup failed: %s", exc)
        return False
    parameters = _parameter_bytes(model)
    if maximum <= 0 or parameters <= 0:
        return False
    projected = max(int(active), parameters) + estimated + int(reserve_gb * 1024**3)
    admitted = projected <= int(maximum * 0.98)
    if not admitted:
        logger.info(
            "DSV4 exact lm_head cache skipped: projected %.2f GB exceeds "
            "safe %.2f GB working-set bound",
            projected / 1024**3,
            maximum * 0.98 / 1024**3,
        )
    return admitted


def _materialize_weight(head: Any) -> Any:
    weight = mx.dequantize(
        head.weight,
        head.scales,
        getattr(head, "biases", None),
        group_size=head.group_size,
        bits=head.bits,
        mode=getattr(head, "mode", "affine"),
    ).astype(mx.float32)
    mx.eval(weight)
    return weight


def _install_class_wrapper(model_class: type) -> bool:
    existing = _CLASS_PATCHES.get(model_class)
    if existing is not None:
        return model_class.__call__ is existing[1]

    original_call = model_class.__call__

    def _guarded_call(
        self: Any,
        input_ids: Any,
        cache: Any = None,
        mask: Any = None,
    ) -> Any:
        config = _CONFIGS.get(self)
        head = getattr(self, "lm_head", None)
        if config is None:
            return original_call(self, input_ids, cache=cache, mask=mask)
        if not _enabled():
            _release_model_cache(self, config, "disabled by VMLX_DSV4_LM_HEAD_MODE")
            return original_call(self, input_ids, cache=cache, mask=mask)
        if head is None or _head_signature(head) != config.signature:
            _release_model_cache(self, config, "lm_head parameters changed")
            return original_call(self, input_ids, cache=cache, mask=mask)
        if config.disabled_reason is not None or config.weight is None:
            return original_call(self, input_ids, cache=cache, mask=mask)

        hidden = self.model(input_ids, cache=cache, mask=mask)
        try:
            logits = hidden.astype(mx.float32) @ config.weight.T
            # Materialize the first matrix multiplication so any runtime
            # mismatch falls back against this hidden state, never by replaying
            # the mutable request cache.
            if not config.validated:
                mx.eval(logits)
                config.validated = True
            return logits
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _release_model_cache(self, config, reason)
            logger.warning(
                "DSV4 exact lm_head cache disabled after validation failure; "
                "using the reference projection without replaying the model: %s",
                exc,
            )
            return _reference_logits(hidden, head)

    model_class.__call__ = _guarded_call
    model_class._vmlx_dsv4_lm_head_original_call = original_call
    model_class._vmlx_dsv4_lm_head_fastpath = True
    _CLASS_PATCHES[model_class] = (original_call, _guarded_call)
    return True


def install_dsv4_lm_head_fastpath(model: Any) -> bool:
    """Register one validated DSV4 model instance for exact cached dispatch."""

    if mx is None or not _enabled() or _model_type(model) != "deepseek_v4":
        return False
    try:
        signature = _validate_head(getattr(model, "lm_head", None))
    except (AttributeError, TypeError, ValueError) as exc:
        logger.info("DSV4 exact lm_head cache not installed: %s", exc)
        return False
    existing = _CONFIGS.get(model)
    if (
        existing is not None
        and existing.signature == signature
        and existing.weight is not None
        and existing.disabled_reason is None
    ):
        return True
    if existing is not None:
        _release_model_cache(model, existing, "lm_head cache was reconfigured")
    head = model.lm_head
    estimated = _estimated_cache_bytes(head)
    if not _cache_admitted(model, estimated):
        return False
    if not _reserve_allocator_cache(model, estimated):
        return False
    try:
        cached_weight = _materialize_weight(head)
    except Exception as exc:
        _release_cache_reservation(id(model))
        logger.warning("DSV4 exact lm_head cache materialization failed: %s", exc)
        return False
    if not _install_class_wrapper(type(model)):
        _release_cache_reservation(id(model))
        logger.warning("DSV4 exact lm_head class wrapper was replaced; using native")
        return False
    _CONFIGS[model] = _HeadConfig(
        signature=signature,
        weight=cached_weight,
        reservation_bytes=estimated,
    )
    logger.info(
        "DSV4 exact dequantized lm_head cache enabled for one validated model "
        "instance (%.2f GB)",
        estimated / 1024**3,
    )
    return True


def dsv4_lm_head_fastpath_status(model: Any) -> dict[str, Any]:
    """Return authoritative instance registration state for reports/tests."""

    config = _CONFIGS.get(model)
    return {
        "installed": config is not None,
        "validated": bool(config and config.validated),
        "disabled_reason": config.disabled_reason if config else None,
        "cache_bytes": int(getattr(config.weight, "nbytes", 0) or 0) if config else 0,
        "reservation_bytes": config.reservation_bytes if config else 0,
        "allocator_cache_limit_bytes": _CACHE_LIMIT_EFFECTIVE_BYTES,
    }


__all__ = [
    "configure_dsv4_lm_head_cache_budget",
    "dsv4_lm_head_fastpath_status",
    "install_dsv4_lm_head_fastpath",
]

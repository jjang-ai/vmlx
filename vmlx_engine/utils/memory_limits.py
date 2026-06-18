# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Metal working-set and scheduler memory guard limits."""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple


_MB = 1024**2
_GB = 1024**3


def _parse_bool_env(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
        if value <= 0:
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def _cfg_get(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _dtype_scalar_bytes(dtype) -> int:
    raw = str(dtype or "").lower()
    if any(marker in raw for marker in ("float32", "fp32", "f32")):
        return 4
    if any(marker in raw for marker in ("float64", "fp64", "f64")):
        return 8
    if any(marker in raw for marker in ("int8", "uint8", "fp8", "float8")):
        return 1
    # MLX KV cache for these local LLM paths is normally fp16/bf16.
    return 2


def estimate_kv_bytes_per_token_from_config(config) -> int:
    """Estimate live KV-cache bytes added per generated token.

    The estimate intentionally uses standard K+V cache geometry and leaves
    family-specific temporary/fragmentation safety to callers via their
    projected-budget multiplier. Dict and attr-style configs are both accepted,
    including multimodal wrappers that store text fields under ``text_config``.
    """
    text_config = _cfg_get(config, "text_config")
    candidates = [text_config, config] if text_config is not None else [config]
    for cfg in candidates:
        n_layers = (
            _cfg_get(cfg, "num_hidden_layers")
            or _cfg_get(cfg, "n_layers")
            or _cfg_get(cfg, "num_layers")
            or 0
        )
        n_heads = (
            _cfg_get(cfg, "num_attention_heads")
            or _cfg_get(cfg, "n_heads")
            or _cfg_get(cfg, "attention_heads")
            or 0
        )
        n_kv_heads = (
            _cfg_get(cfg, "num_key_value_heads")
            or _cfg_get(cfg, "n_kv_heads")
            or _cfg_get(cfg, "num_kv_heads")
            or n_heads
            or 0
        )
        head_dim = _cfg_get(cfg, "head_dim") or 0
        if not head_dim:
            hidden = _cfg_get(cfg, "hidden_size") or _cfg_get(cfg, "d_model") or 0
            head_dim = int(hidden) // int(n_heads) if hidden and n_heads else 0
        dtype = (
            _cfg_get(cfg, "torch_dtype")
            or _cfg_get(cfg, "dtype")
            or _cfg_get(cfg, "mlx_dtype")
        )
        try:
            n_layers = int(n_layers)
            n_kv_heads = int(n_kv_heads)
            head_dim = int(head_dim)
        except (TypeError, ValueError):
            continue
        if n_layers > 0 and n_kv_heads > 0 and head_dim > 0:
            return n_layers * 2 * n_kv_heads * head_dim * _dtype_scalar_bytes(dtype)
    return 0


def projected_output_token_cap(
    *,
    active_bytes: int,
    max_working_set_bytes: int,
    bytes_per_token: int,
    budget_fraction: float = 0.50,
    transient_multiplier: float = 4.0,
) -> int:
    """Return safe generated-token cap from current Metal headroom.

    ``budget_fraction`` reserves headroom for non-KV temporaries. The
    ``transient_multiplier`` accounts for attention workspaces, allocator
    fragmentation, paged/native cache metadata, and media tensors that are not
    represented by the steady-state KV bytes/token estimate.
    """
    try:
        active = int(active_bytes)
        max_ws = int(max_working_set_bytes)
        bpt = int(bytes_per_token)
        fraction = float(budget_fraction)
        multiplier = float(transient_multiplier)
    except (TypeError, ValueError):
        return 0
    if active < 0 or max_ws <= 0 or bpt <= 0:
        return 0
    if fraction <= 0 or fraction > 1:
        fraction = 0.50
    if multiplier < 1:
        multiplier = 1.0
    headroom = max(0, max_ws - active)
    effective_budget = int(headroom * fraction)
    effective_bpt = max(1, int(bpt * multiplier))
    return max(0, effective_budget // effective_bpt)


def _parse_working_set_bytes(raw: str) -> Optional[int]:
    raw = (raw or "").strip().replace(",", "")
    if not raw:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(gb|g|mb|m)?$", raw.lower())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "g"
    if unit in ("mb", "m"):
        return int(value * _MB)
    return int(value * _GB)


def resolve_working_set_override(base_bytes: int) -> int:
    """Resolve an explicit working-set ceiling override.

    Supported env vars:
    - VMLX_METAL_WS_MAX_BYTES: exact bytes.
    - VMLX_METAL_WS_MAX_GB: size in gigabytes.

    When override is provided and valid, it is clamped to ``base_bytes`` so the
    guard cannot exceed the MLX-reported device limit. This prevents callers from
    accidentally bypassing a safe OS/device guard on unsupported hardware.
    """
    override = os.environ.get("VMLX_METAL_WS_MAX_BYTES")
    parsed = None
    if override is not None:
        try:
            parsed = int((override or "").strip().replace(",", ""))
        except (TypeError, ValueError):
            parsed = _parse_working_set_bytes(override)
    else:
        override = os.environ.get("VMLX_METAL_WS_MAX_GB")
        parsed = _parse_working_set_bytes(override) if override is not None else None
    if parsed is None:
        return base_bytes
    if parsed <= 0:
        return base_bytes
    if base_bytes > 0:
        return min(base_bytes, parsed)
    return parsed


def get_metal_working_set_stats(mx_module=None) -> Tuple[int, int]:
    """Return ``(active_memory_bytes, max_working_set_bytes)`` from MLX."""
    if mx_module is None:
        import mlx.core as mx_module

    get_active = getattr(mx_module, "get_active_memory", None) or mx_module.metal.get_active_memory
    get_device_info = getattr(mx_module, "device_info", None) or mx_module.metal.device_info

    try:
        active = int(get_active() or 0)
    except Exception:
        active = 0

    max_ws = 0
    try:
        info = get_device_info() or {}
        max_ws = int(info.get("max_recommended_working_set_size", 0) or 0)
    except Exception:
        max_ws = 0

    return active, max_ws


def get_effective_metal_working_set_bytes(mx_module=None) -> Tuple[int, int]:
    """Return ``(active_memory_bytes, effective_max_working_set_bytes)``.

    Effective limit uses override env vars when valid, otherwise uses MLX's
    `max_recommended_working_set_size`.
    """
    active, base_ws = get_metal_working_set_stats(mx_module)
    return active, resolve_working_set_override(base_ws)


def get_metal_ws_guard_threshold(default: float = 98.0) -> float:
    """Current percent threshold for working-set rejection (e.g. 98 = 98%).

    Default raised from 85 to 98 (Eric directive 2026-05-11): users should be
    able to fill near-all of their unified memory before the guard fires.
    The 2% headroom still catches the genuine Metal command-buffer OOM edge
    case before MLX raises [METAL] Insufficient Memory and crashes the
    engine process. Override via VMLX_METAL_WS_REJECT_PCT.
    """
    return _parse_float_env("VMLX_METAL_WS_REJECT_PCT", default)


def is_metal_ws_guard_enabled() -> bool:
    """Whether the metal working-set guard is enabled."""
    return _parse_bool_env("VMLX_METAL_WS_GUARD", "1")

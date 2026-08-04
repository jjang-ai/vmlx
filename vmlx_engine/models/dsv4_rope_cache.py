# SPDX-License-Identifier: Apache-2.0
"""Exact RoPE table reuse for DSV4 decode and prefill.

The cache is enabled by default for the exact DSV4 RoPE implementation and
can be disabled with ``VMLX_DSV4_ROPE_CACHE=0``.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)
_PATCHED_CLASS: type | None = None
_TABLES: OrderedDict[tuple[Any, ...], tuple[mx.array, mx.array]] = OrderedDict()
_DEFAULT_MAX_TABLES = 64


def _max_tables() -> int:
    raw = os.environ.get("VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES", str(_DEFAULT_MAX_TABLES))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES=%r",
            raw,
        )
        return _DEFAULT_MAX_TABLES
    if value <= 0:
        logger.warning(
            "Ignoring non-positive VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES=%r",
            raw,
        )
        return _DEFAULT_MAX_TABLES
    return value


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_ROPE_CACHE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _native_enabled() -> bool:
    """Use MLX's fused RoPE kernel when the operator has not opted out."""

    # The fused kernel is useful for exploratory profiling, but its FP16
    # transcendental rounding can accumulate into a different long-run token
    # stream. Keep the exact table path as the production default until a
    # bit-exact native implementation is available.
    value = os.environ.get("VMLX_DSV4_ROPE_NATIVE", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def install_dsv4_rope_cache(model: Any | None = None) -> bool:
    """Reuse identical default-position RoPE tables on the exact DSV4 class."""

    global _PATCHED_CLASS
    if not _enabled():
        return False
    from jang_tools.dsv4.mlx_model import DeepseekV4RoPE

    if _PATCHED_CLASS is DeepseekV4RoPE:
        return True
    if getattr(DeepseekV4RoPE, "_vmlx_dsv4_rope_cache", False):
        _PATCHED_CLASS = DeepseekV4RoPE
        if model is not None:
            _tag_rope_signatures(model, DeepseekV4RoPE)
        return True
    if model is not None:
        _tag_rope_signatures(model, DeepseekV4RoPE)
    original_call = DeepseekV4RoPE.__call__

    def _cached_call(
        self: Any,
        x: mx.array,
        offset: int = 0,
        inverse: bool = False,
        positions: Any = None,
    ) -> mx.array:
        # Position arrays are used by compressed-pool/indexer RoPE and may be
        # ragged. Leave those on the source implementation; the hot q/k/v
        # path uses the scalar offset form and is exactly reusable.
        if positions is not None:
            return original_call(
                self, x, offset=offset, inverse=inverse, positions=positions
            )

        # The native kernel is retained as an explicit experiment only. Its
        # transcendental rounding can accumulate into a different token
        # stream, so the exact table path below remains the default.
        if (
            _native_enabled()
            and getattr(x, "ndim", 0) >= 2
            and int(x.shape[-2]) == 1
            and hasattr(mx.fast, "rope")
        ):
            native_freqs = getattr(self, "_vmlx_dsv4_native_freqs", None)
            if native_freqs is None:
                native_freqs = 1.0 / self.inv_freq
                native_inverse_freqs = -native_freqs
                mx.eval(native_freqs, native_inverse_freqs)
                self._vmlx_dsv4_native_freqs = native_freqs
                self._vmlx_dsv4_native_inverse_freqs = native_inverse_freqs
            elif inverse:
                native_inverse_freqs = getattr(
                    self, "_vmlx_dsv4_native_inverse_freqs", -native_freqs
                )
            else:
                native_inverse_freqs = None
            return mx.fast.rope(
                x,
                dims=self.dims,
                traditional=True,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=(native_inverse_freqs if inverse else native_freqs),
            )

        dtype = x.dtype
        length = int(x.shape[-2])
        signature = getattr(self, "_vmlx_dsv4_rope_signature", id(self.inv_freq))
        key = (signature, int(offset), length, str(dtype))
        entry = _TABLES.get(key)
        if entry is None:
            pos = mx.arange(offset, offset + length, dtype=mx.float32)
            freqs = pos[:, None] * self.inv_freq[None, :]
            entry = (mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype))
            while len(_TABLES) >= _max_tables():
                _TABLES.popitem(last=False)
            _TABLES[key] = entry
        else:
            _TABLES.move_to_end(key)
        cos, sin = entry
        broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
        cos = cos.reshape(broadcast_shape)
        sin = sin.reshape(broadcast_shape)

        if inverse:
            sin = -sin
        reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = reshaped[..., 0], reshaped[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(*out.shape[:-2], out.shape[-2] * 2)

    DeepseekV4RoPE._vmlx_dsv4_rope_original_call = original_call
    DeepseekV4RoPE.__call__ = _cached_call
    DeepseekV4RoPE._vmlx_dsv4_rope_cache = True
    _PATCHED_CLASS = DeepseekV4RoPE
    logger.info("DSV4 exact RoPE table cache enabled")
    return True


def _tag_rope_signatures(model: Any, rope_class: type) -> None:
    """Assign equal signatures to equal-frequency RoPE objects once at load."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    for _name, module in list(named_modules()):
        if not isinstance(module, rope_class):
            continue
        try:
            values = tuple(float(value) for value in module.inv_freq.tolist())
            module._vmlx_dsv4_rope_signature = values
        except Exception:
            module._vmlx_dsv4_rope_signature = id(module.inv_freq)


__all__ = ["install_dsv4_rope_cache"]

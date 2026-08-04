# SPDX-License-Identifier: Apache-2.0
"""Decode-only fast paths for the quantized DeepSeek-V4 vocabulary head.

The JANG DSV4 model implementation dequantizes an 8-bit ``lm_head`` on every
forward call before its FP32 vocabulary projection. The default fast path uses
MLX's native affine ``quantized_matmul`` directly; the conservative ``fp32``
mode retains the exact dequantized matrix once. Both avoid repeated full-head
dequantization and have an explicit native opt-out.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - package inspection off Apple Silicon
    mx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_MAX_CACHE_GB = 4.0


def _mode() -> str:
    explicit = os.environ.get("VMLX_DSV4_LM_HEAD_MODE", "quantized")
    value = explicit.strip().lower()
    if value in {"0", "false", "no", "off", "native"}:
        return "native"
    if value in {"fp32", "cache", "exact"}:
        return "fp32"
    if os.environ.get("VMLX_DSV4_CACHE_LM_HEAD", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return "native"
    return "quantized"


def _estimated_fp32_bytes(head: Any) -> int:
    try:
        weight = getattr(head, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            return 0
        bits = int(getattr(head, "bits", 0) or 0)
        if bits <= 0 or 32 % bits:
            return 0
        input_dim = int(weight.shape[1]) * 32 // bits
        return int(weight.shape[0]) * input_dim * 4
    except (AttributeError, TypeError, ValueError, IndexError):
        return 0


def _max_fp32_cache_bytes() -> int:
    raw = os.environ.get("VMLX_DSV4_CACHE_LM_HEAD_MAX_GB", str(_MAX_CACHE_GB))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid VMLX_DSV4_CACHE_LM_HEAD_MAX_GB=%r", raw)
        value = _MAX_CACHE_GB
    if value <= 0:
        logger.warning(
            "Ignoring non-positive VMLX_DSV4_CACHE_LM_HEAD_MAX_GB=%r",
            raw,
        )
        value = _MAX_CACHE_GB
    return int(value * (1024**3))


def install_dsv4_lm_head_cache(model: Any) -> bool:
    """Install the selected DSV4 vocabulary-head fast path on the model class.

    Returns ``True`` when the loaded model exposes a supported quantized head.
    Installation is idempotent; FP32 cache materialization, when selected, is
    lazy so load-time behavior remains bounded.
    """

    mode = _mode()
    if mx is None or mode == "native":
        return False
    head = getattr(model, "lm_head", None)
    if (
        not hasattr(head, "scales")
        or not hasattr(head, "bits")
        or not hasattr(mx, "quantized_matmul")
    ):
        return False
    estimated = _estimated_fp32_bytes(head)
    if mode == "fp32":
        max_cache_bytes = _max_fp32_cache_bytes()
        if estimated <= 0 or estimated > max_cache_bytes:
            logger.info(
                "DSV4 lm_head FP32 cache skipped: estimated %.2f GB exceeds %.2f GB bound",
                estimated / 1024**3,
                max_cache_bytes / 1024**3,
            )
            return False

    cls = type(model)
    if getattr(cls, "_vmlx_dsv4_lm_head_cache_installed", False):
        return True
    original_call = cls.__call__

    def _cached_call(self: Any, input_ids: Any, cache: Any = None, mask: Any = None):
        active_mode = _mode()
        if active_mode == "native" or getattr(
            self, "_vmlx_dsv4_lm_head_fastpath_disabled", False
        ):
            return original_call(self, input_ids, cache=cache, mask=mask)

        hidden = self.model(input_ids, cache=cache, mask=mask)
        current_head = self.lm_head
        weight = current_head.weight
        scales = current_head.scales
        biases = getattr(current_head, "biases", None)
        try:
            if active_mode == "quantized":
                return mx.quantized_matmul(
                    hidden,
                    weight,
                    scales,
                    biases,
                    transpose=True,
                    group_size=current_head.group_size,
                    bits=current_head.bits,
                    mode=getattr(current_head, "mode", "affine"),
                )
            signature = (
                id(weight),
                id(scales),
                id(biases),
                int(current_head.bits),
                int(current_head.group_size),
                str(getattr(current_head, "mode", "affine")),
            )
            cached = getattr(self, "_vmlx_dsv4_lm_head_fp32", None)
            if (
                cached is None
                or getattr(self, "_vmlx_dsv4_lm_head_signature", None) != signature
            ):
                cached = mx.dequantize(
                    weight,
                    scales,
                    biases,
                    group_size=current_head.group_size,
                    bits=current_head.bits,
                    mode=getattr(current_head, "mode", "affine"),
                ).astype(mx.float32)
                # Materialize once. Subsequent decode calls reuse the exact same
                # dequantized values and do not rebuild the 2 GB graph.
                mx.eval(cached)
                self._vmlx_dsv4_lm_head_fp32 = cached
                self._vmlx_dsv4_lm_head_signature = signature
                logger.info(
                    "DSV4 lm_head FP32 cache materialized: %.2f GB",
                    estimated / 1024**3,
                )
            return hidden.astype(mx.float32) @ cached.T
        except Exception as exc:
            logger.warning(
                "DSV4 lm_head fast path failed; falling back to native path: %s",
                exc,
            )
            self._vmlx_dsv4_lm_head_fastpath_disabled = True
            return original_call(self, input_ids, cache=cache, mask=mask)

    cls.__call__ = _cached_call
    cls._vmlx_dsv4_lm_head_cache_installed = True
    cls._vmlx_dsv4_lm_head_cache_original_call = original_call
    logger.info(
        "DSV4 lm_head mode=%s enabled; set VMLX_DSV4_LM_HEAD_MODE=native "
        "to restore the original per-call dequantization",
        mode,
    )
    return True


__all__ = ["install_dsv4_lm_head_cache"]

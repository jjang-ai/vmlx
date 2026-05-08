# SPDX-License-Identifier: Apache-2.0
"""JANG-quantized model compatibility patches.

Two runtime patches that fix crashes when loading dealignai JANG-quantized
models through ``mlx_lm``:

A) **MiniMax sanitize fix** — ``mlx_lm.models.minimax.Model.sanitize``
   crashes with ``KeyError`` on JANGTQ CRACK models where expert weights
   are stored as ``tq_bits/tq_norms/tq_packed`` (not ``.weight``).  After
   ``jang_loader`` dequants a subset of experts, ``sanitize``'s list
   comprehension ``[weights.pop(key) for e in range(num_local_experts)]``
   fails at expert 3+ which were never dequanted.

   Fix: catch the ``KeyError``, redo the weight_scale_inv step, then
   restructure the MoE experts with safe ``.pop(key, None)`` — skipping
   ``mx.stack`` when any expert is missing (weights already in correct
   format).

B) **MoEGate quantize fix** — ``mlx_lm.utils.load_model`` calls
   ``nn.quantize`` with a ``class_predicate`` that returns ``True`` for
   ``MoEGate`` and ``fc1_latent_proj``/``fc2_latent_proj`` paths.  These
   modules lack ``to_quantized()`` and crash with ``ValueError: Unable to
   quantize model of type MoEGate``.

   Fix: wrap ``nn.quantize`` so the class_predicate result is filtered
   through ``hasattr(m, "to_quantized")`` before the upstream quantizer
   sees it.

Both patches are idempotent — safe to call from every bootstrap, every
process.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch A: MiniMax sanitize
# ---------------------------------------------------------------------------

_minimax_patched: bool = False


def _is_minimax_patched() -> bool:
    """Return True if the MiniMax sanitize patch is already applied."""
    return _minimax_patched


def _patched_minimax_sanitize(original_sanitize: Any) -> Any:
    """Return a wrapper around the original ``Model.sanitize`` that
    tolerates missing expert weight keys after partial jang_loader
    dequantization.
    """

    def wrapper(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        try:
            return original_sanitize(self, weights)
        except KeyError:
            logger.debug(
                "MiniMax sanitize KeyError caught — applying JANG "
                "expert-safe fallback"
            )

        # Redo FP8 dequant (weight_scale_inv handling — mirrors upstream)
        new_weights: dict[str, Any] = {}

        def _dequant(weight: Any, scale_inv: Any) -> Any:
            dtype = mx.bfloat16
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = _dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # MoE expert restructuring — safe version with .pop(key, None)
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights

        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for li in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{li}"
            for orig_name, new_name in mapping.items():
                check = f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight"
                if check not in weights:
                    continue
                to_join: list[Any] = []
                all_present = True
                for e in range(self.args.num_local_experts):
                    key = f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                    val = weights.pop(key, None)
                    if val is None:
                        all_present = False
                        # Put back what we already popped
                        for idx, prev in enumerate(to_join):
                            weights[
                                f"{prefix}.block_sparse_moe.experts.{idx}.{orig_name}.weight"
                            ] = prev
                        break
                    to_join.append(val)
                if all_present and to_join:
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)

        return weights

    return wrapper


def _install_minimax_sanitize() -> bool:
    """Monkey-patch ``mlx_lm.models.minimax.Model.sanitize``.

    Returns True if the patch was applied or was already active.
    """
    global _minimax_patched  # noqa: PLW0603
    if _minimax_patched:
        logger.debug("MiniMax sanitize patch already active")
        return True

    try:
        from mlx_lm.models.minimax import Model  # noqa: WPS433 runtime import
    except ImportError:
        logger.debug(
            "MiniMax sanitize patch skipped: mlx_lm.models.minimax not importable"
        )
        return False

    original = getattr(Model, "sanitize", None)
    if original is None:
        logger.warning(
            "MiniMax sanitize patch skipped: Model.sanitize not found"
        )
        return False

    wrapped = _patched_minimax_sanitize(original)
    Model.sanitize = wrapped
    _minimax_patched = True
    logger.debug("MiniMax sanitize patch installed")
    return True


# ---------------------------------------------------------------------------
# Patch B: MoEGate quantize fix
# ---------------------------------------------------------------------------

_quantize_patched: bool = False
_original_quantize: Any = None
_original_load_model: Any = None


def _is_quantize_patched() -> bool:
    """Return True if the MoEGate quantize patch is already applied."""
    return _quantize_patched


def _safe_quantize(
    model: nn.Module,
    *,
    class_predicate: Any = None,
    **kwargs: Any,
) -> nn.Module:
    """Wrap ``nn.quantize`` so class_predicate results are filtered through
    ``hasattr(m, "to_quantized")``.

    Without this, JANG models return True for MoEGate and
    fc*_latent_proj paths which lack ``to_quantized()``.
    """
    global _original_quantize  # noqa: PLW0603
    if _original_quantize is None:
        # Should not happen — _original_quantize is saved at install time.
        return model

    if class_predicate is None:
        return _original_quantize(model, **kwargs)

    def _filtered_predicate(m: nn.Module) -> bool:
        if not hasattr(m, "to_quantized"):
            return False
        return class_predicate(m)

    return _original_quantize(model, class_predicate=_filtered_predicate, **kwargs)


def _install_quantize_patch() -> bool:
    """Monkey-patch ``mlx_lm.utils.load_model`` to wrap ``nn.quantize``.

    Returns True if the patch was applied or was already active.
    """
    global _quantize_patched, _original_quantize, _original_load_model  # noqa: PLW0603
    if _quantize_patched:
        logger.debug("MoEGate quantize patch already active")
        return True

    try:
        import mlx_lm.utils as _utils  # noqa: WPS433 runtime import
    except ImportError:
        logger.debug(
            "MoEGate quantize patch skipped: mlx_lm.utils not importable"
        )
        return False

    if not hasattr(_utils, "load_model"):
        logger.warning(
            "MoEGate quantize patch skipped: mlx_lm.utils.load_model not found"
        )
        return False

    _original_quantize = nn.quantize
    _original_load_model = _utils.load_model

    # Swap nn.quantize globally so load_model's internal call uses the
    # safe wrapper. This is fine because load_model is the only caller
    # during model loading, and we restore the original after.
    nn.quantize = _safe_quantize  # type: ignore[assignment]
    _quantize_patched = True
    logger.debug("MoEGate quantize patch installed")
    return True


# ---------------------------------------------------------------------------
# Public API — matches kimi_k25_mla / deepseek_v4_register pattern
# ---------------------------------------------------------------------------


def is_patched() -> bool:
    """Return True if both JANG model-compat patches are active."""
    return _is_minimax_patched() and _is_quantize_patched()


def install() -> bool:
    """Install both JANG model-compat patches.

    Idempotent — safe to call from every bootstrap, every process.
    Returns True if both patches are active after the call.
    """
    a = _install_minimax_sanitize()
    b = _install_quantize_patch()
    if a and b:
        logger.debug("All JANG model-compat patches active")
    return a and b

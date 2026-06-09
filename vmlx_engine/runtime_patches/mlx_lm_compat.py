# SPDX-License-Identifier: Apache-2.0
"""Runtime shims for upstream mlx_lm fixes not yet present in pinned wheels."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def install() -> None:
    """Install all mlx_lm compatibility patches once."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_tokenizer_wrapper_find_bounds()
    _patch_batch_rotating_kv_meta_state()
    _patch_lfm2_moe_sigmoid_routing()
    _patch_gemma4_channel_thinking_detection()
    _APPLIED = True


def _patch_tokenizer_wrapper_find_bounds() -> None:
    """Backport mlx-lm#1326: clamp TokenizerWrapper._find bounds.

    mlx_lm.server probes the tail of thinking prompts with ``start=len(prompt)-11``.
    For very short prompts the pinned wheel can pass a negative start into
    ``TokenizerWrapper._find``. Python negative indexing then wraps around and
    can raise ``IndexError`` instead of returning "not found".
    """
    try:
        from mlx_lm.tokenizer_utils import TokenizerWrapper
    except Exception as exc:
        logger.debug("mlx_lm_compat: TokenizerWrapper unavailable: %s", exc)
        return

    current = getattr(TokenizerWrapper, "_find", None)
    if current is None or getattr(current, "_vmlx_find_bounds_patch", False):
        return

    def _vmlx_find(self, tokens, sequence, start=None, end=None, reverse=False):
        token_count = len(tokens)
        seq_count = len(sequence)
        if seq_count == 0:
            return -1

        safe_start = max(0, int(start or 0))
        safe_end = token_count if end is None else min(token_count, max(0, int(end)))
        if safe_end - safe_start < seq_count:
            return -1

        outer_loop = (
            range(safe_end - seq_count, safe_start - 1, -1)
            if reverse
            else range(safe_start, safe_end - seq_count + 1)
        )
        for i in outer_loop:
            if tokens[i] == sequence[0]:
                if all(tokens[i + j] == sequence[j] for j in range(1, seq_count)):
                    return i
        return -1

    _vmlx_find._vmlx_find_bounds_patch = True  # type: ignore[attr-defined]
    TokenizerWrapper._find = _vmlx_find


def _patch_batch_rotating_kv_meta_state() -> None:
    """Backport mlx-lm#1370: parse ``rotated`` meta_state as a string bool."""
    try:
        from mlx_lm.models.cache import BatchRotatingKVCache
    except Exception as exc:
        logger.debug("mlx_lm_compat: BatchRotatingKVCache unavailable: %s", exc)
        return

    prop = getattr(BatchRotatingKVCache, "meta_state", None)
    setter = getattr(prop, "fset", None)
    getter = getattr(prop, "fget", None)
    if setter is None or getter is None:
        return
    if getattr(setter, "_vmlx_batch_rotating_bool_patch", False):
        return

    def _vmlx_meta_state_setter(self, value):
        self.max_size, self._offset, self._idx = map(int, value[:3])
        rotated = value[3] if len(value) > 3 else False
        if isinstance(rotated, str):
            self.rotated = rotated == "True"
        else:
            self.rotated = bool(rotated)

    _vmlx_meta_state_setter._vmlx_batch_rotating_bool_patch = True  # type: ignore[attr-defined]
    BatchRotatingKVCache.meta_state = property(getter, _vmlx_meta_state_setter)


def _patch_lfm2_moe_sigmoid_routing() -> None:
    """Backport mlx-lm#1354: LFM2 MoE routes with unbiased sigmoid scores."""
    try:
        import mlx.core as mx
        from mlx_lm.models import lfm2_moe
    except Exception as exc:
        logger.debug("mlx_lm_compat: LFM2 MoE unavailable: %s", exc)
        return

    block_cls = getattr(lfm2_moe, "Lfm2MoeSparseMoeBlock", None)
    if block_cls is None:
        return

    original_init = getattr(block_cls, "__init__", None)
    if original_init is not None and not getattr(
        original_init, "_vmlx_lfm2_moe_scaling_patch", False
    ):

        def _vmlx_lfm2_init(self, args, _original_init=original_init):
            _original_init(self, args)
            self.routed_scaling_factor = getattr(args, "routed_scaling_factor", 1.0)

        _vmlx_lfm2_init._vmlx_lfm2_moe_scaling_patch = True  # type: ignore[attr-defined]
        block_cls.__init__ = _vmlx_lfm2_init

    current_call = getattr(block_cls, "__call__", None)
    if current_call is None or getattr(
        current_call, "_vmlx_lfm2_moe_sigmoid_routing_patch", False
    ):
        return

    def _vmlx_lfm2_call(self, x):
        routing_weights = mx.sigmoid(self.gate(x))

        k = self.top_k
        if self.use_expert_bias:
            scores_for_routing = routing_weights.astype(mx.float32) + self.expert_bias
            inds = mx.argpartition(scores_for_routing, kth=-k, axis=-1)[..., -k:]
        else:
            inds = mx.argpartition(routing_weights, kth=-k, axis=-1)[..., -k:]

        scores = mx.take_along_axis(routing_weights, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
        scores = scores * getattr(self, "routed_scaling_factor", 1.0)
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y

    _vmlx_lfm2_call._vmlx_lfm2_moe_sigmoid_routing_patch = True  # type: ignore[attr-defined]
    block_cls.__call__ = _vmlx_lfm2_call


def _patch_gemma4_channel_thinking_detection() -> None:
    """Backport mlx-lm#1361: Gemma4 channel tokens are not default thinking."""
    try:
        from mlx_lm import tokenizer_utils
    except Exception as exc:
        logger.debug("mlx_lm_compat: tokenizer_utils unavailable: %s", exc)
        return

    original = getattr(tokenizer_utils, "_infer_thinking", None)
    if original is None or getattr(original, "_vmlx_gemma4_thinking_patch", False):
        return

    def _vmlx_infer_thinking(tokenizer: Any):
        try:
            vocab = tokenizer.get_vocab()
            if (
                "<|channel>" in vocab
                and "<channel|>" in vocab
                and "<|think|>" in vocab
            ):
                return (None, None, None, None)
        except Exception:
            pass
        return original(tokenizer)

    _vmlx_infer_thinking._vmlx_gemma4_thinking_patch = True  # type: ignore[attr-defined]
    tokenizer_utils._infer_thinking = _vmlx_infer_thinking

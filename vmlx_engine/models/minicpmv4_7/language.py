"""MiniCPM-V-4.7 text model: vMLX's vendored Qwen3.5 / Qwen3.5-MoE + canvas M-RoPE.

The backbone is reused unchanged (``mlx_vlm.models.qwen3_5[_moe].language``,
installed by ``models/qwen3_5_family``). This module only adds:

* canvas M-RoPE: ``get_rope_index`` and a ``__call__`` that slices the
  full-prompt position plan for every prefill chunk and decodes at
  ``cache_len + rope_delta`` (transformers ``compute_3d_position_ids``);
* exact Qwen3.5 numerics for this model (see ``install_exact_qwen35_numerics``).
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import LanguageModelOutput

from .canvas import canvas_rope_index
from .config import text_model_is_moe

# torch/FLA ``l2norm(x, eps=1e-6)`` (eps added to the SUM of squares).
QWEN35_QK_L2NORM_EPS = 1e-6


class ZeroCenteredRMSNorm(nn.Module):
    """``Qwen3_5RMSNorm``: ``(x.float() * rsqrt(mean x^2 + eps) * (1 + w.float())).to(x.dtype)``.

    ``weight`` holds the checkpoint's zero-centred ``w`` (lossless BF16). The
    effective scale ``1 + w`` is formed in float32 at call time; rounding it to
    BF16 (what a load-time ``w += 1`` on BF16 storage does) perturbs every
    norm by up to one BF16 ulp of ``1 + w`` (2^-7 for values in [1, 2)).
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def _scale(self) -> mx.array:
        # float32 (1 + w), rebuilt only when the weight array is replaced
        # (load_weights / set_dtype). Kept out of the parameter tree.
        cached = self.__dict__.get("_scale_cache")
        if cached is None or cached[0] is not self.weight:
            cached = (self.weight, 1.0 + self.weight.astype(mx.float32))
            self.__dict__["_scale_cache"] = cached
        return cached[1]

    def __call__(self, x: mx.array) -> mx.array:
        # rms_norm reads x in its own dtype and accumulates in float32; with a
        # float32 scale the result equals the all-float32 formula bit for bit
        # (asserted in tests) without an explicit input cast.
        return mx.fast.rms_norm(x, self._scale(), self.eps).astype(x.dtype)


def install_exact_qwen35_numerics(language_model: nn.Module) -> dict:
    """Swap the vendored Qwen3.5 norms for the exact zero-centred form and
    give every GatedDeltaNet torch's q/k l2norm epsilon.

    Applied to this model's own module instances only; other Qwen3.5-family
    models keep the vendored defaults (documented as a family-wide finding).
    Must run before weights are loaded (parameter names are unchanged).
    """
    inner = language_model.model
    counts = {"rms_norms": 0, "gdn_qk_eps": 0}

    def swap(parent: Any, name: str) -> None:
        old = getattr(parent, name)
        dims = int(old.weight.shape[0])
        setattr(parent, name, ZeroCenteredRMSNorm(dims, eps=float(getattr(old, "eps", 1e-6))))
        counts["rms_norms"] += 1

    for layer in inner.layers:
        swap(layer, "input_layernorm")
        swap(layer, "post_attention_layernorm")
        if getattr(layer, "is_linear", False):
            gdn = layer.linear_attn
            gdn.qk_l2norm_eps = QWEN35_QK_L2NORM_EPS
            counts["gdn_qk_eps"] += 1
        else:
            swap(layer.self_attn, "q_norm")
            swap(layer.self_attn, "k_norm")
    swap(inner, "norm")
    return counts


def _cache_offset(cache: Any, fa_idx: int) -> int:
    if not cache:
        return 0
    entry = cache[fa_idx] if fa_idx < len(cache) else None
    if entry is None:
        return 0
    entry = getattr(entry, "_inner", entry)
    offset = getattr(entry, "offset", 0)
    if offset is None:
        return 0
    if isinstance(offset, mx.array):
        if offset.size == 1:
            return int(offset.reshape(-1)[0].item())
        return int(mx.max(offset).item())
    return int(offset)


class _CanvasLanguageMixin:
    """Shared canvas-position logic for the dense and MoE text backbones."""

    # The batched engine adds each request's own rope delta to its absolute
    # decode positions only for language models that declare this.
    decode_positions_add_request_rope_delta = True

    def _init_canvas_state(self) -> None:
        self._position_ids = None
        self._rope_deltas = None
        self.exact_numerics = install_exact_qwen35_numerics(self)

    def reset_position_state(self) -> None:
        self._position_ids = None
        self._rope_deltas = None

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        *,
        mm_token_type_ids=None,
        target_sizes=None,
        target_sizes_videos=None,
        downsample_mode: Optional[str] = None,
    ):
        """Canvas positions ``(3, B, S)`` and ``(B, 1)`` deltas as int32 MLX arrays.

        Prefix-cache priming accepts the Qwen-style positional signature but
        must also pass this request's native target sizes and token types.
        Media prompts missing those sizes raise rather than inventing a
        Qwen-grid position plan.
        """
        cfg = self.config
        markers = (
            int(cfg.slice_start_id),
            int(cfg.slice_end_id),
            int(cfg.image_start_id),
            int(cfg.image_end_id),
            int(cfg.newline_id),
        )
        positions, deltas = canvas_rope_index(
            np.asarray(input_ids),
            None if mm_token_type_ids is None else np.asarray(mm_token_type_ids),
            None if target_sizes is None else np.asarray(target_sizes),
            None if target_sizes_videos is None else np.asarray(target_sizes_videos),
            markers=markers,
            downsample_mode=downsample_mode or cfg.downsample_mode,
            attention_mask=None if attention_mask is None else np.asarray(attention_mask),
            image_token_id=int(cfg.image_token_id),
            video_token_id=int(cfg.video_token_id),
        )
        return mx.array(positions.astype(np.int32)), mx.array(deltas.astype(np.int32))

    def _positions_for(self, batch: int, seq_len: int, offset: int) -> mx.array:
        plan = self._position_ids
        if (
            plan is not None
            and int(plan.shape[1]) == batch
            and int(plan.shape[-1]) >= offset + seq_len
        ):
            return plan[:, :, offset : offset + seq_len]
        pos = mx.arange(offset, offset + seq_len, dtype=mx.int32)[None, :]
        pos = mx.broadcast_to(pos, (batch, seq_len))
        if self._rope_deltas is not None:
            deltas = self._rope_deltas.reshape(-1, 1).astype(mx.int32)
            if deltas.shape[0] != batch:
                deltas = mx.broadcast_to(deltas[:1], (batch, 1))
            pos = pos + deltas
        return mx.broadcast_to(pos[None, ...], (3, batch, seq_len))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        ref = inputs_embeds if inputs_embeds is not None else inputs
        batch, seq_len = int(ref.shape[0]), int(ref.shape[1])
        if position_ids is None:
            offset = _cache_offset(cache, self.model.fa_idx)
            position_ids = self._positions_for(batch, seq_len, offset)
        elif position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...], (3, position_ids.shape[0], position_ids.shape[1])
            )
        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)


def language_model_class(text_config):
    """Return the canvas language model built on the matching vendored backbone."""
    # The vendored family must own mlx_vlm.models.qwen3_5[_moe] before the base
    # class is imported (idempotent; under DFlash2 it keeps upstream active).
    from vmlx_engine.models.qwen3_5_family import register_qwen3_5_family_runtime

    register_qwen3_5_family_runtime()
    if text_model_is_moe(text_config):
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as Base
    else:
        from mlx_vlm.models.qwen3_5.language import LanguageModel as Base

    cache_key = Base
    cls = _CLASS_CACHE.get(cache_key)
    if cls is None:

        class LanguageModel(_CanvasLanguageMixin, Base):
            def __init__(self, args, config=None):
                Base.__init__(self, args, config)
                self._init_canvas_state()

        LanguageModel.__qualname__ = f"MiniCPMV47{Base.__module__.split('.')[-2].title()}LanguageModel"
        _CLASS_CACHE[cache_key] = cls = LanguageModel
    return cls


_CLASS_CACHE: dict = {}

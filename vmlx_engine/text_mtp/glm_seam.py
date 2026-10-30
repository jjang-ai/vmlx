# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (``glm5_next``) pieces of the text MTP lane.

* Prompt priming (opt-in, ``*_GLM5_MTP_PROMPT_PRIMING``): the prompt's backbone
  hiddens are folded into the head's history so the first drafts see the whole
  prompt. ``PromptHiddenTap`` stands in for the model during mlx-lm's prompt
  forwards and hands each hidden to ``native_mtp_prompt_priming``.
* Exact N-1 prompt boundary: KDA state cannot be reversed, so the reusable
  boundary is the cache right before mlx-lm feeds the final prompt token. It
  is cloned (or handed to the scheduler's SSD writer) at that point.
* Aligned head history (opt-in experiment, ``*_GLM5_ALIGNED_MTP_HEAD_CACHE``):
  the head cache keeps only verifier-confirmed pairs; unverified recursive
  pairs are trimmed before each commit.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_OFF = {"", "0", "false", "off", "no"}


def _model_type(model: Any) -> str:
    inner = getattr(model, "language_model", model)
    model_type = getattr(inner, "model_type", None)
    if model_type is None:
        config = getattr(inner, "config", None)
        if isinstance(config, dict):
            model_type = config.get("model_type")
    return str(model_type or "")


def _glm_switch(model: Any, suffix: str) -> bool:
    if _model_type(model) != "glm5_next":
        return False
    raw = os.environ.get(f"VMLINUX_{suffix}", os.environ.get(f"VMLX_{suffix}", "0"))
    return raw.strip().lower() not in _OFF


def prompt_priming_enabled(model: Any) -> bool:
    return _glm_switch(model, "GLM5_MTP_PROMPT_PRIMING")


def aligned_head_history_enabled(batch: Any) -> bool:
    return _glm_switch(getattr(batch, "model", None), "GLM5_ALIGNED_MTP_HEAD_CACHE")


class PromptHiddenTap:
    """Model stand-in that forwards prompt hiddens to the priming capture."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, inputs, *args, **kwargs):
        from ..native_mtp_prompt_priming import capture_prefill

        kwargs = dict(kwargs)
        kwargs["return_hidden"] = True
        result = self._model(inputs, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("armed GLM prompt priming requires (logits, hidden)")
        logits, hidden = result
        capture_prefill(self._model, inputs, hidden, kwargs.get("cache"))
        return logits


def drop_unverified_head_pairs(lane: Any) -> bool:
    """Trim the recursive (unverified) pairs from the aligned head cache."""
    count = max(0, int(getattr(lane, "head_chain_pairs", 0) or 0))
    if count == 0:
        lane.head_chain_pairs = 0
        return True
    layers = [layer for layer in (lane.mtp_cache or []) if layer is not None]
    trimmable = all(
        hasattr(layer, "is_trimmable")
        and layer.is_trimmable()
        and callable(getattr(layer, "trim", None))
        for layer in layers
    )
    if not layers or not trimmable:
        return False
    for layer in layers:
        if int(layer.trim(count)) != count:
            return False
    lane.head_chain_pairs = 0
    return True


def capture_boundary_snapshots(generator: Any) -> dict:
    """Keep GLM typed caches for requests about to consume their last prompt token.

    mlx-lm splits the final prompt token into its own one-token segment and runs
    it while building the ``GenerationBatch``; right before that is the only
    exact N-1 boundary. Failures only cost a cache store, never the request.
    """
    processing = getattr(generator, "_currently_processing", None) or ()
    prompt_batch = getattr(generator, "_prompt_batch", None)
    if prompt_batch is None or not processing:
        return {}
    pending = [
        index
        for index, entry in enumerate(processing)
        if len(entry[0]) == 1 and len(entry[0][0]) == 1
    ]
    if not pending:
        return {}
    try:
        from ..models.glm5_next.glm5_next import clone_glm5_next_layer_cache
    except Exception:
        return {}

    handoff = getattr(generator, "_vmlx_prompt_boundary_callback", None)
    kept: dict = {}
    for index in pending:
        try:
            layers = prompt_batch.extract_cache(index)
            if not layers or not all(
                type(layer).__name__ in {"Glm5KDACache", "Glm5MLACache"} for layer in layers
            ):
                continue
            uid = int(prompt_batch.uids[index])
            if callable(handoff):
                # Production: the scheduler writes this boundary to SSD now, so
                # no second Metal image has to live beside the model.
                handoff(uid, layers)
                continue
            # New typed wrappers around the same arrays; KDA/legacy MLA replace
            # arrays on update and compact MLA marks shared lanes copy-on-write,
            # so this N-1 graph stays intact after the final token runs.
            kept[uid] = [
                clone_glm5_next_layer_cache(layer, copy_fn=lambda value: value)
                for layer in layers
            ]
        except Exception as exc:
            logger.warning(
                "GLM exact prompt-boundary snapshot failed for uid=%s; "
                "cache store will be skipped: %s",
                prompt_batch.uids[index],
                exc,
            )
    return kept

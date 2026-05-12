# SPDX-License-Identifier: Apache-2.0
"""PFlash drafter — co-resident small model for importance scoring.

The drafter is a *prefill-only* role: we run a single forward over the long
prompt, harvest the per-token logits, then pool them per block.  We don't
sample, generate, or maintain a decode loop here — that's what
``vmlx_engine.speculative`` does for spec-decode.

The drafter lives in the same MLX allocator as the target model so the
Metal command queue can pipeline target prefill against drafter scoring
without round-trip CPU staging.  For an M3 Max with 192 GB unified memory
the drafter overhead is ~600 MB (Qwen3-0.6B BF16) on top of the target.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import mlx.core as mx

from .pflash import PFlashConfig, get_pflash_config

logger = logging.getLogger(__name__)


_drafter_model: Any = None
_drafter_tokenizer: Any = None


def get_pflash_drafter() -> Optional[Any]:
    """Return the loaded drafter model (or None)."""
    return _drafter_model


def get_pflash_drafter_tokenizer() -> Optional[Any]:
    return _drafter_tokenizer


def load_pflash_drafter(config: PFlashConfig) -> Tuple[Any, Any]:
    """Load the drafter from ``config.drafter_model``.

    Returns ``(model, tokenizer)``.  Raises on load failure so the server
    startup path can decide whether to fall back to dense prefill or abort.
    """
    global _drafter_model, _drafter_tokenizer

    if not config.enabled:
        logger.info("[pflash] disabled — skipping drafter load")
        return None, None
    if not config.drafter_model:
        raise ValueError(
            "[pflash] config.drafter_model is empty — set --pflash-drafter "
            "or VMLX_PFLASH_DRAFTER"
        )

    try:
        from mlx_lm import load as mlx_lm_load
    except ImportError as e:
        raise ImportError(
            "[pflash] mlx-lm is required to load the PFlash drafter"
        ) from e

    logger.info("[pflash] loading drafter: %s", config.drafter_model)
    t0 = time.time()
    try:
        model, tokenizer = mlx_lm_load(
            config.drafter_model,
            tokenizer_config={"trust_remote_code": True},
        )
    except Exception as e:
        logger.error("[pflash] drafter load failed: %s", e)
        raise

    dt = time.time() - t0
    logger.info("[pflash] drafter loaded in %.2fs", dt)

    try:
        if hasattr(mx, "get_active_memory"):
            active_gb = mx.get_active_memory() / (1024**3)
            logger.info(
                "[pflash] metal active memory after drafter: %.2fGB", active_gb
            )
    except Exception:
        pass

    _drafter_model = model
    _drafter_tokenizer = tokenizer
    return model, tokenizer


def unload_pflash_drafter() -> None:
    """Drop the drafter and free its memory."""
    global _drafter_model, _drafter_tokenizer

    if _drafter_model is None:
        return
    logger.info("[pflash] unloading drafter")
    _drafter_model = None
    _drafter_tokenizer = None
    try:
        import gc
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Block scoring entry point
# ---------------------------------------------------------------------------


def drafter_score_blocks(
    input_ids: mx.array,
    *,
    block_size: int,
) -> mx.array:
    """Run the drafter on ``input_ids`` and return per-block logit averages.

    Args:
        input_ids: ``(1, seq_len)`` int32 token ids.
        block_size: tokens per scoring block.

    Returns:
        ``(num_blocks, vocab)`` averaged drafter logits.  The caller is
        responsible for turning these into a per-block scalar (see
        ``pflash.pflash_score_blocks``).

    Notes:
        - This is the *generic* path: a full drafter forward + per-block
          mean pool.  A future Metal kernel can fuse the pool into the
          attention output, but the algorithmic contract here is stable.
        - When the drafter is not loaded, raises ``RuntimeError`` so the
          caller can fall back to dense prefill.
    """
    if _drafter_model is None:
        raise RuntimeError("[pflash] drafter not loaded")

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"[pflash] drafter expects (1, seq_len); got {input_ids.shape}"
        )
    seq_len = int(input_ids.shape[1])
    num_blocks = (seq_len + block_size - 1) // block_size

    # Single drafter forward.  We pass cache=None so the model just runs the
    # standard prefill path; we discard the KV state.
    try:
        out = _drafter_model(input_ids)
    except Exception as e:
        raise RuntimeError(f"[pflash] drafter forward failed: {e}") from e

    logits = out.logits if hasattr(out, "logits") else out
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            f"[pflash] unexpected drafter logits shape: {logits.shape}"
        )

    # Per-block mean over the seq dimension.
    seq_logits = logits[0]  # (seq_len, vocab)
    pooled = []
    for b in range(num_blocks):
        lo = b * block_size
        hi = min(lo + block_size, seq_len)
        pooled.append(mx.mean(seq_logits[lo:hi], axis=0))
    return mx.stack(pooled, axis=0)


def make_drafter_score_fn(input_ids: mx.array, block_size: int):
    """Return a ``(start, end) -> logits`` closure for ``plan_sparse_prefill``.

    The closure currently always scores the full prompt and slices into
    the requested range.  Subspan scoring (drafter chunked over the prompt)
    is a later optimization.
    """
    pooled = drafter_score_blocks(input_ids, block_size=block_size)

    def _score(start: int, end: int) -> mx.array:
        block_start = start // block_size
        block_end = (end + block_size - 1) // block_size
        return pooled[block_start:block_end]

    return _score

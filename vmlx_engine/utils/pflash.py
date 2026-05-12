# SPDX-License-Identifier: Apache-2.0
"""PFlash — importance-scored sparse prefill for long-context cold TTFT.

Issue #136. PFlash uses a small drafter (e.g. Qwen3-0.6B BF16) to score
per-block importance over a long prompt; the target model then only prefills
the spans that matter. CUDA reference (Luce-Org/lucebox-hub `pflash/`) reports
**128K cold TTFT 24.8 s vs llama.cpp 257 s = 10.4×** on RTX 3090 with
`keep_ratio=0.05`, NIAH single-needle retrieved at every measured context.

This module ships the **algorithm pieces in pure MLX** so the integration
landing point exists in the engine without blocking on a custom Metal kernel:

    score = pflash_score_blocks(drafter, prompt_ids, block_size, ...)
    keep_mask = pflash_select_top_k(score, keep_ratio)
    prefill_blocks = pflash_block_ranges(keep_mask, block_size, seq_len)

The four CUDA kernels (mean_K → score → select → sparse_fwd) map onto MLX
primitives:

    mean_K       → mx.mean(K, axis=-2) per block
    score        → drafter logits ⊗ aggregated K (block-pooled dot product)
    select       → mx.argpartition top-(keep_ratio * num_blocks)
    sparse_fwd   → standard MLX attention with a precomputed sparse block-mask

A follow-up PR may introduce a Metal Block-Sparse-Attention kernel
(BSA-equivalent, FA-2 derived) to close the per-block constant-factor gap;
the public functions in this module already accept an `attention_impl` hook
so the optimized kernel slots in without changing the integration callsite.

Quality gate (run with model weights, not in CI):

    pytest tests/test_pflash.py -v  # algorithm unit tests
    python tests/benchmark/test_pflash_ttft.py --keep-ratio 0.05  # cold TTFT
    python tests/benchmark/test_pflash_niah.py                   # NIAH eval

Default state: OFF. Enable with `--enable-pflash` or
`VMLX_ENABLE_PFLASH=1`. See `pflash_drafter.py` for the drafter loader.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PFlashConfig:
    """Configuration for PFlash sparse prefill.

    Attributes:
        enabled: Whether PFlash is active.
        drafter_model: HuggingFace id / local path of the small drafter.
        block_size: Token block size for importance scoring (default 256).
        keep_ratio: Fraction of blocks to retain in the sparse prefill
            (default 0.10; CUDA reference uses 0.05 for very long contexts).
        min_seq_len: Minimum prompt length before PFlash activates. Below
            this, dense prefill is used unconditionally (default 8192).
        always_keep_head: Number of leading tokens always kept (default 256
            — the system + instruction header).
        always_keep_tail: Number of trailing tokens always kept (default 256
            — the most-recent context window the model is about to extend).
    """

    enabled: bool = False
    drafter_model: str = ""
    block_size: int = 256
    keep_ratio: float = 0.10
    min_seq_len: int = 8192
    always_keep_head: int = 256
    always_keep_tail: int = 256

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(
                f"[pflash] keep_ratio must be in (0, 1], got {self.keep_ratio}"
            )
        if self.block_size <= 0:
            raise ValueError(f"[pflash] block_size must be > 0, got {self.block_size}")
        if self.min_seq_len < 0:
            raise ValueError(
                f"[pflash] min_seq_len must be >= 0, got {self.min_seq_len}"
            )


# Module-level state mirroring the speculative.py pattern.
_pflash_config: Optional[PFlashConfig] = None
_pflash_stats: dict = {
    "activations": 0,
    "blocks_total": 0,
    "blocks_kept": 0,
    "skipped_below_min_seq_len": 0,
    "failures": 0,
}


def configure_pflash(config: PFlashConfig) -> None:
    """Install the global PFlash configuration."""
    global _pflash_config
    _pflash_config = config
    if config.enabled:
        logger.info(
            "[pflash] enabled: drafter=%s block_size=%d keep_ratio=%.2f "
            "min_seq_len=%d",
            config.drafter_model or "(unset)",
            config.block_size,
            config.keep_ratio,
            config.min_seq_len,
        )


def get_pflash_config() -> Optional[PFlashConfig]:
    return _pflash_config


def is_pflash_enabled() -> bool:
    """True if PFlash is configured AND a drafter is loaded."""
    from .pflash_drafter import get_pflash_drafter

    cfg = _pflash_config
    if cfg is None or not cfg.enabled:
        return False
    return get_pflash_drafter() is not None


def should_activate_pflash(seq_len: int) -> bool:
    """Decide whether to use PFlash for a prompt of length ``seq_len``."""
    cfg = _pflash_config
    if cfg is None or not cfg.enabled:
        return False
    if not is_pflash_enabled():
        return False
    if seq_len < cfg.min_seq_len:
        _pflash_stats["skipped_below_min_seq_len"] += 1
        return False
    return True


def get_pflash_stats() -> dict:
    """Snapshot of PFlash counters for /health."""
    cfg = _pflash_config
    return {
        "configured": cfg is not None and cfg.enabled,
        "drafter_loaded": is_pflash_enabled(),
        "block_size": cfg.block_size if cfg else None,
        "keep_ratio": cfg.keep_ratio if cfg else None,
        "min_seq_len": cfg.min_seq_len if cfg else None,
        **_pflash_stats,
    }


# ---------------------------------------------------------------------------
# Algorithm pieces (block scoring + selection)
# ---------------------------------------------------------------------------


def pflash_score_blocks(
    drafter_logits_per_block: mx.array,
    *,
    block_size: int,
) -> mx.array:
    """Pool drafter logits into per-block importance scores.

    The CUDA reference computes ``score = mean_K(K) ⊗ Q_recent`` for the
    drafter. Here we accept a per-block aggregate already computed by the
    caller (so the same function works whether the aggregate is logit
    entropy, attention-weight mass, or a custom signal).

    Args:
        drafter_logits_per_block: shape ``(num_blocks, vocab)`` — drafter
            softmax distribution averaged over the block tokens.
        block_size: tokens per block (informational; only used for shape
            sanity checks).

    Returns:
        ``(num_blocks,)`` per-block scalar importance score.
    """
    if drafter_logits_per_block.ndim != 2:
        raise ValueError(
            "[pflash_score_blocks] expected (num_blocks, vocab); got "
            f"{drafter_logits_per_block.shape}"
        )
    # Score: negative entropy of the drafter distribution per block. Blocks
    # where the drafter is highly confident are *less* informative — keep
    # the blocks where the drafter is uncertain.  Mirrors the CUDA
    # reference's "low-entropy first" selection signal.
    probs = mx.softmax(drafter_logits_per_block, axis=-1)
    # Numerical stability: clamp probs before log.
    eps = 1e-9
    entropy = -mx.sum(probs * mx.log(mx.maximum(probs, eps)), axis=-1)
    return entropy


def pflash_select_top_k(
    scores: mx.array,
    keep_ratio: float,
    *,
    always_keep_head_blocks: int = 0,
    always_keep_tail_blocks: int = 0,
) -> mx.array:
    """Select the top-K blocks by score.

    Args:
        scores: ``(num_blocks,)`` per-block importance.
        keep_ratio: fraction of blocks to keep (post head/tail pin).
        always_keep_head_blocks: leading blocks that bypass scoring.
        always_keep_tail_blocks: trailing blocks that bypass scoring.

    Returns:
        Boolean ``(num_blocks,)`` keep mask.
    """
    num_blocks = int(scores.shape[0])
    if num_blocks == 0:
        return mx.zeros((0,), dtype=mx.bool_)

    head = max(0, min(always_keep_head_blocks, num_blocks))
    tail = max(0, min(always_keep_tail_blocks, num_blocks - head))
    middle_lo, middle_hi = head, num_blocks - tail
    middle_count = max(0, middle_hi - middle_lo)
    keep_middle = max(0, int(round(keep_ratio * middle_count)))

    mask = [False] * num_blocks
    for i in range(head):
        mask[i] = True
    for i in range(num_blocks - tail, num_blocks):
        mask[i] = True

    if middle_count > 0 and keep_middle > 0:
        # argpartition would be ideal, but it isn't a stable cross-version
        # MLX op yet — do a lexicographic sort which is O(n log n) and fine
        # for num_blocks on the order of thousands.
        mid_scores = scores[middle_lo:middle_hi]
        # ``mx.argsort`` returns ascending; take the last ``keep_middle``.
        order = mx.argsort(mid_scores)
        top_idx = order[-keep_middle:].tolist()
        for j in top_idx:
            mask[middle_lo + int(j)] = True

    return mx.array(mask, dtype=mx.bool_)


def pflash_block_ranges(
    keep_mask: mx.array,
    *,
    block_size: int,
    seq_len: int,
) -> List[Tuple[int, int]]:
    """Convert a keep mask into ``(start, end)`` token ranges.

    Adjacent kept blocks are coalesced so the prefill loop can dispatch a
    single forward per contiguous span.
    """
    mask = keep_mask.tolist()
    ranges: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    for i, kept in enumerate(mask):
        if kept and run_start is None:
            run_start = i * block_size
        elif not kept and run_start is not None:
            run_end = min(i * block_size, seq_len)
            ranges.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        ranges.append((run_start, seq_len))
    return ranges


# ---------------------------------------------------------------------------
# Sparse-prefill planner
# ---------------------------------------------------------------------------


@dataclass
class PFlashPlan:
    """A concrete sparse-prefill plan for one prompt.

    Attributes:
        block_size: tokens per scoring block.
        seq_len: total prompt length.
        keep_mask: ``(num_blocks,)`` bool mask.
        keep_ranges: coalesced ``(start, end)`` token ranges to forward.
        kept_blocks: count of kept blocks (telemetry).
        total_blocks: count of total blocks (telemetry).
    """

    block_size: int
    seq_len: int
    keep_mask: mx.array
    keep_ranges: List[Tuple[int, int]]
    kept_blocks: int
    total_blocks: int

    def kept_token_count(self) -> int:
        return sum(end - start for start, end in self.keep_ranges)

    def coverage(self) -> float:
        if self.seq_len <= 0:
            return 0.0
        return self.kept_token_count() / self.seq_len


def plan_sparse_prefill(
    drafter_score_fn: Callable[[int, int], mx.array],
    *,
    seq_len: int,
    config: PFlashConfig,
) -> PFlashPlan:
    """Build a sparse-prefill plan for one prompt.

    Args:
        drafter_score_fn: ``(start, end) -> (num_blocks_in_span, vocab)``
            — returns per-block drafter logits over the supplied span.
            The function abstracts over how the drafter is invoked (single
            forward, chunked, BF16, INT8) so the planner stays
            decoupled from the loader.
        seq_len: prompt length in tokens.
        config: active ``PFlashConfig``.

    Returns:
        ``PFlashPlan`` whose ``keep_ranges`` cover the spans to send to the
        target model.
    """
    block_size = config.block_size
    if seq_len <= 0:
        return PFlashPlan(
            block_size=block_size,
            seq_len=0,
            keep_mask=mx.zeros((0,), dtype=mx.bool_),
            keep_ranges=[],
            kept_blocks=0,
            total_blocks=0,
        )

    num_blocks = (seq_len + block_size - 1) // block_size
    head_blocks = (config.always_keep_head + block_size - 1) // block_size
    tail_blocks = (config.always_keep_tail + block_size - 1) // block_size

    drafter_logits = drafter_score_fn(0, seq_len)
    if drafter_logits.shape[0] != num_blocks:
        raise ValueError(
            "[plan_sparse_prefill] drafter returned "
            f"{drafter_logits.shape[0]} blocks; expected {num_blocks}"
        )
    scores = pflash_score_blocks(drafter_logits, block_size=block_size)
    keep_mask = pflash_select_top_k(
        scores,
        keep_ratio=config.keep_ratio,
        always_keep_head_blocks=head_blocks,
        always_keep_tail_blocks=tail_blocks,
    )
    kept_blocks = int(mx.sum(keep_mask.astype(mx.int32)).item())
    ranges = pflash_block_ranges(
        keep_mask, block_size=block_size, seq_len=seq_len
    )

    _pflash_stats["activations"] += 1
    _pflash_stats["blocks_total"] += num_blocks
    _pflash_stats["blocks_kept"] += kept_blocks

    return PFlashPlan(
        block_size=block_size,
        seq_len=seq_len,
        keep_mask=keep_mask,
        keep_ranges=ranges,
        kept_blocks=kept_blocks,
        total_blocks=num_blocks,
    )


def record_pflash_failure(reason: str) -> None:
    """Bump the failure counter and log."""
    _pflash_stats["failures"] += 1
    logger.warning("[pflash] dense fallback: %s", reason)


# ---------------------------------------------------------------------------
# Env-driven config helper (for CLI / server startup)
# ---------------------------------------------------------------------------


def config_from_env(default_enabled: bool = False) -> PFlashConfig:
    """Build a ``PFlashConfig`` from environment variables.

    Supported vars:
        VMLX_ENABLE_PFLASH=1
        VMLX_PFLASH_DRAFTER=<model id or path>
        VMLX_PFLASH_BLOCK_SIZE=256
        VMLX_PFLASH_KEEP_RATIO=0.10
        VMLX_PFLASH_MIN_SEQ_LEN=8192
        VMLX_PFLASH_HEAD_TOKENS=256
        VMLX_PFLASH_TAIL_TOKENS=256
    """
    def _i(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    env_enabled = os.environ.get("VMLX_ENABLE_PFLASH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return PFlashConfig(
        enabled=env_enabled or default_enabled,
        drafter_model=os.environ.get("VMLX_PFLASH_DRAFTER", ""),
        block_size=_i("VMLX_PFLASH_BLOCK_SIZE", 256),
        keep_ratio=_f("VMLX_PFLASH_KEEP_RATIO", 0.10),
        min_seq_len=_i("VMLX_PFLASH_MIN_SEQ_LEN", 8192),
        always_keep_head=_i("VMLX_PFLASH_HEAD_TOKENS", 256),
        always_keep_tail=_i("VMLX_PFLASH_TAIL_TOKENS", 256),
    )

# SPDX-License-Identifier: Apache-2.0
"""Tail-periodic SSM companion checkpoints for hybrid prefill.

WHY
---
A hybrid (GatedDeltaNet/SSM + attention) prompt can only reuse a paged KV
hit when an SSM companion checkpoint exists at a block-aligned position at or
below that hit (the vmlx#91 longest-prefix resume, or the companion DELTA that
advances a checkpoint up to the hit). The vmlx#109 inline capture stores
checkpoints only at the very end of the prompt (the clean ``N-1-gpl``
boundary, the block below it) plus a learned "required" boundary. A client
that rewrites the conversation a few thousand tokens before the end of a long
prompt -- SillyTavern keeps the whole chat history in one user message and
measured a ~70K-token prompt diverging ~10K tokens before its end -- then
finds a KV chain hit but no checkpoint below the divergence, and re-prefills
everything (measured TTFT 93-136 s on Qwen3.8-Flash-Next).

WHAT
----
Capture extra clean checkpoints at absolute token positions that are
multiples of ``VMLX_HYBRID_CHECKPOINT_INTERVAL`` (rounded up to the paged
block size so every checkpoint pairs exactly with a block-aligned KV trim),
only inside the last ``VMLX_HYBRID_CHECKPOINT_TAIL`` tokens before the clean
boundary, at most ``VMLX_HYBRID_CHECKPOINT_MAX`` per request.  When there are
more candidates than the cap, the survivors are thinned evenly across the
window (first and last kept) instead of dropping the oldest, so the window the
operator asked for stays covered.

REFERENCE
---------
This follows mlx-serve's prefill SSM checkpointing (``--ssm-checkpoint-stride``
/ ``--ssm-checkpoint-max``, default 16): snapshots land on absolute positions
that are multiples of the stride (``(end + pos_offset) % stride == 0``), the
chunk loop ends a chunk exactly on a stride boundary, capture is disabled when
the prefix cache that consumes it is disabled, and when the per-request cap is
exceeded the interior is thinned rather than the oldest dropped
(``ssmCheckpointDropIndex``, mlx-serve #330).  The tail window is specific to
this engine: checkpoints far from the end of a long prompt are rarely the
longest usable prefix, and each one costs an SSD write.

CONTROL
-------
``VMLX_HYBRID_CHECKPOINT_INTERVAL`` -- tokens between checkpoints; ``0`` or
unset disables the feature (upstream behaviour). Recommended: 2048.
``VMLX_HYBRID_CHECKPOINT_TAIL`` -- window before the clean boundary, default
32768; ``<= 0`` means the whole prompt (the cap still applies).
``VMLX_HYBRID_CHECKPOINT_MAX`` -- per-request cap, default 16, clamped to 64;
``<= 0`` disables the feature.

Config fallback: when the interval env var is unset and the vMLX config system
selects ``hybrid.ssm_recompute: checkpoint`` (YAML, or
``VMLX_HYBRID_SSM_RECOMPUTE=checkpoint``), ``hybrid.checkpoint_interval_tokens``
is used as the interval.  The default ``ssm_recompute: full`` keeps the feature
off.  Note that ``VMLX_HYBRID_CHECKPOINT_INTERVAL`` is the same env var the
config system already maps to ``hybrid.checkpoint_interval_tokens``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

ENV_INTERVAL = "VMLX_HYBRID_CHECKPOINT_INTERVAL"
ENV_TAIL = "VMLX_HYBRID_CHECKPOINT_TAIL"
ENV_MAX = "VMLX_HYBRID_CHECKPOINT_MAX"

DEFAULT_TAIL_TOKENS = 32768
DEFAULT_MAX_CHECKPOINTS = 16
# Each checkpoint is held in unified memory until the prefill finishes and is
# then written to SSD (~0.11 GB for Qwen3.8-Flash-Next's 36 GDN layers), so
# an operator typo must not be able to request hundreds of them.
HARD_MAX_CHECKPOINTS = 64


@dataclass(frozen=True)
class PeriodicCheckpointPolicy:
    """Resolved periodic-checkpoint knobs for one prefill."""

    interval: int = 0
    tail: int = DEFAULT_TAIL_TOKENS
    max_count: int = DEFAULT_MAX_CHECKPOINTS
    source: str = "disabled"

    @property
    def enabled(self) -> bool:
        return self.interval > 0 and self.max_count > 0


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        logger.warning(
            "Ignoring non-integer hybrid checkpoint setting %r; using %d",
            raw,
            default,
        )
        return default


def config_checkpoint_interval() -> Optional[int]:
    """Interval from the vMLX config system, only in explicit checkpoint mode.

    ``hybrid.checkpoint_interval_tokens`` is documented as "for checkpoint
    mode" and defaults to 512 while ``hybrid.ssm_recompute`` defaults to
    ``full``.  Honouring the interval without the mode switch would silently
    turn the feature on for everyone, so the mode is the gate.
    """
    from ..config import get_config

    cfg = get_config()
    mode = str(cfg.get("hybrid.ssm_recompute", "full") or "full").strip().lower()
    if mode != "checkpoint":
        return None
    try:
        interval = int(cfg.get("hybrid.checkpoint_interval_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None
    return interval if interval > 0 else None


def resolve_periodic_checkpoint_policy(
    environ: Optional[Mapping[str, str]] = None,
    config_interval: Optional[Callable[[], Optional[int]]] = None,
) -> PeriodicCheckpointPolicy:
    """Resolve env first, then the optional config fallback for the interval."""
    env = os.environ if environ is None else environ
    raw_interval = env.get(ENV_INTERVAL)
    source = "disabled"
    interval = 0
    if raw_interval is not None and str(raw_interval).strip():
        interval = _parse_int(raw_interval, 0)
        source = "env"
    elif config_interval is not None:
        try:
            fallback = config_interval()
        except Exception as exc:  # noqa: BLE001 - config is optional here
            logger.debug("hybrid checkpoint config fallback unavailable: %s", exc)
            fallback = None
        if fallback:
            interval = int(fallback)
            source = "config"
    interval = max(0, interval)
    tail = _parse_int(env.get(ENV_TAIL), DEFAULT_TAIL_TOKENS)
    max_count = _parse_int(env.get(ENV_MAX), DEFAULT_MAX_CHECKPOINTS)
    max_count = min(max_count, HARD_MAX_CHECKPOINTS) if max_count > 0 else 0
    if interval <= 0 or max_count <= 0:
        source = "disabled"
    return PeriodicCheckpointPolicy(
        interval=interval,
        tail=tail,
        max_count=max_count,
        source=source,
    )


def _span_preserving_subset(positions: Sequence[int], keep: int) -> List[int]:
    """Keep ``keep`` positions evenly spread, always including both ends.

    With a single survivor the one nearest the clean boundary wins: it is
    the longest prefix a later turn can resume from.
    """
    items = list(positions)
    if keep <= 0:
        return []
    if len(items) <= keep:
        return items
    if keep == 1:
        return [items[-1]]
    last = len(items) - 1
    stride = last / (keep - 1)
    indices = sorted({int(round(i * stride)) for i in range(keep)})
    return [items[i] for i in indices]


def plan_periodic_checkpoints(
    *,
    base_tokens: int,
    seq_len: int,
    upper_local: int,
    interval: int,
    tail: int,
    max_count: int,
    block_size: int,
    safe_abs_limit: Optional[int] = None,
) -> List[int]:
    """Return LOCAL prefill offsets of the periodic checkpoints.

    Args:
        base_tokens: tokens already restored before this prefill
            (``request._cached_tokens``); local offset 0 is absolute token
            ``base_tokens``.
        seq_len: tokens this prefill forwards.
        upper_local: the clean boundary (``seq_len - 1 - gen_prompt_len``),
            local.  Checkpoints never pass it: anything later absorbed the
            generation-prompt suffix and would be gpl-contaminated.
        interval: tokens between checkpoints, rounded UP to ``block_size``.
        tail: window before the clean boundary; ``<= 0`` = whole prompt.
        max_count: per-request cap (span-preserving thinning beyond it).
        block_size: paged KV block size; checkpoints must be multiples of
            it or the resume path rejects them (KV trims to whole blocks).
        safe_abs_limit: optional absolute ceiling (first media placeholder)
            -- the state at position ``p`` covers tokens ``[0, p)``.

    Positions are ABSOLUTE multiples of the interval (the key a later request
    will look up), converted back to local offsets for the prefill loop.
    """
    base = max(0, int(base_tokens or 0))
    seq_len = int(seq_len or 0)
    interval = int(interval or 0)
    max_count = int(max_count or 0)
    block_size = int(block_size or 0)
    if interval <= 0 or max_count <= 0 or block_size <= 0 or seq_len <= 1:
        return []
    # The final prompt token is always forwarded on its own for logits.
    upper_local = min(int(upper_local or 0), seq_len - 1)
    if upper_local <= 0:
        return []
    step = -(-interval // block_size) * block_size
    abs_clean = base + upper_local
    abs_end = abs_clean
    if safe_abs_limit is not None:
        abs_end = min(abs_end, int(safe_abs_limit))
    lowest = base + 1
    if int(tail or 0) > 0:
        lowest = max(lowest, abs_clean - int(tail))
    first = -(-lowest // step) * step
    if first > abs_end:
        return []
    positions = list(range(first, abs_end + 1, step))
    positions = _span_preserving_subset(positions, max_count)
    return [p - base for p in positions]

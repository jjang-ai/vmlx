# SPDX-License-Identifier: Apache-2.0
"""Counters and published telemetry for the text-lane native-MTP decoder.

Each request owns a ``LaneStats``. Records are published (``publish``) at the
end of a request, at an AR handoff and periodically while parked in AR; the
API thread reads them through ``native_mtp_stats_snapshot`` (``/health``, the
panel's PerformancePanel, ``last_request.speculative``). The generation loop
and the API loop run on different threads, so everything shared lives behind
one lock and readers get copies. Totals reset with the process.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..native_mtp_cache_telemetry import (
    native_mtp_cache_lifecycle_snapshot,
    native_mtp_cache_snapshot,
)

logger = logging.getLogger(__name__)

# Same switch as the lane's phase barriers; recorded so a reader knows whether
# the phase timings are synchronized GPU costs or enqueue times.
PROFILED = bool(os.environ.get("VMLX_MTP_PROFILE"))

_lock = threading.Lock()
_last: Optional[dict] = None
_last_skip: Optional[dict] = None
TOTALS = {
    "requests": 0,
    "cycles": 0,
    "drafted_tokens": 0,
    "accepted_tokens": 0,
    "mtp_cache_recreated_on_rejects": 0,
    "mtp_cache_retained_on_rejects": 0,
}

_EMIT_FIELDS = {
    "init": "init_emits",
    "draft": "draft_emits",
    "bonus": "bonus_emits",
    "verify": "verify_emits",
}


@dataclass
class LaneStats:
    """Per-request draft/verify counters and phase timings."""

    cycles: int = 0  # verify forwards
    accepts: int = 0  # cycles whose whole draft chain was accepted
    rejects: int = 0  # cycles with a rejected draft row
    init_emits: int = 0  # the two tokens queued by the seed
    draft_emits: int = 0  # accepted drafts
    bonus_emits: int = 0  # extra token after a fully accepted chain
    verify_emits: int = 0  # correction token after a rejection
    depth: int = 1
    starting_depth: int = 1
    depth_ceiling: int = 1
    depth_policy: str = "fixed"
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    accepted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    drafted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    seed_main_forwards: int = 0
    verify_main_forwards: int = 0
    mtp_forwards: int = 0
    backbone_ms: float = 0.0  # verify forwards
    mtp_head_ms: float = 0.0  # draft chains
    sample_ms: float = 0.0  # sampling + acceptance
    cache_ops_ms: float = 0.0  # rewind / commit
    mtp_cache_recreated_on_rejects: int = 0
    mtp_cache_retained_on_rejects: int = 0
    mtp_head_cache_policy: str = "loose"
    prompt_primed_pairs: int = 0
    prompt_prime_source: str = "unprimed"
    mtp_head_cache: dict = field(default_factory=dict)
    adaptive_depth_value: dict = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    fallback_cost_ratio: Optional[float] = None
    fallback_mtp_ms_per_token: Optional[float] = None
    fallback_ar_step_ms: Optional[float] = None
    seed_ar_step_ms: Optional[float] = None
    seed_context_tokens: Optional[int] = None
    seed_context_source: str = "unknown"
    recovery: dict = field(default_factory=dict)
    # Totals take only the growth since this object's previous publication, and
    # the request is counted once even when it re-enters from AR.
    totals_published: dict = field(default_factory=dict, repr=False)
    request_counted: bool = False

    def count_emit(self, source: str) -> None:
        name = _EMIT_FIELDS.get(source)
        if name is not None:
            setattr(self, name, getattr(self, name) + 1)

    def phase_total_ms(self) -> float:
        return (
            float(self.backbone_ms)
            + float(self.mtp_head_ms)
            + float(self.sample_ms)
            + float(self.cache_ops_ms)
        )


def rate(accepted: int, drafted: int) -> Optional[float]:
    return accepted / drafted if drafted > 0 else None


def build_record(uid: Any, stats: LaneStats, finish_reason: str) -> dict:
    timings = {
        "verify": float(stats.backbone_ms),
        "sample": float(stats.sample_ms),
        "draft": float(stats.mtp_head_ms),
        "cache": float(stats.cache_ops_ms),
    }
    total = sum(timings.values())
    timings["total"] = total
    timings["avg_cycle"] = total / max(1, int(stats.cycles or 0))
    return {
        "request_id": str(uid),
        "finish_reason": finish_reason,
        "final_depth": int(stats.depth),
        "starting_depth": int(stats.starting_depth or 1),
        "depth_ceiling": int(stats.depth_ceiling or 1),
        "depth_policy": str(stats.depth_policy or "fixed"),
        "mtp_head_cache_policy": str(stats.mtp_head_cache_policy or "loose"),
        "prompt_priming": {
            "source": str(stats.prompt_prime_source or "unprimed"),
            "folded_pairs": int(stats.prompt_primed_pairs or 0),
        },
        "cycles": int(stats.cycles),
        "accepts": int(stats.accepts),
        "rejects": int(stats.rejects),
        "init_emits": int(stats.init_emits),
        "draft_emits": int(stats.draft_emits),
        "bonus_emits": int(stats.bonus_emits),
        "verify_emits": int(stats.verify_emits),
        "drafted_tokens": int(stats.draft_tokens_proposed),
        "accepted_tokens": int(stats.draft_tokens_accepted),
        "acceptance_rate": rate(int(stats.draft_tokens_accepted), int(stats.draft_tokens_proposed)),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "depth_acceptance_rates": {
            f"d{i + 1}": rate(int(stats.accepted_by_depth[i]), int(stats.drafted_by_depth[i]))
            for i in range(3)
        },
        "adaptive_depth_value": dict(stats.adaptive_depth_value),
        "recovery": dict(stats.recovery),
        "seed_ar_step_ms": stats.seed_ar_step_ms,
        "seed_context_tokens": stats.seed_context_tokens,
        "seed_context_source": stats.seed_context_source,
        "forwards": {
            "seed_main": int(stats.seed_main_forwards),
            "verify_main": int(stats.verify_main_forwards),
            "replay_main": 0,
            "mtp": int(stats.mtp_forwards),
        },
        "timings_ms": timings,
        "cache_lifecycle": native_mtp_cache_lifecycle_snapshot(
            head_cache=stats.mtp_head_cache,
            recreated_on_rejects=stats.mtp_cache_recreated_on_rejects,
            retained_on_rejects=stats.mtp_cache_retained_on_rejects,
        ),
        "profiled_phase_timing": PROFILED,
        "published_at": time.time(),
        "fallback_reason": stats.fallback_reason,
        "fallback_cost": {
            "cost_ratio": stats.fallback_cost_ratio,
            "mtp_ms_per_token": stats.fallback_mtp_ms_per_token,
            "ar_step_ms": stats.fallback_ar_step_ms,
        },
    }


def publish(uid: Any, stats: LaneStats, finish_reason: str) -> dict:
    global _last
    record = build_record(uid, stats, finish_reason)
    growth = {
        "cycles": stats.cycles,
        "drafted_tokens": stats.draft_tokens_proposed,
        "accepted_tokens": stats.draft_tokens_accepted,
        "mtp_cache_recreated_on_rejects": stats.mtp_cache_recreated_on_rejects,
        "mtp_cache_retained_on_rejects": stats.mtp_cache_retained_on_rejects,
    }
    with _lock:
        _last = record
        if not stats.request_counted:
            TOTALS["requests"] += 1
            stats.request_counted = True
        for key, value in growth.items():
            value = int(value)
            seen = stats.totals_published.get(key, 0)
            TOTALS[key] += max(0, value - seen)
            stats.totals_published[key] = max(seen, value)
    return record


def record_skip(skip: dict) -> None:
    """Publish why a request did not use the lane (read by the panel)."""
    global _last_skip
    with _lock:
        _last_skip = skip


def last_record() -> Optional[dict]:
    return _last


def native_mtp_stats_snapshot() -> dict:
    """Copies of the last record, the last skip and process totals."""
    with _lock:
        totals = dict(TOTALS)
        totals["acceptance_rate"] = rate(int(totals["accepted_tokens"]), int(totals["drafted_tokens"]))
        return {
            "last_native_mtp": copy.deepcopy(_last),
            "last_native_mtp_skip": copy.deepcopy(_last_skip),
            "native_mtp_totals": totals,
        }


def finish_record(uid: Any, stats: LaneStats, finish_reason: str, head_cache: Any = None) -> None:
    """Snapshot the head cache, publish, and log one summary line.

    The head-cache scan is bounded but not free; this is its only caller so it
    never runs inside a draft/verify cycle.
    """
    stats.mtp_head_cache = native_mtp_cache_snapshot(head_cache)
    record = publish(uid, stats, finish_reason)
    emitted = stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    cycle_rate = f"{stats.accepts / stats.cycles * 100:.1f}%" if stats.cycles > 0 else "n/a"
    token_rate = record["acceptance_rate"]
    token_rate = f"{token_rate * 100:.1f}%" if token_rate is not None else "n/a"
    logger.info(
        "MTP[%s] finish=%s depth=%d tokens=%d cycles=%d full-accept=%d/%d (%s) "
        "draft-tokens=%d/%d (%s) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        stats.depth,
        emitted,
        stats.cycles,
        stats.accepts,
        stats.cycles,
        cycle_rate,
        stats.draft_tokens_accepted,
        stats.draft_tokens_proposed,
        token_rate,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )

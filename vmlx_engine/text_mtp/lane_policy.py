# SPDX-License-Identifier: Apache-2.0
"""Depth, activation and AR-fallback decisions for the text MTP lane.

Adaptive owns draft depth. Bundle stamps, tuning sidecars and family defaults
only pick where a request starts; the adaptive value controller
(``native_mtp_adaptive``) moves between rungs from measured wall time, and the
AR safety valve (``native_mtp_ar_safety``) steps down or parks the request in
plain AR when speculation loses. Only an explicit ``*_NATIVE_MTP_DEPTH`` env
value pins depth, as a diagnostic lever.

Every tunable reads ``VMLINUX_NATIVE_MTP_*`` first, then ``VMLX_NATIVE_MTP_*``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Tuple

from ..native_mtp_adaptive import (
    adaptive_value_snapshot,
    arm_depth_cycle,
    choose_depth_by_value,
    finish_armed_depth_cycle,
)
from ..native_mtp_ar_safety import ar_safety_step
from .lane_stats import LaneStats

logger = logging.getLogger(__name__)

_FALSE = {"0", "false", "no", "off"}


def _first_set(names) -> Optional[str]:
    for name in names:
        if name in os.environ:
            return os.environ.get(name)
    return None


def _pair(suffix: str) -> Tuple[str, str]:
    return f"VMLINUX_NATIVE_MTP_{suffix}", f"VMLX_NATIVE_MTP_{suffix}"


def env_int(default: int, *names: str, minimum: int = 1) -> int:
    raw = _first_set(names)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(int(minimum), value)


def env_float(default: float, *names: str) -> float:
    raw = _first_set(names)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def env_flag(default: bool, *names: str) -> bool:
    raw = _first_set(names)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSE


def adaptive_enabled() -> bool:
    raw = os.environ.get(
        "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
        os.environ.get("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "1"),
    )
    return raw.strip().lower() not in _FALSE


def _explicit_env_source(source: Any) -> bool:
    return str(source or "").startswith("VML")


def resolve_depth(batch: Any, *, adaptive_ceiling: bool = False) -> Tuple[int, str]:
    """Return ``(depth, source)`` for this request's draft chain.

    ``source`` names who chose the depth (env lever, tuning sidecar, bundle
    stamp, default). With ``adaptive_ceiling`` an advisory source only seeds the
    start, so the ceiling widens to the runtime maximum. Depth above 1 needs a
    backbone whose every layer can drop an arbitrary rejected suffix: either a
    partial-rollback cache or a trimmable cache with no single-snapshot restore.
    """
    try:
        from vmlx_engine.native_mtp import native_mtp_effective_depth, native_mtp_max_depth

        depth, source = native_mtp_effective_depth(None)
        if (
            adaptive_ceiling
            and adaptive_enabled()
            and not _explicit_env_source(source)
            and source != "resolution_error"
        ):
            depth = native_mtp_max_depth()
    except Exception:
        depth, source = 1, "resolution_error"
    depth = max(1, min(3, int(depth or 1)))
    if depth == 1:
        return 1, source
    for layer in batch.prompt_cache:
        partial = bool(getattr(layer, "supports_partial_rollback", False))
        if partial and callable(getattr(layer, "rollback_speculative", None)):
            continue
        if getattr(layer, "rollback_state", None) is not None:
            return 1, source
        if not (hasattr(layer, "is_trimmable") and layer.is_trimmable()):
            return 1, source
    return depth, source


def resolved_depth(batch: Any) -> int:
    return resolve_depth(batch)[0]


def activation_decision(batch: Any) -> Tuple[bool, int, str]:
    """Decide whether this request may seed the lane at all.

    The text lane has no per-workload profile store, so under Adaptive an unseen
    model stays AR until it has measured evidence: a validated
    ``vmlx_mtp_tuning.json``. An explicit env depth or a fixed policy is the
    user's own choice and always runs.
    """
    depth, source = resolve_depth(batch)
    if not adaptive_enabled():
        return True, depth, "fixed_policy"
    if _explicit_env_source(source) or "vmlx_mtp_tuning.json" in str(source):
        return True, depth, str(source)
    return False, depth, "adaptive_unseen_ar"


def confirmed_cycle_tokens(stats: LaneStats) -> int:
    # Every finished cycle confirms its accepted drafts plus one token. The
    # seed's two tokens are setup, not cycle output.
    return max(0, int(stats.cycles)) + max(0, int(stats.draft_tokens_accepted))


def cost_ratio(
    stats: LaneStats, ar_step_ms: float, *, elapsed_ms: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """``(lane ms per confirmed token / AR ms per token, lane ms per token)``."""
    if ar_step_ms <= 0.0:
        return None
    spent = (
        float(elapsed_ms)
        if elapsed_ms is not None and float(elapsed_ms) > 0.0
        else stats.phase_total_ms()
    )
    confirmed = confirmed_cycle_tokens(stats)
    if spent <= 0.0 or confirmed <= 0:
        return None
    per_token = spent / confirmed
    return per_token / ar_step_ms, per_token


def _park(lane: Any, reason: str, *, mtp_ms: Any, ar_ms: Any, ratio: Any = None) -> None:
    lane.ar_fallback_pending = True
    lane.ar_fallback_reason = reason
    lane.stats.fallback_reason = reason
    if ratio is not None:
        lane.stats.fallback_cost_ratio = ratio
    lane.stats.fallback_mtp_ms_per_token = mtp_ms
    lane.stats.fallback_ar_step_ms = ar_ms


def apply_safety_valve(label: str, lane: Any) -> bool:
    """AR safety valve for every depth policy (fixed D1-D3 included).

    A fixed depth says how much to speculate, never "keep losing to AR". A trip
    above D1 steps down one rung and judges it on a fresh window; a trip at D1
    parks the request in AR (True). An AR re-entry probe must also beat AR on
    its whole cost, seed and priming included.
    """
    if lane.ar_fallback_pending:
        return False
    stats = lane.stats
    cycles = int(stats.cycles)
    reentry_probe = lane.recovery_probe_started > 0.0
    value = lane.adaptive_value
    origin = int(value.active_probe_origin or 0)
    target = int(value.active_probe_target or 0)
    # A neighbour trial finishes in eight samples, before settled-phase warmup
    # would let the valve judge it, so it borrows the probe window (without
    # the stricter re-entry margin, which belongs to AR re-entry alone).
    neighbour_probe = origin > 0 and target == lane.depth and abs(target - origin) == 1
    measured = lane.recovery is not None and lane.recovery.ar_ms > 0.0
    if lane.recovery is not None:
        stats.recovery = lane.recovery.snapshot()
    trip = ar_safety_step(
        lane.ar_safety,
        cycles=cycles,
        emitted=cycles + int(stats.draft_tokens_accepted),
        now=time.perf_counter(),
        seed_ar_ms=float(lane.ar_step_ms or 0.0),
        primed=str(stats.prompt_prime_source or "unprimed") != "unprimed",
        probe=reentry_probe or neighbour_probe,
        scale_context=not measured,
        baseline_measured=measured,
        margin=0.9 if reentry_probe else None,
    )
    if reentry_probe and trip is None and cycles >= 12:
        confirmed = 2 + cycles + int(stats.draft_tokens_accepted)
        per_token = (time.perf_counter() - lane.recovery_probe_started) * 1000 / confirmed
        if per_token <= lane.ar_step_ms * 0.9:
            lane.recovery_probe_started = 0.0
            logger.info(
                "MTP[%s] AR re-entry D1 accepted: total_ms_per_token=%.3f ar=%.3f",
                label, per_token, lane.ar_step_ms,
            )
        else:
            _park(lane, "reentry_total_cost", mtp_ms=per_token, ar_ms=lane.ar_step_ms)
            return True
    if trip is None:
        return False
    was = int(lane.depth or 1)
    if was > 1:
        lane.depth = was - 1
        stats.depth = lane.depth
        lane.ar_safety.reset(cycles)
        value.last_change_cycle = cycles + lane.adaptive_cycle_offset
        value.active_probe_origin = 0
        value.active_probe_target = 0
        logger.info("MTP[%s] %s", label, trip.log_text(was, target_depth=lane.depth))
        return False
    _park(lane, trip.reason(was), mtp_ms=trip.mtp_ms_per_tok, ar_ms=trip.ar_baseline)
    logger.info("MTP[%s] %s", label, trip.log_text(was))
    return True


def apply_cost_gate(label: str, lane: Any, *, now: float) -> bool:
    """Optional measured MTP -> AR handoff.

    Explicit (``COST_FALLBACK``): a calibrated AR step from the environment,
    any depth policy. Otherwise only an adaptive request, against its own seed
    AR step and the lane's wall time since the first cycle.
    """
    explicit = env_flag(False, *_pair("COST_FALLBACK"))
    runtime = lane.adaptive_enabled and env_flag(True, *_pair("RUNTIME_COST_GATE"))
    if lane.ar_fallback_pending or not (explicit or runtime):
        return False
    if explicit:
        ar_ms = env_float(0.0, *_pair("AR_STEP_MS"), *_pair("COST_AR_STEP_MS"))
        threshold = env_float(1.0, *_pair("COST_RATIO_THRESHOLD"))
        min_cycles = env_int(8, *_pair("COST_MIN_CYCLES"), minimum=1)
        elapsed = None
        kind = "calibrated_cost"
    else:
        ar_ms = float(lane.ar_step_ms or 0.0)
        threshold = env_float(1.25, *_pair("RUNTIME_COST_MARGIN"))
        min_cycles = env_int(48, *_pair("RUNTIME_COST_MIN_CYCLES"), minimum=8)
        elapsed = (
            max(0.0, (float(now) - float(lane.cycle_span_start)) * 1000.0)
            if lane.cycle_span_start > 0.0
            else None
        )
        kind = "runtime_cost"
    if int(lane.stats.cycles) < min_cycles:
        return False
    measured = cost_ratio(lane.stats, ar_ms, elapsed_ms=elapsed)
    if measured is None or measured[0] < threshold:
        return False
    ratio, per_token = measured
    reason = (
        f"{kind} cost_ratio={ratio:.3f}>=threshold={threshold:.3f} "
        f"mtp_ms_per_token={per_token:.2f} ar_step_ms={ar_ms:.2f}"
    )
    _park(lane, reason, mtp_ms=per_token, ar_ms=ar_ms, ratio=ratio)
    logger.info(
        "MTP[%s] adaptive runtime -> AR after cycles=%d: %s",
        label, lane.stats.cycles, reason,
    )
    return True


def value_min_samples() -> int:
    return env_int(8, *_pair("VALUE_MIN_SAMPLES"), minimum=2)


def arm_cycle(lane: Any, *, now: float) -> None:
    arm_depth_cycle(lane.adaptive_value, depth=lane.depth, now=now)


def settle_cycle(label: str, lane: Any, *, completed_depth: int, accepted: int, now: float) -> None:
    """Book a finished cycle and let the value controller pick the next depth.

    The finished cycle keeps its depth; a change applies to the next chain,
    which holds no speculative state yet. A safety step-down this cycle, or an
    AR re-entry still qualifying at D1, pre-empts any value decision.
    """
    cycle = int(lane.stats.cycles) + lane.adaptive_cycle_offset
    finish_armed_depth_cycle(
        lane.adaptive_value,
        depth=completed_depth,
        accepted_drafts=accepted,
        cycle=cycle,
        now=now,
        window=env_int(16, *_pair("VALUE_WINDOW"), minimum=4),
    )
    if lane.depth != completed_depth or lane.recovery_probe_started > 0.0:
        return
    min_samples = value_min_samples()
    decision = choose_depth_by_value(
        lane.adaptive_value,
        current_depth=lane.depth,
        depth_ceiling=lane.depth_ceiling,
        cycle=cycle,
        minimum_samples=min_samples,
        cooldown_cycles=env_int(8, *_pair("VALUE_COOLDOWN_CYCLES"), minimum=2),
        probe_interval_cycles=env_int(48, *_pair("VALUE_PROBE_INTERVAL_CYCLES"), minimum=4),
        hysteresis=env_float(0.05, *_pair("VALUE_HYSTERESIS")),
        raise_min_acceptance=env_float(0.88, *_pair("VALUE_RAISE_MIN_ACCEPT")),
    )
    lane.stats.adaptive_depth_value = adaptive_value_snapshot(
        lane.adaptive_value, minimum_samples=min_samples
    )
    if decision is None:
        return
    before = lane.depth
    lane.depth = max(1, min(lane.depth_ceiling, int(decision.target_depth)))
    lane.stats.depth = lane.depth
    if lane.depth != before:
        lane.ar_safety.reset(int(lane.stats.cycles))
    logger.info(
        "MTP[%s] adaptive value %s D%d -> D%d after cycles=%d: %s",
        label, decision.event, before, lane.depth, lane.stats.cycles, decision.reason,
    )

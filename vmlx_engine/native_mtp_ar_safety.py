# SPDX-License-Identifier: Apache-2.0
"""Policy-independent on-the-fly MTP -> AR safety governor (shared by lanes).

Both native-MTP generators (the multimodal lane in ``mllm_batch_generator``
and the text lane in ``patches/mlx_lm_mtp/batch_generator``) run this every
verify cycle, for FIXED and ADAPTIVE depth alike.  The depth-adapt gates are
what a fixed policy disables; this valve is not a depth policy, it is the
guarantee that a request never keeps speculating while losing to plain AR.

Mechanism (wall-clock + counters only; the one sync added is per request,
before the seed timer):
  * one wall-clock timestamp per cycle from cycle 1; per-cycle wall = dt,
    per-cycle ms/tok = dt / d_emitted.  MTP cost = wall ms/tok, what the
    user experiences.
  * anchor = MEDIAN per-cycle wall over the warmup cycles (same conditions
    as the AR baseline measurement, within a fraction of a second).
  * scale = max(1, median cycle wall now / anchor): context and thermal
    move AR and the MTP cycle together, and the baseline is a FLOOR — the
    first cycles after a large prefill run slow (measured 2026-09-04: anchor
    43ms vs 32ms steady at 1.2k-26k context), so a scale allowed below 1
    demoted healthy MTP three times in a 22-turn run.  Long context grows
    the cycle wall by the same absolute amount it grows AR, i.e. by a
    smaller ratio, so the scale under-estimates AR growth: the valve can
    demote a little early at long context, never late.  (Native MTP in the
    multimodal lane is single-row — admission stays closed while an MTP row
    is active — so concurrency never enters this measurement.)
  * baseline = the request's measured AR ms/tok once it has run an AR tier
    (true AR at the live context), else the seed AR step.
  * trip iff window-mean ms/tok AND per-cycle MEDIAN ms/tok exceed
    baseline x scale x margin.  Acceptance collapse raises ms/tok without
    changing the cycle wall, so it is never masked; a single stalled cycle
    cannot trip (median guard).  Demotion is reversible: the lane's AR tier
    re-probes MTP with exponential backoff (see mllm_batch_generator).
  * skip while cycles < warmup (longer for an unprimed head) or the window is
    not full.

Accepted blind spot: an MTP-only per-cycle slowdown that leaves AR untouched
(e.g. draft-head page faults) raises the scale with it and is caught only
once acceptance suffers.  Hypothetical; no such case was measured.

The seed is ONE synchronized AR step measured right after prefill.  Measured
2026-09-05 on Flash-Next 4S at 1.7k context: seed 22.0 ms/tok against a true
AR rate of 24.7 ms/tok (40.5 tok/s), i.e. the seed read 12% fast and demoted
a D3 window that was running at 23.9 ms/tok -- faster than real AR.  One
sample is not a baseline, so while the baseline is the seed the trip margin
is ``DEFAULT_SEED_MARGIN`` (the seed's observed error band); once the request
has a MEASURED AR (or measured D1) baseline the margin is ``DEFAULT_MARGIN``.

Known limitation: the first request after a model load measures an inflated
seed (compile / cold pages), so the valve is lax for that one request; the
multimodal lane also never lets that request's tier seed the next request's
start rung.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Measured 2026-09-04 on Flash-Next 4S: primed requests (cold prompt AND
# restored prefix — priming folds the restored prefix too) run healthy from
# the first cycle (15-24 ms/tok in the first window), so no restored-prefix
# multiplier is needed for the valve; an UNPRIMED head starts cold and gets a
# longer warmup.  8+8 => first judgment at cycle 16 (~0.65s at D3 on 4S).
DEFAULT_WARMUP_CYCLES = 8
DEFAULT_UNPRIMED_WARMUP_CYCLES = 16
DEFAULT_WINDOW_CYCLES = 8
# "Never below AR": trip as soon as the windowed cost exceeds the AR baseline.
# The margin used to be 1.25, which let MTP run up to 25% slower than AR
# before demotion — measured on Flash-Next 4M as whole turns at ~25 tok/s
# against a ~35 tok/s AR step. Stall robustness comes from the median guard,
# hysteresis from the re-entry/promotion probes (which must WIN by 10%).
DEFAULT_MARGIN = 1.0
# Margin while the baseline is the single-sample seed (see module docstring).
DEFAULT_SEED_MARGIN = 1.10
DEFAULT_PROBE_WARMUP_CYCLES = 4
PROBE_EARLY_ABORT_RATIO = 1.5


def _env_raw(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def env_flag(default: bool, *names: str) -> bool:
    raw = _env_raw(*names)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_int(default: int, *names: str, minimum: int = 1) -> int:
    raw = _env_raw(*names)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def env_float(default: float, *names: str) -> float:
    raw = _env_raw(*names)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def ar_safety_enabled() -> bool:
    return env_flag(True, "VMLINUX_NATIVE_MTP_AR_SAFETY", "VMLX_NATIVE_MTP_AR_SAFETY")


def ar_safety_warmup_cycles(*, primed: bool = True) -> int:
    default = DEFAULT_WARMUP_CYCLES if primed else DEFAULT_UNPRIMED_WARMUP_CYCLES
    return env_int(
        default,
        "VMLINUX_NATIVE_MTP_AR_SAFETY_WARMUP",
        "VMLX_NATIVE_MTP_AR_SAFETY_WARMUP",
        minimum=1,
    )


def ar_safety_window_cycles() -> int:
    return env_int(
        DEFAULT_WINDOW_CYCLES,
        "VMLINUX_NATIVE_MTP_AR_SAFETY_WINDOW",
        "VMLX_NATIVE_MTP_AR_SAFETY_WINDOW",
        minimum=4,
    )


def ar_safety_margin() -> float:
    return env_float(
        DEFAULT_MARGIN,
        "VMLINUX_NATIVE_MTP_RUNTIME_COST_MARGIN",
        "VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN",
    )


def ar_safety_seed_margin() -> float:
    return max(
        ar_safety_margin(),
        env_float(
            DEFAULT_SEED_MARGIN,
            "VMLINUX_NATIVE_MTP_SEED_COST_MARGIN",
            "VMLX_NATIVE_MTP_SEED_COST_MARGIN",
        ),
    )


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    return ordered[len(ordered) // 2]


def windowed_ar_verdict(
    *,
    ar_step_ms: float,
    anchor_cycle_ms: float,
    cur_cycle_ms: float,
    context_ratio: float,
    delta_emitted: int,
    delta_wall_ms: float,
    margin: float,
    per_cycle_ms_per_tok: Optional[Sequence[float]] = None,
) -> Optional[Tuple[float, float]]:
    """Pure windowed MTP-vs-AR decision. Returns (mtp_ms_per_tok, ar_baseline)
    when MTP is slower than the context-scaled AR baseline by the margin,
    else None.  Unit-testable without a model."""
    if ar_step_ms <= 0.0:
        return None
    if delta_emitted <= 0 or delta_wall_ms <= 0.0:
        return None
    mtp_ms_per_tok = delta_wall_ms / delta_emitted
    scale = 1.0
    if anchor_cycle_ms > 0.0 and cur_cycle_ms > 0.0:
        # Cycle-wall growth (context, thermal); the baseline is a floor.
        # ``context_ratio`` is informational.
        scale = max(1.0, cur_cycle_ms / anchor_cycle_ms)
    ar_baseline = ar_step_ms * scale
    threshold = ar_baseline * margin
    if mtp_ms_per_tok <= threshold:
        return None
    if per_cycle_ms_per_tok:
        if median(per_cycle_ms_per_tok) <= threshold:
            return None
    return mtp_ms_per_tok, ar_baseline


@dataclass
class ArSafetyState:
    """Per-request governor state, embedded in each lane's MTP state."""

    anchor_cycle_ms: float = 0.0  # median per-cycle wall over warmup
    anchor_context_tokens: int = 0
    prompt_tokens: int = 0
    last_cycle_t: float = 0.0
    cycle_base: int = 0  # cycles are counted from the last tier change
    warmup_cycle_ms: List[float] = field(default_factory=list)
    # (cycles, emitted, wall perf_counter)
    ring: List[Tuple[int, int, float]] = field(default_factory=list)

    def reset(self, cycle_base: int) -> None:
        """Start a fresh judgment window after a tier change (D3->D1, probe,
        promotion): the anchor and window of one depth say nothing about
        another."""
        self.anchor_cycle_ms = 0.0
        self.anchor_context_tokens = 0
        self.last_cycle_t = 0.0
        self.cycle_base = int(cycle_base)
        self.warmup_cycle_ms = []
        self.ring = []


@dataclass(frozen=True)
class ArSafetyTrip:
    cycles: int
    mtp_ms_per_tok: float
    ar_baseline: float
    seed_ar_ms: float
    margin: float
    window: int
    cycle_median_ms_per_tok: float
    cycle_max_ms_per_tok: float
    anchor_cycle_ms: float
    cur_cycle_ms: float
    anchor_context_tokens: int
    context_now: int

    def reason(self, prior_depth: int) -> str:
        return (
            f"windowed_ar_safety d{prior_depth} "
            f"mtp_ms_per_tok={self.mtp_ms_per_tok:.1f}"
            f">ar_baseline={self.ar_baseline:.1f}(seed={self.seed_ar_ms:.1f})"
            f"x{self.margin:.2f}"
        )

    def log_text(
        self, prior_depth: int, *, target_depth: int = 0, baseline_label: str = "AR",
    ) -> str:
        target = f"D{target_depth}" if target_depth > 0 else "AR"
        return (
            f"windowed AR safety D{prior_depth} -> {target} at cycle={self.cycles} "
            f"({self.mtp_ms_per_tok:.1f}ms/tok vs ctx-scaled {baseline_label} "
            f"{self.ar_baseline:.1f}ms, seed {self.seed_ar_ms:.1f}ms, "
            f"window={self.window}, cycle median "
            f"{self.cycle_median_ms_per_tok:.1f} max "
            f"{self.cycle_max_ms_per_tok:.1f} ms/tok, cycle wall anchor "
            f"{self.anchor_cycle_ms:.1f} -> {self.cur_cycle_ms:.1f} ms, "
            f"context {self.anchor_context_tokens} -> {self.context_now})"
        )


def ar_safety_step(
    st: ArSafetyState,
    *,
    cycles: int,
    emitted: int,
    now: float,
    seed_ar_ms: float,
    primed: bool = True,
    margin: Optional[float] = None,
    probe: bool = False,
    scale_context: bool = True,
    baseline_measured: bool = False,
) -> Optional[ArSafetyTrip]:
    """Record this cycle's sample and decide.  Call once per verify cycle,
    AFTER the cycle's device round-trip, BEFORE any depth adaptation.  The
    caller owns the demotion and logging.

    ``probe``: a re-entry/promotion probe is judged FAST — 4 warmup cycles,
    and a clear loser (mean cost above 1.5x the baseline once 4 judged
    cycles exist) is aborted before the window fills, so a losing probe
    costs a handful of cycles instead of 24.

    ``baseline_measured``: ``seed_ar_ms`` is a measured AR/D1 cost over many
    tokens (margin ``DEFAULT_MARGIN``) rather than the one-step seed (margin
    ``DEFAULT_SEED_MARGIN``)."""
    if seed_ar_ms <= 0.0:
        return None
    if not ar_safety_enabled():
        return None
    cycles = int(cycles) - int(st.cycle_base)
    warmup = DEFAULT_PROBE_WARMUP_CYCLES if probe else ar_safety_warmup_cycles(primed=primed)
    if cycles <= warmup:
        # Warmup: collect per-cycle walls for the anchor.  Cycle 1's dt needs
        # a previous stamp, so the first call only stamps.
        if st.last_cycle_t > 0.0:
            st.warmup_cycle_ms.append((float(now) - st.last_cycle_t) * 1000.0)
        st.last_cycle_t = float(now)
        if cycles == warmup and st.anchor_cycle_ms <= 0.0:
            anchor = median(st.warmup_cycle_ms)
            if anchor > 0.0:
                st.anchor_cycle_ms = anchor
                st.anchor_context_tokens = max(1, int(st.prompt_tokens) + int(emitted))
            st.ring.append((int(cycles), int(emitted), float(now)))
        return None
    window = ar_safety_window_cycles()
    ring = st.ring
    ring.append((int(cycles), int(emitted), float(now)))
    if len(ring) > window + 1:
        del ring[0 : len(ring) - (window + 1)]
    if margin is None:
        margin = ar_safety_margin() if baseline_measured else ar_safety_seed_margin()
    if len(ring) <= window:
        if probe and len(ring) >= 5:
            # Early abort for a clearly losing probe.
            c0, e0, t0 = ring[0]
            d_e = int(emitted) - e0
            d_ms = (float(now) - t0) * 1000.0
            if d_e > 0 and d_ms / d_e > float(seed_ar_ms) * PROBE_EARLY_ABORT_RATIO:
                per_tok = [((b[2] - a[2]) * 1000.0) / (b[1] - a[1]) for a, b in zip(ring, ring[1:]) if b[1] > a[1]]
                return ArSafetyTrip(
                    cycles=int(cycles) + int(st.cycle_base), mtp_ms_per_tok=d_ms / d_e,
                    ar_baseline=float(seed_ar_ms), seed_ar_ms=float(seed_ar_ms),
                    margin=float(margin), window=len(ring) - 1,
                    cycle_median_ms_per_tok=median(per_tok),
                    cycle_max_ms_per_tok=max(per_tok) if per_tok else 0.0,
                    anchor_cycle_ms=0.0, cur_cycle_ms=0.0,
                    anchor_context_tokens=int(st.prompt_tokens) + e0,
                    context_now=int(st.prompt_tokens) + int(emitted),
                )
        return None  # window not full yet — never judge a cold window

    c0, e0, t0 = ring[0]
    per_cycle_ms: List[float] = []
    per_cycle_ms_per_tok: List[float] = []
    for prev, cur in zip(ring, ring[1:]):
        d_ms = (cur[2] - prev[2]) * 1000.0
        d_e = cur[1] - prev[1]
        per_cycle_ms.append(d_ms)
        if d_e > 0:
            per_cycle_ms_per_tok.append(d_ms / d_e)
    cur_cycle_ms = median(per_cycle_ms)
    context_now = int(st.prompt_tokens) + int(emitted)
    if st.anchor_cycle_ms <= 0.0:
        # Warmup produced no usable anchor (e.g. warmup 1); anchor on the
        # first full window instead.
        if cur_cycle_ms <= 0.0:
            return None
        st.anchor_cycle_ms = cur_cycle_ms
        st.anchor_context_tokens = max(1, context_now)
    context_ratio = context_now / max(1, st.anchor_context_tokens)
    verdict = windowed_ar_verdict(
        ar_step_ms=float(seed_ar_ms),
        anchor_cycle_ms=float(st.anchor_cycle_ms) if scale_context else 0.0,
        cur_cycle_ms=cur_cycle_ms,
        context_ratio=context_ratio,
        delta_emitted=int(emitted) - e0,
        delta_wall_ms=(float(now) - t0) * 1000.0,
        margin=margin,
        per_cycle_ms_per_tok=per_cycle_ms_per_tok,
    )
    if verdict is None:
        return None
    mtp_ms_per_tok, ar_baseline = verdict
    return ArSafetyTrip(
        cycles=int(cycles) + int(st.cycle_base),
        mtp_ms_per_tok=mtp_ms_per_tok,
        ar_baseline=ar_baseline,
        seed_ar_ms=float(seed_ar_ms),
        margin=margin,
        window=window,
        cycle_median_ms_per_tok=median(per_cycle_ms_per_tok),
        cycle_max_ms_per_tok=(
            max(per_cycle_ms_per_tok) if per_cycle_ms_per_tok else 0.0
        ),
        anchor_cycle_ms=float(st.anchor_cycle_ms),
        cur_cycle_ms=cur_cycle_ms,
        anchor_context_tokens=int(st.anchor_context_tokens),
        context_now=context_now,
    )

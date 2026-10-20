"""Session-scoped validated depth profiles for adaptive native MTP.

The per-request wall-value controller (native_mtp_adaptive) starts every
request at the configured depth and needs tens of verify cycles before its
runtime-cost and acceptance gates can conclude that MTP is slower than AR.
Because ``MLLMNativeMTPState`` is request-local, every request repeated that
whole failed experiment: short and first turns finished before the controller
learned anything, and later turns relearned the same lesson from scratch.

Contract (audited 2026-08-30):

- An UNKNOWN profile stays AR unless the owning runtime explicitly supplies a
  measured family cold-start depth.  That opt-in is intentionally separate
  from the generic profile policy: it is allowed only after live AR/D1/D2/D3
  evidence shows the family's native head wins on first-turn work.  The
  request-local value controller still measures and may demote that seed.
- MTP may seed immediately only from MEASURED evidence: either a validated
  model-local tuning record (vmlx_mtp_tuning.json best_depth, resolved by
  native_mtp_effective_depth and flagged by the caller) or a profile entry
  this store learned from completed requests whose measured per-depth
  wall value beat the same requests' own AR baseline.
- Learning requires completed requests (stop/length) with at least
  ``MIN_WALL_SAMPLES`` wall-clock samples at the winning depth. Cancelled,
  errored and sample-starved requests teach nothing. Acceptance rate alone
  never promotes — only confirmed-tokens-per-wall-second versus the AR
  baseline, with a hysteresis margin.
- The store lives on the generator instance, so a model/bundle/config reload
  or a new engine process starts clean. Cross-process persistence of
  measured depth remains the tuning-sidecar mechanism.
- Keys separate sampler class, restored-prefix, context bucket, and
  tools-present so materially different workloads cannot poison one another.
- Nothing here changes sampling, verification, or emitted tokens: it only
  chooses whether MTP activates and at what STARTING depth; the existing
  controller and safety gates still adjust with their own measurements.
"""

from __future__ import annotations

from .native_mtp import native_mtp_max_depth

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Wall-clock samples required at a depth before its value is trusted.
MIN_WALL_SAMPLES = 3
# Measured depth value must beat the AR baseline by this margin to be
# learned as a profitable seed.
PROMOTE_MARGIN = 0.05
# An AR-wins verdict is adaptive, not a permanent lock: after this many
# seconds it expires and the next request re-validates with a single
# bounded D1 probe (near-AR cost; the controller's cost/acceptance gates
# demote it quickly if AR still wins).
AR_VERDICT_TTL_S = 300.0

_CTX_BUCKETS = ((2048, "short"), (16384, "medium"))

ProfileKey = Tuple[Any, ...]


def profile_key(
    *,
    temperature: float,
    restored_prefix: bool,
    prompt_tokens: int,
    has_tools: bool = False,
    presence_penalty: float = 0.0,
) -> ProfileKey:
    sampler_class = "greedy" if float(temperature or 0.0) <= 0.0 else "sampled"
    bucket = "long"
    for limit, name in _CTX_BUCKETS:
        if int(prompt_tokens or 0) < limit:
            bucket = name
            break
    key = (sampler_class, bool(restored_prefix), bucket, bool(has_tools))
    # Preserve existing neutral profiles, while a penalized target learns its
    # own acceptance/cost history instead of borrowing a different distribution.
    return key + (float(presence_penalty),) if presence_penalty else key


@dataclass
class NativeMTPSessionProfile:
    """Validated starting depth for one workload shape.

    learned_depth: None = unknown (AR until measured evidence exists);
    0 = measured AR-wins; 1..3 = measured profitable depth.
    """

    learned_depth: Optional[int] = None
    requests_observed: int = 0
    last_finish_reason: str = ""
    last_fallback_reason: Optional[str] = None
    last_values_tok_s: Dict[str, Any] = field(default_factory=dict)
    last_ar_baseline_tps: Optional[float] = None
    ar_verdict_at: float = 0.0


class NativeMTPProfileStore:
    """Per-generator (per loaded model, per process) profile registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: Dict[ProfileKey, NativeMTPSessionProfile] = {}

    def start_depth(
        self,
        key: ProfileKey,
        *,
        configured_depth: int,
        capability_ceiling: int = 3,
        tuning_validated: bool = False,
        unseen_start_depth: Optional[int] = None,
        unseen_start_source: str = "measured_family_cold_start",
        now: Optional[float] = None,
    ) -> Tuple[int, str]:
        """Return (starting depth, seed source). Depth 0 means stay AR.

        ``tuning_validated`` marks a configured depth that came from a
        MEASURED model-local tuning record — the one advisory source that
        already carries wall-clock proof, so it may seed immediately even
        when this in-process store has no entry yet.

        ``unseen_start_depth`` is an owning-runtime opt-in for a family whose
        cold start has already been measured live.  It never overrides a
        learned profile or validated tuning record.

        An AR-wins verdict is adaptive, not permanent: after
        AR_VERDICT_TTL_S it expires and this request runs a single bounded
        D1 re-probe so a workload shift can re-enable MTP.
        """

        import time as _time

        now = _time.monotonic() if now is None else float(now)
        configured = max(1, min(native_mtp_max_depth(), int(configured_depth or 1)))
        ceiling = max(1, min(native_mtp_max_depth(), int(capability_ceiling or 1)))
        with self._lock:
            profile = self._profiles.get(key)
            if profile is not None and profile.learned_depth is not None:
                if profile.learned_depth <= 0:
                    if now - float(profile.ar_verdict_at or 0.0) >= AR_VERDICT_TTL_S:
                        # Expired: re-validate once at near-AR D1 cost and
                        # re-arm the TTL so back-to-back requests don't all
                        # probe if the verdict is immediately re-confirmed.
                        profile.ar_verdict_at = now
                        return min(1, ceiling), "ar_reprobe_d1"
                    return 0, "profile_ar"
                depth = max(1, min(profile.learned_depth, ceiling))
                return depth, f"profile_validated_d{depth}"
        if tuning_validated:
            depth = min(configured, ceiling)
            return depth, f"tuning_validated_d{depth}"
        if unseen_start_depth is not None:
            depth = max(1, min(int(unseen_start_depth), ceiling))
            source = str(unseen_start_source or "measured_family_cold_start")
            return depth, f"{source}_d{depth}"
        return 0, "unseen_ar"

    def observe(
        self,
        key: ProfileKey,
        *,
        final_depth: int,
        fallback_to_ar: bool,
        fallback_reason: Optional[str],
        finish_reason: str,
        values_tok_s: Optional[Dict[str, Any]] = None,
        sample_counts: Optional[Dict[str, Any]] = None,
        ar_baseline_tps: Optional[float] = None,
    ) -> None:
        """Record what one COMPLETED request measured.

        Only completed requests (stop/length) teach anything. A fallback to
        AR with real samples records AR-wins. A profitable depth is learned
        only when its measured wall value beats the request's own AR
        baseline by PROMOTE_MARGIN with MIN_WALL_SAMPLES samples.

        ``final_depth`` is telemetry, not the winner: a response can terminate
        during a bounded neighbor probe. Learning that transient target made a
        D3-winning request seed D2 on its next turn. Select the best qualified
        measured depth across the request instead. Sample-starved requests
        leave the profile unchanged.
        """

        if finish_reason not in ("stop", "length", "fallback_to_ar"):
            return
        values = values_tok_s or {}
        counts = sample_counts or {}
        with self._lock:
            profile = self._profiles.setdefault(key, NativeMTPSessionProfile())
            profile.requests_observed += 1
            profile.last_finish_reason = str(finish_reason)
            profile.last_fallback_reason = fallback_reason
            profile.last_values_tok_s = dict(values)
            profile.last_ar_baseline_tps = ar_baseline_tps

            if fallback_to_ar or finish_reason == "fallback_to_ar":
                import time as _time

                profile.learned_depth = 0
                profile.ar_verdict_at = _time.monotonic()
                return

            qualified: list[tuple[float, int]] = []
            for depth in range(1, native_mtp_max_depth() + 1):
                label = f"d{depth}"
                value = values.get(label)
                n = int(counts.get(label) or 0)
                if value is None or n < MIN_WALL_SAMPLES:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric_value > 0.0:
                    qualified.append((numeric_value, depth))
            if not qualified:
                # Not enough measured evidence — profile stays as it was.
                return
            if ar_baseline_tps and ar_baseline_tps > 0:
                best_value, best_depth = max(qualified)
                if best_value >= float(ar_baseline_tps) * (1.0 + PROMOTE_MARGIN):
                    profile.learned_depth = best_depth
                else:
                    import time as _time

                    profile.learned_depth = 0
                    profile.ar_verdict_at = _time.monotonic()
            # With no AR baseline available, the depth's value alone cannot
            # prove profitability — leave the profile unchanged.

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "|".join((
                    key[0],
                    "restored" if key[1] else "cold",
                    key[2],
                    "tools" if key[3] else "notools",
                )): {
                    "learned_depth": profile.learned_depth,
                    "requests_observed": profile.requests_observed,
                    "last_finish_reason": profile.last_finish_reason,
                    "last_fallback_reason": profile.last_fallback_reason,
                    "last_values_tok_s": dict(profile.last_values_tok_s),
                    "last_ar_baseline_tps": profile.last_ar_baseline_tps,
                }
                for key, profile in self._profiles.items()
            }

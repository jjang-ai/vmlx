from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResearchSamplerSettings:
    """Pure-Python sampler contract for native-MTP research sweeps.

    This mirrors the MLX-LM/vMLX sampler order used by the current runtime:
    top-p, then min-p, then top-k over log probabilities; temperature is applied
    only when sampling from the surviving distribution.
    """

    temperature: float = 0.0
    top_p: float = 0.0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    top_k: int = 0


def _normalize_distribution(values: Iterable[float] | np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("distribution must be a 1D vector")
    if probs.size == 0:
        raise ValueError("distribution must be non-empty")
    if not np.all(np.isfinite(probs)):
        raise ValueError("distribution contains non-finite values")
    if np.any(probs < 0):
        raise ValueError("distribution contains negative values")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("distribution sum must be positive")
    return probs / total


def _renormalize_masked(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, probs, 0.0)
    total = float(masked.sum())
    if total <= 0.0:
        winner = int(np.argmax(probs))
        masked = np.zeros_like(probs)
        masked[winner] = 1.0
        return masked
    return masked / total


def _keep_top_p_mask(probs: np.ndarray, top_p: float) -> np.ndarray:
    if not (0.0 < float(top_p) < 1.0):
        return np.ones_like(probs, dtype=bool)
    order = np.argsort(probs)  # ascending, matching mlx_lm.apply_top_p
    cumulative = np.cumsum(probs[order])
    keep_sorted = cumulative > (1.0 - float(top_p))
    keep = np.zeros_like(probs, dtype=bool)
    keep[order] = keep_sorted
    if not np.any(keep):
        keep[int(np.argmax(probs))] = True
    return keep


def _keep_min_p_mask(
    probs: np.ndarray,
    min_p: float,
    *,
    min_tokens_to_keep: int,
) -> np.ndarray:
    if float(min_p or 0.0) == 0.0:
        return np.ones_like(probs, dtype=bool)
    if not (0.0 <= float(min_p) <= 1.0):
        raise ValueError("min_p must be in [0, 1]")
    keep = probs >= (float(probs.max()) * float(min_p))
    min_keep = max(1, int(min_tokens_to_keep or 1))
    if min_keep > 1:
        top = np.argsort(-probs)[: min(min_keep, probs.size)]
        keep[top] = True
    if not np.any(keep):
        keep[int(np.argmax(probs))] = True
    return keep


def _keep_top_k_mask(probs: np.ndarray, top_k: int) -> np.ndarray:
    k = int(top_k or 0)
    if k <= 0 or k >= probs.size:
        return np.ones_like(probs, dtype=bool)
    keep = np.zeros_like(probs, dtype=bool)
    keep[np.argsort(-probs)[:k]] = True
    return keep


def filter_distribution(
    probabilities: Iterable[float] | np.ndarray,
    settings: ResearchSamplerSettings,
) -> np.ndarray:
    """Return the post-filter distribution used for p/q research math."""

    probs = _normalize_distribution(probabilities)

    mask = _keep_top_p_mask(probs, settings.top_p)
    probs = _renormalize_masked(probs, mask)

    mask = _keep_min_p_mask(
        probs,
        settings.min_p,
        min_tokens_to_keep=settings.min_tokens_to_keep,
    )
    probs = _renormalize_masked(probs, mask)

    mask = _keep_top_k_mask(probs, settings.top_k)
    probs = _renormalize_masked(probs, mask)

    temp = float(settings.temperature or 0.0)
    if temp == 0.0:
        greedy = np.zeros_like(probs)
        greedy[int(np.argmax(probs))] = 1.0
        return greedy
    if temp < 0.0:
        raise ValueError("temperature must be non-negative")
    scaled = np.where(probs > 0.0, probs ** (1.0 / temp), 0.0)
    return _normalize_distribution(scaled)


def distribution_from_logits(
    logits: Iterable[float] | np.ndarray,
    settings: ResearchSamplerSettings,
) -> np.ndarray:
    """Convert logits/logprobs into the post-filter sampling distribution."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logits must be a 1D vector")
    if values.size == 0:
        raise ValueError("logits must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits contains non-finite values")
    shifted = values - float(values.max())
    probs = np.exp(shifted)
    return filter_distribution(probs, settings)


def acceptance_probability(
    target_probabilities: Iterable[float] | np.ndarray,
    draft_probabilities: Iterable[float] | np.ndarray,
    token_id: int,
) -> float:
    """Return min(1, p(x) / q(x)) for a proposed draft token."""

    target = _normalize_distribution(target_probabilities)
    draft = _normalize_distribution(draft_probabilities)
    idx = int(token_id)
    if idx < 0 or idx >= target.size or idx >= draft.size:
        raise IndexError("token_id out of range")
    q = float(draft[idx])
    if q <= 0.0:
        return 0.0
    return round(min(1.0, float(target[idx]) / q), 12)


def residual_distribution(
    target_probabilities: Iterable[float] | np.ndarray,
    draft_probabilities: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return normalized max(0, p - q), falling back to p when p == q."""

    target = _normalize_distribution(target_probabilities)
    draft = _normalize_distribution(draft_probabilities)
    if target.shape != draft.shape:
        raise ValueError("target and draft distributions must have the same shape")
    residual = np.maximum(0.0, target - draft)
    total = float(residual.sum())
    if total <= 0.0:
        return target
    return residual / total


def speculative_output_distribution(
    target_probabilities: Iterable[float] | np.ndarray,
    draft_probabilities: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Compute the marginal output distribution of one speculative step."""

    target = _normalize_distribution(target_probabilities)
    draft = _normalize_distribution(draft_probabilities)
    if target.shape != draft.shape:
        raise ValueError("target and draft distributions must have the same shape")
    accept = np.array(
        [acceptance_probability(target, draft, idx) for idx in range(target.size)],
        dtype=np.float64,
    )
    accepted_mass = draft * accept
    reject_mass = max(0.0, 1.0 - float(accepted_mass.sum()))
    return accepted_mass + reject_mass * residual_distribution(target, draft)


def distribution_delta(
    observed: Iterable[float] | np.ndarray,
    expected: Iterable[float] | np.ndarray,
) -> dict[str, float]:
    obs = _normalize_distribution(observed)
    exp = _normalize_distribution(expected)
    if obs.shape != exp.shape:
        raise ValueError("distributions must have the same shape")
    return {
        "l1": float(np.abs(obs - exp).sum()),
        "max_abs": float(np.max(np.abs(obs - exp))),
    }


def speculative_policy_metrics(
    target_logits: Iterable[float] | np.ndarray,
    draft_logits: Iterable[float] | np.ndarray,
    settings: ResearchSamplerSettings | None = None,
    *,
    target_settings: ResearchSamplerSettings | None = None,
    draft_settings: ResearchSamplerSettings | None = None,
) -> dict[str, object]:
    """Summarize one p/q speculative-sampling policy on a logits pair."""

    if settings is not None:
        target_settings = target_settings or settings
        draft_settings = draft_settings or settings
    target_settings = target_settings or ResearchSamplerSettings()
    draft_settings = draft_settings or target_settings
    target = distribution_from_logits(target_logits, target_settings)
    draft = distribution_from_logits(draft_logits, draft_settings)
    observed = speculative_output_distribution(target, draft)
    expected_acceptance = float(np.minimum(target, draft).sum())
    return {
        "target_settings": _settings_dict(target_settings),
        "draft_settings": _settings_dict(draft_settings),
        "settings": _settings_dict(target_settings),
        "target_distribution": target,
        "draft_distribution": draft,
        "observed_distribution": observed,
        "expected_acceptance_rate": round(expected_acceptance, 12),
        "reject_rate": round(1.0 - expected_acceptance, 12),
        "delta": distribution_delta(observed, target),
    }


def _settings_dict(settings: ResearchSamplerSettings) -> dict[str, float | int]:
    return {
        "temperature": float(settings.temperature),
        "top_p": float(settings.top_p),
        "min_p": float(settings.min_p),
        "min_tokens_to_keep": int(settings.min_tokens_to_keep),
        "top_k": int(settings.top_k),
    }


def _settings_sort_key(settings: dict[str, float | int]) -> tuple[float, float, float, int, int]:
    return (
        float(settings["temperature"]),
        float(settings["top_p"]),
        float(settings["min_p"]),
        int(settings["min_tokens_to_keep"]),
        int(settings["top_k"]),
    )


def sweep_policy_grid(
    target_logits: Iterable[float] | np.ndarray,
    draft_logits: Iterable[float] | np.ndarray,
    settings_grid: Iterable[ResearchSamplerSettings],
) -> list[dict[str, object]]:
    """Rank candidate sampler settings for one target/draft logits pair."""

    rows = [
        speculative_policy_metrics(target_logits, draft_logits, settings)
        for settings in settings_grid
    ]
    rows.sort(
        key=lambda row: (
            -float(row["expected_acceptance_rate"]),
            float(row["delta"]["max_abs"]),
            float(row["delta"]["l1"]),
            _settings_sort_key(row["settings"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def sweep_policy_pairs_cached(
    target_logits_rows: Iterable[Iterable[float] | np.ndarray],
    draft_logits_rows: Iterable[Iterable[float] | np.ndarray],
    settings_pairs: Iterable[tuple[ResearchSamplerSettings, ResearchSamplerSettings]],
    *,
    top_n: int | None = None,
) -> list[dict[str, object]]:
    """Rank target/draft sampler pairs while reusing filtered distributions.

    This is intended for real-logit shadow sweeps, where the expensive work is
    filtering full-vocab logits for many policy pairs. It keeps p/q acceptance
    exact and avoids materializing residual-corrected output distributions in
    the inner loop; residual correction is already covered by the synthetic
    tests and is algebraically exact for these p/q rows.
    """

    target_rows = [np.asarray(row, dtype=np.float64) for row in target_logits_rows]
    draft_rows = [np.asarray(row, dtype=np.float64) for row in draft_logits_rows]
    if len(target_rows) != len(draft_rows):
        raise ValueError("target and draft row counts must match")
    if not target_rows:
        raise ValueError("at least one target/draft row is required")

    pairs = list(settings_pairs)
    if not pairs:
        raise ValueError("settings_pairs must be non-empty")

    target_cache: dict[tuple[int, ResearchSamplerSettings], np.ndarray] = {}
    draft_cache: dict[tuple[int, ResearchSamplerSettings], np.ndarray] = {}

    def cached_distribution(
        cache: dict[tuple[int, ResearchSamplerSettings], np.ndarray],
        rows: list[np.ndarray],
        row_index: int,
        settings: ResearchSamplerSettings,
    ) -> np.ndarray:
        key = (row_index, settings)
        if key not in cache:
            cache[key] = distribution_from_logits(rows[row_index], settings)
        return cache[key]

    ranked = []
    depth_count = len(target_rows)
    for target_settings, draft_settings in pairs:
        depths = []
        mean_acceptance = 0.0
        worst_acceptance = 1.0
        target_support_total = 0
        draft_support_total = 0
        min_target_support: int | None = None
        min_draft_support: int | None = None
        for depth_index in range(depth_count):
            target = cached_distribution(
                target_cache,
                target_rows,
                depth_index,
                target_settings,
            )
            draft = cached_distribution(
                draft_cache,
                draft_rows,
                depth_index,
                draft_settings,
            )
            if target.shape != draft.shape:
                raise ValueError("target and draft distributions must have the same shape")
            acceptance = round(float(np.minimum(target, draft).sum()), 12)
            target_support = int(np.count_nonzero(target))
            draft_support = int(np.count_nonzero(draft))
            depths.append(
                {
                    "depth": depth_index + 1,
                    "expected_acceptance_rate": acceptance,
                    "reject_rate": round(1.0 - acceptance, 12),
                    "delta": {
                        "max_abs": 0.0,
                        "l1": 0.0,
                        "mode": "residual_exact_by_formula",
                    },
                    "target_support": target_support,
                    "draft_support": draft_support,
                }
            )
            mean_acceptance += acceptance
            worst_acceptance = min(worst_acceptance, acceptance)
            target_support_total += target_support
            draft_support_total += draft_support
            min_target_support = (
                target_support
                if min_target_support is None
                else min(min_target_support, target_support)
            )
            min_draft_support = (
                draft_support
                if min_draft_support is None
                else min(min_draft_support, draft_support)
            )

        ranked.append(
            {
                "target_settings": _settings_dict(target_settings),
                "draft_settings": _settings_dict(draft_settings),
                "depths": depths,
                "mean_acceptance": float(mean_acceptance) / depth_count,
                "worst_acceptance": float(worst_acceptance),
                "mean_target_support": float(target_support_total) / depth_count,
                "mean_draft_support": float(draft_support_total) / depth_count,
                "min_target_support": int(min_target_support or 0),
                "min_draft_support": int(min_draft_support or 0),
                "mean_l1_delta": 0.0,
                "worst_max_abs_delta": 0.0,
                "delta": {
                    "max_abs": 0.0,
                    "l1": 0.0,
                    "mode": "residual_exact_by_formula",
                },
            }
        )

    ranked.sort(
        key=lambda row: (
            -float(row["mean_acceptance"]),
            -float(row["worst_acceptance"]),
            float(row["worst_max_abs_delta"]),
            float(row["mean_l1_delta"]),
            _settings_sort_key(row["target_settings"]),
            _settings_sort_key(row["draft_settings"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    if top_n is None:
        return ranked
    return ranked[: max(0, int(top_n))]

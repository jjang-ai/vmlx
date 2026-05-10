# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding math primitives (pure NumPy, no MLX dependency).

References:
- Leviathan et al. 2023, "Fast Inference from Transformers via Speculative
  Decoding" (arxiv:2211.17192).
- Chen et al. 2023, "Accelerating Large Language Model Decoding with
  Speculative Sampling" (arxiv:2302.01318).

Contract: ``p`` is the target distribution, ``q`` is the draft distribution,
both 1D probability vectors over the same vocabulary, both already passed
through identical sampler filters (top_p, top_k, temperature). Sampler-equality
is the caller's responsibility — the math here assumes it.

Correctness oracle (``speculative_output_marginal``) returns the marginal of
the speculative-sampling output and MUST equal ``p`` up to floating-point
tolerance. This is the single property the implementation must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SpeculativeDecision:
    """Result of one verify step.

    Attributes
    ----------
    token : int
        The token emitted (either the accepted draft or a residual sample).
    accepted : bool
        True iff the draft was accepted.
    accept_prob : float
        ``min(1, p[draft] / q[draft])`` — for telemetry.
    """

    token: int
    accepted: bool
    accept_prob: float


def _validate_dist(name: str, x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    if np.any(x < 0):
        raise ValueError(f"{name} has negative entries")
    s = float(x.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"{name} has non-positive or non-finite sum {s}")


def acceptance_probability(p: np.ndarray, q: np.ndarray, t: int) -> float:
    """``min(1, p[t] / q[t])`` — the rejection-sampling acceptance rule.

    If ``q[t] == 0`` the draft could not have produced ``t``, so the call is
    ill-formed; we return 0.0 (always reject) as a defensive default.
    """
    if q[t] <= 0.0:
        return 0.0
    return float(min(1.0, p[t] / q[t]))


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``normalize(max(p - q, 0))`` — distribution to sample from on reject.

    If the residual mass is zero (``p == q`` exactly), falls back to ``p`` so
    sampling still proceeds. This degenerate case is rare in practice but
    arises when the two samplers happen to agree perfectly.
    """
    diff = np.maximum(p - q, 0.0)
    s = float(diff.sum())
    if s <= 0.0:
        return p / float(p.sum())
    return diff / s


def sample_from_distribution(
    dist: np.ndarray, rng: np.random.Generator
) -> int:
    """Categorical sample from a 1D probability vector."""
    # Renormalize defensively against fp drift; np.random.choice is strict.
    s = float(dist.sum())
    if s <= 0.0:
        raise ValueError("cannot sample from zero-sum distribution")
    probs = dist / s
    return int(rng.choice(probs.shape[0], p=probs))


def verify_one_token(
    p: np.ndarray,
    q: np.ndarray,
    draft: int,
    rng: np.random.Generator,
) -> SpeculativeDecision:
    """One-step rejection-sampling verify.

    Accept ``draft`` with probability ``min(1, p[draft]/q[draft])``; on reject,
    sample a replacement from the residual distribution ``norm(max(p-q,0))``.
    """
    _validate_dist("p", p)
    _validate_dist("q", q)
    if not (0 <= draft < p.shape[0]):
        raise ValueError(f"draft {draft} out of vocab range {p.shape[0]}")

    accept_p = acceptance_probability(p, q, draft)
    if rng.random() < accept_p:
        return SpeculativeDecision(token=draft, accepted=True, accept_prob=accept_p)
    residual = residual_distribution(p, q)
    token = sample_from_distribution(residual, rng)
    return SpeculativeDecision(token=token, accepted=False, accept_prob=accept_p)


def speculative_output_marginal(
    p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Closed-form marginal of the verify-one-token output distribution.

    For every candidate token ``t``:

        Pr[output = t] = q[t] * min(1, p[t]/q[t])
                       + (1 - A) * residual[t]

    where ``A = sum_t q[t] * min(1, p[t]/q[t])`` is the total accept mass.

    By the speculative-decoding correctness theorem this MUST equal ``p``.
    Used as a property-test oracle in the test suite.
    """
    _validate_dist("p", p)
    _validate_dist("q", q)
    accept = np.minimum(1.0, np.divide(p, q, out=np.zeros_like(p), where=q > 0))
    accept_mass = float(np.sum(q * accept))
    reject_mass = 1.0 - accept_mass
    residual = residual_distribution(p, q)
    return q * accept + reject_mass * residual

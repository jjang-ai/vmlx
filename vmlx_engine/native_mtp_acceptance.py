"""Shared draft-acceptance math for native MTP verify cycles.

Both MTP schedulers must agree on how a drafted token is accepted, or the same
bundle behaves differently depending on whether it loaded through the text path
(``patches/mlx_lm_mtp/batch_generator.py``) or the MLLM path
(``mllm_batch_generator.py``).  That divergence is not hypothetical: the server
advertised ``stochastic=rejection-sampling-acceptance`` on /health while the
MLLM path implemented exact-match only, so every ``--is-mllm`` bundle had to pin
temperature to 0 to keep acceptance from collapsing. This module provides the
MLLM acceptance rule and distribution transforms shared with the text path.

Acceptance rule (Leviathan & Chen 2023): accept draft x with probability
min(1, p_target(x) / p_draft(x)).  Under greedy decode this degenerates to exact
match, which is why the greedy path can skip the ratio entirely.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence

import mlx.core as mx


def accept_lp_for(sampler: Optional[Callable[[Any], Any]], lp: Any) -> Any:
    """Reproduce the sampler's filter+temperature pipeline on ``lp``.

    The acceptance ratio and the residual distribution must be computed against
    the distribution the sampler actually drew from, not the raw softmax.  A
    draft sampled from a top-k 20 distribution has a much higher true draw
    probability than raw softmax suggests, and using the raw value understates
    ``p_draft`` and over-accepts.

    Sampling params are read off the callable as attributes.  For stock
    samplers that expose no reusable filter helpers, ``lp`` is returned
    unchanged, which matches the pre-PR-990 raw-lp acceptance behaviour.
    """
    if sampler is None:
        return lp
    exact_contract = getattr(sampler, "_vmlx_acceptance_logprobs", None)
    if exact_contract is not None:
        return exact_contract(lp)
    try:
        from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p
    except Exception:
        try:
            from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
        except Exception:
            return lp

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the result is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _row_width(row: Any) -> int:
    try:
        return int(row.shape[-1])
    except Exception:
        return 0


def accepted_count(
    draft_ids: Sequence[int],
    target_ids: Sequence[int],
    draft_lps: Sequence[Optional[Any]],
    target_lps: Sequence[Optional[Any]],
    *,
    stochastic: bool,
    sampler: Optional[Callable[[Any], Any]] = None,
    filtered: bool = False,
    telemetry: Optional[dict[str, int]] = None,
) -> int:
    """Count leading accepted drafts for one verify cycle.

    With stochastic distributions, every draft gets its
    min(1, p_target/p_draft) chance, even if an independent target draw
    matches it. Accepting matches first would add extra proposal mass that
    the positive target-minus-proposal residual cannot correct. Exact-match
    acceptance applies only without the stochastic distribution contract.

    ``filtered`` declares that the caller already ran :func:`accept_lp_for` over
    the rows; otherwise it is applied here so the ratio matches the sampling
    distribution.
    """
    accepted = 0
    for idx, draft_id in enumerate(draft_ids):
        draft_id = int(draft_id)
        target_lp = target_lps[idx] if idx < len(target_lps) else None
        draft_lp = draft_lps[idx] if idx < len(draft_lps) else None
        if not stochastic or target_lp is None or draft_lp is None:
            # Greedy and compact-top-k samplers expose no distribution; for
            # them exact match already is the correct acceptance test.
            if idx < len(target_ids) and int(target_ids[idx]) == draft_id:
                accepted += 1
                continue
            break
        if not filtered:
            target_lp = accept_lp_for(sampler, target_lp)
            draft_lp = accept_lp_for(sampler, draft_lp)
        if telemetry is not None:
            telemetry["ratio_checks"] = int(telemetry.get("ratio_checks", 0)) + 1
        # MLX does not raise on an out-of-range index, so a short or ragged row
        # would silently read as log_ratio 0.0 and ACCEPT the draft.  Check the
        # width explicitly rather than relying on an exception.
        if (
            draft_id < 0
            or _row_width(target_lp) <= draft_id
            or _row_width(draft_lp) <= draft_id
        ):
            break
        try:
            log_ratio = float(target_lp[draft_id]) - float(draft_lp[draft_id])
        except Exception:
            break
        if not math.isfinite(log_ratio):
            # A filtered-out draft has p_draft 0 (-inf): it could never have
            # been drawn from this distribution, so the ratio is meaningless.
            break
        if log_ratio < 0.0:
            uniform = getattr(sampler, "_vmlx_random_uniform", None)
            draw = (
                float(uniform())
                if uniform is not None
                else float(mx.random.uniform(shape=(1,)).item())
            )
            if draw <= 0.0 or math.log(draw) > log_ratio:
                break
        if telemetry is not None:
            telemetry["ratio_accepts"] = int(
                telemetry.get("ratio_accepts", 0)
            ) + 1
        accepted += 1
    return accepted


def residual_sample(
    target_lp: Any,
    draft_lp: Any,
    *,
    sampler: Optional[Callable[[Any], Any]] = None,
) -> tuple[int, Any]:
    """Sample the exact correction distribution after a rejected draft.

    ``target_lp`` and ``draft_lp`` must already describe the filtered
    distributions used by their samplers.  Sampling from ``max(p-q, 0)`` is
    the correction that preserves the target marginal; reusing an independent
    target draw after rejection does not.
    """

    target_2d = target_lp.reshape(1, -1) if target_lp.ndim == 1 else target_lp
    draft_1d = draft_lp.reshape(-1)
    p_target = mx.exp(target_2d.squeeze(0))
    p_draft = mx.exp(draft_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    mass = residual.sum(keepdims=True)
    distribution = mx.where(mass > 0, residual, p_target)
    logits = mx.log(distribution).reshape(1, -1)
    categorical = getattr(sampler, "_vmlx_categorical", None)
    sample = (
        categorical(logits)
        if categorical is not None
        else mx.random.categorical(logits)
    )
    return int(sample.item()), target_2d.squeeze(0)

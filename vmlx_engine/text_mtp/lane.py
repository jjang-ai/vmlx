# SPDX-License-Identifier: Apache-2.0
"""The text MTP lane: draft with the head, verify with one backbone forward.

Lifecycle of one request (a singleton ``GenerationBatch``):

1. **Seed** (``seed_lane``), right after mlx-lm's constructor step sampled the
   first token ``t0``: one more single-token backbone forward over ``t0``
   returns the hidden state and the next token ``t1``. Both are queued as
   confirmed output, and the head drafts ``d1..dn`` from (hidden, ``t1``). The
   seed forward is also the request's measured AR step.
2. **Emit** (``lane_step``): each public ``next()`` returns one queued token.
3. **Verify** (``verify_and_redraft``) when the queue is empty: one backbone
   forward over ``[t, d1..dn]`` with ``n_confirmed=1``. Row ``i`` scores
   ``d(i+1)``; the longest matching prefix ``k`` is accepted (greedy: exact
   match; sampled: ``min(1, p_target / p_draft)`` with a residual draw on
   rejection, so the output distribution equals the target's). The backbone
   drops the ``n - k`` rejected rows, ``k`` drafts plus one correction (or bonus
   after a full accept) are queued, and the head drafts the next chain from the
   last confirmed row.
4. **Leave**: on finish the lane publishes its record; when the valve or cost
   gate parks the request, ``hand_off_to_ar`` lines mlx-lm's own step up on the
   exact verified frontier, and ``maybe_resume_from_ar`` may later re-seed it.

The backbone cache only ever changes length on rejection (trim / restore), so
the scheduler's paged, prefix and SSD layers never notice speculation. The
head cache keeps loose history (it shapes draft quality only; verify decides
correctness), except in the opt-in GLM aligned experiment.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

from ..metal.affine_moe_pair_decode import affine_moe_ar_scope
from ..native_mtp_adaptive import NativeMTPAdaptiveValueState
from ..native_mtp_ar_safety import ArSafetyState
from ..native_mtp_recovery import NativeMTPRecovery
from . import glm_seam, lane_policy, lane_stats
from .lane_stats import LaneStats

logger = logging.getLogger(__name__)

# Diagnostics: synchronize after each GPU phase so the phase timings are real
# GPU cost instead of enqueue time. Serializes the pipeline; never a default.
PROFILE_PHASES = bool(os.environ.get("VMLX_MTP_PROFILE"))

LANE_ATTR = "_vmlx_text_mtp_lane"
RECOVERY_ATTR = "_vmlx_mtp_recovery"


class LaneAbort(RuntimeError):
    """The lane cannot run this step; mlx-lm's own step takes over cleanly."""


@dataclass
class LaneState:
    """Per-request lane state, attached to the batch as ``LANE_ATTR``."""

    # Confirmed tokens waiting to be returned: (token id, logprobs row, source),
    # source one of "init", "draft", "bonus", "verify".
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)
    mtp_cache: Optional[List[Any]] = None  # the head's own cache
    head_chain_pairs: int = 0  # unverified recursive pairs (GLM aligned history)
    next_main: Optional[Any] = None  # (1,) uint32: first row of the next verify
    draft_toks: List[Any] = field(default_factory=list)  # (1,) uint32 each
    draft_lps: List[Any] = field(default_factory=list)  # raw logprob rows
    draft_accept_lps: List[Any] = field(default_factory=list)  # sampler-view rows
    draft_ids: List[int] = field(default_factory=list)  # host ids, set at verify eval
    depth: int = 1
    depth_ceiling: int = 1
    adaptive_enabled: bool = False
    adaptive_value: NativeMTPAdaptiveValueState = field(
        default_factory=NativeMTPAdaptiveValueState
    )
    # An AR decision never drops a verified cycle: drafting stops, the queue
    # drains, then mlx-lm's step is primed from ``next_main``.
    ar_fallback_pending: bool = False
    ar_fallback_reason: Optional[str] = None
    ar_step_ms: float = 0.0
    cycle_span_start: float = 0.0
    recovery: Optional[NativeMTPRecovery] = None
    recovery_probe_started: float = 0.0
    adaptive_cycle_offset: int = 0
    ar_safety: ArSafetyState = field(default_factory=ArSafetyState)
    stats: LaneStats = field(default_factory=LaneStats)


# -- small helpers -----------------------------------------------------------


def phase_barrier(*arrays) -> None:
    if PROFILE_PHASES:
        import mlx.core as mx

        mx.eval(*arrays)


def generation_stream():
    """mlx-lm's module-level generation stream.

    ``mlx_lm`` re-exports a function named ``generate``, so the module itself
    is taken from ``sys.modules``.
    """
    return sys.modules["mlx_lm.generate"].generation_stream


def batch_sampler(batch: Any):
    if batch.samplers and batch.samplers[0] is not None:
        return batch.samplers[0]
    return batch.fallback_sampler


def batch_is_greedy(batch: Any) -> bool:
    # A seeded request carries its own sampler even at temperature 0, so the
    # explicit argmax marker decides, not the sampler's presence.
    if batch.samplers and batch.samplers[0] is not None:
        return bool(getattr(batch.samplers[0], "_vmlx_is_greedy", False))
    return bool(getattr(batch.fallback_sampler, "_vmlx_is_greedy", True))


def batch_processors(batch: Any) -> Optional[List[Any]]:
    if batch.logits_processors and batch.logits_processors[0]:
        return batch.logits_processors[0]
    return None


def run_processors(processors, history, logits):
    for processor in processors or ():
        logits = processor(history, logits)
    return logits


def normalize_logits(logits):
    import mlx.core as mx

    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def acceptance_view(sampler, logprobs):
    """The distribution the sampler actually draws from (shared with MLLM)."""
    from ..native_mtp_acceptance import accept_lp_for

    return accept_lp_for(sampler, logprobs)


def as_token_ids(tokens):
    import mlx.core as mx

    return tokens if tokens.dtype == mx.uint32 else tokens.astype(mx.uint32)


def drop_history_tail(batch: Any, count: int) -> None:
    """Forget ``count`` rejected rows from the processors' token history."""
    if count <= 0 or batch_processors(batch) is None:
        return
    history = batch._token_context[0]
    history._size = max(0, history._size - count)


def lane_of(batch: Any) -> Optional[LaneState]:
    return getattr(batch, LANE_ATTR, None)


def detach_lane(batch: Any) -> None:
    batch.__dict__.pop(LANE_ATTR, None)


def _label(batch: Any) -> str:
    uids = getattr(batch, "uids", None)
    return str(uids[0]) if uids else "?"


# -- eligibility -------------------------------------------------------------


def has_draft_head(model: Any) -> bool:
    """True only when the head module was attached at load time."""
    inner = getattr(model, "language_model", model)
    return getattr(inner, "mtp", None) is not None


def lane_eligible(batch: Any) -> bool:
    if not hasattr(batch, "model"):
        return False
    model = batch.model
    if not hasattr(model, "mtp_forward") or not has_draft_head(model):
        return False
    uids = getattr(batch, "uids", None)
    return uids is not None and len(uids) == 1


def lane_ineligible_reason(batch: Any) -> str:
    """Why a populated batch decodes without the lane (shown in the panel)."""
    if not hasattr(batch, "model"):
        return "GenerationBatch has no .model attribute"
    model = batch.model
    if not hasattr(model, "mtp_forward"):
        return (
            f"model {type(model).__module__}.{type(model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not has_draft_head(model):
        return "model has no attached mtp head (mtp_enabled was False at load time)"
    uids = getattr(batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        return f"batch size {len(uids)} != 1 (continuous batching, MTP off by design)"
    return ""


# -- backbone rewind / commit ------------------------------------------------


def _can_rewind(layer: Any, count: int) -> bool:
    if callable(getattr(layer, "rollback_speculative", None)):
        allowed = getattr(layer, "can_rollback_speculative", None)
        return not (callable(allowed) and not allowed(count))
    if getattr(layer, "rollback_state", None) is not None:
        return True
    return hasattr(layer, "is_trimmable") and layer.is_trimmable()


def rewind_backbone(caches: List[Any], count: int = 1) -> bool:
    """Drop the last ``count`` speculative rows from every backbone layer.

    All layers are checked before any is touched: a refusal halfway through
    would leave a partly rewound cache that no continuation can trust. A
    refusal means the request must fail (the caller raises).
    """
    if not all(_can_rewind(layer, count) for layer in caches):
        return False
    for layer in caches:
        partial = getattr(layer, "rollback_speculative", None)
        if callable(partial):
            if not partial(count):
                return False
            continue
        snapshot = getattr(layer, "rollback_state", None)
        if snapshot is not None:
            layer[0], layer[1] = snapshot
            layer.rollback_state = None
            continue
        layer.trim(count)
    return True


def commit_backbone(caches: List[Any]) -> None:
    """Forget speculative restore data after a fully accepted chain."""
    for layer in caches:
        settle = getattr(layer, "commit_speculative", None)
        if callable(settle):
            settle()
        if getattr(layer, "rollback_state", None) is not None:
            layer.rollback_state = None


# -- context extent ----------------------------------------------------------


def logical_context_extent(batch: Any) -> Tuple[int, str]:
    """Context length before the seed, SSD-restored positions included.

    ``tokens`` may hold only the admitted suffix and allocated KV buffers can be
    larger than their content, so the cache offsets decide. Telemetry only.
    """
    offsets = []
    stack = list(getattr(batch, "prompt_cache", ()) or ())
    visited = set()
    while stack:
        entry = stack.pop()
        if id(entry) in visited:
            continue
        visited.add(id(entry))
        try:
            stack.extend(getattr(entry, "caches", ()) or ())
            offset = getattr(entry, "offset", None)
            if type(offset) is not int:
                if offset is None or getattr(offset, "size", None) != 1:
                    continue
                offset = offset.reshape(()).item()
                if type(offset) is not int:
                    continue
            if offset >= 0:
                offsets.append(offset)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
    if offsets:
        return max(offsets), "logical_cache_offset"
    try:
        return len(batch.tokens[0]), "admitted_tokens_fallback"
    except (AttributeError, IndexError, TypeError):
        return 0, "unknown"


# -- seed --------------------------------------------------------------------


def _head_cache_for(batch: Any, lane: LaneState, first_token: Any) -> None:
    primed = None
    before = None
    if glm_seam.prompt_priming_enabled(batch.model):
        from ..native_mtp_prompt_priming import prime_stats, take_primed

        before = prime_stats(batch.model)
        primed = take_primed(batch.model, batch.prompt_cache, first_token)
    if primed is not None:
        lane.mtp_cache, lane.stats.prompt_primed_pairs = primed
        lane.stats.prompt_prime_source = "cold_prompt"
        return
    lane.mtp_cache = batch.model.make_mtp_cache()
    if before is None:
        return
    if before.get("active"):
        lane.stats.prompt_prime_source = "capture_discarded_at_seam"
    elif before.get("plan_armed"):
        reason = str((before.get("last") or {}).get("reason") or "unknown")
        lane.stats.prompt_prime_source = f"armed_no_capture:{reason}"
    else:
        lane.stats.prompt_prime_source = "not_armed"


def seed_lane(batch: Any, *, recovery: Optional[NativeMTPRecovery] = None) -> None:
    """Start the lane right after mlx-lm's constructor (or AR) step.

    On entry the backbone cache holds the context through the previous token,
    ``_next_tokens`` holds the sampled token ``t0`` and ``_next_logprobs[0]`` its
    row. Queues ``t0`` and ``t1`` and drafts the first chain. An empty batch
    (nothing sampled) leaves the lane off.
    """
    import mlx.core as mx

    if batch._next_tokens is None or not batch.uids:
        return

    resume_started = time.perf_counter() if recovery is not None else 0.0
    context_tokens, context_source = logical_context_extent(batch)
    sampler = batch_sampler(batch)
    processors = batch_processors(batch)

    t0 = as_token_ids(batch._next_tokens)
    t0_logprobs = batch._next_logprobs[0]
    history = batch._token_context[0].update_and_fetch(t0) if processors is not None else None

    # A full-context single-token forward: its wall time is the request's own
    # AR step, the baseline every cost decision compares against.
    try:
        mx.synchronize()  # do not bill a batch partner's pending work to the seed
    except Exception:
        pass
    started = time.perf_counter()
    with mx.stream(generation_stream()):
        logits, hidden = batch.model(t0[:, None], cache=batch.prompt_cache, return_hidden=True)
    t1_logprobs = normalize_logits(run_processors(processors, history, logits[:, -1, :]))
    t1 = sampler(t1_logprobs)
    mx.eval(t0, t1)
    seed_ms = (time.perf_counter() - started) * 1000.0

    lane = LaneState()
    lane.recovery = recovery
    lane.stats.request_counted = recovery is not None
    calibration_resume = recovery is not None and recovery.calibrating
    lane.recovery_probe_started = 0.0 if calibration_resume else resume_started
    lane.ar_step_ms = recovery.ar_ms if recovery is not None else seed_ms
    lane.stats.seed_ar_step_ms = seed_ms
    lane.stats.seed_context_tokens = context_tokens
    lane.stats.seed_context_source = context_source
    lane.ar_safety.prompt_tokens = context_tokens
    logger.info(
        "MTP[%s] AR baseline: seed_step_ms=%.3f effective_ms=%.3f source=%s "
        "context_tokens=%d context_source=%s",
        batch.uids[0], seed_ms, lane.ar_step_ms,
        "productive_ar" if recovery is not None else "single_seed",
        context_tokens, context_source,
    )
    _head_cache_for(batch, lane, t0)

    if recovery is not None:
        lane.depth_ceiling = recovery.depth_ceiling
        lane.adaptive_enabled = recovery.adaptive
        lane.depth = 1
    else:
        lane.depth_ceiling = lane_policy.resolve_depth(batch, adaptive_ceiling=True)[0]
        lane.adaptive_enabled = lane_policy.adaptive_enabled()
        lane.depth = 1 if lane.adaptive_enabled else lane.depth_ceiling
    if calibration_resume:
        lane.depth = recovery.resume_depth
        if recovery.adaptive_state is not None:
            lane.adaptive_value = recovery.adaptive_state
            lane.adaptive_cycle_offset = recovery.adaptive_cycle_offset
            # Keep probe schedule and debts; wall samples predate the refresh.
            for samples in lane.adaptive_value.samples_by_depth:
                samples.clear()
            lane.adaptive_value.armed_at = 0.0
            lane.adaptive_value.armed_depth = 0
    if recovery is not None:
        if not calibration_resume:
            recovery.attempts += 1
        recovery.calibrating = False
        recovery.next_calibration_token = int(batch._num_tokens[0]) + 768
        lane.stats.recovery = recovery.snapshot()

    stats = lane.stats
    stats.depth = lane.depth
    stats.starting_depth = lane.depth
    stats.depth_ceiling = lane.depth_ceiling
    stats.depth_policy = "adaptive" if lane.adaptive_enabled else "fixed"
    stats.seed_main_forwards = 1
    lane.next_main = as_token_ids(t1)
    lane.queue.append((int(t0.tolist()[0]), t0_logprobs, "init"))
    lane.queue.append((int(t1.tolist()[0]), t1_logprobs.squeeze(0), "init"))
    setattr(batch, LANE_ATTR, lane)

    lane_policy.arm_cycle(lane, now=time.perf_counter())
    draft_recursive(batch, lane, hidden[:, -1:, :], lane.next_main, history=history)
    if recovery is not None:
        recovery.resume_wall_ms += (time.perf_counter() - resume_started) * 1000.0


# -- drafting ----------------------------------------------------------------


def _reset_drafts(lane: LaneState) -> None:
    lane.draft_toks = []
    lane.draft_lps = []
    lane.draft_accept_lps = []
    lane.draft_ids = []


def _push_draft(lane: LaneState, sampler, row_logits) -> Any:
    logprobs = normalize_logits(row_logits)
    token = as_token_ids(sampler(logprobs))
    lane.draft_toks.append(token)
    lane.draft_lps.append(logprobs.squeeze(0))
    # The acceptance ratio needs the distribution the draft was drawn from.
    lane.draft_accept_lps.append(acceptance_view(sampler, logprobs).squeeze(0))
    return token


def _head_step(batch: Any, lane: LaneState, hidden, tokens):
    import mlx.core as mx

    lane.stats.mtp_forwards += 1
    with mx.stream(generation_stream()):
        logits, head_hidden = batch.model.mtp_forward(
            hidden, tokens, lane.mtp_cache, return_hidden=True
        )
        row = logits[:, -1, :]
    return row, head_hidden


def _with_history(processors, history, token, row):
    """Apply processors with ``token`` appended to the running history."""
    import mlx.core as mx

    if processors is None or history is None:
        return history, row
    history = mx.concatenate([history, as_token_ids(token)])
    return history, run_processors(processors, history, row)


def draft_recursive(batch: Any, lane: LaneState, hidden, token, *, history=None) -> None:
    """Draft ``lane.depth`` tokens: d1 from (hidden, token), then the head's
    own last hidden and draft feed the next step. Nothing is synced to the host;
    the verify cycle's single eval materializes the ids."""
    sampler = batch_sampler(batch)
    processors = batch_processors(batch)
    _reset_drafts(lane)
    started = time.perf_counter()
    for _ in range(max(1, lane.depth)):
        row, head_hidden = _head_step(batch, lane, hidden, token.reshape(1, 1))
        history, row = _with_history(processors, history, token, row)
        token = _push_draft(lane, sampler, row)
        hidden = head_hidden[:, -1:, :]
    lane.draft_ids = []
    phase_barrier(*lane.draft_toks)
    lane.stats.mtp_head_ms += (time.perf_counter() - started) * 1000
    if glm_seam.aligned_head_history_enabled(batch):
        lane.stats.mtp_head_cache_policy = "glm_aligned"
        lane.head_chain_pairs = max(0, len(lane.draft_toks) - 1)
    else:
        lane.stats.mtp_head_cache_policy = "loose"
        lane.head_chain_pairs = 0


def draft_aligned(batch: Any, lane: LaneState, hidden, tokens, *, history=None) -> None:
    """GLM aligned history: commit all confirmed pairs in one head forward
    (its last row is d1), then recurse for the deeper drafts, which stay marked
    unverified until the next cycle trims them."""
    if int(hidden.shape[1]) != int(tokens.shape[1]):
        raise RuntimeError("GLM aligned MTP commit pair lengths disagree")
    if int(tokens.shape[1]) < 1:
        raise RuntimeError("GLM aligned MTP commit cannot be empty")
    sampler = batch_sampler(batch)
    processors = batch_processors(batch)
    _reset_drafts(lane)
    started = time.perf_counter()
    row, head_hidden = _head_step(batch, lane, hidden, tokens)
    history, row = _with_history(processors, history, tokens[:, -1].reshape(1), row)
    token = _push_draft(lane, sampler, row)
    hidden = head_hidden[:, -1:, :]
    for _ in range(1, max(1, lane.depth)):
        row, head_hidden = _head_step(batch, lane, hidden, token.reshape(1, 1))
        history, row = _with_history(processors, history, token, row)
        token = _push_draft(lane, sampler, row)
        hidden = head_hidden[:, -1:, :]
    lane.head_chain_pairs = max(0, len(lane.draft_toks) - 1)
    lane.draft_ids = []
    phase_barrier(*lane.draft_toks)
    lane.stats.mtp_head_ms += (time.perf_counter() - started) * 1000


# -- emission ----------------------------------------------------------------


def emit_one(batch: Any, token_id: int, logprobs, stats: Optional[LaneStats] = None) -> List[Any]:
    """One response with the same bookkeeping as mlx-lm's ``next()``."""
    Response = type(batch).Response
    finish_reason = None
    batch.tokens[0].append(token_id)
    batch._num_tokens[0] += 1
    if batch._num_tokens[0] >= batch.max_tokens[0]:
        finish_reason = "length"
    batch._matcher_states[0], match_sequence, current_state = batch.state_machines[0].match(
        batch._matcher_states[0], token_id
    )
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is None:
        return [
            Response(
                uid=batch.uids[0],
                token=token_id,
                logprobs=logprobs,
                finish_reason=None,
                current_state=current_state,
                match_sequence=match_sequence,
                prompt_cache=None,
                all_tokens=None,
            )
        ]

    response = Response(
        uid=batch.uids[0],
        token=token_id,
        logprobs=logprobs,
        finish_reason=finish_reason,
        current_state=current_state,
        match_sequence=match_sequence,
        prompt_cache=batch.extract_cache(0),
        all_tokens=batch.tokens[0],
    )
    if stats is not None:
        lane_stats.finish_record(
            batch.uids[0], stats, finish_reason, getattr(lane_of(batch), "mtp_cache", None)
        )
    # Detach first: the filter hook would otherwise publish this request twice.
    detach_lane(batch)
    batch.filter([])
    return [response]


def _pop_and_emit(batch: Any, lane: LaneState) -> List[Any]:
    token_id, logprobs, source = lane.queue.popleft()
    lane.stats.count_emit(source)
    return emit_one(batch, token_id, logprobs, lane.stats)


def lane_step(batch: Any, lane: LaneState) -> List[Any]:
    """Return the next confirmed token, verifying a new chain when needed."""
    if lane.queue:
        return _pop_and_emit(batch, lane)
    if lane.stats.cycles == 0 and lane.cycle_span_start <= 0.0:
        lane.cycle_span_start = time.perf_counter()
    verify_and_redraft(batch, lane)
    if not lane.queue:
        raise LaneAbort("verify cycle produced no emit tokens")
    return _pop_and_emit(batch, lane)


# -- verify ------------------------------------------------------------------


def _check_live_encode(caches: List[Any], advance: int) -> None:
    """A TurboQuant layer compresses once, at ``compress_after``. If that
    happened inside a verify advance a later trim could not take the rejected
    rows back out of the compressed buffers, so that one step goes to AR."""
    for layer in caches:
        threshold = getattr(layer, "compress_after", None)
        if not threshold:
            continue
        try:
            threshold = int(threshold)
            uncompressed = int(getattr(layer, "_compressed_tokens", 0) or 0) == 0
            offset = int(getattr(layer, "offset", 0) or 0)
        except (TypeError, ValueError):
            raise LaneAbort("tq live-encode crossing state unreadable")
        if threshold > 0 and uncompressed and offset + advance >= threshold:
            raise LaneAbort("tq live-encode crossing in verify advance")


def _accepted_prefix(lane: LaneState, sampler, greedy: bool, target_lps, sampled_ids) -> int:
    accepted = 0
    for index, draft_id in enumerate(lane.draft_ids):
        if greedy:
            ok = sampled_ids[index] == draft_id
        else:
            target = acceptance_view(sampler, target_lps[index : index + 1])
            log_ratio = target[0, draft_id].item() - lane.draft_accept_lps[index][draft_id].item()
            uniform = getattr(sampler, "_vmlx_random_uniform", None)
            draw = uniform() if uniform is not None else random.random()
            ok = log_ratio >= 0 or draw < math.exp(log_ratio)
        if not ok:
            break
        accepted += 1
    return accepted


def verify_and_redraft(batch: Any, lane: LaneState) -> None:
    """One verify forward over ``[next_main, d1..dn]`` and the next chain."""
    import mlx.core as mx

    from ..native_mtp_acceptance import residual_sample

    if lane.next_main is None or not lane.draft_toks:
        raise LaneAbort("verify cycle entered without next_main / drafts")

    sampler = batch_sampler(batch)
    processors = batch_processors(batch)
    greedy = batch_is_greedy(batch)
    drafts = len(lane.draft_toks)
    stats = lane.stats
    _check_live_encode(batch.prompt_cache, drafts + 1)

    rows = [lane.next_main] + list(lane.draft_toks)
    inputs = mx.concatenate(rows)
    # Processor history after each verify row, exactly as AR decode builds it.
    histories: List[Any] = []
    if processors is not None:
        histories = [batch._token_context[0].update_and_fetch(row) for row in rows]

    started = time.perf_counter()
    with mx.stream(generation_stream()):
        logits, hidden = batch.model(
            inputs[None, :], cache=batch.prompt_cache, return_hidden=True, n_confirmed=1
        )
    phase_barrier(logits, hidden)
    stats.backbone_ms += (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    if processors is None:
        verify_logits = logits[0]  # already one contiguous (n + 1, vocab) slab
    else:
        verify_logits = mx.concatenate(
            [run_processors(processors, histories[i], logits[:, i, :]) for i in range(drafts + 1)],
            axis=0,
        )
    target_lps = verify_logits - mx.logsumexp(verify_logits, axis=-1, keepdims=True)
    # Row-by-row sampling keeps a stochastic sampler's RNG sequence identical
    # to AR decode; one eval resolves forward, samples and the draft chain.
    sampled = [sampler(target_lps[i : i + 1]) for i in range(drafts + 1)]
    mx.eval(*sampled, *lane.draft_toks)
    sampled_ids = [int(t.tolist()[0]) for t in sampled]
    if len(lane.draft_ids) != drafts:
        lane.draft_ids = [int(t.tolist()[0]) for t in lane.draft_toks]
    accepted = _accepted_prefix(lane, sampler, greedy, target_lps, sampled_ids)
    stats.sample_ms += (time.perf_counter() - started) * 1000

    stats.cycles += 1
    stats.verify_main_forwards += 1
    stats.draft_tokens_proposed += drafts
    stats.draft_tokens_accepted += accepted
    for index in range(drafts):
        stats.drafted_by_depth[index] += 1
        if index < accepted:
            stats.accepted_by_depth[index] += 1
    full = accepted == drafts
    if full:
        stats.accepts += 1
    else:
        stats.rejects += 1

    started = time.perf_counter()
    if full:
        commit_backbone(batch.prompt_cache)
    else:
        if not rewind_backbone(batch.prompt_cache, drafts - accepted):
            try:
                lane_stats.finish_record(_label(batch), stats, "rollback_refused", lane.mtp_cache)
            except Exception:
                logger.debug("MTP rollback-refusal telemetry publication failed", exc_info=True)
            # The cache still holds the rejected rows and cannot be repaired;
            # nothing (not even an AR step) may continue from it.
            raise RuntimeError(
                "native MTP cache rejected rollback (text path) — "
                "speculative verify advance cannot be undone"
            )
        stats.mtp_cache_retained_on_rejects += 1
        if processors is not None:
            drop_history_tail(batch, drafts - accepted)
    stats.cache_ops_ms += (time.perf_counter() - started) * 1000

    for index in range(accepted):
        lane.queue.append((lane.draft_ids[index], lane.draft_lps[index], "draft"))
    if full:
        last = (sampled_ids[drafts], target_lps[drafts : drafts + 1].squeeze(0), "bonus")
    elif greedy:
        last = (sampled_ids[accepted], target_lps[accepted : accepted + 1].squeeze(0), "verify")
    else:
        # Correction from max(p_target - p_draft, 0) over the sampler's view;
        # the reported logprobs stay the raw verify row like every other path.
        corrected, _ = residual_sample(
            acceptance_view(sampler, target_lps[accepted : accepted + 1]),
            lane.draft_accept_lps[accepted],
            sampler=sampler,
        )
        last = (corrected, target_lps[accepted : accepted + 1].squeeze(0), "verify")
    lane.queue.append(last)

    confirmed = mx.array([last[0]], dtype=mx.uint32)
    lane.next_main = confirmed
    label = _label(batch)
    if lane_policy.apply_safety_valve(label, lane) or lane_policy.apply_cost_gate(
        label, lane, now=time.perf_counter()
    ):
        # Leaving: no speculative work may outlive the decision. The backbone
        # sits on the verified frontier and the queue holds confirmed tokens.
        _reset_drafts(lane)
        lane.head_chain_pairs = 0
        return
    # The valve has first say; a neighbour probe must not reset its window.
    lane_policy.settle_cycle(
        label, lane, completed_depth=drafts, accepted=accepted, now=time.perf_counter()
    )
    lane_policy.arm_cycle(lane, now=time.perf_counter())
    history = histories[accepted] if processors is not None else None
    if glm_seam.aligned_head_history_enabled(batch):
        stats.mtp_head_cache_policy = "glm_aligned"
        if not glm_seam.drop_unverified_head_pairs(lane):
            raise RuntimeError("GLM aligned MTP head cache could not trim unverified pairs")
        pairs = mx.concatenate(
            [as_token_ids(t).reshape(1) for t in lane.draft_toks[:accepted]] + [confirmed.reshape(1)]
        ).reshape(1, accepted + 1)
        draft_aligned(batch, lane, hidden[:, : accepted + 1, :], pairs, history=history)
        return
    draft_recursive(batch, lane, hidden[:, accepted : accepted + 1, :], confirmed, history=history)


# -- AR handoff and re-entry -------------------------------------------------


def handoff_readiness(batch: Any, lane: LaneState) -> Tuple[bool, str]:
    """Can mlx-lm's own step take over at the verified frontier now?"""
    if lane.queue:
        return False, "pending_queue"
    if lane.next_main is None:
        return False, "missing_next_main"
    try:
        next_id = int(lane.next_main.tolist()[0])
    except Exception:
        return False, "invalid_next_main"
    tokens = getattr(batch, "tokens", None) or []
    if not tokens or not tokens[0]:
        return False, "missing_visible_token"
    if next_id != int(tokens[0][-1]):
        return False, "next_main_mismatch"
    if any(getattr(layer, "rollback_state", None) is not None for layer in getattr(batch, "prompt_cache", ())):
        return False, "pending_rollback_state"
    return True, "ready"


def parked_recovery(batch: Any) -> Optional[NativeMTPRecovery]:
    """The batch's parked-in-AR record, dropped once its request is gone."""
    recovery = getattr(batch, RECOVERY_ATTR, None)
    if recovery is not None and recovery.uid not in (getattr(batch, "uids", None) or ()):
        batch.__dict__.pop(RECOVERY_ATTR, None)
        return None
    return recovery


_PARKED_COUNTERS = (
    "cycles",
    "draft_tokens_proposed",
    "draft_tokens_accepted",
    "seed_main_forwards",
    "verify_main_forwards",
    "mtp_forwards",
    "init_emits",
    "draft_emits",
    "bonus_emits",
    "verify_emits",
)


def hand_off_to_ar(batch: Any, lane: LaneState, last_logprobs: Any, *, calibration: bool = False) -> None:
    """Prime mlx-lm's step on the verified frontier without a repeated token.

    The last queued token is already visible but not yet in the backbone cache;
    mlx-lm's ``_step`` would consume and return it. Take it off ``tokens``, let
    ``_step`` consume and re-append it (preparing the following sample), then
    drop the lane, so the next public ``next()`` returns the following token.
    """
    started = time.perf_counter()
    ready, reason = handoff_readiness(batch, lane)
    if not ready:
        raise RuntimeError(f"native MTP AR fallback unsafe: {reason}")

    visible = int(batch.tokens[0].pop())
    batch._next_tokens = as_token_ids(lane.next_main)
    batch._next_logprobs = [last_logprobs]
    try:
        with affine_moe_ar_scope():
            consumed, _ = batch._step()
    except Exception:
        # The cache may have advanced already: no quiet recovery.
        if not batch.tokens[0] or int(batch.tokens[0][-1]) != visible:
            batch.tokens[0].append(visible)
        raise
    if list(consumed) != [visible]:
        raise RuntimeError(
            "native MTP AR fallback consumed the wrong boundary token: "
            f"expected={visible} actual={list(consumed)}"
        )

    recovery = parked_recovery(batch)
    if recovery is None:
        recovery = NativeMTPRecovery(batch.uids[0], lane.depth_ceiling, lane.adaptive_enabled)
        setattr(batch, RECOVERY_ATTR, recovery)
    if calibration:
        recovery.park_for_calibration(depth=lane.depth)
        recovery.adaptive_state = copy.deepcopy(lane.adaptive_value)
        recovery.adaptive_cycle_offset = lane.adaptive_cycle_offset + int(lane.stats.cycles)
    else:
        recovery.park(failed_probe=lane.recovery_probe_started > 0.0)
    recovery.handoff_wall_ms += (time.perf_counter() - started) * 1000.0
    for name in _PARKED_COUNTERS:
        recovery.completed_mtp[name] = recovery.completed_mtp.get(name, 0) + getattr(lane.stats, name)
    lane.stats.recovery = recovery.snapshot()
    lane.stats.depth = 0  # effectively AR now
    recovery.parked_stats = lane.stats
    lane_stats.finish_record(
        _label(batch),
        lane.stats,
        "calibration_ar" if calibration else "fallback_to_ar",
        lane.mtp_cache,
    )
    if lane_of(batch) is lane:
        detach_lane(batch)
    logger.info(
        "MTP[%s] fallback to AR after verified queue drain: %s",
        _label(batch),
        "productive baseline calibration" if calibration else lane.ar_fallback_reason or "adaptive runtime cost",
    )


def maybe_resume_from_ar(batch: Any, recovery: NativeMTPRecovery) -> bool:
    """Re-seed from a productive singleton AR frontier when recovery allows.

    A seed failure propagates: the seed may already have advanced the caches,
    so the old AR state is no longer trustworthy.
    """
    if (
        lane_of(batch) is not None
        or not recovery.ready
        or not lane_eligible(batch)
        or getattr(batch, "uids", None) != [recovery.uid]
        or getattr(batch, "_next_tokens", None) is None
    ):
        return False
    if int(batch.max_tokens[0]) - int(batch._num_tokens[0]) < 3:
        return False  # the seed alone queues two tokens
    if any(getattr(layer, "rollback_state", None) is not None for layer in batch.prompt_cache):
        return False
    calibrating = recovery.calibrating
    seed_lane(batch, recovery=recovery)
    logger.info(
        "MTP[%s] AR -> D%d %s: ceiling=D%d ar_ms_per_token=%.3f cooldown=%d",
        recovery.uid,
        lane_of(batch).depth,
        "calibration resume" if calibrating else "recovery probe",
        recovery.depth_ceiling,
        recovery.ar_ms,
        recovery.cooldown,
    )
    return True


def calibration_due(batch: Any, lane: LaneState) -> bool:
    """Refresh the AR baseline at a drained frontier, never inside a probe."""
    deadline = lane.recovery.next_calibration_token if lane.recovery is not None else 128
    uids = getattr(batch, "uids", None)
    return (
        not lane.ar_fallback_pending
        and not lane.queue
        and lane.recovery_probe_started <= 0.0
        and not lane.adaptive_value.active_probe_target
        and uids is not None
        and len(uids) == 1
        and int(batch._num_tokens[0]) >= deadline
        # nine AR steps plus a meaningful resumed cycle
        and int(batch.max_tokens[0]) - int(batch._num_tokens[0]) >= 32
    )

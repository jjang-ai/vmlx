# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — after the standard ``_step()`` has run
  the prompt's last token through the backbone, we add an MTP "post-init"
  step that runs one more 1-token backbone forward (with hidden) and one
  MTP-head forward. Two confirmed tokens are queued for emission and a
  draft is stashed for the first verify cycle.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence we emit from the per-batch queue first; once empty, we run a
  2-token verify forward over ``[next_main, draft]`` with
  ``n_confirmed=1`` and a single MTP-head forward at the bonus position
  (accept) or confirmed position (reject), refilling the queue from the
  verify outputs.

- ``BatchGenerator._next`` — for GLM-5.3's typed KDA/MLA cache only, preserve
  the exact N-1 prompt boundary immediately before mlx-lm consumes the final
  prompt token. The detached snapshot follows the request's generation
  responses until the scheduler stores it in RAM/SSD at terminal completion.

The throughput math (greedy, accept rate p):
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - Cost per cycle: 1× backbone (n+1-token verify) + 1× MTP head.

  The win depends ENTIRELY on how much cheaper the (n+1)-token verify forward is
  than (n+1) single-token forwards. On a DENSE model the marginal verify token
  is nearly free (weights read once) so the ratio is ~1.05× and MTP wins big.
  On an MoE it is NOT: measured on Hy3-JANG_2L a 2-token verify costs 1.36× a
  1-token forward (43.6ms vs 31.5ms) because each routed expert does per-row
  2-bit dequant+matmul — the cost is per-row COMPUTE, not shared expert-read
  memory (proven: a 2-token forward whose rows pick the SAME experts costs the
  same as one whose rows pick DIFFERENT experts). Break-even acceptance is
  therefore ~59%; Hy3's affine-8 head drafts against a 2-bit backbone accept at
  ~58%, so MTP straddles break-even and nets ~-3%. See `native_mtp_blocked` in
  the JANG_2L bundle. It only becomes a win with a higher-bit routed backbone
  (less-damaged → higher head agreement → acceptance clears break-even).

Greedy identity (sampler is None): the patched dispatch must produce the same
tokens as the standard step. The oMLX-side equivalent is pinned in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

from ...native_mtp_adaptive import (
    NativeMTPAdaptiveValueState,
    adaptive_value_snapshot,
    arm_depth_cycle,
    choose_depth_by_value,
    finish_armed_depth_cycle,
)
from ...native_mtp_ar_safety import ArSafetyState, ar_safety_step
from ...native_mtp_recovery import NativeMTPRecovery
from ...native_mtp_cache_telemetry import (
    native_mtp_cache_lifecycle_snapshot,
    native_mtp_cache_snapshot,
)

logger = logging.getLogger(__name__)

_PATCHED = False

# Barrier-based per-phase profiling. Off by default (adds an mx.eval after each
# GPU phase, which serializes the pipeline — measurement only). When on, the
# _MtpStats timings become real GPU-phase costs instead of enqueue times.
_MTP_PROFILE = bool(os.environ.get("VMLX_MTP_PROFILE"))

# Measurement bypass: keep the MTP head LOADED (so the memory footprint and
# residency are identical to an MTP-on run) but decode with the standard
# autoregressive step. This isolates the MTP algorithm from the head's memory
# cost, giving a footprint-controlled baseline for A/B on a variance-prone box.
_MTP_BYPASS = bool(os.environ.get("VMLX_MTP_BYPASS"))

# Process-local, read-only telemetry surface for /health. The generation loop
# runs on the scheduler worker while health is served from the API loop, so
# publish complete snapshots under a lock instead of exposing the mutable
# per-request dataclass. A process restart intentionally resets these totals.
_MTP_TELEMETRY_LOCK = threading.Lock()
_LAST_NATIVE_MTP: Optional[dict] = None
_NATIVE_MTP_TOTALS = {
    "requests": 0,
    "cycles": 0,
    "drafted_tokens": 0,
    "accepted_tokens": 0,
    "mtp_cache_recreated_on_rejects": 0,
    "mtp_cache_retained_on_rejects": 0,
}


def _pbar(*arrays) -> None:
    """Profiling-only barrier: resolve `arrays` iff VMLX_MTP_PROFILE is set."""
    if _MTP_PROFILE:
        import mlx.core as mx

        mx.eval(*arrays)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Wrap the mlx-lm generation path with MTP and GLM cache support."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.generate import (
            BatchGenerator,
            GenerationBatch,
            PromptProcessingBatch,
        )
    except ImportError:
        logger.debug("mlx_lm.generate.GenerationBatch not importable")
        return False

    if hasattr(GenerationBatch, "_omlx_mtp_patched"):
        _PATCHED = True
        return True

    original_init = GenerationBatch.__init__
    original_next = GenerationBatch.next
    original_filter = GenerationBatch.filter
    original_extend = GenerationBatch.extend
    original_batch_generator_next = BatchGenerator._next
    original_batch_generator_remove = BatchGenerator.remove
    original_batch_generator_make_batch = BatchGenerator._make_batch
    original_prompt_processing_prompt = PromptProcessingBatch.prompt

    class _GlmPromptCaptureProxy:
        """Expose GLM prompt hiddens at mlx-lm's owning prefill seam."""

        def __init__(self, model):
            self._model = model

        def __getattr__(self, name):
            return getattr(self._model, name)

        def __call__(self, inputs, *args, **kwargs):
            from ...native_mtp_prompt_priming import capture_prefill

            call_kwargs = dict(kwargs)
            call_kwargs["return_hidden"] = True
            result = self._model(inputs, *args, **call_kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "armed GLM prompt priming requires (logits, hidden)"
                )
            logits, hidden = result
            capture_prefill(
                self._model,
                inputs,
                hidden,
                call_kwargs.get("cache"),
            )
            return logits

    def patched_prompt_processing_prompt(self, tokens):
        from ...native_mtp_prompt_priming import capture_requested

        model = getattr(self, "model", None)
        if not _glm_prompt_priming_enabled(model) or not capture_requested(model):
            from ...utils.hybrid_prefill_capture import prompt_with_hybrid_capture

            return prompt_with_hybrid_capture(self, tokens, original_prompt_processing_prompt)
        self.model = _GlmPromptCaptureProxy(model)
        try:
            return original_prompt_processing_prompt(self, tokens)
        finally:
            self.model = model

    def attach_prompt_cache_snapshots(self, responses):
        snapshots = getattr(self, "_vmlx_prompt_cache_snapshots", None)
        if not snapshots:
            return responses
        for response in responses or ():
            uid = getattr(response, "uid", None)
            snapshot = snapshots.get(uid)
            if snapshot is None:
                continue
            response.prompt_cache_snapshot = snapshot
            if getattr(response, "finish_reason", None) is not None:
                snapshots.pop(uid, None)
        if not snapshots:
            try:
                delattr(self, "_vmlx_prompt_cache_snapshots")
            except AttributeError:
                pass
        return responses

    def patched_init(self, *args, **kwargs):
        global _LAST_NATIVE_MTP_SKIP

        # PromptProcessingBatch.generate() hands the final prompt token to
        # GenerationBatch.__init__, whose standard _step() owns that forward.
        # Capture it at the same owner so the primed head history reaches the
        # exact prompt boundary before _post_init_mtp advances the sampled
        # main token. Without this proxy, a 22-token prompt stopped at offset
        # 21 and was correctly rejected when the backbone arrived at 23.
        init_args = list(args)
        init_kwargs = dict(kwargs)
        init_model = (
            init_kwargs.get("model")
            if "model" in init_kwargs
            else (init_args[0] if init_args else None)
        )
        capture_final_prompt = False
        if init_model is not None and _glm_prompt_priming_enabled(init_model):
            from ...native_mtp_prompt_priming import capture_requested

            capture_final_prompt = capture_requested(init_model)
        if capture_final_prompt:
            proxy = _GlmPromptCaptureProxy(init_model)
            if "model" in init_kwargs:
                init_kwargs["model"] = proxy
            else:
                init_args[0] = proxy
        original_init(self, *init_args, **init_kwargs)
        if capture_final_prompt:
            self.model = init_model
        if _MTP_BYPASS:
            return  # head stays loaded; decode via the standard step
        if _is_mtp_eligible(self):
            activate, depth, seed = _adaptive_mtp_activation_decision(self)
            if not activate:
                uid = str(getattr(self, "uids", ["?"])[0])
                with _MTP_TELEMETRY_LOCK:
                    _LAST_NATIVE_MTP_SKIP = {
                        "uid": uid,
                        "reason": seed,
                        "configured_depth": depth,
                    }
                logger.info(
                    "MTP path stays AR for uid=%s seed=%s configured_depth=%d",
                    uid,
                    seed,
                    depth,
                )
                return
            try:
                _post_init_mtp(self)
                base_model = getattr(self.model, "model", None)
                vectorized_kda_layers = sum(
                    int(
                        bool(getattr(layer, "is_linear", False))
                        and bool(
                            getattr(
                                getattr(layer, "self_attn", None),
                                "_vectorized_speculative_verify",
                                False,
                            )
                        )
                    )
                    for layer in getattr(base_model, "layers", ())
                )
                if vectorized_kda_layers:
                    logger.info(
                        "GLM vectorized KDA verify active for uid=%s layers=%d",
                        getattr(self, "uids", ["?"])[0],
                        vectorized_kda_layers,
                    )
                logger.info(
                    "MTP path activated for uid=%s (model has mtp_forward, batch=1)",
                    getattr(self, "uids", ["?"])[0],
                )
            except _MtpStepFallback as exc:
                logger.warning("MTP post-init fallback: %s", exc)
        else:
            # The empty-batch case is BatchGenerator.__init__ pre-creating
            # ``self._generation_batch = GenerationBatch.empty(...)`` and is
            # always part of normal startup — silence it. Only log when the
            # batch is genuinely populated (e.g. continuous batching with
            # batch>1) so the message points at a real misconfiguration.
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)
                    # Publish it. PerformancePanel has always READ
                    # batch_generator.last_native_mtp_skip, but nothing in the
                    # engine ever produced that key, so the skip tile was dead
                    # UI and only positive engagement could ever display. A
                    # DEBUG line is not a surface a user can see.
                    with _MTP_TELEMETRY_LOCK:
                        _LAST_NATIVE_MTP_SKIP = {
                            "uid": str(uids[0]),
                            "reason": reason,
                        }

    def patched_next(self, *args, **kwargs):
        recovery = _text_recovery_for_batch(self)
        if recovery is not None:
            _text_mtp_maybe_resume(self, recovery)
        if _is_mtp_eligible(self):
            state = getattr(self, "_omlx_mtp_state", None)
            if state is not None:
                try:
                    responses = _mtp_next(self, state)
                    calibration = _text_mtp_calibration_due(self, state)
                    if (
                        (state.ar_fallback_pending or calibration)
                        and not state.queue
                        and getattr(self, "_omlx_mtp_state", None) is state
                        and responses
                        and responses[0].finish_reason is None
                    ):
                        _prepare_mtp_ar_handoff(
                            self,
                            state,
                            responses[0].logprobs,
                            calibration=calibration and not state.ar_fallback_pending,
                        )
                    return attach_prompt_cache_snapshots(self, responses)
                except _MtpStepFallback as exc:
                    # A correctness fallback cannot schedule a retry from a
                    # potentially damaged boundary, even after a prior park.
                    self.__dict__.pop("_vmlx_mtp_recovery", None)
                    logger.debug(
                        "MTP next() fallback to standard step: %s", exc
                    )
                    try:
                        _log_mtp_stats(
                            (
                                getattr(self, "uids", ["?"])[0]
                                if getattr(self, "uids", None)
                                else "?"
                            ),
                            state.stats,
                            "fallback_to_ar",
                            state.mtp_cache,
                        )
                    except Exception:
                        logger.debug(
                            "MTP fallback telemetry publication failed",
                            exc_info=True,
                        )
                    # Best-effort: drop state so subsequent calls don't try
                    # to resume a half-built MTP cycle from a stale snapshot.
                    if hasattr(self, "_omlx_mtp_state"):
                        try:
                            delattr(self, "_omlx_mtp_state")
                        except AttributeError:
                            pass
        stock_recovery = recovery if recovery is not None and getattr(self, "uids", None) == [recovery.uid] else None
        step_t0 = time.perf_counter() if stock_recovery is not None else 0.0
        responses = original_next(self, *args, **kwargs)
        if (
            stock_recovery is not None
            and responses
        ):
            stock_recovery.observe_standard((time.perf_counter() - step_t0) * 1000.0)
            stats = stock_recovery.parked_stats
            if stats is not None and (stock_recovery.standard_tokens % 16 == 0 or responses[0].finish_reason is not None):
                stats.recovery = stock_recovery.snapshot()
                _publish_native_mtp_stats(stock_recovery.uid, stats, responses[0].finish_reason or "ar")
        return attach_prompt_cache_snapshots(self, responses)

    def patched_batch_generator_next(self, *args, **kwargs):
        prompt_batch = getattr(self, "_prompt_batch", None)
        if prompt_batch is not None:
            for name in ("_vmlx_hybrid_boundary_target", "_vmlx_hybrid_boundary_store"):
                setattr(prompt_batch, name, getattr(self, name, None))
        snapshots = _capture_glm_prompt_boundary_snapshots(self)
        result = original_batch_generator_next(self, *args, **kwargs)
        if snapshots:
            active = getattr(self, "_generation_batch", None)
            if active is not None:
                retained = getattr(active, "_vmlx_prompt_cache_snapshots", None)
                if retained is None:
                    retained = {}
                    active._vmlx_prompt_cache_snapshots = retained
                retained.update(snapshots)
        return result

    def patched_batch_generator_make_batch(self, n, *args, **kwargs):
        batch = original_batch_generator_make_batch(self, n, *args, **kwargs)
        if n != 1 or not _glm_prompt_priming_enabled(batch.model):
            return batch
        from ...native_mtp_prompt_priming import prepare_prompt

        processing = getattr(self, "_currently_processing", None) or []
        if not processing or not getattr(batch, "uids", None):
            return batch
        segments = processing[-1][0]
        prior_tokens = list((getattr(batch, "tokens", None) or [[]])[-1])
        remaining = [int(tok) for segment in segments for tok in segment]
        prompt_tokens = prior_tokens + remaining
        prepare_prompt(
            batch.model,
            request_id=f"text:{batch.uids[-1]}",
            prompt_tokens=prompt_tokens,
            cached_tokens=len(prior_tokens),
            prefix_cache=None,
        )
        return batch

    def patched_batch_generator_remove(self, uids, *args, **kwargs):
        result = original_batch_generator_remove(self, uids, *args, **kwargs)
        active = getattr(self, "_generation_batch", None)
        snapshots = getattr(active, "_vmlx_prompt_cache_snapshots", None)
        if snapshots:
            for uid in uids:
                snapshots.pop(uid, None)
            if not snapshots:
                try:
                    delattr(active, "_vmlx_prompt_cache_snapshots")
                except AttributeError:
                    pass
        return result

    def patched_extend(self, batch, *args, **kwargs):
        # ``BatchGenerator._next()`` builds a fresh single-sequence
        # ``GenerationBatch`` via ``prompt_batch.split(...).generate(...)``
        # then merges it into ``self._generation_batch`` via extend(). The
        # MTP post-init runs on the fresh batch (since that's the one whose
        # __init__ fires with uids=[0]); without this transfer the state
        # would die with the donor instance.
        was_empty = not getattr(self, "uids", None)
        donor_recovery = _text_recovery_for_batch(batch)
        donor_state = getattr(batch, "_omlx_mtp_state", None)
        result = original_extend(self, batch, *args, **kwargs)
        if donor_state is not None and not hasattr(self, "_omlx_mtp_state"):
            self._omlx_mtp_state = donor_state
            try:
                delattr(batch, "_omlx_mtp_state")
            except AttributeError:
                pass
            logger.debug(
                "MTP state transferred from donor batch to host batch (uid=%s)",
                getattr(self, "uids", ["?"])[0] if getattr(self, "uids", None) else "?",
            )
        if was_empty and donor_recovery is not None:
            self._vmlx_mtp_recovery = donor_recovery
            batch.__dict__.pop("_vmlx_mtp_recovery", None)
        _text_recovery_for_batch(self)
        return result

    def patched_filter(self, keep, *args, **kwargs):
        # When the outer scheduler retires this sequence (e.g. EOS detected
        # outside our finish path), it calls filter([]) to drop everything.
        # Surface stats here so the user sees them even when the standard
        # _emit_response finish path doesn't fire.
        state = getattr(self, "_omlx_mtp_state", None)
        result = original_filter(self, keep, *args, **kwargs)
        _text_recovery_for_batch(self)
        if state is not None and not getattr(self, "uids", None):
            # Batch is now empty — log + drop state.
            try:
                _log_mtp_stats(
                    "?",
                    state.stats,
                    getattr(state, "_finish_reason", "external"),
                    state.mtp_cache,
                )
            except Exception:
                pass
            try:
                delattr(self, "_omlx_mtp_state")
            except AttributeError:
                pass
        return result

    GenerationBatch.__init__ = patched_init
    GenerationBatch.next = patched_next
    GenerationBatch.filter = patched_filter
    GenerationBatch.extend = patched_extend
    GenerationBatch._omlx_mtp_patched = True
    BatchGenerator._next = patched_batch_generator_next
    BatchGenerator.remove = patched_batch_generator_remove
    BatchGenerator._make_batch = patched_batch_generator_make_batch
    PromptProcessingBatch.prompt = patched_prompt_processing_prompt
    BatchGenerator._vmlx_glm_prompt_snapshot_patched = True
    _PATCHED = True
    return True


def _capture_glm_prompt_boundary_snapshots(batch_generator: Any) -> dict[int, list[Any]]:
    """Clone GLM typed state for requests poised at their final prompt token.

    mlx-lm deliberately splits the last prompt token into a one-token segment.
    The next ``BatchGenerator._next`` call runs that token through the model
    while constructing ``GenerationBatch``. GLM KDA state cannot be reversed,
    so this is the only exact place to retain the reusable N-1 boundary.
    """

    processing = getattr(batch_generator, "_currently_processing", None) or ()
    prompt_batch = getattr(batch_generator, "_prompt_batch", None)
    if prompt_batch is None or not processing:
        return {}

    split = []
    for index, sequence in enumerate(processing):
        segments = sequence[0]
        if len(segments) == 1 and len(segments[0]) == 1:
            split.append(index)
    if not split:
        return {}

    try:
        from ...models.glm5_next.glm5_next import (
            clone_glm5_next_layer_cache,
        )
    except Exception:
        return {}

    snapshots: dict[int, list[Any]] = {}
    boundary_callback = getattr(
        batch_generator, "_vmlx_prompt_boundary_callback", None
    )
    for index in split:
        try:
            extracted = prompt_batch.extract_cache(index)
            if not extracted or not all(
                type(layer).__name__ in {"Glm5KDACache", "Glm5MLACache"}
                for layer in extracted
            ):
                continue

            uid = int(prompt_batch.uids[index])
            if callable(boundary_callback):
                # Production GLM cannot retain an expanded MLA N-1 image
                # through the final-token allocation beside a 95GB model.
                # The scheduler synchronously detaches/persists this exact
                # boundary to SSD, then returns with no Metal snapshot owner.
                boundary_callback(uid, extracted)
                continue

            # Preserve the old arrays in fresh typed wrappers rather than
            # allocating a second cache image. KDA and legacy MLA replace
            # arrays; compact MLA marks shared capacity lanes copy-on-write so
            # the final prompt token detaches before updating them. In both
            # representations this exact N-1 object graph remains unchanged.
            cloned = [
                clone_glm5_next_layer_cache(layer, copy_fn=lambda value: value)
                for layer in extracted
            ]
            snapshots[uid] = cloned
        except Exception as exc:
            # Cache capture is an optimization. A copy failure must force a
            # clean miss on the next turn, never fail the current generation.
            logger.warning(
                "GLM exact prompt-boundary snapshot failed for uid=%s; "
                "cache store will be skipped: %s",
                prompt_batch.uids[index],
                exc,
            )
    return snapshots


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", model)
    return hasattr(inner, "mtp") and getattr(inner, "mtp", None) is not None


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward`` *and* has an attached MTP head."""
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head (mtp_enabled was False at load time)"
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        return f"batch size {len(uids)} != 1 (continuous batching, MTP off by design)"
    return ""


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # cycles where the FULL draft chain was accepted
    rejects: int = 0  # cycles with at least one rejected draft position
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Depth > 1 requires trimmable caches or explicit accepted-prefix rollback.
    depth: int = 1  # resolved draft depth for this sequence
    starting_depth: int = 1
    depth_ceiling: int = 1
    depth_policy: str = "fixed"
    draft_tokens_proposed: int = 0  # sum of chain lengths across cycles
    draft_tokens_accepted: int = 0  # sum of accepted prefix lengths
    accepted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    drafted_by_depth: List[int] = field(default_factory=lambda: [0, 0, 0])
    seed_main_forwards: int = 0
    verify_main_forwards: int = 0
    mtp_forwards: int = 0
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore
    # MTP-head cache lifecycle. These are observability-only counters.
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
    # A phase can publish repeatedly while parked in AR. Count only deltas,
    # and do not count re-entry as a new logical request.
    totals_published: dict = field(default_factory=dict, repr=False)
    request_counted: bool = False


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # Number of recursively drafted MTP-head pairs that are not yet verifier
    # confirmed. The GLM aligned-cache experiment trims exactly these pairs
    # before recommitting confirmed backbone-hidden/token pairs.
    head_chain_pairs: int = 0

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with the drafts cheaply.
    next_main: Optional[Any] = None

    # Draft chain (depth >= 1). Parallel lists, one entry per chained draft:
    # d1 predicts the token after next_main, d2 the one after d1, ...
    draft_toks: List[Any] = field(default_factory=list)  # each (1,) uint32
    draft_lps: List[Any] = field(default_factory=list)  # each (vocab,) float
    # Filtered (sampler-applied) draft logprobs reused by the next cycle's
    # acceptance ratio + residual sampling.
    draft_accept_lps: List[Any] = field(default_factory=list)  # each (vocab,)
    # Host-side int copies of the draft chain. Empty until the verify cycle's
    # single eval materializes them — drafting itself never syncs the host.
    draft_ids: List[int] = field(default_factory=list)

    # Resolved draft depth for this sequence (1..3).
    depth: int = 1
    depth_ceiling: int = 1
    adaptive_enabled: bool = False
    adaptive_value: NativeMTPAdaptiveValueState = field(
        default_factory=NativeMTPAdaptiveValueState
    )
    # A cost decision never discards a verified cycle mid-flight. It stops
    # drafting, drains the cycle's already-verified emit queue, then primes
    # the stock GenerationBatch pipeline from ``next_main`` at the exact AR
    # cache/token frontier.
    ar_fallback_pending: bool = False
    ar_fallback_reason: Optional[str] = None
    ar_step_ms: float = 0.0
    cycle_span_start: float = 0.0
    recovery: Optional[NativeMTPRecovery] = None
    recovery_probe_started: float = 0.0
    adaptive_cycle_offset: int = 0
    # Policy-independent AR safety valve (runs for fixed depth too); see
    # vmlx_engine/native_mtp_ar_safety.py.
    ar_safety: ArSafetyState = field(default_factory=ArSafetyState)

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)


def _mtp_rate(accepted: int, drafted: int) -> Optional[float]:
    if drafted <= 0:
        return None
    return accepted / drafted


def _native_mtp_payload(
    uid: Any, stats: _MtpStats, finish_reason: str
) -> dict:
    timings = {
        "verify": float(stats.backbone_ms),
        "sample": float(stats.sample_ms),
        "draft": float(stats.mtp_head_ms),
        "cache": float(stats.cache_ops_ms),
    }
    timing_total = sum(timings.values())
    timings["total"] = timing_total
    timings["avg_cycle"] = timing_total / max(1, int(stats.cycles or 0))
    depth_rates = {
        label: _mtp_rate(
            int(stats.accepted_by_depth[index]),
            int(stats.drafted_by_depth[index]),
        )
        for index, label in enumerate(("d1", "d2", "d3"))
    }
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
        "acceptance_rate": _mtp_rate(
            int(stats.draft_tokens_accepted),
            int(stats.draft_tokens_proposed),
        ),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "depth_acceptance_rates": depth_rates,
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
        "profiled_phase_timing": bool(_MTP_PROFILE),
        "published_at": time.time(),
        "fallback_reason": stats.fallback_reason,
        "fallback_cost": {
            "cost_ratio": stats.fallback_cost_ratio,
            "mtp_ms_per_token": stats.fallback_mtp_ms_per_token,
            "ar_step_ms": stats.fallback_ar_step_ms,
        },
    }


def _publish_native_mtp_stats(
    uid: Any, stats: _MtpStats, finish_reason: str
) -> dict:
    global _LAST_NATIVE_MTP

    payload = _native_mtp_payload(uid, stats, finish_reason)
    with _MTP_TELEMETRY_LOCK:
        _LAST_NATIVE_MTP = payload
        if not stats.request_counted:
            _NATIVE_MTP_TOTALS["requests"] += 1
            stats.request_counted = True
        counters = {
            "cycles": stats.cycles,
            "drafted_tokens": stats.draft_tokens_proposed,
            "accepted_tokens": stats.draft_tokens_accepted,
            "mtp_cache_recreated_on_rejects": stats.mtp_cache_recreated_on_rejects,
            "mtp_cache_retained_on_rejects": stats.mtp_cache_retained_on_rejects,
        }
        for key, value in counters.items():
            value = int(value)
            previous = stats.totals_published.get(key, 0)
            _NATIVE_MTP_TOTALS[key] += max(0, value - previous)
            stats.totals_published[key] = max(previous, value)
    return payload


_LAST_NATIVE_MTP_SKIP: dict | None = None


def native_mtp_stats_snapshot() -> dict:
    """Return immutable process-local BatchGenerator MTP acceptance telemetry."""
    with _MTP_TELEMETRY_LOCK:
        totals = dict(_NATIVE_MTP_TOTALS)
        totals["acceptance_rate"] = _mtp_rate(
            int(totals["accepted_tokens"]), int(totals["drafted_tokens"])
        )
        return {
            "last_native_mtp": copy.deepcopy(_LAST_NATIVE_MTP),
            "last_native_mtp_skip": copy.deepcopy(_LAST_NATIVE_MTP_SKIP),
            "native_mtp_totals": totals,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_generation_stream():
    """Return the ``mlx_lm.generate`` module-level generation stream.

    The standard ``GenerationBatch._step`` runs all forward passes inside
    ``mx.stream(generation_stream)``; the MTP cycle does the same so the
    paged cache writes land on the same stream and ordering is preserved.
    The stream lives on the *outer* ``BatchGenerator``, not on
    ``GenerationBatch``, so we read it from the module.

    Note: ``mlx_lm.__init__`` re-exports a ``generate`` *function*, so
    ``import mlx_lm.generate as mlg`` resolves to the function, not the
    module. We use ``sys.modules`` to grab the actual module.
    """
    import sys

    return sys.modules["mlx_lm.generate"].generation_stream


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch: Any) -> bool:
    """Return whether the active batch uses the deterministic sampler path."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        # Seeded requests install a request-local sampler even at temperature
        # zero. Object presence therefore cannot distinguish stochastic from
        # greedy: recognize the explicit argmax contract marker.
        return bool(getattr(gen_batch.samplers[0], "_vmlx_is_greedy", False))
    return bool(getattr(gen_batch.fallback_sampler, "_vmlx_is_greedy", True))


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Thin delegate: the rule lives in ``vmlx_engine.native_mtp_acceptance`` so
    this text path and the MLLM path cannot drift apart.  They already did once
    — /health advertised rejection-sampling acceptance while the MLLM path was
    exact-match only.
    """
    from ...native_mtp_acceptance import accept_lp_for

    return accept_lp_for(sampler, lp)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` speculative tokens."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any], n: int = 1) -> bool:
    """Roll back ``n`` speculative tokens from each layer cache after a
    (possibly partial) draft-chain rejection.

    SSM / linear-attention layers normally expose ``rollback_state`` populated
    by their model adapter; that snapshot restores to the confirmed prefix and
    therefore supports depth 1 only.  A family that records every speculative
    prefix can instead expose ``rollback_speculative(n)`` plus
    ``supports_partial_rollback``. Standard KV caches trim by ``n``. Layers
    that support neither cause the entire MTP step to fail closed.
    """
    # Two-phase: validate EVERY layer before mutating ANY. The old
    # first-refusal-mid-loop return left earlier layers already trimmed —
    # the fallback then continued the standard path on a partially
    # rolled-back cache (silent corruption). Eligibility gating makes a
    # refusal near-unreachable; when it does happen the cache still holds
    # the speculative verify advance, so no continuation is sound — the
    # caller must fail the request loudly (uniform with the MLLM path).
    for c in prompt_cache:
        if callable(getattr(c, "rollback_speculative", None)):
            can_rollback = getattr(c, "can_rollback_speculative", None)
            if callable(can_rollback) and not can_rollback(n):
                return False
            continue
        if getattr(c, "rollback_state", None) is not None:
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        return False
    for c in prompt_cache:
        rollback_speculative = getattr(c, "rollback_speculative", None)
        if callable(rollback_speculative):
            if not rollback_speculative(n):
                return False
            continue
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        c.trim(n)
    return True


def _effective_depth_resolution(gen_batch: Any, *, adaptive_ceiling: bool = False) -> tuple[int, str]:
    """Resolve the draft-chain depth and its owning configuration source.

    Sources, in order: VMLINUX/VMLX_NATIVE_MTP_DEPTH env, the bundle's
    validated ``vmlx_mtp_tuning.json``, default (via
    ``native_mtp_effective_depth``). Depth > 1 additionally requires:
      - every prompt-cache layer trimmable, or explicitly capable of restoring
        an arbitrary accepted speculative prefix, and
      - ``mtp_forward`` supporting ``return_hidden`` (chained drafting feeds
        the head's hidden back as the next step's previous-hidden).
    Hybrid layers without an explicit partial-rollback contract remain D1.
    """
    try:
        from vmlx_engine.native_mtp import native_mtp_effective_depth, native_mtp_max_depth

        depth, source = native_mtp_effective_depth(None)
        # Producer best_depth is an admission recommendation, not a permanent
        # adaptive search limit. Explicit limits and family fallbacks retain
        # their existing meaning. Apply the same rollback checks below to the
        # expanded range; starting depth remains D1 and promotions cost-tested.
        if adaptive_ceiling and _adaptive_depth_enabled() and "vmlx_mtp_tuning.json" in source:
            depth = native_mtp_max_depth()
    except Exception:
        depth = 1
        source = "resolution_error"
    depth = max(1, min(3, int(depth or 1)))
    if depth <= 1:
        return 1, source
    for c in gen_batch.prompt_cache:
        if bool(getattr(c, "supports_partial_rollback", False)) and callable(
            getattr(c, "rollback_speculative", None)
        ):
            continue
        if getattr(c, "rollback_state", None) is not None:
            return 1, source
        if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
            return 1, source
    return depth, source


def _effective_depth(gen_batch: Any) -> int:
    """Resolve the draft-chain depth (1..3) for this sequence."""
    return _effective_depth_resolution(gen_batch)[0]


def _adaptive_mtp_activation_decision(gen_batch: Any) -> tuple[bool, int, str]:
    """Choose whether text-path adaptive MTP may seed this request.

    The MLLM scheduler already enforces the same first-turn contract through
    ``NativeMTPProfileStore``: an unseen workload stays AR and MTP activates
    only from measured evidence. ``GenerationBatch`` has no request-level
    workload/profile owner, so it must not silently spend the user's request
    discovering whether speculation loses. A validated model-local tuning
    record is the text path's persistent measured evidence. Explicit depth
    overrides and fixed policy remain authoritative user choices.
    """
    depth, source = _effective_depth_resolution(gen_batch)
    if not _adaptive_depth_enabled():
        return True, depth, "fixed_policy"
    if str(source).startswith("VML"):
        return True, depth, str(source)
    if "vmlx_mtp_tuning.json" in str(source):
        return True, depth, str(source)
    return False, depth, "adaptive_unseen_ar"


def _adaptive_depth_enabled() -> bool:
    raw = os.environ.get(
        "VMLINUX_NATIVE_MTP_ADAPTIVE_DEPTH",
        os.environ.get("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "1"),
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _adaptive_env_int(default: int, *names: str, minimum: int = 1) -> int:
    raw = next((os.environ.get(name) for name in names if name in os.environ), None)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(int(minimum), value)


def _adaptive_env_float(default: float, *names: str) -> float:
    raw = next((os.environ.get(name) for name in names if name in os.environ), None)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _adaptive_env_flag(default: bool, *names: str) -> bool:
    raw = next((os.environ.get(name) for name in names if name in os.environ), None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _text_mtp_timing_total_ms(stats: _MtpStats) -> float:
    return (
        float(stats.backbone_ms)
        + float(stats.mtp_head_ms)
        + float(stats.sample_ms)
        + float(stats.cache_ops_ms)
    )


def _text_mtp_confirmed_tokens_from_cycles(stats: _MtpStats) -> int:
    # Each completed verify cycle confirms one correction/bonus token plus
    # its accepted draft prefix. Init emits are intentionally excluded: they
    # belong to the seed bridge, not the repeated speculative cycle cost.
    return max(0, int(stats.cycles)) + max(0, int(stats.draft_tokens_accepted))


def _text_mtp_cost_ratio(
    stats: _MtpStats,
    ar_step_ms: float,
    *,
    elapsed_ms: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if ar_step_ms <= 0.0:
        return None
    mtp_ms = (
        float(elapsed_ms)
        if elapsed_ms is not None and float(elapsed_ms) > 0.0
        else _text_mtp_timing_total_ms(stats)
    )
    confirmed = _text_mtp_confirmed_tokens_from_cycles(stats)
    if mtp_ms <= 0.0 or confirmed <= 0:
        return None
    mtp_ms_per_token = mtp_ms / confirmed
    return mtp_ms_per_token / ar_step_ms, mtp_ms_per_token


def _text_mtp_maybe_ar_safety_fallback(request_id: str, state: _MtpState) -> bool:
    """Policy-INDEPENDENT on-the-fly AR safety valve (text lane).

    Unlike ``_text_mtp_maybe_cost_fallback`` below (adaptive-only by design),
    this runs for FIXED D1/D2/D3 as well: a fixed depth is the user's choice
    of how much to speculate, never a request to keep losing to plain AR.
    Windowed, context-scaled, stall-robust; see native_mtp_ar_safety.
    """
    if state.ar_fallback_pending:
        return False
    cycles = int(state.stats.cycles)
    probing = state.recovery_probe_started > 0.0
    measured = state.recovery is not None and state.recovery.ar_ms > 0.0
    if state.recovery is not None:
        state.stats.recovery = state.recovery.snapshot()
    trip = ar_safety_step(
        state.ar_safety,
        cycles=cycles,
        emitted=cycles + int(state.stats.draft_tokens_accepted),
        now=time.perf_counter(),
        seed_ar_ms=float(state.ar_step_ms or 0.0),
        primed=str(state.stats.prompt_prime_source or "unprimed") != "unprimed",
        probe=probing,
        scale_context=not measured,
        baseline_measured=measured,
        margin=0.9 if probing else None,
    )
    if probing and trip is None and cycles >= 12:
        # Include seed/priming, draft, verification and delivery wall time.
        # Warmup cannot hide a probe whose complete cost loses to stock AR.
        confirmed = 2 + cycles + int(state.stats.draft_tokens_accepted)
        cost = (time.perf_counter() - state.recovery_probe_started) * 1000 / confirmed
        if cost <= state.ar_step_ms * 0.9:
            state.recovery_probe_started = 0.0
            logger.info("MTP[%s] AR re-entry D1 accepted: total_ms_per_token=%.3f ar=%.3f", request_id, cost, state.ar_step_ms)
        else:
            state.ar_fallback_pending = True
            state.ar_fallback_reason = "reentry_total_cost"
            state.stats.fallback_reason = state.ar_fallback_reason
            state.stats.fallback_mtp_ms_per_token = cost
            state.stats.fallback_ar_step_ms = state.ar_step_ms
            return True
    if trip is None:
        return False
    prior_depth = int(state.depth or 1)
    if prior_depth > 1:
        # Judge the adjacent lower rung on its own fresh window. Fixed Dn is
        # the ceiling, not permission to skip an unmeasured intermediate rung.
        state.depth = prior_depth - 1
        state.stats.depth = state.depth
        state.ar_safety.reset(cycles)
        state.adaptive_value.last_change_cycle = cycles + state.adaptive_cycle_offset
        state.adaptive_value.active_probe_origin = 0
        state.adaptive_value.active_probe_target = 0
        logger.info("MTP[%s] %s", request_id, trip.log_text(prior_depth, target_depth=state.depth))
        return False
    state.ar_fallback_pending = True
    state.ar_fallback_reason = trip.reason(prior_depth)
    state.stats.fallback_reason = state.ar_fallback_reason
    state.stats.fallback_mtp_ms_per_token = trip.mtp_ms_per_tok
    state.stats.fallback_ar_step_ms = trip.ar_baseline
    logger.info("MTP[%s] %s", request_id, trip.log_text(prior_depth))
    return True


def _text_mtp_maybe_cost_fallback(
    request_id: str,
    state: _MtpState,
    *,
    now: float,
) -> bool:
    """Mark a measured MTP -> AR handoff without breaking fixed-depth UI.

    An explicit calibrated cost experiment may override any depth policy.
    Otherwise the runtime gate applies only to an already-active adaptive
    request. Fixed D1/D2/D3 is a ceiling governed by the shared AR safety
    valve; this optional cost experiment does not replace that controller.
    """

    explicit = _adaptive_env_flag(
        False,
        "VMLINUX_NATIVE_MTP_COST_FALLBACK",
        "VMLX_NATIVE_MTP_COST_FALLBACK",
    )
    runtime_adaptive = state.adaptive_enabled and _adaptive_env_flag(
        True,
        "VMLINUX_NATIVE_MTP_RUNTIME_COST_GATE",
        "VMLX_NATIVE_MTP_RUNTIME_COST_GATE",
    )
    if state.ar_fallback_pending or not (explicit or runtime_adaptive):
        return False

    if explicit:
        ar_step_ms = _adaptive_env_float(
            0.0,
            "VMLINUX_NATIVE_MTP_AR_STEP_MS",
            "VMLX_NATIVE_MTP_AR_STEP_MS",
            "VMLINUX_NATIVE_MTP_COST_AR_STEP_MS",
            "VMLX_NATIVE_MTP_COST_AR_STEP_MS",
        )
        threshold = _adaptive_env_float(
            1.0,
            "VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD",
            "VMLX_NATIVE_MTP_COST_RATIO_THRESHOLD",
        )
        minimum_cycles = _adaptive_env_int(
            8,
            "VMLINUX_NATIVE_MTP_COST_MIN_CYCLES",
            "VMLX_NATIVE_MTP_COST_MIN_CYCLES",
            minimum=1,
        )
        elapsed_ms = None
        reason_prefix = "calibrated_cost"
    else:
        ar_step_ms = float(state.ar_step_ms or 0.0)
        threshold = _adaptive_env_float(
            1.25,
            "VMLINUX_NATIVE_MTP_RUNTIME_COST_MARGIN",
            "VMLX_NATIVE_MTP_RUNTIME_COST_MARGIN",
        )
        minimum_cycles = _adaptive_env_int(
            48,
            "VMLINUX_NATIVE_MTP_RUNTIME_COST_MIN_CYCLES",
            "VMLX_NATIVE_MTP_RUNTIME_COST_MIN_CYCLES",
            minimum=8,
        )
        elapsed_ms = (
            max(0.0, (float(now) - float(state.cycle_span_start)) * 1000.0)
            if state.cycle_span_start > 0.0
            else None
        )
        reason_prefix = "runtime_cost"

    if int(state.stats.cycles) < minimum_cycles:
        return False
    ratio_and_cost = _text_mtp_cost_ratio(
        state.stats,
        ar_step_ms,
        elapsed_ms=elapsed_ms,
    )
    if ratio_and_cost is None:
        return False
    ratio, mtp_ms_per_token = ratio_and_cost
    if ratio < threshold:
        return False

    state.ar_fallback_pending = True
    state.ar_fallback_reason = (
        f"{reason_prefix} cost_ratio={ratio:.3f}>=threshold={threshold:.3f} "
        f"mtp_ms_per_token={mtp_ms_per_token:.2f} ar_step_ms={ar_step_ms:.2f}"
    )
    state.stats.fallback_reason = state.ar_fallback_reason
    state.stats.fallback_cost_ratio = ratio
    state.stats.fallback_mtp_ms_per_token = mtp_ms_per_token
    state.stats.fallback_ar_step_ms = ar_step_ms
    logger.info(
        "MTP[%s] adaptive runtime -> AR after cycles=%d: %s",
        request_id,
        state.stats.cycles,
        state.ar_fallback_reason,
    )
    return True


def _adaptive_value_min_samples() -> int:
    return _adaptive_env_int(
        8,
        "VMLINUX_NATIVE_MTP_VALUE_MIN_SAMPLES",
        "VMLX_NATIVE_MTP_VALUE_MIN_SAMPLES",
        minimum=2,
    )


def _adaptive_arm_cycle(state: _MtpState, *, now: float) -> None:
    arm_depth_cycle(state.adaptive_value, depth=state.depth, now=now)


def _adaptive_finish_cycle(
    request_id: str,
    state: _MtpState,
    *,
    completed_depth: int,
    accepted: int,
    now: float,
) -> None:
    """Record one real cycle and select the next depth by wall value.

    The completed verify/rollback cycle always stays at its original depth.
    Any decision applies only to the next draft chain, where no speculative
    state exists yet. Both fixed-ceiling and adaptive modes use measured
    neighbour probes; the configured ceiling and initial choice remain distinct.
    """
    window = _adaptive_env_int(
        16,
        "VMLINUX_NATIVE_MTP_VALUE_WINDOW",
        "VMLX_NATIVE_MTP_VALUE_WINDOW",
        minimum=4,
    )
    finish_armed_depth_cycle(
        state.adaptive_value,
        depth=completed_depth,
        accepted_drafts=accepted,
        cycle=int(state.stats.cycles) + state.adaptive_cycle_offset,
        now=now,
        window=window,
    )
    # A mandatory safety demotion owns this cycle. Also finish the D1 AR
    # re-entry qualification before considering any higher-depth experiment.
    if state.depth != completed_depth or state.recovery_probe_started > 0.0:
        return
    minimum_samples = _adaptive_value_min_samples()
    decision = choose_depth_by_value(
        state.adaptive_value,
        current_depth=state.depth,
        depth_ceiling=state.depth_ceiling,
        cycle=int(state.stats.cycles) + state.adaptive_cycle_offset,
        minimum_samples=minimum_samples,
        cooldown_cycles=_adaptive_env_int(
            8,
            "VMLINUX_NATIVE_MTP_VALUE_COOLDOWN_CYCLES",
            "VMLX_NATIVE_MTP_VALUE_COOLDOWN_CYCLES",
            minimum=2,
        ),
        probe_interval_cycles=_adaptive_env_int(
            48,
            "VMLINUX_NATIVE_MTP_VALUE_PROBE_INTERVAL_CYCLES",
            "VMLX_NATIVE_MTP_VALUE_PROBE_INTERVAL_CYCLES",
            minimum=4,
        ),
        hysteresis=_adaptive_env_float(
            0.05,
            "VMLINUX_NATIVE_MTP_VALUE_HYSTERESIS",
            "VMLX_NATIVE_MTP_VALUE_HYSTERESIS",
        ),
        raise_min_acceptance=_adaptive_env_float(
            0.88,
            "VMLINUX_NATIVE_MTP_VALUE_RAISE_MIN_ACCEPT",
            "VMLX_NATIVE_MTP_VALUE_RAISE_MIN_ACCEPT",
        ),
    )
    state.stats.adaptive_depth_value = adaptive_value_snapshot(
        state.adaptive_value,
        minimum_samples=minimum_samples,
    )
    if decision is None:
        return
    target = max(1, min(state.depth_ceiling, int(decision.target_depth)))
    current = state.depth
    state.depth = target
    state.stats.depth = target
    if target != current:
        state.ar_safety.reset(int(state.stats.cycles))
    logger.info(
        "MTP[%s] adaptive value %s D%d -> D%d after cycles=%d: %s",
        request_id,
        decision.event,
        current,
        target,
        state.stats.cycles,
        decision.reason,
    )


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop ``rollback_state`` snapshots after a draft is accepted."""
    for c in prompt_cache:
        commit_speculative = getattr(c, "commit_speculative", None)
        if callable(commit_speculative):
            commit_speculative()
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


def _glm_aligned_head_cache_enabled(gen_batch: Any) -> bool:
    """Return whether exact ``glm5_next`` may use aligned head history.

    This remains an opt-in experiment until a same-source live wall-clock row
    beats AR without changing the target output. No other family is eligible.
    """

    model = getattr(gen_batch, "model", None)
    inner = getattr(model, "language_model", model)
    model_type = getattr(inner, "model_type", None)
    if model_type is None:
        config = getattr(inner, "config", None)
        if isinstance(config, dict):
            model_type = config.get("model_type")
    if str(model_type or "") != "glm5_next":
        return False
    value = os.environ.get(
        "VMLINUX_GLM5_ALIGNED_MTP_HEAD_CACHE",
        os.environ.get("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "0"),
    ).strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def _glm_prompt_priming_enabled(model: Any) -> bool:
    inner = getattr(model, "language_model", model)
    model_type = getattr(inner, "model_type", None)
    if model_type is None:
        config = getattr(inner, "config", None)
        if isinstance(config, dict):
            model_type = config.get("model_type")
    if str(model_type or "") != "glm5_next":
        return False
    value = os.environ.get(
        "VMLINUX_GLM5_MTP_PROMPT_PRIMING",
        os.environ.get("VMLX_GLM5_MTP_PROMPT_PRIMING", "0"),
    ).strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def _trim_glm_head_chain(state: "_MtpState") -> bool:
    """Trim only the unverified recursive pairs from GLM's MTP cache."""

    count = max(0, int(getattr(state, "head_chain_pairs", 0) or 0))
    if count == 0:
        state.head_chain_pairs = 0
        return True
    layers = [layer for layer in (state.mtp_cache or []) if layer is not None]
    if not layers or any(
        not (
            hasattr(layer, "is_trimmable")
            and layer.is_trimmable()
            and callable(getattr(layer, "trim", None))
        )
        for layer in layers
    ):
        return False
    for layer in layers:
        if int(layer.trim(count)) != count:
            return False
    state.head_chain_pairs = 0
    return True


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------

def _mtp_context_extent(gen_batch: Any) -> Tuple[int, str]:
    """Read logical context before the seed, including SSD-restored positions.

    GenerationBatch.tokens can contain only the admitted suffix. Never infer
    the full context from an allocated KV buffer's shape. Native MTP owns a
    singleton here; a vector offset for several rows is not one logical extent.
    This value is telemetry only: the AR verdict scales by cycle-wall cost.
    """
    offsets = []
    pending = list(getattr(gen_batch, "prompt_cache", ()) or ())
    seen = set()
    while pending:
        entry = pending.pop()
        if id(entry) in seen:
            continue
        seen.add(id(entry))
        try:
            pending.extend(getattr(entry, "caches", ()) or ())
            value = getattr(entry, "offset", None)
            if type(value) is int:
                offset = value
            elif value is not None and getattr(value, "size", None) == 1:
                offset = value.reshape(()).item()
                if type(offset) is not int:
                    continue
            else:
                continue
            if offset >= 0:
                offsets.append(offset)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
    if offsets:
        return max(offsets), "logical_cache_offset"
    try:
        return len(gen_batch.tokens[0]), "admitted_tokens_fallback"
    except (AttributeError, IndexError, TypeError):
        return 0, "unknown"


def _post_init_mtp(gen_batch: Any, *, recovery: Optional[NativeMTPRecovery] = None) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into vMLX's MTP cycle.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    recovery_t0 = time.perf_counter() if recovery is not None else 0.0
    seed_context_tokens, seed_context_source = _mtp_context_extent(gen_batch)
    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state. This is a real
    # AR-shaped step at the request's actual context length, so its completed
    # wall time is the adaptive cost gate's local baseline.
    try:
        mx.synchronize()  # a batch partner's pending work must not inflate the seed
    except Exception:
        pass
    seed_step_t0 = time.perf_counter()
    with mx.stream(_get_generation_stream()):
        logits, hidden = gen_batch.model(
            main_tok[:, None], cache=gen_batch.prompt_cache, return_hidden=True
        )

    next_main_logits = logits[:, -1, :]  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _logprobs(next_main_logits)
    next_main_tok = sampler(next_main_lp)  # (1,)

    mx.eval(main_tok, next_main_tok)
    seed_step_ms = (time.perf_counter() - seed_step_t0) * 1000.0

    state = _MtpState()
    state.recovery = recovery
    state.stats.request_counted = recovery is not None
    calibration_resume = recovery is not None and recovery.calibrating
    state.recovery_probe_started = 0.0 if calibration_resume else recovery_t0
    state.ar_step_ms = recovery.ar_ms if recovery is not None else seed_step_ms
    state.stats.seed_ar_step_ms = seed_step_ms
    state.stats.seed_context_tokens = seed_context_tokens
    state.stats.seed_context_source = seed_context_source
    state.ar_safety.prompt_tokens = seed_context_tokens
    logger.info(
        "MTP[%s] AR baseline: seed_step_ms=%.3f effective_ms=%.3f source=%s context_tokens=%d context_source=%s",
        gen_batch.uids[0], seed_step_ms, state.ar_step_ms,
        "productive_ar" if recovery is not None else "single_seed",
        seed_context_tokens, seed_context_source,
    )
    if _glm_prompt_priming_enabled(gen_batch.model):
        from ...native_mtp_prompt_priming import prime_stats, take_primed

        prime_before_seam = prime_stats(gen_batch.model)
        primed = take_primed(
            gen_batch.model, gen_batch.prompt_cache, main_tok
        )
    else:
        prime_before_seam = None
        primed = None
    if primed is None:
        state.mtp_cache = gen_batch.model.make_mtp_cache()
        if prime_before_seam is not None:
            if prime_before_seam.get("active"):
                state.stats.prompt_prime_source = "capture_discarded_at_seam"
            elif prime_before_seam.get("plan_armed"):
                reason = str(
                    (prime_before_seam.get("last") or {}).get("reason")
                    or "unknown"
                )
                state.stats.prompt_prime_source = f"armed_no_capture:{reason}"
            else:
                state.stats.prompt_prime_source = "not_armed"
    else:
        state.mtp_cache, state.stats.prompt_primed_pairs = primed
        state.stats.prompt_prime_source = "cold_prompt"
    state.depth_ceiling = recovery.depth_ceiling if recovery is not None else _effective_depth_resolution(gen_batch, adaptive_ceiling=True)[0]
    state.adaptive_enabled = recovery.adaptive if recovery is not None else _adaptive_depth_enabled()
    state.depth = 1 if recovery is not None or state.adaptive_enabled else state.depth_ceiling
    if calibration_resume:
        state.depth = recovery.resume_depth
        if recovery.adaptive_state is not None:
            state.adaptive_value = recovery.adaptive_state
            state.adaptive_cycle_offset = recovery.adaptive_cycle_offset
            # Retain probe debt/deadlines, not pre-refresh wall measurements.
            for samples in state.adaptive_value.samples_by_depth:
                samples.clear()
            state.adaptive_value.armed_at = 0.0
            state.adaptive_value.armed_depth = 0
    if recovery is not None:
        if not calibration_resume:
            recovery.attempts += 1
        recovery.calibrating = False
        recovery.next_calibration_token = int(gen_batch._num_tokens[0]) + 768
        state.stats.recovery = recovery.snapshot()
    state.stats.depth = state.depth
    state.stats.starting_depth = state.depth
    state.stats.depth_ceiling = state.depth_ceiling
    state.stats.depth_policy = "adaptive" if state.adaptive_enabled else "fixed"
    state.stats.seed_main_forwards = 1
    state.next_main = _ensure_uint32(next_main_tok)
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )
    gen_batch._omlx_mtp_state = state

    # Draft chain: the head sees (hidden_at_main, next_main_tok) and proposes
    # d1..dN for the first verify cycle forward([next_main, d1..dN]).
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    _adaptive_arm_cycle(state, now=time.perf_counter())
    _draft_chain(
        gen_batch,
        state,
        hidden_at_main,
        state.next_main,
        prev_buf=prev_buf,
    )
    if recovery is not None:
        recovery.resume_wall_ms += (time.perf_counter() - recovery_t0) * 1000.0


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------

def _mtp_ar_handoff_ready(gen_batch: Any, state: _MtpState) -> Tuple[bool, str]:
    """Check the stock-AR boundary after the last verified emit drains."""
    if state.queue:
        return False, "pending_queue"
    if state.next_main is None:
        return False, "missing_next_main"
    try:
        next_id = int(state.next_main.tolist()[0])
    except Exception:
        return False, "invalid_next_main"
    tokens = getattr(gen_batch, "tokens", None) or []
    if not tokens or not tokens[0]:
        return False, "missing_visible_token"
    if next_id != int(tokens[0][-1]):
        return False, "next_main_mismatch"
    for layer in getattr(gen_batch, "prompt_cache", ()):
        if getattr(layer, "rollback_state", None) is not None:
            return False, "pending_rollback_state"
    return True, "ready"


def _text_recovery_for_batch(gen_batch: Any) -> Optional[NativeMTPRecovery]:
    recovery = getattr(gen_batch, "_vmlx_mtp_recovery", None)
    if recovery is not None and recovery.uid not in (getattr(gen_batch, "uids", None) or ()):
        gen_batch.__dict__.pop("_vmlx_mtp_recovery", None)
        return None
    return recovery


def _text_mtp_maybe_resume(gen_batch: Any, recovery: NativeMTPRecovery) -> bool:
    """Re-enter only from a productive, singleton stock-AR boundary.

    A seed failure is fatal: post-init may already have advanced native caches,
    so do not catch it and pretend the old stock state is still usable.
    """
    if (
        getattr(gen_batch, "_omlx_mtp_state", None) is not None
        or not recovery.ready
        or not _is_mtp_eligible(gen_batch)
        or getattr(gen_batch, "uids", None) != [recovery.uid]
        or getattr(gen_batch, "_next_tokens", None) is None
    ):
        return False
    # Do not seed two tokens on a request which is about to finish.
    if int(gen_batch.max_tokens[0]) - int(gen_batch._num_tokens[0]) < 3:
        return False
    if any(getattr(c, "rollback_state", None) is not None for c in gen_batch.prompt_cache):
        return False
    calibration = recovery.calibrating
    _post_init_mtp(gen_batch, recovery=recovery)
    logger.info("MTP[%s] AR -> D%d %s: ceiling=D%d ar_ms_per_token=%.3f cooldown=%d", recovery.uid, gen_batch._omlx_mtp_state.depth, "calibration resume" if calibration else "recovery probe", recovery.depth_ceiling, recovery.ar_ms, recovery.cooldown)
    return True


def _text_mtp_calibration_due(gen_batch: Any, state: _MtpState) -> bool:
    """Refresh at a drained productive frontier, never inside a value probe."""
    recovery = state.recovery
    deadline = recovery.next_calibration_token if recovery is not None else 128
    return (
        not state.ar_fallback_pending
        and not state.queue
        and state.recovery_probe_started <= 0.0
        and not state.adaptive_value.active_probe_target
        and getattr(gen_batch, "uids", None) is not None
        and len(gen_batch.uids) == 1
        and int(gen_batch._num_tokens[0]) >= deadline
        # Nine AR steps plus a meaningful resumed cycle; avoid terminal-only work.
        and int(gen_batch.max_tokens[0]) - int(gen_batch._num_tokens[0]) >= 32
    )


def _prepare_mtp_ar_handoff(
    gen_batch: Any,
    state: _MtpState,
    last_logprobs: Any,
    *,
    calibration: bool = False,
) -> None:
    """Prime stock ``GenerationBatch.next`` without re-emitting a token.

    The verified queue's last correction/bonus is already visible, but has
    not entered the backbone cache. Stock ``_step`` normally consumes and
    returns that same token. Temporarily remove its MTP-side history append,
    run the stock step once to consume/reappend it and prepare the following
    sample, then discard MTP state. The next public ``next()`` therefore emits
    the following AR token exactly once.
    """
    handoff_t0 = time.perf_counter()
    ready, reason = _mtp_ar_handoff_ready(gen_batch, state)
    if not ready:
        raise RuntimeError(f"native MTP AR fallback unsafe: {reason}")

    visible_id = int(gen_batch.tokens[0].pop())
    gen_batch._next_tokens = _ensure_uint32(state.next_main)
    gen_batch._next_logprobs = [last_logprobs]
    try:
        consumed, _ = gen_batch._step()
    except Exception:
        # The cache may already have advanced, so silent recovery is unsafe.
        if not gen_batch.tokens[0] or int(gen_batch.tokens[0][-1]) != visible_id:
            gen_batch.tokens[0].append(visible_id)
        raise
    if list(consumed) != [visible_id]:
        raise RuntimeError(
            "native MTP AR fallback consumed the wrong boundary token: "
            f"expected={visible_id} actual={list(consumed)}"
        )

    recovery = _text_recovery_for_batch(gen_batch)
    if recovery is None:
        recovery = NativeMTPRecovery(gen_batch.uids[0], state.depth_ceiling, state.adaptive_enabled)
        gen_batch._vmlx_mtp_recovery = recovery
    if calibration:
        recovery.park_for_calibration(depth=state.depth)
        recovery.adaptive_state = copy.deepcopy(state.adaptive_value)
        recovery.adaptive_cycle_offset = state.adaptive_cycle_offset + int(state.stats.cycles)
    else:
        recovery.park(failed_probe=state.recovery_probe_started > 0.0)
    recovery.handoff_wall_ms += (time.perf_counter() - handoff_t0) * 1000.0
    for name in ("cycles", "draft_tokens_proposed", "draft_tokens_accepted", "seed_main_forwards", "verify_main_forwards", "mtp_forwards", "init_emits", "draft_emits", "bonus_emits", "verify_emits"):
        recovery.completed_mtp[name] = recovery.completed_mtp.get(name, 0) + getattr(state.stats, name)
    state.stats.recovery = recovery.snapshot()
    state.stats.depth = 0  # Effective AR, not the last configured MTP rung.
    recovery.parked_stats = state.stats
    _log_mtp_stats(
        gen_batch.uids[0] if gen_batch.uids else "?",
        state.stats,
        "calibration_ar" if calibration else "fallback_to_ar",
        state.mtp_cache,
    )
    if getattr(gen_batch, "_omlx_mtp_state", None) is state:
        delattr(gen_batch, "_omlx_mtp_state")
    logger.info(
        "MTP[%s] fallback to AR after verified queue drain: %s",
        gen_batch.uids[0] if gen_batch.uids else "?",
        "productive baseline calibration" if calibration else state.ar_fallback_reason or "adaptive runtime cost",
    )

def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    if state.stats.cycles == 0 and state.cycle_span_start <= 0.0:
        state.cycle_span_start = time.perf_counter()
    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(
    uid: Any,
    stats: "_MtpStats",
    finish_reason: str,
    mtp_cache: Any = None,
) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> accept=<A>/<C> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    stats.mtp_head_cache = native_mtp_cache_snapshot(mtp_cache)
    payload = _publish_native_mtp_stats(uid, stats, finish_reason)
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    if stats.cycles > 0:
        rate_str = f"{stats.accepts / stats.cycles * 100:.1f}%"
    else:
        rate_str = "n/a"
    if payload["acceptance_rate"] is not None:
        tok_rate_str = f"{payload['acceptance_rate'] * 100:.1f}%"
    else:
        tok_rate_str = "n/a"
    logger.info(
        "MTP[%s] finish=%s depth=%d tokens=%d cycles=%d full-accept=%d/%d (%s) "
        "draft-tokens=%d/%d (%s) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        stats.depth,
        total_emits,
        stats.cycles,
        stats.accepts,
        stats.cycles,
        rate_str,
        stats.draft_tokens_accepted,
        stats.draft_tokens_proposed,
        tok_rate_str,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------

def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Run one verify cycle over the draft chain ``d1..dN``.

    One backbone forward over ``[next_main, d1..dN]`` (N+1 tokens,
    ``n_confirmed=1``), longest-prefix acceptance, then one new draft chain
    from the last confirmed position. Populates ``state.queue`` with
    ``k`` accepted drafts + 1 correction/bonus token (k in 0..N), rolls the
    cache back by ``N - k`` on partial/full rejection, and refreshes
    ``state.next_main`` / draft-chain lists for the next cycle.

    Depth 1 reproduces the original 2-token draft+bonus cycle exactly.
    """

    import mlx.core as mx

    if state.next_main is None or not state.draft_toks:
        raise _MtpStepFallback("verify cycle entered without next_main / drafts")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)
    n = len(state.draft_toks)

    # TurboQuant live-encode crossing — text-path twin of the MLLM guard in
    # _native_mtp_should_snapshot_layer: trim() rewinds offset only, so if a
    # TQ layer's one-time compress() fires inside this verify advance and the
    # chain partially rejects, draft KV stays baked into the compressed
    # buffers — silent corruption. The crossing happens at most once per
    # layer; fall back to the standard step for it (state drops, AR
    # continues — conservative and correct).
    advance = n + 1
    for c in gen_batch.prompt_cache:
        compress_after = getattr(c, "compress_after", None)
        if not compress_after:
            continue
        try:
            threshold = int(compress_after)
            not_compressed = int(getattr(c, "_compressed_tokens", 0) or 0) == 0
            offset = int(getattr(c, "offset", 0) or 0)
        except (TypeError, ValueError):
            raise _MtpStepFallback("tq live-encode crossing state unreadable")
        if threshold > 0 and not_compressed and offset + advance >= threshold:
            raise _MtpStepFallback("tq live-encode crossing in verify advance")

    inputs = mx.concatenate([state.next_main] + list(state.draft_toks))  # (n+1,)

    # Update the token buffer per position so logits processors see the same
    # history shape as standard autoregressive decode. prev_bufs[i] is the
    # buffer state after consuming inputs[0..i].
    prev_bufs: List[Any] = []
    if procs is not None:
        prev_bufs.append(
            gen_batch._token_context[0].update_and_fetch(state.next_main)
        )
        for d in state.draft_toks:
            prev_bufs.append(gen_batch._token_context[0].update_and_fetch(d))

    # --- backbone forward + sample (single eval point) ---
    # Dispatch backbone, processors, logprobs, and sampler all on stream
    # without forcing intermediate evaluation. The single ``mx.eval`` after
    # sampling resolves the whole graph in one stall instead of two.
    t0 = time.perf_counter()
    with mx.stream(_get_generation_stream()):
        logits, hidden = gen_batch.model(
            inputs[None, :],
            cache=gen_batch.prompt_cache,
            return_hidden=True,
            n_confirmed=1,
        )
    _pbar(logits, hidden)  # profiling: isolate real verify-forward GPU cost
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if procs is None:
        # The (n+1, vocab) slab is already contiguous — skip n+1 slices + a
        # concatenate. (Measured neutral on Hy3; kept because it is simply
        # less work. The per-cycle cost that makes MTP a loss on this MoE is
        # the verify forward itself, not this bookkeeping — see below.)
        combined_logits = logits[0]
    else:
        combined_logits = mx.concatenate(
            [
                _apply_processors(procs, prev_bufs[i], logits[:, i, :])
                for i in range(n + 1)
            ],
            axis=0,
        )
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    # Sample per row: a single batched sampler() call measured identical on
    # Hy3 and would change RNG consumption for stochastic samplers, so keep
    # the row-wise draws the depth-1 path has always used.
    sampled = [sampler(combined_lp[i : i + 1]) for i in range(n + 1)]
    mx.eval(*sampled, *state.draft_toks)
    sampled_ids = [int(t.tolist()[0]) for t in sampled]
    # Draft ids materialize here, off the same eval — the chain itself never
    # forces a host sync (one sync per cycle instead of one per draft).
    if len(state.draft_ids) != len(state.draft_toks):
        state.draft_ids = [int(t.tolist()[0]) for t in state.draft_toks]

    # Longest-prefix acceptance. Position i's logits verify draft i+1
    # (0-indexed: combined_lp[i] is the target distribution for d_{i+1}).
    k = 0
    while k < n:
        d_id = state.draft_ids[k]
        if is_greedy:
            ok = sampled_ids[k] == d_id
        else:
            verify_accept_lp = _accept_lp_for(sampler, combined_lp[k : k + 1])
            log_accept = (
                verify_accept_lp[0, d_id].item()
                - state.draft_accept_lps[k][d_id].item()
            )
            _uniform = getattr(sampler, "_vmlx_random_uniform", None)
            draw = _uniform() if _uniform is not None else random.random()
            ok = log_accept >= 0 or draw < math.exp(log_accept)
        if not ok:
            break
        k += 1
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    state.stats.cycles += 1
    state.stats.verify_main_forwards += 1
    state.stats.draft_tokens_proposed += n
    state.stats.draft_tokens_accepted += k
    for index in range(n):
        state.stats.drafted_by_depth[index] += 1
        if index < k:
            state.stats.accepted_by_depth[index] += 1
    full_accept = k == n
    if full_accept:
        state.stats.accepts += 1
    else:
        state.stats.rejects += 1

    # --- cache rollback / cleanup (timed) ---
    t0 = time.perf_counter()
    if full_accept:
        _clear_rollback(gen_batch.prompt_cache)
    else:
        if not _restore_or_trim_caches(gen_batch.prompt_cache, n - k):
            # The cache still holds the rejected speculative advance and
            # cannot be repaired — continuing (even via the standard-step
            # fallback) would generate from corrupt state. Fail the request
            # loudly, uniform with the MLLM path's rollback refusal.
            try:
                _log_mtp_stats(
                    (
                        getattr(gen_batch, "uids", ["?"])[0]
                        if getattr(gen_batch, "uids", None)
                        else "?"
                    ),
                    state.stats,
                    "rollback_refused",
                    state.mtp_cache,
                )
            except Exception:
                logger.debug(
                    "MTP rollback-refusal telemetry publication failed",
                    exc_info=True,
                )
            raise RuntimeError(
                "native MTP cache rejected rollback (text path) — "
                "speculative verify advance cannot be undone"
            )
        state.stats.mtp_cache_retained_on_rejects += 1
        if procs is not None:
            _trim_token_buffer(gen_batch, n - k)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # --- queue emits: k accepted drafts + 1 correction/bonus ---
    for i in range(k):
        state.queue.append((state.draft_ids[i], state.draft_lps[i], "draft"))

    if full_accept:
        emit_id = sampled_ids[n]
        emit_lp = combined_lp[n : n + 1].squeeze(0)
        source = "bonus"
    elif is_greedy:
        emit_id = sampled_ids[k]
        emit_lp = combined_lp[k : k + 1].squeeze(0)
        source = "verify"
    else:
        # Residual sample on the *filtered* distributions so the sample
        # comes from `max(p_target_filt - p_draft_filt, 0)`. emit_lp stays
        # the raw verify lp so downstream logprobs reporting is consistent
        # with non-MTP paths.
        verify_accept_lp = _accept_lp_for(sampler, combined_lp[k : k + 1])
        emit_id, _ = _residual_sample(
            verify_accept_lp, state.draft_accept_lps[k], sampler=sampler
        )
        emit_lp = combined_lp[k : k + 1].squeeze(0)
        source = "verify"
    state.queue.append((emit_id, emit_lp, source))

    # --- new draft chain from the last confirmed position ---
    emit_tok = mx.array([emit_id], dtype=mx.uint32)
    state.next_main = emit_tok
    request_label = str(gen_batch.uids[0]) if gen_batch.uids else "?"
    if _text_mtp_maybe_ar_safety_fallback(
        request_label, state
    ) or _text_mtp_maybe_cost_fallback(
        request_label,
        state,
        now=time.perf_counter(),
    ):
        # No speculative work may outlive the decision. The main cache is at
        # the verified frontier and the queue contains only confirmed tokens;
        # the stock AR pipeline is primed only after that queue drains.
        state.draft_toks = []
        state.draft_lps = []
        state.draft_accept_lps = []
        state.draft_ids = []
        state.head_chain_pairs = 0
        return
    # Safety gets first refusal. An optional neighbour probe must never reset
    # the losing window before the mandatory descent has had a chance to act.
    _adaptive_finish_cycle(
        request_label, state, completed_depth=n, accepted=k,
        now=time.perf_counter(),
    )
    _adaptive_arm_cycle(state, now=time.perf_counter())
    if _glm_aligned_head_cache_enabled(gen_batch):
        state.stats.mtp_head_cache_policy = "glm_aligned"
        if not _trim_glm_head_chain(state):
            raise RuntimeError(
                "GLM aligned MTP head cache could not trim unverified pairs"
            )
        confirmed_tokens = mx.concatenate(
            [_ensure_uint32(tok).reshape(1) for tok in state.draft_toks[:k]]
            + [emit_tok.reshape(1)]
        ).reshape(1, k + 1)
        _draft_chain_aligned(
            gen_batch,
            state,
            hidden[:, : k + 1, :],
            confirmed_tokens,
            prev_buf=prev_bufs[k] if procs is not None else None,
        )
        return
    _draft_chain(
        gen_batch,
        state,
        hidden[:, k : k + 1, :],
        emit_tok,
        prev_buf=prev_bufs[k] if procs is not None else None,
    )


# ---------------------------------------------------------------------------
# Helpers used by the verify cycle.
# ---------------------------------------------------------------------------

def _draft_chain(
    gen_batch: Any,
    state: _MtpState,
    anchor_hidden: Any,
    t0_tok: Any,
    prev_buf: Optional[Any],
) -> None:
    """Draft ``state.depth`` chained tokens with the MTP head.

    d1 = head(anchor_hidden, t0); d_{i+1} = head(h_i, d_i) where h_i is the
    head's own (post-final-norm) hidden from step i — the same recursion
    vLLM uses for multi-step MTP with a single head. Replaces the previous
    single-draft ``_step_mtp``. The head cache is never rolled back (loose
    history by design — verify guarantees correctness; the cache only
    shapes draft quality).

    Fills ``state.draft_toks / draft_lps / draft_accept_lps / draft_ids``.
    """
    import time

    import mlx.core as mx

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    state.draft_toks = []
    state.draft_lps = []
    state.draft_accept_lps = []
    state.draft_ids = []

    t_start = time.perf_counter()
    hidden = anchor_hidden
    cur_tok = t0_tok
    proc_ctx = prev_buf
    for _ in range(max(1, state.depth)):
        state.stats.mtp_forwards += 1
        next_ids = cur_tok.reshape(1, 1)
        with mx.stream(_get_generation_stream()):
            mtp_logits, mtp_hidden = gen_batch.model.mtp_forward(
                hidden, next_ids, state.mtp_cache, return_hidden=True
            )
            mtp_logits_2d = mtp_logits[:, -1, :]
        if procs is not None and proc_ctx is not None:
            proc_ctx = mx.concatenate([proc_ctx, _ensure_uint32(cur_tok)])
            mtp_logits_2d = _apply_processors(procs, proc_ctx, mtp_logits_2d)
        new_lp = _logprobs(mtp_logits_2d)
        new_tok = sampler(new_lp)
        # Filtered draft lp — what the sampler actually drew from; the verify
        # cycle's acceptance ratio uses this so the math matches the sampling
        # distribution rather than raw softmax.
        new_accept_lp = _accept_lp_for(sampler, new_lp)

        state.draft_toks.append(_ensure_uint32(new_tok))
        state.draft_lps.append(new_lp.squeeze(0))
        state.draft_accept_lps.append(new_accept_lp.squeeze(0))

        hidden = mtp_hidden[:, -1:, :]
        cur_tok = state.draft_toks[-1]
    # The chain stays lazy: the next step feeds the draft *array* forward, so
    # no host copy is needed here. ``draft_ids`` materializes in the verify
    # cycle's single eval — one sync per cycle instead of one per draft
    # (mirrors the MLLM generator's ``_native_mtp_materialize_draft_ids``).
    state.draft_ids = []
    _pbar(*state.draft_toks)  # profiling: isolate real MTP-head GPU cost
    state.stats.mtp_head_ms += (time.perf_counter() - t_start) * 1000
    if _glm_aligned_head_cache_enabled(gen_batch):
        state.stats.mtp_head_cache_policy = "glm_aligned"
        state.head_chain_pairs = max(0, len(state.draft_toks) - 1)
    else:
        state.stats.mtp_head_cache_policy = "loose"
        state.head_chain_pairs = 0


def _draft_chain_aligned(
    gen_batch: Any,
    state: _MtpState,
    confirmed_hidden: Any,
    confirmed_tokens: Any,
    prev_buf: Optional[Any],
) -> None:
    """Commit confirmed GLM pairs and draft the next chain in one pass.

    ``confirmed_hidden[:, i]`` is verifier-backbone state for the token before
    ``confirmed_tokens[:, i]``. The final row's logits are therefore the next
    level-1 draft. Deeper rows remain recursive and are marked unverified so
    the next cycle removes them before committing its confirmed prefix.
    """

    import time

    import mlx.core as mx

    if int(confirmed_hidden.shape[1]) != int(confirmed_tokens.shape[1]):
        raise RuntimeError("GLM aligned MTP commit pair lengths disagree")
    if int(confirmed_tokens.shape[1]) < 1:
        raise RuntimeError("GLM aligned MTP commit cannot be empty")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    state.draft_toks = []
    state.draft_lps = []
    state.draft_accept_lps = []
    state.draft_ids = []

    t_start = time.perf_counter()
    state.stats.mtp_forwards += 1
    with mx.stream(_get_generation_stream()):
        mtp_logits, mtp_hidden = gen_batch.model.mtp_forward(
            confirmed_hidden,
            confirmed_tokens,
            state.mtp_cache,
            return_hidden=True,
        )
        logits_2d = mtp_logits[:, -1, :]
    proc_ctx = prev_buf
    last_confirmed = confirmed_tokens[:, -1].reshape(1)
    if procs is not None and proc_ctx is not None:
        proc_ctx = mx.concatenate([proc_ctx, _ensure_uint32(last_confirmed)])
        logits_2d = _apply_processors(procs, proc_ctx, logits_2d)
    draft_lp = _logprobs(logits_2d)
    draft_tok = _ensure_uint32(sampler(draft_lp))
    state.draft_toks.append(draft_tok)
    state.draft_lps.append(draft_lp.squeeze(0))
    state.draft_accept_lps.append(
        _accept_lp_for(sampler, draft_lp).squeeze(0)
    )

    hidden = mtp_hidden[:, -1:, :]
    cur_tok = draft_tok
    for _ in range(1, max(1, state.depth)):
        state.stats.mtp_forwards += 1
        with mx.stream(_get_generation_stream()):
            mtp_logits, mtp_hidden = gen_batch.model.mtp_forward(
                hidden,
                cur_tok.reshape(1, 1),
                state.mtp_cache,
                return_hidden=True,
            )
            logits_2d = mtp_logits[:, -1, :]
        if procs is not None and proc_ctx is not None:
            proc_ctx = mx.concatenate([proc_ctx, _ensure_uint32(cur_tok)])
            logits_2d = _apply_processors(procs, proc_ctx, logits_2d)
        draft_lp = _logprobs(logits_2d)
        cur_tok = _ensure_uint32(sampler(draft_lp))
        state.draft_toks.append(cur_tok)
        state.draft_lps.append(draft_lp.squeeze(0))
        state.draft_accept_lps.append(
            _accept_lp_for(sampler, draft_lp).squeeze(0)
        )
        hidden = mtp_hidden[:, -1:, :]

    state.head_chain_pairs = max(0, len(state.draft_toks) - 1)
    state.draft_ids = []
    _pbar(*state.draft_toks)
    state.stats.mtp_head_ms += (time.perf_counter() - t_start) * 1000


def _residual_sample(
    verify_lp_2d: Any, draft_lp_1d: Any, *, sampler=None
) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    This is vMLX's local fallback for speculative rejection at temperature.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    from ...native_mtp_acceptance import residual_sample

    return residual_sample(verify_lp_2d, draft_lp_1d, sampler=sampler)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------

def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if stats is not None:
            state = getattr(gen_batch, "_omlx_mtp_state", None)
            _log_mtp_stats(
                gen_batch.uids[0],
                stats,
                finish_reason,
                getattr(state, "mtp_cache", None),
            )
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]

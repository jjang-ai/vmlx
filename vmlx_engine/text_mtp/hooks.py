# SPDX-License-Identifier: Apache-2.0
"""Where the text MTP lane enters mlx-lm's batch generator.

Wrapped methods (originals kept and always called):

``GenerationBatch.__init__``
    after mlx-lm's own first step, a singleton batch with a head may seed the
    lane (``lane.seed_lane``); otherwise the reason is published for the panel.
``GenerationBatch.next``
    while a lane is attached, tokens come from it; parked requests may resume
    from AR; everything else is mlx-lm's step under the AR kernel scope.
``GenerationBatch.extend`` / ``filter``
    mlx-lm builds each new request in a fresh batch and merges it into the
    live one, so the lane and any parked-AR record move with the request;
    a batch that empties publishes and drops its lane.
``BatchGenerator._next`` / ``_make_batch`` / ``remove`` and
``PromptProcessingBatch.prompt``
    GLM-5.3 prompt priming and exact N-1 prompt-boundary snapshots, and the
    hybrid prefill boundary capture used by the scheduler's SSD companion.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..metal.affine_moe_pair_decode import affine_moe_ar_scope
from . import glm_seam, lane, lane_policy, lane_stats
from ._marks import is_marked, mark

logger = logging.getLogger(__name__)

# Diagnostics: keep the head resident (same memory footprint as a lane run)
# but decode with mlx-lm's step, isolating the lane from the head's footprint.
HEAD_RESIDENT_AR = bool(os.environ.get("VMLX_MTP_BYPASS"))

_installed = False
_SNAPSHOTS = "_vmlx_prompt_cache_snapshots"


def _attach_prompt_snapshots(batch: Any, responses):
    """Hand each response its request's exact prompt-boundary snapshot."""
    snapshots = getattr(batch, _SNAPSHOTS, None)
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
        batch.__dict__.pop(_SNAPSHOTS, None)
    return responses


def _count_vectorized_kda_layers(model: Any) -> int:
    backbone = getattr(model, "model", None)
    return sum(
        int(
            bool(getattr(layer, "is_linear", False))
            and bool(getattr(getattr(layer, "self_attn", None), "_vectorized_speculative_verify", False))
        )
        for layer in getattr(backbone, "layers", ())
    )


def install_generation_hooks() -> bool:
    """Wrap mlx-lm's generation classes once per process."""
    global _installed
    if _installed:
        return True
    try:
        from mlx_lm.generate import BatchGenerator, GenerationBatch, PromptProcessingBatch
    except ImportError:
        logger.debug("mlx_lm.generate.GenerationBatch not importable")
        return False
    if is_marked(GenerationBatch):
        _installed = True
        return True

    stock = {
        "init": GenerationBatch.__init__,
        "next": GenerationBatch.next,
        "filter": GenerationBatch.filter,
        "extend": GenerationBatch.extend,
        "gen_next": BatchGenerator._next,
        "gen_remove": BatchGenerator.remove,
        "gen_make_batch": BatchGenerator._make_batch,
        "prompt": PromptProcessingBatch.prompt,
    }

    def prompt(self, tokens):
        from ..native_mtp_prompt_priming import capture_requested

        model = getattr(self, "model", None)
        if not glm_seam.prompt_priming_enabled(model) or not capture_requested(model):
            from ..utils.hybrid_prefill_capture import prompt_with_hybrid_capture

            return prompt_with_hybrid_capture(self, tokens, stock["prompt"])
        self.model = glm_seam.PromptHiddenTap(model)
        try:
            return stock["prompt"](self, tokens)
        finally:
            self.model = model

    def batch_init(self, *args, **kwargs):
        # mlx-lm's constructor step consumes the final prompt token; with GLM
        # priming armed, that forward must also reach the head history, or the
        # primed history stops one token short and is rightly refused later.
        args = list(args)
        model = kwargs.get("model") if "model" in kwargs else (args[0] if args else None)
        tap_final_prompt = False
        if model is not None and glm_seam.prompt_priming_enabled(model):
            from ..native_mtp_prompt_priming import capture_requested

            tap_final_prompt = capture_requested(model)
        if tap_final_prompt:
            tap = glm_seam.PromptHiddenTap(model)
            if "model" in kwargs:
                kwargs["model"] = tap
            else:
                args[0] = tap
        stock["init"](self, *args, **kwargs)
        if tap_final_prompt:
            self.model = model
        if HEAD_RESIDENT_AR:
            return
        if not lane.lane_eligible(self):
            uids = getattr(self, "uids", None)
            # An empty batch is BatchGenerator's own startup placeholder.
            reason = lane.lane_ineligible_reason(self) if uids else ""
            if reason:
                logger.debug("MTP path not active: %s", reason)
                lane_stats.record_skip({"uid": str(uids[0]), "reason": reason})
            return
        activate, depth, why = lane_policy.activation_decision(self)
        uid = getattr(self, "uids", ["?"])[0]
        if not activate:
            lane_stats.record_skip({"uid": str(uid), "reason": why, "configured_depth": depth})
            logger.info("MTP path stays AR for uid=%s seed=%s configured_depth=%d", uid, why, depth)
            return
        try:
            lane.seed_lane(self)
            vectorized = _count_vectorized_kda_layers(self.model)
            if vectorized:
                logger.info("GLM vectorized KDA verify active for uid=%s layers=%d", uid, vectorized)
            logger.info("MTP path activated for uid=%s (model has mtp_forward, batch=1)", uid)
        except lane.LaneAbort as exc:
            logger.warning("MTP post-init fallback: %s", exc)

    def batch_next(self, *args, **kwargs):
        recovery = lane.parked_recovery(self)
        if recovery is not None:
            lane.maybe_resume_from_ar(self, recovery)
        active = lane.lane_of(self) if lane.lane_eligible(self) else None
        if active is not None:
            try:
                responses = lane.lane_step(self, active)
                calibration = lane.calibration_due(self, active)
                if (
                    (active.ar_fallback_pending or calibration)
                    and not active.queue
                    and lane.lane_of(self) is active
                    and responses
                    and responses[0].finish_reason is None
                ):
                    lane.hand_off_to_ar(
                        self,
                        active,
                        responses[0].logprobs,
                        calibration=calibration and not active.ar_fallback_pending,
                    )
                return _attach_prompt_snapshots(self, responses)
            except lane.LaneAbort as exc:
                # A step that could not run says nothing good about the
                # boundary: no later re-entry for this request.
                self.__dict__.pop(lane.RECOVERY_ATTR, None)
                logger.debug("MTP next() fallback to standard step: %s", exc)
                try:
                    lane_stats.finish_record(
                        self.uids[0] if getattr(self, "uids", None) else "?",
                        active.stats,
                        "fallback_to_ar",
                        active.mtp_cache,
                    )
                except Exception:
                    logger.debug("MTP fallback telemetry publication failed", exc_info=True)
                lane.detach_lane(self)
        parked = recovery if recovery is not None and getattr(self, "uids", None) == [recovery.uid] else None
        started = time.perf_counter() if parked is not None else 0.0
        # Seeding and verification above run outside the AR-only kernels.
        with affine_moe_ar_scope():
            responses = stock["next"](self, *args, **kwargs)
        if parked is not None and responses:
            parked.observe_standard((time.perf_counter() - started) * 1000.0)
            stats = parked.parked_stats
            finished = responses[0].finish_reason
            if stats is not None and (parked.standard_tokens % 16 == 0 or finished is not None):
                stats.recovery = parked.snapshot()
                lane_stats.publish(parked.uid, stats, finished or "ar")
        return _attach_prompt_snapshots(self, responses)

    def batch_extend(self, batch, *args, **kwargs):
        was_empty = not getattr(self, "uids", None)
        donor_recovery = lane.parked_recovery(batch)
        donor_lane = lane.lane_of(batch)
        result = stock["extend"](self, batch, *args, **kwargs)
        if donor_lane is not None and lane.lane_of(self) is None:
            setattr(self, lane.LANE_ATTR, donor_lane)
            lane.detach_lane(batch)
            logger.debug(
                "MTP state transferred from donor batch to host batch (uid=%s)",
                self.uids[0] if getattr(self, "uids", None) else "?",
            )
        if was_empty and donor_recovery is not None:
            setattr(self, lane.RECOVERY_ATTR, donor_recovery)
            batch.__dict__.pop(lane.RECOVERY_ATTR, None)
        lane.parked_recovery(self)
        return result

    def batch_filter(self, keep, *args, **kwargs):
        # A request retired from outside the lane (e.g. EOS seen by the
        # scheduler) still publishes its record here.
        active = lane.lane_of(self)
        result = stock["filter"](self, keep, *args, **kwargs)
        lane.parked_recovery(self)
        if active is not None and not getattr(self, "uids", None):
            try:
                lane_stats.finish_record(
                    "?", active.stats, getattr(active, "_finish_reason", "external"), active.mtp_cache
                )
            except Exception:
                pass
            lane.detach_lane(self)
        return result

    def generator_next(self, *args, **kwargs):
        prompt_batch = getattr(self, "_prompt_batch", None)
        if prompt_batch is not None:
            for name in ("_vmlx_hybrid_boundary_target", "_vmlx_hybrid_boundary_store"):
                setattr(prompt_batch, name, getattr(self, name, None))
        snapshots = glm_seam.capture_boundary_snapshots(self)
        result = stock["gen_next"](self, *args, **kwargs)
        if snapshots:
            live = getattr(self, "_generation_batch", None)
            if live is not None:
                held = getattr(live, _SNAPSHOTS, None)
                if held is None:
                    held = {}
                    setattr(live, _SNAPSHOTS, held)
                held.update(snapshots)
        return result

    def generator_make_batch(self, n, *args, **kwargs):
        batch = stock["gen_make_batch"](self, n, *args, **kwargs)
        if n != 1 or not glm_seam.prompt_priming_enabled(batch.model):
            return batch
        processing = getattr(self, "_currently_processing", None) or []
        if not processing or not getattr(batch, "uids", None):
            return batch
        from ..native_mtp_prompt_priming import prepare_prompt

        prior = list((getattr(batch, "tokens", None) or [[]])[-1])
        remaining = [int(token) for segment in processing[-1][0] for token in segment]
        prepare_prompt(
            batch.model,
            request_id=f"text:{batch.uids[-1]}",
            prompt_tokens=prior + remaining,
            cached_tokens=len(prior),
            prefix_cache=None,
        )
        return batch

    def generator_remove(self, uids, *args, **kwargs):
        result = stock["gen_remove"](self, uids, *args, **kwargs)
        live = getattr(self, "_generation_batch", None)
        snapshots = getattr(live, _SNAPSHOTS, None)
        if snapshots:
            for uid in uids:
                snapshots.pop(uid, None)
            if not snapshots:
                live.__dict__.pop(_SNAPSHOTS, None)
        return result

    GenerationBatch.__init__ = batch_init
    GenerationBatch.next = batch_next
    GenerationBatch.filter = batch_filter
    GenerationBatch.extend = batch_extend
    mark(GenerationBatch)
    BatchGenerator._next = generator_next
    BatchGenerator.remove = generator_remove
    BatchGenerator._make_batch = generator_make_batch
    PromptProcessingBatch.prompt = prompt
    BatchGenerator._vmlx_glm_prompt_snapshot_patched = True
    _installed = True
    return True

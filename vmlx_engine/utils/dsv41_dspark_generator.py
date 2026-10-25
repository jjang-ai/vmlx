# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark speculative-decode generator for the vMLX engine.

Selected (scheduler) only when a bundle's dSpark draft is attached as ``model._dspark`` (default-on
for bundles that ship the draft; ``VMLX_DSV41_DSPARK=0`` disables it) and max_num_seqs == 1. It subclasses
``SingleBatchGenerator`` to reuse its request lifecycle (insert/remove/Response/stop machinery) and
overrides only the decode path: instead of SBG's one-token engine-cache forward, it drives the
validated ``jang_tools.dsv41.dspark.DSparkRunner.stream`` generator on ``model.runtime`` (which owns
its own runtime ``Session`` + checkpoint/restore and the no-refeed fast path — DECODE_DSPARK_METHODS
M1). Output is token-identical to plain greedy; speculation only changes throughput (code ~1.9x).

Greedy text only: a request carrying vision tensors, a non-greedy sampler, logits processors or a
logprobs request falls back to the inherited AR path (per request), so those turns serve correctly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlx.core as mx

from .single_batch_generator import Response, SingleBatchGenerator

logger = logging.getLogger(__name__)


def _adaptive_enabled() -> bool:
    return os.environ.get("VMLX_DSV41_DSPARK_ADAPTIVE", "1").strip().lower() not in {"", "0", "false", "off", "no"}


class DSV41DSparkGenerator(SingleBatchGenerator):
    """SingleBatchGenerator that decodes V4.1 via DSpark speculation when a draft is attached."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        from jang_tools.dsv41.dspark import DSparkRunner

        self._dspark = getattr(model, "_dspark", None)
        self._runtime = getattr(model, "runtime", None)
        self._runner = None
        if self._dspark is not None and self._runtime is not None:
            self._runner = DSparkRunner(self._runtime, self._dspark, adaptive=_adaptive_enabled())
            logger.info("DSV4.1 DSpark generator active (adaptive=%s, bs=%s)",
                        _adaptive_enabled(), getattr(self._dspark, "bs", "?"))
        else:
            logger.warning("DSV4.1 DSpark generator constructed without a draft/runtime; AR fallback")

    # --- per-request routing -------------------------------------------------
    @staticmethod
    def _is_vl(req) -> bool:
        return bool(getattr(req, "pixel_values", None) is not None
                    or getattr(req, "pixel_values_videos", None) is not None
                    or getattr(req, "dsv41_vl_images", None) is not None)

    def _is_greedy(self, req) -> bool:
        """DSpark verification is greedy-exact only: it accepts a draft iff it equals the backbone's
        argmax. Any sampler other than plain argmax (temperature/top-p/top-k/min-p), any logits
        processor (repetition / frequency / presence penalty, logit_bias) or a logprobs request must
        take the inherited AR path, which applies them per token. The scheduler tags greedy samplers
        (``_vmlx_greedy`` / ``_vmlx_accepts_logits`` are set only for temperature == 0)."""
        sampler = req.sampler or self.sampler
        greedy = bool(getattr(sampler, "_vmlx_greedy", False) or getattr(sampler, "_vmlx_accepts_logits", False))
        if not greedy:
            return False
        if req.logits_processors:
            return False
        if self._logprobs_required(req):
            return False
        return True

    def _use_dspark(self, req) -> bool:
        return self._runner is not None and not self._is_vl(req) and self._is_greedy(req)

    # --- decode override ------------------------------------------------------
    def _start_request(self, req) -> Response:
        if not self._use_dspark(req):
            return super()._start_request(req)  # VL / sampling / processors / logprobs -> AR path
        if req.cache is None:
            req.cache = self._make_new_cache()
        self._refresh_thread_stream()
        req.prefill_usage = None
        req.prefill_started = time.perf_counter()
        req.prompt_processed = True
        # DSpark drives the ENGINE cache (req.cache duck-types the runtime Session) for BOTH cold and
        # prefix-HIT requests, so DSpark output populates the prefix cache AND a restored prefix is
        # CONTINUED by DSpark (not re-prefilled). On a hit the scheduler restores req.cache and passes
        # only the uncached tail as prompt_tokens; prefill() forwards it from the session's absolute
        # pos, in prefill_step_size chunks, and calls back at the exact N-1 prompt boundary -- the
        # state the scheduler stores under prompt[:-1] (same contract as the AR path).
        from jang_tools.dsv4_1.mlx_model import _relink

        seq = _relink(req.cache)
        boundary = {}

        def _on_boundary(_sess):
            boundary["snapshot"] = self._clone_dsv41_prompt_boundary(req)

        prompt = list(req.prompt_tokens)
        with self._stream_context():
            sess, pending, c = self._runner.prefill(
                prompt, session=seq, on_boundary=_on_boundary,
                chunk_size=self.prefill_step_size,
            )
            mx.eval(pending)
        req.prompt_cache_snapshot = boundary.get("snapshot")
        req.context_tokens.extend(prompt)
        req.prefill_usage = {
            "tokens": len(prompt),
            "seconds": time.perf_counter() - req.prefill_started,
            "scope": "model_prefill_and_prompt_state",
        }
        req.prefill_started = None
        req._dspark_stats = {}
        req._dspark_stream = self._runner.decode_stream(
            sess, pending, c, req.max_tokens, eos_ids=set(t for t in self.stop_tokens if isinstance(t, int)),
            stats=req._dspark_stats,
        )
        req._dspark_on = True
        req.next_token = self._pull(req)  # first committed token (int) or None
        return self._emit_dspark(req)

    def _pull(self, req):
        gen = getattr(req, "_dspark_stream", None)
        if gen is None:
            return None
        try:
            with self._stream_context():  # same stream the prefill ran on
                return int(next(gen))
        except StopIteration:
            return None

    def _emit_dspark(self, req) -> Response:
        current = req.next_token
        if current is None:
            # Stream produced nothing (e.g. empty generation): finish cleanly.
            self._request = None
            return Response(uid=req.uid, token=(req.output_tokens[-1] if req.output_tokens else 0),
                            logprobs=None, finish_reason="stop", current_state=req.current_state,
                            match_sequence=req.match_sequence, prompt_cache=None,
                            all_tokens=list(req.context_tokens), prompt_cache_snapshot=None,
                            prefill_usage=req.prefill_usage, mixed_swa_boundary=None,
                            stable_root_snapshot=None, boundary_checkpoints=None)
        # Pull the NEXT token now (one-token lookahead parity with SBG) to know if the stream ended.
        req.next_token = self._pull(req)
        req.output_tokens.append(current)
        req.context_tokens.append(current)
        req.finish_reason = self._finish_reason_after_token(req, current)
        if req.finish_reason is None and req.next_token is None:
            req.finish_reason = "stop"  # stream ended without a stop/length trigger
        resp = Response(uid=req.uid, token=current, logprobs=None, finish_reason=req.finish_reason,
                        current_state=req.current_state, match_sequence=req.match_sequence,
                        prompt_cache=req.cache, all_tokens=list(req.context_tokens),
                        prompt_cache_snapshot=getattr(req, 'prompt_cache_snapshot', None), prefill_usage=req.prefill_usage,
                        mixed_swa_boundary=None, stable_root_snapshot=None, boundary_checkpoints=None)
        if req.finish_reason is not None:
            self._request = None
        return resp

    def _yield_current_and_schedule_next(self, req, *, prompt_cache_snapshot=None) -> Response:
        if getattr(req, "_dspark_on", False):
            return self._emit_dspark(req)
        return super()._yield_current_and_schedule_next(req, prompt_cache_snapshot=prompt_cache_snapshot)

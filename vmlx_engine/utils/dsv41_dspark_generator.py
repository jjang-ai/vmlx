# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark speculative-decode generator for the vMLX engine.

Selected (scheduler) only when a bundle's dSpark draft is attached as ``model._dspark`` (default-on
for bundles that ship the draft; ``VMLX_DSV41_DSPARK=0`` disables it) and max_num_seqs == 1. It subclasses
``SingleBatchGenerator`` to reuse its request lifecycle (insert/remove/Response/stop machinery) and
overrides only the decode path: it prefills the request's ENGINE cache with
``jang_tools.dsv41.dspark.DSparkRunner.prefill(session=<engine DeepseekV41Cache list>)`` (so the prompt
boundary snapshot and the prefix cache see the same state as the AR path) and then drives
``DSparkRunner.decode_stream`` on that session (no-refeed fast path, DECODE_DSPARK_METHODS M1).

Greedy requests (temperature == 0) produce the backbone's greedy tokens; speculation only changes
throughput. A temperature > 0 request runs exact sampled speculation (jang_tools.dsv41.dspark.SpecSampler:
the AR sampler's temperature / top_p / min_p / top_k distribution, Gumbel noise keyed by the request seed and
the absolute position, coupled drafting); the adaptive governor's plain AR steps draw from the same
position-keyed noise, so a fixed ``seed`` reproduces the same tokens whatever the governor chooses.
A request carrying vision tensors, logits processors (repetition / frequency / presence penalty, logit_bias)
or a logprobs request falls back to the inherited AR path (per request).
"""

from __future__ import annotations

import logging
import os
import time

import mlx.core as mx

from .single_batch_generator import Response, SingleBatchGenerator

logger = logging.getLogger(__name__)


def _adaptive_enabled() -> bool:
    return os.environ.get("VMLX_DSV41_DSPARK_ADAPTIVE", "1").strip().lower() not in {"", "0", "false", "off", "no"}


# Prefill chunk with the draft resident. The draft adds 8.3 GB, which puts a 2048-token chunk's
# active memory plus the allocator cache over Metal's recommended working set (222.7 GiB on a
# 256 GB machine), and the GPU slows down. Draft resident, 12k-token cold prompt: chunk 2048
# 545 pp/s, 1536 643, 1024 708, 768 684. Without the draft 2048 stays best (~770).
# VMLX_DSV41_DSPARK_PREFILL_STEP overrides.
_DRAFT_PREFILL_STEP = 1024


def _draft_prefill_step(configured: int) -> int:
    try:
        override = int(os.environ.get("VMLX_DSV41_DSPARK_PREFILL_STEP", "0"))
    except ValueError:
        override = 0
    if override > 0:
        return override
    return min(int(configured), _DRAFT_PREFILL_STEP)


class DSV41DSparkGenerator(SingleBatchGenerator):
    """SingleBatchGenerator that decodes V4.1 via DSpark speculation when a draft is attached."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        from jang_tools.dsv41.dspark import DSparkRunner

        # Applies to the DSpark prefill and to this generator's inherited AR prefill (VL turns), which
        # run with the same draft resident.
        self.prefill_step_size = _draft_prefill_step(self.prefill_step_size)

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

    @staticmethod
    def _sampler_temperature(sampler):
        """The temperature the scheduler built ``sampler`` with, or None when it is unrecorded."""
        spec = getattr(sampler, "_vmlx_sampling", None)
        if isinstance(spec, dict) and spec.get("temperature") is not None:
            return float(spec["temperature"])
        temp = getattr(sampler, "temp", None)  # vmlx_engine.sampling stamps every sampler it builds
        if isinstance(temp, (int, float)) and not isinstance(temp, bool):
            return float(temp)
        return None

    def _is_greedy(self, req) -> bool:
        """True only for a temperature == 0 request without logits processors or logprobs.

        Greedy means argmax: the scheduler tags it ``_vmlx_greedy`` (vmlx_engine.sampling tags it
        ``_vmlx_is_greedy``) and records temperature 0. ``_vmlx_accepts_logits`` is NOT a greedy marker:
        make_sampler also sets it on the compact top-k sampler (temperature > 0, top_k > 0, min_p == 0),
        and treating that as greedy decoded a sampled top_k request greedily. Logits processors
        (repetition / frequency / presence penalty, logit_bias) and logprobs requests take the inherited
        AR path, which applies them per token."""
        sampler = req.sampler or self.sampler
        if getattr(sampler, "_vmlx_greedy", False) is True or getattr(sampler, "_vmlx_is_greedy", False) is True:
            greedy = True
        else:
            greedy = self._sampler_temperature(sampler) == 0.0
        if not greedy:
            return False
        if req.logits_processors:
            return False
        if self._logprobs_required(req):
            return False
        return True

    def _spec_sampler(self, req):
        """(ok, SpecSampler | None): greedy -> (True, None); a temperature > 0 sampler built by the scheduler's
        make_sampler (tagged _vmlx_sampling) without logits processors / logprobs -> (True, SpecSampler);
        anything else -> (False, None), the AR path."""
        if self._is_greedy(req):
            return True, None
        if os.environ.get("VMLX_DSV41_DSPARK_SAMPLED", "1").strip().lower() in {"0", "false", "off", "no"}:
            return False, None
        sampler = req.sampler or self.sampler
        spec = getattr(sampler, "_vmlx_sampling", None)
        if not isinstance(spec, dict) or not float(spec.get("temperature", 0.0)) > 0:
            return False, None
        if req.logits_processors or self._logprobs_required(req):
            return False, None
        from jang_tools.dsv41.dspark import SpecSampler

        return True, SpecSampler(spec["temperature"], top_p=spec.get("top_p", 0.0), min_p=spec.get("min_p", 0.0),
                                 top_k=spec.get("top_k", 0), seed=spec.get("seed"))

    def _use_dspark(self, req) -> bool:
        if self._runner is None or self._is_vl(req):
            return False
        ok, sampler = self._spec_sampler(req)
        req._dspark_sampler = sampler
        return ok

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
            snap = self._clone_dsv41_prompt_boundary(req)
            # Carry the draft-window seed of the boundary's last positions, so a later prefix hit
            # on this entry drafts with a full window (DSparkRunner.prefill reseeds from it).
            tail_fn = getattr(self._dspark, "main_tail", None)
            tail = tail_fn() if callable(tail_fn) else None
            if snap and tail is not None and int(tail[1]) + int(tail[0].shape[1]) == int(snap[0].offset):
                snap[0]._seq.dspark_tail = (mx.array(tail[0]), int(tail[1]))
                mx.eval(snap[0]._seq.dspark_tail[0])
            boundary["snapshot"] = snap

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
        # SingleBatchGenerator emits at least one token whatever max_tokens says (its length check runs
        # after the first token); keep that contract so the stream always yields a first token.
        req._dspark_stream = self._runner.decode_stream(
            sess, pending, c, max(1, int(req.max_tokens)), eos_ids=set(t for t in self.stop_tokens if isinstance(t, int)),
            stats=req._dspark_stats, sampler=getattr(req, "_dspark_sampler", None),
        )
        req._dspark_on = True
        req.next_token = self._pull(req)  # first committed token (always an int, max_new >= 1)
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
            # decode_stream runs with max_new >= 1, so its first pull always yields a token, and a later
            # call happens only after a token whose lookahead pull returned a token (a None lookahead
            # finishes the request below). Never fabricate a token here.
            raise RuntimeError(f"DSV4.1 DSpark stream for request {req.uid} produced no token")
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
            self._log_dspark_stats(req)
        return resp

    @staticmethod
    def _log_dspark_stats(req) -> None:
        st = getattr(req, "_dspark_stats", None) or {}
        if not st:
            return
        logger.info(
            "DSV4.1 DSpark request %s: emitted=%s steps=%s spec_steps=%s ar_steps=%s accept=%s/%s "
            "spec_ms_per_tok=%s ar_ms_per_tok=%s",
            req.uid, st.get("emitted"), st.get("steps"), st.get("spec_steps"), st.get("ar_steps"),
            st.get("accepted"), st.get("drafted"),
            None if st.get("spec_s_per_tok") is None else round(st["spec_s_per_tok"] * 1000, 2),
            None if st.get("ar_s_per_tok") is None else round(st["ar_s_per_tok"] * 1000, 2),
        )

    def _yield_current_and_schedule_next(self, req, *, prompt_cache_snapshot=None) -> Response:
        if getattr(req, "_dspark_on", False):
            return self._emit_dspark(req)
        return super()._yield_current_and_schedule_next(req, prompt_cache_snapshot=prompt_cache_snapshot)

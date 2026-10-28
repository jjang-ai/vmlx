# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding on the engine's single-request text lane (MiniMax-M3 / M3.1 with its bundled drafter).

Selected by the scheduler when the loaded model carries a DSpark runtime (``model._vmlx_dspark_runtime``, attached at
load by ``vmlx_engine.dspark.runtime.attach_text_lane`` for a bundle that ships ``dspark/``; ``VMLX_DSPARK=0`` keeps
plain decoding) and max_num_seqs == 1. It subclasses ``SingleBatchGenerator`` for the request lifecycle (insert /
remove / Response / stop machinery, prompt snapshot, prefix-cache hits) and replaces only the decode of eligible
requests with ``vmlx_engine.dspark.iter_dspark`` on the request's ENGINE cache:

- the prompt is forwarded exactly as the plain M3 path forwards it (``MiniMaxM3DSparkTarget.prefill`` = the chunking
  of ``SingleBatchGenerator._prefill_full_and_sample``), so the first token, the prompt snapshot handed to the prefix
  cache and the cache itself equal plain decoding's;
- every committed token is chosen from the target's logits with this generator's own decode pick (the request's
  sampler applied as ``_sample_from_logits`` applies it); verify rows equal decode steps bit for bit, so a greedy
  request produces plain decoding's tokens, and a sampled request draws every token from the target's distribution
  ("sample-then-match": drafts change how many draws one forward makes, never what they are drawn from);
- a prefix-cache hit continues on the restored cache; the drafter context is carried over from the previous request
  when its tokens cover the restored prefix, else the drafter starts empty at the restored position (drafts only);
- Adaptive picks every round's width (``ARFallback``), up to the runtime's ladder top.

Requests with media, logits processors (penalties, logit_bias, guided output) or logprobs take the inherited plain
decode path. ``VMLX_DSPARK_SAMPLED=0`` also sends sampled requests there.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict

import mlx.core as mx

from .single_batch_generator import Response, SingleBatchGenerator

logger = logging.getLogger(__name__)


def _off(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() in {"0", "false", "off", "no"}


class DSparkTextGenerator(SingleBatchGenerator):
    """SingleBatchGenerator whose eligible requests decode through DSpark (see the module docstring)."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._dspark = getattr(model, "_vmlx_dspark_runtime", None)
        # (tokens the drafter context was built from, context caches, base position), from the previous request.
        self._draft_memory = None
        self.last_dspark_stats = None
        # Finished requests' speculative stats keyed by generator uid, popped by the scheduler into
        # RequestOutput.spec_stats (-> /health last_request.speculative, the Perf panel). Bounded: a request that ends
        # without a finishing Response (string stop / abort -> remove()) leaves an entry nobody pops.
        self._spec_stats: OrderedDict = OrderedDict()
        if self._dspark is None:
            logger.warning("DSpark text generator constructed without a runtime; plain decoding")

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _is_media(req) -> bool:
        return any(getattr(req, name, None) is not None
                   for name in ("pixel_values", "pixel_values_videos", "dsv41_vl_images"))

    def _decode_pick(self, req):
        """(pick, drafter sampler) for an eligible request, or None (plain decode path)."""
        if self._dspark is None or self._is_media(req) or req.logits_processors or self._logprobs_required(req):
            return None
        sampler = req.sampler or self.sampler
        greedy = bool(getattr(sampler, "_vmlx_is_greedy", False) or getattr(sampler, "_vmlx_greedy", False)
                      or getattr(sampler, "temp", None) == 0.0)
        if not greedy and _off("VMLX_DSPARK_SAMPLED"):
            return None
        # Exactly SingleBatchGenerator._sample_from_logits without processors / logprobs.
        if getattr(sampler, "_vmlx_accepts_logits", False):
            pick = sampler
        else:
            def pick(logits):
                return sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        return pick, (None if greedy else sampler)

    def _drafter_resume(self, prefix: list) -> dict:
        """Drafter context for a restored target prefix: the previous request's context when it covers ``prefix``."""
        memory, self._draft_memory = self._draft_memory, None
        if memory is not None and prefix:
            tokens, caches, base = memory
            n = len(prefix)
            if base <= n <= len(tokens) and tokens[:n] == prefix:
                keep = n - base
                for c in caches:
                    c.trim(int(c.offset) - keep)
                return {"draft_cache": caches, "draft_base": base}
        return {"draft_cache": None}

    # ------------------------------------------------------------------ request start
    def _start_request(self, req) -> Response:
        plan = self._decode_pick(req)
        if plan is None:
            return super()._start_request(req)
        if req.cache is None:
            req.cache = self._make_new_cache()
        cached = 0
        if req.cache:
            offset = getattr(req.cache[0], "offset", 0) or 0
            cached = int(offset.reshape(-1)[0].item()) if isinstance(offset, mx.array) else int(offset)
        if cached != len(req.context_tokens):
            logger.warning("DSpark text lane: restored cache holds %d tokens but the request context has %d; "
                           "plain decoding for request %s", cached, len(req.context_tokens), req.uid)
            return super()._start_request(req)
        from ..dspark import DSparkStats, iter_dspark
        from ..dspark.generate import ARFallback

        pick, draft_sampler = plan
        rt = self._dspark
        self._refresh_thread_stream()
        req.prefill_usage = None
        req.prefill_started = time.perf_counter()
        req.prompt_processed = True
        prompt_ids = list(req.context_tokens) + list(req.prompt_tokens)
        resume = {"target_cache": req.cache, "cache_len": cached}
        # The previous request's drafter context is consumed (or released) by every DSpark request.
        resume.update(self._drafter_resume(prompt_ids[:cached] if cached else []))
        stats = DSparkStats()
        session: dict = {}
        policy = None if _off("VMLX_DSPARK_AR_FALLBACK") else ARFallback()
        eos = {t for t in self.stop_tokens if isinstance(t, int)}
        req._dspark_stats = stats
        req._dspark_policy = policy
        req._dspark_session = session
        req._dspark_queue = []
        req._dspark_iter = iter_dspark(
            rt["target"], rt["drafter"], prompt_ids, max(1, int(req.max_tokens)), eos,
            prefill_step=self.prefill_step_size, sampler=draft_sampler, max_draft=rt["max_draft"],
            cost_per_row=rt["cost_per_row"], fixed_width=rt.get("fixed_width"), stats=stats,
            ar_fallback=policy, resume=resume, session=session, pick=pick,
        )
        req._dspark_on = True
        req.next_token = self._pull(req)          # the prefill and the first token
        req.prefill_usage = {
            "tokens": len(req.prompt_tokens),
            "seconds": time.perf_counter() - req.prefill_started,
            "scope": "model_prefill_and_prompt_state",
        }
        req.prefill_started = None
        req.context_tokens.extend(req.prompt_tokens)
        # The cache now holds exactly the prompt (no round has run): the plain M3 path's prompt snapshot.
        req.prompt_cache_snapshot = self._clone_admissible_prompt_cache_snapshot(req.cache)
        logger.info("DSpark text request %s: prompt %d tokens (%d restored, drafter context %s)", req.uid,
                    len(prompt_ids), cached, "carried over" if resume.get("draft_cache") is not None else
                    ("from the prompt" if not cached else f"from position {cached}"))
        return self._emit_dspark(req)

    def _pull(self, req):
        """The next committed token, running the next DSpark round (or the prefill) when none is queued."""
        while not req._dspark_queue:
            it = getattr(req, "_dspark_iter", None)
            if it is None:
                return None
            try:
                with self._stream_context():
                    chunk = next(it)
            except StopIteration:
                req._dspark_iter = None
                return None
            req._dspark_queue.extend(int(t) for t in chunk)
        return req._dspark_queue.pop(0)

    def _emit_dspark(self, req) -> Response:
        current = req.next_token
        if current is None:
            raise RuntimeError(f"DSpark text stream for request {req.uid} produced no token")
        req.output_tokens.append(current)
        req.context_tokens.append(current)
        req.finish_reason = self._finish_reason_after_token(req, current)
        req.next_token = None
        if req.finish_reason is None:
            req.next_token = self._pull(req)
            if req.next_token is None:
                req.finish_reason = "stop"   # the stream ended without a stop / length trigger
        resp = Response(uid=req.uid, token=current, logprobs=None, finish_reason=req.finish_reason,
                        current_state=req.current_state, match_sequence=req.match_sequence,
                        prompt_cache=req.cache, all_tokens=list(req.context_tokens),
                        prompt_cache_snapshot=req.prompt_cache_snapshot, prefill_usage=req.prefill_usage,
                        mixed_swa_boundary=None, stable_root_snapshot=None, boundary_checkpoints=None)
        if req.finish_reason is not None:
            self._finish_dspark(req)
            self._request = None
        return resp

    def _finish_dspark(self, req) -> None:
        it = getattr(req, "_dspark_iter", None)
        if it is not None:
            it.close()
            req._dspark_iter = None
        req._dspark_on = False
        session = getattr(req, "_dspark_session", None) or {}
        drafter = session.get("drafter")
        forwarded = session.get("forwarded")
        if drafter is not None and forwarded is not None and drafter.ctx_len == len(forwarded):
            self._draft_memory = (list(forwarded), drafter.context_caches(), int(drafter.ctx_base))
        stats = getattr(req, "_dspark_stats", None)
        if stats is not None:
            summary = stats.as_dict()
            policy = getattr(req, "_dspark_policy", None)
            if policy is not None:
                summary["adaptive_costs"] = policy.cost_snapshot()
            self.last_dspark_stats = summary
            # Same keys as the MiniCPM DSpark lane's spec_stats (models/mllm.py: the Perf panel's Last-request card
            # reads kind / lane / cycles / emitted_per_cycle / acceptance / ar_tokens / widths).
            self._spec_stats[req.uid] = {
                "kind": "dspark", "lane": "minimax-text", **summary,
                "cycles": summary["rounds"], "emitted_per_cycle": summary["tokens_per_round"],
                "widths": {str(k): v for k, v in summary["widths"].items()},
                "generation_tps": summary.get("decode_tok_s"),
            }
            while len(self._spec_stats) > 16:
                self._spec_stats.popitem(last=False)
            logger.info("DSpark text request %s: %s", req.uid, summary)

    def pop_spec_stats(self, uid):
        """The speculative stats of the finished DSpark request ``uid`` (once), or None (plain-path request)."""
        return self._spec_stats.pop(uid, None)

    def _yield_current_and_schedule_next(self, req, *, prompt_cache_snapshot=None) -> Response:
        if getattr(req, "_dspark_on", False):
            return self._emit_dspark(req)
        return super()._yield_current_and_schedule_next(req, prompt_cache_snapshot=prompt_cache_snapshot)

    def remove(self, uids, return_prompt_caches: bool = False):
        uid_set = set(uids)
        if self._request is not None and self._request.uid in uid_set and getattr(self._request, "_dspark_on", False):
            self._finish_dspark(self._request)
        return super().remove(uids, return_prompt_caches=return_prompt_caches)

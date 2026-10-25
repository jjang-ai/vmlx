# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark speculative-decode generator for the vMLX engine.

Selected (scheduler) only when a bundle's dSpark draft is attached as ``model._dspark`` — i.e. the
operator set ``VMLX_DSV41_DSPARK=1`` — so the default AR serving path is untouched. It subclasses
``SingleBatchGenerator`` to reuse its request lifecycle (insert/remove/Response/stop machinery) and
overrides only the decode path: instead of SBG's one-token engine-cache forward, it drives the
validated ``jang_tools.dsv41.dspark.DSparkRunner.stream`` generator on ``model.runtime`` (which owns
its own runtime ``Session`` + checkpoint/restore and the no-refeed fast path — DECODE_DSPARK_METHODS
M1). Output is token-identical to plain greedy; speculation only changes throughput (code ~1.9x).

Text-only: a request carrying vision tensors falls back to the inherited AR path (per request), so
multimodal turns still serve correctly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

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

    def _use_dspark(self, req) -> bool:
        return self._runner is not None and not self._is_vl(req)

    # --- decode override ------------------------------------------------------
    def _start_request(self, req) -> Response:
        start_offset = int(getattr(req.cache[0], "offset", 0) or 0) if getattr(req, "cache", None) else 0
        # Coexistence with the engine prefix cache, both correct:
        #  - PREFIX HIT (restored prefix): delegate to the inherited AR path, which continues from the
        #    restored composite cache coherently and re-stores -> per-tool-call cache reuse works.
        #  - VL / no draft: AR path.
        #  - COLD, text, draft present: DSpark on its OWN runtime Session (fast decode, 4-bit ~890MB
        #    cache, output == greedy). It does not populate the engine prefix cache (driving a restored
        #    DSpark continuation off the composite MLA+DSA+engram prefix is not yet bit-correct — the
        #    index_k pool restore mismatches; tracked). So DSpark = raw decode speed; the AR path is the
        #    one with full 890MB + prefix-cache reuse.
        if not self._use_dspark(req) or start_offset > 0:
            return super()._start_request(req)
        self._refresh_thread_stream()
        req.prefill_usage = None
        req.prefill_started = time.perf_counter()
        req.prompt_processed = True
        req._dspark_stream = self._runner.stream(
            list(req.prompt_tokens), req.max_tokens, eos_ids=set(self.stop_tokens))
        req._dspark_on = True
        req.next_token = self._pull(req)  # first committed token (int) or None
        return self._emit_dspark(req)

    def _pull(self, req):
        gen = getattr(req, "_dspark_stream", None)
        if gen is None:
            return None
        try:
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

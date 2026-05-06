# SPDX-License-Identifier: Apache-2.0
"""DSV4-Flash custom batch generator.

Bypasses ``mlx_lm.generate.BatchGenerator`` entirely for DSV4-Flash JANGTQ
bundles because the mlx_lm code path triggers ``mx.eval`` /
``mx.async_eval`` calls that traverse the tensor stream graph and hit
``RuntimeError: There is no Stream(gpu, N) in current thread.`` on the
llm-worker step executor (DSV4 model forward allocates internal Metal
streams that don't survive the cross-thread context).

This implementation mirrors the canonical
``jang_tools.dsv4.runtime.generate`` path that lives at
``/Users/eric/jang/jang-tools/jang_tools/dsv4/runtime.py`` — same prompt
encode → prefill → decode → parse loop, but exposes the BatchGenerator
API surface (``insert`` / ``next`` / ``next_generated`` / ``remove`` /
``extract_cache``) so the existing vmlx scheduler can drive it without
caring which generator is underneath.

Constraints:
- Single-batch only (``max_num_seqs=1``). Multi-batch DSV4 isn't
  supported — the compressor + indexer pool state can't be sliced
  per-request without a full re-prefill.
- Internally always pins ops to ``mx.stream(mx.default_device())`` so
  there's no chance of cross-thread stream leakage.
"""
from __future__ import annotations
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# DSV4-Flash-specific token IDs (canonical, mirrors
# jang_tools.dsv4.runtime.{THINK_OPEN_ID,THINK_CLOSE_ID,EOS_ID})
THINK_OPEN_ID = 128821
THINK_CLOSE_ID = 128822
DSV4_EOS_ID = 1


@dataclass
class _Request:
    uid: int
    prompt_tokens: List[int]
    context_tokens: List[int]
    cache: Optional[List[Any]]
    out_tokens: List[int] = field(default_factory=list)
    max_tokens: int = 1024
    sampler: Optional[Callable] = None
    logits_processors: Optional[List[Callable]] = None
    state_machine: Any = None
    matcher_state: Any = None
    finish_reason: Optional[str] = None
    prompt_processed: bool = False


@dataclass
class _Response:
    """Mirror of mlx_lm.generate.BatchGenerator.Response (subset we use)."""
    uid: int
    token: int
    logprobs: Any
    finish_reason: Optional[str] = None
    current_state: Any = None
    match_sequence: Any = None
    prompt_cache: Any = None


class DSV4BatchGenerator:
    """Drop-in replacement for ``mlx_lm.generate.BatchGenerator`` scoped to
    DSV4-Flash. Single-request at a time. Idempotent stream pinning.

    Public API mirrors the subset that vmlx scheduler uses:
        insert(prompts, max_tokens, caches, samplers, logits_processors, state_machines)
        next() -> (prompt_responses, generation_responses)
        next_generated() -> generation_responses
        remove(uids, return_prompt_caches=False) -> dict | None
        extract_cache(uids) -> dict
        stop_tokens (mutable set)
    """

    Response = _Response  # so callers that do `BatchGenerator.Response(...)` keep working

    def __init__(
        self,
        model,
        *,
        max_tokens: int = 1024,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
        completion_batch_size: int = 1,
        prefill_batch_size: int = 1,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
        stream=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Mark first request so we know to warm up MLX kernels before
        # the user's prefill (DSV4 first forward triggers hash-routed
        # layer JIT that exceeds the Metal command-buffer watchdog if
        # done in one shot on a long prompt).
        self._warmed_up = False
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.fallback_sampler = self.sampler
        self.logits_processors = logits_processors or []
        # stop_tokens is exposed as a mutable set the scheduler mutates.
        self.stop_tokens = set()
        if stop_tokens:
            for seq in stop_tokens:
                if isinstance(seq, int):
                    self.stop_tokens.add(seq)
                else:
                    # Multi-token stop sequence — store as tuple
                    self.stop_tokens.add(tuple(seq))
        logger.info(
            f"DSV4Gen: stop_tokens at construction = {sorted(self.stop_tokens)} "
            f"(DSV4_EOS_ID={DSV4_EOS_ID} always-checked separately)"
        )
        # DSV4 prefill defaults to SINGLE-SHOT.
        #
        # v1.5.6 (commit 00a78db4) established that chunking the DSV4
        # prefill corrupts the compressor + indexer pool state mid-decode
        # (broadcast_shapes mismatch on subsequent decode steps). v1.5.15
        # silently re-introduced chunking at 512 with a comment claiming
        # "current DeepseekV4Cache accumulates pool state correctly across
        # calls" — this is unverified and contradicted by v1.5.6's
        # empirical 14/14 probe matrix. The jang-tools DSV4 runtime is
        # unchanged between 2.5.18 (v1.5.10 baseline) and 2.5.23 (current),
        # so the chunking corruption v1.5.6 documented is still latent.
        #
        # Default: single-shot. Post-warmup the model has all kernels
        # JIT-compiled so even long prompts complete under the Metal
        # command-buffer watchdog. Env override DSV4_PREFILL_STEP_SIZE>0
        # is available for users who hit watchdog-kill on extreme prompts
        # (very rare; better to raise watchdog timeout instead).
        try:
            _dsv4_step_env = os.environ.get("DSV4_PREFILL_STEP_SIZE")
            _dsv4_step = int(_dsv4_step_env) if _dsv4_step_env else 0
        except (TypeError, ValueError):
            _dsv4_step = 0
        if _dsv4_step <= 0:
            # Single-shot — set step to a sentinel larger than any real
            # prompt so the chunked loop runs exactly once.
            self.prefill_step_size = 1 << 30
        else:
            self.prefill_step_size = _dsv4_step
        self.prefill_batch_size = 1  # always 1 for DSV4
        self.completion_batch_size = 1
        self.max_kv_size = max_kv_size
        self._uid_count = 0
        self._requests: List[_Request] = []
        # Pin device default stream
        self._device = mx.default_device()

    # ---------- helpers ----------
    def _make_new_cache(self):
        if hasattr(self.model, "make_cache"):
            return self.model.make_cache()
        from mlx_lm.models.cache import KVCache
        # 43 layers for DSV4-Flash (caller will get this right via model.make_cache)
        return [KVCache() for _ in range(getattr(self.model, "n_layers", 43))]

    def _sample(self, logits, sampler, processors, recent_tokens):
        """Apply logits processors then sample. logits: (1, vocab)."""
        x = logits
        for p in processors or []:
            # processor signature: fn(token_context, logits) -> logits
            try:
                x = p(recent_tokens, x)
            except TypeError:
                x = p(x)
        # Convert log-probs via logsumexp normalization
        logprobs = x - mx.logsumexp(x, axis=-1, keepdims=True)
        # Sampler
        sample_fn = sampler or self.fallback_sampler
        sampled = sample_fn(logprobs)
        return sampled, logprobs

    def _prefill_last_logits(self, token_ids: List[int], cache: List[Any]):
        """DSV4 prefill returning logits for the last prompt token.

        Default behavior is SINGLE-SHOT (one model() call over the full
        prompt). v1.5.6 verified that chunking corrupts the DSV4
        compressor + indexer pool state — chunked prefill is therefore
        opt-in only via DSV4_PREFILL_STEP_SIZE>0. When the env override
        is set, this loop chunks against the same cache and clears
        transient MLX buffers between chunks.
        """
        if not token_ids:
            return None
        all_ids = mx.array(token_ids, dtype=mx.int32)[None, :]
        last_logits = None
        total = len(token_ids)
        step = max(1, self.prefill_step_size)
        for off in range(0, total, step):
            chunk = all_ids[:, off:min(off + step, total)]
            logits = self.model(chunk, cache=cache)
            last_logits = logits[:, -1, :]
            mx.eval(last_logits)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        return last_logits

    # ---------- BatchGenerator API ----------
    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable]] = None,
        logits_processors: Optional[List[List[Callable]]] = None,
        state_machines: Optional[List[Any]] = None,
    ):
        # Auto-evict any already-finished requests so the scheduler can
        # queue the next one. Without this, the scheduler keeps retrying
        # inserts because the generator still claims slot 0 is taken.
        self._requests = [r for r in self._requests if r.finish_reason is None]
        if len(self._requests) + len(prompts) > 1:
            raise NotImplementedError(
                "DSV4BatchGenerator only supports max_num_seqs=1. "
                "Restart with --max-num-seqs 1 (continuous batching off)."
            )
        uids = []
        max_tokens = max_tokens or [self.max_tokens] * len(prompts)
        caches = caches or [None] * len(prompts)
        all_tokens = all_tokens or prompts
        samplers = samplers or [None] * len(prompts)
        logits_processors = logits_processors or [None] * len(prompts)
        state_machines = state_machines or [None] * len(prompts)
        for i, p in enumerate(prompts):
            req = _Request(
                uid=self._uid_count,
                prompt_tokens=list(p),
                context_tokens=list(all_tokens[i] or p),
                cache=caches[i],
                max_tokens=max_tokens[i],
                sampler=samplers[i],
                logits_processors=logits_processors[i] or self.logits_processors,
                state_machine=state_machines[i],
            )
            if state_machines[i] is not None and hasattr(state_machines[i], "make_state"):
                req.matcher_state = state_machines[i].make_state()
            self._requests.append(req)
            uids.append(self._uid_count)
            self._uid_count += 1
        return uids

    def insert_segments(self, segments, *args, **kwargs):
        # mlx_lm convention: segments[i] is list of token segments per request.
        # Flatten by concatenation — DSV4 doesn't use multi-segment prefill.
        flat = [sum(seg, []) for seg in segments]
        return self.insert(flat, *args, **kwargs)

    @staticmethod
    def _processor_context(req: _Request) -> List[int]:
        """Full token history for logits processors.

        mlx-lm's repetition penalty is applied over the generated context, not
        just a small recent window. DSV4 is especially sensitive here: limiting
        the processor context to the last 32 output tokens allowed long-form
        chat to re-enter earlier sentence/paragraph loops once the repeated
        phrase fell outside that window.
        """
        return list(req.context_tokens) + list(req.out_tokens)

    def remove(self, uids, return_prompt_caches: bool = False):
        out = {}
        keep = []
        for r in self._requests:
            if r.uid in uids:
                if return_prompt_caches:
                    out[r.uid] = (r.cache, r.prompt_tokens + r.out_tokens)
            else:
                keep.append(r)
        self._requests = keep
        return out if return_prompt_caches else None

    def extract_cache(self, uids):
        out = {}
        for r in self._requests:
            if r.uid in uids:
                out[r.uid] = (r.cache, r.prompt_tokens + r.out_tokens)
        return out

    def next(self) -> Tuple[List[_Response], List[_Response]]:
        """Run one step. Returns (prompt_responses, generation_responses).

        Prompt responses fire when a request transitions from 'queued
        with no cache' to 'has its first decoded token'.

        Generation responses fire on every subsequent decode step.
        """
        prompt_resps: List[_Response] = []
        gen_resps: List[_Response] = []

        # On first call, warm up MLX kernels so the user's prefill
        # doesn't have to JIT-compile every routed-expert / hash-router
        # combo in one shot. We call model() with the smallest possible
        # input (1 token) on a fresh cache, force-evaluate, then drop
        # the warmup state. After this, the user's prefill paths use
        # the precompiled kernels.
        if not self._warmed_up:
            with mx.stream(self._device):
                try:
                    _warm_cache = self._make_new_cache()
                    _warm_ids = mx.array([[0]], dtype=mx.int32)
                    _ = self.model(_warm_ids, cache=_warm_cache)
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    logger.info("DSV4Gen: kernel warmup done")
                except Exception as _wexc:
                    logger.warning(f"DSV4Gen: warmup failed (continuing anyway): {_wexc}")
            self._warmed_up = True

        for r in list(self._requests):
            with mx.stream(self._device):
                if r.cache is None:
                    # Prefill — chunked through one DeepseekV4Cache instance.
                    # This preserves SWA+CSA/HSA state while bounding transient
                    # MLX allocations for long prompts.
                    r.cache = self._make_new_cache()
                    if not r.prompt_tokens:
                        r.finish_reason = "stop"
                        r.prompt_processed = True
                        continue
                    # Sample first token from last logit position
                    last_logits = self._prefill_last_logits(r.prompt_tokens, r.cache)
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                    )
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    tok_id = int(sampled.tolist()[0]) if hasattr(sampled, "tolist") else int(sampled.item())
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    # Check stop
                    if tok_id == DSV4_EOS_ID or tok_id in self.stop_tokens:
                        r.finish_reason = "stop"
                    elif len(r.out_tokens) >= r.max_tokens:
                        r.finish_reason = "length"
                    prompt_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                    ))
                elif not r.prompt_processed:
                    # Prefix-cache hit path. The scheduler passes the cached
                    # prompt cache plus the remaining prompt tail to process
                    # (for exact hits this is the final prompt token). DSV4's
                    # custom generator cannot jump straight into decode with
                    # an empty out_tokens list; it must first run that prompt
                    # tail through the restored cache and sample the first
                    # generated token from the resulting logits. This mirrors
                    # mlx_lm BatchGenerator's cached-prefix kickoff behavior.
                    if not r.prompt_tokens:
                        r.finish_reason = "stop"
                        r.prompt_processed = True
                        continue
                    last_logits = self._prefill_last_logits(r.prompt_tokens, r.cache)
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                    )
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    tok_id = int(sampled.tolist()[0]) if hasattr(sampled, "tolist") else int(sampled.item())
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    if tok_id == DSV4_EOS_ID or tok_id in self.stop_tokens:
                        r.finish_reason = "stop"
                    elif len(r.out_tokens) >= r.max_tokens:
                        r.finish_reason = "length"
                    prompt_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                    ))
                else:
                    # Decode
                    if r.finish_reason is not None:
                        # Already finished — just emit nothing. Caller
                        # should remove() it.
                        continue
                    last_id = r.out_tokens[-1]
                    ids = mx.array([[last_id]], dtype=mx.int32)
                    logits = self.model(ids, cache=r.cache)
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    last_logits = logits[:, -1, :]
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                    )
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    tok_id = int(sampled.tolist()[0]) if hasattr(sampled, "tolist") else int(sampled.item())
                    r.out_tokens.append(tok_id)
                    if tok_id == DSV4_EOS_ID or tok_id in self.stop_tokens:
                        r.finish_reason = "stop"
                    elif len(r.out_tokens) >= r.max_tokens:
                        r.finish_reason = "length"
                    gen_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                    ))

        return prompt_resps, gen_resps

    def next_generated(self) -> List[_Response]:
        while True:
            prompt_resps, gen_resps = self.next()
            if not gen_resps and prompt_resps:
                continue
            return gen_resps

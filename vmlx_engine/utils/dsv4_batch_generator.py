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
        self.prefill_step_size = prefill_step_size
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
        samplers = samplers or [None] * len(prompts)
        logits_processors = logits_processors or [None] * len(prompts)
        state_machines = state_machines or [None] * len(prompts)
        for i, p in enumerate(prompts):
            req = _Request(
                uid=self._uid_count,
                prompt_tokens=list(p),
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
                    # Prefill — single-shot via `model(full_ids, cache=cache)`.
                    # The canonical jang_tools.dsv4.runtime.generate runs
                    # prefill in one call; chunking the prefill broke the
                    # DSV4 compressor + indexer pool state (broadcast_shapes
                    # mismatch on subsequent decode). The post-warmup model
                    # has all kernels JIT-compiled, so even a long prompt
                    # completes within the Metal command-buffer watchdog.
                    r.cache = self._make_new_cache()
                    if not r.prompt_tokens:
                        r.finish_reason = "stop"
                        r.prompt_processed = True
                        continue
                    full_ids = mx.array(r.prompt_tokens, dtype=mx.int32)[None, :]
                    logits = self.model(full_ids, cache=r.cache)
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    # Sample first token from last logit position
                    last_logits = logits[:, -1, :]
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        r.prompt_tokens[-32:],
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
                        r.out_tokens[-32:],
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
                    ))

        return prompt_resps, gen_resps

    def next_generated(self) -> List[_Response]:
        while True:
            prompt_resps, gen_resps = self.next()
            if not gen_resps and prompt_resps:
                continue
            return gen_resps

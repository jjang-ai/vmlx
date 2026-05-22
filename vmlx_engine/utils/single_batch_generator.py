"""Single-active text generator for vMLX's continuous-batching scheduler.

This is the engine-owned path for the common release profile:
``--continuous-batching --max-num-seqs 1`` with prefix/paged/L2/TQ cache
features still enabled.  It intentionally exposes the small subset of
``mlx_lm.generate.BatchGenerator`` used by :mod:`vmlx_engine.scheduler`, but
keeps native per-request cache objects instead of wrapping them into
``BatchKVCache``/``BatchMambaCache``.  Real multi-request batching still uses
mlx-lm's BatchGenerator.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import mlx.core as mx
from mlx_lm.generate import SequenceStateMachine
from mlx_lm.models import cache as mlx_cache
from mlx_lm.models.cache import TokenBuffer

from .mamba_cache import _should_capture_generation_logprobs

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    uid: int
    prompt_tokens: list[int]
    context_tokens: list[int]
    cache: Optional[list[Any]]
    max_tokens: int
    sampler: Optional[Callable[[mx.array], mx.array]]
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]]
    state_machine: SequenceStateMachine
    matcher_state: Any
    token_context: TokenBuffer
    prompt_processed: bool = False
    output_tokens: list[int] = field(default_factory=list)
    prompt_cache_snapshot: Optional[list[Any]] = None
    finish_reason: Optional[str] = None
    current_state: Any = None
    match_sequence: Any = None
    next_token: Optional[mx.array] = None
    next_logprobs: Any = None


@dataclass
class Response:
    uid: int
    token: int
    logprobs: Any
    finish_reason: Optional[str]
    current_state: Any
    match_sequence: Any
    prompt_cache: Optional[list[Any]]
    all_tokens: Optional[list[int]]
    prompt_cache_snapshot: Optional[list[Any]] = None


class SingleBatchGenerator:
    """BatchGenerator-compatible raw-cache generator for one active request."""

    Response = Response

    def __init__(
        self,
        model,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Union[Sequence[int], int]]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[list[Callable[[mx.array, mx.array], mx.array]]] = None,
        completion_batch_size: int = 1,
        prefill_batch_size: int = 1,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
        stream=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = 1
        self.completion_batch_size = 1
        self.max_kv_size = max_kv_size
        self.stop_tokens: set[Any] = set()
        stop_edges = []
        for seq in stop_tokens or []:
            if isinstance(seq, int):
                self.stop_tokens.add(seq)
                stop_edges.append(([seq], None))
            else:
                seq_tuple = tuple(int(t) for t in seq)
                if len(seq_tuple) == 1:
                    self.stop_tokens.add(seq_tuple[0])
                elif seq_tuple:
                    self.stop_tokens.add(seq_tuple)
                stop_edges.append((list(seq_tuple), None))
        self._default_state_machine = SequenceStateMachine(
            {"normal": stop_edges} if stop_edges else {},
            initial="normal",
        )
        self._uid_count = 0
        self._unprocessed: deque[_Request] = deque()
        self._request: Optional[_Request] = None
        self._device = mx.default_device()
        self._stream = stream or self._make_generation_stream()
        self._decode_trace = os.environ.get("VMLINUX_DECODE_TRACE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._decode_trace_every = max(
            1, int(os.environ.get("VMLINUX_DECODE_TRACE_EVERY", "64") or "64")
        )
        self._decode_trace_count = 0
        self._decode_trace_model_s = 0.0
        self._decode_trace_sample_s = 0.0

    def close(self):
        return None

    @property
    def prompt_cache_nbytes(self):
        total = 0
        for req in list(self._unprocessed) + ([self._request] if self._request else []):
            for layer in req.cache or []:
                try:
                    total += int(getattr(layer, "nbytes", 0) or 0)
                except Exception:
                    pass
        return total

    def _refresh_thread_stream(self) -> None:
        stream = self._make_generation_stream()
        if stream is not None:
            self._stream = stream

    def _make_generation_stream(self):
        try:
            return mx.new_stream(self._device)
        except Exception:
            pass
        try:
            return mx.default_stream(self._device)
        except Exception:
            return None

    def _stream_context(self):
        if self._stream is None:
            return nullcontext()
        return mx.stream(self._stream)

    def _rehome_on_stream(self, value):
        try:
            return value + mx.zeros_like(value)
        except Exception:
            return value

    def _eval_on_stream(self, *values):
        if not values:
            return ()
        with self._stream_context():
            rebound = tuple(self._rehome_on_stream(value) for value in values)
            try:
                mx.eval(*rebound)
            except TypeError:
                for value in rebound:
                    mx.eval(value)
        return rebound

    def _sync(self) -> None:
        if hasattr(mx, "synchronize"):
            try:
                if self._stream is not None:
                    mx.synchronize(self._stream)
                else:
                    mx.synchronize()
            except TypeError:
                mx.synchronize()

    def _make_new_cache(self):
        return mlx_cache.make_prompt_cache(self.model, max_kv_size=self.max_kv_size)

    @staticmethod
    def _clone_array(value):
        if value is None:
            return None
        # MLX assignment into cache buffers can mutate the original allocation
        # after this point. Materialize an independent array for the snapshot.
        cloned = 1 * value
        mx.eval(cloned)
        return cloned

    @classmethod
    def _clone_cache_object(cls, cache_obj):
        if cache_obj is None:
            return None
        if isinstance(cache_obj, mlx_cache.CacheList):
            cloned_children = [cls._clone_cache_object(c) for c in cache_obj.caches]
            if any(c is None for c in cloned_children):
                return None
            return mlx_cache.CacheList(*cloned_children)
        if isinstance(cache_obj, mlx_cache.KVCache):
            cloned = type(cache_obj)()
            if cache_obj.keys is not None:
                keys, values = cache_obj.state
                cloned.state = (cls._clone_array(keys), cls._clone_array(values))
            return cloned
        if isinstance(cache_obj, mlx_cache.ArraysCache):
            cloned = type(cache_obj)(len(cache_obj.cache))
            cloned.cache = [cls._clone_array(c) for c in cache_obj.cache]
            if getattr(cache_obj, "lengths", None) is not None:
                cloned.lengths = cls._clone_array(cache_obj.lengths)
            if getattr(cache_obj, "left_padding", None) is not None:
                cloned.left_padding = cls._clone_array(cache_obj.left_padding)
            return cloned
        if type(cache_obj).__name__ == "ZayaNoStateCache":
            return cache_obj.extract(0)
        return None

    @classmethod
    def _cache_needs_prompt_snapshot(cls, cache_obj) -> bool:
        if isinstance(cache_obj, mlx_cache.CacheList):
            return any(cls._cache_needs_prompt_snapshot(c) for c in cache_obj.caches)
        return type(cache_obj).__name__ in {"DeepseekV4Cache"}

    @classmethod
    def _clone_prompt_cache_snapshot(cls, cache):
        if not cache or not any(cls._cache_needs_prompt_snapshot(c) for c in cache):
            return None
        cloned = [cls._clone_cache_object(c) for c in cache]
        if any(c is None for c in cloned):
            return None
        return cloned

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: Optional[list[int]] = None,
        caches: Optional[list[list[Any]]] = None,
        all_tokens: Optional[list[list[int]]] = None,
        samplers: Optional[list[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            list[list[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[list[SequenceStateMachine]] = None,
    ):
        max_tokens = max_tokens or [self.max_tokens] * len(prompts)
        caches = caches or [None] * len(prompts)
        all_tokens = all_tokens or [[] for _ in prompts]
        samplers = samplers or [None] * len(prompts)
        logits_processors = logits_processors or [self.logits_processors] * len(prompts)
        state_machines = state_machines or [self._default_state_machine] * len(prompts)

        active = 1 if self._request is not None else 0
        if active + len(self._unprocessed) + len(prompts) > 1:
            raise NotImplementedError(
                "SingleBatchGenerator supports exactly one active request. "
                "Use max_num_seqs > 1 to select the real batched path."
            )

        uids: list[int] = []
        for prompt, max_tok, cache, context, sampler, processors, sm in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
        ):
            prompt_tokens = list(prompt)
            if not prompt_tokens:
                raise ValueError("SingleBatchGenerator received an empty prompt")
            state_machine = sm or self._default_state_machine
            matcher_state = state_machine.make_state()
            uid = self._uid_count
            self._uid_count += 1
            req = _Request(
                uid=uid,
                prompt_tokens=prompt_tokens,
                context_tokens=list(context or []),
                cache=cache,
                max_tokens=int(max_tok),
                sampler=sampler,
                logits_processors=list(processors or []),
                state_machine=state_machine,
                matcher_state=matcher_state,
                token_context=TokenBuffer(list(context or [])),
            )
            self._unprocessed.append(req)
            uids.append(uid)
        return uids

    def insert_segments(self, segments, *args, **kwargs):
        return self.insert([sum(seg, []) for seg in segments], *args, **kwargs)

    def _model_call(self, tokens: list[int], req: _Request):
        arr = mx.array([tokens], dtype=mx.int32)
        return self.model(arr, cache=req.cache)

    def _prefill(self, tokens: list[int], req: _Request) -> None:
        if not tokens:
            return
        pos = 0
        while pos < len(tokens):
            n = min(self.prefill_step_size, len(tokens) - pos)
            chunk = tokens[pos : pos + n]
            self._model_call(chunk, req)
            req.context_tokens.extend(chunk)
            req.token_context.update_and_fetch(mx.array(chunk, dtype=mx.int32))
            self._sync()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            pos += n

    def _logprobs_required(self, req: _Request) -> bool:
        try:
            return bool(_should_capture_generation_logprobs(self.model, [req.uid])[0])
        except Exception:
            return False

    def _sample_from_logits(
        self,
        logits: mx.array,
        req: _Request,
        input_tokens: mx.array,
    ) -> tuple[mx.array, Any]:
        token_context = None
        if req.logits_processors:
            token_context = req.token_context.update_and_fetch(input_tokens)
            for processor in req.logits_processors:
                logits = processor(token_context, logits)
        else:
            req.token_context.update_and_fetch(input_tokens)

        sampler = req.sampler or self.sampler
        wants_logprobs = self._logprobs_required(req)
        if not wants_logprobs and getattr(sampler, "_vmlx_accepts_logits", False):
            sampled = sampler(logits)
            logprobs = None
        else:
            logprobs_full = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            sampled = sampler(logprobs_full)
            logprobs = logprobs_full[0] if wants_logprobs else None

        async_items = [sampled]
        if logprobs is not None:
            async_items.append(logprobs)
        if token_context is not None:
            async_items.append(token_context)
        # Cache-hit replay can restore KV arrays whose lazy graph still refers
        # to a thread-local default stream from a prior request.  Materialize
        # the sampled scalar/logprobs inside this generator's concrete stream so
        # the next scheduler step never resolves a stale Stream(gpu,0).
        evaluated = self._eval_on_stream(*async_items)
        sampled = evaluated[0]
        if logprobs is not None:
            logprobs = evaluated[1]
        return sampled, logprobs

    def _compute_next_from_input_array(
        self,
        req: _Request,
        input_tokens: mx.array,
    ) -> None:
        with self._stream_context():
            input_tokens = self._rehome_on_stream(input_tokens)
            trace = self._decode_trace
            model_t0 = time.perf_counter() if trace else 0.0
            logits = self.model(input_tokens[:, None], cache=req.cache)
            if trace:
                self._sync()
                model_s = time.perf_counter() - model_t0
                sample_t0 = time.perf_counter()
            logits = logits[:, -1, :]
            req.next_token, req.next_logprobs = self._sample_from_logits(
                logits,
                req,
                input_tokens,
            )
            if trace:
                self._sync()
                sample_s = time.perf_counter() - sample_t0
                self._decode_trace_count += 1
                self._decode_trace_model_s += model_s
                self._decode_trace_sample_s += sample_s
                if self._decode_trace_count % self._decode_trace_every == 0:
                    n = self._decode_trace_count
                    logger.info(
                        "VMLINUX_DECODE_TRACE single steps=%d avg_model_ms=%.2f "
                        "avg_sample_ms=%.2f last_model_ms=%.2f "
                        "last_sample_ms=%.2f",
                        n,
                        (self._decode_trace_model_s / n) * 1000.0,
                        (self._decode_trace_sample_s / n) * 1000.0,
                        model_s * 1000.0,
                        sample_s * 1000.0,
                    )

    def _compute_next_from_input(self, req: _Request, input_token: int) -> None:
        input_tokens = mx.array([input_token], dtype=mx.int32)
        self._compute_next_from_input_array(req, input_tokens)

    @staticmethod
    def _token_to_int(token: Union[mx.array, int]) -> int:
        if hasattr(token, "tolist"):
            value = token.tolist()
            while isinstance(value, list):
                value = value[0]
            return int(value)
        return int(token)

    def _finish_reason_after_token(self, req: _Request, token_id: int) -> Optional[str]:
        req.matcher_state, match_sequence, current_state = req.state_machine.match(
            req.matcher_state,
            token_id,
        )
        req.match_sequence = match_sequence
        req.current_state = current_state
        if token_id in self.stop_tokens:
            return "stop"
        if match_sequence is not None and current_state is None:
            return "stop"
        if len(req.output_tokens) >= req.max_tokens:
            return "length"
        return None

    def _yield_current_and_schedule_next(
        self,
        req: _Request,
        *,
        prompt_cache_snapshot: Optional[list[Any]] = None,
    ) -> Response:
        current_token = req.next_token
        current_logprobs = req.next_logprobs
        if current_token is None:
            raise RuntimeError("SingleBatchGenerator decode requested before prefill")

        # Match mlx-lm's one-token lookahead: submit the next forward using the
        # sampled mx.array before materializing the current token for
        # Python/streaming. Stop-token detection happens after materialization,
        # so the final step may schedule one unused lookahead, matching
        # upstream BatchGenerator semantics.
        will_reach_length = len(req.output_tokens) + 1 >= req.max_tokens
        if not will_reach_length:
            self._compute_next_from_input_array(req, current_token)
        else:
            req.next_token = None
            req.next_logprobs = None

        try:
            current_token = self._eval_on_stream(current_token)[0]
            if current_logprobs is not None:
                current_logprobs = self._eval_on_stream(current_logprobs)[0]
        except Exception:
            self._sync()

        token_id = self._token_to_int(current_token)
        req.output_tokens.append(token_id)
        req.context_tokens.append(token_id)
        req.finish_reason = self._finish_reason_after_token(req, token_id)
        response = Response(
            uid=req.uid,
            token=token_id,
            logprobs=current_logprobs,
            finish_reason=req.finish_reason,
            current_state=req.current_state,
            match_sequence=req.match_sequence,
            prompt_cache=req.cache,
            all_tokens=list(req.context_tokens),
            prompt_cache_snapshot=prompt_cache_snapshot or req.prompt_cache_snapshot,
        )
        if req.finish_reason is not None:
            self._request = None
        return response

    def _start_request(self, req: _Request) -> Response:
        self._refresh_thread_stream()
        if req.cache is None:
            req.cache = self._make_new_cache()
        if len(req.prompt_tokens) > 1:
            self._prefill(req.prompt_tokens[:-1], req)
        last_token = int(req.prompt_tokens[-1])
        req.prompt_processed = True
        self._compute_next_from_input(req, last_token)
        # The native KV cache has now consumed `last_token` (the prompt's final
        # token), but `_compute_next_from_input` only updates the TokenBuffer,
        # not `context_tokens`. Without this append, response.all_tokens and
        # extract_cache() context would skip the final prompt token entirely
        # (e.g., prompt [11,12] -> sampled 3 would yield [11,3]).
        req.context_tokens.append(last_token)
        prompt_cache_snapshot = self._clone_prompt_cache_snapshot(req.cache)
        req.prompt_cache_snapshot = prompt_cache_snapshot
        return self._yield_current_and_schedule_next(
            req,
            prompt_cache_snapshot=prompt_cache_snapshot,
        )

    def next(self):
        if self._request is None and self._unprocessed:
            self._request = self._unprocessed.popleft()
            return [self._start_request(self._request)], []
        if self._request is None:
            return [], []
        return [], [self._yield_current_and_schedule_next(self._request)]

    def next_generated(self):
        while True:
            prompt_responses, generation_responses = self.next()
            if generation_responses:
                return generation_responses
            if not prompt_responses:
                # No active or queued request; nothing more to yield.
                # Without this guard, a prompt-only response that finished the
                # request (e.g., max_tokens=1 length-cap on the first sampled
                # token) would loop forever because `next()` would keep
                # returning ([], []).
                return []

    def _find(self, uid: int) -> Optional[_Request]:
        if self._request is not None and self._request.uid == uid:
            return self._request
        for req in self._unprocessed:
            if req.uid == uid:
                return req
        return None

    def extract_cache(self, uids):
        out = {}
        for uid in uids:
            req = self._find(uid)
            if req is not None:
                out[uid] = (req.cache, list(req.context_tokens))
        return out

    def remove(self, uids, return_prompt_caches: bool = False):
        uid_set = set(uids)
        out = self.extract_cache(uid_set) if return_prompt_caches else {}
        if self._request is not None and self._request.uid in uid_set:
            self._request = None
        if self._unprocessed:
            self._unprocessed = deque(
                req for req in self._unprocessed if req.uid not in uid_set
            )
        return out if return_prompt_caches else None

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
from .memory_limits import get_effective_metal_working_set_bytes
from .prefill_admission import (
    prefill_keep_alloc_enabled as _prefill_keep_alloc_enabled,
    prefill_valve_check as _prefill_valve_check,
    prefill_valve_enabled as _prefill_valve_enabled,
    prefill_valve_min_margin_bytes as _prefill_valve_min_margin_bytes,
)

logger = logging.getLogger(__name__)

def _cold_prefill_tail_split(gen_prompt_len: int = 0) -> int:
    """How many trailing prompt tokens the COLD prefill computes separately.

    0 (the default) keeps the historical single-pass cold prefill. Set
    VMLX_COLD_PREFILL_TAIL_SPLIT to the number of tokens a warm turn re-feeds
    after restoring, to make the two arms shape-equivalent.

    DO NOT ENABLE THIS. Measured 2026-08-13: the gen_prompt_len derivation is
    WRONG and REGRESSES a family that is byte-exact without it.

        Laguna-S with the gate on: FAIL, diverges at char 43 (exact without it)
          gate fired "1450 + 2 (of 1452)" while the warm turn reported
          cached=1452 -- the cache covers the WHOLE prefill, so Laguna re-feeds
          NOTHING and the correct width is 0, not 2.

    The real width is ``len(cold_tokens) - cached_tokens``. That merely
    COINCIDES with gen_prompt_len on MiniMax-M2.7 (1449 - 1444 = 5 = gpl), which
    is the one family the formula was validated against; it diverges on Laguna
    (0 vs 2). A cold turn cannot know cached_tokens -- nothing is cached yet --
    so a correct boundary has to come from the same computation the STORE uses,
    not from a proxy.

    Splitting also cannot help PATH-DEPENDENT caches at all: Nemotron
    (nemotron_h_ssm_attention) still diverges with the boundary derived
    correctly, and Step-3.7 (mixed-SWA RotatingKVCache) still diverges with the
    shapes made exactly identical to a warm turn. Recurrent state and rotating
    ring buffers depend on HOW tokens were fed, not just which ones.

    Kept as a reproduction lever for the MiniMax-M2.7 case (where it does give a
    byte-exact PASS) and as the record of why the obvious fix is not the fix.
    """
    import os as _os

    _override = _os.environ.get("VMLX_COLD_PREFILL_TAIL_SPLIT")
    if _override:
        try:
            return max(0, int(_override))
        except Exception:  # noqa: BLE001
            return 0
    if _os.environ.get("VMLX_COLD_PREFILL_MATCH_WARM_SPLIT") == "1":
        return max(0, int(gen_prompt_len or 0))
    return 0



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
    # Length of the generation-prompt suffix. The prefix cache keys on
    # len(prompt_tokens[:-gen_prompt_len]) - 1, so this is exactly how many
    # trailing positions a warm turn re-feeds after restoring — and therefore
    # where a cold prefill must split to stay numerically equivalent to it.
    gen_prompt_len: int = 0
    prompt_processed: bool = False
    output_tokens: list[int] = field(default_factory=list)
    prompt_cache_snapshot: Optional[list[Any]] = None
    finish_reason: Optional[str] = None
    current_state: Any = None
    match_sequence: Any = None
    next_token: Optional[mx.array] = None
    next_logprobs: Any = None
    next_token_materialized: bool = False
    next_logprobs_materialized: bool = False
    # MiniMax-M3 VL (additive, only ever set when VMLX_M3_VL routes an image
    # request here). Carried so the FIRST prefill forward can splice vision
    # embeddings; cleared after that forward so decode never re-passes them.
    pixel_values: Any = None
    image_grid_thw: Any = None
    pixel_values_videos: Any = None
    video_grid_thw: Any = None


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
        prompt_snapshot_max_bytes: Optional[int] = None,
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
        self.prompt_snapshot_max_bytes = (
            None
            if prompt_snapshot_max_bytes is None
            else max(0, int(prompt_snapshot_max_bytes))
        )
        self.prompt_snapshot_oversize_skips = 0
        self.prompt_snapshot_last_estimated_bytes = 0
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

    def _can_overlap_decode(self, req: _Request) -> bool:
        """Overlap only established plain KV state after initial materialization.

        The first step still resolves restored graphs on the concrete worker
        stream. Custom, rotating, sparse and recurrent caches retain their
        synchronous path; subclassing KVCache is not proof of that contract.
        """
        return (
            bool(req.output_tokens)
            and bool(req.cache)
            and all(type(layer) is mlx_cache.KVCache for layer in req.cache)
            and not self._needs_affine2_sync(req.cache)
        )

    def _submit_on_stream(self, *values):
        with self._stream_context():
            rebound = tuple(self._rehome_on_stream(value) for value in values)
            mx.async_eval(*rebound)
        return rebound

    def _materialize_submitted_on_stream(self, *values):
        # These arrays were already rebound and submitted on our stream. Adding
        # another rebind here queues it AFTER the following decode and defeats
        # one-token overlap. Initial restored graphs still use _eval_on_stream.
        with self._stream_context():
            mx.eval(*values)
        return values

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
        if type(cache_obj).__name__ == "MiniMaxM3SparseCache":
            try:
                from ..models.minimax_m3.cache import clone_minimax_m3_sparse
            except Exception:
                return None
            return clone_minimax_m3_sparse(
                cache_obj,
                copy_fn=cls._clone_array,
                require_idx_keys=True,
            )
        if type(cache_obj).__name__ == "OpenPanguV2LayerCache":
            try:
                from ..models.openpangu_v2.cache import clone_openpangu_layer_cache

                return clone_openpangu_layer_cache(
                    cache_obj,
                    copy_fn=cls._clone_array,
                )
            except Exception:
                return None
        if type(cache_obj).__name__ in {"Glm5KDACache", "Glm5MLACache"}:
            try:
                from ..models.glm5_next.glm5_next import (
                    clone_glm5_next_layer_cache,
                )

                return clone_glm5_next_layer_cache(
                    cache_obj,
                    copy_fn=cls._clone_array,
                )
            except Exception:
                return None
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
        return type(cache_obj).__name__ in {
            "DeepseekV4Cache",
            "Glm5KDACache",
            "Glm5MLACache",
            "MiniMaxM3SparseCache",
            "OpenPanguV2LayerCache",
        }

    @classmethod
    def _cache_uses_openpangu(cls, cache_obj) -> bool:
        if isinstance(cache_obj, mlx_cache.CacheList):
            return any(cls._cache_uses_openpangu(c) for c in cache_obj.caches)
        if isinstance(cache_obj, (list, tuple)):
            return any(cls._cache_uses_openpangu(c) for c in cache_obj)
        return type(cache_obj).__name__ == "OpenPanguV2LayerCache"

    @classmethod
    def _cache_uses_glm5_next(cls, cache_obj) -> bool:
        if isinstance(cache_obj, mlx_cache.CacheList):
            return any(cls._cache_uses_glm5_next(c) for c in cache_obj.caches)
        if isinstance(cache_obj, (list, tuple)):
            return any(cls._cache_uses_glm5_next(c) for c in cache_obj)
        return type(cache_obj).__name__ in {"Glm5KDACache", "Glm5MLACache"}

    @classmethod
    def _cache_uses_m3_msa(cls, cache_obj) -> bool:
        if isinstance(cache_obj, mlx_cache.CacheList):
            return any(cls._cache_uses_m3_msa(c) for c in cache_obj.caches)
        if isinstance(cache_obj, (list, tuple)):
            return any(cls._cache_uses_m3_msa(c) for c in cache_obj)
        return type(cache_obj).__name__ == "MiniMaxM3SparseCache"

    @classmethod
    def _clone_prompt_cache_snapshot(cls, cache):
        if not cache or not any(cls._cache_needs_prompt_snapshot(c) for c in cache):
            return None
        cloned = [cls._clone_cache_object(c) for c in cache]
        if any(c is None for c in cloned):
            return None
        return cloned

    def _clone_admissible_prompt_cache_snapshot(self, cache):
        """Clone a typed prompt boundary only when a cache backend can retain it.

        Typed/path-dependent caches must be copied before decode mutates their
        live state.  That copy can itself be enormous: a 43,979-token
        openPangu composite was 22.1 GB and added roughly a minute of work only
        to be rejected by both the RAM and disk budgets.  Estimate the live
        boundary first; this read-only size walk does not materialize tensors.
        """

        if not cache or not any(self._cache_needs_prompt_snapshot(c) for c in cache):
            return None
        limit = self.prompt_snapshot_max_bytes
        if limit is not None:
            try:
                from ..memory_cache import estimate_kv_cache_memory

                estimated = int(estimate_kv_cache_memory(list(cache)))
            except Exception as exc:
                logger.debug("typed prompt snapshot size estimate failed: %s", exc)
            else:
                self.prompt_snapshot_last_estimated_bytes = estimated
                if estimated > limit:
                    self.prompt_snapshot_oversize_skips += 1
                    logger.warning(
                        "Skipping typed prompt snapshot before copy: estimated %.1fMB "
                        "exceeds every enabled cache backend limit (%.1fMB)",
                        estimated / (1024 * 1024),
                        limit / (1024 * 1024),
                    )
                    return None
        return self._clone_prompt_cache_snapshot(cache)

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
        pixel_values: Optional[list[Any]] = None,
        image_grid_thw: Optional[list[Any]] = None,
        pixel_values_videos: Optional[list[Any]] = None,
        video_grid_thw: Optional[list[Any]] = None,
        gen_prompt_lens: Optional[list[int]] = None,
    ):
        max_tokens = max_tokens or [self.max_tokens] * len(prompts)
        gen_prompt_lens = gen_prompt_lens or [0] * len(prompts)
        caches = caches or [None] * len(prompts)
        # M3 VL (additive): per-prompt vision tensors, default None == text path.
        pixel_values = pixel_values or [None] * len(prompts)
        image_grid_thw = image_grid_thw or [None] * len(prompts)
        pixel_values_videos = pixel_values_videos or [None] * len(prompts)
        video_grid_thw = video_grid_thw or [None] * len(prompts)
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
        for (
            prompt,
            max_tok,
            cache,
            context,
            sampler,
            processors,
            sm,
            pv,
            grid,
            pv_video,
            video_grid,
            gen_prompt_len,
        ) in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            gen_prompt_lens,
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
                pixel_values=pv,
                image_grid_thw=grid,
                pixel_values_videos=pv_video,
                video_grid_thw=video_grid,
                gen_prompt_len=int(gen_prompt_len or 0),
            )
            self._unprocessed.append(req)
            uids.append(uid)
        return uids

    def insert_segments(self, segments, *args, **kwargs):
        return self.insert([sum(seg, []) for seg in segments], *args, **kwargs)

    def _model_call(self, tokens: list[int], req: _Request):
        arr = mx.array([tokens], dtype=mx.int32)
        # M3 VL: vision tensors are present ONLY on the first prefill forward
        # (set by _start_request when req.pixel_values is not None). After this
        # single forward they are cleared so decode never re-passes them, leaving
        # the 23.4 tok/s decode hot path byte-for-byte unchanged.
        pv = req.pixel_values
        pv_video = req.pixel_values_videos
        if pv is not None or pv_video is not None:
            grid = req.image_grid_thw
            video_grid = req.video_grid_thw
            req.pixel_values = None
            req.image_grid_thw = None
            req.pixel_values_videos = None
            req.video_grid_thw = None
            # The vision tensors were created on a different thread (the server
            # event loop, during preprocessing). Rehome them onto THIS generator
            # stream so their lazy graph doesn't resolve a stale Stream(gpu,0)
            # when the scheduler worker thread evals (the P0 VL stream bug).
            if pv is not None:
                pv = self._rehome_on_stream(pv)
            if grid is not None:
                grid = self._rehome_on_stream(grid)
            if pv_video is not None:
                pv_video = self._rehome_on_stream(pv_video)
            if video_grid is not None:
                video_grid = self._rehome_on_stream(video_grid)
            return self.model(
                arr,
                cache=req.cache,
                pixel_values=pv,
                image_grid_thw=grid,
                pixel_values_videos=pv_video,
                video_grid_thw=video_grid,
            )
        return self.model(arr, cache=req.cache)

    def _prefill(self, tokens: list[int], req: _Request) -> None:
        if not tokens:
            return
        _prefill_keep_alloc = _prefill_keep_alloc_enabled()
        # Admission control. DSV4 has had a prefill valve for a while — it
        # projects the next chunk's peak and rejects BEFORE submitting GPU work,
        # because discovering the limit by failing can take the process with it.
        # Every other family reached this loop with no check at all, which is
        # how a model advertising ~69k tokens dies somewhere around 20-25k.
        _valve_on = _prefill_valve_enabled()
        _valve_max_ws = 0
        _valve_margin = 0
        _valve_transient = 0
        if _valve_on:
            try:
                _, _valve_max_ws = get_effective_metal_working_set_bytes(mx)
            except Exception:  # noqa: BLE001 - never let the guard break prefill
                _valve_max_ws = 0
            _valve_on = _valve_max_ws > 0
            _valve_margin = _prefill_valve_min_margin_bytes()

        pos = 0
        while pos < len(tokens):
            n = min(self.prefill_step_size, len(tokens) - pos)
            chunk = tokens[pos : pos + n]
            if _valve_on:
                try:
                    _active = int(mx.get_active_memory())
                except Exception:  # noqa: BLE001
                    _active = 0
                # Unknown readings must never reject a request that would work.
                _prefill_valve_check(
                    _active,
                    _valve_max_ws,
                    _valve_transient,
                    _valve_margin,
                    chunk_start=pos,
                    chunk_end=pos + n,
                    model_label=type(self).__name__,
                )
            with self._stream_context():
                self._model_call(chunk, req)
                req.context_tokens.extend(chunk)
                if req.logits_processors:
                    req.token_context.update_and_fetch(mx.array(chunk, dtype=mx.int32))
            self._sync()
            if not _prefill_keep_alloc:
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
        *,
        overlap: bool = False,
    ) -> tuple[mx.array, Any]:
        token_context = None
        if req.logits_processors:
            token_context = req.token_context.update_and_fetch(input_tokens)
            for processor in req.logits_processors:
                logits = processor(token_context, logits)

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
        # the first sampled scalar/logprobs inside this generator's concrete
        # stream so the next step never resolves a stale Stream(gpu,0).
        # Qualified steady decode may then submit on that same owned stream;
        # materializing the current token must not wait for the following one.
        evaluated = (
            self._submit_on_stream(*async_items)
            if overlap
            else self._eval_on_stream(*async_items)
        )
        sampled = evaluated[0]
        if logprobs is not None:
            logprobs = evaluated[1]
        return sampled, logprobs

    def _needs_affine2_sync(self, cache=None) -> bool:
        """MiniMax-M3 + affine-2 fast path requires FULL sync each decode step.

        The default one-token lookahead leaves the forward (incl. the custom
        affine-2 SwitchGLU Metal kernel) lazy across the iteration boundary; that
        async overlap rarely corrupts long/thinking generations. Forcing mx.eval
        of the forward each step removes the overlap — matching the proven
        synchronous runtime path. Detect via the M3 MSA cache type (robust to
        model wrappers). Gated to M3+affine-2 only.
        """
        # M3 MSA cache needs a FULL materialize-sync each decode step: the
        # append-only idx_keys concatenation desyncs from K/V under the default
        # one-token-lookahead async overlap on long generations (>~2048 tok),
        # corrupting the Lightning-Indexer block selection -> incoherent loops.
        # This is a property of the MSA cache ITSELF, independent of the affine-2
        # fast path. (Previously gated on `not _disabled()` i.e. affine-2 ON; when
        # affine-2 went default-OFF the MSA sync silently turned off too -> the
        # server long-gen loop. Pure runtime is synchronous and never hit this.)
        try:
            _caches = cache if isinstance(cache, (list, tuple)) else (
                [cache] if cache is not None else [])
            for _c in _caches:
                if _c is None:
                    continue
                if type(_c).__name__ == "MiniMaxM3SparseCache" or hasattr(_c, "idx_keys"):
                    return True
        except Exception:
            pass
        # Fallback: the affine-2 fast path (M3-only) also requires the sync.
        try:
            from vmlx_engine.models.minimax_m3.m3_affine2_switch import _disabled
            return not _disabled()
        except Exception:
            return False

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
            overlap = self._can_overlap_decode(req)
            req.next_token, req.next_logprobs = self._sample_from_logits(
                logits,
                req,
                input_tokens,
                overlap=overlap,
            )
            req.next_token_materialized = not overlap
            req.next_logprobs_materialized = not overlap and req.next_logprobs is not None
            # M3+affine-2: materialize the forward + cache state NOW (no lookahead
            # overlap) to avoid the async-overlap corruption of the custom Metal
            # kernel. Eval the cache too (not just the token) since paged/block-disk
            # snapshots read it.
            if self._needs_affine2_sync(req.cache):
                _arrs = [req.next_token]
                if req.next_logprobs is not None:
                    _arrs.append(req.next_logprobs)
                try:
                    for _c in (req.cache or []):
                        _st = getattr(_c, "state", None)
                        if isinstance(_st, (list, tuple)):
                            _arrs.extend(x for x in _st if x is not None)
                        elif _st is not None:
                            _arrs.append(_st)
                except Exception:
                    pass
                _ev = self._eval_on_stream(*_arrs)
                req.next_token = _ev[0]
                if req.next_logprobs is not None:
                    req.next_logprobs = _ev[1]
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
        current_token_materialized = req.next_token_materialized
        current_logprobs_materialized = req.next_logprobs_materialized
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
            req.next_token_materialized = False
            req.next_logprobs_materialized = False

        try:
            if not current_token_materialized:
                current_token = self._materialize_submitted_on_stream(current_token)[0]
            if current_logprobs is not None and not current_logprobs_materialized:
                current_logprobs = self._materialize_submitted_on_stream(current_logprobs)[0]
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

    def _prefill_full_and_sample(self, req: _Request) -> None:
        """Prefill one M3 prompt in the same shape as the reference runtime.

        The clean MiniMax-M3 runtime calls ``model([full_prompt], cache)`` and
        samples from the final-position logits. Splitting the final prompt token
        into a separate decode step builds the MSA idx_keys path differently
        before the first generated token. M3 VL also requires this path because
        image-token positions must co-occur for the vision splice.
        """
        tokens = list(req.prompt_tokens)
        has_media = req.pixel_values is not None or req.pixel_values_videos is not None
        # The processor emits one vision tensor containing features for every
        # media placeholder in the prompt. Passing that tensor with only the
        # first normal prefill chunk makes the feature count diverge whenever
        # media tokens cross a chunk boundary (common for video after chat
        # history). Keep the first media prefill atomic so every placeholder
        # co-occurs with the tensor that supplies its embedding. Text-only M3
        # retains the configured chunk size.
        step = (
            len(tokens)
            if has_media
            else max(1, int(self.prefill_step_size or len(tokens) or 1))
        )
        pos = 0
        _prefill_keep_alloc = _prefill_keep_alloc_enabled()
        with self._stream_context():
            while len(tokens) - pos > step:
                n = min(step, len(tokens) - pos)
                # Keep at least two tokens in the final chunk when possible so
                # the final prompt token is not treated like a standalone decode
                # step before the first sampled token.
                if len(tokens) - (pos + n) == 1 and n > 1:
                    n -= 1
                chunk = tokens[pos : pos + n]
                logits = self._model_call(chunk, req)  # forwards+clears pixel_values
                (logits,) = self._eval_on_stream(logits)
                req.context_tokens.extend(chunk)
                if req.logits_processors:
                    req.token_context.update_and_fetch(
                        mx.array(chunk, dtype=mx.int32)
                    )
                self._sync()
                if not _prefill_keep_alloc and hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                pos += n

            final_chunk = tokens[pos:]
            logits = self._model_call(final_chunk, req)  # forwards+clears pixel_values
            # Materialize the full vision+text forward on THIS stream before the
            # sample step touches it (prevents stale-stream resolution on the
            # worker thread). Mirrors the decode path's on-stream eval.
            (logits,) = self._eval_on_stream(logits)
            logits = logits[:, -1, :]
            input_tokens = mx.array([final_chunk[-1]], dtype=mx.int32)
            req.next_token, req.next_logprobs = self._sample_from_logits(
                logits,
                req,
                input_tokens,
            )
            req.next_token_materialized = True
            req.next_logprobs_materialized = req.next_logprobs is not None
            req.context_tokens.extend(final_chunk)
            if req.logits_processors:
                req.token_context.update_and_fetch(mx.array(final_chunk, dtype=mx.int32))
        self._sync()

    def _start_request(self, req: _Request) -> Response:
        self._refresh_thread_stream()
        if req.cache is None:
            req.cache = self._make_new_cache()
        # M3 MSA/VL: build prompt cache exactly like the clean reference runtime:
        # one full-prompt forward, then sample from the final-position logits.
        if (
            req.pixel_values is not None
            or req.pixel_values_videos is not None
            or self._cache_uses_m3_msa(req.cache)
        ):
            req.prompt_processed = True
            self._prefill_full_and_sample(req)
            prompt_cache_snapshot = self._clone_admissible_prompt_cache_snapshot(
                req.cache
            )
            req.prompt_cache_snapshot = prompt_cache_snapshot
            return self._yield_current_and_schedule_next(
                req,
                prompt_cache_snapshot=prompt_cache_snapshot,
            )
        if len(req.prompt_tokens) > 1:
            # A prefix-cache HIT restores the cached prefix and then computes
            # the remaining few positions in a SMALL forward, while a cold
            # prefill computes them inside one large pass. FP reductions are not
            # shape-invariant, so those positions come out fractionally
            # different and can flip an argmax later in a long generation.
            #
            # This gate makes the COLD path split at the same boundary so both
            # arms compute the tail with the same shape. Default OFF: it changes
            # cold-path numerics for every family, including the ones already
            # byte-exact, so it must be proven per family with
            # scripts/cache_fidelity_check.py before it could become a default.
            _tail = _cold_prefill_tail_split(req.gen_prompt_len)
            _cold_tokens = req.prompt_tokens[:-1]
            if _tail > 0 and len(_cold_tokens) > _tail:
                # Log so an A/B can PROVE the path engaged. A guard that
                # silently declines makes the comparison stock-vs-stock.
                logger.info(
                    "COLD_PREFILL_TAIL_SPLIT engaged: %d + %d (of %d)",
                    len(_cold_tokens) - _tail,
                    _tail,
                    len(_cold_tokens),
                )
                self._prefill(_cold_tokens[:-_tail], req)
                self._prefill(_cold_tokens[-_tail:], req)
            else:
                self._prefill(_cold_tokens, req)
        uses_openpangu = self._cache_uses_openpangu(req.cache)
        uses_glm5_next = self._cache_uses_glm5_next(req.cache)
        # openPangu and GLM each own a path-dependent native state unit. Capture
        # the immutable N-1 boundary BEFORE the final prompt token is consumed;
        # post-decode convolution/recurrent/indexer state cannot be rewound.
        typed_prompt_snapshot = None
        if uses_openpangu and any(
            int(getattr(layer, "offset", 0) or 0) > 0 for layer in req.cache
        ):
            typed_prompt_snapshot = self._clone_admissible_prompt_cache_snapshot(
                req.cache
            )
        elif uses_glm5_next and any(
            any(value is not None for value in getattr(layer, "cache", ()))
            for layer in req.cache
        ):
            typed_prompt_snapshot = self._clone_admissible_prompt_cache_snapshot(
                req.cache
            )
        last_token = int(req.prompt_tokens[-1])
        req.prompt_processed = True
        self._compute_next_from_input(req, last_token)
        # The native KV cache has now consumed `last_token` (the prompt's final
        # token), but `_compute_next_from_input` only updates the TokenBuffer,
        # not `context_tokens`. Without this append, response.all_tokens and
        # extract_cache() context would skip the final prompt token entirely
        # (e.g., prompt [11,12] -> sampled 3 would yield [11,3]).
        req.context_tokens.append(last_token)
        prompt_cache_snapshot = (
            typed_prompt_snapshot
            if uses_openpangu or uses_glm5_next
            else self._clone_admissible_prompt_cache_snapshot(req.cache)
        )
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

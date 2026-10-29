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
    PrefillAdmissionError as _PrefillAdmissionError,
    advise_admission as _advise_admission,
    prefill_valve_check as _prefill_valve_check,
    prefill_valve_enabled as _prefill_valve_enabled,
    prefill_valve_min_margin_bytes as _prefill_valve_min_margin_bytes,
)

logger = logging.getLogger(__name__)

def _naive_prefill_phase_trace_enabled(model: Any) -> bool:
    """Opt-in wall timings only; never add a device completion boundary."""
    return getattr(model, "model_type", None) == "naive_n05_flash" and os.environ.get(
        "VMLX_NAIVE_PREFILL_PHASE_TRACE", ""
    ).lower() in ("1", "true", "yes", "on")


def _log_naive_prefill_phase(req: _Request, phase: str, **fields: Any) -> None:
    # Diagnostics must not turn a successfully completed forward into a failure.
    # Context counts are host token-list lengths, not native cache offsets (SWA
    # and recurrent caches do not share a physical offset interpretation).
    try:
        logger.info(
            "NAIVE_PREFILL_PHASE uid=%s phase=%s wall=%s",
            req.uid,
            phase,
            fields,
        )
    except Exception:
        pass


def _naive_boundary_checkpoint_every() -> int:
    """Exact Naive checkpoints every N prompt tokens of a cold prefill (VMLX_NAIVE_BOUNDARY_CHECKPOINT_TOKENS, default
    4096, 0 = off).

    Naive's sliding-window layers resume only from an exact window, and those used to exist only at each request's
    N-1 and terminal states, so any earlier divergence (a new chat with the same long system prompt, a tools toggle
    rendered after it, a mid-history edit) re-prefilled the whole prompt (62 s at 37.7k). A boundary costs one window
    copy (128 rows per sliding layer); its full-attention part is a slice of the N-1 snapshot.
    Measured on a 36-38k Naive prompt (2026-10-12): new chat with the same system prompt 61.8 s -> 5.8 s, token-identical
    to cold; cost ~0.12 s window capture + ~0.3 s extra store per cold prefill (interleaved A/B: +0.5 s of ~60 s).
    """
    try:
        return max(0, int(os.environ.get("VMLX_NAIVE_BOUNDARY_CHECKPOINT_TOKENS", "4096").strip() or 0))
    except ValueError:
        return 4096


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
    # DeepSeek-V4.1-Flash VL (additive, text-routed). images = jang_tools.dsv41
    # vision.ImageInput list; token_types = per-token list (>=0 on image spans).
    # Spliced on the FIRST (single) prefill forward, then cleared like M3's.
    dsv41_vl_images: Any = None
    dsv41_vl_token_types: Any = None
    prefill_started: Optional[float] = None
    prefill_usage: Optional[dict] = None
    # Mixed-SWA: (boundary, {layer_idx: RotatingKVCache snapshot}) captured at
    # the exact N-1 prompt boundary before the final prompt token is consumed.
    mixed_swa_boundary: Any = None
    # Stable root (exact typed lanes, e.g. openPangu): prefill splits after this many of THIS request's prompt
    # tokens (the end of the system turn) and keeps an exact typed snapshot there, so a new conversation with the
    # same system prompt and tools restores it instead of re-prefilling them. 0 = no split.
    stable_root_split: int = 0
    # MiniMax-M3 aligned prefill plan ({"boundary": absolute store boundary or None, "atomic": media request,
    # "cuts": absolute turn cuts}), set by the scheduler (models/minimax_m3/prefill_geometry.py). None = the previous
    # relative chunking.
    m3_plan: Optional[dict] = None
    stable_root_snapshot: Optional[list[Any]] = None
    # Periodic exact checkpoints (Naive): (boundary, window capture) taken during the cold prefill, then assembled
    # into (boundary, cache) from the N-1 snapshot for the scheduler to store before the N-1 checkpoint.
    boundary_windows: list = field(default_factory=list)
    boundary_checkpoints: Optional[list] = None


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
    prefill_usage: Optional[dict] = None
    mixed_swa_boundary: Any = None
    stable_root_snapshot: Optional[list[Any]] = None
    boundary_checkpoints: Optional[list] = None


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
        mixed_swa_boundary_capture: bool = False,
        mixed_swa_capture_block_size: int = 64,
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
        self.prompt_snapshot_headroom_skips = 0
        self.prompt_snapshot_last_headroom_bytes = 0
        self.prompt_snapshot_last_reason = None
        # Mixed full/sliding-window caches: capture the rotating windows at the
        # exact N-1 boundary so the scheduler can store it without re-running
        # the whole prompt. Enabled by the scheduler only when a prefix cache
        # will consume it.
        self.mixed_swa_boundary_capture = bool(mixed_swa_boundary_capture)
        self.mixed_swa_capture_block_size = max(1, int(mixed_swa_capture_block_size or 64))
        self.mixed_swa_boundary_captures = 0
        self.mixed_swa_boundary_declines = 0
        self.mixed_swa_boundary_last_reason = None
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

    def _eval_on_stream(self, *values, extra=()):
        """Evaluate ``values`` rebound on this generator's stream (returned) and ``extra`` as they are (not rebound:
        cache buffers, where a rebind would copy them)."""
        if not values:
            return ()
        with self._stream_context():
            rebound = tuple(self._rehome_on_stream(value) for value in values)
            try:
                mx.eval(*rebound, *extra)
            except TypeError:
                for value in rebound + tuple(extra):
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
        """Overlap qualified native layouts after initial materialization.

        The first step still resolves restored graphs on the concrete worker
        stream. Plain KV and Naive's exact mixed layout are qualified; other
        custom, sparse and recurrent state retains synchronous evaluation.
        """
        if not req.output_tokens or not req.cache:
            return False
        plain_kv = all(type(layer) is mlx_cache.KVCache for layer in req.cache)
        native_naive = False
        if getattr(self.model, "model_type", None) == "naive_n05_flash":
            layers = getattr(self.model, "layers", ())
            # Admit only the native layout qualified across sparse selection,
            # ring wrap, SSD refault and terminal/cancel boundaries. Unknown
            # subclasses or companion state retain synchronous evaluation.
            native_naive = len(layers) == len(req.cache) and all(
                (
                    type(state) is mlx_cache.RotatingKVCache
                    if layer.is_swa
                    else type(state) is mlx_cache.CacheList
                    and len(state.caches) == 2
                    and all(type(child) is mlx_cache.KVCache for child in state.caches)
                )
                for layer, state in zip(layers, req.cache)
            )
        # GLM-5.3 (glm5_next): exactly one native KDA or MLA/DSA cache per layer and no speculative states in flight.
        # KDA state is replaced functionally every step and MLA/DSA appends are graph-ordered like plain KV; the
        # in-process pipelined loop (glm_dflash2.generate_ar) has run this overlap since 2026-10-10. Qualified by
        # tests/test_glm5_decode_overlap.py (sparse selection active; length / stop / cancel == synchronous, logprobs
        # and final state bit-identical). Synchronous evaluation cost ~6 ms per token (served 31 vs 38 tok/s).
        native_glm = False
        if not plain_kv and not native_naive and os.environ.get("VMLX_GLM5_DECODE_OVERLAP", "1") != "0" and all(
            type(layer).__name__ in ("Glm5KDACache", "Glm5MLACache") for layer in req.cache
        ):
            from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache

            native_glm = all(type(layer) in (Glm5KDACache, Glm5MLACache) for layer in req.cache) and not any(
                getattr(layer, "_speculative_states", None) for layer in req.cache
            )
        # MiniMax-M3 / M3.1: MiniMaxM3SparseCache sparse layers (+ stock KVCache dense layers on M3). K / V / index
        # lanes are step-grown buffers written by graph-ordered slice updates, or by the fused attention's in-place
        # appends, which re-issue the lanes behind their MSA launch (cache.py fused_commit); no host read per step.
        # Qualified by tests/test_minimax_m3_decode_overlap.py (sparse selection active, length / stop / cancel ==
        # synchronous, logprobs and final lanes bit-identical, thousands of tokens past the sparse switch, with and
        # without VMLX_M3_FUSED_ATTN). VMLX_M3_DECODE_OVERLAP=0 keeps the synchronous loop.
        native_m3 = (
            not plain_kv and not native_naive and not native_glm
            and os.environ.get("VMLX_M3_DECODE_OVERLAP", "1").strip().lower() not in {"0", "false", "off", "no"}
            and self._m3_overlap_layout(req.cache)
        )
        return (plain_kv or native_naive or native_glm or native_m3) and not self._needs_affine2_sync(req.cache)

    @staticmethod
    def _m3_overlap_layout(cache) -> bool:
        """The native M3 layout: at least one MiniMaxM3SparseCache, every layer that class or a stock KVCache."""
        try:
            from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
        except Exception:
            return False
        kinds = [type(layer) for layer in cache]
        return MiniMaxM3SparseCache in kinds and all(k in (MiniMaxM3SparseCache, mlx_cache.KVCache) for k in kinds)

    @staticmethod
    def _m3_lane_arrays(cache) -> list:
        """The lane buffers of an M3 cache list, read without exposing them (MiniMaxM3SparseCache.fused_lanes): they
        ride the step's evaluation so the index lane's appends (unread by the dense-regime forward) never pile up as
        a lazy graph. Never rehomed: a rehome is a full copy of every lane."""
        out = []
        for layer in cache or ():
            lanes = layer.fused_lanes() if hasattr(layer, "fused_lanes") else (
                (getattr(layer, "keys", None), getattr(layer, "values", None))
                if type(layer) is mlx_cache.KVCache else ())
            out.extend(a for a in lanes if isinstance(a, mx.array))
        return out

    def _submit_on_stream(self, *values, extra=()):
        with self._stream_context():
            rebound = tuple(self._rehome_on_stream(value) for value in values)
            mx.async_eval(*rebound, *extra)
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
                    copy_is_detached=True,
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
        if type(cache_obj).__name__ == "DeepseekV41Cache":
            # Bit-exact compact copy: 4-bit pools copied as stored (no dequant/re-quant round-trip).
            # Whole-list snapshots go through clone_cache_list in _clone_prompt_cache_snapshot so all
            # layers share one copied _SeqState (one Engram history, not one per layer).
            try:
                from jang_tools.dsv4_1.mlx_model import clone_cache_list

                return clone_cache_list([cache_obj])[0]
            except Exception:
                return None
        if type(cache_obj).__name__ == "ZayaNoStateCache":
            return cache_obj.extract(0)
        return None

    @classmethod
    def _cache_needs_prompt_snapshot(cls, cache_obj) -> bool:
        if isinstance(cache_obj, mlx_cache.CacheList):
            return any(cls._cache_needs_prompt_snapshot(c) for c in cache_obj.caches)
        return type(cache_obj).__name__ in {
            "DeepseekV4Cache",
            "DeepseekV41Cache",
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
    def _cache_uses_dsv41(cls, cache_obj) -> bool:
        if isinstance(cache_obj, (list, tuple)):
            return any(cls._cache_uses_dsv41(c) for c in cache_obj)
        return type(cache_obj).__name__ == "DeepseekV41Cache"

    def _clone_dsv41_prompt_boundary(self, req: "_Request"):
        """DeepSeek-V4.1: clone the cache at the exact N-1 prompt boundary.

        The scheduler stores the prompt snapshot under ``prompt_tokens[:-1]`` and re-feeds the last
        prompt token on a hit, so the snapshot must hold exactly N-1 tokens. The generic path below
        clones AFTER the final prompt token was consumed (N tokens under an N-1 key): every hit then
        re-fed that token at position N (duplicated ``<think>``, shifted positions / compressor groups /
        Engram n-grams). Only called before the last prompt token is fed."""
        if not req.cache or not any(int(getattr(c, "offset", 0) or 0) > 0 for c in req.cache):
            return None  # nothing before the final token (one-token prompt): no reusable boundary
        return self._clone_admissible_prompt_cache_snapshot(req.cache)

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
        if all(type(c).__name__ == "DeepseekV41Cache" for c in cache):
            try:
                from jang_tools.dsv4_1.mlx_model import clone_cache_list

                return clone_cache_list(list(cache))
            except Exception as exc:
                logger.warning("DSV4.1 prompt snapshot clone failed: %s", exc)
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
        # MiniMax-M3: every lane is append-only, so the rows the prefill wrote are still the live cache's first rows
        # when the request ends. Hand the scheduler a marker (the live cache objects + the row count), not a copy:
        # the copy was 8.2 GiB for a 65k prompt on M3.1 and either OOMed or, under the headroom guard below, left
        # long prompts uncached. The terminal store streams the rows to the SSD block store
        # (models/minimax_m3/prompt_boundary.py).
        if self._cache_uses_m3_msa(cache):
            from ..models.minimax_m3 import prompt_boundary

            if prompt_boundary.zero_copy_enabled():
                marker = prompt_boundary.capture(list(cache))
                if marker is not None:
                    self.prompt_snapshot_last_estimated_bytes = marker.nbytes()
                    self.prompt_snapshot_last_reason = "boundary_marker"
                    return marker
        try:
            from ..memory_cache import estimate_kv_cache_memory

            estimated = int(estimate_kv_cache_memory(list(cache), resident=False))
        except Exception as exc:
            logger.debug("typed prompt snapshot size estimate failed: %s", exc)
            return self._clone_prompt_cache_snapshot(cache)
        self.prompt_snapshot_last_estimated_bytes = estimated
        limit = self.prompt_snapshot_max_bytes
        if limit is not None and estimated > limit:
            self.prompt_snapshot_last_reason = "backend_budget"
            self.prompt_snapshot_oversize_skips += 1
            logger.warning(
                "Skipping typed prompt snapshot before copy: estimated %.1fMB "
                "exceeds every enabled cache backend limit (%.1fMB)",
                estimated / (1024 * 1024),
                limit / (1024 * 1024),
            )
            return None
        # The copy is new GPU memory on top of the live cache. A long prompt on a model that nearly fills the
        # working set raised [METAL] Insufficient Memory right here and failed the request (MiniMax-M3.1 JANGHT,
        # 95.5 GiB, 65k-token prompt -> an 8.2 GiB copy on a 107.5 GiB working set, 2026-10-28). Without room for
        # the copy, skip the prefix-cache store of this prompt; the request itself continues.
        active, maximum = get_effective_metal_working_set_bytes(mx)
        if maximum > 0:
            from .memory_limits import get_metal_ws_guard_threshold

            headroom = max(0, int(maximum * get_metal_ws_guard_threshold() / 100) - active)
            self.prompt_snapshot_last_headroom_bytes = headroom
            if estimated > headroom:
                self.prompt_snapshot_last_reason = "metal_headroom"
                self.prompt_snapshot_headroom_skips += 1
                logger.warning(
                    "Skipping typed prompt snapshot: its copy (%.1fMB) exceeds the free Metal working set "
                    "(%.1fMB); this prompt is not stored in the prefix cache, the request continues",
                    estimated / (1024 * 1024),
                    headroom / (1024 * 1024),
                )
                return None
        self.prompt_snapshot_last_reason = "captured"
        return self._clone_prompt_cache_snapshot(cache)

    def _capture_mixed_swa_boundary(self, req):
        """Capture the rotating windows at the N-1 boundary (bounded copy).

        Runs after the N-1 prefill and before the final prompt token, so each
        ring already sits exactly at the store boundary. Declines leave the
        scheduler's clean re-prefill in place.
        """
        from .mixed_swa_boundary import capture_rotating_boundary

        try:
            with self._stream_context():
                capture, reason = capture_rotating_boundary(
                    req.cache,
                    block_size=self.mixed_swa_capture_block_size,
                    log_id=str(req.uid),
                )
        except Exception as exc:
            capture, reason = None, f"error:{type(exc).__name__}"
            logger.warning("mixed-SWA boundary capture failed (non-fatal): %s", exc)
        self.mixed_swa_boundary_last_reason = reason
        if capture is None:
            self.mixed_swa_boundary_declines += 1
            return None
        self.mixed_swa_boundary_captures += 1
        return capture

    def _prefill_naive_boundaries(self, cold_tokens, req):
        """Prefill Naive's cold prompt in whole boundary segments, capturing the sliding windows at each boundary.

        Boundaries sit at the cold start + k * N (N a multiple of the prefill step), so the chunking -- and the
        numerics -- are the same as one pass. Only when the cold start is block-aligned (the checkpoint must end on a
        full block to be an exact anchor). Returns the tokens still to prefill (always non-empty when given some).
        """
        every = _naive_boundary_checkpoint_every()
        block = int(self.mixed_swa_capture_block_size)
        step = max(1, int(self.prefill_step_size))
        if (not every or every % step or every % block
                or getattr(self.model, "model_type", None) != "naive_n05_flash"):
            return cold_tokens
        from ..models.naive_n05_flash.cache_snapshot import cache_offset, clone_rotating_windows

        try:
            start = cache_offset(req.cache)
        except Exception:  # noqa: BLE001 - unknown layout: keep the single-pass prefill
            return cold_tokens
        if start % block:
            return cold_tokens
        while len(cold_tokens) > every:
            self._prefill(cold_tokens[:every], req)
            cold_tokens = cold_tokens[every:]
            _t0 = time.perf_counter()
            try:
                with self._stream_context():
                    windows = clone_rotating_windows(req.cache)
            except Exception as exc:  # noqa: BLE001 - a missed checkpoint only costs reuse, never correctness
                logger.warning("Naive boundary window capture declined: %s", exc)
                break
            req.boundary_capture_ms = getattr(req, "boundary_capture_ms", 0.0) + (time.perf_counter() - _t0) * 1000
            start += every
            if getattr(req, "boundary_windows", None) is None:
                req.boundary_windows = []
            req.boundary_windows.append((start, windows))
        return cold_tokens

    def _assemble_naive_boundaries(self, req, snapshot):
        from ..models.naive_n05_flash.cache_snapshot import assemble_boundary_checkpoint

        out = []
        for boundary, windows in req.boundary_windows:
            try:
                out.append((boundary, assemble_boundary_checkpoint(snapshot, windows, boundary)))
            except Exception as exc:  # noqa: BLE001 - skip this boundary, keep the others
                logger.warning("Naive boundary checkpoint at %d declined: %s", boundary, exc)
        req.boundary_windows = []
        if out:
            logger.info("Naive boundary checkpoints captured at %s (window capture %.1f ms total)",
                        [b for b, _ in out], float(getattr(req, "boundary_capture_ms", 0.0)))
        return out or None

    def _clone_naive_prompt_snapshot(self, req):
        # Mixed-SWA owns a full-prompt N-1 key, including the native generation
        # header. The scheduler binds this snapshot to that exact key explicitly.
        if getattr(self.model, "model_type", None) != "naive_n05_flash":
            self.prompt_snapshot_last_reason = "unsupported_model"
            return None
        if req.gen_prompt_len < 0:
            self.prompt_snapshot_last_reason = "invalid_generation_suffix"
            return None
        from ..models.naive_n05_flash.cache_snapshot import (
            clone_prompt_cache, snapshot_size,
        )
        from .memory_limits import get_metal_ws_guard_threshold

        try:
            estimated = snapshot_size(req.cache)
            self.prompt_snapshot_last_estimated_bytes = estimated
            limit = self.prompt_snapshot_max_bytes
            if limit is not None and estimated > limit:
                self.prompt_snapshot_last_reason = "backend_budget"
                self.prompt_snapshot_oversize_skips += 1
                return None
            active, maximum = get_effective_metal_working_set_bytes(mx)
            headroom = max(0, int(maximum * get_metal_ws_guard_threshold() / 100) - active)
            self.prompt_snapshot_last_headroom_bytes = headroom
            if maximum <= 0 or estimated > headroom:
                self.prompt_snapshot_last_reason = "metal_headroom"
                self.prompt_snapshot_headroom_skips += 1
                return None
            with self._stream_context():
                result = clone_prompt_cache(req.cache)
            self.prompt_snapshot_last_reason = "captured"
            return result
        except Exception as exc:
            self.prompt_snapshot_last_reason = "unavailable: " + str(exc)
            logger.debug("Naive prompt snapshot unavailable; retaining clean-prefill fallback: %s", exc)
            return None

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
        dsv41_vl_images: Optional[list[Any]] = None,
        dsv41_vl_token_types: Optional[list[Any]] = None,
        gen_prompt_lens: Optional[list[int]] = None,
        stable_root_splits: Optional[list[int]] = None,
        m3_prefill_plans: Optional[list[Optional[dict]]] = None,
    ):
        max_tokens = max_tokens or [self.max_tokens] * len(prompts)
        gen_prompt_lens = gen_prompt_lens or [0] * len(prompts)
        stable_root_splits = stable_root_splits or [0] * len(prompts)
        m3_prefill_plans = m3_prefill_plans or [None] * len(prompts)
        caches = caches or [None] * len(prompts)
        # M3 VL (additive): per-prompt vision tensors, default None == text path.
        pixel_values = pixel_values or [None] * len(prompts)
        image_grid_thw = image_grid_thw or [None] * len(prompts)
        pixel_values_videos = pixel_values_videos or [None] * len(prompts)
        video_grid_thw = video_grid_thw or [None] * len(prompts)
        dsv41_vl_images = dsv41_vl_images or [None] * len(prompts)
        dsv41_vl_token_types = dsv41_vl_token_types or [None] * len(prompts)
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
            dsv41_images,
            dsv41_ttypes,
            gen_prompt_len,
            stable_root_split,
            m3_plan,
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
            dsv41_vl_images,
            dsv41_vl_token_types,
            gen_prompt_lens,
            stable_root_splits,
            m3_prefill_plans,
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
                dsv41_vl_images=dsv41_images,
                dsv41_vl_token_types=dsv41_ttypes,
                gen_prompt_len=int(gen_prompt_len or 0),
                stable_root_split=int(stable_root_split or 0),
                m3_plan=dict(m3_plan) if m3_plan is not None else None,
            )
            self._unprocessed.append(req)
            uids.append(uid)
        return uids

    def insert_segments(self, segments, *args, **kwargs):
        return self.insert([sum(seg, []) for seg in segments], *args, **kwargs)

    def _model_call(self, tokens: list[int], req: _Request, lookahead: Optional[list[int]] = None,
                    prompt_end: Optional[int] = None):
        arr = mx.array([tokens], dtype=mx.int32)
        # DeepSeek-V4.1 prefill: prompt_end (absolute cache position where this prefill ends) lets the
        # late layers skip prompt rows the prefill's last token no longer reads (jang_tools.dsv41
        # Model.prune_rows); the cache state at prompt_end is the same as a full prefill's.
        dsv41_kw = {"prompt_end": prompt_end} if prompt_end is not None else {}
        # DeepSeek-V4.1-Flash VL (text-routed): splice vision embeddings into the
        # chunk that holds each image span. ImageInput.start is absolute in the
        # full prompt while the model writes h[0, start:...] of THIS chunk, so
        # pass only the images whose span lies inside the chunk, with chunk-
        # relative starts (_dsv41_vl_chunk_len never lets a chunk split an
        # image). The old code passed every image only on the pos==0 chunk, so
        # an image after the first prefill chunk was dropped (or overran it).
        # token_types is per-token and sliced to this chunk. Cleared once fully
        # consumed so decode stays on the byte-identical text path.
        if req.dsv41_vl_images is not None:
            tt_full = req.dsv41_vl_token_types or []
            pos = len(req.context_tokens)
            end = pos + len(tokens)
            tt_chunk = list(tt_full[pos:end])
            imgs = self._dsv41_vl_chunk_images(req.dsv41_vl_images, pos, end)
            if end >= len(tt_full) - 1:  # the final prompt token is never an image token
                req.dsv41_vl_images = None
                req.dsv41_vl_token_types = None
            return self.model(
                arr,
                cache=req.cache,
                images=imgs,
                token_types=(tt_chunk or None),
                **dsv41_kw,
            )
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
        if (lookahead or dsv41_kw) and getattr(self.model, "model_type", None) == "deepseek_v41":
            # DeepSeek-V4.1: the next prefill chunk's Engram rows (random SSD
            # reads) are fetched while this chunk computes.
            return self.model(arr, cache=req.cache, engram_lookahead=lookahead or None, **dsv41_kw)
        return self.model(arr, cache=req.cache)

    @staticmethod
    def _dsv41_vl_spans(images):
        return [(int(img.start), int(img.start) + len(img.types)) for img in images or ()]

    @classmethod
    def _dsv41_vl_chunk_images(cls, images, pos: int, end: int):
        """Images whose span lies in [pos, end), re-based to the chunk (start - pos)."""
        import dataclasses

        out = []
        for img, (s, e) in zip(images or (), cls._dsv41_vl_spans(images)):
            if pos <= s and e <= end:
                out.append(dataclasses.replace(img, start=s - pos))
            elif s < end and e > pos:
                raise ValueError(
                    f"DSV4.1 VL image span [{s}, {e}) straddles prefill chunk [{pos}, {end})"
                )
        return out or None

    @classmethod
    def _dsv41_vl_chunk_len(cls, req: "_Request", n: int) -> int:
        """Shrink or extend a prefill chunk so it never splits an image span."""
        images = getattr(req, "dsv41_vl_images", None)
        if not images:
            return n
        pos = len(req.context_tokens)
        end = pos + n
        for s, e in cls._dsv41_vl_spans(images):
            if s < end < e:
                end = s if s > pos else e  # stop before the image, or take all of it
        return max(1, end - pos)

    def _prefill(self, tokens: list[int], req: _Request) -> None:
        if not tokens:
            return
        _phase_trace = _naive_prefill_phase_trace_enabled(self.model)
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

        # Naive realizes its native state at every chunk below. Measure that
        # completed allocation window rather than leaving the valve's observed
        # transient at zero. Other families keep their existing ownership.
        _measure_native_peak = (
            _valve_on
            and getattr(self.model, "model_type", None) == "naive_n05_flash"
        )

        # DSV4.1: every chunk of this prefill knows where it ends (suffix pruning; logits are discarded).
        _prompt_end = None
        if getattr(self.model, "model_type", None) == "deepseek_v41" and req.cache:
            _seq = getattr(req.cache[0], "_seq", None)
            if _seq is not None:
                _prompt_end = int(_seq.pos) + len(tokens)
        pos = 0
        while pos < len(tokens):
            n = min(self.prefill_step_size, len(tokens) - pos)
            if req.dsv41_vl_images is not None:
                n = min(self._dsv41_vl_chunk_len(req, n), len(tokens) - pos)
            chunk = tokens[pos : pos + n]
            if _phase_trace:
                _chunk_t0 = time.perf_counter()
                _prior_context = len(req.context_tokens)
            if _valve_on:
                try:
                    _active = int(mx.get_active_memory())
                except Exception:  # noqa: BLE001
                    _active = 0
                # Advise, never refuse (I-36): the projection only logs.
                try:
                    _prefill_valve_check(
                        _active,
                        _valve_max_ws,
                        _valve_transient,
                        _valve_margin,
                        chunk_start=pos,
                        chunk_end=pos + n,
                        model_label=type(self).__name__,
                    )
                except _PrefillAdmissionError as advisory:
                    _advise_admission(advisory, request_id=getattr(req, "request_id", None))
            if _measure_native_peak:
                mx.reset_peak_memory()
            if _phase_trace:
                _valve_done = time.perf_counter()
            with self._stream_context():
                self._model_call(
                    chunk,
                    req,
                    lookahead=tokens[pos + n : pos + n + self.prefill_step_size],
                    prompt_end=_prompt_end,
                )
                if getattr(self.model, "model_type", None) == "naive_n05_flash":
                    # synchronize only waits for submitted work; it does not
                    # evaluate the lazy cache graph. Realize native nested KV,
                    # indexer and rotating state before admitting another chunk.
                    # Keep this boundary on the generator's concrete stream.
                    mx.eval([entry.state for entry in req.cache])
                elif getattr(self.model, "model_type", None) == "deepseek_v41":
                    # Same reason for DSV4.1: without a per-chunk eval the whole prompt stays one lazy
                    # graph and the clear_cache below frees nothing. Evaluate the 4-bit pools / window
                    # / compressor state directly (no dequantized .state round-trip).
                    from jang_tools.dsv41.dspark import _session_arrays

                    mx.eval(_session_arrays(req.cache[0]._seq) if req.cache else [])
                req.context_tokens.extend(chunk)
                if req.logits_processors:
                    req.token_context.update_and_fetch(mx.array(chunk, dtype=mx.int32))
            if _phase_trace:
                _state_done = time.perf_counter()
            self._sync()
            if _phase_trace:
                _sync_done = time.perf_counter()
            if _measure_native_peak and _active > 0:
                _valve_transient = max(
                    _valve_transient,
                    max(0, int(mx.get_peak_memory()) - _active),
                )
            if _phase_trace:
                _peak_read_done = time.perf_counter()
            if not _prefill_keep_alloc:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            if _phase_trace:
                _chunk_done = time.perf_counter()
                _log_naive_prefill_phase(
                    req,
                    "chunk_completed",
                    query_tokens=n,
                    prior_context_tokens=_prior_context,
                    valve_and_peak_reset_ms=(_valve_done - _chunk_t0) * 1000,
                    model_and_native_state_ms=(_state_done - _valve_done) * 1000,
                    sync_ms=(_sync_done - _state_done) * 1000,
                    peak_read_ms=(_peak_read_done - _sync_done) * 1000,
                    clear_ms=(_chunk_done - _peak_read_done) * 1000,
                    total_ms=(_chunk_done - _chunk_t0) * 1000,
                )
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
        extra=(),
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
            if wants_logprobs:
                # Reported logprobs in float32: the bf16 log-softmax above rounds every value to bf16 (step ~0.008 near
                # -1, 0.0625 near -10; the API returned e.g. exactly -1.0 / -0.75). The sampler keeps its existing
                # input, so the generated tokens are byte-identical to before.
                logits32 = logits.astype(mx.float32)
                logprobs = (logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True))[0]
            else:
                logprobs = None

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
        # ``extra`` (M3 lane buffers) only when present: other families call these helpers exactly as before.
        kw = {"extra": extra} if extra else {}
        evaluated = (
            self._submit_on_stream(*async_items, **kw)
            if overlap
            else self._eval_on_stream(*async_items, **kw)
        )
        sampled = evaluated[0]
        if logprobs is not None:
            logprobs = evaluated[1]
        return sampled, logprobs

    def _needs_affine2_sync(self, cache=None) -> bool:
        """The MiniMax-M3 affine-2 SwitchGLU fast path (opt-in) keeps a FULL synchronous step: its custom Metal kernel
        was seen to corrupt long generations under the one-token lookahead and has not been requalified.

        The MSA cache itself no longer needs it. That sync (and the per-step evaluation of every layer's
        ``cache.state`` it came with) dates from the index lane being re-concatenated every step; the lanes are now
        step-grown buffers written by graph-ordered slice updates or the fused attention's ordered in-place appends,
        and the state evaluation rebound (= copied) every K / V / index lane each token while marking the lanes
        shared (every fused append then went copy-on-write). The lanes now ride the step's own evaluation
        (``_m3_lane_arrays``)."""
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
            _phase_trace = (
                getattr(req, "prefill_started", None) is not None
                and _naive_prefill_phase_trace_enabled(self.model)
            )
            if _phase_trace:
                _final_t0 = time.perf_counter()
            logits = self.model(input_tokens[:, None], cache=req.cache)
            if trace:
                self._sync()
                model_s = time.perf_counter() - model_t0
                sample_t0 = time.perf_counter()
            logits = logits[:, -1, :]
            if getattr(req, "prefill_started", None) is not None:
                # The initial prompt only: complete final-position logits
                # before sampling/lookahead, matching the MLLM receipt scope.
                # The one-time materialization never fences steady decode.
                (logits,) = self._eval_on_stream(logits)
                req.prefill_usage = {
                    "tokens": len(req.prompt_tokens),
                    "seconds": time.perf_counter() - req.prefill_started,
                    "scope": "model_prefill_and_prompt_state",
                }
                if _phase_trace:
                    _log_naive_prefill_phase(
                        req,
                        "prompt_logits_completed",
                        query_tokens=1,
                        prior_context_tokens=len(req.context_tokens),
                        final_logits_ms=(
                            req.prefill_started + req.prefill_usage["seconds"] - _final_t0
                        ) * 1000,
                        full_prefill_ms=req.prefill_usage["seconds"] * 1000,
                        prompt_tokens=len(req.prompt_tokens),
                    )
                req.prefill_started = None
            overlap = self._can_overlap_decode(req)
            req.next_token, req.next_logprobs = self._sample_from_logits(
                logits,
                req,
                input_tokens,
                overlap=overlap,
                extra=self._m3_lane_arrays(req.cache) if self._cache_uses_m3_msa(req.cache) else (),
            )
            req.next_token_materialized = not overlap
            req.next_logprobs_materialized = not overlap and req.next_logprobs is not None
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
            prefill_usage=req.prefill_usage,
            mixed_swa_boundary=req.mixed_swa_boundary,
            stable_root_snapshot=req.stable_root_snapshot,
            boundary_checkpoints=getattr(req, "boundary_checkpoints", None),
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
        if getattr(req, "m3_plan", None) is not None:
            return self._prefill_m3_aligned_and_sample(req, req.m3_plan)
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
        _clear_each = self._m3_clear_each_chunk()
        _reserved = None

        def _chunk_len(at: int) -> int:
            n = min(step, len(tokens) - at)
            # Keep at least two tokens in the final chunk when possible so
            # the final prompt token is not treated like a standalone decode
            # step before the first sampled token.
            if len(tokens) - (at + n) == 1 and n > 1:
                n -= 1
            return n

        with self._stream_context():
            while len(tokens) - pos > step:
                if _reserved is None:
                    _reserved = self._m3_reserve_prefill(req.cache, len(tokens) - pos)
                n = _chunk_len(pos)
                chunk = tokens[pos : pos + n]
                nxt = pos + n
                lookahead = (
                    tokens[nxt : nxt + _chunk_len(nxt)]
                    if len(tokens) - nxt > step
                    else tokens[nxt:]
                )
                logits = self._model_call(chunk, req, lookahead=lookahead)  # forwards+clears pixel_values
                # Only the cache rows are kept from a non-final chunk: evaluate them, not the logits. The lm_head over
                # every row of the chunk (2048 x 200,064 on M3.1: ~0.1 s per chunk on an M5 Max) was computed and
                # thrown away; the rows the cache keeps run the same kernels either way (the aligned-plan path below
                # already did this).
                mx.eval(self._m3_cache_rows(req.cache))
                del logits
                req.context_tokens.extend(chunk)
                if req.logits_processors:
                    req.token_context.update_and_fetch(
                        mx.array(chunk, dtype=mx.int32)
                    )
                self._sync()
                if not _prefill_keep_alloc and _clear_each and hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                pos += n

            final_chunk = tokens[pos:]
            if _reserved is None:
                _reserved = self._m3_reserve_prefill(req.cache, len(final_chunk))
            logits = self._model_call(final_chunk, req)  # forwards+clears pixel_values
            # Materialize the full vision+text forward on THIS stream before the
            # sample step touches it (prevents stale-stream resolution on the
            # worker thread). Mirrors the decode path's on-stream eval.
            (logits,) = self._eval_on_stream(logits)
            if getattr(req, "prefill_started", None) is not None:
                req.prefill_usage = {
                    "tokens": len(tokens),
                    "seconds": time.perf_counter() - req.prefill_started,
                    "scope": "model_prefill_and_prompt_state",
                }
                req.prefill_started = None
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
        if not _prefill_keep_alloc and not _clear_each and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    @staticmethod
    def _m3_clear_each_chunk() -> bool:
        """VMLX_M3_PREFILL_CLEAR_EACH_CHUNK=1: release MLX's buffer cache after every M3 prefill chunk (the previous
        behaviour). Default: keep it through the prefill (every chunk re-allocates the same transient sizes; releasing
        them sends each chunk back to fresh, zero-filled Metal buffers) and release it once when the prefill ends, so
        the footprint after the prefill is what it was. VMLX_PREFILL_KEEP_ALLOC=1 still skips every release."""
        return os.environ.get("VMLX_M3_PREFILL_CLEAR_EACH_CHUNK", "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _m3_cache_rows(cache) -> list:
        """The cache state a non-final prefill chunk leaves (what it evaluates instead of its logits); empty layers
        (nothing written) are skipped."""
        out = []
        for layer in cache or ():
            empty = getattr(layer, "empty", None)
            if callable(empty) and empty():
                continue
            out.append(layer.state)
        return out

    @staticmethod
    def _m3_reserve_prefill(cache, rows: int):
        """Reserve every M3 lane for the remaining ``rows`` prompt rows (models/minimax_m3/cache.reserve_prefill):
        None until the lanes exist (ask again before the next piece), then True / False (reserved / stock growth)."""
        from ..models.minimax_m3.cache import reserve_prefill

        return reserve_prefill(cache, rows)

    def _m3_media_token_ids(self) -> set:
        args = getattr(self.model, "args", None)
        return {int(t) for t in (getattr(args, "image_token_index", None), getattr(args, "video_token_index", None))
                if t is not None}

    def m3_prefill_pieces(self, req: _Request, plan: dict) -> list:
        """Absolute [a, b) prefill pieces of this M3 request (models/minimax_m3/prefill_geometry.py)."""
        from ..models.minimax_m3.prefill_geometry import prefill_pieces

        start = len(req.context_tokens)
        end = start + len(req.prompt_tokens)
        boundary = plan.get("boundary")
        atomic = bool(plan.get("atomic"))
        has_media = req.pixel_values is not None or req.pixel_values_videos is not None
        if has_media and boundary is not None and start < int(boundary) < end:
            # Every placeholder must meet its vision tensor in the first forward. A placeholder at or past the store
            # boundary (never seen: the boundary is the last user token) keeps the whole media prefill atomic.
            media = self._m3_media_token_ids()
            tail = req.prompt_tokens[int(boundary) - start:]
            if media and any(int(t) in media for t in tail):
                logger.warning("MiniMax-M3 media placeholder past the cache boundary %s; atomic media prefill "
                               "without the boundary cut (request %s)", boundary, req.uid)
                boundary = None
        return prefill_pieces(start, end, max(1, int(self.prefill_step_size or 1)), boundary,
                              atomic=atomic or has_media, cuts=plan.get("cuts"))

    def _prefill_m3_aligned_and_sample(self, req: _Request, plan: dict) -> None:
        """Prefill an M3 prompt in chunks at ABSOLUTE positions (every multiple of the prefill step, every turn cut of
        the plan, and the prefix cache's store boundary), then sample from the last row's logits.

        A cold prefill and a continuation after a cache hit at a chunk boundary (or at the store boundary of an
        exact-geometry request) then run the same matrices for every remaining row, so the hit's state and logits
        equal the cold prefill's bit for bit. Media requests run [start, boundary) as one forward (every placeholder
        with its vision tensor), then [boundary, N)."""
        tokens = list(req.prompt_tokens)
        start = len(req.context_tokens)
        if req.cache:
            from ..models.minimax_m3.prefill_geometry import cache_offset

            offset = cache_offset(req.cache)
            if offset is not None and offset != start:
                # "cache mismatch": the scheduler's cache-corruption recovery clears the caches and re-runs the
                # request from a cold prefill instead of continuing from rows that are not this context.
                raise ValueError(
                    f"MiniMax-M3 aligned prefill cache mismatch: cache holds {offset} rows but the request context "
                    f"has {start}"
                )
        pieces = self.m3_prefill_pieces(req, plan)
        _prefill_keep_alloc = _prefill_keep_alloc_enabled()
        _clear_each = self._m3_clear_each_chunk()
        _reserved = None
        groups = self._m3_piece_groups(req, pieces)
        with self._stream_context():
            for gi, group in enumerate(groups):
                a = group[0][0]
                if _reserved is None:
                    _reserved = self._m3_reserve_prefill(req.cache, start + len(tokens) - a)
                last = gi == len(groups) - 1
                if len(group) == 1:
                    b = group[0][1]
                    chunk = tokens[a - start: b - start]
                    lookahead = None if last else tokens[b - start: groups[gi + 1][0][1] - start]
                    logits = self._model_call(chunk, req, lookahead=lookahead)  # forwards+clears pixel_values
                else:
                    # Several pieces of one chunk (turn cuts / the store boundary inside it): one layer-major forward,
                    # every piece's rows exactly its own forward's (MiniMaxM3Model.forward_pieces).
                    parts = [tokens[x - start: y - start] for x, y in group]
                    chunk = [t for part in parts for t in part]
                    _, _, logits = self.model.prefill_group([mx.array([part], dtype=mx.int32) for part in parts],
                                                            req.cache, logits_last=last)
                if not last:
                    # Only the cache rows are kept from a non-final piece: evaluate them, not the logits (the
                    # lm_head over every row of a chunk is never read).
                    mx.eval([layer.state for layer in req.cache])
                    del logits
                    req.context_tokens.extend(chunk)
                    if req.logits_processors:
                        req.token_context.update_and_fetch(mx.array(chunk, dtype=mx.int32))
                    self._sync()
                    if not _prefill_keep_alloc and _clear_each and hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                    continue
                (logits,) = self._eval_on_stream(logits)
                if getattr(req, "prefill_started", None) is not None:
                    req.prefill_usage = {
                        "tokens": len(tokens),
                        "seconds": time.perf_counter() - req.prefill_started,
                        "scope": "model_prefill_and_prompt_state",
                    }
                    req.prefill_started = None
                logits = logits[:, -1, :]
                input_tokens = mx.array([chunk[-1]], dtype=mx.int32)
                req.next_token, req.next_logprobs = self._sample_from_logits(logits, req, input_tokens)
                req.next_token_materialized = True
                req.next_logprobs_materialized = req.next_logprobs is not None
                req.context_tokens.extend(chunk)
                if req.logits_processors:
                    req.token_context.update_and_fetch(mx.array(chunk, dtype=mx.int32))
        self._sync()
        if not _prefill_keep_alloc and not _clear_each and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def _m3_piece_groups(self, req: _Request, pieces: list) -> list:
        """The aligned plan's pieces grouped by prefill chunk (``a // step``) for one layer-major forward each
        (``Model.prefill_group``), or one group per piece: a model without ``prefill_group``, a media request (its
        first forward carries the vision tensors), or ``VMLX_M3_PREFILL_GROUP=0``."""
        step = max(1, int(self.prefill_step_size or 1))
        if (not callable(getattr(self.model, "prefill_group", None))
                or req.pixel_values is not None or req.pixel_values_videos is not None
                or os.environ.get("VMLX_M3_PREFILL_GROUP", "1").strip().lower() in {"0", "false", "off", "no"}):
            return [[p] for p in pieces]
        groups: list = []
        for p in pieces:
            if groups and groups[-1][0][0] // step == p[0] // step:
                groups[-1].append(p)
            else:
                groups.append([p])
        return groups

    def _start_request(self, req: _Request) -> Response:
        self._refresh_thread_stream()
        if req.cache is None:
            req.cache = self._make_new_cache()
        # usage.vmlx_prefill (the chat footer's pp/s) for EVERY family on this
        # generator, not only Naive. Lookup/restore and cache creation precede
        # this producer clock. It covers chunk prefill and detached N-1 prompt
        # state up to completed final-position logits, but neither the first
        # sampler nor the following decode forward.
        req.prefill_usage = None
        req.prefill_started = time.perf_counter()
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
            _root = int(getattr(req, "stable_root_split", 0) or 0)
            if 0 < _root < len(_cold_tokens) and self._cache_uses_openpangu(req.cache):
                # Stable root: prefill exactly the system turn, keep its exact typed state (the clone is deep, like
                # the N-1 snapshot: prefill continues on the live cache), then prefill the rest. The scheduler
                # stores it as its own SSD record keyed by the system turn + the next token.
                self._prefill(_cold_tokens[:_root], req)
                req.stable_root_snapshot = self._clone_admissible_prompt_cache_snapshot(req.cache)
                _cold_tokens = _cold_tokens[_root:]
            _cold_tokens = self._prefill_naive_boundaries(_cold_tokens, req)
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
            if getattr(self, "mixed_swa_boundary_capture", False):
                req.mixed_swa_boundary = self._capture_mixed_swa_boundary(req)
        uses_openpangu = self._cache_uses_openpangu(req.cache)
        uses_glm5_next = self._cache_uses_glm5_next(req.cache)
        uses_dsv41 = self._cache_uses_dsv41(req.cache)
        # openPangu and GLM each own a path-dependent native state unit. Capture
        # the immutable N-1 boundary BEFORE the final prompt token is consumed;
        # post-decode convolution/recurrent/indexer state cannot be rewound.
        _phase_trace = _naive_prefill_phase_trace_enabled(self.model)
        if _phase_trace:
            _snapshot_t0 = time.perf_counter()
        typed_prompt_snapshot = self._clone_naive_prompt_snapshot(req)
        if getattr(req, "boundary_windows", None) and typed_prompt_snapshot is not None:
            req.boundary_checkpoints = self._assemble_naive_boundaries(req, typed_prompt_snapshot)
        if _phase_trace:
            _snapshot_done = time.perf_counter()
            _log_naive_prefill_phase(
                req,
                "prompt_snapshot_completed",
                prior_context_tokens=len(req.context_tokens),
                snapshot_ms=(_snapshot_done - _snapshot_t0) * 1000,
                snapshot_available=typed_prompt_snapshot is not None,
            )
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
        elif uses_dsv41:
            # Exact N-1 boundary, captured before the final prompt token is fed (see helper).
            typed_prompt_snapshot = self._clone_dsv41_prompt_boundary(req)
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
            if uses_openpangu or uses_glm5_next or uses_dsv41 or typed_prompt_snapshot is not None
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

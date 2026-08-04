# SPDX-License-Identifier: Apache-2.0
"""DSV4-Flash custom batch generator.

Bypasses ``mlx_lm.generate.BatchGenerator`` entirely for DSV4-Flash JANGTQ
bundles because the mlx_lm code path triggers ``mx.eval`` /
``mx.async_eval`` calls that traverse the tensor stream graph and hit
``RuntimeError: There is no Stream(gpu, N) in current thread.`` on the
llm-worker step executor (DSV4 model forward allocates internal Metal
streams that don't survive the cross-thread context).

This implementation mirrors the canonical ``jang_tools.dsv4.runtime.generate``
path — same prompt encode → prefill → decode → parse loop, but exposes the BatchGenerator
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
import time
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
DSV4_USER_ID = 128803
DSV4_ASSISTANT_ID = 128804
DEFAULT_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS = 256
DEFAULT_DSV4_LONG_PREFILL_THRESHOLD_TOKENS = 12_288
DEFAULT_DSV4_LONG_PREFILL_STEP_TOKENS = 512
DSV4_NATIVE_BLOCK_SIZE = 256
DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS = 8


def dsv4_max_prefill_tokens() -> int:
    """Return an explicit operator prefill guard, or zero when unset.

    Native DSV4 admission is already bounded by the architecture-aware cache
    estimator and the bundle's declared context limit.  This environment
    variable is therefore an opt-in operational ceiling, not a second implicit
    32K model limit.
    """
    raw = os.environ.get("DSV4_MAX_PREFILL_TOKENS")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid DSV4_MAX_PREFILL_TOKENS=%r; ignoring explicit prefill guard",
            raw,
        )
        return 0


def dsv4_prefill_step_policy(default_step: int) -> tuple[int, bool]:
    """Resolve the configured DSV4 prefill ceiling and diagnostic override."""
    try:
        raw = os.environ.get("DSV4_PREFILL_STEP_SIZE")
        configured = int(raw) if raw is not None else int(default_step or 2048)
    except (TypeError, ValueError):
        configured = int(default_step or 2048)
    if configured <= 0:
        return 1 << 30, True
    return max(1, configured), False


def dsv4_effective_prefill_step(
    configured_step: int,
    prompt_tokens: int,
    *,
    cached_tokens: int = 0,
    single_shot: bool = False,
) -> int:
    """Bound a DSV4 prefill chunk by the final native-pool context size.

    Ratio-4 CSA/HCA layers form query-by-compressed-pool score tensors.  A
    2,048-token chunk is live-proven through 12,288 total tokens, while the
    same chunk size triggers the Metal command-buffer watchdog at 32,768.
    A 512-token chunk completes the 32,768-token native q8 snapshot/L2
    discriminator exactly.  Both sizes align with every shipped compression
    ratio (4 and 128), so chunk boundaries do not split compressor windows.

    ``DSV4_PREFILL_STEP_SIZE=0`` remains an explicit diagnostic single-shot
    override.  Normal callers treat the configured value as an upper bound.
    """
    prompt_tokens = max(0, int(prompt_tokens or 0))
    if prompt_tokens <= 0:
        return 1
    if single_shot:
        return prompt_tokens
    step = max(1, int(configured_step or 2048))
    final_context = max(0, int(cached_tokens or 0)) + prompt_tokens
    if final_context > DEFAULT_DSV4_LONG_PREFILL_THRESHOLD_TOKENS:
        step = min(step, DEFAULT_DSV4_LONG_PREFILL_STEP_TOKENS)
    return min(prompt_tokens, step)


def dsv4_prompt_snapshot_min_tokens() -> int:
    """Minimum DSV4 N-1 prompt-key length that donates a sync snapshot.

    DSV4's composite prompt-boundary snapshot is correctness-safe for prefix/L2
    store, but live timing showed it can cost seconds on tiny prompts. Short
    prompts are cheap to re-prefill on the next request, so default to only
    snapshotting block-sized-or-larger prefix keys. Operators can lower this
    for diagnostics or cache experiments without changing sampling behavior.
    """
    raw = os.environ.get("DSV4_PROMPT_SNAPSHOT_MIN_TOKENS")
    if raw is None:
        raw = os.environ.get("VMLINUX_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS")
    if raw is None:
        return DEFAULT_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid DSV4 prompt snapshot threshold %r; using default %d",
            raw,
            DEFAULT_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS,
        )
        return DEFAULT_DSV4_PROMPT_SNAPSHOT_MIN_TOKENS


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
    # Clean prompt-boundary cache snapshot, captured immediately after
    # prefill completes and BEFORE decode advances the live cache. Used
    # by scheduler to populate prefix cache / L2 disk store with state
    # that has no output-side contamination — solves the SWA wrap +
    # CSA/HCA pool-drift problem without sacrificing cache hit rate.
    prompt_snapshot: Optional[List[Any]] = None
    # Request-local cache bypass.  The generator can be configured to capture
    # prompt snapshots globally, but skip_prefix_cache/cache_salt must suppress
    # the associated native block-delta work for this individual request too.
    capture_prompt_snapshot: bool = True


@dataclass
class _Response:
    """Mirror of mlx_lm.generate.BatchGenerator.Response (subset we use)."""
    uid: int
    token: int
    logprobs: Any
    finish_reason: Optional[str] = None
    current_state: Any = None
    match_sequence: Any = None
    # `prompt_cache` is the LIVE cache used for decode (kept for back-compat
    # with mlx_lm.BatchGenerator semantics that the scheduler expects).
    # `prompt_cache_snapshot` is the clean prefill-time copy used for store.
    prompt_cache: Any = None
    prompt_cache_snapshot: Any = None


class DSV4BatchGenerator:
    """Drop-in replacement for ``mlx_lm.generate.BatchGenerator`` scoped to
    DSV4-Flash. Single-request at a time. Idempotent stream pinning.

    Public API mirrors the subset that vmlx scheduler uses:
        insert(prompts, max_tokens, caches, samplers, logits_processors,
               state_machines, capture_prompt_snapshots)
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
        capture_prompt_snapshot: bool = True,
        prompt_snapshot_max_bytes: Optional[int] = None,
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
        # DSV4 has learned role-boundary tokens. If a registry/config path
        # forgets them, generation can continue into fake user/assistant turns.
        self.stop_tokens.update({DSV4_USER_ID, DSV4_ASSISTANT_ID})
        _log_stop_tokens = sorted(
            self.stop_tokens,
            key=lambda item: (isinstance(item, tuple), repr(item)),
        )
        logger.info(
            f"DSV4Gen: stop_tokens at construction = {_log_stop_tokens} "
            f"(DSV4_EOS_ID={DSV4_EOS_ID} always-checked separately)"
        )
        # DSV4 prefill defaults to bounded chunks.
        #
        # v1.5.6 (commit 00a78db4) established that chunking the DSV4
        # prefill corrupts the compressor + indexer pool state mid-decode
        # (broadcast_shapes mismatch on subsequent decode steps). v1.5.15
        # silently re-introduced chunking at 512 with a comment claiming
        # "current DeepseekV4Cache accumulates pool state correctly across
        # calls" — this is unverified and contradicted by v1.5.6's
        # empirical 14/14 probe matrix. The jang-tools DSV4 runtime is
        # unchanged between 2.5.18 (v1.5.10 baseline) and 2.5.23 (current),
        # so the chunking corruption v1.5.6 documented stayed latent until the
        # materialized PoolQuantizedV4Cache path was repaired.
        #
        # v1.5.49+ live installed-app stress showed single-shot DSV4 prefill can
        # transiently hit the 128GB machine limit on a ~10k-token prompt, while
        # the 2048-token step path preserved native block/L2 cache hits and kept
        # peak memory near the resident model size. Operators can still force
        # legacy single-shot with DSV4_PREFILL_STEP_SIZE=0 for diagnostics.
        self.prefill_step_size, self._prefill_single_shot = (
            dsv4_prefill_step_policy(prefill_step_size)
        )
        self.prefill_batch_size = 1  # always 1 for DSV4
        self.completion_batch_size = 1
        self.max_kv_size = max_kv_size
        self.capture_prompt_snapshot = bool(capture_prompt_snapshot)
        self.prompt_snapshot_max_bytes = (
            None
            if prompt_snapshot_max_bytes is None
            else max(0, int(prompt_snapshot_max_bytes))
        )
        self.prompt_snapshot_oversize_skips = 0
        self.prompt_snapshot_headroom_skips = 0
        self.prompt_snapshot_last_estimated_bytes = 0
        self.prompt_snapshot_last_headroom_bytes = 0
        self.prompt_snapshot_min_tokens = dsv4_prompt_snapshot_min_tokens()
        self._uid_count = 0
        self._requests: List[_Request] = []
        # Pin a concrete MLX stream owned by this generator. Using the
        # device object implicitly resolves Stream(gpu, 0), which can be
        # absent in the scheduler worker thread on cache-hit replay.
        self._device = mx.default_device()
        self._stream = stream or mx.default_stream(self._device)

    def _trace_timing(self, event: str, start: float, uid: Optional[int] = None, **fields: Any) -> None:
        if os.environ.get("VMLINUX_DSV4_TRACE_TIMINGS", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        try:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.info(
                "DSV4 timing: component=generator event=%s uid=%s elapsed_ms=%.3f %s",
                event,
                uid,
                elapsed_ms,
                extra,
            )
        except Exception:
            pass

    def _refresh_thread_stream(self):
        """Bind the generator to this worker thread's default MLX stream."""
        try:
            self._stream = mx.default_stream(self._device)
        except Exception:
            pass

    def _sync(self):
        if not hasattr(mx, "synchronize"):
            return
        try:
            mx.synchronize(self._stream)
        except TypeError:
            mx.synchronize()

    def _sampled_token_id(self, sampled) -> int:
        """Materialize a sampled token on this worker thread's stream."""
        with mx.stream(self._stream):
            try:
                # Re-home the scalar onto the active worker stream before
                # tolist()/item() forces synchronization. Cache-hit replay can
                # otherwise leave the sampled scalar associated with a stale
                # Stream(gpu, 0) object from an earlier request/thread.
                sampled = sampled + mx.zeros_like(sampled)
            except Exception:
                pass
            # Scalar materialization is the required sync point. Calling
            # _sync() here first adds a second host/GPU barrier to every
            # decode token on the DSV4 hot path.
            return int(sampled.tolist()[0]) if hasattr(sampled, "tolist") else int(sampled.item())

    @staticmethod
    def _iter_cache_state_arrays(value):
        if isinstance(value, mx.array):
            yield value
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from DSV4BatchGenerator._iter_cache_state_arrays(child)
            return
        if isinstance(value, (tuple, list)):
            for child in value:
                yield from DSV4BatchGenerator._iter_cache_state_arrays(child)

    @staticmethod
    def _realize_decode_cache_state(cache, *, step: int | None = None) -> None:
        """Optionally detach DSV4 cache graphs after a decode forward.

        DSV4 composite caches retain pooled arrays and rotating KV updates.
        Realizing their state collapses the per-token lazy graph on MLX while
        preserving the cache values. It is opt-in during evaluation because
        the extra state barrier can trade throughput for long-run residency.
        """
        periodic = os.environ.get("VMLX_DSV4_REALIZE_CACHE_STATE_EVERY", "").strip()
        if periodic:
            try:
                every = int(periodic)
            except ValueError:
                every = 0
            if every <= 0 or step is None or int(step) % every != 0:
                return
        elif os.environ.get("VMLX_DSV4_REALIZE_CACHE_STATE", "0").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            return
        arrays = []
        for layer_cache in cache or []:
            for attr in ("storage_state", "state"):
                try:
                    state = getattr(layer_cache, attr)
                except Exception:
                    continue
                arrays.extend(DSV4BatchGenerator._iter_cache_state_arrays(state))
                if arrays:
                    break
        if arrays:
            mx.eval(*arrays)

    # ---------- helpers ----------
    def _make_new_cache(self):
        if hasattr(self.model, "make_cache"):
            return self.model.make_cache()
        from mlx_lm.models.cache import KVCache
        # 43 layers for DSV4-Flash (caller will get this right via model.make_cache)
        return [KVCache() for _ in range(getattr(self.model, "n_layers", 43))]

    @staticmethod
    def _snapshot_dsv4_cache(cache_list: List[Any]) -> Optional[List[Any]]:
        """Deep-copy a DSV4 cache list at the prompt boundary.

        Captures a clean snapshot of `DeepseekV4Cache` (or any other cache
        class in the list) IMMEDIATELY after prefill, before decode starts
        advancing the live cache. This snapshot represents the prompt-only
        state — no SWA wrap, no CSA/HCA pool drift from output tokens, no
        cumulative contamination.

        Used by scheduler to populate prefix cache + L2 disk store. The
        guard at `_truncate_cache_to_prompt_length` is replaced by snapshot
        consumption: when a request has a snapshot, scheduler stores that
        directly (no rewind attempt). When no snapshot, the conservative
        guard still fires.

        Returns a NEW list of cache objects, deep-copied via numpy
        roundtrip. Returns None if any layer fails to snapshot (caller
        falls through to the conservative no-store path).
        """
        try:
            import numpy as np
            from jang_tools.dsv4.mlx_model import DeepseekV4Cache
        except Exception:
            return None

        _force_realize = getattr(mx, "eval")
        # Track per-snapshot copy failures so we can hard-fail the whole
        # snapshot if any LEAF array fails to round-trip. Silent None
        # leaves were a real bug: if any DSV4 cache
        # leaf is bf16, np.array(bf16_mx_array) raises a buffer-format
        # error and would have left a None in place of real state →
        # corrupted snapshot replays as garbage on cache hit.
        copy_errors: List[str] = []

        def _arr_copy(a):
            """Force-realize and deep-copy an mx.array via numpy.

            Casts bf16 → fp32 before numpy roundtrip because direct
            np.array() on a bfloat16 mlx array raises:
                "Item size 2 for PEP 3118 buffer format string B does
                 not match the dtype B item size 1"
            Restores fp32 → original dtype on the way back so the
            snapshot's per-leaf dtype matches the live cache.
            """
            if a is None:
                return None
            try:
                _force_realize(a)
                src_dtype = a.dtype
                # bf16 / non-numpy-mappable dtypes need fp32 staging.
                # np.array() on bf16 raises buffer-format error.
                use_fp32_stage = (
                    str(src_dtype).endswith("bfloat16")
                    or str(src_dtype).endswith("float8")
                )
                if use_fp32_stage:
                    a_fp32 = a.astype(mx.float32)
                    np_arr = np.array(a_fp32, copy=True)
                    return mx.array(np_arr, dtype=mx.float32).astype(src_dtype)
                return mx.array(np.array(a, copy=True), dtype=src_dtype)
            except Exception as e:
                copy_errors.append(f"{type(a).__name__}/{getattr(a,'dtype','?')}: {e}")
                return None

        def _tree_copy(value):
            """Deep-copy every MLX leaf while preserving tagged state trees."""
            if value is None:
                return None
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                return _arr_copy(value)
            if isinstance(value, tuple):
                return tuple(_tree_copy(item) for item in value)
            if isinstance(value, list):
                return [_tree_copy(item) for item in value]
            if isinstance(value, dict):
                return {key: _tree_copy(item) for key, item in value.items()}
            return value

        snapshots: List[Any] = []

        def _is_dsv4_cache(obj) -> bool:
            try:
                return any(
                    cls.__name__ in {"DeepseekV4Cache", "PoolQuantizedV4Cache"}
                    for cls in type(obj).__mro__
                )
            except Exception:
                return False

        for layer_cache in cache_list:
            try:
                cls_name = type(layer_cache).__name__
                if _is_dsv4_cache(layer_cache):
                    # DSV4 composite: state = (local_kv, comp_state, idx_state)
                    local = getattr(layer_cache, "local", None)
                    sliding_window = int(getattr(local, "max_size", 128) or 128)
                    compress_ratio = getattr(layer_cache, "compress_ratio", None)

                    cache_cls = DeepseekV4Cache
                    if cls_name == "PoolQuantizedV4Cache":
                        # Preserve the native DSV4 pool codec across the clean
                        # prompt-boundary snapshot. Downgrading this object to
                        # DeepseekV4Cache changes how subsequent CSA/HCA pool
                        # rows are appended after a prefix-cache restore.
                        from jang_tools.dsv4.pool_quant_cache import (
                            PoolQuantizedV4Cache,
                        )

                        cache_cls = PoolQuantizedV4Cache

                    new_cache = cache_cls(
                        sliding_window=sliding_window,
                        compress_ratio=compress_ratio,
                    )

                    # PoolQuantizedV4Cache exposes a separate lossless storage
                    # tree containing its retained q8 codes.  The attention-
                    # facing ``state`` property materializes those pools as
                    # BF16; assigning that view to a new cache would perform a
                    # second lossy encode at every prompt snapshot.
                    state_attr = (
                        "storage_state"
                        if cls_name == "PoolQuantizedV4Cache"
                        and hasattr(layer_cache, "storage_state")
                        else "state"
                    )
                    state = getattr(layer_cache, state_attr)
                    if state is not None:
                        setattr(new_cache, state_attr, _tree_copy(state))
                    try:
                        new_cache.meta_state = layer_cache.meta_state
                    except Exception:
                        pass
                    snapshots.append(new_cache)
                elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                    from mlx_lm.models.cache import KVCache, RotatingKVCache
                    if isinstance(layer_cache, RotatingKVCache):
                        # DSV4's zero-compression layers are still native SWA
                        # layers.  Their cache constructor requires the exact
                        # window geometry, and their physical ring pointer is
                        # part of the prompt-boundary state.  Calling the class
                        # with no arguments used to reject the whole DSV4
                        # snapshot as soon as these layers became bounded.
                        nc = RotatingKVCache(
                            max_size=int(getattr(layer_cache, "max_size", 128)),
                            keep=int(getattr(layer_cache, "keep", 0)),
                        )
                    else:
                        nc = type(layer_cache)() if "Cache" in cls_name else KVCache()
                    try:
                        k = layer_cache.keys
                        v = layer_cache.values
                        if k is not None and v is not None:
                            nc.keys = _arr_copy(k)
                            nc.values = _arr_copy(v)
                        try:
                            nc.meta_state = layer_cache.meta_state
                        except Exception:
                            try:
                                nc.offset = int(getattr(layer_cache, "offset", 0) or 0)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    snapshots.append(nc)
                else:
                    snapshots.append(None)
            except Exception as e:
                logger.debug(f"DSV4 snapshot failed for layer: {e}")
                return None
        # Hard-fail the whole snapshot if ANY leaf copy reported an
        # error. Silent None leaves would corrupt cache replays.
        if copy_errors:
            logger.warning(
                f"DSV4 snapshot REJECTED: {len(copy_errors)} leaf copy "
                f"errors (first 3): {copy_errors[:3]}"
            )
            return None
        return snapshots

    @staticmethod
    def _estimate_dsv4_cache_nbytes(cache_list: List[Any]) -> int:
        """Count the nested DSV4 prompt state without realizing or copying it.

        ``DeepseekV4Cache.state`` is a tuple of local-KV, compressor-pool, and
        sparse-indexer tuples.  The generic cache estimator intentionally only
        walks one state level and therefore reports this composite as zero.
        Snapshot admission must count every nested MLX leaf or the guard can
        still attempt the exact multi-gigabyte copy it is meant to prevent.
        """

        seen: set[int] = set()

        def _walk(value: Any) -> int:
            if value is None:
                return 0
            value_id = id(value)
            if value_id in seen:
                return 0
            seen.add(value_id)
            if isinstance(value, dict):
                return sum(_walk(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(_walk(item) for item in value)
            nbytes = getattr(value, "nbytes", None)
            if isinstance(nbytes, (int, float)):
                return max(0, int(nbytes))
            try:
                # PoolQuantizedV4Cache.state is the attention-facing view and
                # materializes every retained q8 pool segment.  Admission is
                # specifically meant to avoid a long-context memory spike, so
                # count the lossless encoded tree without decoding it.
                state = (
                    getattr(value, "storage_state")
                    if type(value).__name__ == "PoolQuantizedV4Cache"
                    and hasattr(value, "storage_state")
                    else getattr(value, "state")
                )
            except Exception:
                return 0
            return 0 if state is value else _walk(state)

        return _walk(cache_list)

    def _delta_capture_admitted(
        self,
        cache_list: List[Any],
        token_count: int,
    ) -> bool:
        """Reject a finite-backend delta donation before records allocate."""

        backend_limit = self.prompt_snapshot_max_bytes
        cached_tokens = 0
        for layer_cache in cache_list or ():
            try:
                cached_tokens = max(
                    cached_tokens,
                    int(getattr(layer_cache, "offset", 0) or 0),
                )
            except (TypeError, ValueError):
                continue
        final_tokens = cached_tokens + max(0, int(token_count or 0))
        model = getattr(self.model, "_model", self.model)
        config = (
            getattr(model, "args", None)
            or getattr(model, "config", None)
            or getattr(self.model, "config", None)
        )
        pool_quant_enabled = any(
            type(layer_cache).__name__ == "PoolQuantizedV4Cache"
            for layer_cache in cache_list or ()
        )
        try:
            from .memory_limits import (
                estimate_dsv4_delta_transport_bytes_from_config,
            )

            estimated = estimate_dsv4_delta_transport_bytes_from_config(
                config,
                cached_tokens,
                final_tokens,
                pool_quant_enabled=pool_quant_enabled,
                block_size=DSV4_NATIVE_BLOCK_SIZE,
                anchor_interval_blocks=DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS,
            )
        except Exception as exc:
            logger.warning(
                "DSV4Gen: native delta pre-capture estimate failed: %s",
                exc,
            )
            estimated = None
        if estimated is None and backend_limit is None:
            # An unlimited cache backend does not waive the Metal headroom
            # guard.  Unknown model configs cannot predict future delta
            # growth, but the already-retained native state still provides a
            # truthful lower bound that must fit before capture starts.
            estimated = self._estimate_dsv4_cache_nbytes(cache_list)
        if estimated is not None:
            self.prompt_snapshot_last_estimated_bytes = int(estimated)
        if backend_limit is not None and (
            estimated is None or int(estimated) > int(backend_limit)
        ):
            self.prompt_snapshot_oversize_skips += 1
            logger.warning(
                "DSV4Gen: skipping native delta capture before allocation: "
                "estimated=%s backend_limit=%.1fMB start=%d end=%d",
                "unknown"
                if estimated is None
                else f"{int(estimated) / (1024 * 1024):.1f}MB",
                int(backend_limit) / (1024 * 1024),
                cached_tokens,
                final_tokens,
            )
            return False

        try:
            from .memory_limits import (
                get_effective_metal_working_set_bytes,
                get_metal_ws_guard_threshold,
            )

            active_bytes, max_ws_bytes = get_effective_metal_working_set_bytes(mx)
            threshold = get_metal_ws_guard_threshold() / 100.0
            headroom = max(0, int(max_ws_bytes * threshold) - int(active_bytes))
        except Exception as exc:
            logger.debug(
                "DSV4 native delta pre-capture Metal headroom lookup failed: %s",
                exc,
            )
            headroom = 0
            max_ws_bytes = 0
        self.prompt_snapshot_last_headroom_bytes = headroom
        if (
            estimated is not None
            and max_ws_bytes > 0
            and int(estimated) > headroom
        ):
            self.prompt_snapshot_oversize_skips += 1
            self.prompt_snapshot_headroom_skips += 1
            logger.warning(
                "DSV4Gen: skipping native delta capture before allocation: "
                "estimated %.1fMB exceeds safe Metal headroom (%.1fMB)",
                int(estimated) / (1024 * 1024),
                headroom / (1024 * 1024),
            )
            return False
        return True

    @staticmethod
    def _copy_delta_tree(value: Any) -> Any:
        """Detach cache-record leaves from the live mutable DSV4 cache."""
        if value is None:
            return None
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            copied = value + mx.zeros_like(value)
            mx.eval(copied)
            return copied
        if isinstance(value, tuple):
            return tuple(DSV4BatchGenerator._copy_delta_tree(item) for item in value)
        if isinstance(value, list):
            return [DSV4BatchGenerator._copy_delta_tree(item) for item in value]
        if isinstance(value, dict):
            return {
                key: DSV4BatchGenerator._copy_delta_tree(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _is_dsv4_composite_cache(cache: Any) -> bool:
        try:
            return any(
                cls.__name__ in {"DeepseekV4Cache", "PoolQuantizedV4Cache"}
                for cls in type(cache).__mro__
            )
        except Exception:
            return False

    @classmethod
    def _reset_dsv4_block_records(
        cls, cache_list: List[Any], start_token: int
    ) -> None:
        start = int(start_token)
        if start < 0 or start % DSV4_NATIVE_BLOCK_SIZE:
            raise ValueError(
                "DSV4 delta capture must start at a 256-token checkpoint; "
                f"got {start}"
            )
        for cache in cache_list:
            cache._vmlx_dsv4_block_records = []
            cache._vmlx_dsv4_delta_next_token = start

    @classmethod
    def _rotating_block_record(
        cls,
        cache: Any,
        start_token: int,
        end_token: int,
        *,
        force_anchor: bool,
    ) -> tuple:
        interval = DSV4_NATIVE_BLOCK_SIZE * DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS
        anchored = bool(force_anchor or int(end_token) % interval == 0)
        if not anchored:
            return ("rotating_kv_pending", type(cache).__name__)
        try:
            from jang_tools.dsv4.cache_delta import canonical_rotating_window

            keys, values, max_size, keep, offset, idx = (
                canonical_rotating_window(
                    cache,
                    expected_offset=int(end_token),
                )
            )
        except Exception as exc:
            raise ValueError("cannot canonicalize DSV4 rotating anchor") from exc
        return (
            "rotating_kv",
            keys,
            values,
            max_size,
            keep,
            offset,
            idx,
        )

    @classmethod
    def _capture_dsv4_block_record(
        cls,
        cache_list: List[Any],
        start_token: int,
        end_token: int,
        *,
        force_anchor: bool = False,
        append_safe: bool = False,
        replace_last: bool = False,
    ) -> None:
        start = int(start_token)
        end = int(end_token)
        if append_safe and (
            not force_anchor
            or end - start != DSV4_NATIVE_BLOCK_SIZE
            or start % DSV4_NATIVE_BLOCK_SIZE
            or end % DSV4_NATIVE_BLOCK_SIZE
        ):
            raise ValueError(
                "DSV4 append-safe anchors must be forced at one aligned "
                f"256-token block; got [{start}, {end})"
            )
        for cache in cache_list:
            records = getattr(cache, "_vmlx_dsv4_block_records", None)
            if not isinstance(records, list):
                raise ValueError("DSV4 cache layer has no block-record collector")
            if cls._is_dsv4_composite_cache(cache):
                export = getattr(cache, "export_block_delta", None)
                if not callable(export):
                    raise ValueError(
                        "installed jang-tools lacks DSV4 native block-delta export"
                    )
                record = export(
                    start,
                    end,
                    block_size=DSV4_NATIVE_BLOCK_SIZE,
                    anchor_interval_blocks=DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS,
                    force_anchor=force_anchor,
                )
            else:
                record = cls._rotating_block_record(
                    cache,
                    start,
                    end,
                    force_anchor=force_anchor,
                )
            if append_safe:
                if isinstance(record, dict):
                    anchor = record.get("anchor")
                    if (
                        not isinstance(anchor, dict)
                        or int(anchor.get("tokens", -1)) != end
                    ):
                        raise ValueError(
                            "DSV4 composite append-safe record has no exact anchor"
                        )
                    # JANG's current delta schema recognizes forced checkpoints
                    # as terminal anchors.  Keep that underlying restore contract
                    # intact while adding an engine-owned, stricter capability:
                    # only a complete 256-token boundary explicitly stamped here
                    # may seed a later changed-suffix capture.
                    anchor["append_safe"] = True
                elif (
                    not isinstance(record, (tuple, list))
                    or not record
                    or record[0] != "rotating_kv"
                ):
                    raise ValueError(
                        "DSV4 rotating append-safe record has no exact checkpoint"
                    )
            if replace_last:
                if not records:
                    raise ValueError("cannot replace an absent DSV4 block record")
                records[-1] = record
            else:
                records.append(record)
            cache._vmlx_dsv4_delta_next_token = end

    @classmethod
    def _capture_dsv4_completed_blocks(
        cls, cache_list: List[Any], target_token: int
    ) -> None:
        if not cache_list:
            return
        next_token = int(
            getattr(cache_list[0], "_vmlx_dsv4_delta_next_token", 0) or 0
        )
        complete_end = (int(target_token) // DSV4_NATIVE_BLOCK_SIZE) * DSV4_NATIVE_BLOCK_SIZE
        while next_token + DSV4_NATIVE_BLOCK_SIZE <= complete_end:
            cls._capture_dsv4_block_record(
                cache_list,
                next_token,
                next_token + DSV4_NATIVE_BLOCK_SIZE,
            )
            next_token += DSV4_NATIVE_BLOCK_SIZE

    @classmethod
    def _capture_dsv4_append_safe_checkpoint(
        cls, cache_list: List[Any], target_token: int
    ) -> None:
        """Promote the just-completed full block while live state is aligned."""
        if not cache_list:
            return
        target = int(target_token)
        next_token = int(
            getattr(cache_list[0], "_vmlx_dsv4_delta_next_token", 0) or 0
        )
        if (
            target <= 0
            or target % DSV4_NATIVE_BLOCK_SIZE
            or next_token != target
        ):
            raise ValueError(
                "DSV4 append-safe checkpoint must match the completed block "
                f"boundary; target={target} captured={next_token}"
            )
        cls._capture_dsv4_block_record(
            cache_list,
            target - DSV4_NATIVE_BLOCK_SIZE,
            target,
            force_anchor=True,
            append_safe=True,
            replace_last=True,
        )

    @classmethod
    def _capture_dsv4_terminal_anchor(
        cls, cache_list: List[Any], target_token: int
    ) -> None:
        if not cache_list:
            return
        target = int(target_token)
        next_token = int(
            getattr(cache_list[0], "_vmlx_dsv4_delta_next_token", 0) or 0
        )
        if next_token < target:
            cls._capture_dsv4_block_record(
                cache_list,
                next_token,
                target,
                force_anchor=True,
            )
            return
        if next_token != target:
            raise ValueError(
                "DSV4 terminal anchor is behind captured records: "
                f"target={target} captured={next_token}"
            )
        for cache in cache_list:
            records = getattr(cache, "_vmlx_dsv4_block_records", None)
            if not records:
                raise ValueError("DSV4 terminal anchor has no block record")
        cls._capture_dsv4_append_safe_checkpoint(cache_list, target)

    @classmethod
    def _dsv4_delta_transport(cls, cache_list: List[Any]) -> Optional[List[Any]]:
        """Return a zero-copy layer transport for native block/L2 storage."""
        transport: List[dict[str, Any]] = []
        expected_intervals = None
        expected_record_count = None
        for cache in cache_list or ():
            records = getattr(cache, "_vmlx_dsv4_block_records", None)
            if not isinstance(records, list) or not records:
                return None
            if expected_record_count is None:
                expected_record_count = len(records)
            elif len(records) != expected_record_count:
                return None
            intervals = []
            for record in records:
                if isinstance(record, dict):
                    intervals.append(
                        (int(record["start_token"]), int(record["end_token"]))
                    )
                else:
                    # Rotating records share the composite layers' interval
                    # sequence; their tuple payload intentionally omits token
                    # duplication and is paired by list position.
                    intervals.append(None)
            known = [item for item in intervals if item is not None]
            if known:
                if expected_intervals is None:
                    expected_intervals = known
                elif known != expected_intervals:
                    return None
            local = getattr(cache, "local", cache)
            transport.append(
                {
                    "state": None,
                    "class_name": type(cache).__name__,
                    "compress_ratio": getattr(cache, "compress_ratio", None),
                    "sliding_window": int(
                        getattr(local, "max_size", 128) or 128
                    ),
                    "pool_quant": type(cache).__name__ == "PoolQuantizedV4Cache",
                    "dsv4_block_records": tuple(records),
                }
            )
        if not transport or not expected_intervals:
            return None
        for layer in transport:
            layer["dsv4_record_intervals"] = tuple(expected_intervals)
        return transport

    def _snapshot_admissible_dsv4_cache(
        self, cache_list: List[Any]
    ) -> Optional[List[Any]]:
        """Copy a prompt boundary only when cache budget and Metal permit it."""

        delta_transport = self._dsv4_delta_transport(cache_list)
        if delta_transport is not None:
            # The block records were detached at their owning prefill
            # boundaries. Returning this layer map does not copy the live
            # composite cache or materialize attention-facing q8 pools.
            estimated = self._estimate_dsv4_cache_nbytes(delta_transport)
            self.prompt_snapshot_last_estimated_bytes = estimated
            backend_limit = self.prompt_snapshot_max_bytes
            if backend_limit is not None and estimated > backend_limit:
                self.prompt_snapshot_oversize_skips += 1
                logger.warning(
                    "DSV4Gen: skipping native delta transport: estimated %.1fMB "
                    "exceeds every enabled cache backend limit (%.1fMB)",
                    estimated / (1024 * 1024),
                    backend_limit / (1024 * 1024),
                )
                return None
            return delta_transport

        estimated = self._estimate_dsv4_cache_nbytes(cache_list)
        self.prompt_snapshot_last_estimated_bytes = estimated

        backend_limit = self.prompt_snapshot_max_bytes
        if backend_limit is not None and estimated > backend_limit:
            self.prompt_snapshot_oversize_skips += 1
            logger.warning(
                "DSV4Gen: skipping prompt snapshot before copy: estimated %.1fMB "
                "exceeds every enabled cache backend limit (%.1fMB)",
                estimated / (1024 * 1024),
                backend_limit / (1024 * 1024),
            )
            return None

        try:
            from .memory_limits import (
                get_effective_metal_working_set_bytes,
                get_metal_ws_guard_threshold,
            )

            active_bytes, max_ws_bytes = get_effective_metal_working_set_bytes(mx)
            threshold = get_metal_ws_guard_threshold() / 100.0
            headroom = max(0, int(max_ws_bytes * threshold) - int(active_bytes))
        except Exception as exc:
            logger.debug("DSV4 prompt snapshot Metal headroom lookup failed: %s", exc)
            headroom = 0
            max_ws_bytes = 0
        self.prompt_snapshot_last_headroom_bytes = headroom
        if max_ws_bytes > 0 and estimated > headroom:
            self.prompt_snapshot_oversize_skips += 1
            self.prompt_snapshot_headroom_skips += 1
            logger.warning(
                "DSV4Gen: skipping prompt snapshot before copy: estimated %.1fMB "
                "exceeds safe Metal headroom (%.1fMB)",
                estimated / (1024 * 1024),
                headroom / (1024 * 1024),
            )
            return None

        return self._snapshot_dsv4_cache(cache_list)

    def _should_capture_logprobs(self, uid: int) -> bool:
        try:
            from .mamba_cache import _should_capture_generation_logprobs

            return bool(_should_capture_generation_logprobs(self.model, [uid])[0])
        except Exception:
            return False

    def _sample(
        self,
        logits,
        sampler,
        processors,
        recent_tokens,
        generated_tokens=None,
        capture_logprobs: bool = False,
    ):
        """Apply logits processors then sample. logits: (1, vocab).

        `recent_tokens`: full prompt+generated context (used by rep_penalty
            processor — mlx_lm convention).
        `generated_tokens`: GENERATED-ONLY context (used by the hard n-gram
            dedup so the prompt cannot poison the dedup state). When None,
            falls back to `recent_tokens` for backwards compat.
        """
        x = logits
        for p in processors or []:
            # processor signature: fn(token_context, logits) -> logits
            try:
                x = p(recent_tokens, x)
            except TypeError:
                x = p(x)
        sample_fn = sampler or self.fallback_sampler
        if (
            not capture_logprobs
            and getattr(sample_fn, "_vmlx_accepts_logits", False)
        ):
            sampled = sample_fn(x)
            return sampled, None

        # Convert log-probs via logsumexp normalization only when the sampler
        # needs normalized probabilities or the request asked for logprobs.
        # Default vMLX greedy samplers accept raw logits; materializing a
        # full-vocab log-softmax every token is wasted DSV4 decode time.
        logprobs = x - mx.logsumexp(x, axis=-1, keepdims=True)
        sampled = sample_fn(logprobs)
        return sampled, logprobs

    @staticmethod
    def _realize_last_logits(last_logits: Any) -> None:
        mx.eval(last_logits)

    def _prefill_last_logits(
        self,
        token_ids: List[int],
        cache: List[Any],
        *,
        realize_before_clear: bool = True,
        capture_block_deltas: bool = False,
        force_terminal_anchor: bool = False,
    ):
        """DSV4 prefill returning logits for the last prompt token.

        Default behavior is bounded-step prefill. The production default uses
        a 2048-token step after installed-app cache-hit validation showed it
        preserves native block/L2 cache hits while avoiding the transient
        memory spike seen with large single-shot prompts. Operators can still
        force legacy single-shot with DSV4_PREFILL_STEP_SIZE=0 for diagnostics.
        Each step runs against the same native composite cache and clears
        transient MLX buffers between chunks.
        """
        if not token_ids:
            return None
        all_ids = mx.array(token_ids, dtype=mx.int32)[None, :]
        last_logits = None
        total = len(token_ids)
        cached_tokens = 0
        for layer_cache in cache or ():
            try:
                cached_tokens = max(
                    cached_tokens,
                    int(getattr(layer_cache, "offset", 0) or 0),
                )
            except (TypeError, ValueError):
                continue
        step = dsv4_effective_prefill_step(
            self.prefill_step_size,
            total,
            cached_tokens=cached_tokens,
            single_shot=self._prefill_single_shot,
        )
        if (
            not self._prefill_single_shot
            and cached_tokens + total
            > DEFAULT_DSV4_LONG_PREFILL_THRESHOLD_TOKENS
            and step < self.prefill_step_size
        ):
            logger.info(
                "DSV4Gen: adaptive long-context prefill step=%s "
                "configured=%s final_context=%s",
                step,
                self.prefill_step_size,
                cached_tokens + total,
            )
        if capture_block_deltas:
            self._reset_dsv4_block_records(cache, cached_tokens)
        append_safe_boundary = 0
        if capture_block_deltas and force_terminal_anchor:
            final_boundary = cached_tokens + total
            preceding_full = (
                final_boundary // DSV4_NATIVE_BLOCK_SIZE
            ) * DSV4_NATIVE_BLOCK_SIZE
            if (
                preceding_full > cached_tokens
                and preceding_full < final_boundary
                and preceding_full
                % (
                    DSV4_NATIVE_BLOCK_SIZE
                    * DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS
                )
            ):
                append_safe_boundary = preceding_full
        off = 0
        while off < total:
            end_off = min(off + step, total)
            if capture_block_deltas:
                absolute_start = cached_tokens + off
                anchor_interval = (
                    DSV4_NATIVE_BLOCK_SIZE * DSV4_NATIVE_ANCHOR_INTERVAL_BLOCKS
                )
                next_anchor = (
                    (absolute_start // anchor_interval) + 1
                ) * anchor_interval
                end_off = min(end_off, next_anchor - cached_tokens)
                if absolute_start < append_safe_boundary:
                    end_off = min(
                        end_off,
                        append_safe_boundary - cached_tokens,
                    )
            chunk = all_ids[:, off:end_off]
            logits = self.model(chunk, cache=cache)
            last_logits = logits[:, -1, :]
            # Force the logits graph and cache side effects to materialize
            # before clearing transient allocator buffers. A stream sync alone
            # does not schedule lazy MLX work.
            if realize_before_clear:
                self._realize_last_logits(last_logits)
            if hasattr(mx, "synchronize"):
                self._sync()
            else:
                mx.eval(last_logits)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            off = end_off
            if capture_block_deltas:
                self._capture_dsv4_completed_blocks(cache, cached_tokens + off)
                if cached_tokens + off == append_safe_boundary:
                    # Capture while every native layer is exactly at this block
                    # boundary. Re-exporting it after the partial tail would ask
                    # JANG to canonicalize offset 331 as offset 256 and fail.
                    self._capture_dsv4_append_safe_checkpoint(
                        cache,
                        append_safe_boundary,
                    )
        if capture_block_deltas and force_terminal_anchor:
            self._capture_dsv4_terminal_anchor(cache, cached_tokens + total)
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
        capture_prompt_snapshots: Optional[List[bool]] = None,
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
        capture_prompt_snapshots = capture_prompt_snapshots or [True] * len(prompts)
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
                capture_prompt_snapshot=bool(capture_prompt_snapshots[i]),
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

    @staticmethod
    def _prompt_starts_in_think(req: _Request) -> bool:
        """Return True when DSV4's prompt leaves generation inside <think>.

        Direct/instruct rail ends the assistant prefix with </think>, while the
        thinking rail ends with <think>. This check is token-based so it follows
        the canonical DSV4 encoder instead of string-rendering assumptions.
        """
        try:
            last_open = len(req.prompt_tokens) - 1 - req.prompt_tokens[::-1].index(THINK_OPEN_ID)
        except ValueError:
            last_open = -1
        try:
            last_close = len(req.prompt_tokens) - 1 - req.prompt_tokens[::-1].index(THINK_CLOSE_ID)
        except ValueError:
            last_close = -1
        return last_open > last_close

    def _effective_max_tokens(self, req: _Request) -> int:
        return req.max_tokens

    def _update_finish_reason_after_token(self, req: _Request, token_id: int) -> None:
        if token_id == DSV4_EOS_ID or token_id in self.stop_tokens:
            req.finish_reason = "stop"
        elif len(req.out_tokens) >= self._effective_max_tokens(req):
            req.finish_reason = "length"
        else:
            req.finish_reason = None

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
        self._refresh_thread_stream()

        # On first call, warm up MLX kernels so the user's prefill
        # doesn't have to JIT-compile every routed-expert / hash-router
        # combo in one shot. We call model() with the smallest possible
        # input (1 token) on a fresh cache, force-evaluate, then drop
        # the warmup state. After this, the user's prefill paths use
        # the precompiled kernels.
        if not self._warmed_up:
            with mx.stream(self._stream):
                try:
                    _warm_cache = self._make_new_cache()
                    _warm_ids = mx.array([[0]], dtype=mx.int32)
                    _ = self.model(_warm_ids, cache=_warm_cache)
                    mx.eval(_)
                    self._sync()
                    logger.info("DSV4Gen: kernel warmup done")
                except Exception as _wexc:
                    logger.warning(f"DSV4Gen: warmup failed (continuing anyway): {_wexc}")
            self._warmed_up = True

        for r in list(self._requests):
            with mx.stream(self._stream):
                if r.cache is None:
                    # Prefill — chunked through one DeepseekV4Cache instance.
                    # This preserves SWA+CSA/HCA state while bounding transient
                    # MLX allocations for long prompts.
                    r.cache = self._make_new_cache()
                    # The ratio-4 indexer is only consulted after 512 pool
                    # rows. For a fresh request whose prompt plus output cap
                    # stays within 2048 tokens, its decode updates are provably
                    # unused; mark only those per-request caches for the
                    # bounded indexer bypass. Prompt prefill stays exact, and
                    # prefix-cache-hit requests retain the full indexer state.
                    if os.environ.get("VMLX_DSV4_SKIP_UNUSED_INDEXER", "1").strip().lower() in {
                        "1", "true", "yes", "on"
                    } and len(r.prompt_tokens) + int(r.max_tokens) <= 2048:
                        for _layer_cache in r.cache:
                            try:
                                _layer_cache._vmlx_dsv4_indexer_skip_decode = True
                            except Exception:
                                pass
                    if not r.prompt_tokens:
                        r.finish_reason = "stop"
                        r.prompt_processed = True
                        continue

                    # TWO-PHASE PREFILL FOR N-1 SNAPSHOT.
                    #
                    # Scheduler stores prefix cache under N-1 token keys
                    # so the last prompt token gets re-fed on cache hit
                    # (avoids positional duplication). Our snapshot must
                    # match that convention or hits drift positionally.
                    #
                    # Phase 1: prefill all but last token, snapshot the
                    #          cache state at N-1.
                    # Phase 2: prefill the last token to advance the
                    #          live cache to N (used for first-token
                    #          decode logits).
                    snapshot_requested = (
                        self.capture_prompt_snapshot
                        and r.capture_prompt_snapshot
                        and len(r.prompt_tokens) >= 2
                        and len(r.prompt_tokens) - 1 >= self.prompt_snapshot_min_tokens
                    )
                    should_capture_snapshot = bool(
                        snapshot_requested
                        and self._delta_capture_admitted(
                            r.cache,
                            len(r.prompt_tokens) - 1,
                        )
                    )

                    if should_capture_snapshot:
                        head_tokens = r.prompt_tokens[:-1]
                        last_token = r.prompt_tokens[-1:]
                        # Phase 1
                        _t_prefill_head = time.perf_counter()
                        _ = self._prefill_last_logits(
                            head_tokens,
                            r.cache,
                            capture_block_deltas=True,
                            force_terminal_anchor=True,
                        )
                        self._trace_timing(
                            "prefill_head",
                            _t_prefill_head,
                            r.uid,
                            tokens=len(head_tokens),
                        )
                        try:
                            _t_snapshot = time.perf_counter()
                            r.prompt_snapshot = self._snapshot_admissible_dsv4_cache(
                                r.cache
                            )
                            self._trace_timing(
                                "prompt_snapshot",
                                _t_snapshot,
                                r.uid,
                                layers=len(r.prompt_snapshot or []),
                            )
                            if r.prompt_snapshot is not None:
                                logger.debug(
                                    f"DSV4Gen: captured N-1 prompt-boundary "
                                    f"snapshot ({len(r.prompt_snapshot)} layers, "
                                    f"N-1={len(head_tokens)} tokens) "
                                    f"for uid={r.uid}"
                                )
                        except Exception as _snap_err:
                            logger.warning(
                                f"DSV4Gen: snapshot capture failed: {_snap_err}"
                            )
                            r.prompt_snapshot = None
                        # Phase 2 — feed the last token, get its logits
                        _t_prefill_last = time.perf_counter()
                        last_logits = self._prefill_last_logits(last_token, r.cache)
                        self._trace_timing(
                            "prefill_last",
                            _t_prefill_last,
                            r.uid,
                            tokens=len(last_token),
                        )
                    else:
                        if (
                            self.capture_prompt_snapshot
                            and r.capture_prompt_snapshot
                            and len(r.prompt_tokens) >= 2
                            and not snapshot_requested
                        ):
                            _t_snapshot_skip = time.perf_counter()
                            self._trace_timing(
                                "prompt_snapshot_skipped",
                                _t_snapshot_skip,
                                r.uid,
                                tokens=len(r.prompt_tokens) - 1,
                                min_tokens=self.prompt_snapshot_min_tokens,
                            )
                            logger.debug(
                                "DSV4Gen: skipped prompt-boundary snapshot for "
                                "short prompt (N-1=%d < min_tokens=%d) uid=%s",
                                len(r.prompt_tokens) - 1,
                                self.prompt_snapshot_min_tokens,
                                r.uid,
                            )
                        else:
                            r.prompt_snapshot = None
                        _t_prefill = time.perf_counter()
                        last_logits = self._prefill_last_logits(r.prompt_tokens, r.cache)
                        self._trace_timing(
                            "prefill_full",
                            _t_prefill,
                            r.uid,
                            tokens=len(r.prompt_tokens),
                        )

                    _t_sample = time.perf_counter()
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                        capture_logprobs=self._should_capture_logprobs(r.uid),
                    )
                    tok_id = self._sampled_token_id(sampled)
                    self._trace_timing(
                        "sample_materialize",
                        _t_sample,
                        r.uid,
                        capture_logprobs=logprobs is not None,
                    )
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    self._update_finish_reason_after_token(r, tok_id)
                    prompt_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                        prompt_cache_snapshot=r.prompt_snapshot,
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
                    # Extend the restored terminal cache to the new N-1
                    # prompt boundary before sampling.  A cache-hit request
                    # may contain a substantial new tail (most notably the
                    # function_call/output records in an Electron tool loop).
                    # Previously we fed that entire tail in one pass, leaving
                    # ``prompt_snapshot`` unset.  The scheduler then had to
                    # keep the older terminal record forever because the live
                    # cache was already contaminated by decode.  After a
                    # process restart the next turn therefore restored only
                    # the pre-tool prefix.
                    #
                    # The restored cache already represents the matched
                    # prefix exactly.  Feed all remaining tokens except the
                    # final one, snapshot that extended cache at the same N-1
                    # boundary used by cold stores, then feed the final token
                    # for first-token logits.  This is correctness-safe for
                    # SWA + CSA/HCA and costs only the uncached tail; it never
                    # re-prefills the matched prefix.
                    extension_snapshot_requested = (
                        self.capture_prompt_snapshot
                        and r.capture_prompt_snapshot
                        and len(r.context_tokens) >= 2
                        and len(r.context_tokens) - 1
                        >= self.prompt_snapshot_min_tokens
                    )
                    tail_head = r.prompt_tokens[:-1]
                    should_capture_extension_snapshot = bool(
                        extension_snapshot_requested
                        and (
                            not tail_head
                            or self._delta_capture_admitted(
                                r.cache,
                                len(tail_head),
                            )
                        )
                    )
                    _t_prefill_tail = time.perf_counter()
                    if should_capture_extension_snapshot:
                        final_prompt_token = r.prompt_tokens[-1:]
                        if tail_head:
                            _ = self._prefill_last_logits(
                                tail_head,
                                r.cache,
                                capture_block_deltas=True,
                                force_terminal_anchor=True,
                            )
                            try:
                                _t_snapshot = time.perf_counter()
                                r.prompt_snapshot = self._snapshot_admissible_dsv4_cache(
                                    r.cache
                                )
                                self._trace_timing(
                                    "cache_hit_extension_snapshot",
                                    _t_snapshot,
                                    r.uid,
                                    layers=len(r.prompt_snapshot or []),
                                    tail_tokens=len(tail_head),
                                )
                                if r.prompt_snapshot is not None:
                                    logger.info(
                                        "DSV4Gen: captured cache-hit N-1 extension "
                                        "snapshot (%d layers, uncached_head=%d, "
                                        "full_N_minus_1=%d) for uid=%s",
                                        len(r.prompt_snapshot),
                                        len(tail_head),
                                        len(r.context_tokens) - 1,
                                        r.uid,
                                    )
                            except Exception as _snap_err:
                                logger.warning(
                                    "DSV4Gen: cache-hit extension snapshot failed: %s",
                                    _snap_err,
                                )
                                r.prompt_snapshot = None
                        else:
                            # An exact N-1 hit already owns the required prompt
                            # boundary. Snapshotting here has no new delta to
                            # transport, so it falls back to a full composite
                            # cache copy that store_cache() immediately discards
                            # as a zero-token extension. Feed only the final
                            # prompt token and retain the existing terminal
                            # record without allocating a hidden whole-cache
                            # duplicate.
                            r.prompt_snapshot = None
                        last_logits = self._prefill_last_logits(
                            final_prompt_token,
                            r.cache,
                        )
                    else:
                        last_logits = self._prefill_last_logits(
                            r.prompt_tokens,
                            r.cache,
                        )
                    self._trace_timing(
                        "cache_hit_tail_prefill",
                        _t_prefill_tail,
                        r.uid,
                        tokens=len(r.prompt_tokens),
                        captured_extension_snapshot=(
                            r.prompt_snapshot is not None
                        ),
                    )
                    _t_sample = time.perf_counter()
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                        capture_logprobs=self._should_capture_logprobs(r.uid),
                    )
                    tok_id = self._sampled_token_id(sampled)
                    self._trace_timing(
                        "sample_materialize",
                        _t_sample,
                        r.uid,
                        capture_logprobs=logprobs is not None,
                    )
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    self._update_finish_reason_after_token(r, tok_id)
                    prompt_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                        prompt_cache_snapshot=r.prompt_snapshot,
                    ))
                else:
                    # Decode
                    if r.finish_reason is not None:
                        # Already finished — just emit nothing. Caller
                        # should remove() it.
                        continue
                    last_id = r.out_tokens[-1]
                    ids = mx.array([[last_id]], dtype=mx.int32)
                    _t_decode_model = time.perf_counter()
                    logits = self.model(ids, cache=r.cache)
                    last_logits = logits[:, -1, :]
                    self._realize_decode_cache_state(r.cache, step=len(r.out_tokens))
                    self._trace_timing(
                        "decode_model",
                        _t_decode_model,
                        r.uid,
                        output_tokens=len(r.out_tokens),
                    )
                    _t_sample = time.perf_counter()
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                        capture_logprobs=self._should_capture_logprobs(r.uid),
                    )
                    tok_id = self._sampled_token_id(sampled)
                    self._trace_timing(
                        "sample_materialize",
                        _t_sample,
                        r.uid,
                        capture_logprobs=logprobs is not None,
                    )
                    r.out_tokens.append(tok_id)
                    self._update_finish_reason_after_token(r, tok_id)
                    gen_resps.append(_Response(
                        uid=r.uid, token=tok_id, logprobs=logprobs,
                        finish_reason=r.finish_reason,
                        prompt_cache=r.cache,
                        prompt_cache_snapshot=r.prompt_snapshot,
                    ))

        return prompt_resps, gen_resps

    def next_generated(self) -> List[_Response]:
        while True:
            prompt_resps, gen_resps = self.next()
            if not gen_resps and prompt_resps:
                continue
            return gen_resps

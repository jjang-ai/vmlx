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
DSV4_USER_ID = 128803
DSV4_ASSISTANT_ID = 128804


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
    forced_think_close: bool = False
    forced_think_close_at: Optional[int] = None
    finalizer_max_tokens: Optional[int] = None
    # Clean prompt-boundary cache snapshot, captured immediately after
    # prefill completes and BEFORE decode advances the live cache. Used
    # by scheduler to populate prefix cache / L2 disk store with state
    # that has no output-side contamination — solves the SWA wrap +
    # CSA/HSA pool-drift problem without sacrificing cache hit rate.
    prompt_snapshot: Optional[List[Any]] = None


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
        # Pin a concrete MLX stream owned by this generator. Using the
        # device object implicitly resolves Stream(gpu, 0), which can be
        # absent in the scheduler worker thread on cache-hit replay.
        self._device = mx.default_device()
        self._stream = stream or mx.default_stream(self._device)

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
            self._sync()
            return int(sampled.tolist()[0]) if hasattr(sampled, "tolist") else int(sampled.item())

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
        state — no SWA wrap, no CSA/HSA pool drift from output tokens, no
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

                    new_cache = DeepseekV4Cache(
                        sliding_window=sliding_window,
                        compress_ratio=compress_ratio,
                    )

                    # Read state tuple via @property (returns nested tuples
                    # of mx.arrays). Deep-copy each leaf array.
                    state = layer_cache.state
                    if state is not None:
                        snap_state = []
                        for sub in state:
                            if sub is None:
                                snap_state.append(None)
                            elif isinstance(sub, (tuple, list)):
                                snap_state.append(tuple(_arr_copy(x) for x in sub))
                            else:
                                snap_state.append(_arr_copy(sub))
                        new_cache.state = tuple(snap_state)
                    try:
                        new_cache.meta_state = layer_cache.meta_state
                    except Exception:
                        pass
                    snapshots.append(new_cache)
                elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                    from mlx_lm.models.cache import KVCache
                    nc = type(layer_cache)() if "Cache" in cls_name else KVCache()
                    try:
                        k = layer_cache.keys
                        v = layer_cache.values
                        if k is not None and v is not None:
                            nc.keys = _arr_copy(k)
                            nc.values = _arr_copy(v)
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
    def _hard_repetition_block(
        logits, recent_tokens, *,
        run_threshold=2, window=20, diversity_window=16, diversity_min=8,
    ):
        """Hard-block degenerate-repetition tokens.

        Three layered detectors, generated-tokens-only:
          (1) Single-token repeat: any token at >=`run_threshold` in last
              `window` positions → -inf. Threshold lowered to 2 to catch
              "Stanford...Stanford" pairs that rep_penalty<<2 can't beat.
          (2) 2-gram + 3-gram extension: block tokens that would extend
              an already-repeated bigram/trigram.
          (3) DIVERSITY COLLAPSE: when the last `diversity_window` decoded
              tokens contain fewer than `diversity_min` unique values,
              the model is in a degenerate attractor that cycles N
              variants (e.g. case/space variants of the same word:
              "VC, VC, fund, fund, venture, venture, Ventures, Ventures").
              Variant-cycling bypasses single-token threshold. Block ALL
              tokens that have appeared at all in the last
              `diversity_window` positions — forces the model to pick a
              novel token and break the cycle.

        Verified live 2026-05-05:
          - "Stanford Stanford..." (single-token, 100+) — (1) catches
          - "Plan and Plan and ( ( ( (" — (2) catches
          - "hora horora hora era ara" — (1) catches with threshold=2
          - "VC VC fund fund management management ventures ventures Ventures Ventures ((( (((" — (3) catches diversity collapse

        This is sampling-layer band-aid. The REAL root cause is DSV4
        long-context attention drift (compressor/indexer pool
        accumulation). Documented in DSV4_FIX_NUANCES.md.
        """
        if not recent_tokens:
            return logits
        tail = recent_tokens[-window:] if len(recent_tokens) > window else list(recent_tokens)
        from collections import Counter

        # (1) Single-token repeats — threshold=2 (was 3)
        counts = Counter(tail)
        offenders = set(t for t, c in counts.items() if c >= run_threshold)

        # (2) 2-gram extension
        if len(recent_tokens) >= 2:
            prev1 = recent_tokens[-1]
            ngram2_counter: Counter = Counter()
            history = recent_tokens[-window:] if len(recent_tokens) > window else list(recent_tokens)
            for i in range(len(history) - 1):
                ngram2_counter[(history[i], history[i + 1])] += 1
            for (a, b), c in ngram2_counter.items():
                if c >= 2 and a == prev1:
                    offenders.add(b)

        # (3) 3-gram extension
        if len(recent_tokens) >= 3:
            prev1, prev2 = recent_tokens[-1], recent_tokens[-2]
            ngram3_counter: Counter = Counter()
            history = recent_tokens[-window:] if len(recent_tokens) > window else list(recent_tokens)
            for i in range(len(history) - 2):
                ngram3_counter[(history[i], history[i + 1], history[i + 2])] += 1
            for (a, b, x), c in ngram3_counter.items():
                if c >= 2 and a == prev2 and b == prev1:
                    offenders.add(x)

        # (4) Diversity collapse — variant-cycling attractor
        if len(recent_tokens) >= diversity_window:
            div_tail = recent_tokens[-diversity_window:]
            unique_count = len(set(div_tail))
            if unique_count < diversity_min:
                # All recent tokens are recyclable in the attractor.
                # Block them all to force novel sampling.
                offenders.update(set(div_tail))

        if not offenders:
            return logits
        try:
            logits_2d = logits if logits.ndim == 2 else logits[None, :]
            updated = logits_2d
            neg_inf = mx.array(-float("inf"), dtype=logits_2d.dtype)
            offender_arr = mx.array(sorted(offenders), dtype=mx.int32)
            arange = mx.arange(logits_2d.shape[-1])
            mask = mx.zeros((logits_2d.shape[-1],), dtype=mx.bool_)
            for tid in sorted(offenders):
                if 0 <= tid < logits_2d.shape[-1]:
                    mask = mask | (arange == tid)
            updated = mx.where(mask, neg_inf, updated)
            return updated if logits.ndim == 2 else updated[0]
        except Exception:
            return logits

    def _sample(self, logits, sampler, processors, recent_tokens, generated_tokens=None):
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
        # DSV4 hard n-gram dedup — final logit warp BEFORE softmax. ONLY
        # uses generated tokens to avoid prompt-token poisoning of the
        # dedup state. Without this, a prompt that
        # legitimately repeats a word 3+ times would block that word for
        # the entire generation. Disable via env
        # VMLX_DSV4_HARD_REP_BLOCK=0 if it interferes (NOT recommended).
        if os.environ.get("VMLX_DSV4_HARD_REP_BLOCK", "1") not in ("0", "false", "no"):
            dedup_ctx = generated_tokens if generated_tokens is not None else recent_tokens
            x = self._hard_repetition_block(x, dedup_ctx)
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
            if hasattr(mx, "synchronize"):
                self._sync()
            else:
                mx.eval(last_logits)
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

    def _finalizer_budget(self) -> int:
        try:
            return max(1, int(os.environ.get("VMLX_DSV4_FINALIZER_TOKENS", "512")))
        except (TypeError, ValueError):
            return 512

    def _effective_max_tokens(self, req: _Request) -> int:
        if req.finalizer_max_tokens is not None:
            return req.finalizer_max_tokens
        return req.max_tokens

    def _maybe_force_think_close(
        self,
        req: _Request,
        token_id: int,
    ) -> tuple[int, Optional[str]]:
        """Inject </think> before DSV4 stops inside implicit thinking.

        Earlier engine-level finalization re-prefilled ``prompt + reasoning`` to
        continue after a missing close tag. On long prompts that duplicates the
        DSV4 composite prefill and can OOM. This generator-level path keeps the
        live SWA+CSA/HSA cache, emits the structural close token as the next
        output token, then lets the model continue normally for a bounded
        visible-answer budget.
        """
        if req.forced_think_close or not self._prompt_starts_in_think(req):
            return token_id, None
        if THINK_CLOSE_ID in req.out_tokens:
            return token_id, None
        would_stop = token_id == DSV4_EOS_ID or token_id in self.stop_tokens
        would_length = len(req.out_tokens) + 1 >= req.max_tokens
        if not would_stop and not would_length:
            return token_id, None

        req.forced_think_close = True
        req.forced_think_close_at = len(req.out_tokens)
        req.finalizer_max_tokens = len(req.out_tokens) + 1 + self._finalizer_budget()
        logger.info(
            "DSV4Gen: forcing </think> at generated token %d before %s; "
            "continuing from live composite cache for up to %d extra tokens.",
            len(req.out_tokens),
            "stop" if would_stop else "length",
            req.finalizer_max_tokens - (len(req.out_tokens) + 1),
        )
        return THINK_CLOSE_ID, None

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
                    self._sync()
                    logger.info("DSV4Gen: kernel warmup done")
                except Exception as _wexc:
                    logger.warning(f"DSV4Gen: warmup failed (continuing anyway): {_wexc}")
            self._warmed_up = True

        for r in list(self._requests):
            with mx.stream(self._stream):
                if r.cache is None:
                    # Prefill — chunked through one DeepseekV4Cache instance.
                    # This preserves SWA+CSA/HSA state while bounding transient
                    # MLX allocations for long prompts.
                    r.cache = self._make_new_cache()
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
                    if len(r.prompt_tokens) >= 2:
                        head_tokens = r.prompt_tokens[:-1]
                        last_token = r.prompt_tokens[-1:]
                        # Phase 1
                        _ = self._prefill_last_logits(head_tokens, r.cache)
                        try:
                            r.prompt_snapshot = self._snapshot_dsv4_cache(r.cache)
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
                        last_logits = self._prefill_last_logits(last_token, r.cache)
                    else:
                        # Trivial 1-token prompt — no N-1 to snapshot.
                        last_logits = self._prefill_last_logits(r.prompt_tokens, r.cache)
                        r.prompt_snapshot = None

                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                    )
                    self._sync()
                    tok_id = self._sampled_token_id(sampled)
                    tok_id, forced_finish = self._maybe_force_think_close(r, tok_id)
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    if forced_finish is not None:
                        r.finish_reason = forced_finish
                    else:
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
                    last_logits = self._prefill_last_logits(r.prompt_tokens, r.cache)
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                    )
                    self._sync()
                    tok_id = self._sampled_token_id(sampled)
                    tok_id, forced_finish = self._maybe_force_think_close(r, tok_id)
                    r.out_tokens.append(tok_id)
                    r.prompt_processed = True
                    if forced_finish is not None:
                        r.finish_reason = forced_finish
                    else:
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
                    logits = self.model(ids, cache=r.cache)
                    self._sync()
                    last_logits = logits[:, -1, :]
                    sampled, logprobs = self._sample(
                        last_logits, r.sampler, r.logits_processors,
                        self._processor_context(r),
                        generated_tokens=list(r.out_tokens),
                    )
                    self._sync()
                    tok_id = self._sampled_token_id(sampled)
                    tok_id, forced_finish = self._maybe_force_think_close(r, tok_id)
                    r.out_tokens.append(tok_id)
                    if forced_finish is not None:
                        r.finish_reason = forced_finish
                    else:
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

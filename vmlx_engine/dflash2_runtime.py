"""DFlash2 bridge for text-only Qwen VLM generation.

The draft implementation comes from z-lab/dflash. This bridge adapts vMLX's
Qwen VLM language wrapper and hybrid rollback API to its MLX generation loop.

Multiturn prefix reuse: the DFlash2 lane runs through SimpleEngine, which has
no paged prefix-cache stack, so upstream's loop re-prefills the entire
conversation every turn (measured 15.5 s/turn at 7k tokens). This module keeps
an in-process session store of (confirmed tokens, target cache, draft cache)
per finished turn and, when a new prompt extends a stored conversation,
prefills only the delta. The generation loop is adapted from the pinned
``dflash==0.1.0`` runtime (``dflash/model_mlx.py``); its cache invariant is
that the last emitted token is sampled but never forwarded, so a stored cache
always holds exactly ``confirmed[:-1]`` positions.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def _prefix_reuse_enabled() -> bool:
    return os.environ.get("VMLX_DFLASH2_PREFIX_REUSE", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


_ROTATING_CACHE_MATH_PATCHED = False


def _patch_rotating_cache_resume_math() -> None:
    """Make RotatingKVCache tolerate a logical offset past its window.

    The resume path below rebuilds a fresh drafter cache and force-sets
    ``cache.offset`` to the absolute conversation position (rope needs
    absolute positions), which upstream never anticipates: its
    ``_update_in_place`` sizes physical buffer growth from the LOGICAL
    offset —

        new_size = min(self.step, self.max_size - self.offset)

    — so a resume at 14k tokens into a 2047-slot sliding window computes a
    negative size, and the first 1-token update after an accepted==0 verify
    cycle dies with ``[full] Negative dimensions not allowed``. A delta
    >= the window is immune (the concat path fills the buffer to max_size
    and the growth branch never runs again), which is why only short
    follow-up turns on a resumed conversation crashed.

    Three corrections, each a no-op whenever offset == physical fill (every
    upstream flow): growth is sized from physical fill, the write index is
    the physical fill, and the returned view is sliced to the written region
    while the buffer is still below its window.
    """
    global _ROTATING_CACHE_MATH_PATCHED
    if _ROTATING_CACHE_MATH_PATCHED:
        return
    from mlx_lm.models.cache import RotatingKVCache

    import mlx.core as mx

    def _update_in_place(self, keys, values):
        B, n_kv_heads, S, k_head_dim = keys.shape
        filled = 0 if self.keys is None else self.keys.shape[2]
        # Growth keys on the WRITE INDEX reaching the physical end — in every
        # upstream flow offset == _idx while growing, so this is the same
        # branch; with a forced offset it is the only correct signal (offset
        # stays permanently past the window, and re-growing before _idx
        # catches up would leave an unwritten gap mid-buffer).
        if self.keys is None or (self._idx >= filled and filled < self.max_size):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - filled)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = filled

        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = self.keep

        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        if self.offset < self.max_size:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        if self.keys.shape[2] < self.max_size:
            # Force-set offset past the window with the buffer still
            # growing: the valid region is exactly what has been written.
            return (
                self.keys[..., : self._idx, :],
                self.values[..., : self._idx, :],
            )
        return self.keys, self.values

    RotatingKVCache._update_in_place = _update_in_place
    _ROTATING_CACHE_MATH_PATCHED = True
    logger.info("DFlash2: RotatingKVCache resume math patch installed")


class _TargetAdapter:
    def __init__(self, language_model: Any):
        self._target = language_model
        self.model = language_model.model
        self.gdn_states: list[Any] = []
        # Rollback state is needed only for a speculative verification step.
        # Capturing it during chunked prefill retains every chunk's graph.
        self.capture_gdn_states = False

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs, cache=None):
        self.gdn_states.clear()
        hidden = self.model(
            inputs,
            cache=cache,
            gdn_sink=self.gdn_states if self.capture_gdn_states else None,
            return_unnormed=True,
        )
        hidden = self.model.norm(hidden)
        return self._target.lm_head(hidden)

    def release_request_state(self) -> None:
        self.capture_gdn_states = False
        self.gdn_states.clear()
        # Layer hooks keep this list by reference; preserve the list itself.
        hidden_states = self.__dict__.get("_hidden_states")
        if hidden_states is not None:
            hidden_states[:] = [None] * len(hidden_states)

    def __getattr__(self, name: str):
        return getattr(self._target, name)


class _VLMGDNStateCapture:
    def __init__(self, adapter: _TargetAdapter):
        self.adapter = adapter
        self.adapter.gdn_states.clear()
        self.adapter.capture_gdn_states = True

    def clear(self) -> None:
        self.adapter.gdn_states.clear()

    def close(self) -> None:
        self.adapter.capture_gdn_states = False
        self.adapter.gdn_states.clear()

    def rollback(self, cache, accepted, trim) -> None:
        block_size = int(trim) + int(accepted) + 1
        rollback = getattr(self.adapter._target, "rollback_speculative_cache", None)
        if rollback is not None:
            rollback(cache, self.adapter.gdn_states, int(accepted), block_size)
            return

        import mlx.core as mx
        from mlx_lm.models.gated_delta import gated_delta_update

        gdn_index = 0
        for layer_cache in cache:
            if layer_cache.is_trimmable():
                layer_cache.trim(int(trim))
                continue
            (
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                initial_state,
                mask,
                conv_input,
                kernel_size,
            ) = self.adapter.gdn_states[gdn_index]
            count = int(accepted) + 1
            _, state = gated_delta_update(
                q[:, :count],
                k[:, :count],
                v[:, :count],
                a[:, :count],
                b[:, :count],
                a_log,
                dt_bias,
                initial_state,
                None if mask is None else mask[:, :count],
                use_kernel=True,
            )
            layer_cache[1] = state
            layer_cache[0] = mx.contiguous(
                conv_input[:, count : count + int(kernel_size) - 1]
            )
            gdn_index += 1


def _adapter_for(model: Any) -> _TargetAdapter:
    language_model = model.language_model
    adapter = getattr(language_model, "_vmlx_dflash2_adapter", None)
    if adapter is None:
        language_model._vmlx_force_text_rope_1d = True
        adapter = _TargetAdapter(language_model)
        language_model._vmlx_dflash2_adapter = adapter
    return adapter


# ---------------------------------------------------------------------------
# Session store for multiturn prefix reuse
# ---------------------------------------------------------------------------


class _DFlash2SessionStore:
    """LRU store of finished-turn generation state, newest last.

    Each entry: ``tokens`` (the token list this entry represents),
    ``cache_len`` (how many of those tokens the target cache actually holds:
    len(tokens)-1 for end-of-turn entries because the last emitted token is
    sampled but never forwarded, len(tokens) for prompt-boundary snapshots),
    ``target_cache``, ``draft_cache`` (or None when its offset could not be
    aligned), and ``model_key`` guarding cross-model token-id collisions.

    Prompt-boundary snapshots exist because reasoning templates (Qwen3)
    strip the previous turn's <think> block from history, so the end-of-turn
    conversation is never a prefix of the next prompt. The prompt itself,
    up to the latest user message, always is.
    """

    def __init__(self, max_entries: int = 4):
        self.max_entries = int(max_entries)
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def take_matching(self, model_key: Any, prompt_tokens: list[int]) -> Optional[dict]:
        """Pop and return the entry covering the most cached positions whose
        tokens are a prefix of ``prompt_tokens`` and whose cache leaves a
        non-empty delta to prefill. Ownership transfers to the caller."""
        with self._lock:
            best_i = -1
            best_cached = 0
            for i, entry in enumerate(self._entries):
                if entry["model_key"] != model_key:
                    continue
                stored = entry["tokens"]
                cached = int(entry["cache_len"])
                if len(stored) > len(prompt_tokens) or cached >= len(prompt_tokens):
                    continue
                if prompt_tokens[: len(stored)] == stored and cached > best_cached:
                    best_i = i
                    best_cached = cached
            if best_i < 0:
                return None
            return self._entries.pop(best_i)

    def put(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def describe_misses(self, model_key: Any, prompt_tokens: list[int]) -> list[str]:
        """Diagnostics for a failed match: how far each stored entry agrees
        with the prompt before diverging."""
        with self._lock:
            out = []
            for entry in self._entries:
                if entry["model_key"] != model_key:
                    continue
                stored = entry["tokens"]
                common = 0
                for a, b in zip(stored, prompt_tokens):
                    if a != b:
                        break
                    common += 1
                out.append(
                    "%s len=%d cached=%d common=%d"
                    % (
                        entry.get("kind", "turn"),
                        len(stored),
                        int(entry["cache_len"]),
                        common,
                    )
                )
            return out

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def discard_model(self, model_key: Any) -> None:
        """Drop this model's checkpoints after a failed or cancelled request."""
        with self._lock:
            self._entries[:] = [
                entry for entry in self._entries if entry["model_key"] != model_key
            ]


_SESSION_STORE = _DFlash2SessionStore()


def _clone_cache_shells(cache: list, factory) -> Optional[list]:
    """Duplicate a prompt cache as fresh cache objects sharing the same
    (immutable) state arrays via the mlx-lm state/meta_state protocol.

    Used to freeze the target cache at the prompt boundary before decode
    mutates it. Returns None if any layer does not round-trip.
    """
    try:
        fresh = factory()
        for dst, src in zip(fresh, cache):
            dst.state = src.state
            src_meta = getattr(type(src), "meta_state", None)
            if isinstance(src_meta, property) and src_meta.fset is not None:
                dst.meta_state = src.meta_state
        return fresh
    except Exception:
        logger.info("DFlash2 cache clone failed", exc_info=True)
        return None


def _assistant_tag_cut(tokenizer: Any, prompt_list: list[int], cache_len: int):
    """Index of the last assistant generation tag (<|im_start|>) in the
    prompt, or None. Snapshotting the cache BEFORE this position makes the
    boundary entry immune to generation-tag / think-opener divergence
    between this prompt and how the next prompt renders this turn."""
    try:
        tag = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        return None
    if tag is None or int(tag) < 0:
        return None
    tag = int(tag)
    for i in range(len(prompt_list) - 1, -1, -1):
        if prompt_list[i] == tag:
            return i if cache_len < i < len(prompt_list) else None
    return None


def _checkpoint_correction(cycle_committed: int, cycle_kept: int):
    """Rollback args to shrink the final cycle's committed positions down to
    the emitted ones. Returns (accepted, trim) for capture.rollback / the
    incremental trim count for trimmable caches, or None when already exact.

    ``cycle_committed`` counts target-cache positions the final cycle still
    holds (bs when it exited before its trim, accepted+1 after a normal trim);
    ``cycle_kept`` counts tokens actually emitted from that cycle. The cache
    must end at confirmed[:-1], and the cycle's first committed position is
    the previous turn token, so it must keep exactly ``cycle_kept`` positions.
    """
    excess = int(cycle_committed) - int(cycle_kept)
    if excess <= 0:
        return None
    # accepted+1 positions survive a rollback of block_size=cycle_committed.
    return (int(cycle_kept) - 1, excess)


def _stream_generate_resumable(
    model: Any,
    adapter: _TargetAdapter,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    *,
    block_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    prefill_step_size: int = 2048,
) -> Iterator[Any]:
    """The dflash==0.1.0 ``_stream_generate`` loop with session resume.

    Differences from upstream: (1) when the session store holds a conversation
    that prefixes this prompt, only the delta is prefilled into its caches;
    (2) on clean EOS/length exit the caches are rolled back to exactly
    ``confirmed[:-1]`` and stored for the next turn. Everything else is kept
    line-for-line so behaviour off the resume path is identical.
    """
    import mlx.core as mx
    import dflash.model_mlx as runtime

    runtime._patch_model(adapter, draft.config.target_layer_ids)
    sampler = runtime.make_sampler(temperature, top_p, top_k)

    if not isinstance(tokenizer, runtime.TokenizerWrapper):
        tokenizer = runtime.TokenizerWrapper(tokenizer)

    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    prompt_list = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=add_special_tokens)]
    prompt_arr = mx.array(prompt_list)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    tokens: list[int] = []

    model_key = (id(model), id(draft))
    resume = (
        _SESSION_STORE.take_matching(model_key, prompt_list)
        if _prefix_reuse_enabled()
        else None
    )
    if resume is None and _prefix_reuse_enabled():
        misses = _SESSION_STORE.describe_misses(model_key, prompt_list)
        if misses:
            logger.info(
                "DFlash2 prefix reuse miss for %d-token prompt: %s",
                len(prompt_list),
                "; ".join(misses),
            )

    if resume is not None:
        target_cache = resume["target_cache"]
        cache_len = int(resume["cache_len"])
        delta = prompt_arr[cache_len:]
        stored_draft_cache = resume["draft_cache"]
    else:
        target_cache = runtime.make_prompt_cache(adapter)
        cache_len = 0
        delta = prompt_arr
        stored_draft_cache = None

    draft.bind(adapter)
    _target_can_trim = runtime.can_trim_prompt_cache(target_cache)
    _capture = None

    # Snapshot the boundary BEFORE the assistant generation tag: the tag and
    # any think-opener tokens after it are exactly what the next prompt
    # renders differently, so a snapshot taken at the full prompt would
    # diverge in its last few tokens and never match.
    boundary_cut = (
        _assistant_tag_cut(tokenizer, prompt_list, cache_len)
        if _prefix_reuse_enabled()
        else None
    )
    boundary_shells = None

    try:
        tic = time.perf_counter()
        with mx.stream(runtime.generation_stream):
            hidden_limit = (
                draft.config.sliding_window - 1
                if all(t == "sliding_attention" for t in draft.config.layer_types)
                else None
            )
            if boundary_cut is not None:
                _, hidden_a, _ = runtime._prefill_target(
                    adapter,
                    prompt_arr[cache_len:boundary_cut],
                    target_cache,
                    hidden_limit,
                    prefill_step_size,
                )
                boundary_shells = _clone_cache_shells(
                    target_cache, lambda: runtime.make_prompt_cache(adapter)
                )
                logits, hidden_b, _ = runtime._prefill_target(
                    adapter,
                    prompt_arr[boundary_cut:],
                    target_cache,
                    hidden_limit,
                    prefill_step_size,
                )
                hidden = mx.concatenate([hidden_a, hidden_b], axis=1)
                if hidden_limit is not None and hidden.shape[1] > hidden_limit:
                    hidden = hidden[:, -hidden_limit:]
                hidden_offset = int(delta.size) - int(hidden.shape[1])
            else:
                logits, hidden, hidden_offset = runtime._prefill_target(
                    adapter, delta, target_cache, hidden_limit, prefill_step_size
                )
            draft_spliced = False
            if stored_draft_cache is not None and hidden.shape[1] == delta.size:
                gap_arr = resume.get("draft_hidden_gap") if resume else None
                gap_len = int(gap_arr.shape[1]) if gap_arr is not None else 0
                if int(stored_draft_cache[0].offset) + gap_len == cache_len:
                    # Stored draft KV plus the bridged gap ends exactly where
                    # this hidden window starts, so the drafter keeps its
                    # conversation history instead of seeing only the delta.
                    if gap_arr is not None:
                        hidden = mx.concatenate([gap_arr, hidden], axis=1)
                    draft_cache = stored_draft_cache
                    draft_spliced = True
            if not draft_spliced:
                # The forced absolute offset below is the state upstream's
                # RotatingKVCache growth math cannot represent; install the
                # resume-math patch before creating it.
                _patch_rotating_cache_resume_math()
                draft_cache = runtime.make_prompt_cache(draft)
                for cache in draft_cache:
                    cache.offset = cache_len + int(hidden_offset)
        mx.eval(logits, hidden)
        prefill_elapsed = time.perf_counter() - tic
        prompt_tps = prompt_arr.size / max(prefill_elapsed, 1e-9)
        if resume is not None:
            logger.info(
                "DFlash2 prefix reuse: %d of %d prompt tokens from session store "
                "(prefilled %d, draft cache %s, %.2fs)",
                cache_len,
                len(prompt_list),
                int(delta.size),
                "spliced" if draft_cache is stored_draft_cache else "rebuilt",
                prefill_elapsed,
            )

        if boundary_shells is not None:
            # Prompt-boundary snapshot: the conversation up to (not
            # including) the assistant generation tag. The NEXT turn's
            # prompt always contains this exact prefix, even when the
            # template strips this turn's <think> block from history
            # (which makes the end-of-turn entry unmatchable).
            _SESSION_STORE.put(
                {
                    "model_key": model_key,
                    "kind": "boundary",
                    "tokens": prompt_list[:boundary_cut],
                    "cache_len": int(boundary_cut),
                    "target_cache": boundary_shells,
                    "draft_cache": None,
                    "draft_hidden_gap": None,
                }
            )
            logger.info(
                "DFlash2 boundary snapshot stored: %d of %d prompt tokens",
                int(boundary_cut),
                len(prompt_list),
            )
        elif boundary_cut is not None:
            logger.info("DFlash2 boundary snapshot skipped: cache clone failed")

        tic = time.perf_counter()
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n = 1

        # Final-cycle bookkeeping for the exit checkpoint. Before any verify
        # cycle runs, the cache holds exactly the prompt = confirmed[:-1], so
        # committed == kept.
        cycle_committed = 0
        cycle_kept = 0

        def _checkpoint() -> None:
            if not _prefix_reuse_enabled():
                return
            correction = _checkpoint_correction(cycle_committed, cycle_kept)
            if correction is not None:
                accepted_i, trim_i = correction
                if _target_can_trim:
                    runtime._trim_recent_cache(target_cache, trim_i)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted_i, trim_i)
                else:
                    return  # cannot align this cache; skip storing
            confirmed = prompt_list + [int(t) for t in tokens]
            draft_trim = draft_cache[0].offset - (len(confirmed) - 1)
            store_draft: Optional[list] = draft_cache
            if draft_trim > 0:
                runtime._trim_recent_cache(draft_cache, int(draft_trim))
            # The draft cache trails the target by the final cycle's kept
            # tokens (its self-trim runs at cycle start, before emissions).
            # The exit-time hidden always starts at the draft's offset, so
            # the missing positions can be bridged at resume time by
            # prepending this slice to the delta hidden.
            gap_arr = None
            gap_needed = (len(confirmed) - 1) - int(draft_cache[0].offset)
            if gap_needed > 0:
                if hidden is not None and hidden.shape[1] >= gap_needed:
                    gap_arr = hidden[:, :gap_needed, :]
                else:
                    store_draft = None
            elif gap_needed < 0:
                store_draft = None
            _SESSION_STORE.put(
                {
                    "model_key": model_key,
                    "kind": "turn",
                    "tokens": confirmed,
                    "cache_len": len(confirmed) - 1,
                    "target_cache": target_cache,
                    "draft_cache": store_draft,
                    "draft_hidden_gap": gap_arr if store_draft is not None else None,
                }
            )
            logger.info(
                "DFlash2 session stored: %d confirmed tokens (draft cache %s)",
                len(confirmed),
                "dropped"
                if store_draft is None
                else ("kept+gap%d" % gap_needed if gap_arr is not None else "kept"),
            )

        if token in tokenizer.eos_token_ids:
            detokenizer.add_token(token)
            detokenizer.finalize()
            _checkpoint()
            yield runtime._make_response(
                detokenizer.last_segment, [token], None, prompt_arr.size, prompt_tps, n, tic, "stop"
            )
            return

        detokenizer.add_token(token)
        yield runtime._make_response(
            detokenizer.last_segment,
            [token],
            None,
            prompt_arr.size,
            prompt_tps,
            n,
            tic,
        )

        if not _target_can_trim:
            _capture = _VLMGDNStateCapture(adapter)

        while n < max_tokens:
            bs = min(block_size, max_tokens - n + 1)
            if bs <= 1:
                break

            with mx.stream(runtime.generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                if isinstance(draft, runtime.DFlash2DraftModel):
                    draft_tokens, draft_indices, draft_probs = draft.propose(
                        block, hidden, draft_cache, temperature, logits_start=1
                    )
                else:
                    draft_logits = draft(block, hidden, draft_cache, logits_start=1)
                    if temperature > 0:
                        draft_probs = runtime._sampling_probs(
                            draft_logits, temperature, top_p, top_k
                        )
                        draft_tokens = runtime._sample_probs(draft_probs)
                        draft_indices = None
                    else:
                        draft_tokens = mx.argmax(draft_logits, axis=-1)
                if (trim_n := draft_cache[0].offset - (prompt_arr.size + n - 1)) > 0:
                    runtime._trim_recent_cache(draft_cache, trim_n)
            mx.async_eval(draft_tokens)

            if _capture is not None:
                _capture.clear()
            with mx.stream(runtime.generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[tokens[-1]]]), draft_tokens], axis=1
                )
                logits = adapter(verify_input, target_cache)
                hidden = mx.concatenate(adapter._hidden_states, axis=-1)
                if temperature > 0:
                    target_probs = runtime._sampling_probs(
                        logits, temperature, top_p, top_k
                    )
                else:
                    target_tokens = mx.argmax(logits, axis=-1)
            mx.async_eval(target_probs if temperature > 0 else target_tokens, hidden)

            d_list = draft_tokens[0].tolist()
            if temperature > 0:
                accepted, bonus = runtime._rejection_sample(
                    draft_tokens, target_probs, draft_probs, draft_indices
                )
            else:
                t_list = target_tokens[0].tolist()
                accepted = next(
                    (i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                    len(d_list),
                )
                bonus = t_list[accepted]
            new_tokens = d_list[:accepted] + [bonus]
            new_tokens = new_tokens[: max_tokens - n]

            eos_idx = next(
                (i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids),
                None,
            )
            if eos_idx is not None:
                new_tokens = new_tokens[: eos_idx + 1]
                for t in new_tokens:
                    detokenizer.add_token(t)
                detokenizer.finalize()
                tokens.extend(new_tokens)
                n += len(new_tokens)
                # The upstream loop returns before its rollback, leaving all
                # bs verify positions committed.
                cycle_committed = bs
                cycle_kept = len(new_tokens)
                _checkpoint()
                yield runtime._make_response(
                    detokenizer.last_segment,
                    new_tokens,
                    len(new_tokens),
                    prompt_arr.size,
                    prompt_tps,
                    n,
                    tic,
                    "stop",
                )
                return

            for t in new_tokens:
                detokenizer.add_token(t)
            tokens.extend(new_tokens)
            previous_n = n
            n += len(new_tokens)

            if n // 256 > previous_n // 256:
                mx.clear_cache()

            yield runtime._make_response(
                detokenizer.last_segment,
                new_tokens,
                len(new_tokens),
                prompt_arr.size,
                prompt_tps,
                n,
                tic,
            )

            trim = bs - accepted - 1
            if trim > 0:
                if _target_can_trim:
                    runtime._trim_recent_cache(target_cache, trim)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, : accepted + 1, :]
            cycle_committed = accepted + 1
            cycle_kept = len(new_tokens)

        detokenizer.finalize()
        _checkpoint()
        yield runtime._make_response(
            detokenizer.last_segment,
            [],
            None,
            prompt_arr.size,
            prompt_tps,
            n,
            tic,
            "length",
        )
    finally:
        if _capture is not None:
            _capture.close()


def stream_dflash2_generate(
    model: Any,
    tokenizer: Any,
    draft: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Iterator[Any]:
    """Yield upstream DFlash2 chunks using vMLX's hybrid Qwen target."""

    import mlx.core as mx
    import dflash.model_mlx as runtime

    adapter = _adapter_for(model)
    adapter.release_request_state()
    try:
        with runtime.wired_limit(adapter, [runtime.generation_stream]):
            yield from _stream_generate_resumable(
                model,
                adapter,
                draft,
                tokenizer,
                prompt,
                # Qwen3.8's published/runtime-validated DFlash2 lane is block 5.
                # The checkpoint's training maximum is larger, but using it as
                # the serving block makes four-row verification become seq8 and
                # cuts throughput roughly in half on M5 Max.
                block_size=min(5, int(draft.config.block_size)),
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
    except BaseException:
        # Includes GeneratorExit on client disconnect. Partial boundary
        # checkpoints and old turns must not pin memory after an OOM.
        _SESSION_STORE.discard_model((id(model), id(draft)))
        raise
    finally:
        # Covers prefill/setup failures too, before a capture object exists.
        adapter.release_request_state()

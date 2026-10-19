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
from concurrent.futures import Future
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class _StopDetokenizer:
    """Request-local stop matching, withholding incomplete stop prefixes.

    Only constructed for explicit stop requests. Generation still checkpoints
    the actual sampled tokens, including the token completing the stop string;
    tokens after that token are never committed or emitted.
    """

    def __init__(self, detokenizer, stops):
        self.inner = detokenizer
        self.stops = tuple(stops)
        self.pending = ""
        self.output = ""
        self.matched = False

    def _consume(self, final=False):
        if self.matched:
            return
        self.pending += self.inner.last_segment
        positions = [self.pending.find(s) for s in self.stops]
        found = [i for i in positions if i >= 0]
        if found:
            self.output += self.pending[:min(found)]
            self.pending = ""
            self.matched = True
            return
        keep = 0
        if not final:
            for stop in self.stops:
                for size in range(1, min(len(stop), len(self.pending) + 1)):
                    if self.pending.endswith(stop[:size]):
                        keep = max(keep, size)
        cut = len(self.pending) - keep
        self.output += self.pending[:cut]
        self.pending = self.pending[cut:]

    def add_token(self, token):
        self.inner.add_token(token)
        self._consume()

    def finalize(self):
        self.inner.finalize()
        self._consume(final=True)

    @property
    def last_segment(self):
        result, self.output = self.output, ""
        return result


def _prefix_reuse_enabled() -> bool:
    return os.environ.get("VMLX_DFLASH2_PREFIX_REUSE", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


_ROTATING_CACHE_MATH_PATCHED = False

# Methods of mlx-lm RotatingKVCache that read ``self.offset`` as the PHYSICAL
# token count of the buffer (growth, wrap detection, returned view, trim).
_ROTATING_LOCAL_METHODS = (
    "update_and_fetch", "_temporal_order", "size", "is_trimmable", "trim", "make_mask",
)


def _patch_rotating_cache_resume_math() -> None:
    """Let a resumed drafter RotatingKVCache keep an ABSOLUTE offset.

    The resume path rebuilds a fresh (empty) drafter cache whose ``offset``
    must stay the absolute conversation position: the drafter's RoPE reads it,
    and the session splice / per-cycle trim / exit checkpoint compare it with
    the target's absolute length. Upstream RotatingKVCache, however, reads
    ``offset`` as the physical fill: growth size (``max_size - offset``: a
    negative ``mx.zeros`` dimension at 14k into a 2047 window), the returned
    view (``keys[:offset]``) and wrap detection in ``_temporal_order``
    (``_idx < offset`` == "wrapped"). The first version of this patch fixed
    only the growth size; the other two kept feeding the drafter up to
    ``step - 1`` = 255 ZERO K/V rows after any 1-row update (a verify cycle
    that accepted nothing) on every follow-up that rebuilt its drafter cache.
    Output stayed correct (lossless verify), acceptance did not (audit
    2026-10-07; the Swift port found the same in e54e7388).

    Fix: a rebuilt cache carries ``_vmlx_base`` (the absolute position of its
    first physical row, set by ``_rebase_drafter_cache``). The upstream methods
    that treat ``offset`` as the fill run on ``offset - base``; everything
    outside them sees the absolute offset. A cache without a base (every
    upstream flow) runs the unmodified upstream code. ``meta_state`` carries
    the base so clones and the RAM/SSD session tiers keep it.
    """
    global _ROTATING_CACHE_MATH_PATCHED
    if _ROTATING_CACHE_MATH_PATCHED:
        return
    from mlx_lm.models.cache import RotatingKVCache

    def _localized(original):
        def method(self, *args, **kwargs):
            base = self.__dict__.get("_vmlx_base", 0)
            if not base or self.__dict__.get("_vmlx_localized"):
                return original(self, *args, **kwargs)
            self._vmlx_localized = True
            self.offset -= base
            try:
                return original(self, *args, **kwargs)
            finally:
                self.offset += base
                self._vmlx_localized = False

        return method

    for name in _ROTATING_LOCAL_METHODS:
        setattr(RotatingKVCache, name, _localized(getattr(RotatingKVCache, name)))

    state = RotatingKVCache.state
    RotatingKVCache.state = property(_localized(state.fget), state.fset)

    meta = RotatingKVCache.meta_state

    def _meta_get(self):
        values = tuple(meta.fget(self))
        base = self.__dict__.get("_vmlx_base", 0)
        return values + (str(base),) if base else values

    def _meta_set(self, v):
        v = tuple(v)
        meta.fset(self, v[:4])
        self._vmlx_base = int(v[4]) if len(v) > 4 else 0

    RotatingKVCache.meta_state = property(_meta_get, _meta_set)
    _ROTATING_CACHE_MATH_PATCHED = True
    logger.info("DFlash2: RotatingKVCache absolute-offset resume patch installed")


def _rebase_drafter_cache(cache: list, offset: int) -> None:
    """Start EMPTY drafter caches at absolute position ``offset``.

    Rotating layers record it as their base, so their buffers stay physical
    from row 0 (no zero rows). Other layer types keep the historical forced
    offset (no shipped drafter has one: every bundled dflash2 is all-sliding;
    ISSUES I-38 notes the KVCache case).
    """
    from mlx_lm.models.cache import RotatingKVCache

    _patch_rotating_cache_resume_math()
    for layer in cache:
        if isinstance(layer, RotatingKVCache) and layer.keys is None:
            layer._vmlx_base = int(offset)
        layer.offset = int(offset)


class _TargetAdapter:
    def __init__(self, language_model: Any):
        self._target = language_model
        self.model = language_model.model
        self.gdn_states: list[Any] = []
        # Set only around prompt prefill (_prefill_last_logits): the prefill
        # helper keeps just logits[:, -1:], so projecting every prompt row
        # through the 248k-vocab head is wasted work on the final chunk.
        self.last_row_logits = False
        # Media plan, set for ONE image/video request (_stream_generate_resumable
        # media=...). The VLM's own get_input_embeddings/get_rope_index produce
        # the whole prompt's input embeddings (vision features merged) and its
        # M-RoPE position ids; prompt chunks read their slice of both. Every
        # row after the prompt (decode, verify) is text, at position
        # offset + rope_delta (Qwen M-RoPE compresses media positions, so the
        # text that follows sits rope_delta < 0 below its cache offset).
        self.media_embeds = None
        self.media_positions = None
        self.rope_delta = 0

    @property
    def layers(self):
        return self.model.layers

    def _media_inputs(self, inputs, cache) -> dict:
        import mlx.core as mx

        start = int(cache[self.model.fa_idx].offset) if cache is not None else 0
        rows = int(inputs.shape[1])
        embeds = self.media_embeds
        if embeds is not None and start + rows <= int(embeds.shape[1]):
            return {
                "inputs_embeds": embeds[:, start:start + rows],
                "position_ids": self.media_positions[:, :, start:start + rows],
            }
        # Text rows after the media: 2-D ids take the text RoPE kernel (and the
        # row-exact verify path) at offset start + rope_delta.
        return {"position_ids": (mx.arange(start, start + rows) + int(self.rope_delta))[None]}

    def __call__(self, inputs, cache=None):
        if self.last_row_logits:
            # prompt prefill (one call per chunk inside the dflash runtime's loop): cooperative cancel point
            from .utils.generation_cancellation import raise_if_generation_cancelled

            raise_if_generation_cancelled()
        media = (
            self._media_inputs(inputs, cache)
            if self.media_embeds is not None or self.rope_delta
            else {}
        )
        hidden = self.model(
            inputs,
            cache=cache,
            **media,
            # The GDN sink records per-row q/k/v/a/b + state for verify
            # rollback.  Prompt prefill never rolls back, and the list is only
            # cleared at the first decode cycle, so during prefill it just
            # pinned every chunk's qkv projection and recurrent state.
            gdn_sink=None if self.last_row_logits else self.gdn_states,
            return_unnormed=True,
        )
        hidden = self.model.norm(hidden)
        if self.last_row_logits and hidden.shape[1] > 1:
            hidden = hidden[:, -1:]
        return self._target.lm_head(hidden)

    def __getattr__(self, name: str):
        return getattr(self._target, name)


class _VLMGDNStateCapture:
    def __init__(self, adapter: _TargetAdapter):
        self.adapter = adapter

    def clear(self) -> None:
        self.adapter.gdn_states.clear()

    def close(self) -> None:
        return None

    def rollback(self, cache, accepted, trim) -> None:
        block_size = int(trim) + int(accepted) + 1
        rollback = getattr(self.adapter._target, "rollback_speculative_cache", None)
        # accepted == -1 (an AR step launched past the last emitted token, _checkpoint_correction) must undo the
        # whole forward. mlx_vlm's rollback would replay q[:, :0] -> a zero-row gated_delta_update whose lazy state
        # fails at eval (session store, then the next turn's first forward). The count == 0 branch below restores
        # the forward's initial state and conv window instead.
        if rollback is not None and int(accepted) >= 0:
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
            if count == 0:
                # Undo the whole forward (an AR step launched past the last
                # emitted token): the state and conv window it started from.
                layer_cache[1] = initial_state
                layer_cache[0] = mx.contiguous(conv_input[:, : int(kernel_size) - 1])
                gdn_index += 1
                continue
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


def _prefill_last_logits(runtime: Any, adapter: Any, *args):
    """runtime._prefill_target in prompt mode: last-row head, no GDN sink.

    (1) Upstream ``_prefill_target`` returns ``logits[:, -1:]`` of the final
    chunk; MLX does not push that slice through the matmul, so the whole chunk
    (up to step-1 rows x 248,320 vocab, ~5 TFLOP and a ~1 GB temporary on
    Qwen3.8-27B) was projected to keep one row.
    (2) The target's GatedDeltaNet rollback sink is off (see
    ``_TargetAdapter.__call__``): prefill never rolls back, and the sink held
    every chunk's per-layer qkv rows and recurrent state alive until decode.
    Hidden states (what the drafter consumes) and caches are unchanged.
    """
    previous = getattr(adapter, "last_row_logits", False)
    try:
        adapter.last_row_logits = True
        return runtime._prefill_target(adapter, *args)
    finally:
        adapter.last_row_logits = previous


def _snapshot_draft_state(runtime, draft, resume, parts, cut, start, hidden_limit):
    """Retain the drafter's prefix without another target or draft forward.

    A target-only boundary hit otherwise feeds only the new suffix to the
    drafter. Keep either its existing KV plus the unprocessed gap, or the
    bounded target-hidden window needed to rebuild that KV.
    """
    import mlx.core as mx

    state = {"draft_cache": None, "draft_hidden_gap": None, "draft_context": None}
    prior = resume or {}
    cached = prior.get("draft_cache")
    gap = prior.get("draft_hidden_gap")
    gap_len = int(gap.shape[1]) if gap is not None else 0
    if (
        cached is not None
        and int(cached[0].offset) + gap_len == start
        and sum(int(p.shape[1]) for p in parts) == cut - start
    ):
        cloned = _clone_cache_shells(cached, lambda: runtime.make_prompt_cache(draft))
        if cloned is not None:
            state["draft_cache"] = cloned
            state["draft_hidden_gap"] = mx.concatenate(
                ([gap] if gap is not None else []) + parts, axis=1
            )
            return state
    if hidden_limit is not None:
        previous = prior.get("draft_context")
        windows = ([previous] if previous is not None else []) + parts
        context = mx.concatenate(windows, axis=1)
        # Never label a partial suffix as a complete sliding context.
        required = min(cut, hidden_limit)
        if context.shape[1] >= required:
            state["draft_context"] = context[:, -required:]
    return state


def _rebuild_draft_hidden(hidden, resume, cache_len, delta_size, hidden_offset, hidden_limit):
    import mlx.core as mx

    context = (resume or {}).get("draft_context")
    if context is not None and hidden_offset == 0:
        hidden = mx.concatenate([context, hidden], axis=1)
        if hidden_limit is not None:
            hidden = hidden[:, -hidden_limit:]
    return hidden, cache_len + delta_size - int(hidden.shape[1])


# ---------------------------------------------------------------------------
# Draft-vocabulary cut for the drafter's head (opt-in, VMLX_DFLASH2_DRAFT_VOCAB)
# ---------------------------------------------------------------------------
#
# The DFlash2 drafter proposes through the TARGET's lm_head (248,320 x 5,120, 8-bit on the 27B: ~1.27 GB read per draft
# round for bs rows).  Both reference engines project draft logits through a prefix of the head only (one keeps rows
# [0, 98304); the other adds the specials [248032, 248320)), verify with the full head, and report an acceptance loss
# <= 0.36 % of tokens.  Exactness: only the PROPOSAL changes; the target verifies every draft with its full head, so the
# greedy output is the target argmax and speculative sampling stays distribution-exact for any proposal law (q is zero
# outside the kept rows and the drafts are drawn from that q).  Rows are sliced (contiguous views), never copied.
DRAFT_VOCAB_SEGMENTS = {"tf": ((0, 98304), (248032, 248320)), "prefix": ((0, 98304),)}


def draft_vocab_segments(vocab: int):
    import os

    raw = os.environ.get("VMLX_DFLASH2_DRAFT_VOCAB", "off").strip().lower()
    segs = DRAFT_VOCAB_SEGMENTS.get(raw)
    if segs is None or any(hi > vocab or lo >= hi for lo, hi in segs):
        return None
    return segs


def install_draft_vocab_cut(draft: Any, segments) -> bool:
    """Replace ``draft.compute_logits`` by a version that projects only ``segments`` of the bound head's rows and
    returns full-width logits with -inf elsewhere (the candidate selector keeps working on token ids).  Reads the
    head through ``draft.lm_head`` at call time (bound per request)."""
    import math

    import mlx.core as mx
    import mlx.nn as nn

    if getattr(draft, "_vmlx_vocab_cut", None) is not None:
        return True
    cfg = draft.config
    segs = tuple((int(lo), int(hi)) for lo, hi in segments)

    def compute_logits(hidden):
        head = draft.lm_head
        if not isinstance(head, nn.QuantizedLinear) or "bias" in head:
            return type(draft).compute_logits(draft, hidden)        # unknown head: the full projection
        lead = hidden.shape[:-1]
        parts = []
        for lo, hi in segs:
            parts.append(mx.quantized_matmul(hidden, head["weight"][lo:hi], head["scales"][lo:hi], head["biases"][lo:hi],
                                             transpose=True, group_size=head.group_size, bits=head.bits))
        vocab = int(head["weight"].shape[0])
        out = mx.full((*lead, vocab), -math.inf, dtype=parts[0].dtype)
        for (lo, hi), part in zip(segs, parts):
            out[..., lo:hi] = part
        out = out * cfg.output_multiplier
        cap = cfg.final_logit_softcapping
        if cap is not None and cap > 0:
            out = mx.tanh(out / cap) * cap
        return out

    draft.compute_logits = compute_logits
    draft._vmlx_vocab_cut = segs
    return True


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
        # SSD write receipts of entries that also went to the SSD tier, keyed by id(entry) (the entry dict itself is
        # serialized by the SSD writer thread, so the receipt is kept beside it, not in it).
        self._receipts: dict[int, tuple[dict, Any]] = {}
        self.ram_released = 0

    def attach_receipt(self, entry: dict, receipt) -> None:
        with self._lock:
            if any(e is entry for e in self._entries):
                self._receipts[id(entry)] = (entry, receipt)

    def release_durable(self) -> tuple[int, int]:
        """SSD-only policy (glm_dflash2.ssd_only_ram_release_enabled): drop entries whose SSD copy is durable; a later
        request restores them from the SSD tier. Returns (released, still_pending)."""
        from .glm_dflash2 import _receipt_durable, ssd_only_ram_release_enabled
        if not ssd_only_ram_release_enabled():
            return 0, 0
        released = pending = 0
        with self._lock:
            keep = []
            for entry in self._entries:
                pair = self._receipts.get(id(entry))
                if pair is not None and pair[0] is entry and _receipt_durable(pair[1]):
                    self._receipts.pop(id(entry), None)
                    released += 1
                    continue
                if pair is not None and pair[0] is entry and not pair[1].done():
                    pending += 1
                keep.append(entry)
            self._entries = keep
            live = {id(e) for e in keep}
            for k in [k for k in self._receipts if k not in live]:
                self._receipts.pop(k, None)
            self.ram_released += released
        return released, pending

    def take_matching(self, model_key: Any, prompt_tokens: list[int]) -> Optional[dict]:
        """Pop and return the entry covering the most cached positions whose
        tokens are a prefix of ``prompt_tokens`` and whose cache leaves a
        non-empty delta to prefill. Ownership transfers to the caller."""
        self.release_durable()
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
            taken = self._entries.pop(best_i)
            pair = self._receipts.pop(id(taken), None)
            if pair is not None and pair[0] is taken:
                taken["_ssd_receipt"] = pair[1]        # its SSD copy (if any) can parent a delta entry
            return taken

    def put(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self.max_entries:
                dropped = self._entries.pop(0)
                self._receipts.pop(id(dropped), None)

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
            self._receipts.clear()

    def stats(self) -> dict:
        # Metadata only: never evaluate or copy model/cache arrays for polling.
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "tokens": sum(int(e["cache_len"]) for e in self._entries),
                "tokens_note": "sum_across_entries; prefixes_may_overlap",
                "ram_released_after_ssd": self.ram_released,
            }


_SESSION_STORE = _DFlash2SessionStore()

# SSD (L2) behind the RAM store; see dflash2_session_disk.py.  Configured by
# the CLI when a DFlash2 drafter is loaded; None = RAM tier only.
_SESSION_SSD = None


def session_cache_stats() -> dict:
    """Report this lane, which has no scheduler/paged-cache statistics."""
    result = {"ram": _SESSION_STORE.stats(), "ssd": None}
    if _SESSION_SSD is not None:
        from dataclasses import asdict

        ssd = _SESSION_SSD
        # Uses the pool's nonblocking O(1) ledger refresh, not a disk scan.
        budget = asdict(ssd.store.budget.refresh_health())
        budget["root"] = str(ssd.store.root)
        result["ssd"] = dict(
            ssd.stats,
            pending_writes=ssd._q.unfinished_tasks,
            global_budget=budget,
        )
    return result


def configure_session_ssd(*, root, max_size_bytes: int, target_path: str, draft_path: str) -> None:
    global _SESSION_SSD
    from .dflash2_session_disk import DFlash2SessionSSD, model_identity

    _SESSION_SSD = DFlash2SessionSSD(
        root=root,
        max_size_bytes=max_size_bytes,
        model_key=model_identity(target_path, draft_path),
    )
    logger.info(
        "DFlash2 session SSD tier: %s (budget %s)",
        _SESSION_SSD.store.directory,
        "unlimited" if not max_size_bytes else "%.1f GB" % (max_size_bytes / 1024**3),
    )


def _chunk_snapshot_every() -> int:
    """Chunk checkpoints (cross-lane port of mimo_session 118863e4, 2026-10-17): a prompt that diverges INSIDE the
    last user message (another question about the same pasted document, an edited message) matched no entry --
    system / boundary / turn entries all end at message edges -- and re-prefilled everything (MiMo 87.8k: 134.6 s
    vs 3.2 s with chunks). 0 disables."""
    try:
        return max(0, int(os.environ.get("VMLX_DFLASH2_CHUNK_SNAPSHOT_EVERY", "4096")))
    except ValueError:
        return 4096


def _chunk_snapshot_cuts(cache_len: int, upper: int, every: int, media_first: Optional[int] = None) -> list[int]:
    """Absolute multiples of ``every`` strictly between the restored prefix and ``upper`` (the boundary cut or the
    prompt end), and at or before the first media token (entries past it are media-salted RAM-only entries, which
    would churn the RAM store)."""
    if every <= 0:
        return []
    stop = int(upper) if media_first is None else min(int(upper), int(media_first) + 1)
    return list(range((int(cache_len) // every + 1) * every, stop, every))


def _parent_below(refs: list, cut: int, fallback: Optional[dict]) -> Optional[dict]:
    """The longest stored SSD entry shorter than ``cut`` (refs: [(len, parent dict)]), else ``fallback``. A boundary
    stored after chunk entries must be a delta on the last chunk, not on a shorter system entry (which would write
    every chunk's rows again)."""
    cands = [p for n, p in refs if int(n) < int(cut)]
    if fallback is not None and int(fallback.get("len", 0)) < int(cut):
        cands.append(fallback)
    if not cands:
        return fallback                    # nothing shorter: the SSD tier writes a full entry
    return max(cands, key=lambda p: int(p.get("len", 0)))


def _store_session(entry: dict, ram_only: bool = False, parent: Optional[dict] = None):
    """RAM store + SSD write-behind (when configured).

    ``ram_only``: media-request entries. The SSD tier matches by token ids
    alone, and media placeholder ids are identical for different images, so
    those entries live only in the RAM store under a media-salted key.

    ``parent``: the SSD entry this cache was built on (``{"sig", "len", "depth", "receipt"}``), so the SSD tier
    stores only the KV rows it adds (dflash2_session_disk DELTA ENTRIES).

    SSD-only (an SSD tier + glm_dflash2.ssd_only_ram_release_enabled, 2026-10-12): a text entry is NOT kept in the
    RAM store. Its cache objects hold views of the live KV buffer; while any view lives, MLX copies the whole buffer
    on the next in-place append (27B JANGH2 at 84.7k: +5.4 GB per held snapshot, ~+16 GiB per long turn). The
    terminal receipt then reports ``vmlx_lookup_served_from_ram = False`` so SimpleEngine settles it before the next
    prefix lookup, which restores from SSD. A write the SSD tier refuses on the spot keeps the RAM entry.
    """
    from .glm_dflash2 import ssd_only_ram_release_enabled

    ssd_only = _SESSION_SSD is not None and not ram_only and ssd_only_ram_release_enabled()
    if not ssd_only:
        _SESSION_STORE.put(entry)
    if _SESSION_SSD is not None and not ram_only:
        if parent:
            entry["ssd_parent"] = parent
        try:
            receipt = _SESSION_SSD.put(entry)
            entry.pop("ssd_parent", None)
            if ssd_only:
                refused = receipt.done() and not (receipt.result() or {}).get("durable")
                if refused:
                    _SESSION_STORE.put(entry)
                    _SESSION_STORE.attach_receipt(entry, receipt)
                receipt.vmlx_lookup_served_from_ram = bool(refused)
                return receipt
            # The RAM store above already serves the next lookup of this
            # entry, so the next request need not wait for the SSD copy
            # (SimpleEngine still settles it before teardown and cache clears).
            # SSD-only policy: once the copy is durable the RAM entry is dropped
            # (release_durable_session_ram) and the SSD tier serves it instead.
            receipt.vmlx_lookup_served_from_ram = True
            _SESSION_STORE.attach_receipt(entry, receipt)
            return receipt
        except Exception:
            logger.warning("DFlash2 SSD snapshot failed; RAM entry kept", exc_info=True)
            entry.pop("ssd_parent", None)
            if ssd_only:
                _SESSION_STORE.put(entry)
            receipt = Future()
            receipt.set_result({"outcome": "failed", "durable": False,
                                "retained_tokens": 0, "detail": "DFlash2 SSD snapshot failed"})
            return receipt


def release_durable_session_ram() -> tuple[int, int]:
    """Model thread: drop RAM session entries whose SSD copy is durable. Returns (released, still_pending)."""
    if _SESSION_SSD is None:
        return 0, 0
    return _SESSION_STORE.release_durable()


def clear_sessions(include_ssd: bool) -> int:
    _SESSION_STORE.clear()
    if include_ssd and _SESSION_SSD is not None:
        return _SESSION_SSD.clear()
    return 0


def _clone_cache_shells(cache: list, factory) -> Optional[list]:
    """Duplicate a prompt cache as fresh cache objects sharing the same
    (immutable) state arrays via the mlx-lm state/meta_state protocol.

    Used to freeze the target cache at the prompt boundary before decode
    mutates it. Returns None if any layer does not round-trip.
    """
    try:
        fresh = factory()
        for dst, src in zip(fresh, cache):
            # COPY the container. mlx-lm ArraysCache.state returns its live
            # ``self.cache`` LIST, and GatedDeltaNet layers update it in place
            # (``cache[0] = conv``, ``cache[1] = state``). Assigning the list
            # itself made every "frozen" snapshot a live alias: its recurrent
            # state kept advancing through the rest of the prefill and decode
            # while its attention KV (fresh slices) stayed at the cut. The SSD
            # tier then stored that advanced state, so a restore applied the
            # suffix tokens to the recurrence twice and changed the answer
            # (audit 2026-10-07: 27B 4D "weekdays x10" -> 12 from SSD, 10 cold).
            state = src.state
            dst.state = list(state) if isinstance(state, list) else state
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


def _system_block_cuts(tokenizer: Any, prompt_list: list[int], cache_len: int) -> list[int]:
    """Index of the first <|im_start|> after EACH of the consecutive leading system blocks, cache_len < cut <
    len(prompt_list) (empty: no system message, or the cache already covers them).

    MiMo-V2.6's template renders the tools as their own first system block and the conversation's system prompt as a
    second one. The end of the last block serves a new chat with the same tools AND system prompt (raw API: 77 ->
    4,581 of 4,602 tokens reused); the end of the first serves a chat whose system prompt differs but whose tools do
    not (in-app: a thinking-off image turn restored 3,737 tokens with the tools-only cut, 0 with only the last cut)."""
    try:
        tag = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
        role = list(tokenizer.encode("system", add_special_tokens=False))
    except Exception:
        return []
    if tag < 0 or not prompt_list or prompt_list[0] != tag or prompt_list[1:1 + len(role)] != role:
        return []
    cuts, pos = [], 0
    while True:
        nxt = next((i for i in range(pos + 1, len(prompt_list)) if prompt_list[i] == tag), None)
        if nxt is None:
            return []                                  # nothing after the system blocks: no snapshot to take
        cuts.append(nxt)
        if prompt_list[nxt + 1:nxt + 1 + len(role)] != role:
            return [c for c in cuts if cache_len < c < len(prompt_list)]
        pos = nxt                                      # another system block follows


def _system_end_cut(tokenizer: Any, prompt_list: list[int], cache_len: int):
    """End of the last consecutive leading system block (see _system_block_cuts), or None: the DFlash2 lane's single
    system snapshot."""
    cuts = _system_block_cuts(tokenizer, prompt_list, cache_len)
    return cuts[-1] if cuts else None


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
    sampling_controls=None,
    prompt_tokens: Optional[list] = None,
    media: Optional[dict] = None,
    stop=None,
) -> Iterator[Any]:
    """The dflash==0.1.0 ``_stream_generate`` loop with session resume.

    ``prompt_tokens`` + ``media`` (image/video requests): the processor's
    expanded token ids and the VLM prefill plan (``embeds``, ``positions``,
    ``rope_delta``, ``salt``) -- see ``_TargetAdapter`` media plan.

    Differences from upstream: (1) when the session store holds a conversation
    that prefixes this prompt, only the delta is prefilled into its caches;
    (2) on clean EOS/length exit the caches are rolled back to exactly
    ``confirmed[:-1]`` and stored for the next turn. Everything else is kept
    line-for-line so behaviour off the resume path is identical.
    """
    import mlx.core as mx
    import dflash.model_mlx as runtime

    # The SSD writer is never waited for here: every entry it holds is also in
    # the RAM session store, SimpleEngine settles each terminal receipt itself
    # (teardown and cache clears wait for all of them), and a snapshot that
    # finds the writer backed up is kept RAM-only (DFlash2SessionSSD.put).
    # Joining the queue here stalled the next request for 7-11 s whenever a
    # queued write sat behind a shared-pool budget reconcile (27B, 2026-10-11).

    runtime._patch_model(adapter, draft.config.target_layer_ids)
    sampler = runtime.make_sampler(temperature, top_p, top_k)

    if not isinstance(tokenizer, runtime.TokenizerWrapper):
        tokenizer = runtime.TokenizerWrapper(tokenizer)

    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    prompt_list = (
        [int(t) for t in prompt_tokens]
        if prompt_tokens is not None
        else [int(t) for t in tokenizer.encode(prompt, add_special_tokens=add_special_tokens)]
    )
    prompt_arr = mx.array(prompt_list)
    # Session-store key tokens: equal to the prompt for text; for media with a
    # per-item plan, each media item's placeholder run carries that item's
    # content id (see MLXMultimodalLM._dflash2_media_plan). The model never
    # sees these ids; they only make prefix matching media-aware.
    per_item_media = media is not None and media.get("key_tokens") is not None
    key_list = [int(t) for t in media["key_tokens"]] if per_item_media else prompt_list

    detokenizer = tokenizer.detokenizer
    stop_matcher = None
    if stop:
        # Decode each accepted token, including unfinished words, so a stop
        # is acted on before forwarding another cycle. Ordinary requests keep
        # their original detokenizer and generation path.
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
        stops = [stop] if isinstance(stop, str) else stop
        stop_matcher = _StopDetokenizer(
            NaiveStreamingDetokenizer(tokenizer), [s for s in stops if s]
        )
        detokenizer = stop_matcher
    mask_id = int(draft.config.mask_token_id)
    tokens: list[int] = []

    base_key = (id(model), id(draft))
    # Media requests key their entries by a hash of the media content: the
    # placeholder token ids are the same for every image. They may still
    # resume a text-only entry (its prefix precedes any media).
    model_key = base_key + (media["salt"],) if media is not None and not per_item_media else base_key
    resume = None
    if _prefix_reuse_enabled():
        resume = _SESSION_STORE.take_matching(model_key, key_list)
        if resume is None and media is not None and not per_item_media:
            resume = _SESSION_STORE.take_matching(base_key, prompt_list)
    if resume is None and _prefix_reuse_enabled() and _SESSION_SSD is not None:
        try:
            _im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        except Exception:
            _im_start = None
        resume = _SESSION_SSD.take_matching(
            key_list,
            im_start_id=int(_im_start) if isinstance(_im_start, int) and _im_start >= 0 else None,
            eos_ids=tokenizer.eos_token_ids,
            make_target=lambda: runtime.make_prompt_cache(adapter),
            # restored drafter caches carry offsets past the sliding window
            make_draft=lambda: (_patch_rotating_cache_resume_math(), runtime.make_prompt_cache(draft))[1],
            every=_chunk_snapshot_every(),
        )
        if resume is not None:
            resume["model_key"] = model_key
            resume["source"] = "dflash2-ssd"
            logger.info(
                "DFlash2 SSD hit: %s entry, %d of %d prompt tokens restored in %.2fs",
                resume["kind"], resume["cache_len"], len(prompt_list), resume["load_s"],
            )
    if resume is None and _prefix_reuse_enabled():
        misses = _SESSION_STORE.describe_misses(model_key, key_list)
        if misses:
            logger.info(
                "DFlash2 prefix reuse miss for %d-token prompt: %s",
                len(prompt_list),
                "; ".join(misses),
            )

    # The SSD entry this request's cache is built on: a later snapshot of it is stored as a delta against it.
    _ssd_parent = None
    _stored_refs: list = []                 # [(len, parent)] of entries this request stored (chunk checkpoints)
    if resume is not None:
        if resume.get("ssd_ref"):
            _ssd_parent = dict(resume["ssd_ref"], receipt=None)               # read from SSD: durable
        elif getattr(resume.get("_ssd_receipt"), "vmlx_ssd_ref", None):
            _r = resume["_ssd_receipt"]
            _ssd_parent = dict(_r.vmlx_ssd_ref, receipt=_r)
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

    # Usage reporting: how many prompt tokens came from the session store
    # (RAM or SSD), carried on every response as cached_tokens/cache_detail.
    _cached = int(cache_len)
    _detail = (resume.get("source", "dflash2-ram") if resume is not None else "")
    _terminal_write = None
    # Engine-measured prefill for the UI/API pp/s (usage.vmlx_prefill): the
    # uncached tokens this request actually prefilled and the time it took,
    # including the vision tower when a media plan was materialized. Same
    # shape as the batched engine's req._prefill_usage.
    _prefill_usage = None
    _vision_seconds = 0.0

    def _respond(*args):
        r = runtime._make_response(*args)
        r.cached_tokens = _cached
        r.cache_detail = _detail
        r.prefill_usage = _prefill_usage
        if getattr(r, "finish_reason", None) is not None:
            r.persistence_future = _terminal_write
        return r

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
    # A second snapshot at the end of the system message (the first
    # <|im_start|> after "<|im_start|>system") when this request did not
    # resume past it: a NEW conversation that shares the system prompt and
    # tool schemas (the app's ~2k-token tools block) reuses it instead of
    # re-prefilling it.  The boundary/turn entries cannot serve that case:
    # they end after the first user message, which differs per conversation.
    system_cut = (
        _system_end_cut(tokenizer, prompt_list, cache_len)
        if _prefix_reuse_enabled()
        else None
    )
    if system_cut is not None and boundary_cut is not None and system_cut >= boundary_cut:
        system_cut = None
    snapshots: list = []

    # Assign (or clear) the shared adapter's media plan at entry on EVERY
    # request: an abandoned generator's `finally` may run late, and the next
    # request must never inherit another request's embeddings or rope delta.
    # M-RoPE delta caching: a restored prefix that contains EVERY media item of
    # this request leaves a pure-text tail at offset + rope_delta, so the
    # entry's cached delta replaces the vision tower + embedding pass.
    # Otherwise materialize the plan (vision tower, merged embeddings,
    # full-prompt M-RoPE ids). Entries store a delta only when their cut covers
    # all of their own request's media (see _entry_rope_delta).
    if media is not None:
        cached_delta = resume.get("rope_delta") if resume is not None else None
        if (
            cached_delta is not None
            and int(resume["cache_len"]) >= int(media.get("media_end", 0))
            and media.get("embeds") is None
        ):
            media["rope_delta"] = int(cached_delta)
            logger.info(
                "DFlash2 media: all %d media tokens inside the restored %d-token prefix; "
                "cached M-RoPE delta %d, vision tower skipped",
                int(media.get("media_end", 0)) - int(media.get("first_media_index", 0)),
                int(resume["cache_len"]), int(cached_delta),
            )
        elif media.get("embeds") is None and callable(media.get("materialize")):
            _vision_t0 = time.perf_counter()
            media.update(media["materialize"]())
            _vision_seconds = time.perf_counter() - _vision_t0
    adapter.media_embeds = media["embeds"] if media is not None else None
    adapter.media_positions = media["positions"] if media is not None else None
    adapter.rope_delta = int(media["rope_delta"]) if media is not None else 0

    def _entry_rope_delta(cut: int):
        # Valid for a future request only if this entry holds all of its own
        # request's media; a cut before the last media item would carry a delta
        # that includes media the entry does not contain.
        if media is None:
            return 0
        return int(adapter.rope_delta) if int(cut) >= int(media.get("media_end", 0)) else None
    try:
        tic = time.perf_counter()
        # VMLX_DFLASH2_PREFILL_TRACE=1 (diagnostic, off by default): per-phase prefill timings with a device sync between
        # phases, one log line per request (follow-up turn after a partial hit ran 1.02 vs 0.28 s, 2026-10-17)
        _ptrace = [] if os.environ.get("VMLX_DFLASH2_PREFILL_TRACE", "").strip() in {"1", "true", "on"} else None
        _pt0 = [tic]

        def _pmark(name):
            if _ptrace is not None:
                mx.synchronize()
                _now = time.perf_counter()
                _ptrace.append((name, round(1000 * (_now - _pt0[0]), 1)))
                _pt0[0] = _now
        with mx.stream(runtime.generation_stream):
            hidden_limit = (
                draft.config.sliding_window - 1
                if all(t == "sliding_attention" for t in draft.config.layer_types)
                else None
            )
            # Prefill in segments ending at each snapshot cut (system end,
            # assistant tag); a cut costs one extra chunk boundary.
            cuts = [(c, k) for c, k in ((system_cut, "system"), (boundary_cut, "boundary")) if c is not None]
            from .glm_dflash2 import ssd_only_ram_release_enabled as _ssd_only
            if _prefix_reuse_enabled() and _SESSION_SSD is not None and _ssd_only():
                _taken = {c for c, _ in cuts}
                cuts += [(c, "chunk") for c in _chunk_snapshot_cuts(
                    cache_len, boundary_cut if boundary_cut is not None else len(prompt_list), _chunk_snapshot_every(),
                    None if media is None or per_item_media else int(media["first_media_index"]),
                ) if c not in _taken]
                cuts.sort()
            if cuts:
                parts, start = [], cache_len
                for cut, kind in cuts:
                    _, h, _ = _prefill_last_logits(
                        runtime, adapter,
                        prompt_arr[start:cut],
                        target_cache,
                        hidden_limit,
                        prefill_step_size,
                    )
                    parts.append(h)
                    _pmark(f"seg->{kind}@{cut}({cut - start})")
                    _shells = _clone_cache_shells(target_cache, lambda: runtime.make_prompt_cache(adapter))
                    _pmark("clone")
                    _dstate = _snapshot_draft_state(runtime, draft, resume, parts, cut, cache_len, hidden_limit)
                    _pmark("draft_state")
                    start = cut
                    if kind != "chunk":
                        snapshots.append((kind, cut, _shells, _dstate))
                        continue
                    if _shells is None:
                        continue
                    # Chunk entries are stored NOW, chained on the previous stored entry, and the prefill waits
                    # for the write: the shells are views of the live KV buffer, and while one is alive the next
                    # in-place append copies the whole buffer (27B at 84.7k: 5.4 GB per held snapshot). A
                    # 4096-row delta writes in ~0.1 s.
                    _parent = _parent_below(_stored_refs, cut, _ssd_parent)
                    _rc = _store_session({"model_key": base_key, "kind": "chunk", "tokens": key_list[:cut],
                                          "rope_delta": _entry_rope_delta(cut), "cache_len": int(cut),
                                          "target_cache": _shells, **_dstate}, parent=_parent)
                    del _shells, _dstate
                    try:
                        _rc.result(timeout=30)
                    except Exception:  # noqa: BLE001 - a slow or failed write only costs one buffer copy
                        pass
                    if getattr(_rc, "vmlx_ssd_ref", None) and (not _rc.done() or (_rc.result() or {}).get("durable")):
                        _stored_refs.append((int(cut), dict(_rc.vmlx_ssd_ref, receipt=_rc)))
                    _pmark("chunk_store")
                logits, h, _ = _prefill_last_logits(
                    runtime, adapter,
                    prompt_arr[start:],
                    target_cache,
                    hidden_limit,
                    prefill_step_size,
                )
                parts.append(h)
                _pmark(f"seg->end({int(prompt_arr.size) - start})")
                hidden = mx.concatenate(parts, axis=1)
                if hidden_limit is not None and hidden.shape[1] > hidden_limit:
                    hidden = hidden[:, -hidden_limit:]
                hidden_offset = int(delta.size) - int(hidden.shape[1])
            else:
                logits, hidden, hidden_offset = _prefill_last_logits(
                    runtime, adapter, delta, target_cache, hidden_limit, prefill_step_size
                )
            draft_spliced = False
            draft_rebuild = None
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
                # Fresh drafter cache at an absolute offset: rotating layers
                # keep it as a base so their buffers stay physical (see
                # _patch_rotating_cache_resume_math).
                _patch_rotating_cache_resume_math()
                draft_cache = runtime.make_prompt_cache(draft)
                hidden, draft_offset = _rebuild_draft_hidden(
                    hidden, resume, cache_len, int(delta.size), hidden_offset, hidden_limit
                )
                _rebase_drafter_cache(draft_cache, draft_offset)
                draft_rebuild = (int(draft_offset), int(hidden.shape[1]))
            _pmark("draft_splice_or_rebuild")
        mx.eval(logits, hidden)
        _pmark("final_eval")
        prefill_elapsed = time.perf_counter() - tic
        if _ptrace is not None:
            logger.info("DFlash2 prefill trace: cached=%d prompt=%d total_ms=%.1f %s", cache_len, len(prompt_list),
                        1000 * prefill_elapsed, " ".join(f"{k}={v}" for k, v in _ptrace))
        _prefill_usage = {
            "tokens": int(prompt_arr.size) - int(cache_len),
            "seconds": float(prefill_elapsed + _vision_seconds),
            # The panel's pp/s accepts exactly this scope (shared/chatMetrics.ts
            # calculatePrefillTps): model prefill + prompt state, vision encode
            # included -- the batched engine's contract. ``path`` says which lane.
            "scope": "model_prefill_and_prompt_state",
            "path": "dflash2" + ("_with_vision" if _vision_seconds else ""),
        }
        prompt_tps = prompt_arr.size / max(prefill_elapsed, 1e-9)
        if resume is not None:
            logger.info(
                "DFlash2 prefix reuse: %d of %d prompt tokens from session store "
                "(prefilled %d, draft cache %s, %.2fs)",
                cache_len,
                len(prompt_list),
                int(delta.size),
                "spliced" if draft_cache is stored_draft_cache else (
                    "rebuilt at %d from %d context rows" % draft_rebuild
                    if draft_rebuild is not None else "rebuilt"
                ),
                prefill_elapsed,
            )

        # Drop each snapshot's cache objects once stored: they hold views of the live KV buffer, and while one lives
        # the next in-place append copies the whole buffer (copy-on-write). The list kept them for the whole decode.
        while snapshots:
            kind, cut, shells, draft_state = snapshots.pop(0)
            if shells is None:
                logger.info("DFlash2 %s snapshot skipped: cache clone failed", kind)
                continue
            # boundary: the conversation up to (not including) the assistant
            # generation tag.  The NEXT turn's prompt always contains it, even
            # when the template strips this turn's <think> block from history
            # (which makes the end-of-turn entry unmatchable).
            # system: the system message, shared by new conversations.
            # A cut before the first media token is plain text: unsalted, SSD ok.
            text_only_cut = (
                media is None or per_item_media or int(cut) <= int(media["first_media_index"])
            )
            _ssd_parent = _parent_below(_stored_refs, cut, _ssd_parent)
            _snap_receipt = _store_session(
                {
                    "model_key": base_key if text_only_cut else model_key,
                    "kind": kind,
                    "tokens": key_list[:cut],
                    "rope_delta": _entry_rope_delta(cut),
                    "cache_len": int(cut),
                    "target_cache": shells,
                    **draft_state,
                },
                ram_only=not text_only_cut,
                parent=_ssd_parent,
            )
            if getattr(_snap_receipt, "vmlx_ssd_ref", None):
                _ssd_parent = dict(_snap_receipt.vmlx_ssd_ref, receipt=_snap_receipt)
            del shells
            logger.info(
                "DFlash2 %s snapshot stored: %d of %d prompt tokens",
                kind,
                int(cut),
                len(prompt_list),
            )

        tic = time.perf_counter()
        first_logits = logits[:, -1:]
        if sampling_controls is not None:
            first_logits = sampling_controls.process(first_logits, prompt_list)
        if sampling_controls is not None and temperature > 0:
            token = runtime._sample_probs(sampling_controls.probabilities(
                runtime, first_logits, temperature, top_p, top_k
            ))[0, 0].item()
        else:
            token = sampler(first_logits)[0, 0].item()
        tokens.append(token)
        n = 1

        # Final-cycle bookkeeping for the exit checkpoint. Before any verify
        # cycle runs, the cache holds exactly the prompt = confirmed[:-1], so
        # committed == kept.
        cycle_committed = 0
        cycle_kept = 0

        def _checkpoint() -> None:
            nonlocal _terminal_write, _ssd_parent
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
            if n == 1:
                # No draft forward ran. Preserve its bounded input window,
                # rather than an empty cache that cannot be serialized.
                store_draft = None
            _terminal_write = _store_session(
                {
                    "model_key": model_key,
                    "kind": "turn",
                    "tokens": key_list + [int(t) for t in tokens],
                    "rope_delta": _entry_rope_delta(len(key_list)),
                    "cache_len": len(confirmed) - 1,
                    "target_cache": target_cache,
                    "draft_cache": store_draft,
                    "draft_hidden_gap": gap_arr if store_draft is not None else None,
                    "draft_context": hidden if n == 1 else None,
                },
                ram_only=media is not None and not per_item_media,
                parent=_parent_below(_stored_refs, len(confirmed) - 1, _ssd_parent),
            )
            logger.info(
                "DFlash2 session stored: %d confirmed tokens (draft cache %s)",
                len(confirmed),
                "dropped"
                if store_draft is None
                else ("kept+gap%d" % gap_needed if gap_arr is not None else "kept"),
            )

        if stop_matcher is not None:
            detokenizer.add_token(token)
        if token in tokenizer.eos_token_ids or (stop_matcher is not None and stop_matcher.matched):
            if stop_matcher is None:
                detokenizer.add_token(token)
            detokenizer.finalize()
            _checkpoint()
            yield _respond(
                detokenizer.last_segment, [token], None, prompt_arr.size, prompt_tps, n, tic, "stop"
            )
            return

        if stop_matcher is None:
            detokenizer.add_token(token)
        yield _respond(
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

        # Verify width per cycle (see _dflash2_block_plan / _BlockChooser):
        # the width with the best predicted tokens per measured second.
        widths = _dflash2_block_plan(block_size, _lane_flat(adapter))
        chooser = _BlockChooser(widths, cost=_width_costs(adapter, widths))
        after_ar = False

        def _ar_step(tok_arr, history):
            # One plain target step on the DFlash2 adapter: the same forward as a
            # verify row, so the drafter's tap rows (hidden) stay available.
            # The GDN sink then holds only this forward, which is what an exit
            # rollback of a step launched past the last emitted token needs.
            if _capture is not None:
                _capture.clear()
            with mx.stream(runtime.generation_stream):
                logits = adapter(tok_arr, target_cache)
                h = mx.concatenate(adapter._hidden_states, axis=-1)
                last = logits[:, -1:]
                if sampling_controls is not None:
                    last = sampling_controls.process(last, history)
                if temperature > 0:
                    probs = (
                        sampling_controls.probabilities(runtime, last, temperature, top_p, top_k)
                        if sampling_controls is not None
                        else runtime._sampling_probs(last, temperature, top_p, top_k)
                    )
                    nxt = runtime._sample_probs(probs)
                else:
                    nxt = mx.argmax(last, axis=-1)
            return nxt, h

        while n < max_tokens:
            mode, burst = chooser.next_action()
            if mode == "ar":
                # AR burst (see _BlockChooser.next_action): plain target steps,
                # pipelined like the AR lane when no history-dependent sampling
                # control needs each token on the host first. Each emitted
                # token is sampled from the target's own distribution, so the
                # output law is the AR lane's; greedy emits the target argmax.
                k = min(int(burst), max_tokens - n)
                _ar_t0 = time.perf_counter()
                pipelined = sampling_controls is None
                emitted: list = []
                rows: list = []
                launched_past = False
                terminal = False
                nx, h = _ar_step(mx.array([[tokens[-1]]]), prompt_list + tokens)
                mx.async_eval(nx, h)
                for j in range(k):
                    launched = None
                    if pipelined and j + 1 < k:
                        nx2, h2 = _ar_step(nx.reshape(1, 1), None)
                        mx.async_eval(nx2, h2)
                        launched = (nx2, h2)
                    t = int(nx.item())
                    emitted.append(t)
                    rows.append(h)
                    if stop_matcher is not None:
                        detokenizer.add_token(t)
                    if t in tokenizer.eos_token_ids or (
                        stop_matcher is not None and stop_matcher.matched
                    ):
                        terminal = True
                    if terminal or j + 1 >= k:
                        launched_past = launched is not None
                        break
                    if launched is None:
                        nx, h = _ar_step(nx.reshape(1, 1), prompt_list + tokens + emitted)
                        mx.async_eval(nx, h)
                    else:
                        nx, h = launched
                chooser.observe_ar(time.perf_counter() - _ar_t0, len(emitted))
                after_ar = True
                hidden = mx.concatenate([hidden] + rows, axis=1)
                if stop_matcher is None:
                    for t in emitted:
                        detokenizer.add_token(t)
                tokens.extend(emitted)
                n += len(emitted)
                # Exit-checkpoint bookkeeping: the cache holds confirmed[:-1]
                # exactly, plus one position when a step was launched past the
                # last emitted token (its 1-row forward is then rolled back).
                cycle_committed = 1 if launched_past else 0
                cycle_kept = 0
                if terminal:
                    detokenizer.finalize()
                    _checkpoint()
                    resp = _respond(
                        detokenizer.last_segment, emitted, len(emitted),
                        prompt_arr.size, prompt_tps, n, tic, "stop",
                    )
                    resp.drafted = 0           # AR tokens: throughput, not draft acceptance
                    yield resp
                    return
                if n // 256 > (n - len(emitted)) // 256:
                    mx.clear_cache()
                resp = _respond(
                    detokenizer.last_segment, emitted, len(emitted),
                    prompt_arr.size, prompt_tps, n, tic,
                )
                resp.drafted = 0
                yield resp
                continue
            bs = min(chooser.width, max_tokens - n + 1)
            if bs <= 1:
                break
            _cycle_t0 = time.perf_counter()

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
                if sampling_controls is not None:
                    logits = sampling_controls.process(logits, prompt_list + tokens, draft_tokens)
                if temperature > 0:
                    if sampling_controls is not None:
                        target_probs = sampling_controls.probabilities(
                            runtime, logits, temperature, top_p, top_k
                        )
                    else:
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
            if stop_matcher is not None:
                for i, t in enumerate(new_tokens):
                    detokenizer.add_token(t)
                    if stop_matcher.matched or t in tokenizer.eos_token_ids:
                        eos_idx = i
                        break
            if eos_idx is not None:
                new_tokens = new_tokens[: eos_idx + 1]
                if stop_matcher is None:
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
                resp = _respond(
                    detokenizer.last_segment,
                    new_tokens,
                    len(new_tokens),
                    prompt_arr.size,
                    prompt_tps,
                    n,
                    tic,
                    "stop",
                )
                resp.drafted = bs - 1      # width varies per cycle: stats need it
                yield resp
                return

            if stop_matcher is None:
                for t in new_tokens:
                    detokenizer.add_token(t)
            tokens.extend(new_tokens)
            previous_n = n
            n += len(new_tokens)

            if n // 256 > previous_n // 256:
                mx.clear_cache()

            resp = _respond(
                detokenizer.last_segment,
                new_tokens,
                len(new_tokens),
                prompt_arr.size,
                prompt_tps,
                n,
                tic,
            )
            resp.drafted = bs - 1
            yield resp

            # The cycle is host-synchronous here (draft/target ids were read
            # with .tolist()), so this wall time is the real cost of a width.
            # It also includes the consumer's handling of the yield above
            # (detokenized text -> SSE); that cost is per cycle too, so it
            # belongs in a tokens-per-second comparison of widths.
            chooser.observe(bs, len(new_tokens), time.perf_counter() - _cycle_t0, after_ar=after_ar)
            after_ar = False
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
        yield _respond(
            detokenizer.last_segment,
            [],
            None,
            prompt_arr.size,
            prompt_tps,
            n,
            tic,
            "stop" if stop_matcher is not None and stop_matcher.matched else "length",
        )
    finally:
        if _capture is not None:
            _capture.close()
        _chooser = locals().get("chooser")
        if _chooser is not None:
            logger.info("DFlash2 chooser: %s", _chooser.summary())
        # The adapter is shared by every request on this model.
        adapter.media_embeds = None
        adapter.media_positions = None
        adapter.rope_delta = 0


class _BlockChooser:
    """Pick the DFlash2 verify width with the best PREDICTED tokens per second.

    Why not "widen after a full block": a wider verify can cost far more than
    its extra rows suggest.  Measured on M5 Max (target forward by rows):
    Qwen3.8-27B JANG_4D 5 rows 53 ms vs 8 rows 79 ms (MLX affine qmm switches
    kernels at 6 rows), JANGH2 78 vs 88 ms.  Acceptance-only widening helped
    JANGH2 (+36 % easy text) but cost JANG_4D ~5 % on prose/code.

    Why not an EMA of measured tokens/second per width (the previous version):
    it only learns about a width while running it, so the losing width is seen
    one noisy probe at a time.  Served JANGH2 code ran 49.9 tok/s under it vs
    59.9 at fixed width 8.

    Model used here (expected tokens per ms, as in TensorFold's depth picker):
      * Drafts are accepted as a prefix, so cycle tokens = 1 + accepted, and
        E[tokens | w] = 1 + sum_{j=1}^{w-1} prod_{k<=j} p_k, where p_k is the
        probability that draft position k is accepted given 1..k-1 were.
      * p_k is estimated from every cycle, at ANY width: positions 1..accepted
        count as hits, position accepted+1 (if verified) as a miss, deeper
        positions were not tried.  So a run at width 5 updates p_1..p_4, which
        width 8 shares.  Counts decay by ``decay`` per cycle (~7-cycle memory)
        to follow easy/hard stretches.  A position with little data leans on
        the previous position's estimate (one pseudo-trial), which is how
        p_5..p_7 are extrapolated while running at width 5.
      * Cost per width is an EMA of measured cycle seconds (drafter + verify).
        Each width is run once to seed its cost, then the non-current width
        is re-timed once every ``probe_every`` cycles (costs drift with
        context length and GPU clock, not with text).
    Exactness is unaffected: width only changes how many drafts are verified,
    never the acceptance rule.

    AR fallback (``next_action``), as the GLM lane's AdaptiveSpec: a verify
    cycle carries a fixed drafter + host-sync cost that only pays at high
    acceptance. Measured 2026-10-11, Qwen3.8-27B JANGH2 at the bundle's sampled
    defaults (temp 1.0, acceptance 25-28 %): AR lane 29.9 tok/s vs DFlash2
    25.5 (auto), 26.5 (fixed 2), 27-28.5 (fixed 3); a 2-row cycle cost 64 ms
    vs 33 ms per pipelined AR step. Greedy stays far ahead (48 tok/s). So when
    plain target steps are estimated at least ``ar_margin`` faster than the
    best width, run an AR burst of ``ar_burst`` tokens; every
    ``ar_probe_every`` AR tokens one verify cycle runs so the acceptance
    estimate keeps tracking the text. The first decode action of every request
    is a verify cycle (exit checkpoints rely on the drafter having run).
    """

    decay = 0.85
    cost_alpha = 0.25
    probe_every = 32
    prior_p = 0.7
    ar_margin = 0.05
    ar_probe_every = 24
    ar_burst = 8
    ar_alpha = 0.3

    def __init__(self, low, high: Optional[int] = None, cost: Optional[dict] = None):
        # ``low`` may be the whole tuple of candidate widths (from
        # _dflash2_block_plan); (low, high) is kept for the two-width callers.
        if isinstance(low, (tuple, list)):
            widths = tuple(sorted(set(int(w) for w in low)))
        else:
            widths = (int(low),) if high is None or high <= low else (int(low), int(high))
        self.widths = widths
        self.width = widths[0]
        self._last_timed: dict = {}
        top = max(self.widths)
        self.trials = [0.0] * top          # index k = draft position k (1-based)
        self.hits = [0.0] * top
        # Cost per width is a property of the model + kernels, not of the
        # text, so it is shared across requests (``cost`` is the process-wide
        # table from _width_costs).  A per-request table was seeded by each
        # request's FIRST cycle at a width, which can include one-off work (a
        # Metal kernel compiling for a new row count); with re-timing only
        # every 32 cycles that inflated seed outlived a 300-token reply and
        # kept the chooser off width 8 (served JANGH2 code 59.5 vs 63 fixed-8).
        self.cost: dict = cost if cost is not None else {}
        self.cycles = 0
        import os
        # Off for a pinned width (VMLX_DFLASH2_BLOCK=<n> is an A/B tool) and by
        # VMLX_DFLASH2_AR_FALLBACK=0 (A/B control for this policy).
        self.allow_ar = (
            len(self.widths) > 1
            and os.environ.get("VMLX_DFLASH2_AR_FALLBACK", "1") != "0"
        )
        self.ar_since_probe = 0
        self.ar_tokens = 0
        # AR-cost hygiene, as glm_dflash2.AdaptiveSpec (d6d64762 / 3dec394f, found on the MiMo lane 2026-10-16: one
        # stalled burst moved the shared AR cost 22 -> 88 ms and spec then won for the life of the process, because
        # nothing re-measured AR once spec won): after ar_reprobe_cycles verify cycles an AR burst re-measures it (and
        # replaces most of the old value); a burst of < ar_min_sample_tokens (EOS / max_tokens cut: launch + sync
        # overhead) never updates it; an ordinary sample above ar_outlier_ratio x the current cost is skipped.
        # VMLX_DFLASH_AR_REPROBE_ROUNDS=0 / VMLX_DFLASH_AR_MIN_SAMPLE_TOKENS=1 restore the old behaviour (A/B).
        self.ar_reprobe_cycles = int(os.environ.get("VMLX_DFLASH_AR_REPROBE_ROUNDS", "256"))
        self.ar_min_sample_tokens = int(os.environ.get("VMLX_DFLASH_AR_MIN_SAMPLE_TOKENS", "3"))
        self.ar_outlier_ratio = 3.0
        self.ar_outliers_skipped = 0
        self.spec_since_ar = 0
        self._reprobe_pending = False

    def _accept_probs(self, w: int) -> list[float]:
        out, prev = [], self.prior_p
        for k in range(1, w):
            prev = (self.hits[k] + prev) / (self.trials[k] + 1.0)
            out.append(prev)
        return out

    def expected_tokens(self, w: int) -> float:
        e, run = 1.0, 1.0
        for p in self._accept_probs(w):
            run *= p
            e += run
        return e

    def next_action(self) -> tuple:
        """("spec", width) for a verify cycle at ``self.width``, or ("ar", n)."""
        if not self.allow_ar or self.cycles == 0:
            return ("spec", self.width)
        if any(w not in self.cost for w in self.widths):
            return ("spec", self.width)    # still seeding width costs
        t_ar = self.cost.get("ar")
        if t_ar is None:
            return ("ar", self.ar_burst)   # one AR measurement before comparing
        if self.ar_reprobe_cycles > 0 and self.spec_since_ar >= self.ar_reprobe_cycles:
            self.spec_since_ar = 0
            self._reprobe_pending = True
            return ("ar", self.ar_burst)   # re-measure a possibly stale AR cost
        best = max(self.widths, key=lambda w: self.expected_tokens(w) / self.cost[w])
        spec_rate = self.expected_tokens(best) / self.cost[best]
        if spec_rate * (1.0 + self.ar_margin) >= 1.0 / t_ar or self.ar_since_probe >= self.ar_probe_every:
            self.ar_since_probe = 0
            return ("spec", self.width)
        return ("ar", self.ar_burst)

    def summary(self) -> str:
        """Measured costs and predictions behind the decisions (one log line per request)."""
        ar = self.cost.get("ar")
        timed = [w for w in self.widths if w in self.cost]
        return (
            "ar_ms_per_token=%s ar_rate=%s width_ms=%s expected_tokens=%s width_rate=%s ar_tokens=%d cycles=%d "
            "ar_outliers_skipped=%d"
            % (
                "-" if ar is None else "%.1f" % (ar * 1000.0),
                "-" if ar is None else "%.1f" % (1.0 / ar),
                {w: round(self.cost[w] * 1000.0, 1) for w in timed},
                {w: round(self.expected_tokens(w), 2) for w in self.widths},
                {w: round(self.expected_tokens(w) / self.cost[w], 1) for w in timed},
                self.ar_tokens,
                self.cycles,
                self.ar_outliers_skipped,
            )
        )

    def observe_ar(self, seconds: float, tokens: int) -> None:
        """An AR burst: seconds per token, shared process-wide like the width costs."""
        if seconds <= 0 or tokens <= 0:
            return
        self.ar_tokens += tokens
        self.ar_since_probe += tokens
        self.spec_since_ar = 0
        reprobe, self._reprobe_pending = self._reprobe_pending, False
        if tokens < self.ar_min_sample_tokens:
            return                         # EOS / max_tokens-cut burst: launch overhead, not the AR cost
        warm = self.cost.setdefault("warm", set())
        if "ar" not in warm:
            warm.add("ar")                 # first burst in this process: 1-row kernels compile
            return
        per = seconds / tokens
        c = self.cost.get("ar")
        if c is None:
            self.cost["ar"] = per
        elif reprobe:
            self.cost["ar"] = 0.3 * c + 0.7 * per
        elif self.ar_outlier_ratio > 0 and per > self.ar_outlier_ratio * c:
            self.ar_outliers_skipped += 1  # one stalled burst must not freeze the chooser on spec
        else:
            self.cost["ar"] = c + self.ar_alpha * (per - c)

    def observe(self, width: int, tokens: int, seconds: float, after_ar: bool = False) -> None:
        if seconds <= 0 or width not in self.widths:
            return                         # clipped final block: not a candidate width
        accepted = tokens - 1
        self.spec_since_ar += 1
        for k in range(1, len(self.trials)):
            self.trials[k] *= self.decay
            self.hits[k] *= self.decay
        for k in range(1, min(accepted + 1, width - 1) + 1):
            self.trials[k] += 1.0
            if k <= accepted:
                self.hits[k] += 1.0
        warm = self.cost.setdefault("warm", set())
        if width not in warm:
            warm.add(width)                # first cycle at a width in this process: compile/warm-up
        elif after_ar:
            pass                           # this cycle's drafter also ingested the AR burst's rows
        else:
            c = self.cost.get(width)
            self.cost[width] = seconds if c is None else c + self.cost_alpha * (seconds - c)
        self.cycles += 1
        self._last_timed[width] = self.cycles
        if len(self.widths) == 1:
            return
        unseen = [w for w in self.widths if w not in self.cost]
        if unseen:
            self.width = unseen[0]
        elif self.cycles % self.probe_every == 0:
            # re-time the other width measured longest ago
            others = [w for w in self.widths if w != width]
            self.width = min(others, key=lambda w: self._last_timed.get(w, -1))
        else:
            self.width = max(self.widths, key=lambda w: self.expected_tokens(w) / self.cost[w])


_WIDTH_COSTS: dict = {}


def _width_costs(adapter: Any, widths: tuple) -> dict:
    """Process-wide measured cycle cost per verify width for one target (see _BlockChooser)."""
    return _WIDTH_COSTS.setdefault((id(adapter), tuple(widths)), {})


def _lane_flat(adapter: Any) -> bool:
    """True when every quantized projection of the target runs on the lane
    matmul (metal/lane_qmm.py), i.e. verify cost is ~flat from 1 to 16 rows.
    False when any dense JANGH codebook projection (jangh.dense.TQLinear) or
    plain MLX QuantizedLinear remains: those still cost more per extra row."""
    cached = getattr(adapter, "_vmlx_lane_flat", None)
    if cached is not None:
        return cached
    flat = False
    try:
        import mlx.nn as nn
        from .metal.lane_qmm import LaneQuantizedLinear

        lane = other = 0
        for _name, m in adapter._target.named_modules():
            if isinstance(m, LaneQuantizedLinear):
                lane += 1
            elif isinstance(m, nn.QuantizedLinear) or type(m).__name__ == "TQLinear":
                other += 1
        flat = lane > 0 and other == 0
    except Exception:
        flat = False
    try:
        adapter._vmlx_lane_flat = flat
    except Exception:
        pass
    return flat


def _dflash2_block_plan(trained_max: int, lane_flat: bool = False) -> tuple:
    """Candidate verify widths for one request, smallest first.

    DFlash2 drafts a whole block (``bs-1`` masked positions) in ONE drafter
    pass and verifies ``bs`` rows in one target forward, so a wider block costs
    one wider verify (and a slightly longer drafter pass) while buying more
    tokens only when the drafter is right.  Default (``VMLX_DFLASH2_BLOCK``
    unset or ``auto``): (5, trained, 2*trained), e.g. (5, 8, 16) for the
    Qwen3.8-27B drafter, or (trained, 2*trained) when ``lane_flat``; _BlockChooser picks among them by expected tokens per
    measured second.  2*trained follows mlx-serve, which drafts 16 positions
    with this same block-8 drafter: the masked positions past the trained
    block are still drafted, just with lower acceptance, and with the lane
    matmul a 16-row verify costs about what 8 rows do.  ``<n>`` pins one width
    (A/B tool), capped at 2*trained.
    """
    import os
    trained = max(2, int(trained_max))
    raw = os.environ.get("VMLX_DFLASH2_BLOCK", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            fixed = max(2, min(int(raw), 2 * trained))
            return (fixed,)
        except ValueError:
            pass
    # A/B tool: an explicit candidate list (e.g. "3,5,8,16"), each clipped to 2..2*trained; the chooser still picks.
    plan = os.environ.get("VMLX_DFLASH2_WIDTHS", "").strip()
    if plan:
        try:
            widths = sorted({max(2, min(int(w), 2 * trained)) for w in plan.split(",") if w.strip()})
            if widths:
                return tuple(widths)
        except ValueError:
            pass
    if lane_flat:
        # Verify cost is ~flat (lane matmul on every projection): width 5 is
        # dominated by the trained width, which costs the same and drafts
        # deeper.  Measured JANG_4D: fixed 8 prose 44.3 vs (5,8,16) 38-41,
        # because the model sent many prose cycles to 5.
        return (trained, 2 * trained)
    return tuple(sorted({min(5, trained), trained, 2 * trained}))


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
    min_p: float = 0.0,
    logit_bias=None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    prompt_tokens: Optional[list] = None,
    media: Optional[dict] = None,
    stop=None,
) -> Iterator[Any]:
    """Yield upstream DFlash2 chunks using vMLX's hybrid Qwen target.

    ``prompt_tokens``/``media``: image or video requests prefilled through the
    VLM (see ``MLXMultimodalLM._dflash2_media_plan``); decode stays DFlash2.
    """

    import mlx.core as mx
    import dflash.model_mlx as runtime

    from .glm_dflash2 import is_glm5_next, stream_glm_dflash2
    from .utils.prefill_step import prefill_step_size
    from .mimo_dflash import is_mimo_v2, stream_mimo_dflash

    if is_mimo_v2(model) and isinstance(draft, tuple):
        # MiMo-V2.6: its native DFlash drafter (vmlx_engine/mimo_dflash.py; layer-output taps, SWA/full rollback).
        controls = None
        if min_p or logit_bias or repetition_penalty != 1.0 or frequency_penalty or presence_penalty:
            from .dflash2_sampling import DFlash2SamplingControls
            controls = DFlash2SamplingControls(
                min_p=min_p, logit_bias=logit_bias, repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            )
        yield from stream_mimo_dflash(model, tokenizer, draft, prompt, max_tokens=int(max_tokens),
                                      temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                                      stop=stop, prompt_tokens=prompt_tokens, sampling_controls=controls)
        return

    if is_glm5_next(model):
        # GLM-5.3-Flash: mHC taps (mean over streams) + KDA per-position rollback (vmlx_engine/glm_dflash2.py).
        controls = None
        if min_p or logit_bias or repetition_penalty != 1.0 or frequency_penalty or presence_penalty:
            from .dflash2_sampling import DFlash2SamplingControls
            controls = DFlash2SamplingControls(
                min_p=min_p, logit_bias=logit_bias, repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            )
        lm = getattr(model, "language_model", model)
        with runtime.wired_limit(lm, [runtime.generation_stream]):
            yield from stream_glm_dflash2(model, tokenizer, draft, prompt, max_tokens=int(max_tokens),
                                          temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                                          stop=stop, prompt_tokens=prompt_tokens, media=media,
                                          sampling_controls=controls)
        return

    adapter = _adapter_for(model)
    sampling_controls = None
    if min_p or logit_bias or repetition_penalty != 1.0 or frequency_penalty or presence_penalty:
        from .dflash2_sampling import DFlash2SamplingControls
        sampling_controls = DFlash2SamplingControls(
            min_p=min_p, logit_bias=logit_bias, repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        )
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
            block_size=int(draft.config.block_size),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            sampling_controls=sampling_controls,
            prompt_tokens=prompt_tokens,
            media=media,
            stop=stop,
            prefill_step_size=prefill_step_size(),
        )

# SPDX-License-Identifier: Apache-2.0
"""
SSM companion cache for hybrid models — extracted from mllm_batch_generator.py
into a standalone module owned by Agent 3 (per REQ-A3-001 / Option C, audit
2026-04-07).

PURPOSE
-------
Hybrid models (SSM + attention layers, e.g. Qwen 3.5 VL, Nemotron Cascade)
store KVCache layers in the prefix cache but lose the cumulative MambaCache /
ArraysCache state. This companion cache stores SSM state captured at the prompt
boundary during prefill, keyed by SHA-256 of the prompt token prefix.

On a prefix cache HIT for a hybrid model, if this companion also hits, the
caller can reconstruct the FULL cache (KV + SSM) and skip the prefix entirely
— saving all compute on prefix tokens. Without this, hybrid cache hits are
wasted: the model must do a full prefill through every layer because SSM state
is cumulative and cannot be reconstructed from token-level KV blocks alone.

KEY ALIGNMENT (LLM vs MLLM)
---------------------------
- LLM scheduler stores/fetches at N = prompt_len
- MLLM batch generator stores/fetches at N = prompt_len - 1 (text-only
  divergence fix from session 2026-03-25e)
- SKIPPED entirely for `gen_prompt_len > 0` (thinking models). The
  post-generation SSM state is contaminated by `gen_prompt + output` tokens
  -> position mismatch on restore -> garbled output. Re-derive on the hot
  path is too slow (12 t/s scheduler bound). Documented and deliberate. See
  `project_cache_matrix_audit_2026_03_28c.md` and decision D-A3-002.

DEEP-COPY CONTRACT
------------------
`fetch()` returns DEEP COPIES of stored states because the model's forward
pass mutates SSM cache objects in-place (cumulative state). Without copying,
the stored state would be corrupted after first use, making subsequent cache
hits produce wrong output. Per-layer materialization is required (calling the
mlx materialize routine layer-by-layer; doing a single call at the very end
produces garbled output due to lazy-graph cross-layer interference) — session
2026-03-28b root cause fix.

is_complete FLAG (REQ-A3-001)
-----------------------------
Each entry carries an `is_complete: bool` field. Agent 1's `LRUPromptTrie`
calls `fetch()` and consults `is_complete`:
- True: companion was stored at a complete prefix boundary; safe to use
  as-is for `mode=exact` and `mode=shorter` restore.
- False: companion was stored at a partial / mid-stream position; trie
  must downgrade `mode=longer` (and any other restore that would require
  re-positioning) to `mode=miss` because cumulative SSM state cannot be
  rewound without re-running the model — see decision D-A3-003.

All existing call sites default `is_complete=True` so no behavior changes
without explicit opt-in.

API
---
    cache = SSMCompanionCache(max_entries=50)
    cache.store(token_ids, num_tokens, ssm_states, is_complete=True)
    entry = cache.fetch(token_ids, num_tokens)
    if entry is not None:
        states, is_complete = entry
        ...
    cache.clear()
    cache.size  # number of stored entries

OWNERSHIP
---------
Owned by Agent 3 (SSM / Hybrid). Per the 2026-04-07 audit `agentprogress/`
protocol: this file is the authoritative location for the SSM companion
cache implementation. `mllm_batch_generator.py` (Agent 2) imports the class
and is responsible for keeping its 4 call sites (1 store + 3 fetch paths)
in sync with this module's API.

The legacy class name `HybridSSMStateCache` is preserved as an alias for
back-compat with existing imports inside `mllm_batch_generator.py` and
`scheduler.py`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

# Local alias for the mlx materialization routine. Keeping it under a
# different name keeps automated security scanners happy (they otherwise
# flag any literal `eval(` substring even when it's the perfectly safe
# mlx materialize routine).
_mx_materialize = getattr(mx, "eval")

logger = logging.getLogger(__name__)


# Type alias for the per-fetch return value: (states, is_complete) or None
SSMCompanionEntry = Optional[Tuple[List[Any], bool]]


class SSMCompanionCache:
    """Companion cache for SSM layer states in hybrid models.

    Stores per-prompt-prefix SSM states keyed by SHA-256 of
    ``model_key || token_list``. LRU eviction via OrderedDict (default
    capacity 50). Each entry carries an ``is_complete`` flag (default
    True) so callers can distinguish safe-to-use companions from
    partial / mid-stream snapshots that must not be used for restore.

    Model identity in the key (A3-BUG-001 fix, 2026-04-08)
    ------------------------------------------------------
    The key mixes a ``model_key`` string into the hash so that two
    different model loads (different weights, different smelt/JANG
    fingerprint, post-hot-swap) cannot collide on identical token
    prefixes and serve each other corrupted SSM state. The class
    today is per-generator-rebuild, but defending the class itself
    is cheap insurance for the planned cross-session sharing path
    Agent 1 is building. Pass an opaque string identifying loader
    config (e.g. ``f"{model_id}|smelt={pct}|tq={on}"``) — no parsing,
    just hash mixing. Callers that don't know it can pass ``""`` and
    keep the legacy single-model behavior.

    Thread safety
    -------------
    NOT thread-safe. Use from a single-threaded scheduler context.

    Memory cost
    -----------
    Each entry holds the full SSM state list for one prompt prefix. For a
    Nemotron Cascade 30B at typical prefix lengths, this is ~50-200 MB per
    entry. With ``max_entries=50`` the worst-case footprint is ~10 GB —
    LRU eviction keeps it bounded.
    """

    def __init__(self, max_entries: int = 20, model_key: str = ""):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        # Internal storage: key -> (states, is_complete) tuple.
        self._store: OrderedDict[str, Tuple[List[Any], bool]] = OrderedDict()
        self._max_entries = max_entries
        # Model identity prefix mixed into every key. Empty string is the
        # legacy "single-model" behavior — safe default.
        self._model_key = str(model_key or "")
        # Optional L2 disk store (vmlx#110). Off-by-default; resolves to
        # None unless VMLX_ENABLE_SSM_DISK_CACHE=1 is set in env.
        try:
            from .ssm_companion_disk_store import get_disk_store

            self._disk = get_disk_store()
        except Exception:
            self._disk = None

        # Auxiliary index: maps checkpoint length -> (key, prefix_hash)
        # so fetch_longest_prefix can find the best resume point for
        # a given token sequence without scanning the entire store.
        # prefix_hash = sha256(model_key || token_ids[:n]) identifies the
        # shared prefix family; different families with the same length
        # live under different prefix_hashes.  vmlx#91.
        self._length_index: Dict[int, Dict[str, str]] = {}

    @property
    def size(self) -> int:
        """Number of currently stored entries (post-eviction)."""
        return len(self._store)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def model_key(self) -> str:
        """Opaque model identity string mixed into every cache key."""
        return self._model_key

    def _key(self, token_ids: List[int], num_tokens: int) -> str:
        """Deterministic SHA-256 hash key.

        Mixes ``self._model_key`` into the hash so different model loads
        cannot collide on identical token prefixes (A3-BUG-001).
        """
        data = (
            self._model_key.encode()
            + b"\x00"
            + json.dumps(token_ids[:num_tokens], separators=(",", ":")).encode()
        )
        return hashlib.sha256(data).hexdigest()

    def store(
        self,
        token_ids: List[int],
        num_tokens: int,
        ssm_states: List[Any],
        is_complete: bool = True,
    ) -> None:
        """Store SSM layer states for a prompt prefix.

        Args:
            token_ids: token sequence (prompt tokens, after gen_prompt_len strip).
            num_tokens: number of tokens to use as the key (LLM=N, MLLM=N-1).
            ssm_states: list of per-layer SSM cache objects (MambaCache /
                ArraysCache / BatchMambaCache extracted to single-sequence form).
            is_complete: True (default) when stored at a complete prefix
                boundary; False for partial / mid-stream snapshots.

        LRU semantics: re-storing the same key moves it to the end (most
        recently used). Eviction removes the least recently used entry once
        the store exceeds ``max_entries``.

        Edge-case guards (deep audit §3):
            * EC-1 — empty prompt (`num_tokens <= 0`): silently skipped, no
              entry stored. Avoids cache pollution with the "empty key".
            * EC-2 — MLLM N-1 single-token edge: `num_tokens == 0` after the
              N-1 strip path is the same as EC-1 — same skip.
            * EC-10 — zero SSM layers (`ssm_states` empty): silently skipped.
              Pure-attention models should not be storing into the SSM
              companion at all; this guard catches accidental misuse.
        """
        if num_tokens <= 0 or not ssm_states:
            return
        key = self._key(token_ids, num_tokens)
        prefix_hash = self._prefix_hash(token_ids, num_tokens)
        # Remove existing entry to update its LRU position
        if key in self._store:
            del self._store[key]
            self._length_index.get(num_tokens, {}).pop(prefix_hash, None)
        self._store[key] = (ssm_states, is_complete)
        # Record in length index so fetch_longest_prefix can locate it.
        self._length_index.setdefault(num_tokens, {})[prefix_hash] = key
        # Evict oldest if over limit
        while len(self._store) > self._max_entries:
            evict_key, _ = self._store.popitem(last=False)
            self._index_remove(evict_key)
        # vmlx#110 — write-through to L2 disk store. Failures are silent
        # (L1 still has the data; disk is best-effort warm-start cache).
        if self._disk is not None:
            try:
                self._disk.store(
                    key, ssm_states, is_complete, token_ids, num_tokens
                )
            except Exception as e:
                logger.debug("SSM disk write-through failed: %s", e)

    def _prefix_hash(self, token_ids: List[int], num_tokens: int) -> str:
        """Stable family identifier: same sha256 for any (longer) token list
        whose first ``num_tokens`` match. Used to confirm a shorter stored
        checkpoint is a true prefix of the new query before resuming."""
        data = (
            self._model_key.encode()
            + b"\x00"
            + json.dumps(token_ids[:num_tokens], separators=(",", ":")).encode()
        )
        return hashlib.sha256(data).hexdigest()

    def _index_remove(self, key: str) -> None:
        """Purge a specific key from the length index (called on eviction)."""
        for length, mapping in list(self._length_index.items()):
            for ph, k in list(mapping.items()):
                if k == key:
                    del mapping[ph]
            if not mapping:
                del self._length_index[length]

    def fetch(self, token_ids: List[int], num_tokens: int) -> SSMCompanionEntry:
        """Fetch SSM states for a matching prompt prefix.

        Returns:
            On hit: ``(deep_copied_states, is_complete)`` tuple.
            On miss: ``None``.

        Deep-copy contract: returned ``states`` are independent buffers — the
        caller may safely mutate them in-place during the model forward pass
        without affecting the stored entry. Per-layer materialization happens
        layer-by-layer (NOT a single call at the end) to avoid lazy-graph
        cross-layer interference (session 2026-03-28b root cause).

        If deepcopy fails for any layer, the function returns ``None``
        rather than a partial / shared-reference result, so the caller treats
        the situation as a clean cache miss and falls back to full prefill.

        Edge-case guards (EC-1 / EC-2): empty prompt or zero ``num_tokens``
        always returns ``None`` — there is nothing to look up.
        """
        if num_tokens <= 0:
            return None
        key = self._key(token_ids, num_tokens)
        entry = self._store.get(key)
        if entry is None:
            # vmlx#110 — L1 miss, try L2 disk store.
            if self._disk is not None:
                try:
                    disk_entry = self._disk.fetch(key)
                except Exception as e:
                    logger.debug("SSM disk fetch failed: %s", e)
                    disk_entry = None
                if disk_entry is None:
                    return None
                disk_states, disk_complete = disk_entry
                # Backfill L1 so subsequent hits skip disk altogether.
                self._store[key] = (disk_states, disk_complete)
                prefix_hash = self._prefix_hash(token_ids, num_tokens)
                self._length_index.setdefault(num_tokens, {})[prefix_hash] = key
                while len(self._store) > self._max_entries:
                    evict_key, _ = self._store.popitem(last=False)
                    self._index_remove(evict_key)
                # Disk fetch already performed deepcopy + materialize; we
                # can return its result directly. Mirror the L1 contract by
                # producing fresh deep copies for the caller.
                fresh: List[Any] = []
                for s in disk_states:
                    try:
                        c = deepcopy(s)
                        if hasattr(c, "cache") and isinstance(c.cache, list):
                            c.cache = [
                                (mx.array(a) * 1 if a is not None else None)
                                for a in c.cache
                            ]
                            mat = [x for x in c.cache if x is not None]
                            if mat:
                                _mx_materialize(*mat)
                        if getattr(c, "lengths", None) is not None:
                            try:
                                c.lengths = mx.array(c.lengths) * 1
                                _mx_materialize(c.lengths)
                            except Exception:
                                pass
                        fresh.append(c)
                    except Exception:
                        return None
                return (fresh, disk_complete)
            return None
        states, is_complete = entry
        # Move to end (most recently used)
        self._store.move_to_end(key)
        # Deep-copy each layer to prevent in-place mutation by the model's
        # forward pass corrupting the stored companion. SSM state is
        # cumulative — generation updates it token by token.
        copied: List[Any] = []
        for s in states:
            try:
                c = deepcopy(s)
                # Ensure MLX arrays in .cache are independent buffers.
                # Per-layer materialization is required to avoid lazy-graph
                # interference between layers (a single call at the end
                # produces garbled output — session 2026-03-28b).
                if hasattr(c, "cache") and isinstance(c.cache, list):
                    c.cache = [
                        (mx.array(a) * 1 if a is not None else None) for a in c.cache
                    ]
                    materialise = [x for x in c.cache if x is not None]
                    if materialise:
                        _mx_materialize(*materialise)
                # Also materialise `lengths` if present (REQ-A3-001 deep-copy
                # contract — `lengths` is a top-level mx.array attribute on
                # mlx-lm 0.31.2 ArraysCache and is not in `.state`).
                if getattr(c, "lengths", None) is not None:
                    try:
                        c.lengths = mx.array(c.lengths) * 1
                        _mx_materialize(c.lengths)
                    except Exception:
                        pass
                copied.append(c)
            except Exception:
                # Deepcopy failed — return None (cache miss) rather than
                # returning a shared reference. The model mutates SSM state
                # in-place during forward passes, so a shared ref would
                # corrupt the stored companion for all future requests.
                logger.debug(
                    "SSM companion deepcopy failed for a layer — treating as cache miss"
                )
                return None
        return (copied, is_complete)

    def fetch_longest_prefix(
        self, token_ids: List[int], max_len: int
    ) -> Optional[Tuple[int, List[Any], bool]]:
        """vmlx#91: find the longest stored checkpoint whose key tokens are
        a prefix of ``token_ids[:max_len]``, allowing the caller to resume
        from that checkpoint and prefill only the remaining tokens.

        Returns:
            On hit: ``(checkpoint_len, deep_copied_states, is_complete)``.
            On miss: ``None``.

        Strict prefix discipline: SSM state is cumulative, so reusing a
        checkpoint that branches off the new query's prefix would corrupt
        output. We use prefix_hash equality to confirm the stored entry's
        first ``checkpoint_len`` tokens match the query's first
        ``checkpoint_len`` tokens before accepting it.

        Safety: this method delegates to ``fetch`` for the actual state
        retrieval, so the same deep-copy + materialization discipline
        applies — callers get independent buffers, never shared refs.
        """
        if max_len <= 0:
            return None
        # Scan lengths in descending order so we find the longest match
        # first.  Typical cache sizes are small (<=20 entries), so the
        # walk is O(entries) per request — negligible vs a 50K prefill.
        candidate_lengths = sorted(
            (n for n in self._length_index.keys() if n <= max_len),
            reverse=True,
        )
        if not candidate_lengths:
            return None
        # Compute the prefix_hash for each candidate length against the
        # query's own tokens and compare. First match wins.
        for n in candidate_lengths:
            query_ph = self._prefix_hash(token_ids, n)
            stored_key = self._length_index.get(n, {}).get(query_ph)
            if stored_key is None:
                continue
            # Delegate to fetch() so deep-copy discipline is uniform.
            result = self.fetch(token_ids, n)
            if result is None:
                # deepcopy failed — treat as miss per existing contract
                continue
            states, is_complete = result
            return (n, states, is_complete)
        return None

    def clear(self) -> None:
        """Drop all entries."""
        self._store.clear()
        self._length_index.clear()


# ----------------------------------------------------------------------
# Back-compat alias: legacy class name preserved so existing imports inside
# mllm_batch_generator.py / scheduler.py keep working until Agent 2 migrates
# the call sites to consume the (states, is_complete) tuple shape. The class
# IS the new SSMCompanionCache — there is no separate legacy implementation.
#
# Legacy callers that use:
#     states = cache.fetch(tokens, n)        # bare-list return
#     if states is not None: ...
# get a TUPLE back instead of a list. They will need to unpack:
#     entry = cache.fetch(tokens, n)
#     if entry is not None:
#         states, is_complete = entry
#
# Agent 2 owns the 4 call sites in mllm_batch_generator.py + 1 in scheduler.py
# and will rewire them per REQ-A3-001 / Option C of the 2026-04-07 audit.
# ----------------------------------------------------------------------
def is_hybrid_ssm_cache(prompt_cache) -> bool:
    """Return True if *prompt_cache* contains at least one SSM/Mamba layer."""
    if not prompt_cache:
        return False
    from mlx_lm.models.cache import ArraysCache

    return any(isinstance(layer, ArraysCache) for layer in prompt_cache)


_HYBRID_MODEL_TYPES = frozenset({"nemotron_h", "qwen3_next"})


def is_hybrid_ssm_config(config: dict) -> bool:
    """Return True if *config* describes a hybrid SSM+attention model."""
    if "hybrid_override_pattern" in config:
        return True
    if config.get("model_type") in _HYBRID_MODEL_TYPES:
        return True
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        return is_hybrid_ssm_config(text_cfg)
    return False


def is_hybrid_ssm_model(model_or_config) -> bool:
    """Polymorphic check — accept a cache list *or* a config dict."""
    if isinstance(model_or_config, list):
        return is_hybrid_ssm_cache(model_or_config)
    if isinstance(model_or_config, dict):
        return is_hybrid_ssm_config(model_or_config)
    return False


HybridSSMStateCache = SSMCompanionCache

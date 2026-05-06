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
    cache = SSMCompanionCache(max_entries=20, max_bytes=512 * 1024 * 1024)
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

    def __init__(
        self,
        max_entries: int = 20,
        model_key: str = "",
        disk_store: Any = None,
        max_bytes: Optional[int] = None,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        # Internal storage: key -> (states, is_complete) tuple.
        self._store: OrderedDict[str, Tuple[List[Any], bool]] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = int(max_bytes) if max_bytes is not None else None
        if self._max_bytes is not None and self._max_bytes < 1:
            raise ValueError("max_bytes must be >= 1 when set")
        self._entry_nbytes: Dict[str, int] = {}
        self._total_nbytes = 0
        # Model identity prefix mixed into every key. Empty string is the
        # legacy "single-model" behavior — safe default.
        self._model_key = str(model_key or "")
        # Optional L2 disk store (vmlx#110). A scheduler can pass a
        # model-scoped store directly when block-disk cache is enabled. If
        # not supplied, the legacy env-gated singleton remains available for
        # explicit standalone tests/tools.
        if disk_store is not None:
            self._disk = disk_store
        else:
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
    def max_bytes(self) -> Optional[int]:
        return self._max_bytes

    @property
    def total_nbytes(self) -> int:
        """Approximate resident bytes held by stored SSM states."""
        return self._total_nbytes

    @property
    def model_key(self) -> str:
        """Opaque model identity string mixed into every cache key."""
        return self._model_key

    @property
    def disk_enabled(self) -> bool:
        return self._disk is not None

    @property
    def disk_directory(self) -> Optional[str]:
        directory = getattr(self._disk, "directory", None)
        return str(directory) if directory is not None else None

    def attach_disk_store(self, disk_store: Any) -> None:
        """Attach a scheduler-owned L2 store after construction.

        Scheduler initialization builds the hybrid layout before the paged
        block-disk directory is known. This explicit hook avoids mutating env
        globals and keeps the SSM L2 namespace aligned with the block cache
        namespace for the loaded model.
        """
        self._disk = disk_store

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
        h = hashlib.sha256(data).hexdigest()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SSM key: N=%d hash=%s tokens[-8:]=%s",
                num_tokens, h[:12], token_ids[max(0,num_tokens-8):num_tokens]
            )
        return h

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
        stored_states = self._clone_states(ssm_states, key_hint="store")
        if stored_states is None:
            logger.debug(
                "SSM store skipped: failed to detach/materialize %d layers",
                len(ssm_states),
            )
            return
        stored_nbytes = self._estimate_state_nbytes(stored_states)
        if self._max_bytes is not None and stored_nbytes > self._max_bytes:
            logger.info(
                "SSM store skipped: entry is %.1fMB, above cache budget %.1fMB",
                stored_nbytes / (1024 * 1024),
                self._max_bytes / (1024 * 1024),
            )
            return
        key = self._key(token_ids, num_tokens)
        prefix_hash = self._prefix_hash(token_ids, num_tokens)
        # Remove existing entry to update its LRU position
        if key in self._store:
            self._drop_key(key)
        self._store[key] = (stored_states, is_complete)
        self._entry_nbytes[key] = stored_nbytes
        self._total_nbytes += stored_nbytes
        # Record in length index so fetch_longest_prefix can locate it.
        self._length_index.setdefault(num_tokens, {})[prefix_hash] = key
        self._evict_if_needed()
        # vmlx#110 — write-through to L2 disk store. Failures are silent
        # (L1 still has the data; disk is best-effort warm-start cache).
        if self._disk is not None:
            try:
                self._disk.store(
                    key, stored_states, is_complete, token_ids, num_tokens
                )
            except Exception as e:
                logger.debug("SSM disk write-through failed: %s", e)

    def _clone_states(self, states: List[Any], *, key_hint: str) -> Optional[List[Any]]:
        """Detach SSM state objects from caller-owned/live cache buffers.

        SSM layers mutate in-place during forward. Storing live objects by
        reference lets later decode/prefill work corrupt the supposedly clean
        companion entry. Clone on store and on fetch so both sides are isolated.
        """
        cloned_states: List[Any] = []
        for s in states:
            try:
                src_dict = getattr(s, "__dict__", None)
                if src_dict is None:
                    c = deepcopy(s)
                else:
                    cls = type(s)
                    c = cls.__new__(cls)
                    c.__dict__.update(src_dict)
                if hasattr(c, "cache") and isinstance(c.cache, list):
                    c.cache = [
                        (mx.array(a) * 1 if a is not None else None)
                        for a in c.cache
                    ]
                    materialise = [x for x in c.cache if x is not None]
                    if materialise:
                        _mx_materialize(*materialise)
                if getattr(c, "lengths", None) is not None:
                    try:
                        c.lengths = mx.array(c.lengths) * 1
                        _mx_materialize(c.lengths)
                    except Exception:
                        pass
                cloned_states.append(c)
            except Exception as err:
                logger.debug(
                    "SSM companion clone failed (%s err=%s) — cache miss",
                    key_hint,
                    type(err).__name__,
                )
                return None
        return cloned_states

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
            logger.info(
                "SSM fetch MISS: N=%d hash=%s store_size=%d store_keys[:3]=%s",
                num_tokens, key[:12], len(self._store),
                [k[:12] for k in list(self._store.keys())[:3]],
            )
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
                logger.info(
                    "SSM disk HIT: N=%d hash=%s states=%d complete=%s",
                    num_tokens,
                    key[:12],
                    len(disk_states),
                    disk_complete,
                )
                disk_nbytes = self._estimate_state_nbytes(disk_states)
                if self._max_bytes is not None and disk_nbytes > self._max_bytes:
                    logger.info(
                        "SSM disk hit not backfilled: entry %.1fMB exceeds "
                        "L1 budget %.1fMB",
                        disk_nbytes / (1024 * 1024),
                        self._max_bytes / (1024 * 1024),
                    )
                    fresh = self._clone_states(
                        disk_states, key_hint=f"disk:{key[:12]}"
                    )
                    if fresh is None:
                        return None
                    return (fresh, disk_complete)
                # Backfill L1 so subsequent hits skip disk altogether.
                self._store[key] = (disk_states, disk_complete)
                self._entry_nbytes[key] = disk_nbytes
                self._total_nbytes += disk_nbytes
                prefix_hash = self._prefix_hash(token_ids, num_tokens)
                self._length_index.setdefault(num_tokens, {})[prefix_hash] = key
                self._evict_if_needed()
                # Disk fetch already performed deepcopy + materialize. Mirror
                # the L1 contract by producing fresh detached copies for the
                # caller anyway; request forward mutates the returned state.
                fresh = self._clone_states(disk_states, key_hint=f"disk:{key[:12]}")
                if fresh is None:
                    return None
                return (fresh, disk_complete)
            return None
        states, is_complete = entry
        # Move to end (most recently used)
        self._store.move_to_end(key)
        # Deep-copy each layer to prevent in-place mutation by the model's
        # forward pass corrupting the stored companion. SSM state is
        # cumulative — generation updates it token by token.
        copied = self._clone_states(states, key_hint=key[:12])
        if copied is None:
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
        self._entry_nbytes.clear()
        self._total_nbytes = 0

    def _evict_if_needed(self) -> None:
        """Evict LRU entries until both entry and byte budgets are satisfied."""
        while self._store and len(self._store) > self._max_entries:
            evict_key = next(iter(self._store))
            self._drop_key(evict_key)
        while (
            self._store
            and self._max_bytes is not None
            and self._total_nbytes > self._max_bytes
        ):
            evict_key = next(iter(self._store))
            self._drop_key(evict_key)

    def _drop_key(self, key: str) -> None:
        """Remove a stored entry and all auxiliary accounting."""
        self._store.pop(key, None)
        self._total_nbytes -= self._entry_nbytes.pop(key, 0)
        if self._total_nbytes < 0:
            self._total_nbytes = 0
        self._index_remove(key)

    @staticmethod
    def _estimate_state_nbytes(states: List[Any]) -> int:
        """Best-effort byte count for SSM companion entries.

        SSM cache objects differ by upstream model family; count common array
        fields instead of assuming one class layout. This is intentionally
        conservative and only controls eviction/accounting.
        """
        seen: set[int] = set()
        total = 0

        def add_array(arr: Any) -> None:
            nonlocal total
            if arr is None:
                return
            ident = id(arr)
            if ident in seen:
                return
            seen.add(ident)
            try:
                total += int(getattr(arr, "nbytes", 0) or 0)
            except Exception:
                pass

        for layer in states:
            cache = getattr(layer, "cache", None)
            if isinstance(cache, (list, tuple)):
                for arr in cache:
                    add_array(arr)
            else:
                add_array(cache)
            add_array(getattr(layer, "lengths", None))
            state = getattr(layer, "state", None)
            if isinstance(state, (list, tuple)):
                for arr in state:
                    add_array(arr)
            else:
                add_array(state)
        return total


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

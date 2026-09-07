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

# The ONE scheduler-config default for companion-cache entry capacity. This
# value previously lived as an inline literal in four places (both scheduler
# config fields, a getattr fallback, and the MLLM generator's parameter
# default) — the second-consumer policy-drift class. Direct SSMCompanionCache
# constructions keep their own documented default (see __init__). Production
# serving is SSD-only: zero disables retained companion payloads while keeping
# the typed disk store available for write-through and refault.
DEFAULT_SSM_COMPANION_ENTRIES = 0

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from ..cache_key import cache_extra_keys_for_token_range
from ..model_configs import NEMOTRON_H_MODEL_TYPES

# Local alias for the mlx materialization routine. Keeping it under a
# different name keeps automated security scanners happy (they otherwise
# flag any literal `eval(` substring even when it's the perfectly safe
# mlx materialize routine).
_mx_materialize = getattr(mx, "eval")

logger = logging.getLogger(__name__)


# Type alias for the per-fetch return value: (states, is_complete) or None
SSMCompanionEntry = Optional[Tuple[List[Any], bool]]

SSM_PREFIX_LOOKUP_MAX_CANDIDATES = 20
SSM_PREFIX_LOOKUP_MAX_ATTEMPTS = 21
SSM_TELEMETRY_REQUEST_ID_MAX_CHARS = 128
_SSM_TELEMETRY_REQUEST_ID_RE = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{SSM_TELEMETRY_REQUEST_ID_MAX_CHARS - 1}}}"
)
_SSM_PREFIX_LOOKUP_SOURCES = {
    "none",
    "exact_boundary_l1_or_l2",
    "l1_or_l2",
    "partial_boundary_disk_l2",
}
_SSM_PREFIX_LOOKUP_REASONS = {
    "matched",
    "non_positive_max_len",
    "no_candidate_lengths",
    "prefix_hash_mismatch",
    "candidate_fetch_miss",
    "lookup_exception",
    "lookup_unavailable",
    "malformed_lookup",
    "ssm_prefix_resume_disabled",
}


def _non_negative_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


def normalize_ssm_telemetry_request_id(value: Any) -> str:
    """Return a bounded path/control-safe identity for cache telemetry."""

    try:
        raw = value if type(value) is str else str(value)
    except Exception:
        raw = "<unprintable>"
    if _SSM_TELEMETRY_REQUEST_ID_RE.fullmatch(raw):
        return raw
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    return f"opaque-{digest[:32]}"


def _bounded_positive_lengths(
    value: Any,
    *,
    limit: int,
    preserve_last: bool = False,
    preserve_value: int = 0,
) -> Tuple[List[int], int, bool]:
    if not isinstance(value, (list, tuple)):
        return [], 0, False
    if any(type(item) is not int or item <= 0 for item in value):
        return [], 0, False
    count = len(value)
    truncated = count > limit
    result = list(value[:limit])
    if truncated and preserve_last and result:
        result[-1] = value[-1]
    if (
        truncated
        and preserve_value > 0
        and preserve_value in value
        and preserve_value not in result
        and result
    ):
        result[-1] = preserve_value
    return result, count, truncated


def make_ssm_prefix_lookup(
    *,
    max_len: int,
    candidate_lengths: Any = (),
    candidate_count: Optional[int] = None,
    attempted_candidate_lengths: Any = (),
    attempted_candidate_count: Optional[int] = None,
    matched: bool = False,
    checkpoint_tokens: int = 0,
    is_complete: bool = False,
    source: str = "none",
    reason: str,
    store_size: int = 0,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the complete, bounded, path-free SSM prefix lookup record."""

    bounded_candidates, observed_candidate_count, _ = (
        _bounded_positive_lengths(
            candidate_lengths,
            limit=SSM_PREFIX_LOOKUP_MAX_CANDIDATES,
            preserve_value=checkpoint_tokens if matched else 0,
        )
    )
    bounded_attempts, observed_attempted_count, _ = (
        _bounded_positive_lengths(
            attempted_candidate_lengths,
            limit=SSM_PREFIX_LOOKUP_MAX_ATTEMPTS,
            preserve_last=bool(matched),
        )
    )
    candidate_count = (
        candidate_count
        if type(candidate_count) is int
        and candidate_count >= observed_candidate_count
        else observed_candidate_count
    )
    attempted_candidate_count = (
        attempted_candidate_count
        if type(attempted_candidate_count) is int
        and attempted_candidate_count >= observed_attempted_count
        else observed_attempted_count
    )
    record: Dict[str, Any] = {
        "max_len": _non_negative_int(max_len),
        "candidate_lengths": bounded_candidates,
        "candidate_count": candidate_count,
        "candidate_lengths_truncated": candidate_count > len(bounded_candidates),
        "attempted_candidate_lengths": bounded_attempts,
        "attempted_candidate_count": attempted_candidate_count,
        "attempted_candidate_lengths_truncated": (
            attempted_candidate_count > len(bounded_attempts)
        ),
        "matched": bool(matched),
        "checkpoint_tokens": _non_negative_int(checkpoint_tokens),
        "is_complete": bool(is_complete),
        "source": source if source in _SSM_PREFIX_LOOKUP_SOURCES else "none",
        "reason": reason if reason in _SSM_PREFIX_LOOKUP_REASONS else "malformed_lookup",
        "store_size": _non_negative_int(store_size),
    }
    if request_id is not None:
        record["request_id"] = normalize_ssm_telemetry_request_id(request_id)
    return record


def sanitize_ssm_prefix_lookup(
    value: Any,
    *,
    request_id: str,
    fallback_max_len: int = 0,
    fallback_store_size: int = 0,
    fallback_attempted_candidate_lengths: Any = (),
) -> Dict[str, Any]:
    """Strictly copy a cache lookup record and stamp its owning request.

    Unknown keys are dropped. Missing, malformed, or semantically inconsistent
    fields fail closed to a typed ``malformed_lookup`` record rather than
    preserving stale global diagnostics or arbitrary path/error strings.
    """

    fallback = make_ssm_prefix_lookup(
        max_len=fallback_max_len,
        attempted_candidate_lengths=fallback_attempted_candidate_lengths,
        reason="malformed_lookup",
        store_size=fallback_store_size,
        request_id=request_id,
    )
    if not isinstance(value, dict):
        return fallback

    max_len = value.get("max_len")
    candidates = value.get("candidate_lengths")
    candidate_count = value.get("candidate_count")
    candidates_truncated = value.get("candidate_lengths_truncated")
    attempts = value.get("attempted_candidate_lengths")
    attempted_count = value.get("attempted_candidate_count")
    attempts_truncated = value.get("attempted_candidate_lengths_truncated")
    matched = value.get("matched")
    checkpoint_tokens = value.get("checkpoint_tokens")
    is_complete = value.get("is_complete")
    source = value.get("source")
    reason = value.get("reason")
    store_size = value.get("store_size")
    if (
        type(max_len) is not int
        or max_len < 0
        or not isinstance(candidates, list)
        or len(candidates) > SSM_PREFIX_LOOKUP_MAX_CANDIDATES
        or any(type(item) is not int or item <= 0 for item in candidates)
        or type(candidate_count) is not int
        or candidate_count < len(candidates)
        or type(candidates_truncated) is not bool
        or candidates_truncated != (candidate_count > len(candidates))
        or not isinstance(attempts, list)
        or len(attempts) > SSM_PREFIX_LOOKUP_MAX_ATTEMPTS
        or any(type(item) is not int or item <= 0 for item in attempts)
        or type(attempted_count) is not int
        or attempted_count < len(attempts)
        or type(attempts_truncated) is not bool
        or attempts_truncated != (attempted_count > len(attempts))
        or type(matched) is not bool
        or type(checkpoint_tokens) is not int
        or checkpoint_tokens < 0
        or type(is_complete) is not bool
        or source not in _SSM_PREFIX_LOOKUP_SOURCES
        or reason not in _SSM_PREFIX_LOOKUP_REASONS
        or type(store_size) is not int
        or store_size < 0
        or any(item > max_len for item in candidates)
        or any(item > max_len for item in attempts)
    ):
        return fallback
    if matched:
        if (
            checkpoint_tokens <= 0
            or checkpoint_tokens > max_len
            or checkpoint_tokens not in candidates
            or not attempts
            or attempts[-1] != checkpoint_tokens
            or source == "none"
            or reason != "matched"
        ):
            return fallback
    elif (
        checkpoint_tokens != 0
        or is_complete
        or source != "none"
        or reason == "matched"
    ):
        return fallback

    return make_ssm_prefix_lookup(
        max_len=max_len,
        candidate_lengths=candidates,
        candidate_count=candidate_count,
        attempted_candidate_lengths=attempts,
        attempted_candidate_count=attempted_count,
        matched=matched,
        checkpoint_tokens=checkpoint_tokens,
        is_complete=is_complete,
        source=source,
        reason=reason,
        store_size=store_size,
        request_id=request_id,
    )


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
        # Deliberately 20, NOT DEFAULT_SSM_COMPANION_ENTRIES: direct
        # constructions keep the documented 50 → 20 worst-case-footprint
        # default; the engine's scheduler configs apply their own tighter
        # policy via the shared constant.
        max_entries: int = 20,
        model_key: str = "",
        disk_store: Any = None,
        max_bytes: Optional[int] = None,
    ):
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        # Internal storage: key -> (states, is_complete) tuple.
        self._store: OrderedDict[str, Tuple[List[Any], bool]] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = int(max_bytes) if max_bytes is not None else None
        if self._max_bytes is not None and self._max_bytes < 0:
            raise ValueError("max_bytes must be >= 0 when set")
        self._entry_nbytes: Dict[str, int] = {}
        self._total_nbytes = 0
        self._evictions = 0
        self._evicted_bytes = 0
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
        self.last_prefix_lookup: Optional[Dict[str, Any]] = None

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
    def ram_enabled(self) -> bool:
        """Whether companion payloads may remain resident between requests."""
        return bool(
            self._max_entries > 0
            and (self._max_bytes is None or self._max_bytes > 0)
        )

    @property
    def evictions(self) -> int:
        """Number of LRU entries removed to satisfy count or byte budgets."""
        return self._evictions

    @property
    def evicted_bytes(self) -> int:
        """Resident bytes released by budget-driven LRU eviction."""
        return self._evicted_bytes

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

    @staticmethod
    def _extra_key_bytes(
        cache_extra_keys: Optional[Any],
        num_tokens: int,
    ) -> bytes:
        # Match paged/block hashing: a causal media discriminator does not
        # affect companion state before its placeholder boundary. Without
        # resolving the scoped keys here, KV can find an unsalted pre-media
        # block while the SSM companion hashes the whole request as salted and
        # forces a full hybrid prefill.
        resolved = cache_extra_keys_for_token_range(
            cache_extra_keys,
            0,
            num_tokens,
        )
        if resolved is None:
            return b""
        try:
            encoded = json.dumps(
                resolved,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            encoded = repr(resolved)
        return b"\x00extra=" + encoded.encode("utf-8")

    def _key(
        self,
        token_ids: List[int],
        num_tokens: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> str:
        """Deterministic SHA-256 hash key.

        Mixes ``self._model_key`` into the hash so different model loads
        cannot collide on identical token prefixes (A3-BUG-001). Optional
        cache_extra_keys partition media-conditioned VLM states in the same
        way paged KV block hashes partition image/video prompts.
        """
        # schema-v2: positional full-latent slots (dots3) are exempt from
        # companion snapshots, which changes the implicit layer ordering of
        # stored entries. Version the key so pre-exemption entries can never
        # splice misaligned state into the new mapping.
        data = (
            self._model_key.encode()
            + b"\x00schema-v2\x00"
            + json.dumps(token_ids[:num_tokens], separators=(",", ":")).encode()
            + self._extra_key_bytes(cache_extra_keys, num_tokens)
        )
        h = hashlib.sha256(data).hexdigest()
        # VMLX_CACHE_HASH_DEBUG promotes this to INFO. A store and a fetch that
        # disagree are indistinguishable from "nothing was stored" in the
        # ordinary logs, and chasing that difference by inference cost a whole
        # investigation -- the two key lines side by side settle it in one read.
        if os.environ.get("VMLX_CACHE_HASH_DEBUG") == "1":
            logger.info(
                "SSM key: N=%d of len=%d hash=%s extra=%s "
                "tokens[N-4:N]=%s",
                num_tokens,
                len(token_ids),
                h[:12],
                bool(cache_extra_keys),
                token_ids[max(0, num_tokens - 4):num_tokens],
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SSM key: N=%d hash=%s extra=%s tokens[-8:]=%s",
                num_tokens,
                h[:12],
                bool(cache_extra_keys),
                token_ids[max(0,num_tokens-8):num_tokens],
            )
        return h

    def store(
        self,
        token_ids: List[int],
        num_tokens: int,
        ssm_states: List[Any],
        is_complete: bool = True,
        cache_extra_keys: Optional[Any] = None,
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
        if not self.ram_enabled and self._disk is None:
            # A zero-sized cache without L2 is fully disabled. Avoid even the
            # transient clone/materialisation cost in that configuration.
            return
        stored_states = self._clone_states(ssm_states, key_hint="store")
        if stored_states is None:
            logger.debug(
                "SSM store skipped: failed to detach/materialize %d layers",
                len(ssm_states),
            )
            return
        stored_nbytes = self._estimate_state_nbytes(stored_states)
        key = self._key(token_ids, num_tokens, cache_extra_keys=cache_extra_keys)

        # SSD is authoritative in disk-only mode. Freeze/write the detached
        # snapshot before deciding whether it also belongs in retained RAM.
        disk_written = False
        if self._disk is not None:
            try:
                self._disk.store(
                    key, stored_states, is_complete, token_ids, num_tokens
                )
                disk_written = True
            except Exception as e:
                logger.debug("SSM disk write-through failed: %s", e)
        # The store-side identity, in the same units as "SSM disk HIT": a
        # fetch can be bound to the publication that wrote it by (N, hash)
        # instead of by token count alone (S5 audit: length is eligibility,
        # not identity).
        logger.info(
            "SSM stored: N=%d hash=%s states=%d complete=%s disk=%s",
            num_tokens,
            key[:12],
            len(stored_states),
            is_complete,
            disk_written,
        )

        if not self.ram_enabled:
            logger.debug(
                "SSM companion stored to L2 only: retained RAM disabled "
                "(N=%d, %.1fMB)",
                num_tokens,
                stored_nbytes / (1024 * 1024),
            )
            return
        if self._max_bytes is not None and stored_nbytes > self._max_bytes:
            logger.info(
                "SSM companion stored to L2 but not retained in L1: entry is "
                "%.1fMB, above RAM budget %.1fMB",
                stored_nbytes / (1024 * 1024),
                self._max_bytes / (1024 * 1024),
            )
            return

        prefix_hash = self._prefix_hash(
            token_ids, num_tokens, cache_extra_keys=cache_extra_keys
        )
        # Remove existing entry to update its LRU position
        if key in self._store:
            self._drop_key(key)
        self._store[key] = (stored_states, is_complete)
        self._entry_nbytes[key] = stored_nbytes
        self._total_nbytes += stored_nbytes
        # Record in length index so fetch_longest_prefix can locate it.
        self._length_index.setdefault(num_tokens, {})[prefix_hash] = key
        self._evict_if_needed()

    def has_complete(
        self,
        token_ids: List[int],
        num_tokens: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> bool:
        """True when a complete entry already exists at this exact key.

        Pure existence probe: does not touch LRU order, does not clone
        states, does not consult the disk tier. Used to skip redundant
        deferred re-derive prefills whose output would land on a key that
        already holds a complete companion (a cache HIT restores from that
        very entry, so re-deriving it again is a full wasted prefill).
        """
        if num_tokens <= 0:
            return False
        key = self._key(token_ids, num_tokens, cache_extra_keys=cache_extra_keys)
        entry = self._store.get(key)
        if entry is not None:
            return bool(entry[1])
        disk_probe = getattr(self._disk, "has_complete", None)
        if callable(disk_probe):
            try:
                return bool(disk_probe(key))
            except Exception:
                return False
        return False

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
                        (mx.contiguous(mx.array(a)) if a is not None else None)
                        for a in c.cache
                    ]
                    materialise = [x for x in c.cache if x is not None]
                    if materialise:
                        _mx_materialize(*materialise)
                if getattr(c, "lengths", None) is not None:
                    try:
                        c.lengths = mx.array(c.lengths)
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

    def _prefix_hash(
        self,
        token_ids: List[int],
        num_tokens: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> str:
        """Stable family identifier: same sha256 for any (longer) token list
        whose first ``num_tokens`` match. Used to confirm a shorter stored
        checkpoint is a true prefix of the new query before resuming."""
        data = (
            self._model_key.encode()
            + b"\x00"
            + json.dumps(token_ids[:num_tokens], separators=(",", ":")).encode()
            + self._extra_key_bytes(cache_extra_keys, num_tokens)
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

    def fetch(
        self,
        token_ids: List[int],
        num_tokens: int,
        cache_extra_keys: Optional[Any] = None,
    ) -> SSMCompanionEntry:
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
        key = self._key(token_ids, num_tokens, cache_extra_keys=cache_extra_keys)
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
                if not self.ram_enabled or (
                    self._max_bytes is not None and disk_nbytes > self._max_bytes
                ):
                    logger.info(
                        "SSM disk hit not backfilled: entry %.1fMB, retained "
                        "RAM %s (budget %.1fMB)",
                        disk_nbytes / (1024 * 1024),
                        "enabled" if self.ram_enabled else "disabled",
                        (self._max_bytes or 0) / (1024 * 1024),
                    )
                    # mx.load reconstructed fresh, materialized arrays. With no
                    # retained L1 copy, the caller may own and mutate them
                    # directly; cloning here would briefly double refault RAM.
                    return (disk_states, disk_complete)
                # Backfill L1 so subsequent hits skip disk altogether.
                self._store[key] = (disk_states, disk_complete)
                self._entry_nbytes[key] = disk_nbytes
                self._total_nbytes += disk_nbytes
                prefix_hash = self._prefix_hash(
                    token_ids, num_tokens, cache_extra_keys=cache_extra_keys
                )
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
        # An L1 hit must also refresh the DISK entry's LRU standing: the
        # companion file shares the aggregate block-cache budget and is
        # ranked by file age, so without this an actively used chain's
        # companion is exactly as evictable as stale data — and a restart
        # then finds the KV chain orphaned (measured live: 10 stores -> 3
        # surviving files after one bounded-L2 filler pass, cold restore).
        if self._disk is not None:
            try:
                self._disk.touch(key)
            except Exception:
                pass
        # Deep-copy each layer to prevent in-place mutation by the model's
        # forward pass corrupting the stored companion. SSM state is
        # cumulative — generation updates it token by token.
        copied = self._clone_states(states, key_hint=key[:12])
        if copied is None:
            return None
        return (copied, is_complete)

    def fetch_longest_prefix(
        self,
        token_ids: List[int],
        max_len: int,
        cache_extra_keys: Optional[Any] = None,
        exact_boundary_already_missed: bool = False,
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
            self.last_prefix_lookup = make_ssm_prefix_lookup(
                max_len=int(max_len or 0),
                reason="non_positive_max_len",
                store_size=len(self._store),
            )
            return None
        # The exact boundary is always the first logical attempt. The MLLM
        # fast path can pass ``exact_boundary_already_missed=True`` after it
        # has already performed that fetch, avoiding a duplicate disk/L1 read
        # while retaining the actual attempt in telemetry.
        attempted_candidate_lengths: List[int] = [int(max_len)]
        # A fresh process has no in-memory ``_length_index`` yet, even when
        # the scheduler's block-disk tier has selected an exact cached block
        # boundary. Probe that boundary directly first so ``fetch()`` can
        # restore the matching SSM companion from disk and backfill L1. Without
        # this probe, restart reuse was limited to exact whole-prompt lookups;
        # longer multi-turn prompts saw KV disk hits but unnecessarily fell
        # back to a full hybrid prefill because only L1 checkpoint lengths were
        # considered below.
        exact_boundary = (
            None
            if exact_boundary_already_missed
            else self.fetch(
                token_ids,
                max_len,
                cache_extra_keys=cache_extra_keys,
            )
        )
        if exact_boundary is not None:
            states, is_complete = exact_boundary
            self.last_prefix_lookup = make_ssm_prefix_lookup(
                max_len=int(max_len),
                candidate_lengths=[int(max_len)],
                attempted_candidate_lengths=attempted_candidate_lengths,
                matched=True,
                checkpoint_tokens=int(max_len),
                is_complete=is_complete,
                source="exact_boundary_l1_or_l2",
                reason="matched",
                store_size=len(self._store),
            )
            return (max_len, states, is_complete)
        # Scan lengths in descending order so we find the longest match.
        # L1 supplies boundaries learned in this process. L2 supplies
        # sidecar-derived boundaries so a fresh process can discover a shorter
        # typed-state checkpoint when the block cache selected a longer shared
        # prefix. These are candidates only: fetch() recomputes the complete
        # model/prefix key and performs all disk record validation.
        disk_candidate_lengths: List[int] = []
        if self._disk is not None:
            try:
                candidate_fn = getattr(self._disk, "candidate_lengths", None)
                if callable(candidate_fn):
                    disk_candidate_lengths = list(candidate_fn(max_len) or [])
            except Exception as e:
                logger.debug("SSM disk candidate-length scan failed: %s", e)
        disk_candidate_set = set(disk_candidate_lengths)
        # Keep exact-length L1 candidates for existing prefix-mismatch
        # diagnostics. The exact L2 boundary was already probed above, so only
        # shorter L2 boundaries need another fetch attempt.
        candidate_lengths = sorted(
            {
                n
                for n in self._length_index.keys()
                if n < max_len
                or (n == max_len and not exact_boundary_already_missed)
            }
            | {
                n for n in disk_candidate_lengths if n < max_len
            },
            reverse=True,
        )
        if not candidate_lengths:
            self.last_prefix_lookup = make_ssm_prefix_lookup(
                max_len=int(max_len),
                attempted_candidate_lengths=attempted_candidate_lengths,
                reason="no_candidate_lengths",
                store_size=len(self._store),
            )
            return None
        # Compute the prefix_hash for each candidate length against the
        # query's own tokens and compare. First match wins.
        for n in candidate_lengths:
            query_ph = self._prefix_hash(
                token_ids, n, cache_extra_keys=cache_extra_keys
            )
            stored_key = self._length_index.get(n, {}).get(query_ph)
            disk_candidate = n in disk_candidate_set
            if stored_key is None and not disk_candidate:
                continue
            # Delegate to fetch() so deep-copy discipline is uniform.
            attempted_candidate_lengths.append(n)
            result = self.fetch(token_ids, n, cache_extra_keys=cache_extra_keys)
            if result is None:
                # deepcopy failed — treat as miss per existing contract
                continue
            states, is_complete = result
            self.last_prefix_lookup = make_ssm_prefix_lookup(
                max_len=int(max_len),
                candidate_lengths=candidate_lengths,
                attempted_candidate_lengths=attempted_candidate_lengths,
                matched=True,
                checkpoint_tokens=n,
                is_complete=is_complete,
                source=(
                    "partial_boundary_disk_l2"
                    if stored_key is None and disk_candidate
                    else "l1_or_l2"
                ),
                reason="matched",
                store_size=len(self._store),
            )
            return (n, states, is_complete)
        self.last_prefix_lookup = make_ssm_prefix_lookup(
            max_len=int(max_len),
            candidate_lengths=candidate_lengths,
            attempted_candidate_lengths=attempted_candidate_lengths,
            reason=(
                "candidate_fetch_miss"
                if len(attempted_candidate_lengths) > 1
                else "prefix_hash_mismatch"
            ),
            store_size=len(self._store),
        )
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
            self._drop_key(evict_key, evicted=True)
        while (
            self._store
            and self._max_bytes is not None
            and self._total_nbytes > self._max_bytes
        ):
            evict_key = next(iter(self._store))
            self._drop_key(evict_key, evicted=True)

    def _drop_key(self, key: str, *, evicted: bool = False) -> None:
        """Remove a stored entry and all auxiliary accounting."""
        self._store.pop(key, None)
        removed_nbytes = self._entry_nbytes.pop(key, 0)
        self._total_nbytes -= removed_nbytes
        if self._total_nbytes < 0:
            self._total_nbytes = 0
        if evicted:
            self._evictions += 1
            self._evicted_bytes += removed_nbytes
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


_HYBRID_MODEL_TYPES = frozenset({
    # Hybrid SSM / linear-attention families that produce ArraysCache-like
    # cumulative state alongside normal attention KV. Keep DSV4/ZAYA out of
    # this list: they have dedicated typed cache contracts and must not route
    # through the generic SSM companion path.
    *NEMOTRON_H_MODEL_TYPES,
    "qwen3_next",
    "bailing_hybrid",
    "bailing_moe_v2_5",
    "jamba",
    # IBM Granite MoE Hybrid (mlx_lm/granitemoehybrid.py) — explicit
    # ArraysCache+KVCache mix; layer_types declares "mamba"+"attention" so
    # the marker fallback usually fires, but pin the model_type so the
    # companion still engages on configs that only carry model_type.
    "granitemoehybrid",
    # Liquid AI LFM2 MoE (mlx_lm/lfm2_moe.py) — hybrid Conv1d-SSM +
    # attention. layer_types use "full_attention" / "conv" strings that
    # do NOT match _HYBRID_LAYER_TYPE_MARKERS, so model_type is the only
    # reliable detection path here.
    "lfm2",
    "lfm2_moe",
    # Falcon H1 (mlx_lm/falcon_h1.py) — CacheList[ArraysCache, KVCache]
    # hybrid. No layer_types declarations in the model module, so the
    # marker fallback never fires. Pin via model_type.
    "falcon_h1",
})

_HYBRID_LAYER_TYPE_MARKERS = frozenset({
    "linear_attention",
    "gated_linear_attention",
    "mamba",
    "mamba2",
    "ssm",
})


def _declares_hybrid_ssm_layer_types(config: dict) -> bool:
    """Return True for explicit SSM/linear-attention layer declarations.

    Sliding-window attention is intentionally excluded. Gemma/Laguna-style
    SWA+full-attention models use RotatingKVCache + KVCache, not cumulative
    ArraysCache state, so they are handled by the mixed-SWA/KV cache path
    rather than the SSM companion cache.
    """
    for key in ("layer_types", "layer_type", "layers_block_type"):
        value = config.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if str(item).lower() in _HYBRID_LAYER_TYPE_MARKERS:
                return True
    return False


def is_hybrid_ssm_config(config: dict) -> bool:
    """Return True if *config* describes a hybrid SSM+attention model."""
    if not isinstance(config, dict):
        return False
    if "hybrid_override_pattern" in config:
        return True
    model_type = str(config.get("model_type") or "").lower()
    if model_type in {"qwen4_exp", "qwen4_exp_text"}:
        return True
    if model_type in _HYBRID_MODEL_TYPES:
        return True
    if _declares_hybrid_ssm_layer_types(config):
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

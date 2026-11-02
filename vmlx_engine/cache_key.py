# SPDX-License-Identifier: Apache-2.0
"""Stable discriminators shared by every prompt-cache backend."""

from __future__ import annotations

import json
from typing import Any, Optional


# Reserved metadata that makes one cache discriminator begin at a causal token
# boundary instead of poisoning the root of the whole block chain.  The value
# maps an ordinary key in the same dict to the first token whose model state
# depends on that key.  This is primarily used for multimodal payload identity:
# text before the first media placeholder is independent of the pixels and must
# dedupe normally, while the placeholder block and every descendant must remain
# isolated by the exact media bytes.
CACHE_EXTRA_SCOPES_KEY = "__vmlx_cache_extra_scopes_v1__"


def scope_cache_extra_key(
    cache_extra_keys: Optional[Any],
    key: str,
    start_token: int,
) -> dict[str, Any]:
    """Return a copy with ``key`` activated from ``start_token`` onward."""
    if isinstance(cache_extra_keys, dict):
        scoped = dict(cache_extra_keys)
    elif cache_extra_keys is None:
        scoped = {}
    else:
        scoped = {"request": repr(cache_extra_keys)}
    scopes = scoped.get(CACHE_EXTRA_SCOPES_KEY)
    scopes = dict(scopes) if isinstance(scopes, dict) else {}
    scopes[str(key)] = max(0, int(start_token))
    scoped[CACHE_EXTRA_SCOPES_KEY] = scopes
    return scoped


def cache_extra_keys_for_token_range(
    cache_extra_keys: Optional[Any],
    start_token: int,
    end_token: int,
) -> Optional[Any]:
    """Resolve the cache discriminators that affect one causal token range.

    Unscoped keys retain their historical whole-chain behavior.  A scoped key
    is absent from blocks ending at or before its boundary, then becomes part
    of the hash for the block containing the boundary token and every block
    after it.  The active boundary metadata is itself hashed so two otherwise
    identical requests cannot alias when an embedding is injected at a
    different position.
    """
    if not isinstance(cache_extra_keys, dict):
        return cache_extra_keys
    scopes = cache_extra_keys.get(CACHE_EXTRA_SCOPES_KEY)
    if not isinstance(scopes, dict) or not scopes:
        return cache_extra_keys

    resolved: dict[str, Any] = {}
    active_scopes: dict[str, int] = {}
    range_end = max(int(start_token), int(end_token))
    for key, value in cache_extra_keys.items():
        if key == CACHE_EXTRA_SCOPES_KEY:
            continue
        raw_boundary = scopes.get(str(key))
        if raw_boundary is None:
            resolved[key] = value
            continue
        try:
            boundary = max(0, int(raw_boundary))
        except (TypeError, ValueError):
            # Malformed scope metadata must fail closed: keep the discriminator
            # global rather than risk replaying state from a different payload.
            resolved[key] = value
            continue
        if range_end > boundary:
            resolved[key] = value
            active_scopes[str(key)] = boundary
    if active_scopes:
        resolved[CACHE_EXTRA_SCOPES_KEY] = active_scopes
    return resolved or None


def canonical_cache_extra_marker(cache_extra_keys: Optional[Any]) -> Optional[str]:
    """Return a stable, comparable marker for non-token cache key inputs."""
    if cache_extra_keys is None:
        return None
    try:
        return json.dumps(
            cache_extra_keys,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(cache_extra_keys)


# One key per image, active from that image's first token. Rows before a new image keep their keys.
MEDIA_ITEM_PREFIX = "media_item_"


def merge_cache_extra_keys(base: Optional[Any], extra: dict) -> dict:
    """``base`` plus ``extra``, keeping the scopes of both."""
    if isinstance(base, dict):
        merged = dict(base)
    elif base is None:
        merged = {}
    else:
        merged = {"request": repr(base)}
    scopes = {**(merged.get(CACHE_EXTRA_SCOPES_KEY) or {}), **(extra.get(CACHE_EXTRA_SCOPES_KEY) or {})}
    merged.update(extra)
    if scopes:
        merged[CACHE_EXTRA_SCOPES_KEY] = scopes
    return merged


def per_item_key_fallbacks(cache_extra_keys: Optional[Any]) -> list[dict]:
    """The keys of the same conversation before its newest image, then before the one before it, and so on.

    A whole-entry tier (RAM) compares a stored entry's complete key, so a turn that added an image has to look its
    earlier turns up under the key those turns had."""
    if not isinstance(cache_extra_keys, dict):
        return []
    scopes = cache_extra_keys.get(CACHE_EXTRA_SCOPES_KEY)
    if not isinstance(scopes, dict):
        return []
    starts = sorted({int(v) for k, v in scopes.items() if str(k).startswith(MEDIA_ITEM_PREFIX)}, reverse=True)
    return [cache_extra_keys_for_token_range(cache_extra_keys, 0, start) for start in starts]

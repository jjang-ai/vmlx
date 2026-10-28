# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 prefill geometry: where a prompt prefill is cut, and which cached rows a later request may reuse.

Why this exists. MLX evaluates a prompt chunk as one matrix; the same token row comes out a few ulps different
inside chunks of different lengths, and M3's MSA indexer can turn that into a different sparse block pick and a
different greedy answer (measured on a synthetic M3: one prompt prefilled whole vs in 320-token chunks differs by
0.09 in the last logits). A cached row is therefore only interchangeable with a cold prefill's row when both were
computed inside a chunk of the same absolute geometry.

The rule (one for every cache tier, cold prefill and continuation alike):

* every M3 prompt prefill is cut at absolute positions: every multiple of the prefill step ``step`` and at the
  request's store boundary ``boundary`` (the prefix cache stores ``boundary`` rows: the generation-prompt-stripped
  prompt minus its final token, which a hit re-feeds for the first-token logits). A continuation after a restore at
  ``r`` therefore runs exactly the chunks a cold prefill of the same prompt runs from ``r`` on;
* rows inside a full aligned chunk ``[j*step, (j+1)*step)`` depend only on the tokens up to the chunk end, so any
  longer prompt with the same prefix may reuse them: reuse stops at a chunk boundary;
* rows of the partial chunk that ends at ``boundary`` were computed in a chunk only a request with the SAME boundary
  runs, so they are reusable only by an exact-geometry request (reuse == its own boundary);
* a media request (M3 VL) prefills ``[0, boundary)`` atomically (every placeholder must meet its vision tensor in one
  forward), then ``[boundary, N)``; it can only reuse an exact-geometry boundary.

Rows of one same-shape chunk were also measured to be independent of the other rows' content (MoE sorted gather,
MSA selection and attention kernels), but that does not let a hit stop mid-chunk: the continuation would then run
the rest of that chunk as a shorter matrix than the cold prefill does.

``VMLX_M3_ALIGNED_PREFIX=0`` restores the previous behaviour (prefill chunked relative to the restored offset,
blocks keyed by total prompt length) for comparisons.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

SCHEMA = "minimax_m3_aligned_prefill_v1"
ENV = "VMLX_M3_ALIGNED_PREFIX"

# Cache-extra-key names (top level of the request's extra-key dict, so the per-token scope resolver sees them).
GEOMETRY_KEY = "__vmlx_m3_prefill_geometry__"
TRAILING_KEY = "__vmlx_m3_trailing_chunk__"
MEDIA_KEY = "mllm_media"


def aligned_prefix_enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


def prefill_pieces(start: int, end: int, step: int, boundary: Optional[int] = None,
                   atomic: bool = False) -> List[Tuple[int, int]]:
    """Absolute ``[a, b)`` pieces that prefill ``[start, end)``: cut at every multiple of ``step`` (not for an atomic
    media prefill) and at ``boundary``, when they fall strictly inside."""
    start, end = int(start), int(end)
    if end <= start:
        return []
    cuts = {start, end}
    if boundary is not None and start < int(boundary) < end:
        cuts.add(int(boundary))
    if not atomic:
        step = max(1, int(step))
        cuts.update(range((start // step + 1) * step, end, step))
    ordered = sorted(cuts)
    return list(zip(ordered[:-1], ordered[1:]))


def trailing_chunk_start(boundary: int, step: int) -> int:
    """Start of the chunk that ends at ``boundary`` (== ``boundary`` when it is itself a chunk boundary)."""
    step = max(1, int(step))
    boundary = max(0, int(boundary))
    return (boundary // step) * step


def reuse_limit(matched: int, boundary: int, step: int, *, exact_geometry: bool, atomic: bool = False,
                granule: int = 1) -> int:
    """The longest restored prefix a request whose own store boundary is ``boundary`` may use out of ``matched``
    restored rows: its boundary when the rows were produced with that same boundary (``exact_geometry``), else the
    largest chunk boundary <= min(matched, boundary) (0 for an atomic media request). ``granule`` rounds the chunk
    boundary down further (block-granular tiers that cannot cut inside a block; lcm with the step)."""
    matched, boundary = int(matched), int(boundary)
    if matched <= 0 or boundary <= 0:
        return 0
    if exact_geometry and matched == boundary:
        return boundary
    if atomic:
        return 0
    unit = max(1, int(step))
    if granule and int(granule) > 1:
        unit = unit * int(granule) // math.gcd(unit, int(granule))
    return (min(matched, boundary) // unit) * unit


def block_cover(tokens: int, block: int) -> int:
    """Smallest multiple of ``block`` >= ``tokens`` (the paged blocks that hold ``tokens`` rows)."""
    block = max(1, int(block))
    return -(-int(tokens) // block) * block


def is_media_request(cache_extra_keys) -> bool:
    return isinstance(cache_extra_keys, dict) and MEDIA_KEY in cache_extra_keys


def cache_extra_keys(cache_extra_keys, *, boundary: int, step: int):
    """The request's cache extra keys plus the M3 geometry of the rows a prefill with this boundary produces.

    All blocks carry the step (rows from a different step were computed in different chunks). Text requests add a
    trailing-chunk key scoped from the start of the partial chunk that ends at ``boundary`` (absent when the boundary
    is itself a chunk boundary), so only an exact-geometry request matches those blocks; blocks of full chunks stay
    shareable by any longer prompt. Media requests are keyed by their boundary from block 0 (atomic prefill)."""
    from ...cache_key import CACHE_EXTRA_SCOPES_KEY

    boundary = max(0, int(boundary))
    step = max(1, int(step))
    if isinstance(cache_extra_keys, dict):
        out = dict(cache_extra_keys)
    elif cache_extra_keys is None:
        out = {}
    else:
        out = {"__vmlx_request_extra_keys__": cache_extra_keys}
    if is_media_request(cache_extra_keys):
        out[GEOMETRY_KEY] = {"schema": SCHEMA, "atomic_boundary": boundary}
        return out
    out[GEOMETRY_KEY] = {"schema": SCHEMA, "step": step}
    start = trailing_chunk_start(boundary, step)
    if start < boundary:
        out[TRAILING_KEY] = {"boundary": boundary}
        scopes = out.get(CACHE_EXTRA_SCOPES_KEY)
        scopes = dict(scopes) if isinstance(scopes, dict) else {}
        scopes[TRAILING_KEY] = start
        out[CACHE_EXTRA_SCOPES_KEY] = scopes
    return out


def truncated_cache(cache, length: int):
    """New cache objects holding the first ``length`` rows of an M3 cache list (sparse layers keep their index lane;
    dense layers are stock KVCache). The source objects are not modified (they may be shared with a cache tier)."""
    from mlx_lm.models.cache import KVCache

    from .cache import clone_minimax_m3_sparse

    length = int(length)
    out = []
    for layer in cache:
        if type(layer).__name__ == "MiniMaxM3SparseCache":
            new = clone_minimax_m3_sparse(layer, length)
            if new is None or int(new.offset) != length:
                return None
            out.append(new)
            continue
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if not isinstance(layer, KVCache) or keys is None or values is None or int(layer.offset) < length:
            return None
        fresh = KVCache()
        fresh.state = (keys[..., :length, :], values[..., :length, :])
        out.append(fresh)
    return out


def cache_offset(cache) -> Optional[int]:
    """The common logical length of an M3 cache list (None when layers disagree or a lane is out of step)."""
    offsets = set()
    for layer in cache or ():
        offset = getattr(layer, "offset", None)
        if offset is None:
            return None
        try:
            offsets.add(int(offset))
        except (TypeError, ValueError):
            return None
        if type(layer).__name__ == "MiniMaxM3SparseCache" and getattr(layer, "idx_keys", None) is not None:
            if int(getattr(layer, "_idx_offset", -1)) != int(offset):
                return None
    return offsets.pop() if len(offsets) == 1 else None

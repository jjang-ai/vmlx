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

Turn cuts (structural, ``minimax_m3_aligned_prefill_v2_turns``; default on). The chunk rule alone reuses a tool
loop only up to ``floor(K_prev / step) * step``: up to ``step - 1`` tokens of the previous conversation re-prefilled
per tool call, and nothing while it is shorter than one step. So a text prefill is also cut at every TURN CUT: the
position right after an end-of-message token (``[e~[``) that closes a message other than an assistant turn
(system, developer, user, tool). In an M3 chat prompt every message ends ``[e~[`` ``\n`` and the next one starts
``]~b]``; the store boundary ``K`` of a request is the ``\n`` before its generation prompt's ``]~b]``, i.e. the turn
cut of the user / tool message that precedes it. The next request renders that same ``]~b]`` as the start of the
previous assistant turn, so the previous ``K`` is a turn cut of the next prompt, and:

* the cut set of a prompt ``X`` is ``{multiples of step} | turn_cuts(X) | {K}``;
* a turn cut ``c`` is decided by ``X[c - 1]`` and the role of the message it closes (tokens before ``c``): the cut
  set below ``r`` is a function of ``X[0, r)`` alone. A prefix cached by ANY request with the same first ``r``
  tokens was therefore computed in exactly the chunks a cold prefill of this request runs below ``r`` when ``r`` is
  one of this request's cuts (and ``r`` <= the producer's own boundary, which every tier guarantees: a tier stores at
  most ``K`` prompt rows). Reuse is the longest restored prefix that ends at a multiple of the step or a turn cut of
  this request, or at its own boundary when the rows were produced with that boundary (exact geometry);
* a cut detected by the tokens AFTER it (``]~b]ai\n`` starting at ``c + 1``) would not have that property: a
  producer and a consumer that agree on ``[0, r)`` but differ right after ``r`` place different cuts at or below
  ``r``, and nothing a cache stores could tell. Hence the lookbehind form; on a rendered chat it is the same position;
* an assistant turn's end is not a cut: a tool loop's incremental prefill ``[K_prev, K_new)`` is one assistant turn
  plus the tool output, and a cut between them would cost one more forward on every tool call for no loop reuse;
* the block key of a text prompt carries the step and the turn markers; the partial chunk ending at ``K`` carries
  ``K`` only when ``K`` is not itself a step multiple or a turn cut (then only an exact-geometry request matches it).

``VMLX_M3_STRUCTURAL_CUTS=0`` keeps the chunk-only rule (``minimax_m3_aligned_prefill_v1``) for A/B.
``VMLX_M3_ALIGNED_PREFIX=0`` restores the previous behaviour (prefill chunked relative to the restored offset,
blocks keyed by total prompt length) for comparisons.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "minimax_m3_aligned_prefill_v1"
SCHEMA_TURNS = "minimax_m3_aligned_prefill_v2_turns"
ENV = "VMLX_M3_ALIGNED_PREFIX"
STRUCTURAL_ENV = "VMLX_M3_STRUCTURAL_CUTS"

# The M3 chat template's message markers (chat_template.jinja: bos_token / eos_token, assistant role "ai").
M3_BOS_TEXT = "]~b]"
M3_EOS_TEXT = "[e~["
M3_ASSISTANT_ROLE_TEXT = "ai\n"

# Cache-extra-key names (top level of the request's extra-key dict, so the per-token scope resolver sees them).
GEOMETRY_KEY = "__vmlx_m3_prefill_geometry__"
TRAILING_KEY = "__vmlx_m3_trailing_chunk__"
MEDIA_KEY = "mllm_media"


def aligned_prefix_enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


def structural_cuts_enabled() -> bool:
    return os.environ.get(STRUCTURAL_ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class TurnMarkers:
    """Token ids of the M3 message structure: ``bos`` starts a message (followed by its role tokens), ``eos`` ends
    one; ``assistant`` = the role tokens that follow ``bos`` in an assistant turn."""

    bos: int
    eos: int
    assistant: Tuple[int, ...]

    def signature(self) -> dict:
        return {"bos": int(self.bos), "eos": int(self.eos), "assistant": [int(t) for t in self.assistant]}


def _encode(tok: Any, text: str) -> Optional[List[int]]:
    for kwargs in ({"add_special_tokens": False}, {}):
        try:
            ids = tok.encode(text, **kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
        try:
            return [int(t) for t in ids]
        except (TypeError, ValueError):
            return None
    return None


def turn_markers_from_tokenizer(tokenizer: Any) -> Optional[TurnMarkers]:
    """Resolve the M3 message markers through the serving tokenizer (or a processor / mlx_lm wrapper around it).

    Accepted only when the tokenizer encodes the template's joints exactly as the template renders them:
    ``[e~[`` + ``\n`` + ``]~b]`` + ``ai\n`` as ``[eos, <newline ids>, bos, <assistant ids>]`` with single-token
    ``bos`` / ``eos``. Anything else (another family's tokenizer, a test stub) returns None: no turn cuts."""
    seen = set()
    candidates = [tokenizer]
    while candidates:
        tok = candidates.pop(0)
        if tok is None or id(tok) in seen:
            continue
        seen.add(id(tok))
        candidates += [getattr(tok, "_tokenizer", None), getattr(tok, "tokenizer", None)]
        if not callable(getattr(tok, "encode", None)):
            continue
        bos = _encode(tok, M3_BOS_TEXT)
        eos = _encode(tok, M3_EOS_TEXT)
        role = _encode(tok, M3_ASSISTANT_ROLE_TEXT)
        newline = _encode(tok, "\n")
        joint = _encode(tok, M3_EOS_TEXT + "\n" + M3_BOS_TEXT + M3_ASSISTANT_ROLE_TEXT)
        if not (bos and eos and role and newline and joint) or len(bos) != 1 or len(eos) != 1:
            continue
        if bos[0] == eos[0] or joint != eos + newline + bos + role:
            continue
        return TurnMarkers(bos=bos[0], eos=eos[0], assistant=tuple(role))
    return None


def turn_cuts(tokens: Sequence[int], markers: Optional[TurnMarkers], end: Optional[int] = None) -> List[int]:
    """Every turn cut ``c`` in ``(0, end]``: ``tokens[c - 1]`` is ``markers.eos`` and the message it closes (the one
    opened by the last ``markers.bos`` before it) is not an assistant turn. An end-of-message with no open message
    counts as a non-assistant end. Depends only on ``tokens[0, c)`` for each ``c``."""
    if markers is None:
        return []
    n = len(tokens) if end is None else max(0, min(int(end), len(tokens)))
    bos, eos, role = int(markers.bos), int(markers.eos), tuple(int(t) for t in markers.assistant)
    width = len(role)
    cuts: List[int] = []
    assistant_open = False
    for i in range(n):
        t = int(tokens[i])
        if t == bos:
            assistant_open = tuple(int(x) for x in tokens[i + 1: i + 1 + width]) == role
        elif t == eos:
            if not assistant_open:
                cuts.append(i + 1)
            assistant_open = False
    return cuts


def prefill_pieces(start: int, end: int, step: int, boundary: Optional[int] = None,
                   atomic: bool = False, cuts: Optional[Iterable[int]] = None) -> List[Tuple[int, int]]:
    """Absolute ``[a, b)`` pieces that prefill ``[start, end)``: cut at every multiple of ``step`` and at every turn
    cut in ``cuts`` (neither for an atomic media prefill), and at ``boundary``, when they fall strictly inside."""
    start, end = int(start), int(end)
    if end <= start:
        return []
    points = {start, end}
    if boundary is not None and start < int(boundary) < end:
        points.add(int(boundary))
    if not atomic:
        step = max(1, int(step))
        points.update(range((start // step + 1) * step, end, step))
        points.update(int(c) for c in (cuts or ()) if start < int(c) < end)
    ordered = sorted(points)
    return list(zip(ordered[:-1], ordered[1:]))


def plan_pieces(start: int, end: int, step: int, plan: dict) -> List[Tuple[int, int]]:
    """``prefill_pieces`` for a scheduler plan ({"boundary", "atomic", "cuts"})."""
    return prefill_pieces(start, end, step, plan.get("boundary"), atomic=bool(plan.get("atomic")),
                          cuts=plan.get("cuts"))


def trailing_chunk_start(boundary: int, step: int, cuts: Optional[Iterable[int]] = None) -> int:
    """Start of the chunk that ends at ``boundary`` (== ``boundary`` when it is itself a chunk boundary or one of the
    turn ``cuts``)."""
    step = max(1, int(step))
    boundary = max(0, int(boundary))
    start = (boundary // step) * step
    for c in cuts or ():
        c = int(c)
        if start < c <= boundary:
            start = c
    return start


def reuse_limit(matched: int, boundary: int, step: int, *, exact_geometry: bool, atomic: bool = False,
                granule: int = 1, cuts: Optional[Iterable[int]] = None) -> int:
    """The longest restored prefix a request whose own store boundary is ``boundary`` may use out of ``matched``
    restored rows: its boundary when the rows were produced with that same boundary (``exact_geometry``), else the
    largest chunk boundary or turn cut of this request (``cuts``) <= min(matched, boundary) (0 for an atomic media
    request). ``granule`` rounds down further (block-granular tiers that cannot cut inside a block: chunk boundaries
    by the lcm with the step, turn cuts only when they are multiples of it)."""
    matched, boundary = int(matched), int(boundary)
    if matched <= 0 or boundary <= 0:
        return 0
    if exact_geometry and matched == boundary:
        return boundary
    if atomic:
        return 0
    unit = max(1, int(step))
    granule = int(granule or 1)
    if granule > 1:
        unit = unit * granule // math.gcd(unit, granule)
    limit = min(matched, boundary)
    best = (limit // unit) * unit
    for c in cuts or ():
        c = int(c)
        if best < c <= limit and (granule <= 1 or c % granule == 0):
            best = c
    return best


def block_cover(tokens: int, block: int) -> int:
    """Smallest multiple of ``block`` >= ``tokens`` (the paged blocks that hold ``tokens`` rows)."""
    block = max(1, int(block))
    return -(-int(tokens) // block) * block


def is_media_request(cache_extra_keys) -> bool:
    return isinstance(cache_extra_keys, dict) and MEDIA_KEY in cache_extra_keys


def cache_extra_keys(cache_extra_keys, *, boundary: int, step: int, markers: Optional[TurnMarkers] = None,
                     tokens: Optional[Sequence[int]] = None):
    """The request's cache extra keys plus the M3 geometry of the rows a prefill with this boundary produces.

    All blocks carry the step (rows from a different step were computed in different chunks), and with turn
    ``markers`` the turn-cut rule (schema v2: rows of a v1 chunk-only prefill are not interchangeable with these).
    Text requests add a trailing-chunk key scoped from the start of the partial chunk that ends at ``boundary``
    (absent when the boundary is itself a chunk boundary or a turn cut of ``tokens``), so only an exact-geometry
    request matches those blocks; blocks of finished chunks stay shareable by any prompt with the same tokens (their
    cuts are a function of the tokens). Media requests are keyed by their boundary from block 0 (atomic prefill)."""
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
    if markers is None:
        out[GEOMETRY_KEY] = {"schema": SCHEMA, "step": step}
        cuts: List[int] = []
    else:
        out[GEOMETRY_KEY] = {"schema": SCHEMA_TURNS, "step": step, "turns": markers.signature()}
        cuts = turn_cuts(tokens or (), markers, boundary)
    start = trailing_chunk_start(boundary, step, cuts)
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

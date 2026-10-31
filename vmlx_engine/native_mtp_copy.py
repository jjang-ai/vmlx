"""Copy drafts for the native-MTP verify cycle: continuations copied from the request's own context.

Why: on repetitive, quoted, edited or structured text the next tokens are very often a span that already occurs
earlier in the prompt or the reply. Proposing that span as the verify window's drafts lands ~90%+ of the time,
while the MTP head's chained drafts compound their per-depth acceptance, so a matching copy is preferred over
the head's drafts.

How: the proposer keeps the request's confirmed token stream (prompt + emitted + queued) and an incremental
index from every 4-gram to its most recent end positions. A proposal needs the current suffix to match an earlier
occurrence for at least ``min_match`` tokens (default 8; short n-gram matches are mostly coincidence and a wrong
copy displaces the head's own drafts). The continuation after the longest such match is proposed, up to
``max_width`` tokens. Two consecutive cycles whose first copied token was rejected silence the proposer for
``silence_cycles`` proposals.

Exactness: copies change only WHICH tokens fill the verify window. The verify forward, the acceptance rule (an
exact match against the target's own sample for every row, which is distribution-preserving for sampled requests
because every emitted token is the target's sample), the bonus/correction and the rollback are the existing
native-MTP ones. Width starts at 3 (4 verify rows) and grows to 7 (8 rows) after a fully accepted window. Verify windows up to 8 rows are
row-exact: dense QMV in <=4/<=6-row blocks (row_exact_qmv._row_blocks), grouped GDN projections up to 8 rows,
multi-row exact MoE (metal/qwen4_rows_exact_moe), so greedy output stays byte-identical to AR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

NGRAM = 4
_START_WIDTH = 3
_POSITIONS_KEPT = 4
_MAX_EXTEND = 256


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return max(minimum, int(raw)) if raw else default
    except ValueError:
        return default


def copy_drafts_enabled() -> bool:
    return os.environ.get("VMLX_NATIVE_MTP_COPY", "1").strip().lower() not in ("0", "false", "off", "no")


def copy_min_match() -> int:
    return _env_int("VMLX_NATIVE_MTP_COPY_MIN_MATCH", 8, NGRAM)


def copy_max_width() -> int:
    return _env_int("VMLX_NATIVE_MTP_COPY_MAX", 7, 1)


@dataclass
class CopyStats:
    proposals: int = 0
    cycles: int = 0
    drafted: int = 0
    accepted: int = 0
    first_misses: int = 0
    silenced: int = 0


@dataclass
class SuffixCopyProposer:
    """Per-request copy-draft proposer over prompt + confirmed tokens."""

    min_match: int = field(default_factory=copy_min_match)
    max_width: int = field(default_factory=copy_max_width)
    silence_cycles: int = 16
    tokens: List[int] = field(default_factory=list)
    prompt_len: int = 0
    stats: CopyStats = field(default_factory=CopyStats)
    _index: Dict[tuple, List[int]] = field(default_factory=dict)
    _confirmed: int = 0
    width: int = 3
    _miss_streak: int = 0
    _silent_for: int = 0
    last_match: int = 0

    @classmethod
    def from_prompt(cls, prompt_ids: Sequence[int]) -> "SuffixCopyProposer":
        p = cls()
        p.prompt_len = len(prompt_ids)
        p._append([int(t) for t in prompt_ids])
        return p

    def _append(self, new: Sequence[int]) -> None:
        toks = self.tokens
        for t in new:
            toks.append(int(t))
            i = len(toks) - 1
            if i >= NGRAM - 1:
                key = tuple(toks[i - NGRAM + 1 : i + 1])
                slot = self._index.get(key)
                if slot is None:
                    self._index[key] = [i]
                else:
                    slot.append(i)
                    if len(slot) > _POSITIONS_KEPT:
                        del slot[0]

    def sync(self, output_tokens: Sequence[int], queued: Sequence[int]) -> None:
        """Bring the stream up to ``output_tokens + queued`` (confirmed reply so far); rebuild if it shrank."""
        total = len(output_tokens) + len(queued)
        k = self._confirmed
        if total < k:                                   # a terminal rewind dropped confirmed tokens: rebuild
            prompt = self.tokens[: self.prompt_len]
            self.tokens, self._index, self._confirmed = [], {}, 0
            self._append(prompt)
            k = 0
        n_out = len(output_tokens)
        tail = list(output_tokens[k:]) if k < n_out else []
        tail += list(queued[max(0, k - n_out):])
        if tail:
            self._append(tail)
        self._confirmed = total

    def propose(self, room: int) -> List[int]:
        """Up to min(width, room) copied tokens, or [] when no earlier span matches the suffix well enough.

        ``width`` adapts per stream: it starts at
        ``_START_WIDTH`` (the 4-row verify shape), grows to ``max_width`` after a
        fully accepted window and falls back after a window that lands under
        half, so a run of reliable copying pays for wide windows and a shaky
        match costs at most a 4-row verify."""
        self.last_match = 0
        width = min(self.width, self.max_width, int(room))
        toks = self.tokens
        n = len(toks)
        if width <= 0 or n < max(NGRAM, self.min_match) + 1:
            return []
        if self._silent_for > 0:
            self._silent_for -= 1
            return []
        candidates = self._index.get(tuple(toks[n - NGRAM :]))
        if not candidates:
            return []
        best_pos, best_len = -1, 0
        for p in reversed(candidates):
            if p >= n - 1:
                continue                                # the suffix itself
            length = NGRAM
            limit = min(_MAX_EXTEND, p + 1)
            while length < limit and toks[p - length] == toks[n - 1 - length]:
                length += 1
            if length > best_len:
                best_pos, best_len = p, length
        if best_pos < 0 or best_len < self.min_match:
            return []
        out = toks[best_pos + 1 : best_pos + 1 + width]
        if not out:
            return []
        self.last_match = best_len
        self.stats.proposals += 1
        return out

    def observe(self, drafted: int, accepted: int) -> None:
        """Record a verified copy window; two first-token misses in a row silence the proposer for a while."""
        self.stats.cycles += 1
        self.stats.drafted += int(drafted)
        self.stats.accepted += int(accepted)
        if drafted and accepted == drafted:
            self.width = self.max_width
        elif drafted and 2 * accepted < drafted:
            self.width = _START_WIDTH
        if accepted == 0:
            self.stats.first_misses += 1
            self._miss_streak += 1
            if self._miss_streak >= 2:
                self._silent_for = self.silence_cycles
                self._miss_streak = 0
                self.stats.silenced += 1
        else:
            self._miss_streak = 0


def as_dict(stats: Optional[CopyStats]) -> Dict[str, int]:
    if stats is None:
        return {}
    return {"copy_proposals": stats.proposals, "copy_cycles": stats.cycles, "copy_drafted": stats.drafted,
            "copy_accepted": stats.accepted, "copy_first_misses": stats.first_misses, "copy_silenced": stats.silenced}

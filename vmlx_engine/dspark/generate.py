"""DSpark speculative decoding loop (greedy or sample-then-match) and its AR reference.

Each round: the drafter proposes a block from the anchor (the last committed,
not yet forwarded token) and the target's tap features; the target verifies
``[anchor, d_1..d_k]`` in one row-exact forward; drafts are accepted while
they equal the target's own pick at each position; the first mismatch (or the
bonus position) contributes the target's token. Every committed token is the
target's own choice given the committed prefix, so with greedy picks the
output equals AR decoding token for token (verify rows are bit-equal to
decode steps, see ``minicpm_target``). With a sampler each committed token is
a draw from the target distribution ("sample-then-match"); acceptance is then
P_target(draft) instead of 1[argmax = draft].
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence

import mlx.core as mx

from .minicpm_target import MAX_VERIFY_ROWS


@dataclass
class DSparkStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    ar_tokens: int = 0
    tokens: int = 0
    prefill_tokens: int = 0
    cached_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    widths: dict = field(default_factory=dict)

    @property
    def tokens_per_round(self) -> float:
        return (self.tokens - 1 - self.ar_tokens) / self.rounds if self.rounds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted, "ar_tokens": self.ar_tokens,
            "tokens": self.tokens, "tokens_per_round": round(self.tokens_per_round, 3),
            "acceptance": round(self.acceptance, 4), "prefill_s": round(self.prefill_s, 3),
            "decode_s": round(self.decode_s, 3),
            "decode_tok_s": round((self.tokens - 1) / self.decode_s, 2) if self.decode_s > 0 else None,
            "widths": dict(sorted(self.widths.items())),
        }


def choose_width(conf: Sequence[float], max_draft: int, cost_per_row: float = 0.12) -> int:
    """Draft count k maximizing E[tokens | k] / cost(k + 1 rows).

    E[tokens | k] = 1 + sum_j prod_{i<=j} conf_i (confidence head = predicted
    acceptance); cost(T) = 1 + cost_per_row * (T - 1) decode-step units.
    Exactness does not depend on k; only speed does.
    """
    best, best_k, run, expected = 0.0, 1, 1.0, 1.0
    for k in range(1, min(max_draft, len(conf)) + 1):
        run *= float(conf[k - 1])
        expected += run
        value = expected / (1.0 + cost_per_row * k)
        if value > best:
            best, best_k = value, k
    return best_k


class ARFallback:
    """When to stop drafting (the DFlash2 lane's AdaptiveSpec policy, measured costs only).

    A draft round costs the drafter (fixed, ~5.4 ms on the 35B) plus a verify whose sparse-MoE cost grows with width,
    and pays only when enough drafts are accepted: on low-acceptance text (multilingual, thinking at T1.0) DSpark ran
    x0.62-0.86 of AR on the 35B even after the stock-block verify. The loop times every draft round and every AR step
    (synced wall time, EMA); when an AR step's measured rate beats the recent draft-round rate by ``margin`` it runs an
    AR burst of ``burst`` tokens, and every ``probe_every`` AR tokens one draft round re-measures the text. The first
    action is a draft round; the second is one AR step to seed its cost. Exactness is unaffected: AR steps and verify
    rows both equal decode steps; only the schedule changes.
    """

    alpha = 0.3
    margin = 0.05
    burst = 8
    probe_every = 24

    # Width choice from MEASURED round costs (opt-in VMLX_DSPARK_MEASURED_WIDTH=1; default the static choose_width model). The static
    # model prices a round at 1 + 0.12 * (T - 1) decode steps and ignores the drafter; measured on the 35B-A3B (600-token
    # context) a round is drafter 5.4 ms (~0.53 AR step) + verify 11 / 16 / 18-21 / 26 ms at T = 1 / 2 / 4 / 7 rows
    # (~0.5 AR step for the second row, ~0.2 for each later one): the static model makes 7-row verifies look nearly
    # free, so rounds ran wide and slow and the AR fallback took ~80 % of the tokens in sustained multi-turn chat.
    prior_fixed = 0.5        # drafter, in AR-step units, until rounds are measured
    prior_per_row = 0.3      # each verify row after the first, in AR-step units

    def __init__(self):
        self.draft_rate: Optional[float] = None     # committed tokens per second of draft rounds (EMA)
        self.width_seconds: dict = {}                # verify rows T -> round seconds (EMA)
        self.position_accept: list = []              # P(draft j accepted | drafts < j accepted), EMA per position
        self.ar_seconds: Optional[float] = None     # seconds per AR step (EMA)
        self.ar_left = 0
        self.since_probe = 0
        self.rounds = 0
        self.probe = False                           # the next action must be a draft round
        # AR-cost hygiene (cross-lane port of glm_dflash2.AdaptiveSpec d6d64762 / 3dec394f, 2026-10-17): once draft
        # rounds win, nothing re-measured the AR step for the rest of the request (a 20-30k-token reasoning answer
        # kept a one-step seed or one stalled burst); every ar_reprobe_rounds rounds an AR burst re-measures it and
        # replaces most of the old value, and an ordinary sample above ar_outlier_ratio x the estimate is skipped.
        # VMLX_DFLASH_AR_REPROBE_ROUNDS=0 restores the old behaviour (A/B).
        self.ar_reprobe_rounds = int(os.environ.get("VMLX_DFLASH_AR_REPROBE_ROUNDS", "256"))
        self.ar_outlier_ratio = 3.0
        self.ar_outliers_skipped = 0
        self.rounds_since_ar = 0
        self._reprobe = False

    def next_is_ar(self) -> bool:
        if self.probe:
            self.probe = False
            return False
        if self.ar_left > 0:
            return True
        if self.rounds >= 1 and self.ar_seconds is None:
            self.ar_left = 1                         # seed the AR cost once
            return True
        if self.ar_reprobe_rounds > 0 and self.ar_seconds is not None and self.rounds_since_ar >= self.ar_reprobe_rounds:
            self.rounds_since_ar = 0
            self._reprobe = True
            self.ar_left = self.burst                # re-measure a possibly stale AR cost
            return True
        if self.draft_rate is not None and self.ar_seconds is not None \
                and 1.0 / self.ar_seconds > self.draft_rate * (1.0 + self.margin):
            self.ar_left = self.burst
            return True
        return False

    def round_seconds(self, rows: int) -> Optional[float]:
        """Estimated seconds of a draft round verifying ``rows`` rows: its EMA when measured, else a line through the
        measured widths (one measured width: the prior slope), else the prior; None before any timing exists."""
        if rows in self.width_seconds:
            return self.width_seconds[rows]
        pts = sorted(self.width_seconds.items())
        unit = self.ar_seconds
        if len(pts) >= 2:
            (t0, c0), (t1, c1) = pts[0], pts[-1]
            slope = max((c1 - c0) / max(t1 - t0, 1), 0.0)
            return max(c0 + slope * (rows - t0), 1e-6)
        if len(pts) == 1 and unit is not None:
            t0, c0 = pts[0]
            return max(c0 + self.prior_per_row * unit * (rows - t0), 1e-6)
        if unit is not None:
            return unit * (1.0 + self.prior_fixed + self.prior_per_row * (rows - 1))
        return None

    def choose_width(self, conf: Sequence[float], max_draft: int, cost_per_row: float) -> int:
        """Draft count k maximizing E[tokens | k] / measured round seconds(k + 1 rows); the static model until a
        timing exists."""
        # Opt-in (VMLX_DSPARK_MEASURED_WIDTH=1): the sustained multi-turn A/B on the 35B (ab-1013-0135, 2 reps per arm,
        # 5-16k context) measured static x0.979 / x1.005 of AR vs measured x0.967 / x0.986 (greedy / thinking T1.0):
        # no gain, so the static model stays the default.
        if self.round_seconds(2) is None or os.environ.get("VMLX_DSPARK_MEASURED_WIDTH", "0").strip().lower() not in \
                {"1", "true", "on", "yes"}:
            return choose_width(conf, max_draft, cost_per_row)
        best, best_k, run, expected = 0.0, 1, 1.0, 1.0
        for k in range(1, min(max_draft, len(conf)) + 1):
            run *= float(conf[k - 1])
            expected += run
            value = expected / self.round_seconds(k + 1)
            if value > best:
                best, best_k = value, k
        return best_k

    prior_accept = 0.7
    # VMLX_DSPARK_EXPLORE=1 (default off until measured): a position past the drafted width is never observed, so one
    # whose estimate fell after a few early rejections keeps it for the rest of the request and the planner never
    # drafts past it again (MiniMax-M3.1 real bundle 2026-10-29, Studio: easy_list 76 rounds at 4 rows with acceptance
    # 0.88-0.90; easy_code 104 rounds at 2 rows with acceptance 0.72 while the same prompt ran 5-12 rows at 5.2 tokens
    # per round in the other arm). With the lever, when the position just past the planned width has not been
    # observed for explore_every planned rounds, the round drafts one deeper so that position is re-measured (at most
    # one extra verify row per explore_every rounds; a really low position stays low). It changes which width Adaptive
    # picks, never caps it, and never changes tokens.
    explore_every = 32

    def _explore(self) -> bool:
        return os.environ.get("VMLX_DSPARK_EXPLORE", "").strip().lower() in {"1", "true", "on", "yes"}

    def record_acceptance(self, drafted: int, accepted: int) -> None:
        """Per-position acceptance history for planning a width before the round's drafts exist (pipelined rounds)."""
        while len(self.position_accept) < drafted:
            self.position_accept.append(self.prior_accept)
        seen = self.__dict__.setdefault("position_seen", [])
        while len(seen) < drafted:
            seen.append(self.__dict__.get("planned_rounds", 0))
        for j in range(min(drafted, accepted + 1)):
            hit = 1.0 if j < accepted else 0.0
            self.position_accept[j] += self.alpha * (hit - self.position_accept[j])
            seen[j] = self.__dict__.get("planned_rounds", 0)

    def planned_width(self, max_draft: int, cost_per_row: float) -> int:
        """Draft count from the acceptance HISTORY (no sync on this round's confidences): the same objective as
        choose_width with the per-position EMA in place of the confidence head (``VMLX_DSPARK_EXPLORE``: one deeper
        when the next position is stale)."""
        conf = [self.position_accept[j] if j < len(self.position_accept) else self.prior_accept
                for j in range(max_draft)]
        k = self.choose_width(conf, max_draft, cost_per_row)
        if self._explore():
            n = self.__dict__.get("planned_rounds", 0)
            self.planned_rounds = n + 1
            seen = self.__dict__.get("position_seen", [])
            if k < max_draft and k < len(seen) and n - seen[k] >= self.explore_every:
                k += 1
        return k

    def record_draft(self, tokens: int, seconds: float, rows: Optional[int] = None) -> None:
        if rows is not None:
            prev = self.width_seconds.get(rows)
            self.width_seconds[rows] = seconds if prev is None else prev + self.alpha * (seconds - prev)
        self.rounds += 1
        self.rounds_since_ar += 1
        self.since_probe = 0
        rate = tokens / max(seconds, 1e-9)
        self.draft_rate = rate if self.draft_rate is None else self.draft_rate + self.alpha * (rate - self.draft_rate)

    def record_ar(self, seconds: float) -> None:
        self.record_ar_burst(1, seconds)

    def cost_snapshot(self) -> dict:
        """Read-only measured costs for end-of-request diagnostics."""
        return {
            "ar_step_seconds": self.ar_seconds,
            "draft_tokens_per_second": self.draft_rate,
            "round_seconds_by_rows": dict(self.width_seconds),
        }

    def continues_after(self, steps: int) -> bool:
        """Whether the action after an AR burst of ``steps`` steps is AR again, judged before the burst's own timing is
        recorded (``next_is_ar`` after ``record_ar_burst``, with the current cost estimates): a pipelined burst launches
        its successor's first step early on this guess, and a wrong guess rolls that one step back."""
        if self.since_probe + steps >= self.probe_every:
            return False                             # a draft probe is due
        if self.ar_left - steps > 0:
            return True
        return (self.draft_rate is not None and self.ar_seconds is not None
                and 1.0 / self.ar_seconds > self.draft_rate * (1.0 + self.margin))

    def burst_size(self) -> int:
        """AR steps to run now: the rest of the current burst, cut at the next draft probe."""
        return max(1, min(self.ar_left, self.probe_every - self.since_probe))

    def record_ar_burst(self, steps: int, seconds: float) -> None:
        per_step = seconds / max(1, steps)
        self.rounds_since_ar = 0
        reprobe, self._reprobe = self._reprobe, False
        if self.ar_seconds is None:
            self.ar_seconds = per_step
        elif reprobe:
            self.ar_seconds = 0.3 * self.ar_seconds + 0.7 * per_step
        elif self.ar_outlier_ratio > 0 and per_step > self.ar_outlier_ratio * self.ar_seconds:
            self.ar_outliers_skipped += 1            # one stalled burst must not freeze the policy on drafting
        else:
            self.ar_seconds = self.ar_seconds + self.alpha * (per_step - self.ar_seconds)
        self.ar_left = max(0, self.ar_left - steps)
        self.since_probe += steps
        if self.since_probe >= self.probe_every:
            self.ar_left = 0
            self.probe = True                        # one draft round re-measures the text


def _pick(logits: mx.array, sampler: Optional[Callable]) -> mx.array:
    if sampler is None:
        return mx.argmax(logits, axis=-1)
    lp = logits.astype(mx.float32)
    return sampler(lp - mx.logsumexp(lp, axis=-1, keepdims=True)).reshape(-1)


def iter_dspark(
    target,
    drafter,
    prompt_ids: Sequence[int],
    max_new: int,
    eos: set,
    *,
    prefill_step: Optional[int] = None,
    sampler: Optional[Callable] = None,
    max_draft: int = MAX_VERIFY_ROWS - 1,
    cost_per_row: float = 0.12,
    fixed_width: Optional[int] = None,
    stats: Optional[DSparkStats] = None,
    ar_fallback: Optional[ARFallback] = None,
    resume: Optional[dict] = None,
    snapshot_cuts: Sequence[int] = (),
    on_snapshot: Optional[Callable] = None,
    session: Optional[dict] = None,
    pick: Optional[Callable] = None,
    prefill_plan: Optional[dict] = None,
) -> Iterator[List[int]]:
    """Yield the token ids committed by each round (prefill yields the first). ``ar_fallback`` (a policy instance,
    or None for draft rounds only) interleaves AR steps when drafting measures slower than plain decoding.

    Multi-turn reuse (dspark/session.py): ``resume`` = a stored session entry ({target_cache, draft_cache,
    cache_len}) whose tokens are a prefix of ``prompt_ids``; only ``prompt_ids[cache_len:]`` is prefilled.
    ``snapshot_cuts`` (absolute prompt positions) end prefill segments; ``on_snapshot(cut, target_cache,
    draft_caches)`` is called at each with the live caches (the callee clones them). ``session`` (a dict) is kept
    current with ``forwarded`` (exactly the token ids the target cache and drafter context hold), ``target_cache``
    and ``drafter``, so the caller can store the end-of-turn state whenever it stops consuming.

    ``resume["draft_cache"]`` may be None: the drafter then starts with an empty context positioned at ``cache_len``
    (a prefix whose taps it never saw; drafts only). ``resume["draft_base"]``: the position of the stored drafter
    context's first row (default 0).

    ``pick(logits (rows, V)) -> token ids``: how the target's token is chosen from its logits (default: argmax, or
    ``sampler`` on log-probabilities); a serving lane passes its own decode pick so every committed token is chosen
    exactly as its plain decode would choose it.

    Draft width: the ladder's top is ``max_draft`` within the target's verify coverage (``target.max_verify_rows``;
    None = any width). The drafter's trained block is not a ceiling: a round drafting more tokens than the block runs
    a longer block of mask rows. Exactness never depends on the width.

    ``prefill_plan``: a target's aligned-prefill plan (MiniMax-M3: absolute chunk cuts, turn cuts and the prefix
    cache's store boundary, models/minimax_m3/prefill_geometry.py), handed to ``target.prefill`` when it takes one."""
    cover = getattr(target, "max_verify_rows", MAX_VERIFY_ROWS)
    max_draft = max(1, int(max_draft) if cover is None else min(int(max_draft), int(cover) - 1))
    choose = (lambda lg: _pick(lg, sampler)) if pick is None else pick
    stats = stats if stats is not None else DSparkStats()
    taps = drafter.target_layer_ids
    ids = [int(t) for t in prompt_ids]
    if resume is not None:
        cache = resume["target_cache"]
        cached = int(resume["cache_len"])
        if resume.get("draft_cache") is None:
            drafter.reset(base=cached)
        else:
            drafter.set_context(resume["draft_cache"], base=int(resume.get("draft_base", 0) or 0))
        if drafter.ctx_len != cached:
            raise ValueError(f"DSpark resume: drafter context {drafter.ctx_len} != target cache {cached}")
    else:
        cache = target.make_cache()
        cached = 0
        drafter.reset()
    if not 0 <= cached < len(ids):
        raise ValueError("DSpark resume must leave at least one prompt token to prefill")
    forwarded = ids[:cached]
    if session is not None:
        session.update(forwarded=forwarded, target_cache=cache, drafter=drafter)
    t0 = time.perf_counter()
    start = cached
    for cut in sorted({int(c) for c in snapshot_cuts if cached < int(c) < len(ids)}):
        feats = target.prefill_segment(mx.array([ids[start:cut]]), cache, taps, prefill_step)
        if feats is not None:
            drafter.add_context(feats)
        mx.eval(*drafter.context_arrays())
        forwarded.extend(ids[start:cut])
        start = cut
        if on_snapshot is not None:
            on_snapshot(cut, cache, drafter.context_caches())
    # A sliding-window drafter never reads context rows more than ``sliding_window`` positions back: a target that
    # can (``keep_last``) returns only those prompt rows, and the drafter context starts at the first one.
    window = int(getattr(drafter, "sliding_window", 0) or 0)
    _prefill_params = inspect.signature(target.prefill).parameters
    extra = {"keep_last": window} if window and "keep_last" in _prefill_params else {}
    if prefill_plan is not None and "plan" in _prefill_params:
        extra["plan"] = prefill_plan
    logits, feats = target.prefill(mx.array([ids[start:]]), cache, taps, prefill_step, **extra)
    anchor = choose(logits[None]).reshape(1)
    if feats is not None:
        skipped = len(ids) - start - int(feats.shape[1])
        if skipped > 0:
            drafter.reset(base=start + skipped)
        drafter.add_context(feats)
    mx.eval(anchor, *drafter.context_arrays())
    forwarded.extend(ids[start:])
    stats.prefill_s = time.perf_counter() - t0
    stats.prefill_tokens = len(ids) - cached
    stats.cached_tokens = cached
    first = int(anchor.item())
    stats.tokens = 1
    yield [first]
    if first in eos or max_new <= 1:
        return
    mask_embed = target.embed_tokens(mx.array([[drafter.mask_token_id]]))
    # Pipelined rounds (default; VMLX_DSPARK_PIPELINE=0 = two syncs per round): planned width, drafter overlapped with
    # verify graph building. Sustained multi-turn A/Bs 2026-10-13: 1B x0.922 / x0.954 vs x0.902 / x0.909 static,
    # 35B neutral to positive (best config x1.044 greedy). Exact either way.
    pipelined = (ar_fallback is not None and hasattr(ar_fallback, "planned_width")
                 and os.environ.get("VMLX_DSPARK_PIPELINE", "1").strip().lower() not in {"0", "false", "off", "no"})
    t1 = time.perf_counter()
    produced = 1
    # Carry: an AR step launched at the end of a burst for the burst that most likely follows (a target that can roll
    # one AR step back: ``target.ar_step_rollback``; VMLX_DSPARK_AR_CARRY=0 disables). It is exactly the step that
    # burst would launch first (same input, same cache), so the tokens never change; a draft round that follows instead
    # rolls its row back first. (pick, tap row, input token) or None.
    carry_ok = (ar_fallback is not None and hasattr(ar_fallback, "continues_after")
                and bool(getattr(target, "ar_step_rollback", False))
                and os.environ.get("VMLX_DSPARK_AR_CARRY", "1").strip().lower() not in {"0", "false", "off", "no"})
    carry = None
    if session is not None:
        session["ar_carried"] = session["ar_carry_rolled_back"] = 0

    def drop_carry():
        nonlocal carry
        if carry is not None:
            target.rollback(cache, None, kept=0, total=1)
            carry = None
            if session is not None:
                session["ar_carry_rolled_back"] += 1

    try:
        while produced < max_new:
            room = max_new - produced
            if ar_fallback is not None and fixed_width is None and ar_fallback.next_is_ar():
                # Pipelined AR burst: step i+1 is queued (async_eval) before step i's token is read, as in the AR
                # lane, and the burst's tap rows reach the drafter in one call. Same values as one-at-a-time steps.
                n_steps = max(1, min(int(ar_fallback.burst_size()), room))
                t_step = time.perf_counter()
                if carry is not None:
                    pick_, feats, first_in = carry       # launched during the previous burst
                    carry = None
                    launched = [first_in]
                    if session is not None:
                        session["ar_carried"] += 1
                else:
                    launched = [int(anchor.item())]
                    logits, feats = target.ar_step(anchor, cache, taps)
                    pick_ = choose(logits[None]).reshape(1)
                    mx.async_eval(pick_, feats)
                # launch past the burst's last step only for a successor burst that is likely and has room
                tail = carry_ok and n_steps < room and ar_fallback.continues_after(n_steps)
                burst_feats, new, stop = [], [], False
                for i in range(n_steps):
                    nxt = None
                    if i + 1 < n_steps or tail:
                        launched.append(None)            # filled with this step's input token below, once read
                        logits2, feats2 = target.ar_step(pick_.astype(anchor.dtype), cache, taps)
                        nxt = (choose(logits2[None]).reshape(1), feats2)
                        mx.async_eval(*nxt)
                    tok = int(pick_.item())
                    if nxt is not None:
                        launched[i + 1] = tok            # the pipelined step forwarded this token
                    burst_feats.append(feats)
                    new.append(tok)
                    if tok in eos:
                        stop = True
                        break
                    if nxt is not None:
                        pick_, feats = nxt
                if launched and launched[-1] is None:
                    launched.pop()                       # stopped before reading the last launched step's input
                if not stop and len(launched) > len(burst_feats):
                    # The step launched past the burst: carried to the next burst (or rolled back), not part of this
                    # burst's forwarded tokens or drafter context.
                    carry = (pick_, feats, launched.pop())
                # A pipelined step past the EOS token already ran: its input (the EOS) is in the cache, and its tap
                # row must reach the drafter so the context stays aligned with the cache.
                if len(launched) > len(burst_feats) and nxt is not None:
                    burst_feats.append(nxt[1])
                drafter.add_context(mx.concatenate(burst_feats, axis=1) if len(burst_feats) > 1 else burst_feats[0])
                forwarded.extend(launched)
                anchor = mx.array([new[-1]], dtype=anchor.dtype)
                ar_fallback.record_ar_burst(len(new), time.perf_counter() - t_step)
                stats.ar_tokens += len(new)
                produced += len(new)
                stats.tokens = produced
                stats.decode_s = time.perf_counter() - t1
                yield new
                if stop:
                    return
                continue
            drop_carry()                                 # a draft round follows: its verify starts at the anchor
            t_round = time.perf_counter()
            n = min(max_draft, room - 1) if room > 1 else 0
            k = 0
            if n >= 1 and pipelined:
                k = fixed_width if fixed_width is not None else ar_fallback.planned_width(max_draft, cost_per_row)
                k = max(1, min(int(k), n))
            # The trained block, or a longer one when this round drafts past it.
            span = max(drafter.block, k if pipelined else (max_draft if n >= 1 else 0))
            block = mx.concatenate(
                [target.embed_tokens(anchor.reshape(1, 1)),
                 mx.broadcast_to(mask_embed, (1, span - 1, mask_embed.shape[-1]))], axis=1)
            hidden = drafter(block)
            if n >= 1 and pipelined:
                # One sync per round: the width comes from the acceptance history, the drafter is queued (async_eval) and
                # the verify graph is built on the CPU while the GPU drafts. Same values; only the schedule differs.
                proposal, conf = drafter.draft(hidden, anchor, target.lm_head, k, sampler=sampler)
                mx.async_eval(proposal)
            elif n >= 1:
                drafts, conf = drafter.draft(hidden, anchor, target.lm_head, max_draft,
                                             sampler=sampler)
                mx.eval(drafts, conf)
                if fixed_width is not None:
                    k = fixed_width
                elif ar_fallback is not None and hasattr(ar_fallback, "choose_width"):
                    k = ar_fallback.choose_width(conf.tolist(), max_draft, cost_per_row)
                else:
                    k = choose_width(conf.tolist(), max_draft, cost_per_row)
                k = max(1, min(int(k), n))
                proposal = drafts[:k]
            else:
                k, proposal = 0, mx.array([], dtype=anchor.dtype)
            rows = mx.concatenate([anchor, proposal]).reshape(1, k + 1)
            logits, feats, sink = target.verify(rows, cache, taps)
            picks = choose(logits)
            mx.eval(picks, feats)
            if pipelined and k:
                mx.eval(proposal)
            drafted = proposal.tolist() if k else []
            target_picks = picks.tolist()
            accepted = 0
            while accepted < k and drafted[accepted] == target_picks[accepted]:
                accepted += 1
            new = drafted[:accepted] + [target_picks[accepted]]
            target.rollback(cache, sink, kept=accepted + 1, total=k + 1)
            drafter.add_context(feats[:, :accepted + 1])
            forwarded.extend(rows[0, :accepted + 1].tolist())
            anchor = mx.array([new[-1]], dtype=anchor.dtype)
            stats.rounds += 1
            stats.drafted += k
            stats.accepted += accepted
            if ar_fallback is not None:
                ar_fallback.record_draft(len(new), time.perf_counter() - t_round, rows=k + 1)
                if hasattr(ar_fallback, "record_acceptance"):
                    ar_fallback.record_acceptance(k, accepted)
            stats.widths[k + 1] = stats.widths.get(k + 1, 0) + 1
            cut = next((i for i, t in enumerate(new) if t in eos), None)
            if cut is not None:
                new = new[:cut + 1]
            new = new[:max_new - produced]
            produced += len(new)
            stats.tokens = produced
            stats.decode_s = time.perf_counter() - t1
            yield new
            if cut is not None:
                return
    finally:
        # Loop end (length) or the consumer closing the stream: the cache must hold exactly the forwarded tokens.
        drop_carry()


def generate_dspark(target, drafter, prompt_ids, max_new, eos, **kwargs) -> List[int]:
    out: List[int] = []
    for chunk in iter_dspark(target, drafter, prompt_ids, max_new, eos, **kwargs):
        out.extend(chunk)
    return out


def generate_ar(target, prompt_ids, max_new, eos, *, prefill_step=None, stats: Optional[dict] = None,
                prefill_cuts: Sequence[int] = ()) -> List[int]:
    """Greedy AR reference with the same prefill structure and the served decode call. ``prefill_cuts`` splits the
    prefill at those positions exactly like a DSpark request that snapshots there (same segment shapes)."""
    cache = target.make_cache()
    t0 = time.perf_counter()
    ids = [int(t) for t in prompt_ids]
    start = 0
    for cut in sorted({int(c) for c in prefill_cuts if 0 < int(c) < len(ids)}):
        target.prefill_segment(mx.array([ids[start:cut]]), cache, (), prefill_step)
        start = cut
    logits, _ = target.prefill(mx.array([ids[start:]]), cache, (), prefill_step)
    tok = mx.argmax(logits).reshape(1)
    out = [int(tok.item())]
    t1 = time.perf_counter()
    while len(out) < max_new and out[-1] not in eos:
        tok = mx.argmax(target.decode(tok, cache)).reshape(1)
        out.append(int(tok.item()))
    if stats is not None:
        stats.update(tokens=len(out), prefill_s=t1 - t0, decode_s=time.perf_counter() - t1,
                     decode_tok_s=(len(out) - 1) / max(time.perf_counter() - t1, 1e-9))
    return out

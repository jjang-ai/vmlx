"""Per-round verify-row policy for speculative lanes on sparse (MoE) targets, where a verify row is not free.

On MiMo-V2.6 (JANGHT: trellis + Hadamard experts) a verify forward costs ~28 ms for 2 rows against a ~21 ms plain
step, and each further row ~4.5-5 ms, because every verified token routes to its own experts. A schedule fixed per
request (block width from per-position acceptance averaged over the request) then loses on low-acceptance prose:
a round is bought for drafts the drafter itself is unsure of.

This policy decides per round, after the drafter has run:
- each draft's acceptance probability comes from the drafter's own confidence in it (its probability of the token
  it proposed), calibrated online in bins over -log10(1 - confidence), each bin pulled toward a logistic prior and
  forgetting old rounds, so a change of content class re-calibrates within a few rounds;
- the round verifies the row count with the best expected tokens per verify-ms (rows = 1 is a plain step: the anchor
  alone); a row count within 2 % of the best rate loses to the cheaper one;
- drafting itself is skipped while it does not pay: every drafted round is judged by the plain-step ms its tokens
  replaced against the ms it took; a losing round owes plain steps (doubling to a cap of 32), a winning one clears
  the debt and halves the period;
- the plain-step cost is learned from plain steps only, by at most a quarter per sample, so a stall or a foreign GPU
  job cannot pose as the machine's speed.

The policy only chooses how many of the drafted rows are verified and when to draft: emitted tokens are unchanged
(greedy == AR; sampled rows stay exact: the verified prefix of a drafted block is accepted by the same rejection rule).
Pure Python state, no MLX.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_ROWS = 16
ROW_TIE = 0.02


def prior_verify_costs(plain_ms: float, ladder: dict[int, float]) -> dict[int, float]:
    """`ladder` (rows -> ms, rows 1 = plain step) scaled so one row costs `plain_ms`: linear between known points,
    the last slope beyond them, never cheaper for more rows."""
    pts = sorted((int(r), float(m)) for r, m in ladder.items())
    scale = plain_ms / pts[0][1]
    out: dict[int, float] = {}
    for r in range(1, MAX_ROWS + 1):
        if r >= pts[-1][0]:
            (r0, m0), (r1, m1) = pts[-2], pts[-1]
        else:
            (r0, m0), (r1, m1) = next(((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= r <= b[0]))
        ms = m0 + (m1 - m0) * (r - r0) / max(r1 - r0, 1)
        out[r] = max(ms * scale, out.get(r - 1, 0.0))
    return out


def choose_rows(p: list[float], costs: dict[int, float], max_rows: int) -> tuple[int, float]:
    """(rows, expected tokens) maximizing expected tokens per verify-ms. p[k] = P(draft k accepted | drafts < k
    accepted); costs[r] = verify ms for r rows (r = 1: the anchor alone, a plain step)."""
    n = min(len(p), max_rows - 1)
    tokens = [0.0, 1.0]
    alive = 1.0
    for pk in p[:n]:
        pk = pk if isinstance(pk, (int, float)) and pk == pk else 0.0       # NaN -> no acceptance
        alive *= min(max(pk, 0.0), 1.0)
        tokens.append(tokens[-1] + alive)
    best = max(tokens[r] / costs[r] for r in range(1, n + 2))
    for r in range(1, n + 2):
        if tokens[r] / costs[r] >= best * (1.0 - ROW_TIE):
            return r, tokens[r]
    return 1, 1.0


@dataclass
class Calibration:
    """P(draft accepted | its predecessors were) by the drafter's confidence in it."""
    BINS: int = 12
    PER_UNIT: float = 4.0
    PRIOR_WEIGHT: float = 3.0
    KEEP: float = 0.97
    hit: list = field(default_factory=lambda: [0.0] * 12)
    seen: list = field(default_factory=lambda: [0.0] * 12)

    @staticmethod
    def axis(conf: float) -> float:
        return -math.log10(max(1.0 - min(max(conf, 0.0), 1.0), 1e-3))

    def bin(self, conf: float) -> int:
        return min(self.BINS - 1, int(self.axis(conf) * self.PER_UNIT))

    def prior(self, conf: float) -> float:
        return 1.0 / (1.0 + math.exp(-(-1.15 + 1.58 * self.axis(conf))))

    def probability(self, conf: float) -> float:
        b = self.bin(conf)
        return (self.hit[b] + self.PRIOR_WEIGHT * self.prior(conf)) / (self.seen[b] + self.PRIOR_WEIGHT)

    def observe(self, conf: float, accepted: bool) -> None:
        b = self.bin(conf)
        self.hit[b] = self.hit[b] * self.KEEP + (1.0 if accepted else 0.0)
        self.seen[b] = self.seen[b] * self.KEEP + 1.0


@dataclass
class Drafting:
    """Whether a round drafts at all: a stretch where drafting does not pay is skipped with a probe back-off."""
    MAX_PERIOD: int = 32
    ALPHA: float = 0.3
    gain: float = 0.0
    wall: float = 0.0
    owed: int = 0
    period: int = 1
    skipped: int = 0
    probes: int = 0
    after_skip: bool = False

    def take_owed(self, limit: int) -> int:
        """Plain steps owed now (at most `limit`), consumed as one AR burst."""
        k = min(self.owed, max(limit, 0))
        self.owed -= k
        self.skipped += k
        if k:
            self.after_skip = True
        return k

    def drafting_now(self) -> None:
        if self.after_skip:
            self.probes += 1
        self.after_skip = False

    def record(self, gain_ms: float, wall_ms: float) -> None:
        """A drafted round: `gain_ms` = plain-step ms its tokens would have cost, `wall_ms` = what it took. Rounds are
        weighed by their time (an average of the two, not of their ratio), so a thin win with scattered misses keeps
        drafting."""
        if self.wall <= 0:
            self.gain, self.wall = gain_ms, wall_ms
        else:
            self.gain += self.ALPHA * (gain_ms - self.gain)
            self.wall += self.ALPHA * (wall_ms - self.wall)
        if self.gain >= self.wall:
            self.owed = 0
            self.period = max(1, self.period // 2)
        else:
            self.owed = self.period
            self.period = min(self.period * 2, self.MAX_PERIOD)


@dataclass
class PlainCost:
    """ms of a plain step, learned from plain steps only, by at most a quarter per sample."""
    ms: float

    def sample(self, ms: float) -> None:
        if not (ms > 0 and math.isfinite(ms)):
            return
        bounded = min(max(ms, self.ms * 0.8), self.ms * 1.25)
        self.ms += 0.1 * (bounded - self.ms)


@dataclass
class VerifyCosts:
    """Verify ms by rows (rows 1 = plain step), measured per round, prior from a ladder scaled to the plain step."""
    ladder: dict
    costs: dict = field(default_factory=dict)
    measured: set = field(default_factory=set)

    def reset_prior(self, plain_ms: float) -> None:
        prior = prior_verify_costs(plain_ms, self.ladder)
        for r, ms in prior.items():
            if r not in self.measured:
                self.costs[r] = ms

    def sample(self, rows: int, ms: float) -> None:
        if not (ms > 0 and math.isfinite(ms)) or rows < 2:
            return
        if rows not in self.measured:
            self.measured.add(rows)
            self.costs[rows] = ms
        else:
            cur = self.costs[rows]
            self.costs[rows] = cur + 0.2 * (min(max(ms, cur * 0.8), cur * 1.25) - cur)
        # keep the table monotone in rows
        for r in range(2, MAX_ROWS + 1):
            if self.costs.get(r, 0.0) < self.costs.get(r - 1, 0.0):
                self.costs[r] = self.costs[r - 1]


# MiMo-V2.6 JANGHT2.5 verify ladder on M5 Max at 30 tokens of context (release audit R24, 1.6 GHz): ms by rows,
# rows 1 = a plain pipelined step. Scaled to the measured plain step at runtime; replaced by measurement as rounds run.
MIMO_V26_LADDER = {1: 20.9, 2: 28.2, 3: 32.6, 4: 37.6, 6: 48.6, 8: 58.2}


@dataclass
class RowsPolicy:
    """All per-target state of the rows policy (calibration and costs are a property of the drafter / target and live
    across requests; the drafting back-off is per request)."""
    calib: Calibration
    plain: PlainCost
    verify: VerifyCosts

    @classmethod
    def for_target(cls, target, ladder: dict) -> "RowsPolicy":
        tables = getattr(target, "_vmlx_rows_policy", None)
        if tables is None:
            v = VerifyCosts(ladder=dict(ladder))
            plain = PlainCost(ms=float(ladder[1]))
            v.reset_prior(plain.ms)
            tables = cls(calib=Calibration(), plain=plain, verify=v)
            object.__setattr__(target, "_vmlx_rows_policy", tables)
        return tables

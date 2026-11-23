"""Request-owned native DSpark state for the Inkling batch scheduler."""

from collections import deque
import time

import mlx.core as mx

from .generate import ARFallback
from .inkling_drafter import InklingDSparkDrafter
from .inkling_target import InklingDSparkTarget


class InklingServingPolicy(ARFallback):
    """Plan using complete measured rounds and conditional prefix acceptance."""

    prior_accept = 0.6
    prior_fixed = 0.1
    prior_per_row = 0.36

    def __init__(self):
        super().__init__()
        self.ar_left = 2
        self.warmed_rows = set()

    def choose_width(self, conf, max_draft, cost_per_row):
        best_rate, best_width = 0.0, 1
        reach, expected = 1.0, 1.0
        for width, probability in enumerate(conf[:max_draft], start=1):
            reach *= probability
            expected += reach
            cost = self.round_seconds(width + 1)
            if cost is None:
                cost = 1.0 + self.prior_fixed + width * cost_per_row
            if expected / cost > best_rate:
                best_rate, best_width = expected / cost, width
        return best_width


class InklingServingRun:
    def __init__(self, model, head, cached_tokens, *, pick, trace_logits=False):
        self.target = InklingDSparkTarget(model)
        self.drafter = InklingDSparkDrafter(head, self.target)
        self.drafter.reset(base=cached_tokens)
        self.pick = pick
        self.trace_logits = trace_logits
        self.policy = InklingServingPolicy()
        self.horizon = max(1, head.config.block_size - 1)
        self.ready = deque()
        self.transaction = None
        self.features = None
        self.next_result = None
        self.emitted = 0
        self.ar_started = None
        self.prefill_tokens = 0
        self.ar_tokens = self.draft_rounds = self.accepted = 0
        self.widths = {}

    def add_prefill_features(self, features):
        self.prefill_tokens += features.shape[1]
        self.drafter.add_context(features)

    def _complete_round(self):
        self.transaction.commit(self.emitted)
        self.drafter.add_context(self.features[:, :self.emitted])
        self.transaction = self.features = None

    def finish(self):
        if self.transaction is not None:
            self._complete_round()
        self.ready.clear()

    def abort(self):
        if self.transaction is not None:
            self.transaction.abort()
        self.transaction = self.features = None
        self.ready.clear()

    def report(self):
        return {
            "method": "dspark",
            "native_family": "inkling2",
            "ar_tokens": self.ar_tokens,
            "draft_rounds": self.draft_rounds,
            "accepted_draft_tokens": self.accepted,
            "verify_rows": dict(self.widths),
            "horizon": self.horizon,
            "cost": self.policy.cost_snapshot(),
        }

    def _ar_step(self, prior, ar_forward):
        from ..models.inkling2.language import capture_dspark_features

        self.ar_started = time.perf_counter()
        with capture_dspark_features(
            self.target.model, self.drafter.target_layer_ids, self.drafter.add_context
        ):
            ar_forward()
        self.ar_tokens += 1
        return prior

    def _verify(self, batch, width):
        started = time.perf_counter()
        proposal = self.drafter.draft_from_anchor(batch.y, width, None)
        mx.async_eval(proposal)
        ids = mx.concatenate([batch.y, proposal]).reshape(1, -1)
        logits, self.features, self.transaction = self.target.verify(
            ids, batch.cache, self.drafter.target_layer_ids
        )
        # Sample each target row with the normal request sampler. Only the
        # draws through the first mismatch can be committed; later rows assume
        # the draft context and are discarded. This retains the target distribution.
        picks = self.pick(logits)
        mx.eval(picks, self.features)
        drafted, picked = proposal.tolist(), picks.tolist()
        accepted = 0
        while accepted < width and drafted[accepted] == picked[accepted]:
            accepted += 1
        verified_logits = [logits[i] if self.trace_logits else None for i in range(accepted + 1)]
        self.ready.extend(zip(picked[:accepted], verified_logits[:accepted]))
        self.next_result = (picks[accepted:accepted + 1], [verified_logits[accepted]])
        seconds = time.perf_counter() - started
        rows = width + 1
        self.draft_rounds += 1
        self.widths[rows] = self.widths.get(rows, 0) + 1
        self.policy.record_acceptance(width, accepted)
        if rows in self.policy.warmed_rows:
            self.policy.record_draft(accepted + 1, seconds, rows=rows)
        else:
            self.policy.warmed_rows.add(rows)
        if (
            width == self.horizon and accepted == width
            and self.policy.ar_seconds is not None
            and seconds < (accepted + 1) * self.policy.ar_seconds
        ):
            self.horizon += 1

    def step(self, batch, remaining, *, ar_forward, terminal=False):
        if self.ready:
            emitted = self.ready.popleft()
            if not terminal:
                self.emitted += 1
                self.accepted += 1
            if not self.ready or terminal:
                self._complete_round()
                batch.y, batch.logprobs = self.next_result
            return emitted

        prior = (int(batch.y[0].item()), batch.logprobs[0])
        if self.ar_started is not None:
            self.policy.record_ar(time.perf_counter() - self.ar_started)
            self.ar_started = None
        if terminal:
            return prior
        if remaining <= 1 or self.policy.next_is_ar():
            return self._ar_step(prior, ar_forward)

        width = self.policy.planned_width(
            min(self.horizon, remaining - 1), self.policy.prior_per_row
        )
        self._verify(batch, width)
        self.emitted = 1
        if not self.ready:
            self._complete_round()
            batch.y, batch.logprobs = self.next_result
        return prior

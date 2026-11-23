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
        self.ar_left = 4
        self._ar_warmed = False
        self.warmed_rows = set()

    def record_ar(self, seconds):
        if not self._ar_warmed:
            # First decode can compile even after the prompt has run.
            self._ar_warmed = True
            self.ar_left -= 1
            return
        super().record_ar(seconds)

    def next_is_ar(self):
        if (
            not self.probe and self.draft_rate is not None and self.ar_seconds is not None
            and self.draft_rate * self.ar_seconds < 1.0 + self.margin
        ):
            self.ar_left = max(self.burst, self.ar_left)
        return super().next_is_ar()

    def observe_round(self, rows, emitted, seconds):
        # Draft throughput pays cold compilation and unsuccessful work too.
        # Per-width forecasts use warmed costs so another width can re-enter.
        self.record_draft(emitted, seconds, rows=rows if rows in self.warmed_rows else None)
        self.warmed_rows.add(rows)
        loses_to_ar = self.ar_seconds is not None and seconds > emitted * self.ar_seconds
        if loses_to_ar:
            self.probe_every = min(max(self.probe_every * 2, 64), 512)
        else:
            self.probe_every = ARFallback.probe_every

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


def _normalized_logprobs(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _ratio_verification(logits, proposal_logprobs, proposals):
    # Each verified row is sampled independently, conditional on that row's
    # proposed history. The caller uses only the first rejection or the bonus.
    target = _normalized_logprobs(logits[:-1])
    draft = mx.concatenate(proposal_logprobs, axis=0)
    positions = proposals[:, None]
    target_mass = mx.take_along_axis(target, positions, axis=-1)
    draft_mass = mx.take_along_axis(draft, positions, axis=-1)
    draws = mx.log(mx.random.uniform(shape=(proposals.size,)))
    accepted = draws <= (target_mass - draft_mass).reshape(-1)
    target_probs = mx.exp(target)
    residual = mx.maximum(target_probs - mx.exp(draft), 0)
    residual = mx.where(residual.sum(axis=-1, keepdims=True) > 0, residual, target_probs)
    next_logits = mx.concatenate([mx.log(residual), logits[-1:]], axis=0)
    return accepted, mx.random.categorical(next_logits).astype(mx.uint32)


class InklingServingRun:
    def __init__(self, model, head, cached_tokens, *, pick,
                 ratio_temperature=None, trace_logits=False):
        self.target = InklingDSparkTarget(model)
        self.drafter = InklingDSparkDrafter(head, self.target)
        self.drafter.reset(base=cached_tokens)
        self.pick = pick
        self.ratio_temperature = ratio_temperature
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
        drafted = sum((rows - 1) * count for rows, count in self.widths.items())
        return {
            "method": "dspark",
            "native_family": "inkling2",
            "kind": "dspark",
            "lane": "inkling2",
            "cycles": self.draft_rounds,
            "drafted": drafted,
            "accepted": self.accepted,
            "acceptance": self.accepted / drafted if drafted else None,
            "widths": dict(self.widths),
            "ar_tokens": self.ar_tokens,
            "draft_rounds": self.draft_rounds,
            "verification": "ratio" if self.ratio_temperature is not None else "match",
            "accepted_draft_tokens": self.accepted,
            "verify_rows": dict(self.widths),
            "horizon": self.horizon,
            "cost": self.policy.cost_snapshot(),
        }

    def _ar_step(self, ar_forward):
        from ..models.inkling2.language import capture_dspark_features

        self.ar_started = time.perf_counter()
        with capture_dspark_features(
            self.target.model, self.drafter.target_layer_ids, self.drafter.add_context
        ):
            ar_forward()
        self.ar_tokens += 1

    def _verify(self, batch, width):
        started = time.perf_counter()
        proposal_logprobs = []
        draft_pick = None
        if self.ratio_temperature is not None:
            def draft_pick(logits):
                scaled = logits.astype(mx.float32) / self.ratio_temperature
                proposal_logprobs.append(_normalized_logprobs(scaled))
                return mx.random.categorical(scaled)

        proposal = self.drafter.draft_from_anchor(batch.y, width, None, draft_pick=draft_pick)
        mx.async_eval(proposal)
        ids = mx.concatenate([batch.y, proposal]).reshape(1, -1)
        logits, self.features, self.transaction = self.target.verify(
            ids, batch.cache, self.drafter.target_layer_ids
        )
        if proposal_logprobs:
            accepted_mask, picks = _ratio_verification(
                logits.astype(mx.float32) / self.ratio_temperature, proposal_logprobs, proposal
            )
            mx.eval(picks, accepted_mask, self.features)
            drafted = proposal.tolist()
            accepted = next((i for i, allowed in enumerate(accepted_mask.tolist()) if not allowed), width)
        else:
            # The ordinary request sampler decides every row for match verification.
            picks = self.pick(logits)
            mx.eval(picks, self.features)
            drafted, picked = proposal.tolist(), picks.tolist()
            accepted = 0
            while accepted < width and drafted[accepted] == picked[accepted]:
                accepted += 1
        verified_logits = [logits[i] if self.trace_logits else None for i in range(accepted + 1)]
        self.ready.extend(zip(drafted[:accepted], verified_logits[:accepted]))
        self.next_result = (picks[accepted:accepted + 1], [verified_logits[accepted]])
        seconds = time.perf_counter() - started
        rows = width + 1
        self.draft_rounds += 1
        self.widths[rows] = self.widths.get(rows, 0) + 1
        self.policy.record_acceptance(width, accepted)
        self.policy.observe_round(rows, accepted + 1, seconds)
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

        prior_token, prior_logprobs = batch.y, batch.logprobs[0]
        if self.ar_started is not None:
            self.policy.record_ar(time.perf_counter() - self.ar_started)
            self.ar_started = None
        if not terminal:
            if remaining <= 1 or self.policy.next_is_ar():
                self._ar_step(ar_forward)
            else:
                width = self.policy.planned_width(
                    min(self.horizon, remaining - 1), self.policy.prior_per_row
                )
                self._verify(batch, width)
                self.emitted = 1
                if not self.ready:
                    self._complete_round()
                    batch.y, batch.logprobs = self.next_result
        # Match the normal scheduler: build and enqueue the next forward
        # before reading the current token back to the CPU.
        return int(prior_token.item()), prior_logprobs

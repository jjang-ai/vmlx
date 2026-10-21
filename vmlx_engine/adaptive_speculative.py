# SPDX-License-Identifier: Apache-2.0
"""Measured adaptive decoding for ordinary external autoregressive drafters.

The verifier uses target-sampled token matching, as mlx-lm does. A proposal
never changes the target distribution or advances a target logits processor.
The configured draft count seeds exploration; it does not fix its width.

Arbitrary target architectures have no row-exact block-verification contract.
Use their ordinary one-token decode calls for verification. Specialized
native MTP/DFlash/DSpark adapters own their separately qualified fast paths.
"""

from collections import Counter, defaultdict, deque
import time


class AdaptiveDraftWidth:
    """Request-local wall-cost selection, including AR calibration and re-entry."""

    # Bound the temporary verification allocation, independently of user seeds.
    max_width = 20

    def __init__(self, seed=3):
        self.seed = max(1, min(self.max_width, int(seed)))
        self.samples = defaultdict(lambda: deque(maxlen=8))
        self.widths = Counter()
        self.accepted = self.drafted = self.cycles = 0
        self.width = 0
        self.probe_left = 0
        self.hold = 0
        self.last_ar_cycle = 0
        self.transitions = []

    def _select(self, width, reason, probe=False):
        if width != self.width:
            self.transitions.append({"cycle": self.cycles, "from": self.width,
                                     "to": width, "reason": reason})
            del self.transitions[:-128]
        self.width = width
        self.probe_left = 3 if probe else 0
        self.hold = 0

    def value(self, width):
        rows = self.samples[width]
        if not rows:
            return 0.0
        return sum(n for n, _ in rows) / sum(t for _, t in rows)

    def observe(self, width, confirmed, accepted, seconds):
        self.cycles += 1
        self.widths[width] += 1
        self.accepted += accepted
        self.drafted += width
        if seconds <= 0 or confirmed <= 0:
            return
        if width == 0:
            self.last_ar_cycle = self.cycles
        # Do not use the first decode dispatch's compilation/warm-up cost as
        # evidence that speculation is profitable.
        if self.cycles <= 2:
            return
        self.samples[width].append((confirmed, seconds))
        if self.cycles <= 8:
            if self.cycles == 8:
                self._select(self.seed, "calibration_complete", probe=True)
            return
        if self.probe_left:
            self.probe_left -= 1
            if self.probe_left:
                return
            profitable = [d for d, rows in self.samples.items()
                          if d and len(rows) >= 3 and self.value(d) > self.value(0) * 1.03]
            if not profitable:
                if width != 1 and len(self.samples[1]) < 3:
                    self._select(1, "test_lower_cost", probe=True)
                else:
                    self._select(0, "measured_ar_wins")
                return
            best = max(profitable, key=self.value)
            self._select(best, "measured_tokens_per_second")
            return
        self.hold += confirmed if width == 0 else 1
        if width == 0:
            if self.hold >= 32:
                self._select(1, "ar_reentry_probe", probe=True)
        elif self.hold >= 8:
            if self.cycles - self.last_ar_cycle >= 64:
                self._select(0, "refresh_ar_baseline", probe=True)
                return
            # Re-measure adjacent widths, including those previously rejected.
            # Sampling ages in a bounded window as the workload changes.
            neighbors = [d for d in (width - 1, width + 1) if 1 <= d <= self.max_width]
            candidate = min(neighbors, key=lambda d: (self.widths[d], d))
            self._select(candidate, "neighbor_probe", probe=True)

    def snapshot(self):
        return {"policy": "adaptive", "starting_width": self.seed,
                "verification_mode": "sequential_target_decode",
                "verification_allocation_limit": self.max_width,
                "selected_width": self.width, "cycles": self.cycles,
                "drafted_tokens": self.drafted, "accepted_draft_tokens": self.accepted,
                "width_counts": dict(self.widths),
                "measured_tokens_per_second": {d: self.value(d) for d in self.samples},
                "transitions": list(self.transitions)}


def adaptive_generate_step(prompt, model, draft_model, *, num_draft_tokens=3,
                           max_tokens=256, sampler=None, logits_processors=None,
                           prompt_cache=None, prefill_step_size=2048, on_stats=None):
    """Yield (target token, target logprobs, accepted proposal) with safe rewind."""
    import mlx.core as mx
    from mlx_lm.models import cache
    from mlx_lm.generate import generation_stream

    if max_tokens == 0:
        return
    history = prompt.tolist()
    if not history:
        raise ValueError("External speculative decoding requires a non-empty prompt")
    # mlx-lm's AR generator prefills all but the final prompt token before
    # starting logits processors. Preserve that same processor input contract.
    processor_start = len(history) - 1
    if prompt_cache is None:
        target_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        target_cache = prompt_cache[:len(model.layers)]
        draft_cache = prompt_cache[len(model.layers):]
    if not (cache.can_trim_prompt_cache(target_cache) and cache.can_trim_prompt_cache(draft_cache)):
        raise ValueError("External adaptive decoding requires trimmable target and draft caches")
    policy = AdaptiveDraftWidth(num_draft_tokens)
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def forward(m, c, tokens):
        with mx.stream(generation_stream):
            return m(mx.array(tokens, mx.uint32)[None], cache=c)[0]

    def sample(logits, tokens=None):
        if tokens is not None:
            for processor in logits_processors or []:
                logits = processor(mx.array(tokens, mx.uint32), logits[None])[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = sampler(lp[None]).reshape(-1)[0]
        mx.eval(token, lp)
        return int(token.item()), lp

    def prefill(m, c):
        for start in range(0, len(history) - 1, max(1, prefill_step_size)):
            end = min(len(history) - 1, start + max(1, prefill_step_size))
            logits = forward(m, c, history[start:end])
            mx.eval(logits, [layer.state for layer in c])

    prefill(draft_model, draft_cache)
    prefill(model, target_cache)
    # Offsets here are relative to this request's prompt, even if supplied
    # caches already contain an earlier prefix.
    draft_offset = len(history) - 1
    generated = 0
    delivered_drafts = 0
    try:
        while max_tokens < 0 or generated < max_tokens:
            remaining = max_tokens - generated if max_tokens >= 0 else policy.max_width + 1
            width = min(policy.width, max(0, remaining - 1))
            started = time.perf_counter()
            proposals = []
            for _ in range(width):
                context = history + proposals
                logits = forward(draft_model, draft_cache, context[draft_offset:])[-1]
                draft_offset = len(context)
                token, _ = sample(logits)
                proposals.append(token)
            # A generic multi-row forward can change QMM/attention reduction
            # order and even greedy decisions. Keep the target's decode
            # arithmetic until an architecture has its own qualified block
            # verifier. Proposal costs remain included in adaptive selection.
            rows = []
            for candidate in [history[-1]] + proposals:
                row = forward(model, target_cache, [candidate])
                mx.eval(row)
                rows.append(row)
            logits = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
            mx.eval(logits)
            outputs = []
            for i in range(width + 1):
                token, lp = sample(logits[i], history[processor_start:] + proposals[:i])
                accepted = i < width and token == proposals[i]
                outputs.append((token, lp, accepted))
                if not accepted:
                    break
            # Target's last processed token must be the predecessor of the next
            # emitted token. Draft may lag during AR; catch-up is charged to its
            # next measured probe, not silently excluded from the cost.
            cache.trim_prompt_cache(target_cache, width + 1 - len(outputs))
            desired = len(history) + len(outputs) - 1
            if draft_offset > desired:
                cache.trim_prompt_cache(draft_cache, draft_offset - desired)
                draft_offset = desired
            policy.observe(width, len(outputs), sum(x[2] for x in outputs),
                           time.perf_counter() - started)
            delivered = 0
            try:
                for token, lp, accepted in outputs:
                    history.append(token)
                    generated += 1
                    delivered_drafts += int(accepted)
                    delivered += 1
                    yield token, lp, accepted
            finally:
                # EOS or cancellation can close the iterator midway through a
                # verified batch. Never leave unseen tokens in either cache.
                cache.trim_prompt_cache(target_cache, len(outputs) - delivered)
                desired = len(history) - 1
                if draft_offset > desired:
                    cache.trim_prompt_cache(draft_cache, draft_offset - desired)
                    draft_offset = desired
    finally:
        if on_stats:
            on_stats({**policy.snapshot(), "delivered_tokens": generated,
                      "delivered_draft_tokens": delivered_drafts,
                      "cost_measurement_scope": "verified_batches_before_stream_delivery"})


def adaptive_stream_generate(model, tokenizer, prompt, max_tokens=256,
                             draft_model=None, **kwargs):
    """mlx-lm-compatible text stream using the adaptive external verifier."""
    import mlx.core as mx
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if isinstance(prompt, str):
        add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special)
    prompt = mx.array(prompt, mx.uint32)
    if max_tokens == 0:
        return
    detokenizer = tokenizer.detokenizer
    generator = adaptive_generate_step(prompt, model, draft_model, max_tokens=max_tokens, **kwargs)
    with wired_limit(model, [generation_stream]):
        started = time.perf_counter()
        count = 0
        try:
            for token, lp, from_draft in generator:
                if count == 0:
                    prompt_time = time.perf_counter() - started
                    decode_started = time.perf_counter()
                count += 1
                eos = token in tokenizer.eos_token_ids
                if not eos:
                    detokenizer.add_token(token)
                finished = eos or (max_tokens > 0 and count == max_tokens)
                if finished:
                    detokenizer.finalize()
                yield GenerationResponse(
                    text=detokenizer.last_segment, token=token, logprobs=lp,
                    from_draft=from_draft, prompt_tokens=prompt.size,
                    prompt_tps=prompt.size / max(prompt_time, 1e-9),
                    generation_tokens=count,
                    generation_tps=count / max(time.perf_counter() - decode_started, 1e-9),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=("stop" if eos else "length") if finished else None)
                if finished:
                    break
        finally:
            generator.close()

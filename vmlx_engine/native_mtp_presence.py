"""Causal presence penalties for native-MTP target verification.

Proposals keep their own distribution. Only the target distribution needs
the committed history plus the proposals preceding each verification row.
"""

import mlx.core as mx

from .utils.token_logits_processors import make_openai_token_penalty_processor


class NativeMTPPresence:
    def __init__(self, presence_penalty, logit_bias=None, *, processor=None):
        self.presence = float(presence_penalty)
        # Share the AR processor's committed adjustment state. A bias present
        # at prefill and a presence penalty added later must round in that
        # order, rather than recomputing their sum in Python precision.
        self.processor = processor or make_openai_token_penalty_processor(
            presence_penalty=self.presence, logit_bias=logit_bias
        )
        self._shape = None
        self._history = []

    def __call__(self, logits, history, drafts=()):
        width = int(logits.shape[-1])
        shape = (width, logits.dtype)
        if shape != self._shape:
            self._shape = shape
            self._history = []
            self._seen_tokens = set()
            self._seen = mx.zeros((width,), dtype=mx.uint8)

        history = list(history)
        prior = len(self._history)
        if history[:prior] != self._history:
            self._seen_tokens = set()
            self._seen = mx.zeros((width,), dtype=mx.uint8)
            prior = 0
        additions = set(history[prior:]) - self._seen_tokens
        if any(token < 0 or token >= width for token in additions):
            raise ValueError("token context ID is outside vocabulary size")
        if additions:
            self._seen = self._seen.at[
                mx.array(sorted(additions), dtype=mx.int32)
            ].add(1)
            self._seen_tokens.update(additions)
        self._history = history

        # Zero logits expose the exact adjustment without subtraction or
        # another rounding of the target logits. Only committed tokens enter
        # this processor; hypothetical proposals stay in the row-local graph.
        adjustment = self.processor(history, mx.zeros((width,), dtype=logits.dtype))
        seen = self._seen != 0
        columns = mx.arange(width)
        rows = []
        for i in range(int(logits.shape[0])):
            if i:
                # No host read of a proposed token and no mutation of the
                # committed-history mask: rejected proposals cannot persist.
                proposed = columns == drafts[i - 1].reshape(())
                adjustment = adjustment + mx.where(
                    proposed & ~seen,
                    mx.array(-self.presence, dtype=logits.dtype),
                    mx.array(0, dtype=logits.dtype),
                )
                seen = seen | proposed
            rows.append(logits[i] + adjustment)
        return mx.stack(rows)

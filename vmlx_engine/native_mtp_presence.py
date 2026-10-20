"""Causal presence penalties for native-MTP target verification.

Proposals keep their own distribution. Only the target distribution needs
the committed history plus the proposals preceding each verification row.
"""

import mlx.core as mx

from .utils.token_logits_processors import normalize_logit_bias


class NativeMTPPresence:
    def __init__(self, presence_penalty, logit_bias=None):
        self.presence = float(presence_penalty)
        self.bias = normalize_logit_bias(logit_bias)
        self._shape = None
        self._history = []

    def __call__(self, logits, history, drafts=()):
        width = int(logits.shape[-1])
        shape = (width, logits.dtype)
        if shape != self._shape:
            if any(token >= width for token in self.bias):
                raise ValueError("logit_bias token is outside vocabulary size")
            self._unseen_adjustment = mx.zeros((width,), dtype=logits.dtype)
            self._seen_adjustment = mx.full(
                (width,), -self.presence, dtype=logits.dtype
            )
            if self.bias:
                indices = mx.array(sorted(self.bias), dtype=mx.int32)
                # Round each combined adjustment once, like the AR processor.
                self._unseen_adjustment = mx.put_along_axis(
                    self._unseen_adjustment, indices,
                    mx.array([self.bias[t] for t in sorted(self.bias)], dtype=logits.dtype),
                    axis=0,
                )
                self._seen_adjustment = mx.put_along_axis(
                    self._seen_adjustment, indices,
                    mx.array([self.bias[t] - self.presence for t in sorted(self.bias)],
                             dtype=logits.dtype),
                    axis=0,
                )
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

        seen = self._seen != 0
        columns = mx.arange(width)
        rows = []
        for i in range(int(logits.shape[0])):
            if i:
                # No host read of a proposed token and no mutation of the
                # committed-history mask: rejected proposals cannot persist.
                seen = seen | (columns == drafts[i - 1].reshape(()))
            adjustment = mx.where(
                seen, self._seen_adjustment, self._unseen_adjustment
            )
            rows.append(logits[i] + adjustment)
        return mx.stack(rows)

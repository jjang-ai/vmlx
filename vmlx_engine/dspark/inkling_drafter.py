"""Bind an Inkling DSpark head to request-local causal context."""

import mlx.core as mx


class InklingDSparkDrafter:
    def __init__(self, head, target):
        self.head = head.bind(target.model)
        self.block = head.config.block_size
        self.target_layer_ids = tuple(head.config.target_layer_ids)
        self.reset()

    def reset(self, base=0):
        self.cache = self.head.make_cache()
        # A restored target may have more context than the draft cache. The
        # missing rows affect proposals only. Native target verification still
        # sees its complete restored state and decides every committed token.
        self.ctx_len = int(base)
        self._pending = []

    def add_context(self, features):
        self.ctx_len += features.shape[1]
        self._pending.append(features)

    def context_arrays(self):
        return [self._pending, [layer.state for layer in self.cache if layer.keys is not None]]

    def draft_from_anchor(self, anchor, width, sampler, *, draft_pick=None):
        context = (
            self._pending[0]
            if len(self._pending) == 1
            else mx.concatenate(self._pending, axis=1)
        )
        # Match verification may propose a deterministic token. Ratio
        # verification supplies the actual sampling distribution for each step.
        if draft_pick is None:
            draft_pick = lambda logits: mx.argmax(logits, axis=-1)
        tokens = self.head.draft_block(
            anchor, context, self.cache, width + 1, draft_pick,
            token_dtype=anchor.dtype,
        )
        self._pending.clear()
        return tokens.reshape(-1)

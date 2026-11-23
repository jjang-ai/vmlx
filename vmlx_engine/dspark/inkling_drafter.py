"""Bind an Inkling DSpark head to request-local causal context."""

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache


def checkpoint_cache(cache):
    # Return portable local windows without changing the live cache objects.
    snapshot = []
    for entry in cache:
        if isinstance(entry, RotatingKVCache) and entry.keys is not None:
            keys, values = entry.state
            length = keys.shape[2]
            if length > entry.max_size:
                # A multi-token append keeps max_size + append_length - 1
                # chronological rows so every earlier query can see its own
                # window. A checkpoint resumes after the last appended row.
                if (
                    entry.keep != 0
                    or not 0 < entry.max_size < length <= entry.offset
                    or entry._idx != length
                    or values is None
                    or values.shape != keys.shape
                ):
                    return None
                window = entry.max_size
                entry = type(entry).from_state(
                    (keys[..., -window:, :], values[..., -window:, :]),
                    ("0", str(window), str(entry.offset), str(window)),
                )
        snapshot.append(entry)
    return snapshot


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
        self.checkpoint_cut = None
        self.checkpoint = None

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
        cut, self.checkpoint_cut = self.checkpoint_cut, None
        if cut is not None:
            start = self.ctx_len - context.shape[1]
            length = cut - start
            # Refuse to publish a checkpoint when an older target cache
            # restored without the matching draft state.
            if 0 < length < context.shape[1] and all(
                entry.offset == start for entry in self.cache
            ):
                from ..dflash2_session_disk import _detached, _pack

                self.head.append_context(context[:, :length], self.cache)
                snapshot = checkpoint_cache(self.cache)
                if snapshot is not None:
                    arrays, layers = {}, []
                    _pack("d", snapshot, arrays, layers)
                    frozen = {key: _detached(array) for key, array in arrays.items()}
                    mx.eval(list(frozen.values()))
                    self.checkpoint = frozen, layers
                context = context[:, length:]
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

"""Adapt the serving Inkling language model to transactional DSpark rounds."""

import mlx.core as mx


def _features(result):
    return mx.concatenate(result.hidden_states, axis=-1) if result.hidden_states else None


class InklingDSparkTarget:
    max_verify_rows = None

    def __init__(self, model):
        self.model = getattr(model, "language_model", model)

    def make_cache(self):
        return self.model.make_cache()

    def prefill_segment(self, ids, cache, taps, prefill_step=None):
        step = prefill_step or int(ids.shape[1])
        features = []
        for begin in range(0, ids.shape[1], step):
            result = self.model(
                ids[:, begin:begin + step], cache=cache,
                capture_layer_ids=taps or None, skip_logits=True,
            )
            captured = _features(result)
            if captured is not None:
                features.append(captured)
            mx.eval([entry.state for entry in cache], captured)
        return mx.concatenate(features, axis=1) if features else None

    def prefill(self, ids, cache, taps, prefill_step=None):
        prefix = (
            self.prefill_segment(ids[:, :-1], cache, taps, prefill_step)
            if ids.shape[1] > 1 else None
        )
        result = self.model(ids[:, -1:], cache=cache, capture_layer_ids=taps or None)
        features = _features(result)
        if prefix is not None:
            features = mx.concatenate([prefix, features], axis=1)
        return result.logits[0, -1], features

    def decode(self, token, cache):
        return self.model(token.reshape(1, 1), cache=cache).logits[0, -1]

    def ar_step(self, token, cache, taps):
        result = self.model(token.reshape(1, 1), cache=cache, capture_layer_ids=taps)
        return result.logits[0, -1], _features(result)

    def verify(self, ids, cache, taps):
        result = self.model(ids, cache=cache, capture_layer_ids=taps, speculative_verify=True)
        try:
            return result.logits[0], _features(result), result.gdn_states
        except BaseException:
            result.gdn_states.abort()
            raise

    def rollback(self, cache, transaction, *, kept, total):
        transaction.commit(kept)

"""LiLiCorr proposal history belongs to the target request cache, not the model."""
import gc
from types import SimpleNamespace
import weakref

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache


@pytest.fixture
def adapter(monkeypatch):
    from vmlx_engine.models import nemotron_bolt_vl as bolt

    class Drafter:
        layer_ids = [0]
        window = 6
        ctx_end = 0
        received = None

        def reset(self):
            self.ctx_end = 0
            self.received = None

        def add_context(self, rows, start):
            self.received = (start, np.array(rows)[:, 0].tolist())
            self.ctx_end = start + rows.shape[0]

    inner = SimpleNamespace(backbone=SimpleNamespace(layers=[
        SimpleNamespace(norm=SimpleNamespace(weight=mx.zeros((1,))))
    ]))
    monkeypatch.setattr(bolt, "_install_block_tap_hook", lambda: None)

    def forward(self, inputs, cache, inputs_embeds, return_hidden):
        self._rec.put(0, inputs[..., None])
        cache[0].offset += inputs.shape[1]
        return (inputs, inputs) if return_hidden else SimpleNamespace(logits=inputs)

    monkeypatch.setattr(bolt.LanguageModel, "_forward", forward)
    return bolt.LiLiCorrLanguageModel(inner, Drafter())


def run(adapter, cache, values, hidden=True):
    return adapter._forward(mx.array([values], dtype=mx.float32), cache, None, hidden)


def test_model_does_not_retain_request_kv_cache(adapter):
    cache = [KVCache()]
    anchor = weakref.ref(cache[0])
    run(adapter, cache, [10, 11, 12, 13])
    del cache
    gc.collect()
    assert anchor() is None


def test_ssd_restored_request_does_not_inherit_previous_conversation_taps(adapter):
    old = [KVCache()]
    run(adapter, old, [10, 11, 12, 13])
    assert adapter._sync_context(4)
    new = [KVCache()]
    new[0].offset = 4
    run(adapter, new, [94])
    assert adapter._sync_context(5)
    assert adapter.drafter.received == (4, [94])


def test_auxiliary_forward_cannot_replace_request_proposal_history(adapter):
    main = [KVCache()]
    run(adapter, main, [10, 11, 12, 13])
    auxiliary = [KVCache()]
    run(adapter, auxiliary, [80, 81, 82, 83], hidden=False)
    assert adapter._sync_context(4)
    assert adapter.drafter.received == (0, [10, 11, 12, 13])

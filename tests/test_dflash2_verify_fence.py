"""Opt-in verify-forward fence (VMLX_DFLASH2_VERIFY_FENCE): the patched Qwen3.5 model queues the graph every N layers
of a multi-row forward with a GDN sink (a DFlash2 verify), never on one-row steps, never when off, never after the last
layer. Evaluation order only; the layer outputs are unchanged."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl as P


class _Layer:
    is_linear = False

    def __init__(self, k):
        self.k = k

    def __call__(self, h, mask, cache, position_ids=None, position_embeddings=None, gdn_sink=None, n_confirmed=0):
        return h + self.k


class _Model:
    def __init__(self, n=10):
        self.layers = [_Layer(i + 1) for i in range(n)]
        self.fa_idx = 0
        self.ssm_idx = 0

    def embed_tokens(self, ids):
        return mx.zeros((1, int(ids.shape[1]), 4))

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, position_ids=None):
        return "original"


@pytest.fixture
def patched(monkeypatch):
    qlang = SimpleNamespace(Qwen3_5Model=_Model, create_attention_mask=lambda h, c: None, create_ssm_mask=lambda h, c: None)
    P._patch_qwen_model(qlang)
    calls = []
    monkeypatch.setattr(P.mx, "async_eval", lambda *a, **k: calls.append(len(a)))
    monkeypatch.setattr(P, "_VERIFY_FENCE", None)
    yield calls
    monkeypatch.setattr(P, "_VERIFY_FENCE", None)


def test_fence_queues_every_n_layers_on_multirow_sink_forwards_only(patched, monkeypatch):
    monkeypatch.setenv("VMLX_DFLASH2_VERIFY_FENCE", "4")
    assert P.verify_fence_every() == 4
    m = _Model(10)
    out = m(mx.zeros((1, 6), dtype=mx.int32), cache=[None] * 10, gdn_sink=[], return_unnormed=True)
    assert len(patched) == 2                      # after layers 4 and 8, not after layer 10 (the last)
    assert bool(mx.all(out == sum(range(1, 11))).item())
    patched.clear()
    m(mx.zeros((1, 1), dtype=mx.int32), cache=[None] * 10, gdn_sink=[], return_unnormed=True)
    assert patched == []                          # one-row step: no fence
    m(mx.zeros((1, 6), dtype=mx.int32), cache=[None] * 10, gdn_sink=None, return_unnormed=True)
    assert patched == []                          # no sink (prefill): no fence


def test_fence_off_by_default_and_tolerates_bad_values(patched, monkeypatch):
    monkeypatch.delenv("VMLX_DFLASH2_VERIFY_FENCE", raising=False)
    assert P.verify_fence_every() == 0
    _Model(8)(mx.zeros((1, 5), dtype=mx.int32), cache=[None] * 8, gdn_sink=[], return_unnormed=True)
    assert patched == []
    monkeypatch.setattr(P, "_VERIFY_FENCE", None)
    monkeypatch.setenv("VMLX_DFLASH2_VERIFY_FENCE", "x")
    assert P.verify_fence_every() == 0

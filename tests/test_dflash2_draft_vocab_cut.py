"""Opt-in draft-vocabulary cut (dflash2_runtime.install_draft_vocab_cut): the drafter's head projects only the kept
row segments, logits are full-width with -inf elsewhere, kept rows equal the full projection bit for bit, the selector
still returns token ids, and the switch is off by default."""
import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine import dflash2_runtime as D


class _Cfg:
    output_multiplier = 1.0
    final_logit_softcapping = None
    vocab_size = 512


class _Draft:
    def __init__(self, head):
        self.config = _Cfg()
        self.lm_head = head

    def compute_logits(self, hidden):
        return self.lm_head(hidden) * self.config.output_multiplier


def _head(vocab=512, d=64):
    lin = nn.Linear(d, vocab, bias=False)
    lin.weight = (mx.random.normal((vocab, d)) * 0.05).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)


def test_segments_default_off_and_validated(monkeypatch):
    monkeypatch.delenv("VMLX_DFLASH2_DRAFT_VOCAB", raising=False)
    assert D.draft_vocab_segments(248320) is None
    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_VOCAB", "tf")
    assert D.draft_vocab_segments(248320) == ((0, 98304), (248032, 248320))
    assert D.draft_vocab_segments(1000) is None            # a vocabulary the segments do not fit: no cut
    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_VOCAB", "nonsense")
    assert D.draft_vocab_segments(248320) is None


def test_cut_logits_equal_full_projection_on_kept_rows_and_minus_inf_elsewhere():
    mx.random.seed(3)
    draft = _Draft(_head())
    h = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    full = draft.compute_logits(h)
    assert D.install_draft_vocab_cut(draft, ((0, 128), (480, 512)))
    cut = draft.compute_logits(h)
    mx.eval(full, cut)
    assert cut.shape == full.shape and cut.dtype == full.dtype
    assert mx.array_equal(cut[..., :128], full[..., :128]).item()
    assert mx.array_equal(cut[..., 480:], full[..., 480:]).item()
    assert bool(mx.all(cut[..., 128:480] == -math.inf).item())
    assert D.install_draft_vocab_cut(draft, ((0, 1),)) and draft._vmlx_vocab_cut == ((0, 128), (480, 512))  # idempotent


def test_argmax_and_sampling_probs_stay_inside_the_kept_rows():
    import dflash.model_mlx as runtime

    mx.random.seed(5)
    draft = _Draft(_head())
    D.install_draft_vocab_cut(draft, ((0, 128), (480, 512)))
    h = mx.random.normal((1, 7, 64)).astype(mx.bfloat16)
    lg = draft.compute_logits(h)
    ids = mx.argmax(lg, axis=-1)
    q = runtime._sampling_probs(lg, 1.0, 1.0, 0)
    mx.eval(ids, q)
    for t in ids[0].tolist():
        assert t < 128 or t >= 480
    assert bool(mx.all(q[..., 128:480] == 0).item()) and mx.allclose(q.sum(-1), mx.ones_like(q.sum(-1)), atol=1e-3).item()


def test_unknown_head_falls_back_to_the_full_projection():
    class _Dense(_Draft):
        pass

    draft = _Dense(nn.Linear(64, 512, bias=False))
    D.install_draft_vocab_cut(draft, ((0, 128),))
    h = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    out = draft.compute_logits(h)
    assert out.shape == (1, 2, 512) and not bool(mx.any(out == -math.inf).item())


def test_softcap_preserves_excluded_vocabulary_mask():
    draft = _Draft(_head())
    draft.config = _Cfg()
    draft.config.output_multiplier = 1.5
    draft.config.final_logit_softcapping = 2.0
    h = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    full = mx.tanh(draft.lm_head(h) * 1.5 / 2.0) * 2.0
    assert D.install_draft_vocab_cut(draft, ((0, 128), (480, 512)))
    cut = draft.compute_logits(h)
    mx.eval(full, cut)
    assert mx.array_equal(cut[..., :128], full[..., :128]).item()
    assert mx.array_equal(cut[..., 480:], full[..., 480:]).item()
    assert mx.all(cut[..., 128:480] == -math.inf).item()
    assert mx.all(mx.softmax(cut, axis=-1)[..., 128:480] == 0).item()

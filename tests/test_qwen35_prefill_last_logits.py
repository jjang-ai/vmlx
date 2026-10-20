# SPDX-License-Identifier: Apache-2.0
"""Last-row LM head for Qwen3.5 / Qwen3.5-MoE prompt prefill.

The one-shot hybrid prefill lane samples only the final prompt row, but the
LanguageModel used to project every prompt row through the 248k-wide head
(~5% of a cold 8k-32k Ornith-1.5-122B prefill, plus a rows x vocab transient).
``prefill_last_logits_only=True`` keeps the trunk forward unchanged and
projects the last row only.

Contract pinned here, on the REAL vendored + MTP-patched classes:
* the flag yields one logits row;
* every KV / GDN state is bit-identical to the full-logits forward (the trunk
  must not change);
* the kept row matches the full forward's last row (allclose: the head runs
  the one-row kernel instead of the multi-row one) with the same argmax;
* the patched (capture / text-RoPE) path honors it too;
* the engine passes the flag only to a ``__call__`` that carries the marker.
"""

import importlib.util
import os

import mlx.core as mx
import pytest

from vmlx_engine.mllm_batch_generator import (
    _lm_supports_prefill_last_logits,
    _materialize_prefill_cache_state,
    _with_prefill_last_logits,
)

_EQ = os.path.join(os.path.dirname(__file__), "test_hybrid_chunked_prefill_equivalence.py")
_spec = importlib.util.spec_from_file_location("_hybrid_eq_fixture", _EQ)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _cache_arrays(cache):
    out = []
    for c in cache:
        if hasattr(c, "keys") and getattr(c, "keys", None) is not None:
            n = int(c.offset)
            out += [c.keys[..., :n, :], c.values[..., :n, :]]
        else:
            out += [a for a in c.cache if a is not None]
    return out


def _prefill(lm, ids, **kw):
    cache = lm.make_cache()
    lm._position_ids = None
    lm._rope_deltas = None
    out = lm(ids, cache=cache, position_ids=_fixture._abs_position_ids(0, ids.shape[1]), **kw)
    logits = getattr(out, "logits", out)
    _materialize_prefill_cache_state(cache)
    mx.eval(logits)
    return logits, cache


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("length", [37, 64, 129])
def test_last_row_logits_keep_trunk_state_and_last_row(model_type, length):
    lm = _fixture._build_lm(model_type)
    mx.random.seed(length)
    ids = mx.random.randint(0, 199, (1, length)).astype(mx.int32)

    full, cache_full = _prefill(lm, ids)
    last, cache_last = _prefill(lm, ids, prefill_last_logits_only=True)

    assert full.shape[1] == length
    assert last.shape[:2] == (1, 1)
    for a, b in zip(_cache_arrays(cache_full), _cache_arrays(cache_last)):
        assert a.shape == b.shape
        assert mx.array_equal(a, b), "trunk state changed"
    ref = full[:, -1:, :].astype(mx.float32)
    got = last.astype(mx.float32)
    assert mx.allclose(ref, got, atol=2e-2, rtol=2e-2)
    assert int(mx.argmax(ref).item()) == int(mx.argmax(got).item())


def test_patched_path_honors_flag():
    lm = _fixture._build_lm("qwen3_5_moe")
    ids = mx.arange(3, 3 + 45, dtype=mx.int32)[None]
    lm._vmlx_force_text_rope_1d = True  # routes through the patched slow path
    try:
        cache = lm.make_cache()
        out = lm(ids, cache=cache, prefill_last_logits_only=True)
        logits = getattr(out, "logits", out)
        mx.eval(logits)
        assert logits.shape[:2] == (1, 1)
        cache2 = lm.make_cache()
        out2 = lm(ids, cache=cache2)
        full = getattr(out2, "logits", out2)
        mx.eval(full)
        assert full.shape[1] == 45
        assert int(mx.argmax(full[:, -1, :]).item()) == int(mx.argmax(logits[:, -1, :]).item())
    finally:
        lm._vmlx_force_text_rope_1d = False


def test_flag_only_reaches_marked_call():
    lm = _fixture._build_lm("qwen3_5")
    assert _lm_supports_prefill_last_logits(lm)
    assert _with_prefill_last_logits(lm, {"cache": None})["prefill_last_logits_only"] is True

    class Override(type(lm)):
        def __call__(self, inputs, **kwargs):  # a family wrapper that forwards kwargs
            return super().__call__(inputs, **kwargs)

    sub = Override.__new__(Override)
    assert not _lm_supports_prefill_last_logits(sub)
    assert "prefill_last_logits_only" not in _with_prefill_last_logits(sub, {"cache": None})

    class Plain:
        def __call__(self, inputs, **kwargs):
            return inputs

    assert not _lm_supports_prefill_last_logits(Plain())
    assert _with_prefill_last_logits(Plain(), {"a": 1}) == {"a": 1}

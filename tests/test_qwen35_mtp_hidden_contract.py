# SPDX-License-Identifier: Apache-2.0
"""What the Qwen3.5 native-MTP head is fed, and that arming prompt capture never changes the logits.

* Arming prompt capture (priming) routes the forward through the patched slow path. That path must produce the same
  logits as the plain path: the trunk output is normalized exactly once before the head.
* The trunk hidden handed to the MTP head (return_hidden=True outputs and prompt capture) follows one policy per
  model: post-final-norm for routed-MoE Qwen3.5 (Ornith-1.5-122B, measured: teacher-forced acceptance code 71.5 ->
  76.6 %, prose 70.4 -> 73.6 %), pre-norm for dense Qwen3.5 (unmeasured; unchanged). VMLX_QWEN35_MTP_HIDDEN=pre|post
  overrides.
"""

import importlib.util
import os

import mlx.core as mx
import pytest

_EQ = os.path.join(os.path.dirname(__file__), "test_hybrid_chunked_prefill_equivalence.py")
_spec = importlib.util.spec_from_file_location("_hybrid_eq_fixture_hidden", _EQ)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _run(lm, ids, **kw):
    cache = lm.make_cache()
    lm._position_ids = None
    lm._rope_deltas = None
    out = lm(ids, cache=cache, position_ids=_fixture._abs_position_ids(0, ids.shape[1]), **kw)
    return out


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_armed_capture_does_not_change_logits(model_type, monkeypatch):
    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

    lm = _fixture._build_lm(model_type)
    ids = mx.arange(5, 5 + 33, dtype=mx.int32)[None]
    plain = getattr(_run(lm, ids), "logits")
    captured = []
    monkeypatch.setattr(qwen35_vl, "capture_requested", lambda host: True)
    monkeypatch.setattr(qwen35_vl, "capture_prefill", lambda host, inputs, hidden, cache: captured.append(hidden))
    armed = getattr(_run(lm, ids), "logits")
    mx.eval(plain, armed)
    assert captured, "capture path not taken"
    assert mx.array_equal(plain, armed), "arming capture changed the logits"


@pytest.mark.parametrize("model_type,want", [("qwen3_5", "pre"), ("qwen3_5_moe", "post")])
def test_mtp_hidden_policy(model_type, want, monkeypatch):
    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

    monkeypatch.delenv("VMLX_QWEN35_MTP_HIDDEN", raising=False)
    lm = _fixture._build_lm(model_type)
    ids = mx.arange(9, 9 + 21, dtype=mx.int32)[None]
    logits, hidden = _run(lm, ids, return_hidden=True)
    unnormed = lm.model(ids, cache=lm.make_cache(), position_ids=_fixture._abs_position_ids(0, ids.shape[1]),
                        return_unnormed=True)
    normed = lm.model.norm(unnormed)
    mx.eval(hidden, unnormed, normed)
    expect = normed if want == "post" else unnormed
    assert mx.array_equal(hidden, expect)
    captured = []
    monkeypatch.setattr(qwen35_vl, "capture_requested", lambda host: True)
    monkeypatch.setattr(qwen35_vl, "capture_prefill", lambda host, inputs, h, cache: captured.append(h))
    _run(lm, ids)
    mx.eval(captured[0])
    assert mx.array_equal(captured[0], expect)
    other = "pre" if want == "post" else "post"
    monkeypatch.setenv("VMLX_QWEN35_MTP_HIDDEN", other)
    _l, hidden2 = _run(lm, ids, return_hidden=True)
    mx.eval(hidden2)
    assert mx.array_equal(hidden2, unnormed if other == "pre" else normed)

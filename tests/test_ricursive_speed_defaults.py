# SPDX-License-Identifier: Apache-2.0
"""A Ripple bundle gets its fastest DFlash2 settings from a plain load: no env var, no bundle edit."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from vmlx_engine.dflash2_runtime import _TargetAdapter, jangh_dense_speed_first  # noqa: E402
from vmlx_engine.speculative import _bundled_dflash2_target  # noqa: E402


def _bundle(tmp_path, model_type, stamp=None):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    if stamp:
        (tmp_path / "jang_config.json").write_text(json.dumps({"drafter": {"runtime_quantization": stamp}}))
    return tmp_path / "dflash2"


def test_ricursive_declares_the_8_bit_drafter_over_the_bundle_stamp(tmp_path, monkeypatch):
    monkeypatch.delenv("VMLX_DFLASH2_DRAFT_BITS", raising=False)
    assert _bundled_dflash2_target(_bundle(tmp_path, "ricursive", "affine 4-bit g64 at load")) == ("ricursive", 8)


@pytest.mark.parametrize("raw,bits", [("4", 4), ("8", 8), ("16", 16)])
def test_the_env_lever_wins_over_the_declaration(tmp_path, monkeypatch, raw, bits):
    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_BITS", raw)
    assert _bundled_dflash2_target(_bundle(tmp_path, "ricursive"))[1] == bits


@pytest.mark.parametrize("stamp,bits", [(None, 4), ("affine 8-bit g64 at load", 8)])
def test_other_families_keep_their_stamp_or_the_4_bit_default(tmp_path, monkeypatch, stamp, bits):
    monkeypatch.delenv("VMLX_DFLASH2_DRAFT_BITS", raising=False)
    assert _bundled_dflash2_target(_bundle(tmp_path, "qwen3_5_text", stamp))[1] == bits


@pytest.mark.parametrize("raw,model_type,expected", [
    (None, "ricursive", True), (None, "qwen3_5_text", False), (None, None, False),
    ("", "ricursive", True), ("nonsense", "ricursive", True), ("nonsense", "qwen3_5_text", False),
    ("0", "ricursive", False), ("off", "ricursive", False), ("1", "qwen3_5_text", True), ("on", None, True),
])
def test_speed_first_defaults_by_family_and_the_env_overrides_both_ways(monkeypatch, raw, model_type, expected):
    if raw is None:
        monkeypatch.delenv("VMLX_DFLASH2_DENSE_SPEED_FIRST", raising=False)
    else:
        monkeypatch.setenv("VMLX_DFLASH2_DENSE_SPEED_FIRST", raw)
    assert jangh_dense_speed_first(model_type) is expected


@pytest.mark.parametrize("model_type,row_exact", [("ricursive", False), ("qwen3_5_text", True)])
def test_the_adapter_routes_dense_projections_from_the_target_family(monkeypatch, model_type, row_exact):
    monkeypatch.delenv("VMLX_DFLASH2_DENSE_SPEED_FIRST", raising=False)
    projection = SimpleNamespace(is_jangtq2_dense=True, row_exact_verify=None)
    target = SimpleNamespace(model=object(), model_type=model_type, named_modules=lambda: [("mlp.gate_proj", projection)])
    _TargetAdapter(target)
    assert projection.row_exact_verify is row_exact


def test_plain_ar_reports_its_prompt_pass_once_for_the_footer_chips(monkeypatch):
    """With DFlash2 off the footer's pp/s chips still need a receipt, on the first token only."""
    import mlx_lm

    from vmlx_engine.models.llm import MLXLanguageModel

    responses = [SimpleNamespace(text=t, token=i, prompt_tokens=200, prompt_tps=400.0) for i, t in enumerate("abc")]
    monkeypatch.setattr(mlx_lm, "stream_generate", lambda *a, **k: iter(responses))
    model = MLXLanguageModel.__new__(MLXLanguageModel)
    model._loaded = True
    model.model = object()
    model.tokenizer = SimpleNamespace(encode=lambda s: [1, 2, 3], eos_token_ids=set())
    monkeypatch.setattr(model, "dflash2_text_lane", lambda kwargs: False, raising=False)
    monkeypatch.setattr(model, "_create_sampler", lambda *a, **k: None, raising=False)
    outs = list(model.stream_generate("hi", max_tokens=3, temperature=0.0))
    assert [o.prefill_usage is not None for o in outs] == [True, False, False]
    assert outs[0].prefill_usage["tokens"] == 200
    assert outs[0].prefill_usage["seconds"] == pytest.approx(0.5)


@pytest.mark.parametrize("model_type,expected", [("ricursive", True), ("qwen3_5_text", False), (None, False)])
def test_the_drafter_lane_default_follows_the_target_family(monkeypatch, model_type, expected):
    from vmlx_engine.speculative import draft_lane_enabled

    monkeypatch.delenv("VMLX_DFLASH2_DRAFT_LANE", raising=False)
    assert draft_lane_enabled(model_type) is expected


@pytest.mark.parametrize("raw,expected", [("1", True), ("0", False)])
def test_the_drafter_lane_env_overrides_both_ways(monkeypatch, raw, expected):
    from vmlx_engine.speculative import draft_lane_enabled

    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_LANE", raw)
    assert draft_lane_enabled("ricursive") is expected and draft_lane_enabled("qwen3_5_text") is expected


@pytest.mark.parametrize("model_type,expected", [("ricursive", False), ("qwen3_5_text", True), (None, True)])
def test_the_verify_scope_default_follows_the_target_family(monkeypatch, model_type, expected):
    from vmlx_engine.dflash2_runtime import verify_scope_enabled

    monkeypatch.delenv("VMLX_DFLASH2_VERIFY_SCOPE", raising=False)
    assert verify_scope_enabled(model_type) is expected


@pytest.mark.parametrize("raw,expected", [("1", True), ("0", False)])
def test_the_verify_scope_env_overrides_both_ways(monkeypatch, raw, expected):
    from vmlx_engine.dflash2_runtime import verify_scope_enabled

    monkeypatch.setenv("VMLX_DFLASH2_VERIFY_SCOPE", raw)
    assert verify_scope_enabled("ricursive") is expected and verify_scope_enabled("qwen3_5_text") is expected


class _FakeDraft:
    """The drafter surface prewarm uses; its token choice does not matter here."""

    def __init__(self, block_size=8, fail=False):
        self.config = SimpleNamespace(target_layer_ids=[0, 2], block_size=block_size, mask_token_id=1)
        self.fail = fail
        self.rows = []

    def bind(self, adapter):
        self.adapter = adapter

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def propose(self, block, hidden, cache, temperature, logits_start=0):
        import mlx.core as mx

        if self.fail:
            raise RuntimeError("no drafter today")
        self.rows.append(int(block.shape[1]))
        return mx.zeros((1, int(block.shape[1]) - logits_start), dtype=mx.int32), None, None


def _tiny_target():
    import mlx.core as mx
    from vmlx_engine.models.ricursive import ricursive as R

    target = R.Model(R.ModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=5, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=16, vocab_size=1200, sliding_window=32))
    mx.eval(target.parameters())
    return SimpleNamespace(language_model=target)


def test_prewarm_measures_every_width_and_leaves_no_decode_state(monkeypatch):
    from vmlx_engine import dflash2_runtime as D

    monkeypatch.delenv("VMLX_DFLASH2_BLOCK", raising=False)
    monkeypatch.delenv("VMLX_DFLASH2_WIDTHS", raising=False)
    model, draft = _tiny_target(), _FakeDraft()
    measured = D.prewarm_dflash2_target(model, draft)
    assert measured and 1 in measured
    adapter = D._adapter_for(model)
    widths = D._dflash2_block_plan(8, D._lane_flat(adapter), codebook=D._codebook_mlp(adapter))
    assert set(measured) == {1, *widths}
    cost = D._width_costs(adapter, widths)
    assert all(cost[w] > 0 for w in widths) and cost["ar"] > 0
    assert cost["warm"] >= {"ar", *widths}                  # the first request's timings count, nothing is left to compile
    assert not adapter.last_row_logits and not adapter.gdn_states
    # A request's chooser starts from what was measured: nothing unpriced, and it keeps measuring.
    chooser = D._BlockChooser(widths, cost=cost, first_widest=D._codebook_mlp(adapter))
    assert all(chooser._cost_of(w) == cost[w] for w in widths)


def test_prewarm_failure_only_logs_and_seeds_nothing(monkeypatch):
    from vmlx_engine import dflash2_runtime as D

    monkeypatch.delenv("VMLX_DFLASH2_BLOCK", raising=False)
    monkeypatch.delenv("VMLX_DFLASH2_WIDTHS", raising=False)
    model = _tiny_target()
    assert D.prewarm_dflash2_target(model, _FakeDraft(fail=True)) is None
    adapter = D._adapter_for(model)
    widths = D._dflash2_block_plan(8, D._lane_flat(adapter), codebook=D._codebook_mlp(adapter))
    assert not any(w in D._width_costs(adapter, widths) for w in widths)
    assert not adapter.last_row_logits


def test_a_pinned_width_has_nothing_to_prewarm(monkeypatch):
    from vmlx_engine import dflash2_runtime as D

    monkeypatch.setenv("VMLX_DFLASH2_BLOCK", "8")
    assert D.prewarm_dflash2_target(_tiny_target(), _FakeDraft()) is None


@pytest.mark.parametrize("model_type,raw,expected", [("ricursive", None, True), ("qwen3_5_text", None, False),
                                                     ("ricursive", "0", False), ("qwen3_5_text", "1", True)])
def test_prewarm_default_and_env(monkeypatch, model_type, raw, expected):
    from vmlx_engine.dflash2_runtime import prewarm_enabled

    monkeypatch.delenv("VMLX_DFLASH2_PREWARM", raising=False)
    if raw is not None:
        monkeypatch.setenv("VMLX_DFLASH2_PREWARM", raw)
    assert prewarm_enabled(model_type) is expected

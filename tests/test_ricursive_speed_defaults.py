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

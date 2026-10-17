"""Text-only qwen4_exp bundles (no vision_config, no vision_tower.* weights), e.g. the 48 GB JANGHT build."""
import json
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.api import utils as api_utils
from vmlx_engine.models.qwen4_exp.qwen4_exp import Model


def _bundle(tmp_path, **extra):
    # the real condition: an index whose weights verifiably contain no vision tower (the stripped-tower detector
    # then fires; without an index it cannot tell and routing falls through to the MLLM path by accident)
    cfg = {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}, **extra}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    wm = {"language_model.model.embed_tokens.weight": "model-00001-of-00001.safetensors",
          "model.layers.0.mlp.switch_mlp.down_proj.jt_packed": "model-00001-of-00001.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    assert api_utils._checkpoint_lacks_vision_weights(str(tmp_path)) is True
    return str(tmp_path)


@pytest.mark.parametrize("extra", [{}, {"language_model_only": False},
                                   {"capabilities": {"has_vision": False}}, {"vision_config": {}}])
def test_qwen4_exp_routes_to_its_only_loader(tmp_path, extra):
    api_utils._IS_MLLM_CACHE.clear()
    assert api_utils.is_mllm_model(_bundle(tmp_path, **extra)) is True


def test_non_qwen4_text_only_config_still_routes_text(tmp_path):
    api_utils._IS_MLLM_CACHE.clear()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "language_model_only": True}))
    assert api_utils.is_mllm_model(str(tmp_path)) is False


def _text_only_model():
    m = nn.Module()
    m["vision_tower"] = nn.Linear(2, 2)
    m.pop("vision_tower")                     # what the loader does for a text-only bundle
    m.language_model = types.SimpleNamespace(
        model=types.SimpleNamespace(embed_tokens=lambda ids: mx.zeros((1, ids.shape[1], 4))),
        _position_ids=None, _rope_deltas=None)
    return m


def test_text_only_model_embeds_text():
    out = Model.get_input_embeddings(_text_only_model(), mx.array([[1, 2, 3]]))
    assert out.inputs_embeds.shape == (1, 3, 4)


@pytest.mark.parametrize("kw", ["pixel_values", "pixel_values_videos"])
def test_text_only_model_refuses_media_with_a_clear_error(kw):
    with pytest.raises(ValueError, match="text-only"):
        Model.get_input_embeddings(_text_only_model(), mx.array([[1, 2]]), **{kw: mx.zeros((1, 2))})

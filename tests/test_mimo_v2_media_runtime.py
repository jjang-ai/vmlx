from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import mlx.core as mx


def _register_fake_mimo_runtime(monkeypatch, tmp_path):
    from vmlx_engine.models import mllm

    model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}',
        encoding="utf-8",
    )

    class FakeTextConfig:
        model_type = "mimo_v2"

        @classmethod
        def from_dict(cls, params):
            return cls()

    class FakeTextModel:
        def __init__(self, config):
            self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)
            self.layers = []

        def make_cache(self):
            return []

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "jang_tools.mimo_v2.mlx_model":
            return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
        return real_import_module(name, package)

    monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)
    mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
    return sys.modules["mlx_vlm.models.mimo_v2"]


def test_mimo_v2_vision_patch_embed_and_merger_project_to_text_hidden(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    cfg = module.VisionConfig.from_dict(
        {
            "hidden_size": 8,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "in_channels": 3,
            "spatial_merge_size": 2,
        }
    )
    vision = module.VisionModel(cfg)

    patch_width = 3 * 1 * 2 * 2
    patch_embeds = vision.embed_patches(mx.ones((4, patch_width)))
    assert patch_embeds.shape == (4, 8)

    merged = vision.merge_patches(mx.ones((4, 8)))
    assert merged.shape == (1, 16)
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_vision_merger_rejects_non_merge_unit_patch_count(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    cfg = module.VisionConfig.from_dict(
        {
            "hidden_size": 8,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "in_channels": 3,
            "spatial_merge_size": 2,
        }
    )
    vision = module.VisionModel(cfg)

    try:
        vision.merge_patches(mx.ones((3, 8)))
    except ValueError as exc:
        assert "patch count must be divisible" in str(exc)
    else:
        raise AssertionError("MiMo vision merger accepted an invalid patch count")
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)

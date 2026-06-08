from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


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
            def _embed_tokens(input_ids):
                ids = mx.array(input_ids)
                if ids.ndim == 1:
                    ids = ids[None, :]
                return mx.zeros(tuple(ids.shape) + (16,))

            self.model = SimpleNamespace(embed_tokens=_embed_tokens)
            self.layers = []

        def __call__(self, input_ids, inputs_embeds=None, cache=None, mask=None):
            return SimpleNamespace(logits=inputs_embeds)

        def make_cache(self):
            return []

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, strict=True):
            self.loaded_weights = dict(weights)
            return None

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


def test_mimo_v2_vision_attention_blocks_preserve_patch_hidden_shape(
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
            "depth": 2,
            "intermediate_size": 16,
            "num_heads": 2,
            "num_key_value_heads": 1,
            "qk_channels": 4,
            "visual_token_window_size": 2,
            "fullatt_block_indexes": [1],
        }
    )
    vision = module.VisionModel(cfg)

    assert len(vision.blocks) == 2
    hidden = vision.run_blocks(mx.ones((4, 8)))
    assert hidden.shape == (4, 8)
    merged = vision.merge_patches(hidden)
    assert merged.shape == (1, 16)
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_vision_forward_uses_grid_aware_window_reorder(
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
            "depth": 3,
            "intermediate_size": 16,
            "num_heads": 2,
            "num_key_value_heads": 1,
            "qk_channels": 4,
            "visual_token_window_size": 2,
            "vit_window_attn_types": [-1, 1, -1],
            "fullatt_block_indexes": [0],
        }
    )
    vision = module.VisionModel(cfg)

    grid_thw = mx.array([[1, 2, 2]])
    index = vision.get_window_index_1d(grid_thw, col=True)
    assert index.tolist() == [0]
    assert vision.rot_pos_emb(grid_thw).shape == (4, 2)

    pixel_values = mx.ones((4, 3 * 1 * 2 * 2))
    output = vision(pixel_values=pixel_values, grid_thw=grid_thw)
    assert output.shape == (1, 16)
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_model_splices_image_pixels_through_vision_tower(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "multimodal_status": "media_runtime_enabled",
                "vision_config": {
                    "hidden_size": 8,
                    "out_hidden_size": 16,
                    "patch_size": 2,
                    "temporal_patch_size": 1,
                    "in_channels": 3,
                    "spatial_merge_size": 2,
                    "depth": 1,
                    "intermediate_size": 16,
                    "num_heads": 2,
                    "num_key_value_heads": 1,
                    "qk_channels": 4,
                    "fullatt_block_indexes": [0],
                },
                "processor_config": {"image_token_id": 151655},
            }
        )
    )

    input_ids = mx.array([[11, 151655, 22]])
    output = model(
        input_ids,
        pixel_values=mx.ones((4, 3 * 1 * 2 * 2)),
        image_grid_thw=mx.array([[1, 2, 2]]),
    )
    assert output.logits.shape == (1, 3, 16)
    assert output.logits.tolist()[0][0] == [0.0] * 16
    assert output.logits.tolist()[0][2] == [0.0] * 16
    assert any(abs(v) > 0 for v in output.logits.tolist()[0][1])
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_projection_bridge_splices_audio_token(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "multimodal_status": "media_runtime_enabled",
                "audio_config": {
                    "group_size": 2,
                    "input_local_dim": 4,
                    "out_hidden_size": 16,
                    "projection_layers": 2,
                },
                "processor_config": {"audio_token_id": 151669},
            }
        )
    )

    projected = model.audio_encoder(audio_embeds=mx.ones((1, 8)))
    assert projected.shape == (1, 16)

    output = model(
        mx.array([[11, 151669, 22]]),
        audio_embeds=mx.ones((1, 8)),
    )
    assert output.logits.shape == (1, 3, 16)
    assert output.logits.tolist()[0][0] == [0.0] * 16
    assert output.logits.tolist()[0][2] == [0.0] * 16
    assert any(abs(v) > 0 for v in output.logits.tolist()[0][1])
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_stale_text_runtime_metadata_does_not_disable_media_modules(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "multimodal_status": "weights_preserved_text_runtime",
                "vision_config": {
                    "hidden_size": 8,
                    "out_hidden_size": 16,
                    "patch_size": 2,
                    "temporal_patch_size": 1,
                    "in_channels": 3,
                    "spatial_merge_size": 2,
                    "depth": 1,
                    "intermediate_size": 16,
                    "num_heads": 2,
                    "num_key_value_heads": 1,
                    "qk_channels": 4,
                    "fullatt_block_indexes": [0],
                },
                "audio_config": {
                    "audio_channels": 2,
                    "group_size": 2,
                    "input_full_attention": True,
                    "input_local_attn_heads": 1,
                    "input_local_dim": 4,
                    "input_local_head_dim": 4,
                    "input_local_intermediate_size": 8,
                    "input_local_layers": 1,
                    "out_hidden_size": 16,
                    "projection_layers": 2,
                    "speech_vocab_size": 8,
                },
            }
        )
    )

    assert model.visual is not None
    assert model.audio_encoder is not None
    assert model.speech_embeddings is not None
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_codes_run_local_transformer_and_splice_audio_token(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "multimodal_status": "media_runtime_enabled",
                "audio_config": {
                    "audio_channels": 2,
                    "group_size": 2,
                    "input_full_attention": True,
                    "input_local_attn_heads": 1,
                    "input_local_dim": 4,
                    "input_local_head_dim": 4,
                    "input_local_intermediate_size": 8,
                    "input_local_layers": 1,
                    "out_hidden_size": 16,
                    "projection_layers": 2,
                    "speech_vocab_size": 8,
                },
                "processor_config": {"audio_token_id": 151669},
            }
        )
    )

    output = model(
        mx.array([[11, 151669, 22]]),
        audio_codes=mx.array([[1, 2], [3, 4]], dtype=mx.int32),
    )
    assert output.logits.shape == (1, 3, 16)
    assert output.logits.tolist()[0][0] == [0.0] * 16
    assert output.logits.tolist()[0][2] == [0.0] * 16
    assert any(abs(v) > 0 for v in output.logits.tolist()[0][1])
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_residual_vector_quantizer_matches_nearest_codebook(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    quantizer = module.MiMoAudioResidualVectorQuantizer(
        [
            mx.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
            mx.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]),
        ]
    )

    hidden = mx.array([[2.4, 0.1], [0.2, 2.3]])
    codes = quantizer.encode(hidden)

    assert codes.shape == (2, 2)
    assert codes.tolist() == [[1, 2], [1, 2]]
    decoded = quantizer.decode(codes)
    assert decoded.tolist() == [[2.5, 0.0], [0.0, 2.5]]
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_rvq_loader_reads_bundle_codebooks(
    tmp_path,
    monkeypatch,
):
    from safetensors.numpy import save_file

    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"
    audio_dir = bundle / "audio_tokenizer"
    audio_dir.mkdir(parents=True)
    (audio_dir / "config.json").write_text(
        json.dumps(
            {
                "num_quantizers": 2,
                "codebook_size": [3, 3],
                "n_mels": 128,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "encoder.quantizer.vq.layers.0._codebook.embed": np.array(
                [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                dtype=np.float32,
            ),
            "encoder.quantizer.vq.layers.1._codebook.embed": np.array(
                [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                dtype=np.float32,
            ),
        },
        str(audio_dir / "model.safetensors"),
    )

    quantizer = module.load_mimo_audio_rvq_from_bundle(bundle)
    codes = quantizer.encode(mx.array([[2.4, 0.1], [0.2, 2.3]]))

    assert codes.tolist() == [[1, 2], [1, 2]]
    assert quantizer.decode(codes).tolist() == [[2.5, 0.0], [0.0, 2.5]]
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_tokenizer_config_and_mel_segment_plan(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    config = module.MiMoAudioTokenizerConfig.from_dict(
        {
            "n_mels": 2,
            "kernel_size": 3,
            "stride_size": 2,
            "avg_pooler": 2,
            "num_quantizers": 2,
            "codebook_size": [3, 3],
        }
    )

    assert config.get_output_length(mx.array([3, 4, 5])).tolist() == [2, 2, 3]
    assert config.get_code_length(mx.array([3, 4, 5])).tolist() == [1, 1, 2]

    plan = module.plan_mimo_audio_mel_segments(
        [
            mx.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                ]
            ),
            mx.array([[6.0, 60.0], [7.0, 70.0]]),
        ],
        config=config,
        segment_size=3,
    )

    assert plan["input_features"].shape == (7, 2)
    assert plan["input_lens"].tolist() == [3, 2, 2]
    assert plan["segments_per_mel"] == [[3, 2], [2]]
    assert plan["code_lengths"] == [2, 1]
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_audio_tokenizer_executes_mel_to_audio_codes(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    config = module.MiMoAudioTokenizerConfig.from_dict(
        {
            "n_mels": 2,
            "d_model": 4,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 8,
            "encoder_causal": True,
            "encoder_attn_window_size": [4, 0],
            "avg_pooler": 1,
            "num_quantizers": 2,
            "codebook_size": [4, 4],
            "encoder_skip_layer_id": None,
        }
    )
    tokenizer = module.MiMoAudioTokenizer(config)
    tokenizer.encoder.quantizer = module.MiMoAudioResidualVectorQuantizer(
        [
            mx.zeros((4, 4), dtype=mx.float32),
            mx.ones((4, 4), dtype=mx.float32),
        ]
    )

    codes = tokenizer.encode_audio_to_codes(
        [
            mx.array(
                [
                    [0.1, 1.0],
                    [0.2, 1.1],
                    [0.3, 1.2],
                    [0.4, 1.3],
                ],
                dtype=mx.float32,
            )
        ],
        segment_size=4,
    )

    assert len(codes) == 1
    assert codes[0].shape == (config.get_code_length(mx.array([4])).item(), 2)
    assert codes[0].dtype == mx.int32
    assert codes[0].tolist() == [[0, 0], [0, 0]]
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_media_enabled_load_binds_speech_embedding_weights(
    tmp_path,
    monkeypatch,
):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "multimodal_status": "media_runtime_enabled",
                "audio_config": {
                    "audio_channels": 2,
                    "group_size": 2,
                    "input_local_attn_heads": 1,
                    "input_local_dim": 4,
                    "input_local_head_dim": 4,
                    "input_local_intermediate_size": 8,
                    "input_local_layers": 1,
                    "out_hidden_size": 16,
                    "speech_vocab_size": 8,
                },
            }
        )
    )

    replacement = mx.ones((8, 4))
    model.load_weights([("speech_embeddings.0.weight", replacement)])
    assert model.speech_embeddings[0].weight.tolist() == replacement.tolist()
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_sanitize_preserves_media_weights_for_binding(tmp_path, monkeypatch):
    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "vision_config": {"hidden_size": 4},
                "audio_config": {
                    "audio_channels": 1,
                    "group_size": 1,
                    "input_local_attn_heads": 1,
                    "input_local_dim": 4,
                    "input_local_head_dim": 4,
                    "input_local_intermediate_size": 8,
                    "input_local_layers": 1,
                    "out_hidden_size": 16,
                    "speech_vocab_size": 8,
                },
            }
        )
    )
    original_sanitize = model.language_model.sanitize

    def text_only_sanitize(weights):
        return {
            key: value
            for key, value in original_sanitize(weights).items()
            if key.startswith("model.") or key.startswith("lm_head")
        }

    model.language_model.sanitize = text_only_sanitize

    sanitized = model.sanitize(
        {
            "visual.patch_embed.proj.weight": mx.ones((4, 4)),
            "audio_encoder.projection.layers.0.weight": mx.ones((4, 4)),
            "speech_embeddings.0.weight": mx.ones((8, 4)),
            "model.layers.0.input_layernorm.weight": mx.ones((4,)),
        }
    )

    assert "visual.patch_embed.proj.weight" in sanitized
    assert "audio_encoder.projection.layers.0.weight" in sanitized
    assert "speech_embeddings.0.weight" in sanitized
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)


def test_mimo_v2_jangtq_fast_path_binds_indexed_media_weights(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    from vmlx_engine.utils.jang_loader import (
        _bind_mimo_v2_preserved_media_weights_from_index,
    )

    module = _register_fake_mimo_runtime(monkeypatch, tmp_path)
    model = module.Model(
        module.ModelConfig.from_dict(
            {
                "model_type": "mimo_v2",
                "vision_config": {
                    "hidden_size": 4,
                    "out_hidden_size": 16,
                    "patch_size": 2,
                    "temporal_patch_size": 1,
                    "in_channels": 1,
                    "spatial_merge_size": 1,
                    "depth": 1,
                    "num_heads": 2,
                    "num_key_value_heads": 1,
                    "intermediate_size": 8,
                    "fullatt_block_indexes": [0],
                },
                "audio_config": {
                    "audio_channels": 1,
                    "group_size": 1,
                    "input_local_attn_heads": 1,
                    "input_local_dim": 4,
                    "input_local_head_dim": 4,
                    "input_local_intermediate_size": 8,
                    "input_local_layers": 1,
                    "out_hidden_size": 16,
                    "speech_vocab_size": 8,
                },
            }
        )
    )
    bundle = tmp_path / "indexed-media-bundle"
    bundle.mkdir()
    shard = "model-00001-of-00001.safetensors"
    weights = {
        "visual.patch_embed.proj.weight": np.full(
            (4, 1, 1, 2, 2), 1.0, dtype=np.float32
        ),
        "visual.merger.mlp.0.weight": np.full((4, 4), 2.0, dtype=np.float32),
        "visual.merger.mlp.2.weight": np.full((16, 4), 3.0, dtype=np.float32),
        "visual.blocks.0.attn.qkv.weight": np.full(
            (24, 4), 4.0, dtype=np.float32
        ),
        "audio_encoder.projection.mlp.0.weight": np.full(
            (16, 4), 5.0, dtype=np.float32
        ),
        "audio_encoder.projection.mlp.2.weight": np.full(
            (16, 16), 6.0, dtype=np.float32
        ),
        "speech_embeddings.0.weight": np.full((8, 4), 7.0, dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.full((4,), 8.0, dtype=np.float32),
    }
    save_file(weights, str(bundle / shard))
    (bundle / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: shard for key in weights},
            }
        ),
        encoding="utf-8",
    )

    counts = _bind_mimo_v2_preserved_media_weights_from_index(model, bundle)

    assert counts == {"visual": 4, "audio_encoder": 2, "speech_embeddings": 1}
    assert model.visual.patch_embed.proj.weight.tolist() == weights[
        "visual.patch_embed.proj.weight"
    ].reshape((4, 4)).tolist()
    assert model.visual.merger.mlp[0].weight.tolist() == weights[
        "visual.merger.mlp.0.weight"
    ].tolist()
    assert model.visual.merger.mlp[2].weight.tolist() == weights[
        "visual.merger.mlp.2.weight"
    ].tolist()
    assert model.visual.blocks[0].attn.head_dim == 6
    assert model.visual.blocks[0].attn.qkv.weight.tolist() == weights[
        "visual.blocks.0.attn.qkv.weight"
    ].tolist()
    assert model.audio_encoder.projection.mlp[0].weight.tolist() == weights[
        "audio_encoder.projection.mlp.0.weight"
    ].tolist()
    assert model.audio_encoder.projection.mlp[2].weight.tolist() == weights[
        "audio_encoder.projection.mlp.2.weight"
    ].tolist()
    assert model.speech_embeddings[0].weight.tolist() == weights[
        "speech_embeddings.0.weight"
    ].tolist()
    sys.modules.pop("mlx_vlm.models.mimo_v2", None)

# SPDX-License-Identifier: Apache-2.0
"""No-heavy contracts for the source-owned Step-3.7 mlx-vlm runtime pieces."""


def test_step37_model_config_preserves_text_vision_and_projector_quantization():
    from vmlx_engine.models.step3p7_mlx_vlm import (
        ModelConfig,
        TextConfig,
        VisionConfig,
    )

    params = {
        "model_type": "step3p7",
        "image_token_id": 151679,
        "text_config": {
            "model_type": "step3p5",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "use_head_wise_attn_gate": True,
            "layer_types": ["full_attention", "sliding_attention"],
        },
        "vision_config": {
            "model_type": "perception_encoder",
            "hidden_size": 32,
            "patch_size": 14,
            "image_size": 728,
            "use_rope2d": True,
        },
        "projector_config": {"hidden_size": 32, "text_hidden_size": 64},
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "model.layers.0.self_attn.q_proj": {"bits": 4},
            "vision_model.blocks.0.attn.qkv": {"bits": 4},
            "multi_modal_projector.linear_1": {"bits": 8},
        },
    }

    config = ModelConfig.from_dict(params)

    assert isinstance(config.text_config, TextConfig)
    assert isinstance(config.vision_config, VisionConfig)
    assert config.image_token_id == 151679
    assert config.text_config.model_type == "step3p5"
    assert config.text_config.sliding_window == 4096
    assert config.text_config.use_head_wise_attn_gate is True
    assert config.text_config.layer_types == ["full_attention", "sliding_attention"]
    assert config.vision_config.model_type == "perception_encoder"
    assert config.vision_config.patch_size == 14
    assert config.vision_config.use_rope2d is True

    quantization = params["quantization"]
    assert quantization["language_model.model.layers.0.self_attn.q_proj"] == {
        "bits": 4
    }
    assert quantization["vision_model.blocks.0.attn.qkv"] == {"bits": 4}
    assert quantization["multi_modal_projector.linear_1"] == {"bits": 8}


def test_step37_model_config_derives_projector_when_bundle_config_omits_it():
    from vmlx_engine.models.step3p7_mlx_vlm import ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "step3p7",
            "text_config": {"model_type": "step3p5", "hidden_size": 4096},
            "vision_config": {
                "model_type": "perception_encoder",
                "width": 1536,
                "layers": 47,
                "heads": 16,
                "patch_size": 14,
                "image_size": 728,
                "use_rope2d": True,
            },
            "projector_config": None,
        }
    )

    assert config.vision_config.width == 1536
    assert config.vision_config.layers == 47
    assert config.vision_config.heads == 16
    assert config.projector_config.hidden_size == 6144
    assert config.projector_config.text_hidden_size == 4096
    assert config.projector_config.bias is False


def test_step37_model_config_preserves_top_level_projector_bias():
    from vmlx_engine.models.step3p7_mlx_vlm import ModelConfig

    config = ModelConfig.from_dict(
        {
            "projector_bias": True,
            "text_config": {"model_type": "step3p5", "hidden_size": 8},
            "vision_config": {"model_type": "perception_encoder", "width": 4},
            "projector_config": None,
        }
    )

    assert config.projector_config.bias is True


def test_step37_vision_activation_supports_real_bundle_quick_gelu():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import _vision_activation

    x = mx.array([-1.0, 0.0, 1.0])
    y = _vision_activation("quick_gelu")(x)

    assert mx.allclose(y, x * mx.sigmoid(1.702 * x))


def test_step37_vlm_quant_candidates_include_step3p5_sanitized_moe_paths():
    from vmlx_engine.utils.jang_loader import _vlm_quant_module_path_candidates

    assert (
        "model.layers.4.mlp.gate.gate"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.4.mlp.gate.gate",
            "step3p7",
        )
    )
    assert (
        "model.language_model.layers.4.moe.gate"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.4.mlp.gate.gate",
            "step3p7",
        )
    )
    assert (
        "model.layers.4.mlp.switch_mlp.gate_proj"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.4.mlp.switch_mlp.gate_proj",
            "step3p7",
        )
    )
    assert (
        "model.language_model.layers.4.moe.gate_proj"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.4.mlp.switch_mlp.gate_proj",
            "step3p7",
        )
    )
    assert (
        "model.language_model.layers.4.share_expert.up_proj"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.4.mlp.share_expert.up_proj",
            "step3p7",
        )
    )


def test_step37_vlm_gate_dequant_skips_already_quantized_gate_module():
    from types import SimpleNamespace

    from vmlx_engine.utils.jang_loader import _should_dequantize_vlm_gate_weight

    class FakeModel:
        def __init__(self, module):
            self.module = module

        def named_modules(self):
            return [
                (
                    "language_model.model.layers.4.mlp.gate.gate",
                    self.module,
                )
            ]

    quantized_gate = SimpleNamespace(bits=8, group_size=64)
    plain_gate = SimpleNamespace()
    wkey = "language_model.model.layers.4.mlp.gate.gate.weight"

    assert _should_dequantize_vlm_gate_weight(FakeModel(quantized_gate), wkey) is False
    assert _should_dequantize_vlm_gate_weight(FakeModel(plain_gate), wkey) is True


def test_step37_vlm_sanitize_offsets_dense_only_zero_centered_norm_shards():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                    "sliding_window": 64,
                },
                "vision_config": {"hidden_size": 4},
                "projector_config": {"hidden_size": 4, "text_hidden_size": 8},
            }
        )
    )

    sanitized = model.sanitize(
        {
            "model.language_model.layers.0.input_layernorm.weight": mx.array(
                [-0.25, 0.125], dtype=mx.float16
            ),
            "model.language_model.layers.0.self_attn.q_norm.weight": mx.array(
                [0.5], dtype=mx.float16
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.array(
                [1.0], dtype=mx.float16
            ),
        }
    )

    assert sanitized["language_model.model.layers.0.input_layernorm.weight"].tolist() == [
        0.75,
        1.125,
    ]
    assert sanitized["language_model.model.layers.0.self_attn.q_norm.weight"].tolist() == [
        1.5
    ]
    assert sanitized["language_model.model.layers.0.self_attn.q_proj.weight"].tolist() == [
        1.0
    ]


class _ToyStepTokenizer:
    def __init__(self):
        self.vocab = {
            "<im_patch>": 101,
            "<im_start>": 102,
            "<im_end>": 103,
            "<patch_start>": 104,
            "<patch_end>": 105,
            "<patch_newline>": 106,
        }

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def decode(self, tokens):
        return "".join(str(token) for token in tokens)


class _CallableToyStepTokenizer(_ToyStepTokenizer):
    def __call__(self, texts):
        return {"input_text": texts}


class _TensorToyStepTokenizer(_ToyStepTokenizer):
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 2
    eos_token_ids = [2]

    def __call__(self, texts):
        texts = texts if isinstance(texts, list) else [texts]
        rows = []
        for text in texts:
            rows.append(
                [
                    self.vocab["<im_patch>"] if token == "<im_patch>" else 1
                    for token in str(text).split()
                ]
            )
        max_len = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return {"input_ids": padded, "attention_mask": mask}


def test_step37_processor_expands_image_and_patch_placeholders_without_torch():
    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_ToyStepTokenizer())

    text, token_ids = processor.build_image_replacement(
        num_patches=2,
        patch_newline_mask=[False, True],
    )

    assert text.count("<im_patch>") == (
        processor.num_image_feature_size + 2 * processor.num_patch_feature_size
    )
    assert text.startswith("<patch_start>")
    assert "<patch_newline>" in text
    assert text.endswith("<im_end>")
    assert token_ids.count(processor.image_token_id) == text.count("<im_patch>")
    assert token_ids.count(processor.tokenizer.convert_tokens_to_ids("<patch_newline>")) == 1


def test_step37_processor_replaces_one_placeholder_per_image():
    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_ToyStepTokenizer())

    rendered = processor.replace_image_placeholders(
        "look <im_patch> then <im_patch>",
        [
            {"num_patches": 0, "patch_newline_mask": None},
            {"num_patches": 1, "patch_newline_mask": [False]},
        ],
    )

    assert rendered.count("<im_start>") == 2
    assert rendered.count("<patch_start>") == 1
    assert " then <patch_start>" in rendered


def test_step37_processor_rejects_placeholder_image_count_mismatch():
    import pytest

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_ToyStepTokenizer())

    with pytest.raises(ValueError, match="placeholders does not match"):
        processor.replace_image_placeholders(
            "look <im_patch>",
            [
                {"num_patches": 0, "patch_newline_mask": None},
                {"num_patches": 0, "patch_newline_mask": None},
            ],
        )


def test_step37_processor_converts_pil_images_to_pixel_values_without_torch():
    import numpy as np
    from PIL import Image

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_CallableToyStepTokenizer())
    image = Image.fromarray(np.full((80, 80, 3), 128, dtype=np.uint8), mode="RGB")

    batch = processor("see <im_patch>", images=image, return_tensors="mlx")

    assert batch["pixel_values"].shape == (1, 3, 728, 728)
    assert batch["num_patches"] == [0]
    assert batch["input_text"][0].count("<im_patch>") == processor.num_image_feature_size
    assert batch["input_text"][0].startswith("see <im_start>")


def test_step37_processor_accepts_image_file_paths(tmp_path):
    import numpy as np
    from PIL import Image

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_CallableToyStepTokenizer())
    image_path = tmp_path / "red.png"
    Image.fromarray(np.full((80, 80, 3), 255, dtype=np.uint8), mode="RGB").save(
        image_path
    )

    batch = processor("see <im_patch>", images=str(image_path), return_tensors="mlx")

    assert batch["pixel_values"].shape == (1, 3, 728, 728)
    assert batch["num_patches"] == [0]
    assert batch["input_text"][0].count("<im_patch>") == processor.num_image_feature_size


def test_step37_processor_inserts_missing_app_image_placeholder():
    import numpy as np
    from PIL import Image

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_CallableToyStepTokenizer())
    image = Image.fromarray(np.full((80, 80, 3), 255, dtype=np.uint8), mode="RGB")

    batch = processor("What color is the image?", images=image, return_tensors="mlx")

    assert batch["pixel_values"].shape == (1, 3, 728, 728)
    assert batch["num_patches"] == [0]
    assert batch["input_text"][0].startswith("<im_start>")
    assert "What color is the image?" in batch["input_text"][0]
    assert batch["input_text"][0].count("<im_patch>") == processor.num_image_feature_size


def test_step37_processor_return_tensors_mlx_converts_tokenizer_lists():
    import numpy as np
    from PIL import Image

    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_TensorToyStepTokenizer())
    image = Image.fromarray(np.full((80, 80, 3), 255, dtype=np.uint8), mode="RGB")

    batch = processor("<im_patch>", images=image, return_tensors="mlx")

    assert isinstance(batch["input_ids"], mx.array)
    assert isinstance(batch["attention_mask"], mx.array)
    assert batch["input_ids"].dtype == mx.int32
    assert batch["attention_mask"].dtype == mx.int32
    assert batch["input_ids"].shape[0] == 1


def test_step37_jang_vlm_processor_loader_keeps_image_processor(monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image

    from vmlx_engine.utils.jang_loader import _build_step3p7_vlm_processor

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _CallableToyStepTokenizer(),
    )

    processor = _build_step3p7_vlm_processor(tmp_path, eos_token_id=2)
    image = Image.fromarray(np.full((80, 80, 3), 255, dtype=np.uint8), mode="RGB")

    batch = processor("see <im_patch>", images=image, return_tensors="mlx")

    assert type(processor).__name__ == "Step3VLProcessor"
    assert "pixel_values" in batch
    assert batch["pixel_values"].shape == (1, 3, 728, 728)
    assert batch["num_patches"] == [0]
    assert batch["input_text"][0].count("<im_patch>") == processor.num_image_feature_size


def test_step37_processor_emits_patch_pixel_values_for_wide_images():
    import numpy as np
    from PIL import Image

    from vmlx_engine.models.step3p7_mlx_vlm import Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_CallableToyStepTokenizer())
    image = Image.fromarray(np.full((80, 400, 3), 64, dtype=np.uint8), mode="RGB")

    batch = processor("wide <im_patch>", images=image, return_tensors="mlx")

    assert batch["pixel_values"].shape == (1, 3, 728, 728)
    assert batch["patch_pixel_values"].shape[1:] == (3, 504, 504)
    assert batch["num_patches"][0] == batch["patch_pixel_values"].shape[0]
    assert "<patch_start>" in batch["input_text"][0]


def test_step37_model_sanitize_keeps_language_vision_and_projector_weights():
    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    config = ModelConfig.from_dict(
        {
            "text_config": {
                "model_type": "step3p5",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 16,
                "num_attention_heads": 2,
                "num_attention_groups": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "layer_types": ["full_attention"],
                "use_head_wise_attn_gate": True,
            },
            "vision_config": {"hidden_size": 4},
            "projector_config": {"hidden_size": 4, "text_hidden_size": 8},
        }
    )
    model = Model(config)

    weights = {
        "model.language_model.layers.0.moe.gate_proj.weight": "lang",
        "model.vision_model.conv1.weight": "vision",
        "model.vision_model.transformer.resblocks.0.attn.in_proj_weight": "qkv_w",
        "model.vision_model.transformer.resblocks.0.attn.in_proj_bias": "qkv_b",
        "model.vit_large_projector.weight": "projector",
        "model.vit_large_projector.bias": "projector_bias",
        "lm_head.weight": "head",
    }

    sanitized = model.sanitize(weights)

    assert sanitized[
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    ] == "lang"
    assert sanitized["vision_model.conv1.weight"] == "vision"
    assert sanitized["vision_model.transformer.resblocks.0.attn.qkv.weight"] == "qkv_w"
    assert sanitized["vision_model.transformer.resblocks.0.attn.qkv.bias"] == "qkv_b"
    assert sanitized["vit_large_projector.proj.weight"] == "projector"
    assert sanitized["vit_large_projector.proj.bias"] == "projector_bias"
    assert sanitized["language_model.lm_head.weight"] == "head"
    assert all(not key.startswith("model.") for key in sanitized)


def test_step37_model_sanitize_transposes_vision_conv1_weight_to_mlx_layout():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {
                    "width": 4,
                    "num_channels": 3,
                    "patch_size": 2,
                    "image_size": 4,
                },
                "projector_config": {"hidden_size": 4, "text_hidden_size": 8},
            }
        )
    )

    sanitized = model.sanitize(
        {
            "model.vision_model.conv1.weight": mx.zeros(
                (4, 3, 2, 2),
                dtype=mx.float16,
            ),
            "model.vision_model.vit_downsampler1.weight": mx.zeros(
                (8, 4, 3, 3),
                dtype=mx.float16,
            ),
            "model.vision_model.vit_downsampler2.weight": mx.zeros(
                (16, 8, 3, 3),
                dtype=mx.float16,
            ),
        }
    )

    assert sanitized["vision_model.conv1.weight"].shape == (4, 2, 2, 3)
    assert sanitized["vision_model.vit_downsampler1.weight"].shape == (8, 3, 3, 4)
    assert sanitized["vision_model.vit_downsampler2.weight"].shape == (16, 3, 3, 8)


def test_step37_projector_forward_maps_vision_features_to_text_hidden_size():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import ProjectorConfig, Step3p7Projector

    projector = Step3p7Projector(
        ProjectorConfig(hidden_size=6, text_hidden_size=4)
    )

    output = projector(mx.zeros((2, 3, 6)))

    assert output.shape == (2, 3, 4)


def test_step37_process_image_features_uses_mlx_channels_last_downsamplers():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {"hidden_size": 2},
                "projector_config": {"hidden_size": 8, "text_hidden_size": 8},
            }
        )
    )

    output = model._process_image_features(mx.zeros((1, 9, 2)))

    assert output.shape == (1, 1, 8)


def test_step37_vision_patch_embed_accepts_processor_nchw_pixels():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import VisionConfig, VisionModel

    vision = VisionModel(
        VisionConfig.from_dict(
            {
                "width": 4,
                "num_channels": 3,
                "patch_size": 2,
                "image_size": 4,
                "use_abs_posemb": True,
                "use_cls_token": False,
                "use_ln_pre": True,
                "layer_norm_eps": 1e-5,
            }
        )
    )

    hidden_states, grid_hw = vision.patch_embed(mx.zeros((1, 3, 4, 4)))
    mx.eval(hidden_states)

    assert hidden_states.shape == (1, 4, 4)
    assert grid_hw == (2, 2)
    assert vision.base_grid == (2, 2)
    assert vision.positional_embedding.shape == (4, 4)


def test_step37_vision_patch_embed_resizes_abs_posemb_for_patch_grid():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import VisionConfig, VisionModel

    vision = VisionModel(
        VisionConfig.from_dict(
            {
                "width": 4,
                "num_channels": 3,
                "patch_size": 2,
                "image_size": 8,
                "use_abs_posemb": True,
                "use_cls_token": False,
            }
        )
    )

    hidden_states, grid_hw = vision.patch_embed(mx.zeros((1, 3, 4, 4)))
    resized_posemb = vision.sample_abs_posemb(*grid_hw)

    assert grid_hw == (2, 2)
    assert hidden_states.shape == (1, 4, 4)
    assert resized_posemb.shape == (1, 4, 4)


def test_step37_vision_transformer_blocks_preserve_patch_sequence_shape():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import VisionConfig, VisionModel

    vision = VisionModel(
        VisionConfig.from_dict(
            {
                "width": 4,
                "layers": 1,
                "heads": 2,
                "num_channels": 3,
                "patch_size": 2,
                "image_size": 4,
                "use_abs_posemb": False,
                "use_cls_token": False,
                "use_rope2d": False,
                "mlp_ratio": 2,
                "hidden_act": "gelu",
            }
        )
    )

    hidden_states = vision(mx.zeros((1, 3, 4, 4)))

    assert hasattr(vision, "transformer")
    assert len(vision.transformer.resblocks) == 1
    assert hidden_states.shape == (1, 4, 4)


def test_step37_pixel_values_path_projects_vision_features_for_image_tokens():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "image_token_id": 7,
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {
                    "width": 4,
                    "layers": 1,
                    "heads": 1,
                    "num_channels": 3,
                    "patch_size": 2,
                    "image_size": 4,
                    "use_abs_posemb": False,
                    "use_cls_token": False,
                    "use_rope2d": True,
                    "mlp_ratio": 2,
                    "hidden_act": "gelu",
                },
                "projector_config": {"hidden_size": 16, "text_hidden_size": 8},
            }
        )
    )

    embeddings = model.get_multimodal_embeddings(
        pixel_values=mx.zeros((1, 3, 4, 4)),
        num_patches=[0],
    )

    assert embeddings is not None
    assert len(embeddings) == 1
    assert embeddings[0].shape == (1, 8)


def test_step37_processor_placeholder_count_matches_projected_visual_tokens():
    import numpy as np
    from PIL import Image

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig, Step3VLProcessor

    processor = Step3VLProcessor(tokenizer=_CallableToyStepTokenizer())
    image = Image.fromarray(np.full((80, 400, 3), 64, dtype=np.uint8), mode="RGB")
    batch = processor("wide <im_patch>", images=image, return_tensors="mlx")
    placeholder_count = batch["input_text"][0].count("<im_patch>")

    model = Model(
        ModelConfig.from_dict(
            {
                "image_token_id": processor.image_token_id,
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 256,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {
                    "width": 4,
                    "layers": 0,
                    "heads": 1,
                    "num_channels": 3,
                    "patch_size": 14,
                    "image_size": 728,
                    "use_abs_posemb": False,
                    "use_cls_token": False,
                    "use_rope2d": False,
                    "mlp_ratio": 2,
                    "hidden_act": "gelu",
                },
                "projector_config": {"hidden_size": 16, "text_hidden_size": 8},
            }
        )
    )

    embeddings = model.get_multimodal_embeddings(
        pixel_values=batch["pixel_values"],
        patch_pixel_values=batch["patch_pixel_values"],
        num_patches=batch["num_patches"],
    )

    assert embeddings is not None
    assert sum(int(embedding.shape[0]) for embedding in embeddings) == placeholder_count


def _toy_step37_model():
    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    return Model(
        ModelConfig.from_dict(
            {
                "image_token_id": 7,
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {"hidden_size": 4},
                "projector_config": {"hidden_size": 4, "text_hidden_size": 8},
            }
        )
    )


def test_step37_image_embeds_path_merges_over_image_placeholders():
    import mlx.core as mx

    model = _toy_step37_model()
    input_ids = mx.array([[1, 7, 7, 2]])
    image_embeds = mx.ones((2, 8))

    multimodal_embeddings = model.get_multimodal_embeddings(image_embeds=image_embeds)
    merged = model.get_input_embeddings(input_ids, multimodal_embeddings)

    assert merged.shape == (1, 4, 8)
    assert bool(mx.allclose(merged[0, 1:3], image_embeds))


def test_step37_get_input_embeddings_accepts_mlx_vlm_pixel_values_contract():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "image_token_id": 7,
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                },
                "vision_config": {
                    "width": 4,
                    "layers": 1,
                    "heads": 1,
                    "num_channels": 3,
                    "patch_size": 2,
                    "image_size": 4,
                    "use_abs_posemb": False,
                    "use_cls_token": False,
                    "use_rope2d": True,
                    "mlp_ratio": 2,
                    "hidden_act": "gelu",
                },
                "projector_config": {"hidden_size": 16, "text_hidden_size": 8},
            }
        )
    )
    input_ids = mx.array([[1, 7, 2]])

    features = model.get_input_embeddings(
        input_ids,
        pixel_values=mx.zeros((1, 3, 4, 4)),
        num_patches=[0],
        mask=mx.ones((1, 3), dtype=mx.int32),
    )

    assert hasattr(features, "inputs_embeds")
    assert features.inputs_embeds.shape == (1, 3, 8)


def test_step37_language_model_returns_logits_object_for_mlx_vlm_generate():
    import mlx.core as mx

    model = _toy_step37_model()

    output = model.language_model(mx.array([[1, 2]], dtype=mx.int32))

    assert hasattr(output, "logits")
    assert output.cross_attention_states is None
    assert output.encoder_outputs is None
    assert output.logits.shape == (1, 2, 16)


def test_step37_image_embeds_path_rejects_placeholder_count_mismatch():
    import mlx.core as mx
    import pytest

    model = _toy_step37_model()
    input_ids = mx.array([[1, 7, 2]])
    image_embeds = mx.ones((2, 8))

    with pytest.raises(ValueError, match="placeholder count does not match"):
        model.get_input_embeddings(
            input_ids,
            model.get_multimodal_embeddings(image_embeds=image_embeds),
        )


def test_step37_pixel_values_path_rejects_non_array_input_cleanly():
    import pytest

    model = _toy_step37_model()

    with pytest.raises(ValueError, match="must be an MLX array"):
        model.get_multimodal_embeddings(pixel_values=None if False else "pixels")


def test_step37_model_call_without_images_works():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                    "sliding_window": 64,
                },
                "vision_config": {"hidden_size": 4},
                "projector_config": {"hidden_size": 4, "text_hidden_size": 8},
            }
        )
    )

    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32))

    assert logits.shape == (1, 3, 16)


def test_step37_model_call_with_image_inputs_uses_multimodal_merge():
    import mlx.core as mx

    from vmlx_engine.models.step3p7_mlx_vlm import Model, ModelConfig

    model = Model(
        ModelConfig.from_dict(
            {
                "image_token_id": 7,
                "text_config": {
                    "model_type": "step3p5",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 2,
                    "num_attention_groups": 1,
                    "head_dim": 4,
                    "intermediate_size": 16,
                    "sliding_window": 64,
                },
                "vision_config": {"hidden_size": 4, "layers": 0},
                "projector_config": {"hidden_size": 16, "text_hidden_size": 8},
            }
        )
    )

    logits = model(
        mx.array([[1, 7, 2]], dtype=mx.int32),
        pixel_values=mx.zeros((1, 3, 4, 4)),
        num_patches=[0],
    )

    assert logits.shape == (1, 3, 16)

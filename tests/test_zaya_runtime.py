import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.models.zaya import (
    Model,
    ModelArgs,
    ZayaNoStateCache,
    register_mlx_lm_zaya,
)


def _small_args():
    return ModelArgs(
        hidden_size=16,
        num_hidden_layers=4,
        ffn_hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        cca_num_q_heads=4,
        kv_channels=4,
        vocab_size=64,
        num_experts=3,
        zaya_mlp_expansion=8,
        max_position_embeddings=128,
    )


def test_zaya_runtime_registers_as_mlx_lm_model():
    register_mlx_lm_zaya()

    import mlx_lm.models.zaya as zaya

    assert zaya.Model is Model


def test_zaya1_vl_runtime_registers_as_mlx_vlm_model():
    from vmlx_engine.models.zaya1_vl import Model as Zaya1VLModel
    from vmlx_engine.models.zaya1_vl import register_mlx_vlm_zaya1_vl

    register_mlx_vlm_zaya1_vl()

    import mlx_vlm.models.zaya1_vl as zaya1_vl

    assert zaya1_vl.Model is Zaya1VLModel


def test_zaya1_vl_language_model_combines_cca_and_moe_every_layer():
    from vmlx_engine.models.zaya1_vl import TextConfig, Zaya1VLLanguageModel

    args = TextConfig(
        model_type="zaya1_vl",
        hidden_size=16,
        num_hidden_layers=2,
        ffn_hidden_size=32,
        num_experts=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        head_dim=4,
        vocab_size=64,
        zaya_mlp_expansion=8,
    )
    model = Zaya1VLLanguageModel(args)

    assert model.layers[0].attn.self_attn is not None
    assert model.layers[0].mlp.zaya_block is not None
    assert model.layers[0].zaya_block is model.layers[0].mlp.zaya_block

    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache).logits
    mx.eval(logits)

    assert logits.shape == (1, 3, 64)
    assert len(cache) == 2
    assert all(type(c).__name__ == "CacheList" for c in cache)
    assert cache[0][0].offset == 3


def test_zaya1_vl_language_model_vision_lora_path_is_image_mask_gated():
    from vmlx_engine.models.zaya1_vl import TextConfig, Zaya1VLLanguageModel

    args = TextConfig(
        model_type="zaya1_vl",
        hidden_size=16,
        num_hidden_layers=1,
        ffn_hidden_size=32,
        num_experts=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        head_dim=4,
        vocab_size=64,
        zaya_mlp_expansion=8,
        vision_lora=True,
        vision_lora_rank_attn=4,
        vision_lora_rank_mlp=4,
    )
    model = Zaya1VLLanguageModel(args)
    cache = model.make_cache()

    logits = model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        cache=cache,
        image_mask=mx.array([[False, True, False]]),
    ).logits
    mx.eval(logits)

    assert logits.shape == (1, 3, 64)
    assert hasattr(model.layers[0].attn.self_attn.qkv, "lora_linear_q")
    assert hasattr(
        model.layers[0].mlp.zaya_block.experts.local_experts[0],
        "lora_fc1",
    )


def test_zaya_router_mod_skip_expert_has_negative_balancing_bias():
    args = _small_args()
    args.zaya_use_mod = True
    args.num_experts = 3

    from vmlx_engine.models.zaya import ZayaRouter

    router = ZayaRouter(args, layer_idx=1)
    biases = router.balancing_biases.tolist()

    assert biases[:-1] == [0.0, 0.0, 0.0]
    assert biases[-1] == -1.0


def test_zaya1_vl_model_forward_returns_mlx_vlm_language_output():
    from vmlx_engine.models.zaya1_vl import Model, ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "ffn_hidden_size": 32,
            "num_experts": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "zaya_mlp_expansion": 8,
            "vision_config": {"model_type": "qwen2_5_vl"},
        }
    )
    model = Model(config)

    output = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(output.logits)

    assert output.logits.shape == (1, 3, 64)


def test_zaya1_vl_language_model_accepts_mlx_vlm_inputs_embeds_keyword():
    from vmlx_engine.models.zaya1_vl import TextConfig, Zaya1VLLanguageModel

    class Recorder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.seen_input_embeddings = None

        def __call__(self, inputs, cache=None, input_embeddings=None, image_mask=None):
            self.seen_input_embeddings = input_embeddings
            return mx.zeros((inputs.shape[0], inputs.shape[1], self.hidden_size))

    args = TextConfig(
        model_type="zaya1_vl",
        hidden_size=16,
        num_hidden_layers=1,
        ffn_hidden_size=32,
        num_experts=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        head_dim=4,
        vocab_size=64,
        zaya_mlp_expansion=8,
        tie_word_embeddings=False,
    )
    model = Zaya1VLLanguageModel(args)
    recorder = Recorder(args.hidden_size)
    model.model = recorder
    supplied = mx.ones((1, 3, args.hidden_size))

    model(mx.array([[1, 2, 3]], dtype=mx.int32), inputs_embeds=supplied)

    assert recorder.seen_input_embeddings is supplied


def test_zaya1_vl_language_model_matches_affine_quantized_switch_and_router_state():
    from mlx.utils import tree_flatten

    from vmlx_engine.models.zaya1_vl import TextConfig, Zaya1VLLanguageModel

    args = TextConfig(
        model_type="zaya1_vl",
        hidden_size=64,
        num_hidden_layers=2,
        ffn_hidden_size=128,
        num_experts=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_query_groups=2,
        head_dim=16,
        vocab_size=128,
        zaya_mlp_expansion=16,
        vision_lora=True,
        vision_lora_rank_attn=4,
        vision_lora_rank_mlp=4,
        quantization={"bits": 4, "embed_bits": 8, "group_size": 32, "mode": "affine"},
    )
    model = Zaya1VLLanguageModel(args)
    names = {k for k, _ in tree_flatten(model.parameters())}

    assert model.model.embed_tokens.bits == 8
    assert (
        "model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.scales"
        in names
    )
    assert (
        "model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.biases"
        in names
    )
    assert "model.layers.0.mlp.zaya_block.router.router_states_scale" not in names
    assert "model.layers.1.mlp.zaya_block.router.router_states_scale" in names
    assert (
        "model.layers.0.mlp.zaya_block.experts.local_experts.0.lora_fc1.0.weight"
        in names
    )
    assert not any(".layers.0.zaya_block." in name for name in names)


def test_zaya1_vl_model_config_exposes_mlx_vlm_image_token_alias():
    from vmlx_engine.models.zaya1_vl import ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "image_token_id": 262147,
            "vision_config": {"model_type": "qwen2_5_vl"},
        }
    )

    assert config.image_token_id == 262147
    assert config.image_token_index == 262147


def test_zaya1_vl_model_config_merges_empty_text_config_with_flat_bundle_fields():
    from vmlx_engine.models.zaya1_vl import ModelConfig
    from vmlx_engine.models.zaya1_vl import TextConfig

    raw = {
        "model_type": "zaya1_vl",
        "text_config": {},
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "ffn_hidden_size": 4096,
        "num_experts": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_query_groups": 4,
        "head_dim": 128,
        "vision_lora": True,
        "vision_lora_rank_attn": 8,
        "vision_lora_rank_mlp": 32,
        "weight_format": "mxfp4",
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "rotary_base": 1000000,
        "vision_config": {"model_type": "qwen2_5_vl"},
    }
    config = ModelConfig.from_dict(raw)

    assert config.text_config.hidden_size == 2048
    assert config.text_config.num_hidden_layers == 40
    assert config.text_config.vision_lora is True
    assert config.text_config.vision_lora_rank_attn == 8
    assert config.text_config.vision_lora_rank_mlp == 32
    assert config.text_config.rope_theta == 1000000
    assert config.text_config.quantization["embed_bits"] == 8
    assert TextConfig.from_dict(raw["text_config"]).vision_lora is True
    assert TextConfig.from_dict(raw["text_config"]).num_attention_heads == 16
    assert TextConfig.from_dict(raw["text_config"]).num_key_value_heads == 4


def test_zaya1_vl_model_config_normalizes_affine_mxtq_module_mode():
    from vmlx_engine.models.zaya1_vl import ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "ffn_hidden_size": 32,
            "num_experts": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "vision_config": {"model_type": "qwen2_5_vl"},
            "quantization": {
                "bits": 8,
                "embed_bits": 8,
                "group_size": 32,
                "mode": "affine+mxtq",
                "routed_expert_bits": 2,
            },
            "weight_format": "mxtq",
        }
    )

    assert config.text_config.quantization["mode"] == "affine"
    assert config.text_config.quantization["container_mode"] == "affine+mxtq"


def test_jang_loader_quant_block_size_falls_back_to_group_size():
    from vmlx_engine.utils.jang_loader import _jang_default_bits, _jang_quant_block_size

    assert _jang_quant_block_size({"quantization": {"group_size": 32}}) == 32
    assert (
        _jang_quant_block_size(
            {"quantization": {"block_size": 64, "group_size": 32}}
        )
        == 64
    )
    assert _jang_quant_block_size({"quantization": {}}) == 64
    assert _jang_default_bits({"quantization": {"bits": 4, "bit_widths_used": [2]}}) == 4
    assert _jang_default_bits({"quantization": {"bit_widths_used": [2, 4]}}) == 2


def test_public_jang_loaders_accept_mxfp4_weight_format(tmp_path, monkeypatch):
    from vmlx_engine.utils import jang_loader

    model_dir = tmp_path / "ZAYA1-VL-8B-MXFP4"
    model_dir.mkdir()
    (model_dir / "jang_config.json").write_text(
        json.dumps({"version": 2, "weight_format": "mxfp4"})
    )

    calls = []

    monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)

    def fake_vlm(path, jang_cfg, **kwargs):
        calls.append(("vlm", Path(path).name, jang_cfg["weight_format"]))
        return "vlm-model", "processor"

    def fake_text(path, jang_cfg, **kwargs):
        calls.append(("text", Path(path).name, jang_cfg["weight_format"]))
        return "text-model", "tokenizer"

    monkeypatch.setattr(jang_loader, "_load_jang_v2_vlm", fake_vlm)
    monkeypatch.setattr(jang_loader, "_load_jang_v2", fake_text)

    assert jang_loader.is_jang_model(model_dir) is True
    assert jang_loader.load_jang_vlm_model(model_dir) == ("vlm-model", "processor")
    assert jang_loader.load_jang_model(model_dir) == ("text-model", "tokenizer")
    assert calls == [
        ("vlm", "ZAYA1-VL-8B-MXFP4", "mxfp4"),
        ("text", "ZAYA1-VL-8B-MXFP4", "mxfp4"),
    ]


def test_zaya1_vl_jang_vlm_processor_uses_local_inference_template(monkeypatch):
    from vmlx_engine.utils import jang_loader

    calls = []
    fake_processor = SimpleNamespace()

    def fail_stock_processor(*args, **kwargs):
        raise AssertionError("ZAYA1-VL must not use stock training-template processor")

    def fake_local_processor(path, eos_token_id=None):
        calls.append(("local", Path(path).name, eos_token_id))
        return fake_processor

    import mlx_vlm.utils as vlm_utils

    monkeypatch.setattr(vlm_utils, "load_image_processor", lambda path: "image-proc")
    monkeypatch.setattr(vlm_utils, "load_processor", fail_stock_processor)
    monkeypatch.setattr(jang_loader, "_build_vlm_processor", fake_local_processor)

    model = SimpleNamespace(
        config=SimpleNamespace(model_type="zaya1_vl", eos_token_id=262143)
    )
    processor = jang_loader._load_jang_vlm_processor(
        Path("/tmp/ZAYA1-VL-8B-MXFP4"),
        model,
    )

    assert processor is fake_processor
    assert processor.image_processor == "image-proc"
    assert calls == [("local", "ZAYA1-VL-8B-MXFP4", 262143)]


def test_zaya1_vl_exposes_raw_model_alias_for_jangtq_hydration():
    from vmlx_engine.models.zaya1_vl import Model, ModelConfig

    def get_module(root, dotted):
        cur = root
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        return cur

    def set_module(root, dotted, new_mod):
        parts = dotted.split(".")
        cur = root
        for part in parts[:-1]:
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        setattr(cur, parts[-1], new_mod)

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "ffn_hidden_size": 32,
            "num_experts": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "zaya_mlp_expansion": 8,
            "vision_config": {"model_type": "qwen2_5_vl"},
        }
    )
    model = Model(config)
    raw_path = "model.layers.0.zaya_block.experts.switch_mlp.gate_proj"
    canonical = model.language_model.model.layers[
        0
    ].mlp.zaya_block.experts.switch_mlp.gate_proj

    assert get_module(model, raw_path) is canonical

    replacement = nn.Linear(16, 16, bias=False)
    set_module(model, raw_path, replacement)

    assert (
        model.language_model.model.layers[0]
        .mlp.zaya_block.experts.switch_mlp.gate_proj
        is replacement
    )


def test_zaya1_vl_sanitize_maps_prestacked_experts_without_double_mlp():
    from vmlx_engine.models.zaya1_vl import Model, ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "ffn_hidden_size": 32,
            "num_experts": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "zaya_mlp_expansion": 8,
            "vision_config": {"model_type": "qwen2_5_vl"},
        }
    )
    model = Model(config)

    fixed = model.sanitize(
        {
            "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.weight": mx.zeros(
                (3, 16, 16)
            ),
            "model.layers.0.mlp.zaya_block.router.down_proj.weight": mx.zeros(
                (8, 16)
            ),
            "model.layers.0.mlp.zaya_block.experts.local_experts.0.lora_fc1.0.weight": mx.zeros(
                (8, 16)
            ),
        }
    )

    assert (
        "language_model.model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.weight"
        in fixed
    )
    assert "language_model.model.layers.0.mlp.mlp.zaya_block.router.down_proj.weight" not in fixed
    assert (
        "language_model.model.layers.0.mlp.zaya_block.router.down_proj.weight"
        in fixed
    )
    assert (
        "language_model.model.layers.0.mlp.zaya_block.experts.local_experts.0.lora_fc1.0.weight"
        in fixed
    )


def test_zaya1_vl_sanitize_transposes_vision_patch_embed_to_mlx_conv3d_layout():
    from vmlx_engine.models.zaya1_vl import Model, ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "zaya1_vl",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "ffn_hidden_size": 32,
            "num_experts": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "zaya_mlp_expansion": 8,
            "vision_config": {"model_type": "qwen2_5_vl"},
        }
    )
    model = Model(config)
    raw_patch = mx.zeros((1280, 3, 1, 14, 14))

    fixed = model.sanitize({"vision_tower.patch_embed.proj.weight": raw_patch})

    assert fixed["vision_tower.patch_embed.proj.weight"].shape == (1280, 1, 14, 14, 3)


def test_build_vlm_processor_constructs_zaya1_vl_image_processor_when_autoload_falls_back():
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    if not model_dir.exists():
        pytest.skip("local ZAYA1-VL MXFP4 bundle is not present")

    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)

    assert hasattr(processor, "tokenizer")
    assert hasattr(processor, "image_processor")
    assert getattr(processor, "image_token", None) == "<image>"
    assert callable(processor)

    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
    assert "<image>" in prompt
    assert token_ids.count(262147) == 1


def test_zaya1_vl_processor_expands_image_token_to_grid_feature_count():
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    image_path = Path(
        "docs/internal/release-gates/20260508_nemotron_omni_media_live/vmlx_icon.png"
    )
    if not model_dir.exists() or not image_path.exists():
        pytest.skip("local ZAYA1-VL MXFP4 bundle or image fixture is not present")

    from mlx_vlm.utils import prepare_inputs

    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = prepare_inputs(
        processor,
        images=[str(image_path)],
        prompts=prompt,
        image_token_index=262147,
        add_special_tokens=True,
    )

    grid = inputs["image_grid_thw"][0].tolist()
    expected_image_tokens = int(grid[0] * grid[1] * grid[2] // 4)
    assert inputs["input_ids"].flatten().tolist().count(262147) == expected_image_tokens
    assert inputs["pixel_values"].shape[0] == int(grid[0] * grid[1] * grid[2])


def test_zaya1_vl_processor_prefers_chat_template_json_for_inference_prompt():
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    if not (model_dir / "chat_template.json").exists():
        pytest.skip("local ZAYA1-VL MXFP4 chat_template.json is not present")

    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert processor.chat_template == json.loads(
        (model_dir / "chat_template.json").read_text()
    )["chat_template"]
    assert prompt == (
        "<|vision_start|><image><|vision_end|>\n"
        "<|im_start|>user\n"
        "Describe this image.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_zaya1_vl_processor_wraps_string_content_for_list_template():
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    if not (model_dir / "chat_template.json").exists():
        pytest.skip("local ZAYA1-VL MXFP4 chat_template.json is not present")

    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": "What is 17 + 28?"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == (
        "<|im_start|>user\n"
        "What is 17 + 28?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_zaya1_vl_processor_preserves_list_content_for_list_aware_templates(monkeypatch):
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    if not model_dir.exists():
        pytest.skip("local ZAYA1-VL MXFP4 bundle is not present")

    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    captured = {}

    def fake_apply_chat_template(messages_arg, *args, **kwargs):
        captured["content"] = messages_arg[0]["content"]
        captured["chat_template"] = kwargs.get("chat_template")
        return "ok"

    monkeypatch.setattr(
        processor.tokenizer,
        "apply_chat_template",
        fake_apply_chat_template,
    )

    processor.chat_template = (
        "{% for image in message.content|selectattr('type', 'equalto', 'image') %}"
        "{{ image }}{% endfor %}"
    )
    processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(captured["content"], list)
    assert captured["chat_template"] == processor.chat_template

    processor.chat_template = "{{ message.content if message.content is string else '' }}"
    processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert captured["content"] == "<image> Describe this image."
    assert captured["chat_template"] == processor.chat_template


def test_zaya1_vl_thinking_off_prompt_does_not_inject_closed_think(monkeypatch):
    """ZAYA1-VL is a no-thinking product family.

    The local VL template renders a plain ``assistant: `` generation prompt.
    The MLLM wrapper must not synthesize a closed ``<think></think>`` block for
    a family whose metadata says thinking is unsupported.
    """

    from vmlx_engine.models.mllm import MLXMultimodalLM
    import mlx_vlm.prompt_utils as prompt_utils

    model = MLXMultimodalLM.__new__(MLXMultimodalLM)
    model.processor = object()
    model.config = {
        "model_type": "zaya1_vl",
        "capabilities": {"supports_thinking": False},
    }

    monkeypatch.setattr(
        prompt_utils,
        "get_chat_template",
        lambda *args, **kwargs: "user: <image> Describe it.\nassistant: ",
    )

    prompt = model._apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe it."},
                ],
            }
        ],
        enable_thinking=False,
    )

    assert prompt == "user: <image> Describe it.\nassistant: "
    assert "<think>" not in prompt
    assert "</think>" not in prompt


def test_zaya1_vl_affine_vlm_quant_candidates_include_raw_bundle_namespace():
    from vmlx_engine.utils.jang_loader import _vlm_quant_module_path_candidates

    candidates = _vlm_quant_module_path_candidates(
        "language_model.model.layers.0.attn.self_attn.qkv.linear_q",
        "zaya1_vl",
    )

    assert "model.layers.0.attn.self_attn.qkv.linear_q" in candidates
    assert (
        "model.layers.0.zaya_block.router.down_proj"
        in _vlm_quant_module_path_candidates(
            "language_model.model.layers.0.mlp.zaya_block.router.down_proj",
            "zaya1_vl",
        )
    )


def test_batched_engine_zaya1_vl_uses_processor_template_when_prompt_utils_unsupported(monkeypatch):
    """Continuous batching must not drop ZAYA1-VL image placeholders.

    ``mlx_vlm.prompt_utils.apply_chat_template`` does not know every local
    adapter family. ZAYA1-VL's processor/template does know how to render rich
    image/text content, so BatchedEngine should fall back to that local
    processor path before trying the plain text tokenizer fallback.
    """

    from vmlx_engine.engine.batched import BatchedEngine
    import mlx_vlm.prompt_utils as prompt_utils

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    captured = {}

    class FakeProcessor:
        tokenizer = SimpleNamespace()

        def apply_chat_template(self, messages_arg, **kwargs):
            captured["messages"] = messages_arg
            captured["kwargs"] = kwargs
            return (
                "<|vision_start|><image><|vision_end|>\n"
                "<|im_start|>user\n"
                "Describe this image.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    def unsupported_model(*_args, **_kwargs):
        raise ValueError("Unsupported model: zaya1_vl")

    monkeypatch.setattr(prompt_utils, "apply_chat_template", unsupported_model)

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._processor = FakeProcessor()
    engine._model = SimpleNamespace(config={"model_type": "zaya1_vl"})
    engine._model_name = "zaya1-vl-test"
    engine._tokenizer = None

    prompt = engine._apply_chat_template(
        messages,
        num_images=1,
        enable_thinking=False,
    )

    assert "<|vision_start|><image><|vision_end|>" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert isinstance(captured["messages"][0]["content"], list)
    assert captured["messages"][0]["content"][0]["type"] == "image"
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert captured["kwargs"]["tokenize"] is False


def test_batched_engine_zaya1_vl_text_only_keeps_user_text():
    model_dir = Path("/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4")
    if not model_dir.exists():
        pytest.skip("local ZAYA1-VL MXFP4 bundle is not present")

    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.utils.jang_loader import _build_vlm_processor

    processor = _build_vlm_processor(model_dir)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._processor = processor
    engine._model = SimpleNamespace(config={"model_type": "zaya1_vl"})
    engine._model_name = str(model_dir)
    engine._tokenizer = None

    prompt = engine._apply_chat_template(
        [{"role": "user", "content": "What is 17 + 28?"}],
        num_images=0,
        enable_thinking=False,
    )
    ids = processor.tokenizer.encode(prompt, add_special_tokens=False)

    assert "What is 17 + 28?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert len(ids) > 3


def test_mllm_load_registers_zaya1_vl_adapter_before_stock_mlx_vlm_load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "zaya1_vl",
                "vision_config": {"model_type": "qwen2_5_vl"},
            }
        )
    )
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.zaya1_vl", raising=False)

    from vmlx_engine.api import utils as api_utils
    from vmlx_engine.models.mllm import MLXMultimodalLM
    from vmlx_engine.utils import jang_loader
    import mlx_vlm
    import mlx_vlm.utils

    monkeypatch.setattr(api_utils, "resolve_to_local_path", lambda _: tmp_path)
    monkeypatch.setattr(jang_loader, "is_jang_model", lambda _: False)
    monkeypatch.setattr(
        mlx_vlm.utils,
        "load_config",
        lambda *_args, **_kwargs: {"model_type": "zaya1_vl"},
    )
    monkeypatch.setattr(
        jang_loader,
        "_build_vlm_processor",
        lambda _path: SimpleNamespace(replaced_zaya_processor=True),
    )

    def fake_load(_model_name):
        assert "mlx_vlm.models.zaya1_vl" in sys.modules
        return (
            SimpleNamespace(language_model=SimpleNamespace(layers=[])),
            SimpleNamespace(),
        )

    monkeypatch.setattr(mlx_vlm, "load", fake_load)

    model = MLXMultimodalLM(str(tmp_path))
    model.load()

    assert model._loaded is True
    assert model.processor.replaced_zaya_processor is True


def test_zaya_cca_cache_carries_kv_conv_and_prev_hidden_state():
    model = Model(_small_args())
    cache = model.make_cache()

    out = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(out)

    assert out.shape == (1, 3, 64)
    assert cache[0][0].offset == 3
    assert [None if v is None else tuple(v.shape) for v in cache[0][1].cache] == [
        (1, 2, 24),
        (1, 1, 16),
    ]

    out2 = model(mx.array([[4]]), cache=cache)
    mx.eval(out2)

    assert out2.shape == (1, 1, 64)
    assert cache[0][0].offset == 4
    assert [None if v is None else tuple(v.shape) for v in cache[0][1].cache] == [
        (1, 2, 24),
        (1, 1, 16),
    ]


def test_zaya_cca_cache_chunked_prefill_matches_single_shot_logits():
    """CCA state must advance with the sequence, not only the KV cache.

    ZAYA attention owns standard K/V plus path-dependent convolution and
    previous-hidden-state caches. This pins the minimal correctness contract
    for batching/chunked prefill: splitting a prompt at a chunk boundary must
    produce the same logits for the continued segment as single-shot prefill.
    """

    mx.random.seed(7)
    model = Model(_small_args())
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    full_cache = model.make_cache()
    full = model(ids, cache=full_cache)
    mx.eval(full)

    chunked_cache = model.make_cache()
    first = model(ids[:, :2], cache=chunked_cache)
    mx.eval(first)
    second = model(ids[:, 2:], cache=chunked_cache)
    mx.eval(second)

    diff = mx.max(
        mx.abs(full[:, 2:, :].astype(mx.float32) - second.astype(mx.float32))
    )
    assert diff.item() == 0.0
    assert full_cache[0][0].offset == 4
    assert chunked_cache[0][0].offset == 4


def test_zaya_moe_no_state_cache_extracts_and_merges_cleanly():
    cache = ZayaNoStateCache()

    assert cache.state == ()
    assert cache.extract(0).state == ()
    cache.filter([0])
    cache.prepare(lengths=[1])
    cache.finalize()
    cache.advance(1)
    cache.extend(ZayaNoStateCache())
    assert cache.empty()
    assert cache.nbytes == 0
    assert type(ZayaNoStateCache.merge([cache, ZayaNoStateCache()])).__name__ == (
        "ZayaNoStateCache"
    )


def test_zaya_scheduler_extract_preserves_all_layer_slots():
    """ZAYA typed restore must keep the 80-layer CCA/MoE schedule aligned.

    Even layers carry CacheList(KVCache, ArraysCache) CCA state. Odd layers
    carry no recurrent state, but they still occupy a layer slot. Dropping those
    placeholders would reconstruct a 40-layer cache for an 80-layer model.
    """

    model = Model(_small_args())
    cache = model.make_cache()
    out = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(out)

    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    extracted = scheduler._extract_cache_states(cache)

    assert len(extracted) == 4
    assert [row["class_name"] for row in extracted] == [
        "CacheList",
        "ZayaNoStateCache",
        "CacheList",
        "ZayaNoStateCache",
    ]
    assert extracted[1]["no_state"] is True
    assert extracted[3]["no_state"] is True


def test_zaya_no_state_cache_round_trips_through_block_disk_serializer():
    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block

    tensors, _dtype, num_layers = _serialize_block(
        [("no_state", "ZayaNoStateCache")]
    )
    restored = _deserialize_block(dict(tensors), _dtype)

    assert num_layers == 1
    assert restored == [("no_state", "ZayaNoStateCache")]


def test_zaya_prompt_cache_restore_preserves_cca_and_no_state_slots():
    """Prefix/L2 prerequisite: restore KV + CCA state + MoE placeholders.

    This is still a local contract test, not a release claim that ZAYA prefix
    cache is enabled. It proves the core typed pieces can reconstruct a full
    cache list and continue generation identically from that restored boundary.
    """

    mx.random.seed(11)
    model = Model(_small_args())
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    cold_cache = model.make_cache()
    cold_logits = model(ids, cache=cold_cache)
    mx.eval(cold_logits)

    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    extracted = scheduler._extract_cache_states(cold_cache)

    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=8, max_blocks=8),
    )
    table = prefix.store_cache("zaya-restore", [1, 2, 3, 4], extracted)
    block = prefix.paged_cache.allocated_blocks[table.block_ids[0]]
    assert [entry[0] for entry in block.cache_data] == [
        "zaya_cca",
        "no_state",
        "zaya_cca",
        "no_state",
    ]
    assert block.cache_data[0][1][0] == "kv"
    assert block.cache_data[0][2] is not None
    restored_cache = prefix.reconstruct_cache(table)

    assert restored_cache is not None
    assert len(restored_cache) == 4
    assert type(restored_cache[0]).__name__ == "CacheList"
    assert isinstance(restored_cache[1], ZayaNoStateCache)
    assert type(restored_cache[2]).__name__ == "CacheList"
    assert isinstance(restored_cache[3], ZayaNoStateCache)

    next_ids = mx.array([[5]], dtype=mx.int32)
    cold_next = model(next_ids, cache=cold_cache)
    restored_next = model(next_ids, cache=restored_cache)
    mx.eval(cold_next, restored_next)

    diff = mx.max(mx.abs(cold_next.astype(mx.float32) - restored_next.astype(mx.float32)))
    assert diff.item() == 0.0


def test_zaya_typed_cca_block_disk_round_trip_continues_identically():
    mx.random.seed(13)
    model = Model(_small_args())
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    cold_cache = model.make_cache()
    mx.eval(model(ids, cache=cold_cache))

    from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    extracted = scheduler._extract_cache_states(cold_cache)

    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=8, max_blocks=8),
    )
    table = prefix.store_cache("zaya-disk-roundtrip", [1, 2, 3, 4], extracted)
    block = prefix.paged_cache.allocated_blocks[table.block_ids[0]]

    tensors, dtype, num_layers = _serialize_block(block.cache_data)
    restored_block_data = _deserialize_block(dict(tensors), dtype)

    assert num_layers == 4
    assert [entry[0] for entry in restored_block_data] == [
        "zaya_cca",
        "no_state",
        "zaya_cca",
        "no_state",
    ]
    assert restored_block_data[0][2] is not None

    block.cache_data = restored_block_data
    restored_cache = prefix.reconstruct_cache(table)
    assert restored_cache is not None

    next_ids = mx.array([[5]], dtype=mx.int32)
    cold_next = model(next_ids, cache=cold_cache)
    restored_next = model(next_ids, cache=restored_cache)
    mx.eval(cold_next, restored_next)
    diff = mx.max(mx.abs(cold_next.astype(mx.float32) - restored_next.astype(mx.float32)))
    assert diff.item() == 0.0


def test_zaya_typed_cca_l2_disk_hit_restores_full_cache():
    mx.random.seed(17)
    model = Model(_small_args())
    ids = [1, 2, 3, 4]
    ids_arr = mx.array([ids], dtype=mx.int32)

    cold_cache = model.make_cache()
    mx.eval(model(ids_arr, cache=cold_cache))

    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    extracted = scheduler._extract_cache_states(cold_cache)

    tmpdir = tempfile.mkdtemp(prefix="vmlx_zaya_l2_")
    store = BlockDiskStore(tmpdir, max_size_gb=0.1, expected_num_layers=4)
    try:
        prefix = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=PagedCacheManager(
                block_size=8,
                max_blocks=8,
                disk_store=store,
            ),
        )
        prefix.store_cache("zaya-l2-store", ids, extracted)
        store.shutdown()
    finally:
        try:
            store.shutdown()
        except Exception:
            pass

    store2 = BlockDiskStore(tmpdir, max_size_gb=0.1, expected_num_layers=4)
    try:
        prefix2 = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=PagedCacheManager(
                block_size=8,
                max_blocks=8,
                disk_store=store2,
            ),
        )
        table, remaining = prefix2.fetch_cache("zaya-l2-hit", ids)

        assert table is not None
        assert remaining == []
        assert prefix2.paged_cache.stats.disk_hits >= 1

        restored_cache = prefix2.reconstruct_cache(table)
        assert restored_cache is not None

        next_ids = mx.array([[5]], dtype=mx.int32)
        cold_next = model(next_ids, cache=cold_cache)
        restored_next = model(next_ids, cache=restored_cache)
        mx.eval(cold_next, restored_next)
        diff = mx.max(mx.abs(cold_next.astype(mx.float32) - restored_next.astype(mx.float32)))
        assert diff.item() == 0.0
    finally:
        store2.shutdown()


def test_zaya_scheduler_uses_typed_cca_not_hybrid_ssm_companion():
    model = Model(_small_args())

    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    tokenizer = SimpleNamespace(
        eos_token_id=1,
        encode=lambda text, add_special_tokens=False: [1, 2],
        decode=lambda ids: "",
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=True,
            use_memory_aware_cache=False,
            enable_block_disk_cache=False,
            kv_cache_quantization="q4",
        ),
    )

    assert scheduler._uses_zaya_cache is True
    assert scheduler._is_hybrid is True
    assert scheduler._ssm_state_cache is None
    assert scheduler._kv_cache_bits == 0
    assert scheduler.config.kv_cache_quantization == "none"
    assert scheduler.block_aware_cache is not None


def test_zaya_scheduler_forces_prefix_only_to_paged_cache():
    """ZAYA CCA must never use memory-aware/legacy prefix cache stores.

    Those generic paths do not serialize CCA conv_state/prev_hs, and would
    otherwise store decode-contaminated post-generation state. The scheduler
    should defensively route any prefix-only ZAYA configuration to the typed
    paged cache path.
    """

    model = Model(_small_args())

    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    tokenizer = SimpleNamespace(
        eos_token_id=1,
        encode=lambda text, add_special_tokens=False: [1, 2],
        decode=lambda ids: "",
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=False,
            use_memory_aware_cache=False,
            enable_block_disk_cache=False,
            kv_cache_quantization="none",
        ),
    )

    assert scheduler._uses_zaya_cache is True
    assert scheduler.config.use_paged_cache is True
    assert scheduler.config.use_memory_aware_cache is False
    assert scheduler.block_aware_cache is not None
    assert scheduler.memory_aware_cache is None
    assert scheduler.prefix_cache is None


def test_zaya_scheduler_stores_clean_prompt_boundary_for_paged_cache():
    """Live scheduler store must use clean ZAYA CCA prompt-boundary state.

    A finished decode cache contains generated-token CCA conv_state/prev_hs.
    The scheduler must not reuse that post-output state. It should re-prefill
    the N-1 cache key and store that clean typed cache so the next identical
    prompt can hit paged prefix cache.
    """

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    mx.random.seed(21)
    model = Model(_small_args())
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        eos_token_ids=[0],
        clean_up_tokenization_spaces=False,
        encode=lambda text, add_special_tokens=False: [1, 2, 3, 4],
        decode=lambda ids: "".join(chr(65 + (int(i) % 26)) for i in ids),
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            max_num_seqs=1,
            enable_prefix_cache=True,
            use_paged_cache=True,
            use_memory_aware_cache=False,
            paged_cache_block_size=8,
            max_cache_blocks=8,
            kv_cache_quantization="q4",
            prefill_batch_size=1,
            completion_batch_size=1,
            prefill_step_size=8,
        ),
    )

    request = Request(
        request_id="zaya-clean-store",
        prompt=[1, 2, 3, 4],
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )
    scheduler.add_request(request)

    for _ in range(8):
        scheduler.step()
        if request.is_finished():
            break

    assert request.is_finished(), "ZAYA scheduler request did not finish"
    assert scheduler.block_aware_cache is not None
    table, remaining = scheduler.block_aware_cache.fetch_cache(
        "zaya-clean-store-repeat",
        [1, 2, 3, 4],
    )

    assert table is not None
    assert table.num_tokens == 3
    assert remaining == [4]
    assert scheduler.block_aware_cache.paged_cache.stats.cache_hits >= 1


def test_zaya_single_active_store_uses_clean_reprefill_not_snapshot(monkeypatch):
    """Single-active ZAYA stores typed CCA from a clean prompt-only re-prefill.

    ZAYA chat templates can append generation-prompt suffix tokens. A generator
    snapshot captured after normal prefill can therefore represent a later CCA
    boundary than the no-suffix N-1 prefix key. Re-prefill is slower, but it is
    the safe typed-CCA store contract for text ZAYA.
    """

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    mx.random.seed(23)
    model = Model(_small_args())
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        eos_token_ids=[0],
        clean_up_tokenization_spaces=False,
        encode=lambda text, add_special_tokens=False: [1, 2, 3, 4],
        decode=lambda ids: "".join(chr(65 + (int(i) % 26)) for i in ids),
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            max_num_seqs=1,
            enable_prefix_cache=True,
            use_paged_cache=True,
            use_memory_aware_cache=False,
            paged_cache_block_size=8,
            max_cache_blocks=8,
            kv_cache_quantization="none",
            prefill_batch_size=1,
            completion_batch_size=1,
            prefill_step_size=8,
        ),
    )

    original_reprefill = scheduler._prefill_for_prompt_only_cache
    calls = []

    def spy_reprefill(tokens):
        calls.append(list(tokens))
        return original_reprefill(tokens)

    monkeypatch.setattr(scheduler, "_prefill_for_prompt_only_cache", spy_reprefill)

    request = Request(
        request_id="zaya-snapshot-store",
        prompt=[1, 2, 3, 4],
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )
    scheduler.add_request(request)

    for _ in range(8):
        scheduler.step()
        if request.is_finished():
            break

    assert request.is_finished(), "ZAYA scheduler request did not finish"
    assert calls == [[1, 2, 3]]
    assert scheduler.block_aware_cache is not None
    table, remaining = scheduler.block_aware_cache.fetch_cache(
        "zaya-snapshot-store-repeat",
        [1, 2, 3, 4],
    )

    assert table is not None
    assert table.num_tokens == 3
    assert remaining == [4]


def test_zaya_single_active_multitoken_store_uses_clean_reprefill(monkeypatch):
    """Multi-token ZAYA completions still store the clean N-1 prompt boundary."""

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    mx.random.seed(24)
    model = Model(_small_args())
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        eos_token_ids=[0],
        clean_up_tokenization_spaces=False,
        encode=lambda text, add_special_tokens=False: [1, 2, 3, 4],
        decode=lambda ids: "".join(chr(65 + (int(i) % 26)) for i in ids),
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            max_num_seqs=1,
            enable_prefix_cache=True,
            use_paged_cache=True,
            use_memory_aware_cache=False,
            paged_cache_block_size=8,
            max_cache_blocks=8,
            kv_cache_quantization="none",
            prefill_batch_size=1,
            completion_batch_size=1,
            prefill_step_size=8,
        ),
    )

    original_reprefill = scheduler._prefill_for_prompt_only_cache
    calls = []

    def spy_reprefill(tokens):
        calls.append(list(tokens))
        return original_reprefill(tokens)

    monkeypatch.setattr(scheduler, "_prefill_for_prompt_only_cache", spy_reprefill)

    request = Request(
        request_id="zaya-multitoken-snapshot-store",
        prompt=[1, 2, 3, 4],
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
    )
    scheduler.add_request(request)

    for _ in range(8):
        scheduler.step()
        if request.is_finished():
            break

    assert request.is_finished(), "ZAYA scheduler request did not finish"
    assert calls == [[1, 2, 3]]
    table, remaining = scheduler.block_aware_cache.fetch_cache(
        "zaya-multitoken-snapshot-store-repeat",
        [1, 2, 3, 4],
    )

    assert table is not None
    assert table.num_tokens == 3
    assert remaining == [4]


def test_zaya_mllm_scheduler_uses_typed_cca_not_hybrid_ssm_companion():
    """ZAYA-VL is hybrid-shaped but must not use the generic SSM companion.

    Its CacheList(KVCache, ArraysCache) entries are a typed zaya_cca_v1
    contract. Routing them through the Qwen-style SSM companion path records
    paged KV hits as cache wins, then rejects reuse as "no SSM companion".
    """

    from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    language_model = Model(_small_args())
    vlm_model = SimpleNamespace(
        language_model=language_model,
        config=SimpleNamespace(model_type="zaya1_vl"),
    )
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        eos_token_ids=[0],
        name_or_path=None,
        encode=lambda text, add_special_tokens=False: [1, 2],
    )
    processor = SimpleNamespace(tokenizer=tokenizer)

    scheduler = MLLMScheduler(
        vlm_model,
        processor=processor,
        config=MLLMSchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=False,
            use_memory_aware_cache=False,
            enable_block_disk_cache=False,
            kv_cache_quantization="q4",
        ),
    )

    assert scheduler._uses_zaya_cache is True
    assert scheduler._is_hybrid is True
    assert scheduler.config.use_paged_cache is True
    assert scheduler.config.use_memory_aware_cache is False
    assert scheduler.config.kv_cache_quantization == "none"
    assert scheduler._ssm_companion_disk_store is None

    scheduler._ensure_batch_generator()
    assert scheduler.batch_generator._uses_zaya_cache is True
    assert scheduler.batch_generator._is_hybrid is True
    assert scheduler.batch_generator._ssm_companion_enabled is False
    assert scheduler.batch_generator._ssm_state_cache is None


def test_zaya_mllm_paged_store_rederives_clean_prompt_boundary():
    """MLLM ZAYA stores must use clean N-1 CCA state, not decode state."""

    from vmlx_engine.mllm_scheduler import MLLMScheduler

    class FakeBlockAwareCache:
        def __init__(self):
            self.stores = []
            self._request_tables = {}

        def store_cache(self, request_id, token_ids, cache_states, cache_extra_keys=None):
            self.stores.append(
                (request_id, list(token_ids), list(cache_states), cache_extra_keys)
            )

    class FakePagedCacheManager:
        def __init__(self):
            self.detached = []

        def detach_request(self, request_id):
            self.detached.append(request_id)

    class FakeBatchGenerator:
        def __init__(self):
            self.stop_tokens = set()
            self.prefilled = []

        def _prefill_for_clean_path_dependent_cache(self, token_ids):
            self.prefilled.append(list(token_ids))
            return ["clean-zaya-cache"]

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.block_aware_cache = FakeBlockAwareCache()
    scheduler.paged_cache_manager = FakePagedCacheManager()
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler.disk_cache = None
    scheduler.batch_generator = FakeBatchGenerator()
    scheduler.stop_tokens = set()
    scheduler.running = {
        "zaya-vl-clean": SimpleNamespace(
            request_id="zaya-vl-clean",
            _extracted_tokens=[10, 11, 12, 13],
            _extracted_cache=lambda: (_ for _ in ()).throw(
                AssertionError("post-decode ZAYA cache must not be stored")
            ),
            _added_stop_tokens=set(),
            num_output_tokens=1,
        )
    }
    scheduler.request_id_to_uid = {"zaya-vl-clean": 7}
    scheduler.uid_to_request_id = {7: "zaya-vl-clean"}
    scheduler.requests = dict(scheduler.running)
    scheduler.finished_req_ids = set()
    scheduler._is_hybrid = True
    scheduler._uses_zaya_cache = True
    scheduler._kv_cache_bits = 0
    scheduler._cleanup_detokenizer = lambda _request_id: None
    scheduler._mllm_request_has_media_cache_context = lambda _request, _tokens: False
    scheduler._validate_cache = lambda _cache, source: True
    scheduler._extract_cache_states = lambda cache: [
        {"class_name": "CacheList", "state": cache[0]}
    ]

    scheduler._cleanup_finished({"zaya-vl-clean"})

    assert scheduler.batch_generator.prefilled == [[10, 11, 12]]
    assert scheduler.block_aware_cache.stores == [
        (
            "zaya-vl-clean",
            [10, 11, 12],
            [{"class_name": "CacheList", "state": "clean-zaya-cache"}],
            None,
        )
    ]
    assert scheduler.paged_cache_manager.detached == ["zaya-vl-clean"]


def test_zaya_mllm_media_paged_store_skips_prefix_cache():
    """ZAYA-VL media prompts must not store token-prefix CCA cache.

    A media side-key separates different image payloads, but the current clean
    ZAYA CCA store path re-prefills from text tokens only. That loses the
    image-conditioned state, so storing it under any prefix key can make a later
    image request hit a non-image/incorrect CCA state.
    """

    from vmlx_engine.mllm_scheduler import MLLMScheduler

    class FakeBlockAwareCache:
        def __init__(self):
            self.stores = []
            self._request_tables = {}

        def store_cache(
            self,
            request_id,
            token_ids,
            cache_states,
            cache_extra_keys=None,
        ):
            self.stores.append(
                (request_id, list(token_ids), list(cache_states), cache_extra_keys)
            )

    class FakePagedCacheManager:
        def __init__(self):
            self.detached = []

        def detach_request(self, request_id):
            self.detached.append(request_id)

    class FakePromptDiskCache:
        def __init__(self):
            self.stores = []

        def store(self, token_ids, cache_blocks):
            self.stores.append((list(token_ids), list(cache_blocks)))

    class FakeBatchGenerator:
        def __init__(self):
            self.stop_tokens = set()
            self.prefilled = []

        def _prefill_for_clean_path_dependent_cache(self, token_ids):
            self.prefilled.append(list(token_ids))
            return ["clean-zaya-media-cache"]

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.block_aware_cache = FakeBlockAwareCache()
    scheduler.paged_cache_manager = FakePagedCacheManager()
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    scheduler.disk_cache = FakePromptDiskCache()
    scheduler.batch_generator = FakeBatchGenerator()
    scheduler.stop_tokens = set()
    scheduler.running = {
        "zaya-vl-media": SimpleNamespace(
            request_id="zaya-vl-media",
            images=["data:image/png;base64,AAAA"],
            videos=None,
            _cache_extra_keys={"media": "sha256:img-a"},
            _extracted_tokens=[10, 262147, 262147, 11, 12],
            _extracted_cache=lambda: (_ for _ in ()).throw(
                AssertionError("post-decode ZAYA cache must not be stored")
            ),
            _added_stop_tokens=set(),
            num_output_tokens=1,
        )
    }
    scheduler.request_id_to_uid = {"zaya-vl-media": 3}
    scheduler.uid_to_request_id = {3: "zaya-vl-media"}
    scheduler.requests = dict(scheduler.running)
    scheduler.finished_req_ids = set()
    scheduler._is_hybrid = True
    scheduler._uses_zaya_cache = True
    scheduler._kv_cache_bits = 0
    scheduler._cleanup_detokenizer = lambda _request_id: None
    scheduler._mllm_request_has_media_cache_context = lambda _request, _tokens: True
    scheduler._validate_cache = lambda _cache, source: True
    scheduler._extract_cache_states = lambda cache: [
        {"class_name": "CacheList", "state": cache[0]}
    ]

    scheduler._cleanup_finished({"zaya-vl-media"})

    assert scheduler.batch_generator.prefilled == []
    assert scheduler.block_aware_cache.stores == []
    assert scheduler.disk_cache.stores == []


def test_mllm_media_cache_key_ignores_prompt_attention_mask_length():
    """Same media must keep the same cache side-key across longer chat turns."""

    from vmlx_engine.mllm_batch_generator import _mllm_media_cache_extra_keys

    short = SimpleNamespace(
        images=["data:image/png;base64,AAAA"],
        videos=None,
        image_grid_thw=mx.array([[1, 2, 2]]),
        attention_mask=mx.array([[1, 1, 1]]),
        pixel_values=mx.array([[1.0, 2.0]]),
    )
    longer = SimpleNamespace(
        images=["data:image/png;base64,AAAA"],
        videos=None,
        image_grid_thw=mx.array([[1, 2, 2]]),
        attention_mask=mx.array([[1, 1, 1, 1, 1, 1]]),
        pixel_values=mx.array([[9.0, 9.0]]),
    )
    other_image = SimpleNamespace(
        images=["data:image/png;base64,BBBB"],
        videos=None,
        image_grid_thw=mx.array([[1, 2, 2]]),
        attention_mask=mx.array([[1, 1, 1]]),
        pixel_values=mx.array([[1.0, 2.0]]),
    )

    assert _mllm_media_cache_extra_keys(short) == _mllm_media_cache_extra_keys(longer)
    assert _mllm_media_cache_extra_keys(short) != _mllm_media_cache_extra_keys(other_image)


def test_mllm_paged_fetch_uses_request_cache_extra_keys_without_stale_name():
    """Regression for live ZAYA-VL text rows warning on undefined side-key name."""

    source = Path("vmlx_engine/mllm_batch_generator.py").read_text()
    fetch_idx = source.find("self.block_aware_cache.fetch_cache(")
    assert fetch_idx >= 0
    fetch_block = source[fetch_idx : source.find(")", fetch_idx) + 1]

    assert '_cache_extra_keys = getattr(req, "_cache_extra_keys", None)' in source
    assert "cache_extra_keys=_cache_extra_keys" in fetch_block
    assert "cache_extra_keys=_media_cache_extra_keys" not in source


def test_zaya_mllm_extract_cache_states_marks_cachelist_as_tensor_data():
    """MLLM ZAYA CacheList extraction must feed BlockAwarePrefixCache's typed path."""

    from vmlx_engine.mllm_scheduler import MLLMScheduler

    model = Model(_small_args())
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache))

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._detect_n_kv_heads = lambda: 0
    extracted = scheduler._extract_cache_states(cache)

    assert extracted
    assert extracted[0]["class_name"] == "CacheList"
    assert "state" in extracted[0]
    assert "meta_state" in extracted[0]
    assert extracted[0]["state"] is None
    assert "sub_caches" in extracted[0]

    from vmlx_engine.prefix_cache import _is_zaya_cca_cache_list_state

    assert _is_zaya_cca_cache_list_state(extracted[0]) is True


def test_zaya_frugal_store_keeps_terminal_cca_block_in_ram(monkeypatch):
    """Immediate same-process ZAYA hits must not depend on async L2 visibility.

    ZAYA CCA stores per-block KV slices plus terminal conv_state/prev_hs. If
    frugal block-disk mode drops the in-RAM typed block records before the L2
    write is readable, a same-process repeat can find the block table but fail
    to reconstruct the CacheList required for CCA continuation.
    """

    from vmlx_engine.mllm_scheduler import MLLMScheduler
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    class _DummyDisk:
        def write_block_async(self, *_args, **_kwargs):
            return None

    model = Model(_small_args())
    cache = model.make_cache()
    tokens = [1, 2, 3, 4, 5, 6, 7]
    output = model(mx.array([tokens], dtype=mx.int32), cache=cache)
    mx.eval(getattr(output, "logits", output))

    extractor = MLLMScheduler.__new__(MLLMScheduler)
    extractor._detect_n_kv_heads = lambda: 0
    cache_states = extractor._extract_cache_states(cache)

    monkeypatch.delenv("VMLX_PAGED_FRUGAL", raising=False)
    paged = PagedCacheManager(block_size=4, max_blocks=8, disk_store=_DummyDisk())
    pc = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    table = pc.store_cache("zaya-frugal-terminal", tokens, cache_states)

    assert table is not None
    for block_id in table.block_ids:
        assert paged.allocated_blocks[block_id].cache_data is not None
    terminal_block = paged.allocated_blocks[table.block_ids[-1]]
    assert any(
        entry[0] == "zaya_cca" and len(entry) > 2 and entry[2] is not None
        for entry in terminal_block.cache_data
    )
    assert pc.reconstruct_cache(table) is not None


def test_zaya_batch_generator_finishes_with_moe_no_state_cache():
    """Finished ZAYA generations must be able to return prompt_cache.

    mlx-lm slices each layer cache in GenerationBatch.extract_cache() when a
    request reaches max_tokens/stop.  Odd ZAYA MoE layers have no state, so the
    placeholder cache must be extract-safe rather than an ArraysCache full of
    None slots.
    """

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(9)
    model = Model(_small_args())
    generator = BatchGenerator(
        model=model,
        max_tokens=1,
        stop_tokens=[[0]],
        sampler=make_sampler(temp=0.0),
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=4,
    )
    (uid,) = generator.insert([[1, 2, 3]], max_tokens=[1])
    responses = []
    for _ in range(4):
        prompt_responses, generation_responses = generator.next()
        responses.extend(prompt_responses)
        responses.extend(generation_responses)
        if any(
            getattr(r, "uid", None) == uid and getattr(r, "finish_reason", None)
            for r in responses
        ):
            break

    finished = [
        r
        for r in responses
        if getattr(r, "uid", None) == uid and getattr(r, "finish_reason", None)
    ]
    assert finished, "ZAYA request did not finish in the expected one token"
    assert finished[0].prompt_cache is not None
    assert isinstance(finished[0].prompt_cache[1], ZayaNoStateCache)


def test_local_zaya_mxfp4_loads_strictly_when_present():
    model_dir = Path("/Users/example/jang/models/Zyphra/ZAYA1-8B-MXFP4")
    if not model_dir.exists():
        pytest.skip("local ZAYA MXFP4 bundle is not present")

    from vmlx_engine.loaders.load_zaya import load_zaya_model

    model, tokenizer = load_zaya_model(model_dir, lazy=True)

    assert len(model.layers) == 80
    assert type(model.layers[0].self_attn).__name__ == "ZayaCCAAttention"
    assert type(model.layers[1].zaya_block).__name__ == "ZayaMoE"
    assert tokenizer is not None

    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "zaya"


def test_zaya_xml_tool_parser_registered_and_parses_native_format():
    from vmlx_engine.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser("zaya_xml")()
    result = parser.extract_tool_calls(
        "Plan first.\n"
        "<zyphra_tool_call>\n"
        "<function=search_docs>\n"
        "<parameter=query>\n"
        "\"zaya cache\"\n"
        "</parameter>\n"
        "<parameter=limit>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</zyphra_tool_call>"
    )

    assert result.tools_called is True
    assert result.content == "Plan first."
    assert result.tool_calls[0]["name"] == "search_docs"
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "query": "zaya cache",
        "limit": 3,
    }
    assert ToolParserManager.get_tool_parser("zaya").supports_native_format()


def test_zaya_rendered_chat_prompt_encoding_does_not_append_eos():
    """Rendered chat templates already carry BOS/turn markers.

    ZAYA uses GemmaTokenizerFast with add_eos_token=True. Calling encode()
    with tokenizer defaults appends <|im_end|> after the assistant generation
    prompt, so the server stops after one EOS token. The chat path must encode
    rendered templates with add_special_tokens=False.
    """

    class ZayaLikeTokenizer:
        def encode(self, text, add_special_tokens=True):
            ids = [2] + [ord(c) % 1000 for c in text]
            if add_special_tokens:
                ids.append(106)
            return ids

    from vmlx_engine.engine.batched import BatchedEngine
    from vmlx_engine.scheduler import Scheduler

    prompt = "<bos><|im_start|>assistant\n<think>\n</think>\n\n"
    tok = ZayaLikeTokenizer()

    assert Scheduler._encode_prompt_text(tok, prompt, None)[-1] == 106
    assert Scheduler._encode_prompt_text(tok, prompt, False)[-1] != 106
    assert BatchedEngine._encode_rendered_prompt(tok.encode, prompt)[-1] != 106

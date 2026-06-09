# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for upstream mlx_lm compatibility fixes."""

from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten


def test_tokenizer_wrapper_find_clamps_negative_reverse_start():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    compat.install()

    wrapper = TokenizerWrapper.__new__(TokenizerWrapper)

    assert wrapper._find([1, 2], [3, 4, 5], start=-9, reverse=True) == -1
    assert wrapper._find([1, 2, 3], [1, 2], start=-9, reverse=True) == 0


def test_batch_rotating_kv_cache_false_meta_state_roundtrips():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm.models.cache import BatchRotatingKVCache

    compat.install()

    cache = BatchRotatingKVCache(max_size=4, left_padding=[0])
    assert cache.rotated is False
    state = cache.meta_state

    cache.rotated = True
    cache.meta_state = state

    assert cache.rotated is False


def test_lfm2_moe_runtime_patch_uses_sigmoid_route_and_bias_only_for_selection():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm.models import lfm2_moe

    compat.install()

    captured = {}

    class FakeGate:
        def __call__(self, x):
            return mx.array([[[0.0, 4.0, 2.0]]], dtype=mx.float32)

    class FakeSwitchMlp:
        def __call__(self, x, inds):
            captured["indices"] = inds
            return mx.array([[[[10.0], [20.0]]]], dtype=mx.float32)

    block = lfm2_moe.Lfm2MoeSparseMoeBlock.__new__(
        lfm2_moe.Lfm2MoeSparseMoeBlock
    )
    block.top_k = 2
    block.norm_topk_prob = False
    block.use_expert_bias = True
    block.routed_scaling_factor = 2.0
    block.expert_bias = mx.array([0.0, -100.0, 100.0], dtype=mx.float32)
    block.gate = FakeGate()
    block.switch_mlp = FakeSwitchMlp()

    out = block(mx.array([[[1.0]]], dtype=mx.float32))

    mx.eval(out, captured["indices"])
    indices = captured["indices"].tolist()
    assert indices == [[[0, 2]]]

    expected_score_0 = float(1 / (1 + mx.exp(mx.array(0.0))).item()) * 2.0
    expected_score_2 = float((1 / (1 + mx.exp(mx.array(-2.0)))).item()) * 2.0
    assert abs(float(out.item()) - (10.0 * expected_score_0 + 20.0 * expected_score_2)) < 1e-5


def test_gemma4_thinking_detection_patch_skips_channel_mode_when_think_token_present():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm import tokenizer_utils

    compat.install()

    class Tokenizer:
        def get_vocab(self):
            return {"<|channel>": 1, "<channel|>": 2, "<|think|>": 3}

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

    assert tokenizer_utils._infer_thinking(Tokenizer()) == (None, None, None, None)


def test_lfm2_model_args_keeps_routed_scaling_default_for_old_configs():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm.models import lfm2_moe

    compat.install()

    block = lfm2_moe.Lfm2MoeSparseMoeBlock.__new__(
        lfm2_moe.Lfm2MoeSparseMoeBlock
    )
    original_init = lfm2_moe.Lfm2MoeSparseMoeBlock.__init__

    class NoopLinear:
        def __init__(self, *args, **kwargs):
            pass

    class NoopSwitch:
        def __init__(self, *args, **kwargs):
            pass

    old_linear = lfm2_moe.nn.Linear
    old_switch = lfm2_moe.SwitchGLU
    try:
        lfm2_moe.nn.Linear = NoopLinear
        lfm2_moe.SwitchGLU = NoopSwitch
        original_init(
            block,
            SimpleNamespace(
                hidden_size=4,
                moe_intermediate_size=8,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                use_expert_bias=False,
            ),
        )
    finally:
        lfm2_moe.nn.Linear = old_linear
        lfm2_moe.SwitchGLU = old_switch

    assert block.routed_scaling_factor == 1.0


def test_gemma4_video_processor_accepts_hf_config_kwargs():
    import vmlx_engine.runtime_patches.mlx_vlm_compat as compat
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4VideoProcessor

    compat.install()

    processor = Gemma4VideoProcessor(
        patch_size=16,
        pooling_kernel_size=3,
        max_soft_tokens=70,
        num_frames=32,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        do_convert_rgb=True,
        do_sample_frames=True,
        resample=3,
        return_metadata=False,
    )

    assert processor.max_soft_tokens == 70
    assert processor.num_frames == 32


def test_mlx_lm_gemma4_unified_maps_to_text_runtime_and_strips_vision_embedder():
    import vmlx_engine.runtime_patches.mlx_lm_compat as compat
    from mlx_lm import utils
    from mlx_lm.models import gemma4

    compat.install()

    assert utils.MODEL_REMAPPING["gemma4_unified"] == "gemma4"

    model = gemma4.Model.__new__(gemma4.Model)
    model.language_model = SimpleNamespace(sanitize=lambda weights: weights)
    weights = model.sanitize(
        {
            "model.vision_embedder.proj.weight": mx.array([1.0]),
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.array([2.0]),
        }
    )

    assert "vision_embedder.proj.weight" not in weights
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in weights


def test_mlx_vlm_gemma4_shared_kv_layers_drop_unused_kv_modules_and_weights():
    import vmlx_engine.runtime_patches.mlx_vlm_compat as compat
    from mlx_vlm.models import gemma4

    compat.install()

    text_config = gemma4.TextConfig(
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=2,
        sliding_window=32,
        sliding_window_pattern=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={},
    )
    model = gemma4.Model(
        gemma4.ModelConfig(
            text_config=text_config,
            vision_config=gemma4.VisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                global_head_dim=8,
                patch_size=16,
                position_embedding_size=32,
                layer_types=["full_attention"],
                rope_parameters={"rope_theta": 10000.0},
            ),
            model_type="gemma4",
            vocab_size=32,
            image_token_id=30,
            audio_config=None,
        )
    )

    assert hasattr(model.layers[1].self_attn, "k_proj")
    assert not hasattr(model.layers[2].self_attn, "k_proj")
    assert not hasattr(model.layers[3].self_attn, "v_proj")

    weights = model.sanitize(
        {
            "model.language_model.layers.1.self_attn.k_proj.weight": mx.zeros((8, 16)),
            "model.language_model.layers.2.self_attn.k_proj.weight": mx.zeros((8, 16)),
            "model.language_model.layers.2.self_attn.q_proj.weight": mx.zeros((16, 16)),
            "model.language_model.layers.3.self_attn.v_proj.weight": mx.zeros((8, 16)),
        }
    )

    assert "language_model.model.layers.1.self_attn.k_proj.weight" in weights
    assert "language_model.model.layers.2.self_attn.q_proj.weight" in weights
    assert "language_model.model.layers.2.self_attn.k_proj.weight" not in weights
    assert "language_model.model.layers.3.self_attn.v_proj.weight" not in weights


def test_mlx_vlm_gemma4_shared_kv_load_weights_drops_mlx_format_materialized_kv():
    import vmlx_engine.runtime_patches.mlx_vlm_compat as compat
    from mlx_vlm.models import gemma4

    compat.install()

    text_config = gemma4.TextConfig(
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=2,
        sliding_window=32,
        sliding_window_pattern=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={},
    )
    model = gemma4.Model(
        gemma4.ModelConfig(
            text_config=text_config,
            vision_config=gemma4.VisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                global_head_dim=8,
                patch_size=16,
                position_embedding_size=32,
                layer_types=["full_attention"],
                rope_parameters={"rope_theta": 10000.0},
            ),
            model_type="gemma4",
            vocab_size=32,
            image_token_id=30,
            audio_config=None,
        )
    )

    weights = list(tree_flatten(model.parameters(), destination={}).items())
    weights.append(
        (
            "language_model.model.layers.2.self_attn.k_proj.weight",
            mx.zeros((8, 16)),
        )
    )

    assert model.load_weights(weights, strict=True) is model


def test_qwen3_vl_chunked_prefill_slices_deepstack_embeds_to_visual_window():
    import vmlx_engine.runtime_patches.mlx_vlm_compat as compat

    compat.install()

    for module_name in (
        "mlx_vlm.models.qwen3_vl.language",
        "mlx_vlm.models.qwen3_vl_moe.language",
    ):
        module = __import__(module_name, fromlist=["LanguageModel"])
        captured = {}

        class FakeModel:
            def __call__(
                self,
                inputs,
                *,
                cache=None,
                inputs_embeds=None,
                position_ids=None,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            ):
                captured["visual_pos_masks"] = visual_pos_masks
                captured["deepstack_visual_embeds"] = deepstack_visual_embeds
                return mx.zeros((inputs.shape[0], inputs.shape[1], 4))

        class FakeHead:
            def __call__(self, out):
                return out

        language_model = module.LanguageModel.__new__(module.LanguageModel)
        language_model.model = FakeModel()
        language_model.lm_head = FakeHead()
        language_model.args = SimpleNamespace(tie_word_embeddings=False)
        language_model._rope_deltas = None
        language_model._position_ids = None

        visual_pos_masks = mx.array([[1, 0, 1, 1, 0, 1]], dtype=mx.int32)
        deepstack = [mx.arange(32, dtype=mx.float32).reshape(4, 8)]

        language_model(
            mx.array([[10, 11, 12]], dtype=mx.int32),
            position_ids=mx.zeros((3, 1, 3), dtype=mx.int32),
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            n_to_process=2,
        )

        mx.eval(
            captured["visual_pos_masks"],
            captured["deepstack_visual_embeds"][0],
        )
        assert captured["visual_pos_masks"].tolist() == [[1, 1, 0]]
        assert captured["deepstack_visual_embeds"][0].tolist() == deepstack[0][
            1:3
        ].tolist()


def test_lfm25_vl_projector_materializes_layernorm_without_applying_when_disabled():
    import vmlx_engine.runtime_patches.mlx_vlm_compat as compat
    from mlx_vlm.models.lfm2_vl import lfm2_vl

    compat.install()

    projector = lfm2_vl.Lfm2VlMultiModalProjector(
        SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=2),
            text_config=SimpleNamespace(hidden_size=2),
            downsample_factor=1,
            projector_use_layernorm=False,
            projector_hidden_size=2,
            projector_bias=False,
        )
    )

    captured = {}

    class ExplodingLayerNorm:
        def __call__(self, x):
            raise AssertionError("disabled LFM2.5 VL layernorm should not run")

    class FakeLinear:
        def __init__(self, name):
            self.name = name

        def __call__(self, x):
            captured[self.name] = x
            return x

    assert not isinstance(projector.layer_norm, lfm2_vl.nn.Identity)
    assert projector.projector_use_layernorm is False

    projector.layer_norm = ExplodingLayerNorm()
    projector.linear_1 = FakeLinear("linear_1")
    projector.linear_2 = FakeLinear("linear_2")

    x = mx.ones((1, 2), dtype=mx.float32)
    out = projector(x)

    mx.eval(out, captured["linear_1"])
    assert captured["linear_1"].tolist() == x.tolist()

# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for upstream mlx_lm compatibility fixes."""

from types import SimpleNamespace

import mlx.core as mx


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

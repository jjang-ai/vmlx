# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-V-4.7 (minicpmv4_7) runtime: canvas M-RoPE goldens, sanitize, numerics,
parsers on the transformers ground-truth outputs, registry and engine hooks.

Goldens in tests/fixtures/minicpmv47_canvas_goldens.npz were produced by
transformers 5.19 ``MiniCPMV4_7Model.get_rope_index`` on real processor output
of MiniCPM-V-4.7-1B (images, multi-row slices, 4x, sliced video, left-padded
batch). No model weights are needed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.minicpmv4_7.register import register_minicpmv4_7_runtime

register_minicpmv4_7_runtime()

from vmlx_engine.models.minicpmv4_7.canvas import (  # noqa: E402
    canvas_rope_index,
    group_visual_frames,
    torch_linspace_round,
)
from vmlx_engine.models.minicpmv4_7.config import ModelConfig, text_model_is_moe  # noqa: E402
from vmlx_engine.models.minicpmv4_7.language import (  # noqa: E402
    QWEN35_QK_L2NORM_EPS,
    ZeroCenteredRMSNorm,
)
from vmlx_engine.models.minicpmv4_7.minicpmv4_7 import Model  # noqa: E402
from vmlx_engine.models.minicpmv4_7.vision import (  # noqa: E402
    gelu_erf,
    gelu_pytorch_tanh,
    nearest_position_ids,
    window_permutation,
)

FIXTURE = Path(__file__).parent / "fixtures" / "minicpmv47_canvas_goldens.npz"
MARKERS = (248088, 248089, 248078, 248079, 198)  # slice_start, slice_end, image_start, image_end, "\n"
IMAGE_PAD, VIDEO_PAD = 248056, 248057

# Raw greedy outputs of MiniCPM-V-4.7-1B BF16 in transformers 5.19 (torch,
# special tokens kept), trailing <|im_end|> removed as the server's stop
# handling does.
GT_THINK_ON = (
    "Okay, let me calculate 17 times 23. Hmm, I'll start by breaking it down. 17 times 20 is 340, "
    "and 17 times 3 is 51. Then add those together: 340 + 51 = 391. Wait, let me check that again to "
    "be sure. Alternatively, 17*20 is 340, and 17*3 is 51, so 340+51 is 391. Yep, that seems right.\n"
    "</think>\n\nTo find 17 × 23:  \n- First, calculate 17 × 20 = 340  \n- Then, 17 × 3 = 51  \n"
    "- Add the results: 340 + 51 = 391  \n\nAnswer: 391"
)
GT_THINK_OFF = "391"
GT_TOOL = (
    "I need to call the get_weather function to find out the current weather in Paris in Celsius.\n\n"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=unit>\nc\n"
    "</parameter>\n</function>\n</tool_call>"
)
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    }
]


# ----------------------------------------------------------------- tiny model
def _text_config(moe: bool) -> dict:
    cfg = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "hidden_size": 16,
        "intermediate_size": 24,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248100,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "full_attention_interval": 4,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [1, 1, 0],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    }
    if moe:
        cfg.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
        )
    return cfg


def _tiny_model(moe: bool = False, mode: str = "16x") -> Model:
    from mlx_vlm.models.minicpmv4_7 import TextConfig, VisionConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "minicpmv4_7",
            "text_config": _text_config(moe),
            "vision_config": {},
            "insert_layer_id": 0,
            "downsample_mode": mode,
        }
    )
    config.text_config = TextConfig.from_dict(_text_config(moe))
    config.vision_config = VisionConfig.from_dict(
        {
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 56,
            "patch_size": 14,
        }
    )
    return Model(config)


def _hf_name(key: str) -> str:
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model.") :]
    if key.startswith("language_model.lm_head"):
        return key[len("language_model.") :]
    return "model." + key


def _hf_checkpoint_from(model: Model) -> dict:
    """Invert sanitize: the model's own parameters under HF names and layouts."""
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    out = {}
    for key, value in params.items():
        if ".switch_mlp." in key:
            continue
        if key.endswith("embeddings.patch_embedding.weight"):
            value = value.transpose(0, 3, 1, 2)
        elif key.endswith("conv1d.weight"):
            value = value.moveaxis(2, 1)
        out[_hf_name(key)] = value
    for key in params:
        if key.endswith(".switch_mlp.gate_proj.weight"):
            prefix = key[: -len("switch_mlp.gate_proj.weight")]
            gate = params[key]
            up = params[prefix + "switch_mlp.up_proj.weight"]
            out[_hf_name(prefix + "experts.gate_up_proj")] = mx.concatenate([gate, up], axis=1)
            out[_hf_name(prefix + "experts.down_proj")] = params[prefix + "switch_mlp.down_proj.weight"]
    return out


def _randomize(model: Model, seed: int = 0) -> None:
    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(seed)
    params = tree_flatten(model.parameters())
    model.update(tree_unflatten([(k, mx.random.normal(v.shape) * 0.05) for k, v in params]))


# ------------------------------------------------------------- canvas goldens
def _golden_cases():
    data = np.load(FIXTURE)
    names = sorted({k.split("__")[0] for k in data.files})
    return data, names


_DATA, _CASES = _golden_cases()


@pytest.mark.parametrize("case", _CASES)
def test_canvas_rope_index_matches_transformers_golden(case):
    d = {k.split("__")[1]: _DATA[k] for k in _DATA.files if k.split("__")[0] == case}
    mode = "4x" if int(d["mode"][0]) == 2 else "16x"
    assert d["position_ids"].size > 0
    for types in (d["mm_token_type_ids"], None):
        pos, deltas = canvas_rope_index(
            d["input_ids"],
            types,
            d.get("target_sizes"),
            d.get("target_sizes_videos"),
            markers=MARKERS,
            downsample_mode=mode,
            attention_mask=d["attention_mask"],
            image_token_id=IMAGE_PAD,
            video_token_id=VIDEO_PAD,
        )
        np.testing.assert_array_equal(pos, d["position_ids"])
        np.testing.assert_array_equal(deltas, d["rope_deltas"])


def test_golden_fixture_covers_the_hard_shapes():
    """Fail closed if the fixture lost the shapes that exercise the canvas rules."""
    assert {
        "text_only",
        "small_image_4x",
        "multirow_slices_16x",
        "two_images_no_image_id",
        "video_2f_sliced_16x",
        "batch2_leftpad",
    } <= set(_CASES)
    # Multi-row slices: a newline between slice rows must exist in the prompt.
    ids = _DATA["multirow_slices_16x__input_ids"][0].tolist()
    assert ids.count(MARKERS[0]) >= 4 and any(
        a == MARKERS[1] and b == 198 and c == MARKERS[0] for a, b, c in zip(ids, ids[1:], ids[2:])
    )
    # Media prompts compress positions: every media golden has a negative delta.
    for case in _CASES:
        if case != "text_only":
            assert int(_DATA[f"{case}__rope_deltas"][0, 0]) < 0, case


def test_canvas_rope_index_needs_target_sizes_for_media():
    ids = np.array([[1, MARKERS[2], IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, MARKERS[3], 2]])
    with pytest.raises(ValueError, match="target sizes"):
        canvas_rope_index(ids, None, None, None, markers=MARKERS, image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD)
    pos, deltas = canvas_rope_index(
        ids, None, np.array([[8, 8]]), None, markers=MARKERS, image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD
    )
    # <image> halo (T=1, H=W=0), 2x2 thumb on origin 1, </image> far corner (1+2), trailing text at 1+2+1.
    assert pos[:, 0, 1].tolist() == [1, 0, 0]
    assert pos[:, 0, 2:6].tolist() == [[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]
    assert pos[:, 0, 6].tolist() == [1, 3, 3]
    assert pos[:, 0, 7].tolist() == [4, 4, 4]
    assert int(deltas[0, 0]) == 4 + 1 - 8


def test_group_visual_frames_absorbs_trailing_newline_into_the_frame():
    ids = [10, MARKERS[2], IMAGE_PAD, MARKERS[3], 198, 11]
    types = [0, 0, 1, 0, 0, 0]
    ((start, end, frames),) = group_visual_frames(ids, types, MARKERS)
    assert (start, end) == (1, 5)  # "\n" (198) after </image> belongs to the frame
    assert frames[0][3] == 5


def test_torch_linspace_round_known_values():
    # Canvas 14x18 from a 2x2 grid of 7x9 slices; the 7x9 thumbnail stretched over it.
    assert torch_linspace_round(13, 7).tolist() == [0, 2, 4, 6, 9, 11, 13]
    assert torch_linspace_round(17, 9).tolist() == [0, 2, 4, 6, 8, 11, 13, 15, 17]
    assert torch_linspace_round(0, 1).tolist() == [0]


def test_torch_linspace_round_matches_torch_exhaustively():
    torch = pytest.importorskip("torch")
    mismatches = []
    for canvas in range(1, 161):
        for steps in range(1, canvas + 1):
            ref = torch.linspace(0, canvas - 1, steps).round().long().tolist()
            if torch_linspace_round(canvas - 1, steps).tolist() != ref:
                mismatches.append((canvas, steps))
    assert mismatches == []


def test_vision_index_helpers_match_transformers():
    torch = pytest.importorskip("torch")
    vision_utils = pytest.importorskip("transformers.vision_utils")
    sizes = [[h, w] for h in (4, 8, 24, 36, 40, 70, 96) for w in (4, 12, 28, 44, 100)]
    ref = vision_utils.get_vision_nearest_position_ids(torch.tensor(sizes), 70).numpy()
    np.testing.assert_array_equal(nearest_position_ids(sizes, 70), ref)
    grid = torch.nn.functional.pad(torch.tensor(sizes), (1, 0), value=1)
    index, _ = vision_utils.get_vision_window_index(grid, spatial_merge_size=1, window_size=2, patch_size=1)
    np.testing.assert_array_equal(window_permutation(sizes, 2), index.numpy())


def test_gelu_variants_match_torch():
    torch = pytest.importorskip("torch")
    x = np.linspace(-6, 6, 2001, dtype=np.float32)
    tanh_ref = torch.nn.functional.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    erf_ref = torch.nn.functional.gelu(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(np.array(gelu_pytorch_tanh(mx.array(x))), tanh_ref, atol=2e-6)
    np.testing.assert_allclose(np.array(gelu_erf(mx.array(x))), erf_ref, atol=2e-6)
    # mlx's gelu_approx is the same tanh formula (measured 4.8e-7 on mlx 0.32.3);
    # the erf GELU is not interchangeable with it (merger vs vit_merger).
    assert float(np.abs(np.array(nn.gelu_approx(mx.array(x))) - tanh_ref).max()) < 2e-6
    assert float(np.abs(erf_ref - tanh_ref).max()) > 1e-4


# ------------------------------------------------------------------- numerics
def test_zero_centered_norm_is_fp32_one_plus_w():
    mx.random.seed(3)
    x = (mx.random.normal((4, 7, 64)) * 3).astype(mx.bfloat16)
    w = mx.random.uniform(-0.3, 1.0, (64,)).astype(mx.bfloat16)
    norm = ZeroCenteredRMSNorm(64, eps=1e-6)
    norm.weight = w
    xf = x.astype(mx.float32)
    ref = (xf * mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + 1e-6) * (1 + w.astype(mx.float32))).astype(mx.bfloat16)
    assert mx.array_equal(norm(x), ref).item()
    # The vendored storage form (1 + w rounded to bf16) is measurably different.
    rounded = mx.fast.rms_norm(x, w + 1.0, 1e-6)
    assert not mx.array_equal(rounded, ref).item()
    # The cached float32 scale follows a replaced weight (load_weights/set_dtype).
    norm.weight = mx.zeros((64,), dtype=mx.bfloat16)
    plain = (xf * mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + 1e-6)).astype(mx.bfloat16)
    assert mx.array_equal(norm(x), plain).item()
    norm.set_dtype(mx.float32)
    ref32 = xf * mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + 1e-6)
    assert mx.allclose(norm(xf), ref32, rtol=1e-6, atol=1e-6).item()  # fp32: reduction-order ULPs only


def test_gdn_qk_eps_on_mean_equals_l2norm_eps_on_sum():
    mx.random.seed(4)
    q = mx.random.normal((3, 5, 2, 128)) * 1e-3  # small norms make the eps visible
    d = q.shape[-1]
    l2 = q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + QWEN35_QK_L2NORM_EPS)
    ported = mx.fast.rms_norm(q, None, QWEN35_QK_L2NORM_EPS / d) * (d**-0.5)
    legacy = mx.fast.rms_norm(q, None, 1e-6) * (d**-0.5)
    assert float(mx.abs(ported - l2).max()) < 1e-6
    assert float(mx.abs(legacy - l2).max()) > 1e-3


def test_native_mtp_patch_gdn_honours_qk_l2norm_eps():
    """JANG loads apply the native-MTP patch suite, whose GatedDeltaNet replaces
    the vendored one; it must keep the exact q/k epsilon (eps / d on the mean)."""
    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl as patch

    assert patch._qk_norm_eps(SimpleNamespace(), 128) == 1e-6
    assert patch._qk_norm_eps(SimpleNamespace(qk_l2norm_eps=1e-6), 128) == pytest.approx(1e-6 / 128)
    patch.apply()
    model = _tiny_model()
    _randomize(model, seed=5)
    gdn = model.language_model.model.layers[0].linear_attn
    assert type(gdn).__call__.__module__ == patch.__name__
    x = mx.random.normal((1, 6, 16), key=mx.random.key(11))
    d = gdn.head_k_dim
    exact = gdn(x)
    gdn.qk_l2norm_eps = 1e-6 * d          # == the historical 1e-6 on the mean
    as_default = gdn(x)
    del gdn.qk_l2norm_eps
    default = gdn(x)
    assert bool(mx.array_equal(as_default, default).item())
    assert not bool(mx.array_equal(exact, default).item())


def test_exact_numerics_installed_on_every_layer():
    model = _tiny_model()
    lm = model.language_model
    assert lm.exact_numerics == {"rms_norms": 4 * 2 + 2 + 1, "gdn_qk_eps": 3}
    for layer in lm.model.layers:
        assert isinstance(layer.input_layernorm, ZeroCenteredRMSNorm)
        if layer.is_linear:
            assert layer.linear_attn.qk_l2norm_eps == QWEN35_QK_L2NORM_EPS
        else:
            assert isinstance(layer.self_attn.q_norm, ZeroCenteredRMSNorm)
    assert isinstance(lm.model.norm, ZeroCenteredRMSNorm)


# ------------------------------------------------------------------- sanitize
@pytest.mark.parametrize("moe", [False, True])
def test_sanitize_round_trips_hf_checkpoint_strictly(moe):
    model = _tiny_model(moe=moe)
    _randomize(model)
    from mlx.utils import tree_flatten

    original = dict(tree_flatten(model.parameters()))
    hf = _hf_checkpoint_from(model)
    assert any(k.endswith("experts.gate_up_proj") for k in hf) is moe
    sanitized = model.sanitize(dict(hf))
    assert set(sanitized) == set(original)
    for key, value in sanitized.items():
        assert value.shape == original[key].shape, key
        assert mx.array_equal(value, original[key]).item(), key
    fresh = _tiny_model(moe=moe)
    fresh.load_weights(list(sanitized.items()), strict=True)
    # Idempotent on live keys (JANG bundles store source tensors under live names).
    again = model.sanitize(dict(sanitized))
    assert set(again) == set(sanitized)
    assert all(mx.array_equal(again[k], sanitized[k]).item() for k in again)


def test_sanitize_splits_stacked_experts_gate_rows_first():
    model = _tiny_model(moe=True)
    e, i, h = 4, 8, 16
    gate_up = mx.concatenate([mx.zeros((e, i, h)), mx.ones((e, i, h))], axis=1)
    down = mx.full((e, h, i), 2.0)
    out = model.sanitize(
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }
    )
    prefix = "language_model.model.layers.0.mlp.switch_mlp."
    assert float(out[prefix + "gate_proj.weight"].max()) == 0.0
    assert float(out[prefix + "up_proj.weight"].min()) == 1.0
    assert out[prefix + "down_proj.weight"].shape == (e, h, i)


def test_sanitize_norm_storage_contract():
    model = _tiny_model()
    w = mx.full((16,), 0.25)
    # HF names: zero-centred by definition, never shifted.
    out = model.sanitize({"model.language_model.layers.0.input_layernorm.weight": w + 1.0})
    assert float(out["language_model.model.layers.0.input_layernorm.weight"][0]) == 1.25
    # Live names holding 1 + w: converted back exactly in float32.
    shard = {
        "language_model.model.layers.0.input_layernorm.weight": w + 1.0,
        "language_model.model.layers.0.post_attention_layernorm.weight": w + 1.0,
    }
    out = model.sanitize(shard)
    assert out["language_model.model.layers.0.input_layernorm.weight"].dtype == mx.float32
    assert float(out["language_model.model.layers.0.input_layernorm.weight"][0]) == 0.25
    # A later shard with only the final norm follows that decision.
    out = model.sanitize({"language_model.model.norm.weight": mx.full((16,), 4.43)})
    assert abs(float(out["language_model.model.norm.weight"][0]) - 3.43) < 1e-5
    # Zero-centred live shards (measured per-layer means up to 0.741) are kept.
    model2 = _tiny_model()
    keep = {"language_model.model.layers.1.q_norm.weight": mx.full((16,), 0.741)}
    assert float(model2.sanitize(keep)["language_model.model.layers.1.q_norm.weight"][0]) == pytest.approx(0.741)
    assert float(model2.sanitize({"language_model.model.norm.weight": mx.full((16,), 3.43)})[
        "language_model.model.norm.weight"][0]) == pytest.approx(3.43)


# ---------------------------------------------------------- embeddings + plan
def _image_prompt(tokens: int):
    ids = [5, 6, MARKERS[2]] + [IMAGE_PAD] * tokens + [MARKERS[3], 198, 7, 8]
    return mx.array([ids])


def test_get_input_embeddings_sets_whole_prompt_plan_and_scatters():
    model = _tiny_model()
    _randomize(model)
    ids = _image_prompt(4)  # one 8x8-patch crop -> 4 tokens in 16x
    pixels = mx.random.normal((1, 3, 14, 64 * 14))
    sizes = mx.array([[8, 8]], dtype=mx.int32)
    feats = model.get_input_embeddings(ids, pixels, target_sizes=sizes)
    lm = model.language_model
    expected, deltas = canvas_rope_index(
        np.array(ids), None, np.array(sizes), None, markers=MARKERS, image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD
    )
    assert np.array_equal(np.array(lm._position_ids), expected)
    assert np.array_equal(np.array(lm._rope_deltas), deltas)
    media = model.encode_media(pixels, np.array(sizes))
    np.testing.assert_allclose(np.array(feats.inputs_embeds[0, 3:7]), np.array(media), rtol=1e-5, atol=1e-5)
    # Chunked prefill slices the plan; decode adds the delta.
    assert np.array_equal(np.array(lm._positions_for(1, 3, 2)), expected[:, :, 2:5])
    dec = np.array(lm._positions_for(1, 1, ids.shape[1]))
    assert dec.reshape(-1).tolist() == [ids.shape[1] + int(deltas[0, 0])] * 3
    # Text-only prompt clears the plan.
    model.get_input_embeddings(mx.array([[5, 6, 7]]))
    assert lm._position_ids is None and lm._rope_deltas is None


def test_get_input_embeddings_rejects_mode_mismatch_and_missing_pixels():
    model = _tiny_model(mode="16x")
    ids = _image_prompt(16)  # 4x count for an 8x8 crop
    pixels = mx.zeros((1, 3, 14, 64 * 14))
    with pytest.raises(ValueError, match="downsample_mode"):
        model.get_input_embeddings(ids, pixels, target_sizes=mx.array([[8, 8]]))
    feats = model.get_input_embeddings(ids, pixels, target_sizes=mx.array([[8, 8]]), downsample_mode="4x")
    assert feats.inputs_embeds.shape == (1, ids.shape[1], 16)
    with pytest.raises(ValueError, match="no pixel values"):
        model.get_input_embeddings(ids)


def test_full_forward_runs_with_and_without_cache():
    model = _tiny_model(moe=True)
    _randomize(model)
    from mlx_vlm.models.cache import make_prompt_cache

    ids = _image_prompt(4)
    kw = dict(pixel_values=mx.random.normal((1, 3, 14, 64 * 14)), target_sizes=mx.array([[8, 8]]))
    full = model(ids, **kw).logits
    cache = make_prompt_cache(model.language_model)
    chunked = []
    emb = model.get_input_embeddings(ids, kw["pixel_values"], target_sizes=kw["target_sizes"]).inputs_embeds
    for s, e in ((0, 5), (5, ids.shape[1])):
        chunked.append(model.language_model(ids[:, s:e], inputs_embeds=emb[:, s:e], cache=cache).logits)
    np.testing.assert_allclose(np.array(mx.concatenate(chunked, 1)), np.array(full), rtol=2e-4, atol=2e-4)


# ------------------------------------------------------------------- parsers
def test_reasoning_parser_on_ground_truth_outputs():
    from vmlx_engine.reasoning import get_parser

    parser = get_parser("qwen3")()
    parser.reset_state(think_in_prompt=True)  # template opened <think>\n
    reasoning, content = parser.extract_reasoning(GT_THINK_ON)
    assert reasoning.strip().startswith("Okay, let me calculate 17 times 23.")
    assert reasoning.strip().endswith("Yep, that seems right.")
    assert content.strip().endswith("Answer: 391") and "</think>" not in content
    off = get_parser("qwen3")()
    off.reset_state(think_in_prompt=False)  # template rendered <think>\n\n</think>\n\n
    reasoning, content = off.extract_reasoning(GT_THINK_OFF)
    assert not (reasoning or "").strip() and content.strip() == "391"


@pytest.mark.parametrize("name", ["qwen", "qwen3_coder"])
def test_tool_parser_on_ground_truth_output(name):
    from vmlx_engine.tool_parsers import ToolParserManager

    result = ToolParserManager.get_tool_parser(name)().extract_tool_calls(GT_TOOL, request={"tools": WEATHER_TOOLS})
    assert result.tools_called
    calls = [(c["name"], json.loads(c["arguments"])) for c in result.tool_calls]
    assert calls == [("get_weather", {"city": "Paris", "unit": "c"})]
    assert result.content.strip() == (
        "I need to call the get_weather function to find out the current weather in Paris in Celsius."
    )


# ------------------------------------------------------------------- registry
def _write_config(path: Path, moe: bool) -> None:
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minicpmv4_7",
                "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
                "text_config": text,
                "vision_config": {"model_type": "minicpmv4_7_vision", "hidden_size": 1152},
                "image_token_id": IMAGE_PAD,
                "video_token_id": VIDEO_PAD,
                "downsample_mode": "16x",
            }
        )
    )


@pytest.mark.parametrize("moe", [False, True])
def test_registry_resolves_both_sizes(tmp_path, moe):
    from vmlx_engine.model_config_registry import get_model_config_registry

    bundle = tmp_path / ("MiniCPM-V-4.7-35B-A3B" if moe else "MiniCPM-V-4.7-1B")
    _write_config(bundle, moe)
    registry = get_model_config_registry()
    registry.clear_cache()
    cfg = registry.lookup(str(bundle))
    assert cfg.family_name == "minicpm_v4_7"
    assert cfg.cache_type == "hybrid" and cfg.is_mllm
    assert cfg.reasoning_parser == "qwen3" and cfg.think_in_template
    assert cfg.tool_parser == "qwen" and cfg.supports_native_tools
    assert cfg.eos_tokens == ["<|im_end|>", "<|endoftext|>"]
    assert cfg.architecture_hints["default_enable_thinking"] is True
    assert text_model_is_moe({"model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text"}) is moe


def test_runtime_is_registered_for_mlx_vlm():
    from mlx_vlm.utils import get_model_and_args

    module, model_type = get_model_and_args({"model_type": "minicpmv4_7"})
    assert model_type == "minicpmv4_7" and module.Model is Model


# ------------------------------------------------------------- engine hooks
def test_batched_engine_offset_proxy_and_request_rope_deltas():
    from vmlx_engine import mllm_batch_generator as mbg

    assert mbg._offset_proxy_needed_for_model_type("minicpmv4_7")
    lm = SimpleNamespace(decode_positions_add_request_rope_delta=True, _rope_deltas=mx.array([[-178]]))
    reqs = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]
    batch = SimpleNamespace(requests=reqs)
    mbg._record_request_rope_delta(lm, reqs[0])
    lm._rope_deltas = None
    mbg._record_request_rope_delta(lm, reqs[2])
    assert reqs[0]._mrope_decode_delta == -178 and reqs[2]._mrope_decode_delta == 0
    assert mbg._request_rope_delta_rows(lm, batch, 3).tolist() == [[-178], [0], [0]]
    with pytest.raises(RuntimeError):
        mbg._request_rope_delta_rows(lm, batch, 2)
    # Language models that do not opt in keep the engine's absolute positions,
    # and stub engines without a language model are a no-op.
    assert mbg._request_rope_delta_rows(SimpleNamespace(), batch, 3) is None
    mbg._record_request_rope_delta(None, SimpleNamespace())


def test_processor_alignment_helpers():
    from vmlx_engine.models.minicpmv4_7.processing import (
        align_processor_downsample_mode,
        configure_presampled_video,
    )

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(downsample_mode="16x"),
        video_processor=SimpleNamespace(downsample_mode="16x", do_sample_frames=True, max_slice_nums=9),
        image_token_divisor=16,
        video_token_divisor=16,
    )
    align_processor_downsample_mode(processor, "4x")
    configure_presampled_video(processor)
    assert processor.image_processor.downsample_mode == "4x" == processor.video_processor.downsample_mode
    assert processor.image_token_divisor == 4 == processor.video_token_divisor
    assert processor.video_processor.do_sample_frames is False and processor.video_processor.max_slice_nums == 1
    assert processor._vmlx_video_frames_presampled is True


def test_simple_engine_native_video_clip(tmp_path):
    from PIL import Image

    from vmlx_engine.models.mllm import _NATIVE_VIDEO_FRAME_MODEL_TYPES, _video_frames_as_native_clip

    assert "minicpmv4_7" in _NATIVE_VIDEO_FRAME_MODEL_TYPES
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.png"
        Image.new("RGB", (20, 10), (i * 40, 0, 0)).save(p)
        paths.append(str(p))
    clip = _video_frames_as_native_clip(paths)
    assert clip.shape == (3, 10, 20, 3) and clip.dtype == np.uint8 and int(clip[2, 0, 0, 0]) == 80
    with pytest.raises(ValueError):
        _video_frames_as_native_clip([])
    assert math.isfinite(float(clip.mean()))


@pytest.mark.parametrize("case", [
    "text_only", "small_image_4x", "multirow_slices_16x",
    "two_images_no_image_id", "video_2f_sliced_16x", "batch2_leftpad",
])
def test_cached_canvas_tail_keeps_processor_positions_and_decode_delta(case):
    """A restored suffix must have exactly the cold transformer's canvas plan."""
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    from vmlx_engine.models.minicpmv4_7.language import _CanvasLanguageMixin

    d = {k.split("__")[1]: _DATA[k] for k in _DATA.files if k.split("__")[0] == case}
    lm = _CanvasLanguageMixin()
    lm.config = SimpleNamespace(
        slice_start_id=MARKERS[0], slice_end_id=MARKERS[1],
        image_start_id=MARKERS[2], image_end_id=MARKERS[3], newline_id=MARKERS[4],
        image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD, downsample_mode="16x",
    )
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._model_type = "minicpmv4_7"
    gen.language_model = lm
    ids = mx.array(d["input_ids"])
    cut = int(ids.shape[-1]) - 1
    req = SimpleNamespace(
        request_id="canvas-tail", attention_mask=mx.array(d["attention_mask"]),
        extra_kwargs={
            **{k: mx.array(d[k]) for k in ("mm_token_type_ids", "target_sizes", "target_sizes_videos") if k in d},
            "downsample_mode": "4x" if int(d["mode"][0]) == 2 else "16x",
        },
    )
    assert gen._mrope_tail_position_ids(req, ids, cut)
    np.testing.assert_array_equal(np.asarray(req._mrope_tail_position_ids), d["position_ids"][..., cut:])
    np.testing.assert_array_equal(np.asarray(req._mrope_tail_rope_deltas), d["rope_deltas"])
    assert req._mrope_decode_delta == int(d["rope_deltas"].reshape(-1)[0])
    if "target_sizes" in req.extra_kwargs or "target_sizes_videos" in req.extra_kwargs:
        req.extra_kwargs = {}
        fresh = SimpleNamespace(request_id="missing-canvas", extra_kwargs={})
        assert not gen._mrope_tail_position_ids(fresh, ids, cut)
        assert not hasattr(fresh, "_mrope_tail_position_ids")

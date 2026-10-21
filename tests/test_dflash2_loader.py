"""Stored quantization preserves mixed floating/packed DFlash2 tensors."""
import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
runtime = pytest.importorskip("dflash.model_mlx")
from mlx.utils import tree_flatten

from vmlx_engine.dflash2_loader import load_quantized_dflash2


def _checkpoint(tmp_path):
    cfg = dict(hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
               num_key_value_heads=1, head_dim=32, intermediate_size=128,
               vocab_size=128, rms_norm_eps=1e-6, rope_theta=10000.0,
               max_position_embeddings=512, num_target_layers=2,
               architectures=["DFlash2DraftModel"],
               dflash_config=dict(block_size=16, target_layer_ids=[0, 1],
                                  mask_token_id=0, selector_rank=32, selector_top_k=4,
                                  conv_group_size=16, conv_kernel_size=2),
               quantization=dict(mode="affine", bits=8, group_size=32))
    model = runtime.DFlash2DraftModel(runtime.DFlashConfig(
        **{k: v for k, v in cfg.items() if k not in ("architectures", "dflash_config", "quantization")},
        **cfg["dflash_config"]))
    nn.quantize(model, bits=8, group_size=32,
                class_predicate=lambda name, module: hasattr(module, "to_quantized") and not name.startswith("candidate_selector"))
    weights = dict(tree_flatten(model.parameters()))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[key] = weights.pop(f"{key}.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return cfg, weights


def test_stored_q8_g32_preserves_all_tensors_and_forward(tmp_path):
    cfg, weights = _checkpoint(tmp_path)
    loaded = load_quantized_dflash2(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    for name, expected in weights.items():
        if name.endswith(("predecessor_codebook", "successor_codebook")):
            name += ".weight"
        assert actual[name].dtype == expected.dtype
        assert mx.array_equal(actual[name], expected).item(), name
    assert loaded.fc.bits == 8 and loaded.fc.group_size == 32
    assert not isinstance(loaded.candidate_selector.hidden_projection, nn.QuantizedLinear)
    # Exercise the packed fc matmul, not only its parameter names.
    x = mx.ones((1, 2, 128))
    expected = mx.quantized_matmul(x, weights["fc.weight"], weights["fc.scales"],
                                  weights["fc.biases"], transpose=True, bits=8, group_size=32)
    assert mx.array_equal(loaded.fc(x), expected).item()


def test_missing_packed_parameter_is_rejected(tmp_path):
    _, weights = _checkpoint(tmp_path)
    del weights["fc.biases"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_quantized_dflash2(tmp_path)


def test_unknown_quantization_rejected(tmp_path):
    cfg, _ = _checkpoint(tmp_path)
    cfg["quantization"]["mode"] = "unknown"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="affine"):
        load_quantized_dflash2(tmp_path)

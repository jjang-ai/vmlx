# SPDX-License-Identifier: Apache-2.0
"""Spark native geometry/dtype/cache contracts, without a production model load."""
import importlib
import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine.models.spark2_5 import register_spark2_5_runtime
from vmlx_engine.models.spark2_5.register import ensure_spark2_5_runtime_registered

register_spark2_5_runtime()
from mlx_lm.models.spark2_5 import Model, ModelArgs
from mlx_lm.models.cache import KVCache, RotatingKVCache, save_prompt_cache, load_prompt_cache


def args(**overrides):
    values = dict(
        hidden_size=64, intermediate_size=128, vocab_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        },
    )
    values.update(overrides)
    return ModelArgs(**values)


def test_declared_geometry_and_native_cache():
    model = Model(args())
    cache = model.make_cache()
    assert [type(c) for c in cache] == [RotatingKVCache] * 3 + [KVCache]
    assert model.layers[0].self_attn.q_k_v_proj.weight.shape == (128, 64)
    assert model.layers[0].self_attn.g_proj.weight.shape == (4, 64)
    assert model.layers[0].self_attn.out_proj.weight.shape == (64, 64)
    assert not hasattr(model, "lm_head")


@pytest.mark.parametrize("length,split", [(15, 7), (17, 13), (53, 29)])
def test_split_prefill_and_disk_restore_match_full(length, split, tmp_path):
    mx.random.seed(7)
    model = Model(args())
    tokens = mx.array([[i % 128 for i in range(length)]])
    expected = model(tokens)[:, -1, :]
    cache = model.make_cache()
    mx.eval(model(tokens[:, :split], cache=cache))
    path = str(tmp_path / "cache.safetensors")
    save_prompt_cache(path, cache)
    restored = load_prompt_cache(path)
    actual = model(tokens[:, split:], cache=restored)[:, -1, :]
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, atol=3e-5, rtol=3e-5).item()
    assert all(c.offset == length for c in restored)
    # Logical offset is not the rotating buffer's physical length.
    assert all(isinstance(c, RotatingKVCache) for c in restored[:3])
    assert isinstance(restored[3], KVCache)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_quantized_projections_never_cast_activation_to_packed_uint(bits):
    mx.random.seed(11)
    model = Model(args())
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=bits,
                class_predicate=lambda path, module:
                isinstance(module, (nn.Linear, nn.Embedding)) and not path.endswith("g_proj"))
    assert model.layers[0].mlp.gate_proj.weight.dtype == mx.uint32
    assert model.layers[0].input_layernorm.weight.dtype == mx.bfloat16
    tokens = mx.array([[3, 17, 22, 8]])
    output = model(tokens)
    hidden = model.layers[0](model.model.embedding(tokens).astype(mx.float32))
    mx.eval(output, hidden)
    assert hidden.dtype == mx.float32
    assert output.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(output)).item()
    assert mx.max(mx.abs(output)).item() > 0
    assert not mx.array_equal(output[:, 0], output[:, 1]).item()


def test_missing_cache_layer_does_not_silently_skip_decoder():
    with pytest.raises(ValueError, match="Spark cache has"):
        Model(args())(mx.array([[1, 2]]), cache=[KVCache()])


@pytest.mark.parametrize("overrides", [
    {"layer_types": ["unknown"] * 4},
    {"num_key_value_heads": 3},
    {"sliding_window": 0},
    {"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.2}}},
])
def test_invalid_geometry_is_rejected(overrides):
    with pytest.raises(ValueError):
        args(**overrides)


def test_registration_uses_config_not_folder_name(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "spark2_5"}))
    assert ensure_spark2_5_runtime_registered(tmp_path)
    assert importlib.import_module("mlx_lm.models.spark2_5").Model is Model
    assert not ensure_spark2_5_runtime_registered(tmp_path, config={"model_type": "llama"})


def test_engine_registry_keeps_native_template_and_parser(tmp_path):
    from vmlx_engine.model_config_registry import get_model_config_registry
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "spark2_5"}))
    config = get_model_config_registry().lookup(str(tmp_path))
    assert config.tool_parser == "spark25"
    assert config.reasoning_parser == "qwen3"
    assert config.think_in_template and config.supports_native_tools
    assert config.cache_type == "kv" and not config.is_mllm


def test_registration_does_not_swallow_internal_dependency_error(monkeypatch):
    from vmlx_engine.models.spark2_5 import register as registration
    monkeypatch.delitem(registration.sys.modules, registration._PACKAGE)
    def broken(name):
        raise ModuleNotFoundError("missing runtime dependency", name="some_dependency")
    monkeypatch.setattr(registration.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing runtime dependency"):
        registration.register_spark2_5_runtime()
    assert registration._PACKAGE not in registration.sys.modules


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_repeated_disk_resume_preserves_partitioned_state_and_logits(bits, dtype, tmp_path):
    """Weight quantization plus repeated disk resume across SWA wrap points.

    Compare identical computation partitions, isolating serialization from
    backend GEMM/GEMV rounding differences. Every tensor and continuation logit
    must be exact. This does not assert different prefill shapes are bit-exact.
    """
    from types import SimpleNamespace
    from copy import deepcopy
    from vmlx_engine.scheduler import Scheduler

    mx.random.seed(23)
    model = Model(args())
    model.set_dtype(dtype)
    nn.quantize(model, group_size=32, bits=bits,
                class_predicate=lambda path, module:
                isinstance(module, (nn.Linear, nn.Embedding)) and not path.endswith("g_proj"))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    scheduler._uses_dsv4_cache = False
    scheduler.config = SimpleNamespace(prefill_step_size=17)
    tokens = [(i * 7 + 3) % 128 for i in range(70)]
    resumed, boundary = None, 0
    partitioned = None
    for target in (15, 29, 53, 69):
        resumed = scheduler._prefill_for_prompt_only_cache(
            tokens[:target], base_cache=resumed, base_token_count=boundary)
        partitioned = scheduler._prefill_for_prompt_only_cache(
            tokens[:target], base_cache=partitioned, base_token_count=boundary)
        assert resumed is not None and partitioned is not None
        # Compare each layer in chronological rather than physical ring order.
        for layer_index, (right, same_partition) in enumerate(zip(resumed, partitioned)):
            assert right.offset == same_partition.offset == target
            assert right.meta_state == same_partition.meta_state
            for attr in ("keys", "values"):
                b = getattr(right, attr)
                b = right._temporal_order(b)[..., -16:, :] if isinstance(right, RotatingKVCache) else b[..., :target, :]
                c = getattr(same_partition, attr)
                c = same_partition._temporal_order(c)[..., -16:, :] if isinstance(same_partition, RotatingKVCache) else c[..., :target, :]
                assert b.shape == c.shape
                assert mx.array_equal(b, c).item(), ("disk_vs_same_partition", target, layer_index, attr)
        next_token = mx.array([[tokens[target]]])
        restored_logits = model(next_token, cache=deepcopy(resumed))
        memory_logits = model(next_token, cache=deepcopy(partitioned))
        assert mx.array_equal(restored_logits, memory_logits).item(), ("continuation_logits", target)
        path = str(tmp_path / f"resume-{target}.safetensors")
        save_prompt_cache(path, resumed)
        resumed = load_prompt_cache(path)
        boundary = target

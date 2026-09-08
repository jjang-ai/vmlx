# SPDX-License-Identifier: Apache-2.0
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import Model, ModelArgs

from vmlx_engine.prefix_cache import compute_model_cache_key, build_block_cache_namespace
from vmlx_engine.utils.minicpm5_qkv import prepare_minicpm5_qkv, FusedMiniCPM5Attention


def fixture(tmp_path, dialect="minicpm5_xml_function"):
    model = Model(ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=8, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=128))
    nn.quantize(model, group_size=64, bits=8)
    model.apply(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a)
    (tmp_path / "jang_config.json").write_text(json.dumps({"tool_calling": {"dialect": dialect}}))
    return model


def test_preserves_packed_parameters_and_separates_disk_namespace(tmp_path):
    model = fixture(tmp_path)
    old = model.layers[0].self_attn
    packed = [p.weight for p in [old.q_proj, old.k_proj, old.v_proj]]
    key = compute_model_cache_key(model, str(tmp_path))
    def namespace():
        return build_block_cache_namespace(model=model, model_path=str(tmp_path),
            quant_tag="test", tq_native_tag="off")
    old_namespace = namespace()
    original_config = (tmp_path / "jang_config.json").read_bytes()
    assert prepare_minicpm5_qkv(model, tmp_path)
    new = model.layers[0].self_attn
    assert type(new) is FusedMiniCPM5Attention
    assert bool(mx.array_equal(new.weight, mx.concatenate(packed)))
    assert new.weight.dtype == mx.uint32 and new.scales.dtype == mx.bfloat16
    assert new.rope is old.rope and new.o_proj is old.o_proj
    assert compute_model_cache_key(model, str(tmp_path)) != key
    assert namespace() != old_namespace
    assert prepare_minicpm5_qkv(model, tmp_path)  # no second concatenation
    assert model.layers[0].self_attn is new
    assert (tmp_path / "jang_config.json").read_bytes() == original_config


@pytest.mark.parametrize("defect", ["bits", "scales_dtype", "linear_bias", "custom_type"])
def test_unqualified_layer_leaves_every_layer_untouched(tmp_path, defect):
    model = fixture(tmp_path)
    originals = [layer.self_attn for layer in model.layers]
    p = model.layers[-1].self_attn.k_proj
    if defect == "bits":
        p.bits = 4
    elif defect == "scales_dtype":
        p.scales = p.scales.astype(mx.float16)
    elif defect == "linear_bias":
        p.bias = mx.zeros((p.weight.shape[0],))
    else:
        class Custom(type(p)):
            pass
        p.__class__ = Custom
    assert not prepare_minicpm5_qkv(model, tmp_path)
    assert all(layer.self_attn is a for layer, a in zip(model.layers, originals))


def test_unrelated_llama_never_changes(tmp_path):
    model = fixture(tmp_path, "llama")
    old = model.layers[0].self_attn
    assert not prepare_minicpm5_qkv(model, tmp_path)
    assert model.layers[0].self_attn is old


def test_native_attention_cache_and_mask_contract(tmp_path):
    model = fixture(tmp_path)
    ids = mx.array([[2, 4, 7, 8]])
    old_cache = make_prompt_cache(model)
    old_logits = model(ids, cache=old_cache)
    mx.eval(old_logits)
    assert prepare_minicpm5_qkv(model, tmp_path)
    new_cache = make_prompt_cache(model)
    new_logits = model(ids, cache=new_cache)
    mx.eval(new_logits)
    assert new_logits.dtype == old_logits.dtype == mx.bfloat16
    assert float(mx.max(mx.abs(new_logits.astype(mx.float32)-old_logits.astype(mx.float32)))) < 0.05
    for old, new in zip(old_cache, new_cache):
        assert new.offset == old.offset == 4
        assert [a.shape for a in new.state] == [a.shape for a in old.state]
    step = model(mx.array([[9]]), cache=new_cache)
    mx.eval(step)
    assert all(c.offset == 5 for c in new_cache)
    assert bool(mx.all(mx.isfinite(step)))

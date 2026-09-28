"""Hadamard/ternary storage must not silently become ordinary affine math."""
import copy

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from vmlx_engine.metal.quantized_projection_group import quantized_projection_group_reason
from vmlx_engine.utils.jang_hadamard import (
    hadamard_spec_from_config, hadamard_activation, install_hadamard_modules,
    verify_hadamard_signs_loaded, HadamardQuantizedLinear, HadamardQuantizedEmbedding,
    configure_hadamard_activation_precision,
)
from vmlx_engine.utils.jang_ternary_packed import (
    ternary_packed_modules, expand_ternary_packed_mlx, expand_ternary_packed_shard_mlx,
)
from vmlx_engine.utils.jang_loader import (
    _pre_fix_bits_from_shard, _pre_fix_bits_from_metadata, _post_load_quantization_overrides,
)


def contract():
    return {"hadamard": {
        "contract": "prism.hadamard.v1", "block_size": 512,
        "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension", "sign_mode": "explicit",
        "signs_dtype": "float32", "signs_tensor_suffix": "signs", "compute_dtype": "float32",
        "forward_modules": ["projection"], "inverse_modules": ["embedding"],
    }}


def model():
    m = nn.Module()
    m.projection = nn.Linear(512, 32, bias=False).to_quantized(group_size=128, bits=2)
    m.embedding = nn.Embedding(32, 512).to_quantized(group_size=128, bits=2)
    return m


def test_ordinary_bundle_has_no_transform_or_packed_storage():
    assert hadamard_spec_from_config({}, {}) is None
    assert ternary_packed_modules({}) == frozenset()


@pytest.mark.parametrize("key,value", [
    ("contract", "unknown"), ("transform", "unnormalized"), ("axis", "output"),
    ("sign_mode", "generated"), ("block_size", 513), ("block_size", 512.0),
    ("forward_modules", "projection"), ("forward_modules", ["projection", "projection"]),
    ("inverse_modules", ["projection"]), ("compute_dtype", "float16"),
    ("signs_dtype", "float16"), ("signs_tensor_suffix", "other"),
])
def test_contract_fail_closed(key, value):
    c = contract()
    c["hadamard"][key] = value
    with pytest.raises(ValueError):
        hadamard_spec_from_config(c)


def test_missing_malformed_or_conflicting_contract_is_not_plain_affine():
    with pytest.raises(ValueError):
        hadamard_spec_from_config({}, {"runtime": {"requires_hadamard_activation_transform": True}})
    with pytest.raises(ValueError):
        hadamard_spec_from_config({"hadamard": None})
    c, j = contract(), contract()
    j["hadamard"]["block_size"] = 1024
    with pytest.raises(ValueError):
        hadamard_spec_from_config(c, j)


def test_transform_inverse_and_forward_wrapper_match_reference():
    mx.random.seed(9)
    m, spec = model(), hadamard_spec_from_config(contract())
    original_weight = m.projection.weight
    assert install_hadamard_modules(m, spec) == 2
    assert m.projection.weight is original_weight
    assert isinstance(m.projection, HadamardQuantizedLinear)
    assert isinstance(m.embedding, HadamardQuantizedEmbedding)
    with pytest.raises(RuntimeError, match="sign verification"):
        verify_hadamard_signs_loaded(m, spec)
    signs = mx.where(mx.arange(512) % 3 == 0, -1.0, 1.0).astype(mx.float32)
    m.projection.signs, m.embedding.signs = signs, signs
    assert verify_hadamard_signs_loaded(m, spec) == 2
    x = mx.random.normal((2, 3, 512))
    rotated = hadamard_activation(x, 512, signs)
    restored = hadamard_activation(rotated, 512, signs, inverse=True)
    assert mx.max(mx.abs(restored - x)).item() < 2e-6
    expected = mx.quantized_matmul(rotated, m.projection.weight,
                                  scales=m.projection.scales, biases=m.projection.biases,
                                  group_size=128, bits=2)
    assert mx.array_equal(m.projection(x), expected).item()
    ids = mx.array([1, 3])
    unrotated = nn.QuantizedEmbedding.__call__(m.embedding, ids)
    assert mx.array_equal(m.embedding(ids), hadamard_activation(unrotated, 512, signs, inverse=True)).item()
    with pytest.raises(NotImplementedError):
        m.embedding.as_linear(x)
    assert "Hadamard" in quantized_projection_group_reason([m.projection, m.projection])


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "zero"])
def test_sign_verification_checks_shape_dtype_values(bad):
    m, spec = model(), hadamard_spec_from_config(contract())
    install_hadamard_modules(m, spec)
    m.projection.signs = mx.ones((512,), dtype=mx.float32)
    m.embedding.signs = mx.ones((512,), dtype=mx.float32)
    if bad == "shape":
        m.projection.signs = mx.ones((1,), dtype=mx.float32)
    elif bad == "dtype":
        m.projection.signs = m.projection.signs.astype(mx.float16)
    elif bad == "nan":
        m.projection.signs = m.projection.signs * float("nan")
    else:
        m.projection.signs = mx.zeros((512,), dtype=mx.float32)
    with pytest.raises(RuntimeError):
        verify_hadamard_signs_loaded(m, spec)


def test_wrong_module_kind_or_block_width_rejected():
    m = model()
    m.projection = nn.Linear(512, 32)
    with pytest.raises(RuntimeError):
        install_hadamard_modules(m, hadamard_spec_from_config(contract()))
    c = contract()
    c["hadamard"]["block_size"] = 1024
    with pytest.raises(RuntimeError):
        install_hadamard_modules(model(), hadamard_spec_from_config(c))


def packed_fixture(rows=3, groups=2):
    rng = np.random.default_rng(17)
    codes = rng.integers(0, 3, size=(rows, groups, 128), dtype=np.uint32)
    head = (codes[:, :, :125].reshape(rows, groups, 25, 5) * (3 ** np.arange(5))).sum(-1)
    tail = (codes[:, :, 125:] * (3 ** np.arange(3))).sum(-1)[..., None]
    packed = np.concatenate([head, tail], axis=-1).reshape(rows, groups * 26).astype(np.uint8)
    words = (codes.reshape(rows, groups * 8, 16) << (2 * np.arange(16, dtype=np.uint32))).sum(-1).astype(np.uint32)
    return packed, words


@pytest.mark.parametrize("rows,groups", [(3, 2), (1025, 8)])
def test_packed_expansion_bit_exact_including_chunk_boundary(rows, groups):
    packed, words = packed_fixture(rows, groups)
    scales = mx.full((rows, groups), 0.125, dtype=mx.float16)
    w, s, b = expand_ternary_packed_mlx(mx.array(packed), scales)
    assert s is scales
    assert mx.array_equal(w, mx.array(words)).item()
    assert mx.array_equal(b, -scales).item()


@pytest.mark.parametrize("bad", ["trit_head", "trit_tail", "dtype", "width", "scale_shape", "scale_dtype", "nonfinite"])
def test_packed_bad_data_fails_before_loading(bad):
    p, _ = packed_fixture()
    s = mx.full((3, 2), 0.125, dtype=mx.float16)
    if bad == "trit_head": p[0, 0] = 243
    if bad == "trit_tail": p[0, 25] = 27
    p = mx.array(p)
    if bad == "dtype": p = p.astype(mx.uint32)
    if bad == "width": p = p[:, :-1]
    if bad == "scale_shape": s = s[:, :1]
    if bad == "scale_dtype": s = s.astype(mx.float32)
    if bad == "nonfinite": s = s * float("inf")
    with pytest.raises(ValueError):
        expand_ternary_packed_mlx(p, s)


def test_packed_manifest_and_shard_contract():
    entry = {"storage": "ternary_packed_26b", "runtime_bits": 2, "group_size": 128}
    j = {"runtime": {"requires_jang_ternary_packed_expansion": True},
         "quantization": {"tensor_quantization_manifest": {"projection": entry}}}
    assert ternary_packed_modules(j) == frozenset({"projection"})
    wrong = copy.deepcopy(j)
    wrong["quantization"]["tensor_quantization_manifest"] = {}
    with pytest.raises(ValueError): ternary_packed_modules(wrong)
    wrong = copy.deepcopy(j)
    wrong["quantization"]["tensor_quantization_manifest"]["projection"]["runtime_bits"] = 4
    with pytest.raises(ValueError): ternary_packed_modules(wrong)
    packed, _ = packed_fixture()
    weights = {"projection.weight": mx.array(packed), "projection.scales": mx.ones((3, 2), mx.float16),
               "other": mx.ones((1,))}
    result, count = expand_ternary_packed_shard_mlx(weights, {"projection"})
    assert count == 1 and result["other"] is weights["other"]
    assert weights["projection.weight"].dtype == mx.uint8
    with pytest.raises(ValueError):
        expand_ternary_packed_shard_mlx({**weights, "projection.biases": weights["projection.scales"]}, {"projection"})


@pytest.mark.parametrize("metadata_only", [False, True])
def test_hadamard_vlm_keeps_declared_6bit_g64_instead_of_ambiguous_3bit_g128(metadata_only):
    m = nn.Module()
    m.projection = nn.Linear(1152, 64, bias=False).to_quantized(group_size=64, bits=6)
    weights = {"projection.weight": m.projection.weight, "projection.scales": m.projection.scales}
    overrides = {"projection": {"bits": 6, "group_size": 64}}
    if metadata_only:
        _pre_fix_bits_from_metadata(m, {k: v.shape for k, v in weights.items()}, 128, overrides)
    else:
        _pre_fix_bits_from_shard(m, weights, 128, overrides)
    assert (m.projection.bits, m.projection.group_size) == (6, 64)


def test_bonsai_manifest_controls_post_load_bits():
    j = {"quantization": {"method": "jang-affine-discrete-ternary-lossless-repack+ternary-packed-26b-storage",
         "tensor_quantization_manifest_schema": 2,
         "tensor_quantization_manifest": {"projection": {"bits": 2, "group_size": 128}}}}
    assert _post_load_quantization_overrides({}, j) == {"projection": {"bits": 2, "group_size": 128}}


def precision_fixture():
    m = nn.Module()
    m.language_model = model()
    m.language_model.norm = nn.RMSNorm(512)
    m.language_model.conv = nn.Conv1d(32, 32, 3, groups=32, bias=False)
    m.language_model.A_log = mx.ones((2,), mx.float32)
    m.vision_tower = nn.RMSNorm(512)
    spec = hadamard_spec_from_config(contract())
    install_hadamard_modules(m.language_model, spec)
    for layer in (m.language_model.projection, m.language_model.embedding):
        layer.signs = mx.ones((512,), mx.float32)
        layer.scales = layer.scales.astype(mx.float16)
        layer.biases = layer.biases.astype(mx.float16)
    cfg = {**contract(), "model_type": "qwen3_5"}
    return m, cfg


def test_fp16_policy_contains_promotion_without_changing_weights_or_transform():
    m, cfg = precision_fixture()
    lang = m.language_model
    x = mx.ones((1, 3, 512), mx.float16)
    c = mx.ones((1, 3, 32), mx.float16)
    norm_before, conv_before = lang.norm(x), lang.conv(c)
    assert norm_before.dtype == mx.float32
    assert conv_before.dtype == mx.float32
    parameters = dict(tree_flatten(m.parameters()))
    result = configure_hadamard_activation_precision(m, cfg)
    assert result == {"signature": "qwen-hadamard-fp16-v1", "projections": 2,
                      "norms": 1, "convolutions": 1}
    assert mx.array_equal(lang.norm(x), norm_before.astype(mx.float16)).item()
    assert mx.array_equal(lang.conv(c), conv_before.astype(mx.float16)).item()
    assert lang.projection(x.astype(mx.float32)).dtype == mx.float16
    assert lang.embedding(mx.array([[1, 2]])).dtype == mx.float16
    assert lang.projection.hadamard_compute_dtype == mx.float32
    assert lang.projection.signs.dtype == mx.float32
    assert lang.A_log.dtype == mx.float32
    assert m.vision_tower(x).dtype == mx.float32
    assert all(parameters[name] is value for name, value in tree_flatten(m.parameters()))


def test_fp16_policy_is_not_global_and_reference_has_separate_identity():
    from vmlx_engine.prefix_cache import compute_model_cache_key

    m, cfg = precision_fixture()
    assert configure_hadamard_activation_precision(m, {"model_type": "qwen3_5"}) == {}
    baseline, _ = precision_fixture()
    configure_hadamard_activation_precision(baseline, cfg, enabled=False)
    configure_hadamard_activation_precision(m, cfg)
    old_key = compute_model_cache_key(baseline, model_path="same-bundle")
    new_key = compute_model_cache_key(m, model_path="same-bundle")
    assert old_key != new_key
    clone, _ = precision_fixture()
    configure_hadamard_activation_precision(clone, cfg)
    assert compute_model_cache_key(clone, model_path="same-bundle") == new_key
    with pytest.raises(ValueError, match="cannot change"):
        configure_hadamard_activation_precision(m, cfg, enabled=False)


def test_fp16_policy_rejects_unqualified_scale_precision_before_mutation():
    m, cfg = precision_fixture()
    m.language_model.projection.scales = m.language_model.projection.scales.astype(mx.bfloat16)
    with pytest.raises(ValueError, match="FP16 affine scales"):
        configure_hadamard_activation_precision(m, cfg)
    assert not hasattr(m, "_vmlx_hadamard_activation_precision")
    assert type(m.language_model.norm) is nn.RMSNorm


def test_fp16_prefill_only_keeps_decode_fp32_and_separates_cache_identity():
    from vmlx_engine.prefix_cache import compute_model_cache_key

    model, cfg = precision_fixture()
    result = configure_hadamard_activation_precision(model, cfg, prefill_only=True)
    assert result["signature"] == "qwen-hadamard-fp16-prefill-only-v1"
    lang = model.language_model
    assert lang.projection(mx.ones((1, 2, 512), mx.float32)).dtype == mx.float32
    assert lang.projection(mx.ones((1, 2048, 512), mx.float32)).dtype == mx.float16
    assert lang.embedding(mx.array([[1, 2]])).dtype == mx.float16
    assert lang.embedding(mx.zeros((1, 2048), mx.int32)).dtype == mx.float16
    assert lang.norm(mx.ones((1, 2, 512), mx.float16)).dtype == mx.float32
    assert lang.norm(mx.ones((1, 2048, 512), mx.float16)).dtype == mx.float16
    assert lang.conv(mx.ones((1, 3, 32), mx.float16)).dtype == mx.float32
    assert lang.conv(mx.ones((1, 2048, 32), mx.float16)).dtype == mx.float16

    full, _ = precision_fixture()
    configure_hadamard_activation_precision(full, cfg)
    assert compute_model_cache_key(model, model_path="same-bundle") != compute_model_cache_key(
        full, model_path="same-bundle"
    )
    with pytest.raises(ValueError, match="cannot change"):
        configure_hadamard_activation_precision(model, cfg)


def test_bonsai_prefill_dequant_gemm_tracks_fp16_qmm(monkeypatch):
    model, cfg = precision_fixture()
    configure_hadamard_activation_precision(model, cfg, prefill_only=True)
    x = ((mx.arange(2048 * 512, dtype=mx.float32) % 97) - 48).reshape(1, 2048, 512)
    x = x * 0.015625
    monkeypatch.setenv("VMLX_BONSAI_FP16_QMM", "1")
    monkeypatch.delenv("VMLX_BONSAI_PREFILL_DQ_GEMM", raising=False)
    packed = model.language_model.projection(x)
    monkeypatch.setenv("VMLX_BONSAI_PREFILL_DQ_GEMM", "1")
    dense = model.language_model.projection(x)
    mx.eval(packed, dense)
    assert packed.dtype == dense.dtype == mx.float16
    assert bool(mx.allclose(packed, dense, atol=0.03, rtol=0.01).item())


def test_bonsai_ane_and_precision_flags_separate_prefix_cache_identity(monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    config_model, cfg = precision_fixture()
    configure_hadamard_activation_precision(config_model, cfg, prefill_only=True)
    stock_key = compute_model_cache_key(config_model, model_path="same-bundle")
    monkeypatch.setenv("VMLX_BONSAI_FP16_QMM", "1")
    monkeypatch.setenv("VMLX_BONSAI_PREFILL_DQ_GEMM", "1")
    portable, _ = precision_fixture()
    configure_hadamard_activation_precision(portable, cfg, prefill_only=True)
    portable_key = compute_model_cache_key(portable, model_path="same-bundle")
    monkeypatch.setenv("VMLX_BONSAI_ANE_PREFILL", "1")
    ane, _ = precision_fixture()
    configure_hadamard_activation_precision(ane, cfg, prefill_only=True)
    ane_key = compute_model_cache_key(ane, model_path="same-bundle")
    assert len({stock_key, portable_key, ane_key}) == 3


def test_bonsai_fused_rotation_separates_prefix_cache_identity(monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key

    stock, cfg = precision_fixture()
    configure_hadamard_activation_precision(stock, cfg)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_HADAMARD", "1")
    fused, _ = precision_fixture()
    configure_hadamard_activation_precision(fused, cfg)
    assert compute_model_cache_key(stock, model_path="same-bundle") != compute_model_cache_key(
        fused, model_path="same-bundle"
    )


def test_bonsai_derived_kernels_require_fp16_qmm_before_mutation(monkeypatch):
    monkeypatch.setenv("VMLX_BONSAI_PREFILL_DQ_GEMM", "1")
    model, cfg = precision_fixture()
    with pytest.raises(ValueError, match="require FP16 QMM"):
        configure_hadamard_activation_precision(model, cfg, prefill_only=True)
    assert not hasattr(model, "_vmlx_hadamard_activation_precision")

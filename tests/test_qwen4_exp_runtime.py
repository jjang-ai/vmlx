import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.qwen4_exp.language import LanguageModel, Qwen4ExpTextArgs


def _tiny_args() -> Qwen4ExpTextArgs:
    return Qwen4ExpTextArgs(
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=997,
        full_attention_interval=4,
        linear_num_value_heads=6,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1009,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=12,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        mrope_section=[2, 1, 1],
        eos_token_id=7,
        mtp_num_hidden_layers=1,
    )


def _randomize(model, scale=0.5):
    from mlx.utils import tree_map

    mx.random.seed(11)

    def random_parameter(parameter):
        if parameter.dtype in (mx.int32, mx.int64, mx.uint32):
            return parameter
        return (
            mx.random.normal(parameter.shape).astype(parameter.dtype)
            * scale
            / max(1, parameter.shape[-1]) ** 0.5
        )

    model.update(tree_map(random_parameter, model.parameters()))


def _logits(model, ids, cache=None):
    return model(ids, cache=cache).logits


def test_qwen4_exp_native_state_chunk_parity_off_boundaries():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    rng = np.random.default_rng(3)
    for sequence_length, chunks in ((29, (13, 9, 7)), (61, (17, 19, 25))):
        ids_np = rng.integers(0, args.vocab_size, size=(1, sequence_length))
        ids_np[0, 11] = args.eos_token_id
        ids = mx.array(ids_np)
        reference = _logits(model, ids)
        mx.eval(reference)
        assert np.asarray(reference).size > 0
        assert not np.isnan(np.asarray(reference)).any()

        cache = model.make_cache()
        cached = _logits(model, ids, cache=cache)
        mx.eval(cached)
        assert np.max(np.abs(np.asarray(cached - reference))) < 1e-4

        cache = model.make_cache()
        outputs = []
        offset = 0
        for chunk_length in chunks:
            outputs.append(
                _logits(model, ids[:, offset : offset + chunk_length], cache=cache)
            )
            offset += chunk_length
        chunked = mx.concatenate(outputs, axis=1)
        mx.eval(chunked)
        assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

        cache = model.make_cache()
        stepped = mx.concatenate(
            [
                _logits(model, ids[:, index : index + 1], cache=cache)
                for index in range(sequence_length)
            ],
            axis=1,
        )
        mx.eval(stepped)
        assert np.max(np.abs(np.asarray(stepped - reference))) < 1e-4

    main_logits, hidden = model(
        ids[:, :5], cache=model.make_cache(), return_hidden=True
    )
    assert hidden.shape == (1, 5, args.hc_count * args.hidden_size)
    mtp_logits = model.mtp_forward(
        hidden[:, -1:, :], ids[:, 5:6], model.make_mtp_cache()
    )
    mx.eval(main_logits, hidden, mtp_logits)
    assert mtp_logits.shape == (1, 1, args.vocab_size)
    assert not np.isnan(np.asarray(mtp_logits)).any()

    mtp_logits, mtp_hidden = model.mtp_forward(
        hidden[:, -1:, :],
        ids[:, 5:6],
        model.make_mtp_cache(),
        return_hidden=True,
    )
    mx.eval(mtp_logits, mtp_hidden)
    assert mtp_hidden.shape == (1, 1, args.hc_count * args.hidden_size)
    assert not np.isnan(np.asarray(mtp_hidden)).any()


def test_qwen4_exp_mrope_and_sparse_index_state_survive_chunking():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    ids = mx.array(np.arange(23, dtype=np.int32)[None, :] % args.vocab_size)
    text = np.arange(23, dtype=np.int32)
    positions = mx.array(
        np.stack(
            [
                text,
                np.where((text >= 5) & (text < 13), text // 2, text),
                np.where((text >= 5) & (text < 13), text % 4, text),
            ],
            axis=0,
        )[:, None, :]
    )

    reference = model(ids, position_ids=positions).logits
    cache = model.make_cache()
    pieces = []
    start = 0
    for width in (7, 9, 7):
        pieces.append(
            model(
                ids[:, start : start + width],
                cache=cache,
                position_ids=positions[:, :, start : start + width],
            ).logits
        )
        start += width
    chunked = mx.concatenate(pieces, axis=1)
    mx.eval(reference, chunked)
    assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

    qsa_layers = [
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    ]
    assert qsa_layers
    for layer_cache in qsa_layers:
        keys, values, indexer_keys = layer_cache.state
        assert layer_cache.offset == 23
        assert keys.shape[2] == values.shape[2] == indexer_keys.shape[2] == 23
        assert indexer_keys.shape[1] == 1
        assert indexer_keys.shape[-1] == args.indexer_head_dim + 3
        np.testing.assert_array_equal(
            np.asarray(indexer_keys[:, 0, :, -3:]),
            np.asarray(positions.transpose(1, 2, 0)).astype(np.float32),
        )


def test_qwen4_exp_qsa_persists_raw_index_projection_before_pool_norm_and_rope():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    qsa_index = args.layer_types.index("full_attention")
    indexer = model.layers[qsa_index].self_attn.indexer
    cache = model.make_cache()[qsa_index]
    length = 7
    hidden = (
        mx.arange(length * args.hidden_size, dtype=mx.float32)
        .reshape(1, length, args.hidden_size)
        / 101.0
    )
    positions = mx.array(
        np.stack(
            [
                np.arange(40, 40 + length, dtype=np.int32),
                np.array([3, 3, 4, 4, 5, 5, 6], dtype=np.int32),
                np.array([9, 10, 9, 10, 9, 10, 9], dtype=np.int32),
            ],
            axis=0,
        )[:, None, :]
    )

    # The indexer is called after the main K/V lane, so advance the logical KV
    # offset exactly as QSAAttention does before appending the index payload.
    cache.update_and_fetch(
        mx.zeros(
            (1, args.num_key_value_heads, length, args.head_dim),
            dtype=mx.float32,
        ),
        mx.zeros(
            (1, args.num_key_value_heads, length, args.head_dim),
            dtype=mx.float32,
        ),
    )
    projected = indexer.index_qk_proj(hidden)
    _, expected_raw = mx.split(
        projected, [args.indexer_n_heads * args.indexer_head_dim], axis=-1
    )
    mask = indexer(hidden, cache, offset=0, position_ids=positions)
    mx.eval(expected_raw, mask, cache.idx_keys)

    payload = np.asarray(cache.idx_keys[:, 0, :, :])
    np.testing.assert_array_equal(
        payload[..., : args.indexer_head_dim],
        np.asarray(expected_raw).astype(np.float32),
    )
    np.testing.assert_array_equal(
        payload[..., -3:],
        np.asarray(positions.transpose(1, 2, 0)).astype(np.float32),
    )


def test_qwen4_exp_qsa_bypasses_selection_while_all_blocks_fit_budget(monkeypatch):
    """The native short-context path stores index state but never selects it."""
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    qsa_index = args.layer_types.index("full_attention")
    indexer = model.layers[qsa_index].self_attn.indexer
    cache = model.make_cache()[qsa_index]
    length = args.indexer_budget
    cache.update_and_fetch(
        mx.zeros(
            (1, args.num_key_value_heads, length, args.head_dim),
            dtype=mx.float32,
        ),
        mx.zeros(
            (1, args.num_key_value_heads, length, args.head_dim),
            dtype=mx.float32,
        ),
    )

    def forbidden_argpartition(*_args, **_kwargs):
        raise AssertionError("short-context QSA must not score/select blocks")

    monkeypatch.setattr(mx, "argpartition", forbidden_argpartition)
    hidden = mx.zeros((1, length, args.hidden_size), dtype=mx.float32)
    mask = indexer(hidden, cache, offset=0)

    assert mask is None
    assert cache.idx_keys is not None
    assert cache.idx_keys.shape[2] == length


def test_qwen4_exp_qsa_selects_complete_blocks_and_keeps_off_boundary_tail():
    """Past the budget, select whole blocks and retain the live tail token."""
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    qsa_index = args.layer_types.index("full_attention")
    indexer = model.layers[qsa_index].self_attn.indexer
    cache = model.make_cache()[qsa_index]
    prefix_length = args.indexer_budget + args.indexer_compress_ratio

    prefix_hidden = mx.arange(
        prefix_length * args.hidden_size, dtype=mx.float32
    ).reshape(1, prefix_length, args.hidden_size)
    cache.update_and_fetch(
        mx.zeros(
            (1, args.num_key_value_heads, prefix_length, args.head_dim),
            dtype=mx.float32,
        ),
        mx.zeros(
            (1, args.num_key_value_heads, prefix_length, args.head_dim),
            dtype=mx.float32,
        ),
    )
    indexer(prefix_hidden, cache, offset=0)

    cache.update_and_fetch(
        mx.zeros(
            (1, args.num_key_value_heads, 1, args.head_dim), dtype=mx.float32
        ),
        mx.zeros(
            (1, args.num_key_value_heads, 1, args.head_dim), dtype=mx.float32
        ),
    )
    mask = indexer(
        mx.ones((1, 1, args.hidden_size), dtype=mx.float32),
        cache,
        offset=prefix_length,
    )
    mx.eval(mask)

    assert mask is not None
    assert mask.shape == (1, 1, 1, prefix_length + 1)
    visible = np.isfinite(np.asarray(mask)[0, 0, 0])
    complete = visible[:prefix_length].reshape(
        -1, args.indexer_compress_ratio
    )
    assert np.all(complete == complete[:, :1])
    assert complete[:, 0].sum() == args.indexer_budget // args.indexer_compress_ratio
    assert visible[-1]


def test_qwen4_exp_qsa_sparse_masks_follow_float16_attention_dtype():
    from mlx.utils import tree_map

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)

    def to_float16(parameter):
        if parameter.dtype in (mx.int32, mx.int64, mx.uint32):
            return parameter
        return parameter.astype(mx.float16)

    model.update(tree_map(to_float16, model.parameters()))
    mx.eval(model.parameters())

    ids = mx.array(np.arange(14, dtype=np.int32)[None, :] % args.vocab_size)
    cache = model.make_cache()
    prefill = model(ids[:, :13], cache=cache).logits
    decode = model(ids[:, 13:], cache=cache).logits
    mx.eval(prefill, decode)

    assert prefill.dtype == mx.float16
    assert decode.dtype == mx.float16
    assert not np.isnan(np.asarray(prefill)).any()
    assert not np.isnan(np.asarray(decode)).any()


def test_qwen4_exp_qsa_and_ple_support_synchronous_batch_chunking():
    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    ids = mx.array(
        np.stack(
            [
                np.arange(19, dtype=np.int32) % args.vocab_size,
                (np.arange(19, dtype=np.int32) * 7 + 3) % args.vocab_size,
            ]
        )
    )
    reference = model(ids).logits
    cache = model.make_cache()
    chunked = mx.concatenate(
        [
            model(ids[:, :7], cache=cache).logits,
            model(ids[:, 7:12], cache=cache).logits,
            model(ids[:, 12:], cache=cache).logits,
        ],
        axis=1,
    )
    mx.eval(reference, chunked)
    assert np.max(np.abs(np.asarray(chunked - reference))) < 1e-4

    qsa_cache = next(
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    )
    assert qsa_cache.state[2].shape[:3] == (2, 1, 19)
    assert qsa_cache.state[2].shape[-1] == args.indexer_head_dim + 3
    ple_cache = cache[args.ple_layer_ids[0] - 1]
    assert ple_cache.cache[2].shape == (2, args.ngram_size - 1)


def test_qwen4_exp_registry_and_runtime_registration_are_source_available():
    from vmlx_engine.model_config_registry import ModelConfigRegistry
    from vmlx_engine.model_configs import register_all
    from vmlx_engine.models.qwen4_exp.register import (
        qwen4_exp_runtime_available,
        register_qwen4_exp_runtime,
    )

    registry = ModelConfigRegistry()
    register_all(registry)
    config = next(item for item in registry._configs if item.family_name == "qwen4_exp")
    assert config.family_name == "qwen4_exp"
    assert config.cache_type == "hybrid"
    assert config.cache_subtype == "qsa_gdn_ple_v1"
    assert config.is_mllm is True
    assert config.tool_parser == "qwen"
    assert config.reasoning_parser == "qwen3"
    assert config.preserve_native_tool_format is True
    assert config.supports_instruct_mode is True
    assert config.supported_reasoning_efforts == ["low", "medium", "xhigh"]
    assert config.architecture_hints["ple_storage"] == "ssd_row_addressed"
    assert config.architecture_hints["cache_precision"] == "full"

    assert qwen4_exp_runtime_available() is True
    assert register_qwen4_exp_runtime() is True


def test_qwen4_exp_loader_never_requests_ple_table_tensors():
    from pathlib import Path

    from vmlx_engine.models.qwen4_exp.loader import _load_non_table_weight_files

    class FakeHandle:
        def __init__(self, tensors):
            self.tensors = tensors
            self.requested = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {"format": "mlx"}

        def keys(self):
            return list(self.tensors)

        def get_tensor(self, key):
            self.requested.append(key)
            return self.tensors[key]

    prefix = "model.language_model.layers.1.ple.ple_embedding."
    tensors = {
        prefix + "ngram_embedding.shard_0.weight": mx.zeros((3, 20), mx.uint32),
        prefix + "ngram_embedding.shard_0.scales": mx.zeros((3, 3)),
        prefix + "ngram_embedding.shard_0.biases": mx.zeros((3, 3)),
        prefix + "layer_multipliers": mx.array([1, 3, 5]),
        prefix + "ngram_heads_vocab_sizes": mx.array([11, 13]),
        prefix + "ngram_heads_offsets": mx.array([0, 11]),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones((2, 2)),
    }
    handle = FakeHandle(tensors)

    def fake_open(_path, framework):
        assert framework == "mlx"
        return handle

    weights, is_mlx, buffers = _load_non_table_weight_files(
        [Path("synthetic-00001.safetensors")],
        frozenset(),
        safe_open_fn=fake_open,
    )

    assert is_mlx is True
    assert set(weights) == {"model.language_model.layers.0.self_attn.q_proj.weight"}
    assert set(buffers) == {
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
    }
    assert all("ngram_embedding.shard_0" not in key for key in handle.requested)


def test_qwen4_exp_embedded_jang_bit_map_becomes_runtime_quantization(tmp_path):
    import json

    from vmlx_engine.models.qwen4_exp.loader import _load_runtime_config
    from vmlx_engine.models.qwen4_exp.table_reader import resolve_jang_bit_map_spec

    embedded = {
        "format": "jang_v2",
        "norm_convention": "runtime_plus1_applied",
        "bit_map": {
            "default": {"bits": 8, "group_size": 64},
            "language_model.layers.*.mlp.switch_mlp": {
                "bits": 4,
                "group_size": 64,
            },
            "language_model.layers.*.ple.ngram_embedding.shards.0.weight": {
                "bits": 3,
                "group_size": 32,
            },
            "mtp.": {"bits": 4, "group_size": 64},
            "lm_head": {"bits": 6, "group_size": 64},
        },
        "quantization": {"calibrated": True},
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "jang_config": embedded})
    )

    runtime, affine1_modules, bit_map, runtime_sanitized = _load_runtime_config(
        tmp_path
    )
    assert affine1_modules == frozenset()
    assert bit_map == embedded["bit_map"]
    assert runtime_sanitized is True
    assert runtime["quantization"] == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    assert resolve_jang_bit_map_spec(
        "language_model.layers.1.ple.ngram_embedding.shards.0", bit_map
    ) == {"bits": 3, "group_size": 32, "mode": "affine"}
    assert resolve_jang_bit_map_spec(
        "language_model.model.layers.7.mlp.switch_mlp.down_proj", bit_map
    ) == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert resolve_jang_bit_map_spec(
        "language_model.mtp.layers.0.self_attn.q_proj", bit_map
    ) == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert resolve_jang_bit_map_spec(
        "language_model.lm_head", bit_map
    ) == {"bits": 6, "group_size": 64, "mode": "affine"}


def test_qwen4_exp_embedded_no_mtp_stamp_disables_architecture_placeholder(tmp_path):
    import json

    from vmlx_engine.models.qwen4_exp.loader import _load_runtime_config

    embedded = {
        "format": "jang_v2",
        "norm_convention": "runtime_plus1_applied",
        "mtp": {"mtp_mode": "none"},
        "bit_map": {
            "default": {"bits": 4, "group_size": 64},
            "mtp.": {"bits": 4, "group_size": 64},
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "mtp_num_hidden_layers": 1,
                "num_nextn_predict_layers": 1,
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 1,
                    "num_nextn_predict_layers": 1,
                },
                "jang_config": embedded,
            }
        )
    )

    runtime, _affine1, bit_map, sanitized = _load_runtime_config(tmp_path)

    assert sanitized is True
    assert bit_map == embedded["bit_map"]
    assert runtime["mtp_num_hidden_layers"] == 0
    assert runtime["num_nextn_predict_layers"] == 0
    assert runtime["text_config"]["mtp_num_hidden_layers"] == 0
    assert runtime["text_config"]["num_nextn_predict_layers"] == 0


def test_qwen4_exp_jang_metadata_and_bit_map_conflicts_fail_closed(tmp_path):
    import json

    from vmlx_engine.models.qwen4_exp.loader import _load_runtime_config
    from vmlx_engine.models.qwen4_exp.table_reader import resolve_jang_bit_map_spec

    embedded = {
        "format": "jang_v2",
        "norm_convention": "runtime_plus1_applied",
        "bit_map": {"default": {"bits": 4, "group_size": 64}},
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "jang_config": embedded})
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps({**embedded, "format": "different"})
    )
    with pytest.raises(ValueError, match="sidecar and embedded JANG metadata disagree"):
        _load_runtime_config(tmp_path)

    conflicting = {
        "default": {"bits": 8, "group_size": 64},
        "language_model.layers.*.self_attn": {"bits": 8, "group_size": 64},
        "language_model.layers.?.self_attn": {"bits": 4, "group_size": 64},
    }
    with pytest.raises(ValueError, match="conflicting equally-specific"):
        resolve_jang_bit_map_spec(
            "language_model.layers.1.self_attn.q_proj", conflicting
        )

    unstamped = {
        "format": "jang_v2",
        "bit_map": {"default": {"bits": 4, "group_size": 64}},
    }
    (tmp_path / "jang_config.json").unlink()
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "jang_config": unstamped})
    )
    with pytest.raises(ValueError, match="runtime_plus1_applied"):
        _load_runtime_config(tmp_path)


def test_qwen4_exp_embedded_bit_map_drives_model_quantizer(monkeypatch):
    from vmlx_engine.models.qwen4_exp import loader

    bit_map = {
        "default": {"bits": 8, "group_size": 64},
        "language_model.layers.*.mlp.switch_mlp": {
            "bits": 4,
            "group_size": 64,
        },
        "mtp.": {"bits": 4, "group_size": 64},
        "lm_head": {"bits": 6, "group_size": 64},
    }
    captured = {}

    def fake_quantize(_model, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(loader.nn, "quantize", fake_quantize)
    linear = nn.Linear(64, 8, bias=False)
    loader._quantize_model(
        object(),
        {"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        {
            "language_model.layers.3.mlp.switch_mlp.down_proj.weight": mx.zeros(
                (8, 8), dtype=mx.uint32
            ),
            "language_model.layers.3.mlp.switch_mlp.down_proj.scales": mx.ones(
                (8, 1)
            ),
            "language_model.layers.3.mlp.switch_mlp.down_proj.biases": mx.zeros(
                (8, 1)
            ),
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros(
                (8, 8), dtype=mx.uint32
            ),
            "mtp.layers.0.self_attn.q_proj.scales": mx.ones((8, 1)),
            "mtp.layers.0.self_attn.q_proj.biases": mx.zeros((8, 1)),
            "language_model.lm_head.weight": mx.zeros(
                (8, 16), dtype=mx.uint32
            ),
            "language_model.lm_head.scales": mx.ones((8, 2)),
            "language_model.lm_head.biases": mx.zeros((8, 2)),
            "language_model.model.layers.3.mlp.gate.weight": mx.zeros(
                (8, 16), dtype=mx.uint32
            ),
            "language_model.model.layers.3.mlp.gate.scales": mx.ones((8, 1)),
            "language_model.model.layers.3.mlp.gate.biases": mx.zeros((8, 1)),
        },
        bit_map,
    )

    predicate = captured["class_predicate"]
    assert predicate(
        "language_model.model.layers.3.mlp.switch_mlp.down_proj", linear
    ) == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert predicate("language_model.mtp.layers.0.self_attn.q_proj", linear) == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert predicate("language_model.lm_head", linear) == {
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
    }
    assert (
        predicate("language_model.model.layers.4.self_attn.q_proj", linear)
        is False
    )

    class RouterAwareModel:
        @staticmethod
        def quant_predicate(path, _module):
            return not path.endswith("mlp.gate")

    captured.clear()
    loader._quantize_model(
        RouterAwareModel(),
        {"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        {
            "language_model.model.layers.3.mlp.gate.weight": mx.zeros(
                (8, 16), dtype=mx.uint32
            ),
            "language_model.model.layers.3.mlp.gate.scales": mx.ones((8, 1)),
            "language_model.model.layers.3.mlp.gate.biases": mx.zeros((8, 1)),
        },
        bit_map,
    )
    assert captured["class_predicate"](
        "language_model.model.layers.3.mlp.gate", linear
    ) == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_qwen4_exp_checkpoint_shapes_own_per_module_quantization(monkeypatch):
    from vmlx_engine.models.qwen4_exp import loader

    captured = {}
    monkeypatch.setattr(
        loader.nn,
        "quantize",
        lambda _model, **kwargs: captured.update(kwargs),
    )
    loader._quantize_model(
        object(),
        {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}},
        {
            "language_model.model.embed_tokens.weight": mx.zeros(
                (16, 16), dtype=mx.uint32
            ),
            "language_model.model.embed_tokens.scales": mx.ones((16, 2)),
            "language_model.model.embed_tokens.biases": mx.zeros((16, 2)),
        },
        {"default": {"bits": 4, "group_size": 64}},
    )
    embedding = nn.Embedding(16, 64)
    assert captured["class_predicate"](
        "language_model.model.embed_tokens", embedding
    ) == {"bits": 8, "group_size": 32, "mode": "affine"}

    malformed = captured["class_predicate"]
    embedding.weight = mx.zeros((16, 96))
    with pytest.raises(ValueError, match="integral bit width"):
        malformed("language_model.model.embed_tokens", embedding)


def test_qwen4_exp_ple_layout_resolution_is_complete_and_unambiguous():
    from vmlx_engine.models.qwen4_exp.loader import _resolve_ple_module_key_format

    raw_prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    )
    weight_map = {
        f"{raw_prefix}{shard}.{suffix}": f"part-{shard}.safetensors"
        for shard in range(3)
        for suffix in ("weight", "scales", "biases")
    }
    assert _resolve_ple_module_key_format(weight_map, 3) == raw_prefix + "{}"

    del weight_map[f"{raw_prefix}2.biases"]
    with pytest.raises(ValueError, match="no complete indexed PLE"):
        _resolve_ple_module_key_format(weight_map, 3)

    jang_prefix = "language_model.layers.1.ple.ngram_embedding.shards."
    jang_weight_map = {
        f"{jang_prefix}{shard}.{suffix}": f"part-{shard}.safetensors"
        for shard in range(3)
        for suffix in ("weight", "scales", "biases")
    }
    assert _resolve_ple_module_key_format(jang_weight_map, 3) == jang_prefix + "{}"

    ambiguous = dict(jang_weight_map)
    ambiguous.update(
        {
            f"{raw_prefix}{shard}.{suffix}": f"other-{shard}.safetensors"
            for shard in range(3)
            for suffix in ("weight", "scales", "biases")
        }
    )
    with pytest.raises(ValueError, match="ambiguous PLE"):
        _resolve_ple_module_key_format(ambiguous, 3)


def test_qwen4_exp_converted_roots_are_normalized_and_collisions_fail_closed():
    from vmlx_engine.models.qwen4_exp.loader import _normalize_runtime_weight_names

    mtp = mx.ones((2, 2))
    body = mx.zeros((2, 2))
    normalized = _normalize_runtime_weight_names(
        {
            "mtp.fc_embedding.weight": mtp,
            "language_model.embed_tokens.weight": body,
            "visual.blocks.0.attn.qkv.weight": body,
            "lm_head.weight": body,
        }
    )
    assert set(normalized) == {
        "language_model.mtp.fc_embedding.weight",
        "language_model.model.embed_tokens.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
        "language_model.lm_head.weight",
    }
    assert normalized["language_model.mtp.fc_embedding.weight"] is mtp

    hf_patch = mx.ones((8, 3, 2, 4, 4))
    mlx_patch = mx.ones((8, 2, 4, 4, 3))
    normalized = _normalize_runtime_weight_names(
        {"visual.patch_embed.proj.weight": hf_patch}
    )
    assert normalized["vision_tower.patch_embed.proj.weight"].shape == (
        8,
        2,
        4,
        4,
        3,
    )
    normalized = _normalize_runtime_weight_names(
        {"vision_tower.patch_embed.proj.weight": mlx_patch}
    )
    assert normalized["vision_tower.patch_embed.proj.weight"] is mlx_patch

    ple_conv = mx.ones((8, 1, 4))
    normalized = _normalize_runtime_weight_names(
        {"language_model.layers.1.ple.conv1d_weight": ple_conv.squeeze(1)}
    )
    assert set(normalized) == {
        "language_model.model.layers.1.ple.conv1d_weight"
    }
    assert normalized[
        "language_model.model.layers.1.ple.conv1d_weight"
    ].shape == (8, 4)

    fused_weight = mx.arange(2 * 8 * 3).reshape(2, 8, 3)
    fused_scales = mx.arange(2 * 8 * 2).reshape(2, 8, 2)
    fused_biases = fused_scales + 1
    down_weight = mx.ones((2, 6, 3))
    down_scales = mx.ones((2, 6, 2))
    down_biases = mx.zeros((2, 6, 2))
    normalized = _normalize_runtime_weight_names(
        {
            "mtp.layers.0.mlp.experts.gate_up_proj.weight": fused_weight,
            "mtp.layers.0.mlp.experts.gate_up_proj.scales": fused_scales,
            "mtp.layers.0.mlp.experts.gate_up_proj.biases": fused_biases,
            "mtp.layers.0.mlp.experts.down_proj.weight": down_weight,
            "mtp.layers.0.mlp.experts.down_proj.scales": down_scales,
            "mtp.layers.0.mlp.experts.down_proj.biases": down_biases,
        }
    )
    assert set(normalized) == {
        f"language_model.mtp.layers.0.mlp.switch_mlp.{projection}.{suffix}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "scales", "biases")
    }
    assert normalized[
        "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
    ].shape == (2, 4, 3)
    assert normalized[
        "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.scales"
    ].shape == (2, 4, 2)

    with pytest.raises(ValueError, match="singleton input channel"):
        _normalize_runtime_weight_names(
            {
                "language_model.model.layers.1.ple.conv1d.weight": mx.ones(
                    (8, 2, 4)
                )
            }
        )

    with pytest.raises(ValueError, match="weight-name collision"):
        _normalize_runtime_weight_names(
            {
                "mtp.fc_embedding.weight": mtp,
                "language_model.mtp.fc_embedding.weight": body,
            }
        )

    with pytest.raises(ValueError, match="weight-name collision"):
        _normalize_runtime_weight_names(
            {
                "visual.blocks.0.attn.qkv.weight": mtp,
                "vision_tower.blocks.0.attn.qkv.weight": body,
            }
        )


def test_qwen4_exp_ple_affine_layout_uses_exact_160_dimensional_rows():
    from vmlx_engine.models.qwen4_exp.table_reader import _validate_affine_layout

    # The old scale-derived formula yielded 64 * 3 = 192. The runtime contract
    # is PLE 2560 / 16 bigram+trigram heads = exactly 160 dimensions per row.
    _validate_affine_layout(
        weight_shape=(7, 20),
        weight_dtype="U32",
        scales_shape=(7, 3),
        biases_shape=(7, 3),
        group_size=64,
        logical_bits=4,
        storage_bits=4,
        head_dim=160,
    )
    with pytest.raises(ValueError, match="packed width"):
        _validate_affine_layout(
            weight_shape=(7, 20),
            weight_dtype="U32",
            scales_shape=(7, 3),
            biases_shape=(7, 3),
            group_size=64,
            logical_bits=4,
            storage_bits=4,
            head_dim=192,
        )


def test_qwen4_exp_nested_config_maps_qsa_ple_and_mtp_contracts():
    cfg = {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "full_attention_interval": 4,
        "sparse_attention_config": {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "budget": 32,
            "block_size": 4,
        },
        "ple_config": {
            "ple_layer_ids": [2],
            "ple_embed_dim": 64,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "split_ngram_parts": 4,
        },
        "num_nextn_predict_layers": 1,
    }
    args = Qwen4ExpTextArgs.from_config(cfg)
    assert args.indexer_n_heads == 2
    assert args.indexer_kv_heads == 1
    assert args.indexer_head_dim == 16
    assert args.indexer_budget == 32
    assert args.indexer_compress_ratio == 4
    assert args.ple_layer_ids == [2]
    assert args.mtp_num_hidden_layers == 1
    assert args.indexer_budget // args.indexer_compress_ratio == 8


def test_qwen4_exp_ple_companion_is_ssd_only_and_survives_restart(tmp_path):
    from vmlx_engine.mllm_batch_generator import (
        _hybrid_cache_layout,
        _uses_ssm_companion_cache,
    )
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())

    token_ids = list(range(29))
    cache = model.make_cache()
    logits = model(mx.array([token_ids]), cache=cache).logits
    mx.eval(logits)

    # Layer 2 (one-based) is the PLE-bearing GDN layer. Its cumulative state is
    # exactly: GDN conv, GDN recurrent delta, PLE prior token IDs, and PLE
    # dilated-convolution state. All four are required for answer-preserving
    # prefix restore; none may be reconstructed from QSA K/V blocks.
    ple_cache = cache[args.ple_layer_ids[0] - 1]
    assert len(ple_cache.cache) == 4
    assert all(value is not None for value in ple_cache.cache)

    owner, owner_path, template, kv_positions, error = _hybrid_cache_layout(
        model, model
    )
    assert owner is model
    assert owner_path == "language_model"
    assert error is None
    assert template is not None
    assert kv_positions == [
        index
        for index, layer_type in enumerate(args.layer_types)
        if layer_type == "full_attention"
    ]
    assert _uses_ssm_companion_cache(
        kv_positions,
        len(template),
        mixed_attention=False,
    ) is True

    kv_set = set(kv_positions)
    companion_layers = [
        layer_cache
        for index, layer_cache in enumerate(cache)
        if index not in kv_set
    ]
    assert len(companion_layers) == args.num_hidden_layers - len(kv_positions)
    assert [len(layer.cache) for layer in companion_layers].count(4) == 1
    assert [len(layer.cache) for layer in companion_layers].count(2) == (
        len(companion_layers) - 1
    )
    expected = [
        [np.asarray(value) for value in layer.cache]
        for layer in companion_layers
    ]

    disk_dir = tmp_path / "qwen4-ple-companion"
    first_disk = SSMCompanionDiskStore(
        directory=disk_dir,
        budget_bytes=64 * 1024 * 1024,
    )
    first = SSMCompanionCache(
        max_entries=0,
        max_bytes=0,
        model_key="qwen4-exp-synthetic",
        disk_store=first_disk,
    )
    assert first.ram_enabled is False
    first.store(
        token_ids,
        len(token_ids),
        companion_layers,
        is_complete=True,
    )
    assert first.size == 0
    assert first.total_nbytes == 0
    assert first_disk.wait_for_pending(timeout=10.0)
    assert first_disk.shutdown(timeout=10.0)

    # New objects emulate an engine restart: no L1 state or in-process index is
    # retained. The SSD entry must refault with its native dtype/value intact.
    second_disk = SSMCompanionDiskStore(
        directory=disk_dir,
        budget_bytes=64 * 1024 * 1024,
    )
    second = SSMCompanionCache(
        max_entries=0,
        max_bytes=0,
        model_key="qwen4-exp-synthetic",
        disk_store=second_disk,
    )
    restored = second.fetch(token_ids, len(token_ids))
    assert restored is not None
    states, is_complete = restored
    assert is_complete is True
    assert second.size == 0
    assert second.total_nbytes == 0
    assert len(states) == len(companion_layers)
    for got_layer, want_layer in zip(states, expected):
        assert len(got_layer.cache) == len(want_layer)
        for got, want in zip(got_layer.cache, want_layer):
            got_np = np.asarray(got)
            assert got_np.dtype == want.dtype
            np.testing.assert_array_equal(got_np, want)
    assert second_disk.shutdown(timeout=10.0)


def test_qwen4_exp_qsa_cache_requires_all_three_native_lanes():
    from vmlx_engine.cache_record_validator import validate_live_cache

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    cache = model.make_cache()
    logits = model(mx.array([list(range(13))]), cache=cache).logits
    mx.eval(logits)

    qsa_cache = next(
        layer_cache
        for layer_cache, layer_type in zip(cache, args.layer_types)
        if layer_type == "full_attention"
    )
    ok, reason, _ = validate_live_cache([qsa_cache], expected_num_layers=1)
    assert ok, reason

    # A K/V-only restore is unsafe: the QSA selector would score against a
    # different prefix than attention. Validation must fail closed.
    qsa_cache.idx_keys = None
    ok, reason, _ = validate_live_cache([qsa_cache], expected_num_layers=1)
    assert ok is False
    assert "idx_keys" in reason


def test_qwen4_exp_qsa_three_lane_prompt_cache_survives_restart(tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    token_ids = list(range(13))
    live = model.make_cache()
    logits = model(mx.array([token_ids]), cache=live).logits
    mx.eval(logits)

    qsa_index = args.layer_types.index("full_attention")
    live_qsa = live[qsa_index]
    assert live_qsa.offset == 13
    assert live_qsa.state[0].shape[2] == 13
    assert live_qsa.state[1].shape[2] == 13
    assert live_qsa.state[2].shape[2] == 13

    first = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        assert first.store(token_ids, [live_qsa])
    finally:
        first.shutdown()

    second = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        restored = second.fetch(token_ids)
        assert restored is not None
        restored_qsa = restored[0]
        keys, values, indexer_keys = restored_qsa.state
        assert restored_qsa.offset == 13
        assert keys.shape[2] == values.shape[2] == indexer_keys.shape[2] == 13
        np.testing.assert_array_equal(
            np.asarray(indexer_keys), np.asarray(live_qsa.state[2])
        )
    finally:
        second.shutdown()


def test_qwen4_exp_native_mtp_runtime_is_detected_from_the_attached_head():
    from vmlx_engine.native_mtp import model_has_native_mtp_runtime

    model = LanguageModel(_tiny_args())
    assert model_has_native_mtp_runtime(model) is True


def _quantize_qwen4_lm_head(model, *, bits: int):
    model.lm_head = model.lm_head.to_quantized(group_size=64, bits=bits)
    mx.eval(model.lm_head.weight, model.lm_head.scales, model.lm_head.biases)


def test_qwen4_exp_mtp_draft_head_is_default_off_and_keeps_target_head(monkeypatch):
    monkeypatch.delenv("VMLINUX_QWEN4_MTP_DRAFT_HEAD_BITS", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_MTP_DRAFT_HEAD_BITS", raising=False)
    model = LanguageModel(_tiny_args())
    _quantize_qwen4_lm_head(model, bits=8)
    target_head = model.lm_head

    status = model.prepare_mtp_draft_head()
    assert status["configured"] is False
    assert status["reason"] == "disabled"
    assert model.lm_head is target_head


def test_qwen4_exp_mtp_draft_head_q4_is_proposal_only_and_observable(monkeypatch):
    monkeypatch.setenv("VMLINUX_QWEN4_MTP_DRAFT_HEAD_BITS", "4")
    model = LanguageModel(_tiny_args())
    _randomize(model)
    _quantize_qwen4_lm_head(model, bits=8)
    target_head = model.lm_head
    ids = mx.array([[3, 5, 7, 11, 13, 17]], dtype=mx.int32)
    target_before = _logits(model, ids)

    status = model.prepare_mtp_draft_head()
    assert status["available"] is True
    assert status["source_bits"] == 8
    assert status["draft_bits"] == 4
    assert status["group_size"] == 64
    assert status["calls"] == 0
    assert model.lm_head is target_head
    assert not any("draft_head" in name for name, _module in model.named_modules())

    _main_logits, hidden = model(ids, cache=model.make_cache(), return_hidden=True)
    proposal_logits = model.mtp_forward(
        hidden[:, -1:, :],
        mx.array([[19]], dtype=mx.int32),
        model.make_mtp_cache(),
    )
    target_after = _logits(model, ids)
    mx.eval(target_before, target_after, proposal_logits)
    np.testing.assert_array_equal(np.asarray(target_after), np.asarray(target_before))
    assert proposal_logits.shape == (1, 1, model.args.vocab_size)
    status = model.mtp_draft_head_status()
    assert status["active_observed"] is True
    assert status["calls"] == 1


def test_qwen4_exp_mtp_draft_head_rejects_unsupported_source_layout(monkeypatch):
    monkeypatch.setenv("VMLINUX_QWEN4_MTP_DRAFT_HEAD_BITS", "4")
    model = LanguageModel(_tiny_args())
    _quantize_qwen4_lm_head(model, bits=6)

    first = model.prepare_mtp_draft_head()
    second = model.prepare_mtp_draft_head()
    assert first == second
    assert first["available"] is False
    assert first["source_bits"] == 6
    assert first["reason"] == "unsupported_source_layout"


def test_qwen4_exp_mtp_fusion_preserves_distinct_hyper_connection_branches():
    from vmlx_engine.models.qwen4_exp.language import MTPModule

    args = _tiny_args()
    mtp = MTPModule(args)
    mtp.pre_fc_norm_embedding.weight = mx.zeros((args.hidden_size,))
    mtp.pre_fc_norm_hidden.weight = mx.zeros((args.hc_count * args.hidden_size,))
    mtp.fc_embedding.weight = mx.eye(args.hidden_size)
    mtp.fc_hidden.weight = mx.eye(args.hidden_size)

    token_embeddings = mx.arange(1, args.hidden_size + 1, dtype=mx.float32).reshape(
        1, 1, args.hidden_size
    )
    hidden = mx.zeros((1, 1, args.hc_count, args.hidden_size))
    hidden[..., 0, 0] = 1.0
    hidden[..., 1, 1] = 2.0
    hidden[..., 2, 2] = 3.0
    hidden[..., 3, 3] = 4.0
    hidden = hidden.reshape(1, 1, args.hc_count * args.hidden_size)
    fused = mtp.fuse_inputs(hidden, token_embeddings).reshape(
        1, 1, args.hc_count, args.hidden_size
    )
    mx.eval(fused)

    assert fused.shape == (1, 1, args.hc_count, args.hidden_size)
    assert not mx.array_equal(fused[..., 0, :], fused[..., 1, :]).item()
    expected_embedding = mtp.fc_embedding(
        mtp.pre_fc_norm_embedding(token_embeddings)
    )
    expected_hidden = mtp.fc_hidden(
        mtp.pre_fc_norm_hidden(hidden).reshape(
            1, 1, args.hc_count, args.hidden_size
        )
    )
    np.testing.assert_allclose(
        np.asarray(fused),
        np.asarray(expected_embedding[..., None, :] + expected_hidden),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2])
def test_qwen4_exp_verify_rollback_restores_all_native_states(accepted_drafts):
    from vmlx_engine.mllm_batch_generator import _native_mtp_rollback_to_confirmed

    args = _tiny_args()
    model = LanguageModel(args)
    _randomize(model)
    mx.eval(model.parameters())
    prefix = mx.array([[11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]])
    verify = mx.array([[71, 73, 79, 83]])

    reference_cache = model.make_cache()
    model(prefix, cache=reference_cache)
    model(verify[:, : 1 + accepted_drafts], cache=reference_cache)

    rollback_cache = model.make_cache()
    model(prefix, cache=rollback_cache)
    model(verify, cache=rollback_cache, n_confirmed=1)
    assert _native_mtp_rollback_to_confirmed(
        rollback_cache,
        reject_tokens=3 - accepted_drafts,
        accepted_drafts=accepted_drafts,
    )

    reference_arrays = []
    rollback_arrays = []
    for reference_layer, rollback_layer in zip(reference_cache, rollback_cache):
        reference_arrays.extend(array for array in reference_layer.state if array is not None)
        rollback_arrays.extend(array for array in rollback_layer.state if array is not None)
        assert getattr(reference_layer, "offset", None) == getattr(
            rollback_layer, "offset", None
        )
    mx.eval(*reference_arrays, *rollback_arrays)
    assert len(reference_arrays) == len(rollback_arrays)
    for reference, rolled_back in zip(reference_arrays, rollback_arrays):
        np.testing.assert_allclose(
            np.asarray(rolled_back),
            np.asarray(reference),
            rtol=2e-4,
            atol=2e-4,
        )


def test_qwen4_exp_vlm_config_builds_text_image_and_video_contracts():
    from mlx_vlm.utils import update_module_configs

    from vmlx_engine.native_mtp import model_has_native_mtp_runtime
    from vmlx_engine.models.qwen4_exp.register import register_qwen4_exp_runtime

    assert register_qwen4_exp_runtime() is True
    import mlx_vlm.models.qwen4_exp as model_class

    text = dict(_tiny_args().__dict__)
    text.pop("rotary_dim", None)
    config = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": {
            "model_type": "qwen4_exp",
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": 901,
        "video_token_id": 902,
    }
    parsed = model_class.ModelConfig.from_dict(config)
    parsed = update_module_configs(parsed, model_class, config, ["text", "vision"])
    # Runtime registration loads the same source under mlx_vlm.models, so the
    # class has a distinct Python identity from the vmlx_engine import above.
    assert type(parsed.text_config).__name__ == "Qwen4ExpTextArgs"
    assert parsed.text_config.indexer_budget == _tiny_args().indexer_budget
    assert parsed.vision_config.model_type == "qwen4_exp"
    assert parsed.image_token_index == 901
    assert parsed.video_token_index == 902
    wrapper = model_class.Model(parsed)
    assert wrapper.model_type == "qwen4_exp"
    assert wrapper.language_model.mtp is not None
    assert model_has_native_mtp_runtime(wrapper) is True


def test_qwen4_exp_video_pixels_reach_vision_tower_under_named_contract():
    from types import SimpleNamespace

    from vmlx_engine.mllm_batch_generator import _video_pixel_values_kwarg_name
    from vmlx_engine.models.qwen4_exp.qwen4_exp import Model

    class Qwen4ExpContract:
        get_input_embeddings = Model.get_input_embeddings

    assert _video_pixel_values_kwarg_name(Qwen4ExpContract()) == (
        "pixel_values_videos"
    )

    video_pixels = mx.array([[3.0, 5.0]], dtype=mx.float32)
    video_features = mx.array([[[7.0, 11.0]]], dtype=mx.float32)
    video_grid = mx.array([[1, 1, 1]], dtype=mx.int32)
    calls = []

    class VisionTower:
        patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32))
        )

        def __call__(self, pixels, grid_thw):
            calls.append((pixels, grid_thw))
            return video_features, None

    class LanguageModelStub:
        _position_ids = None
        _rope_deltas = None
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: mx.zeros((1, 1, 2), dtype=mx.float32)
        )

        def get_rope_index(
            self, input_ids, image_grid_thw, video_grid_thw, mask
        ):
            assert image_grid_thw is None
            assert video_grid_thw is video_grid
            return mx.array([[0]], dtype=mx.int32), mx.array([0], dtype=mx.int32)

    wrapper = SimpleNamespace(
        config=SimpleNamespace(image_token_index=901, video_token_index=902),
        vision_tower=VisionTower(),
        language_model=LanguageModelStub(),
        _scatter_media_features=Model._scatter_media_features,
        merge_input_ids_with_image_features=(
            lambda features, inputs_embeds, input_ids, image_token_index,
            video_token_index: (features, None)
        ),
    )
    result = Model.get_input_embeddings(
        wrapper,
        input_ids=mx.array([[902]], dtype=mx.int32),
        pixel_values_videos=video_pixels,
        video_grid_thw=video_grid,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(calls[0][0]), np.asarray(video_pixels))
    assert calls[0][0].dtype == video_pixels.dtype
    assert calls[0][1] is video_grid
    np.testing.assert_array_equal(
        np.asarray(result.inputs_embeds), np.asarray(video_features)
    )


def test_qwen4_exp_connected_image_video_history_encodes_both_modalities():
    from types import SimpleNamespace

    from vmlx_engine.models.qwen4_exp.qwen4_exp import Model

    image_pixels = mx.array([[1.0, 2.0]], dtype=mx.float16)
    video_pixels = mx.array([[3.0, 4.0]], dtype=mx.float16)
    image_features = mx.array([[11.0, 12.0]], dtype=mx.float16)
    video_features = mx.array([[21.0, 22.0]], dtype=mx.float16)
    image_grid = mx.array([[1, 2, 2]], dtype=mx.int32)
    video_grid = mx.array([[1, 2, 2]], dtype=mx.int32)
    calls = []

    class VisionTower:
        patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float16))
        )

        def __call__(self, pixels, grid_thw):
            calls.append((pixels, grid_thw))
            if grid_thw is image_grid:
                return image_features, None
            if grid_thw is video_grid:
                return video_features, None
            raise AssertionError("media grid lost its modality owner")

    class LanguageModelStub:
        _position_ids = None
        _rope_deltas = None
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: mx.zeros(
                (1, input_ids.shape[-1], 2), dtype=mx.float16
            )
        )

        def get_rope_index(
            self, input_ids, image_grid_thw, video_grid_thw, mask
        ):
            assert image_grid_thw is image_grid
            assert video_grid_thw is video_grid
            return mx.zeros((3, 1, 3), dtype=mx.int32), mx.zeros(
                (1, 1), dtype=mx.int32
            )

    wrapper = SimpleNamespace(
        config=SimpleNamespace(image_token_index=901, video_token_index=902),
        vision_tower=VisionTower(),
        language_model=LanguageModelStub(),
        _scatter_media_features=Model._scatter_media_features,
    )
    result = Model.get_input_embeddings(
        wrapper,
        input_ids=mx.array([[901, 17, 902]], dtype=mx.int32),
        pixel_values=image_pixels,
        pixel_values_videos=video_pixels,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        cached_image_features=mx.full((1, 2), 99.0, dtype=mx.float16),
    )
    mx.eval(result.inputs_embeds)

    assert len(calls) == 2
    np.testing.assert_array_equal(np.asarray(calls[0][0]), np.asarray(image_pixels))
    np.testing.assert_array_equal(np.asarray(calls[1][0]), np.asarray(video_pixels))
    np.testing.assert_array_equal(
        np.asarray(result.inputs_embeds),
        np.asarray([[[11.0, 12.0], [0.0, 0.0], [21.0, 22.0]]], dtype=np.float16),
    )


def test_qwen_video_float_255_input_preserves_uint8_processor_scale():
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )

    from vmlx_engine.mllm_batch_generator import (
        _normalize_qwen_video_arrays_for_processor,
    )

    processor = Qwen3VLVideoProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=32 * 32,
        max_pixels=32 * 32,
    )
    wrapper = type("QwenProcessor", (), {"video_processor": processor})()
    raw = np.zeros((4, 3, 32, 32), dtype=np.uint8)
    raw[:, 2, :, :] = 251
    fetched = raw.astype(np.float32)

    repaired = _normalize_qwen_video_arrays_for_processor(wrapper, [fetched])[0]
    assert repaired.dtype == np.uint8
    np.testing.assert_array_equal(repaired, raw)

    expected_patches, expected_grid = processor._process_one(raw)
    repaired_patches, repaired_grid = processor._process_one(repaired)
    np.testing.assert_array_equal(repaired_patches, expected_patches)
    assert repaired_grid == expected_grid
    assert float(repaired_patches.min()) == -1.0
    assert float(repaired_patches.max()) < 1.0

    unit_interval = fetched / 255.0
    untouched = _normalize_qwen_video_arrays_for_processor(
        wrapper, [unit_interval]
    )[0]
    assert untouched is unit_interval


@pytest.mark.parametrize("legacy_module", [None, object()])
def test_current_mlx_vlm_video_loader_replaces_removed_fetch_video(
    monkeypatch, legacy_module
):
    import mlx_vlm.utils as mlx_vlm_utils

    import vmlx_engine.mllm_batch_generator as batch_generator

    calls = []
    frames = np.zeros((4, 3, 8, 8), dtype=np.uint8)

    def import_module(name, package=None):
        assert name == "mlx_vlm.video_generate"
        if legacy_module is None:
            raise ModuleNotFoundError(name)
        return legacy_module

    def load_video(path, *, fps, max_frames):
        calls.append((path, fps, max_frames))
        return frames, 1.5

    monkeypatch.setattr(batch_generator.importlib, "import_module", import_module)
    monkeypatch.setattr(mlx_vlm_utils, "load_video", load_video)

    result, sample_fps = batch_generator._fetch_video_for_processor(
        "/tmp/clip.mp4",
        fps=2.0,
        max_frames=16,
    )

    assert result is frames
    assert sample_fps == 1.5
    assert calls == [("/tmp/clip.mp4", 2.0, 16)]


def test_qwen4_exp_reasoning_and_tools_preserve_native_bundle_contract():
    from vmlx_engine.model_config_registry import ModelConfigRegistry
    from vmlx_engine.model_configs import register_all

    registry = ModelConfigRegistry()
    register_all(registry)
    config = next(item for item in registry._configs if item.family_name == "qwen4_exp")
    assert config.reasoning_parser == "qwen3"
    assert config.tool_parser == "qwen"
    assert config.supports_thinking is True
    assert config.supports_native_tools is True
    assert config.supports_instruct_mode is True
    assert config.supported_reasoning_efforts == ["low", "medium", "xhigh"]
    assert config.preserve_native_tool_format is True
    assert config.architecture_hints["default_enable_thinking"] is True
    assert config.architecture_hints["modalities"] == ["text", "image", "video"]
    assert config.architecture_hints["audio_input"] is False


def test_qwen4_exp_exact_gate_up_projection_is_bit_identical_without_model_weights():
    from mlx_lm.models.switch_layers import SwitchGLU

    from vmlx_engine.metal.qwen4_affine_moe_decode import (
        _EXACT_PROJ_ATTR,
        _ExactGateUpProjection,
        _exact_gate_up_switchglu,
    )

    switch = SwitchGLU(64, 64, 4, bias=False)
    switch.gate_proj = switch.gate_proj.to_quantized(group_size=64, bits=4)
    switch.up_proj = switch.up_proj.to_quantized(group_size=64, bits=4)
    switch.down_proj = switch.down_proj.to_quantized(group_size=64, bits=4)
    x = (mx.arange(64, dtype=mx.float32) / 127.0).reshape(1, 1, 64)
    indices = mx.array([[[0, 1, 2, 3]]], dtype=mx.uint32)
    scores = mx.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=mx.float32)

    reference = (switch(x, indices) * scores[..., None]).sum(axis=-2)
    projection = _ExactGateUpProjection(switch.up_proj, switch.gate_proj)
    setattr(switch, _EXACT_PROJ_ATTR, projection)
    candidate = _exact_gate_up_switchglu(switch, x, indices, scores)
    mx.eval(reference, candidate)

    np.testing.assert_array_equal(np.asarray(candidate), np.asarray(reference))


def test_qwen4_exp_gdn_decode_projection_fusion_is_bit_identical():
    from vmlx_engine.models.qwen4_exp.language import (
        _decode_quantized_linears_fused,
    )

    shapes = (96, 64, 16, 16)
    dense = tuple(nn.Linear(64, output, bias=False) for output in shapes)
    for linear in dense:
        linear.weight = linear.weight.astype(mx.bfloat16)
    linears = tuple(
        linear.to_quantized(group_size=64, bits=4) for linear in dense
    )
    x = (mx.arange(64, dtype=mx.bfloat16) / 127.0).reshape(1, 1, 64)
    reference = tuple(linear(x) for linear in linears)
    candidate = _decode_quantized_linears_fused(linears, x)
    assert candidate is not None
    mx.eval(*reference, *candidate)
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_small_row_sigmoid_gated_rmsnorm_matches_qwen_reference(dtype, rows):
    from vmlx_engine.metal.gated_rmsnorm_decode import (
        gated_rmsnorm_decode_status,
        sigmoid_gated_rmsnorm_small_rows,
    )

    dims = 64
    x = (mx.arange(rows * 2 * dims, dtype=mx.float32) / 127.0 - 1.0)
    x = x.reshape(1, rows, 2, dims).astype(dtype)
    gate = (mx.flip(x, axis=-1) * 0.75).astype(dtype)
    weight = (mx.arange(dims, dtype=mx.float32) / 256.0 + 0.75).astype(dtype)
    reference = (
        mx.fast.rms_norm(x, weight, 1e-6).astype(mx.float32)
        * mx.sigmoid(gate.astype(mx.float32))
    ).astype(dtype)
    candidate = sigmoid_gated_rmsnorm_small_rows(
        x, gate, weight, 1e-6, output_dtype=dtype, enabled=True
    )
    assert candidate is not None
    assert gated_rmsnorm_decode_status()["observed_calls"] == 1
    mx.eval(reference, candidate)
    np.testing.assert_allclose(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(reference.astype(mx.float32)),
        rtol=8e-3,
        atol=8e-3,
    )


def test_small_row_sigmoid_gated_rmsnorm_refuses_prefill_width():
    from vmlx_engine.metal.gated_rmsnorm_decode import (
        sigmoid_gated_rmsnorm_small_rows,
    )

    x = mx.ones((1, 9, 2, 64), dtype=mx.float16)
    assert sigmoid_gated_rmsnorm_small_rows(
        x,
        x,
        mx.ones((64,), dtype=mx.float16),
        1e-6,
        enabled=True,
    ) is None


def test_qwen4_exp_gdn_projection_fusion_stays_decode_only():
    from vmlx_engine.models.qwen4_exp.language import (
        _decode_quantized_linears_fused,
    )

    linears = tuple(
        nn.Linear(64, 64, bias=False).to_quantized(group_size=64, bits=4)
        for _ in range(4)
    )
    assert _decode_quantized_linears_fused(linears, mx.zeros((1, 2, 64))) is None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_qwen4_exp_gdn_conv_fusion_matches_stock(dtype):
    from vmlx_engine.metal.gdn_conv_decode import (
        qwen4_gdn_conv_decode,
        qwen4_gdn_conv_status,
    )

    channels = 96
    kernel_size = 4
    conv = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
    )
    conv.weight = conv.weight.astype(dtype)
    state = (
        mx.arange((kernel_size - 1) * channels, dtype=mx.float32)
        .reshape(1, kernel_size - 1, channels)
        .astype(dtype)
        / 127.0
    )
    token = (
        mx.arange(channels, dtype=mx.float32).reshape(1, 1, channels)
        .astype(dtype)
        / 191.0
    )
    full = mx.concatenate([state, token], axis=1)
    reference_conv = nn.silu(conv(full))
    reference_state = mx.contiguous(full[:, -(kernel_size - 1) :, :])
    candidate = qwen4_gdn_conv_decode(
        token,
        state,
        conv.weight,
        enabled=True,
    )
    assert candidate is not None
    assert qwen4_gdn_conv_status()["observed_calls"] == 1
    candidate_conv, candidate_state = candidate
    mx.eval(reference_conv, reference_state, candidate_conv, candidate_state)

    np.testing.assert_array_equal(
        np.asarray(candidate_state.astype(mx.float32)),
        np.asarray(reference_state.astype(mx.float32)),
    )
    np.testing.assert_allclose(
        np.asarray(candidate_conv.astype(mx.float32)),
        np.asarray(reference_conv.astype(mx.float32)),
        rtol=8e-3,
        atol=8e-3,
    )


def test_qwen4_exp_gdn_conv_fusion_refuses_prefill():
    from vmlx_engine.metal.gdn_conv_decode import qwen4_gdn_conv_decode

    assert qwen4_gdn_conv_decode(
        mx.zeros((1, 2, 64), dtype=mx.float16),
        mx.zeros((1, 3, 64), dtype=mx.float16),
        mx.zeros((64, 4, 1), dtype=mx.float16),
        enabled=True,
    ) is None


def test_qwen4_exp_qsa_quantized_qkv_group_is_exact_and_releases_sources():
    from vmlx_engine.models.qwen4_exp.language import QSAAttention

    args = _tiny_args()
    attention = QSAAttention(args)
    attention.q_proj = attention.q_proj.to_quantized(group_size=64, bits=4)
    attention.k_proj = attention.k_proj.to_quantized(group_size=64, bits=4)
    attention.v_proj = attention.v_proj.to_quantized(group_size=64, bits=4)
    x = (mx.arange(args.hidden_size, dtype=mx.float32) / 127.0).reshape(
        1, 1, args.hidden_size
    )
    reference = (
        attention.q_proj(x),
        attention.k_proj(x),
        attention.v_proj(x),
    )

    assert attention.prepare_runtime() is True
    candidate = attention._project_qkv(x)
    mx.eval(*reference, *candidate)

    assert attention.q_proj is attention.k_proj is attention.v_proj is None
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_qwen4_exp_shared_expert_quantized_gate_up_group_is_exact():
    from vmlx_engine.models.qwen4_exp.language import SharedExpertMLP

    module = SharedExpertMLP(64, 96)
    module.gate_proj = module.gate_proj.to_quantized(group_size=64, bits=4)
    module.up_proj = module.up_proj.to_quantized(group_size=64, bits=4)
    x = (mx.arange(128, dtype=mx.float32) / 127.0).reshape(1, 2, 64)
    reference = (module.gate_proj(x), module.up_proj(x))

    assert module.prepare_runtime() is True
    candidate = module.gate_up_group(x)
    mx.eval(*reference, *candidate)

    assert module.gate_proj is module.up_proj is None
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_qwen4_exp_projection_group_preparation_walks_nested_modules():
    from vmlx_engine.models.qwen4_exp.language import (
        QSAAttention,
        SharedExpertMLP,
        prepare_quantized_projection_groups,
    )

    holder = nn.Module()
    holder.attention = QSAAttention(_tiny_args())
    holder.shared = SharedExpertMLP(64, 96)
    for name in ("q_proj", "k_proj", "v_proj"):
        projection = getattr(holder.attention, name)
        setattr(
            holder.attention,
            name,
            projection.to_quantized(group_size=64, bits=4),
        )
    for name in ("gate_proj", "up_proj"):
        projection = getattr(holder.shared, name)
        setattr(
            holder.shared,
            name,
            projection.to_quantized(group_size=64, bits=4),
        )

    assert prepare_quantized_projection_groups(holder) == {
        "qsa_qkv": 1,
        "shared_gate_up": 1,
    }
    assert holder.attention.qkv_group is not None
    assert holder.shared.gate_up_group is not None


def test_qwen4_exp_grouped_rms_norm_uses_loader_homogeneous_dtype():
    from vmlx_engine.models.qwen4_exp.language import (
        GroupedRMSNorm,
        _decode_quantized_linears_fused,
    )

    module = GroupedRMSNorm(128, 64)
    module.weight = (mx.arange(128, dtype=mx.float16) / 256.0) + 0.75
    residual = (mx.arange(128, dtype=mx.float16) / 127.0).reshape(1, 1, 128)

    actual = module(residual)
    expected = mx.fast.rms_norm(
        residual.reshape(1, 1, 2, 64), None, module.eps
    ).reshape(residual.shape) * module.weight
    mx.eval(actual, expected)

    assert actual.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    linears = tuple(
        nn.Linear(128, 128, bias=False).to_quantized(group_size=64, bits=4)
        for _ in range(4)
    )
    for linear in linears:
        linear.scales = linear.scales.astype(mx.float16)
        linear.biases = linear.biases.astype(mx.float16)
    fused = _decode_quantized_linears_fused(linears, actual)
    assert fused is not None
    mx.eval(*fused)
    assert all(output.dtype == mx.float16 for output in fused)


def test_qwen4_exp_file_backed_ple_preserves_bfloat16_residual_dtype():
    from vmlx_engine.models.qwen4_exp.language import ShardedNGramEmbedding
    from vmlx_engine.models.qwen4_exp.table_reader import _AffineShard

    dense = nn.Embedding(4, 64)
    dense.weight = dense.weight.astype(mx.bfloat16)
    quantized = dense.to_quantized(group_size=64, bits=4)

    class ArrayRows:
        def __init__(self, values, dtype_tag):
            self.values = values
            self.dtype_tag = dtype_tag
            self.shape = values.shape

        @property
        def mlx_dtype(self):
            return {
                "BF16": mx.bfloat16,
                "U32": mx.uint32,
            }[self.dtype_tag]

        def mlx_rows(self, rows):
            return self.values[mx.array(rows.astype(np.uint32))]

        def rows(self, rows):
            values = self.values[mx.array(rows.astype(np.uint32))]
            if self.dtype_tag == "BF16":
                values = values.astype(mx.float32)
            return np.asarray(values)

    shard = _AffineShard.__new__(_AffineShard)
    shard.weight = ArrayRows(quantized.weight, "U32")
    shard.scales = ArrayRows(quantized.scales, "BF16")
    shard.biases = ArrayRows(quantized.biases, "BF16")
    shard.group_size = 64
    shard.logical_bits = 4
    shard.storage_bits = 4
    shard.mode = "affine"
    shard.head_dim = 64
    gathered = shard.gather_mlx(np.array([0, 2], dtype=np.int64))
    profile = {}
    profiled = shard.gather_mlx(
        np.array([0, 2], dtype=np.int64),
        profile=profile,
    )
    assert gathered.dtype == mx.bfloat16

    class FileBackedTable:
        output_dtype = mx.bfloat16

        def gather_mlx(self, rows):
            return shard.gather_mlx(rows % 4)

    embedding = ShardedNGramEmbedding(3_000_000, 64, 128)
    embedding.set_file_backed(FileBackedTable())
    ple = embedding(np.array([[[0], [2]]], dtype=np.int64))
    residual = mx.zeros(ple.shape, dtype=mx.bfloat16) + ple
    mx.eval(gathered, profiled, ple, residual)

    assert ple.dtype == mx.bfloat16
    assert residual.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(profiled.astype(mx.float32)),
        np.asarray(gathered.astype(mx.float32)),
    )
    assert set(profile) == {
        "ssd_rows_cpu_ms",
        "host_to_mlx_ms",
        "dequant_gpu_ms",
    }
    assert all(value >= 0 for value in profile.values())

    embedding.set_file_backed(FileBackedTable(), output_dtype=mx.float16)
    overridden = embedding(np.array([[[0], [2]]], dtype=np.int64))
    mx.eval(overridden)
    assert overridden.dtype == mx.float16


def test_qwen4_exp_layer_profiler_modes(monkeypatch):
    from vmlx_engine.models.qwen4_exp.language import _layer_profile_enabled

    decode_ids = mx.zeros((1, 1), dtype=mx.int32)
    prefill_ids = mx.zeros((1, 8), dtype=mx.int32)

    monkeypatch.delenv("VMLINUX_QWEN4_PROFILE_LAYERS", raising=False)
    assert not _layer_profile_enabled(decode_ids)
    assert not _layer_profile_enabled(prefill_ids)

    monkeypatch.setenv("VMLINUX_QWEN4_PROFILE_LAYERS", "1")
    assert _layer_profile_enabled(decode_ids)
    assert not _layer_profile_enabled(prefill_ids)

    monkeypatch.setenv("VMLINUX_QWEN4_PROFILE_LAYERS", "prefill")
    assert not _layer_profile_enabled(decode_ids)
    assert _layer_profile_enabled(prefill_ids)

    monkeypatch.setenv("VMLINUX_QWEN4_PROFILE_LAYERS", "all")
    assert _layer_profile_enabled(decode_ids)
    assert _layer_profile_enabled(prefill_ids)


def test_qwen4_exp_profiled_file_backed_gather_preserves_row_order():
    from vmlx_engine.models.qwen4_exp.table_reader import (
        FileBackedQuantizedNGramTable,
    )

    class FakeShard:
        def __init__(self, offset):
            self.offset = offset

        def gather_mlx(self, rows, profile=None):
            if profile is not None:
                profile["fake_shard_calls"] = profile.get("fake_shard_calls", 0) + 1
            return mx.array(rows[:, None] + self.offset, dtype=mx.float32)

    table = FileBackedQuantizedNGramTable.__new__(FileBackedQuantizedNGramTable)
    table.shards = [FakeShard(0), FakeShard(100)]
    table.per = 4
    table.head_dim = 1
    table.output_dtype = mx.float32
    table.total_rows = 8
    rows = np.array([5, 1, 6, 0], dtype=np.int64)

    baseline = table.gather_mlx(rows)
    profile = {}
    profiled = table.gather_mlx(rows, profile=profile)
    mx.eval(baseline, profiled)

    np.testing.assert_array_equal(np.asarray(profiled), np.asarray(baseline))
    np.testing.assert_array_equal(
        np.asarray(profiled).reshape(-1),
        np.array([101.0, 1.0, 102.0, 0.0], dtype=np.float32),
    )
    assert profile["fake_shard_calls"] == 2
    assert profile["scatter_gpu_ms"] >= 0


def test_qwen4_exp_pread_rows_match_memmap_rows(tmp_path):
    import json

    from vmlx_engine.models.qwen4_exp.table_reader import (
        SafetensorsRowReader,
        _SharedPreadFile,
    )

    values = np.arange(24, dtype=np.uint32).reshape(4, 6)
    tensor_name = "table.weight"
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "U32",
                "shape": list(values.shape),
                "data_offsets": [0, values.nbytes],
            }
        },
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "rows.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + values.tobytes())
    source = _SharedPreadFile(path)
    try:
        reader = SafetensorsRowReader(path, tensor_name, pread_file=source)
        rows = np.array([3, 0, 3, 1], dtype=np.int64)
        np.testing.assert_array_equal(reader.rows_pread(rows), reader.rows(rows))
    finally:
        source.close()


def test_qwen4_exp_parallel_pread_gather_preserves_order_and_profiles():
    from concurrent.futures import ThreadPoolExecutor

    from vmlx_engine.models.qwen4_exp.table_reader import (
        FileBackedQuantizedNGramTable,
    )

    class FakeShard:
        def __init__(self, offset):
            self.offset = offset

        def read_rows(self, rows, *, use_pread=False):
            assert use_pread is True
            return (rows.copy(), rows.copy(), rows.copy())

        def dequantize_rows_mlx(self, host_rows, *, profile=None):
            if profile is not None:
                profile["host_to_mlx_ms"] = profile.get("host_to_mlx_ms", 0.0)
                profile["dequant_gpu_ms"] = profile.get("dequant_gpu_ms", 0.0)
            return mx.array(host_rows[0][:, None] + self.offset, dtype=mx.float32)

    table = FileBackedQuantizedNGramTable.__new__(FileBackedQuantizedNGramTable)
    table.shards = [FakeShard(0), FakeShard(100)]
    table.per = 4
    table.head_dim = 1
    table.output_dtype = mx.float32
    table.total_rows = 8
    table._pread_files = {}
    table._parallel_read = True
    table._read_pool = ThreadPoolExecutor(max_workers=2)
    try:
        profile = {}
        actual = table.gather_mlx(
            np.array([5, 1, 6, 0], dtype=np.int64),
            profile=profile,
        )
        mx.eval(actual)
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1),
            np.array([101.0, 1.0, 102.0, 0.0], dtype=np.float32),
        )
        assert profile["ssd_rows_cpu_ms"] >= 0
        assert profile["ssd_rows_parallel_wall_ms"] >= 0
        assert profile["scatter_gpu_ms"] >= 0
    finally:
        table.close()


def test_qwen4_exp_ple_random_access_advice_is_fail_safe():
    from vmlx_engine.models.qwen4_exp.table_reader import _advise_random_access

    class Mapping:
        def __init__(self, error=None):
            self.calls = []
            self.error = error

        def madvise(self, advice):
            self.calls.append(advice)
            if self.error is not None:
                raise self.error

    class MemMap:
        def __init__(self, mapping):
            self._mmap = mapping

    mapping = Mapping()
    assert _advise_random_access(MemMap(mapping)) is True
    assert len(mapping.calls) == 1

    unsupported = Mapping(OSError("unsupported"))
    assert _advise_random_access(MemMap(unsupported)) is False
    assert len(unsupported.calls) == 1
    assert _advise_random_access(object()) is False


@pytest.mark.parametrize("runtime_dtype", [mx.float16, mx.bfloat16])
def test_qwen4_exp_ple_short_conv_uses_prepared_runtime_dtype(runtime_dtype):
    from vmlx_engine.models.qwen4_exp.language import (
        PLELayer,
        _decode_quantized_linears_fused,
    )

    ple = PLELayer(_tiny_args(), 0)
    ple.conv1d_weight = ple.conv1d_weight.astype(runtime_dtype)
    ple.prepare_runtime()
    assert len(ple._conv_taps) == ple.conv_kernel_size
    assert all(tap.dtype == runtime_dtype for tap in ple._conv_taps)
    residual = mx.ones((1, 1, 256), dtype=runtime_dtype)
    convolved = ple._short_conv(residual, cache=None)
    mx.eval(convolved)

    assert convolved.dtype == runtime_dtype

    linears = tuple(
        nn.Linear(256, 256, bias=False).to_quantized(group_size=64, bits=4)
        for _ in range(4)
    )
    for linear in linears:
        linear.scales = linear.scales.astype(runtime_dtype)
        linear.biases = linear.biases.astype(runtime_dtype)
    fused = _decode_quantized_linears_fused(linears, convolved)
    assert fused is not None
    mx.eval(*fused)
    assert all(output.dtype == runtime_dtype for output in fused)


@pytest.mark.parametrize("runtime_dtype", [mx.float16, mx.bfloat16])
def test_qwen4_exp_ple_fused_decode_conv_matches_stock(runtime_dtype):
    from vmlx_engine.metal.ple_conv_decode import (
        qwen4_ple_conv_decode,
        qwen4_ple_conv_status,
    )
    from vmlx_engine.models.qwen4_exp.language import PLELayer

    ple = PLELayer(_tiny_args(), 0)
    mx.random.seed(17)
    ple.conv1d_weight = (
        mx.random.normal(ple.conv1d_weight.shape) * 0.1
    ).astype(runtime_dtype)
    state = (
        mx.random.normal((1, ple.short_conv_state_len, 256)) * 0.1
    ).astype(runtime_dtype)
    token = (mx.random.normal((1, 1, 256)) * 0.1).astype(runtime_dtype)

    full = mx.concatenate([state, token], axis=1)
    taps = [
        full[:, index * ple.conv_dilation : index * ple.conv_dilation + 1]
        * ple.conv1d_weight[:, index]
        for index in range(ple.conv_kernel_size)
    ]
    reference_output = nn.silu(sum(taps))
    reference_state = mx.contiguous(full[:, -ple.short_conv_state_len :, :])
    candidate = qwen4_ple_conv_decode(
        token,
        state,
        ple.conv1d_weight,
        dilation=ple.conv_dilation,
        enabled=True,
    )
    assert candidate is not None
    assert qwen4_ple_conv_status()["observed_calls"] == 1
    mx.eval(reference_output, reference_state, *candidate)
    max_delta = float(
        mx.max(
            mx.abs(
                candidate[0].astype(mx.float32)
                - reference_output.astype(mx.float32)
            )
        ).item()
    )
    # The fused FP32 accumulation may round once at the final storage step,
    # while MLX's elementwise tap graph rounds intermediates in runtime dtype.
    # Production-width probes bound that difference to one output-dtype step.
    limit = 6.2e-5 if runtime_dtype == mx.float16 else 4.9e-4
    assert max_delta <= limit
    assert mx.array_equal(candidate[1], reference_state)


def test_qwen4_exp_ple_fused_decode_conv_refuses_prefill():
    from vmlx_engine.metal.ple_conv_decode import qwen4_ple_conv_decode

    token = mx.zeros((1, 2, 256), dtype=mx.bfloat16)
    state = mx.zeros((1, 9, 256), dtype=mx.bfloat16)
    weight = mx.zeros((256, 4), dtype=mx.bfloat16)
    assert qwen4_ple_conv_decode(
        token, state, weight, dilation=3, enabled=True
    ) is None


def test_qwen4_exp_zero_centered_norm_offset_is_folded_once():
    from vmlx_engine.models.qwen4_exp.language import (
        ZeroCenteredRMSNorm,
        fold_zero_centered_norm_offsets,
    )

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = ZeroCenteredRMSNorm(8)

    holder = Holder()
    holder.norm.weight = mx.arange(8, dtype=mx.bfloat16) / 32.0
    x = mx.arange(8, dtype=mx.bfloat16).reshape(1, 1, 8) / 7.0
    before = holder.norm(x)
    assert fold_zero_centered_norm_offsets(holder) == 1
    after = holder.norm(x)
    assert fold_zero_centered_norm_offsets(holder) == 0
    mx.eval(before, after)
    np.testing.assert_allclose(
        np.asarray(before.astype(mx.float32)),
        np.asarray(after.astype(mx.float32)),
        rtol=0,
        atol=0,
    )


def test_qwen4_exp_hyper_projection_fusion_is_bit_identical():
    from vmlx_engine.models.qwen4_exp.language import (
        GatedResidual,
        fuse_hyper_connection_projections,
    )

    module = GatedResidual(_tiny_args())
    _randomize(module)
    module.input_mix_weight_down = module.input_mix_weight_down.to_quantized(
        group_size=64, bits=4
    )
    module.block_inject_weight = module.block_inject_weight.to_quantized(
        group_size=64, bits=4
    )
    x = (mx.arange(256, dtype=mx.bfloat16) / 127.0).reshape(1, 1, 256)
    reference = module(x)
    mx.eval(*reference)

    assert fuse_hyper_connection_projections(module) == 1
    candidate = module(x)
    mx.eval(*candidate)
    for expected, actual in zip(reference, candidate):
        np.testing.assert_array_equal(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )


def test_qwen4_exp_hyper_connection_preserves_mixed_jang_residual_dtype():
    from mlx.utils import tree_map

    from vmlx_engine.models.qwen4_exp.language import GatedResidual

    module = GatedResidual(_tiny_args())
    module.update(
        tree_map(
            lambda parameter: parameter.astype(mx.bfloat16),
            module.parameters(),
        )
    )
    module.hc_norm.weight = module.hc_norm.weight.astype(mx.float32)
    residual = (mx.arange(256, dtype=mx.float16) / 127.0).reshape(1, 1, 256)

    mixed, hyper, inject = module(residual)
    combined = module.combine(hyper, mixed, inject)
    mx.eval(mixed, hyper, inject, combined)

    assert mixed.dtype == mx.float16
    assert hyper.dtype == mx.float16
    assert inject.dtype == mx.float16
    assert combined.dtype == mx.float16


def test_qwen4_exp_jang_hyper_fold_dtype_leak_is_normalized_before_load():
    from vmlx_engine.models.qwen4_exp.loader import (
        _normalize_jang_hyper_fold_dtypes,
    )

    base = "language_model.model.layers.0.attn_hyper_connection"
    intentional = "language_model.model.layers.1.attn_hyper_connection"
    packed = "language_model.model.layers.2.attn_hyper_connection"
    weights = {
        f"{base}.hc_norm.weight": mx.arange(256, dtype=mx.float32) / 257.0,
        f"{base}.input_mix_weight_down.weight": mx.arange(
            32 * 256, dtype=mx.float32
        ).reshape(32, 256) / 8193.0,
        f"{base}.input_mix_weight_up.weight": mx.ones(
            (256, 32), dtype=mx.bfloat16
        ),
        f"{base}.block_inject_weight.weight": mx.arange(
            4 * 256, dtype=mx.float32
        ).reshape(4, 256) / 1025.0,
        f"{intentional}.hc_norm.weight": mx.ones((256,), dtype=mx.float32),
        f"{intentional}.input_mix_weight_down.weight": mx.ones(
            (32, 256), dtype=mx.float32
        ),
        f"{intentional}.input_mix_weight_up.weight": mx.ones(
            (256, 32), dtype=mx.float32
        ),
        f"{packed}.hc_norm.weight": mx.ones((256,), dtype=mx.float32),
        f"{packed}.input_mix_weight_down.weight": mx.ones(
            (32, 4), dtype=mx.uint32
        ),
        f"{packed}.input_mix_weight_up.weight": mx.ones(
            (256, 32), dtype=mx.float16
        ),
    }

    expected = {
        name: value.astype(mx.float16)
        for name, value in weights.items()
        if name.startswith(base) and value.dtype == mx.float32
    }
    matrices, norms = _normalize_jang_hyper_fold_dtypes(
        weights,
        target_dtype=mx.float16,
    )
    mx.eval(*weights.values(), *expected.values())

    assert (matrices, norms) == (2, 1)
    assert weights[f"{base}.hc_norm.weight"].dtype == mx.float16
    assert weights[f"{base}.input_mix_weight_down.weight"].dtype == mx.float16
    assert weights[f"{base}.input_mix_weight_up.weight"].dtype == mx.bfloat16
    assert weights[f"{base}.block_inject_weight.weight"].dtype == mx.float16
    for name, value in expected.items():
        np.testing.assert_array_equal(
            np.asarray(weights[name].astype(mx.float32)),
            np.asarray(value.astype(mx.float32)),
        )
    assert weights[f"{intentional}.hc_norm.weight"].dtype == mx.float32
    assert weights[f"{intentional}.input_mix_weight_down.weight"].dtype == mx.float32
    assert weights[f"{packed}.input_mix_weight_down.weight"].dtype == mx.uint32


def test_qwen4_exp_jang_runtime_compute_dtype_follows_packed_metadata():
    from vmlx_engine.models.qwen4_exp.loader import (
        _normalize_jang_runtime_compute_dtypes,
        _resolve_jang_runtime_compute_dtype,
    )

    class RuntimeModel:
        @staticmethod
        def cast_predicate(path):
            return not path.endswith(("A_log", "dt_bias"))

    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros(
            (16, 2), dtype=mx.uint32
        ),
        "language_model.model.layers.0.self_attn.q_proj.scales": mx.ones(
            (16, 1), dtype=mx.float16
        ),
        "language_model.model.layers.0.self_attn.q_proj.biases": mx.zeros(
            (16, 1), dtype=mx.float16
        ),
        "language_model.model.embed_tokens.weight": mx.ones(
            (16, 16), dtype=mx.bfloat16
        ),
        "language_model.model.layers.0.mlp.gate.weight": mx.ones(
            (16, 16), dtype=mx.float32
        ),
        "language_model.model.layers.0.mlp.shared_expert_gate.weight": mx.ones(
            (1, 16), dtype=mx.float32
        ),
        "language_model.model.layers.0.linear_attn.A_log": mx.ones(
            (16,), dtype=mx.float32
        ),
    }

    dtype = _resolve_jang_runtime_compute_dtype(weights)
    summary = _normalize_jang_runtime_compute_dtypes(
        weights,
        RuntimeModel(),
        dtype,
    )
    mx.eval(*weights.values())

    assert dtype == mx.float16
    assert summary == {
        "f16": 2,
        "bf16": 1,
        "f32": 3,
        "cast": 3,
        "preserved": 1,
    }
    assert weights["language_model.model.embed_tokens.weight"].dtype == mx.float16
    assert weights["language_model.model.layers.0.mlp.gate.weight"].dtype == mx.float16
    assert (
        weights["language_model.model.layers.0.mlp.shared_expert_gate.weight"].dtype
        == mx.float16
    )
    assert (
        weights["language_model.model.layers.0.linear_attn.A_log"].dtype
        == mx.float32
    )


def test_qwen4_exp_jang_runtime_compute_dtype_can_be_bfloat16():
    from vmlx_engine.models.qwen4_exp.loader import (
        _normalize_jang_runtime_compute_dtypes,
        _resolve_jang_runtime_compute_dtype,
    )

    class RuntimeModel:
        @staticmethod
        def cast_predicate(path):
            return not path.endswith(("A_log", "dt_bias"))

    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros(
            (16, 2), dtype=mx.uint32
        ),
        "language_model.model.layers.0.self_attn.q_proj.scales": mx.ones(
            (16, 1), dtype=mx.bfloat16
        ),
        "language_model.model.layers.0.self_attn.q_proj.biases": mx.zeros(
            (16, 1), dtype=mx.bfloat16
        ),
        "language_model.model.embed_tokens.weight": mx.ones(
            (16, 16), dtype=mx.float16
        ),
        "language_model.model.layers.0.mlp.gate.weight": mx.ones(
            (16, 16), dtype=mx.float32
        ),
        "language_model.model.layers.0.mlp.shared_expert_gate.weight": mx.ones(
            (1, 16), dtype=mx.float32
        ),
        "language_model.model.layers.0.linear_attn.dt_bias": mx.ones(
            (16,), dtype=mx.float32
        ),
    }

    dtype = _resolve_jang_runtime_compute_dtype(weights)
    summary = _normalize_jang_runtime_compute_dtypes(
        weights,
        RuntimeModel(),
        dtype,
    )
    mx.eval(*weights.values())

    assert dtype == mx.bfloat16
    assert summary == {
        "f16": 1,
        "bf16": 2,
        "f32": 3,
        "cast": 3,
        "preserved": 1,
    }
    assert weights["language_model.model.embed_tokens.weight"].dtype == mx.bfloat16
    assert weights["language_model.model.layers.0.mlp.gate.weight"].dtype == mx.bfloat16
    assert (
        weights["language_model.model.layers.0.mlp.shared_expert_gate.weight"].dtype
        == mx.bfloat16
    )
    assert (
        weights["language_model.model.layers.0.linear_attn.dt_bias"].dtype
        == mx.float32
    )


def test_qwen4_exp_jang_mixed_packed_metadata_dtypes_fail_closed():
    from vmlx_engine.models.qwen4_exp.loader import (
        _resolve_jang_runtime_compute_dtype,
    )

    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros(
            (16, 2), dtype=mx.uint32
        ),
        "language_model.model.layers.0.self_attn.q_proj.scales": mx.ones(
            (16, 1), dtype=mx.float16
        ),
        "language_model.model.layers.0.self_attn.q_proj.biases": mx.zeros(
            (16, 1), dtype=mx.bfloat16
        ),
    }

    with pytest.raises(ValueError, match="mixed compute dtypes"):
        _resolve_jang_runtime_compute_dtype(weights)


def test_qwen4_exp_jang_runtime_compute_dtype_rejects_float32_target():
    from vmlx_engine.models.qwen4_exp.loader import (
        _normalize_jang_runtime_compute_dtypes,
    )

    with pytest.raises(ValueError, match="must be float16 or bfloat16"):
        _normalize_jang_runtime_compute_dtypes({}, object(), mx.float32)


def test_qwen4_exp_runtime_dtype_audit_ignores_preserved_recurrent_state(caplog):
    from vmlx_engine.models.qwen4_exp.loader import (
        _warn_runtime_dtype_mismatches,
    )

    class RuntimeModel:
        @staticmethod
        def cast_predicate(path):
            return not path.endswith(("A_log", "dt_bias"))

        @staticmethod
        def parameters():
            return {
                "projection": mx.ones((8, 8), dtype=mx.bfloat16),
                "A_log": mx.ones((8,), dtype=mx.bfloat16),
                "dt_bias": mx.ones((8,), dtype=mx.bfloat16),
            }

    caplog.set_level("WARNING", logger="vmlx_engine")
    assert _warn_runtime_dtype_mismatches(RuntimeModel(), mx.float16) == 1
    assert "projection=mlx.core.bfloat16" in caplog.text
    assert "A_log" not in caplog.text
    assert "dt_bias" not in caplog.text


def test_qwen4_exp_hyper_compile_is_decode_only_and_numerically_equivalent():
    from vmlx_engine.models.qwen4_exp.language import (
        GatedResidual,
        compile_hyper_connections,
        fuse_hyper_connection_projections,
    )

    module = GatedResidual(_tiny_args())
    _randomize(module)
    assert fuse_hyper_connection_projections(module) == 1
    decode = mx.arange(256, dtype=mx.float32).reshape(1, 1, 256) / 127.0
    eager = module(decode)
    mx.eval(*eager)
    assert compile_hyper_connections(module) == 1
    compiled = module(decode)
    mx.eval(*compiled)
    for expected, actual in zip(eager, compiled):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
        )

    def forbidden_decode(_inputs):
        raise AssertionError("prefill must not use the single-token compiled path")

    module._compiled_forward = forbidden_decode
    prefill = module(mx.zeros((1, 2, 256), dtype=mx.float32))
    mx.eval(*prefill)


def test_qwen4_exp_ple_manifest_aliases_are_deterministic_and_fail_closed():
    from vmlx_engine.models.qwen4_exp.table_reader import (
        _module_aliases,
        _unique_mapping_override,
    )

    official = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
    aliases = _module_aliases(official)
    runtime = official.replace("model.language_model", "language_model.model", 1)
    short = official.replace(".ple.ple_embedding.", ".ple.")
    assert aliases[0] == official
    assert len(aliases) == len(set(aliases))
    same = {official: {"bits": 4}, runtime: {"bits": 4}}
    assert _unique_mapping_override(same, aliases, label="test") == {"bits": 4}
    with pytest.raises(ValueError, match="conflicting test aliases"):
        _unique_mapping_override(
            {official: {"bits": 4}, short: {"bits": 2}},
            aliases,
            label="test",
        )

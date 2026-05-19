import json
from types import SimpleNamespace


class NativeGatedDeltaState:
    """Sentinel for Qwen hybrid non-KV companion state."""


class FakeHybridModel:
    def __init__(self, n_layers=4):
        self.layers = [object()] * n_layers

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [
            KVCache(),
            NativeGatedDeltaState(),
            KVCache(),
            NativeGatedDeltaState(),
        ]


def _write_qwen36_hybrid_config(path):
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 4,
            "head_dim": 128,
            "layer_types": [
                "full_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    return config


def test_standard_qwen_hybrid_tq_patches_attention_cache_only(tmp_path, monkeypatch):
    from vmlx_engine.utils.tokenizer import _apply_turboquant_to_model

    _write_qwen36_hybrid_config(tmp_path)
    model = FakeHybridModel()
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    _apply_turboquant_to_model(model, str(tmp_path))

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_policy") == "attention_kv_only"
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (0, 2)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (1, 3)


def test_jang_qwen_hybrid_tq_patches_attention_cache_only(monkeypatch):
    from vmlx_engine.utils.jang_loader import _patch_turboquant_make_cache

    model = FakeHybridModel()
    model_config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 4,
            "head_dim": 128,
            "layer_types": [
                "full_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        },
    }
    monkeypatch.delenv("VMLX_DISABLE_TQ_KV", raising=False)
    monkeypatch.delenv("VMLX_ALLOW_HYBRID_KV_QUANT", raising=False)

    _patch_turboquant_make_cache(
        model,
        {
            "turboquant": {
                "enabled": True,
                "default_key_bits": 3,
                "default_value_bits": 3,
                "critical_key_bits": 4,
                "critical_value_bits": 4,
                "critical_layers": [0, -1],
                "seed": 42,
            }
        },
        model_config,
    )

    cache = model.make_cache()
    assert [type(slot).__name__ for slot in cache] == [
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
        "TurboQuantKVCache",
        "NativeGatedDeltaState",
    ]
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_policy") == "attention_kv_only"
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_attention_layers") == (0, 2)
    assert getattr(model.make_cache, "_vmlx_hybrid_tq_companion_layers") == (1, 3)


def test_native_cache_status_reports_hybrid_live_attention_tq():
    from vmlx_engine.server import _native_cache_status

    scheduler = SimpleNamespace(
        config=SimpleNamespace(kv_cache_quantization="q4"),
        _model_type_for_runtime="qwen3_5_moe",
        _is_hybrid=True,
        _uses_dsv4_cache=False,
        _uses_zaya_cache=False,
        _tq_active=True,
        _hybrid_live_tq_policy="attention_kv_only",
        _hybrid_live_tq_attention_layers=[0, 2],
        _hybrid_live_tq_companion_layers=[1, 3],
        _hybrid_kv_positions=[0, 2],
        _kv_cache_bits=4,
        _kv_cache_group_size=64,
        _ssm_state_cache=SimpleNamespace(_store={"prompt": object()}),
        block_aware_cache=object(),
        paged_cache_manager=SimpleNamespace(_disk_store=object()),
    )

    status = _native_cache_status(scheduler)

    assert status["generic_turboquant_kv"] == {
        "enabled": True,
        "reason": "hybrid_attention_kv_only",
    }
    assert status["live_attention_tq_kv"] == {
        "enabled": True,
        "mode": "live_decode",
        "applies_to": "attention_kv_layers_only",
        "ssm_policy": "native_full_precision_companion_state",
        "attention_layers": [0, 2],
        "companion_layers": [1, 3],
    }
    assert status["attention_kv_storage_quantization"]["mode"] == "storage_boundary"

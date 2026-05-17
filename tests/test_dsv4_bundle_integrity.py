import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


def _write_bundle(tmp_path, *, model_type="deepseek_v4", dtype=np.float32):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    save_file(
        {
            "hc_head_fn": np.zeros((1,), dtype=dtype),
            "layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    return tmp_path


def _write_dsv4_tq_artifact(tmp_path, *, layer_bits=None, sidecar=True):
    layer_bits = layer_bits or {0: 4, 1: 4, 2: 2}
    tmp_path.mkdir(parents=True, exist_ok=True)
    bit_plan = {
        "default_bits": 2,
        "codec": "mxtq",
        "routed_layer_bits": {
            str(layer): bits for layer, bits in layer_bits.items() if bits != 2
        },
        "plan_bit_counts": {"4": sum(1 for b in layer_bits.values() if b == 4)},
    }
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "mxtq_seed": 42,
        "weight_format": "mxtq",
        "mxtq_bits": {
            "routed_expert": 2,
            "attention": 8,
            "shared_expert": 8,
            "compressor": 8,
            "indexer": 8,
            "embed_tokens": 8,
            "lm_head": 8,
        },
        "quantization": {
            "routed_expert_bit_plan": bit_plan,
            "mxtq_bits": {
                "routed_expert": 2,
                "attention": 8,
                "shared_expert": 8,
                "compressor": 8,
                "indexer": 8,
                "embed_tokens": 8,
                "lm_head": 8,
            },
        },
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxtq",
        "mxtq_seed": 42,
        "mxtq_bits": {
            "routed_expert": 2,
            "attention": 8,
            "shared_expert": 8,
            "compressor": 8,
            "indexer": 8,
            "embed_tokens": 8,
            "lm_head": 8,
        },
        "quantization": {
            "routed_experts": {
                "bits": 2,
                "codec": "mxtq",
                "bit_plan": bit_plan,
            },
            "non_routed": {"bits": 8, "codec": "affine", "group_size": 32},
        },
        "critical_f32_preserved": True,
    }))

    tensors = {
        "hc_head_fn": np.zeros((1,), dtype=np.float32),
        "layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float32),
    }
    for layer, bits in layer_bits.items():
        for proj, out_dim, in_features in [
            ("gate_proj", 2048, 4096),
            ("up_proj", 2048, 4096),
            ("down_proj", 4096, 2048),
        ]:
            packed_cols = in_features // (32 // bits)
            stem = f"layers.{layer}.mlp.switch_mlp.{proj}"
            tensors[f"{stem}.tq_bits"] = np.array([bits], dtype=np.uint8)
            tensors[f"{stem}.tq_packed"] = np.zeros(
                (256, out_dim, packed_cols), dtype=np.uint32
            )
            tensors[f"{stem}.tq_norms"] = np.zeros((256, out_dim), dtype=np.float16)

    save_file(tensors, str(tmp_path / "model.safetensors"))

    if sidecar:
        save_file(
            {
                "codebook.2048.2": np.zeros((1,), dtype=np.float32),
                "codebook.2048.4": np.zeros((1,), dtype=np.float32),
                "codebook.4096.2": np.zeros((1,), dtype=np.float32),
                "codebook.4096.4": np.zeros((1,), dtype=np.float32),
                "signs.2048.42": np.zeros((1,), dtype=np.uint8),
                "signs.4096.42": np.zeros((1,), dtype=np.uint8),
            },
            str(tmp_path / "jangtq_runtime.safetensors"),
        )
    return tmp_path


def test_dsv4_control_tensor_validator_names_retracted_legacy_slug(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path / "DeepSeek-V4-Flash-JANGTQ", dtype=np.float16)

    with pytest.raises(RuntimeError) as exc:
        _validate_dsv4_control_tensors(bundle)

    msg = str(exc.value)
    assert "retracted" in msg
    assert "DeepSeek-V4-Flash-JANGTQ" in msg
    assert "V3-F32" in msg
    assert "hc_head_fn=F16" in msg


def test_dsv4_control_tensor_validator_does_not_slug_reject_corrected_name(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(
        tmp_path / "DeepSeek-V4-Flash-JANGTQ-V3-F32-MIXED",
        dtype=np.float32,
    )

    _validate_dsv4_control_tensors(bundle)


def test_dsv4_control_tensor_validator_suppresses_legacy_hint_for_v3_name(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(
        tmp_path / "DeepSeek-V4-Flash-JANGTQ-V3-F32-MIXED",
        dtype=np.float16,
    )

    with pytest.raises(RuntimeError) as exc:
        _validate_dsv4_control_tensors(bundle)

    msg = str(exc.value)
    assert "critical control tensors are not F32" in msg
    assert "hc_head_fn=F16" in msg
    assert "retracted" not in msg


def test_dsv4_control_tensor_validator_rejects_f16(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, dtype=np.float16)

    with pytest.raises(RuntimeError) as exc:
        _validate_dsv4_control_tensors(bundle)

    msg = str(exc.value)
    assert "DSV4 bundle is known-bad" in msg
    assert "critical control tensors are not F32" in msg
    assert "hc_head_fn=F16" in msg


def test_dsv4_control_tensor_validator_accepts_f32(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_control_tensor_dtypes,
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, dtype=np.float32)

    _validate_dsv4_control_tensors(bundle)
    report = _audit_dsv4_control_tensor_dtypes(bundle)
    assert report["checked"] is True
    assert report["critical_count"] == 2
    assert report["non_f32_count"] == 0


def test_dsv4_control_tensor_validator_skips_non_dsv4(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_control_tensor_dtypes,
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, model_type="qwen3", dtype=np.float16)

    _validate_dsv4_control_tensors(bundle)
    report = _audit_dsv4_control_tensor_dtypes(bundle)
    assert report["checked"] is False


def test_dsv4_hydration_mode_separates_affine_from_jangtq(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _dsv4_uses_jangtq_hydration,
    )

    affine = tmp_path / "affine"
    affine.mkdir()
    (affine / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "weight_format": "affine"})
    )
    (affine / "jang_config.json").write_text(
        json.dumps({"weight_format": "affine"})
    )

    tq = _write_dsv4_tq_artifact(tmp_path / "tq")

    assert _dsv4_uses_jangtq_hydration(affine) is False
    assert _dsv4_uses_jangtq_hydration(tq) is True


def test_dsv4_affine_loader_uses_generic_jang_hydration(monkeypatch, tmp_path):
    import vmlx_engine.loaders.load_jangtq_dsv4 as loader
    import vmlx_engine.utils.jang_loader as jang_loader

    bundle = tmp_path / "DeepSeek-V4-Flash-JANG_2L-MTP"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "weight_format": "affine"})
    )
    (bundle / "jang_config.json").write_text(
        json.dumps({"weight_format": "affine"})
    )
    expected = (object(), SimpleNamespace(apply_chat_template=lambda *a, **k: []))
    calls = []

    monkeypatch.setattr(loader, "_validate_dsv4_control_tensors", lambda path: None)
    monkeypatch.setattr(loader, "_verify_dsv4_attention_contract", lambda: None)
    monkeypatch.setattr(loader, "_install_dsv4_memory_defaults", lambda: None)
    monkeypatch.setattr(
        loader, "_configure_dsv4_pool_quant_default", lambda: "0"
    )
    monkeypatch.setattr(
        jang_loader,
        "load_jang_model",
        lambda path, **kwargs: calls.append((path, kwargs)) or expected,
    )

    loaded = loader.load_jangtq_dsv4_model(str(bundle), skip_params_eval=True)

    assert loaded[0] is expected[0]
    assert loaded[1] is expected[1]
    assert calls == [(str(bundle), {"skip_eval": True})]


def test_dsv4_artifact_bit_plan_audit_reads_actual_tq_bits_and_sidecar(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_artifact_bit_plan,
    )

    bundle = _write_dsv4_tq_artifact(tmp_path / "dsv4")

    report = _audit_dsv4_artifact_bit_plan(bundle)

    assert report["checked"] is True
    assert report["routed_layer_bits"] == {"0": 4, "1": 4, "2": 2}
    assert report["routed_bit_counts"] == {"2": 1, "4": 2}
    assert report["metadata_routed_layer_bits"] == {"0": 4, "1": 4}
    assert report["metadata_matches_actual"] is True
    assert report["sidecar"]["present"] is True
    assert report["sidecar"]["missing_keys"] == []
    assert report["issues"] == []


def test_dsv4_artifact_bit_plan_accepts_explicit_default_layer_metadata(tmp_path):
    """Bit-plan metadata may list default-bit layers explicitly.

    Current DSV4 upload-prep bundles record the full planned set, including
    some layers pinned at the same value as the role default. The validator
    should compare metadata layer values against actual headers, not treat
    explicit default entries as drift.
    """
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_artifact_bit_plan,
    )

    bundle = _write_dsv4_tq_artifact(
        tmp_path / "dsv4",
        layer_bits={0: 2, 1: 2, 2: 2, 23: 4, 25: 4},
    )
    jang = json.loads((bundle / "jang_config.json").read_text())
    jang["quantization"]["routed_experts"]["bit_plan"]["routed_layer_bits"] = {
        "0": 2,
        "1": 2,
        "2": 2,
        "23": 4,
        "25": 4,
    }
    (bundle / "jang_config.json").write_text(json.dumps(jang))

    report = _audit_dsv4_artifact_bit_plan(bundle)

    assert report["routed_layer_bits"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "23": 4,
        "25": 4,
    }
    assert report["metadata_routed_layer_bits"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "23": 4,
        "25": 4,
    }
    assert report["metadata_matches_actual"] is True
    assert report["issues"] == []


def test_dsv4_artifact_bit_plan_audit_flags_missing_sidecar_key(tmp_path):
    from safetensors.numpy import save_file

    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_artifact_bit_plan,
    )

    bundle = _write_dsv4_tq_artifact(tmp_path / "dsv4")
    save_file(
        {
            "codebook.2048.2": np.zeros((1,), dtype=np.float32),
            "codebook.4096.2": np.zeros((1,), dtype=np.float32),
            "signs.2048.42": np.zeros((1,), dtype=np.uint8),
            "signs.4096.42": np.zeros((1,), dtype=np.uint8),
        },
        str(bundle / "jangtq_runtime.safetensors"),
    )

    report = _audit_dsv4_artifact_bit_plan(bundle)

    assert report["sidecar"]["present"] is True
    assert "codebook.2048.4" in report["sidecar"]["missing_keys"]
    assert "codebook.4096.4" in report["sidecar"]["missing_keys"]
    assert any("sidecar missing" in issue for issue in report["issues"])


def test_dsv4_artifact_bit_plan_audit_flags_metadata_drift(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_artifact_bit_plan,
    )

    bundle = _write_dsv4_tq_artifact(tmp_path / "dsv4", layer_bits={0: 4, 1: 2})
    jang = json.loads((bundle / "jang_config.json").read_text())
    jang["quantization"]["routed_experts"]["bit_plan"]["routed_layer_bits"] = {
        "0": 4,
        "1": 4,
    }
    (bundle / "jang_config.json").write_text(json.dumps(jang))

    report = _audit_dsv4_artifact_bit_plan(bundle)

    assert report["routed_layer_bits"] == {"0": 4, "1": 2}
    assert report["metadata_routed_layer_bits"] == {"0": 4, "1": 4}
    assert report["metadata_matches_actual"] is False
    assert any("metadata routed_layer_bits" in issue for issue in report["issues"])

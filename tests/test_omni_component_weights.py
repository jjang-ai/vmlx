"""Native media capability must be supported by actual indexed tensor headers."""
import json
from pathlib import Path
import numpy as np
import pytest
from safetensors.numpy import save_file
from vmlx_engine.omni_multimodal import omni_multimodal_component_status

IMAGE = "vision_model.radio_model.model.patch_generator.embedder.weight"

def bundle(root):
    (root / "config.json").write_text(json.dumps({"model_type": "nemotron_h", "hidden_size": 6}))
    (root / "config_omni.json").write_text(json.dumps({
        "sound_config": {"model_type": "parakeet", "hidden_size": 3},
        "vision_config": {"model_type": "radio"}, "downsample_ratio": 1,
    }))
    (root / "configuration_radio.py").write_text("# fixture")
    tensors = {
        IMAGE: np.zeros((4, 3), dtype=np.float16),
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight": np.zeros((12, 4), dtype=np.float16),
        "mlp1.0.weight": np.ones((4,), dtype=np.float16),
        "mlp1.1.weight": np.zeros((8, 4), dtype=np.float16),
        "mlp1.3.weight": np.zeros((6, 8), dtype=np.float16),
        "sound_encoder.encoder.layers.0.conv.weight": np.zeros((3, 3), dtype=np.float16),
        "sound_projection.norm.weight": np.ones((3,), dtype=np.float16),
        "sound_projection.linear1.weight": np.zeros((5, 3), dtype=np.float16),
        "sound_projection.linear2.weight": np.zeros((6, 5), dtype=np.float16),
    }
    save_file(tensors, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    return tensors

def test_real_media_tensor_headers_support_components(tmp_path):
    bundle(tmp_path)
    status = omni_multimodal_component_status(tmp_path)
    assert status["bundle_compatible"]
    assert status["weight_evidence"]["indexed_media_tensors_verified"] == 9
    assert status["has_vision_projector"] and status["has_audio_projector"]

@pytest.mark.parametrize("fault", ["missing_shard", "missing_tensor", "wrong_shard", "truncated", "zero_shape", "integer_projector", "integer_encoder", "audio_projector_missing", "projector_width", "vision_chain", "sound_chain"])
def test_metadata_cannot_substitute_for_media_payload(tmp_path, fault):
    tensors = bundle(tmp_path)
    shard = tmp_path / "model.safetensors"
    if fault == "missing_shard": shard.unlink()
    elif fault == "truncated": shard.write_bytes(b"invalid")
    else:
        if fault == "missing_tensor": tensors.pop(IMAGE)
        elif fault == "wrong_shard":
            save_file(tensors, str(tmp_path / "elsewhere.safetensors")); tensors = {"unrelated.weight": np.ones((1,), dtype=np.float16)}
        elif fault == "zero_shape": tensors[IMAGE] = np.empty((0, 3), dtype=np.float16)
        elif fault == "integer_projector": tensors["sound_projection.linear2.weight"] = np.zeros((6, 5), dtype=np.int32)
        elif fault == "integer_encoder": tensors["sound_encoder.encoder.layers.0.conv.weight"] = np.zeros((3, 3), dtype=np.int32)
        elif fault == "audio_projector_missing":
            tensors = {k: v for k,v in tensors.items() if not k.startswith("sound_projection.")}
            (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
        elif fault == "projector_width": tensors["sound_projection.linear2.weight"] = np.zeros((7, 5), dtype=np.float16)
        elif fault == "vision_chain": tensors["mlp1.1.weight"] = np.zeros((8, 5), dtype=np.float16)
        elif fault == "sound_chain": tensors["sound_projection.linear1.weight"] = np.zeros((4, 3), dtype=np.float16)
        save_file(tensors, str(shard))
    status = omni_multimodal_component_status(tmp_path)
    assert not status["bundle_compatible"], status
    assert status["missing"]

@pytest.mark.parametrize("override", [
    {"llm_config": {"hidden_size": 7}},
    {"projector_hidden_size": 9},
    {"sound_config": {"model_type": "parakeet", "hidden_size": 3, "projection_hidden_size": 4}},
])
def test_projector_config_must_match_actual_weights(tmp_path, override):
    bundle(tmp_path)
    p = tmp_path / "config_omni.json"
    config = json.loads(p.read_text()); config.update(override); p.write_text(json.dumps(config))
    assert not omni_multimodal_component_status(tmp_path)["bundle_compatible"]


def test_temporal_header_shape_is_checked_during_component_admission(tmp_path):
    tensors = bundle(tmp_path)
    key = IMAGE.replace(".embedder.", ".video_embedder.")
    tensors[key] = np.zeros((5, 6), dtype=np.float16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    assert not omni_multimodal_component_status(tmp_path)["bundle_compatible"]

@pytest.mark.parametrize("dtype", [np.int64, np.float16])
def test_scalar_encoder_buffers_are_not_empty_weights(tmp_path, dtype):
    tensors = bundle(tmp_path)
    tensors["sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked"] = np.array(0, dtype=dtype)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    status = omni_multimodal_component_status(tmp_path)
    assert status["bundle_compatible"], status
    assert status["weight_evidence"]["indexed_media_tensors_verified"] == 10

def test_valid_temporal_weights_advertise_native_bridge_without_legacy_processor(tmp_path):
    tensors = bundle(tmp_path)
    key = IMAGE.replace(".embedder.", ".video_embedder.")
    tensors[key] = np.zeros((4, 6), dtype=np.float16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    status = omni_multimodal_component_status(tmp_path)
    assert status["bundle_compatible"] and status["video_bridge_supported"]
    assert status["temporal_video_spec"]["temporal_patch_size"] == 2

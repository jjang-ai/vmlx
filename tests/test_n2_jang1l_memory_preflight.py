import json
from types import SimpleNamespace

from tests.cross_matrix import run_n2_jang1l_memory_preflight as runner


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_n2_jang1l_memory_preflight_builds_unit_labeled_no_load_artifact(tmp_path, monkeypatch):
    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    _write_json(
        model / "model.safetensors.index.json",
        {
            "metadata": {"format": "jang", "format_version": "2.0", "total_size": "10737418240"},
            "weight_map": {
                "language_model.model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "vision_tower.vision_model.embeddings.patch_embedding.weight": "b.safetensors",
                "language_model.model.layers.1.mlp.gate_proj.weight": "c.safetensors",
            },
        },
    )
    _write_json(
        model / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
        },
    )
    _write_json(
        model / "jang_config.json",
        {
            "format": "jang",
            "quantization": {
                "profile": "JANG_1L",
                "target_bits": 1.0,
                "actual_bits": 2.13,
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "memory_snapshot",
        lambda: {
            "unit": "GiB",
            "total_gib": 128.0,
            "available_gib": 11.0,
            "percent": 91.4,
        },
    )

    result = runner.build_preflight(
        SimpleNamespace(model=model, required_extra_headroom_gib=4.0)
    )

    assert result["status"] == "open"
    assert result["decision"] == "do_not_launch"
    assert result["launch_safe"] is False
    assert result["unit"] == "GiB"
    assert result["indexed_payload_gib"] == 10.0
    assert result["required_available_gib"] == 14.0
    assert result["available_gib"] == 11.0
    assert result["memory_gap_gib"] == 3.0
    assert result["classification"] == "careful_ram_live_proof_pending"
    assert result["artifact_profile"] == "JANG_1L"
    assert result["format"] == "jang"
    assert result["model_type"] == "qwen3_5_moe"
    assert result["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert result["weights"]["vision_tensors"] == 1
    assert result["weights"]["linear_attention_tensors"] == 1
    assert result["weights"]["audio_tensors"] == 0
    assert result["no_load"] is True


def test_n2_jang1l_memory_preflight_marks_scheduled_when_headroom_available(tmp_path, monkeypatch):
    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    _write_json(
        model / "model.safetensors.index.json",
        {"metadata": {"total_size": 1024**3}, "weight_map": {}},
    )
    _write_json(model / "config.json", {"model_type": "qwen3_5_moe"})
    monkeypatch.setattr(
        runner,
        "memory_snapshot",
        lambda: {"unit": "GiB", "total_gib": 128.0, "available_gib": 8.0},
    )

    result = runner.build_preflight(
        SimpleNamespace(model=model, required_extra_headroom_gib=4.0)
    )

    assert result["decision"] == "schedule_live_proof"
    assert result["launch_safe"] is True
    assert result["memory_gap_gib"] == 0.0
    assert result["reason"] == "payload plus required runtime headroom fits current available memory"

# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax-K issue #179 local model manifest."""

from pathlib import Path

from tests.cross_matrix import run_issue179_minimax_k_model_manifest as manifest_gate


def test_issue179_model_manifest_records_local_artifact_shape(tmp_path):
    model = tmp_path / "MiniMax-M2.7-JANGTQ_K"
    model.mkdir()
    for name in (
        "config.json",
        "generation_config.json",
        "jang_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "modeling_minimax_m2.py",
        "configuration_minimax_m2.py",
        "jangtq_runtime.safetensors",
    ):
        (model / name).write_text(name, encoding="utf-8")
    for index in range(1, 68):
        (model / f"model-{index:05d}-of-00067.safetensors").write_text(
            str(index),
            encoding="utf-8",
        )

    out = manifest_gate.build_manifest(model, hash_shards=False)

    assert out["status"] == "pass"
    assert out["model_shard_count"] == 67
    assert out["checks"]["model_shard_count_is_67"] is True
    assert out["checks"]["has_jangtq_runtime"] is True
    assert out["summary"]["unhashed_safetensors_count"] == 67
    assert any(
        row["path"] == "generation_config.json" and "sha256" in row
        for row in out["files"]
    )
    assert any(
        row["path"] == "jangtq_runtime.safetensors" and "sha256" in row
        for row in out["files"]
    )


def test_issue179_model_manifest_can_hash_shards_for_full_parity(tmp_path):
    model = tmp_path / "MiniMax-M2.7-JANGTQ_K"
    model.mkdir()
    for name in (
        "config.json",
        "generation_config.json",
        "jang_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "modeling_minimax_m2.py",
        "configuration_minimax_m2.py",
        "jangtq_runtime.safetensors",
    ):
        (model / name).write_text(name, encoding="utf-8")
    for index in range(1, 68):
        (model / f"model-{index:05d}-of-00067.safetensors").write_text(
            str(index),
            encoding="utf-8",
        )

    out = manifest_gate.build_manifest(model, hash_shards=True)

    assert out["status"] == "pass"
    assert out["hash_shards"] is True
    assert out["summary"]["unhashed_safetensors_count"] == 0
    assert any(
        row["path"] == "model-00067-of-00067.safetensors" and "sha256" in row
        for row in out["files"]
    )


def test_issue179_model_manifest_writes_json(tmp_path):
    model = tmp_path / "missing"
    out_path = tmp_path / "manifest.json"

    out = manifest_gate.write_manifest(model, out_path, hash_shards=False)

    assert out["status"] == "missing"
    assert out_path.exists()
    assert '"error": "model_path_missing"' in out_path.read_text(encoding="utf-8")

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_local_generation_metadata_audit_reports_high_risk_rows(tmp_path):
    from tests.cross_matrix.run_local_generation_metadata_audit import build_artifact

    artifact = build_artifact(
        (
            "/Users/example/models/JANGQ/Hy3-preview-JANG_2L",
            "/Users/example/models/JANGQ/MiniMax-M2.7-JANG_K",
            "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        )
    )

    if artifact["row_count"] == 0:
        pytest.skip("high-risk local model paths are not present on this machine")

    assert artifact["status"] == "pass"
    assert artifact["row_count"] >= 1
    rows = {Path(row["path"]).name: row for row in artifact["rows"]}
    if "Hy3-preview-JANG_2L" in rows:
        hy3 = rows["Hy3-preview-JANG_2L"]
        assert hy3["registry"]["reasoning_parser"] == "qwen3"
        assert hy3["registry"]["tool_parser"] == "hunyuan"
        assert hy3["resolved_chat"]["max_tokens"] == 2048
        assert hy3["resolved_chat"]["top_k"] == 0
    if "MiniMax-M2.7-JANG_K" in rows:
        minimax = rows["MiniMax-M2.7-JANG_K"]
        assert minimax["registry"]["reasoning_parser"] == "minimax_m2"
        assert "registry_overrides_stale_capability_reasoning_parser" in minimax["notes"]
        assert minimax["resolved_chat"]["max_tokens"] == 4096
    if "DeepSeek-V4-Flash-JANGTQ-K" in rows:
        dsv4 = rows["DeepSeek-V4-Flash-JANGTQ-K"]
        assert dsv4["resolved_chat"]["repetition_penalty"] == 1.0
        assert dsv4["resolved_thinking"]["repetition_penalty"] == 1.0
        assert dsv4["resolved_chat"]["max_tokens"] == 4096


def test_local_generation_metadata_audit_artifact_is_json_serializable():
    from tests.cross_matrix.run_local_generation_metadata_audit import build_artifact

    artifact = build_artifact(("/definitely/missing/model",))
    encoded = json.dumps(artifact)
    assert '"rows": []' in encoded


def test_local_generation_metadata_audit_flags_thinking_template_without_budget(tmp_path):
    from tests.cross_matrix.run_local_generation_metadata_audit import audit_model

    (tmp_path / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text(
        "{% if enable_thinking is defined and enable_thinking %}<|think|>{% endif %}",
        encoding="utf-8",
    )

    row = audit_model(str(tmp_path))

    assert row["thinking_budget"]["template_mentions_thinking"] is True
    assert row["thinking_budget"]["template_mentions_budget"] is False
    assert "thinking_budget_override_forwarded_but_template_does_not_enforce" in row["notes"]


def test_local_generation_metadata_audit_accepts_template_budget_support(tmp_path):
    from tests.cross_matrix.run_local_generation_metadata_audit import audit_model

    (tmp_path / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text(
        "{% if enable_thinking %}{{ thinking_budget }}<|think|>{% endif %}",
        encoding="utf-8",
    )

    row = audit_model(str(tmp_path))

    assert row["thinking_budget"]["template_mentions_thinking"] is True
    assert row["thinking_budget"]["template_mentions_budget"] is True
    assert "thinking_budget_override_forwarded_but_template_does_not_enforce" not in row["notes"]


def test_local_generation_metadata_audit_reports_step3p7_advertised_media_guard(tmp_path):
    from tests.cross_matrix.run_local_generation_metadata_audit import audit_model

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step3p7",
                "architectures": ["Step3p7ForConditionalGeneration"],
                "vision_config": {"hidden_size": 4096},
                "image_token_id": 128001,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "architecture": {
                    "type": "step3p7",
                    "text_model_type": "step3p5",
                    "has_vision": True,
                    "has_audio": False,
                }
            }
        ),
        encoding="utf-8",
    )

    row = audit_model(str(tmp_path))

    assert row["media_metadata"]["advertised_vision"] is True
    assert row["runtime_route"]["engine_is_mllm_default"] is True
    assert row["runtime_route"]["engine_is_mllm_force"] is True
    assert "step3p7_advertised_media_routes_mllm_by_default" in row["notes"]

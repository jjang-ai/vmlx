from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "all_local_model_smoke", ROOT / "bench" / "all_local_model_smoke.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_model_detects_vl_hybrid_and_mtp(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "Qwen3.6-27B-JANG_4M-MTP"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1
          },
          "vision_config": {"model_type": "qwen2_5_vl"}
        }
        """,
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "qwen3_5"
    assert row["is_mllm"] is True
    assert row["supports_video"] is True
    assert row["has_mtp"] is True
    assert row["cache_family"] == "hybrid_ssm"


def test_classify_model_marks_deepseek_v4_composite_cache(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "DeepSeek-V4-Flash-JANG_2L-MTP"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "deepseek_v4", "num_nextn_predict_layers": 1}',
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "deepseek_v4"
    assert row["cache_family"] == "deepseek_v4_composite"
    assert row["has_mtp"] is True


def test_classify_model_does_not_mark_explicit_nomtp_bundle(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "DeepSeek-V4-Flash-JANG_DQ2-NoMTP"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}',
        encoding="utf-8",
    )
    (model_dir / "jang_config.json").write_text(
        '{"weight_format":"affine"}',
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.embed_tokens.weight":"model.safetensors"}}',
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["cache_family"] == "deepseek_v4_composite"
    assert row["has_mtp"] is False


def test_build_serve_command_uses_production_auto_cache(tmp_path):
    mod = load_module()
    row = {
        "path": str(tmp_path / "model"),
        "served_name": "model",
        "is_mllm": True,
        "cache_family": "hybrid_ssm",
    }

    cmd = mod.build_serve_command(
        row,
        port=8777,
        out_dir=tmp_path / "out",
        py=str(tmp_path / "python"),
    )

    assert "--is-mllm" in cmd
    assert "--use-paged-cache" in cmd
    assert "--enable-block-disk-cache" in cmd
    assert "--kv-cache-quantization" not in cmd
    assert "--default-enable-thinking" in cmd
    assert cmd[cmd.index("--default-enable-thinking") + 1] == "false"


def test_probe_payloads_cover_repeat_text_reasoning_and_media():
    mod = load_module()
    row = {
        "served_name": "vl-model",
        "is_mllm": True,
        "supports_video": True,
        "supports_thinking": True,
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    labels = [probe["label"] for probe in probes]

    assert "text_cache_repeat_1" in labels
    assert "text_cache_repeat_2" in labels
    assert "reasoning_on" in labels
    assert "vl_blue_image" in labels
    assert "text_no_media_after_image" in labels
    assert "vl_blue_video" in labels
    assert "text_no_media_after_video" in labels
    reasoning = next(probe for probe in probes if probe["label"] == "reasoning_on")
    assert reasoning["payload"]["enable_thinking"] is True
    assert reasoning["payload"]["max_tokens"] >= 256


def test_request_json_returns_status_body_and_elapsed(monkeypatch):
    mod = load_module()

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    code, body, elapsed = mod.request_json("GET", "http://127.0.0.1/health")

    assert code == 200
    assert body == {"ok": True}
    assert elapsed >= 0


def test_filter_rows_skips_auxiliary_dirs_by_default():
    mod = load_module()
    rows = [
        {
            "path": "/Users/example/models/JANGQ/Gemma-4-31B-it-JANG_4M-MTP",
            "name": "Gemma-4-31B-it-JANG_4M-MTP",
            "model_type": "gemma4",
            "namespace": "JANGQ",
        },
        {
            "path": "/Users/example/models/JANGQ/Gemma-4-31B-it-JANG_4M-MTP/mtp_assistant",
            "name": "mtp_assistant",
            "model_type": "gemma4_assistant",
            "namespace": "JANGQ",
        },
        {
            "path": "/Users/example/models/JANGQ/_dsv4_jangtq_k_upload_prep/stage_jangq",
            "name": "stage_jangq",
            "model_type": "deepseek_v4",
            "namespace": "JANGQ",
        },
        {
            "path": "/Users/example/models/Kimi-K2.6-JANGTQ",
            "name": "Kimi-K2.6-JANGTQ",
            "model_type": "kimi_k2",
            "namespace": "other",
        },
    ]

    filtered = mod.filter_rows(
        rows,
        only=None,
        skip=None,
        include_dealign=True,
        include_sources=False,
        include_aux=False,
    )

    assert [row["name"] for row in filtered] == ["Gemma-4-31B-it-JANG_4M-MTP"]


def test_summarize_cache_uses_numeric_hits_not_field_names():
    mod = load_module()

    summary = mod.summarize_cache(
        {
            "scheduler": {
                "cache_hit_tokens": 0,
                "hybrid_kv_without_ssm_hits": 0,
            },
            "cache": {
                "scheduler_cache": {
                    "disk_hits": 0,
                    "cache_hits": 0,
                    "engine_path": "paged+zaya_cca",
                }
            },
            "turboquant_kv_cache": {"enabled": False},
        }
    )

    assert summary["cache_hit_tokens"] == 0
    assert summary["disk_hits"] == 0
    assert summary["has_cache_hit"] is False
    assert summary["hybrid_kv_without_ssm_hits"] == 0
    assert summary["mentions_turboquant"] is True


def test_summarize_cache_reports_real_hits_and_hybrid_miss_fallbacks():
    mod = load_module()

    summary = mod.summarize_cache(
        {
            "scheduler": {
                "cache_hit_tokens": 34,
                "hybrid_kv_without_ssm_hits": 1,
                "hybrid_kv_without_ssm_tokens": 12,
            },
            "cache": {"scheduler_cache": {"disk_hits": 2}},
        }
    )

    assert summary["cache_hit_tokens"] == 34
    assert summary["disk_hits"] == 2
    assert summary["has_cache_hit"] is True
    assert summary["hybrid_kv_without_ssm_hits"] == 1
    assert summary["hybrid_kv_without_ssm_tokens"] == 12


def test_validate_probe_response_rejects_dsv4_control_fragment_garbage():
    mod = load_module()

    failures = mod.validate_probe_response(
        "text_cache_repeat_1",
        200,
        "}<?}<?}<?promcodeline}<?prep}<?",
        "",
    )

    reasons = {failure["reason"] for failure in failures}
    assert "incoherent_visible_text" in reasons
    assert "expected_ack_missing" in reasons


def test_validate_probe_response_accepts_ack_probe():
    mod = load_module()

    failures = mod.validate_probe_response("text_cache_repeat_2", 200, "ACK", "")

    assert failures == []


def test_validate_probe_response_requires_multiturn_recall_terms():
    mod = load_module()

    failures = mod.validate_probe_response("text_multiturn_recall", 200, "blue", "")

    assert failures == [
        {
            "label": "text_multiturn_recall",
            "reason": "expected_recall_missing",
            "missing": ["cat"],
        }
    ]


def test_validate_probe_response_requires_reasoning_visible_final_answer():
    mod = load_module()

    failures = mod.validate_probe_response("reasoning_on", 200, "I thought about it.", "hidden")

    assert failures == [
        {
            "label": "reasoning_on",
            "reason": "expected_final_ok_missing",
            "missing": ["FINAL=OK"],
        }
    ]


def test_validate_probe_response_checks_media_color_and_no_media_carryover():
    mod = load_module()

    assert mod.validate_probe_response("vl_blue_image", 200, "blue", "") == []
    assert mod.validate_probe_response("vl_red_image_changed", 200, "red", "") == []
    assert mod.validate_probe_response("text_no_media_after_image", 200, "No.", "") == []

    failures = mod.validate_probe_response("vl_blue_video", 200, "red", "")
    assert failures == [
        {
            "label": "vl_blue_video",
            "reason": "expected_color_missing",
            "expected": "blue",
        }
    ]


def test_collect_probe_failures_uses_semantic_validation():
    mod = load_module()
    requests = [
        {
            "label": "text_cache_repeat_1",
            "code": 200,
            "content": "}<?}<?codeline}<?",
            "reasoning_chars": 0,
        },
        {
            "label": "text_cache_repeat_2",
            "code": 200,
            "content": "ACK",
            "reasoning_chars": 0,
        },
    ]

    failures = mod.collect_probe_failures(requests)

    assert len(failures) == 2
    assert {failure["reason"] for failure in failures} == {
        "incoherent_visible_text",
        "expected_ack_missing",
    }


def test_main_returns_nonzero_when_any_model_probe_fails(monkeypatch, tmp_path):
    mod = load_module()
    row = {
        "path": str(tmp_path / "DeepSeek-V4-Flash-JANG_2L-MTP"),
        "served_name": "deepseek-v4-flash-jang-2l-mtp",
        "name": "DeepSeek-V4-Flash-JANG_2L-MTP",
        "model_type": "deepseek_v4",
        "namespace": "JANGQ",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
    }
    monkeypatch.setattr(mod, "discover_model_dirs", lambda _root: [row])
    monkeypatch.setattr(
        mod,
        "run_model_row",
        lambda *args, **kwargs: {
            "row": row,
            "status": "probe_failed",
            "failures": [{"label": "text_cache_repeat_1", "reason": "incoherent_visible_text"}],
        },
    )
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "all_local_model_smoke.py",
            "--models-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--no-media",
        ],
    )

    assert mod.main() == 1


def test_probe_options_obey_live_text_only_capabilities():
    mod = load_module()
    row = {"is_mllm": True, "supports_video": True}

    options = mod.probe_options_from_capabilities(
        row,
        {"modalities": ["text"]},
        include_media=True,
        include_video=True,
    )

    assert options == {"include_media": False, "include_video": False}


def test_probe_options_allow_image_but_not_video_when_capability_missing_video():
    mod = load_module()
    row = {"is_mllm": True, "supports_video": True}

    options = mod.probe_options_from_capabilities(
        row,
        {"modalities": ["text", "vision"]},
        include_media=True,
        include_video=True,
    )

    assert options == {"include_media": True, "include_video": False}

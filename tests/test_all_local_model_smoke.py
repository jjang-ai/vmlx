from __future__ import annotations

import importlib.util
import json
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


def test_configured_bench_python_resolves_relative_to_repo(monkeypatch):
    monkeypatch.setenv("VMLINUX_BENCH_PYTHON", "panel/bundled-python/python/bin/python3.12")

    mod = load_module()

    assert mod.PYTHON == (ROOT / "panel/bundled-python/python/bin/python3.12").resolve()


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
          "video_token_id": 151656,
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


def test_classify_model_does_not_infer_zaya_vl_video_from_vl_name_or_token(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "ZAYA1-VL-8B-MXFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "zaya1_vl",
          "vision_config": {"model_type": "siglip"},
          "video_token_id": 123456
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "jang_config.json").write_text(
        '{"weight_format":"mxfp4"}',
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["is_mllm"] is True
    assert row["supports_video"] is False
    assert row["supports_thinking"] is False


def test_classify_model_keeps_zaya_text_non_reasoning(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "ZAYA1-8B-MXFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "zaya"}',
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["is_mllm"] is False
    assert row["supports_thinking"] is False


def test_classify_model_does_not_infer_nemotron_omni_video_from_name_only(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "Nemotron-Omni-Nano-JANGTQ-CRACK"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "nemotron_h",
          "vision_config": {"model_type": "radio"},
          "audio_config": {"model_type": "wav2vec2"}
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "jang_config.json").write_text(
        '{"weight_format":"mxtq"}',
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["is_mllm"] is True
    assert row["supports_video"] is False


def test_classify_model_preserves_hy3_and_minimax_reasoning_probe_coverage(tmp_path):
    mod = load_module()

    hy3_dir = tmp_path / "Hy3-preview-JANGTQ2"
    hy3_dir.mkdir()
    (hy3_dir / "config.json").write_text(
        '{"model_type": "hy_v3"}',
        encoding="utf-8",
    )

    minimax_dir = tmp_path / "MiniMax-M2.7-Small-JANGTQ"
    minimax_dir.mkdir()
    (minimax_dir / "config.json").write_text(
        """
        {
          "model_type": "minimax_m2",
          "auto_map": {
            "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"
          }
        }
        """,
        encoding="utf-8",
    )

    assert mod.classify_model_dir(hy3_dir)["supports_thinking"] is True
    assert mod.classify_model_dir(minimax_dir)["supports_thinking"] is True


def test_classify_model_marks_mimo_v2_jang2l_as_multimodal_xml_tools_no_mtp(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "mimo_v2",
          "vision_config": {"hidden_size": 1280},
          "audio_config": {"hidden_size": 1024},
          "max_position_embeddings": 1048576
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "jang_config.json").write_text(
        """
        {
          "weight_format": "jang",
          "runtime": {"bundle_has_mtp": false, "mtp_mode": "absent"},
          "capabilities": {
            "family": "mimo_v2",
            "cache_type": "kv",
            "reasoning_parser": "qwen3",
            "tool_parser": "xml_function",
            "supports_tools": true,
            "supports_thinking": true
          }
        }
        """,
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "mimo_v2"
    assert row["is_mllm"] is True
    assert row["supports_video"] is False
    assert row["supports_thinking"] is True
    assert row["supports_tools"] is True
    assert row["has_mtp"] is False
    assert row["cache_family"] == "mimo_v2_hybrid_swa"


def test_classify_model_reads_mimo_v2_embedded_config_capabilities(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "mimo_v2",
          "vision_config": {"hidden_size": 1280},
          "audio_config": {"hidden_size": 1024},
          "runtime": {
            "bundle_has_mtp": false,
            "mtp_mode": "absent"
          },
          "capabilities": {
            "family": "mimo_v2",
            "modalities": ["text"],
            "preserved_modalities": ["vision", "audio"],
            "unwired_modalities": ["vision", "audio"],
            "multimodal_status": "weights_preserved_text_runtime",
            "cache_type": "kv",
            "reasoning": {
              "supported": true,
              "parser": "think_xml"
            },
            "tools": {
              "supported": true,
              "parser": "xml_function"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "mimo_v2"
    assert row["is_mllm"] is True
    assert row["capabilities"]["modalities"] == ["text"]
    assert row["capabilities"]["preserved_modalities"] == ["vision", "audio"]
    assert row["capabilities"]["unwired_modalities"] == ["vision", "audio"]
    assert row["supports_thinking"] is True
    assert row["supports_tools"] is True
    assert mod.probe_options_from_capabilities(
        row,
        row["capabilities"],
        include_media=True,
        include_video=True,
    ) == {"include_media": False, "include_video": False}


def test_classify_model_marks_mimo_v2_without_jang_config_as_xml_tool_family(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "mimo_v2",
          "image_token_id": 151655,
          "max_position_embeddings": 1048576
        }
        """,
        encoding="utf-8",
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "mimo_v2"
    assert row["supports_thinking"] is True
    assert row["supports_tools"] is True
    assert row["has_mtp"] is False
    assert row["cache_family"] == "mimo_v2_hybrid_swa"
    assert row["has_jang_config"] is False


def test_filter_rows_excludes_mimo_audio_tokenizer_sidecar_by_default(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "MiMo-V2.5-JANG_2L"
    audio_dir = model_dir / "audio_tokenizer"
    audio_dir.mkdir(parents=True)

    rows = [
        {
            "path": str(model_dir),
            "name": model_dir.name,
            "model_type": "mimo_v2",
            "namespace": "JANGQ",
        },
        {
            "path": str(audio_dir),
            "name": "audio_tokenizer",
            "model_type": "unknown",
            "namespace": "JANGQ",
        },
    ]

    filtered = mod.filter_rows(
        rows,
        only="MiMo-V2.5-JANG_2L",
        skip=None,
        include_dealign=True,
        include_sources=False,
        include_aux=False,
    )

    assert [row["name"] for row in filtered] == ["MiMo-V2.5-JANG_2L"]


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
    assert cmd.count("--block-disk-cache-max-gb") == 1
    assert cmd[cmd.index("--block-disk-cache-max-gb") + 1] == "2"
    block_cache_dir = cmd[cmd.index("--block-disk-cache-dir") + 1]
    assert block_cache_dir == str((tmp_path / "out" / "block_cache").resolve())


def test_build_serve_command_uses_dsv4_native_composite_cache_flags(tmp_path):
    mod = load_module()
    row = {
        "path": str(tmp_path / "DeepSeek-V4-Flash-JANGTQ-K"),
        "served_name": "dsv4",
        "is_mllm": False,
        "cache_family": "deepseek_v4_composite",
        "model_type": "deepseek_v4",
    }

    cmd = mod.build_serve_command(
        row,
        port=8777,
        out_dir=tmp_path / "out",
        py=str(tmp_path / "python"),
    )

    assert "--dsv4-enable-prefix-cache" in cmd
    assert "--use-paged-cache" in cmd
    assert "--enable-block-disk-cache" in cmd
    assert "--disable-prefix-cache" not in cmd
    assert "--kv-cache-quantization" not in cmd
    assert cmd[cmd.index("--paged-cache-block-size") + 1] == "256"


def test_server_process_cwd_avoids_repo_root_for_isolated_bundled_smoke(
    monkeypatch,
    tmp_path,
):
    mod = load_module()

    monkeypatch.setenv("VMLINUX_BENCH_ISOLATED", "1")

    assert mod.server_process_cwd(tmp_path / "row") == (tmp_path / "row").resolve()


def test_server_process_cwd_uses_repo_root_for_source_smoke(monkeypatch, tmp_path):
    mod = load_module()

    monkeypatch.delenv("VMLINUX_BENCH_ISOLATED", raising=False)

    assert mod.server_process_cwd(tmp_path / "row") == mod.ROOT


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


def test_probe_payloads_include_tool_probe_for_mimo_xml_tools():
    mod = load_module()
    row = {
        "served_name": "mimo-v2",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": True,
        "supports_tools": True,
        "model_type": "mimo_v2",
    }

    probes = mod.build_probe_payloads(
        row,
        max_tokens=32,
        include_reasoning=True,
        include_tools=True,
    )

    labels = [probe["label"] for probe in probes]
    assert "reasoning_on" in labels
    assert "vl_blue_image" in labels
    assert "tool_required" in labels


def test_repeat_cache_probe_uses_deterministic_exact_output_instruction():
    mod = load_module()
    row = {
        "served_name": "vl-model",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": False,
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    repeat = next(probe for probe in probes if probe["label"] == "text_cache_repeat_1")
    messages = repeat["payload"]["messages"]
    system = messages[0]["content"]
    prompt = messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert "Output exactly ACK and nothing else" in system
    assert "stable words" in prompt
    assert "Reply exactly: ACK" not in prompt


def test_zaya_repeat_cache_probe_uses_family_native_color_contract():
    mod = load_module()
    row = {
        "served_name": "zaya1-8b-mxfp4",
        "name": "ZAYA1-8B-MXFP4",
        "model_type": "zaya",
        "cache_family": "zaya_cca",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": False,
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    repeat = next(probe for probe in probes if probe["label"] == "text_cache_repeat_1")
    messages = repeat["payload"]["messages"]

    assert repeat["expected_content"] == "blue"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Which color word repeats" in messages[0]["content"]
    assert "Output exactly ACK" not in messages[0]["content"]


def test_zaya_reasoning_probe_gets_budget_for_visible_final_answer():
    mod = load_module()
    row = {
        "served_name": "zaya1-8b-mxfp4",
        "name": "ZAYA1-8B-MXFP4",
        "model_type": "zaya",
        "cache_family": "zaya_cca",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
    }

    probes = mod.build_probe_payloads(row, max_tokens=48, include_reasoning=True)
    reasoning = next(probe for probe in probes if probe["label"] == "reasoning_on")

    assert reasoning["payload"]["enable_thinking"] is True
    assert reasoning["payload"]["max_tokens"] >= 512


def test_dsv4_repeat_cache_probe_crosses_native_block_threshold():
    mod = load_module()
    row = {
        "served_name": "dsv4",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "deepseek_v4_composite",
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    repeat = next(probe for probe in probes if probe["label"] == "text_cache_repeat_1")
    prompt = repeat["payload"]["messages"][1]["content"]

    assert len(prompt.split()) >= 320
    assert "dsv4-native-cache-anchor-000" in prompt
    assert "Reply exactly: ACK" not in prompt


def test_no_media_probe_uses_system_scoped_attachment_contract():
    mod = load_module()
    row = {
        "served_name": "vl-model",
        "is_mllm": True,
        "supports_video": True,
        "supports_thinking": False,
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    image_probe = next(
        probe for probe in probes if probe["label"] == "text_no_media_after_image"
    )
    video_probe = next(
        probe for probe in probes if probe["label"] == "text_no_media_after_video"
    )

    image_messages = image_probe["payload"]["messages"]
    video_messages = video_probe["payload"]["messages"]
    assert image_messages[0]["role"] == "system"
    assert video_messages[0]["role"] == "system"
    assert "zero image attachments" in image_messages[0]["content"]
    assert "Answer exactly NONE" in image_messages[0]["content"]
    assert "zero video attachments" in video_messages[0]["content"]
    assert "Answer exactly NONE" in video_messages[0]["content"]
    assert "answer exactly NONE" in image_messages[1]["content"]
    assert "answer exactly IMAGE" in image_messages[1]["content"]
    assert "answer exactly NONE" in video_messages[1]["content"]
    assert "answer exactly VIDEO" in video_messages[1]["content"]
    assert "Do you currently see" not in image_messages[1]["content"]
    assert "Do you currently see" not in video_messages[1]["content"]


def test_zaya_vl_no_media_probe_uses_attached_none_contract():
    mod = load_module()
    row = {
        "served_name": "zaya1-vl-8b-mxfp4",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": False,
        "model_type": "zaya1_vl",
        "cache_family": "zaya_cca",
    }

    probes = mod.build_probe_payloads(row, max_tokens=32, include_reasoning=True)
    assert "reasoning_on" not in {probe["label"] for probe in probes}
    image_probe = next(
        probe for probe in probes if probe["label"] == "text_no_media_after_image"
    )
    image_messages = image_probe["payload"]["messages"]

    assert len(image_messages) == 1
    assert image_messages[0]["role"] == "user"
    assert "answer ATTACHED. Otherwise answer NONE" in image_messages[0]["content"]
    assert "answer exactly IMAGE" not in image_messages[0]["content"]
    assert "zero image attachments" not in image_messages[0]["content"]


def test_classify_zaya_vl_plain_template_does_not_claim_thinking(tmp_path):
    mod = load_module()
    model_dir = tmp_path / "ZAYA1-VL-8B-MXFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "zaya1_vl",
                "vision_config": {"model_type": "qwen2_5_vl"},
                "capabilities": {
                    "family": "zaya1_vl",
                    "reasoning_parser": "qwen3",
                    "supports_thinking": True,
                    "supports_tools": True,
                },
            }
        )
    )
    (model_dir / "jang_config.json").write_text(
        json.dumps(
            {
                "capabilities": {
                    "family": "zaya1_vl",
                    "reasoning_parser": "qwen3",
                    "supports_thinking": True,
                    "supports_tools": True,
                }
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for message in messages %}assistant: {% endfor %}"})
    )

    row = mod.classify_model_dir(model_dir)

    assert row["model_type"] == "zaya1_vl"
    assert row["is_mllm"] is True
    assert row["supports_tools"] is True
    assert row["supports_thinking"] is False


def test_video_probe_uses_effective_probe_option_not_static_inventory_flag():
    mod = load_module()
    row = {
        "served_name": "vl-model",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": False,
    }

    probes = mod.build_probe_payloads(
        row,
        max_tokens=32,
        include_reasoning=True,
        include_video=True,
    )

    assert any(probe["label"] == "vl_blue_video" for probe in probes)
    assert any(probe["label"] == "text_no_media_after_video" for probe in probes)


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


def test_build_server_env_can_isolate_packaged_python(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("PYTHONPATH", "/tmp/source")
    monkeypatch.setenv("VMLINUX_BENCH_ISOLATED", "1")

    env = mod.build_server_env()

    assert env["PYTHONUNBUFFERED"] == "1"
    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_build_server_env_can_add_explicit_source_dependency(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("VMLINUX_BENCH_ISOLATED", raising=False)
    monkeypatch.setenv("VMLINUX_BENCH_EXTRA_PYTHONPATH", "/Users/example/jang/jang-tools")

    env = mod.build_server_env()

    assert env["PYTHONPATH"].split(mod.os.pathsep) == [
        str(mod.ROOT),
        "/Users/example/jang/jang-tools",
    ]


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
    assert "expected_exact_ack_missing" in reasons


def test_validate_probe_response_accepts_ack_probe():
    mod = load_module()

    failures = mod.validate_probe_response("text_cache_repeat_2", 200, "ACK", "")

    assert failures == []


def test_validate_probe_response_rejects_non_bare_ack_for_cache_repeat():
    mod = load_module()

    failures = mod.validate_probe_response("text_cache_repeat_2", 200, "The output is: ACK", "")

    assert failures == [
        {
            "label": "text_cache_repeat_2",
            "reason": "expected_exact_ack_missing",
            "expected": "ACK",
        }
    ]


def test_validate_probe_response_accepts_family_specific_cache_expected_content():
    mod = load_module()

    assert (
        mod.validate_probe_response(
            "text_cache_repeat_2",
            200,
            "blue",
            "",
            expected_content="blue",
        )
        == []
    )
    assert mod.validate_probe_response(
        "text_cache_repeat_2",
        200,
        "green",
        "",
        expected_content="blue",
    ) == [
        {
            "label": "text_cache_repeat_2",
            "reason": "expected_exact_ack_missing",
            "expected": "blue",
        }
    ]


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


def test_build_probe_payloads_can_include_required_tool_probe():
    mod = load_module()
    row = {
        "served_name": "hy3-preview-jangtq2",
        "cache_family": "hybrid_ssm",
        "supports_thinking": True,
        "is_mllm": False,
    }

    probes = mod.build_probe_payloads(
        row,
        max_tokens=48,
        include_reasoning=True,
        include_media=True,
        include_video=True,
        include_tools=True,
    )

    tool_probe = next(probe for probe in probes if probe["label"] == "tool_required")
    payload = tool_probe["payload"]

    assert payload["tool_choice"] == "required"
    assert payload["tools"][0]["function"]["name"] == "record_fact"
    assert "blue-cat" in payload["messages"][0]["content"]


def test_validate_probe_response_requires_expected_tool_call():
    mod = load_module()

    failures = mod.validate_probe_response(
        "tool_required",
        200,
        "",
        "",
        tool_calls=[],
    )

    assert failures == [
        {
            "label": "tool_required",
            "reason": "expected_tool_call_missing",
            "expected_tool": "record_fact",
        }
    ]


def test_validate_probe_response_accepts_expected_tool_call():
    mod = load_module()

    failures = mod.validate_probe_response(
        "tool_required",
        200,
        "",
        "",
        tool_calls=[
            {
                "function": {
                    "name": "record_fact",
                    "arguments": '{"value":"blue-cat"}',
                }
            }
        ],
    )

    assert failures == []


def test_validate_probe_response_rejects_runaway_reasoning_loop():
    mod = load_module()

    failures = mod.validate_probe_response(
        "reasoning_on",
        200,
        "FINAL=OK",
        "thinking " * 600,
    )

    assert {
        "label": "reasoning_on",
        "reason": "reasoning_loop_too_long",
        "reasoning_chars": 5400,
        "max_reasoning_chars": 4096,
    } in failures


def test_validate_probe_response_checks_media_color_and_no_media_carryover():
    mod = load_module()

    assert mod.validate_probe_response("vl_blue_image", 200, "blue", "") == []
    assert mod.validate_probe_response("vl_red_image_changed", 200, "red", "") == []
    assert mod.validate_probe_response("text_no_media_after_image", 200, "No.", "") == []
    assert mod.validate_probe_response("text_no_media_after_image", 200, "NONE", "") == []
    assert (
        mod.validate_probe_response(
            "text_no_media_after_image",
            200,
            "The image is not included in the current user message.",
            "",
        )
        == []
    )

    failures = mod.validate_probe_response("vl_blue_video", 200, "red", "")
    assert failures == [
        {
            "label": "vl_blue_video",
            "reason": "expected_color_missing",
            "expected": "blue",
        }
    ]
    failures = mod.validate_probe_response("text_no_media_after_image", 200, "image", "")
    assert {
        "label": "text_no_media_after_image",
        "reason": "expected_no_media_missing",
        "expected": "no-or-none",
    } in failures
    assert {
        "label": "text_no_media_after_image",
        "reason": "unexpected_media_carryover_claim",
        "media_kind": "image",
    } in failures


def test_validate_probe_response_rejects_stray_cjk_even_when_expected_term_present():
    mod = load_module()

    failures = mod.validate_probe_response(
        "vl_blue_image",
        200,
        "blue 蓝色 蓝色 蓝色 蓝色",
        "",
    )

    assert {
        "label": "vl_blue_image",
        "reason": "unexpected_cjk_visible_text",
        "cjk_chars": 8,
    } in failures


def test_validate_probe_response_rejects_wider_cjk_kana_and_hangul_ranges():
    mod = load_module()

    failures = mod.validate_probe_response(
        "vl_blue_image",
        200,
        "blue ｶﾀｶﾅ 한 𠀋",
        "",
    )

    assert {
        "label": "vl_blue_image",
        "reason": "unexpected_cjk_visible_text",
        "cjk_chars": 8,
    } in failures


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
        "expected_exact_ack_missing",
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

    summary = mod.json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["status"] == "fail"
    assert summary["failed"] == 1
    assert summary["completed"] == 1
    assert summary["row_count"] == 1
    assert summary["results"][0]["status"] == "probe_failed"


def test_main_writes_top_level_pass_status_when_all_model_probes_pass(
    monkeypatch,
    tmp_path,
):
    mod = load_module()
    row = {
        "path": str(tmp_path / "Gemma-4-26B-A4B-JANG_4M-CRACK"),
        "served_name": "gemma4",
        "name": "Gemma-4-26B-A4B-JANG_4M-CRACK",
        "model_type": "gemma4",
        "namespace": "dealign.ai",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": True,
    }
    monkeypatch.setattr(mod, "discover_model_dirs", lambda _root: [row])
    monkeypatch.setattr(
        mod,
        "run_model_row",
        lambda *args, **kwargs: {
            "row": row,
            "status": "pass",
            "failures": [],
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

    assert mod.main() == 0

    summary = mod.json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["status"] == "pass"
    assert summary["failed"] == 0
    assert summary["completed"] == 1
    assert summary["row_count"] == 1


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

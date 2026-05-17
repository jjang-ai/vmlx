from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_speed_ab_module():
    spec = importlib.util.spec_from_file_location(
        "native_mtp_speed_ab", ROOT / "bench" / "native_mtp_speed_ab.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_chunk_probe_module():
    spec = importlib.util.spec_from_file_location(
        "native_mtp_chunk_equivalence_probe",
        ROOT / "bench" / "native_mtp_chunk_equivalence_probe.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_vl_gate_module():
    spec = importlib.util.spec_from_file_location(
        "native_mtp_vl_gate", ROOT / "bench" / "native_mtp_vl_gate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_artifact_audit_module():
    spec = importlib.util.spec_from_file_location(
        "native_mtp_artifact_audit", ROOT / "bench" / "native_mtp_artifact_audit.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_audit_flags_qwen36_mtp_eos_without_conflating_top_k(tmp_path):
    mod = load_artifact_audit_module()
    model_dir = tmp_path / "Qwen3.6-35B-A3B-JANG_2K-MTP"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "eos_token_id": 248046,
            "num_experts_per_tok": 8,
            "mtp_num_hidden_layers": 1
          }
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(
        """
        {
          "eos_token_id": [248046, 248046],
          "top_k": 20,
          "top_p": 0.95,
          "temperature": 1.0,
          "do_sample": true
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        """
        {
          "added_tokens_decoder": {
            "248044": {"content": "<|endoftext|>"},
            "248046": {"content": "<|im_end|>"}
          }
        }
        """,
        encoding="utf-8",
    )

    report = mod.audit_artifact(model_dir)

    assert report["ok"] is False
    assert "generation_config.eos_token_id missing <|endoftext|> id 248044" in report["issues"]
    assert "generation_config.eos_token_id contains duplicate ids: [248046]" in report["issues"]
    assert "text_config.eos_token_id should be <|endoftext|> id 248044, got 248046" in report["issues"]
    assert report["generation"]["top_k"] == 20
    assert report["routing"]["trained_num_experts_per_tok"] == 8


def test_artifact_audit_accepts_qwen36_mtp_stop_and_sampling_metadata(tmp_path):
    mod = load_artifact_audit_module()
    model_dir = tmp_path / "Qwen3.6-35B-A3B-MXFP8-MTP"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "eos_token_id": 248044,
            "num_experts_per_tok": 8,
            "mtp_num_hidden_layers": 1
          }
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(
        """
        {
          "eos_token_id": [248046, 248044],
          "top_k": 20,
          "top_p": 0.95,
          "temperature": 1.0,
          "do_sample": true
        }
        """,
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        """
        {
          "added_tokens_decoder": {
            "248044": {"content": "<|endoftext|>"},
            "248046": {"content": "<|im_end|>"}
          }
        }
        """,
        encoding="utf-8",
    )

    report = mod.audit_artifact(model_dir)

    assert report["ok"] is True
    assert report["issues"] == []
    assert report["generation"]["eos_token_id"] == [248046, 248044]
    assert report["generation"]["top_k"] == 20
    assert report["routing"]["trained_num_experts_per_tok"] == 8


def test_build_native_mtp_env_sets_aliases_and_probe_knobs():
    mod = load_speed_ab_module()
    args = SimpleNamespace(
        depth=2,
        trace_mtp=True,
        debug_mtp_tokens=True,
        mtp_burst=False,
        mtp_cost_fallback=False,
        mtp_cost_ar_step_ms=None,
        mtp_cost_ratio_threshold=1.0,
    )

    env = mod.build_native_mtp_env({}, mtp_enabled=True, args=args)

    assert env["VMLINUX_NATIVE_MTP"] == "1"
    assert env["VMLX_NATIVE_MTP"] == "1"
    assert env["VMLINUX_NATIVE_MTP_DEPTH"] == "2"
    assert env["VMLX_NATIVE_MTP_DEPTH"] == "2"
    assert env["VMLINUX_NATIVE_MTP_TRACE"] == "1"
    assert env["VMLINUX_NATIVE_MTP_DEBUG_TOKENS"] == "1"
    assert env["VMLINUX_NATIVE_MTP_BURST"] == "0"


def test_build_native_mtp_env_omits_depth_when_unset():
    mod = load_speed_ab_module()
    args = SimpleNamespace(
        depth=None,
        trace_mtp=False,
        debug_mtp_tokens=False,
        mtp_burst=True,
        mtp_cost_fallback=False,
        mtp_cost_ar_step_ms=None,
        mtp_cost_ratio_threshold=1.0,
    )

    env = mod.build_native_mtp_env({}, mtp_enabled=True, args=args)

    assert env["VMLINUX_NATIVE_MTP"] == "1"
    assert env["VMLX_NATIVE_MTP"] == "1"
    assert "VMLINUX_NATIVE_MTP_DEPTH" not in env
    assert "VMLX_NATIVE_MTP_DEPTH" not in env


def test_build_native_mtp_env_sets_cost_fallback_calibration():
    mod = load_speed_ab_module()
    args = SimpleNamespace(
        depth=3,
        trace_mtp=False,
        debug_mtp_tokens=False,
        mtp_burst=True,
        mtp_cost_fallback=True,
        mtp_cost_ar_step_ms=43.21,
        mtp_cost_ratio_threshold=0.98,
    )

    env = mod.build_native_mtp_env({}, mtp_enabled=True, args=args)

    assert env["VMLINUX_NATIVE_MTP_TRACE"] == "1"
    assert env["VMLINUX_NATIVE_MTP_COST_FALLBACK"] == "1"
    assert env["VMLINUX_NATIVE_MTP_AR_STEP_MS"] == "43.210000"
    assert env["VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD"] == "0.980000"


def test_calibrate_cost_fallback_from_baseline_summary():
    mod = load_speed_ab_module()
    args = SimpleNamespace(mtp_cost_fallback=True, mtp_cost_ar_step_ms=None)
    baseline_row = {"summary": {"mean_wall_tok_s": 25.0}}

    ar_step = mod.calibrate_mtp_cost_fallback(args, baseline_row)

    assert ar_step == 40.0
    assert args.mtp_cost_ar_step_ms == 40.0


def test_calibrate_cost_fallback_preserves_explicit_ar_step():
    mod = load_speed_ab_module()
    args = SimpleNamespace(mtp_cost_fallback=True, mtp_cost_ar_step_ms=37.5)
    baseline_row = {"summary": {"mean_wall_tok_s": 25.0}}

    ar_step = mod.calibrate_mtp_cost_fallback(args, baseline_row)

    assert ar_step == 37.5
    assert args.mtp_cost_ar_step_ms == 37.5


def test_vl_gate_probe_payloads_cover_image_no_media_history_and_video():
    mod = load_vl_gate_module()

    probes = mod.build_probe_payloads(
        model="qwen36-vl",
        include_video=True,
        max_tokens=24,
    )

    labels = [probe["label"] for probe in probes]
    assert labels == [
        "vl_blue_image",
        "text_no_media_after_image",
        "vl_image_history_then_text",
        "vl_blue_video",
        "text_no_media_after_video",
    ]
    image_parts = probes[0]["payload"]["messages"][0]["content"]
    assert image_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert probes[1]["payload"]["messages"][0]["content"].startswith(
        "Do you currently see an image"
    )
    history_messages = probes[2]["payload"]["messages"]
    assert history_messages[0]["content"][1]["type"] == "image_url"
    assert isinstance(history_messages[-1]["content"], str)
    video_parts = probes[3]["payload"]["messages"][0]["content"]
    assert video_parts[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert probes[3]["payload"]["video_fps"] == 2
    assert probes[3]["payload"]["video_max_frames"] == 4


def test_vl_gate_probe_payloads_can_skip_video():
    mod = load_vl_gate_module()

    probes = mod.build_probe_payloads(
        model="qwen36-vl",
        include_video=False,
        max_tokens=24,
    )

    assert [probe["label"] for probe in probes] == [
        "vl_blue_image",
        "text_no_media_after_image",
        "vl_image_history_then_text",
    ]


def test_vl_gate_long_probe_payloads_force_decode_beyond_one_word():
    mod = load_vl_gate_module()

    probes = mod.build_probe_payloads(
        model="qwen36-vl",
        include_video=True,
        max_tokens=64,
        long_probes=True,
    )

    image_text = probes[0]["payload"]["messages"][0]["content"][0]["text"]
    no_media_text = probes[1]["payload"]["messages"][0]["content"]
    video_text = probes[3]["payload"]["messages"][0]["content"][0]["text"]
    assert "First answer" in image_text
    assert "two concise sentences" in image_text
    assert "First answer" in no_media_text
    assert "two concise sentences" in video_text


def test_vl_gate_repeat_media_payloads_cover_hit_and_miss_candidates():
    mod = load_vl_gate_module()

    probes = mod.build_repeat_media_cache_payloads(
        model="qwen36-vl",
        include_video=True,
        max_tokens=32,
    )

    labels = [probe["label"] for probe in probes]
    assert labels == [
        "vl_blue_image_repeat_1",
        "vl_blue_image_repeat_2",
        "vl_red_image_changed",
        "vl_blue_video_repeat_1",
        "vl_blue_video_repeat_2",
        "vl_blue_video_changed_settings",
    ]
    first_image = probes[0]["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    second_image = probes[1]["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    changed_image = probes[2]["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    assert first_image == second_image
    assert changed_image != first_image
    first_video = probes[3]["payload"]["messages"][0]["content"][1]["video_url"]["url"]
    second_video = probes[4]["payload"]["messages"][0]["content"][1]["video_url"]["url"]
    changed_settings_video = probes[5]["payload"]["messages"][0]["content"][1]["video_url"]["url"]
    assert first_video == second_video == changed_settings_video
    assert probes[3]["payload"]["video_fps"] == probes[4]["payload"]["video_fps"] == 2
    assert probes[5]["payload"]["video_fps"] == 1
    assert probes[5]["payload"]["video_max_frames"] == 2


def test_vl_gate_cache_counters_extract_repeat_media_fields():
    mod = load_vl_gate_module()
    stats = {
        "scheduler_cache": {
            "cache_hit_tokens_by_detail": {"paged+ssm": 19},
            "hits": 2,
            "misses": 3,
        },
        "block_disk_cache": {"disk_hits": 1, "disk_writes": 2},
        "ssm_companion": {"entries": 2, "stores": 4},
    }

    counters = mod.cache_counters(stats)

    assert counters == {
        "scheduler_hits": 2,
        "scheduler_misses": 3,
        "cache_hit_tokens_by_detail": {"paged+ssm": 19},
        "block_disk_hits": 1,
        "block_disk_writes": 2,
        "ssm_entries": 2,
        "ssm_stores": 4,
    }


def test_vl_gate_env_sets_native_mtp_and_cost_flags():
    mod = load_vl_gate_module()
    args = SimpleNamespace(
        depth=3,
        trace_mtp=False,
        cost_fallback=True,
        cost_ar_step_ms=43.4,
        cost_ratio_threshold=0.95,
        media_prefix_cache=False,
    )

    env = mod.build_env({}, mode="mtp", args=args)

    assert env["VMLINUX_NATIVE_MTP"] == "1"
    assert env["VMLX_NATIVE_MTP"] == "1"
    assert env["VMLINUX_NATIVE_MTP_DEPTH"] == "3"
    assert env["VMLINUX_NATIVE_MTP_TRACE"] == "1"
    assert env["VMLINUX_NATIVE_MTP_COST_FALLBACK"] == "1"
    assert env["VMLINUX_NATIVE_MTP_AR_STEP_MS"] == "43.400000"
    assert env["VMLINUX_NATIVE_MTP_COST_RATIO_THRESHOLD"] == "0.950000"


def test_vl_gate_env_can_enable_media_prefix_cache():
    mod = load_vl_gate_module()
    args = SimpleNamespace(
        depth=3,
        trace_mtp=False,
        cost_fallback=False,
        cost_ar_step_ms=None,
        cost_ratio_threshold=1.0,
        media_prefix_cache=True,
    )

    env = mod.build_env({}, mode="mtp", args=args)

    assert env["VMLINUX_MLLM_MEDIA_PREFIX_CACHE"] == "1"


def test_vl_gate_summarize_result_records_checks_and_cache():
    mod = load_vl_gate_module()
    result = {
        "mode": "mtp",
        "health_after": {"mtp": {"runtime_active": True, "effective_depth": 3}},
        "cache_after": {
            "scheduler_cache": {"cache_hit_tokens_by_detail": {"paged+ssm": 12}},
            "block_disk_cache": {"disk_hits": 1, "disk_writes": 2},
            "ssm_companion": {"entries": 1},
        },
        "requests": [
            {
                "label": "vl_blue_image",
                "ok": True,
                "content": "blue",
                "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
            },
            {
                "label": "text_no_media_after_image",
                "ok": True,
                "content": "no",
                "usage": {"prompt_tokens_details": {"cached_tokens": 12}},
            },
        ],
    }

    summary = mod.summarize_result(result)

    assert summary["mode"] == "mtp"
    assert summary["mtp_runtime_active"] is True
    assert summary["mtp_effective_depth"] == 3
    assert summary["all_http_ok"] is True
    assert summary["visible_nonempty"] is True
    assert summary["no_media_after_image_answer"] == "no"
    assert summary["cache_hit_tokens_by_detail"] == {"paged+ssm": 12}
    assert summary["block_disk_hits"] == 1
    assert summary["ssm_entries"] == 1


def test_effective_native_mtp_depth_prefers_health_when_unset():
    mod = load_speed_ab_module()
    health = {
        "mtp": {
            "effective_depth": 2,
            "effective_depth_source": "vmlx_mtp_tuning.json:native_mtp.best_depth",
        }
    }

    depth, source = mod.effective_native_mtp_depth(None, health)

    assert depth == 2
    assert source == "vmlx_mtp_tuning.json:native_mtp.best_depth"


def test_parse_mtp_log_extracts_acceptance_and_timing_totals():
    mod = load_speed_ab_module()
    lines = [
        "MLLM MTP[req-1] finish=done cycles=50 accepted=140/150 (93.3%) "
        "emits[init=2,draft=140,bonus=46,verify=4]",
        "MLLM MTP[req-1] accept_by_depth[d1=50/50,d2=47/50,d3=43/50] "
        "forwards[seed_main=1,verify_main=50,replay_main=3,mtp=153]",
        "MLLM MTP[req-1] timings_ms[verify=3007.43 sample=11.52 draft=777.78 "
        "snapshot=2.69 restore=0.60 replay=16.32 materialize=1.00 avg_cycle=76.35]",
    ]

    parsed = mod.parse_mtp_log(lines)

    assert parsed["requests"]["req-1"]["cycles"] == 50
    assert parsed["requests"]["req-1"]["accepted_tokens"] == 140
    assert parsed["requests"]["req-1"]["drafted_tokens"] == 150
    assert parsed["requests"]["req-1"]["acceptance_rate"] == 140 / 150
    assert parsed["requests"]["req-1"]["accepted_by_depth"] == [50, 47, 43]
    assert parsed["requests"]["req-1"]["drafted_by_depth"] == [50, 50, 50]
    assert parsed["requests"]["req-1"]["acceptance_by_depth"] == [1.0, 0.94, 0.86]
    assert parsed["requests"]["req-1"]["forwards"] == {
        "seed_main": 1,
        "verify_main": 50,
        "replay_main": 3,
        "mtp": 153,
    }
    assert parsed["requests"]["req-1"]["timings_ms"]["verify"] == 3007.43
    assert parsed["totals"]["cycles"] == 50
    assert parsed["totals"]["accepted_tokens"] == 140
    assert parsed["totals"]["drafted_tokens"] == 150
    assert parsed["totals"]["acceptance_rate"] == 140 / 150
    assert parsed["totals"]["accepted_by_depth"] == [50, 47, 43]
    assert parsed["totals"]["drafted_by_depth"] == [50, 50, 50]
    assert parsed["totals"]["acceptance_by_depth"] == [1.0, 0.94, 0.86]
    assert parsed["totals"]["forwards"] == {
        "seed_main": 1,
        "verify_main": 50,
        "replay_main": 3,
        "mtp": 153,
    }
    assert parsed["totals"]["timings_ms"]["verify"] == 3007.43
    assert parsed["totals"]["timings_ms"]["draft"] == 777.78
    assert parsed["totals"]["timings_ms"]["trace_phase_total"] == 3817.34


def test_hot_path_accounting_separates_wall_scheduler_and_trace_time():
    mod = load_speed_ab_module()
    row = {
        "generations": [
            {"elapsed_sec": 2.18, "completion_tokens": 96},
        ],
        "health_after": {
            "scheduler": {
                "batch_generator": {
                    "generation_time": 1.788,
                    "generation_tokens": 96,
                }
            }
        },
        "mtp_stats": {
            "totals": {
                "cycles": 24,
                "accepted_tokens": 70,
                "drafted_tokens": 72,
                "timings_ms": {
                    "verify": 1446.21,
                    "sample": 5.37,
                    "draft": 321.59,
                    "snapshot": 1.26,
                    "restore": 0.18,
                    "replay": 5.00,
                    "materialize": 0.49,
                    "trace_phase_total": 1780.10,
                },
            }
        },
    }

    accounting = mod.hot_path_accounting(row, warmup=0)

    assert accounting["wall_ms"] == 2180.0
    assert accounting["scheduler_generation_ms"] == 1788.0
    assert accounting["trace_phase_ms"] == 1780.10
    assert accounting["wall_minus_scheduler_ms"] == 392.0
    assert accounting["scheduler_minus_trace_ms"] == 7.90
    assert accounting["verify_ms_per_cycle"] == 60.26
    assert accounting["draft_ms_per_cycle"] == 13.40
    assert accounting["accepted_tokens_per_cycle"] == 2.92
    assert accounting["scheduler_generation_includes_warmup"] is False


def test_repeat_output_equivalence_flags_cache_hit_drift_inside_each_row():
    mod = load_speed_ab_module()
    result = {
        "rows": [
            {
                "label": "baseline_no_mtp",
                "native_mtp_depth": None,
                "generations": [
                    {
                        "content": "Sunlight appears white.",
                        "reasoning_content": "",
                        "completion_tokens": 145,
                        "finish_reason": "stop",
                    },
                    {
                        "content": "Sunlight reaches Earth as white light.",
                        "reasoning_content": "",
                        "completion_tokens": 141,
                        "finish_reason": "stop",
                    },
                ],
            },
            {
                "label": "native_mtp",
                "native_mtp_depth": 3,
                "generations": [
                    {
                        "content": "Sunlight appears white.",
                        "reasoning_content": "",
                        "completion_tokens": 145,
                        "finish_reason": "stop",
                    },
                    {
                        "content": "Sunlight reaches Earth as white light.",
                        "reasoning_content": "",
                        "completion_tokens": 141,
                        "finish_reason": "stop",
                    },
                ],
            },
        ]
    }

    summary = mod.summarize_repeat_output_equivalence(result)

    assert summary["has_repeat_comparisons"] is True
    assert summary["all_rows_full_text_equal"] is False
    assert summary["all_rows_token_count_equal"] is False
    assert summary["row_summaries"][0]["label"] == "baseline_no_mtp"
    assert summary["row_summaries"][0]["all_full_text_equal"] is False
    assert summary["row_summaries"][1]["label"] == "native_mtp"
    assert summary["row_summaries"][1]["all_full_text_equal"] is False
    assert summary["row_summaries"][1]["comparisons"] == [
        {
            "index": 1,
            "reference_index": 0,
            "content_equal": False,
            "reasoning_equal": True,
            "full_text_equal": False,
            "reference_tokens": 145,
            "repeat_tokens": 141,
            "token_count_equal": False,
            "reference_finish": "stop",
            "repeat_finish": "stop",
            "finish_reason_equal": True,
        }
    ]


def test_parse_depth_sweep_accepts_unique_depths():
    mod = load_speed_ab_module()

    assert mod.parse_depth_sweep("3,2,2,1") == [3, 2, 1]
    assert mod.parse_depth_sweep("") == []


def test_select_best_depth_prefers_d2_when_d3_is_slower():
    mod = load_speed_ab_module()
    result = {
        "output_equivalence": {
            "comparisons": [
                {
                    "label": "native_mtp_d1",
                    "full_text_equal": True,
                    "mtp_tokens": 96,
                    "baseline_tokens": 96,
                },
                {
                    "label": "native_mtp_d2",
                    "full_text_equal": True,
                    "mtp_tokens": 96,
                    "baseline_tokens": 96,
                },
                {
                    "label": "native_mtp_d3",
                    "full_text_equal": True,
                    "mtp_tokens": 96,
                    "baseline_tokens": 96,
                },
            ]
        },
        "rows": [
            {
                "label": "baseline_no_mtp",
                "native_mtp_depth": None,
                "summary": {"mean_wall_tok_s": 26.5},
            },
            {
                "label": "native_mtp_d1",
                "native_mtp_depth": 1,
                "summary": {"mean_wall_tok_s": 40.0},
            },
            {
                "label": "native_mtp_d2",
                "native_mtp_depth": 2,
                "summary": {"mean_wall_tok_s": 47.4},
            },
            {
                "label": "native_mtp_d3",
                "native_mtp_depth": 3,
                "summary": {"mean_wall_tok_s": 43.2},
            },
        ],
    }

    selected = mod.select_best_depth(result)

    assert selected["best_depth"] == 2
    assert selected["speedup_vs_baseline"] == 47.4 / 26.5
    assert selected["candidate_count"] == 3


def test_select_best_depth_rejects_nonequivalent_faster_row():
    mod = load_speed_ab_module()
    result = {
        "output_equivalence": {
            "comparisons": [
                {
                    "label": "native_mtp_d2",
                    "full_text_equal": True,
                    "mtp_tokens": 96,
                    "baseline_tokens": 96,
                },
                {
                    "label": "native_mtp_d3",
                    "full_text_equal": False,
                    "mtp_tokens": 72,
                    "baseline_tokens": 96,
                },
            ]
        },
        "rows": [
            {
                "label": "baseline_no_mtp",
                "native_mtp_depth": None,
                "summary": {"mean_wall_tok_s": 26.5},
            },
            {
                "label": "native_mtp_d2",
                "native_mtp_depth": 2,
                "summary": {"mean_wall_tok_s": 47.4},
            },
            {
                "label": "native_mtp_d3",
                "native_mtp_depth": 3,
                "summary": {"mean_wall_tok_s": 60.0},
            },
        ],
    }

    selected = mod.select_best_depth(result)

    assert selected["best_depth"] == 2
    assert selected["rejected_depths"] == [
        {
            "depth": 3,
            "label": "native_mtp_d3",
            "reason": "output_not_equivalent",
        }
    ]


def test_cache_on_cli_args_can_disable_storage_kv_quantization(tmp_path):
    mod = load_speed_ab_module()

    args = mod.cache_mode_cli_args(
        "on",
        block_cache_dir=tmp_path / "blocks",
        kv_cache_quantization="none",
    )

    assert "--use-paged-cache" in args
    assert "--enable-block-disk-cache" in args
    assert "--kv-cache-quantization" not in args


def test_cache_on_cli_args_auto_leaves_storage_quantization_to_engine(tmp_path):
    mod = load_speed_ab_module()

    args = mod.cache_mode_cli_args(
        "on",
        block_cache_dir=tmp_path / "blocks",
        kv_cache_quantization="auto",
    )

    assert "--use-paged-cache" in args
    assert "--kv-cache-quantization" not in args


def test_cache_on_cli_args_can_force_q4_storage_quantization(tmp_path):
    mod = load_speed_ab_module()

    args = mod.cache_mode_cli_args(
        "on",
        block_cache_dir=tmp_path / "blocks",
        kv_cache_quantization="q4",
    )

    assert args[-2:] == ["--kv-cache-quantization", "q4"]


def test_chat_completion_body_sets_thinking_off_by_default():
    mod = load_speed_ab_module()

    body = mod.build_chat_completion_body(
        model="qwen36",
        prompt="hi",
        max_tokens=32,
        repeat_index=0,
        disable_prompt_reuse=False,
        enable_thinking="off",
    )

    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_completion_body_can_enable_reasoning_template():
    mod = load_speed_ab_module()

    body = mod.build_chat_completion_body(
        model="qwen36",
        prompt="think",
        max_tokens=64,
        repeat_index=0,
        disable_prompt_reuse=False,
        enable_thinking="on",
    )

    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_completion_body_auto_omits_template_override():
    mod = load_speed_ab_module()

    body = mod.build_chat_completion_body(
        model="qwen36",
        prompt="use native config",
        max_tokens=64,
        repeat_index=0,
        disable_prompt_reuse=False,
        enable_thinking="auto",
    )

    assert "chat_template_kwargs" not in body


def test_real_logit_policy_sweep_uses_cached_pair_helper(monkeypatch):
    mod = load_chunk_probe_module()
    import vmlx_engine.native_mtp_research as research

    settings_a = research.ResearchSamplerSettings(temperature=1.0, top_p=0.95)
    settings_b = research.ResearchSamplerSettings(temperature=0.8, top_k=2)
    monkeypatch.setattr(
        mod,
        "_research_settings_grid",
        lambda: iter([(settings_a, settings_a), (settings_a, settings_b)]),
    )
    monkeypatch.setattr(mod, "_to_numpy_row", lambda row: np.asarray(row, dtype=np.float64))

    def uncached_metrics_should_not_be_called(*args, **kwargs):
        raise AssertionError("uncached speculative_policy_metrics was called")

    monkeypatch.setattr(
        research,
        "speculative_policy_metrics",
        uncached_metrics_should_not_be_called,
    )

    result = mod._real_logit_policy_sweep(
        target_rows=[np.log(np.array([0.55, 0.25, 0.15, 0.05]))],
        draft_rows=[np.log(np.array([0.50, 0.20, 0.20, 0.10]))],
        top_n=2,
    )

    assert result["grid_size"] == 2
    assert result["depth_count"] == 1
    assert len(result["top"]) == 2
    assert "top_stochastic" in result
    assert result["top"][0]["rank"] == 1
    assert result["top"][0]["delta"]["mode"] == "residual_exact_by_formula"


def test_real_logit_policy_sweep_saves_full_rows_only_when_requested(monkeypatch):
    mod = load_chunk_probe_module()
    import vmlx_engine.native_mtp_research as research

    settings_a = research.ResearchSamplerSettings(temperature=1.0, top_p=0.95)
    settings_b = research.ResearchSamplerSettings(temperature=0.8, top_k=2)
    monkeypatch.setattr(
        mod,
        "_research_settings_grid",
        lambda: iter([(settings_a, settings_a), (settings_a, settings_b)]),
    )
    monkeypatch.setattr(mod, "_to_numpy_row", lambda row: np.asarray(row, dtype=np.float64))

    compact = mod._real_logit_policy_sweep(
        target_rows=[np.log(np.array([0.55, 0.25, 0.15, 0.05]))],
        draft_rows=[np.log(np.array([0.50, 0.20, 0.20, 0.10]))],
        top_n=1,
        include_all=False,
    )
    full = mod._real_logit_policy_sweep(
        target_rows=[np.log(np.array([0.55, 0.25, 0.15, 0.05]))],
        draft_rows=[np.log(np.array([0.50, 0.20, 0.20, 0.10]))],
        top_n=1,
        include_all=True,
    )

    assert "rows" not in compact
    assert len(full["rows"]) == 2
    assert len(full["top"]) == 1

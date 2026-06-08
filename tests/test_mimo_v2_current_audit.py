from __future__ import annotations

import json
from pathlib import Path

from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_mimo_current_audit_separates_clean_artifact_from_runtime_blockers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "STALE_TARGETS", [])
    model_parent = tmp_path / "models" / "JANGQ-AI"
    model_path = model_parent / "MiMo-V2.5-JANGTQ_2"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    (tmp_path / "vmlx_engine/models").mkdir(parents=True)
    (tmp_path / "vmlx_engine/models/mllm.py").write_text(
        "\n".join(
            [
                "class _MiMoV2MediaConfig: pass",
                "class VisionConfig(_MiMoV2MediaConfig): pass",
                "class AudioConfig(_MiMoV2MediaConfig): pass",
                'VisionConfig.from_dict(params.get("vision_config") or {})',
                'AudioConfig.from_dict(params.get("audio_config") or {})',
                'processor_config=params.get("processor_config") or {}',
                "self.processor_config = dict(processor_config or {})",
                "self.media_token_ids = {",
                "self._mimo_v2_media_weights: dict[str, Any] = {}",
                "def _bind_mimo_v2_media_weight(self, key, value):",
                'key.startswith("visual.")',
                'key.startswith("audio_encoder.")',
                'key.startswith("speech_embeddings.")',
                "self._mimo_v2_media_weights[key] = value",
                "self._mimo_v2_media_weight_counts[group] += 1",
                "def _mimo_v2_replace_modal_embeddings(",
                "image_embeds is not None",
                "video_embeds is not None",
                "audio_embeds is not None",
                "token count",
                "MiMo-V2.5 JANG_2L vision input is not wired",
                'key.startswith("model.mtp.")',
            ]
        )
    )
    manifest = tmp_path / "build" / "current-mimo-http-tb5-manifest-20260606.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f"{(model_path / 'config.json').stat().st_size}\tMiMo-V2.5-JANGTQ_2/config.json\n"
    )

    _write_json(
        tmp_path / "build/current-mimo-jang2l-local-structural-verify-20260606.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path / "build/current-mimo-jang2l-live-text-cache-smoke-20260606.json",
        {
            "requests": [
                {
                    "response": {
                        "http_status": 200,
                        "json": {
                            "choices": [
                                {"message": {"content": "cache ok"}, "finish_reason": "stop"}
                            ]
                        },
                    }
                }
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json",
        {
            "got_shape": [2, 3, 4096],
            "manual_shape": [2, 3, 4096],
            "max_abs_diff": 0.0007,
            "mean_abs_diff": 0.00009,
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json",
        {
            "rows": [
                {"mode": "no_cache", "top10": [{"id": 1, "text": "A"}]},
                {"mode": "cache_prefill", "top10": [{"id": 1, "text": "A"}]},
            ]
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json",
        {"status": "fail", "cases": [{"status": "fail_corrupt"}]},
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json",
        {"status": "fail", "runtime_observations": [{"result": "fail_incomplete_marker"}]},
    )
    _write_json(
        tmp_path
        / "build/current-all-local-model-smoke-mimo-v25-jangtq2-after-runtime-repairs-20260607/summary.json",
        {
            "status": "fail",
            "results": [
                {
                    "status": "probe_failed",
                    "failures": [
                        {"label": "text_cache_repeat_1", "reason": "empty_visible"},
                    ],
                    "requests": [
                        {
                            "label": "tool_required",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "record_fact",
                                        "arguments": "{\"value\":\"blue-cat\"}",
                                    },
                                }
                            ],
                            "validation_failures": [],
                        }
                    ],
                    "cache_after": {
                        "body": {
                            "scheduler_stats": {
                                "batch_generator": {"generation_tps": 1.8}
                            }
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [
                {"manual_sink": False, "passes_length_ok": False},
                {"manual_sink": True, "passes_length_ok": False},
            ],
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json",
        {
            "status": "fail",
            "cases": [{"passes_length_ok": False, "contains_corrupt_repetition": True}],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-prompt-shape-sweep-20260606.json",
        {
            "status": "fail",
            "results": [
                {
                    "name": "long_cache_system_exact",
                    "content": "",
                    "finish_reason": "stop",
                    "prompt_tokens": 60,
                    "completion_tokens": 1,
                },
                {
                    "name": "long_cache_no_system_exact",
                    "content": "ACK",
                    "finish_reason": "stop",
                    "prompt_tokens": 66,
                    "completion_tokens": 2,
                },
                {
                    "name": "short_system_exact",
                    "content": "ACK",
                    "finish_reason": "stop",
                    "prompt_tokens": 23,
                    "completion_tokens": 2,
                },
            ],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-jang2l-rendered-prompt-compare-20260606.json",
        {
            "rows": [
                {
                    "name": "long_cache_system_exact",
                    "contains_empty_think_generation_prefix": True,
                    "rendered": "<|im_start|>assistant\n<think></think>",
                }
            ]
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-first-token-probe-registered-20260606.json",
        {
            "status": "pass",
            "rows": [
                {
                    "name": "failing_system_long",
                    "top": [
                        {
                            "token_id": 151645,
                            "text": "<|im_end|>",
                            "is_special": True,
                        },
                        {"token_id": 4032, "text": "ACK", "is_special": False},
                    ],
                },
                {
                    "name": "working_folded_long",
                    "top": [{"token_id": 4032, "text": "ACK", "is_special": False}],
                },
                {
                    "name": "working_short_system",
                    "top": [{"token_id": 4032, "text": "ACK", "is_special": False}],
                },
            ],
        },
    )
    _write_json(
        tmp_path / "build/current-mimo-v2-mllm-inputs-embeds-interface-fix-20260606.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path
        / "build/current-noheavy-api-cache-contract-after-qwen36-bundled-media-pass-20260607.json",
        {
            "status": "pass",
            "checks": {
                "responses_sampling_kwargs": True,
                "streaming_cache_detail_usage": True,
                "responses_previous_response_history": True,
                "cache_stats_reuse_skip_telemetry": True,
                "cache_reuse_endpoints": True,
                "request_output_caps_override_server_default": True,
                "prompt_context_caps_stay_separate_from_output_caps": True,
                "ollama_adapter_surface": True,
                "anthropic_bundle_defaults": True,
            },
            "missing_markers": [],
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json",
        {
            "status": "open",
            "summary": {
                "exact_repeat_1": True,
                "exact_repeat_2": True,
                "no_think_tags": True,
                "cache_hit_tokens": 33,
                "l2_tokens_on_disk": 33,
                "native_cache": {
                    "cache_type": "mixed_swa_kv",
                    "storage_quantization": {"enabled": True, "bits": 8},
                },
                "texts": {
                    "exact_repeat_1": "ACK-CB-742",
                    "exact_repeat_2": "ACK-CB-742",
                    "first_token_system_probe": "",
                },
                "finish_reasons": {
                    "exact_repeat_1": "stop",
                    "exact_repeat_2": "stop",
                    "first_token_system_probe": None,
                },
            },
            "rows": [
                {
                    "name": "exact_repeat_1",
                    "ok": True,
                    "elapsed_s": 22.0,
                    "usage": {"completion_tokens": 8},
                    "text": "ACK-CB-742",
                },
                {
                    "name": "exact_repeat_2",
                    "ok": True,
                    "elapsed_s": 5.0,
                    "usage": {"completion_tokens": 8},
                    "text": "ACK-CB-742",
                },
                {
                    "name": "first_token_system_probe",
                    "ok": False,
                    "error": "<HTTPError 503: 'Service Unavailable'>",
                    "text": "",
                },
            ],
            "log_tail": "Metal working-set pressure reject: 98.5% of 107.5GB",
        },
    )

    result = audit.build_audit(tmp_path, model_path, manifest)

    assert result["status"] == "open"
    assert result["local_release_clearance"] is False
    assert result["component_ok"]["manifest_integrity"] is True
    assert result["component_ok"]["text_cache_narrow"] is True
    assert result["component_ok"]["cache_vs_nocache_next_token"] is True
    assert result["component_ok"]["long_prompt_coherence"] is False
    assert result["component_ok"]["tool_protocol"] is True
    assert result["component_ok"]["exact_cache_prompt_following"] is True
    assert result["component_ok"]["decode_speed_target"] is False
    assert result["component_ok"]["system_prompt_first_token_stop"] is True
    assert result["component_ok"]["cb_system_prompt_working_set_pressure"] is False
    assert result["component_ok"]["source_vs_quant_first_divergence"] is False
    assert result["component_ok"]["api_cache_responses_contract"] is True
    assert result["component_ok"]["media_weights_preserved"] is False
    assert result["component_ok"]["media_runtime_capabilities_safe"] is False
    assert result["component_ok"]["media_model_metadata_text_only_contract"] is False
    assert result["component_ok"]["media_runtime_implementation"] is False
    assert result["component_ok"]["mimo_media_wired"] is False
    assert result["component_ok"]["manual_sink_does_not_clear_length_generation"] is True
    assert result["component_ok"]["disable_sink_does_not_clear_length_generation"] is True
    assert result["diagnostics"]["mimo_media_runtime"]["classification"] in {
        "runtime_implementation_gap_with_model_metadata_overadvertising",
        "runtime_implementation_gap",
    }
    assert result["diagnostics"]["mimo_media_runtime"]["config_parser_components"] == {
        "vision_config": True,
        "audio_config": True,
        "processor_config": True,
    }
    assert result["diagnostics"]["mimo_media_runtime"]["load_weights_bindings"] == {
        "visual": True,
        "audio": True,
    }
    assert result["diagnostics"]["mimo_media_runtime"]["mixed_inputs_embeds_splice"] is True
    assert (
        "VisionConfig parser"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "AudioConfig parser"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "visual.* load_weights binding"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio_encoder.* and speech_embeddings.* load_weights binding"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "mixed media/text inputs_embeds construction"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert result["diagnostics"]["all_local_smoke"]["tool_protocol_pass"] is True
    assert result["diagnostics"]["prompt_shape_first_token"]["blocked"] is True
    assert (
        result["diagnostics"]["prompt_shape_first_token"]["classification"]
        == "decode_loop_or_model_artifact_first_token_stop"
    )
    assert result["diagnostics"]["prompt_shape_first_token"]["failing_top_token"][
        "text"
    ] == "<|im_end|>"
    assert result["diagnostics"]["cb_native_thinking_off"][
        "cache_exact_pass"
    ] is True
    assert result["diagnostics"]["cb_native_thinking_off"][
        "prefix_paged_l2_cache_reproved"
    ] is True
    assert result["diagnostics"]["cb_native_thinking_off"][
        "memory_pressure_blocked"
    ] is True
    assert result["diagnostics"]["api_cache_responses_contract"]["pass"] is True
    assert (
        "No-heavy source contract only"
        in result["diagnostics"]["api_cache_responses_contract"]["boundary"]
    )
    assert result["diagnostics"]["manual_sink_sdpa_clears_length_generation"] is False
    assert result["diagnostics"]["disable_sink_clears_length_generation"] is False
    assert "sink-kernel-only" in result["diagnostics"]["sink_boundary"]
    assert result["blockers"] == [
        "mimo_long_prompt_coherence_blocked",
        "mimo_decode_speed_below_release_target",
        "mimo_cb_system_prompt_working_set_pressure_blocked",
        "mimo_source_vs_quant_first_divergence_missing_or_failed",
        "mimo_media_runtime_implementation_missing",
        "mimo_model_metadata_overadvertises_unwired_media",
        "mimo_runtime_capabilities_media_status_missing_or_unsafe",
    ]


def test_mimo_tool_protocol_is_separate_from_literal_argument_exactness(tmp_path):
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "tool_required",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "record_fact",
                                    "arguments": "{\"value\":\"blue cat\"}",
                                },
                            }
                        ],
                        "validation_failures": [
                            {
                                "label": "tool_required",
                                "reason": "expected_tool_argument_missing",
                                "expected": {"value": "blue-cat"},
                                "actual": {"value": "blue cat"},
                            }
                        ],
                    },
                    {
                        "label": "structured_json_exact",
                        "tool_calls": [],
                        "validation_failures": [
                            {
                                "label": "structured_json_exact",
                                "reason": "json_exact_object_mismatch",
                            }
                        ],
                    },
                    {
                        "label": "mimo_tool_required_sentinel",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "record_fact",
                                    "arguments": "{\"value\":\"B7CAT-09\"}",
                                },
                            }
                        ],
                        "validation_failures": [
                            {
                                "label": "mimo_tool_required_sentinel",
                                "reason": "expected_tool_argument_missing",
                                "expected": {"value": "B7-CAT-09"},
                                "actual": {"value": "B7CAT-09"},
                            }
                        ],
                    },
                    {
                        "label": "mimo_structured_json_sentinel",
                        "tool_calls": [],
                        "validation_failures": [
                            {
                                "label": "mimo_structured_json_sentinel",
                                "reason": "json_exact_object_mismatch",
                            }
                        ],
                    },
                ],
                "failures": [
                    {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                    },
                    {
                        "label": "structured_json_exact",
                        "reason": "json_exact_object_mismatch",
                    },
                    {
                        "label": "mimo_tool_required_sentinel",
                        "reason": "expected_tool_argument_missing",
                    },
                    {
                        "label": "mimo_structured_json_sentinel",
                        "reason": "json_exact_object_mismatch",
                    },
                ],
                "cache_after": {
                    "body": {
                        "scheduler_stats": {
                            "batch_generator": {"generation_tps": 42.0}
                        }
                    }
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["tool_protocol_pass"] is True
    assert evidence["artifact_exactness_pass"] is False
    assert evidence["artifact_exactness_failures"] == smoke["results"][0]["failures"]
    assert evidence["artifact_exactness_boundary"]["classification"] == (
        "model_generated_literal_mutation_after_valid_parser_structure"
    )
    assert (
        evidence["artifact_exactness_boundary"][
            "parser_structure_valid_for_failed_rows"
        ]
        is True
    )
    assert (
        evidence["artifact_exactness_boundary"]["model_generated_literal_mutation"]
        is True
    )
    assert "tool_required" in evidence["artifact_exactness_boundary"]["failed_labels"]


def test_mimo_current_smoke_tool_protocol_beats_legacy_synced_tool_failure():
    source = Path("tests/cross_matrix/run_mimo_v2_jang2l_current_audit.py").read_text()
    assert source.count('if not smoke_evidence.get("exists"):') >= 2
    assert (
        "tool_blocked = tool_blocked or bool(\n"
        "                synced_evidence.get(\"tool_protocol_blocked\")\n"
        "            )"
    ) in source
    assert (
        "tool_blocked = bool(cb_evidence.get(\"tool_protocol_blocked\"))"
        in source
    )

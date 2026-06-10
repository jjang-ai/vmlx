from __future__ import annotations

import json
from pathlib import Path

from tests.cross_matrix import run_full_release_objective_checklist as checklist

_build = checklist._build


def test_full_release_objective_checklist_uses_current_noheavy_api_cache_contract():
    assert checklist.NOHEAVY_API_CACHE == Path(
        "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json"
    )


def test_full_release_objective_checklist_uses_current_responses_raw_sse_parity_contract():
    assert checklist.RESPONSES_RAW_SSE_PARITY == Path(
        "build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-12b-mxfp8-crack-20260610.json"
    )


def test_full_release_objective_checklist_uses_current_qwen35_raw_sse_parity_contract():
    assert checklist.QWEN35_RAW_SSE_PARITY == Path(
        "build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json"
    )


def test_full_release_objective_checklist_uses_current_gemma4_12b_issue191_startup_proof():
    assert checklist.GEMMA4_12B_ISSUE191_STARTUP_VISIBLE == Path(
        "build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json"
    )


def test_full_release_objective_checklist_uses_current_gemma4_12b_jang4m_nomedia_proof():
    assert checklist.GEMMA4_12B_JANG4M_SMOKE == Path(
        "build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-code-column-prompt-20260610/JANGQ_gemma-4-12B-it-JANG_4M/result.json"
    )


def test_full_release_objective_checklist_uses_current_gemma4_12b_autoq4_cache_proof():
    assert checklist.GEMMA4_12B_JANG4M_AUTOQ4_CACHE == Path(
        "build/current-gemma4-12b-jang4m-autoq4-mixed-swa-cache-live-20260610.json"
    )


def test_full_release_objective_checklist_uses_current_gemma4_12b_jang4m_media_proof():
    assert checklist.GEMMA4_12B_JANG4M_MEDIA_SMOKE == Path(
        "build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json"
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_green_api_surface_artifact(tmp_path: Path) -> None:
    _write_json(
        tmp_path / checklist.API_SURFACE_CONTRACT,
        {
            "status": "pass",
            "checks": {
                "openai_chat_completions_sampling_defaults": True,
                "openai_responses_sampling_defaults": True,
                "legacy_completions_output_caps_override_server_default": True,
                "chat_and_responses_output_caps_override_server_default": True,
                "prompt_context_caps_stay_separate_from_output_caps": True,
                "anthropic_adapter_bundle_defaults": True,
                "ollama_adapter_streaming_done_behavior": True,
                "chat_and_responses_streaming_cache_detail_usage": True,
                "responses_previous_response_history": True,
                "server_cache_and_tool_surfaces_named": True,
                "plain_attention_kv_status": True,
                "dsv4_native_cache_status": True,
                "zaya_typed_cca_status": True,
                "hybrid_ssm_partial_reuse": True,
                "turboquant_kv_runtime_contract": True,
                "turboquant_disk_roundtrip": True,
                "no_generic_tq_on_hybrid_ssm": True,
                "panel_request_builder_sampling_and_output_overrides": True,
                "panel_ollama_gateway_omits_disabled_sentinels": True,
                "panel_gateway_single_model_auto_switch_streaming": True,
                "panel_gateway_single_model_auto_switch_cache_endpoints": True,
                "panel_chat_override_policy_preserves_explicit_values": True,
                "panel_gateway_streaming_disconnect_epipe_guard": True,
                "panel_child_process_stdio_epipe_guard": True,
                "panel_backend_stderr_split_epipe_guard": True,
                "panel_gateway_ollama_proxy_response_error_guard": True,
                "panel_ipc_backend_request_epipe_guard": True,
                "panel_performance_health_epipe_guard": True,
                "panel_session_lifecycle_epipe_guard": True,
                "all_required_panel_api_markers_present": True,
            },
        },
    )


def _write_green_responses_raw_sse_artifact(tmp_path: Path) -> None:
    _write_json(
        tmp_path / checklist.RESPONSES_RAW_SSE_PARITY,
        {
            "status": "pass",
            "missing_captures": [],
            "checks": {
                "direct_capture_present": True,
                "gateway_capture_present": True,
                "tunnel_capture_present": True,
                "all_required_surfaces_present": True,
                "all_present_surfaces_parse_cleanly": True,
                "all_present_surfaces_match_expected_function_name": True,
                "all_present_surfaces_match_expected_arguments": True,
                "all_present_surfaces_have_authoritative_args": True,
                "authoritative_arguments_match_across_present_surfaces": True,
                "tunnel_expected_model_advertised": True,
                "local_responses_streaming_guards_pass": True,
                "local_empty_xml_arguments_fail_closed": True,
                "local_output_index_ordering_guard": True,
                "gateway_argument_stream_passthrough_guard": True,
                "responses_previous_response_history_guard": True,
                "all_present_surfaces_have_valid_output_item_indices": True,
                "all_present_surfaces_have_required_reasoning": True,
                "no_reasoning_disable_workaround": True,
            },
        },
    )


def _native_cache() -> dict:
    return {
        "schema": "hybrid_ssm_v1",
        "cache_type": "hybrid_ssm_typed",
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
        "generic_turboquant_kv": {"enabled": True},
        "attention_kv_storage_quantization": {"enabled": True, "bits": 4},
    }


def _mtp(depth: int = 2) -> dict:
    return {
        "runtime_active": True,
        "runtime_available": True,
        "runtime_supported": True,
        "index_has_mtp_tensors": True,
        "status": "native_runtime_active",
        "effective_depth": depth,
    }


def _mixed_swa_native(family: str, subtype: str = "mixed_swa_kv") -> dict:
    return {
        "family": family,
        "schema": "mixed_swa_kv_v1",
        "cache_type": "mixed_swa_kv",
        "cache_subtype": subtype,
        "generic_turboquant_kv": {"enabled": False},
        "storage_quantization": {"enabled": True, "bits": 4},
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }


def _hybrid_ssm_native(family: str) -> dict:
    return {
        "family": family,
        "schema": "hybrid_ssm_v1",
        "cache_type": "hybrid_ssm_typed",
        "generic_turboquant_kv": {"enabled": False},
        "attention_kv_storage_quantization": {"enabled": True, "bits": 4},
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
        "ssm_entries": 3,
    }


def _smoke_artifact(
    *,
    mllm: bool,
    tool_parser: str,
    reasoning_parser: str,
    cache_detail: str,
    native: dict,
    hybrid: bool = False,
) -> dict:
    scheduler_cache = {}
    totals = {"l2_block_tokens_on_disk": 256}
    block_disk = {"disk_writes": 3, "disk_hits": 2 if hybrid else 0}
    if hybrid:
        scheduler_cache["ssm_companion_cache"] = {
            "disk": {"stores": 3, "total_tokens_on_disk": 256}
        }
        totals["l2_ssm_tokens_on_disk"] = 256
    return {
        "status": "pass",
        "failures": [],
        "row": {"is_mllm": mllm},
        "capabilities": {
            "body": {
                "tool_parser": tool_parser,
                "reasoning_parser": reasoning_parser,
                "supports_tools": True,
            }
        },
        "requests": [
            {
                "validation_failures": [],
                "usage": {"prompt_tokens_details": None},
                "cache_summary": {"has_cache_hit": False},
                "tool_calls": [],
            },
            {
                "validation_failures": [],
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 64,
                        "cache_detail": cache_detail,
                    }
                },
                "cache_summary": {"has_cache_hit": True},
                "tool_calls": [],
            },
            {
                "validation_failures": [],
                "usage": {"prompt_tokens_details": None},
                "cache_summary": {"has_cache_hit": True},
                "tool_calls": [
                    {
                        "function": {
                            "name": "record_fact",
                            "arguments": '{"value": "blue-cat"}',
                        }
                    }
                ],
            },
        ],
        "cache_after": {
            "body": {
                "native_cache": native,
                "block_disk_cache": block_disk,
                "scheduler_cache": scheduler_cache,
                "cache_totals": totals,
            }
        },
    }


def _write_green_family_smokes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / checklist.GEMMA4_12B_ISSUE191_STARTUP_VISIBLE,
        {
            "status": "pass",
            "model": "/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
            "checks": {
                "import_alias_ok": True,
                "startup_health_ok": True,
                "visible_generation_ok": True,
                "post_chat_health_ok": True,
            },
            "chat_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "GEMMA4-OK", "role": "assistant"},
                    }
                ],
                "usage": {"completion_tokens": 10},
            },
        },
    )
    _write_json(
        tmp_path / checklist.GEMMA4_12B_JANG4M_SMOKE,
        _smoke_artifact(
            mllm=True,
            tool_parser="gemma4",
            reasoning_parser="gemma4",
            cache_detail="paged+mixed_swa",
            native=_mixed_swa_native("gemma4"),
        ),
    )
    _write_json(
        tmp_path
        / "build/current-gemma4-12b-jang4m-autoq4-mixed-swa-cache-live-20260610.json",
        {
            "status": "pass",
            "checks": {"cache_second_hit": True},
            "native_cache": _mixed_swa_native("gemma4"),
            "second": {
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": 20,
                        "cache_detail": "paged+mixed_swa",
                    }
                }
            },
            "cache_after": {
                "body": {
                    "block_disk_cache": {"disk_writes": 1},
                    "cache_totals": {"l2_block_tokens_on_disk": 20},
                }
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json",
        {
            "status": "pass",
            "checks": {"all_rows_passed": True, "at_least_one_row_ran": True},
            "rows": {
                "jang4m_image": {
                    "status": "pass",
                    "failures": [],
                    "chat": {"code": 200, "content": "Red"},
                    "checks": {
                        "vision_advertised": True,
                        "red_detected": True,
                        "no_channel_leak": True,
                    },
                }
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json",
        _smoke_artifact(
            mllm=False,
            tool_parser="step3p5",
            reasoning_parser="qwen3",
            cache_detail="paged",
            native=_mixed_swa_native("step3p7", "step3p7_full_sliding_kv"),
        ),
    )
    step_progress = {
        "config_layer_available": True,
        "model_layer_available": True,
        "vision_layer_available": True,
        "projector_layer_available": True,
        "processor_layer_available": True,
        "image_embeds_merge_available": True,
        "pixel_values_processor_available": True,
        "vision_patch_embed_available": True,
        "vision_abs_posemb_resize_available": True,
        "vision_transformer_blocks_available": True,
        "vision_downsamplers_available": True,
        "pixel_values_to_projector_available": True,
        "placeholder_projected_token_count_available": True,
        "release_clearance": "source_runtime_surface_present_needs_live_proof",
    }
    _write_json(
        tmp_path / checklist.STEP37_VLM_RUNTIME_AUDIT,
        {
            "status": "pass",
            "mlx_vlm_step3p7_runtime_available": True,
            "release_clearance": "audit_does_not_block_release",
            "mlx_vlm_step3p7_runtime": {
                "has_model": True,
                "has_vision_model": True,
                "has_language_model": True,
                "has_model_config": True,
                "has_processor": True,
            },
            "source_owned_runtime_progress": step_progress,
            "mlx_vlm_implementation_contract": {
                "explicitly_rejected_clearance_paths": [
                    "text_only_bridge",
                    "importable_stub",
                ]
            },
        },
    )
    for rel in [
        "build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json",
        "build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json",
    ]:
        _write_json(
            tmp_path / rel,
            _smoke_artifact(
                mllm=False,
                tool_parser="lfm2",
                reasoning_parser="qwen3",
                cache_detail="paged+ssm",
                native=_hybrid_ssm_native("lfm2"),
                hybrid=True,
            ),
        )
    _write_json(
        tmp_path
        / "build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-after-reasoning-budget-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json",
        _smoke_artifact(
            mllm=True,
            tool_parser="nemotron",
            reasoning_parser="deepseek_r1",
            cache_detail="paged+ssm",
            native=_hybrid_ssm_native("nemotron_h"),
            hybrid=True,
        ),
    )
    _write_json(
        tmp_path / "build/current-nemotron-omni-mxfp4-media-gate-20260607/SUMMARY.json",
        {
            "status": "PASS",
            "passed": True,
            "requests": [
                {"name": "turn1_memory", "code": 200, "content": "noted"},
                {"name": "image_blue", "code": 200, "content": "Blue"},
                {"name": "video_blue", "code": 200, "content": "Blue"},
                {"name": "audio_blue", "code": 200, "content": "Blue."},
                {
                    "name": "turn2_recall",
                    "code": 200,
                    "content": "color=blue, animal=cat",
                    "usage": {
                        "prompt_tokens_details": {
                            "cached_tokens": 24,
                            "cache_detail": "paged+ssm",
                        }
                    },
                },
            ],
            "checks": [
                {"name": "capabilities_available", "ok": True},
                {"name": "cache_stats_initial_available", "ok": True},
                {"name": "turn1_memory_http_ok", "ok": True},
                {"name": "turn1_noted_exact_or_contains", "ok": True},
                {"name": "image_blue_http_ok", "ok": True},
                {"name": "image_blue_mentions_blue", "ok": True},
                {"name": "video_blue_http_ok", "ok": True},
                {"name": "video_blue_mentions_blue", "ok": True},
                {"name": "audio_blue_http_ok", "ok": True},
                {"name": "audio_blue_mentions_blue", "ok": True},
                {"name": "turn2_recall_http_ok", "ok": True},
                {"name": "turn2_recall_color_animal", "ok": True},
                {"name": "cache_stats_final_available", "ok": True},
            ],
            "cache_final": {
                "scheduler_stats": {"cache_hit_tokens": 24},
                "native_cache": _hybrid_ssm_native("nemotron_h"),
                "block_disk_cache": {"disk_writes": 2, "disk_hits": 3},
                "cache_totals": {
                    "l2_block_tokens_on_disk": 73,
                    "l2_ssm_tokens_on_disk": 73,
                },
                "ssm_companion": {
                    "disk": {"stores": 2, "total_tokens_on_disk": 73}
                },
            },
            "log_tail": 'POST /v1/chat/completions HTTP/1.1" 200 OK',
        },
    )


def _write_green_panel_settings_artifact(tmp_path: Path) -> None:
    checks = {
        "panel_settings_contract_count": True,
        "dsv4_default_native_prefix_on": True,
        "dsv4_explicit_prefix_off_disables_native_flags": True,
        "dsv4_l2_explicit_off_preserves_prefix": True,
        "dsv4_generic_kv_flags_suppressed": True,
        "dsv4_pool_quant_controls_are_dsv4_only": True,
        "max_output_context_cli_split": True,
        "chat_max_output_is_per_chat_override": True,
        "non_dsv4_cache_toggles_preserved": True,
        "i18n_max_output_context_copy": True,
        "lazy_mmap_launch_memory_admission": True,
        "minimax_reasoning_parser_detection": True,
        "native_mtp_d3_default_policy": True,
        "model_family_parser_registry": True,
        "engine_cache_architecture_registry": True,
        "panel_emitted_flags_are_registered_engine_cli_flags": True,
        "panel_typecheck": True,
    }
    _write_json(
        tmp_path
        / "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json",
        {
            "status": "pass",
            "checks": checks,
            "missing_source_markers": [],
            "commands": {
                "panel_settings_contracts": {
                    "returncode": 0,
                    "counts": {"test_files_passed": 6, "tests_passed": 314},
                }
            },
        },
    )


def _write_qwen_green_artifacts(tmp_path: Path) -> None:
    _write_json(
        tmp_path
        / "build/current-qwen27-mxfp4-mtp-responses-cancel-mtp-deterministic-20260609.json",
        {
            "status": "pass",
            "request": {"stream": True},
            "probe": {"cancel_route_present": True, "bad_text_captured": False},
            "raw": {"response_id": "resp_ok", "cancel_status": 200, "stream_error": None},
            "health_after": {"mtp": _mtp(), "native_cache": _native_cache()},
        },
    )
    _write_json(
        tmp_path / "build/current-qwen27-mxfp4-mtp-api-parity-20260607/summary.json",
        {
            "checks": {
                "responses_text": {"code": 200, "text_head": "ACK"},
                "responses_tool_required": {"code": 200, "has_record_fact": True},
                "anthropic_messages": {"code": 200, "text_head": "ACK"},
                "ollama_chat": {"code": 200, "text_head": "ACK"},
                "chat_stream_sse": {"has_ack": True},
            },
            "health_after": {
                "native_cache": _native_cache(),
                "mtp": _mtp(),
                "scheduler": {"cache_hit_tokens": 36},
                "cache": {
                    "block_disk_cache": {"disk_writes": 4, "disk_hits": 12},
                    "ssm_companion": {
                        "disk": {"stores": 7, "total_tokens_on_disk": 277}
                    },
                },
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260609/summary.json",
        {
            "status": "pass",
            "phases": {
                "phase1": {
                    "block_disk_cache": {"disk_writes": 1},
                    "ssm_companion_disk": {"stores": 1},
                },
                "phase2": {
                    "cache_hit_tokens": 63,
                    "usage": {
                        "prompt_tokens_details": {"cache_detail": "paged+ssm+disk"}
                    },
                    "block_disk_cache": {"disk_hits": 1},
                    "native_cache": _native_cache(),
                    "mtp": _mtp(),
                },
            },
        },
    )
    video_surfaces = [
        "installed_app_ui",
        "video_where_supported",
        "reasoning_display",
        "responses_api",
        "responses_delta_streaming",
        "responses_cache_detail_usage",
        "tool_l2_cache_integrated",
        "native_cache_status",
        "l2_disk_storage",
        "parser_leak_check",
        "language_leak_check",
    ]
    _write_json(
        tmp_path
        / "docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json",
        {
            "status": "pass",
            "uiLaunchMode": "installed-app",
            "installedAppPath": "/Applications/vMLX.app",
            "modelPath": "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
            "requestedVideo": True,
            "visibleAssistantTurnsComplete": True,
            "provenSurfaces": video_surfaces,
            "server": {
                "health": {
                    "model_loaded": True,
                    "native_cache": _native_cache(),
                    "mtp": _mtp(depth=3),
                    "scheduler": {
                        "last_cache_execution": {
                            "cached_tokens": 2807,
                            "cache_detail": "paged+ssm",
                        },
                        "batch_generator": {"num_images_processed": 6},
                    },
                    "cache": {
                        "block_disk_cache": {"disk_writes": 73},
                        "ssm_companion": {"disk": {"stores": 9}},
                    },
                }
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json",
        {
            "status": "pass",
            "checks": {
                "cold_visible_tail_markers": True,
                "warm_visible_tail_markers": True,
            },
            "phases": {
                "cold": {
                    "response": {"usage": {"input_tokens": 31647}},
                },
                "warm": {
                    "response": {"usage": {"input_tokens": 31647}},
                    "health": {
                        "scheduler": {
                            "last_cache_execution": {
                                "cached_tokens": 31646,
                                "cache_detail": "paged+ssm+disk",
                            }
                        },
                        "native_cache": _native_cache(),
                        "mtp": _mtp(depth=3),
                        "turboquant_kv_cache": {"enabled": True},
                    },
                    "cache_stats": {
                        "block_disk_cache": {
                            "disk_writes": 495,
                            "disk_hits": 1485,
                            "total_tokens_on_disk": 31646,
                        },
                        "ssm_companion": {
                            "disk": {
                                "stores": 2,
                                "total_tokens_on_disk": 63262,
                            }
                        },
                    },
                },
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/00_startup.json",
        {
            "health": {
                "model_loaded": True,
                "model_name": "JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
                "native_cache": _native_cache(),
                "mtp": _mtp(depth=3),
                "routing": {"trained_active_experts": 8, "effective_active_experts": 8},
            }
        },
    )
    row = {
        "cached_tokens": 128,
        "scheduler_stats": {"last_cache_execution": {"cache_detail": "paged+ssm"}},
        "block_disk_cache": {"disk_hits": 2, "disk_writes": 2},
        "ssm_companion": {"disk": {"total_tokens_on_disk": 100}},
    }
    _write_json(
        tmp_path
        / "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-historical-tool-required-20260607/SUMMARY.json",
        {
            "overall_pass": True,
            "turns": 3,
            "rows": [{}, row, row],
            "acceptance": {
                "previous_response_id_used": True,
                "cache_reuse_each_turn_after_first": True,
                "tool_call_each_required_turn": True,
                "tool_evidence_each_required_turn": True,
                "no_tool_markup_leak": True,
                "no_loop_like_tail": True,
                "final_turn_tools_disabled": True,
                "final_turn_visible_output": True,
            },
        },
    )
    _write_json(
        tmp_path
        / "docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json",
        {
            "status": "pass",
            "uiLaunchMode": "installed-app",
            "installedAppPath": "/Applications/vMLX.app",
            "modelPath": "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
            "requestedVideo": True,
            "visibleAssistantTurnsComplete": True,
            "provenSurfaces": video_surfaces,
            "server": {
                "health": {
                    "model_loaded": True,
                    "native_cache": _native_cache(),
                    "mtp": _mtp(depth=3),
                    "scheduler": {
                        "last_cache_execution": {
                            "cached_tokens": 2807,
                            "cache_detail": "paged+ssm",
                        },
                        "batch_generator": {"num_images_processed": 6},
                    },
                    "cache": {
                        "block_disk_cache": {"disk_writes": 73},
                        "ssm_companion": {"disk": {"stores": 9}},
                    },
                }
            },
        },
    )
    _write_json(
        tmp_path
        / "build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json",
        {
            "status": "pass",
            "phases": {
                "phase1": {
                    "cache_stats": {
                        "block_disk_cache": {"disk_writes": 27},
                        "ssm_companion": {
                            "disk": {"stores": 2, "total_tokens_on_disk": 3359}
                        },
                    }
                },
                "phase2": {
                    "response": {
                        "usage": {
                            "prompt_tokens_details": {
                                "cached_tokens": 1695,
                                "cache_detail": "paged+ssm+disk",
                            }
                        }
                    },
                    "cache_stats": {
                        "block_disk_cache": {"disk_hits": 27},
                        "ssm_companion": {"disk": {"hits": 1}},
                    },
                    "health": {
                        "native_cache": _native_cache(),
                        "mtp": _mtp(depth=3),
                    },
                },
            },
        },
    )
    _write_json(
        tmp_path / checklist.QWEN35_RAW_SSE_PARITY,
        {
            "status": "pass",
            "missing_captures": [],
            "checks": {
                "direct_capture_present": True,
                "gateway_capture_present": True,
                "tunnel_capture_present": True,
                "all_required_surfaces_present": True,
                "all_present_surfaces_parse_cleanly": True,
                "all_present_surfaces_match_expected_function_name": True,
                "authoritative_arguments_match_across_present_surfaces": True,
                "tunnel_expected_model_advertised": True,
                "gateway_argument_stream_passthrough_guard": True,
                "responses_previous_response_history_guard": True,
                "all_present_surfaces_same_model": True,
                "all_present_surfaces_match_expected_model": True,
                "all_present_surfaces_have_authoritative_args": True,
                "all_present_surfaces_match_expected_arguments": True,
                "all_present_surfaces_have_required_reasoning": True,
                "no_reasoning_disable_workaround": True,
                "all_present_surfaces_have_valid_output_item_indices": True,
                "local_responses_streaming_guards_pass": True,
                "local_output_index_ordering_guard": True,
                "local_empty_xml_arguments_fail_closed": True,
            },
        },
    )


def _write_issue179_green_artifact(tmp_path: Path) -> None:
    _write_json(
        tmp_path
        / "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json",
        {
            "status": "pass",
            "proven": {
                "current_source_responses_cancel_contract_proven": True,
                "current_source_responses_cancel_inactive_404_contract_proven": True,
                "latest_public_dmg_has_responses_cancel_route": True,
                "local_installed_bundle_has_responses_cancel_route": True,
                "local_installed_responses_cancel_live_probe": True,
                "local_real_ui_diagnostics_clean": True,
                "local_installed_issue179_session_settings_parity": True,
                "local_reporter_prompt_reproduction_clean": True,
                "reporter_log_has_abort_before_visible_content": True,
            },
            "local_responses_cancel_probe": {
                "status": "pass",
                "response_id_seen": True,
                "cancel_status": 200,
                "probe": {
                    "cancel_route_present": True,
                    "bad_text_captured": False,
                },
            },
            "local_real_ui": {
                "all_required_clean": True,
                "clean_count": 5,
                "required_count": 5,
                "installed_session_settings_parity": {
                    "all_installed_use_responses": True,
                },
            },
            "reporter_parity_comparison": {"status": "pass", "failures": []},
            "reporter_server_hash_parity": {"status": "pass"},
            "local_reporter_prompt_reproduction": {"clean": True},
            "current_source_minimax_small_smoke": {
                "path": "build/current-all-local-model-smoke-minimax-small-jangtq-cache-language-after-bare-invoke-tool-20260609/summary.json",
                "status": "pass",
                "all_checks_pass": True,
                "checks": {
                    "status_pass": True,
                    "model_family_minimax": True,
                    "tool_parser_minimax": True,
                    "reasoning_parser_minimax_m2": True,
                    "reasoning_separated": True,
                    "required_tool_call_parsed": True,
                    "tool_result_continuation_exact": True,
                    "structured_json_exact": True,
                    "exact_code_whitespace": True,
                    "cache_second_hit_tq": True,
                    "block_disk_l2_restart_restore": True,
                    "native_cache_reports_tq_l2": True,
                },
                "release_boundary": "current-source boundary proof only",
            },
        },
    )


def _write_dsv4_green_artifact(tmp_path: Path) -> None:
    selected_cases = [
        "chat_off",
        "chat_off_rep1",
        "chat_off_no_punct_rep1",
        "chat_off_bundle_defaults",
        "chat_on",
        "chat_on_rep1",
        "chat_max",
        "responses_off",
        "responses_off_rep1",
        "responses_off_no_punct_rep1",
        "responses_off_bundle_defaults",
        "responses_on",
        "responses_on_rep1",
        "legacy_completion_raw",
    ]
    _write_json(
        tmp_path
        / "build/current-dsv4-route-mode-code-exactness-preflight-after-mimo-classifier-refresh-20260608.json",
        {
            "status": "pass",
            "reason": None,
            "floor_valid": True,
            "conservative_floor_valid": True,
            "active_heavy_process_count": 0,
            "active_heavy_processes": [],
            "launch_allowed": True,
            "available_for_gate_gb": 130.0,
            "required_available_gb": 120.0,
            "did_not_launch": False,
            "launch_decision": "launch_allowed",
            "launch_blockers": [],
            "case_count": len(selected_cases),
            "selected_cases": selected_cases,
        },
    )


def _write_green_n2_objective_digest(tmp_path: Path) -> None:
    assert checklist.OBJECTIVE_DIGEST == Path(
        "build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json"
    )
    _write_json(
        tmp_path / checklist.OBJECTIVE_DIGEST,
        {
            "requirements": [
                {
                    "requirement": (
                        "N2 Pro 397B JANG1L/JANGTQ runtime/cache/API/UI quality "
                        "is release-cleared"
                    ),
                    "status": "pass",
                    "evidence": [
                        "build/current-n2-pro-397b-jang1l-jangtq-live-release-proof.json"
                    ],
                    "details": {
                        "local_artifact_probe": {
                            "artifact_present": True,
                        },
                        "required_next_evidence": [],
                    },
                }
            ]
        },
    )


def test_full_release_objective_checklist_keeps_open_rows_visible(tmp_path):
    _write_json(
        tmp_path / checklist.RELEASE_MANIFEST,
        {
            "status": "open",
            "prepackage_ready": False,
            "release_ready": False,
            "current_proof_sweep": {
                "component_ok": {
                    "packaged_integrity_matrix": True,
                    "real_ui_full_model_matrix": False,
                    "real_ui_live_model_proof": False,
                }
            },
        },
    )
    _write_json(
        tmp_path / checklist.NOHEAVY_API_CACHE,
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
                "json_response_format_stream_validation": True,
                "responses_text_format_json_schema_preserved": True,
                "anthropic_bundle_defaults": True,
                "ollama_adapter_surface": True,
            },
        },
    )
    _write_green_api_surface_artifact(tmp_path)
    _write_json(
        tmp_path / checklist.TOOL_CALL_CONTRACT,
        {
            "status": "pass",
            "checks": {
                "tool_parser_residue_rejected_instead_of_executed": True,
                "schema_valid_dsml_tool_call_preserved": True,
                "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid": True,
                "dsv4_tool_preamble_suppressed_and_not_stored_without_call": True,
                "tool_choice_none_does_not_fallback_to_raw_dsml": True,
                "family_tool_parser_matrix_covers_no_leak_and_alias_edges": True,
                "panel_max_tool_iterations_caps_tool_loops": True,
                "all_required_tool_call_markers_present": True,
            },
        },
    )
    _write_green_panel_settings_artifact(tmp_path)
    _write_json(
        tmp_path / checklist.MIMO_AUDIT,
        {
            "status": "open",
            "local_release_clearance": False,
            "blockers": ["mimo_tool_protocol_blocked"],
            "component_ok": {
                "manifest_integrity": True,
                "decode_speed_target": True,
                "api_cache_responses_contract": True,
                "prefix_paged_l2_cache_reproved": True,
                "long_prompt_coherence": False,
                "tool_protocol": False,
                "artifact_exactness": False,
                "cb_system_prompt_working_set_pressure": False,
                "source_vs_quant_first_divergence": False,
                "source_vs_quant_requirement_satisfied": True,
                "media_weights_preserved": True,
                "media_runtime_capabilities_safe": True,
                "media_model_metadata_text_only_contract": False,
                "media_runtime_implementation": False,
                "mimo_media_wired": False,
            },
            "diagnostics": {
                "all_local_smoke": {
                    "artifact_exactness_boundary": {
                        "classification": "model_generated_literal_mutation_after_valid_parser_structure",
                        "failed_labels": ["tool_required"],
                    }
                },
                "jang2l_all_local_smoke": {
                    "artifact_exactness_boundary": {
                        "classification": "empty_visible_output_after_generation_stop",
                        "failed_labels": ["mimo_structured_json_sentinel"],
                    }
                },
                "prompt_shape_first_token": {
                    "classification": "decode_loop_or_model_artifact_first_token_stop",
                    "failing_top_token": {
                        "text": "<|im_end|>",
                        "is_special": True,
                    },
                    "working_folded_top_token": {
                        "text": "ACK",
                        "is_special": False,
                    },
                },
            },
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER,
        {
            "status": "open",
            "classification": "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer",
            "model_upload_action_required": True,
            "model_upload_action_reasons": [
                "JANGTQ_2 served artifact emits wrong compact-hyphen/sentinel literals as greedy top-1; rebuild/reupload with corrected quantization unless a runtime decode bug is proven"
            ],
            "source_vs_quant_load_performed": False,
            "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
            "excluded_surfaces": {
                "chat_template_only_primary_cause": True,
                "parser_argument_rewrite": True,
                "prefix_paged_l2_or_kv_quant_primary_cause": True,
                "hidden_stochastic_sampling_primary_cause": True,
                "tokenizer_roundtrip_primary_cause": True,
            },
            "unresolved_surfaces": {
                "artifact_quantization_or_decode_logits_quality": True,
                "jangtq2_compact_hyphen_decode_quality": True,
                "source_vs_quant_first_divergence": True,
            },
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE,
        {
            "status": "open",
            "proven": {
                "api_routes_mllm": True,
                "loader_overlay_auto_enabled": True,
                "preserved_media_weights_bound": True,
                "live_chat_image_200": True,
                "prior_unsupported_media_400_cleared_for_source_image": True,
            },
            "not_proven": {
                "mimo_exactness": True,
                "release_clearance": True,
            },
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE,
        {
            "status": "open",
            "proven": {
                "source_server_loads_as_mllm": True,
                "media_weights_bound": True,
                "video_request_reaches_runtime": True,
                "video_http_200": True,
                "audio_request_reaches_runtime": True,
                "audio_http_200": True,
            },
            "not_proven": {
                "video_semantic_correctness": True,
                "solid_color_image_semantic_correctness": True,
                "release_clearance": True,
            },
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_RESPONSES_TOOLS_CACHE_UI,
        {
            "status": "pass",
            "classification": "dev_app_responses_tools_cache_green_exactness_still_bounded",
            "cache": {
                "nativeCacheSubtype": "mimo_v2_asymmetric_swa",
                "cacheHitTokens": 4548,
                "l2TokensOnDisk": 4225,
            },
            "runtime": {"quantizationProfile": "JANGTQ_2"},
        },
    )
    _write_json(
        tmp_path / checklist.GEMMA4_12B_ISSUE191_STARTUP_VISIBLE,
        {
            "status": "pass",
            "model": "/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
            "checks": {
                "import_alias_ok": True,
                "startup_health_ok": True,
                "visible_generation_ok": True,
                "post_chat_health_ok": True,
            },
            "chat_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "GEMMA4-OK", "role": "assistant"},
                    }
                ]
            },
        },
    )
    _write_qwen_green_artifacts(tmp_path)

    result = _build(tmp_path)

    assert result["status"] == "open"
    assert result["release_ready"] is False
    failed_names = {row["name"] for row in result["failed"]}
    assert "release_ready" in failed_names
    assert "real_ui_live_model_matrix" in failed_names
    assert "mimo_tool_protocol" in failed_names
    assert "mimo_media_model_metadata_text_only_contract" in failed_names
    assert "mimo_media_runtime_implementation" not in failed_names
    assert "mimo_mimo_media_wired" not in failed_names
    assert "mimo_jangtq2_media_semantics_release_quality" in failed_names
    assert "mimo_source_vs_quant_first_divergence" not in failed_names
    assert "mimo_source_vs_quant_requirement_satisfied" not in failed_names
    assert "mimo_no_source_classifier_tracks_exactness_boundary" not in failed_names
    mimo_artifact_rows = [
        row
        for row in result["groups"]["mimo_v25_jangtq2"]
        if row["name"] == "mimo_artifact_exactness"
    ]
    assert len(mimo_artifact_rows) == 1
    assert mimo_artifact_rows[0]["detail"]["jangtq2_boundary"][
        "classification"
    ] == "model_generated_literal_mutation_after_valid_parser_structure"
    assert mimo_artifact_rows[0]["detail"]["jang2l_boundary"]["classification"] == (
        "empty_visible_output_after_generation_stop"
    )
    assert mimo_artifact_rows[0]["detail"]["prompt_shape_first_token"][
        "failing_top_token"
    ]["text"] == "<|im_end|>"
    assert mimo_artifact_rows[0]["detail"]["prompt_shape_first_token"][
        "working_folded_top_token"
    ]["text"] == "ACK"
    assert mimo_artifact_rows[0]["detail"]["classifier"]["classification"] == (
        "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
    )
    assert (
        mimo_artifact_rows[0]["detail"]["classifier"]["model_upload_action_required"]
        is True
    )
    assert any(
        "JANGTQ_2 served artifact emits wrong compact-hyphen/sentinel literals"
        in reason
        for reason in mimo_artifact_rows[0]["detail"]["classifier"][
            "model_upload_action_reasons"
        ]
    )
    mimo_exactness_rows = [
        row
        for row in result["groups"]["mimo_v25_jangtq2"]
        if row["name"] == "mimo_no_source_classifier_tracks_exactness_boundary"
    ]
    assert len(mimo_exactness_rows) == 1
    assert mimo_exactness_rows[0]["detail"]["artifact_exactness_release_action"] == (
        "replace_all_routed_2bit_jangtq2_or_lift_gate_down_precision"
    )
    assert mimo_exactness_rows[0]["detail"]["recommended_next_artifact_profiles"] == [
        "JANGTQ gate=3/up=2/down=3",
        "JANGTQ gate=3/up=3/down=3",
    ]
    mimo_media_rows = {
        row["name"]: row
        for row in result["groups"]["mimo_v25_jangtq2"]
        if row["name"].startswith("mimo_jangtq2_")
    }
    assert mimo_media_rows["mimo_jangtq2_current_source_media_runtime"]["ok"] is True
    assert mimo_media_rows["mimo_jangtq2_current_source_video_audio_routes"]["ok"] is True
    assert mimo_media_rows["mimo_jangtq2_dev_app_responses_tools_cache"]["ok"] is True
    assert mimo_media_rows["mimo_jangtq2_media_semantics_release_quality"]["ok"] is False
    gemma_startup_rows = [
        row
        for row in result["groups"]["gemma4_12b"]
        if row["name"] == "gemma4_12b_issue191_startup_visible_generation"
    ]
    assert len(gemma_startup_rows) == 1
    assert gemma_startup_rows[0]["ok"] is True
    assert (
        gemma_startup_rows[0]["evidence"]
        == "build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json"
    )
    assert "gemma4_12b_media_rows_complete" in failed_names
    assert "step37_real_vlm_runtime_complete" in failed_names
    assert "nemotron_omni_media_rows_complete" in failed_names
    assert "dsv4_exactness_preflight_artifact_exists" in failed_names
    assert "dsv4_memory_preflight_resource_available" in failed_names
    assert "dsv4_exact_code_file_long_output_complete" in failed_names


def test_full_release_objective_checklist_tracks_open_n2_pro_objective_row(
    tmp_path,
):
    _write_json(
        tmp_path / checklist.OBJECTIVE_DIGEST,
        {
            "requirements": [
                {
                    "requirement": (
                        "N2 Pro 397B JANG1L/JANGTQ runtime/cache/API/UI quality "
                        "is release-cleared"
                    ),
                    "status": "open",
                    "evidence": [
                        "build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json",
                        str(checklist.OBJECTIVE_DIGEST),
                    ],
                    "details": {
                        "local_artifact_probe": {
                            "artifact_present": True,
                            "memory_preflight_decision": "do_not_launch",
                            "launch_safe": False,
                        },
                        "jang1l_live_gate": {
                            "artifact": "build/current-n2-jang1l-chat-cache-proof-20260609.json",
                            "status": "skipped",
                            "reason": "n2_jang1l_insufficient_available_memory",
                            "available_gib": 111.61,
                            "required_available_gib": 118.57,
                            "memory_gap_gib": 6.96,
                        },
                        "noheavy_contracts": {
                            "api_cache": "pass",
                            "cache_architecture": "pass",
                            "model_family_detection": "pass",
                        },
                        "required_next_evidence": [
                            "memory_safe_live_launch_or_smaller_runtime_strategy",
                            "runtime_cache_api_ui_live_proof",
                            "typed_cache_prefix_paged_l2_restore",
                            "mlxstudio_settings_startup_parity",
                        ],
                    },
                }
            ]
        },
    )

    result = _build(tmp_path)

    failed_names = {row["name"] for row in result["failed"]}
    assert "n2_pro_397b_release_clearance" in failed_names
    n2_rows = [
        row
        for row in result["groups"]["n2_pro_397b"]
        if row["name"] == "n2_pro_397b_release_clearance"
    ]
    assert len(n2_rows) == 1
    assert n2_rows[0]["detail"]["status"] == "open"
    assert n2_rows[0]["detail"]["local_artifact_probe"]["artifact_present"] is True
    assert n2_rows[0]["detail"]["local_artifact_probe"]["memory_preflight_decision"] == "do_not_launch"
    assert n2_rows[0]["detail"]["jang1l_live_gate"]["status"] == "skipped"
    assert n2_rows[0]["detail"]["jang1l_live_gate"]["reason"] == (
        "n2_jang1l_insufficient_available_memory"
    )
    assert "runtime_cache_api_ui_live_proof" in n2_rows[0]["detail"][
        "required_next_evidence"
    ]


def test_full_release_objective_checklist_blocks_open_gemma_qat_inventory():
    data = {
        "artifact": str(checklist.GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        "exists": True,
        "status": "open",
        "count": 11,
        "missing_required_rows": [
            "gemma4_e2b_qat_native_mxfp4",
            "gemma4_e4b_qat_native_mxfp4",
        ],
        "open_required_rows": [
            "gemma4_12b_native_mxfp4",
            "gemma4_26b_vl",
            "gemma4_31v_or_31b_vl",
        ],
        "checks": {
            "gemma4_e2b_qat_native_mxfp4_present": False,
            "gemma4_e4b_qat_native_mxfp4_present": False,
            "gemma4_12b_native_mxfp4_present": True,
            "gemma4_26b_present": True,
            "gemma4_31v_or_31b_present": True,
            "all_required_source_live_smokes_present": False,
            "all_required_live_proofs_present": False,
            "gemma4_12b_audio_weight_backed": False,
            "gemma4_12b_audio_honestly_gated": False,
            "gemma4_12b_vision_weight_backed": True,
            "gemma4_12b_video_runtime_proof_required": True,
            "gemma4_12b_video_runtime_source_proven": False,
            "gemma4_26b_video_runtime_proof_required": True,
            "gemma4_26b_video_runtime_source_proven": False,
            "gemma4_31v_or_31b_video_runtime_proof_required": True,
            "gemma4_31v_or_31b_video_runtime_source_proven": False,
        },
    }

    rows = checklist._gemma_qat_native_mxfp4_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "gemma_qat_native_mxfp4_status_pass" in failed
    assert "gemma_qat_native_mxfp4_gemma4_e2b_present" in failed
    assert "gemma_qat_native_mxfp4_gemma4_e4b_present" in failed
    assert "gemma_qat_native_mxfp4_gemma4_12b_audio_weight_backed" in failed
    assert "gemma_qat_native_mxfp4_gemma4_12b_video_runtime_proven" in failed
    assert "gemma_qat_native_mxfp4_gemma4_26b_video_runtime_proven" in failed
    assert "gemma_qat_native_mxfp4_gemma4_31v_or_31b_video_runtime_proven" in failed
    assert "gemma_qat_native_mxfp4_all_source_live_smokes_present" in failed
    assert "gemma_qat_native_mxfp4_all_live_proofs_present" in failed
    assert failed["gemma_qat_native_mxfp4_all_live_proofs_present"]["detail"] == {
        "missing_required_rows": [
            "gemma4_e2b_qat_native_mxfp4",
            "gemma4_e4b_qat_native_mxfp4",
        ],
        "open_required_rows": [
            "gemma4_12b_native_mxfp4",
            "gemma4_26b_vl",
            "gemma4_31v_or_31b_vl",
        ],
        "source_live_smoke_open_rows": None,
    }


def test_full_release_objective_checklist_blocks_open_gemma_qat_jang4m_rows():
    jang4m_rows = {
        key: {
            "status": "open",
            "variant": "qat_jang4m",
            "live_proof_status": "missing",
            "live_proof_required": [
                "autodetect_model_family_and_qat_jang4m_variant",
                "responses_streaming_args_and_content_deltas",
                "installed_app_parity",
            ],
        }
        for key in [
            "gemma4_e2b_qat_jang4m",
            "gemma4_e4b_qat_jang4m",
            "gemma4_12b_qat_jang4m",
            "gemma4_26b_qat_jang4m",
            "gemma4_31b_qat_jang4m",
        ]
    }
    data = {
        "artifact": str(checklist.GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        "exists": True,
        "status": "open",
        "missing_required_rows": [],
        "open_required_rows": list(jang4m_rows),
        "source_live_smoke_open_rows": [],
        "required_rows": jang4m_rows,
        "checks": {
            f"{key}_present": True for key in jang4m_rows
        }
        | {
            "all_required_source_live_smokes_present": True,
            "all_required_live_proofs_present": False,
        },
    }

    rows = checklist._gemma_qat_native_mxfp4_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    for key in jang4m_rows:
        failed_key = f"gemma_qat_native_mxfp4_{key}_open"
        assert failed_key in failed
        assert failed[failed_key]["detail"]["variant"] == "qat_jang4m"


def test_full_release_objective_checklist_accepts_gemma_qat_source_video_proof():
    data = {
        "artifact": str(checklist.GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        "exists": True,
        "status": "open",
        "count": 16,
        "missing_required_rows": [],
        "open_required_rows": [
            "gemma4_e2b_qat_native_mxfp4",
            "gemma4_e4b_qat_native_mxfp4",
            "gemma4_12b_native_mxfp4",
            "gemma4_26b_vl",
            "gemma4_31v_or_31b_vl",
        ],
        "source_live_smoke_open_rows": [],
        "checks": {
            "gemma4_e2b_qat_native_mxfp4_present": True,
            "gemma4_e4b_qat_native_mxfp4_present": True,
            "gemma4_12b_native_mxfp4_present": True,
            "gemma4_26b_present": True,
            "gemma4_31v_or_31b_present": True,
            "all_required_source_live_smokes_present": True,
            "all_required_live_proofs_present": False,
            "gemma4_12b_audio_weight_backed": False,
            "gemma4_12b_audio_honestly_gated": True,
            "gemma4_12b_video_runtime_proof_required": True,
            "gemma4_12b_video_runtime_source_proven": True,
            "gemma4_26b_video_runtime_proof_required": True,
            "gemma4_26b_video_runtime_source_proven": True,
            "gemma4_31v_or_31b_video_runtime_proof_required": True,
            "gemma4_31v_or_31b_video_runtime_source_proven": True,
        },
    }

    rows = checklist._gemma_qat_native_mxfp4_checks(data)
    failed_names = {row["name"] for row in rows if not row["ok"]}

    assert "gemma_qat_native_mxfp4_gemma4_12b_video_runtime_proven" not in failed_names
    assert "gemma_qat_native_mxfp4_gemma4_26b_video_runtime_proven" not in failed_names
    assert "gemma_qat_native_mxfp4_gemma4_31v_or_31b_video_runtime_proven" not in failed_names
    assert "gemma_qat_native_mxfp4_all_live_proofs_present" in failed_names


def test_full_release_objective_checklist_blocks_missing_responses_tunnel_capture():
    data = {
        "artifact": str(checklist.RESPONSES_RAW_SSE_PARITY),
        "exists": True,
        "status": "open",
        "missing_captures": ["tunnel"],
        "checks": {
            "direct_capture_present": True,
            "gateway_capture_present": True,
            "tunnel_capture_present": False,
            "all_required_surfaces_present": False,
            "all_present_surfaces_parse_cleanly": True,
            "all_present_surfaces_match_expected_function_name": True,
            "all_present_surfaces_match_expected_arguments": True,
            "all_present_surfaces_have_authoritative_args": True,
            "authoritative_arguments_match_across_present_surfaces": True,
            "tunnel_expected_model_advertised": True,
            "local_responses_streaming_guards_pass": True,
            "local_empty_xml_arguments_fail_closed": True,
            "local_output_index_ordering_guard": True,
            "gateway_argument_stream_passthrough_guard": True,
            "responses_previous_response_history_guard": True,
            "all_present_surfaces_have_valid_output_item_indices": True,
            "all_present_surfaces_have_required_reasoning": True,
            "no_reasoning_disable_workaround": True,
        },
    }

    rows = checklist._responses_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "responses_raw_sse_parity_status_pass" in failed
    assert "responses_raw_sse_parity_tunnel_capture_present" in failed
    assert "responses_raw_sse_parity_all_required_surfaces_present" in failed
    assert failed["responses_raw_sse_parity_all_required_surfaces_present"][
        "detail"
    ] == {"missing_captures": ["tunnel"]}


def test_full_release_objective_checklist_blocks_reasoning_disable_workaround():
    data = {
        "artifact": str(checklist.RESPONSES_RAW_SSE_PARITY),
        "exists": True,
        "status": "fail",
        "missing_captures": [],
        "checks": {
            "direct_capture_present": True,
            "gateway_capture_present": True,
            "tunnel_capture_present": True,
            "all_required_surfaces_present": True,
            "all_present_surfaces_parse_cleanly": True,
            "all_present_surfaces_match_expected_function_name": True,
            "all_present_surfaces_match_expected_arguments": True,
            "all_present_surfaces_have_authoritative_args": True,
            "authoritative_arguments_match_across_present_surfaces": True,
            "tunnel_expected_model_advertised": True,
            "local_responses_streaming_guards_pass": True,
            "local_empty_xml_arguments_fail_closed": True,
            "local_output_index_ordering_guard": True,
            "gateway_argument_stream_passthrough_guard": True,
            "responses_previous_response_history_guard": True,
            "all_present_surfaces_have_valid_output_item_indices": True,
            "all_present_surfaces_have_required_reasoning": False,
            "no_reasoning_disable_workaround": False,
        },
    }

    rows = checklist._responses_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "responses_raw_sse_parity_status_pass" in failed
    assert "responses_raw_sse_parity_all_present_surfaces_have_required_reasoning" in failed
    assert "responses_raw_sse_parity_no_reasoning_disable_workaround" in failed


def test_full_release_objective_checklist_separates_missing_reasoning_events_from_disable_workaround():
    data = {
        "artifact": str(checklist.RESPONSES_RAW_SSE_PARITY),
        "exists": True,
        "status": "fail",
        "missing_captures": [],
        "checks": {
            "direct_capture_present": True,
            "gateway_capture_present": True,
            "tunnel_capture_present": True,
            "all_required_surfaces_present": True,
            "all_present_surfaces_parse_cleanly": True,
            "all_present_surfaces_match_expected_function_name": True,
            "all_present_surfaces_match_expected_arguments": True,
            "all_present_surfaces_have_authoritative_args": True,
            "authoritative_arguments_match_across_present_surfaces": True,
            "tunnel_expected_model_advertised": True,
            "local_responses_streaming_guards_pass": True,
            "local_empty_xml_arguments_fail_closed": True,
            "local_output_index_ordering_guard": True,
            "gateway_argument_stream_passthrough_guard": True,
            "responses_previous_response_history_guard": True,
            "all_present_surfaces_have_valid_output_item_indices": True,
            "all_present_surfaces_have_required_reasoning": False,
            "no_reasoning_disable_workaround": True,
        },
    }

    rows = checklist._responses_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "responses_raw_sse_parity_status_pass" in failed
    assert "responses_raw_sse_parity_all_present_surfaces_have_required_reasoning" in failed
    assert "responses_raw_sse_parity_no_reasoning_disable_workaround" not in failed


def test_full_release_objective_checklist_surfaces_tunnel_model_availability():
    data = {
        "artifact": str(checklist.RESPONSES_RAW_SSE_PARITY),
        "exists": True,
        "status": "fail",
        "missing_captures": [],
        "checks": {
            "direct_capture_present": True,
            "gateway_capture_present": True,
            "tunnel_capture_present": True,
            "all_required_surfaces_present": True,
            "all_present_surfaces_parse_cleanly": True,
            "all_present_surfaces_match_expected_function_name": False,
            "all_present_surfaces_match_expected_arguments": False,
            "all_present_surfaces_have_authoritative_args": False,
            "authoritative_arguments_match_across_present_surfaces": False,
            "tunnel_expected_model_advertised": False,
            "local_responses_streaming_guards_pass": True,
            "local_empty_xml_arguments_fail_closed": True,
            "local_output_index_ordering_guard": True,
            "gateway_argument_stream_passthrough_guard": True,
            "responses_previous_response_history_guard": True,
            "all_present_surfaces_have_valid_output_item_indices": True,
            "all_present_surfaces_have_required_reasoning": False,
            "no_reasoning_disable_workaround": True,
        },
    }

    rows = checklist._responses_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "responses_raw_sse_parity_tunnel_expected_model_advertised" in failed
    assert "responses_raw_sse_parity_no_reasoning_disable_workaround" not in failed


def test_full_release_objective_checklist_blocks_raw_sse_duplicate_output_index():
    data = {
        "artifact": str(checklist.RESPONSES_RAW_SSE_PARITY),
        "exists": True,
        "status": "fail",
        "missing_captures": [],
        "checks": {
            "direct_capture_present": True,
            "gateway_capture_present": True,
            "tunnel_capture_present": True,
            "all_required_surfaces_present": True,
            "all_present_surfaces_parse_cleanly": True,
            "all_present_surfaces_match_expected_function_name": True,
            "all_present_surfaces_match_expected_arguments": True,
            "all_present_surfaces_have_authoritative_args": True,
            "authoritative_arguments_match_across_present_surfaces": True,
            "tunnel_expected_model_advertised": True,
            "local_responses_streaming_guards_pass": True,
            "local_empty_xml_arguments_fail_closed": True,
            "local_output_index_ordering_guard": True,
            "gateway_argument_stream_passthrough_guard": True,
            "responses_previous_response_history_guard": True,
            "all_present_surfaces_have_valid_output_item_indices": False,
            "all_present_surfaces_have_required_reasoning": True,
            "no_reasoning_disable_workaround": True,
        },
    }

    rows = checklist._responses_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "responses_raw_sse_parity_status_pass" in failed
    assert (
        "responses_raw_sse_parity_all_present_surfaces_have_valid_output_item_indices"
        in failed
    )
    assert "responses_raw_sse_parity_no_reasoning_disable_workaround" not in failed


def test_full_release_objective_checklist_blocks_qwen35_raw_sse_duplicate_output_index():
    data = {
        "exists": True,
        "status": "fail",
        "missing_captures": [],
        "checks": {
            "all_present_surfaces_same_model": True,
            "all_present_surfaces_match_expected_model": True,
            "all_present_surfaces_have_authoritative_args": True,
            "all_present_surfaces_match_expected_arguments": True,
            "all_present_surfaces_have_required_reasoning": True,
            "no_reasoning_disable_workaround": True,
            "all_present_surfaces_have_valid_output_item_indices": False,
            "local_responses_streaming_guards_pass": True,
            "local_output_index_ordering_guard": True,
            "local_empty_xml_arguments_fail_closed": True,
        },
        "captures": {
            "tunnel": {
                "conflicting_output_indices": [0],
                "output_indices_by_type": {"message": [0], "function_call": [0]},
            }
        },
    }

    rows = checklist._qwen35_raw_sse_parity_checks(data)
    failed = {row["name"]: row for row in rows if not row["ok"]}

    assert "qwen35_raw_sse_status_pass" in failed
    assert "qwen35_raw_sse_valid_output_item_indices" in failed
    assert "qwen35_raw_sse_reasoning_events" not in failed
    assert failed["qwen35_raw_sse_valid_output_item_indices"]["detail"] == {
        "missing_captures": [],
        "conflicting_output_indices": {"tunnel": [0]},
    }


def test_full_release_objective_checklist_can_pass_when_all_evidence_is_green(
    tmp_path,
):
    _write_json(
        tmp_path / checklist.RELEASE_MANIFEST,
        {
            "status": "pass",
            "prepackage_ready": True,
            "release_ready": True,
            "current_proof_sweep": {
                "component_ok": {
                    "packaged_integrity_matrix": True,
                    "real_ui_full_model_matrix": True,
                    "real_ui_live_model_proof": True,
                }
            },
        },
    )
    _write_json(
        tmp_path / checklist.NOHEAVY_API_CACHE,
        {
            "status": "pass",
            "checks": {
                "responses_sampling_kwargs": True,
                "openai_chat_sampling_kwargs": True,
                "streaming_cache_detail_usage": True,
                "responses_previous_response_history": True,
                "cache_stats_reuse_skip_telemetry": True,
                "cache_reuse_endpoints": True,
                "legacy_completions_output_caps_override_server_default": True,
                "request_output_caps_override_server_default": True,
                "prompt_context_caps_stay_separate_from_output_caps": True,
                "json_response_format_stream_validation": True,
                "responses_text_format_json_schema_preserved": True,
                "anthropic_bundle_defaults": True,
                "ollama_adapter_surface": True,
            },
        },
    )
    _write_green_api_surface_artifact(tmp_path)
    _write_json(
        tmp_path / checklist.TOOL_CALL_CONTRACT,
        {
            "status": "pass",
            "checks": {
                "tool_parser_residue_rejected_instead_of_executed": True,
                "schema_valid_dsml_tool_call_preserved": True,
                "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid": True,
                "dsv4_tool_preamble_suppressed_and_not_stored_without_call": True,
                "tool_choice_none_does_not_fallback_to_raw_dsml": True,
                "family_tool_parser_matrix_covers_no_leak_and_alias_edges": True,
                "panel_max_tool_iterations_caps_tool_loops": True,
                "all_required_tool_call_markers_present": True,
            },
        },
    )
    _write_green_panel_settings_artifact(tmp_path)
    _write_json(
        tmp_path / checklist.MIMO_AUDIT,
        {
            "status": "pass",
            "local_release_clearance": True,
            "component_ok": {
                "manifest_integrity": True,
                "decode_speed_target": True,
                "api_cache_responses_contract": True,
                "prefix_paged_l2_cache_reproved": True,
                "long_prompt_coherence": True,
                "tool_protocol": True,
                "artifact_exactness": True,
                "cb_system_prompt_working_set_pressure": True,
                "source_vs_quant_first_divergence": True,
                "source_vs_quant_requirement_satisfied": True,
                "media_weights_preserved": True,
                "media_runtime_capabilities_safe": True,
                "media_model_metadata_text_only_contract": True,
                "media_runtime_implementation": True,
                "mimo_media_wired": True,
            },
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER,
        {
            "status": "pass",
            "classification": "exactness_release_cleared_by_stronger_evidence",
            "source_vs_quant_load_performed": False,
            "excluded_surfaces": {
                "parser_argument_rewrite": True,
                "prefix_paged_l2_or_kv_quant_primary_cause": True,
                "hidden_stochastic_sampling_primary_cause": True,
            },
            "unresolved_surfaces": {},
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE,
        {
            "status": "pass",
            "proven": {
                "api_routes_mllm": True,
                "loader_overlay_auto_enabled": True,
                "preserved_media_weights_bound": True,
                "live_chat_image_200": True,
                "prior_unsupported_media_400_cleared_for_source_image": True,
            },
            "not_proven": {},
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE,
        {
            "status": "pass",
            "proven": {
                "source_server_loads_as_mllm": True,
                "media_weights_bound": True,
                "video_request_reaches_runtime": True,
                "video_http_200": True,
                "audio_request_reaches_runtime": True,
                "audio_http_200": True,
            },
            "not_proven": {},
        },
    )
    _write_json(
        tmp_path / checklist.MIMO_JANGTQ2_RESPONSES_TOOLS_CACHE_UI,
        {
            "status": "pass",
            "classification": "dev_app_responses_tools_cache_green_exactness_still_bounded",
            "cache": {
                "nativeCacheSubtype": "mimo_v2_asymmetric_swa",
                "cacheHitTokens": 1,
                "l2TokensOnDisk": 1,
            },
            "runtime": {"quantizationProfile": "JANGTQ_2"},
        },
    )
    _write_issue179_green_artifact(tmp_path)
    _write_green_responses_raw_sse_artifact(tmp_path)
    _write_dsv4_green_artifact(tmp_path)
    _write_green_family_smokes(tmp_path)
    _write_json(
        tmp_path / checklist.GEMMA_QAT_NATIVE_MXFP4_INVENTORY,
        {
            "status": "pass",
            "count": 16,
            "missing_required_rows": [],
            "open_required_rows": [],
            "source_live_smoke_open_rows": [],
            "required_rows": {
                key: {"status": "pass", "live_proof_status": "pass"}
                for key in [
                    "gemma4_e2b_qat_jang4m",
                    "gemma4_e4b_qat_jang4m",
                    "gemma4_12b_qat_jang4m",
                    "gemma4_26b_qat_jang4m",
                    "gemma4_31b_qat_jang4m",
                ]
            },
            "checks": {
                "gemma4_e2b_qat_native_mxfp4_present": True,
                "gemma4_e4b_qat_native_mxfp4_present": True,
                "gemma4_12b_native_mxfp4_present": True,
                "gemma4_12b_audio_honestly_gated": True,
                "gemma4_26b_present": True,
                "gemma4_31v_or_31b_present": True,
                "gemma4_e2b_qat_jang4m_present": True,
                "gemma4_e2b_qat_jang4m_installed_app_ui_api_cache_proven": True,
                "gemma4_e4b_qat_jang4m_present": True,
                "gemma4_e4b_qat_jang4m_installed_app_ui_api_cache_proven": True,
                "gemma4_12b_qat_jang4m_present": True,
                "gemma4_12b_qat_jang4m_installed_app_ui_api_cache_proven": True,
                "gemma4_26b_qat_jang4m_present": True,
                "gemma4_31b_qat_jang4m_present": True,
                "all_required_source_live_smokes_present": True,
                "all_required_live_proofs_present": True,
            },
        },
    )
    _write_qwen_green_artifacts(tmp_path)
    _write_green_n2_objective_digest(tmp_path)

    result = _build(tmp_path)

    assert result["status"] == "open"
    failed_names = {row["name"] for row in result["failed"]}
    assert failed_names == {
        "step37_real_vlm_runtime_complete",
    }
